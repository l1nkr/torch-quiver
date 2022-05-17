#pragma once
#include <quiver/algorithm.cu.hpp>
#include <quiver/common.hpp>
#include <quiver/cuda_random.cu.hpp>
#include <quiver/functor.cu.hpp>
#include <quiver/quiver.hpp>
#include <quiver/trace.hpp>
#include <quiver/zip.cu.hpp>

#include <thrust/binary_search.h>
#include <thrust/device_vector.h>

#include <cuda_runtime.h>

#define quiverRegister(ptr, size, flag)                                        \
    ;                                                                          \
    {                                                                          \
        size_t BLOCK = 1000000000;                                             \
        void *register_ptr = (void *)ptr;                                      \
        for (size_t pos = 0; pos < size; pos += BLOCK) {                       \
            size_t s = BLOCK;                                                  \
            if (size - pos < BLOCK) { s = size - pos; }                        \
            cudaHostRegister(register_ptr + pos, s, flag);                     \
        }                                                                      \
    }

namespace quiver
{
template <typename T>
class get_adj_diff
{

    // x 节点
    // n 节点数
    // tot 边数
    const T *x;
    const size_t n;
    const size_t tot;

  public:
    get_adj_diff(const T *x, const size_t n, const size_t tot)
        : x(x), n(n), tot(tot)
    {
    }
    // thrust::transform(
    //     thrust::cuda::par.on(stream), input_begin, input_end,
    //     output_begin,
    //     get_adj_diff<T>(row_ptr_mapped_, node_count_, edge_count_));

    __host__ __device__ T operator()(T i) const
    {   
        
        const T end = i + 1 < n ? x[i + 1] : tot;
        return end - x[i];
    }
};

struct sample_option {
    sample_option(bool w, bool u, bool p)
        : weighted(w), use_id(u), partitioned(p)
    {
    }
    bool weighted;
    bool use_id;
    bool partitioned;
};

// make edge weight a nomalized prefix sum within each node
template <typename T, typename W>
class bucket_weight_functor
{
    const W *src_;
    W *dst_;

  public:
    bucket_weight_functor(const W *src, W *dst) : src_(src), dst_(dst) {}
    template <typename P>
    __device__ void operator()(P ptrs)
    {
        T prev = thrust::get<0>(ptrs);
        T next = thrust::get<1>(ptrs);
        if (prev == next) { return; }
        W sum = 0;
        for (T temp = prev; temp != next; temp++) {
            dst_[temp] = sum;
            sum += src_[temp];
        }
        for (T temp = prev; temp != next; temp++) { dst_[temp] /= sum; }
    }
};

template <typename T, typename W>
class sample_functor
{
    const T *row_ptr;
    const size_t n;
    const T *col_idx;
    const T *edge_id;
    const W *edge_weight;
    const size_t m;

    T *output;
    T *output_id;

    bool weighted;
    bool use_id;

  public:
    sample_functor(const T *row_ptr, size_t n, const T *col_idx,
                   const T *edge_id, const W *edge_weight, size_t m, T *output,
                   T *output_id, bool weighted, bool use_id)
        : row_ptr(row_ptr),
          n(n),
          col_idx(col_idx),
          edge_id(edge_id),
          edge_weight(edge_weight),
          m(m),
          output(output),
          output_id(output_id),
          weighted(weighted),
          use_id(use_id)
    {
    }

    __device__ void operator()(const thrust::tuple<size_t, T, T, T> &t) const
    {
        const T &v = thrust::get<1>(t);
        const T &count = thrust::get<2>(t);
        const T &out_ptr = thrust::get<3>(t);

        const T begin = row_ptr[v];
        const T end = v + 1 < n ? row_ptr[v + 1] : m;
        const W *begin_weight = weighted ? edge_weight + begin : nullptr;
        const T *begin_id = use_id ? edge_id + begin : nullptr;

        if (!weighted) {
            cuda_random_generator g(thrust::get<0>(t));
            safe_sample(col_idx + begin, col_idx + end, begin_id, begin_weight,
                        count, output + out_ptr, output_id + out_ptr, &g);
        } else {
            cuda_uniform_generator g(thrust::get<0>(t));
            safe_sample(col_idx + begin, col_idx + end, begin_id, begin_weight,
                        count, output + out_ptr, output_id + out_ptr, &g);
        }
    }
};

template <typename T, typename W>
void bucket_weight(const thrust::device_vector<T> &prev,
                   const thrust::device_vector<T> &next,
                   const thrust::device_vector<W> &src,
                   thrust::device_vector<W> &dst)
{
    thrust::for_each(
        thrust::make_zip_iterator(
            thrust::make_tuple(prev.begin(), next.begin())),
        thrust::make_zip_iterator(thrust::make_tuple(prev.end(), next.end())),
        bucket_weight_functor<T, W>(thrust::raw_pointer_cast(src.data()),
                                    thrust::raw_pointer_cast(dst.data())));
}

template <typename T>
class quiver<T, CUDA>
{
    using self = quiver<T, CUDA>;
    using W = float;

    const thrust::device_vector<T> row_ptr_;
    const thrust::device_vector<T> col_idx_;
    const thrust::device_vector<T> edge_idx_;  // TODO: make it optional

    const thrust::device_vector<W> edge_weight_;         // optional
    const thrust::device_vector<W> bucket_edge_weight_;  // optional

    // Zero-Map Mode Parameters Begin
    const T *row_ptr_mapped_;
    const T *col_idx_mapped_;
    const T *edge_idx_mapped;
    const W *edge_weight_mapped_;
    const W *bucket_edge_weight_mapped_;
    int node_count_;
    int edge_count_;
    // Zero-Map Mode Parameters End

    // Quiver Mode
    QuiverMode quiver_mode;

    const sample_option opt_;

    quiver(thrust::device_vector<T> row_ptr, thrust::device_vector<T> col_idx,
           thrust::device_vector<T> edge_idx)
        : row_ptr_(std::move(row_ptr)),
          col_idx_(std::move(col_idx)),
          edge_idx_(std::move(edge_idx)),
          opt_(false, !edge_idx.empty(), true),
          quiver_mode(DMA)
    {
    }

    quiver(T *row_ptr, T *col_idx, T *edge_idx, T node_count, T edge_count)
        : row_ptr_mapped_(std::move(row_ptr)),
          col_idx_mapped_(std::move(col_idx)),
          edge_idx_mapped(std::move(edge_idx)),
          node_count_(node_count),
          edge_count_(edge_count),
          opt_(false, edge_idx != nullptr, false),
          quiver_mode(ZERO_COPY)
    {
    }

  public:
    static self New(T n, thrust::device_vector<T> row_idx,
                    thrust::device_vector<T> col_idx,
                    thrust::device_vector<T> edge_idx)
    {
        if (!edge_idx.empty()) {
            thrust::device_vector<thrust::tuple<T, T, T>> edges(row_idx.size());
            zip(row_idx, col_idx, edge_idx, edges);
            thrust::sort(edges.begin(), edges.end());
            unzip(edges, row_idx, col_idx, edge_idx);
        } else {
            thrust::device_vector<thrust::tuple<T, T>> edges(row_idx.size());
            zip(row_idx, col_idx, edges);
            thrust::sort(edges.begin(), edges.end());
            unzip(edges, row_idx, col_idx);
        }
        thrust::device_vector<T> row_ptr(n);
        thrust::sequence(row_ptr.begin(), row_ptr.end());
        thrust::lower_bound(row_idx.begin(), row_idx.end(), row_ptr.begin(),
                            row_ptr.end(), row_ptr.begin());
        return self(row_ptr, col_idx, edge_idx);
    }

    static self New(T *row_idx, T *col_idx, T *edge_idx, T node_count,
                    T edge_count)
    {
        return self(row_idx, col_idx, edge_idx, node_count, edge_count);
    }

    virtual ~quiver() = default;

    size_t size() const
    {
        if (quiver_mode == DMA) { return row_ptr_.size(); }
        return node_count_;
    }

    size_t edge_counts() const
    {
        if (quiver_mode == DMA) { return col_idx_.size(); }
        return edge_count_;
    }

    sample_option get_option() const { return opt_; }

    // device_t device() const   { return CUDA; }

    void degree(const cudaStream_t stream,
                thrust::device_ptr<const T> input_begin,
                thrust::device_ptr<const T> input_end,
                thrust::device_ptr<T> output_begin) const
    {
        if (quiver_mode == DMA) {
            thrust::transform(
                thrust::cuda::par.on(stream), input_begin, input_end,
                output_begin,
                get_adj_diff<T>(thrust::raw_pointer_cast(row_ptr_.data()),
                                row_ptr_.size(), col_idx_.size()));
        } else {
            thrust::transform(
                thrust::cuda::par.on(stream), input_begin, input_end,
                output_begin,
                get_adj_diff<T>(row_ptr_mapped_, node_count_, edge_count_));
        }
    }

    void async_degree(const cudaStream_t stream,
                      thrust::device_ptr<const T> input_begin,
                      thrust::device_ptr<const T> input_end,
                      thrust::device_ptr<T> output_begin) const
    {
        if (quiver_mode == DMA) {
            async_transform(
                kernal_option(stream), input_begin, input_end, output_begin,
                get_adj_diff<T>(thrust::raw_pointer_cast(row_ptr_.data()),
                                row_ptr_.size(), col_idx_.size()));
        } else {
            async_transform(
                kernal_option(stream), input_begin, input_end, output_begin,
                get_adj_diff<T>(row_ptr_mapped_, node_count_, edge_count_));
        }
    }

    template <typename Iter>
    void sample(const cudaStream_t stream, Iter input_begin, Iter input_end,
                Iter output_ptr_begin, Iter output_count_begin,
                thrust::device_ptr<T> output_begin,
                thrust::device_ptr<T> output_id_begin) const
    {
        const size_t len = input_end - input_begin;
        thrust::counting_iterator<size_t> i(0);
        auto begin = thrust::make_zip_iterator(thrust::make_tuple(
            i, input_begin, output_count_begin, output_ptr_begin));
        auto end = thrust::make_zip_iterator(
            thrust::make_tuple(i + len, input_end, output_count_begin + len,
                               output_ptr_begin + len));
        if (quiver_mode == DMA) {
            thrust::for_each(
                thrust::cuda::par.on(stream), begin, end,
                sample_functor<T, W>(
                    thrust::raw_pointer_cast(row_ptr_.data()), row_ptr_.size(),
                    thrust::raw_pointer_cast(col_idx_.data()),
                    thrust::raw_pointer_cast(edge_idx_.data()),
                    thrust::raw_pointer_cast(bucket_edge_weight_.data()),
                    col_idx_.size(), thrust::raw_pointer_cast(output_begin),
                    thrust::raw_pointer_cast(output_id_begin), opt_.weighted,
                    opt_.use_id));
        } else {
            thrust::for_each(
                thrust::cuda::par.on(stream), begin, end,
                sample_functor<T, W>(row_ptr_mapped_, node_count_,
                                     col_idx_mapped_, edge_idx_mapped,
                                     bucket_edge_weight_mapped_, edge_count_,
                                     thrust::raw_pointer_cast(output_begin),
                                     thrust::raw_pointer_cast(output_id_begin),
                                     opt_.weighted, opt_.use_id));
        }
    }

    std::pair<const T *, const T *> csr()
    {
        if (quiver_mode == DMA) {
            return std::make_pair(
                (const T *)thrust::raw_pointer_cast(row_ptr_.data()),
                (const T *)thrust::raw_pointer_cast(col_idx_.data()));
        } else {
            return std::make_pair(row_ptr_mapped_, col_idx_mapped_);
        }
    }
    // k 采样节点数 （25）
    // input_begin 需要进行采样的节点
    // input_size = input_begin.size()
    // output_ptr_begin 在这个节点之前需要采样多少个节点
    // output_count_begin 对应 input_begin 中每个节点需要采样多少个节点
    // output_begin 大小为总采样节点个数的空 vector，返回最终结果，需要扩散到哪些节点
    // output_idx 大小为总采样节点个数的空 vector
    void new_sample(const cudaStream_t stream, int k, T *input_begin,
                    int input_size, T *output_ptr_begin, T *output_count_begin,
                    T *output_begin, T *output_idx) const
    {
        constexpr int BLOCK_WARPS = 128 / WARP_SIZE;
        // the number of rows each thread block will cover
        constexpr int TILE_SIZE = BLOCK_WARPS * 16;
        const dim3 block(WARP_SIZE, BLOCK_WARPS);
        const dim3 grid((input_size + TILE_SIZE - 1) / TILE_SIZE);
        if (quiver_mode == DMA) {
            CSRRowWiseSampleKernel<T, BLOCK_WARPS, TILE_SIZE>
                <<<grid, block, 0, stream>>>(
                    0, k, input_size, input_begin,
                    thrust::raw_pointer_cast(row_ptr_.data()),
                    thrust::raw_pointer_cast(col_idx_.data()), output_ptr_begin,
                    output_count_begin, output_begin, output_idx);
        } else {
            CSRRowWiseSampleKernel<T, BLOCK_WARPS, TILE_SIZE>
                <<<grid, block, 0, stream>>>(
                    0, k, input_size, input_begin, row_ptr_mapped_,
                    col_idx_mapped_, output_ptr_begin, output_count_begin,
                    output_begin, output_idx);
        }
    }

    template <typename Iter>
    void async_sample(const cudaStream_t stream, Iter input_begin,
                      Iter input_end, Iter output_ptr_begin,
                      Iter output_count_begin,
                      thrust::device_ptr<T> output_begin,
                      thrust::device_ptr<T> output_id_begin) const
    {
        const size_t len = input_end - input_begin;
        thrust::counting_iterator<size_t> i(0);
        auto begin = thrust::make_zip_iterator(thrust::make_tuple(
            i, input_begin, output_count_begin, output_ptr_begin));
        auto end = thrust::make_zip_iterator(
            thrust::make_tuple(i + len, input_end, output_count_begin + len,
                               output_ptr_begin + len));
        if (quiver_mode == DMA) {
            async_for_each(
                kernal_option(stream), begin, end,
                sample_functor<T, W>(
                    thrust::raw_pointer_cast(row_ptr_.data()), row_ptr_.size(),
                    thrust::raw_pointer_cast(col_idx_.data()),
                    thrust::raw_pointer_cast(edge_idx_.data()),
                    thrust::raw_pointer_cast(bucket_edge_weight_.data()),
                    col_idx_.size(), thrust::raw_pointer_cast(output_begin),
                    thrust::raw_pointer_cast(output_id_begin), opt_.weighted));
        } else {
            async_for_each(
                kernal_option(stream), begin, end,
                sample_functor<T, W>(
                    row_ptr_mapped_, node_count_, col_idx_mapped_,
                    edge_idx_mapped, bucket_edge_weight_mapped_, edge_count_,
                    thrust::raw_pointer_cast(output_begin),
                    thrust::raw_pointer_cast(output_id_begin), opt_.weighted));
        }
    }

};
}  // namespace quiver
