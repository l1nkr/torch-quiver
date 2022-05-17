#include <algorithm>
#include <numeric>

#include <thrust/device_vector.h>

#include <pybind11/numpy.h>
#include <torch/extension.h>

#include <quiver/common.hpp>
#include <quiver/functor.cu.hpp>
#include <quiver/quiver.cu.hpp>
#include <quiver/reindex.cu.hpp>
#include <quiver/stream_pool.hpp>
#include <quiver/trace.hpp>
#include <quiver/zip.hpp>
#include <thrust/remove.h>
namespace quiver
{
using T = int64_t;
template <typename IdType>
HostOrderedHashTable<IdType> *
FillWithDuplicates(const IdType *const input, const size_t num_input,
                   cudaStream_t stream,
                   thrust::device_vector<IdType> &unique_items)
{
    const auto policy = thrust::cuda::par.on(stream);
    const int64_t num_tiles = (num_input + TILE_SIZE - 1) / TILE_SIZE;

    const dim3 grid(num_tiles);
    const dim3 block(BLOCK_SIZE);

    auto host_table = new HostOrderedHashTable<IdType>(num_input, 1);
    DeviceOrderedHashTable<IdType> device_table = host_table->DeviceHandle();

    generate_hashmap_duplicates<IdType, BLOCK_SIZE, TILE_SIZE>
        <<<grid, block, 0, stream>>>(input, num_input, device_table);
    thrust::device_vector<int> item_prefix(num_input + 1, 0);

    using it = thrust::counting_iterator<IdType>;
    using Mapping = typename DeviceOrderedHashTable<IdType>::Mapping;
    thrust::for_each(it(0), it(num_input),
                     [count = thrust::raw_pointer_cast(item_prefix.data()),
                      table = device_table,
                      in = input] __device__(IdType i) mutable {
                         Mapping &mapping = *(table.Search(in[i]));
                         if (mapping.index == i) { count[i] = 1; }
                     });
    thrust::exclusive_scan(item_prefix.begin(), item_prefix.end(),
                           item_prefix.begin());
    size_t tot = item_prefix[num_input];
    unique_items.resize(tot);

    thrust::for_each(it(0), it(num_input),
                     [prefix = thrust::raw_pointer_cast(item_prefix.data()),
                      table = device_table, in = input,
                      u = thrust::raw_pointer_cast(
                          unique_items.data())] __device__(IdType i) mutable {
                         Mapping &mapping = *(table.Search(in[i]));
                         if (mapping.index == i) {
                             mapping.local = prefix[i];
                             u[prefix[i]] = in[i];
                         }
                     });
    return host_table;
}


static void reindex_kernel(const cudaStream_t stream,
                               thrust::device_vector<T> &inputs,
                               thrust::device_vector<T> &outputs,
                               thrust::device_vector<T> &subset)
{
    const auto policy = thrust::cuda::par.on(stream);
    HostOrderedHashTable<T> *table;
    // reindex
    {
        {
            TRACE_SCOPE("reindex 0");
            subset.resize(inputs.size() + outputs.size());
            thrust::copy(policy, inputs.begin(), inputs.end(),
                            subset.begin());
            thrust::copy(policy, outputs.begin(), outputs.end(),
                            subset.begin() + inputs.size());
            thrust::device_vector<T> unique_items;
            unique_items.clear();
            table =
                FillWithDuplicates(thrust::raw_pointer_cast(subset.data()),
                                    subset.size(), stream, unique_items);
            subset.resize(unique_items.size());
            thrust::copy(policy, unique_items.begin(), unique_items.end(),
                            subset.begin());
            // thrust::sort(policy, subset.begin(), subset.end());
            // subset.erase(
            //     thrust::unique(policy, subset.begin(), subset.end()),
            //     subset.end());
            // _reindex_with(policy, outputs, subset, outputs);
        }
        {
            TRACE_SCOPE("permute");
            // thrust::device_vector<T> s1;
            // s1.reserve(subset.size());
            // _reindex_with(policy, inputs, subset, s1);
            // complete_permutation(s1, subset.size(), stream);
            // subset = permute(s1, subset, stream);

            // thrust::device_vector<T> s2;
            // inverse_permutation(s1, s2, stream);
            // permute_value(s2, outputs, stream);
            DeviceOrderedHashTable<T> device_table = table->DeviceHandle();
            thrust::for_each(
                policy, outputs.begin(), outputs.end(),
                [device_table] __device__(T & id) mutable {
                    using Iterator =
                        typename DeviceOrderedHashTable<T>::Iterator;
                    Iterator iter = device_table.Search(id);
                    id = static_cast<T>((*iter).local);
                });
        }
        delete table;
    }
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
reindex_single(torch::Tensor inputs, torch::Tensor outputs, torch::Tensor count)
{
    using T = int64_t;
    cudaStream_t stream = 0;
    const auto policy = thrust::cuda::par.on(stream);
    thrust::device_vector<T> total_inputs(inputs.size(0));
    thrust::device_vector<T> total_outputs(outputs.size(0));
    thrust::device_vector<T> input_prefix(inputs.size(0));
    const T *ptr;
    size_t bs;
    ptr = count.data_ptr<T>();
    bs = inputs.size(0);
    thrust::copy(ptr, ptr + bs, input_prefix.begin());
    ptr = inputs.data_ptr<T>();
    thrust::copy(ptr, ptr + bs, total_inputs.begin());
    thrust::exclusive_scan(policy, input_prefix.begin(), input_prefix.end(),
                            input_prefix.begin());
    ptr = outputs.data_ptr<T>();
    bs = outputs.size(0);
    thrust::copy(ptr, ptr + bs, total_outputs.begin());

    const size_t m = inputs.size(0);
    using it = thrust::counting_iterator<T>;

    thrust::device_vector<T> subset;
    reindex_kernel(stream, total_inputs, total_outputs, subset);

    int tot = total_outputs.size();
    torch::Tensor out_vertices = torch::empty(subset.size(), inputs.options());
    torch::Tensor row_idx = torch::empty(tot, inputs.options());
    torch::Tensor col_idx = torch::empty(tot, inputs.options());
    {
        thrust::device_vector<T> seq(count.size(0));
        thrust::sequence(policy, seq.begin(), seq.end());

        thrust::for_each(
            policy, it(0), it(m),
            [prefix = thrust::raw_pointer_cast(input_prefix.data()),
                count = count.data_ptr<T>(),
                in = thrust::raw_pointer_cast(seq.data()),
                out = thrust::raw_pointer_cast(
                    row_idx.data_ptr<T>())] __device__(T i) {
                for (int j = 0; j < count[i]; j++) {
                    out[prefix[i] + j] = in[i];
                }
            });
        thrust::copy(subset.begin(), subset.end(), out_vertices.data_ptr<T>());
        thrust::copy(total_outputs.begin(), total_outputs.end(),
                        col_idx.data_ptr<T>());
    }
    return std::make_tuple(out_vertices, row_idx, col_idx);
}

template <typename T>
void replicate_fill(size_t n, const T *counts, const T *values, T *outputs)
{
    for (size_t i = 0; i < n; ++i) {
        const size_t c = counts[i];
        std::fill(outputs, outputs + c, values[i]);
        outputs += c;
    }
}





class TorchQuiver
{
    using torch_quiver_t = quiver<int64_t, CUDA>;
    torch_quiver_t quiver_;
    stream_pool pool_;

  public:
    TorchQuiver(torch_quiver_t quiver, int device = 0, int num_workers = 4)
        : quiver_(std::move(quiver))
    {
        pool_ = stream_pool(num_workers);
    }

    using T = int64_t;
    using W = float;


    void cal_neighbor_prob(int stream_num, const torch::Tensor last_prob,
                           torch::Tensor cur_prob, int k)
    {
        cudaStream_t stream = 0;
        auto p = quiver_.csr();
        const T *indptr = p.first;
        const T *indices = p.second;
        size_t block = (last_prob.size(0) + 127) / 128;
        cal_next<<<block, 128, 0, stream>>>(
            last_prob.data_ptr<float>(), cur_prob.data_ptr<float>(),
            cur_prob.size(0), k, indptr, indices);
    }
    
};

/*

TorchQuiver new_quiver_from_csr_array()
{

    // initialize Quiver instance
    using Q = quiver<int64_t, CUDA>;
    // 在成功获得数据之后，使用这个数据构造 quiver 
    // 由 quiver 构造函数可见，quiver 这个类应该就是用来存储图数据的类
    Q quiver = Q::New(indptr_device_pointer, indices_device_pointer,
                    edge_id_device_pointer, node_count - 1, edge_count);
    return TorchQuiver(std::move(quiver), device);
}*/

TorchQuiver new_quiver_from_edge_index(size_t n,
                                       py::array_t<int64_t> &input_edges,
                                       py::array_t<int64_t> &input_edge_idx,
                                       int device = 0)
{
    cudaSetDevice(device);
    TRACE_SCOPE(__func__);
    using T = typename TorchQuiver::T;
    py::buffer_info edges = input_edges.request();
    py::buffer_info edge_idx = input_edge_idx.request();
    check_eq<int64_t>(edges.ndim, 2);
    check_eq<int64_t>(edges.shape[0], 2);
    const size_t m = edges.shape[1];
    check_eq<int64_t>(edge_idx.ndim, 1);

    bool use_eid = edge_idx.shape[0] == m;

    thrust::device_vector<T> row_idx(m);
    thrust::device_vector<T> col_idx(m);
    {
        const T *p = reinterpret_cast<const T *>(edges.ptr);
        thrust::copy(p, p + m, row_idx.begin());
        thrust::copy(p + m, p + m * 2, col_idx.begin());
    }
    thrust::device_vector<T> edge_idx_;
    if (use_eid) {
        edge_idx_.resize(m);
        const T *p = reinterpret_cast<const T *>(edge_idx.ptr);
        thrust::copy(p, p + m, edge_idx_.begin());
    }
    using Q = quiver<int64_t, CUDA>;
    Q quiver = Q::New(static_cast<T>(n), std::move(row_idx), std::move(col_idx),
                      std::move(edge_idx_));
    return TorchQuiver(std::move(quiver), device);
}
}  // namespace quiver

void register_cuda_quiver_sample(pybind11::module &m)
{
    
    // m.def("device_quiver_from_csr_array", &quiver::new_quiver_from_csr_array);
    m.def("device_quiver_from_edge_index", &quiver::new_quiver_from_edge_index);
    m.def("reindex_single", &quiver::reindex_single);
    py::class_<quiver::TorchQuiver>(m, "Quiver")
        .def("cal_neighbor_prob", &quiver::TorchQuiver::cal_neighbor_prob,
             py::call_guard<py::gil_scoped_release>());
        // .def("reindex_single", &quiver::TorchQuiver::reindex_single,
        //      py::call_guard<py::gil_scoped_release>());
}
