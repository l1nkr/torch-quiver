#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <pybind11/numpy.h>
#include <quiver/common.hpp>
#include <quiver/quiver.cu.hpp>
// #include <quiver/shard_tensor.cu.hpp>
#include <torch/extension.h>

#include <atomic>
#include <iostream>
#include <string>
#include <torch/csrc/utils/python_numbers.h>
#include <unordered_map>
#include <vector>
#include <thrust/binary_search.h>
#include <thrust/device_vector.h>
#include <quiver/stream_pool.hpp>
#include <quiver/cuda_stream.hpp>
#include <quiver/functor.cu.hpp>
namespace quiver{

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

#define CHECK_CPU(x) AT_ASSERTM(!x.device().is_cuda(), #x " must be CPU tensor")
// CSRRowWiseSampleKernel<T, BLOCK_WARPS, TILE_SIZE>
//     <<<grid, block, 0, stream>>>(
//         0, k, inputs_size, gpu_memory_budget_,
//         inputs, indptr_device_map_, cpu_part, gpu_part,
//         output_ptr, output_counts, outputs, output_idx);

// 核心就在于这个 kernel ，只要这个函数能够支持从 dev_indices_ 中读取数据
// 分配内存这个问题就基本解决了，接下来就只需要想办法编译通过就可以了
template <typename T, int BLOCK_WARPS, int TILE_SIZE>
__global__ void CSRRowWiseSampleKernel(
    const uint64_t rand_seed, int num_picks, const int64_t num_rows, int gpu_memory_budget,
    const T * const in_rows, const T * const in_ptr, const T * const cpu_part, const T * const gpu_part,
    T * const out_ptr, T * const out_count_ptr, T * const out, T * const out_idxs)
{
    assert(blockDim.x == WARP_SIZE);
    assert(blockDim.y == BLOCK_WARPS);

    int64_t out_row = blockIdx.x * TILE_SIZE + threadIdx.y;

    const int64_t last_row = min(static_cast<int64_t>(blockIdx.x + 1) * TILE_SIZE, num_rows);

    curandState rng;
    curand_init(rand_seed * gridDim.x + blockIdx.x,
                threadIdx.y * WARP_SIZE + threadIdx.x, 0, &rng);
    while (out_row < last_row) {
        const int64_t row = in_rows[out_row];
        // row：需要采样节点的序号
        const int64_t in_row_start = in_ptr[row];
        // degree?
        const int64_t deg = in_ptr[row + 1] - in_row_start;
        const int64_t out_row_start = out_ptr[out_row];
        if (deg <= num_picks) {
            // 邻居节点个数小于要求的节点数量
            // just copy row
            for (int idx = threadIdx.x; idx < deg; idx += WARP_SIZE) {
                const T in_idx = in_row_start + idx;
                // in_index[in_ptr[in_rows[blockIdx.x * TILE_SIZE + threadIdx.y]] + idx]
                // 依据 thread id 和 memory budget 就可以算出使用哪一个指针
                // in_ptr[in_rows[blockIdx.x * TILE_SIZE + threadIdx.y]] + idx > gpu_memory_budget 
                //      then use cpu
                //      else use gpu
                if (in_idx > gpu_memory_budget) {
                    out[out_row_start + idx] = cpu_part[in_idx - gpu_memory_budget];
                } else {
                    out[out_row_start + idx] = gpu_part[in_idx];
                }
                // out[out_row_start + idx] = in_index[in_idx];
            }
        } else {
            // 邻居节点个数大于要求的节点数量，随机采样
            // generate permutation list via reservoir algorithm
            for (int idx = threadIdx.x; idx < num_picks; idx += WARP_SIZE) {
                out_idxs[out_row_start + idx] = idx;
            }
            __syncwarp();
            // 使用水库算法从邻居节点中采样 num_picks 个节点
            for (int idx = num_picks + threadIdx.x; idx < deg; idx += WARP_SIZE) {
                const int num = curand(&rng) % (idx + 1);
                if (num < num_picks) {
                    // use max so as to achieve the replacement order the serial algorithm would have
                    using Type = unsigned long long int;
                    // out_idxs[out_row_start + num] = max(out_idxs[out_row_start + num], idx)
                    atomicMax(reinterpret_cast<Type *>(out_idxs + out_row_start + num),
                              static_cast<Type>(idx));
                }
            }
            __syncwarp();
            // copy permutation over
            for (int idx = threadIdx.x; idx < num_picks; idx += WARP_SIZE) {
                const T perm_idx = out_idxs[out_row_start + idx] + in_row_start;
                // in_index[out_idxs[out_ptr[blockIdx.x * TILE_SIZE + threadIdx.y] + idx] + in_ptr[row]]
                // if (out_idxs[out_ptr[blockIdx.x * TILE_SIZE + threadIdx.y] + idx] + in_ptr[row] > gpu_memory_budget) 
                //      then use cpu
                //      else use gpu 
                if (perm_idx > gpu_memory_budget) {
                    out[out_row_start + idx] = cpu_part[perm_idx - gpu_memory_budget];
                } else {
                    out[out_row_start + idx] = gpu_part[perm_idx];
                }
                // out[out_row_start + idx] = in_index[perm_idx];
            }
        }
        out_row += BLOCK_WARPS;
    }
}

// 因此 ShardNode 应该具有如下方法：
// device_quiver_from_csr_array, sample_neighbor, reindex_single
class ShardNode
{
public:
    using T = int64_t;
    
    ShardNode(T *row_ptr, T *edge_idx, 
              T node_count, T edge_count, T gpu_memory_budget, T device = 0, T num_workers = 4)
            : indptr_device_map_(row_ptr),
              edge_id_device_map_(edge_idx),
              device_(device),
              node_count_(node_count),
              edge_count_(edge_count),
              gpu_memory_budget_(gpu_memory_budget)
    {
        pool_ = stream_pool(num_workers);
    }
    // static ShardNode New(T *row_idx, T *col_idx, T *edge_idx, T node_count, T edge_count) 
    // {

    // }
    size_t get_tensor_bytes(torch::Tensor tensor)
    {
        int dim = tensor.dim();
        size_t total_bytes = 4;
        for (int index = 0; index < dim; index++) {
            total_bytes *= tensor.sizes()[index];
        }
        return total_bytes;
    }
    void append(torch::Tensor &tensor, int target_device) 
    {
        CHECK_CPU(tensor);
        void *ptr = NULL;
        size_t data_size = get_tensor_bytes(tensor);
        if (target_device >= 0) {
            cudaSetDevice(target_device);
            cudaMalloc(&ptr, data_size);
            cudaMemcpy(ptr, tensor.data_ptr<int64_t>(), data_size,
                        cudaMemcpyHostToDevice);
            cudaSetDevice(device_);
        } else {
            cudaSetDevice(device_);
            quiverRegister(tensor.data_ptr<int64_t>(), data_size,
                            cudaHostRegisterMapped);
            cudaHostGetDevicePointer(&ptr, (void *)tensor.data_ptr<int64_t>(), 0);
        }
        dev_indices_.push_back((T *)ptr);
    }
    std::tuple<torch::Tensor, torch::Tensor>
    sample_neighbor(int stream_num, const torch::Tensor &vertices, int k)
    {
        cudaStream_t stream = 0;
        if (!pool_.empty()) { stream = (pool_)[stream_num]; }
        const auto policy = thrust::cuda::par.on(stream);
        const size_t bs = vertices.size(0);

        // 下面的改动就是最关键的了
        thrust::device_vector<T> inputs;
        thrust::device_vector<T> outputs;
        thrust::device_vector<T> output_counts;

        sample_kernel(stream, vertices, k, inputs, outputs, output_counts);

        torch::Tensor neighbors = torch::empty(outputs.size(), vertices.options());
        torch::Tensor counts = torch::empty(vertices.size(0), vertices.options());

        thrust::copy(outputs.begin(), outputs.end(), neighbors.data_ptr<T>());
        thrust::copy(output_counts.begin(), output_counts.end(), counts.data_ptr<T>());

        return std::make_tuple(neighbors, counts);
    }


    void degree(const cudaStream_t stream, 
                thrust::device_ptr<const T> input_begin,
                thrust::device_ptr<const T> input_end,
                thrust::device_ptr<T> output_begin) const
    {
        thrust::transform(thrust::cuda::par.on(stream), input_begin, input_end, output_begin,
                          get_adj_diff<T>(indptr_device_map_, node_count_, edge_count_));
    }
    // k 采样节点数 （25）
    // inputs 需要进行采样的节点
    // inputs_size = inputs.size()
    // output_ptr 在这个节点之前需要采样多少个节点
    // output_counts 对应 inputs 中每个节点需要采样多少个节点
    // outputs 大小为总采样节点个数的空 vector，返回最终结果，需要扩散到哪些节点
    // output_idx 大小为总采样节点个数的空 vector
    void new_sample(const cudaStream_t stream, int k,
                    T *inputs, int inputs_size, T *output_ptr,
                    T *output_counts, T *outputs, T *output_idx) const
    {
        constexpr int BLOCK_WARPS = 128 / WARP_SIZE;
        constexpr int TILE_SIZE = BLOCK_WARPS * 16;
        const dim3 block(WARP_SIZE, BLOCK_WARPS);
        const dim3 grid((inputs_size + TILE_SIZE - 1) / TILE_SIZE);
        // col_idx_mapped 存在 dev_indices_ 中
        const T *gpu_part;
        const T *cpu_part;
        if (dev_indices_.size() > 1) {
            // budget 适中。两者都有
            gpu_part = dev_indices_[0];
            cpu_part = dev_indices_[1];
        } else if (gpu_memory_budget_ == 0) {
            // budget 为0， gpu 为空
            gpu_part = nullptr;
            cpu_part = dev_indices_[0];
        } else {
            // budget 很大，cpu 为空
            gpu_part = dev_indices_[0];
            cpu_part = nullptr;
        }
        
        std::vector<T *> dev_indices_; // column
        CSRRowWiseSampleKernel<T, BLOCK_WARPS, TILE_SIZE>
            <<<grid, block, 0, stream>>>(
                0, k, inputs_size, gpu_memory_budget_,
                inputs, indptr_device_map_, cpu_part, gpu_part,
                output_ptr, output_counts, outputs, output_idx);
            
    }

    // return: std::tuple<torch::Tensor, torch::Tensor>
    void sample_kernel(const cudaStream_t stream, const torch::Tensor &vertices,
                  int k, thrust::device_vector<T> &inputs,
                  thrust::device_vector<T> &outputs,
                  thrust::device_vector<T> &output_counts) const
    {
        
        const auto policy = thrust::cuda::par.on(stream);
        const size_t bs = vertices.size(0);
        
        TRACE_SCOPE("alloc_1");
        inputs.resize(bs);
        output_counts.resize(bs);

        TRACE_SCOPE("prepare");
        const T *vertices_ptr = vertices.data_ptr<T>();
        thrust::copy(vertices_ptr, vertices_ptr + bs, inputs.begin());
        // 1. 这里并没有 quiver_ 这个对象
        // 2. 即使有，也不能直接拿来用，因为我们的参数已经发生了变化
        // 3. 所以，需要自己重写这个方法
        // 已重写
        degree(stream, inputs.data(), inputs.data() + inputs.size(),
               output_counts.data());
        if (k >= 0) {
            // 这个函数不需要修改，没有涉及之前存入的数据
            thrust::transform(policy, output_counts.begin(), output_counts.end(),
                              output_counts.begin(),
                              cap_by<T>(k)); 
        }
        thrust::device_vector<T> output_ptr;
        output_ptr.resize(bs);
        thrust::exclusive_scan(policy, output_counts.begin(), output_counts.end(),
                              output_ptr.begin());
        T tot = 0;
        tot = thrust::reduce(policy, output_counts.begin(), output_counts.end());

        TRACE_SCOPE("alloc_2");
        outputs.resize(tot);
        thrust::device_vector<T> output_idx;
        output_idx.resize(tot);

        TRACE_SCOPE("sample");

        new_sample(stream, k,
                   thrust::raw_pointer_cast(inputs.data()), inputs.size(),
                   thrust::raw_pointer_cast(output_ptr.data()),
                   thrust::raw_pointer_cast(output_counts.data()),
                   thrust::raw_pointer_cast(outputs.data()),
                   thrust::raw_pointer_cast(output_idx.data()));
        // 1. 这里并没有 quiver_ 这个对象
        // 2. 即使有，也不能直接拿来用，因为我们的参数已经发生了变化
        // 3. 所以，需要自己重写这个方法
        // quiver_.new_sample(
        //     stream, k, thrust::raw_pointer_cast(inputs.data()), inputs.size(), 
        //     thrust::raw_pointer_cast(output_ptr.data()),
        //     thrust::raw_pointer_cast(output_counts.data()),
        //     thrust::raw_pointer_cast(outputs.data()),
        //     thrust::raw_pointer_cast(output_idx.data()));
    }

private:
    std::vector<T *> dev_indices_; // column
    const T *indptr_device_map_; // row
    const T *edge_id_device_map_;
    int device_;
    int node_count_;
    int edge_count_;
    int gpu_memory_budget_;
    // https://zhuanlan.zhihu.com/p/51402722
    stream_pool pool_;

};
// 这个函数不负责 indices(edge) 的内存分配
__host__ ShardNode new_node_from_csr_array(torch::Tensor &input_indptr,
                                    torch::Tensor &input_indices,
                                    torch::Tensor &input_edge_idx,
                                    int gpu_memory_budget,
                                    int device = 0)
{
    cudaSetDevice(device);
    TRACE_SCOPE(__func__);
    using T = typename ShardNode::T;
    // 这个 check_eq 是在哪里定义的
    check_eq<int64_t>(input_indptr.dim(), 1);
    const size_t node_count = input_indptr.size(0);

    check_eq<int64_t>(input_indices.dim(), 1);
    const size_t edge_count = input_indices.size(0);

    

    T *indptr_device_pointer = nullptr;
    T *edge_id_device_pointer = nullptr;

    const T *indptr_original = 
        reinterpret_cast<const T*>(input_indptr.data_ptr<T>());
    T *indptr_copy = nullptr;
    cudaMalloc((void **)&indptr_copy, sizeof(T) * node_count);
    cudaMemcpy((void *)indptr_copy, (void *)indptr_original,
               sizeof(T) * node_count, cudaMemcpyDefault);
    indptr_device_pointer = indptr_copy;
    
    bool use_eid = input_edge_idx.size(0) == edge_count;
    if (use_eid) {
        const T *id_original = 
            reinterpret_cast<const T*>(input_edge_idx.data_ptr<T>());
        T *id_copy;
        cudaMalloc((void **)&id_copy, sizeof(T) * edge_count);
        cudaMemcpy((void *)id_copy, (void *)id_original,
                    sizeof(T) * edge_count, cudaMemcpyDefault);
        edge_id_device_pointer = id_copy;
    }
    // 这样构造很有可能存在问题
    return ShardNode(indptr_device_pointer, edge_id_device_pointer,
                     node_count, edge_count, gpu_memory_budget, 0, 4);
}

}   
// namespace quiver
// // 核心就在于这个 kernel ，只要这个函数能够支持从 dev_indices_ 中读取数据
// // 分配内存这个问题就基本解决了，接下来就只需要想办法编译通过就可以了
// template <typename T, int BLOCK_WARPS, int TILE_SIZE>
// __global__ void CSRRowWiseSampleKernel(
//     const uint64_t rand_seed, int num_picks, const int64_t num_rows, const T *const in_rows, 
//     const T *const in_ptr, const T *const in_index,
//     T *const out_ptr, T *const out_count_ptr, T *const out, T *const out_idxs)
// {
//     assert(blockDim.x == WARP_SIZE);
//     assert(blockDim.y == BLOCK_WARPS);

//     int64_t out_row = blockIdx.x * TILE_SIZE + threadIdx.y;

//     const int64_t last_row =
//         min(static_cast<int64_t>(blockIdx.x + 1) * TILE_SIZE, num_rows);

//     // 这是个什么东西，随机初始化？
//     curandState rng;
//     curand_init(rand_seed * gridDim.x + blockIdx.x,
//                 threadIdx.y * WARP_SIZE + threadIdx.x, 0, &rng);
//     while (out_row < last_row) {
//         const int64_t row = in_rows[out_row];
//         const int64_t in_row_start = in_ptr[row];
//         const int64_t deg = in_ptr[row + 1] - in_row_start;
//         const int64_t out_row_start = out_ptr[out_row];
//         if (deg <= num_picks) {
//             for (int idx = threadIdx.x; idx < deg; idx += WARP_SIZE) {
//                 const T in_idx = in_row_start + idx;
//                 out[out_row_start + idx] = in_index[in_idx];
//             }
//         } else {
//             for (int idx = threadIdx.x; idx < num_picks; idx += WARP_SIZE) {
//                 out_idxs[out_row_start + idx] = idx;
//             }
//             __syncwarp();
//             for (int idx = num_picks + threadIdx.x; idx < deg; idx += WARP_SIZE) {
//                 const int num = curand(&rng) % (idx + 1);
//                 if (num < num_picks) {
//                     using Type = unsigned long long int;
//                     atomicMax(reinterpret_cast<Type *>(out_idxs + out_row_start + num),
//                               static_cast<Type>(idx));
//                 }
//             }
//             __syncwarp();
//             for (int idx = threadIdx.x; idx < num_picks; idx += WARP_SIZE) {
//                 const T perm_idx = out_idxs[out_row_start + idx] + in_row_start;
//                 out[out_row_start + idx] = in_index[perm_idx];
//             }
//         }
//         out_row += BLOCK_WARPS;
//     }
// }
    // ShardNode(T *row_ptr, T *edge_idx, 
    //           T node_count, T edge_count, T gpu_memory_budget, T device = 0, T num_workers = 4)
void register_cuda_quiver_node(pybind11::module &m)
{
    m.def("device_node_from_csr_array", &quiver::new_node_from_csr_array);

    py::class_<quiver::ShardNode>(m, "ShardNode")
    .def(py::init<int64_t*, int64_t*, int64_t, int64_t, int64_t, int64_t, int64_t>())
    .def("append", &quiver::ShardNode::append,
        py::call_guard<py::gil_scoped_release>())
    .def("sample_neighbor", &quiver::ShardNode::sample_neighbor,
        py::call_guard<py::gil_scoped_release>());
    // .def("reindex_single", &quiver::ShardNode::reindex_single,
    //    py::call_guard<py::gil_scoped_release>())
}