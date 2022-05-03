#pragma once
#include <quiver/algorithm.cu.hpp>
#include <quiver/common.hpp>
#include <quiver/copy.cu.hpp>
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

#define CHECK_CPU(x) AT_ASSERTM(!x.device().is_cuda(), #x " must be CPU tensor")
class ShardNode
{
private:
    std::vector<int *> dev_ptrs_;
    int device_;
    stream_pool pool_;

public:
    ShardNode(int device = 0, int num_workers = 4) : device_(device) 
    {
        pool_ = stream_pool(num_workers);
    }
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
            cudaSetDevice(target_device)
            cudaMalloc(&ptr, data_size);
            cudaMemcpy(ptr, tensor.data_ptr<int>(), data_size,
                        cudaMemcpyHostToDevice);
            cudaSetDevice(device_);
        } else {
            cudaSetDevice(device_);
            quiverRegister(tensor.data_ptr<int>(), data_size,
                            cudaHostRegisterMapped);
            cudaHostGetDevicePointer(&ptr, (void *)tensor.data_ptr<int>(), 0);
        }
        dev_ptrs_.push_back(ptr);
    }
    std::tuple<torch::Tensor, torch::Tensor>
    sample_neighbor(int stream_num, const torch::Tensor &vertices, int k)
    {
        cudaStream_t stream = 0;
        if (!pool_.empty(0)) { stream = (pool_)[stream_num]; }
        const auto

    }
}
