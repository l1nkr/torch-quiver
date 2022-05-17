#pragma once
#include <vector>
#include <cuda_runtime.h>
#include <iostream>

namespace quiver
{
class cuda_stream
{
    cudaStream_t stream_;

  public:
    //   TODO: check errors

    cuda_stream() { cudaStreamCreate(&stream_); }

    ~cuda_stream() { cudaStreamDestroy(stream_); }

    operator cudaStream_t() const { return stream_; }
};

class stream_pool
{
    std::vector<cuda_stream> streams_;

  public:
    stream_pool() {}
    stream_pool(int size) : streams_(size) {}
    cudaStream_t operator[](int i) const
    {
        if (i < 0 || i >= streams_.size()) { return 0; }
        return streams_[i];
    }
    bool empty() const { return streams_.empty(); }
};
}  // namespace quiver
