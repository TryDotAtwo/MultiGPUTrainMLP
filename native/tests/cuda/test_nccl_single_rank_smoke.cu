#include "mgt_cuda/allreduce_nccl.cuh"

#include <cuda_runtime.h>
#include <cmath>
#include <cstdlib>
#include <vector>

namespace {

int Check(cudaError_t status) {
    return status == cudaSuccess ? 0 : 1;
}

}  // namespace

int main() {
    constexpr std::size_t kCount = 8;
    std::vector<float> host(kCount);
    for (std::size_t i = 0; i < kCount; ++i) {
        host[i] = static_cast<float>(i + 1);
    }

    float* device = nullptr;
    if (Check(cudaMalloc(&device, kCount * sizeof(float))) != 0) return EXIT_FAILURE;
    if (Check(cudaMemcpy(device, host.data(), kCount * sizeof(float), cudaMemcpyHostToDevice)) != 0) return EXIT_FAILURE;

    mgt_cuda::NcclSingleRankContext* context = nullptr;
    if (mgt_cuda::CreateNcclSingleRankContext(0, &context) != mgt::Status::kOk) return EXIT_FAILURE;

    const mgt::AllreduceConfig config{1, 0, 1, kCount};
    if (mgt_cuda::NcclAllreduceAverageFloat(config, device, context, 0) != mgt::Status::kOk) return EXIT_FAILURE;
    if (Check(cudaDeviceSynchronize()) != 0) return EXIT_FAILURE;

    std::vector<float> reduced(kCount);
    if (Check(cudaMemcpy(reduced.data(), device, kCount * sizeof(float), cudaMemcpyDeviceToHost)) != 0) return EXIT_FAILURE;
    for (std::size_t i = 0; i < kCount; ++i) {
        if (std::fabs(reduced[i] - host[i]) > 1.0e-6f) return EXIT_FAILURE;
    }

    if (mgt_cuda::DestroyNcclSingleRankContext(context) != mgt::Status::kOk) return EXIT_FAILURE;
    if (Check(cudaFree(device)) != 0) return EXIT_FAILURE;
    return EXIT_SUCCESS;
}