#include "mgt_cuda/finite_check.cuh"
#include <cmath>

namespace mgt_cuda {
namespace {
__global__ void FiniteTrainingCheckKernel(const float* loss,
                                          const float* grad,
                                          const float* weights,
                                          std::size_t param_count,
                                          int* nonfinite) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i == 0 && !isfinite(loss[0])) atomicExch(nonfinite, 1);
    if (i < param_count && (!isfinite(grad[i]) || !isfinite(weights[i]))) {
        atomicExch(nonfinite, 1);
    }
}
}  // namespace

mgt::Status LaunchFiniteTrainingCheck(const float* device_loss,
                                      const float* device_grad,
                                      const float* device_weights,
                                      std::size_t param_count,
                                      int* device_nonfinite,
                                      cudaStream_t stream) {
    if (device_loss == nullptr || device_grad == nullptr || device_weights == nullptr ||
        device_nonfinite == nullptr || param_count == 0) return mgt::Status::kInvalidConfig;
    if (cudaMemsetAsync(device_nonfinite, 0, sizeof(int), stream) != cudaSuccess) {
        return mgt::Status::kCudaFailure;
    }
    constexpr unsigned int kThreads = 256;
    const auto blocks = static_cast<unsigned int>((param_count + kThreads - 1) / kThreads);
    FiniteTrainingCheckKernel<<<blocks, kThreads, 0, stream>>>(
        device_loss, device_grad, device_weights, param_count, device_nonfinite);
    return cudaGetLastError() == cudaSuccess ? mgt::Status::kOk : mgt::Status::kCudaFailure;
}
}  // namespace mgt_cuda
