#pragma once

#include "mgt/static_contracts.hpp"
#include <cuda_runtime.h>

namespace mgt_cuda {

mgt::Status LaunchBatchNormForward(
    const float* x, int rows, int cols,
    const float* gamma, const float* beta,
    float* running_mean, float* running_var,
    float momentum, float epsilon,
    float* y, float* mean, float* inv_std,
    float* normalized, cudaStream_t stream);

mgt::Status LaunchBatchNormBackward(
    const float* dy, int rows, int cols,
    const float* gamma, const float* inv_std,
    const float* normalized,
    float* dx, float* dgamma, float* dbeta,
    cudaStream_t stream);

}  // namespace mgt_cuda