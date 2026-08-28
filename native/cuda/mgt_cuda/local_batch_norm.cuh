#pragma once

#include "mgt/status.hpp"

#include <cuda_runtime.h>

namespace mgt_cuda {

mgt::Status LaunchLocalStridedBatchNormForward(
    const float* x, int rows, int cols, int row_stride,
    const float* gamma, const float* beta,
    float* running_mean, float* running_var,
    float momentum, float epsilon,
    float* y, float* mean, float* inv_std, float* normalized,
    float* stats_workspace, cudaStream_t stream);

mgt::Status LaunchLocalStridedBatchNormBackward(
    const float* dy, int rows, int cols, int row_stride,
    const float* gamma, const float* inv_std, const float* normalized,
    float* dx, float* dgamma, float* dbeta,
    float* stats_workspace, cudaStream_t stream);

}  // namespace mgt_cuda
