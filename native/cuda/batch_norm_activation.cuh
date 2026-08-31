#pragma once
#include "mgt/status.hpp"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstddef>
namespace mgt_cuda {
mgt::Status LaunchBatchNormReluForward(float* output, std::size_t count, cudaStream_t stream);
mgt::Status LaunchBatchNormReluForwardHalfMirror(float* output, __half* half_mirror, std::size_t count, cudaStream_t stream);
mgt::Status LaunchBatchNormReluBackward(const float* activated, const float* grad_output, float* grad_batch_norm, std::size_t count, cudaStream_t stream);
mgt::Status LaunchBatchNormResidualReluForward(const float* residual, float* output, std::size_t count, cudaStream_t stream);
mgt::Status LaunchBatchNormResidualReluForwardHalfMirror(const float* residual, float* output, __half* half_mirror, std::size_t count, cudaStream_t stream);
mgt::Status LaunchBatchNormResidualReluBackward(const float* activated, const float* grad_output, float* grad_batch_norm, float* grad_residual, std::size_t count, cudaStream_t stream);
}
