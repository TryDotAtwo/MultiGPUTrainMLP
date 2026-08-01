#pragma once

#include "mgt/status.hpp"
#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace mgt_cuda {

struct AdamWKernelConfig {
    std::uint64_t param_count;
    std::uint64_t step;
    float learning_rate;
    float beta1;
    float beta2;
    float eps;
    float weight_decay;
};

__host__ mgt::Status ValidateAdamWKernelConfig(const AdamWKernelConfig& config);
__host__ mgt::Status LaunchAdamWKernel(const AdamWKernelConfig& config,
                                       float* device_weights,
                                       const float* device_grad,
                                       float* device_m,
                                       float* device_v,
                                       cudaStream_t stream);

__host__ mgt::Status LaunchAdamWKernelWithHalfMirror(const AdamWKernelConfig& config,
                                                     float* device_weights,
                                                     __half* device_weights_half,
                                                     const float* device_grad,
                                                     float* device_m,
                                                     float* device_v,
                                                     cudaStream_t stream);
__host__ mgt::Status LaunchFloatToBfloat16Mirror(const float* master,
                                                 __nv_bfloat16* mirror,
                                                 std::uint64_t count,
                                                 cudaStream_t stream);
__host__ mgt::Status LaunchAdamWKernelWithBfloat16Mirror(const AdamWKernelConfig& config,
                                                         float* master,
                                                         __nv_bfloat16* mirror,
                                                         const float* grad,
                                                         float* moment1,
                                                         float* moment2,
                                                         cudaStream_t stream);
}  // namespace mgt_cuda