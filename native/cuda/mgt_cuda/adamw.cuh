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
    // Optional logical-to-physical map for a structurally sparse input-table
    // prefix. param_count is then the launched live count. The first
    // sparse_active_bin_count * sparse_row_width logical elements map through
    // sparse_active_bins; the remaining logical elements map densely after
    // sparse_full_prefix_count. The caller owns a device-readable, unique,
    // in-range bin map and guarantees every skipped grad/m/v is persistent +0
    // with an already coherent low-precision mirror. Weight decay must be zero.
    // Zero/default fields retain dense indexing.
    const std::uint16_t* sparse_active_bins = nullptr;
    std::uint64_t sparse_full_prefix_count = 0;
    std::uint32_t sparse_row_width = 0;
    std::uint32_t sparse_active_bin_count = 0;
};

__host__ mgt::Status ValidateAdamWKernelConfig(const AdamWKernelConfig& config);
__host__ mgt::Status QueryAdamWPhysicalParameterCount(
    const AdamWKernelConfig& config, std::uint64_t* count);
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
