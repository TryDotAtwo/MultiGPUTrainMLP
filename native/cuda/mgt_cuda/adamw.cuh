#pragma once

#include "mgt/status.hpp"
#include <cmath>
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
    // Optional dense subrange. Logical element zero maps to this physical
    // offset. This is mutually exclusive with the sparse map above.
    std::uint64_t dense_physical_offset = 0;
};

namespace detail {

__device__ __forceinline__ std::uint64_t AdamWPhysicalIndex(
    const AdamWKernelConfig& config, std::uint64_t logical) {
    if (!config.sparse_active_bins)
        return config.dense_physical_offset + logical;
    const std::uint64_t active_elements =
        static_cast<std::uint64_t>(config.sparse_active_bin_count) *
        config.sparse_row_width;
    if (logical >= active_elements)
        return config.sparse_full_prefix_count + logical - active_elements;
    const unsigned active_index =
        static_cast<unsigned>(logical / config.sparse_row_width);
    const unsigned column = static_cast<unsigned>(
        logical - static_cast<std::uint64_t>(active_index) *
                      config.sparse_row_width);
    return static_cast<std::uint64_t>(
               config.sparse_active_bins[active_index]) *
               config.sparse_row_width +
           column;
}

__device__ __forceinline__ float AdamWUpdateOneWithBias(
    AdamWKernelConfig config, float bias1, float bias2, float weight,
    float grad_value, float* m_value, float* v_value) {
    const float next_m =
        config.beta1 * *m_value + (1.0f - config.beta1) * grad_value;
    const float next_v = config.beta2 * *v_value +
        (1.0f - config.beta2) * grad_value * grad_value;
    *m_value = next_m;
    *v_value = next_v;
    const float m_hat = next_m / bias1;
    const float v_hat = next_v / bias2;
    return weight - config.learning_rate *
        (m_hat / (sqrtf(v_hat) + config.eps) +
         config.weight_decay * weight);
}

}  // namespace detail

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
