#pragma once

#include "mgt/a100_bf16_policy.hpp"
#include "mgt/status.hpp"
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace mgt_cuda {
struct NcclRankContext;

struct TiledSyncBatchNormConfig {
    std::uint32_t row_chunk = 0;
    std::uint32_t feature_tile = 0;
    mgt::A100XhatStorage xhat_storage = mgt::A100XhatStorage::kBf16;
};

struct TiledSyncBatchNormWorkspace {
    float* partials = nullptr;
    std::uint64_t partial_count = 0;
    float* reduced = nullptr;
    std::uint64_t reduced_count = 0;
};

std::uint64_t TiledSyncBatchNormPartialFloats(
    std::uint32_t capacity_rows, std::uint32_t logical_features,
    const TiledSyncBatchNormConfig& config);

mgt::Status LaunchTiledSyncBatchNormForward(
    const TiledSyncBatchNormConfig& config, const float* preactivation,
    const __nv_bfloat16* residual, std::uint32_t active_rows,
    std::uint32_t capacity_rows, std::uint32_t global_rows,
    std::uint32_t logical_features, std::uint32_t physical_features,
    const float* gamma, const float* beta, float* running_mean,
    float* running_variance, float momentum, float epsilon,
    float* saved_mean, float* saved_inv_std, void* saved_xhat,
    __nv_bfloat16* activation, std::uint32_t* relu_mask,
    TiledSyncBatchNormWorkspace workspace, NcclRankContext* context,
    cudaStream_t stream);

mgt::Status LaunchTiledSyncBatchNormBackward(
    const TiledSyncBatchNormConfig& config, const float* upstream,
    const std::uint32_t* relu_mask, const void* saved_xhat,
    std::uint32_t active_rows, std::uint32_t capacity_rows,
    std::uint32_t global_rows, std::uint32_t logical_features,
    std::uint32_t physical_features, const float* gamma,
    const float* saved_inv_std, float* dgamma, float* dbeta,
    __nv_bfloat16* dz, float* residual_grad,
    TiledSyncBatchNormWorkspace workspace, NcclRankContext* context,
    cudaStream_t stream);
}  // namespace mgt_cuda
