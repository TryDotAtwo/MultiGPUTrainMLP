#pragma once

#include "sync_batch_norm.cuh"

#include <cstdlib>
#include <cstring>

namespace mgt_cuda {

inline bool UseFusedBatchNormEpilogue() {
    static const bool enabled = [] {
        const char* value = std::getenv("MGT_BN_FUSED_EPILOGUE");
        return value != nullptr && std::strcmp(value, "1") == 0;
    }();
    return enabled;
}

inline mgt::Status LaunchSelectedStridedSyncBatchNormForward(
    const float* x, int local_rows, int global_rows, int cols, int row_stride,
    const float* gamma, const float* beta, float* running_mean, float* running_var,
    float momentum, float epsilon, float* y, float* mean, float* inv_std,
    float* normalized, float* stats_workspace, NcclRankContext* context,
    cudaStream_t stream) {
    if (UseFusedBatchNormEpilogue()) {
        return LaunchStridedSyncBatchNormForwardFused(
            x, local_rows, global_rows, cols, row_stride, gamma, beta,
            running_mean, running_var, momentum, epsilon, y, mean, inv_std,
            normalized, stats_workspace, context, stream);
    }
    return LaunchStridedSyncBatchNormForward(
        x, local_rows, global_rows, cols, row_stride, gamma, beta, running_mean,
        running_var, momentum, epsilon, y, mean, inv_std, normalized,
        stats_workspace, context, stream);
}

inline mgt::Status LaunchSelectedStridedSyncBatchNormBackward(
    const float* dy, int local_rows, int global_rows, int cols, int row_stride,
    const float* gamma, const float* inv_std, const float* normalized, float* dx,
    float* dgamma, float* dbeta, float* stats_workspace,
    NcclRankContext* context, cudaStream_t stream) {
    if (UseFusedBatchNormEpilogue()) {
        return LaunchStridedSyncBatchNormBackwardFused(
            dy, local_rows, global_rows, cols, row_stride, gamma, inv_std,
            normalized, dx, dgamma, dbeta, stats_workspace, context, stream);
    }
    return LaunchStridedSyncBatchNormBackward(
        dy, local_rows, global_rows, cols, row_stride, gamma, inv_std,
        normalized, dx, dgamma, dbeta, stats_workspace, context, stream);
}

}  // namespace mgt_cuda
