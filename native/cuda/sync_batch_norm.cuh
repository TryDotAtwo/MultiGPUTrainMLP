#pragma once
#include "mgt/static_contracts.hpp"
#include <cuda_runtime.h>
#include <nccl.h>
namespace mgt_cuda {
struct NcclRankContext;
mgt::Status LaunchSyncBatchNormForward(const float* x, int local_rows, int global_rows, int cols, const float* gamma, const float* beta, float* running_mean, float* running_var, float momentum, float epsilon, float* y, float* mean, float* inv_std, float* normalized, float* stats_workspace, ncclComm_t comm, cudaStream_t stream);
mgt::Status LaunchSyncBatchNormBackward(const float* dy, int local_rows, int global_rows, int cols, const float* gamma, const float* inv_std, const float* normalized, float* dx, float* dgamma, float* dbeta, float* stats_workspace, ncclComm_t comm, cudaStream_t stream);
mgt::Status LaunchSyncBatchNormForward(const float* x, int local_rows, int global_rows, int cols, const float* gamma, const float* beta, float* running_mean, float* running_var, float momentum, float epsilon, float* y, float* mean, float* inv_std, float* normalized, float* stats_workspace, NcclRankContext* context, cudaStream_t stream);
mgt::Status LaunchSyncBatchNormBackward(const float* dy, int local_rows, int global_rows, int cols, const float* gamma, const float* inv_std, const float* normalized, float* dx, float* dgamma, float* dbeta, float* stats_workspace, NcclRankContext* context, cudaStream_t stream);
mgt::Status LaunchStridedSyncBatchNormForward(const float* x, int local_rows, int global_rows, int cols, int row_stride, const float* gamma, const float* beta, float* running_mean, float* running_var, float momentum, float epsilon, float* y, float* mean, float* inv_std, float* normalized, float* stats_workspace, NcclRankContext* context, cudaStream_t stream);
mgt::Status LaunchStridedSyncBatchNormBackward(const float* dy, int local_rows, int global_rows, int cols, int row_stride, const float* gamma, const float* inv_std, const float* normalized, float* dx, float* dgamma, float* dbeta, float* stats_workspace, NcclRankContext* context, cudaStream_t stream);
}