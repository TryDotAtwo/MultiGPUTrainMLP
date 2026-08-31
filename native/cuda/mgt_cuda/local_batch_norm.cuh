#pragma once

#include "mgt/status.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace mgt_cuda {

// Plain BN by default. Residual/half output require relu=true. The FP32
// activation remains authoritative; half_output is only a GEMM operand mirror.
struct LocalBatchNormForwardEpilogue {
    bool relu = false;
    const float* residual = nullptr;
    __half* half_output = nullptr;
};

// Apply supplied full-batch statistics without updating running state. x==y is
// supported; partial x/y overlap is not. normalized and optional residual must
// be separate from x/y and each other. half_output must not overlap float data.
// Padding gets normalized=0 and affine=0, then the same residual/ReLU epilogue.
mgt::Status LaunchLocalStridedBatchNormApply(
    const float* x, int rows, int cols, int row_stride,
    const float* gamma, const float* beta, const float* mean, const float* inv_std,
    float* y, float* normalized,
    LocalBatchNormForwardEpilogue epilogue, cudaStream_t stream);

// Same epilogue contract as Apply; half_output must also be disjoint from
// running_mean, running_var, and stats_workspace. Invalid epilogues are rejected
// before statistics or running-state writes are enqueued.
mgt::Status LaunchLocalStridedBatchNormForward(
    const float* x, int rows, int cols, int row_stride,
    const float* gamma, const float* beta,
    float* running_mean, float* running_var,
    float momentum, float epsilon,
    float* y, float* mean, float* inv_std, float* normalized,
    float* stats_workspace, cudaStream_t stream,
    LocalBatchNormForwardEpilogue epilogue = {});

// FP32 dx remains authoritative. The optional half output is only a mirror
// for a subsequent GEMM, not a lower-precision BN or gradient accumulator.
struct LocalBatchNormBackwardEpilogue {
    __half* half_output = nullptr;
};

// Apply supplied full-batch dgamma/dbeta without updating them. Exact dy==dx
// is supported; partial overlap is rejected. dx must not overlap normalized
// or feature inputs. half_output must be disjoint from all float tensors.
mgt::Status LaunchLocalStridedBatchNormBackwardApply(
    const float* dy, int rows, int cols, int row_stride,
    const float* gamma, const float* inv_std, const float* normalized,
    const float* dgamma, const float* dbeta, float* dx,
    LocalBatchNormBackwardEpilogue epilogue, cudaStream_t stream);

// Validate the optional mirror (including statistics/output-gradient ranges)
// before any work is enqueued. Preserve the existing reduction and copy order.
mgt::Status LaunchLocalStridedBatchNormBackward(
    const float* dy, int rows, int cols, int row_stride,
    const float* gamma, const float* inv_std, const float* normalized,
    float* dx, float* dgamma, float* dbeta,
    float* stats_workspace, cudaStream_t stream,
    LocalBatchNormBackwardEpilogue epilogue = {});

}  // namespace mgt_cuda
