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
    // Optional logical-feature bias before BN. Both statistics and apply use
    // the same rounded FP32 x+bias value; no materialized biased matrix.
    const float* input_bias = nullptr;
};

// Apply supplied full-batch statistics without updating running state. x==y is
// supported; partial x/y overlap is not. normalized and optional residual must
// be separate from x/y and each other. half_output must not overlap float data.
// Padding gets normalized=0 and affine=0, then the same residual/ReLU epilogue.
// input_bias has cols elements and must not overlap writable outputs. It may
// alias read-only inputs; it is not the post-normalization affine beta.
mgt::Status LaunchLocalStridedBatchNormApply(
    const float* x, int rows, int cols, int row_stride,
    const float* gamma, const float* beta, const float* mean, const float* inv_std,
    float* y, float* normalized,
    LocalBatchNormForwardEpilogue epilogue, cudaStream_t stream);

// Same epilogue contract as Apply; half_output must also be disjoint from
// running_mean, running_var, and stats_workspace. Invalid epilogues are rejected
// before statistics or running-state writes are enqueued.
// input_bias must also be disjoint from running state, mean/inv_std and workspace.
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
// A nonnull activated input applies the old activated>0 ? dy : +0 select in
// both partial statistics and apply. Use the original FP32 post-ReLU values.
// residual_grad requires activated and receives incoming masked dy (not dx),
// including all physical padding lanes. BN dx/half padding remains +0.
struct LocalBatchNormBackwardEpilogue {
    __half* half_output = nullptr;
    const float* activated = nullptr;
    float* residual_grad = nullptr;
};

// Apply supplied full-batch dgamma/dbeta without updating them. Exact dy==dx
// is supported; partial overlap is rejected. dx must not overlap normalized
// or feature inputs. half_output must be disjoint from all float tensors.
// activated may alias read-only dy/normalized, but not dx. residual_grad must
// be disjoint from all inputs/outputs, including the optional half mirror.
mgt::Status LaunchLocalStridedBatchNormBackwardApply(
    const float* dy, int rows, int cols, int row_stride,
    const float* gamma, const float* inv_std, const float* normalized,
    const float* dgamma, const float* dbeta, float* dx,
    LocalBatchNormBackwardEpilogue epilogue, cudaStream_t stream);

// Validate the optional mirror (including statistics/output-gradient ranges)
// before any work is enqueued. Reductions remain observable in stats_workspace.
// The apply kernel publishes disjoint outputs; legacy aliases retain copies.
// activated/residual_grad must also be disjoint from writable dgamma/dbeta and
// stats_workspace; the mask is consumed before the in-place dx write.
mgt::Status LaunchLocalStridedBatchNormBackward(
    const float* dy, int rows, int cols, int row_stride,
    const float* gamma, const float* inv_std, const float* normalized,
    float* dx, float* dgamma, float* dbeta,
    float* stats_workspace, cudaStream_t stream,
    LocalBatchNormBackwardEpilogue epilogue = {});

}  // namespace mgt_cuda
