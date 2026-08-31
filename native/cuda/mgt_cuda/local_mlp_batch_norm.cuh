#pragma once

#include "mgt_cuda/mlp_batch_norm_forward.cuh"
#include "mgt_cuda/fp16_linear_train_ops.cuh"

namespace mgt_cuda {

std::uint64_t MlpBatchNormParameterCount(const CudaMlpShape& physical_shape);
std::uint64_t LocalMlpBatchNormForwardWorkspaceFloats(
    const CudaMlpShape& physical_shape,
    const mgt::BatchNormTrainingPlan& plan,
    std::uint32_t rows);

mgt::Status LaunchLocalMlpInputHalf(
    const CudaMlpShape& physical_shape,
    std::uint32_t logical_hd1,
    const __half* weights,
    const mgt::TrainStateStorage* states,
    std::uint32_t rows,
    float* output,
    cudaStream_t stream);

mgt::Status LaunchLocalMlpBatchNormTrainStep(
    const CudaMlpShape& physical_shape,
    std::uint32_t logical_hd1,
    std::uint32_t logical_hd2,
    const mgt::TrainStateStorage* states,
    const float* labels,
    std::uint32_t rows,
    const mgt::BatchNormTrainingPlan& plan,
    std::uint64_t forward_workspace_floats,
    const AdamWKernelConfig& adam,
    MlpBatchNormStepBuffers buffers,
    cublasHandle_t blas,
    cudaStream_t stream);

// operand_b is a step-local gradient mirror of rows*max(hd2,output_dim) halfs.
// Its live range must be disjoint from model, inputs, activations and gradients;
// unused capacity suffixes may share allocation storage. Cache tags are cleared
// on entry and every return, including validation failures. Device storage still
// requires same-stream ordering (or an explicit completion dependency) for reuse.
mgt::Status LaunchLocalMlpBatchNormTrainStepFp16(
    const CudaMlpShape& physical_shape,
    std::uint32_t logical_hd1,
    std::uint32_t logical_hd2,
    const mgt::TrainStateStorage* states,
    const float* labels,
    std::uint32_t rows,
    const mgt::BatchNormTrainingPlan& plan,
    std::uint64_t forward_workspace_floats,
    const AdamWKernelConfig& adam,
    MlpBatchNormStepBuffers buffers,
    LocalMlpFp16Context* fp16,
    cublasHandle_t blas,
    cudaStream_t stream);

}  // namespace mgt_cuda
