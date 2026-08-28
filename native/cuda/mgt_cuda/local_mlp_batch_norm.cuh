#pragma once

#include "mgt_cuda/mlp_batch_norm_forward.cuh"

namespace mgt_cuda {

std::uint64_t MlpBatchNormParameterCount(const CudaMlpShape& physical_shape);

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

}  // namespace mgt_cuda
