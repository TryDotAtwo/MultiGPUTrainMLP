#define MGT_LOCAL_MLP_IMPLEMENTATION
#include "mlp_batch_norm_forward.cu"
#include "mgt_cuda/local_mlp_batch_norm.cuh"

namespace mgt_cuda {

std::uint64_t MlpBatchNormParameterCount(const CudaMlpShape& shape) {
    return ValidateCudaMlpShape(shape) == mgt::Status::kOk
        ? OB(shape) + shape.output_dim : 0;
}

std::uint64_t LocalMlpBatchNormForwardWorkspaceFloats(
    const CudaMlpShape& shape,
    const mgt::BatchNormTrainingPlan& plan,
    std::uint32_t rows) {
    return LocalMlpBatchNormForwardWorkspaceFloatsImpl(shape, plan, rows);
}

mgt::Status LaunchLocalMlpInputHalf(
    const CudaMlpShape& shape, std::uint32_t logical_hd1,
    const __half* weights, const mgt::TrainStateStorage* states,
    std::uint32_t rows, float* output, cudaStream_t stream) {
    if (ValidateCudaMlpShape(shape) != mgt::Status::kOk)
        return mgt::Status::kInvalidConfig;
    return LaunchInputHalfInternal(
        shape, logical_hd1, weights, states, rows, output, stream);
}

mgt::Status LaunchLocalMlpBatchNormTrainStep(
    const CudaMlpShape& shape,
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
    cudaStream_t stream) {
    auto* local_backend = LocalBackendContext();
    return LaunchLocalMlpBatchNormTrainStepImpl(
        shape, logical_hd1, logical_hd2, states, labels, rows, rows, plan,
        forward_workspace_floats, adam, buffers, local_backend, blas, stream);
}

mgt::Status LaunchLocalMlpBatchNormTrainStepFp16(
    const CudaMlpShape& shape, std::uint32_t logical_hd1,
    std::uint32_t logical_hd2, const mgt::TrainStateStorage* states,
    const float* labels, std::uint32_t rows,
    const mgt::BatchNormTrainingPlan& plan,
    std::uint64_t forward_workspace_floats, const AdamWKernelConfig& adam,
    MlpBatchNormStepBuffers buffers, LocalMlpFp16Context* fp16,
    cublasHandle_t blas, cudaStream_t stream) {
    if (!fp16 || fp16->master_weights != buffers.weights || !fp16->weight_mirror)
        return mgt::Status::kInvalidConfig;
    auto* local_backend = LocalBackendContext(fp16);
    return LaunchLocalMlpBatchNormTrainStepImpl(
        shape, logical_hd1, logical_hd2, states, labels, rows, rows, plan,
        forward_workspace_floats, adam, buffers, local_backend, blas, stream);
}

}  // namespace mgt_cuda
