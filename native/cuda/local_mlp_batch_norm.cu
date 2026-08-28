#define MGT_LOCAL_MLP_IMPLEMENTATION
#include "mlp_batch_norm_forward.cu"
#include "mgt_cuda/local_mlp_batch_norm.cuh"

namespace mgt_cuda {

std::uint64_t MlpBatchNormParameterCount(const CudaMlpShape& shape) {
    return ValidateCudaMlpShape(shape) == mgt::Status::kOk
        ? OB(shape) + shape.output_dim : 0;
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

}  // namespace mgt_cuda
