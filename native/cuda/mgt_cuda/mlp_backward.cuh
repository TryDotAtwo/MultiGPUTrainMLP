#pragma once

#include "mgt_cuda/mlp_forward.cuh"

namespace mgt_cuda {

__host__ mgt::Status LaunchMlpLossGradKernel(const CudaMlpShape& shape,
                                             const float* device_weights,
                                             const mgt::TrainState80* device_states,
                                             const float* device_labels,
                                             std::uint32_t sample_count,
                                             float* device_loss,
                                             float* device_grad,
                                             cudaStream_t stream);

}  // namespace mgt_cuda