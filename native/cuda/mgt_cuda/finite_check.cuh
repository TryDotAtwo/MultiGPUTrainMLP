#pragma once
#include "mgt/static_contracts.hpp"
#include <cuda_runtime.h>
#include <cstddef>

namespace mgt_cuda {
mgt::Status LaunchFiniteTrainingCheck(const float* device_loss,
                                      const float* device_grad,
                                      const float* device_weights,
                                      std::size_t param_count,
                                      int* device_nonfinite,
                                      cudaStream_t stream);
}  // namespace mgt_cuda
