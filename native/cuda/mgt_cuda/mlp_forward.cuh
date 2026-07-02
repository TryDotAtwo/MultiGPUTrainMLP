#pragma once

#include "mgt/static_contracts.hpp"
#include <cstdint>
#include <cuda_runtime.h>

namespace mgt_cuda {

struct CudaMlpShape {
    std::uint32_t state_len;
    std::uint32_t state_value_pad;
    std::uint32_t hd1;
    std::uint32_t hd2;
};

__host__ mgt::Status ValidateCudaMlpShape(const CudaMlpShape& shape);

__host__ mgt::Status LaunchMlpForwardKernel(const CudaMlpShape& shape,
                                            const float* device_weights,
                                            const mgt::TrainState80* device_states,
                                            std::uint32_t sample_count,
                                            float* device_outputs,
                                            cudaStream_t stream);

}  // namespace mgt_cuda