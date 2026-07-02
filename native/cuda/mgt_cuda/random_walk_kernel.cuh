#pragma once

#include "mgt/static_contracts.hpp"
#include <cstdint>
#include <cuda_runtime.h>

namespace mgt_cuda {

struct RandomWalkKernelConfig {
    std::uint32_t sample_count;
    std::uint32_t k_min;
    std::uint32_t k_max;
};

__host__ mgt::Status ValidateRandomWalkKernelConfig(const RandomWalkKernelConfig& config);
__host__ mgt::Status LaunchRandomWalkKernel(const RandomWalkKernelConfig& config,
                                            std::uint64_t base_seed,
                                            std::uint64_t epoch,
                                            std::uint64_t step,
                                            std::uint32_t global_rank,
                                            const mgt::TrainState80* device_moves,
                                            const mgt::TrainState80* device_target,
                                            mgt::TrainState80* device_states,
                                            float* device_labels,
                                            mgt::WalkMeta* device_meta,
                                            cudaStream_t stream);

}  // namespace mgt_cuda