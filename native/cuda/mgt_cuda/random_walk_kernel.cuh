#pragma once

#include "mgt/static_contracts.hpp"
#include <cstdint>
#include <cuda_runtime.h>

namespace mgt_cuda {

struct RandomWalkKernelConfig {
    std::uint32_t sample_count = 0;
    std::uint32_t k_min = 0;
    std::uint32_t k_max = 0;
    std::uint32_t move_count = mgt::kMoveCount;
    std::uint32_t state_len = mgt::kStateLen;
    std::uint32_t state_storage_len = mgt::kStateStorageLen;
};

__host__ mgt::Status ValidateRandomWalkKernelConfig(const RandomWalkKernelConfig& config);
__host__ mgt::Status LaunchRandomWalkKernel(const RandomWalkKernelConfig& config,
                                            std::uint64_t base_seed,
                                            std::uint64_t epoch,
                                            std::uint64_t step,
                                            std::uint32_t global_rank,
                                            const mgt::TrainStateStorage* device_moves,
                                            const mgt::TrainStateStorage* device_target,
                                            mgt::TrainStateStorage* device_states,
                                            float* device_labels,
                                            mgt::WalkMeta* device_meta,
                                            cudaStream_t stream);

}  // namespace mgt_cuda