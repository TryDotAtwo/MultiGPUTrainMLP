#pragma once

#include "mgt/static_contracts.hpp"
#include <cstdint>

namespace mgt_cuda {

struct RandomWalkKernelConfig {
    std::uint32_t sample_count;
    std::uint32_t k_min;
    std::uint32_t k_max;
};

__host__ mgt::Status ValidateRandomWalkKernelConfig(const RandomWalkKernelConfig& config);

}  // namespace mgt_cuda