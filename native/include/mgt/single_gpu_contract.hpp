#pragma once

#include "mgt/status.hpp"

#include <cstdint>

namespace mgt {

struct SingleGpuModelContract {
    std::uint32_t schema_version = 0;
    std::uint32_t state_len = 0;
    std::uint32_t state_value_count = 0;
    std::uint32_t input_features = 0;
    std::uint32_t logical_hd1 = 0;
    std::uint32_t physical_hd1 = 0;
    std::uint32_t logical_hd2 = 0;
    std::uint32_t physical_hd2 = 0;
    std::uint32_t residual_blocks = 0;
    std::uint32_t batch_norm_sites = 0;
    std::uint32_t output_dim = 0;
};

SingleGpuModelContract OriginalP888SingleGpuContract();
Status ValidateSingleGpuModelContract(const SingleGpuModelContract& contract);

}  // namespace mgt
