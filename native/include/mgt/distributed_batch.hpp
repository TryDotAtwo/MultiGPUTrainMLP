#pragma once

#include "mgt/status.hpp"

#include <cstdint>

namespace mgt {

struct DistributedBatchSlice {
    std::uint32_t active_rows = 0;
    std::uint64_t global_offset = 0;
};

Status PartitionGlobalBatch(
    std::uint32_t global_rows,
    std::uint32_t world,
    std::uint32_t rank,
    DistributedBatchSlice* out);

}  // namespace mgt
