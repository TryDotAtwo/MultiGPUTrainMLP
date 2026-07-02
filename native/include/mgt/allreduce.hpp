#pragma once

#include "mgt/status.hpp"
#include <cstddef>
#include <cstdint>

namespace mgt {

struct AllreduceConfig {
    std::uint32_t world_size;
    std::uint32_t global_rank;
    std::uint64_t collective_seq;
    std::size_t element_count;
};

struct AllreduceLogEntry {
    std::uint32_t global_rank;
    std::uint64_t collective_seq;
    std::size_t bytes;
    bool begin;
};

Status ValidateAllreduceConfig(const AllreduceConfig& config);
Status AverageReducedGradients(float* gradients, std::size_t element_count, std::uint32_t world_size);
AllreduceLogEntry MakeAllreduceLogEntry(const AllreduceConfig& config, bool begin);
bool SameCollectiveOrder(const AllreduceLogEntry* entries, std::size_t entry_count, std::uint32_t world_size);

}  // namespace mgt