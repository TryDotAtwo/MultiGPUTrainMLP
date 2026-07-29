#pragma once

#include "mgt/status.hpp"

#include <algorithm>
#include <cstdint>

namespace mgt {

struct InputGradGroupingLimits {
    std::uint64_t max_shared_bytes_per_block;
    std::uint64_t max_shared_bytes_per_multiprocessor;
    std::uint32_t target_resident_blocks;
    std::uint32_t max_positions_per_block;
};

struct InputGradGroupingConfig {
    std::uint32_t positions_per_block;
    std::uint64_t dynamic_shared_bytes;
};

inline Status ResolveInputGradGrouping(
    std::uint32_t state_len,
    std::uint32_t requested_positions,
    std::uint64_t shared_bytes_per_position,
    const InputGradGroupingLimits& limits,
    InputGradGroupingConfig* config) {
    if (config == nullptr || state_len == 0 || shared_bytes_per_position == 0 ||
        limits.max_shared_bytes_per_block == 0 ||
        limits.max_shared_bytes_per_multiprocessor == 0 ||
        limits.target_resident_blocks == 0 || limits.max_positions_per_block == 0) {
        return Status::kInvalidConfig;
    }
    if (requested_positions > limits.max_positions_per_block) {
        return Status::kInvalidConfig;
    }

    const std::uint64_t positions_by_block =
        limits.max_shared_bytes_per_block / shared_bytes_per_position;
    if (requested_positions != 0) {
        if (requested_positions > positions_by_block) return Status::kCapacityExceeded;
        config->positions_per_block = requested_positions;
        config->dynamic_shared_bytes = requested_positions * shared_bytes_per_position;
        return Status::kOk;
    }

    std::uint64_t positions_by_residency = 0;
    if (shared_bytes_per_position <=
        limits.max_shared_bytes_per_multiprocessor / limits.target_resident_blocks) {
        positions_by_residency = limits.max_shared_bytes_per_multiprocessor /
                                 limits.target_resident_blocks /
                                 shared_bytes_per_position;
    }
    const std::uint64_t chosen = (std::min)(
        static_cast<std::uint64_t>(state_len),
        (std::min)(
            static_cast<std::uint64_t>(limits.max_positions_per_block),
            (std::min)(positions_by_block, positions_by_residency)));
    if (chosen == 0) return Status::kCapacityExceeded;

    config->positions_per_block = static_cast<std::uint32_t>(chosen);
    config->dynamic_shared_bytes = chosen * shared_bytes_per_position;
    return Status::kOk;
}

}  // namespace mgt
