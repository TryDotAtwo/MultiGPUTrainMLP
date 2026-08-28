#pragma once

#include "mgt/puzzle_io.hpp"
#include <cstdint>

namespace mgt {

struct WalkRequest {
    std::uint64_t base_seed;
    std::uint64_t epoch;
    std::uint64_t step;
    std::uint32_t global_rank;
    std::uint32_t k_min;
    std::uint32_t k_max;
    std::uint32_t sample_count;
    std::uint64_t epoch_sample_offset = 0;
    bool original_p888_schedule = false;
};

std::uint32_t OriginalP888SourceIdAtPosition(
    std::uint64_t base_seed,
    std::uint64_t semantic_epoch,
    std::uint64_t epoch_position);

Status GenerateRandomWalksCpu(const PuzzleDefinition& puzzle,
                              const WalkRequest& request,
                              TrainStateStorage* states,
                              float* labels,
                              WalkMeta* meta);

}  // namespace mgt
