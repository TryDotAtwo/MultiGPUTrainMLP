#pragma once

#include "mgt/static_contracts.hpp"
#include <array>
#include <filesystem>

namespace mgt {

struct PuzzleDefinition {
    std::array<TrainStateStorage, kMoveCount> moves;
    TrainStateStorage target;
};

Status LoadPuzzleDefinition(const std::filesystem::path& group_json,
                            const std::filesystem::path& target_bin,
                            PuzzleDefinition* out);
bool HasNonIdentityMove(const PuzzleDefinition& puzzle);
bool HasCanonicalInverseMovePairs(const PuzzleDefinition& puzzle);

}  // namespace mgt
