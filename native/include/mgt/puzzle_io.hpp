#pragma once

#include "mgt/static_contracts.hpp"
#include <array>
#include <filesystem>

namespace mgt {

struct PuzzleDefinition {
    std::array<TrainState80, kMoveCount> moves;
    TrainState80 target;
};

Status LoadPuzzleDefinition(const std::filesystem::path& group_json,
                            const std::filesystem::path& target_bin,
                            PuzzleDefinition* out);

}  // namespace mgt