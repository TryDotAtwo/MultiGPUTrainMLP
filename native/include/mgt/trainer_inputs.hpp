#pragma once

#include "mgt/puzzle_io.hpp"
#include "mgt/train_plan.hpp"

#include <filesystem>
#include <string>

namespace mgt {

struct TrainerPuzzleInputs {
    std::filesystem::path group_json;
    std::filesystem::path target_bin;
    bool synthetic_benchmark = false;
};

Status LoadTrainerPuzzle(const TrainerPuzzleInputs& inputs,
                         const TrainConfig& config,
                         PuzzleDefinition* puzzle,
                         std::string* fingerprint);

}  // namespace mgt
