#include "mgt/trainer_inputs.hpp"

#include "mgt/training_artifacts.hpp"

#include <fstream>
#include <iomanip>
#include <iterator>
#include <sstream>
#include <vector>

namespace mgt {
namespace {

bool SupportedPuzzleConfig(const TrainConfig& config) {
    return config.puzzle.group_id == 888 &&
           config.puzzle.raw_state_dim == kStateLen &&
           config.puzzle.state_value_count > 0 &&
           config.puzzle.state_value_count <= kStateValuePad &&
           config.puzzle.move_count == kMoveCount &&
           config.puzzle.state_alignment == kStateAlignment;
}

Status HashFile(const std::filesystem::path& path, std::uint64_t* hash) {
    if (hash == nullptr) return Status::kInvalidConfig;
    std::ifstream in(path, std::ios::binary);
    if (!in) return Status::kIoFailure;
    const std::vector<unsigned char> bytes(
        (std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    if (in.bad()) return Status::kIoFailure;
    *hash = Fnv1a64(bytes.data(), bytes.size());
    return Status::kOk;
}

void BuildSyntheticPuzzle(const TrainConfig& config, PuzzleDefinition* puzzle) {
    const std::uint32_t state_len = config.puzzle.raw_state_dim;
    for (std::uint32_t move = 0; move < kMoveCount; ++move) {
        for (std::uint32_t i = 0; i < state_len; ++i) {
            puzzle->moves[move].v[i] =
                static_cast<StateValue>((i + move + 1) % state_len);
        }
        for (std::uint32_t i = state_len; i < kStateStorageLen; ++i) {
            puzzle->moves[move].v[i] = 0;
        }
    }
    for (std::uint32_t i = 0; i < state_len; ++i) {
        puzzle->target.v[i] = static_cast<StateValue>(i);
    }
    for (std::uint32_t i = state_len; i < kStateStorageLen; ++i) {
        puzzle->target.v[i] = 0;
    }
}

std::string Fingerprint(std::string_view mode,
                        const TrainConfig& config,
                        std::uint64_t group_hash,
                        std::uint64_t target_hash) {
    std::ostringstream out;
    out << mode << ':' << std::hex << std::setfill('0')
        << std::setw(16) << group_hash << ':'
        << std::setw(16) << target_hash << std::dec
        << ":g=" << config.puzzle.group_id
        << ":t=" << config.puzzle.target_id
        << ":s=" << config.puzzle.raw_state_dim
        << ":v=" << config.puzzle.state_value_count
        << ":m=" << config.puzzle.move_count;
    return out.str();
}

}  // namespace

Status LoadTrainerPuzzle(const TrainerPuzzleInputs& inputs,
                         const TrainConfig& config,
                         PuzzleDefinition* puzzle,
                         std::string* fingerprint) {
    if (puzzle == nullptr || fingerprint == nullptr ||
        !SupportedPuzzleConfig(config)) {
        return Status::kInvalidConfig;
    }
    const bool has_group = !inputs.group_json.empty();
    const bool has_target = !inputs.target_bin.empty();
    if (inputs.synthetic_benchmark) {
        if (has_group || has_target) return Status::kInvalidConfig;
        BuildSyntheticPuzzle(config, puzzle);
        *fingerprint = Fingerprint("synthetic", config, 0, 0);
        return Status::kOk;
    }
    if (!has_group || !has_target) return Status::kInvalidConfig;

    const Status load_status =
        LoadPuzzleDefinition(inputs.group_json, inputs.target_bin, puzzle);
    if (load_status != Status::kOk) return load_status;
    for (std::uint32_t i = 0; i < config.puzzle.raw_state_dim; ++i) {
        if (puzzle->target.v[i] >= config.puzzle.state_value_count) {
            return Status::kInvalidPuzzle;
        }
    }
    std::uint64_t group_hash = 0;
    std::uint64_t target_hash = 0;
    const Status group_status = HashFile(inputs.group_json, &group_hash);
    if (group_status != Status::kOk) return group_status;
    const Status target_status = HashFile(inputs.target_bin, &target_hash);
    if (target_status != Status::kOk) return target_status;
    *fingerprint = Fingerprint("real", config, group_hash, target_hash);
    return Status::kOk;
}

}  // namespace mgt
