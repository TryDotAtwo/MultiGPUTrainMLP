#include "mgt/trainer_inputs.hpp"

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

int main() {
    const std::filesystem::path fixtures = "native/tests/fixtures";
    mgt::TrainConfig config = mgt::MakeArchiveP888TrainConfig();
    mgt::PuzzleDefinition puzzle{};
    std::string fingerprint;

    mgt::TrainerPuzzleInputs missing{};
    if (mgt::LoadTrainerPuzzle(missing, config, &puzzle, &fingerprint) !=
        mgt::Status::kInvalidConfig) {
        return EXIT_FAILURE;
    }
    missing.group_json = fixtures / "p888.json";
    if (mgt::LoadTrainerPuzzle(missing, config, &puzzle, &fingerprint) !=
        mgt::Status::kInvalidConfig) {
        return EXIT_FAILURE;
    }

    mgt::TrainerPuzzleInputs real{};
    real.group_json = fixtures / "p888.json";
    real.target_bin = fixtures / "p888-target.bin";
    const mgt::Status real_status =
        mgt::LoadTrainerPuzzle(real, config, &puzzle, &fingerprint);
    if (real_status != mgt::Status::kOk) {
        std::cerr << "real_status=" << static_cast<unsigned>(real_status) << "\n";
        return EXIT_FAILURE;
    }
    if (fingerprint.empty()) return EXIT_FAILURE;
    const std::string first_fingerprint = fingerprint;
    if (mgt::LoadTrainerPuzzle(real, config, &puzzle, &fingerprint) !=
        mgt::Status::kOk ||
        fingerprint != first_fingerprint) {
        return EXIT_FAILURE;
    }

    mgt::TrainerPuzzleInputs conflicting = real;
    conflicting.synthetic_benchmark = true;
    if (mgt::LoadTrainerPuzzle(conflicting, config, &puzzle, &fingerprint) !=
        mgt::Status::kInvalidConfig) {
        return EXIT_FAILURE;
    }

    mgt::TrainerPuzzleInputs synthetic{};
    synthetic.synthetic_benchmark = true;
    if (mgt::LoadTrainerPuzzle(synthetic, config, &puzzle, &fingerprint) !=
        mgt::Status::kOk ||
        fingerprint.rfind("synthetic:", 0) != 0) {
        return EXIT_FAILURE;
    }

    mgt::TrainConfig wrong_shape = config;
    wrong_shape.puzzle.raw_state_dim -= 1;
    if (mgt::LoadTrainerPuzzle(real, wrong_shape, &puzzle, &fingerprint) !=
        mgt::Status::kInvalidConfig) {
        return EXIT_FAILURE;
    }

    const auto nonce = std::chrono::steady_clock::now().time_since_epoch().count();
    const std::filesystem::path changed_target =
        std::filesystem::temp_directory_path() / ("mgt-target-" + std::to_string(nonce) + ".bin");
    {
        std::ifstream in(real.target_bin, std::ios::binary);
        std::ofstream out(changed_target, std::ios::binary);
        out << in.rdbuf();
    }
    {
        std::fstream io(changed_target, std::ios::binary | std::ios::in | std::ios::out);
        const unsigned char changed = 1;
        io.write(reinterpret_cast<const char*>(&changed), 1);
    }
    mgt::TrainerPuzzleInputs changed = real;
    changed.target_bin = changed_target;
    std::string changed_fingerprint;
    if (mgt::LoadTrainerPuzzle(changed, config, &puzzle, &changed_fingerprint) !=
        mgt::Status::kOk ||
        changed_fingerprint == first_fingerprint) {
        return EXIT_FAILURE;
    }
    std::filesystem::remove(changed_target);
    return EXIT_SUCCESS;
}
