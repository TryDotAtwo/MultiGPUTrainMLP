#include "mgt/puzzle_io.hpp"
#include <cstdlib>
#include <filesystem>
#include <fstream>

int main() {
    const std::filesystem::path fixture_dir = "native/tests/fixtures";
    mgt::PuzzleDefinition puzzle{};
    const auto status = mgt::LoadPuzzleDefinition(
        fixture_dir / "p888.json",
        fixture_dir / "p888-target.bin",
        &puzzle);
    if (status != mgt::Status::kOk) return EXIT_FAILURE;
    for (std::uint32_t i = 0; i < mgt::kStateLen; ++i) {
        if (puzzle.target.v[i] != static_cast<mgt::StateValue>(i)) return EXIT_FAILURE;
    }
    for (std::uint32_t i = mgt::kStateLen; i < mgt::kStateStorageLen; ++i) {
        if (puzzle.target.v[i] != 0) return EXIT_FAILURE;
        if (puzzle.moves[0].v[i] != static_cast<mgt::StateValue>(i)) return EXIT_FAILURE;
    }
    if (mgt::HasNonIdentityMove(puzzle)) return EXIT_FAILURE;

    mgt::PuzzleDefinition production{};
    const auto production_status = mgt::LoadPuzzleDefinition(
        "native/production_inputs/p888.json",
        fixture_dir / "p888-target.bin",
        &production);
    if (production_status != mgt::Status::kOk ||
        !mgt::HasNonIdentityMove(production)) return EXIT_FAILURE;
    if (!mgt::HasCanonicalInverseMovePairs(production)) return EXIT_FAILURE;
    auto wrong_inverse_order = production;
    std::swap(wrong_inverse_order.moves[1], wrong_inverse_order.moves[2]);
    if (mgt::HasCanonicalInverseMovePairs(wrong_inverse_order)) return EXIT_FAILURE;

    const std::filesystem::path bad_json = "build-native/bad-p888.json";
    {
        std::ofstream out(bad_json, std::ios::binary);
        out << "{\"group_id\":888,\"state_len\":72,\"move_count\":18,\"moves\":[";
        for (std::uint32_t move = 0; move < mgt::kMoveCount; ++move) {
            if (move != 0) out << ",";
            out << "[";
            for (std::uint32_t i = 0; i < mgt::kStateLen; ++i) {
                if (i != 0) out << ",";
                out << (i == 1 ? 0 : i);
            }
            out << "]";
        }
        out << "]}";
    }
    const auto bad_status = mgt::LoadPuzzleDefinition(
        bad_json,
        fixture_dir / "p888-target.bin",
        &puzzle);
    if (bad_status != mgt::Status::kInvalidPuzzle) return EXIT_FAILURE;

    return EXIT_SUCCESS;
}
