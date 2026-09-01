#include "mgt/input_active_bins.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <filesystem>
#include <vector>

namespace {

std::vector<std::uint16_t> DirectedOracle(
    const mgt::PuzzleDefinition& puzzle, unsigned state_len, unsigned value_pad) {
    std::vector<std::uint16_t> bins;
    for (unsigned position = 0; position < state_len; ++position) {
        std::array<bool, mgt::kStateLen> seen{};
        std::array<unsigned, mgt::kStateLen> queue{};
        unsigned first = 0, last = 0;
        seen[position] = true;
        queue[last++] = position;
        while (first != last) {
            const unsigned current = queue[first++];
            for (const auto& move : puzzle.moves) {
                const unsigned next = move.v[current];
                if (!seen[next]) {
                    seen[next] = true;
                    queue[last++] = next;
                }
            }
        }
        std::array<bool, mgt::kStateLen> values{};
        for (unsigned source = 0; source < state_len; ++source)
            if (seen[source]) values[puzzle.target.v[source]] = true;
        for (unsigned value = 0; value < value_pad; ++value)
            if (values[value]) bins.push_back(
                static_cast<std::uint16_t>(position * value_pad + value));
    }
    return bins;
}

mgt::PuzzleDefinition FullOrbitPuzzle() {
    mgt::PuzzleDefinition puzzle{};
    for (unsigned move = 0; move < mgt::kMoveCount; ++move)
        for (unsigned position = 0; position < mgt::kStateLen; ++position)
            puzzle.moves[move].v[position] = static_cast<mgt::StateValue>(position);
    for (unsigned position = 0; position < mgt::kStateLen; ++position) {
        puzzle.moves[0].v[position] =
            static_cast<mgt::StateValue>((position + 1U) % mgt::kStateLen);
        puzzle.target.v[position] = static_cast<mgt::StateValue>(position);
    }
    return puzzle;
}

}  // namespace

int main() {
    mgt::PuzzleDefinition production{};
    if (mgt::LoadPuzzleDefinition(
            std::filesystem::path("native/production_inputs/p888.json"),
            std::filesystem::path("native/tests/fixtures/p888-target.bin"),
            &production) != mgt::Status::kOk) return 1;
    std::vector<std::uint16_t> bins;
    if (mgt::BuildInputActiveBins(production, 72, 72, &bins) != mgt::Status::kOk ||
        bins.size() != 72U * 24U || bins != DirectedOracle(production, 72, 72) ||
        !std::is_sorted(bins.begin(), bins.end())) return 2;
    for (unsigned position = 0; position < 72; ++position) {
        const auto first = std::lower_bound(
            bins.begin(), bins.end(), static_cast<std::uint16_t>(position * 72U));
        const auto last = std::lower_bound(
            bins.begin(), bins.end(), static_cast<std::uint16_t>((position + 1U) * 72U));
        if (last - first != 24) return 3;
    }

    const auto full = FullOrbitPuzzle();
    if (mgt::BuildInputActiveBins(full, 72, 72, &bins) != mgt::Status::kOk ||
        bins.size() != 72U * 72U || bins != DirectedOracle(full, 72, 72)) return 4;

    if (mgt::BuildInputActiveBins(full, 0, 72, &bins) != mgt::Status::kInvalidConfig ||
        mgt::BuildInputActiveBins(full, 72, 0, &bins) != mgt::Status::kInvalidConfig ||
        mgt::BuildInputActiveBins(full, 73, 72, &bins) != mgt::Status::kInvalidConfig ||
        mgt::BuildInputActiveBins(full, 72, 73, &bins) != mgt::Status::kInvalidConfig ||
        mgt::BuildInputActiveBins(full, 72, 72, nullptr) != mgt::Status::kInvalidConfig)
        return 5;

    auto invalid = full;
    invalid.moves[0].v[0] = 255;
    if (mgt::BuildInputActiveBins(invalid, 72, 72, &bins) != mgt::Status::kInvalidPuzzle)
        return 6;
    invalid = full;
    invalid.target.v[0] = 255;
    if (mgt::BuildInputActiveBins(invalid, 72, 72, &bins) != mgt::Status::kInvalidPuzzle)
        return 7;
    return 0;
}
