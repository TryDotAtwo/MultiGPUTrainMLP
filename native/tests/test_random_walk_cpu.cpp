#include "mgt/random_walk.hpp"
#include <cstdlib>
#include <filesystem>
#include <array>
#include <vector>

int main() {
    mgt::PuzzleDefinition puzzle{};
    for (std::uint32_t move = 0; move < mgt::kMoveCount; ++move) {
        for (std::uint32_t i = 0; i < mgt::kStateLen; ++i) {
            puzzle.moves[move].v[i] = static_cast<mgt::StateValue>((i + move + 1) % mgt::kStateLen);
        }
        for (std::uint32_t i = mgt::kStateLen; i < mgt::kStateStorageLen; ++i) {
            puzzle.moves[move].v[i] = static_cast<mgt::StateValue>(i);
        }
    }
    for (std::uint32_t i = 0; i < mgt::kStateLen; ++i) {
        puzzle.target.v[i] = static_cast<mgt::StateValue>(i);
    }
    for (std::uint32_t i = mgt::kStateLen; i < mgt::kStateStorageLen; ++i) {
        puzzle.target.v[i] = 0;
    }

    constexpr std::uint32_t kSamples = 64;
    mgt::TrainStateStorage states_a[kSamples]{};
    mgt::TrainStateStorage states_b[kSamples]{};
    float labels_a[kSamples]{};
    float labels_b[kSamples]{};
    mgt::WalkMeta meta_a[kSamples]{};
    mgt::WalkMeta meta_b[kSamples]{};

    const mgt::WalkRequest request{1234, 5, 7, 2, 1, 9, kSamples};
    if (mgt::GenerateRandomWalksCpu(puzzle, request, states_a, labels_a, meta_a) != mgt::Status::kOk) {
        return EXIT_FAILURE;
    }
    if (mgt::GenerateRandomWalksCpu(puzzle, request, states_b, labels_b, meta_b) != mgt::Status::kOk) {
        return EXIT_FAILURE;
    }

    bool saw_non_target = false;
    for (std::uint32_t sample = 0; sample < kSamples; ++sample) {
        if (meta_a[sample].depth < request.k_min || meta_a[sample].depth > request.k_max) return EXIT_FAILURE;
        if (labels_a[sample] != static_cast<float>(meta_a[sample].depth)) return EXIT_FAILURE;
        if (meta_a[sample].last_move >= mgt::kMoveCount) return EXIT_FAILURE;
        if (meta_a[sample].rng_counter == 0) return EXIT_FAILURE;
        for (std::uint32_t i = 0; i < mgt::kStateStorageLen; ++i) {
            if (states_a[sample].v[i] != states_b[sample].v[i]) return EXIT_FAILURE;
            if (i >= mgt::kStateLen && states_a[sample].v[i] != 0) return EXIT_FAILURE;
        }
        if (labels_a[sample] != labels_b[sample]) return EXIT_FAILURE;
        if (meta_a[sample].depth != meta_b[sample].depth) return EXIT_FAILURE;
        for (std::uint32_t i = 0; i < mgt::kStateLen; ++i) {
            if (states_a[sample].v[i] != puzzle.target.v[i]) saw_non_target = true;
        }
    }
    if (!saw_non_target) return EXIT_FAILURE;

    mgt::TrainStateStorage states_rank3[kSamples]{};
    float labels_rank3[kSamples]{};
    mgt::WalkMeta meta_rank3[kSamples]{};
    const mgt::WalkRequest rank3_request{1234, 5, 7, 3, 1, 9, kSamples};
    if (mgt::GenerateRandomWalksCpu(puzzle, rank3_request, states_rank3, labels_rank3, meta_rank3) != mgt::Status::kOk) {
        return EXIT_FAILURE;
    }
    bool rank_differs = false;
    for (std::uint32_t sample = 0; sample < kSamples; ++sample) {
        if (meta_a[sample].rng_counter != meta_rank3[sample].rng_counter ||
            meta_a[sample].depth != meta_rank3[sample].depth) {
            rank_differs = true;
            break;
        }
    }
    if (!rank_differs) return EXIT_FAILURE;

    mgt::PuzzleDefinition production{};
    if (mgt::LoadPuzzleDefinition(
            "native/production_inputs/p888.json",
            "native/tests/fixtures/p888-target.bin",
            &production) != mgt::Status::kOk) return EXIT_FAILURE;
    constexpr std::uint32_t kOriginalSamples = 4096;
    mgt::TrainStateStorage original_states[kOriginalSamples]{};
    float original_labels[kOriginalSamples]{};
    mgt::WalkMeta original_meta[kOriginalSamples]{};
    const mgt::WalkRequest original_request{
        0x8881, 0, 1, 0, 1, 1, kOriginalSamples};
    if (mgt::GenerateRandomWalksCpu(
            production, original_request, original_states, original_labels,
            original_meta) != mgt::Status::kOk) return EXIT_FAILURE;
    for (const auto& item : original_meta) {
        // Original trainer initializes last_move=-1, therefore its first legal
        // set excludes inverse_moves[-1], which is move 16 for canonical p888.
        if (item.last_move == 16) return EXIT_FAILURE;
    }

    std::vector<bool> seen(mgt::P888TrainingContract::kSamplesPerEpoch, false);
    std::array<std::uint32_t, 29> depth_counts{};
    for (std::uint64_t position = 0;
         position < mgt::P888TrainingContract::kSamplesPerEpoch; ++position) {
        const auto source_id = mgt::OriginalP888SourceIdAtPosition(
            0x8881, 7, position);
        if (source_id >= seen.size() || seen[source_id]) return EXIT_FAILURE;
        seen[source_id] = true;
        ++depth_counts[source_id / mgt::P888TrainingContract::kWalkersPerDepth];
    }
    for (const auto count : depth_counts) {
        if (count != mgt::P888TrainingContract::kWalkersPerDepth)
            return EXIT_FAILURE;
    }
    constexpr std::uint32_t kSlice = 257;
    mgt::TrainStateStorage scheduled_states[kSlice]{};
    float scheduled_labels[kSlice]{};
    mgt::WalkMeta scheduled_meta[kSlice]{};
    const mgt::WalkRequest scheduled_request{
        0x8881, 7, 0, 0, 1, 29, kSlice, 123456, true};
    if (mgt::GenerateRandomWalksCpu(
            production, scheduled_request, scheduled_states, scheduled_labels,
            scheduled_meta) != mgt::Status::kOk) return EXIT_FAILURE;
    for (std::uint32_t row = 0; row < kSlice; ++row) {
        const auto source_id = mgt::OriginalP888SourceIdAtPosition(
            scheduled_request.base_seed, scheduled_request.epoch,
            scheduled_request.epoch_sample_offset + row);
        const auto expected_depth = 1U +
            source_id / mgt::P888TrainingContract::kWalkersPerDepth;
        if (scheduled_meta[row].depth != expected_depth ||
            scheduled_labels[row] != static_cast<float>(expected_depth))
            return EXIT_FAILURE;
    }
    const mgt::WalkRequest bad_request{1234, 5, 7, 2, 9, 1, kSamples};
    if (mgt::GenerateRandomWalksCpu(puzzle, bad_request, states_a, labels_a, meta_a) != mgt::Status::kInvalidConfig) {
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
