#include "mgt/random_walk.hpp"
#include <cstdlib>

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
    const mgt::WalkRequest bad_request{1234, 5, 7, 2, 9, 1, kSamples};
    if (mgt::GenerateRandomWalksCpu(puzzle, bad_request, states_a, labels_a, meta_a) != mgt::Status::kInvalidConfig) {
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}