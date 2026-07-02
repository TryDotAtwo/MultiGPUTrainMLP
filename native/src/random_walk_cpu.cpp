#include "mgt/random_walk.hpp"
#include <array>

namespace mgt {
namespace {

std::uint64_t Mix64(std::uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

std::uint32_t NextBounded(std::uint64_t* state, std::uint32_t bound) {
    *state = Mix64(*state);
    return static_cast<std::uint32_t>(*state % bound);
}

TrainState80 ApplyMove(const TrainState80& current, const TrainState80& move) {
    TrainState80 next{};
    for (std::uint32_t i = 0; i < kStateLen; ++i) {
        next.v[i] = current.v[move.v[i]];
    }
    for (std::uint32_t i = kStateLen; i < kStateStorageLen; ++i) {
        next.v[i] = 0;
    }
    return next;
}

}  // namespace

Status GenerateRandomWalksCpu(const PuzzleDefinition& puzzle,
                              const WalkRequest& request,
                              TrainState80* states,
                              float* labels,
                              WalkMeta* meta) {
    if (states == nullptr || labels == nullptr || meta == nullptr) return Status::kInvalidConfig;
    if (request.sample_count == 0) return Status::kInvalidConfig;
    if (request.k_min == 0 || request.k_min > request.k_max) return Status::kInvalidConfig;

    const std::uint32_t depth_span = request.k_max - request.k_min + 1U;
    for (std::uint32_t sample = 0; sample < request.sample_count; ++sample) {
        std::uint64_t rng = request.base_seed;
        rng ^= Mix64(request.epoch + 0x100000001b3ULL);
        rng ^= Mix64(request.step + 0x9e3779b97f4a7c15ULL);
        rng ^= Mix64(static_cast<std::uint64_t>(request.global_rank) << 32);
        rng ^= Mix64(sample);

        const std::uint32_t depth = request.k_min + NextBounded(&rng, depth_span);
        TrainState80 state = puzzle.target;
        std::uint32_t last_move = kMoveCount;
        for (std::uint32_t d = 0; d < depth; ++d) {
            std::uint32_t move = NextBounded(&rng, kMoveCount);
            if (last_move != kMoveCount && move == last_move) {
                move = (move + 1U + NextBounded(&rng, kMoveCount - 1U)) % kMoveCount;
            }
            state = ApplyMove(state, puzzle.moves[move]);
            last_move = move;
        }
        states[sample] = state;
        labels[sample] = static_cast<float>(depth);
        meta[sample] = WalkMeta{depth, last_move, rng};
    }

    return Status::kOk;
}

}  // namespace mgt