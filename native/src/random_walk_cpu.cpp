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

TrainStateStorage ApplyMove(const TrainStateStorage& current, const TrainStateStorage& move) {
    TrainStateStorage next{};
    for (std::uint32_t i = 0; i < kStateLen; ++i) {
        next.v[i] = current.v[move.v[i]];
    }
    for (std::uint32_t i = kStateLen; i < kStateStorageLen; ++i) {
        next.v[i] = 0;
    }
    return next;
}

}  // namespace

std::uint32_t OriginalP888SourceIdAtPosition(
    std::uint64_t base_seed,
    std::uint64_t semantic_epoch,
    std::uint64_t epoch_position) {
    constexpr std::uint64_t kMultiplier = 48271;
    constexpr std::uint64_t kCount = P888TrainingContract::kSamplesPerEpoch;
    const std::uint64_t shift = Mix64(base_seed ^ Mix64(semantic_epoch)) % kCount;
    return static_cast<std::uint32_t>(
        (kMultiplier * (epoch_position % kCount) + shift) % kCount);
}

Status GenerateRandomWalksCpu(const PuzzleDefinition& puzzle,
                              const WalkRequest& request,
                              TrainStateStorage* states,
                              float* labels,
                              WalkMeta* meta) {
    if (states == nullptr || labels == nullptr || meta == nullptr) return Status::kInvalidConfig;
    if (request.sample_count == 0) return Status::kInvalidConfig;
    if (request.k_min == 0 || request.k_min > request.k_max) return Status::kInvalidConfig;
    if (request.original_p888_schedule &&
        (request.k_min != P888TrainingContract::kMinDepth ||
         request.k_max != P888TrainingContract::kMaxDepth ||
         request.epoch_sample_offset > P888TrainingContract::kSamplesPerEpoch ||
         request.sample_count > P888TrainingContract::kSamplesPerEpoch -
             request.epoch_sample_offset)) return Status::kInvalidConfig;

    const std::uint32_t depth_span = request.k_max - request.k_min + 1U;
    for (std::uint32_t sample = 0; sample < request.sample_count; ++sample) {
        const std::uint32_t source_id = request.original_p888_schedule
            ? OriginalP888SourceIdAtPosition(
                  request.base_seed, request.epoch,
                  request.epoch_sample_offset + sample)
            : sample;
        std::uint64_t rng = request.base_seed;
        rng ^= Mix64(request.epoch + 0x100000001b3ULL);
        if (!request.original_p888_schedule) {
            rng ^= Mix64(request.step + 0x9e3779b97f4a7c15ULL);
            rng ^= Mix64(static_cast<std::uint64_t>(request.global_rank) << 32);
        }
        rng ^= Mix64(source_id);

        const std::uint32_t depth = request.original_p888_schedule
            ? P888TrainingContract::kMinDepth +
                source_id / P888TrainingContract::kWalkersPerDepth
            : request.k_min + NextBounded(&rng, depth_span);
        TrainStateStorage state = puzzle.target;
        // Original source starts with last_moves=-1 and indexes the inverse
        // table with it, so canonical p888 excludes inverse(17)==16 first.
        std::uint32_t last_move = kMoveCount - 1U;
        for (std::uint32_t d = 0; d < depth; ++d) {
            const std::uint32_t forbidden = last_move ^ 1U;
            std::uint32_t move = NextBounded(&rng, kMoveCount - 1U);
            if (move >= forbidden) ++move;
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
