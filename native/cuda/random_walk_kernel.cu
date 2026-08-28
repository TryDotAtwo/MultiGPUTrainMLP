#include "mgt_cuda/random_walk_kernel.cuh"
#include "mgt_cuda/device_context.cuh"

namespace mgt_cuda {
namespace {

__host__ __device__ std::uint64_t Mix64(std::uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

__device__ std::uint32_t NextBounded(std::uint64_t* state, std::uint32_t bound) {
    *state = Mix64(*state);
    return static_cast<std::uint32_t>(*state % bound);
}

__host__ __device__ std::uint32_t OriginalSourceId(
    std::uint64_t base_seed, std::uint64_t epoch, std::uint64_t position) {
    constexpr std::uint64_t kMultiplier = 48271;
    constexpr std::uint64_t kCount = mgt::P888TrainingContract::kSamplesPerEpoch;
    const std::uint64_t shift = Mix64(base_seed ^ Mix64(epoch)) % kCount;
    return static_cast<std::uint32_t>(
        (kMultiplier * (position % kCount) + shift) % kCount);
}

__global__ void RandomWalkKernel(RandomWalkKernelConfig config,
                                 std::uint64_t base_seed,
                                 std::uint64_t epoch,
                                 std::uint64_t step,
                                 std::uint32_t global_rank,
                                 const mgt::TrainStateStorage* moves,
                                 const mgt::TrainStateStorage* target,
                                 mgt::TrainStateStorage* states,
                                 float* labels,
                                 mgt::WalkMeta* meta) {
    const std::uint32_t sample = blockIdx.x * blockDim.x + threadIdx.x;
    if (sample >= config.sample_count) return;

    const std::uint32_t source_id = config.original_p888_schedule
        ? OriginalSourceId(base_seed, epoch, config.epoch_sample_offset + sample)
        : sample;
    std::uint64_t rng = base_seed;
    rng ^= Mix64(epoch + 0x100000001b3ULL);
    if (!config.original_p888_schedule) {
        rng ^= Mix64(step + 0x9e3779b97f4a7c15ULL);
        rng ^= Mix64(static_cast<std::uint64_t>(global_rank) << 32);
    }
    rng ^= Mix64(source_id);

    const std::uint32_t depth_span = config.k_max - config.k_min + 1U;
    const std::uint32_t depth = config.original_p888_schedule
        ? mgt::P888TrainingContract::kMinDepth +
            source_id / mgt::P888TrainingContract::kWalkersPerDepth
        : config.k_min + NextBounded(&rng, depth_span);

    mgt::TrainStateStorage current = *target;
    std::uint32_t last_move = config.move_count - 1U;
    for (std::uint32_t d = 0; d < depth; ++d) {
        const std::uint32_t forbidden = last_move ^ 1U;
        std::uint32_t move = NextBounded(&rng, config.move_count - 1U);
        if (move >= forbidden) ++move;

        mgt::TrainStateStorage next{};
        const mgt::TrainStateStorage move_def = moves[move];
        for (std::uint32_t i = 0; i < config.state_len; ++i) {
            next.v[i] = current.v[move_def.v[i]];
        }
        for (std::uint32_t i = config.state_len; i < config.state_storage_len; ++i) {
            next.v[i] = 0;
        }
        current = next;
        last_move = move;
    }

    states[sample] = current;
    labels[sample] = static_cast<float>(depth);
    meta[sample] = mgt::WalkMeta{depth, last_move, rng};
}

}  // namespace

__host__ mgt::Status ValidateRandomWalkKernelConfig(const RandomWalkKernelConfig& config) {
    if (config.sample_count == 0 || config.k_min == 0 || config.k_min > config.k_max ||
        config.move_count == 0 || config.move_count > mgt::kMoveCount ||
        config.state_len == 0 || config.state_len > mgt::kStateLen ||
        config.state_storage_len < config.state_len || config.state_storage_len > mgt::kStateStorageLen ||
        (config.original_p888_schedule &&
         (config.k_min != mgt::P888TrainingContract::kMinDepth ||
          config.k_max != mgt::P888TrainingContract::kMaxDepth ||
          config.epoch_sample_offset > mgt::P888TrainingContract::kSamplesPerEpoch ||
          config.sample_count > mgt::P888TrainingContract::kSamplesPerEpoch -
              config.epoch_sample_offset))) {
        return mgt::Status::kInvalidConfig;
    }
    return mgt::Status::kOk;
}

__host__ mgt::Status LaunchRandomWalkKernel(const RandomWalkKernelConfig& config,
                                            std::uint64_t base_seed,
                                            std::uint64_t epoch,
                                            std::uint64_t step,
                                            std::uint32_t global_rank,
                                            const mgt::TrainStateStorage* device_moves,
                                            const mgt::TrainStateStorage* device_target,
                                            mgt::TrainStateStorage* device_states,
                                            float* device_labels,
                                            mgt::WalkMeta* device_meta,
                                            cudaStream_t stream) {
    if (ValidateRandomWalkKernelConfig(config) != mgt::Status::kOk ||
        device_moves == nullptr || device_target == nullptr || device_states == nullptr ||
        device_labels == nullptr || device_meta == nullptr) {
        return mgt::Status::kInvalidConfig;
    }
    const DeviceLaunchConfig launch = Build1DLaunchConfig(config.sample_count, 128);
    RandomWalkKernel<<<launch.blocks, launch.threads, 0, stream>>>(
        config, base_seed, epoch, step, global_rank, device_moves, device_target,
        device_states, device_labels, device_meta);
    return cudaGetLastError() == cudaSuccess ? mgt::Status::kOk : mgt::Status::kCudaFailure;
}

}  // namespace mgt_cuda
