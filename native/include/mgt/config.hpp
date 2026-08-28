#pragma once

#include <array>
#include <cstdint>

namespace mgt {

inline constexpr std::uint32_t kStateLen = 72;
inline constexpr std::uint32_t kStateAlignment = 16;
inline constexpr std::uint32_t kStateValuePad = 72;
inline constexpr std::uint32_t kMoveCount = 18;
inline constexpr std::uint32_t kOutputDim = 1;
inline constexpr std::uint32_t kHd1 = 2556;
inline constexpr std::uint32_t kHd2 = 218;
inline constexpr std::uint32_t kResidualBlocks = 16;
inline constexpr std::uint32_t kHiddenAlignment = 8;
inline constexpr std::uint32_t kGradientCarouselSlots = 3;
inline constexpr std::uint32_t kInputGradPartialChunks = 1;
inline constexpr std::uint32_t kInputGradPositionsPerBlock = 1;
inline constexpr std::uint64_t kDefaultAllreduceBucketBytes = 4ULL * 1024ULL * 1024ULL;
inline constexpr std::uint64_t kDefaultLtWorkspaceBytes = 0ULL;
inline constexpr std::uint32_t kMaxRuntimeDevices = 128;

constexpr std::uint32_t RoundUp(std::uint32_t value, std::uint32_t alignment) {
    return alignment == 0 ? value : ((value + alignment - 1U) / alignment) * alignment;
}

constexpr std::uint64_t RoundUp64(std::uint64_t value, std::uint64_t alignment) {
    return alignment == 0 ? value : ((value + alignment - 1ULL) / alignment) * alignment;
}

inline constexpr std::uint32_t kStateStorageLen =
    RoundUp(kStateLen, kStateAlignment);

static_assert(kStateStorageLen == 80);

enum class PrecisionMode : std::uint32_t {
    kFloat32 = 1,
    kFloat16ComputeFloat32Master = 2
};

enum class RuntimeMode : std::uint32_t {
    kSingleGpu = 1,
    kMultiGpu = 2
};

struct PuzzleSpec {
    std::uint32_t group_id = 888;
    std::uint32_t target_id = 0;
    std::uint32_t raw_state_dim = kStateLen;
    std::uint32_t state_value_count = kStateValuePad;
    std::uint32_t move_count = kMoveCount;
    std::uint32_t state_alignment = kStateAlignment;
};

struct ModelSpec {
    std::uint32_t hd1 = kHd1;
    std::uint32_t hd2 = kHd2;
    std::uint32_t residual_blocks = kResidualBlocks;
    std::uint32_t output_dim = kOutputDim;
    std::uint32_t hidden_alignment = kHiddenAlignment;
    PrecisionMode precision = PrecisionMode::kFloat16ComputeFloat32Master;
};

struct P888TrainingContract {
    static constexpr std::uint32_t kStateValueCount = 72;
    static constexpr std::uint32_t kInputFeatures = kStateLen * kStateValueCount;
    static constexpr std::uint32_t kHidden1 = 2556;
    static constexpr std::uint32_t kHidden2 = 218;
    static constexpr std::uint32_t kResidualBlocks = 16;
    static constexpr std::uint32_t kBatchNormSites = 2 + 2 * kResidualBlocks;
    static constexpr std::uint32_t kMinDepth = 1;
    static constexpr std::uint32_t kMaxDepth = 29;
    static constexpr std::uint32_t kWalkersPerDepth = 34482;
    static constexpr std::uint32_t kSamplesPerEpoch = kWalkersPerDepth * (kMaxDepth - kMinDepth + 1);
    static constexpr std::uint32_t kGlobalBatch = 100000;
    static constexpr std::uint32_t kFinalGlobalBatch = kSamplesPerEpoch % kGlobalBatch;
    static constexpr std::uint32_t kOptimizerStepsPerEpoch = (kSamplesPerEpoch + kGlobalBatch - 1) / kGlobalBatch;
    static constexpr std::uint32_t kEpochs = 32692;
    static constexpr std::uint32_t kOptimizerSteps = kEpochs * kOptimizerStepsPerEpoch;
    static constexpr float kLearningRate = 1.0e-4f;
    static constexpr float kWeightDecay = 0.0f;
    static constexpr float kBatchNormEpsilon = 1.0e-5f;
    static constexpr float kBatchNormMomentum = 0.1f;
};
struct TrainSpec {
    std::uint32_t epochs = 32692;
    std::uint32_t batch_size = 100000;
    std::uint32_t k_min = 1;
    std::uint32_t k_max = 29;
    std::uint32_t walkers = 34482;
    std::uint32_t save_interval = 4096;
    std::uint64_t seed = 1234;
    float lr = 0.0001f;
    float weight_decay = 0.0f;
};

struct RuntimeSpec {
    std::array<std::uint32_t, kMaxRuntimeDevices> devices{};
    std::uint32_t device_count = 1;
    std::uint32_t gradient_carousel_slots = kGradientCarouselSlots;
    std::uint32_t input_grad_partial_chunks = kInputGradPartialChunks;
    std::uint32_t input_grad_positions_per_block = kInputGradPositionsPerBlock;
    std::uint32_t input_grad_position_tile = 0;
    bool input_grad_sparse = false;
    std::uint64_t allreduce_bucket_bytes = kDefaultAllreduceBucketBytes;
    std::uint64_t lt_workspace_bytes = kDefaultLtWorkspaceBytes;
    RuntimeMode mode = RuntimeMode::kSingleGpu;
};

struct TrainConfig {
    PuzzleSpec puzzle{};
    ModelSpec model{};
    TrainSpec train{};
    RuntimeSpec runtime{};
};

TrainConfig MakeArchiveP888TrainConfig();

}  // namespace mgt
