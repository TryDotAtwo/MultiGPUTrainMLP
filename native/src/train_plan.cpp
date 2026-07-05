#include "mgt/train_plan.hpp"

namespace mgt {
namespace {

bool IsPowerOfTwo(std::uint32_t value) {
    return value != 0 && (value & (value - 1U)) == 0;
}

std::vector<GradientBucketPlan> BuildGradientBuckets(std::uint64_t total_bytes, std::uint64_t requested_bucket_bytes) {
    std::vector<GradientBucketPlan> buckets;
    if (total_bytes == 0 || requested_bucket_bytes == 0) return buckets;
    const std::uint64_t bucket_bytes = RoundUp64(requested_bucket_bytes, sizeof(float));
    std::uint64_t offset = 0;
    std::uint32_t bucket_id = 0;
    while (offset < total_bytes) {
        const std::uint64_t remaining = total_bytes - offset;
        const std::uint64_t size = remaining < bucket_bytes ? remaining : bucket_bytes;
        buckets.push_back(GradientBucketPlan{
            bucket_id,
            offset,
            size,
            offset / sizeof(float),
            size / sizeof(float)});
        offset += size;
        ++bucket_id;
    }
    return buckets;
}

bool TensorCoreEligible(std::uint32_t m, std::uint32_t n, std::uint32_t k) {
    return m != 0 && n != 0 && k != 0 && (m % 8U) == 0 && (n % 8U) == 0 && (k % 8U) == 0;
}

LinearBackendClass ChooseLinearBackend(ParamBlockRole role,
                                       LinearOpPhase phase,
                                       std::uint32_t m,
                                       std::uint32_t n,
                                       std::uint32_t k,
                                       std::uint32_t output_dim) {
    if (role == ParamBlockRole::kInputEmbedding && phase == LinearOpPhase::kForward) return LinearBackendClass::kCustomKernel;
    if (role == ParamBlockRole::kOutputWeight && output_dim == 1U) return LinearBackendClass::kCustomKernel;
    if (role == ParamBlockRole::kInputEmbedding && phase == LinearOpPhase::kBackwardWeights && m >= 4096U && n >= 4096U && k >= 8192U) {
        return LinearBackendClass::kCutlassLargeKReductionCandidate;
    }
    return LinearBackendClass::kLibraryTensorOp;
}

void AddLinearOp(std::vector<LinearOpPlan>* ops,
                 LinearOpPhase phase,
                 ParamBlockRole role,
                 std::uint32_t block_index,
                 MatrixTranspose transpose_a,
                 MatrixTranspose transpose_b,
                 std::uint32_t m,
                 std::uint32_t n,
                 std::uint32_t k,
                 LinearBackendClass backend,
                 bool logical_tail_masked) {
    LinearOpPlan op{};
    op.op_id = static_cast<std::uint32_t>(ops->size());
    op.phase = phase;
    op.role = role;
    op.block_index = block_index;
    op.transpose_a = transpose_a;
    op.transpose_b = transpose_b;
    op.m = m;
    op.n = n;
    op.k = k;
    op.lhs_rows = transpose_a == MatrixTranspose::kNo ? m : k;
    op.lhs_cols = transpose_a == MatrixTranspose::kNo ? k : m;
    op.rhs_rows = transpose_b == MatrixTranspose::kNo ? k : n;
    op.rhs_cols = transpose_b == MatrixTranspose::kNo ? n : k;
    op.output_rows = m;
    op.output_cols = n;
    op.backend = backend;
    op.tensor_core_eligible = TensorCoreEligible(m, n, k);
    op.logical_tail_masked = logical_tail_masked;
    ops->push_back(op);
}

std::vector<LinearOpPlan> BuildLinearOps(const TrainConfig& config, const ModelLayout& layout) {
    std::vector<LinearOpPlan> ops;
    const std::uint32_t batch = config.train.batch_size;
    const std::uint32_t input_dim = config.puzzle.raw_state_dim * config.puzzle.state_value_count;
    const std::uint32_t hd1 = layout.physical_hd1;
    const std::uint32_t hd2 = layout.physical_hd2;
    const std::uint32_t output_dim = layout.output_dim;
    const bool hd1_tail = layout.logical_hd1 != layout.physical_hd1;
    const bool hd2_tail = layout.logical_hd2 != layout.physical_hd2;

    AddLinearOp(&ops, LinearOpPhase::kForward, ParamBlockRole::kInputEmbedding, 0, MatrixTranspose::kNo, MatrixTranspose::kNo,
                batch, hd1, config.puzzle.raw_state_dim,
                ChooseLinearBackend(ParamBlockRole::kInputEmbedding, LinearOpPhase::kForward, batch, hd1, config.puzzle.raw_state_dim, output_dim),
                hd1_tail);
    AddLinearOp(&ops, LinearOpPhase::kForward, ParamBlockRole::kHiddenWeight, 0, MatrixTranspose::kNo, MatrixTranspose::kNo,
                batch, hd2, hd1,
                ChooseLinearBackend(ParamBlockRole::kHiddenWeight, LinearOpPhase::kForward, batch, hd2, hd1, output_dim),
                hd1_tail || hd2_tail);
    for (std::uint32_t block = 0; block < layout.residual_blocks; ++block) {
        AddLinearOp(&ops, LinearOpPhase::kForward, ParamBlockRole::kResidualFc1Weight, block, MatrixTranspose::kNo, MatrixTranspose::kNo,
                    batch, hd2, hd2,
                    ChooseLinearBackend(ParamBlockRole::kResidualFc1Weight, LinearOpPhase::kForward, batch, hd2, hd2, output_dim),
                    hd2_tail);
        AddLinearOp(&ops, LinearOpPhase::kForward, ParamBlockRole::kResidualFc2Weight, block, MatrixTranspose::kNo, MatrixTranspose::kNo,
                    batch, hd2, hd2,
                    ChooseLinearBackend(ParamBlockRole::kResidualFc2Weight, LinearOpPhase::kForward, batch, hd2, hd2, output_dim),
                    hd2_tail);
    }
    AddLinearOp(&ops, LinearOpPhase::kForward, ParamBlockRole::kOutputWeight, 0, MatrixTranspose::kNo, MatrixTranspose::kNo,
                batch, output_dim, hd2,
                ChooseLinearBackend(ParamBlockRole::kOutputWeight, LinearOpPhase::kForward, batch, output_dim, hd2, output_dim),
                hd2_tail);

    AddLinearOp(&ops, LinearOpPhase::kBackwardWeights, ParamBlockRole::kOutputWeight, 0, MatrixTranspose::kYes, MatrixTranspose::kNo,
                hd2, output_dim, batch,
                ChooseLinearBackend(ParamBlockRole::kOutputWeight, LinearOpPhase::kBackwardWeights, hd2, output_dim, batch, output_dim),
                hd2_tail);
    AddLinearOp(&ops, LinearOpPhase::kBackwardInput, ParamBlockRole::kOutputWeight, 0, MatrixTranspose::kNo, MatrixTranspose::kYes,
                batch, hd2, output_dim,
                ChooseLinearBackend(ParamBlockRole::kOutputWeight, LinearOpPhase::kBackwardInput, batch, hd2, output_dim, output_dim),
                hd2_tail);
    for (std::uint32_t rblock = layout.residual_blocks; rblock > 0; --rblock) {
        const std::uint32_t block = rblock - 1U;
        AddLinearOp(&ops, LinearOpPhase::kBackwardWeights, ParamBlockRole::kResidualFc2Weight, block, MatrixTranspose::kYes, MatrixTranspose::kNo,
                    hd2, hd2, batch,
                    ChooseLinearBackend(ParamBlockRole::kResidualFc2Weight, LinearOpPhase::kBackwardWeights, hd2, hd2, batch, output_dim),
                    hd2_tail);
        AddLinearOp(&ops, LinearOpPhase::kBackwardInput, ParamBlockRole::kResidualFc2Weight, block, MatrixTranspose::kNo, MatrixTranspose::kYes,
                    batch, hd2, hd2,
                    ChooseLinearBackend(ParamBlockRole::kResidualFc2Weight, LinearOpPhase::kBackwardInput, batch, hd2, hd2, output_dim),
                    hd2_tail);
        AddLinearOp(&ops, LinearOpPhase::kBackwardWeights, ParamBlockRole::kResidualFc1Weight, block, MatrixTranspose::kYes, MatrixTranspose::kNo,
                    hd2, hd2, batch,
                    ChooseLinearBackend(ParamBlockRole::kResidualFc1Weight, LinearOpPhase::kBackwardWeights, hd2, hd2, batch, output_dim),
                    hd2_tail);
        AddLinearOp(&ops, LinearOpPhase::kBackwardInput, ParamBlockRole::kResidualFc1Weight, block, MatrixTranspose::kNo, MatrixTranspose::kYes,
                    batch, hd2, hd2,
                    ChooseLinearBackend(ParamBlockRole::kResidualFc1Weight, LinearOpPhase::kBackwardInput, batch, hd2, hd2, output_dim),
                    hd2_tail);
    }
    AddLinearOp(&ops, LinearOpPhase::kBackwardWeights, ParamBlockRole::kHiddenWeight, 0, MatrixTranspose::kYes, MatrixTranspose::kNo,
                hd1, hd2, batch,
                ChooseLinearBackend(ParamBlockRole::kHiddenWeight, LinearOpPhase::kBackwardWeights, hd1, hd2, batch, output_dim),
                hd1_tail || hd2_tail);
    AddLinearOp(&ops, LinearOpPhase::kBackwardInput, ParamBlockRole::kHiddenWeight, 0, MatrixTranspose::kNo, MatrixTranspose::kYes,
                batch, hd1, hd2,
                ChooseLinearBackend(ParamBlockRole::kHiddenWeight, LinearOpPhase::kBackwardInput, batch, hd1, hd2, output_dim),
                hd1_tail || hd2_tail);
    AddLinearOp(&ops, LinearOpPhase::kBackwardWeights, ParamBlockRole::kInputEmbedding, 0, MatrixTranspose::kYes, MatrixTranspose::kNo,
                input_dim, hd1, batch,
                ChooseLinearBackend(ParamBlockRole::kInputEmbedding, LinearOpPhase::kBackwardWeights, input_dim, hd1, batch, output_dim),
                hd1_tail);
    return ops;
}
}  // namespace

TrainConfig MakeArchiveP888TrainConfig() {
    TrainConfig config{};
    config.puzzle = PuzzleSpec{};
    config.model = ModelSpec{};
    config.train = TrainSpec{};
    config.runtime = RuntimeSpec{};
    config.runtime.devices[0] = 0;
    return config;
}

Status ValidateTrainConfig(const TrainConfig& config) {
    const PuzzleSpec& puzzle = config.puzzle;
    const ModelSpec& model = config.model;
    const TrainSpec& train = config.train;
    const RuntimeSpec& runtime = config.runtime;

    if (puzzle.raw_state_dim == 0 || puzzle.state_value_count == 0 || puzzle.move_count == 0) return Status::kInvalidConfig;
    if (puzzle.state_alignment == 0 || !IsPowerOfTwo(puzzle.state_alignment)) return Status::kInvalidConfig;
    if (model.hd1 == 0 || model.hd2 == 0 || model.output_dim == 0) return Status::kInvalidConfig;
    if (model.hidden_alignment == 0 || !IsPowerOfTwo(model.hidden_alignment)) return Status::kInvalidConfig;
    if (train.epochs == 0 || train.batch_size == 0 || train.k_min == 0 || train.k_min > train.k_max) return Status::kInvalidConfig;
    if (train.lr <= 0.0f || train.weight_decay < 0.0f) return Status::kInvalidConfig;
    if (runtime.device_count == 0 || runtime.device_count > kMaxRuntimeDevices) return Status::kInvalidConfig;
    if (runtime.gradient_carousel_slots == 0 || runtime.input_grad_partial_chunks == 0 || runtime.input_grad_positions_per_block == 0 || runtime.allreduce_bucket_bytes == 0) return Status::kInvalidConfig;
    if (runtime.allreduce_bucket_bytes % sizeof(float) != 0) return Status::kInvalidConfig;
    if (runtime.mode == RuntimeMode::kSingleGpu && runtime.device_count != 1) return Status::kInvalidConfig;
    return Status::kOk;
}

RankMemoryRequest BuildRankMemoryRequest(const TrainConfig& config,
                                         const ModelLayout& layout,
                                         std::uint32_t padded_state_dim) {
    RankMemoryRequest request{};
    request.batch_states_per_rank = config.train.batch_size;
    request.model_bytes = layout.total_bytes;
    request.model_params = layout.total_params;
    request.state_storage_bytes = padded_state_dim * sizeof(StateValue);
    request.output_dim = layout.output_dim;
    request.physical_hd1 = layout.physical_hd1;
    request.physical_hd2 = layout.physical_hd2;
    request.residual_blocks = layout.residual_blocks;
    request.gradient_carousel_slots = config.runtime.gradient_carousel_slots;
    request.allreduce_bucket_bytes = config.runtime.allreduce_bucket_bytes;
    return request;
}

Status BuildTrainPlan(const TrainConfig& config, TrainPlan* out) {
    if (out == nullptr) return Status::kInvalidConfig;
    *out = TrainPlan{};
    const Status status = ValidateTrainConfig(config);
    if (status != Status::kOk) return status;

    const std::uint32_t padded_state_dim = RoundUp(config.puzzle.raw_state_dim, config.puzzle.state_alignment);
    ModelLayout layout = BuildModelLayout(config.puzzle, config.model);
    if (layout.blocks.empty() || layout.total_bytes == 0 || layout.total_params == 0) return Status::kInvalidConfig;

    RankMemoryRequest request = BuildRankMemoryRequest(config, layout, padded_state_dim);
    if (ValidateRankMemoryRequest(request) != Status::kOk) return Status::kInvalidConfig;

    TrainPlan plan{};
    plan.config = config;
    plan.layout = std::move(layout);
    plan.memory = BuildRankMemoryPlan(request);
    plan.gradient_buckets = BuildGradientBuckets(plan.layout.total_bytes, config.runtime.allreduce_bucket_bytes);
    plan.linear_ops = BuildLinearOps(config, plan.layout);
    plan.padded_state_dim = padded_state_dim;
    plan.state_capacity_bytes = static_cast<std::uint64_t>(padded_state_dim) * sizeof(StateValue);
    plan.gradient_carousel_slots = config.runtime.gradient_carousel_slots;
    plan.compatible_with_legacy_state_storage = plan.state_capacity_bytes <= sizeof(TrainStateStorage);
    if (plan.memory.total_rank_bytes == 0 || plan.gradient_buckets.empty() || plan.linear_ops.empty()) return Status::kInvalidConfig;

    *out = std::move(plan);
    return Status::kOk;
}

}  // namespace mgt