#include "mgt/memory_plan.hpp"
#include <algorithm>

namespace mgt {
namespace {

std::uint64_t Align256(std::uint64_t value) {
    return ((value + 255ULL) / 256ULL) * 256ULL;
}

}  // namespace

Status ValidateRankMemoryRequest(const RankMemoryRequest& request) {
    if (request.batch_states_per_rank == 0 || request.model_bytes == 0 || request.model_params == 0) {
        return Status::kInvalidConfig;
    }
    if (request.model_bytes % alignof(TensorBlockHeader) != 0 && request.model_bytes % sizeof(float) != 0) {
        return Status::kInvalidConfig;
    }
    return Status::kOk;
}

RankMemoryPlan BuildRankMemoryPlan(const RankMemoryRequest& request) {
    RankMemoryPlan plan{};
    if (ValidateRankMemoryRequest(request) != Status::kOk) return plan;

    plan.weights_bytes = request.model_bytes;
    plan.optimizer_m_bytes = request.model_bytes;
    plan.optimizer_v_bytes = request.model_bytes;
    plan.gradient_bytes = request.model_bytes;

    plan.states_bytes = request.batch_states_per_rank * sizeof(TrainState80);
    plan.labels_bytes = request.batch_states_per_rank * sizeof(float);
    plan.walk_meta_bytes = request.batch_states_per_rank * sizeof(WalkMeta);

    const std::uint64_t batch = request.batch_states_per_rank;
    const std::uint64_t hd1_activations = Align256(batch * kHd1 * sizeof(float));
    const std::uint64_t hd2_activations = Align256(batch * kHd2 * sizeof(float));
    const std::uint64_t output_activations = Align256(batch * kOutputDim * sizeof(float));

    plan.layout_generate_bytes = Align256(plan.states_bytes + plan.labels_bytes + plan.walk_meta_bytes);
    plan.layout_forward_bytes = Align256(hd1_activations + hd2_activations + output_activations);
    plan.layout_backward_bytes = Align256(output_activations + hd2_activations + hd1_activations);
    plan.layout_allreduce_bytes = Align256(plan.gradient_bytes * 2ULL);
    plan.layout_checkpoint_bytes = Align256(plan.weights_bytes + 4096ULL);
    plan.training_scratch_bytes = std::max({
        plan.layout_generate_bytes,
        plan.layout_forward_bytes,
        plan.layout_backward_bytes,
        plan.layout_allreduce_bytes,
        plan.layout_checkpoint_bytes});

    plan.total_rank_bytes = Align256(
        plan.weights_bytes +
        plan.optimizer_m_bytes +
        plan.optimizer_v_bytes +
        plan.gradient_bytes +
        plan.training_scratch_bytes);
    return plan;
}

}  // namespace mgt