#include "mgt/memory_plan.hpp"

#include <algorithm>

namespace mgt {
namespace {

std::uint64_t Align256(std::uint64_t value) {
    return RoundUp64(value, 256ULL);
}

}  // namespace

Status ValidateRankMemoryRequest(const RankMemoryRequest& request) {
    if (request.batch_states_per_rank == 0 || request.model_bytes == 0 || request.model_params == 0) {
        return Status::kInvalidConfig;
    }
    if (request.state_storage_bytes == 0 || request.output_dim == 0 || request.physical_hd1 == 0 || request.physical_hd2 == 0) {
        return Status::kInvalidConfig;
    }
    if (request.gradient_carousel_slots == 0 || request.allreduce_bucket_bytes == 0) {
        return Status::kInvalidConfig;
    }
    if (request.model_bytes % sizeof(float) != 0) {
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
    plan.gradient_carousel_bytes = plan.gradient_bytes * request.gradient_carousel_slots;

    const std::uint64_t batch = request.batch_states_per_rank;
    plan.states_bytes = batch * request.state_storage_bytes;
    plan.labels_bytes = batch * request.output_dim * sizeof(float);
    plan.walk_meta_bytes = batch * sizeof(WalkMeta);

    const std::uint64_t hd1_activations = Align256(batch * request.physical_hd1 * sizeof(float));
    const std::uint64_t hd2_activations = Align256(batch * request.physical_hd2 * sizeof(float));
    const std::uint64_t residual_tape = Align256(batch * request.physical_hd2 * request.residual_blocks * 3ULL * sizeof(float));
    const std::uint64_t output_activations = Align256(batch * request.output_dim * sizeof(float));

    plan.layout_generate_bytes = Align256(plan.states_bytes + plan.labels_bytes + plan.walk_meta_bytes);
    plan.layout_forward_bytes = Align256(hd1_activations + hd2_activations + residual_tape + output_activations);
    plan.layout_backward_bytes = Align256(output_activations + hd2_activations + residual_tape + hd1_activations);
    plan.layout_allreduce_bytes = plan.gradient_bytes * request.gradient_carousel_slots;
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
        plan.gradient_carousel_bytes +
        plan.training_scratch_bytes);
    return plan;
}

}  // namespace mgt