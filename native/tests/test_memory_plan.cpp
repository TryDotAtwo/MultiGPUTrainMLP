#include "mgt/memory_plan.hpp"
#include <cstdlib>

int main() {
    const auto layout = mgt::BuildModelLayout();
    mgt::RankMemoryRequest request{};
    request.batch_states_per_rank = 100021;
    request.model_bytes = layout.total_bytes;
    request.model_params = layout.total_params;
    request.state_storage_bytes = mgt::kStateStorageLen;
    request.output_dim = layout.output_dim;
    request.physical_hd1 = layout.physical_hd1;
    request.physical_hd2 = layout.physical_hd2;
    request.residual_blocks = layout.residual_blocks;
    request.gradient_carousel_slots = 3;
    request.allreduce_bucket_bytes = 4ULL * 1024ULL * 1024ULL;

    const auto plan = mgt::BuildRankMemoryPlan(request);
    if (plan.weights_bytes < layout.total_bytes) return EXIT_FAILURE;
    if (plan.gradient_bytes < layout.total_bytes) return EXIT_FAILURE;
    if (plan.gradient_carousel_bytes != plan.gradient_bytes * request.gradient_carousel_slots) return EXIT_FAILURE;
    if (plan.optimizer_m_bytes < layout.total_bytes) return EXIT_FAILURE;
    if (plan.optimizer_v_bytes < layout.total_bytes) return EXIT_FAILURE;
    if (plan.states_bytes < 100021ULL * mgt::kStateStorageLen) return EXIT_FAILURE;
    if (plan.labels_bytes < 100021ULL * sizeof(float)) return EXIT_FAILURE;
    if (plan.walk_meta_bytes < 100021ULL * sizeof(mgt::WalkMeta)) return EXIT_FAILURE;
    if (plan.layout_generate_bytes < plan.states_bytes + plan.labels_bytes + plan.walk_meta_bytes) return EXIT_FAILURE;
    if (plan.training_scratch_bytes < plan.layout_generate_bytes) return EXIT_FAILURE;
    if (plan.training_scratch_bytes < plan.layout_forward_bytes) return EXIT_FAILURE;
    if (plan.training_scratch_bytes < plan.layout_backward_bytes) return EXIT_FAILURE;
    if (plan.training_scratch_bytes < plan.layout_allreduce_bytes) return EXIT_FAILURE;
    if (plan.layout_allreduce_bytes != plan.gradient_bytes * request.gradient_carousel_slots) return EXIT_FAILURE;
    if (plan.total_rank_bytes <= plan.training_scratch_bytes) return EXIT_FAILURE;

    mgt::RankMemoryRequest bad_request = request;
    bad_request.batch_states_per_rank = 0;
    if (mgt::ValidateRankMemoryRequest(bad_request) != mgt::Status::kInvalidConfig) return EXIT_FAILURE;
    if (mgt::ValidateRankMemoryRequest(request) != mgt::Status::kOk) return EXIT_FAILURE;
    return EXIT_SUCCESS;
}