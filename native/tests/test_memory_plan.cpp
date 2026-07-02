#include "mgt/memory_plan.hpp"
#include <cstdlib>

int main() {
    const auto layout = mgt::BuildModelLayout();
    const mgt::RankMemoryRequest request{100021, layout.total_bytes, layout.total_params};
    const auto plan = mgt::BuildRankMemoryPlan(request);
    if (plan.weights_bytes != layout.total_bytes) return EXIT_FAILURE;
    if (plan.gradient_bytes != layout.total_bytes) return EXIT_FAILURE;
    if (plan.optimizer_m_bytes != layout.total_bytes) return EXIT_FAILURE;
    if (plan.optimizer_v_bytes != layout.total_bytes) return EXIT_FAILURE;
    if (plan.states_bytes != 100021ULL * sizeof(mgt::TrainState80)) return EXIT_FAILURE;
    if (plan.labels_bytes != 100021ULL * sizeof(float)) return EXIT_FAILURE;
    if (plan.walk_meta_bytes != 100021ULL * sizeof(mgt::WalkMeta)) return EXIT_FAILURE;
    if (plan.layout_generate_bytes < plan.states_bytes + plan.labels_bytes + plan.walk_meta_bytes) return EXIT_FAILURE;
    if (plan.training_scratch_bytes < plan.layout_generate_bytes) return EXIT_FAILURE;
    if (plan.training_scratch_bytes < plan.layout_forward_bytes) return EXIT_FAILURE;
    if (plan.training_scratch_bytes < plan.layout_backward_bytes) return EXIT_FAILURE;
    if (plan.training_scratch_bytes < plan.layout_allreduce_bytes) return EXIT_FAILURE;
    if (plan.total_rank_bytes <= plan.training_scratch_bytes) return EXIT_FAILURE;

    const mgt::RankMemoryRequest bad_request{0, layout.total_bytes, layout.total_params};
    if (mgt::ValidateRankMemoryRequest(bad_request) != mgt::Status::kInvalidConfig) return EXIT_FAILURE;
    if (mgt::ValidateRankMemoryRequest(request) != mgt::Status::kOk) return EXIT_FAILURE;
    return EXIT_SUCCESS;
}