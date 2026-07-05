#include "mgt/train_plan.hpp"

#include <cstdlib>

namespace {

bool CheckArchiveP888Plan() {
    const mgt::TrainConfig config = mgt::MakeArchiveP888TrainConfig();
    mgt::TrainPlan plan{};
    if (mgt::BuildTrainPlan(config, &plan) != mgt::Status::kOk) return false;

    if (plan.padded_state_dim != 80) return false;
    if (plan.state_capacity_bytes != 80) return false;
    if (!plan.compatible_with_legacy_state_storage) return false;
    if (plan.layout.logical_hd2 != 218) return false;
    if (plan.layout.physical_hd2 != 224) return false;
    if (plan.layout.output_dim != 1) return false;
    if (plan.layout.residual_blocks != 16) return false;
    if (plan.layout.blocks.size() != 70) return false;
    if (plan.gradient_buckets.empty()) return false;
    if (plan.gradient_carousel_slots != config.runtime.gradient_carousel_slots) return false;
    if (plan.memory.gradient_bytes != plan.layout.total_bytes) return false;
    if (plan.memory.layout_allreduce_bytes != plan.layout.total_bytes * config.runtime.gradient_carousel_slots) return false;
    if (plan.memory.total_rank_bytes <= plan.memory.training_scratch_bytes) return false;
    return true;
}

bool CheckSyntheticPuzzlePlan() {
    mgt::TrainConfig config = mgt::MakeArchiveP888TrainConfig();
    config.puzzle.group_id = 999;
    config.puzzle.raw_state_dim = 31;
    config.puzzle.state_value_count = 37;
    config.puzzle.move_count = 11;
    config.puzzle.state_alignment = 16;
    config.model.hd1 = 64;
    config.model.hd2 = 33;
    config.model.residual_blocks = 2;
    config.train.batch_size = 1234;
    config.runtime.allreduce_bucket_bytes = 4096;

    mgt::TrainPlan plan{};
    if (mgt::BuildTrainPlan(config, &plan) != mgt::Status::kOk) return false;
    if (plan.padded_state_dim != 32) return false;
    if (plan.state_capacity_bytes != 32) return false;
    if (plan.compatible_with_legacy_state_storage != true) return false;
    if (plan.layout.logical_hd2 != 33) return false;
    if (plan.layout.physical_hd2 != 40) return false;
    if (plan.layout.state_value_count != 37) return false;
    if (plan.layout.move_count != 11) return false;
    if (plan.layout.blocks.size() != 14) return false;
    if (plan.gradient_buckets.size() < 2) return false;
    if (plan.memory.states_bytes != 1234ULL * 32ULL) return false;
    return true;
}
bool CheckLinearOpManifest() {
    mgt::TrainConfig config = mgt::MakeArchiveP888TrainConfig();
    config.train.batch_size = 32768;

    mgt::TrainPlan plan{};
    if (mgt::BuildTrainPlan(config, &plan) != mgt::Status::kOk) return false;
    if (plan.linear_ops.size() != 104) return false;

    const mgt::LinearOpPlan& input_forward = plan.linear_ops.front();
    if (input_forward.backend != mgt::LinearBackendClass::kCustomKernel) return false;
    if (input_forward.output_rows != config.train.batch_size) return false;
    if (input_forward.output_cols != plan.layout.physical_hd1) return false;

    const mgt::LinearOpPlan& hidden_forward = plan.linear_ops[1];
    if (hidden_forward.role != mgt::ParamBlockRole::kHiddenWeight) return false;
    if (hidden_forward.phase != mgt::LinearOpPhase::kForward) return false;
    if (hidden_forward.m != config.train.batch_size || hidden_forward.n != plan.layout.physical_hd2 || hidden_forward.k != plan.layout.physical_hd1) return false;
    if (hidden_forward.backend != mgt::LinearBackendClass::kLibraryTensorOp) return false;

    const mgt::LinearOpPlan& input_grad = plan.linear_ops.back();
    if (input_grad.role != mgt::ParamBlockRole::kInputEmbedding) return false;
    if (input_grad.phase != mgt::LinearOpPhase::kBackwardWeights) return false;
    if (input_grad.m != config.puzzle.raw_state_dim * config.puzzle.state_value_count) return false;
    if (input_grad.n != plan.layout.physical_hd1) return false;
    if (input_grad.k != config.train.batch_size) return false;
    if (input_grad.backend != mgt::LinearBackendClass::kLibraryTensorOp) return false;

    config.puzzle.raw_state_dim = 17;
    config.puzzle.state_value_count = 9;
    config.model.hd1 = 31;
    config.model.hd2 = 19;
    config.model.residual_blocks = 2;
    config.train.batch_size = 512;
    if (mgt::BuildTrainPlan(config, &plan) != mgt::Status::kOk) return false;
    if (plan.linear_ops.size() != 20) return false;
    if (plan.linear_ops.back().backend != mgt::LinearBackendClass::kLibraryTensorOp) return false;
    if (plan.linear_ops.back().m != 153) return false;
    if (plan.linear_ops.back().n != plan.layout.physical_hd1) return false;
    if (plan.linear_ops.back().k != 512) return false;

    config.puzzle.raw_state_dim = 128;
    config.puzzle.state_value_count = 64;
    config.model.hd1 = 4097;
    config.model.hd2 = 513;
    config.model.residual_blocks = 1;
    config.train.batch_size = 16384;
    if (mgt::BuildTrainPlan(config, &plan) != mgt::Status::kOk) return false;
    if (plan.linear_ops.back().backend != mgt::LinearBackendClass::kCutlassLargeKReductionCandidate) return false;
    if (plan.linear_ops.back().m != 8192) return false;
    if (plan.linear_ops.back().n != plan.layout.physical_hd1) return false;
    if (plan.linear_ops.back().k != 16384) return false;
    return true;
}

bool CheckInvalidPlans() {
    mgt::TrainConfig config = mgt::MakeArchiveP888TrainConfig();
    config.puzzle.raw_state_dim = 0;
    mgt::TrainPlan plan{};
    if (mgt::BuildTrainPlan(config, &plan) != mgt::Status::kInvalidConfig) return false;

    config = mgt::MakeArchiveP888TrainConfig();
    config.model.output_dim = 0;
    if (mgt::BuildTrainPlan(config, &plan) != mgt::Status::kInvalidConfig) return false;

    config = mgt::MakeArchiveP888TrainConfig();
    config.runtime.gradient_carousel_slots = 0;
    if (mgt::BuildTrainPlan(config, &plan) != mgt::Status::kInvalidConfig) return false;

    return true;
}

}  // namespace

int main() {
    if (!CheckArchiveP888Plan()) return EXIT_FAILURE;
    if (!CheckSyntheticPuzzlePlan()) return EXIT_FAILURE;
    if (!CheckLinearOpManifest()) return EXIT_FAILURE;
    if (!CheckInvalidPlans()) return EXIT_FAILURE;
    return EXIT_SUCCESS;
}