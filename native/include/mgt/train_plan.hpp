#pragma once

#include "mgt/memory_plan.hpp"

#include <cstdint>
#include <vector>

namespace mgt {

struct GradientBucketPlan {
    std::uint32_t bucket_id = 0;
    std::uint64_t offset_bytes = 0;
    std::uint64_t size_bytes = 0;
    std::uint64_t param_offset = 0;
    std::uint64_t param_count = 0;
};

enum class GradientSlotState : std::uint32_t {
    kFree = 0,
    kFilling = 1,
    kReady = 2,
    kInComm = 3,
    kReduced = 4,
    kInAdam = 5
};

enum class LinearOpPhase : std::uint32_t {
    kForward = 1,
    kBackwardWeights = 2,
    kBackwardInput = 3
};

enum class MatrixTranspose : std::uint32_t {
    kNo = 0,
    kYes = 1
};

enum class LinearBackendClass : std::uint32_t {
    kCustomKernel = 1,
    kLibraryTensorOp = 2,
    kCutlassLargeKReductionCandidate = 3
};

struct LinearOpPlan {
    std::uint32_t op_id = 0;
    LinearOpPhase phase = LinearOpPhase::kForward;
    ParamBlockRole role = ParamBlockRole::kHiddenWeight;
    std::uint32_t block_index = 0;
    MatrixTranspose transpose_a = MatrixTranspose::kNo;
    MatrixTranspose transpose_b = MatrixTranspose::kNo;
    std::uint32_t m = 0;
    std::uint32_t n = 0;
    std::uint32_t k = 0;
    std::uint32_t lhs_rows = 0;
    std::uint32_t lhs_cols = 0;
    std::uint32_t rhs_rows = 0;
    std::uint32_t rhs_cols = 0;
    std::uint32_t output_rows = 0;
    std::uint32_t output_cols = 0;
    LinearBackendClass backend = LinearBackendClass::kLibraryTensorOp;
    bool tensor_core_eligible = false;
    bool logical_tail_masked = false;
};
struct TrainPlan {
    TrainConfig config{};
    ModelLayout layout{};
    RankMemoryPlan memory{};
    std::vector<GradientBucketPlan> gradient_buckets;
    std::vector<LinearOpPlan> linear_ops;
    std::uint32_t padded_state_dim = 0;
    std::uint64_t state_capacity_bytes = 0;
    std::uint32_t gradient_carousel_slots = 0;
    bool compatible_with_legacy_state_storage = false;
};

Status ValidateTrainConfig(const TrainConfig& config);
Status BuildTrainPlan(const TrainConfig& config, TrainPlan* out);
RankMemoryRequest BuildRankMemoryRequest(const TrainConfig& config,
                                         const ModelLayout& layout,
                                         std::uint32_t padded_state_dim);

}  // namespace mgt