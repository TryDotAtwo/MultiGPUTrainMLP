#pragma once

#include "mgt/model_layout.hpp"

#include <cstdint>

namespace mgt {

struct RankMemoryRequest {
    std::uint64_t batch_states_per_rank = 0;
    std::uint64_t model_bytes = 0;
    std::uint64_t model_params = 0;
    std::uint32_t state_storage_bytes = kStateStorageLen;
    std::uint32_t output_dim = kOutputDim;
    std::uint32_t physical_hd1 = RoundUp(kHd1, kHiddenAlignment);
    std::uint32_t physical_hd2 = RoundUp(kHd2, kHiddenAlignment);
    std::uint32_t residual_blocks = kResidualBlocks;
    std::uint32_t gradient_carousel_slots = kGradientCarouselSlots;
    std::uint64_t allreduce_bucket_bytes = kDefaultAllreduceBucketBytes;
};

struct RankMemoryPlan {
    std::uint64_t weights_bytes = 0;
    std::uint64_t optimizer_m_bytes = 0;
    std::uint64_t optimizer_v_bytes = 0;
    std::uint64_t gradient_bytes = 0;
    std::uint64_t gradient_carousel_bytes = 0;
    std::uint64_t states_bytes = 0;
    std::uint64_t labels_bytes = 0;
    std::uint64_t walk_meta_bytes = 0;
    std::uint64_t layout_generate_bytes = 0;
    std::uint64_t layout_forward_bytes = 0;
    std::uint64_t layout_backward_bytes = 0;
    std::uint64_t layout_allreduce_bytes = 0;
    std::uint64_t layout_checkpoint_bytes = 0;
    std::uint64_t training_scratch_bytes = 0;
    std::uint64_t total_rank_bytes = 0;
};

Status ValidateRankMemoryRequest(const RankMemoryRequest& request);
RankMemoryPlan BuildRankMemoryPlan(const RankMemoryRequest& request);

}  // namespace mgt