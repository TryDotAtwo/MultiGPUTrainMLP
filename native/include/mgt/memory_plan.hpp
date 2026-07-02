#pragma once

#include "mgt/model_layout.hpp"
#include <cstdint>

namespace mgt {

struct RankMemoryRequest {
    std::uint64_t batch_states_per_rank;
    std::uint64_t model_bytes;
    std::uint64_t model_params;
};

struct RankMemoryPlan {
    std::uint64_t weights_bytes;
    std::uint64_t optimizer_m_bytes;
    std::uint64_t optimizer_v_bytes;
    std::uint64_t gradient_bytes;
    std::uint64_t states_bytes;
    std::uint64_t labels_bytes;
    std::uint64_t walk_meta_bytes;
    std::uint64_t layout_generate_bytes;
    std::uint64_t layout_forward_bytes;
    std::uint64_t layout_backward_bytes;
    std::uint64_t layout_allreduce_bytes;
    std::uint64_t layout_checkpoint_bytes;
    std::uint64_t training_scratch_bytes;
    std::uint64_t total_rank_bytes;
};

Status ValidateRankMemoryRequest(const RankMemoryRequest& request);
RankMemoryPlan BuildRankMemoryPlan(const RankMemoryRequest& request);

}  // namespace mgt