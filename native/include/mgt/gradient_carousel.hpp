#pragma once

#include "mgt/train_plan.hpp"

#include <cstdint>
#include <vector>

namespace mgt {

struct GradientSlotPlan {
    std::uint32_t slot_id = 0;
    std::uint32_t bucket_id = 0;
    std::uint64_t offset_bytes = 0;
    std::uint64_t size_bytes = 0;
    GradientSlotState state = GradientSlotState::kFree;
};

struct GradientCarouselPlan {
    std::uint32_t slots_per_bucket = 0;
    std::vector<GradientSlotPlan> slots;
};

Status BuildGradientCarouselPlan(const TrainPlan& train_plan,
                                 GradientCarouselPlan* out);
Status AdvanceGradientSlot(GradientSlotPlan* slot,
                           GradientSlotState expected,
                           GradientSlotState next);
const char* GradientSlotStateName(GradientSlotState state);

}  // namespace mgt