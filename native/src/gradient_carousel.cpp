#include "mgt/gradient_carousel.hpp"

#include <cstddef>
#include <utility>

namespace mgt {

const char* GradientSlotStateName(GradientSlotState state) {
    switch (state) {
        case GradientSlotState::kFree: return "FREE";
        case GradientSlotState::kFilling: return "FILLING";
        case GradientSlotState::kReady: return "READY";
        case GradientSlotState::kInComm: return "IN_COMM";
        case GradientSlotState::kReduced: return "REDUCED";
        case GradientSlotState::kInAdam: return "IN_ADAM";
    }
    return "UNKNOWN";
}

Status BuildGradientCarouselPlan(const TrainPlan& train_plan,
                                 GradientCarouselPlan* out) {
    if (out == nullptr || train_plan.gradient_carousel_slots == 0 || train_plan.gradient_buckets.empty()) {
        return Status::kInvalidConfig;
    }
    GradientCarouselPlan plan{};
    plan.slots_per_bucket = train_plan.gradient_carousel_slots;
    plan.slots.reserve(static_cast<std::size_t>(plan.slots_per_bucket) * train_plan.gradient_buckets.size());
    for (const GradientBucketPlan& bucket : train_plan.gradient_buckets) {
        if (bucket.size_bytes == 0 || bucket.param_count == 0) return Status::kInvalidConfig;
        for (std::uint32_t slot = 0; slot < plan.slots_per_bucket; ++slot) {
            plan.slots.push_back(GradientSlotPlan{
                slot,
                bucket.bucket_id,
                bucket.offset_bytes,
                bucket.size_bytes,
                GradientSlotState::kFree});
        }
    }
    *out = std::move(plan);
    return Status::kOk;
}

Status AdvanceGradientSlot(GradientSlotPlan* slot,
                           GradientSlotState expected,
                           GradientSlotState next) {
    if (slot == nullptr || slot->state != expected) return Status::kInvalidConfig;
    const bool allowed =
        (expected == GradientSlotState::kFree && next == GradientSlotState::kFilling) ||
        (expected == GradientSlotState::kFilling && next == GradientSlotState::kReady) ||
        (expected == GradientSlotState::kReady && next == GradientSlotState::kInComm) ||
        (expected == GradientSlotState::kInComm && next == GradientSlotState::kReduced) ||
        (expected == GradientSlotState::kReduced && next == GradientSlotState::kInAdam) ||
        (expected == GradientSlotState::kInAdam && next == GradientSlotState::kFree);
    if (!allowed) return Status::kInvalidConfig;
    slot->state = next;
    return Status::kOk;
}

}  // namespace mgt