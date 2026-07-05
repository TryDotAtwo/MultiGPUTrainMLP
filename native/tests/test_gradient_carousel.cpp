#include "mgt/gradient_carousel.hpp"

#include <cstdlib>

namespace {

bool CheckCarouselForArchivePlan() {
    mgt::TrainConfig config = mgt::MakeArchiveP888TrainConfig();
    config.runtime.allreduce_bucket_bytes = 4096;
    mgt::TrainPlan train_plan{};
    if (mgt::BuildTrainPlan(config, &train_plan) != mgt::Status::kOk) return false;

    mgt::GradientCarouselPlan carousel{};
    if (mgt::BuildGradientCarouselPlan(train_plan, &carousel) != mgt::Status::kOk) return false;
    if (carousel.slots_per_bucket != train_plan.gradient_carousel_slots) return false;
    if (carousel.slots.size() != train_plan.gradient_buckets.size() * train_plan.gradient_carousel_slots) return false;

    for (std::size_t i = 0; i < carousel.slots.size(); ++i) {
        const auto& slot = carousel.slots[i];
        const auto& bucket = train_plan.gradient_buckets[slot.bucket_id];
        if (slot.slot_id >= train_plan.gradient_carousel_slots) return false;
        if (slot.offset_bytes != bucket.offset_bytes) return false;
        if (slot.size_bytes != bucket.size_bytes) return false;
        if (slot.state != mgt::GradientSlotState::kFree) return false;
    }
    return true;
}

bool CheckStateMachine() {
    mgt::GradientSlotPlan slot{};
    if (mgt::GradientSlotStateName(slot.state)[0] == '\0') return false;
    if (mgt::AdvanceGradientSlot(&slot, mgt::GradientSlotState::kReady, mgt::GradientSlotState::kInComm) != mgt::Status::kInvalidConfig) return false;
    if (mgt::AdvanceGradientSlot(&slot, mgt::GradientSlotState::kFree, mgt::GradientSlotState::kFilling) != mgt::Status::kOk) return false;
    if (mgt::AdvanceGradientSlot(&slot, mgt::GradientSlotState::kFilling, mgt::GradientSlotState::kReady) != mgt::Status::kOk) return false;
    if (mgt::AdvanceGradientSlot(&slot, mgt::GradientSlotState::kReady, mgt::GradientSlotState::kReduced) != mgt::Status::kInvalidConfig) return false;
    if (mgt::AdvanceGradientSlot(&slot, mgt::GradientSlotState::kReady, mgt::GradientSlotState::kInComm) != mgt::Status::kOk) return false;
    if (mgt::AdvanceGradientSlot(&slot, mgt::GradientSlotState::kInComm, mgt::GradientSlotState::kReduced) != mgt::Status::kOk) return false;
    if (mgt::AdvanceGradientSlot(&slot, mgt::GradientSlotState::kReduced, mgt::GradientSlotState::kInAdam) != mgt::Status::kOk) return false;
    if (mgt::AdvanceGradientSlot(&slot, mgt::GradientSlotState::kInAdam, mgt::GradientSlotState::kFree) != mgt::Status::kOk) return false;
    return slot.state == mgt::GradientSlotState::kFree;
}

}  // namespace

int main() {
    if (!CheckCarouselForArchivePlan()) return EXIT_FAILURE;
    if (!CheckStateMachine()) return EXIT_FAILURE;
    return EXIT_SUCCESS;
}