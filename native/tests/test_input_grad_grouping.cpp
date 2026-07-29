#include "mgt/input_grad_grouping.hpp"

#include <cstdint>

namespace {

bool ResolvesTo(
    std::uint32_t state_len,
    std::uint32_t state_value_pad,
    std::uint32_t threads,
    std::uint32_t requested_positions,
    const mgt::InputGradGroupingLimits& limits,
    std::uint32_t expected_positions,
    std::uint64_t expected_shared_bytes) {
    mgt::InputGradGroupingConfig config{};
    return mgt::ResolveInputGradGrouping(
               state_len,
               state_value_pad,
               threads,
               requested_positions,
               limits,
               &config) == mgt::Status::kOk &&
           config.positions_per_block == expected_positions &&
           config.dynamic_shared_bytes == expected_shared_bytes;
}

}  // namespace

int main() {
    constexpr mgt::InputGradGroupingLimits a100{
        163840,
        163840,
        2,
        5,
    };
    constexpr mgt::InputGradGroupingLimits t4{
        65536,
        65536,
        2,
        5,
    };

    if (!ResolvesTo(72, 72, 96, 0, a100, 2, 56064)) return 1;
    if (!ResolvesTo(72, 72, 96, 0, t4, 1, 28032)) return 1;
    if (!ResolvesTo(72, 72, 96, 5, a100, 5, 140160)) return 1;
    if (!ResolvesTo(2, 4, 96, 0, a100, 2, 3840)) return 1;

    mgt::InputGradGroupingConfig config{};
    if (mgt::ResolveInputGradGrouping(72, 72, 96, 3, t4, &config) !=
        mgt::Status::kCapacityExceeded) {
        return 1;
    }
    if (mgt::ResolveInputGradGrouping(0, 72, 96, 0, a100, &config) !=
        mgt::Status::kInvalidConfig) {
        return 1;
    }
    if (mgt::ResolveInputGradGrouping(72, 72, 96, 6, a100, &config) !=
        mgt::Status::kInvalidConfig) {
        return 1;
    }
    return 0;
}
