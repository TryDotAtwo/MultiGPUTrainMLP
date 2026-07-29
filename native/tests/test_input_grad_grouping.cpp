#include "mgt/input_grad_grouping.hpp"

#include <cstdint>

namespace {

bool ResolvesTo(
    std::uint32_t state_len,
    std::uint32_t requested_positions,
    std::uint64_t shared_bytes_per_position,
    const mgt::InputGradGroupingLimits& limits,
    std::uint32_t expected_positions,
    std::uint64_t expected_shared_bytes) {
    mgt::InputGradGroupingConfig config{};
    return mgt::ResolveInputGradGrouping(
               state_len,
               requested_positions,
               shared_bytes_per_position,
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
        4,
    };
    constexpr mgt::InputGradGroupingLimits t4{
        65536,
        65536,
        2,
        4,
    };
    constexpr mgt::InputGradGroupingLimits constrained{
        32768,
        65536,
        2,
        4,
    };

    constexpr std::uint64_t shared_bytes_per_position = 72ULL * 32ULL * sizeof(float);
    if (!ResolvesTo(72, 0, shared_bytes_per_position, a100, 4, 36864)) return 1;
    if (!ResolvesTo(72, 0, shared_bytes_per_position, t4, 3, 27648)) return 1;
    if (!ResolvesTo(72, 4, shared_bytes_per_position, t4, 4, 36864)) return 1;
    if (!ResolvesTo(2, 0, shared_bytes_per_position, a100, 2, 18432)) return 1;

    mgt::InputGradGroupingConfig config{};
    if (mgt::ResolveInputGradGrouping(72, 4, shared_bytes_per_position, constrained, &config) !=
        mgt::Status::kCapacityExceeded) {
        return 1;
    }
    if (mgt::ResolveInputGradGrouping(0, 0, shared_bytes_per_position, a100, &config) !=
        mgt::Status::kInvalidConfig) {
        return 1;
    }
    if (mgt::ResolveInputGradGrouping(72, 5, shared_bytes_per_position, a100, &config) !=
        mgt::Status::kInvalidConfig) {
        return 1;
    }
    return 0;
}
