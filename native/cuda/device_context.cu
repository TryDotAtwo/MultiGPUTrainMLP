#include "mgt_cuda/device_context.cuh"

namespace mgt_cuda {

DeviceLaunchConfig Build1DLaunchConfig(std::uint64_t items, std::uint32_t threads_per_block) {
    if (threads_per_block == 0) {
        return DeviceLaunchConfig{0, 0};
    }
    const std::uint64_t blocks = (items + threads_per_block - 1ULL) / threads_per_block;
    return DeviceLaunchConfig{static_cast<std::uint32_t>(blocks), threads_per_block};
}

}  // namespace mgt_cuda