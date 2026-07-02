#pragma once

#include <cstdint>

namespace mgt_cuda {

struct DeviceLaunchConfig {
    std::uint32_t blocks;
    std::uint32_t threads;
};

DeviceLaunchConfig Build1DLaunchConfig(std::uint64_t items, std::uint32_t threads_per_block);

}  // namespace mgt_cuda