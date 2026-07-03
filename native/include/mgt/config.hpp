#pragma once

#include <cstdint>

namespace mgt {

inline constexpr std::uint32_t kStateLen = 72;
inline constexpr std::uint32_t kStateAlignment = 16;
inline constexpr std::uint32_t kStateValuePad = 72;
inline constexpr std::uint32_t kMoveCount = 18;
inline constexpr std::uint32_t kOutputDim = 1;
inline constexpr std::uint32_t kHd1 = 2556;
inline constexpr std::uint32_t kHd2 = 218;
inline constexpr std::uint32_t kResidualBlocks = 16;

constexpr std::uint32_t RoundUp(std::uint32_t value, std::uint32_t alignment) {
    return ((value + alignment - 1U) / alignment) * alignment;
}

inline constexpr std::uint32_t kStateStorageLen =
    RoundUp(kStateLen + 4U, kStateAlignment);

static_assert(kStateStorageLen == 80);

}  // namespace mgt