#pragma once

#include "mgt/static_contracts.hpp"
#include <array>
#include <cstddef>

namespace mgt {

enum class DType : std::uint32_t {
    kFloat32 = 1
};

inline constexpr std::uint32_t kResidualLinearBlocks = kResidualBlocks * 2U;
inline constexpr std::uint32_t kParamBlockCount = 4U + kResidualLinearBlocks * 2U + 2U;

struct ModelLayout {
    std::array<TensorBlockHeader, kParamBlockCount> blocks;
    std::uint64_t total_bytes;
    std::uint64_t total_params;
};

ModelLayout BuildModelLayout();

}  // namespace mgt