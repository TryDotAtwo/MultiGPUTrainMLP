#pragma once

#include "mgt/static_contracts.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace mgt {

enum class DType : std::uint32_t {
    kFloat32 = 1,
    kFloat16 = 2
};

enum class ParamBlockRole : std::uint32_t {
    kInputEmbedding = 1,
    kInputBias = 2,
    kHiddenWeight = 3,
    kHiddenBias = 4,
    kResidualFc1Weight = 5,
    kResidualFc1Bias = 6,
    kResidualFc2Weight = 7,
    kResidualFc2Bias = 8,
    kOutputWeight = 9,
    kOutputBias = 10
};

inline constexpr std::uint32_t kResidualLinearBlocks = kResidualBlocks * 2U;
inline constexpr std::uint32_t kParamBlockCount = 4U + kResidualLinearBlocks * 2U + 2U;

struct ParamBlockPlan {
    TensorBlockHeader header{};
    ParamBlockRole role = ParamBlockRole::kInputEmbedding;
    std::uint32_t logical_rows = 0;
    std::uint32_t logical_cols = 0;
    std::uint32_t block_index = 0;
};

struct ModelLayout {
    std::vector<TensorBlockHeader> blocks;
    std::vector<ParamBlockPlan> param_blocks;
    std::uint64_t total_bytes = 0;
    std::uint64_t total_params = 0;
    std::uint64_t logical_params = 0;
    std::uint32_t state_dim = 0;
    std::uint32_t padded_state_dim = 0;
    std::uint32_t state_value_count = 0;
    std::uint32_t move_count = 0;
    std::uint32_t logical_hd1 = 0;
    std::uint32_t physical_hd1 = 0;
    std::uint32_t logical_hd2 = 0;
    std::uint32_t physical_hd2 = 0;
    std::uint32_t residual_blocks = 0;
    std::uint32_t output_dim = 0;
};

ModelLayout BuildModelLayout();
ModelLayout BuildModelLayout(const PuzzleSpec& puzzle, const ModelSpec& model);

}  // namespace mgt