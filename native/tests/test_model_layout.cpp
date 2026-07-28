#include "mgt/model_layout.hpp"
#include <cstdlib>

int main() {
    const auto layout = mgt::BuildModelLayout();
    if (layout.blocks.size() != mgt::kParamBlockCount) return EXIT_FAILURE;
    if (layout.blocks[0].rows != mgt::kStateLen * mgt::kStateValuePad) return EXIT_FAILURE;
    if (layout.blocks[0].cols != layout.physical_hd1) return EXIT_FAILURE;
    if (layout.logical_hd1 != mgt::kHd1) return EXIT_FAILURE;
    if (layout.physical_hd1 != mgt::RoundUp(mgt::kHd1, mgt::kHiddenAlignment)) return EXIT_FAILURE;
    if (layout.logical_hd2 != mgt::kHd2) return EXIT_FAILURE;
    if (layout.physical_hd2 != mgt::RoundUp(mgt::kHd2, mgt::kHiddenAlignment)) return EXIT_FAILURE;
    if (layout.blocks[layout.blocks.size() - 2].cols != mgt::kOutputDim) return EXIT_FAILURE;
    if (layout.batch_norms.size() != mgt::P888TrainingContract::kBatchNormSites) return EXIT_FAILURE;
    if (layout.batch_norms.front().name != "input") return EXIT_FAILURE;
    if (layout.batch_norms.front().features != mgt::kHd1) return EXIT_FAILURE;
    if (layout.batch_norms[1].name != "hidden") return EXIT_FAILURE;
    if (layout.batch_norms[1].features != mgt::kHd2) return EXIT_FAILURE;
    if (layout.batch_norms[2].name != "residual.0.fc1") return EXIT_FAILURE;
    if (layout.batch_norms[3].name != "residual.0.fc2") return EXIT_FAILURE;
    if (layout.batch_norms.back().name != "residual.15.fc2") return EXIT_FAILURE;
    std::uint64_t affine_cursor = 0;
    std::uint64_t running_cursor = 0;
    for (const auto& bn : layout.batch_norms) {
        if (bn.gamma_offset != affine_cursor) return EXIT_FAILURE;
        affine_cursor += bn.features;
        if (bn.beta_offset != affine_cursor) return EXIT_FAILURE;
        affine_cursor += bn.features;
        if (bn.running_mean_offset != running_cursor) return EXIT_FAILURE;
        running_cursor += bn.features;
        if (bn.running_var_offset != running_cursor) return EXIT_FAILURE;
        running_cursor += bn.features;
    }
    if (layout.batch_norm_trainable_params != affine_cursor) return EXIT_FAILURE;
    if (layout.batch_norm_running_state_params != running_cursor) return EXIT_FAILURE;
    if (layout.total_params == 0) return EXIT_FAILURE;
    if (layout.total_bytes % 64 != 0) return EXIT_FAILURE;
    for (std::uint32_t i = 1; i < layout.blocks.size(); ++i) {
        if (layout.blocks[i].offset_bytes <= layout.blocks[i - 1].offset_bytes) return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}