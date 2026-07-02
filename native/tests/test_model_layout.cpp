#include "mgt/model_layout.hpp"
#include <cstdlib>

int main() {
    const auto layout = mgt::BuildModelLayout();
    if (layout.blocks[0].rows != mgt::kStateLen * mgt::kStateValuePad) return EXIT_FAILURE;
    if (layout.blocks[0].cols != mgt::kHd1) return EXIT_FAILURE;
    if (layout.blocks[mgt::kParamBlockCount - 2].cols != mgt::kOutputDim) return EXIT_FAILURE;
    if (layout.total_params == 0) return EXIT_FAILURE;
    if (layout.total_bytes % 64 != 0) return EXIT_FAILURE;
    for (std::uint32_t i = 1; i < mgt::kParamBlockCount; ++i) {
        if (layout.blocks[i].offset_bytes <= layout.blocks[i - 1].offset_bytes) return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}