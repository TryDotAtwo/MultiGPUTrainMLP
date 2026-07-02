#include "mgt/model_layout.hpp"

namespace mgt {
namespace {

std::uint64_t Align64(std::uint64_t value) {
    return ((value + 63ULL) / 64ULL) * 64ULL;
}

void AddBlock(ModelLayout* layout, std::uint32_t* index, std::uint32_t rows, std::uint32_t cols) {
    const std::uint64_t size = static_cast<std::uint64_t>(rows) * cols * sizeof(float);
    layout->blocks[*index] = TensorBlockHeader{
        layout->total_bytes,
        size,
        rows,
        cols,
        static_cast<std::uint32_t>(DType::kFloat32),
        0};
    layout->total_bytes = Align64(layout->total_bytes + size);
    layout->total_params += static_cast<std::uint64_t>(rows) * cols;
    ++(*index);
}

}  // namespace

ModelLayout BuildModelLayout() {
    ModelLayout layout{};
    std::uint32_t index = 0;
    AddBlock(&layout, &index, kStateLen * kStateValuePad, kHd1);
    AddBlock(&layout, &index, 1, kHd1);
    AddBlock(&layout, &index, kHd1, kHd2);
    AddBlock(&layout, &index, 1, kHd2);
    for (std::uint32_t block = 0; block < kResidualBlocks; ++block) {
        AddBlock(&layout, &index, kHd2, kHd2);
        AddBlock(&layout, &index, 1, kHd2);
        AddBlock(&layout, &index, kHd2, kHd2);
        AddBlock(&layout, &index, 1, kHd2);
    }
    AddBlock(&layout, &index, kHd2, kOutputDim);
    AddBlock(&layout, &index, 1, kOutputDim);
    return layout;
}

}  // namespace mgt