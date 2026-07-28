#include "mgt/model_layout.hpp"

namespace mgt {
namespace {

std::uint64_t Align64(std::uint64_t value) {
    return RoundUp64(value, 64ULL);
}

void AddBatchNorm(ModelLayout* layout, std::string name, std::uint32_t features) {
    BatchNormSlice slice{};
    slice.name = std::move(name);
    slice.features = features;
    slice.gamma_offset = layout->batch_norm_trainable_params;
    layout->batch_norm_trainable_params += features;
    slice.beta_offset = layout->batch_norm_trainable_params;
    layout->batch_norm_trainable_params += features;
    slice.running_mean_offset = layout->batch_norm_running_state_params;
    layout->batch_norm_running_state_params += features;
    slice.running_var_offset = layout->batch_norm_running_state_params;
    layout->batch_norm_running_state_params += features;
    layout->batch_norms.push_back(std::move(slice));
}
void AddBlock(ModelLayout* layout,
              ParamBlockRole role,
              std::uint32_t logical_rows,
              std::uint32_t logical_cols,
              std::uint32_t physical_rows,
              std::uint32_t physical_cols,
              std::uint32_t block_index) {
    const std::uint64_t physical_params = static_cast<std::uint64_t>(physical_rows) * physical_cols;
    const std::uint64_t logical_params = static_cast<std::uint64_t>(logical_rows) * logical_cols;
    const std::uint64_t size = physical_params * sizeof(float);
    TensorBlockHeader header{
        layout->total_bytes,
        size,
        physical_rows,
        physical_cols,
        static_cast<std::uint32_t>(DType::kFloat32),
        0};
    layout->blocks.push_back(header);
    layout->param_blocks.push_back(ParamBlockPlan{header, role, logical_rows, logical_cols, block_index});
    layout->total_bytes = Align64(layout->total_bytes + size);
    layout->total_params += physical_params;
    layout->logical_params += logical_params;
}

}  // namespace

ModelLayout BuildModelLayout() {
    return BuildModelLayout(PuzzleSpec{}, ModelSpec{});
}

ModelLayout BuildModelLayout(const PuzzleSpec& puzzle, const ModelSpec& model) {
    ModelLayout layout{};
    layout.state_dim = puzzle.raw_state_dim;
    layout.padded_state_dim = RoundUp(puzzle.raw_state_dim, puzzle.state_alignment);
    layout.state_value_count = puzzle.state_value_count;
    layout.move_count = puzzle.move_count;
    layout.logical_hd1 = model.hd1;
    layout.physical_hd1 = RoundUp(model.hd1, model.hidden_alignment);
    layout.logical_hd2 = model.hd2;
    layout.physical_hd2 = RoundUp(model.hd2, model.hidden_alignment);
    layout.residual_blocks = model.residual_blocks;
    layout.output_dim = model.output_dim;

    AddBlock(&layout,
             ParamBlockRole::kInputEmbedding,
             puzzle.raw_state_dim * puzzle.state_value_count,
             model.hd1,
             puzzle.raw_state_dim * puzzle.state_value_count,
             layout.physical_hd1,
             0);
    AddBlock(&layout,
             ParamBlockRole::kInputBias,
             1,
             model.hd1,
             1,
             layout.physical_hd1,
             0);
    AddBlock(&layout,
             ParamBlockRole::kHiddenWeight,
             model.hd1,
             model.hd2,
             layout.physical_hd1,
             layout.physical_hd2,
             0);
    AddBlock(&layout,
             ParamBlockRole::kHiddenBias,
             1,
             model.hd2,
             1,
             layout.physical_hd2,
             0);
    for (std::uint32_t block = 0; block < model.residual_blocks; ++block) {
        AddBlock(&layout,
                 ParamBlockRole::kResidualFc1Weight,
                 model.hd2,
                 model.hd2,
                 layout.physical_hd2,
                 layout.physical_hd2,
                 block);
        AddBlock(&layout,
                 ParamBlockRole::kResidualFc1Bias,
                 1,
                 model.hd2,
                 1,
                 layout.physical_hd2,
                 block);
        AddBlock(&layout,
                 ParamBlockRole::kResidualFc2Weight,
                 model.hd2,
                 model.hd2,
                 layout.physical_hd2,
                 layout.physical_hd2,
                 block);
        AddBlock(&layout,
                 ParamBlockRole::kResidualFc2Bias,
                 1,
                 model.hd2,
                 1,
                 layout.physical_hd2,
                 block);
    }
    AddBatchNorm(&layout, "input", model.hd1);
    AddBatchNorm(&layout, "hidden", model.hd2);
    for (std::uint32_t block = 0; block < model.residual_blocks; ++block) {
        AddBatchNorm(&layout, "residual." + std::to_string(block) + ".fc1", model.hd2);
        AddBatchNorm(&layout, "residual." + std::to_string(block) + ".fc2", model.hd2);
    }
    AddBlock(&layout,
             ParamBlockRole::kOutputWeight,
             model.hd2,
             model.output_dim,
             layout.physical_hd2,
             model.output_dim,
             0);
    AddBlock(&layout,
             ParamBlockRole::kOutputBias,
             1,
             model.output_dim,
             1,
             model.output_dim,
             0);
    return layout;
}

}  // namespace mgt