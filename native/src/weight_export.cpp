#include "mgt/weight_export.hpp"
#include <filesystem>
#include <fstream>

namespace mgt {

Status ExportInferenceWeights(const std::filesystem::path& output_dir,
                              const PuzzleSpec& puzzle,
                              const ModelLayout& layout,
                              std::span<const float> weights) {
    if (weights.size() != layout.total_params) return Status::kInvalidConfig;

    std::error_code ec;
    std::filesystem::create_directories(output_dir, ec);
    if (ec) return Status::kIoFailure;

    std::ofstream data(output_dir / "weights.f32.bin", std::ios::binary);
    if (!data) return Status::kIoFailure;
    data.write(reinterpret_cast<const char*>(weights.data()),
               static_cast<std::streamsize>(weights.size() * sizeof(float)));
    if (!data) return Status::kIoFailure;

    std::ofstream manifest(output_dir / "manifest.json", std::ios::binary);
    if (!manifest) return Status::kIoFailure;
    manifest
        << "{\n"
        << "  \"format\": \"stream1_weights\",\n"
        << "  \"version\": 1,\n"
        << "  \"group_id\": " << puzzle.group_id << ",\n"
        << "  \"target_id\": " << puzzle.target_id << ",\n"
        << "  \"state_len\": " << layout.state_dim << ",\n"
        << "  \"state_storage_len\": " << layout.padded_state_dim << ",\n"
        << "  \"state_value_pad\": " << layout.state_value_count << ",\n"
        << "  \"move_count\": " << puzzle.move_count << ",\n"
        << "  \"output_dim\": " << layout.output_dim << ",\n"
        << "  \"hd1\": " << layout.logical_hd1 << ",\n"
        << "  \"physical_hd1\": " << layout.physical_hd1 << ",\n"
        << "  \"hd2\": " << layout.logical_hd2 << ",\n"
        << "  \"physical_hd2\": " << layout.physical_hd2 << ",\n"
        << "  \"residual_blocks\": " << layout.residual_blocks << ",\n"
        << "  \"dtype\": \"float32\",\n"
        << "  \"data\": \"weights.f32.bin\",\n"
        << "  \"total_params\": " << layout.total_params << "\n"
        << "}\n";
    return manifest ? Status::kOk : Status::kIoFailure;
}

Status ExportInferenceWeights(const std::filesystem::path& output_dir,
                              const ModelLayout& layout,
                              std::span<const float> weights) {
    return ExportInferenceWeights(output_dir, PuzzleSpec{}, layout, weights);
}

}  // namespace mgt
