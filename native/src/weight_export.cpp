#include "mgt/weight_export.hpp"
#include <filesystem>
#include <fstream>

namespace mgt {

Status ExportInferenceWeights(const std::filesystem::path& output_dir,
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
        << "  \"group_id\": 888,\n"
        << "  \"target_id\": 0,\n"
        << "  \"state_len\": " << kStateLen << ",\n"
        << "  \"state_storage_len\": " << kStateStorageLen << ",\n"
        << "  \"state_value_pad\": " << kStateValuePad << ",\n"
        << "  \"move_count\": " << kMoveCount << ",\n"
        << "  \"output_dim\": " << kOutputDim << ",\n"
        << "  \"hd1\": " << kHd1 << ",\n"
        << "  \"hd2\": " << kHd2 << ",\n"
        << "  \"residual_blocks\": " << kResidualBlocks << ",\n"
        << "  \"dtype\": \"float32\",\n"
        << "  \"data\": \"weights.f32.bin\",\n"
        << "  \"total_params\": " << layout.total_params << "\n"
        << "}\n";
    return manifest ? Status::kOk : Status::kIoFailure;
}

}  // namespace mgt