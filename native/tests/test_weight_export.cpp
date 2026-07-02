#include "mgt/weight_export.hpp"
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <vector>

int main() {
    const auto layout = mgt::BuildModelLayout();
    std::vector<float> weights(layout.total_params, 0.125f);
    const std::filesystem::path out = "build-native/test_weight_export_out";
    const auto status = mgt::ExportInferenceWeights(out, layout, weights);
    if (status != mgt::Status::kOk) return EXIT_FAILURE;
    if (!std::filesystem::exists(out / "manifest.json")) return EXIT_FAILURE;
    if (!std::filesystem::exists(out / "weights.f32.bin")) return EXIT_FAILURE;
    if (std::filesystem::file_size(out / "weights.f32.bin") != weights.size() * sizeof(float)) {
        return EXIT_FAILURE;
    }
    std::ifstream manifest(out / "manifest.json", std::ios::binary);
    const std::string text((std::istreambuf_iterator<char>(manifest)),
                           std::istreambuf_iterator<char>());
    if (text.find("\"format\": \"stream1_weights\"") == std::string::npos) return EXIT_FAILURE;
    if (text.find("\"output_dim\": 1") == std::string::npos) return EXIT_FAILURE;
    return EXIT_SUCCESS;
}