#include "mgt/weight_export.hpp"
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

namespace {

std::string ReadText(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    return std::string((std::istreambuf_iterator<char>(input)),
                       std::istreambuf_iterator<char>());
}

bool TestDefaultExport() {
    const auto layout = mgt::BuildModelLayout();
    std::vector<float> weights(layout.total_params, 0.125f);
    const std::filesystem::path out = "build-native/test_weight_export_out";
    const auto status = mgt::ExportInferenceWeights(out, layout, weights);
    if (status != mgt::Status::kOk) return false;
    if (!std::filesystem::exists(out / "manifest.json")) return false;
    if (!std::filesystem::exists(out / "weights.f32.bin")) return false;
    if (std::filesystem::file_size(out / "weights.f32.bin") != weights.size() * sizeof(float)) {
        return false;
    }
    const std::string text = ReadText(out / "manifest.json");
    if (text.find("\"format\": \"stream1_weights\"") == std::string::npos) return false;
    if (text.find("\"output_dim\": 1") == std::string::npos) return false;
    if (text.find("\"state_storage_len\": 80") == std::string::npos) return false;
    return true;
}

bool TestConfigExport() {
    mgt::PuzzleSpec puzzle{};
    puzzle.group_id = 123;
    puzzle.target_id = 7;
    puzzle.raw_state_dim = 31;
    puzzle.state_value_count = 11;
    puzzle.move_count = 5;
    puzzle.state_alignment = 16;

    mgt::ModelSpec model{};
    model.hd1 = 33;
    model.hd2 = 37;
    model.residual_blocks = 2;
    model.output_dim = 1;
    model.hidden_alignment = 8;

    const auto layout = mgt::BuildModelLayout(puzzle, model);
    std::vector<float> weights(layout.total_params, 0.25f);
    const std::filesystem::path out = "build-native/test_weight_export_config_out";
    if (mgt::ExportInferenceWeights(out, puzzle, layout, weights) != mgt::Status::kOk) return false;
    const std::string text = ReadText(out / "manifest.json");
    if (text.find("\"group_id\": 123") == std::string::npos) return false;
    if (text.find("\"target_id\": 7") == std::string::npos) return false;
    if (text.find("\"state_len\": 31") == std::string::npos) return false;
    if (text.find("\"state_storage_len\": 32") == std::string::npos) return false;
    if (text.find("\"hd1\": 33") == std::string::npos) return false;
    if (text.find("\"physical_hd1\": 40") == std::string::npos) return false;
    if (text.find("\"hd2\": 37") == std::string::npos) return false;
    if (text.find("\"physical_hd2\": 40") == std::string::npos) return false;
    return true;
}

}  // namespace

int main() {
    if (!TestDefaultExport()) return EXIT_FAILURE;
    if (!TestConfigExport()) return EXIT_FAILURE;
    return EXIT_SUCCESS;
}
