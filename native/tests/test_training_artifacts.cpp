#include "mgt/training_artifacts.hpp"

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace {
bool WriteText(const std::filesystem::path& path, const std::string& text) {
    std::ofstream out(path, std::ios::binary);
    out.write(text.data(), static_cast<std::streamsize>(text.size()));
    return static_cast<bool>(out);
}
}  // namespace

int main() {
    const auto nonce = std::chrono::steady_clock::now().time_since_epoch().count();
    const std::filesystem::path root =
        std::filesystem::temp_directory_path() / ("mgt-training-artifacts-" + std::to_string(nonce));
    std::filesystem::create_directories(root);

    const std::vector<unsigned char> payload{0, 1, 2, 3, 254, 255};
    const std::uint64_t checksum = mgt::Fnv1a64(payload.data(), payload.size());
    if (checksum != mgt::Fnv1a64(payload.data(), payload.size())) return EXIT_FAILURE;

    mgt::CheckpointMetadata expected{};
    expected.completed_steps = 17;
    expected.param_count = 23;
    expected.payload_bytes = payload.size();
    expected.payload_checksum = checksum;
    expected.fingerprint = "puzzle=abc;model=def";
    expected.seed = 1234;
    expected.learning_rate = 0.0001f;
    expected.weight_decay = 0.01f;
    expected.adam_beta1 = 0.9f;
    expected.adam_beta2 = 0.999f;
    expected.adam_eps = 1.0e-8f;

    const std::filesystem::path manifest = root / "manifest.env";
    if (mgt::WriteCheckpointMetadata(manifest, expected) != mgt::Status::kOk) return EXIT_FAILURE;
    mgt::CheckpointMetadata actual{};
    if (mgt::ReadCheckpointMetadata(manifest, &actual) != mgt::Status::kOk) return EXIT_FAILURE;
    if (mgt::ValidateCheckpointMetadata(actual, expected) != mgt::Status::kOk) return EXIT_FAILURE;

    mgt::CheckpointMetadata incompatible = expected;
    incompatible.fingerprint = "puzzle=changed;model=def";
    if (mgt::ValidateCheckpointMetadata(actual, incompatible) != mgt::Status::kInvalidConfig) return EXIT_FAILURE;
    incompatible = expected;
    incompatible.param_count += 1;
    if (mgt::ValidateCheckpointMetadata(actual, incompatible) != mgt::Status::kInvalidConfig) return EXIT_FAILURE;
    incompatible = expected;
    incompatible.payload_checksum ^= 1;
    if (mgt::ValidateCheckpointMetadata(actual, incompatible) != mgt::Status::kInvalidConfig) return EXIT_FAILURE;

    if (!mgt::ShouldWritePeriodicArtifact(16, 8, false)) return EXIT_FAILURE;
    if (mgt::ShouldWritePeriodicArtifact(17, 8, false)) return EXIT_FAILURE;
    if (mgt::ShouldWritePeriodicArtifact(17, 0, false)) return EXIT_FAILURE;
    if (!mgt::ShouldWritePeriodicArtifact(17, 0, true)) return EXIT_FAILURE;

    const float finite_values[] = {0.0f, -1.0f, 3.5f};
    if (!mgt::AllFinite(finite_values, 3)) return EXIT_FAILURE;
    const float nan_values[] = {0.0f, std::numeric_limits<float>::quiet_NaN()};
    if (mgt::AllFinite(nan_values, 2)) return EXIT_FAILURE;
    const float inf_values[] = {std::numeric_limits<float>::infinity()};
    if (mgt::AllFinite(inf_values, 1)) return EXIT_FAILURE;
    if (mgt::AllFinite(nullptr, 1)) return EXIT_FAILURE;

    const std::filesystem::path current = root / "latest";
    const std::filesystem::path staged = root / "latest.tmp";
    std::filesystem::create_directories(current);
    std::filesystem::create_directories(staged);
    if (!WriteText(current / "generation.txt", "old")) return EXIT_FAILURE;
    if (!WriteText(staged / "generation.txt", "new")) return EXIT_FAILURE;
    if (mgt::PublishDirectoryAtomically(staged, current) != mgt::Status::kOk) return EXIT_FAILURE;
    if (std::filesystem::exists(staged) || std::filesystem::exists(root / "latest.old")) return EXIT_FAILURE;
    std::ifstream generation(current / "generation.txt", std::ios::binary);
    std::string value;
    generation >> value;
    if (value != "new") return EXIT_FAILURE;
    generation.close();

    std::filesystem::remove_all(root);
    return EXIT_SUCCESS;
}
