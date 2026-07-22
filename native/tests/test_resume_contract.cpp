#include "mgt/training_artifacts.hpp"

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <vector>

int main() {
    if (mgt::GlobalStep(0, 0) != 0 ||
        mgt::GlobalStep(17, 0) != 17 ||
        mgt::GlobalStep(17, 2) != 19) {
        return EXIT_FAILURE;
    }

    const auto nonce = std::chrono::steady_clock::now().time_since_epoch().count();
    const std::filesystem::path root =
        std::filesystem::temp_directory_path() / ("mgt-resume-contract-" + std::to_string(nonce));
    std::filesystem::create_directories(root);
    const std::filesystem::path payload_path = root / "state.f32.bin";

    const std::vector<float> weights{1.0f, 2.0f, 3.0f};
    const std::vector<float> adam_m{0.1f, 0.2f, 0.3f};
    const std::vector<float> adam_v{0.01f, 0.02f, 0.03f};
    mgt::CheckpointMetadata metadata{};
    metadata.completed_steps = 11;
    metadata.param_count = weights.size();
    metadata.fingerprint = "real:fixture:model";
    metadata.seed = 1234;
    metadata.learning_rate = 0.0001f;
    metadata.weight_decay = 0.01f;
    metadata.adam_beta1 = 0.9f;
    metadata.adam_beta2 = 0.999f;
    metadata.adam_eps = 1.0e-8f;

    if (mgt::WriteCheckpointPayload(
            payload_path, weights, adam_m, adam_v, &metadata) != mgt::Status::kOk) {
        return EXIT_FAILURE;
    }
    if (metadata.payload_bytes != 9 * sizeof(float) ||
        metadata.payload_checksum == 0) {
        return EXIT_FAILURE;
    }

    std::vector<float> read_weights(weights.size());
    std::vector<float> read_m(weights.size());
    std::vector<float> read_v(weights.size());
    if (mgt::ReadCheckpointPayload(
            payload_path, metadata, &read_weights, &read_m, &read_v) != mgt::Status::kOk ||
        read_weights != weights || read_m != adam_m || read_v != adam_v) {
        return EXIT_FAILURE;
    }

    mgt::CheckpointMetadata compatible = metadata;
    compatible.completed_steps = 0;
    compatible.payload_checksum = 0;
    if (mgt::ValidateCheckpointCompatibility(metadata, compatible) != mgt::Status::kOk) {
        return EXIT_FAILURE;
    }
    compatible.fingerprint = "changed";
    if (mgt::ValidateCheckpointCompatibility(metadata, compatible) !=
        mgt::Status::kInvalidConfig) {
        return EXIT_FAILURE;
    }
    compatible = metadata;
    compatible.adam_beta1 = 0.8f;
    if (mgt::ValidateCheckpointCompatibility(metadata, compatible) !=
        mgt::Status::kInvalidConfig) {
        return EXIT_FAILURE;
    }

    mgt::CheckpointMetadata wrong_checksum = metadata;
    wrong_checksum.payload_checksum ^= 1;
    if (mgt::ReadCheckpointPayload(
            payload_path, wrong_checksum, &read_weights, &read_m, &read_v) !=
        mgt::Status::kInvalidConfig) {
        return EXIT_FAILURE;
    }
    {
        std::ofstream append(payload_path, std::ios::binary | std::ios::app);
        const char extra = 0;
        append.write(&extra, 1);
    }
    if (mgt::ReadCheckpointPayload(
            payload_path, metadata, &read_weights, &read_m, &read_v) !=
        mgt::Status::kInvalidConfig) {
        return EXIT_FAILURE;
    }
    std::filesystem::resize_file(payload_path, metadata.payload_bytes - 1);
    if (mgt::ReadCheckpointPayload(
            payload_path, metadata, &read_weights, &read_m, &read_v) !=
        mgt::Status::kInvalidConfig) {
        return EXIT_FAILURE;
    }

    std::filesystem::remove_all(root);
    return EXIT_SUCCESS;
}
