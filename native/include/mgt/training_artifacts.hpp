#pragma once

#include "mgt/static_contracts.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace mgt {

struct CheckpointMetadata {
    std::uint32_t version = 2;
    std::uint64_t completed_steps = 0;
    std::uint64_t param_count = 0;
    std::uint64_t payload_bytes = 0;
    std::uint64_t payload_checksum = 0;
    std::string fingerprint;
    std::uint64_t seed = 0;
    float learning_rate = 0.0f;
    float weight_decay = 0.0f;
    float adam_beta1 = 0.9f;
    float adam_beta2 = 0.999f;
    float adam_eps = 1.0e-8f;
};

std::uint64_t Fnv1a64(const void* data, std::size_t size);

Status WriteCheckpointMetadata(const std::filesystem::path& path,
                               const CheckpointMetadata& metadata);

Status ReadCheckpointMetadata(const std::filesystem::path& path,
                              CheckpointMetadata* metadata);

Status ValidateCheckpointMetadata(const CheckpointMetadata& actual,
                                  const CheckpointMetadata& expected);

Status ValidateCheckpointCompatibility(const CheckpointMetadata& actual,
                                       const CheckpointMetadata& expected);
std::uint64_t GlobalStep(std::uint64_t completed_steps,
                         std::uint64_t local_step);
Status WriteCheckpointPayload(const std::filesystem::path& path,
                              const std::vector<float>& weights,
                              const std::vector<float>& adam_m,
                              const std::vector<float>& adam_v,
                              CheckpointMetadata* metadata);
Status ReadCheckpointPayload(const std::filesystem::path& path,
                             const CheckpointMetadata& metadata,
                             std::vector<float>* weights,
                             std::vector<float>* adam_m,
                             std::vector<float>* adam_v);

bool ShouldWritePeriodicArtifact(std::uint64_t completed_steps,
                                 std::uint64_t period_steps,
                                 bool final_step);

Status PublishDirectoryAtomically(const std::filesystem::path& staged,
                                  const std::filesystem::path& current);

}  // namespace mgt
