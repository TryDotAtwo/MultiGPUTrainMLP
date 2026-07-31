#pragma once

#include "mgt/a100_bf16_policy.hpp"

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace mgt {

enum class A100ExecutionProfileState : std::uint32_t { kCandidate = 1, kAccepted = 2 };
enum class A100ExecutionProfileUse : std::uint32_t { kCandidateForTuner = 1, kAcceptedForProduction = 2 };

struct P888A100ExecutionProfileV1 {
    std::uint32_t schema_version = 1;
    std::uint32_t canonical_serialization_version = 1;
    A100ExecutionProfileState profile_state = A100ExecutionProfileState::kCandidate;
    std::uint32_t world = 8;
    std::vector<std::uint32_t> active_rows;
    A100Bf16Policy policy;
    std::uint64_t ordinary_arena_bytes = 0;
    std::uint64_t symmetric_arena_bytes = 0;
    std::uint64_t pinned_host_bytes = 0;
    std::array<std::uint8_t,32> source_sha256{};
    std::array<std::uint8_t,32> binary_sha256{};
    std::array<std::uint8_t,32> arena_layout_sha256{};
    std::array<std::uint8_t,32> gate_artifact_sha256{};
    std::array<std::uint8_t,32> acceptance_report_sha256{};
};

Status ValidateP888A100ExecutionProfile(const P888A100ExecutionProfileV1& profile, A100ExecutionProfileUse use);
Status CanonicalSerializeP888A100ExecutionProfile(const P888A100ExecutionProfileV1& profile, A100ExecutionProfileUse use, std::vector<std::uint8_t>* out);
Status CanonicalP888A100ExecutionProfileSha256(const P888A100ExecutionProfileV1& profile, A100ExecutionProfileUse use, std::string* out_hex);

}  // namespace mgt
