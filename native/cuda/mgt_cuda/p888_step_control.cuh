#pragma once

#include "mgt/status.hpp"

#include <cstdint>
#include <cuda_runtime.h>

namespace mgt_cuda {

enum class A100LocalGraphSlot : std::uint32_t { kFull = 1, kTail = 2 };

enum class P888StepFatal : std::uint32_t {
    kHealthy = 0,
    kBadSchema = 1,
    kCursorOverflow = 2,
    kInflightExists = 3,
    kBadShape = 4,
    kBadBatchSlot = 5,
    kCommitWithoutInflight = 6,
};

struct P888StepControlV1 {
    std::uint32_t schema_version;
    std::uint32_t graph_slot;
    std::uint32_t active_rows;
    std::uint32_t global_rows;
    std::uint64_t global_offset;
    std::uint64_t committed_sequence;
    std::uint64_t committed_optimizer_step;
    std::uint64_t inflight_sequence;
    std::uint64_t inflight_optimizer_step;
    std::uint64_t semantic_epoch;
    std::uint32_t batch_in_epoch;
    std::uint32_t batch_slot;
    std::uint64_t generation_seed;
    std::uint32_t fatal_health;
};

mgt::Status InitializeP888StepControl(P888StepControlV1* control,
    std::uint64_t initial_sequence,std::uint64_t initial_optimizer_step,
    std::uint64_t generation_seed);
mgt::Status LaunchBeginP888StepControl(P888StepControlV1* control,
    A100LocalGraphSlot slot,cudaStream_t stream);
mgt::Status LaunchCommitP888StepControl(P888StepControlV1* control,cudaStream_t stream);

}  // namespace mgt_cuda
