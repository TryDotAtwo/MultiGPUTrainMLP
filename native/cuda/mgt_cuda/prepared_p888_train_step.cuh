#pragma once

#include "mgt/batch_norm_training.hpp"
#include "mgt/static_contracts.hpp"
#include "mgt_cuda/adamw.cuh"
#include "mgt_cuda/mlp_batch_norm_forward.cuh"

#include <array>
#include <cstdint>
#include <cuda_runtime.h>

namespace mgt_cuda {

struct PreparedTrainStepRequest {
    std::uint32_t batch_slot = 0;
    std::uint32_t active_rows = 0;
    std::uint32_t global_rows = 0;
    std::uint64_t global_offset = 0;
    std::uint64_t optimizer_step = 0;
    bool publish_metrics_record = false;
};

struct PreparedTrainStepTicket {
    cudaEvent_t completion_event = nullptr;
    std::uint64_t sequence = 0;
};

struct PreparedP888StrictRuntimeCreateInfo {
    CudaMlpShape shape{};
    std::uint32_t capacity_rows = 0;
    const std::uint32_t* supported_active_rows = nullptr;
    std::uint32_t supported_active_row_count = 0;
    std::uint32_t device_id = 0;
    std::uint32_t rank = 0;
    std::uint32_t world = 0;
    const char* strict_nccl_id_file = nullptr;
    MlpBatchNormStepBuffers buffers{};
    mgt::BatchNormTrainingPlan batch_norm_plan{};
    AdamWKernelConfig adam{};
    std::array<const mgt::TrainStateStorage*, 2> state_slots{};
    std::array<const float*, 2> label_slots{};
};

struct PreparedP888TrainRuntime;

mgt::Status QueryPreparedP888StrictRuntimeBytes(
    const PreparedP888StrictRuntimeCreateInfo& info,
    std::uint64_t* bytes);
mgt::Status CreatePreparedP888StrictRuntime(
    const PreparedP888StrictRuntimeCreateInfo& info,
    PreparedP888TrainRuntime** out);
mgt::Status DestroyPreparedP888TrainRuntime(PreparedP888TrainRuntime* runtime);
mgt::Status LaunchPreparedP888TrainStep(
    PreparedP888TrainRuntime* runtime,
    const PreparedTrainStepRequest& request,
    PreparedTrainStepTicket* ticket);

mgt::Status RecordPreparedP888TrainEvent(
    PreparedP888TrainRuntime* runtime,
    cudaEvent_t event);
}  // namespace mgt_cuda
