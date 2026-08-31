#pragma once

#include "mgt/single_gpu_contract.hpp"
#include "mgt/puzzle_io.hpp"
#include "mgt/status.hpp"
#include "mgt_cuda/adamw.cuh"

#include <cuda_runtime.h>

#include <cstdint>

namespace mgt_cuda {

struct SingleGpuTrainer;

enum class SingleGpuExecutionMode : std::uint32_t {
    kEager = 0,
    // Explicit opt-in: capture capacity_rows at prepare; smaller batches use
    // eager on the same stream. Requires a CUDA >=12.8 native build.
    kFixedBatchGraph = 1,
};

struct SingleGpuTrainerCreateInfo {
    mgt::SingleGpuModelContract contract{};
    std::uint32_t device_id = 0;
    std::uint32_t capacity_rows = 0;
    AdamWKernelConfig adam{};
    const mgt::PuzzleDefinition* puzzle = nullptr;
    std::uint64_t base_seed = 0;
    std::uint32_t k_min = 1;
    std::uint32_t k_max = 29;
    std::uint32_t global_rank = 0;
    SingleGpuExecutionMode execution_mode = SingleGpuExecutionMode::kEager;
};

struct SingleGpuTrainStepRequest {
    std::uint32_t active_rows = 0;
    std::uint64_t optimizer_step = 0;
    std::uint64_t epoch = 0;
    std::uint64_t epoch_sample_offset = 0;
};

struct SingleGpuTrainStepTicket {
    cudaEvent_t completion_event = nullptr;
    std::uint64_t sequence = 0;
};

struct SingleGpuTrainerMetrics {
    std::uint64_t completed_sequence = 0;
    std::uint64_t optimizer_step = 0;
    float loss = 0.0f;
};

mgt::Status QuerySingleGpuTrainerBytes(
    const SingleGpuTrainerCreateInfo& info, std::uint64_t* bytes);
mgt::Status CreateSingleGpuTrainer(
    const SingleGpuTrainerCreateInfo& info, SingleGpuTrainer** out);
mgt::Status PrepareSingleGpuTrainer(SingleGpuTrainer* trainer);
mgt::Status LaunchSingleGpuTrainStep(
    SingleGpuTrainer* trainer, const SingleGpuTrainStepRequest& request,
    SingleGpuTrainStepTicket* ticket);
mgt::Status ReadSingleGpuMetrics(
    SingleGpuTrainer* trainer, SingleGpuTrainerMetrics* metrics);
mgt::Status DestroySingleGpuTrainer(SingleGpuTrainer** trainer);

std::uint64_t SingleGpuTrainerAllocationCountForTest();

}  // namespace mgt_cuda
