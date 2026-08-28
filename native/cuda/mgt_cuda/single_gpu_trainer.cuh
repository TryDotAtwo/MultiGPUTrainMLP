#pragma once

#include "mgt/single_gpu_contract.hpp"
#include "mgt/status.hpp"
#include "mgt_cuda/adamw.cuh"

#include <cuda_runtime.h>

#include <cstdint>

namespace mgt_cuda {

struct SingleGpuTrainer;

struct SingleGpuTrainerCreateInfo {
    mgt::SingleGpuModelContract contract{};
    std::uint32_t device_id = 0;
    std::uint32_t capacity_rows = 0;
    AdamWKernelConfig adam{};
};

struct SingleGpuTrainStepRequest {
    std::uint32_t active_rows = 0;
    std::uint64_t optimizer_step = 0;
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
