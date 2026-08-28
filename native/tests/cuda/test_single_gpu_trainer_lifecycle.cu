#include "mgt_cuda/single_gpu_trainer.cuh"

#include <cuda_runtime.h>
#include <limits>
#include <cstdio>

int main() {
    mgt_cuda::SingleGpuTrainerCreateInfo info{};
    info.contract = mgt::OriginalP888SingleGpuContract();
    info.device_id = 0;
    info.capacity_rows = 4;
    info.adam = {0, 1, .001f, .9f, .999f, 1e-8f, 0.f};
    std::uint64_t bytes = 0;
    const auto query_status = mgt_cuda::QuerySingleGpuTrainerBytes(info, &bytes);
    if (query_status != mgt::Status::kOk || !bytes) {
        std::fprintf(stderr, "query status=%d bytes=%llu\n", static_cast<int>(query_status),
                     static_cast<unsigned long long>(bytes));
        return 1;
    }
    auto bad = info;
    bad.capacity_rows = 0;
    if (mgt_cuda::QuerySingleGpuTrainerBytes(bad, &bytes) != mgt::Status::kInvalidConfig)
        return 2;
    bad = info;
    bad.contract.physical_hd2 = 218;
    if (mgt_cuda::QuerySingleGpuTrainerBytes(bad, &bytes) != mgt::Status::kInvalidConfig)
        return 3;
    bad = info;
    bad.device_id = std::numeric_limits<std::uint32_t>::max();
    mgt_cuda::SingleGpuTrainer* invalid = nullptr;
    if (mgt_cuda::CreateSingleGpuTrainer(bad, &invalid) != mgt::Status::kCudaFailure || invalid)
        return 3;
    mgt_cuda::SingleGpuTrainer* trainer = nullptr;
    if (mgt_cuda::CreateSingleGpuTrainer(info, &trainer) != mgt::Status::kOk || !trainer)
        return 4;
    mgt_cuda::SingleGpuTrainStepTicket ticket{};
    if (mgt_cuda::LaunchSingleGpuTrainStep(trainer, {4, 1}, &ticket) !=
        mgt::Status::kInvalidConfig) return 5;
    if (mgt_cuda::PrepareSingleGpuTrainer(trainer) != mgt::Status::kOk)
        return 6;
    if (mgt_cuda::PrepareSingleGpuTrainer(trainer) != mgt::Status::kInvalidConfig)
        return 7;
    const auto allocations = mgt_cuda::SingleGpuTrainerAllocationCountForTest();
    for (std::uint64_t step = 1; step <= 3; ++step) {
        if (mgt_cuda::LaunchSingleGpuTrainStep(trainer, {4, step}, &ticket) !=
            mgt::Status::kOk) return 8;
        if (!ticket.completion_event || ticket.sequence != step ||
            cudaEventSynchronize(ticket.completion_event) != cudaSuccess) return 9;
    }
    if (mgt_cuda::SingleGpuTrainerAllocationCountForTest() != allocations)
        return 10;
    if (mgt_cuda::LaunchSingleGpuTrainStep(trainer, {3, 4}, &ticket) != mgt::Status::kOk ||
        cudaEventSynchronize(ticket.completion_event) != cudaSuccess) return 11;
    if (mgt_cuda::LaunchSingleGpuTrainStep(trainer, {5, 5}, &ticket) !=
        mgt::Status::kCapacityExceeded) return 11;
    if (mgt_cuda::LaunchSingleGpuTrainStep(trainer, {4, 6}, &ticket) !=
        mgt::Status::kInvalidConfig) return 12;
    mgt_cuda::SingleGpuTrainerMetrics metrics{};
    if (mgt_cuda::ReadSingleGpuMetrics(trainer, &metrics) != mgt::Status::kOk ||
        metrics.completed_sequence != 4 || metrics.optimizer_step != 4)
        return 13;
    if (mgt_cuda::DestroySingleGpuTrainer(&trainer) != mgt::Status::kOk || trainer)
        return 14;
    if (mgt_cuda::DestroySingleGpuTrainer(&trainer) != mgt::Status::kInvalidConfig)
        return 15;
    return 0;
}
