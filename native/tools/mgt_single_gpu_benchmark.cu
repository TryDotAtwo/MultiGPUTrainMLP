#include "mgt/puzzle_io.hpp"
#include "mgt/single_gpu_contract.hpp"
#include "mgt_cuda/single_gpu_trainer.cuh"

#include <cuda_runtime.h>

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>

int main(int argc, char** argv) {
    if (argc != 6) {
        std::cerr << "usage: batch warmup steps group_json target_bin\n";
        return 2;
    }
    const auto batch = static_cast<std::uint32_t>(std::strtoul(argv[1], nullptr, 10));
    const auto warmup = static_cast<std::uint32_t>(std::strtoul(argv[2], nullptr, 10));
    const auto steps = static_cast<std::uint32_t>(std::strtoul(argv[3], nullptr, 10));
    if (!batch || !steps) return 2;
    mgt::PuzzleDefinition puzzle{};
    if (mgt::LoadPuzzleDefinition(argv[4], argv[5], &puzzle) != mgt::Status::kOk)
        return 3;
    if (!mgt::HasNonIdentityMove(puzzle)) {
        std::cerr << "refusing degenerate identity-only move set\n";
        return 12;
    }
    mgt_cuda::SingleGpuTrainerCreateInfo info{};
    info.contract = mgt::OriginalP888SingleGpuContract();
    info.device_id = 0;
    info.capacity_rows = batch;
    info.adam = {0, 1, 1e-4f, .9f, .999f, 1e-8f, 0.f};
    info.puzzle = &puzzle;
    info.base_seed = 0x0888000000000001ULL;
    info.k_min = 1;
    info.k_max = 29;
    std::uint64_t arena_bytes = 0;
    if (mgt_cuda::QuerySingleGpuTrainerBytes(info, &arena_bytes) != mgt::Status::kOk)
        return 4;
    mgt_cuda::SingleGpuTrainer* trainer = nullptr;
    if (mgt_cuda::CreateSingleGpuTrainer(info, &trainer) != mgt::Status::kOk ||
        mgt_cuda::PrepareSingleGpuTrainer(trainer) != mgt::Status::kOk)
        return 5;
    mgt_cuda::SingleGpuTrainStepTicket ticket{};
    std::uint64_t sequence = 0;
    for (std::uint32_t i = 0; i < warmup; ++i) {
        ++sequence;
        if (mgt_cuda::LaunchSingleGpuTrainStep(
                trainer, {batch, sequence, 0, (sequence - 1) * batch}, &ticket) !=
            mgt::Status::kOk) return 6;
    }
    if (warmup && cudaEventSynchronize(ticket.completion_event) != cudaSuccess) return 7;
    const auto begin = std::chrono::steady_clock::now();
    for (std::uint32_t i = 0; i < steps; ++i) {
        ++sequence;
        if (mgt_cuda::LaunchSingleGpuTrainStep(
                trainer, {batch, sequence, 0, (sequence - 1) * batch}, &ticket) !=
            mgt::Status::kOk) return 8;
    }
    if (cudaEventSynchronize(ticket.completion_event) != cudaSuccess) return 9;
    const auto end = std::chrono::steady_clock::now();
    const double total_ms = std::chrono::duration<double, std::milli>(end - begin).count();
    mgt_cuda::SingleGpuTrainerMetrics metrics{};
    if (mgt_cuda::ReadSingleGpuMetrics(trainer, &metrics) != mgt::Status::kOk) return 10;
    cudaDeviceProp device{};
    cudaGetDeviceProperties(&device, 0);
    const double step_ms = total_ms / steps;
    std::cout << "{\"gpu\":\"" << device.name << "\",\"arch\":"
              << device.major * 10 + device.minor << ",\"batch\":" << batch
              << ",\"warmup\":" << warmup << ",\"steps\":" << steps
              << ",\"step_ms\":" << step_ms << ",\"samples_s\":"
              << batch * 1000.0 / step_ms << ",\"memory_bytes\":" << arena_bytes
              << ",\"loss\":" << metrics.loss << ",\"status\":\"ok\"}\n";
    return mgt_cuda::DestroySingleGpuTrainer(&trainer) == mgt::Status::kOk ? 0 : 11;
}
