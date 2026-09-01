#include "mgt/puzzle_io.hpp"
#include "mgt/single_gpu_contract.hpp"
#include "mgt_cuda/single_gpu_trainer.cuh"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>

namespace {

struct TrainerOwner {
    mgt_cuda::SingleGpuTrainer* value = nullptr;
    ~TrainerOwner() {
        if (value) mgt_cuda::DestroySingleGpuTrainer(&value);
    }
};

bool ParsePositiveU32(const char* text, std::uint32_t* out) {
    if (!text || !*text || *text == '-' || !out) return false;
    char* end = nullptr;
    const auto value = std::strtoull(text, &end, 10);
    if (!end || *end || !value || value > UINT32_MAX) return false;
    *out = static_cast<std::uint32_t>(value);
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 6) {
        std::cerr << "usage: batch epochs group_json target_bin fp32|fp16\n";
        return 2;
    }
    std::uint32_t batch = 0;
    std::uint32_t epochs = 0;
    const std::string precision = argv[5];
    if (!ParsePositiveU32(argv[1], &batch) ||
        !ParsePositiveU32(argv[2], &epochs) ||
        batch > mgt::P888TrainingContract::kSamplesPerEpoch ||
        (precision != "fp32" && precision != "fp16")) return 2;

    mgt::PuzzleDefinition puzzle{};
    if (mgt::LoadPuzzleDefinition(argv[3], argv[4], &puzzle) != mgt::Status::kOk)
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
    info.execution_mode = mgt_cuda::SingleGpuExecutionMode::kFixedBatchGraph;
    info.input_gradient_precision = precision == "fp16"
        ? mgt_cuda::SingleGpuInputGradientPrecision::kFp16Mirror
        : mgt_cuda::SingleGpuInputGradientPrecision::kFp32;

    std::uint64_t arena_bytes = 0;
    if (mgt_cuda::QuerySingleGpuTrainerBytes(info, &arena_bytes) != mgt::Status::kOk)
        return 4;
    TrainerOwner owner;
    if (mgt_cuda::CreateSingleGpuTrainer(info, &owner.value) != mgt::Status::kOk ||
        mgt_cuda::PrepareSingleGpuTrainer(owner.value) != mgt::Status::kOk) return 5;

    constexpr std::uint64_t kSamples = mgt::P888TrainingContract::kSamplesPerEpoch;
    constexpr std::uint32_t kLossWindow = 16;
    const auto full_steps = static_cast<std::uint32_t>(kSamples / batch);
    const auto tail = static_cast<std::uint32_t>(kSamples % batch);
    const auto sampled_full_steps = full_steps < kLossWindow ? full_steps : kLossWindow;
    std::uint64_t optimizer_step = 0;
    mgt_cuda::SingleGpuTrainStepTicket ticket{};

    std::cout << std::setprecision(9);
    for (std::uint32_t epoch = 0; epoch < epochs; ++epoch) {
        const auto begin = std::chrono::steady_clock::now();
        double sampled_loss_sum = 0.0;
        std::uint32_t sampled_loss_count = 0;
        float last_full_loss = 0.0f;
        float tail_loss = 0.0f;

        for (std::uint32_t step = 0; step < full_steps; ++step) {
            ++optimizer_step;
            const mgt_cuda::SingleGpuTrainStepRequest request{
                batch, optimizer_step, epoch, static_cast<std::uint64_t>(step) * batch};
            if (mgt_cuda::LaunchSingleGpuTrainStep(owner.value, request, &ticket) !=
                mgt::Status::kOk) return 6;
            if (step + sampled_full_steps >= full_steps) {
                mgt_cuda::SingleGpuTrainerMetrics metrics{};
                if (mgt_cuda::ReadSingleGpuMetrics(owner.value, &metrics) != mgt::Status::kOk ||
                    metrics.optimizer_step != optimizer_step || !std::isfinite(metrics.loss))
                    return 7;
                sampled_loss_sum += metrics.loss;
                ++sampled_loss_count;
                last_full_loss = metrics.loss;
            }
        }
        if (tail) {
            ++optimizer_step;
            const mgt_cuda::SingleGpuTrainStepRequest request{
                tail, optimizer_step, epoch, static_cast<std::uint64_t>(full_steps) * batch};
            if (mgt_cuda::LaunchSingleGpuTrainStep(owner.value, request, &ticket) !=
                mgt::Status::kOk) return 8;
            mgt_cuda::SingleGpuTrainerMetrics metrics{};
            if (mgt_cuda::ReadSingleGpuMetrics(owner.value, &metrics) != mgt::Status::kOk ||
                metrics.optimizer_step != optimizer_step || !std::isfinite(metrics.loss))
                return 9;
            tail_loss = metrics.loss;
        } else {
            tail_loss = last_full_loss;
        }
        if (!sampled_loss_count) return 10;
        const auto end = std::chrono::steady_clock::now();
        const double mean_last_full = sampled_loss_sum / sampled_loss_count;
        if (!std::isfinite(mean_last_full)) return 10;
        std::cout << "{\"precision\":\"" << precision << "\",\"epoch\":"
                  << epoch + 1 << ",\"optimizer_step\":" << optimizer_step
                  << ",\"full_steps\":" << full_steps << ",\"tail_rows\":" << tail
                  << ",\"loss_window\":" << sampled_loss_count
                  << ",\"mean_last_full_loss\":" << mean_last_full
                  << ",\"last_full_loss\":" << last_full_loss
                  << ",\"tail_loss\":" << tail_loss
                  << ",\"epoch_ms\":"
                  << std::chrono::duration<double, std::milli>(end - begin).count()
                  << ",\"arena_bytes\":" << arena_bytes << ",\"status\":\"ok\"}\n";
    }
    return mgt_cuda::DestroySingleGpuTrainer(&owner.value) == mgt::Status::kOk ? 0 : 11;
}
