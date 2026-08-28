#include "mgt/single_gpu_trainer_ffi.h"
#include "mgt/single_gpu_contract.hpp"
#include "mgt/puzzle_io.hpp"
#include "mgt_cuda/single_gpu_trainer.cuh"

#include <algorithm>
#include <cstring>
#include <new>
#include <string>

struct MgtSingleGpuHandle {
    mgt_cuda::SingleGpuTrainer* trainer = nullptr;
    std::string error;
};

namespace {
thread_local std::string g_error;
MgtStatus Convert(mgt::Status status) { return static_cast<MgtStatus>(static_cast<int>(status)); }
MgtStatus Fail(MgtSingleGpuHandle* h, MgtStatus status, const char* message) {
    (h ? h->error : g_error) = message;
    return status;
}
bool ReservedZero(const MgtSingleGpuConfigV1& c) {
    return !c.reserved_u32[0] && !c.reserved_u32[1] && !c.reserved_u32[2];
}
template <class F> MgtStatus Guard(MgtSingleGpuHandle* h, F&& function) noexcept {
    try { return function(); }
    catch (const std::bad_alloc&) { return Fail(h, MGT_STATUS_CAPACITY_EXCEEDED, "host allocation failed"); }
    catch (...) { return Fail(h, MGT_STATUS_INVALID_CONFIG, "unexpected native exception"); }
}
}

extern "C" uint32_t mgt_single_gpu_v1_abi_version(void) { return MGT_SINGLE_GPU_ABI_V1; }

extern "C" MgtStatus mgt_single_gpu_v1_create(
    const MgtSingleGpuConfigV1* c, MgtSingleGpuHandle** out) {
    if (!out) return Fail(nullptr, MGT_STATUS_INVALID_CONFIG, "out handle is null");
    *out = nullptr;
    if (!c || c->struct_size != sizeof(*c) || c->abi_version != MGT_SINGLE_GPU_ABI_V1 ||
        !ReservedZero(*c) || !c->group_json_utf8 || !c->group_json_utf8[0] ||
        !c->target_bin_utf8 || !c->target_bin_utf8[0] || !c->base_seed || !c->k_min ||
        c->k_min > c->k_max)
        return Fail(nullptr, MGT_STATUS_INVALID_CONFIG, "invalid ABI config layout");
    return Guard(nullptr, [&] {
        auto* h = new MgtSingleGpuHandle;
        mgt::PuzzleDefinition puzzle{};
        const auto load = mgt::LoadPuzzleDefinition(
            c->group_json_utf8, c->target_bin_utf8, &puzzle);
        if (load != mgt::Status::kOk) {
            delete h;
            return Fail(nullptr, Convert(load), "puzzle load failed");
        }
        mgt_cuda::SingleGpuTrainerCreateInfo info{};
        info.contract = mgt::OriginalP888SingleGpuContract();
        info.device_id = c->device_id;
        info.capacity_rows = c->capacity_rows;
        info.adam = {0, 1, c->learning_rate, c->beta1, c->beta2, c->epsilon, c->weight_decay};
        info.puzzle = &puzzle;
        info.base_seed = c->base_seed;
        info.k_min = c->k_min;
        info.k_max = c->k_max;
        const auto status = mgt_cuda::CreateSingleGpuTrainer(info, &h->trainer);
        if (status != mgt::Status::kOk) { delete h; return Fail(nullptr, Convert(status), "native trainer creation failed"); }
        *out = h;
        return MGT_STATUS_OK;
    });
}

extern "C" MgtStatus mgt_single_gpu_v1_prepare(MgtSingleGpuHandle* h) {
    if (!h || !h->trainer) return Fail(h, MGT_STATUS_INVALID_CONFIG, "invalid trainer handle");
    return Guard(h, [&] {
        const auto status = mgt_cuda::PrepareSingleGpuTrainer(h->trainer);
        return status == mgt::Status::kOk ? MGT_STATUS_OK : Fail(h, Convert(status), "trainer prepare failed");
    });
}

extern "C" MgtStatus mgt_single_gpu_v1_train_step(
    MgtSingleGpuHandle* h, const MgtSingleGpuStepV1* step, MgtSingleGpuMetricsV1* metrics) {
    if (!h || !h->trainer || !step || step->struct_size != sizeof(*step) ||
        step->reserved_u64[0] || step->reserved_u64[1] ||
        (metrics && metrics->struct_size != sizeof(*metrics)))
        return Fail(h, MGT_STATUS_INVALID_CONFIG, "invalid train-step layout");
    return Guard(h, [&] {
        mgt_cuda::SingleGpuTrainStepTicket ticket{};
        auto status = mgt_cuda::LaunchSingleGpuTrainStep(
            h->trainer, {step->active_rows, step->optimizer_step}, &ticket);
        if (status != mgt::Status::kOk) return Fail(h, Convert(status), "train-step enqueue failed");
        if (metrics) {
            mgt_cuda::SingleGpuTrainerMetrics native{};
            status = mgt_cuda::ReadSingleGpuMetrics(h->trainer, &native);
            if (status != mgt::Status::kOk) return Fail(h, Convert(status), "metrics read failed");
            metrics->reserved_u32 = 0;
            metrics->completed_sequence = native.completed_sequence;
            metrics->optimizer_step = native.optimizer_step;
            metrics->loss = native.loss;
            metrics->reserved_f32 = 0;
        }
        return MGT_STATUS_OK;
    });
}

extern "C" MgtStatus mgt_single_gpu_v1_checkpoint(MgtSingleGpuHandle* h, const char* path) {
    if (!h || !h->trainer || !path || !path[0])
        return Fail(h, MGT_STATUS_INVALID_CONFIG, "checkpoint directory must be non-empty UTF-8");
    return Fail(h, MGT_STATUS_IO_FAILURE, "checkpoint is not implemented yet");
}

extern "C" MgtStatus mgt_single_gpu_v1_destroy(MgtSingleGpuHandle** pointer) {
    if (!pointer || !*pointer) return Fail(nullptr, MGT_STATUS_INVALID_CONFIG, "invalid destroy handle");
    auto* h = *pointer;
    return Guard(h, [&] {
        const auto status = mgt_cuda::DestroySingleGpuTrainer(&h->trainer);
        if (status != mgt::Status::kOk) return Fail(h, Convert(status), "native trainer destroy failed");
        delete h;
        *pointer = nullptr;
        return MGT_STATUS_OK;
    });
}

extern "C" size_t mgt_single_gpu_v1_last_error(
    MgtSingleGpuHandle* h, char* destination, size_t capacity) {
    const auto& message = h ? h->error : g_error;
    const size_t required = message.size() + 1;
    if (destination && capacity) {
        const size_t count = std::min(message.size(), capacity - 1);
        std::memcpy(destination, message.data(), count);
        destination[count] = '\0';
    }
    return required;
}
