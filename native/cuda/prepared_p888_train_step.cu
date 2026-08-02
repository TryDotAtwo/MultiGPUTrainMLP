#include "mgt_cuda/prepared_p888_train_step.cuh"

#include "mgt_cuda/allreduce_nccl.cuh"
#include "mgt_cuda/bf16_batch_norm_sites.cuh"
#include "mgt_cuda/bf16_linear_train_ops.cuh"
#include "mgt_cuda/input_embedding_bf16.cuh"

#include <algorithm>
#include <filesystem>
#include <limits>
#include <new>
#include <vector>

namespace mgt_cuda {

enum class PreparedRuntimeMode { kStrict, kBf16 };

struct PreparedP888TrainRuntime {
    PreparedRuntimeMode mode = PreparedRuntimeMode::kStrict;
    PreparedP888StrictRuntimeCreateInfo info{};
    PreparedP888Bf16RuntimeCreateInfo bf16{};
    std::vector<std::uint32_t> supported_rows;
    NcclRankContext* context = nullptr;
    cublasHandle_t blas = nullptr;
    cudaStream_t stream = nullptr;
    cudaEvent_t completion = nullptr;
    std::uint64_t sequence = 0;
};

namespace {

bool HasAllBuffers(const MlpBatchNormStepBuffers& b) {
    return b.weights && b.weight_grad && b.weight_m && b.weight_v && b.affine &&
           b.affine_grad && b.affine_m && b.affine_v && b.running && b.outputs &&
           b.forward_workspace && b.loss && b.output_dy && b.block_grad && b.fc1_grad &&
           b.residual_grad && b.input_grad;
}

mgt::Status ValidateRows(std::uint32_t capacity,const std::uint32_t* values,std::uint32_t count) {
    if (!capacity || !values || !count) return mgt::Status::kInvalidConfig;
    std::vector<std::uint32_t> rows(values, values + count);
    if (std::any_of(rows.begin(), rows.end(), [&](std::uint32_t value) {
            return value == 0 || value > capacity;
        })) return mgt::Status::kInvalidConfig;
    std::sort(rows.begin(), rows.end());
    return std::adjacent_find(rows.begin(), rows.end()) == rows.end()
        ? mgt::Status::kOk : mgt::Status::kInvalidConfig;
}

mgt::Status ValidateCreateInfo(const PreparedP888StrictRuntimeCreateInfo& info) {
    if (ValidateCudaMlpShape(info.shape) != mgt::Status::kOk || info.capacity_rows == 0 ||
        info.supported_active_rows == nullptr || info.supported_active_row_count == 0 ||
        info.world == 0 || info.rank >= info.world || !HasAllBuffers(info.buffers) ||
        !info.state_slots[0] || !info.state_slots[1] || !info.label_slots[0] ||
        !info.label_slots[1] || info.batch_norm_plan.sites.size() !=
            2ULL + 2ULL * info.shape.residual_blocks ||
        info.batch_norm_plan.workspace_floats == 0 ||
        ValidateAdamWKernelConfig(info.adam) != mgt::Status::kOk) {
        return mgt::Status::kInvalidConfig;
    }
    if (info.world > 1 &&
        (info.strict_nccl_id_file == nullptr || info.strict_nccl_id_file[0] == '\0')) {
        return mgt::Status::kInvalidConfig;
    }
    std::vector<std::uint32_t> rows(
        info.supported_active_rows,
        info.supported_active_rows + info.supported_active_row_count);
    if (std::any_of(rows.begin(), rows.end(), [&](std::uint32_t value) {
            return value == 0 || value > info.capacity_rows;
        })) {
        return mgt::Status::kInvalidConfig;
    }
    std::sort(rows.begin(), rows.end());
    if (std::adjacent_find(rows.begin(), rows.end()) != rows.end()) {
        return mgt::Status::kInvalidConfig;
    }
    const auto required = MlpBatchNormForwardWorkspaceFloats(
        info.shape, info.batch_norm_plan, info.capacity_rows);
    if (required == 0) return mgt::Status::kInvalidConfig;
    return mgt::Status::kOk;
}

mgt::Status ValidateBf16CreateInfo(const PreparedP888Bf16RuntimeCreateInfo& info) {
    const auto& h = info.hidden;
    const auto& input = info.input;
    if (!info.runtime || !info.output_upstream || !input.state_slots[0] ||
        !input.state_slots[1] || !input.table || !input.table_grad || !input.gamma ||
        !input.beta || !input.running_mean || !input.running_variance ||
        !input.saved_mean || !input.saved_inv_std || !input.dgamma || !input.dbeta ||
        input.batch_norm_momentum < 0.0f || input.batch_norm_momentum > 1.0f ||
        input.batch_norm_epsilon <= 0.0f || !h.weight ||
        !h.weight_grad || !h.gamma || !h.beta || !h.running_mean ||
        !h.running_variance || !h.saved_mean || !h.saved_inv_std || !h.dgamma ||
        !h.dbeta || h.batch_norm_momentum < 0.0f || h.batch_norm_momentum > 1.0f ||
        h.batch_norm_epsilon <= 0.0f ||
        ValidateBf16ResidualStackBindings(info.residual_stack, true) != mgt::Status::kOk)
        return mgt::Status::kInvalidConfig;
    return ValidateRows(info.capacity_rows, info.supported_active_rows,
                        info.supported_active_row_count);
}

}  // namespace

mgt::Status QueryPreparedP888StrictRuntimeBytes(
    const PreparedP888StrictRuntimeCreateInfo& info,
    std::uint64_t* bytes) {
    if (bytes == nullptr) return mgt::Status::kInvalidConfig;
    const auto status = ValidateCreateInfo(info);
    if (status != mgt::Status::kOk) return status;
    const std::uint64_t rows_bytes =
        static_cast<std::uint64_t>(info.supported_active_row_count) * sizeof(std::uint32_t);
    if (rows_bytes > std::numeric_limits<std::uint64_t>::max() -
                         sizeof(PreparedP888TrainRuntime)) {
        return mgt::Status::kCapacityExceeded;
    }
    *bytes = sizeof(PreparedP888TrainRuntime) + rows_bytes;
    return mgt::Status::kOk;
}

mgt::Status CreatePreparedP888StrictRuntime(
    const PreparedP888StrictRuntimeCreateInfo& info,
    PreparedP888TrainRuntime** out) {
    if (out == nullptr) return mgt::Status::kInvalidConfig;
    *out = nullptr;
    std::uint64_t ignored = 0;
    auto status = QueryPreparedP888StrictRuntimeBytes(info, &ignored);
    if (status != mgt::Status::kOk) return status;
    if (cudaSetDevice(static_cast<int>(info.device_id)) != cudaSuccess)
        return mgt::Status::kCudaFailure;

    auto* runtime = new (std::nothrow) PreparedP888TrainRuntime;
    if (!runtime) return mgt::Status::kCapacityExceeded;
    runtime->info = info;
    runtime->supported_rows.assign(
        info.supported_active_rows,
        info.supported_active_rows + info.supported_active_row_count);
    runtime->info.supported_active_rows = runtime->supported_rows.data();

    if (cudaStreamCreateWithFlags(&runtime->stream, cudaStreamNonBlocking) != cudaSuccess) {
        delete runtime;
        return mgt::Status::kCudaFailure;
    }
    if (cudaEventCreateWithFlags(&runtime->completion, cudaEventDisableTiming) != cudaSuccess) {
        cudaStreamDestroy(runtime->stream);
        delete runtime;
        return mgt::Status::kCudaFailure;
    }
    if (cublasCreate(&runtime->blas) != CUBLAS_STATUS_SUCCESS ||
        cublasSetMathMode(runtime->blas, CUBLAS_DEFAULT_MATH) != CUBLAS_STATUS_SUCCESS) {
        DestroyPreparedP888TrainRuntime(runtime);
        return mgt::Status::kCudaFailure;
    }
    status = info.world == 1
        ? CreateNcclSingleRankContext(info.device_id, &runtime->context)
        : CreateNcclRankContext(
              info.device_id, info.world, info.rank,
              std::filesystem::path(info.strict_nccl_id_file), &runtime->context);
    if (status != mgt::Status::kOk) {
        DestroyPreparedP888TrainRuntime(runtime);
        return status;
    }
    *out = runtime;
    return mgt::Status::kOk;
}

mgt::Status CreatePreparedP888Bf16Runtime(
    const PreparedP888Bf16RuntimeCreateInfo& info,
    PreparedP888TrainRuntime** out) {
    if (!out) return mgt::Status::kInvalidConfig;
    *out = nullptr;
    const auto status = ValidateBf16CreateInfo(info);
    if (status != mgt::Status::kOk) return status;
    auto* runtime = new (std::nothrow) PreparedP888TrainRuntime;
    if (!runtime) return mgt::Status::kCapacityExceeded;
    runtime->mode = PreparedRuntimeMode::kBf16;
    runtime->bf16 = info;
    runtime->supported_rows.assign(info.supported_active_rows,
                                   info.supported_active_rows + info.supported_active_row_count);
    runtime->bf16.supported_active_rows = runtime->supported_rows.data();
    runtime->stream = A100Bf16RuntimeComputeStream(info.runtime);
    if (!runtime->stream ||
        cudaEventCreateWithFlags(&runtime->completion, cudaEventDisableTiming) != cudaSuccess) {
        runtime->stream = nullptr;
        delete runtime;
        return mgt::Status::kCudaFailure;
    }
    *out = runtime;
    return mgt::Status::kOk;
}
mgt::Status DestroyPreparedP888TrainRuntime(PreparedP888TrainRuntime* runtime) {
    if (!runtime) return mgt::Status::kInvalidConfig;
    mgt::Status result = mgt::Status::kOk;
    if (runtime->stream && cudaStreamSynchronize(runtime->stream) != cudaSuccess)
        result = mgt::Status::kCudaFailure;
    if (runtime->context && DestroyNcclRankContext(runtime->context) != mgt::Status::kOk)
        result = mgt::Status::kNcclFailure;
    if (runtime->blas && cublasDestroy(runtime->blas) != CUBLAS_STATUS_SUCCESS)
        result = mgt::Status::kCudaFailure;
    if (runtime->completion && cudaEventDestroy(runtime->completion) != cudaSuccess)
        result = mgt::Status::kCudaFailure;
    if (runtime->mode == PreparedRuntimeMode::kStrict && runtime->stream &&
        cudaStreamDestroy(runtime->stream) != cudaSuccess)
        result = mgt::Status::kCudaFailure;
    delete runtime;
    return result;
}

mgt::Status LaunchPreparedP888TrainStep(
    PreparedP888TrainRuntime* runtime,
    const PreparedTrainStepRequest& request,
    PreparedTrainStepTicket* ticket) {
    if (!runtime || !ticket || request.batch_slot >= 2 || request.active_rows == 0 ||
        request.global_rows < request.active_rows || request.optimizer_step == 0 ||
        request.global_offset > request.global_rows - request.active_rows ||
        std::find(runtime->supported_rows.begin(), runtime->supported_rows.end(),
                  request.active_rows) == runtime->supported_rows.end()) {
        return mgt::Status::kInvalidConfig;
    }
    if (runtime->mode == PreparedRuntimeMode::kBf16) {
        const std::uint64_t next_sequence = runtime->sequence + 1;
        if (request.optimizer_step != next_sequence ||
            request.batch_slot != static_cast<std::uint32_t>(next_sequence & 1ULL))
            return mgt::Status::kInvalidConfig;
        auto* control = A100Bf16RuntimeStepControl(runtime->bf16.runtime);
        const auto slot = request.active_rows == runtime->bf16.capacity_rows
            ? A100LocalGraphSlot::kFull : A100LocalGraphSlot::kTail;
        auto status = LaunchConfigureP888StepControl(
            control, request.active_rows, request.global_rows, request.global_offset,
            runtime->stream);
        if (status != mgt::Status::kOk) return status;
        status = LaunchBeginP888StepControl(control, slot, runtime->stream);
        if (status != mgt::Status::kOk) return status;
        const Bf16LinearProblem hidden_problem{
            request.active_rows, 2, request.active_rows, 2560, 224};
        const auto* linear = A100Bf16RuntimeLinearPlan(runtime->bf16.runtime);
        const auto* input_plan = A100Bf16RuntimeInputEmbeddingPlan(runtime->bf16.runtime);
        float* preactivation = A100Bf16RuntimePreactivationScratch(runtime->bf16.runtime);
        if (!linear || !input_plan || !preactivation) return mgt::Status::kInvalidConfig;
        const auto& input = runtime->bf16.input;
        status = LaunchInputEmbeddingForwardBf16(
            input_plan, input.state_slots[request.batch_slot], input.table,
            request.active_rows, preactivation, runtime->stream);
        if (status != mgt::Status::kOk) return status;
        status = LaunchA100TiledSyncBatchNormForwardSite(
            runtime->bf16.runtime, 0, preactivation, nullptr, request.active_rows,
            request.global_rows, input.gamma, input.beta, input.running_mean,
            input.running_variance, input.batch_norm_momentum,
            input.batch_norm_epsilon, input.saved_mean, input.saved_inv_std);
        if (status != mgt::Status::kOk) return status;
        Bf16BatchNormSiteView input_site{};
        status = QueryA100Bf16RuntimeBatchNormSite(runtime->bf16.runtime, 0, 0, &input_site);
        if (status != mgt::Status::kOk) return status;
        const auto& hidden = runtime->bf16.hidden;
        status = LaunchBf16LinearForwardToFloat(
            linear, hidden_problem, input_site.activation, hidden.weight,
            preactivation, runtime->stream);
        if (status != mgt::Status::kOk) return status;
        status = LaunchA100TiledSyncBatchNormForwardSite(
            runtime->bf16.runtime, 1, preactivation, nullptr, request.active_rows,
            request.global_rows, hidden.gamma, hidden.beta, hidden.running_mean,
            hidden.running_variance, hidden.batch_norm_momentum,
            hidden.batch_norm_epsilon, hidden.saved_mean, hidden.saved_inv_std);
        if (status != mgt::Status::kOk) return status;
        Bf16BatchNormSiteView hidden_site{};
        status = QueryA100Bf16RuntimeBatchNormSite(
            runtime->bf16.runtime, 1, 0, &hidden_site);
        if (status != mgt::Status::kOk) return status;
        auto residual = runtime->bf16.residual_stack;
        residual.input_activation = hidden_site.activation;
        const __nv_bfloat16* output = nullptr;
        status = LaunchA100Bf16ResidualStackForward(
            runtime->bf16.runtime, residual, request.active_rows,
            request.global_rows, &output);
        if (status != mgt::Status::kOk) return status;
        if (!output) return mgt::Status::kInvalidConfig;
        float* input_gradient = nullptr;
        status = LaunchA100Bf16ResidualStackBackward(
            runtime->bf16.runtime, residual, runtime->bf16.output_upstream,
            request.active_rows, request.global_rows, &input_gradient);
        if (status != mgt::Status::kOk) return status;
        if (!input_gradient) return mgt::Status::kInvalidConfig;
        status = LaunchA100TiledSyncBatchNormBackwardSite(
            runtime->bf16.runtime, 1, 0, input_gradient, hidden.saved_inv_std,
            hidden.gamma, hidden.dgamma, hidden.dbeta, request.active_rows,
            request.global_rows, nullptr);
        if (status != mgt::Status::kOk) return status;
        status = QueryA100Bf16RuntimeBatchNormSite(
            runtime->bf16.runtime, 1, 0, &hidden_site);
        if (status != mgt::Status::kOk) return status;
        status = LaunchBf16LinearGradWeightToFloat(
            linear, hidden_problem, input_site.activation, hidden_site.dz,
            hidden.weight_grad, runtime->stream);
        if (status != mgt::Status::kOk) return status;
        status = LaunchBf16LinearGradInputToFloat(
            linear, hidden_problem, hidden_site.dz, hidden.weight,
            A100Bf16RuntimeGradInputScratch(runtime->bf16.runtime), 0.0f,
            runtime->stream);
        if (status != mgt::Status::kOk) return status;
        status = LaunchA100TiledSyncBatchNormBackwardSite(
            runtime->bf16.runtime, 0, 0,
            A100Bf16RuntimeGradInputScratch(runtime->bf16.runtime), input.saved_inv_std,
            input.gamma, input.dgamma, input.dbeta, request.active_rows,
            request.global_rows, nullptr);
        if (status != mgt::Status::kOk) return status;
        status = QueryA100Bf16RuntimeBatchNormSite(runtime->bf16.runtime, 0, 0, &input_site);
        if (status != mgt::Status::kOk) return status;
        status = LaunchInputEmbeddingTableGradBf16(
            input_plan, input.state_slots[request.batch_slot], input_site.dz,
            request.active_rows, input.table_grad, runtime->stream);
        if (status != mgt::Status::kOk) return status;
        status = LaunchCommitP888StepControl(control, runtime->stream);
        if (status != mgt::Status::kOk) return status;
        if (cudaEventRecord(runtime->completion, runtime->stream) != cudaSuccess)
            return mgt::Status::kCudaFailure;
        runtime->sequence = next_sequence;
        ticket->completion_event = runtime->completion;
        ticket->sequence = next_sequence;
        return mgt::Status::kOk;
    }

    AdamWKernelConfig adam = runtime->info.adam;
    adam.step = request.optimizer_step;
    const auto workspace_floats = MlpBatchNormForwardWorkspaceFloats(
        runtime->info.shape, runtime->info.batch_norm_plan, runtime->info.capacity_rows);
    const auto status = LaunchMlpBatchNormTrainStep(
        runtime->info.shape,
        runtime->info.batch_norm_plan.sites[0].logical_features,
        runtime->info.batch_norm_plan.sites[1].logical_features,
        runtime->info.state_slots[request.batch_slot],
        runtime->info.label_slots[request.batch_slot],
        request.active_rows,
        request.global_rows,
        runtime->info.batch_norm_plan,
        workspace_floats,
        adam,
        runtime->info.buffers,
        runtime->context,
        runtime->blas,
        runtime->stream);
    if (status != mgt::Status::kOk) return status;
    if (cudaEventRecord(runtime->completion, runtime->stream) != cudaSuccess)
        return mgt::Status::kCudaFailure;
    ticket->completion_event = runtime->completion;
    ticket->sequence = ++runtime->sequence;
    return mgt::Status::kOk;
}

mgt::Status RecordPreparedP888TrainEvent(
    PreparedP888TrainRuntime* runtime,
    cudaEvent_t event) {
    if (!runtime || !event) return mgt::Status::kInvalidConfig;
    return cudaEventRecord(event, runtime->stream) == cudaSuccess
        ? mgt::Status::kOk
        : mgt::Status::kCudaFailure;
}

}  // namespace mgt_cuda
