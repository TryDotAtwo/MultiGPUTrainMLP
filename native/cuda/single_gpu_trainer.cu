#include "mgt_cuda/single_gpu_trainer.cuh"

#include "mgt/batch_norm_training.hpp"
#include "mgt/input_active_bins.hpp"
#include "mgt_cuda/local_mlp_batch_norm.cuh"
#include "mgt_cuda/random_walk_kernel.cuh"
#include "mgt_cuda/single_gpu_train_graph.cuh"

#include <atomic>
#include <limits>
#include <new>

namespace mgt_cuda {
namespace {

constexpr std::uint64_t kAlignment = 256;
std::atomic<std::uint64_t> g_allocation_count{0};

struct ArenaLayout {
    std::uint64_t weights, weight_half, weight_grad, weight_m, weight_v;
    std::uint64_t affine, affine_grad, affine_m, affine_v, running;
    std::uint64_t moves, target, states, labels, walk_meta, outputs, forward_workspace, loss;
    std::uint64_t output_dy, block_grad, fc1_grad, residual_grad, input_grad;
    std::uint64_t fp16_operand_a, fp16_operand_b;
    std::uint64_t input_active_bins;
    std::uint64_t blas_workspace;
    std::uint64_t bytes, parameter_count, forward_workspace_floats;
};

bool AddSlice(std::uint64_t count, std::uint64_t element_bytes,
              std::uint64_t* cursor, std::uint64_t* offset) {
    if (!cursor || !offset || count > std::numeric_limits<std::uint64_t>::max() / element_bytes)
        return false;
    const std::uint64_t aligned = (*cursor + kAlignment - 1) & ~(kAlignment - 1);
    const std::uint64_t bytes = count * element_bytes;
    if (aligned < *cursor || bytes > std::numeric_limits<std::uint64_t>::max() - aligned)
        return false;
    *offset = aligned;
    *cursor = aligned + bytes;
    return true;
}

CudaMlpShape Shape(const mgt::SingleGpuModelContract& c) {
    return {c.state_len, c.state_value_count, c.physical_hd1, c.physical_hd2,
            c.residual_blocks, c.output_dim};
}

mgt::Status BuildLayout(const SingleGpuTrainerCreateInfo& info,
                        mgt::BatchNormTrainingPlan* plan, ArenaLayout* layout) {
    const bool graph_mode = info.execution_mode == SingleGpuExecutionMode::kFixedBatchGraph;
    if ((info.execution_mode != SingleGpuExecutionMode::kEager && !graph_mode) ||
        (info.input_gradient_precision != SingleGpuInputGradientPrecision::kFp32 &&
         info.input_gradient_precision !=
             SingleGpuInputGradientPrecision::kFp16Mirror) ||
        (graph_mode && (!detail::kSingleGpuTrainGraphSupported ||
                       info.capacity_rows > mgt::P888TrainingContract::kSamplesPerEpoch)))
        return mgt::Status::kInvalidConfig;
    auto adam = info.adam;
    adam.param_count = 1;
    adam.step = 1;
    if (!plan || !layout || !info.capacity_rows || !info.puzzle || !info.base_seed ||
        !info.k_min || info.k_min > info.k_max ||
        mgt::ValidateSingleGpuModelContract(info.contract) != mgt::Status::kOk ||
        ValidateAdamWKernelConfig(adam) != mgt::Status::kOk)
        return mgt::Status::kInvalidConfig;
    const auto shape = Shape(info.contract);
    const auto parameter_count = MlpBatchNormParameterCount(shape);
    if (!parameter_count || mgt::BuildBatchNormTrainingPlan(
            info.contract.logical_hd1, info.contract.logical_hd2,
            info.contract.physical_hd1, info.contract.physical_hd2,
            info.contract.residual_blocks, info.capacity_rows, plan) != mgt::Status::kOk)
        return mgt::Status::kInvalidConfig;
    const auto workspace = LocalMlpBatchNormForwardWorkspaceFloats(
        shape, *plan, info.capacity_rows);
    const std::uint64_t activation_tape_halfs =
        static_cast<std::uint64_t>(info.capacity_rows) * shape.hd1 +
        2ULL * shape.residual_blocks * info.capacity_rows * shape.hd2;
    if (!workspace) return mgt::Status::kCapacityExceeded;
    ArenaLayout out{};
    out.parameter_count = parameter_count;
    out.forward_workspace_floats = workspace;
    std::uint64_t cursor = 0;
    auto floats = [&](std::uint64_t count, std::uint64_t* offset) {
        return AddSlice(count, sizeof(float), &cursor, offset);
    };
    if (!floats(parameter_count, &out.weights) ||
        !AddSlice(parameter_count, sizeof(__half), &cursor, &out.weight_half) ||
        !floats(parameter_count, &out.weight_grad) ||
        !floats(parameter_count, &out.weight_m) ||
        !floats(parameter_count, &out.weight_v) ||
        !floats(plan->trainable_count, &out.affine) ||
        !floats(plan->trainable_count, &out.affine_grad) ||
        !floats(plan->trainable_count, &out.affine_m) ||
        !floats(plan->trainable_count, &out.affine_v) ||
        !floats(plan->running_count, &out.running) ||
        !AddSlice(mgt::kMoveCount, sizeof(mgt::TrainStateStorage), &cursor, &out.moves) ||
        !AddSlice(1, sizeof(mgt::TrainStateStorage), &cursor, &out.target) ||
        !AddSlice(info.capacity_rows, sizeof(mgt::TrainStateStorage), &cursor, &out.states) ||
        !floats(static_cast<std::uint64_t>(info.capacity_rows) * shape.output_dim, &out.labels) ||
        !AddSlice(info.capacity_rows, sizeof(mgt::WalkMeta), &cursor, &out.walk_meta) ||
        !floats(static_cast<std::uint64_t>(info.capacity_rows) * shape.output_dim, &out.outputs) ||
        !floats(workspace, &out.forward_workspace) || !floats(1, &out.loss) ||
        !floats(static_cast<std::uint64_t>(info.capacity_rows) * shape.output_dim, &out.output_dy) ||
        !floats(static_cast<std::uint64_t>(info.capacity_rows) * shape.hd2, &out.block_grad) ||
        !floats(static_cast<std::uint64_t>(info.capacity_rows) * shape.hd2, &out.fc1_grad) ||
        !floats(static_cast<std::uint64_t>(info.capacity_rows) * shape.hd2, &out.residual_grad) ||
        !floats(static_cast<std::uint64_t>(info.capacity_rows) * shape.hd1, &out.input_grad) ||
        !AddSlice(activation_tape_halfs,
                  sizeof(__half), &cursor, &out.fp16_operand_a) ||
        !AddSlice(static_cast<std::uint64_t>(info.capacity_rows) * shape.hd2,
                  sizeof(__half), &cursor, &out.fp16_operand_b) ||
        !AddSlice(static_cast<std::uint64_t>(shape.state_len) * shape.state_value_pad,
                  sizeof(std::uint16_t), &cursor, &out.input_active_bins))
        return mgt::Status::kCapacityExceeded;
    if (graph_mode && !AddSlice(detail::kSingleGpuBlasWorkspaceBytes, 1,
                               &cursor, &out.blas_workspace))
        return mgt::Status::kCapacityExceeded;
    out.bytes = (cursor + kAlignment - 1) & ~(kAlignment - 1);
    *layout = out;
    return mgt::Status::kOk;
}

__global__ void FillOnes(float* values, std::uint64_t count) {
    const auto index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) values[index] = 1.0f;
}

__global__ void InitializeWeights(float* values, std::uint64_t count, std::uint64_t seed) {
    const auto index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) {
        std::uint64_t bits = index + seed + 0x9e3779b97f4a7c15ULL;
        bits = (bits ^ (bits >> 30)) * 0xbf58476d1ce4e5b9ULL;
        bits = (bits ^ (bits >> 27)) * 0x94d049bb133111ebULL;
        bits ^= bits >> 31;
        values[index] = (static_cast<int>(bits & 2047U) - 1024) * (1.0f / 1048576.0f);
    }
}

template <class T>
T* At(void* arena, std::uint64_t offset) {
    return reinterpret_cast<T*>(static_cast<unsigned char*>(arena) + offset);
}

}  // namespace

struct SingleGpuTrainer {
    SingleGpuTrainerCreateInfo info{};
    mgt::PuzzleDefinition puzzle{};
    CudaMlpShape shape{};
    mgt::BatchNormTrainingPlan plan{};
    std::vector<std::uint16_t> input_active_bins;
    ArenaLayout layout{};
    void* arena = nullptr;
    cudaStream_t stream = nullptr;
    cudaEvent_t completion = nullptr;
    cublasHandle_t blas = nullptr;
    detail::SingleGpuTrainGraph graph{};
    bool prepared = false;
    bool failed = false;
    bool in_flight = false;
    std::uint64_t sequence = 0;
    std::uint64_t completed = 0;
    float last_loss = 0.0f;
};

namespace {
mgt::Status EnqueueSingleGpuTrainStep(
    SingleGpuTrainer* trainer, const SingleGpuTrainStepRequest& request);

mgt::Status FailTrainer(SingleGpuTrainer* trainer,
                        mgt::Status status = mgt::Status::kCudaFailure) {
    trainer->failed = true;
    return status;
}
}  // namespace

mgt::Status QuerySingleGpuTrainerBytes(
    const SingleGpuTrainerCreateInfo& info, std::uint64_t* bytes) {
    if (!bytes) return mgt::Status::kInvalidConfig;
    mgt::BatchNormTrainingPlan plan;
    ArenaLayout layout{};
    const auto status = BuildLayout(info, &plan, &layout);
    if (status != mgt::Status::kOk) return status;
    *bytes = layout.bytes;
    return mgt::Status::kOk;
}

mgt::Status CreateSingleGpuTrainer(
    const SingleGpuTrainerCreateInfo& info, SingleGpuTrainer** out) {
    if (!out) return mgt::Status::kInvalidConfig;
    *out = nullptr;
    auto* trainer = new (std::nothrow) SingleGpuTrainer;
    if (!trainer) return mgt::Status::kCapacityExceeded;
    mgt::Status status;
    try { status = BuildLayout(info, &trainer->plan, &trainer->layout); }
    catch (const std::bad_alloc&) { delete trainer; return mgt::Status::kCapacityExceeded; }
    catch (...) { delete trainer; return mgt::Status::kInvalidConfig; }
    if (status != mgt::Status::kOk) { delete trainer; return status; }
    trainer->info = info;
    trainer->puzzle = *info.puzzle;
    trainer->info.puzzle = &trainer->puzzle;
    trainer->shape = Shape(info.contract);
    status = mgt::BuildInputActiveBins(
        trainer->puzzle, trainer->shape.state_len, trainer->shape.state_value_pad,
        &trainer->input_active_bins);
    if (status != mgt::Status::kOk) {
        delete trainer;
        return status;
    }
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess ||
        info.device_id >= static_cast<std::uint32_t>(device_count)) {
        delete trainer;
        return mgt::Status::kCudaFailure;
    }
    if (cudaSetDevice(static_cast<int>(info.device_id)) != cudaSuccess ||
        cudaStreamCreateWithFlags(&trainer->stream, cudaStreamNonBlocking) != cudaSuccess ||
        cudaEventCreateWithFlags(&trainer->completion, cudaEventDisableTiming) != cudaSuccess ||
        cublasCreate(&trainer->blas) != CUBLAS_STATUS_SUCCESS ||
        cublasSetStream(trainer->blas, trainer->stream) != CUBLAS_STATUS_SUCCESS ||
        cublasSetMathMode(trainer->blas, CUBLAS_DEFAULT_MATH) != CUBLAS_STATUS_SUCCESS) {
        DestroySingleGpuTrainer(&trainer);
        return mgt::Status::kCudaFailure;
    }
    if (cudaMalloc(&trainer->arena, trainer->layout.bytes) != cudaSuccess) {
        DestroySingleGpuTrainer(&trainer);
        return mgt::Status::kCapacityExceeded;
    }
    ++g_allocation_count;
    *out = trainer;
    return mgt::Status::kOk;
}

mgt::Status PrepareSingleGpuTrainer(SingleGpuTrainer* trainer) {
    if (!trainer || !trainer->arena) return mgt::Status::kInvalidConfig;
    if (trainer->failed) return mgt::Status::kCudaFailure;
    if (trainer->prepared) return mgt::Status::kInvalidConfig;
    if (cudaMemsetAsync(trainer->arena, 0, trainer->layout.bytes, trainer->stream) != cudaSuccess)
        return FailTrainer(trainer);
    if (cudaMemcpyAsync(At<mgt::TrainStateStorage>(trainer->arena, trainer->layout.moves),
                        trainer->puzzle.moves.data(), sizeof(trainer->puzzle.moves),
                        cudaMemcpyHostToDevice, trainer->stream) != cudaSuccess ||
        cudaMemcpyAsync(At<mgt::TrainStateStorage>(trainer->arena, trainer->layout.target),
                        &trainer->puzzle.target, sizeof(trainer->puzzle.target),
                        cudaMemcpyHostToDevice, trainer->stream) != cudaSuccess ||
        cudaMemcpyAsync(At<std::uint16_t>(trainer->arena, trainer->layout.input_active_bins),
                        trainer->input_active_bins.data(),
                        trainer->input_active_bins.size() * sizeof(std::uint16_t),
                        cudaMemcpyHostToDevice, trainer->stream) != cudaSuccess)
        return FailTrainer(trainer);
    auto* affine = At<float>(trainer->arena, trainer->layout.affine);
    auto* running = At<float>(trainer->arena, trainer->layout.running);
    const auto logical = trainer->plan.logical_feature_count;
    const unsigned threads = 256;
    const unsigned blocks = static_cast<unsigned>((logical + threads - 1) / threads);
    const unsigned weight_blocks = static_cast<unsigned>(
        (trainer->layout.parameter_count + threads - 1) / threads);
    InitializeWeights<<<weight_blocks, threads, 0, trainer->stream>>>(
        At<float>(trainer->arena, trainer->layout.weights),
        trainer->layout.parameter_count, trainer->info.base_seed);
    FillOnes<<<blocks, threads, 0, trainer->stream>>>(affine, logical);
    FillOnes<<<blocks, threads, 0, trainer->stream>>>(running + logical, logical);
    if (LaunchFloatToHalf(
            At<float>(trainer->arena, trainer->layout.weights),
            At<__half>(trainer->arena, trainer->layout.weight_half),
            trainer->layout.parameter_count, trainer->stream) != mgt::Status::kOk ||
        cudaPeekAtLastError() != cudaSuccess ||
        cudaStreamSynchronize(trainer->stream) != cudaSuccess) return FailTrainer(trainer);
    if (trainer->info.execution_mode == SingleGpuExecutionMode::kFixedBatchGraph) {
        if (cublasSetWorkspace(trainer->blas,
                At<unsigned char>(trainer->arena, trainer->layout.blas_workspace),
                detail::kSingleGpuBlasWorkspaceBytes) != CUBLAS_STATUS_SUCCESS ||
            cudaStreamBeginCapture(trainer->stream, cudaStreamCaptureModeGlobal) != cudaSuccess)
            return FailTrainer(trainer);
        cudaGraph_t captured = nullptr;
        const auto body = EnqueueSingleGpuTrainStep(
            trainer, {trainer->info.capacity_rows, 1, 0, 0});
        const auto ended = cudaStreamEndCapture(trainer->stream, &captured);
        if (body != mgt::Status::kOk || ended != cudaSuccess) {
            trainer->graph.source = captured;
            return FailTrainer(trainer, body == mgt::Status::kOk ? mgt::Status::kCudaFailure : body);
        }
        const auto status = detail::InstantiateSingleGpuTrainGraph(
            captured, trainer->info.capacity_rows, trainer->layout.parameter_count,
            trainer->plan.trainable_count, &trainer->graph);
        if (status != mgt::Status::kOk) return FailTrainer(trainer, status);
    }
    trainer->prepared = true;
    return mgt::Status::kOk;
}

namespace {
mgt::Status EnqueueSingleGpuTrainStep(
    SingleGpuTrainer* trainer, const SingleGpuTrainStepRequest& request) try {
    auto* arena = trainer->arena;
    MlpBatchNormStepBuffers buffers{
        At<float>(arena, trainer->layout.weights), At<float>(arena, trainer->layout.weight_grad),
        At<float>(arena, trainer->layout.weight_m), At<float>(arena, trainer->layout.weight_v),
        At<float>(arena, trainer->layout.affine), At<float>(arena, trainer->layout.affine_grad),
        At<float>(arena, trainer->layout.affine_m), At<float>(arena, trainer->layout.affine_v),
        At<float>(arena, trainer->layout.running), At<float>(arena, trainer->layout.outputs),
        At<float>(arena, trainer->layout.forward_workspace), At<float>(arena, trainer->layout.loss),
        At<float>(arena, trainer->layout.output_dy), At<float>(arena, trainer->layout.block_grad),
        At<float>(arena, trainer->layout.fc1_grad), At<float>(arena, trainer->layout.residual_grad),
        At<float>(arena, trainer->layout.input_grad)};
    auto adam = trainer->info.adam;
    adam.step = request.optimizer_step;
    RandomWalkKernelConfig walk{
        request.active_rows, trainer->info.k_min, trainer->info.k_max,
        mgt::kMoveCount, trainer->info.contract.state_len, mgt::kStateStorageLen};
    walk.epoch_sample_offset = request.epoch_sample_offset;
    walk.original_p888_schedule = 1;
    auto status = LaunchRandomWalkKernel(
        walk, trainer->info.base_seed, request.epoch, request.optimizer_step,
        trainer->info.global_rank,
        At<mgt::TrainStateStorage>(arena, trainer->layout.moves),
        At<mgt::TrainStateStorage>(arena, trainer->layout.target),
        At<mgt::TrainStateStorage>(arena, trainer->layout.states),
        At<float>(arena, trainer->layout.labels),
        At<mgt::WalkMeta>(arena, trainer->layout.walk_meta), trainer->stream);
    if (status != mgt::Status::kOk) return status;
    LocalMlpFp16Context fp16{
        buffers.weights, At<__half>(arena, trainer->layout.weight_half),
        At<__half>(arena, trainer->layout.fp16_operand_a),
        At<__half>(arena, trainer->layout.fp16_operand_b),
        static_cast<std::uint64_t>(trainer->info.capacity_rows) * trainer->shape.hd1 +
            2ULL * trainer->shape.residual_blocks * trainer->info.capacity_rows *
                trainer->shape.hd2,
        static_cast<std::uint64_t>(trainer->info.capacity_rows) * trainer->shape.hd2};
    fp16.input_active_bins =
        At<std::uint16_t>(arena, trainer->layout.input_active_bins);
    fp16.input_active_bin_count =
        static_cast<std::uint32_t>(trainer->input_active_bins.size());
    fp16.input_inactive_gradients_are_persistent_zero =
        trainer->input_active_bins.size() <
        static_cast<std::uint64_t>(trainer->shape.state_len) *
            trainer->shape.state_value_pad;
    if (trainer->info.input_gradient_precision ==
        SingleGpuInputGradientPrecision::kFp16Mirror) {
        fp16.input_gradient_half =
            At<__half>(arena, trainer->layout.fp16_operand_a);
        fp16.input_gradient_half_capacity =
            static_cast<std::uint64_t>(trainer->info.capacity_rows) *
                trainer->shape.hd1;
    }
    status = LaunchLocalMlpBatchNormTrainStepFp16(
        trainer->shape, trainer->info.contract.logical_hd1,
        trainer->info.contract.logical_hd2,
        At<mgt::TrainStateStorage>(arena, trainer->layout.states),
        At<float>(arena, trainer->layout.labels), request.active_rows, trainer->plan,
        trainer->layout.forward_workspace_floats, adam, buffers,
        &fp16, trainer->blas, trainer->stream);
    return status;
} catch (const std::bad_alloc&) {
    return mgt::Status::kCapacityExceeded;
} catch (...) {
    return mgt::Status::kCudaFailure;
}
}  // namespace

mgt::Status LaunchSingleGpuTrainStep(
    SingleGpuTrainer* trainer, const SingleGpuTrainStepRequest& request,
    SingleGpuTrainStepTicket* ticket) {
    if (!trainer || !ticket) return mgt::Status::kInvalidConfig;
    if (trainer->failed) return mgt::Status::kCudaFailure;
    if (!trainer->prepared || !request.active_rows || !request.optimizer_step ||
        trainer->sequence == std::numeric_limits<std::uint64_t>::max() ||
        request.optimizer_step != trainer->sequence + 1) return mgt::Status::kInvalidConfig;
    if (request.active_rows > trainer->info.capacity_rows) return mgt::Status::kCapacityExceeded;
    if (request.epoch_sample_offset > mgt::P888TrainingContract::kSamplesPerEpoch ||
        request.active_rows > mgt::P888TrainingContract::kSamplesPerEpoch -
            request.epoch_sample_offset) return mgt::Status::kInvalidConfig;
    const bool replay = trainer->info.execution_mode == SingleGpuExecutionMode::kFixedBatchGraph &&
                        request.active_rows == trainer->info.capacity_rows;
    const auto status = replay
        ? detail::LaunchSingleGpuTrainGraph(&trainer->graph, request, trainer->stream)
        : EnqueueSingleGpuTrainStep(trainer, request);
    if (status != mgt::Status::kOk) return FailTrainer(trainer, status);
    if (cudaEventRecord(trainer->completion, trainer->stream) != cudaSuccess)
        return FailTrainer(trainer);
    trainer->sequence = request.optimizer_step;
    trainer->in_flight = true;
    ticket->completion_event = trainer->completion;
    ticket->sequence = trainer->sequence;
    return mgt::Status::kOk;
}

mgt::Status ReadSingleGpuMetrics(
    SingleGpuTrainer* trainer, SingleGpuTrainerMetrics* metrics) {
    if (!trainer || !metrics) return mgt::Status::kInvalidConfig;
    if (trainer->failed) return mgt::Status::kCudaFailure;
    if (!trainer->prepared) return mgt::Status::kInvalidConfig;
    if (trainer->in_flight) {
        if (cudaEventSynchronize(trainer->completion) != cudaSuccess) return FailTrainer(trainer);
        trainer->completed = trainer->sequence;
        trainer->in_flight = false;
    }
    if (trainer->completed && cudaMemcpyAsync(
            &trainer->last_loss, At<float>(trainer->arena, trainer->layout.loss),
            sizeof(float), cudaMemcpyDeviceToHost, trainer->stream) != cudaSuccess)
        return FailTrainer(trainer);
    if (trainer->completed && cudaStreamSynchronize(trainer->stream) != cudaSuccess)
        return FailTrainer(trainer);
    *metrics = {trainer->completed, trainer->completed, trainer->last_loss};
    return mgt::Status::kOk;
}

mgt::Status DestroySingleGpuTrainer(SingleGpuTrainer** pointer) {
    if (!pointer || !*pointer) return mgt::Status::kInvalidConfig;
    auto* trainer = *pointer;
    mgt::Status result = mgt::Status::kOk;
    if (trainer->stream && cudaStreamSynchronize(trainer->stream) != cudaSuccess)
        result = mgt::Status::kCudaFailure;
    if (detail::DestroySingleGpuTrainGraph(&trainer->graph) != mgt::Status::kOk)
        result = mgt::Status::kCudaFailure;
    if (trainer->blas && cublasDestroy(trainer->blas) != CUBLAS_STATUS_SUCCESS)
        result = mgt::Status::kCudaFailure;
    if (trainer->arena && cudaFree(trainer->arena) != cudaSuccess) result = mgt::Status::kCudaFailure;
    if (trainer->completion && cudaEventDestroy(trainer->completion) != cudaSuccess)
        result = mgt::Status::kCudaFailure;
    if (trainer->stream && cudaStreamDestroy(trainer->stream) != cudaSuccess)
        result = mgt::Status::kCudaFailure;
    delete trainer;
    *pointer = nullptr;
    return result;
}

std::uint64_t SingleGpuTrainerAllocationCountForTest() {
    return g_allocation_count.load();
}

}  // namespace mgt_cuda
