#define MGT_LOCAL_MLP_IMPLEMENTATION
#include "mlp_batch_norm_forward.cu"
#include "mgt_cuda/local_mlp_batch_norm.cuh"
#include <climits>
#include <limits>

namespace mgt_cuda::detail {
const void* FusedInputAdamTrainingKernel() {
    return reinterpret_cast<const void*>(
        SparseInputGradCompactActiveAdjacent2PackedHalfU16U32AdamW);
}
}  // namespace mgt_cuda::detail

namespace mgt_cuda {
namespace {

// Compare byte ranges, including separately allocated device buffers, without
// pointer subtraction or end-address overflow. Only the new mirror's writes
// are constrained: unrelated float/float lifetime reuse remains unchanged.
bool GradientMirrorDisjoint(const void* mirror, std::uint64_t mirror_bytes,
                            const void* other, std::uint64_t count,
                            std::uint64_t element_bytes) {
    if (!other || count == 0) return true;
    constexpr auto max_address = std::numeric_limits<std::uintptr_t>::max();
    if (count > max_address / element_bytes) return false;
    const auto other_bytes = count * element_bytes;
    const auto a = reinterpret_cast<std::uintptr_t>(mirror);
    const auto b = reinterpret_cast<std::uintptr_t>(other);
    if (mirror_bytes - 1 > max_address - a || other_bytes - 1 > max_address - b)
        return false;
    return a <= b ? b - a >= mirror_bytes : a - b >= other_bytes;
}

bool FitsSlice(std::uint64_t offset, std::uint64_t count, std::uint64_t capacity) {
    return offset <= capacity && count <= capacity - offset;
}

mgt::Status ValidateGradientMirrors(
    const CudaMlpShape& shape, std::uint32_t logical_hd1,
    std::uint32_t logical_hd2, std::uint32_t rows,
    const mgt::BatchNormTrainingPlan& plan, std::uint64_t workspace_capacity,
    const mgt::TrainStateStorage* states, const float* labels,
    MlpBatchNormStepBuffers b, const LocalMlpFp16Context& fp16,
    std::uint64_t tape_halfs) {
    const auto bh1 = static_cast<std::uint64_t>(rows) * shape.hd1;
    const auto bh2 = static_cast<std::uint64_t>(rows) * shape.hd2;
    const auto outputs = static_cast<std::uint64_t>(rows) * shape.output_dim;
    const auto mirror_halfs = std::max(bh2, outputs);
    if (!fp16.operand_b || mirror_halfs > fp16.operand_b_capacity)
        return mgt::Status::kCapacityExceeded;
    if (!logical_hd1 || !logical_hd2 || logical_hd1 > shape.hd1 ||
        logical_hd2 > shape.hd2 || plan.sites.size() != 2 + 2 * shape.residual_blocks)
        return mgt::Status::kInvalidConfig;
    // Active rows may be smaller than the plan's allocated row capacity. Check
    // actual views, not canonical packed offsets or the caller's larger suffix.
    for (std::size_t index = 0; index < plan.sites.size(); ++index) {
        const auto& q = plan.sites[index];
        const auto logical = index == 0 ? logical_hd1 : logical_hd2;
        const auto stride = index == 0 ? shape.hd1 : shape.hd2;
        if (q.logical_features != logical || q.physical_stride != stride ||
            !FitsSlice(q.affine_offset, logical, plan.logical_feature_count) ||
            !FitsSlice(q.running_offset, logical, plan.logical_feature_count) ||
            !FitsSlice(q.normalized_offset, static_cast<std::uint64_t>(rows) * stride,
                       plan.workspace_floats) ||
            !FitsSlice(q.mean_offset, logical, plan.workspace_floats) ||
            !FitsSlice(q.inv_std_offset, logical, plan.workspace_floats))
            return mgt::Status::kInvalidConfig;
    }
    if (!FitsSlice(plan.reduction_offset, 2ULL * std::max(logical_hd1, logical_hd2),
                   plan.workspace_floats)) return mgt::Status::kInvalidConfig;
    const auto activations = bh1 + (2ULL * shape.residual_blocks + 1) * bh2;
    if (plan.workspace_floats > std::numeric_limits<std::uint64_t>::max() - activations)
        return mgt::Status::kInvalidConfig;
    const auto workspace = activations + plan.workspace_floats;
    if (workspace > workspace_capacity) return mgt::Status::kInvalidConfig;
    const auto parameters = OB(shape) + shape.output_dim;
    const auto mirror_bytes = mirror_halfs * sizeof(__half);
    const auto separate = [&](const void* other, std::uint64_t count,
                              std::uint64_t bytes) {
        return GradientMirrorDisjoint(fp16.operand_b, mirror_bytes, other, count, bytes);
    };
    for (const float* tensor : {b.weights, b.weight_grad, b.weight_m, b.weight_v})
        if (!separate(tensor, parameters, sizeof(float))) return mgt::Status::kInvalidConfig;
    for (const float* tensor : {b.affine, b.affine_grad, b.affine_m, b.affine_v, b.running})
        if (!separate(tensor, plan.logical_feature_count, 2 * sizeof(float)))
            return mgt::Status::kInvalidConfig;
    for (const float* tensor : {static_cast<const float*>(b.outputs),
                               static_cast<const float*>(b.output_dy), labels})
        if (!separate(tensor, outputs, sizeof(float))) return mgt::Status::kInvalidConfig;
    if (!separate(fp16.weight_mirror, parameters, sizeof(__half)) ||
        !separate(fp16.operand_a, tape_halfs, sizeof(__half)) ||
        !separate(b.forward_workspace, workspace, sizeof(float)) ||
        !separate(b.loss, 1, sizeof(float)) ||
        !separate(b.block_grad, bh2, sizeof(float)) ||
        !separate(b.input_grad, bh1, sizeof(float)) ||
        !separate(states, rows, sizeof(mgt::TrainStateStorage)))
        return mgt::Status::kInvalidConfig;
    if (shape.residual_blocks &&
        (!separate(b.fc1_grad, bh2, sizeof(float)) ||
         !separate(b.residual_grad, bh2, sizeof(float))))
        return mgt::Status::kInvalidConfig;
    if (!fp16.input_gradient_half) {
        if (fp16.input_gradient_half_capacity != 0)
            return mgt::Status::kInvalidConfig;
        return mgt::Status::kOk;
    }
    if (fp16.input_gradient_half != fp16.operand_a)
        return mgt::Status::kInvalidConfig;
    if (bh1 > fp16.input_gradient_half_capacity ||
        bh1 > fp16.operand_a_capacity)
        return mgt::Status::kCapacityExceeded;
    const auto input_mirror_bytes = bh1 * sizeof(__half);
    const auto input_separate = [&](const void* other, std::uint64_t count,
                                    std::uint64_t bytes) {
        return GradientMirrorDisjoint(fp16.input_gradient_half,
                                      input_mirror_bytes, other, count, bytes);
    };
    for (const float* tensor : {b.weights, b.weight_grad, b.weight_m, b.weight_v})
        if (!input_separate(tensor, parameters, sizeof(float)))
            return mgt::Status::kInvalidConfig;
    for (const float* tensor : {b.affine, b.affine_grad, b.affine_m,
                                b.affine_v, b.running})
        if (!input_separate(tensor, plan.logical_feature_count,
                            2 * sizeof(float)))
            return mgt::Status::kInvalidConfig;
    for (const float* tensor : {static_cast<const float*>(b.outputs),
                               static_cast<const float*>(b.output_dy), labels})
        if (!input_separate(tensor, outputs, sizeof(float)))
            return mgt::Status::kInvalidConfig;
    if (!input_separate(fp16.weight_mirror, parameters, sizeof(__half)) ||
        !input_separate(fp16.operand_b, mirror_halfs, sizeof(__half)) ||
        !input_separate(fp16.input_active_bins, fp16.input_active_bin_count,
                        sizeof(std::uint16_t)) ||
        !input_separate(b.forward_workspace, workspace, sizeof(float)) ||
        !input_separate(b.loss, 1, sizeof(float)) ||
        !input_separate(b.block_grad, bh2, sizeof(float)) ||
        !input_separate(b.input_grad, bh1, sizeof(float)) ||
        !input_separate(states, rows, sizeof(mgt::TrainStateStorage)))
        return mgt::Status::kInvalidConfig;
    if (shape.residual_blocks &&
        (!input_separate(b.fc1_grad, bh2, sizeof(float)) ||
         !input_separate(b.residual_grad, bh2, sizeof(float))))
        return mgt::Status::kInvalidConfig;
    return mgt::Status::kOk;
}

}  // namespace

std::uint64_t MlpBatchNormParameterCount(const CudaMlpShape& shape) {
    return ValidateCudaMlpShape(shape) == mgt::Status::kOk
        ? OB(shape) + shape.output_dim : 0;
}

std::uint64_t LocalMlpBatchNormForwardWorkspaceFloats(
    const CudaMlpShape& shape,
    const mgt::BatchNormTrainingPlan& plan,
    std::uint32_t rows) {
    return LocalMlpBatchNormForwardWorkspaceFloatsImpl(shape, plan, rows);
}

mgt::Status LaunchLocalMlpInputHalf(
    const CudaMlpShape& shape, std::uint32_t logical_hd1,
    const __half* weights, const mgt::TrainStateStorage* states,
    std::uint32_t rows, float* output, cudaStream_t stream) {
    if (ValidateCudaMlpShape(shape) != mgt::Status::kOk)
        return mgt::Status::kInvalidConfig;
    return LaunchInputHalfInternal(
        shape, logical_hd1, weights, states, rows, output, stream);
}

mgt::Status LaunchLocalMlpBatchNormTrainStep(
    const CudaMlpShape& shape,
    std::uint32_t logical_hd1,
    std::uint32_t logical_hd2,
    const mgt::TrainStateStorage* states,
    const float* labels,
    std::uint32_t rows,
    const mgt::BatchNormTrainingPlan& plan,
    std::uint64_t forward_workspace_floats,
    const AdamWKernelConfig& adam,
    MlpBatchNormStepBuffers buffers,
    cublasHandle_t blas,
    cudaStream_t stream) {
    auto* local_backend = LocalBackendContext();
    return LaunchLocalMlpBatchNormTrainStepImpl(
        shape, logical_hd1, logical_hd2, states, labels, rows, rows, plan,
        forward_workspace_floats, adam, buffers, local_backend, blas, stream);
}

mgt::Status LaunchLocalMlpBatchNormTrainStepFp16(
    const CudaMlpShape& shape, std::uint32_t logical_hd1,
    std::uint32_t logical_hd2, const mgt::TrainStateStorage* states,
    const float* labels, std::uint32_t rows,
    const mgt::BatchNormTrainingPlan& plan,
    std::uint64_t forward_workspace_floats, const AdamWKernelConfig& adam,
    MlpBatchNormStepBuffers buffers, LocalMlpFp16Context* fp16,
    cublasHandle_t blas, cudaStream_t stream) {
    ClearGradientHalfCache(fp16);
    struct CacheLifetime {
        LocalMlpFp16Context* context;
        ~CacheLifetime() { ClearGradientHalfCache(context); }
    } cache_lifetime{fp16};
    if (!fp16 || fp16->master_weights != buffers.weights || !fp16->weight_mirror)
        return mgt::Status::kInvalidConfig;
    // CUDA BN indexing and cuBLAS dimensions use signed int. This also bounds
    // every row-product below before deriving byte ranges for the new mirror.
    if (ValidateCudaMlpShape(shape) != mgt::Status::kOk || rows == 0 ||
        rows > static_cast<std::uint32_t>(INT_MAX) /
                   std::max({shape.hd1, shape.hd2, shape.output_dim}))
        return mgt::Status::kInvalidConfig;
    const std::uint64_t activation_tape_halfs =
        static_cast<std::uint64_t>(rows) * shape.hd1 +
        (2ULL * shape.residual_blocks + (shape.output_dim > 1 ? 1ULL : 0ULL)) *
            rows * shape.hd2;
    if (!fp16->operand_a || fp16->operand_a_capacity < activation_tape_halfs)
        return mgt::Status::kCapacityExceeded;
    const auto scratch_status = ValidateGradientMirrors(shape, logical_hd1, logical_hd2,
        rows, plan, forward_workspace_floats, states, labels, buffers, *fp16,
        activation_tape_halfs);
    if (scratch_status != mgt::Status::kOk) return scratch_status;
    fp16->input_adam = adam;
    fp16->weight_grad = buffers.weight_grad;
    fp16->weight_m = buffers.weight_m;
    fp16->weight_v = buffers.weight_v;
    fp16->input_active_adam_fused = false;
    fp16->activation_workspace = buffers.forward_workspace;
    fp16->activation_rows = rows;
    fp16->activation_hd1 = shape.hd1;
    fp16->activation_hd2 = shape.hd2;
    fp16->activation_residual_blocks = shape.residual_blocks;
    fp16->activation_has_final = shape.output_dim > 1;
    auto* local_backend = LocalBackendContext(fp16);
    return LaunchLocalMlpBatchNormTrainStepImpl(
        shape, logical_hd1, logical_hd2, states, labels, rows, rows, plan,
        forward_workspace_floats, adam, buffers, local_backend, blas, stream);
}

}  // namespace mgt_cuda
