#include "mgt_cuda/bf16_linear_train_ops.cuh"
#include "mgt_cuda/a100_bf16_runtime.cuh"

#include <cublasLt.h>
#include <cuda_runtime.h>

#include <cstring>
#include <new>
#include <vector>

namespace {
std::uint32_t FloatBits(float value) {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

void SetCommon(const mgt_cuda::Bf16LinearProblem& problem,
               mgt::Bf16GemmRole role,
               float beta,
               mgt::Bf16GemmKeyV1* key) {
    key->schema_version = 1;
    key->role = role;
    key->site_id = problem.site_id;
    key->order_a = key->order_b = key->order_c = key->order_d = CUBLASLT_ORDER_ROW;
    key->alignment_a_bytes = key->alignment_b_bytes = 16;
    key->alignment_c_bytes = key->alignment_d_bytes = 16;
    key->batch_count = 1;
    key->active_rows = problem.active_rows;
    key->compute_rows = problem.compute_rows;
    key->a_type = key->b_type = CUDA_R_16BF;
    key->c_type = key->d_type = CUDA_R_32F;
    key->compute_type = CUBLAS_COMPUTE_32F;
    key->beta_bits = FloatBits(beta);
}
}

mgt::Status mgt_cuda::BuildBf16LinearGemmKey(const Bf16LinearProblem& problem,
                                              mgt::Bf16GemmRole role,
                                              float beta,
                                              mgt::Bf16GemmKeyV1* out) {
    if (out == nullptr || problem.active_rows == 0 || problem.compute_rows < problem.active_rows ||
        problem.input_features == 0 || problem.output_features == 0 ||
        (beta != 0.0f && beta != 1.0f)) return mgt::Status::kInvalidConfig;
    *out = {};
    SetCommon(problem, role, beta, out);
    switch (role) {
        case mgt::Bf16GemmRole::kInputForward:
            out->m = problem.compute_rows;
            out->n = problem.output_features;
            out->k = problem.input_features;
            out->op_a = CUBLAS_OP_N;
            out->op_b = CUBLAS_OP_N;
            out->lda = problem.input_features;
            out->ldb = problem.output_features;
            out->ldc = out->ldd = problem.output_features;
            break;
        case mgt::Bf16GemmRole::kHiddenForward:
        case mgt::Bf16GemmRole::kResidualForward:
            out->m = problem.compute_rows;
            out->n = problem.output_features;
            out->k = problem.input_features;
            out->op_a = CUBLAS_OP_N;
            out->op_b = CUBLAS_OP_T;
            out->lda = problem.input_features;
            out->ldb = problem.input_features;
            out->ldc = out->ldd = problem.output_features;
            break;
        case mgt::Bf16GemmRole::kGradWeight:
            out->m = problem.output_features;
            out->n = problem.input_features;
            out->k = problem.compute_rows;
            out->op_a = CUBLAS_OP_T;
            out->op_b = CUBLAS_OP_N;
            out->lda = problem.output_features;
            out->ldb = problem.input_features;
            out->ldc = out->ldd = problem.input_features;
            break;
        case mgt::Bf16GemmRole::kInputTableGrad:
            out->m = problem.input_features;
            out->n = problem.output_features;
            out->k = problem.compute_rows;
            out->op_a = CUBLAS_OP_T;
            out->op_b = CUBLAS_OP_N;
            out->lda = problem.input_features;
            out->ldb = problem.output_features;
            out->ldc = out->ldd = problem.output_features;
            break;
        case mgt::Bf16GemmRole::kGradInput:
            out->m = problem.compute_rows;
            out->n = problem.input_features;
            out->k = problem.output_features;
            out->op_a = CUBLAS_OP_N;
            out->op_b = CUBLAS_OP_N;
            out->lda = problem.output_features;
            out->ldb = problem.input_features;
            out->ldc = out->ldd = problem.input_features;
            break;
        default:
            *out = {};
            return mgt::Status::kInvalidConfig;
    }
    return mgt::Status::kOk;
}
mgt::Status mgt_cuda::BuildP888A100Sm80Cuda124Bf16AlgorithmTable(
    mgt::Bf16AlgorithmTable* out) {
    if (out == nullptr) return mgt::Status::kInvalidConfig;
    out->records.clear();
    out->records.reserve(3U * (33U * 3U + 9U * 2U));

    const auto choice = [](std::uint32_t tile, std::uint32_t stages,
                           std::uint32_t split_k, std::uint32_t reduction,
                           std::uint32_t swizzle, std::uint64_t workspace_bytes) {
        mgt::Bf16GemmChoiceV1 value{};
        value.backend = mgt::Bf16GemmBackend::kCublasLt;
        value.cublaslt_algo_id = 6;
        value.tile_id = tile;
        value.stages_id = stages;
        value.split_k = split_k;
        value.reduction_scheme = reduction;
        value.cta_swizzle = swizzle;
        value.workspace_bytes = workspace_bytes;
        value.workspace_alignment = 256;
        value.split_k_contract.split_count = split_k;
        value.split_k_contract.reduction_scheme = reduction;
        value.split_k_contract.scratch_alignment = 256;
        value.split_k_contract.slot_count = 1;
        return value;
    };
    const auto input_embedding_forward = choice(20, 11, 1, 0, 1, 0);
    const auto input_embedding_dw = choice(24, 15, 2, 1, 1, 200);
    const auto input_forward = choice(23, 15, 1, 0, 1, 0);
    const auto hidden_forward = choice(24, 9, 1, 0, 1, 0);
    const auto input_dw = choice(20, 15, 2, 1, 0, 160);
    const auto hidden_dw = choice(20, 15, 24, 4, 0, 4816896);
    const auto input_dx = choice(20, 11, 1, 0, 1, 0);
    const auto hidden_dx = choice(24, 9, 1, 0, 1, 0);

    const auto add = [&](const Bf16LinearProblem& problem, mgt::Bf16GemmRole role,
                         const mgt::Bf16GemmChoiceV1& selected) {
        mgt::Bf16AlgorithmRecordV1 record{};
        if (BuildBf16LinearGemmKey(problem, role, 0.0f, &record.key) != mgt::Status::kOk)
            return false;
        record.choice = selected;
        out->records.push_back(record);
        return true;
    };

    constexpr std::uint32_t rows[] = {12497, 12498, 12500};
    for (std::uint32_t active_rows : rows) {
        for (std::uint32_t tile = 0; tile < 9; ++tile) {
            const Bf16LinearProblem input{active_rows, 100U + tile, active_rows, 576, 2560};
            mgt::Bf16AlgorithmRecordV1 record{};
            if (BuildBf16LinearGemmKey(input, mgt::Bf16GemmRole::kInputForward,
                                       tile == 0 ? 0.0f : 1.0f, &record.key) != mgt::Status::kOk)
                return mgt::Status::kInvalidConfig;
            record.choice = input_embedding_forward;
            out->records.push_back(record);
            if (!add(input, mgt::Bf16GemmRole::kInputTableGrad, input_embedding_dw))
                return mgt::Status::kInvalidConfig;
        }
        const Bf16LinearProblem hidden{active_rows, 2, active_rows, 2560, 224};
        if (!add(hidden, mgt::Bf16GemmRole::kHiddenForward, input_forward) ||
            !add(hidden, mgt::Bf16GemmRole::kGradWeight, input_dw) ||
            !add(hidden, mgt::Bf16GemmRole::kGradInput, input_dx))
            return mgt::Status::kInvalidConfig;

        for (std::uint32_t site = 3; site <= 34; ++site) {
            const Bf16LinearProblem residual{active_rows, site, active_rows, 224, 224};
            if (!add(residual, mgt::Bf16GemmRole::kResidualForward, hidden_forward) ||
                !add(residual, mgt::Bf16GemmRole::kGradWeight, hidden_dw) ||
                !add(residual, mgt::Bf16GemmRole::kGradInput, hidden_dx))
                return mgt::Status::kInvalidConfig;
        }
    }
    return mgt::ValidateBf16AlgorithmTable(*out);
}
namespace mgt_cuda {
struct Bf16LinearTrainOpsPlan {
    struct Entry {
        Bf16LinearProblem problem{};
        FixedBf16GemmPlan* forward = nullptr;
        FixedBf16GemmPlan* grad_weight = nullptr;
        FixedBf16GemmPlan* grad_input_zero = nullptr;
    };
    std::uint32_t device_id = 0;
    std::vector<Entry> entries;
};

namespace {
bool SameProblem(const Bf16LinearProblem& a, const Bf16LinearProblem& b) {
    return a.active_rows == b.active_rows && a.site_id == b.site_id &&
           a.compute_rows == b.compute_rows && a.input_features == b.input_features &&
           a.output_features == b.output_features;
}

const mgt::A100ArenaSliceV1* FindLtWorkspace(const A100StaticArenaView& arena) {
    if (arena.plan == nullptr) return nullptr;
    for (std::uint32_t i = 0; i < arena.plan->slice_count; ++i)
        if (arena.plan->slices[i].kind == mgt::A100ArenaSliceKind::kLtWorkspace)
            return &arena.plan->slices[i];
    return nullptr;
}

mgt::Bf16GemmRole ForwardRole(std::uint32_t site_id) {
    if (site_id == 2) return mgt::Bf16GemmRole::kHiddenForward;
    return mgt::Bf16GemmRole::kResidualForward;
}

const Bf16LinearTrainOpsPlan::Entry* FindEntry(
    const Bf16LinearTrainOpsPlan* plan, const Bf16LinearProblem& problem) {
    if (plan == nullptr || problem.site_id < 2 || problem.site_id > 34) return nullptr;
    for (const auto& entry : plan->entries)
        if (SameProblem(entry.problem, problem)) return &entry;
    return nullptr;
}

mgt::Status PrepareOne(cublasLtHandle_t handle, const mgt::Bf16AlgorithmTable& algorithms,
                       const Bf16LinearProblem& problem, mgt::Bf16GemmRole role,
                       void* workspace, std::uint64_t workspace_bytes,
                       FixedBf16GemmPlan** out) {
    mgt::Bf16GemmKeyV1 key{};
    if (BuildBf16LinearGemmKey(problem, role, 0.0f, &key) != mgt::Status::kOk)
        return mgt::Status::kInvalidConfig;
    const mgt::Bf16GemmChoiceV1* choice = nullptr;
    if (mgt::LookupBf16GemmChoice(algorithms, key, &choice) != mgt::Status::kOk || choice == nullptr)
        return mgt::Status::kInvalidConfig;
    return CreateFixedBf16GemmPlan(handle, key, *choice, workspace, workspace_bytes, out);
}
}

mgt::Status CreateBf16LinearTrainOpsPlan(
    const Bf16LinearProblem* problems, std::uint32_t problem_count,
    std::uint32_t device_id, cublasLtHandle_t handle,
    const mgt::Bf16AlgorithmTable& algorithms, const A100StaticArenaView& arena,
    Bf16LinearTrainOpsPlan** out) {
    if (out == nullptr) return mgt::Status::kInvalidConfig;
    *out = nullptr;
    if (problems == nullptr || problem_count == 0 || handle == nullptr ||
        arena.ordinary_base == nullptr ||
        mgt::ValidateBf16AlgorithmTable(algorithms) != mgt::Status::kOk)
        return mgt::Status::kInvalidConfig;
    int current_device = -1;
    if (cudaGetDevice(&current_device) != cudaSuccess) return mgt::Status::kCudaFailure;
    if (current_device != static_cast<int>(device_id)) return mgt::Status::kInvalidConfig;
    const auto* workspace_slice = FindLtWorkspace(arena);
    if (workspace_slice == nullptr || workspace_slice->bytes == 0 ||
        workspace_slice->offset > arena.ordinary_bytes ||
        workspace_slice->bytes > arena.ordinary_bytes - workspace_slice->offset)
        return mgt::Status::kInvalidConfig;
    void* workspace = static_cast<std::uint8_t*>(arena.ordinary_base) + workspace_slice->offset;
    if (reinterpret_cast<std::uintptr_t>(workspace) % 256 != 0)
        return mgt::Status::kInvalidConfig;

    auto* plan = new (std::nothrow) Bf16LinearTrainOpsPlan;
    if (plan == nullptr) return mgt::Status::kCapacityExceeded;
    plan->device_id = device_id;
    plan->entries.reserve(problem_count);
    for (std::uint32_t i = 0; i < problem_count; ++i) {
        for (const auto& existing : plan->entries) {
            if (SameProblem(existing.problem, problems[i])) {
                DestroyBf16LinearTrainOpsPlan(plan);
                return mgt::Status::kInvalidConfig;
            }
        }
        Bf16LinearTrainOpsPlan::Entry entry{};
        entry.problem = problems[i];
        auto status = PrepareOne(handle, algorithms, entry.problem, ForwardRole(entry.problem.site_id),
                                 workspace, workspace_slice->bytes, &entry.forward);
        if (status == mgt::Status::kOk)
            status = PrepareOne(handle, algorithms, entry.problem, mgt::Bf16GemmRole::kGradWeight,
                                workspace, workspace_slice->bytes, &entry.grad_weight);
        const auto dx_role = mgt::Bf16GemmRole::kGradInput;
        if (status == mgt::Status::kOk)
            status = PrepareOne(handle, algorithms, entry.problem, dx_role,
                                workspace, workspace_slice->bytes, &entry.grad_input_zero);
        plan->entries.push_back(entry);
        if (status != mgt::Status::kOk) {
            DestroyBf16LinearTrainOpsPlan(plan);
            return status;
        }
    }
    *out = plan;
    return mgt::Status::kOk;
}

mgt::Status DestroyBf16LinearTrainOpsPlan(Bf16LinearTrainOpsPlan* plan) {
    if (plan == nullptr) return mgt::Status::kInvalidConfig;
    mgt::Status result = mgt::Status::kOk;
    for (auto& entry : plan->entries) {
        if (entry.grad_input_zero && DestroyFixedBf16GemmPlan(entry.grad_input_zero) != mgt::Status::kOk)
            result = mgt::Status::kCudaFailure;
        if (entry.grad_weight && DestroyFixedBf16GemmPlan(entry.grad_weight) != mgt::Status::kOk)
            result = mgt::Status::kCudaFailure;
        if (entry.forward && DestroyFixedBf16GemmPlan(entry.forward) != mgt::Status::kOk)
            result = mgt::Status::kCudaFailure;
    }
    delete plan;
    return result;
}

mgt::Status LaunchBf16LinearForwardToFloat(
    const Bf16LinearTrainOpsPlan* plan, Bf16LinearProblem problem,
    const __nv_bfloat16* input, const __nv_bfloat16* weight,
    float* output, cudaStream_t stream) {
    const auto* entry = FindEntry(plan, problem);
    return entry ? LaunchFixedBf16Gemm(entry->forward, input, weight, output, stream)
                 : mgt::Status::kInvalidConfig;
}

mgt::Status LaunchBf16LinearGradWeightToFloat(
    const Bf16LinearTrainOpsPlan* plan, Bf16LinearProblem problem,
    const __nv_bfloat16* input, const __nv_bfloat16* grad_output,
    float* grad_weight, cudaStream_t stream) {
    const auto* entry = FindEntry(plan, problem);
    return entry ? LaunchFixedBf16Gemm(entry->grad_weight, grad_output, input, grad_weight, stream)
                 : mgt::Status::kInvalidConfig;
}

mgt::Status LaunchBf16LinearGradInputToFloat(
    const Bf16LinearTrainOpsPlan* plan, Bf16LinearProblem problem,
    const __nv_bfloat16* grad_output, const __nv_bfloat16* weight,
    float* grad_input, float beta, cudaStream_t stream) {
    if (beta != 0.0f) return mgt::Status::kInvalidConfig;
    const auto* entry = FindEntry(plan, problem);
    return entry ? LaunchFixedBf16Gemm(entry->grad_input_zero, grad_output, weight, grad_input, stream)
                 : mgt::Status::kInvalidConfig;
}
}  // namespace mgt_cuda
