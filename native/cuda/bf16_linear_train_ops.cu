#include "mgt_cuda/bf16_linear_train_ops.cuh"

#include <cublasLt.h>
#include <cuda_runtime.h>

#include <cstring>

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
        case mgt::Bf16GemmRole::kGradInput:
        case mgt::Bf16GemmRole::kInputTableGrad:
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
    out->records.reserve(3U * 34U * 3U);

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
        const Bf16LinearProblem input{active_rows, 1, active_rows, 2560, 224};
        if (!add(input, mgt::Bf16GemmRole::kInputForward, input_forward) ||
            !add(input, mgt::Bf16GemmRole::kGradWeight, input_dw) ||
            !add(input, mgt::Bf16GemmRole::kInputTableGrad, input_dx))
            return mgt::Status::kInvalidConfig;

        const Bf16LinearProblem hidden{active_rows, 2, active_rows, 224, 224};
        if (!add(hidden, mgt::Bf16GemmRole::kHiddenForward, hidden_forward) ||
            !add(hidden, mgt::Bf16GemmRole::kGradWeight, hidden_dw) ||
            !add(hidden, mgt::Bf16GemmRole::kGradInput, hidden_dx))
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
