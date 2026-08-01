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
