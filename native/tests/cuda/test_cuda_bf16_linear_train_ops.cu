#include "mgt_cuda/bf16_linear_train_ops.cuh"

#include <cublasLt.h>
#include <cuda_bf16.h>

#include <cstdint>
#include <cstring>

namespace {
std::uint32_t Bits(float value) {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

bool Common(const mgt::Bf16GemmKeyV1& key, std::uint32_t site, std::uint32_t rows) {
    return key.schema_version == 1 && key.site_id == site && key.active_rows == rows &&
           key.compute_rows == rows && key.batch_count == 1 && key.stride_a_bytes == 0 &&
           key.stride_b_bytes == 0 && key.stride_c_bytes == 0 && key.stride_d_bytes == 0 &&
           key.order_a == CUBLASLT_ORDER_ROW && key.order_b == CUBLASLT_ORDER_ROW &&
           key.order_c == CUBLASLT_ORDER_ROW && key.order_d == CUBLASLT_ORDER_ROW &&
           key.alignment_a_bytes == 16 && key.alignment_b_bytes == 16 &&
           key.alignment_c_bytes == 16 && key.alignment_d_bytes == 16 &&
           key.a_type == CUDA_R_16BF && key.b_type == CUDA_R_16BF &&
           key.c_type == CUDA_R_32F && key.d_type == CUDA_R_32F &&
           key.compute_type == CUBLAS_COMPUTE_32F;
}
}

int main() {
    const mgt_cuda::Bf16LinearProblem problem{12500, 7, 12500, 2560, 224};
    mgt::Bf16GemmKeyV1 forward{};
    if (mgt_cuda::BuildBf16LinearGemmKey(problem, mgt::Bf16GemmRole::kHiddenForward, 0.0f,
                                          &forward) != mgt::Status::kOk ||
        !Common(forward, 7, 12500) || forward.m != 12500 || forward.n != 224 ||
        forward.k != 2560 || forward.op_a != CUBLAS_OP_N || forward.op_b != CUBLAS_OP_T ||
        forward.lda != 2560 || forward.ldb != 2560 || forward.ldc != 224 || forward.ldd != 224 ||
        forward.beta_bits != Bits(0.0f)) return 1;

    mgt::Bf16GemmKeyV1 grad_weight{};
    if (mgt_cuda::BuildBf16LinearGemmKey(problem, mgt::Bf16GemmRole::kGradWeight, 0.0f,
                                          &grad_weight) != mgt::Status::kOk ||
        !Common(grad_weight, 7, 12500) || grad_weight.m != 224 || grad_weight.n != 2560 ||
        grad_weight.k != 12500 || grad_weight.op_a != CUBLAS_OP_T ||
        grad_weight.op_b != CUBLAS_OP_N || grad_weight.lda != 224 ||
        grad_weight.ldb != 2560 || grad_weight.ldc != 2560 || grad_weight.ldd != 2560) return 2;

    mgt::Bf16GemmKeyV1 grad_input{};
    if (mgt_cuda::BuildBf16LinearGemmKey(problem, mgt::Bf16GemmRole::kGradInput, 1.0f,
                                          &grad_input) != mgt::Status::kOk ||
        !Common(grad_input, 7, 12500) || grad_input.m != 12500 || grad_input.n != 2560 ||
        grad_input.k != 224 || grad_input.op_a != CUBLAS_OP_N || grad_input.op_b != CUBLAS_OP_N ||
        grad_input.lda != 224 || grad_input.ldb != 2560 || grad_input.ldc != 2560 ||
        grad_input.ldd != 2560 || grad_input.beta_bits != Bits(1.0f)) return 3;

    auto invalid = problem;
    invalid.compute_rows = invalid.active_rows - 1;
    if (mgt_cuda::BuildBf16LinearGemmKey(invalid, mgt::Bf16GemmRole::kHiddenForward, 0.0f,
                                          &forward) == mgt::Status::kOk) return 4;
    return 0;
}
