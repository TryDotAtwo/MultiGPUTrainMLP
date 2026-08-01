#include "mgt_cuda/bf16_linear_train_ops.cuh"

#include <cublasLt.h>

#include <cstdint>
#include <cstring>
#include <new>

namespace mgt_cuda {

struct FixedBf16GemmPlan {
    cublasLtHandle_t handle = nullptr;
    cublasLtMatmulDesc_t operation = nullptr;
    cublasLtMatrixLayout_t a_layout = nullptr;
    cublasLtMatrixLayout_t b_layout = nullptr;
    cublasLtMatrixLayout_t c_layout = nullptr;
    cublasLtMatmulAlgo_t algorithm{};
    void* workspace = nullptr;
    std::size_t workspace_bytes = 0;
    float beta = 0.0f;
};

namespace {
bool Aligned(const void* pointer, std::uint64_t alignment) {
    return pointer != nullptr && alignment != 0 &&
           reinterpret_cast<std::uintptr_t>(pointer) % alignment == 0;
}

void DestroyDescriptors(FixedBf16GemmPlan* plan) {
    if (plan->c_layout != nullptr) cublasLtMatrixLayoutDestroy(plan->c_layout);
    if (plan->b_layout != nullptr) cublasLtMatrixLayoutDestroy(plan->b_layout);
    if (plan->a_layout != nullptr) cublasLtMatrixLayoutDestroy(plan->a_layout);
    if (plan->operation != nullptr) cublasLtMatmulDescDestroy(plan->operation);
}

cublasStatus_t SetAlgorithmAttributes(cublasLtMatmulAlgo_t* algorithm,
                                      const mgt::Bf16GemmChoiceV1& choice) {
    cublasStatus_t status = cublasLtMatmulAlgoConfigSetAttribute(
        algorithm, CUBLASLT_ALGO_CONFIG_TILE_ID, &choice.tile_id, sizeof(choice.tile_id));
    if (status == CUBLAS_STATUS_SUCCESS) status = cublasLtMatmulAlgoConfigSetAttribute(
        algorithm, CUBLASLT_ALGO_CONFIG_STAGES_ID, &choice.stages_id, sizeof(choice.stages_id));
    if (status == CUBLAS_STATUS_SUCCESS) status = cublasLtMatmulAlgoConfigSetAttribute(
        algorithm, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &choice.split_k, sizeof(choice.split_k));
    if (status == CUBLAS_STATUS_SUCCESS) status = cublasLtMatmulAlgoConfigSetAttribute(
        algorithm, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, &choice.reduction_scheme,
        sizeof(choice.reduction_scheme));
    if (status == CUBLAS_STATUS_SUCCESS) status = cublasLtMatmulAlgoConfigSetAttribute(
        algorithm, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING, &choice.cta_swizzle,
        sizeof(choice.cta_swizzle));
    if (status == CUBLAS_STATUS_SUCCESS) status = cublasLtMatmulAlgoConfigSetAttribute(
        algorithm, CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION, &choice.custom_option,
        sizeof(choice.custom_option));
    return status;
}

bool ValidContract(const mgt::Bf16GemmKeyV1& key,
                   const mgt::Bf16GemmChoiceV1& choice,
                   const void* workspace,
                   std::uint64_t capacity) {
    return key.schema_version == 1 && key.batch_count == 1 && key.m != 0 && key.n != 0 && key.k != 0 &&
           key.a_type == CUDA_R_16BF && key.b_type == CUDA_R_16BF &&
           key.c_type == CUDA_R_32F && key.d_type == CUDA_R_32F &&
           key.compute_type == CUBLAS_COMPUTE_32F &&
           key.alignment_a_bytes >= 16 && key.alignment_b_bytes >= 16 &&
           key.alignment_c_bytes >= 16 && key.alignment_d_bytes >= 16 &&
           choice.backend == mgt::Bf16GemmBackend::kCublasLt && choice.cublaslt_algo_id >= 0 &&
           choice.workspace_alignment >= 256 &&
           choice.workspace_offset <= capacity && choice.workspace_bytes <= capacity - choice.workspace_offset &&
           (choice.workspace_bytes == 0 || Aligned(static_cast<const std::uint8_t*>(workspace) +
                                                    choice.workspace_offset,
                                                    choice.workspace_alignment));
}
}

mgt::Status CreateFixedBf16GemmPlan(cublasLtHandle_t handle,
                                    const mgt::Bf16GemmKeyV1& key,
                                    const mgt::Bf16GemmChoiceV1& choice,
                                    void* workspace_base,
                                    std::uint64_t workspace_capacity,
                                    FixedBf16GemmPlan** out) {
    if (out == nullptr) return mgt::Status::kInvalidConfig;
    *out = nullptr;
    if (handle == nullptr || !ValidContract(key, choice, workspace_base, workspace_capacity))
        return mgt::Status::kInvalidConfig;
    auto* plan = new (std::nothrow) FixedBf16GemmPlan;
    if (plan == nullptr) return mgt::Status::kCapacityExceeded;
    plan->handle = handle;
    plan->workspace = choice.workspace_bytes == 0 ? nullptr :
        static_cast<std::uint8_t*>(workspace_base) + choice.workspace_offset;
    plan->workspace_bytes = static_cast<std::size_t>(choice.workspace_bytes);
    std::memcpy(&plan->beta, &key.beta_bits, sizeof(plan->beta));

    const auto op_a = static_cast<cublasOperation_t>(key.op_a);
    const auto op_b = static_cast<cublasOperation_t>(key.op_b);
    const auto order = CUBLASLT_ORDER_ROW;
    const std::uint64_t a_rows = op_a == CUBLAS_OP_N ? key.m : key.k;
    const std::uint64_t a_cols = op_a == CUBLAS_OP_N ? key.k : key.m;
    const std::uint64_t b_rows = op_b == CUBLAS_OP_N ? key.k : key.n;
    const std::uint64_t b_cols = op_b == CUBLAS_OP_N ? key.n : key.k;
    cublasStatus_t status = cublasLtMatmulDescCreate(&plan->operation, CUBLAS_COMPUTE_32F, CUDA_R_32F);
    if (status == CUBLAS_STATUS_SUCCESS) status = cublasLtMatmulDescSetAttribute(
        plan->operation, CUBLASLT_MATMUL_DESC_TRANSA, &op_a, sizeof(op_a));
    if (status == CUBLAS_STATUS_SUCCESS) status = cublasLtMatmulDescSetAttribute(
        plan->operation, CUBLASLT_MATMUL_DESC_TRANSB, &op_b, sizeof(op_b));
    if (status == CUBLAS_STATUS_SUCCESS) status = cublasLtMatrixLayoutCreate(
        &plan->a_layout, CUDA_R_16BF, a_rows, a_cols, static_cast<std::int64_t>(key.lda));
    if (status == CUBLAS_STATUS_SUCCESS) status = cublasLtMatrixLayoutSetAttribute(
        plan->a_layout, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order));
    if (status == CUBLAS_STATUS_SUCCESS) status = cublasLtMatrixLayoutCreate(
        &plan->b_layout, CUDA_R_16BF, b_rows, b_cols, static_cast<std::int64_t>(key.ldb));
    if (status == CUBLAS_STATUS_SUCCESS) status = cublasLtMatrixLayoutSetAttribute(
        plan->b_layout, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order));
    if (status == CUBLAS_STATUS_SUCCESS) status = cublasLtMatrixLayoutCreate(
        &plan->c_layout, CUDA_R_32F, key.m, key.n, static_cast<std::int64_t>(key.ldc));
    if (status == CUBLAS_STATUS_SUCCESS) status = cublasLtMatrixLayoutSetAttribute(
        plan->c_layout, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order));
    if (status == CUBLAS_STATUS_SUCCESS) status = cublasLtMatmulAlgoInit(
        handle, CUBLAS_COMPUTE_32F, CUDA_R_32F, CUDA_R_16BF, CUDA_R_16BF, CUDA_R_32F,
        CUDA_R_32F, choice.cublaslt_algo_id, &plan->algorithm);
    if (status == CUBLAS_STATUS_SUCCESS) status = SetAlgorithmAttributes(&plan->algorithm, choice);
    cublasLtMatmulHeuristicResult_t checked{};
    if (status == CUBLAS_STATUS_SUCCESS) status = cublasLtMatmulAlgoCheck(
        handle, plan->operation, plan->a_layout, plan->b_layout, plan->c_layout, plan->c_layout,
        &plan->algorithm, &checked);
    if (status != CUBLAS_STATUS_SUCCESS || checked.workspaceSize > plan->workspace_bytes) {
        DestroyDescriptors(plan);
        delete plan;
        return mgt::Status::kInvalidConfig;
    }
    *out = plan;
    return mgt::Status::kOk;
}

mgt::Status LaunchFixedBf16Gemm(const FixedBf16GemmPlan* plan,
                                const __nv_bfloat16* a,
                                const __nv_bfloat16* b,
                                float* output,
                                cudaStream_t stream) {
    if (plan == nullptr || !Aligned(a, 16) || !Aligned(b, 16) || !Aligned(output, 16))
        return mgt::Status::kInvalidConfig;
    const float alpha = 1.0f;
    const cublasStatus_t status = cublasLtMatmul(
        plan->handle, plan->operation, &alpha, a, plan->a_layout, b, plan->b_layout,
        &plan->beta, output, plan->c_layout, output, plan->c_layout, &plan->algorithm,
        plan->workspace, plan->workspace_bytes, stream);
    return status == CUBLAS_STATUS_SUCCESS ? mgt::Status::kOk : mgt::Status::kCudaFailure;
}

mgt::Status DestroyFixedBf16GemmPlan(FixedBf16GemmPlan* plan) {
    if (plan == nullptr) return mgt::Status::kInvalidConfig;
    DestroyDescriptors(plan);
    delete plan;
    return mgt::Status::kOk;
}

}  // namespace mgt_cuda
