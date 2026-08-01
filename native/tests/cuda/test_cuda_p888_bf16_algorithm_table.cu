#include "mgt_cuda/bf16_linear_train_ops.cuh"

#include <cstdint>
#include <cstdio>

namespace {
bool Expect(const mgt::Bf16AlgorithmTable& table, const mgt_cuda::Bf16LinearProblem& problem,
            mgt::Bf16GemmRole role, std::uint32_t tile, std::uint32_t stages,
            std::uint32_t split_k, std::uint32_t reduction, std::uint32_t swizzle,
            std::uint64_t workspace) {
    mgt::Bf16GemmKeyV1 key{};
    if (mgt_cuda::BuildBf16LinearGemmKey(problem, role, 0.0f, &key) != mgt::Status::kOk)
        return false;
    const mgt::Bf16GemmChoiceV1* choice = nullptr;
    return mgt::LookupBf16GemmChoice(table, key, &choice) == mgt::Status::kOk &&
           choice != nullptr && choice->cublaslt_algo_id == 6 && choice->tile_id == tile &&
           choice->stages_id == stages && choice->split_k == split_k &&
           choice->reduction_scheme == reduction && choice->cta_swizzle == swizzle &&
           choice->workspace_bytes == workspace;
}

bool Reconstruct(const mgt::Bf16AlgorithmTable& table,
                 const mgt_cuda::Bf16LinearProblem& problem,
                 mgt::Bf16GemmRole role, cublasLtHandle_t handle,
                 void* workspace, std::uint64_t workspace_bytes) {
    mgt::Bf16GemmKeyV1 key{};
    if (mgt_cuda::BuildBf16LinearGemmKey(problem, role, 0.0f, &key) != mgt::Status::kOk)
        return false;
    const mgt::Bf16GemmChoiceV1* choice = nullptr;
    if (mgt::LookupBf16GemmChoice(table, key, &choice) != mgt::Status::kOk || choice == nullptr)
        return false;
    mgt_cuda::FixedBf16GemmPlan* plan = nullptr;
    if (mgt_cuda::CreateFixedBf16GemmPlan(handle, key, *choice, workspace,
                                           workspace_bytes, &plan) != mgt::Status::kOk)
        return false;
    return mgt_cuda::DestroyFixedBf16GemmPlan(plan) == mgt::Status::kOk;
}
}

int main() {
    mgt::Bf16AlgorithmTable table;
    if (mgt_cuda::BuildP888A100Sm80Cuda124Bf16AlgorithmTable(&table) != mgt::Status::kOk ||
        table.records.size() != 306) return 1;
    for (std::uint32_t rows : {12497U, 12498U, 12500U}) {
        const mgt_cuda::Bf16LinearProblem input{rows, 1, rows, 2560, 224};
        if (!Expect(table, input, mgt::Bf16GemmRole::kInputForward, 23, 15, 1, 0, 1, 0) ||
            !Expect(table, input, mgt::Bf16GemmRole::kGradWeight, 20, 15, 2, 1, 0, 160) ||
            !Expect(table, input, mgt::Bf16GemmRole::kInputTableGrad, 20, 11, 1, 0, 1, 0))
            return 2;
        for (std::uint32_t site : {2U, 3U, 18U, 34U}) {
            const mgt_cuda::Bf16LinearProblem hidden{rows, site, rows, 224, 224};
            const auto forward_role = site == 2 ? mgt::Bf16GemmRole::kHiddenForward
                                                : mgt::Bf16GemmRole::kResidualForward;
            if (!Expect(table, hidden, forward_role, 24, 9, 1, 0, 1, 0) ||
                !Expect(table, hidden, mgt::Bf16GemmRole::kGradWeight, 20, 15, 24, 4, 0, 4816896) ||
                !Expect(table, hidden, mgt::Bf16GemmRole::kGradInput, 24, 9, 1, 0, 1, 0))
                return 3;
        }
    }
    cublasLtHandle_t handle = nullptr;
    void* workspace = nullptr;
    constexpr std::uint64_t workspace_bytes = 256ULL << 20;
    if (cublasLtCreate(&handle) != CUBLAS_STATUS_SUCCESS ||
        cudaMalloc(&workspace, workspace_bytes) != cudaSuccess) return 4;
    for (std::uint32_t rows : {12497U, 12498U, 12500U}) {
        const mgt_cuda::Bf16LinearProblem input{rows, 1, rows, 2560, 224};
        const mgt_cuda::Bf16LinearProblem hidden{rows, 2, rows, 224, 224};
        if (!Reconstruct(table, input, mgt::Bf16GemmRole::kInputForward, handle, workspace, workspace_bytes) ||
            !Reconstruct(table, input, mgt::Bf16GemmRole::kGradWeight, handle, workspace, workspace_bytes) ||
            !Reconstruct(table, input, mgt::Bf16GemmRole::kInputTableGrad, handle, workspace, workspace_bytes) ||
            !Reconstruct(table, hidden, mgt::Bf16GemmRole::kHiddenForward, handle, workspace, workspace_bytes) ||
            !Reconstruct(table, hidden, mgt::Bf16GemmRole::kGradWeight, handle, workspace, workspace_bytes) ||
            !Reconstruct(table, hidden, mgt::Bf16GemmRole::kGradInput, handle, workspace, workspace_bytes))
            return 5;
    }
    cudaFree(workspace);
    cublasLtDestroy(handle);

    std::string hash;
    if (mgt::CanonicalBf16AlgorithmTableSha256(table, &hash) != mgt::Status::kOk) return 4;
    std::printf("p888_a100_bf16_records=%zu sha256=%s\n", table.records.size(), hash.c_str());
    return 0;
}