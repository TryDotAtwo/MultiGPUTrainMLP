#pragma once

#include "mgt/a100_bf16_algorithm_table.hpp"
#include "mgt/status.hpp"

#include <cublasLt.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace mgt_cuda {

struct Bf16LinearProblem {
    std::uint32_t active_rows;
    std::uint32_t site_id;
    std::uint32_t compute_rows;
    std::uint32_t input_features;
    std::uint32_t output_features;
};

mgt::Status BuildP888A100Sm80Cuda124Bf16AlgorithmTable(mgt::Bf16AlgorithmTable* out);

mgt::Status BuildBf16LinearGemmKey(const Bf16LinearProblem& problem,
                                    mgt::Bf16GemmRole role,
                                    float beta,
                                    mgt::Bf16GemmKeyV1* out);

struct A100StaticArenaView;
struct Bf16LinearTrainOpsPlan;

mgt::Status CreateBf16LinearTrainOpsPlan(
    const Bf16LinearProblem* problems,
    std::uint32_t problem_count,
    std::uint32_t device_id,
    cublasLtHandle_t handle,
    const mgt::Bf16AlgorithmTable& algorithms,
    const A100StaticArenaView& arena,
    Bf16LinearTrainOpsPlan** out);
mgt::Status CreateBf16LinearTrainOpsPlanInWorkspace(
    const Bf16LinearProblem* problems,
    std::uint32_t problem_count,
    std::uint32_t device_id,
    cublasLtHandle_t handle,
    const mgt::Bf16AlgorithmTable& algorithms,
    const A100StaticArenaView& arena,
    std::uint64_t workspace_offset,
    std::uint64_t workspace_bytes,
    Bf16LinearTrainOpsPlan** out);
mgt::Status DestroyBf16LinearTrainOpsPlan(Bf16LinearTrainOpsPlan* plan);
mgt::Status LaunchBf16LinearForwardToFloat(
    const Bf16LinearTrainOpsPlan* plan, Bf16LinearProblem problem,
    const __nv_bfloat16* input, const __nv_bfloat16* weight,
    float* output, cudaStream_t stream);
mgt::Status LaunchBf16LinearGradWeightToFloat(
    const Bf16LinearTrainOpsPlan* plan, Bf16LinearProblem problem,
    const __nv_bfloat16* input, const __nv_bfloat16* grad_output,
    float* grad_weight, cudaStream_t stream);
mgt::Status LaunchBf16LinearGradInputToFloat(
    const Bf16LinearTrainOpsPlan* plan, Bf16LinearProblem problem,
    const __nv_bfloat16* grad_output, const __nv_bfloat16* weight,
    float* grad_input, float beta, cudaStream_t stream);

struct FixedBf16GemmPlan;

mgt::Status CreateFixedBf16GemmPlan(cublasLtHandle_t handle,
                                         const mgt::Bf16GemmKeyV1& key,
                                         const mgt::Bf16GemmChoiceV1& choice,
                                         void* workspace_base,
                                         std::uint64_t workspace_capacity,
                                         FixedBf16GemmPlan** out);
mgt::Status LaunchFixedBf16Gemm(const FixedBf16GemmPlan* plan,
                                const __nv_bfloat16* a,
                                const __nv_bfloat16* b,
                                float* output,
                                cudaStream_t stream);
mgt::Status DestroyFixedBf16GemmPlan(FixedBf16GemmPlan* plan);

}  // namespace mgt_cuda
