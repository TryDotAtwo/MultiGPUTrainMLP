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
