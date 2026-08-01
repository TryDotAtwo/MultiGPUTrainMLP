#pragma once
#include "mgt/a100_bf16_policy.hpp"
#include "mgt_cuda/mlp_forward.cuh"
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>
namespace mgt_cuda {
struct MlpBatchNormBf16WorkspacePlan {
 std::uint32_t capacity_rows=0;
 std::uint64_t saved_activation_bf16_offset=0,saved_activation_bf16_count=0;
 std::uint64_t saved_xhat_offset=0,saved_xhat_count=0;
 std::uint64_t relu_mask_offset_bytes=0,relu_mask_bytes=0;
 std::uint64_t dz_ring_bf16_offset=0,dz_ring_bf16_count=0;
 std::uint64_t preactivation_f32_offset=0,grad_input_f32_offset=0,total_bytes=0;
};
mgt::Status BuildMlpBatchNormBf16WorkspacePlan(const CudaMlpShape&,std::uint32_t,std::uint32_t,mgt::A100XhatStorage,MlpBatchNormBf16WorkspacePlan*);
mgt::Status LaunchPackReluActivationBf16(const float*,std::uint32_t,std::uint32_t,std::uint32_t,std::uint32_t,__nv_bfloat16*,std::uint32_t*,cudaStream_t);
mgt::Status LaunchGateGradientToBf16(const float*,const std::uint32_t*,std::uint32_t,std::uint32_t,std::uint32_t,std::uint32_t,__nv_bfloat16*,cudaStream_t);
}  // namespace mgt_cuda
