#pragma once
#include "mgt/a100_bf16_policy.hpp"
#include "mgt/status.hpp"
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>
namespace mgt_cuda {
mgt::Status LaunchBf16BatchNormForwardEpilogue(
    const float* preactivation,const float* mean,const float* inv_std,
    const float* gamma,const float* beta,const __nv_bfloat16* residual,
    std::uint32_t active_rows,std::uint32_t capacity_rows,
    std::uint32_t logical_features,std::uint32_t physical_features,
    mgt::A100XhatStorage xhat_storage,void* saved_xhat,
    __nv_bfloat16* activation,std::uint32_t* relu_mask,cudaStream_t stream);
mgt::Status LaunchBf16BatchNormBackwardEpilogue(
    const float* upstream,const std::uint32_t* relu_mask,const void* saved_xhat,
    const float* inv_std,const float* gamma,const float* global_dgamma,
    const float* global_dbeta,std::uint32_t active_rows,std::uint32_t capacity_rows,
    std::uint32_t global_rows,std::uint32_t logical_features,
    std::uint32_t physical_features,mgt::A100XhatStorage xhat_storage,
    __nv_bfloat16* dz,float* residual_grad,cudaStream_t stream);
} // namespace mgt_cuda
