#pragma once
#include "mgt_cuda/a100_bf16_runtime.cuh"
#include <cuda_bf16.h>
#include <array>
#include <cstdint>
namespace mgt_cuda {
constexpr std::uint32_t kMaxBf16BatchNormSites=130;
struct Bf16BatchNormSiteLayout {std::uint32_t site_id=0,logical_features=0,physical_features=0;std::uint64_t activation_element_offset=0,xhat_byte_offset=0,mask_byte_offset=0,mask_bytes=0;};
struct Bf16BatchNormSitesPlan {std::uint32_t site_count=0,capacity_rows=0,dz_ring_slots=0,max_physical_features=0;std::uint64_t xhat_element_bytes=0;std::array<Bf16BatchNormSiteLayout,kMaxBf16BatchNormSites> sites{};};
struct Bf16BatchNormSiteView {const Bf16BatchNormSiteLayout* layout=nullptr;__nv_bfloat16* activation=nullptr;void* xhat=nullptr;std::uint32_t* relu_mask=nullptr;__nv_bfloat16* dz=nullptr;};
mgt::Status BuildBf16BatchNormSitesPlan(const CudaMlpShape& shape,std::uint32_t logical_hd1,std::uint32_t logical_hd2,std::uint32_t capacity_rows,std::uint32_t dz_ring_slots,mgt::A100XhatStorage xhat_storage,Bf16BatchNormSitesPlan* out);
mgt::Status QueryA100Bf16RuntimeBatchNormSite(A100Bf16Runtime* runtime,std::uint32_t site_id,std::uint32_t dz_slot,Bf16BatchNormSiteView* out);
const Bf16BatchNormSitesPlan* A100Bf16RuntimeBatchNormSitesPlan(const A100Bf16Runtime* runtime);
mgt::Status LaunchA100Bf16BatchNormForwardSite(A100Bf16Runtime* runtime,std::uint32_t site_id,const float* preactivation,const float* mean,const float* inv_std,const float* gamma,const float* beta,const __nv_bfloat16* residual,std::uint32_t active_rows);
mgt::Status LaunchA100Bf16BatchNormBackwardSite(A100Bf16Runtime* runtime,std::uint32_t site_id,std::uint32_t dz_slot,const float* upstream,const float* inv_std,const float* gamma,const float* global_dgamma,const float* global_dbeta,std::uint32_t active_rows,std::uint32_t global_rows,float* residual_grad);
} // namespace mgt_cuda
