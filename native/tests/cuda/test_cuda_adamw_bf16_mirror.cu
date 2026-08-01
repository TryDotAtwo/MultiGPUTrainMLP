#include "mgt_cuda/adamw.cuh"

#include <cuda_bf16.h>
#include <cmath>
#include <cstring>
#include <vector>

namespace {
std::uint16_t Bits(const __nv_bfloat16& value){std::uint16_t bits=0;std::memcpy(&bits,&value,sizeof(bits));return bits;}
}
int main(){
 constexpr std::uint64_t n=257;std::vector<float>w(n),g(n),m(n),v(n);for(std::uint64_t i=0;i<n;++i){w[i]=(static_cast<int>(i%31)-15)*0.0137f;g[i]=(static_cast<int>(i%19)-9)*0.0023f;}
 float *dw=nullptr,*dg=nullptr,*dm=nullptr,*dv=nullptr;__nv_bfloat16* mirror=nullptr;if(cudaMalloc(&dw,n*4)!=cudaSuccess||cudaMalloc(&dg,n*4)!=cudaSuccess||cudaMalloc(&dm,n*4)!=cudaSuccess||cudaMalloc(&dv,n*4)!=cudaSuccess||cudaMalloc(&mirror,n*sizeof(*mirror))!=cudaSuccess)return 1;if(cudaMemcpy(dw,w.data(),n*4,cudaMemcpyHostToDevice)!=cudaSuccess||cudaMemcpy(dg,g.data(),n*4,cudaMemcpyHostToDevice)!=cudaSuccess||cudaMemset(dm,0,n*4)!=cudaSuccess||cudaMemset(dv,0,n*4)!=cudaSuccess)return 2;
 if(mgt_cuda::LaunchFloatToBfloat16Mirror(dw,mirror,n,0)!=mgt::Status::kOk)return 3;std::vector<__nv_bfloat16> got(n);if(cudaMemcpy(got.data(),mirror,n*sizeof(*mirror),cudaMemcpyDeviceToHost)!=cudaSuccess)return 4;for(std::uint64_t i=0;i<n;++i)if(Bits(got[i])!=Bits(__nv_bfloat16(w[i])))return 5;
 const mgt_cuda::AdamWKernelConfig cfg{n,1,1e-3f,.9f,.999f,1e-8f,.01f};if(mgt_cuda::LaunchAdamWKernelWithBfloat16Mirror(cfg,dw,mirror,dg,dm,dv,0)!=mgt::Status::kOk||cudaDeviceSynchronize()!=cudaSuccess)return 6;std::vector<float> updated(n);if(cudaMemcpy(updated.data(),dw,n*4,cudaMemcpyDeviceToHost)!=cudaSuccess||cudaMemcpy(got.data(),mirror,n*sizeof(*mirror),cudaMemcpyDeviceToHost)!=cudaSuccess)return 7;for(std::uint64_t i=0;i<n;++i){if(!std::isfinite(updated[i])||Bits(got[i])!=Bits(__nv_bfloat16(updated[i])))return 8;}cudaFree(mirror);cudaFree(dv);cudaFree(dm);cudaFree(dg);cudaFree(dw);return 0;
}
