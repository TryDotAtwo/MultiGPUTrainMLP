#include "batch_norm_activation.cuh"
namespace mgt_cuda { namespace {
constexpr unsigned int T=256;
__global__ void ReluF(float*v,std::size_t n){std::size_t i=(std::size_t)blockIdx.x*blockDim.x+threadIdx.x;if(i<n)v[i]=v[i]>0?v[i]:0;}
__global__ void ReluFHalf(float*v,__half*h,std::size_t n){std::size_t i=(std::size_t)blockIdx.x*blockDim.x+threadIdx.x;if(i<n){float z=v[i]>0?v[i]:0;v[i]=z;h[i]=__float2half_rn(z);}}
__global__ void ReluB(const float*a,const float*g,float*d,std::size_t n){std::size_t i=(std::size_t)blockIdx.x*blockDim.x+threadIdx.x;if(i<n)d[i]=a[i]>0?g[i]:0;}
__global__ void ResF(const float*r,float*v,std::size_t n){std::size_t i=(std::size_t)blockIdx.x*blockDim.x+threadIdx.x;if(i<n){float z=r[i]+v[i];v[i]=z>0?z:0;}}
__global__ void ResFHalf(const float*r,float*v,__half*h,std::size_t n){std::size_t i=(std::size_t)blockIdx.x*blockDim.x+threadIdx.x;if(i<n){float z=r[i]+v[i];z=z>0?z:0;v[i]=z;h[i]=__float2half_rn(z);}}
__global__ void ResB(const float*a,const float*g,float*db,float*dr,std::size_t n){std::size_t i=(std::size_t)blockIdx.x*blockDim.x+threadIdx.x;if(i<n){float d=a[i]>0?g[i]:0;db[i]=d;dr[i]=d;}}
unsigned int blocks(std::size_t n){return static_cast<unsigned int>((n+T-1)/T);} mgt::Status done(){return cudaPeekAtLastError()==cudaSuccess?mgt::Status::kOk:mgt::Status::kCudaFailure;}
}
mgt::Status LaunchBatchNormReluForward(float*v,std::size_t n,cudaStream_t s){if(!v||!n)return mgt::Status::kInvalidConfig;ReluF<<<blocks(n),T,0,s>>>(v,n);return done();}
mgt::Status LaunchBatchNormReluForwardHalfMirror(float*v,__half*h,std::size_t n,cudaStream_t s){if(!v||!h||!n)return mgt::Status::kInvalidConfig;ReluFHalf<<<blocks(n),T,0,s>>>(v,h,n);return done();}
mgt::Status LaunchBatchNormReluBackward(const float*a,const float*g,float*d,std::size_t n,cudaStream_t s){if(!a||!g||!d||!n)return mgt::Status::kInvalidConfig;ReluB<<<blocks(n),T,0,s>>>(a,g,d,n);return done();}
mgt::Status LaunchBatchNormResidualReluForward(const float*r,float*v,std::size_t n,cudaStream_t s){if(!r||!v||!n)return mgt::Status::kInvalidConfig;ResF<<<blocks(n),T,0,s>>>(r,v,n);return done();}
mgt::Status LaunchBatchNormResidualReluForwardHalfMirror(const float*r,float*v,__half*h,std::size_t n,cudaStream_t s){if(!r||!v||!h||!n)return mgt::Status::kInvalidConfig;ResFHalf<<<blocks(n),T,0,s>>>(r,v,h,n);return done();}
mgt::Status LaunchBatchNormResidualReluBackward(const float*a,const float*g,float*db,float*dr,std::size_t n,cudaStream_t s){if(!a||!g||!db||!dr||db==dr||!n)return mgt::Status::kInvalidConfig;ResB<<<blocks(n),T,0,s>>>(a,g,db,dr,n);return done();}
}
