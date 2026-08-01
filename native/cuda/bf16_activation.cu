#include "mgt_cuda/bf16_activation.cuh"
#include <algorithm>
#include <limits>
namespace mgt_cuda { namespace {
constexpr std::uint64_t A=256;
bool Add(std::uint64_t a,std::uint64_t b,std::uint64_t*o){if(a>~std::uint64_t{0}-b)return false;*o=a+b;return true;}
bool Mul(std::uint64_t a,std::uint64_t b,std::uint64_t*o){if(a&&b>~std::uint64_t{0}/a)return false;*o=a*b;return true;}
bool Align(std::uint64_t v,std::uint64_t*o){std::uint64_t t;if(!Add(v,A-1,&t))return false;*o=t&~(A-1);return true;}
bool Place(std::uint64_t bytes,std::uint64_t*c,std::uint64_t*o){return Align(*c,o)&&Add(*o,bytes,c);}
__global__ void Pack(const float*x,unsigned active,unsigned cap,unsigned logical,unsigned physical,__nv_bfloat16*y,unsigned*mask){unsigned r=blockIdx.x,w=blockIdx.y,c=w*32+threadIdx.x;bool stored=r<cap&&c<physical,valid=stored&&r<active&&c<logical;float v=valid?x[(std::uint64_t)r*physical+c]:0.0f;bool on=valid&&v>0.0f;if(stored)y[(std::uint64_t)r*physical+c]=__float2bfloat16_rn(on?v:0.0f);unsigned bits=__ballot_sync(0xffffffffU,on);if(threadIdx.x==0)mask[(std::uint64_t)r*((physical+31)/32)+w]=bits;}
__global__ void Gate(const float*x,const unsigned*mask,unsigned active,unsigned cap,unsigned logical,unsigned physical,__nv_bfloat16*y){std::uint64_t q=(std::uint64_t)blockIdx.x*blockDim.x+threadIdx.x,n=(std::uint64_t)cap*physical;if(q>=n)return;unsigned r=q/physical,c=q-(std::uint64_t)r*physical;bool on=r<active&&c<logical&&((mask[(std::uint64_t)r*((physical+31)/32)+c/32]>>(c&31))&1U);y[q]=__float2bfloat16_rn(on?x[q]:0.0f);}
} // namespace
mgt::Status BuildMlpBatchNormBf16WorkspacePlan(const CudaMlpShape&s,unsigned rows,unsigned slots,mgt::A100XhatStorage xs,MlpBatchNormBf16WorkspacePlan*out){
 if(!out||!rows||slots<2||slots>4||(xs!=mgt::A100XhatStorage::kFp32&&xs!=mgt::A100XhatStorage::kBf16))return mgt::Status::kInvalidConfig;*out={};
 // Preserve overflow classification even when an intentionally impossible shape also violates semantic limits.
 std::uint64_t sites,rf,per,count;if(!Mul(s.residual_blocks,2,&sites)||!Add(sites,1,&sites)||!Mul(sites,s.hd2,&rf)||!Add(s.hd1,rf,&per)||!Mul(rows,per,&count))return mgt::Status::kCapacityExceeded;
 if(ValidateCudaMlpShape(s)!=mgt::Status::kOk)return mgt::Status::kInvalidConfig;
 std::uint64_t mw2,mwpr,mw,mb;if(!Mul(sites,(s.hd2+31ULL)/32,&mw2)||!Add((s.hd1+31ULL)/32,mw2,&mwpr)||!Mul(rows,mwpr,&mw)||!Mul(mw,4,&mb))return mgt::Status::kCapacityExceeded;
 std::uint64_t width=std::max(s.hd1,s.hd2),scratch,dzc,ab,xb,db,sb;if(!Mul(rows,width,&scratch)||!Mul(scratch,slots,&dzc)||!Mul(count,2,&ab)||!Mul(count,xs==mgt::A100XhatStorage::kFp32?4:2,&xb)||!Mul(dzc,2,&db)||!Mul(scratch,4,&sb))return mgt::Status::kCapacityExceeded;
 std::uint64_t cur=0;out->capacity_rows=rows;out->saved_activation_bf16_count=count;out->saved_xhat_count=count;out->relu_mask_bytes=mb;out->dz_ring_bf16_count=dzc;
 if(!Place(ab,&cur,&out->saved_activation_bf16_offset)||!Place(xb,&cur,&out->saved_xhat_offset)||!Place(mb,&cur,&out->relu_mask_offset_bytes)||!Place(db,&cur,&out->dz_ring_bf16_offset)||!Place(sb,&cur,&out->preactivation_f32_offset)||!Place(sb,&cur,&out->grad_input_f32_offset)||!Align(cur,&out->total_bytes))return mgt::Status::kCapacityExceeded;return mgt::Status::kOk;
}
mgt::Status LaunchPackReluActivationBf16(const float*x,unsigned active,unsigned cap,unsigned logical,unsigned physical,__nv_bfloat16*y,unsigned*mask,cudaStream_t stream){if(!x||!y||!mask||!cap||!physical||active>cap||logical>physical)return mgt::Status::kInvalidConfig;Pack<<<dim3(cap,(physical+31)/32),32,0,stream>>>(x,active,cap,logical,physical,y,mask);return cudaPeekAtLastError()==cudaSuccess?mgt::Status::kOk:mgt::Status::kCudaFailure;}
mgt::Status LaunchGateGradientToBf16(const float*x,const unsigned*mask,unsigned active,unsigned cap,unsigned logical,unsigned physical,__nv_bfloat16*y,cudaStream_t stream){if(!x||!y||!mask||!cap||!physical||active>cap||logical>physical)return mgt::Status::kInvalidConfig;std::uint64_t n=(std::uint64_t)cap*physical,b=(n+255)/256;if(b>std::numeric_limits<unsigned>::max())return mgt::Status::kCapacityExceeded;Gate<<<(unsigned)b,256,0,stream>>>(x,mask,active,cap,logical,physical,y);return cudaPeekAtLastError()==cudaSuccess?mgt::Status::kOk:mgt::Status::kCudaFailure;}
} // namespace mgt_cuda
