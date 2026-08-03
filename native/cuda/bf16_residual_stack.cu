#include "mgt_cuda/bf16_residual_stack.cuh"
#include "mgt_cuda/a100_bf16_runtime.cuh"
#include "mgt_cuda/bf16_batch_norm_sites.cuh"
#include "mgt_cuda/bf16_linear_train_ops.cuh"
#include <limits>

namespace mgt_cuda { namespace {
constexpr unsigned kResidualLayers=32;
__global__ void AddInPlace(float* destination,const float* source,std::uint64_t count){std::uint64_t q=static_cast<std::uint64_t>(blockIdx.x)*blockDim.x+threadIdx.x;if(q<count)destination[q]+=source[q];}
const __nv_bfloat16* BlockInput(A100Bf16Runtime*r,const Bf16ResidualStackBindings&b,unsigned block){if(block==0)return b.input_activation;Bf16BatchNormSiteView v{};return QueryA100Bf16RuntimeBatchNormSite(r,2*block+1,0,&v)==mgt::Status::kOk?v.activation:nullptr;}
}

mgt::Status ValidateBf16ResidualStackBindings(const Bf16ResidualStackBindings&b,bool backward){if(!b.input_activation||b.batch_norm_momentum<0.0f||b.batch_norm_momentum>1.0f||b.batch_norm_epsilon<=0.0f)return mgt::Status::kInvalidConfig;for(unsigned layer=0;layer<kResidualLayers;++layer){unsigned site=layer+2;if(!b.weights[layer]||!b.gamma[site]||!b.saved_inv_std[site]||(backward&&(!b.weight_grads[layer]||!b.dgamma[site]||!b.dbeta[site]))||(!backward&&(!b.beta[site]||!b.running_mean[site]||!b.running_variance[site]||!b.saved_mean[site])))return mgt::Status::kInvalidConfig;}return mgt::Status::kOk;}

mgt::Status LaunchA100Bf16ResidualStackForward(A100Bf16Runtime*r,const Bf16ResidualStackBindings&b,unsigned rows,unsigned global,const __nv_bfloat16**output){if(!r||!output||!rows||global<rows||ValidateBf16ResidualStackBindings(b,false)!=mgt::Status::kOk)return mgt::Status::kInvalidConfig;const auto*plan=A100Bf16RuntimeLinearPlan(r);float*pre=A100Bf16RuntimePreactivationScratch(r);auto stream=A100Bf16RuntimeComputeStream(r);if(!plan||!pre||!stream)return mgt::Status::kInvalidConfig;const __nv_bfloat16*current=b.input_activation;for(unsigned layer=0;layer<kResidualLayers;++layer){const unsigned bn_site=layer+2,linear_site=layer+3;Bf16LinearProblem problem{rows,linear_site,rows,224,224};auto z=LaunchBf16LinearForwardToFloat(plan,problem,current,b.weights[layer],pre,stream);if(z!=mgt::Status::kOk)return z;const __nv_bfloat16*residual=(layer&1U)?BlockInput(r,b,layer/2):nullptr;z=LaunchA100TiledSyncBatchNormForwardSite(r,bn_site,pre,residual,rows,global,b.gamma[bn_site],b.beta[bn_site],b.running_mean[bn_site],b.running_variance[bn_site],b.batch_norm_momentum,b.batch_norm_epsilon,b.saved_mean[bn_site],b.saved_inv_std[bn_site]);if(z!=mgt::Status::kOk)return z;Bf16BatchNormSiteView v{};z=QueryA100Bf16RuntimeBatchNormSite(r,bn_site,0,&v);if(z!=mgt::Status::kOk)return z;current=v.activation;}*output=current;return mgt::Status::kOk;}

mgt::Status LaunchA100Bf16ResidualStackBackward(A100Bf16Runtime*r,const Bf16ResidualStackBindings&b,const float*output_up,unsigned rows,unsigned global,float**input_grad){
 if(!r||!output_up||!input_grad||!rows||global<rows||ValidateBf16ResidualStackBindings(b,true)!=mgt::Status::kOk)return mgt::Status::kInvalidConfig;
 const auto*plan=A100Bf16RuntimeLinearPlan(r);const bool concurrent=A100Bf16RuntimeDwDxSchedule(r)==mgt::A100DwDxSchedule::kConcurrentProtected;const auto*weight_plan=concurrent?A100Bf16RuntimeWeightLinearPlan(r):plan;
 float*ping=A100Bf16RuntimePreactivationScratch(r),*pong=A100Bf16RuntimeGradInputScratch(r);auto stream=A100Bf16RuntimeComputeStream(r);auto weight_stream=concurrent?A100Bf16RuntimeWeightStream(r):stream;
 if(!plan||!weight_plan||!ping||!pong||!stream||!weight_stream)return mgt::Status::kInvalidConfig;
 cudaEvent_t ready[2]{},reuse[2]{};if(concurrent)for(unsigned slot=0;slot<2;++slot){ready[slot]=A100Bf16RuntimeDzReadyEvent(r,slot);reuse[slot]=A100Bf16RuntimeDzReuseEvent(r,slot);if(!ready[slot]||!reuse[slot])return mgt::Status::kInvalidConfig;}
 const float*upstream=output_up;const std::uint64_t elements=static_cast<std::uint64_t>(rows)*224;const std::uint64_t blocks=(elements+255)/256;if(blocks>std::numeric_limits<unsigned>::max())return mgt::Status::kCapacityExceeded;
 for(int block=15;block>=0;--block){
  unsigned fc1_site=2+2*block,fc2_site=fc1_site+1,fc1_layer=2*block,fc2_layer=fc1_layer+1;
  if(concurrent&&cudaStreamWaitEvent(stream,reuse[0],0)!=cudaSuccess)return mgt::Status::kCudaFailure;
  auto z=LaunchA100TiledSyncBatchNormBackwardSite(r,fc2_site,0,upstream,b.saved_inv_std[fc2_site],b.gamma[fc2_site],b.dgamma[fc2_site],b.dbeta[fc2_site],rows,global,ping);if(z!=mgt::Status::kOk)return z;
  Bf16BatchNormSiteView fc2{};z=QueryA100Bf16RuntimeBatchNormSite(r,fc2_site,0,&fc2);if(z!=mgt::Status::kOk)return z;Bf16BatchNormSiteView fc1{};z=QueryA100Bf16RuntimeBatchNormSite(r,fc1_site,1,&fc1);if(z!=mgt::Status::kOk)return z;
  Bf16LinearProblem p2{rows,fc2_layer+3,rows,224,224};
  if(concurrent){if(cudaEventRecord(ready[0],stream)!=cudaSuccess||cudaStreamWaitEvent(weight_stream,ready[0],0)!=cudaSuccess)return mgt::Status::kCudaFailure;}
  z=LaunchBf16LinearGradWeightToFloat(weight_plan,p2,fc1.activation,fc2.dz,b.weight_grads[fc2_layer],weight_stream);if(z!=mgt::Status::kOk)return z;
  if(concurrent&&cudaEventRecord(reuse[0],weight_stream)!=cudaSuccess)return mgt::Status::kCudaFailure;
  z=LaunchBf16LinearGradInputToFloat(plan,p2,fc2.dz,b.weights[fc2_layer],pong,0.0f,stream);if(z!=mgt::Status::kOk)return z;
  AddInPlace<<<static_cast<unsigned>(blocks),256,0,stream>>>(pong,ping,elements);if(cudaPeekAtLastError()!=cudaSuccess)return mgt::Status::kCudaFailure;
  if(concurrent&&cudaStreamWaitEvent(stream,reuse[1],0)!=cudaSuccess)return mgt::Status::kCudaFailure;
  z=LaunchA100TiledSyncBatchNormBackwardSite(r,fc1_site,1,pong,b.saved_inv_std[fc1_site],b.gamma[fc1_site],b.dgamma[fc1_site],b.dbeta[fc1_site],rows,global,nullptr);if(z!=mgt::Status::kOk)return z;
  const auto*block_input=BlockInput(r,b,block);if(!block_input)return mgt::Status::kInvalidConfig;Bf16LinearProblem p1{rows,fc1_layer+3,rows,224,224};
  if(concurrent){if(cudaEventRecord(ready[1],stream)!=cudaSuccess||cudaStreamWaitEvent(weight_stream,ready[1],0)!=cudaSuccess)return mgt::Status::kCudaFailure;}
  z=LaunchBf16LinearGradWeightToFloat(weight_plan,p1,block_input,fc1.dz,b.weight_grads[fc1_layer],weight_stream);if(z!=mgt::Status::kOk)return z;
  if(concurrent&&cudaEventRecord(reuse[1],weight_stream)!=cudaSuccess)return mgt::Status::kCudaFailure;
  z=LaunchBf16LinearGradInputToFloat(plan,p1,fc1.dz,b.weights[fc1_layer],ping,0.0f,stream);if(z!=mgt::Status::kOk)return z;upstream=ping;
 }
 if(concurrent)for(unsigned slot=0;slot<2;++slot)if(cudaStreamWaitEvent(stream,reuse[slot],0)!=cudaSuccess)return mgt::Status::kCudaFailure;
 *input_grad=ping;return mgt::Status::kOk;
}
}
