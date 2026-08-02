#include "mgt_cuda/a100_bf16_runtime.cuh"
#include "mgt_cuda/allreduce_nccl.cuh"
#include "mgt_cuda/bf16_linear_train_ops.cuh"
#include "mgt_cuda/bf16_batch_norm_sites.cuh"
#include "mgt_cuda/sync_batch_norm_tiled.cuh"

#include <cublasLt.h>

#include <array>
#include <cstring>
#include <new>
#include <vector>
#include <string>

namespace mgt_cuda {
struct A100Bf16Runtime{
 mgt::P888A100ExecutionProfileV1 profile{};mgt::A100StaticArenaPlanV1 plan{};A100StaticArenaView view{};
 std::array<cudaStream_t,5> streams{};std::array<cudaEvent_t,16> events{};std::array<cublasLtHandle_t,2> lt{};std::array<NcclRankContext*,3> contexts{};P888StepControlV1* control=nullptr;
 mgt::Bf16AlgorithmTable algorithms{};Bf16LinearTrainOpsPlan* linear_plan=nullptr;MlpBatchNormBf16WorkspacePlan activation_plan{};Bf16BatchNormSitesPlan bn_sites{};TiledSyncBatchNormConfig bn_config{};TiledSyncBatchNormWorkspace bn_workspace{};
};
namespace {
bool SameHash(const std::array<std::uint8_t,32>&a,const std::array<std::uint8_t,32>&b){return a==b;}
bool Distinct(const char*a,const char*b,const char*c){return a&&b&&c&&*a&&*b&&*c&&std::strcmp(a,b)&&std::strcmp(a,c)&&std::strcmp(b,c);}
const mgt::A100ArenaSliceV1* Find(const mgt::A100StaticArenaPlanV1&p,mgt::A100ArenaSliceKind k){for(std::uint32_t i=0;i<p.slice_count;++i)if(p.slices[i].kind==k)return&p.slices[i];return nullptr;}
std::string Hex(const std::array<std::uint8_t,32>& value){static constexpr char digits[]="0123456789abcdef";std::string out(64,'0');for(std::size_t i=0;i<value.size();++i){out[2*i]=digits[value[i]>>4];out[2*i+1]=digits[value[i]&15];}return out;}
mgt::Status PrepareActivation(A100Bf16Runtime*r,const A100Bf16RuntimeCreateInfo&i){
 const CudaMlpShape shape{i.arena_info.state_len,i.arena_info.state_value_pad,i.arena_info.hd1,i.arena_info.hd2,i.arena_info.residual_blocks,i.arena_info.output_dim};
 auto status=BuildMlpBatchNormBf16WorkspacePlan(shape,i.arena_info.capacity_rows,i.execution_profile->policy.dz_ring_slots,i.execution_profile->policy.xhat_storage,&r->activation_plan);if(status!=mgt::Status::kOk)return status;
 const auto*acts=Find(r->plan,mgt::A100ArenaSliceKind::kActivationsBf16);const auto*xhat=Find(r->plan,mgt::A100ArenaSliceKind::kXhat);const auto*masks=Find(r->plan,mgt::A100ArenaSliceKind::kReluMasks);const auto*dz=Find(r->plan,mgt::A100ArenaSliceKind::kDzRingBf16);const auto*scratch=Find(r->plan,mgt::A100ArenaSliceKind::kScratchFp32);
 const std::uint64_t xhat_bytes=r->activation_plan.saved_xhat_count*(i.execution_profile->policy.xhat_storage==mgt::A100XhatStorage::kFp32?4ULL:2ULL);const std::uint64_t scratch_bytes=static_cast<std::uint64_t>(i.arena_info.capacity_rows)*std::max(i.arena_info.hd1,i.arena_info.hd2)*8ULL;
 if(!acts||acts->bytes!=r->activation_plan.saved_activation_bf16_count*2ULL||!xhat||xhat->bytes!=xhat_bytes||!masks||masks->bytes!=r->activation_plan.relu_mask_bytes||!dz||dz->bytes!=r->activation_plan.dz_ring_bf16_count*2ULL||!scratch||scratch->bytes!=scratch_bytes)return mgt::Status::kInvalidConfig;
 r->bn_config={i.execution_profile->policy.bn_row_chunk,i.execution_profile->policy.bn_feature_tile,i.execution_profile->policy.xhat_storage};const auto partials=TiledSyncBatchNormPartialFloats(i.arena_info.capacity_rows,std::max(i.arena_info.hd1,i.arena_info.hd2),r->bn_config);const auto reduced=2ULL*std::max(i.arena_info.hd1,i.arena_info.hd2);if(!partials||(partials+reduced)*sizeof(float)>scratch->bytes)return mgt::Status::kInvalidConfig;r->bn_workspace={nullptr,partials,nullptr,reduced};return mgt::Status::kOk;
}
mgt::Status PrepareLinear(A100Bf16Runtime*r,const A100Bf16RuntimeCreateInfo&i){
 if(i.arena_info.hd1!=2560||i.arena_info.hd2!=224||i.arena_info.residual_blocks!=16)return mgt::Status::kInvalidConfig;
 auto status=BuildP888A100Sm80Cuda124Bf16AlgorithmTable(&r->algorithms);if(status!=mgt::Status::kOk)return status;
 std::string hash;if(mgt::CanonicalBf16AlgorithmTableSha256(r->algorithms,&hash)!=mgt::Status::kOk||hash!=Hex(i.execution_profile->policy.algorithm_table_sha256))return mgt::Status::kInvalidConfig;
 std::vector<Bf16LinearProblem> problems;problems.reserve(i.execution_profile->active_rows.size()*33U);
 for(auto rows:i.execution_profile->active_rows){problems.push_back({rows,2,rows,2560,224});for(std::uint32_t site=3;site<=34;++site)problems.push_back({rows,site,rows,224,224});}
 return CreateBf16LinearTrainOpsPlan(problems.data(),static_cast<std::uint32_t>(problems.size()),i.device_id,r->lt[0],r->algorithms,r->view,&r->linear_plan);
}
mgt::Status Validate(const A100Bf16RuntimeCreateInfo&i,const mgt::A100StaticArenaPlanV1&p){
 if(!i.execution_profile||i.arena_info.profile!=i.execution_profile||!i.world||i.rank>=i.world||i.world!=i.execution_profile->world||mgt::ValidateP888A100ExecutionProfile(*i.execution_profile,i.profile_use)!=mgt::Status::kOk||mgt::ValidateA100StaticArenaPlan(p)!=mgt::Status::kOk||!SameHash(p.layout_sha256,i.execution_profile->arena_layout_sha256)||p.ordinary_bytes!=i.execution_profile->ordinary_arena_bytes||p.symmetric_bytes!=i.execution_profile->symmetric_arena_bytes||p.pinned_host_bytes!=i.execution_profile->pinned_host_bytes)return mgt::Status::kInvalidConfig;
 if(i.world>1&&!Distinct(i.bn_nccl_id_file,i.weight_nccl_id_file,i.metrics_nccl_id_file))return mgt::Status::kInvalidConfig;

 if(p.symmetric_bytes)return mgt::Status::kInvalidConfig;
 int current=-1;if(cudaGetDevice(&current)!=cudaSuccess)return mgt::Status::kCudaFailure;if(current!=(int)i.device_id)return mgt::Status::kInvalidConfig;return mgt::Status::kOk;
}
}
mgt::Status CreateA100Bf16Runtime(const A100Bf16RuntimeCreateInfo&i,const mgt::A100StaticArenaPlanV1&p,A100Bf16Runtime**out){
 if(!out)return mgt::Status::kInvalidConfig;*out=nullptr;auto status=Validate(i,p);if(status!=mgt::Status::kOk)return status;auto*r=new(std::nothrow)A100Bf16Runtime;if(!r)return mgt::Status::kCapacityExceeded;r->profile=*i.execution_profile;r->plan=p;if(PrepareActivation(r,i)!=mgt::Status::kOk){delete r;return mgt::Status::kInvalidConfig;}r->view.plan=&r->plan;r->view.ordinary_bytes=p.ordinary_bytes;r->view.pinned_host_bytes=p.pinned_host_bytes;
 if(cudaMalloc(&r->view.ordinary_base,p.ordinary_bytes)!=cudaSuccess||cudaHostAlloc(&r->view.pinned_host_base,p.pinned_host_bytes,cudaHostAllocPortable)!=cudaSuccess){DestroyA100Bf16Runtime(r);return mgt::Status::kCudaFailure;}
 const auto*bn_scratch=Find(r->plan,mgt::A100ArenaSliceKind::kScratchFp32);if(!bn_scratch){DestroyA100Bf16Runtime(r);return mgt::Status::kInvalidConfig;}auto*bn_base=static_cast<std::uint8_t*>(r->view.ordinary_base)+bn_scratch->offset;r->bn_workspace.partials=reinterpret_cast<float*>(bn_base);r->bn_workspace.reduced=r->bn_workspace.partials+r->bn_workspace.partial_count;
 for(auto&s:r->streams)if(cudaStreamCreateWithFlags(&s,cudaStreamNonBlocking)!=cudaSuccess){DestroyA100Bf16Runtime(r);return mgt::Status::kCudaFailure;}
 for(auto&e:r->events)if(cudaEventCreateWithFlags(&e,cudaEventDisableTiming)!=cudaSuccess){DestroyA100Bf16Runtime(r);return mgt::Status::kCudaFailure;}
 for(auto&h:r->lt)if(cublasLtCreate(&h)!=CUBLAS_STATUS_SUCCESS){DestroyA100Bf16Runtime(r);return mgt::Status::kCudaFailure;}
 if(BuildBf16BatchNormSitesPlan({i.arena_info.state_len,i.arena_info.state_value_pad,i.arena_info.hd1,i.arena_info.hd2,i.arena_info.residual_blocks,i.arena_info.output_dim},i.arena_info.hd1,i.arena_info.hd2,i.arena_info.capacity_rows,i.execution_profile->policy.dz_ring_slots,i.execution_profile->policy.xhat_storage,&r->bn_sites)!=mgt::Status::kOk){DestroyA100Bf16Runtime(r);return mgt::Status::kInvalidConfig;}
 if(PrepareLinear(r,i)!=mgt::Status::kOk){DestroyA100Bf16Runtime(r);return mgt::Status::kInvalidConfig;}
#ifdef MGT_HAS_NCCL
 const char* ids[3]={i.bn_nccl_id_file,i.weight_nccl_id_file,i.metrics_nccl_id_file};
 for(int n=0;n<3;++n){const auto cs=i.world==1?CreateNcclSingleRankContext(i.device_id,&r->contexts[n]):CreateNcclRankContext(i.device_id,i.world,i.rank,std::filesystem::path(ids[n]),&r->contexts[n]);if(cs!=mgt::Status::kOk){DestroyA100Bf16Runtime(r);return cs;}}
#else
 if(i.world>1){DestroyA100Bf16Runtime(r);return mgt::Status::kInvalidConfig;}
#endif
 const auto*s=Find(p,mgt::A100ArenaSliceKind::kStepControl);if(!s||s->bytes<sizeof(P888StepControlV1)){DestroyA100Bf16Runtime(r);return mgt::Status::kInvalidConfig;}r->control=reinterpret_cast<P888StepControlV1*>(static_cast<std::uint8_t*>(r->view.ordinary_base)+s->offset);if(InitializeP888StepControl(r->control,0,0,888)!=mgt::Status::kOk){DestroyA100Bf16Runtime(r);return mgt::Status::kCudaFailure;}*out=r;return mgt::Status::kOk;
}
mgt::Status DestroyA100Bf16Runtime(A100Bf16Runtime*r){if(!r)return mgt::Status::kInvalidConfig;mgt::Status result=mgt::Status::kOk;for(auto s:r->streams)if(s&&cudaStreamSynchronize(s)!=cudaSuccess)result=mgt::Status::kCudaFailure;
if(r->linear_plan&&DestroyBf16LinearTrainOpsPlan(r->linear_plan)!=mgt::Status::kOk)result=mgt::Status::kCudaFailure;
#ifdef MGT_HAS_NCCL
 for(auto c:r->contexts)if(c&&DestroyNcclRankContext(c)!=mgt::Status::kOk)result=mgt::Status::kNcclFailure;
#endif
for(auto h:r->lt)if(h&&cublasLtDestroy(h)!=CUBLAS_STATUS_SUCCESS)result=mgt::Status::kCudaFailure;for(auto e:r->events)if(e&&cudaEventDestroy(e)!=cudaSuccess)result=mgt::Status::kCudaFailure;for(auto s:r->streams)if(s&&cudaStreamDestroy(s)!=cudaSuccess)result=mgt::Status::kCudaFailure;if(r->view.pinned_host_base&&cudaFreeHost(r->view.pinned_host_base)!=cudaSuccess)result=mgt::Status::kCudaFailure;if(r->view.ordinary_base&&cudaFree(r->view.ordinary_base)!=cudaSuccess)result=mgt::Status::kCudaFailure;delete r;return result;}
mgt::Status QueryA100Bf16RuntimeView(const A100Bf16Runtime*r,A100StaticArenaView*out){if(!r||!out)return mgt::Status::kInvalidConfig;*out=r->view;return mgt::Status::kOk;}
P888StepControlV1* A100Bf16RuntimeStepControl(A100Bf16Runtime*r){return r?r->control:nullptr;}
cudaStream_t A100Bf16RuntimeComputeStream(A100Bf16Runtime*r){return r?r->streams[0]:nullptr;}
}

bool mgt_cuda::A100Bf16RuntimeHasCommunicators(const A100Bf16Runtime*r){
#ifdef MGT_HAS_NCCL
 return r&&r->contexts[0]&&r->contexts[1]&&r->contexts[2];
#else
 (void)r;return false;
#endif
}
const mgt_cuda::Bf16LinearTrainOpsPlan* mgt_cuda::A100Bf16RuntimeLinearPlan(
    const A100Bf16Runtime* runtime) {
    return runtime ? runtime->linear_plan : nullptr;
}
const mgt_cuda::MlpBatchNormBf16WorkspacePlan* mgt_cuda::A100Bf16RuntimeActivationPlan(const A100Bf16Runtime* runtime) {
    return runtime ? &runtime->activation_plan : nullptr;
}
const mgt_cuda::Bf16BatchNormSitesPlan* mgt_cuda::A100Bf16RuntimeBatchNormSitesPlan(const A100Bf16Runtime* runtime){return runtime?&runtime->bn_sites:nullptr;}

const mgt_cuda::TiledSyncBatchNormConfig* mgt_cuda::A100Bf16RuntimeBatchNormConfig(const A100Bf16Runtime*r){return r?&r->bn_config:nullptr;}
const mgt_cuda::TiledSyncBatchNormWorkspace* mgt_cuda::A100Bf16RuntimeBatchNormWorkspace(const A100Bf16Runtime*r){return r?&r->bn_workspace:nullptr;}
mgt_cuda::NcclRankContext* mgt_cuda::A100Bf16RuntimeBatchNormContext(A100Bf16Runtime*r){return r?r->contexts[0]:nullptr;}
