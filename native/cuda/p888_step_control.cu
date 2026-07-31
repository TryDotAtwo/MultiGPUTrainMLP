#include "mgt_cuda/p888_step_control.cuh"

#include <limits>

namespace {
__global__ void Begin(mgt_cuda::P888StepControlV1* c,std::uint32_t slot){
 if(blockIdx.x||threadIdx.x)return;
 if(c->fatal_health)return;
 if(c->schema_version!=1){c->fatal_health=(std::uint32_t)mgt_cuda::P888StepFatal::kBadSchema;return;}
 if(c->inflight_sequence!=c->committed_sequence){c->fatal_health=(std::uint32_t)mgt_cuda::P888StepFatal::kInflightExists;return;}
 if(c->committed_sequence==~std::uint64_t{0}||c->committed_optimizer_step==~std::uint64_t{0}){c->fatal_health=(std::uint32_t)mgt_cuda::P888StepFatal::kCursorOverflow;return;}
 if(!c->active_rows||c->active_rows>c->global_rows||c->global_offset>c->global_rows-c->active_rows){c->fatal_health=(std::uint32_t)mgt_cuda::P888StepFatal::kBadShape;return;}
 c->graph_slot=slot;c->inflight_sequence=c->committed_sequence+1;c->inflight_optimizer_step=c->committed_optimizer_step+1;c->batch_slot=(std::uint32_t)(c->inflight_sequence&1ULL);
}
__global__ void Commit(mgt_cuda::P888StepControlV1* c){
 if(blockIdx.x||threadIdx.x||c->fatal_health)return;
 if(c->schema_version!=1){c->fatal_health=(std::uint32_t)mgt_cuda::P888StepFatal::kBadSchema;return;}
 if(c->inflight_sequence!=c->committed_sequence+1||c->inflight_optimizer_step!=c->committed_optimizer_step+1){c->fatal_health=(std::uint32_t)mgt_cuda::P888StepFatal::kCommitWithoutInflight;return;}
 c->committed_sequence=c->inflight_sequence;c->committed_optimizer_step=c->inflight_optimizer_step;
}
}
mgt::Status mgt_cuda::InitializeP888StepControl(P888StepControlV1* c,std::uint64_t seq,std::uint64_t step,std::uint64_t seed){if(!c||!seed)return mgt::Status::kInvalidConfig;P888StepControlV1 h{};h.schema_version=1;h.committed_sequence=seq;h.committed_optimizer_step=step;h.inflight_sequence=seq;h.inflight_optimizer_step=step;h.generation_seed=seed;return cudaMemcpy(c,&h,sizeof(h),cudaMemcpyHostToDevice)==cudaSuccess?mgt::Status::kOk:mgt::Status::kCudaFailure;}
mgt::Status mgt_cuda::LaunchBeginP888StepControl(P888StepControlV1* c,A100LocalGraphSlot slot,cudaStream_t s){if(!c||!s||(slot!=A100LocalGraphSlot::kFull&&slot!=A100LocalGraphSlot::kTail))return mgt::Status::kInvalidConfig;Begin<<<1,1,0,s>>>(c,(std::uint32_t)slot);return cudaPeekAtLastError()==cudaSuccess?mgt::Status::kOk:mgt::Status::kCudaFailure;}
mgt::Status mgt_cuda::LaunchCommitP888StepControl(P888StepControlV1* c,cudaStream_t s){if(!c||!s)return mgt::Status::kInvalidConfig;Commit<<<1,1,0,s>>>(c);return cudaPeekAtLastError()==cudaSuccess?mgt::Status::kOk:mgt::Status::kCudaFailure;}
