#include "mgt/a100_static_arena.hpp"
#include "mgt_cuda/a100_bf16_runtime.cuh"

#include <cstdlib>

namespace {
mgt::P888A100ExecutionProfileV1 Profile(){mgt::P888A100ExecutionProfileV1 p{};p.world=1;p.active_rows={4};p.policy.algorithm_table_sha256.fill(1);p.ordinary_arena_bytes=256;p.pinned_host_bytes=64;p.source_sha256.fill(2);p.binary_sha256.fill(3);p.arena_layout_sha256.fill(4);return p;}
}
int main(){const char* local=std::getenv("SLURM_LOCALID");const int device=local?std::atoi(local):0;if(cudaSetDevice(device)!=cudaSuccess)return 10;
 auto profile=Profile();mgt::A100StaticArenaBuildInfo arena_info{&profile,72,16,32,32,1,24,4,4096,256,288};mgt::A100StaticArenaPlanV1 plan{};if(mgt::BuildA100StaticArenaPlan(arena_info,&plan)!=mgt::Status::kOk)return 1;profile.arena_layout_sha256=plan.layout_sha256;profile.ordinary_arena_bytes=plan.ordinary_bytes;profile.pinned_host_bytes=plan.pinned_host_bytes;
 mgt_cuda::A100Bf16RuntimeCreateInfo info{};info.arena_info=arena_info;info.arena_info.profile=&profile;info.device_id=device;info.rank=0;info.world=1;info.execution_profile=&profile;info.profile_use=mgt::A100ExecutionProfileUse::kCandidateForTuner;
 mgt_cuda::A100Bf16Runtime* runtime=nullptr;if(mgt_cuda::CreateA100Bf16Runtime(info,plan,&runtime)!=mgt::Status::kOk||!runtime)return 2;mgt_cuda::A100StaticArenaView view{};if(mgt_cuda::QueryA100Bf16RuntimeView(runtime,&view)!=mgt::Status::kOk||!view.ordinary_base||!view.pinned_host_base||view.ordinary_bytes!=plan.ordinary_bytes||view.pinned_host_bytes!=plan.pinned_host_bytes||view.symmetric_base)return 3;if(!mgt_cuda::A100Bf16RuntimeStepControl(runtime))return 4;if(mgt_cuda::DestroyA100Bf16Runtime(runtime)!=mgt::Status::kOk)return 5;return 0;
}
