#include "mgt/a100_static_arena.hpp"

namespace {
mgt::P888A100ExecutionProfileV1 P(){mgt::P888A100ExecutionProfileV1 p{};p.active_rows={12497,12498,12500};p.policy.algorithm_table_sha256.fill(1);p.ordinary_arena_bytes=256;p.pinned_host_bytes=64;p.source_sha256.fill(2);p.binary_sha256.fill(3);p.arena_layout_sha256.fill(4);return p;}
const mgt::A100ArenaSliceV1* Find(const mgt::A100StaticArenaPlanV1& p,mgt::A100ArenaSliceKind kind){for(std::uint32_t n=0;n<p.slice_count;++n)if(p.slices[n].kind==kind)return &p.slices[n];return nullptr;}
}

int main(){
 auto p=P();mgt::A100StaticArenaBuildInfo i{&p,72,16,512,1024,16,24,12500,10000000,100000,288};mgt::A100StaticArenaPlanV1 a{},b{};
 if(mgt::BuildA100StaticArenaPlan(i,&a)!=mgt::Status::kOk||mgt::BuildA100StaticArenaPlan(i,&b)!=mgt::Status::kOk||a.layout_sha256!=b.layout_sha256||a.slice_count<10||a.symmetric_bytes)return 1;
 const auto* acts=Find(a,mgt::A100ArenaSliceKind::kActivationsBf16);const auto* xhat=Find(a,mgt::A100ArenaSliceKind::kXhat);const auto* masks=Find(a,mgt::A100ArenaSliceKind::kReluMasks);const auto* dz=Find(a,mgt::A100ArenaSliceKind::kDzRingBf16);const auto* scratch=Find(a,mgt::A100ArenaSliceKind::kScratchFp32);
 const std::uint64_t activation_count=12500ULL*(512ULL+33ULL*1024ULL);const std::uint64_t mask_bytes=12500ULL*((512ULL+31)/32+33ULL*((1024ULL+31)/32))*4;
 if(!acts||acts->bytes!=activation_count*2||!xhat||xhat->bytes!=activation_count*4||xhat->dtype!=mgt::A100ArenaDtype::kFp32||!masks||masks->bytes!=mask_bytes||!dz||dz->bytes!=12500ULL*1024*2*2||!scratch||scratch->bytes!=(2ULL*12500*1024+(49ULL+1)*2*1024)*4)return 2;
 auto invalid=a;invalid.slices[1].offset=invalid.slices[0].offset;if(mgt::ValidateA100StaticArenaPlan(invalid)!=mgt::Status::kInvalidConfig)return 7;invalid=a;invalid.slices[0].bytes=invalid.ordinary_bytes+1;if(mgt::ValidateA100StaticArenaPlan(invalid)!=mgt::Status::kInvalidConfig)return 8;
 auto old=a.layout_sha256;p.policy.xhat_storage=mgt::A100XhatStorage::kBf16;p.policy.dz_ring_slots=3;if(mgt::BuildA100StaticArenaPlan(i,&a)!=mgt::Status::kOk||a.layout_sha256==old)return 3;
 xhat=Find(a,mgt::A100ArenaSliceKind::kXhat);if(!xhat||xhat->bytes!=activation_count*2||xhat->dtype!=mgt::A100ArenaDtype::kBf16)return 4;
 p.policy.bn_collective=mgt::A100BnCollectiveBackend::kNcclDeviceLsa;p.symmetric_arena_bytes=256;if(mgt::BuildA100StaticArenaPlan(i,&a)!=mgt::Status::kOk||!a.symmetric_bytes)return 5;
 i.parameter_count=~0ULL;if(mgt::BuildA100StaticArenaPlan(i,&a)!=mgt::Status::kCapacityExceeded)return 6;
 return 0;
}
