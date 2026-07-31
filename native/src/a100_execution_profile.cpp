#include "mgt/a100_execution_profile.hpp"

#include <algorithm>

namespace {
bool Nz(const std::array<std::uint8_t,32>& x){return std::any_of(x.begin(),x.end(),[](auto b){return b!=0;});}
void U32(std::vector<std::uint8_t>& o,std::uint32_t v){for(int i=0;i<4;++i)o.push_back(static_cast<std::uint8_t>(v>>(8*i)));}
void U64(std::vector<std::uint8_t>& o,std::uint64_t v){for(int i=0;i<8;++i)o.push_back(static_cast<std::uint8_t>(v>>(8*i)));}
void Hash(std::vector<std::uint8_t>& o,const std::array<std::uint8_t,32>& h){o.insert(o.end(),h.begin(),h.end());}
}

mgt::Status mgt::ValidateP888A100ExecutionProfile(const P888A100ExecutionProfileV1& p,A100ExecutionProfileUse use){
 const bool candidate=use==A100ExecutionProfileUse::kCandidateForTuner&&p.profile_state==A100ExecutionProfileState::kCandidate;
 const bool accepted=use==A100ExecutionProfileUse::kAcceptedForProduction&&p.profile_state==A100ExecutionProfileState::kAccepted;
 if(p.schema_version!=1||p.canonical_serialization_version!=1||(!candidate&&!accepted)||
    !(p.world==1||p.world==2||p.world==4||p.world==8)||p.active_rows.empty()||
    !std::is_sorted(p.active_rows.begin(),p.active_rows.end())||
    std::adjacent_find(p.active_rows.begin(),p.active_rows.end())!=p.active_rows.end()||p.active_rows.front()==0||
    ValidateA100Bf16Policy(p.policy)!=Status::kOk||p.ordinary_arena_bytes==0||(p.ordinary_arena_bytes&255)||
    p.pinned_host_bytes==0||(p.pinned_host_bytes&63)||!Nz(p.source_sha256)||!Nz(p.binary_sha256)||!Nz(p.arena_layout_sha256))return Status::kInvalidConfig;
 if((p.policy.bn_collective==A100BnCollectiveBackend::kNcclHostAllReduce&&p.symmetric_arena_bytes!=0)||
    (p.policy.bn_collective==A100BnCollectiveBackend::kNcclDeviceLsa&&(p.symmetric_arena_bytes==0||(p.symmetric_arena_bytes&255))))return Status::kInvalidConfig;
 if(accepted&&(!Nz(p.gate_artifact_sha256)||!Nz(p.acceptance_report_sha256)))return Status::kInvalidConfig;
 if(candidate&&(Nz(p.gate_artifact_sha256)||Nz(p.acceptance_report_sha256)))return Status::kInvalidConfig;
 return Status::kOk;
}
mgt::Status mgt::CanonicalSerializeP888A100ExecutionProfile(const P888A100ExecutionProfileV1& p,A100ExecutionProfileUse use,std::vector<std::uint8_t>* out){
 if(!out||ValidateP888A100ExecutionProfile(p,use)!=Status::kOk)return Status::kInvalidConfig;std::vector<std::uint8_t> policy;if(CanonicalSerializeA100Bf16Policy(p.policy,&policy)!=Status::kOk)return Status::kInvalidConfig;
 out->clear();const std::uint8_t tag[]={'M','G','T','P','8','8','8','A'};out->insert(out->end(),std::begin(tag),std::end(tag));U32(*out,p.schema_version);U32(*out,p.canonical_serialization_version);U32(*out,(std::uint32_t)p.profile_state);U32(*out,p.world);U32(*out,(std::uint32_t)p.active_rows.size());for(auto n:p.active_rows)U32(*out,n);U32(*out,(std::uint32_t)policy.size());out->insert(out->end(),policy.begin(),policy.end());U64(*out,p.ordinary_arena_bytes);U64(*out,p.symmetric_arena_bytes);U64(*out,p.pinned_host_bytes);Hash(*out,p.source_sha256);Hash(*out,p.binary_sha256);Hash(*out,p.arena_layout_sha256);Hash(*out,p.gate_artifact_sha256);Hash(*out,p.acceptance_report_sha256);return Status::kOk;
}
mgt::Status mgt::CanonicalP888A100ExecutionProfileSha256(const P888A100ExecutionProfileV1& p,A100ExecutionProfileUse use,std::string* out){std::vector<std::uint8_t>b;if(CanonicalSerializeP888A100ExecutionProfile(p,use,&b)!=Status::kOk)return Status::kInvalidConfig;return CanonicalBytesSha256(b,out);}
