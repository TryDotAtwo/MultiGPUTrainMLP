#include "mgt/a100_bf16_policy.hpp"

#include <iostream>

namespace {
mgt::A100Bf16Policy Valid(){mgt::A100Bf16Policy p{};p.algorithm_table_sha256.fill(0x5a);return p;}
bool Reject(const mgt::A100Bf16Policy& p){return mgt::ValidateA100Bf16Policy(p)!=mgt::Status::kOk;}
}
int main(){
 auto p=Valid();if(Reject(p))return 1;
 auto b=p;b.schema_version=2;if(!Reject(b))return 2;
 b=p;b.linear=static_cast<mgt::A100LinearBackend>(0);if(!Reject(b))return 3;
 b=p;b.input_positions_per_tile=7;if(!Reject(b))return 4;
 b=p;b.bn_row_chunk=128;if(!Reject(b))return 5;
 b=p;b.bn_feature_tile=16;if(!Reject(b))return 6;
 b=p;b.dz_ring_slots=1;if(!Reject(b))return 7;
 b=p;b.padded_rows_multiple=8;if(!Reject(b))return 8;
 b=p;b.lt_workspace_bytes=(16ULL<<20)+256;if(!Reject(b))return 9;
 b=p;b.algorithm_table_sha256.fill(0);if(!Reject(b))return 10;
 std::vector<std::uint8_t>a,c;std::string ha,hb;
 if(mgt::CanonicalSerializeA100Bf16Policy(p,&a)!=mgt::Status::kOk||mgt::CanonicalSerializeA100Bf16Policy(p,&c)!=mgt::Status::kOk||a!=c||mgt::CanonicalA100Bf16PolicySha256(p,&ha)!=mgt::Status::kOk||ha.size()!=64)return 11;
 b=p;b.xhat_storage=mgt::A100XhatStorage::kBf16;if(mgt::CanonicalA100Bf16PolicySha256(b,&hb)!=mgt::Status::kOk||ha==hb)return 12;
 b=p;b.algorithm_table_sha256[0]^=1;if(mgt::CanonicalA100Bf16PolicySha256(b,&hb)!=mgt::Status::kOk||ha==hb)return 13;
 std::cout<<"a100 bf16 policy ok\n";return 0;
}
