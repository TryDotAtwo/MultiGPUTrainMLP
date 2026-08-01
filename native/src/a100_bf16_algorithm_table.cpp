#include "mgt/a100_bf16_algorithm_table.hpp"

#include "mgt/a100_bf16_policy.hpp"

#include <algorithm>

namespace {
void U32(std::vector<std::uint8_t>& out, std::uint32_t value) {
    for (int i = 0; i != 4; ++i) out.push_back(static_cast<std::uint8_t>(value >> (8 * i)));
}
void U64(std::vector<std::uint8_t>& out, std::uint64_t value) {
    for (int i = 0; i != 8; ++i) out.push_back(static_cast<std::uint8_t>(value >> (8 * i)));
}
void KeyBytes(const mgt::Bf16GemmKeyV1& k, std::vector<std::uint8_t>& o) {
    U32(o,k.schema_version); U32(o,static_cast<std::uint32_t>(k.role)); U32(o,k.site_id);
    U32(o,k.order_a); U32(o,k.order_b); U32(o,k.order_c); U32(o,k.order_d);
    U32(o,k.alignment_a_bytes); U32(o,k.alignment_b_bytes); U32(o,k.alignment_c_bytes); U32(o,k.alignment_d_bytes);
    U32(o,k.batch_count); U64(o,k.stride_a_bytes); U64(o,k.stride_b_bytes); U64(o,k.stride_c_bytes); U64(o,k.stride_d_bytes);
    U32(o,k.active_rows); U32(o,k.compute_rows); U32(o,k.m); U32(o,k.n); U32(o,k.k); U32(o,k.op_a); U32(o,k.op_b);
    U64(o,k.lda); U64(o,k.ldb); U64(o,k.ldc); U64(o,k.ldd);
    U32(o,k.a_type); U32(o,k.b_type); U32(o,k.c_type); U32(o,k.d_type); U32(o,k.compute_type); U32(o,k.epilogue); U32(o,k.beta_bits);
}
std::vector<std::uint8_t> KeyBytes(const mgt::Bf16GemmKeyV1& key) { std::vector<std::uint8_t> out; KeyBytes(key,out); return out; }
bool Pow2(std::uint64_t value) { return value != 0 && (value & (value - 1)) == 0; }
bool ValidKey(const mgt::Bf16GemmKeyV1& k) {
    const auto role = static_cast<std::uint32_t>(k.role);
    return k.schema_version == 1 && role >= 1 && role <= 6 && k.site_id != 0 && k.batch_count != 0 &&
           k.active_rows != 0 && k.compute_rows >= k.active_rows && k.m != 0 && k.n != 0 && k.k != 0 &&
           k.lda != 0 && k.ldb != 0 && k.ldc != 0 && k.ldd != 0 && k.a_type != k.c_type &&
           k.b_type != k.d_type && k.alignment_a_bytes >= 16 && k.alignment_b_bytes >= 16 &&
           k.alignment_c_bytes >= 16 && k.alignment_d_bytes >= 16;
}
bool ValidChoice(const mgt::Bf16GemmChoiceV1& c) {
    const bool aligned = Pow2(c.workspace_alignment) && c.workspace_alignment >= 256 &&
                         c.workspace_offset % c.workspace_alignment == 0;
    const bool split = c.split_k_contract.split_count >= 1 && c.split_k_contract.slot_count >= 1 &&
                       Pow2(c.split_k_contract.scratch_alignment) && c.split_k_contract.scratch_alignment >= 256 &&
                       c.split_k_contract.scratch_offset % c.split_k_contract.scratch_alignment == 0;
    const bool backend = (c.backend == mgt::Bf16GemmBackend::kCublasLt && c.cublaslt_algo_id >= 0 && c.custom_kernel_version == 0) ||
                         (c.backend == mgt::Bf16GemmBackend::kCutlass && c.cublaslt_algo_id == -1 && c.custom_kernel_version != 0);
    return aligned && split && backend;
}
void ChoiceBytes(const mgt::Bf16GemmChoiceV1& c, std::vector<std::uint8_t>& o) {
    U32(o,static_cast<std::uint32_t>(c.backend)); U32(o,static_cast<std::uint32_t>(c.cublaslt_algo_id));
    U32(o,c.tile_id); U32(o,c.stages_id); U32(o,c.split_k); U32(o,c.reduction_scheme); U32(o,c.cta_swizzle); U32(o,c.custom_option); U32(o,c.custom_kernel_version);
    U64(o,c.workspace_offset); U64(o,c.workspace_bytes); U64(o,c.workspace_alignment);
    const auto& s=c.split_k_contract; U32(o,s.split_count); U32(o,s.partition_kind); U32(o,s.k_granularity); U32(o,s.reduction_scheme); U32(o,s.finalize_kernel_version); U32(o,s.scratch_layout_version); U64(o,s.scratch_offset); U64(o,s.scratch_bytes); U64(o,s.scratch_alignment); U32(o,s.slot_count);
}
}

mgt::Status mgt::ValidateBf16AlgorithmTable(const Bf16AlgorithmTable& table) {
    if (table.records.empty()) return Status::kInvalidConfig;
    std::vector<std::vector<std::uint8_t>> keys;
    keys.reserve(table.records.size());
    for (const auto& record : table.records) {
        if (!ValidKey(record.key) || !ValidChoice(record.choice)) return Status::kInvalidConfig;
        keys.push_back(KeyBytes(record.key));
    }
    std::sort(keys.begin(), keys.end());
    return std::adjacent_find(keys.begin(), keys.end()) == keys.end() ? Status::kOk : Status::kInvalidConfig;
}

mgt::Status mgt::CanonicalSerializeBf16AlgorithmTable(const Bf16AlgorithmTable& table, std::vector<std::uint8_t>* out) {
    if (out == nullptr || ValidateBf16AlgorithmTable(table) != Status::kOk) return Status::kInvalidConfig;
    std::vector<std::pair<std::vector<std::uint8_t>, const Bf16GemmChoiceV1*>> rows;
    for (const auto& record : table.records) rows.push_back({KeyBytes(record.key), &record.choice});
    std::sort(rows.begin(), rows.end(), [](const auto& a,const auto& b){ return a.first < b.first; });
    out->clear(); const std::uint8_t tag[]={'M','G','T','B','F','1','6','A'}; out->insert(out->end(),std::begin(tag),std::end(tag)); U32(*out,1); U32(*out,static_cast<std::uint32_t>(rows.size()));
    for (const auto& row : rows) { out->insert(out->end(),row.first.begin(),row.first.end()); ChoiceBytes(*row.second,*out); }
    return Status::kOk;
}

mgt::Status mgt::CanonicalBf16AlgorithmTableSha256(const Bf16AlgorithmTable& table, std::string* out_hex) {
    std::vector<std::uint8_t> bytes;
    if (CanonicalSerializeBf16AlgorithmTable(table,&bytes) != Status::kOk) return Status::kInvalidConfig;
    return CanonicalBytesSha256(bytes,out_hex);
}

mgt::Status mgt::LookupBf16GemmChoice(const Bf16AlgorithmTable& table, const Bf16GemmKeyV1& key, const Bf16GemmChoiceV1** out) {
    if (out == nullptr || ValidateBf16AlgorithmTable(table) != Status::kOk || !ValidKey(key)) return Status::kInvalidConfig;
    *out = nullptr; const auto wanted=KeyBytes(key);
    for (const auto& record : table.records) if (KeyBytes(record.key) == wanted) { *out=&record.choice; return Status::kOk; }
    return Status::kInvalidConfig;
}
