#include "mgt/a100_bf16_algorithm_table.hpp"

#include <algorithm>

namespace {
mgt::Bf16AlgorithmRecordV1 Record(std::uint32_t site) {
    mgt::Bf16AlgorithmRecordV1 record{};
    record.key.schema_version = 1;
    record.key.role = mgt::Bf16GemmRole::kHiddenForward;
    record.key.site_id = site;
    record.key.order_a = record.key.order_b = record.key.order_c = record.key.order_d = 1;
    record.key.alignment_a_bytes = record.key.alignment_b_bytes = 16;
    record.key.alignment_c_bytes = record.key.alignment_d_bytes = 16;
    record.key.batch_count = 1;
    record.key.active_rows = record.key.compute_rows = 12500;
    record.key.m = 12500;
    record.key.n = record.key.k = 224;
    record.key.lda = record.key.ldc = record.key.ldd = 224;
    record.key.ldb = 224;
    record.key.a_type = record.key.b_type = 14;
    record.key.c_type = record.key.d_type = 0;
    record.key.compute_type = 68;
    record.choice.backend = mgt::Bf16GemmBackend::kCublasLt;
    record.choice.cublaslt_algo_id = 23;
    record.choice.tile_id = 20;
    record.choice.stages_id = 18;
    record.choice.workspace_offset = 512;
    record.choice.workspace_bytes = 1U << 20;
    record.choice.workspace_alignment = 256;
    record.choice.split_k_contract.split_count = 1;
    record.choice.split_k_contract.k_granularity = 16;
    record.choice.split_k_contract.scratch_alignment = 256;
    record.choice.split_k_contract.slot_count = 1;
    return record;
}
}

int main() {
    mgt::Bf16AlgorithmTable a{{Record(2), Record(1)}};
    mgt::Bf16AlgorithmTable b{{Record(1), Record(2)}};
    std::string hash_a;
    std::string hash_b;
    if (mgt::CanonicalBf16AlgorithmTableSha256(a, &hash_a) != mgt::Status::kOk ||
        mgt::CanonicalBf16AlgorithmTableSha256(b, &hash_b) != mgt::Status::kOk || hash_a != hash_b) return 1;
    const mgt::Bf16GemmChoiceV1* choice = nullptr;
    if (mgt::LookupBf16GemmChoice(a, Record(1).key, &choice) != mgt::Status::kOk ||
        choice == nullptr || choice->cublaslt_algo_id != 23) return 2;
    auto missing = Record(3).key;
    if (mgt::LookupBf16GemmChoice(a, missing, &choice) == mgt::Status::kOk || choice != nullptr) return 3;
    mgt::Bf16AlgorithmTable duplicate{{Record(1), Record(1)}};
    if (mgt::ValidateBf16AlgorithmTable(duplicate) == mgt::Status::kOk) return 4;
    auto bad_alignment = a;
    bad_alignment.records[0].choice.workspace_alignment = 128;
    if (mgt::ValidateBf16AlgorithmTable(bad_alignment) == mgt::Status::kOk) return 5;
    auto changed = a;
    changed.records[0].choice.custom_option = 1;
    if (mgt::CanonicalBf16AlgorithmTableSha256(changed, &hash_b) != mgt::Status::kOk || hash_a == hash_b) return 6;
    auto bad_key = a;
    bad_key.records[0].key.batch_count = 0;
    if (mgt::ValidateBf16AlgorithmTable(bad_key) == mgt::Status::kOk) return 7;
    return 0;
}
