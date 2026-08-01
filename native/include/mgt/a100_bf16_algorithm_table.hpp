#pragma once

#include "mgt/status.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace mgt {

enum class Bf16GemmRole : std::uint32_t {
    kInputForward = 1,
    kHiddenForward = 2,
    kResidualForward = 3,
    kGradWeight = 4,
    kGradInput = 5,
    kInputTableGrad = 6,
};

enum class Bf16GemmBackend : std::uint32_t { kCublasLt = 1, kCutlass = 2 };

struct Bf16GemmKeyV1 {
    std::uint32_t schema_version = 1;
    Bf16GemmRole role = Bf16GemmRole::kHiddenForward;
    std::uint32_t site_id = 0;
    std::uint32_t order_a = 0, order_b = 0, order_c = 0, order_d = 0;
    std::uint32_t alignment_a_bytes = 0, alignment_b_bytes = 0;
    std::uint32_t alignment_c_bytes = 0, alignment_d_bytes = 0;
    std::uint32_t batch_count = 0;
    std::uint64_t stride_a_bytes = 0, stride_b_bytes = 0, stride_c_bytes = 0, stride_d_bytes = 0;
    std::uint32_t active_rows = 0, compute_rows = 0, m = 0, n = 0, k = 0;
    std::uint32_t op_a = 0, op_b = 0;
    std::uint64_t lda = 0, ldb = 0, ldc = 0, ldd = 0;
    std::uint32_t a_type = 0, b_type = 0, c_type = 0, d_type = 0;
    std::uint32_t compute_type = 0, epilogue = 0, beta_bits = 0;
};

struct Bf16SplitKContractV1 {
    std::uint32_t split_count = 1, partition_kind = 0, k_granularity = 0;
    std::uint32_t reduction_scheme = 0, finalize_kernel_version = 0, scratch_layout_version = 0;
    std::uint64_t scratch_offset = 0, scratch_bytes = 0, scratch_alignment = 256;
    std::uint32_t slot_count = 1;
};

struct Bf16GemmChoiceV1 {
    Bf16GemmBackend backend = Bf16GemmBackend::kCublasLt;
    std::int32_t cublaslt_algo_id = -1;
    std::uint32_t tile_id = 0, stages_id = 0, split_k = 0, reduction_scheme = 0;
    std::uint32_t cta_swizzle = 0, custom_option = 0, custom_kernel_version = 0;
    std::uint64_t workspace_offset = 0, workspace_bytes = 0, workspace_alignment = 256;
    Bf16SplitKContractV1 split_k_contract;
};

struct Bf16AlgorithmRecordV1 { Bf16GemmKeyV1 key; Bf16GemmChoiceV1 choice; };
struct Bf16AlgorithmTable { std::vector<Bf16AlgorithmRecordV1> records; };

Status ValidateBf16AlgorithmTable(const Bf16AlgorithmTable& table);
Status CanonicalSerializeBf16AlgorithmTable(const Bf16AlgorithmTable& table, std::vector<std::uint8_t>* out);
Status CanonicalBf16AlgorithmTableSha256(const Bf16AlgorithmTable& table, std::string* out_hex);
Status LookupBf16GemmChoice(const Bf16AlgorithmTable& table, const Bf16GemmKeyV1& key,
                            const Bf16GemmChoiceV1** out);

}  // namespace mgt
