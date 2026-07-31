#pragma once

#include "mgt/status.hpp"

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace mgt {

enum class A100LinearBackend : std::uint32_t { kCublasLtBf16 = 1, kCutlassBf16 = 2 };
enum class A100InputForwardBackend : std::uint32_t { kGatherFromBf16Table = 1, kPositionTiledBf16Gemm = 2 };
enum class A100InputGradBackend : std::uint32_t { kPositionTiledBf16Gemm = 1 };
enum class A100BatchNormBackend : std::uint32_t { kStridedFp32Reference = 1, kRowTiledFp32StatsBf16Output = 2, kCutlassPartialMomentsBf16 = 3 };
enum class A100WeightReduceBackend : std::uint32_t { kSectionedReference = 1, kTailPlusInputTilesOverlap = 2 };
enum class A100GraphBackend : std::uint32_t { kDisabled = 1, kComputeAndBn = 2, kFullMultistream = 3 };
enum class A100XhatStorage : std::uint32_t { kFp32 = 1, kBf16 = 2 };
enum class A100DwDxSchedule : std::uint32_t { kSerial = 1, kConcurrentProtected = 2 };
enum class A100InputGradReduction : std::uint32_t { kOneContiguousSum = 1, kPositionTiledSums = 2 };
enum class A100InputTileMaterialization : std::uint32_t { kExplicitBf16OneHot = 1, kImplicitCutlassIterator = 2 };
enum class A100BnCollectiveBackend : std::uint32_t { kNcclHostAllReduce = 1, kNcclDeviceLsa = 2 };

struct A100Bf16Policy {
    std::uint32_t schema_version = 1;
    A100LinearBackend linear = A100LinearBackend::kCublasLtBf16;
    A100InputForwardBackend input_forward = A100InputForwardBackend::kGatherFromBf16Table;
    A100InputGradBackend input_grad = A100InputGradBackend::kPositionTiledBf16Gemm;
    A100BatchNormBackend batch_norm = A100BatchNormBackend::kStridedFp32Reference;
    A100WeightReduceBackend weight_reduce = A100WeightReduceBackend::kSectionedReference;
    A100GraphBackend graph = A100GraphBackend::kDisabled;
    A100XhatStorage xhat_storage = A100XhatStorage::kFp32;
    A100DwDxSchedule dw_dx_schedule = A100DwDxSchedule::kSerial;
    A100InputGradReduction input_grad_reduction = A100InputGradReduction::kOneContiguousSum;
    A100InputTileMaterialization input_tile_materialization = A100InputTileMaterialization::kExplicitBf16OneHot;
    A100BnCollectiveBackend bn_collective = A100BnCollectiveBackend::kNcclHostAllReduce;
    std::uint32_t input_positions_per_tile = 8;
    std::uint32_t bn_row_chunk = 256;
    std::uint32_t bn_feature_tile = 32;
    std::uint32_t dz_ring_slots = 2;
    std::uint32_t padded_rows_multiple = 1;
    std::uint64_t lt_workspace_bytes = 16ULL << 20;
    std::array<std::uint8_t, 32> algorithm_table_sha256{};
};

Status ValidateA100Bf16Policy(const A100Bf16Policy& policy);
Status CanonicalSerializeA100Bf16Policy(const A100Bf16Policy& policy, std::vector<std::uint8_t>* out);
Status CanonicalA100Bf16PolicySha256(const A100Bf16Policy& policy, std::string* out_hex);
Status CanonicalBytesSha256(const std::vector<std::uint8_t>& bytes, std::string* out_hex);

}  // namespace mgt
