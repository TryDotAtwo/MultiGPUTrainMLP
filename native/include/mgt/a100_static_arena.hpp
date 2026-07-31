#pragma once

#include "mgt/a100_execution_profile.hpp"

#include <array>
#include <cstdint>

namespace mgt {

enum class A100ArenaDomain : std::uint32_t { kOrdinaryDevice = 1, kSymmetricDevice = 2, kPinnedHost = 3 };
enum class A100ArenaDtype : std::uint32_t { kBytes = 1, kFp32 = 2, kBf16 = 3, kU32 = 4, kState = 5 };
enum class A100ArenaSliceKind : std::uint32_t {
    kStepControl=1,kWeightsFp32=2,kWeightGradFp32=3,kWeightMFp32=4,kWeightVFp32=5,
    kWeightsBf16=6,kAffineStateFp32=7,kBatchStates=8,kBatchLabels=9,kActivationsBf16=10,
    kDzRingBf16=11,kScratchFp32=12,kLtWorkspace=13,kFatalHealth=14,kTelemetryDevice=15,
    kBnSymmetric=16,kTelemetryHost=17
};

struct A100ArenaSliceV1 {
    A100ArenaSliceKind kind{};
    A100ArenaDomain domain{};
    A100ArenaDtype dtype{};
    std::uint64_t offset=0;
    std::uint64_t bytes=0;
    std::uint32_t alignment=0;
};

struct A100StaticArenaBuildInfo {
    const P888A100ExecutionProfileV1* profile=nullptr;
    std::uint32_t state_len=0,state_value_pad=0,hd1=0,hd2=0,residual_blocks=0,output_dim=0;
    std::uint32_t capacity_rows=0;
    std::uint64_t parameter_count=0;
    std::uint64_t affine_value_count=0;
    std::uint64_t state_storage_bytes=0;
};

struct A100StaticArenaPlanV1 {
    std::uint32_t schema_version=1;
    std::uint32_t slice_count=0;
    std::array<A100ArenaSliceV1,24> slices{};
    std::uint64_t ordinary_bytes=0,symmetric_bytes=0,pinned_host_bytes=0;
    std::array<std::uint8_t,32> layout_sha256{};
};

Status BuildA100StaticArenaPlan(const A100StaticArenaBuildInfo& info,A100StaticArenaPlanV1* out);
Status ValidateA100StaticArenaPlan(const A100StaticArenaPlanV1& plan);

}  // namespace mgt
