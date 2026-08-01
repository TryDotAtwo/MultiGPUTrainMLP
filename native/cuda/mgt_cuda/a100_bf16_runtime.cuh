#pragma once

#include "mgt/a100_static_arena.hpp"
#include "mgt_cuda/p888_step_control.cuh"

#include <cstdint>

namespace mgt_cuda {

struct A100Bf16RuntimeCreateInfo {
    mgt::A100StaticArenaBuildInfo arena_info{};
    std::uint32_t device_id=0,rank=0,world=0;
    const char* bn_nccl_id_file=nullptr;
    const char* weight_nccl_id_file=nullptr;
    const char* metrics_nccl_id_file=nullptr;
    const mgt::P888A100ExecutionProfileV1* execution_profile=nullptr;
    mgt::A100ExecutionProfileUse profile_use=mgt::A100ExecutionProfileUse::kCandidateForTuner;
};

struct A100StaticArenaView {
    void* ordinary_base=nullptr;
    std::uint64_t ordinary_bytes=0;
    void* symmetric_base=nullptr;
    std::uint64_t symmetric_bytes=0;
    void* pinned_host_base=nullptr;
    std::uint64_t pinned_host_bytes=0;
    const mgt::A100StaticArenaPlanV1* plan=nullptr;
};

struct A100Bf16Runtime;
mgt::Status CreateA100Bf16Runtime(const A100Bf16RuntimeCreateInfo& info,
    const mgt::A100StaticArenaPlanV1& arena,A100Bf16Runtime** out);
mgt::Status DestroyA100Bf16Runtime(A100Bf16Runtime* runtime);
mgt::Status QueryA100Bf16RuntimeView(const A100Bf16Runtime* runtime,A100StaticArenaView* out);
P888StepControlV1* A100Bf16RuntimeStepControl(A100Bf16Runtime* runtime);

}  // namespace mgt_cuda
