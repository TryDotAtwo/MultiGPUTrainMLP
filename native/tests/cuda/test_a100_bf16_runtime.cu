#include "mgt/a100_static_arena.hpp"
#include "mgt_cuda/a100_bf16_runtime.cuh"
#include "mgt_cuda/bf16_linear_train_ops.cuh"
#include "mgt_cuda/bf16_batch_norm_sites.cuh"

#include <cstdlib>
#include <string>

namespace {
bool ParseHex(const std::string& value, std::array<std::uint8_t, 32>* out) {
    if (value.size() != 64 || out == nullptr) return false;
    const auto nibble = [](char c) -> int {
        if (c >= '0' && c <= '9') return c - '0';
        if (c >= 'a' && c <= 'f') return c - 'a' + 10;
        return -1;
    };
    for (std::size_t i = 0; i < out->size(); ++i) {
        const int hi = nibble(value[2 * i]);
        const int lo = nibble(value[2 * i + 1]);
        if (hi < 0 || lo < 0) return false;
        (*out)[i] = static_cast<std::uint8_t>((hi << 4) | lo);
    }
    return true;
}

mgt::P888A100ExecutionProfileV1 Profile(std::uint32_t world) {
    mgt::P888A100ExecutionProfileV1 profile{};
    profile.world = world;
    profile.active_rows = {12497, 12498, 12500};
    mgt::Bf16AlgorithmTable table;
    std::string hash;
    if (mgt_cuda::BuildP888A100Sm80Cuda124Bf16AlgorithmTable(&table) != mgt::Status::kOk ||
        mgt::CanonicalBf16AlgorithmTableSha256(table, &hash) != mgt::Status::kOk ||
        !ParseHex(hash, &profile.policy.algorithm_table_sha256))
        profile.policy.algorithm_table_sha256.fill(0);
    profile.ordinary_arena_bytes = 256;
    profile.pinned_host_bytes = 64;
    profile.source_sha256.fill(2);
    profile.binary_sha256.fill(3);
    profile.arena_layout_sha256.fill(4);
    return profile;
}
}

int main() {
    const char* local = std::getenv("SLURM_LOCALID");
    const int device = local ? std::atoi(local) : 0;
    const char* world_env = std::getenv("SLURM_NTASKS");
    const char* rank_env = std::getenv("SLURM_PROCID");
    const bool multirank = local && world_env && rank_env && std::getenv("MGT_BN_ID") &&
                           std::getenv("MGT_WEIGHT_ID") && std::getenv("MGT_METRICS_ID");
    const std::uint32_t world = multirank ? static_cast<std::uint32_t>(std::atoi(world_env)) : 1;
    const std::uint32_t rank = multirank ? static_cast<std::uint32_t>(std::atoi(rank_env)) : 0;
    if (cudaSetDevice(device) != cudaSuccess) return 10;

    auto profile = Profile(world);
    constexpr std::uint64_t input_parameters = 72ULL * 72ULL * 2560ULL + 2560ULL;
    constexpr std::uint64_t hidden_parameters = 2560ULL * 224ULL + 224ULL;
    constexpr std::uint64_t residual_parameters = 16ULL * 2ULL * (224ULL * 224ULL + 224ULL);
    constexpr std::uint64_t output_parameters = 224ULL + 1ULL;
    const std::uint64_t parameter_count = input_parameters + hidden_parameters +
                                          residual_parameters + output_parameters;
    mgt::A100StaticArenaBuildInfo arena_info{
        &profile, 72, 72, 2560, 224, 16, 1, 12500,
        parameter_count, 10000, 320};
    mgt::A100StaticArenaPlanV1 plan{};
    if (mgt::BuildA100StaticArenaPlan(arena_info, &plan) != mgt::Status::kOk) return 1;
    profile.arena_layout_sha256 = plan.layout_sha256;
    profile.ordinary_arena_bytes = plan.ordinary_bytes;
    profile.pinned_host_bytes = plan.pinned_host_bytes;

    mgt_cuda::A100Bf16RuntimeCreateInfo info{};
    info.arena_info = arena_info;
    info.arena_info.profile = &profile;
    info.device_id = device;
    info.rank = rank;
    info.world = world;
    info.bn_nccl_id_file = std::getenv("MGT_BN_ID");
    info.weight_nccl_id_file = std::getenv("MGT_WEIGHT_ID");
    info.metrics_nccl_id_file = std::getenv("MGT_METRICS_ID");
    info.execution_profile = &profile;
    info.profile_use = mgt::A100ExecutionProfileUse::kCandidateForTuner;

    mgt_cuda::A100Bf16Runtime* runtime = nullptr;
    if (mgt_cuda::CreateA100Bf16Runtime(info, plan, &runtime) != mgt::Status::kOk || !runtime)
        return 2;
    mgt_cuda::A100StaticArenaView view{};
    if (mgt_cuda::QueryA100Bf16RuntimeView(runtime, &view) != mgt::Status::kOk ||
        !view.ordinary_base || !view.pinned_host_base ||
        view.ordinary_bytes != plan.ordinary_bytes ||
        view.pinned_host_bytes != plan.pinned_host_bytes || view.symmetric_base)
        return 3;
    const auto* activation_plan = mgt_cuda::A100Bf16RuntimeActivationPlan(runtime);
    if (!mgt_cuda::A100Bf16RuntimeStepControl(runtime) ||
        !mgt_cuda::A100Bf16RuntimeLinearPlan(runtime) || !activation_plan ||
        activation_plan->saved_activation_bf16_count != 12500ULL * (2560ULL + 33ULL * 224ULL))
        return 4;
    mgt_cuda::Bf16BatchNormSiteView first{},last{};
    const auto* sites=mgt_cuda::A100Bf16RuntimeBatchNormSitesPlan(runtime);
    if(!sites||sites->site_count!=34||mgt_cuda::QueryA100Bf16RuntimeBatchNormSite(runtime,0,0,&first)!=mgt::Status::kOk||mgt_cuda::QueryA100Bf16RuntimeBatchNormSite(runtime,33,profile.policy.dz_ring_slots-1,&last)!=mgt::Status::kOk||!first.activation||!last.activation||first.activation==last.activation||mgt_cuda::QueryA100Bf16RuntimeBatchNormSite(runtime,34,0,&last)!=mgt::Status::kInvalidConfig)return 7;
    const auto* bn_config=mgt_cuda::A100Bf16RuntimeBatchNormConfig(runtime);
    const auto* bn_workspace=mgt_cuda::A100Bf16RuntimeBatchNormWorkspace(runtime);
    if(!bn_config||bn_config->row_chunk!=profile.policy.bn_row_chunk||
       bn_config->feature_tile!=profile.policy.bn_feature_tile||!bn_workspace||
       !bn_workspace->partials||!bn_workspace->reduced||!mgt_cuda::A100Bf16RuntimePreactivationScratch(runtime)||!mgt_cuda::A100Bf16RuntimeGradInputScratch(runtime)||bn_workspace->partials==mgt_cuda::A100Bf16RuntimePreactivationScratch(runtime)||bn_workspace->partials==mgt_cuda::A100Bf16RuntimeGradInputScratch(runtime)||bn_workspace->partial_count==0||
       bn_workspace->reduced_count!=2ULL*2560)return 8;
    if (!mgt_cuda::A100Bf16RuntimeHasCommunicators(runtime)) return 6;
    if (mgt_cuda::DestroyA100Bf16Runtime(runtime) != mgt::Status::kOk) return 5;
    return 0;
}