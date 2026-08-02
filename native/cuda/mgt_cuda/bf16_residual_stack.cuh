#pragma once
#include "mgt/status.hpp"
#include <cuda_bf16.h>
#include <array>
#include <cstdint>
namespace mgt_cuda {
struct A100Bf16Runtime;
struct Bf16ResidualStackBindings {
    const __nv_bfloat16* input_activation=nullptr;
    float batch_norm_momentum=0.1f;
    float batch_norm_epsilon=1e-5f;
    std::array<const __nv_bfloat16*,32> weights{};
    std::array<float*,32> weight_grads{};
    std::array<const float*,34> gamma{};
    std::array<const float*,34> beta{};
    std::array<float*,34> running_mean{};
    std::array<float*,34> running_variance{};
    std::array<float*,34> saved_mean{};
    std::array<float*,34> saved_inv_std{};
    std::array<float*,34> dgamma{};
    std::array<float*,34> dbeta{};
};
mgt::Status ValidateBf16ResidualStackBindings(const Bf16ResidualStackBindings& bindings,bool backward);
mgt::Status LaunchA100Bf16ResidualStackForward(A100Bf16Runtime* runtime,const Bf16ResidualStackBindings& bindings,std::uint32_t active_rows,std::uint32_t global_rows,const __nv_bfloat16** output_activation);
mgt::Status LaunchA100Bf16ResidualStackBackward(A100Bf16Runtime* runtime,const Bf16ResidualStackBindings& bindings,const float* output_upstream,std::uint32_t active_rows,std::uint32_t global_rows,float** input_gradient);
}
