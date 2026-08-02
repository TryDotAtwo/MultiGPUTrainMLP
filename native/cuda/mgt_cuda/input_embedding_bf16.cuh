#pragma once

#include "mgt/static_contracts.hpp"
#include "mgt/status.hpp"
#include "mgt_cuda/a100_bf16_runtime.cuh"
#include "mgt_cuda/bf16_linear_train_ops.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace mgt_cuda {

struct InputEmbeddingBf16Plan;

struct InputEmbeddingBf16Config {
    std::uint32_t state_len = 72;
    std::uint32_t state_value_pad = 72;
    std::uint32_t output_features = 2560;
    std::uint32_t positions_per_tile = 8;
    std::uint32_t capacity_rows = 0;
};

mgt::Status CreateInputEmbeddingBf16Plan(
    InputEmbeddingBf16Config config,
    const std::uint32_t* supported_active_rows,
    std::uint32_t supported_active_row_count,
    cublasLtHandle_t handle,
    const mgt::Bf16AlgorithmTable& algorithms,
    const A100StaticArenaView& arena,
    __nv_bfloat16* one_hot_scratch,
    std::uint64_t one_hot_scratch_elements,
    InputEmbeddingBf16Plan** out);
mgt::Status DestroyInputEmbeddingBf16Plan(InputEmbeddingBf16Plan* plan);

mgt::Status LaunchInputEmbeddingForwardBf16(
    const InputEmbeddingBf16Plan* plan,
    const mgt::TrainStateStorage* states,
    const __nv_bfloat16* table,
    std::uint32_t active_rows,
    float* output,
    cudaStream_t stream);

mgt::Status LaunchInputEmbeddingTableGradBf16(
    const InputEmbeddingBf16Plan* plan,
    const mgt::TrainStateStorage* states,
    const __nv_bfloat16* dz,
    std::uint32_t active_rows,
    float* table_grad,
    cudaStream_t stream);

}  // namespace mgt_cuda