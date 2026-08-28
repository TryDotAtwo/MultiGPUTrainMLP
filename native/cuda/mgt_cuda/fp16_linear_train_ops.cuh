#pragma once

#include "mgt/status.hpp"

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace mgt_cuda {

struct LocalMlpFp16Context {
    float* master_weights = nullptr;
    __half* weight_mirror = nullptr;
    __half* operand_a = nullptr;
    __half* operand_b = nullptr;
    std::uint64_t operand_a_capacity = 0;
    std::uint64_t operand_b_capacity = 0;
};

mgt::Status LaunchFloatToHalf(
    const float* input, __half* output, std::uint64_t count, cudaStream_t stream);
mgt::Status LaunchFp16LinearForward(
    cublasHandle_t blas, const __half* input, const __half* weight,
    float* output, std::uint32_t rows, std::uint32_t input_features,
    std::uint32_t output_features, cudaStream_t stream);
mgt::Status LaunchFp16LinearGradWeight(
    cublasHandle_t blas, const __half* input, const __half* grad_output,
    float* grad_weight, std::uint32_t rows, std::uint32_t input_features,
    std::uint32_t output_features, cudaStream_t stream);
mgt::Status LaunchFp16LinearGradInput(
    cublasHandle_t blas, const __half* grad_output, const __half* weight,
    float* grad_input, std::uint32_t rows, std::uint32_t input_features,
    std::uint32_t output_features, float beta, cudaStream_t stream);

}  // namespace mgt_cuda
