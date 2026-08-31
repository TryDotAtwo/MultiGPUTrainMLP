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
    // Training tape capacity, in half elements, for the maximum active rows:
    // rows * (hd1 + 2*residual_blocks*hd2 + (output_dim > 1 ? hd2 : 0)).
    // Retain device storage through backward; sequential reuse requires the
    // same stream or an explicit completion dependency.
    __half* operand_a = nullptr;
    __half* operand_b = nullptr;
    std::uint64_t operand_a_capacity = 0;
    std::uint64_t operand_b_capacity = 0;
    // Step-local producer tag: BN dX (or an explicit cast) writes operand_b;
    // adjacent dense dW/dX consume it. Public step entry/exit invalidate this
    // tag, including failures. It is not a cross-step pointer-identity cache.
    const float* cached_operand_b_source = nullptr;
    std::uint64_t cached_operand_b_count = 0;
    float* activation_workspace = nullptr;
    std::uint32_t activation_rows = 0;
    std::uint32_t activation_hd1 = 0;
    std::uint32_t activation_hd2 = 0;
    std::uint32_t activation_residual_blocks = 0;
    // Tape: input activation, then each block input/fc1 activation pair.
    // Vector heads also need the final activation for forward and output dW.
    bool activation_has_final = false;
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
