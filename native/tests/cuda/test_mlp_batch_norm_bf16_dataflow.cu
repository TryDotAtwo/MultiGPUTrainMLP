#include "mgt_cuda/bf16_activation.cuh"

#include <cuda_bf16.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

namespace {

std::uint16_t Bits(__nv_bfloat16 value) {
    std::uint16_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

bool Aligned(std::uint64_t value) { return (value & 255ULL) == 0; }

}  // namespace

int main() {
    const mgt_cuda::CudaMlpShape shape{72, 16, 2560, 224, 16, 1};
    mgt_cuda::MlpBatchNormBf16WorkspacePlan fp32{}, bf16{};
    if (mgt_cuda::BuildMlpBatchNormBf16WorkspacePlan(
            shape, 12500, 3, mgt::A100XhatStorage::kFp32, &fp32) != mgt::Status::kOk)
        return 1;
    if (mgt_cuda::BuildMlpBatchNormBf16WorkspacePlan(
            shape, 12500, 3, mgt::A100XhatStorage::kBf16, &bf16) != mgt::Status::kOk)
        return 2;
    const std::uint64_t activation_count = 12500ULL * (2560ULL + 33ULL * 224ULL);
    if (fp32.saved_activation_bf16_count != activation_count ||
        bf16.saved_activation_bf16_count != activation_count ||
        fp32.saved_xhat_count != activation_count || bf16.saved_xhat_count != activation_count)
        return 3;
    if (!Aligned(fp32.saved_activation_bf16_offset) || !Aligned(fp32.saved_xhat_offset) ||
        !Aligned(fp32.relu_mask_offset_bytes) || !Aligned(fp32.dz_ring_bf16_offset) ||
        !Aligned(fp32.preactivation_f32_offset) || !Aligned(fp32.grad_input_f32_offset) ||
        !Aligned(fp32.total_bytes))
        return 4;
    if (fp32.total_bytes <= bf16.total_bytes || fp32.relu_mask_bytes != bf16.relu_mask_bytes)
        return 5;
    if (mgt_cuda::BuildMlpBatchNormBf16WorkspacePlan(
            shape, 0, 3, mgt::A100XhatStorage::kFp32, &fp32) != mgt::Status::kInvalidConfig)
        return 6;
    if (mgt_cuda::BuildMlpBatchNormBf16WorkspacePlan(
            shape, 12500, 1, mgt::A100XhatStorage::kFp32, &fp32) != mgt::Status::kInvalidConfig)
        return 7;
    auto overflow = shape;
    overflow.hd1 = std::numeric_limits<std::uint32_t>::max();
    overflow.hd2 = std::numeric_limits<std::uint32_t>::max();
    overflow.residual_blocks = std::numeric_limits<std::uint32_t>::max();
    if (mgt_cuda::BuildMlpBatchNormBf16WorkspacePlan(
            overflow, std::numeric_limits<std::uint32_t>::max(), 4,
            mgt::A100XhatStorage::kFp32, &fp32) != mgt::Status::kCapacityExceeded)
        return 8;

    constexpr std::uint32_t capacity_rows = 3;
    constexpr std::uint32_t active_rows = 2;
    constexpr std::uint32_t physical_features = 64;
    constexpr std::uint32_t logical_features = 35;
    std::vector<float> host(capacity_rows * physical_features, 7.0f);
    for (std::uint32_t row = 0; row < active_rows; ++row) {
        for (std::uint32_t col = 0; col < logical_features; ++col)
            host[row * physical_features + col] = (col % 3 == 0) ? -0.25f : 0.5f + col;
    }
    float* input = nullptr;
    __nv_bfloat16* activation = nullptr;
    __nv_bfloat16* dz = nullptr;
    std::uint32_t* mask = nullptr;
    const std::size_t values = capacity_rows * physical_features;
    const std::size_t mask_words = capacity_rows * ((physical_features + 31) / 32);
    if (cudaMalloc(&input, values * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&activation, values * sizeof(*activation)) != cudaSuccess ||
        cudaMalloc(&dz, values * sizeof(*dz)) != cudaSuccess ||
        cudaMalloc(&mask, mask_words * sizeof(*mask)) != cudaSuccess)
        return 9;
    if (cudaMemcpy(input, host.data(), values * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess)
        return 10;
    if (mgt_cuda::LaunchPackReluActivationBf16(
            input, active_rows, capacity_rows, logical_features, physical_features,
            activation, mask, nullptr) != mgt::Status::kOk)
        return 11;
    if (mgt_cuda::LaunchGateGradientToBf16(
            input, mask, active_rows, capacity_rows, logical_features, physical_features,
            dz, nullptr) != mgt::Status::kOk || cudaDeviceSynchronize() != cudaSuccess)
        return 12;
    std::vector<__nv_bfloat16> got_activation(values), got_dz(values);
    std::vector<std::uint32_t> got_mask(mask_words);
    if (cudaMemcpy(got_activation.data(), activation, values * sizeof(*activation),
                   cudaMemcpyDeviceToHost) != cudaSuccess ||
        cudaMemcpy(got_dz.data(), dz, values * sizeof(*dz), cudaMemcpyDeviceToHost) != cudaSuccess ||
        cudaMemcpy(got_mask.data(), mask, mask_words * sizeof(*mask),
                   cudaMemcpyDeviceToHost) != cudaSuccess)
        return 13;
    for (std::uint32_t row = 0; row < capacity_rows; ++row) {
        for (std::uint32_t col = 0; col < physical_features; ++col) {
            const bool valid = row < active_rows && col < logical_features;
            const bool positive = valid && host[row * physical_features + col] > 0.0f;
            const auto expected = __nv_bfloat16(positive ? host[row * physical_features + col] : 0.0f);
            if (Bits(got_activation[row * physical_features + col]) != Bits(expected) ||
                Bits(got_dz[row * physical_features + col]) != Bits(expected))
                return 14;
            const bool bit = (got_mask[row * 2 + col / 32] >> (col & 31)) & 1U;
            if (bit != positive) return 15;
        }
    }
    cudaFree(mask);
    cudaFree(dz);
    cudaFree(activation);
    cudaFree(input);
    return 0;
}
