#include "mgt_cuda/adamw.cuh"
#include "mgt_cuda/fp16_linear_train_ops.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

bool Cuda(cudaError_t status) { return status == cudaSuccess; }

template <class T>
bool SameDeviceBytes(const T* left, const T* right, std::uint64_t count) {
    std::vector<T> a(count);
    std::vector<T> b(count);
    return Cuda(cudaMemcpy(a.data(), left, count * sizeof(T), cudaMemcpyDeviceToHost)) &&
           Cuda(cudaMemcpy(b.data(), right, count * sizeof(T), cudaMemcpyDeviceToHost)) &&
           std::memcmp(a.data(), b.data(), count * sizeof(T)) == 0;
}

}  // namespace

int main() {
    constexpr std::uint64_t kCount = 259;
    std::vector<float> weights(kCount), grad(kCount), moment1(kCount), moment2(kCount);
    for (std::uint64_t i = 0; i < kCount; ++i) {
        weights[i] = static_cast<float>(static_cast<int>(i % 31) - 15) * .002f;
        grad[i] = static_cast<float>(static_cast<int>((i * 7) % 29) - 14) * .0003f;
        moment1[i] = static_cast<float>(static_cast<int>((i * 3) % 23) - 11) * .0001f;
        moment2[i] = static_cast<float>((i * 5) % 19) * .00001f;
    }

    float *reference_weights = nullptr, *reference_m = nullptr, *reference_v = nullptr;
    float *fused_weights = nullptr, *fused_m = nullptr, *fused_v = nullptr, *device_grad = nullptr;
    __half *reference_half = nullptr, *fused_half = nullptr;
    if (!Cuda(cudaMalloc(&reference_weights, kCount * sizeof(float))) ||
        !Cuda(cudaMalloc(&reference_m, kCount * sizeof(float))) ||
        !Cuda(cudaMalloc(&reference_v, kCount * sizeof(float))) ||
        !Cuda(cudaMalloc(&fused_weights, kCount * sizeof(float))) ||
        !Cuda(cudaMalloc(&fused_m, kCount * sizeof(float))) ||
        !Cuda(cudaMalloc(&fused_v, kCount * sizeof(float))) ||
        !Cuda(cudaMalloc(&device_grad, kCount * sizeof(float))) ||
        !Cuda(cudaMalloc(&reference_half, kCount * sizeof(__half))) ||
        !Cuda(cudaMalloc(&fused_half, kCount * sizeof(__half)))) return EXIT_FAILURE;

    for (auto* pointer : {reference_weights, fused_weights})
        if (!Cuda(cudaMemcpy(pointer, weights.data(), kCount * sizeof(float), cudaMemcpyHostToDevice)))
            return EXIT_FAILURE;
    for (auto* pointer : {reference_m, fused_m})
        if (!Cuda(cudaMemcpy(pointer, moment1.data(), kCount * sizeof(float), cudaMemcpyHostToDevice)))
            return EXIT_FAILURE;
    for (auto* pointer : {reference_v, fused_v})
        if (!Cuda(cudaMemcpy(pointer, moment2.data(), kCount * sizeof(float), cudaMemcpyHostToDevice)))
            return EXIT_FAILURE;
    if (!Cuda(cudaMemcpy(device_grad, grad.data(), kCount * sizeof(float), cudaMemcpyHostToDevice)))
        return EXIT_FAILURE;

    mgt_cuda::AdamWKernelConfig config{kCount, 1, .0001f, .9f, .999f, 1e-8f, .01f};
    for (const auto step : std::array<std::uint64_t, 4>{1, 2, 997, 65535}) {
        config.step = step;
        if (mgt_cuda::LaunchAdamWKernel(
                config, reference_weights, device_grad, reference_m, reference_v, nullptr) !=
                mgt::Status::kOk ||
            mgt_cuda::LaunchFloatToHalf(reference_weights, reference_half, kCount, nullptr) !=
                mgt::Status::kOk ||
            mgt_cuda::LaunchAdamWKernelWithHalfMirror(
                config, fused_weights, fused_half, device_grad, fused_m, fused_v, nullptr) !=
                mgt::Status::kOk) return EXIT_FAILURE;
    }
    if (!Cuda(cudaDeviceSynchronize()) ||
        !SameDeviceBytes(reference_weights, fused_weights, kCount) ||
        !SameDeviceBytes(reference_m, fused_m, kCount) ||
        !SameDeviceBytes(reference_v, fused_v, kCount) ||
        !SameDeviceBytes(reference_half, fused_half, kCount)) return EXIT_FAILURE;

    constexpr std::uint64_t kSparseCount = 19;
    constexpr std::uint32_t kRowWidth = 3;
    constexpr std::uint32_t kActiveCount = 3;
    constexpr std::uint64_t kInputCount = 15;
    constexpr std::uint64_t kLiveCount = 13;
    const std::array<std::uint16_t, kActiveCount> active_bins{0, 2, 4};
    std::uint16_t* device_active_bins = nullptr;
    if (!Cuda(cudaMalloc(&device_active_bins, sizeof(active_bins))) ||
        !Cuda(cudaMemcpy(device_active_bins, active_bins.data(),
                         sizeof(active_bins), cudaMemcpyHostToDevice)))
        return EXIT_FAILURE;
    std::fill(weights.begin(), weights.end(), 0.0f);
    std::fill(grad.begin(), grad.end(), 0.0f);
    std::fill(moment1.begin(), moment1.end(), 0.0f);
    std::fill(moment2.begin(), moment2.end(), 0.0f);
    for (std::uint64_t i = 0; i < kSparseCount; ++i) {
        weights[i] = static_cast<float>(static_cast<int>(i) - 9) * .003f;
        const bool live = i < 3 || (i >= 6 && i < 9) || i >= 12;
        if (live) {
            grad[i] = static_cast<float>(static_cast<int>((i * 5) % 13) - 6) *
                .0004f;
            moment1[i] = static_cast<float>(static_cast<int>((i * 3) % 11) - 5) *
                .0002f;
            moment2[i] = static_cast<float>((i * 7) % 17) * .00001f;
        }
    }
    for (auto* pointer : {reference_weights, fused_weights})
        if (!Cuda(cudaMemcpy(pointer, weights.data(), kSparseCount * sizeof(float),
                             cudaMemcpyHostToDevice))) return EXIT_FAILURE;
    for (auto* pointer : {reference_m, fused_m})
        if (!Cuda(cudaMemcpy(pointer, moment1.data(), kSparseCount * sizeof(float),
                             cudaMemcpyHostToDevice))) return EXIT_FAILURE;
    for (auto* pointer : {reference_v, fused_v})
        if (!Cuda(cudaMemcpy(pointer, moment2.data(), kSparseCount * sizeof(float),
                             cudaMemcpyHostToDevice))) return EXIT_FAILURE;
    if (!Cuda(cudaMemcpy(device_grad, grad.data(), kSparseCount * sizeof(float),
                         cudaMemcpyHostToDevice)) ||
        mgt_cuda::LaunchFloatToHalf(
            reference_weights, reference_half, kSparseCount, nullptr) !=
            mgt::Status::kOk ||
        mgt_cuda::LaunchFloatToHalf(
            fused_weights, fused_half, kSparseCount, nullptr) != mgt::Status::kOk)
        return EXIT_FAILURE;

    mgt_cuda::AdamWKernelConfig dense{
        kSparseCount, 1, .0001f, .9f, .999f, 1e-8f, 0.0f};
    auto sparse = dense;
    sparse.param_count = kLiveCount;
    sparse.sparse_active_bins = device_active_bins;
    sparse.sparse_full_prefix_count = kInputCount;
    sparse.sparse_row_width = kRowWidth;
    sparse.sparse_active_bin_count = kActiveCount;
    std::uint64_t physical_count = 0;
    if (mgt_cuda::QueryAdamWPhysicalParameterCount(sparse, &physical_count) !=
            mgt::Status::kOk ||
        physical_count != kSparseCount)
        return EXIT_FAILURE;
    auto invalid = sparse;
    invalid.sparse_active_bins = nullptr;
    if (mgt_cuda::ValidateAdamWKernelConfig(invalid) == mgt::Status::kOk)
        return EXIT_FAILURE;
    invalid = sparse;
    invalid.weight_decay = .01f;
    if (mgt_cuda::ValidateAdamWKernelConfig(invalid) == mgt::Status::kOk)
        return EXIT_FAILURE;
    invalid = sparse;
    invalid.sparse_active_bin_count = 6;
    if (mgt_cuda::ValidateAdamWKernelConfig(invalid) == mgt::Status::kOk)
        return EXIT_FAILURE;

    for (const auto step : std::array<std::uint64_t, 4>{1, 2, 997, 65535}) {
        dense.step = step;
        sparse.step = step;
        if (mgt_cuda::LaunchAdamWKernel(
                dense, reference_weights, device_grad, reference_m, reference_v,
                nullptr) != mgt::Status::kOk ||
            mgt_cuda::LaunchFloatToHalf(
                reference_weights, reference_half, kSparseCount, nullptr) !=
                mgt::Status::kOk ||
            mgt_cuda::LaunchAdamWKernelWithHalfMirror(
                sparse, fused_weights, fused_half, device_grad, fused_m, fused_v,
                nullptr) != mgt::Status::kOk)
            return EXIT_FAILURE;
    }
    if (!Cuda(cudaDeviceSynchronize()) ||
        !SameDeviceBytes(reference_weights, fused_weights, kSparseCount) ||
        !SameDeviceBytes(reference_m, fused_m, kSparseCount) ||
        !SameDeviceBytes(reference_v, fused_v, kSparseCount) ||
        !SameDeviceBytes(reference_half, fused_half, kSparseCount))
        return EXIT_FAILURE;

    cudaFree(device_active_bins);
    cudaFree(fused_half);
    cudaFree(reference_half);
    cudaFree(device_grad);
    cudaFree(fused_v);
    cudaFree(fused_m);
    cudaFree(fused_weights);
    cudaFree(reference_v);
    cudaFree(reference_m);
    cudaFree(reference_weights);
    return EXIT_SUCCESS;
}
