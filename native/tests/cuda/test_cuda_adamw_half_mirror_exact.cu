#include "mgt_cuda/adamw.cuh"
#include "mgt_cuda/fp16_linear_train_ops.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

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
