#include "mgt_cuda/adamw.cuh"
#include "mgt_cuda/mlp_backward.cuh"
#include "mgt_cuda/mlp_forward.cuh"
#include <cuda_runtime.h>
#include <cmath>
#include <cstdlib>
#include <vector>

namespace {

int Check(cudaError_t status) { return status == cudaSuccess ? 0 : 1; }

std::uint64_t ParamCount(const mgt_cuda::CudaMlpShape& shape) {
    return static_cast<std::uint64_t>(shape.state_len) * shape.state_value_pad * shape.hd1 +
           shape.hd1 + static_cast<std::uint64_t>(shape.hd1) * shape.hd2 + shape.hd2 + shape.hd2 + 1;
}

}  // namespace

int main() {
    int device_count = 0;
    if (Check(cudaGetDeviceCount(&device_count)) != 0 || device_count <= 0) return EXIT_FAILURE;
    if (Check(cudaSetDevice(0)) != 0) return EXIT_FAILURE;

    const mgt_cuda::CudaMlpShape shape{4, 8, 5, 3};
    constexpr std::uint32_t kSamples = 16;
    const std::uint64_t params = ParamCount(shape);
    std::vector<float> weights(params), labels(kSamples);
    std::vector<mgt::TrainState80> states(kSamples);
    for (std::uint64_t i = 0; i < params; ++i) weights[i] = static_cast<float>((static_cast<int>(i % 29) - 14) * 0.002);
    for (std::uint32_t sample = 0; sample < kSamples; ++sample) {
        labels[sample] = static_cast<float>(sample % 4) * 0.2f;
        for (std::uint32_t i = 0; i < shape.state_len; ++i) states[sample].v[i] = static_cast<mgt::StateValue>((i + sample) % shape.state_value_pad);
    }

    float* d_weights = nullptr;
    float* d_labels = nullptr;
    float* d_loss = nullptr;
    float* d_grad = nullptr;
    float* d_m = nullptr;
    float* d_v = nullptr;
    mgt::TrainState80* d_states = nullptr;
    if (Check(cudaMalloc(&d_weights, params * sizeof(float))) != 0) return EXIT_FAILURE;
    if (Check(cudaMalloc(&d_labels, kSamples * sizeof(float))) != 0) return EXIT_FAILURE;
    if (Check(cudaMalloc(&d_loss, sizeof(float))) != 0) return EXIT_FAILURE;
    if (Check(cudaMalloc(&d_grad, params * sizeof(float))) != 0) return EXIT_FAILURE;
    if (Check(cudaMalloc(&d_m, params * sizeof(float))) != 0) return EXIT_FAILURE;
    if (Check(cudaMalloc(&d_v, params * sizeof(float))) != 0) return EXIT_FAILURE;
    if (Check(cudaMalloc(&d_states, kSamples * sizeof(mgt::TrainState80))) != 0) return EXIT_FAILURE;
    if (Check(cudaMemcpy(d_weights, weights.data(), params * sizeof(float), cudaMemcpyHostToDevice)) != 0) return EXIT_FAILURE;
    if (Check(cudaMemcpy(d_labels, labels.data(), kSamples * sizeof(float), cudaMemcpyHostToDevice)) != 0) return EXIT_FAILURE;
    if (Check(cudaMemcpy(d_states, states.data(), kSamples * sizeof(mgt::TrainState80), cudaMemcpyHostToDevice)) != 0) return EXIT_FAILURE;
    if (Check(cudaMemset(d_m, 0, params * sizeof(float))) != 0) return EXIT_FAILURE;
    if (Check(cudaMemset(d_v, 0, params * sizeof(float))) != 0) return EXIT_FAILURE;

    if (mgt_cuda::LaunchMlpLossGradKernel(shape, d_weights, d_states, d_labels, kSamples, d_loss, d_grad, 0) != mgt::Status::kOk) return EXIT_FAILURE;
    if (Check(cudaDeviceSynchronize()) != 0) return EXIT_FAILURE;
    float loss_before = 0.0f;
    if (Check(cudaMemcpy(&loss_before, d_loss, sizeof(float), cudaMemcpyDeviceToHost)) != 0) return EXIT_FAILURE;
    if (!std::isfinite(loss_before) || loss_before <= 0.0f) return EXIT_FAILURE;

    const mgt_cuda::AdamWKernelConfig adam{params, 1, 0.01f, 0.9f, 0.999f, 1.0e-8f, 0.0f};
    if (mgt_cuda::LaunchAdamWKernel(adam, d_weights, d_grad, d_m, d_v, 0) != mgt::Status::kOk) return EXIT_FAILURE;
    if (Check(cudaDeviceSynchronize()) != 0) return EXIT_FAILURE;
    if (mgt_cuda::LaunchMlpLossGradKernel(shape, d_weights, d_states, d_labels, kSamples, d_loss, d_grad, 0) != mgt::Status::kOk) return EXIT_FAILURE;
    if (Check(cudaDeviceSynchronize()) != 0) return EXIT_FAILURE;
    float loss_after = 0.0f;
    if (Check(cudaMemcpy(&loss_after, d_loss, sizeof(float), cudaMemcpyDeviceToHost)) != 0) return EXIT_FAILURE;
    if (!std::isfinite(loss_after) || loss_after > loss_before) return EXIT_FAILURE;

    cudaFree(d_states);
    cudaFree(d_v);
    cudaFree(d_m);
    cudaFree(d_grad);
    cudaFree(d_loss);
    cudaFree(d_labels);
    cudaFree(d_weights);
    return EXIT_SUCCESS;
}