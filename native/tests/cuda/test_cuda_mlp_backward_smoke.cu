#include "mgt_cuda/mlp_backward.cuh"
#include <cuda_runtime.h>
#include <cmath>
#include <cstdlib>
#include <vector>

namespace {
int Check(cudaError_t status) { return status == cudaSuccess ? 0 : 1; }
std::uint64_t ResidualBlockParams(const mgt_cuda::CudaMlpShape& s) { return 2ULL * (static_cast<std::uint64_t>(s.hd2) * s.hd2 + s.hd2); }
std::uint64_t ParamCount(const mgt_cuda::CudaMlpShape& s) { return static_cast<std::uint64_t>(s.state_len) * s.state_value_pad * s.hd1 + s.hd1 + static_cast<std::uint64_t>(s.hd1) * s.hd2 + s.hd2 + static_cast<std::uint64_t>(s.residual_blocks) * ResidualBlockParams(s) + s.hd2 + 1ULL; }
}  // namespace

int main() {
    int device_count = 0;
    if (Check(cudaGetDeviceCount(&device_count)) != 0 || device_count <= 0) return EXIT_FAILURE;
    if (Check(cudaSetDevice(0)) != 0) return EXIT_FAILURE;
    const mgt_cuda::CudaMlpShape shape{4, 8, 5, 3, 1};
    constexpr std::uint32_t kSamples = 8;
    const std::uint64_t params = ParamCount(shape);
    std::vector<float> weights(params), labels(kSamples), gpu_grad(params);
    std::vector<mgt::TrainState80> states(kSamples);
    for (std::uint64_t i = 0; i < params; ++i) weights[i] = static_cast<float>((static_cast<int>(i % 23) - 11) * 0.004);
    for (std::uint32_t sample = 0; sample < kSamples; ++sample) {
        labels[sample] = static_cast<float>(sample % 5) * 0.25f;
        for (std::uint32_t i = 0; i < shape.state_len; ++i) states[sample].v[i] = static_cast<mgt::StateValue>((i + sample) % shape.state_value_pad);
    }
    float* d_weights = nullptr; float* d_labels = nullptr; float* d_loss = nullptr; float* d_grad = nullptr; mgt::TrainState80* d_states = nullptr;
    if (Check(cudaMalloc(&d_weights, params * sizeof(float))) != 0) return EXIT_FAILURE;
    if (Check(cudaMalloc(&d_labels, kSamples * sizeof(float))) != 0) return EXIT_FAILURE;
    if (Check(cudaMalloc(&d_loss, sizeof(float))) != 0) return EXIT_FAILURE;
    if (Check(cudaMalloc(&d_grad, params * sizeof(float))) != 0) return EXIT_FAILURE;
    if (Check(cudaMalloc(&d_states, kSamples * sizeof(mgt::TrainState80))) != 0) return EXIT_FAILURE;
    if (Check(cudaMemcpy(d_weights, weights.data(), params * sizeof(float), cudaMemcpyHostToDevice)) != 0) return EXIT_FAILURE;
    if (Check(cudaMemcpy(d_labels, labels.data(), kSamples * sizeof(float), cudaMemcpyHostToDevice)) != 0) return EXIT_FAILURE;
    if (Check(cudaMemcpy(d_states, states.data(), kSamples * sizeof(mgt::TrainState80), cudaMemcpyHostToDevice)) != 0) return EXIT_FAILURE;
    if (mgt_cuda::LaunchMlpLossGradKernel(shape, d_weights, d_states, d_labels, kSamples, d_loss, d_grad, 0) != mgt::Status::kOk) return EXIT_FAILURE;
    if (Check(cudaDeviceSynchronize()) != 0) return EXIT_FAILURE;
    float gpu_loss = 0.0f;
    if (Check(cudaMemcpy(&gpu_loss, d_loss, sizeof(float), cudaMemcpyDeviceToHost)) != 0) return EXIT_FAILURE;
    if (!std::isfinite(gpu_loss) || gpu_loss <= 0.0f) return EXIT_FAILURE;
    if (Check(cudaMemcpy(gpu_grad.data(), d_grad, params * sizeof(float), cudaMemcpyDeviceToHost)) != 0) return EXIT_FAILURE;
    bool nonzero_grad = false;
    for (float value : gpu_grad) {
        if (!std::isfinite(value)) return EXIT_FAILURE;
        nonzero_grad = nonzero_grad || std::fabs(value) > 0.0f;
    }
    if (!nonzero_grad) return EXIT_FAILURE;
    cudaFree(d_states); cudaFree(d_grad); cudaFree(d_loss); cudaFree(d_labels); cudaFree(d_weights);
    return EXIT_SUCCESS;
}