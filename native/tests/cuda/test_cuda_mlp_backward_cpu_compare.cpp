#include "mgt/mlp_cpu_ref.hpp"
#include "mgt_cuda/mlp_backward.cuh"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <vector>

namespace {
int Check(cudaError_t status) { return status == cudaSuccess ? 0 : 1; }
std::uint64_t ResidualBlockParams(const mgt_cuda::CudaMlpShape& s) { return 2ULL * (static_cast<std::uint64_t>(s.hd2) * s.hd2 + s.hd2); }
std::uint64_t ParamCount(const mgt_cuda::CudaMlpShape& s) { return static_cast<std::uint64_t>(s.state_len) * s.state_value_pad * s.hd1 + s.hd1 + static_cast<std::uint64_t>(s.hd1) * s.hd2 + s.hd2 + static_cast<std::uint64_t>(s.residual_blocks) * ResidualBlockParams(s) + static_cast<std::uint64_t>(s.hd2) * s.output_dim + s.output_dim; }

bool Close(float lhs, float rhs, float atol, float rtol) {
    return std::fabs(lhs - rhs) <= atol + rtol * std::max(std::fabs(lhs), std::fabs(rhs));
}

int RunCase(const mgt_cuda::CudaMlpShape& shape, std::uint32_t samples) {
    const mgt::CpuMlpShape cpu_shape{shape.state_len, shape.state_value_pad, shape.hd1, shape.hd2, shape.residual_blocks, shape.output_dim};
    const std::uint64_t params = ParamCount(shape);
    if (params != mgt::CpuMlpParamCount(cpu_shape)) {
        std::fprintf(stderr, "param count mismatch output_dim=%u params=%llu cpu=%llu\n", shape.output_dim, static_cast<unsigned long long>(params), static_cast<unsigned long long>(mgt::CpuMlpParamCount(cpu_shape)));
        return EXIT_FAILURE;
    }

    std::vector<float> weights(params), labels(static_cast<std::uint64_t>(samples) * shape.output_dim), gpu_grad(params), cpu_grad(params);
    std::vector<mgt::TrainStateStorage> states(samples);
    for (std::uint64_t i = 0; i < params; ++i) weights[i] = static_cast<float>((static_cast<int>((i * 7) % 29) - 14) * 0.00325);
    for (std::uint32_t sample = 0; sample < samples; ++sample) {
        for (std::uint32_t out = 0; out < shape.output_dim; ++out) {
            const int value = static_cast<int>((sample * 3 + out * 5 + shape.output_dim) % 17) - 8;
            labels[static_cast<std::uint64_t>(sample) * shape.output_dim + out] = static_cast<float>(value) * 0.0625f;
        }
        for (std::uint32_t i = 0; i < shape.state_len; ++i) states[sample].v[i] = static_cast<mgt::StateValue>((i * 3 + sample * 5 + shape.output_dim) % shape.state_value_pad);
        for (std::uint32_t i = shape.state_len; i < mgt::kStateStorageLen; ++i) states[sample].v[i] = 0;
    }

    float cpu_loss = 0.0f;
    if (mgt::CpuMlpLossAndGrad(cpu_shape, weights, states.data(), labels.data(), samples, &cpu_loss, cpu_grad.data()) != mgt::Status::kOk) return EXIT_FAILURE;

    float* d_weights = nullptr;
    float* d_labels = nullptr;
    float* d_loss = nullptr;
    float* d_grad = nullptr;
    mgt::TrainStateStorage* d_states = nullptr;
    if (Check(cudaMalloc(&d_weights, params * sizeof(float))) != 0) return EXIT_FAILURE;
    if (Check(cudaMalloc(&d_labels, labels.size() * sizeof(float))) != 0) return EXIT_FAILURE;
    if (Check(cudaMalloc(&d_loss, sizeof(float))) != 0) return EXIT_FAILURE;
    if (Check(cudaMalloc(&d_grad, params * sizeof(float))) != 0) return EXIT_FAILURE;
    if (Check(cudaMalloc(&d_states, samples * sizeof(mgt::TrainStateStorage))) != 0) return EXIT_FAILURE;
    if (Check(cudaMemcpy(d_weights, weights.data(), params * sizeof(float), cudaMemcpyHostToDevice)) != 0) return EXIT_FAILURE;
    if (Check(cudaMemcpy(d_labels, labels.data(), labels.size() * sizeof(float), cudaMemcpyHostToDevice)) != 0) return EXIT_FAILURE;
    if (Check(cudaMemcpy(d_states, states.data(), samples * sizeof(mgt::TrainStateStorage), cudaMemcpyHostToDevice)) != 0) return EXIT_FAILURE;

    if (mgt_cuda::LaunchMlpLossGradKernel(shape, d_weights, d_states, d_labels, samples, d_loss, d_grad, 0) != mgt::Status::kOk) return EXIT_FAILURE;
    if (Check(cudaDeviceSynchronize()) != 0) return EXIT_FAILURE;

    float gpu_loss = 0.0f;
    if (Check(cudaMemcpy(&gpu_loss, d_loss, sizeof(float), cudaMemcpyDeviceToHost)) != 0) return EXIT_FAILURE;
    if (!std::isfinite(gpu_loss) || !Close(gpu_loss, cpu_loss, 1.0e-5f, 1.0e-5f)) {
        std::fprintf(stderr, "loss mismatch output_dim=%u gpu=%0.9g cpu=%0.9g\n", shape.output_dim, gpu_loss, cpu_loss);
        return EXIT_FAILURE;
    }
    if (Check(cudaMemcpy(gpu_grad.data(), d_grad, params * sizeof(float), cudaMemcpyDeviceToHost)) != 0) return EXIT_FAILURE;
    int mismatches = 0;
    for (std::uint64_t i = 0; i < params; ++i) {
        if (!std::isfinite(gpu_grad[i]) || !Close(gpu_grad[i], cpu_grad[i], 2.0e-5f, 2.0e-5f)) {
            if (mismatches < 16) std::fprintf(stderr, "grad mismatch output_dim=%u i=%llu gpu=%0.9g cpu=%0.9g\n", shape.output_dim, static_cast<unsigned long long>(i), gpu_grad[i], cpu_grad[i]);
            ++mismatches;
        }
    }
    if (mismatches != 0) {
        std::fprintf(stderr, "grad mismatch output_dim=%u count=%d params=%llu\n", shape.output_dim, mismatches, static_cast<unsigned long long>(params));
        return EXIT_FAILURE;
    }

    cudaFree(d_states);
    cudaFree(d_grad);
    cudaFree(d_loss);
    cudaFree(d_labels);
    cudaFree(d_weights);
    return EXIT_SUCCESS;
}
}  // namespace

int main() {
    int device_count = 0;
    if (Check(cudaGetDeviceCount(&device_count)) != 0 || device_count <= 0) return EXIT_FAILURE;
    if (Check(cudaSetDevice(0)) != 0) return EXIT_FAILURE;

    const struct TestCase {
        mgt_cuda::CudaMlpShape shape;
        std::uint32_t samples;
    } cases[] = {
        {{4, 8, 5, 3, 1, 1}, 8},
        {{4, 8, 5, 3, 1, 3}, 8},
        {{6, 8, 9, 7, 2, 11}, 10},
    };

    for (const TestCase& test_case : cases) {
        if (RunCase(test_case.shape, test_case.samples) != EXIT_SUCCESS) return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
