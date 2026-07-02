#include "mgt_cuda/mlp_forward.cuh"
#include <cuda_runtime.h>
#include <cmath>
#include <cstdlib>
#include <vector>

namespace {

int Check(cudaError_t status) {
    return status == cudaSuccess ? 0 : 1;
}

float Relu(float x) {
    return x > 0.0f ? x : 0.0f;
}

std::uint64_t ParamCount(const mgt_cuda::CudaMlpShape& shape) {
    return static_cast<std::uint64_t>(shape.state_len) * shape.state_value_pad * shape.hd1 +
           shape.hd1 +
           static_cast<std::uint64_t>(shape.hd1) * shape.hd2 +
           shape.hd2 +
           shape.hd2 +
           1;
}

void CpuForwardOne(const mgt_cuda::CudaMlpShape& shape,
                   const std::vector<float>& weights,
                   const mgt::TrainState80& state,
                   float* output) {
    const std::uint64_t input_table = 0;
    const std::uint64_t input_bias = input_table + static_cast<std::uint64_t>(shape.state_len) * shape.state_value_pad * shape.hd1;
    const std::uint64_t hidden_weight = input_bias + shape.hd1;
    const std::uint64_t hidden_bias = hidden_weight + static_cast<std::uint64_t>(shape.hd1) * shape.hd2;
    const std::uint64_t output_weight = hidden_bias + shape.hd2;
    const std::uint64_t output_bias = output_weight + shape.hd2;
    std::vector<float> a1(shape.hd1, 0.0f);
    std::vector<float> a2(shape.hd2, 0.0f);
    for (std::uint32_t h = 0; h < shape.hd1; ++h) {
        float sum = weights[input_bias + h];
        for (std::uint32_t pos = 0; pos < shape.state_len; ++pos) {
            const std::uint32_t value = state.v[pos];
            const std::uint64_t row = static_cast<std::uint64_t>(pos) * shape.state_value_pad + value;
            sum += weights[input_table + row * shape.hd1 + h];
        }
        a1[h] = Relu(sum);
    }
    for (std::uint32_t j = 0; j < shape.hd2; ++j) {
        float sum = weights[hidden_bias + j];
        for (std::uint32_t h = 0; h < shape.hd1; ++h) {
            sum += a1[h] * weights[hidden_weight + static_cast<std::uint64_t>(h) * shape.hd2 + j];
        }
        a2[j] = Relu(sum);
    }
    float y = weights[output_bias];
    for (std::uint32_t j = 0; j < shape.hd2; ++j) {
        y += a2[j] * weights[output_weight + j];
    }
    *output = y;
}

}  // namespace

int main() {
    int device_count = 0;
    if (Check(cudaGetDeviceCount(&device_count)) != 0 || device_count <= 0) return EXIT_FAILURE;
    if (Check(cudaSetDevice(0)) != 0) return EXIT_FAILURE;

    const mgt_cuda::CudaMlpShape shape{4, 8, 5, 3};
    constexpr std::uint32_t kSamples = 16;
    const std::uint64_t params = ParamCount(shape);
    std::vector<float> weights(params);
    for (std::uint64_t i = 0; i < params; ++i) {
        weights[i] = static_cast<float>((static_cast<int>(i % 19) - 9) * 0.005);
    }
    std::vector<mgt::TrainState80> states(kSamples);
    for (std::uint32_t sample = 0; sample < kSamples; ++sample) {
        for (std::uint32_t i = 0; i < shape.state_len; ++i) {
            states[sample].v[i] = static_cast<mgt::StateValue>((i + sample) % shape.state_value_pad);
        }
    }
    std::vector<float> cpu(kSamples, 0.0f);
    for (std::uint32_t sample = 0; sample < kSamples; ++sample) {
        CpuForwardOne(shape, weights, states[sample], &cpu[sample]);
    }

    float* d_weights = nullptr;
    mgt::TrainState80* d_states = nullptr;
    float* d_outputs = nullptr;
    if (Check(cudaMalloc(&d_weights, params * sizeof(float))) != 0) return EXIT_FAILURE;
    if (Check(cudaMalloc(&d_states, kSamples * sizeof(mgt::TrainState80))) != 0) return EXIT_FAILURE;
    if (Check(cudaMalloc(&d_outputs, kSamples * sizeof(float))) != 0) return EXIT_FAILURE;
    if (Check(cudaMemcpy(d_weights, weights.data(), params * sizeof(float), cudaMemcpyHostToDevice)) != 0) return EXIT_FAILURE;
    if (Check(cudaMemcpy(d_states, states.data(), kSamples * sizeof(mgt::TrainState80), cudaMemcpyHostToDevice)) != 0) return EXIT_FAILURE;

    if (mgt_cuda::LaunchMlpForwardKernel(shape, d_weights, d_states, kSamples, d_outputs, 0) != mgt::Status::kOk) {
        return EXIT_FAILURE;
    }
    if (Check(cudaDeviceSynchronize()) != 0) return EXIT_FAILURE;

    std::vector<float> gpu(kSamples, 0.0f);
    if (Check(cudaMemcpy(gpu.data(), d_outputs, kSamples * sizeof(float), cudaMemcpyDeviceToHost)) != 0) return EXIT_FAILURE;
    for (std::uint32_t sample = 0; sample < kSamples; ++sample) {
        if (std::fabs(gpu[sample] - cpu[sample]) > 1.0e-5f) return EXIT_FAILURE;
    }

    cudaFree(d_outputs);
    cudaFree(d_states);
    cudaFree(d_weights);
    return EXIT_SUCCESS;
}