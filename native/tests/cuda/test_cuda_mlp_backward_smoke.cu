#include "mgt_cuda/mlp_backward.cuh"
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

float ReluGrad(float x) {
    return x > 0.0f ? 1.0f : 0.0f;
}

std::uint64_t ParamCount(const mgt_cuda::CudaMlpShape& shape) {
    return static_cast<std::uint64_t>(shape.state_len) * shape.state_value_pad * shape.hd1 +
           shape.hd1 + static_cast<std::uint64_t>(shape.hd1) * shape.hd2 + shape.hd2 + shape.hd2 + 1;
}

void CpuLossGrad(const mgt_cuda::CudaMlpShape& shape,
                 const std::vector<float>& weights,
                 const std::vector<mgt::TrainState80>& states,
                 const std::vector<float>& labels,
                 float* loss,
                 std::vector<float>* grad) {
    const std::uint64_t input_table = 0;
    const std::uint64_t input_bias = input_table + static_cast<std::uint64_t>(shape.state_len) * shape.state_value_pad * shape.hd1;
    const std::uint64_t hidden_weight = input_bias + shape.hd1;
    const std::uint64_t hidden_bias = hidden_weight + static_cast<std::uint64_t>(shape.hd1) * shape.hd2;
    const std::uint64_t output_weight = hidden_bias + shape.hd2;
    const std::uint64_t output_bias = output_weight + shape.hd2;
    std::fill(grad->begin(), grad->end(), 0.0f);
    *loss = 0.0f;
    std::vector<float> z1(shape.hd1), a1(shape.hd1), z2(shape.hd2), a2(shape.hd2), dz1(shape.hd1), da1(shape.hd1), dz2(shape.hd2), da2(shape.hd2);
    const float inv_n = 1.0f / static_cast<float>(states.size());
    for (std::uint32_t sample = 0; sample < states.size(); ++sample) {
        for (std::uint32_t h = 0; h < shape.hd1; ++h) {
            float sum = weights[input_bias + h];
            for (std::uint32_t pos = 0; pos < shape.state_len; ++pos) {
                const std::uint32_t value = states[sample].v[pos];
                const std::uint64_t row = static_cast<std::uint64_t>(pos) * shape.state_value_pad + value;
                sum += weights[input_table + row * shape.hd1 + h];
            }
            z1[h] = sum;
            a1[h] = Relu(sum);
        }
        for (std::uint32_t j = 0; j < shape.hd2; ++j) {
            float sum = weights[hidden_bias + j];
            for (std::uint32_t h = 0; h < shape.hd1; ++h) {
                sum += a1[h] * weights[hidden_weight + static_cast<std::uint64_t>(h) * shape.hd2 + j];
            }
            z2[j] = sum;
            a2[j] = Relu(sum);
        }
        float y = weights[output_bias];
        for (std::uint32_t j = 0; j < shape.hd2; ++j) y += a2[j] * weights[output_weight + j];
        const float diff = y - labels[sample];
        *loss += diff * diff * inv_n;
        const float dy = 2.0f * diff * inv_n;
        (*grad)[output_bias] += dy;
        for (std::uint32_t j = 0; j < shape.hd2; ++j) {
            (*grad)[output_weight + j] += a2[j] * dy;
            da2[j] = weights[output_weight + j] * dy;
            dz2[j] = da2[j] * ReluGrad(z2[j]);
        }
        std::fill(da1.begin(), da1.end(), 0.0f);
        for (std::uint32_t h = 0; h < shape.hd1; ++h) {
            for (std::uint32_t j = 0; j < shape.hd2; ++j) {
                (*grad)[hidden_weight + static_cast<std::uint64_t>(h) * shape.hd2 + j] += a1[h] * dz2[j];
                da1[h] += weights[hidden_weight + static_cast<std::uint64_t>(h) * shape.hd2 + j] * dz2[j];
            }
        }
        for (std::uint32_t j = 0; j < shape.hd2; ++j) (*grad)[hidden_bias + j] += dz2[j];
        for (std::uint32_t h = 0; h < shape.hd1; ++h) {
            dz1[h] = da1[h] * ReluGrad(z1[h]);
            (*grad)[input_bias + h] += dz1[h];
        }
        for (std::uint32_t pos = 0; pos < shape.state_len; ++pos) {
            const std::uint32_t value = states[sample].v[pos];
            const std::uint64_t row = static_cast<std::uint64_t>(pos) * shape.state_value_pad + value;
            const std::uint64_t base = input_table + row * shape.hd1;
            for (std::uint32_t h = 0; h < shape.hd1; ++h) (*grad)[base + h] += dz1[h];
        }
    }
}

}  // namespace

int main() {
    int device_count = 0;
    if (Check(cudaGetDeviceCount(&device_count)) != 0 || device_count <= 0) return EXIT_FAILURE;
    if (Check(cudaSetDevice(0)) != 0) return EXIT_FAILURE;

    const mgt_cuda::CudaMlpShape shape{4, 8, 5, 3};
    constexpr std::uint32_t kSamples = 8;
    const std::uint64_t params = ParamCount(shape);
    std::vector<float> weights(params), labels(kSamples), cpu_grad(params), gpu_grad(params);
    std::vector<mgt::TrainState80> states(kSamples);
    for (std::uint64_t i = 0; i < params; ++i) weights[i] = static_cast<float>((static_cast<int>(i % 23) - 11) * 0.004);
    for (std::uint32_t sample = 0; sample < kSamples; ++sample) {
        labels[sample] = static_cast<float>(sample % 5) * 0.25f;
        for (std::uint32_t i = 0; i < shape.state_len; ++i) states[sample].v[i] = static_cast<mgt::StateValue>((i + sample) % shape.state_value_pad);
    }
    float cpu_loss = 0.0f;
    CpuLossGrad(shape, weights, states, labels, &cpu_loss, &cpu_grad);

    float* d_weights = nullptr;
    float* d_labels = nullptr;
    float* d_loss = nullptr;
    float* d_grad = nullptr;
    mgt::TrainState80* d_states = nullptr;
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
    if (Check(cudaMemcpy(gpu_grad.data(), d_grad, params * sizeof(float), cudaMemcpyDeviceToHost)) != 0) return EXIT_FAILURE;
    if (std::fabs(gpu_loss - cpu_loss) > 1.0e-6f) return EXIT_FAILURE;
    for (std::uint64_t i = 0; i < params; ++i) {
        if (std::fabs(gpu_grad[i] - cpu_grad[i]) > 2.0e-6f) return EXIT_FAILURE;
    }

    cudaFree(d_states);
    cudaFree(d_grad);
    cudaFree(d_loss);
    cudaFree(d_labels);
    cudaFree(d_weights);
    return EXIT_SUCCESS;
}