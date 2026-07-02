#include "mgt_cuda/mlp_backward.cuh"
#include <cuda_runtime.h>

namespace mgt_cuda {
namespace {

__device__ float Relu(float x) { return x > 0.0f ? x : 0.0f; }
__device__ float ReluGrad(float x) { return x > 0.0f ? 1.0f : 0.0f; }

__global__ void MlpLossGradKernel(CudaMlpShape shape,
                                  const float* weights,
                                  const mgt::TrainState80* states,
                                  const float* labels,
                                  std::uint32_t sample_count,
                                  float* loss,
                                  float* grad) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    const std::uint64_t input_table = 0;
    const std::uint64_t input_bias = input_table + static_cast<std::uint64_t>(shape.state_len) * shape.state_value_pad * shape.hd1;
    const std::uint64_t hidden_weight = input_bias + shape.hd1;
    const std::uint64_t hidden_bias = hidden_weight + static_cast<std::uint64_t>(shape.hd1) * shape.hd2;
    const std::uint64_t output_weight = hidden_bias + shape.hd2;
    const std::uint64_t output_bias = output_weight + shape.hd2;
    const std::uint64_t param_count = output_bias + 1;
    for (std::uint64_t i = 0; i < param_count; ++i) grad[i] = 0.0f;
    *loss = 0.0f;

    float z1[64];
    float a1[64];
    float z2[64];
    float a2[64];
    float dz1[64];
    float da1[64];
    float dz2[64];
    float da2[64];
    if (shape.hd1 > 64 || shape.hd2 > 64) return;
    const float inv_n = 1.0f / static_cast<float>(sample_count);

    for (std::uint32_t sample = 0; sample < sample_count; ++sample) {
        const mgt::TrainState80 state = states[sample];
        for (std::uint32_t h = 0; h < shape.hd1; ++h) {
            float sum = weights[input_bias + h];
            for (std::uint32_t pos = 0; pos < shape.state_len; ++pos) {
                const std::uint32_t value = state.v[pos];
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

        grad[output_bias] += dy;
        for (std::uint32_t j = 0; j < shape.hd2; ++j) {
            grad[output_weight + j] += a2[j] * dy;
            da2[j] = weights[output_weight + j] * dy;
            dz2[j] = da2[j] * ReluGrad(z2[j]);
        }
        for (std::uint32_t h = 0; h < shape.hd1; ++h) da1[h] = 0.0f;
        for (std::uint32_t h = 0; h < shape.hd1; ++h) {
            for (std::uint32_t j = 0; j < shape.hd2; ++j) {
                grad[hidden_weight + static_cast<std::uint64_t>(h) * shape.hd2 + j] += a1[h] * dz2[j];
                da1[h] += weights[hidden_weight + static_cast<std::uint64_t>(h) * shape.hd2 + j] * dz2[j];
            }
        }
        for (std::uint32_t j = 0; j < shape.hd2; ++j) grad[hidden_bias + j] += dz2[j];
        for (std::uint32_t h = 0; h < shape.hd1; ++h) {
            dz1[h] = da1[h] * ReluGrad(z1[h]);
            grad[input_bias + h] += dz1[h];
        }
        for (std::uint32_t pos = 0; pos < shape.state_len; ++pos) {
            const std::uint32_t value = state.v[pos];
            const std::uint64_t row = static_cast<std::uint64_t>(pos) * shape.state_value_pad + value;
            const std::uint64_t base = input_table + row * shape.hd1;
            for (std::uint32_t h = 0; h < shape.hd1; ++h) grad[base + h] += dz1[h];
        }
    }
}

}  // namespace

__host__ mgt::Status LaunchMlpLossGradKernel(const CudaMlpShape& shape,
                                             const float* device_weights,
                                             const mgt::TrainState80* device_states,
                                             const float* device_labels,
                                             std::uint32_t sample_count,
                                             float* device_loss,
                                             float* device_grad,
                                             cudaStream_t stream) {
    if (ValidateCudaMlpShape(shape) != mgt::Status::kOk || device_weights == nullptr ||
        device_states == nullptr || device_labels == nullptr || device_loss == nullptr ||
        device_grad == nullptr || sample_count == 0) {
        return mgt::Status::kInvalidConfig;
    }
    MlpLossGradKernel<<<1, 1, 0, stream>>>(shape, device_weights, device_states, device_labels, sample_count, device_loss, device_grad);
    return cudaGetLastError() == cudaSuccess ? mgt::Status::kOk : mgt::Status::kCudaFailure;
}

}  // namespace mgt_cuda