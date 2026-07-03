#include "mgt_cuda/mlp_backward.cuh"
#include <cuda_runtime.h>

namespace mgt_cuda {
namespace {

constexpr std::uint32_t kMaxHd1 = 4096;
constexpr std::uint32_t kMaxHd2 = 512;
constexpr std::uint32_t kMaxResidualBlocks = 32;

__device__ float Relu(float x) { return x > 0.0f ? x : 0.0f; }
__device__ float ReluGrad(float x) { return x > 0.0f ? 1.0f : 0.0f; }
__device__ std::uint64_t ResidualBlockParams(CudaMlpShape shape) { return 2ULL * (static_cast<std::uint64_t>(shape.hd2) * shape.hd2 + shape.hd2); }
__device__ std::uint64_t InputBias(CudaMlpShape shape) { return static_cast<std::uint64_t>(shape.state_len) * shape.state_value_pad * shape.hd1; }
__device__ std::uint64_t HiddenWeight(CudaMlpShape shape) { return InputBias(shape) + shape.hd1; }
__device__ std::uint64_t HiddenBias(CudaMlpShape shape) { return HiddenWeight(shape) + static_cast<std::uint64_t>(shape.hd1) * shape.hd2; }
__device__ std::uint64_t ResidualBase(CudaMlpShape shape) { return HiddenBias(shape) + shape.hd2; }
__device__ std::uint64_t OutputWeight(CudaMlpShape shape) { return ResidualBase(shape) + static_cast<std::uint64_t>(shape.residual_blocks) * ResidualBlockParams(shape); }
__device__ std::uint64_t OutputBias(CudaMlpShape shape) { return OutputWeight(shape) + shape.hd2; }
__device__ std::uint64_t ParamCount(CudaMlpShape shape) { return OutputBias(shape) + 1; }
__device__ std::uint64_t ResidualFc1Weight(CudaMlpShape shape, std::uint32_t block) { return ResidualBase(shape) + static_cast<std::uint64_t>(block) * ResidualBlockParams(shape); }
__device__ std::uint64_t ResidualFc1Bias(CudaMlpShape shape, std::uint32_t block) { return ResidualFc1Weight(shape, block) + static_cast<std::uint64_t>(shape.hd2) * shape.hd2; }
__device__ std::uint64_t ResidualFc2Weight(CudaMlpShape shape, std::uint32_t block) { return ResidualFc1Bias(shape, block) + shape.hd2; }
__device__ std::uint64_t ResidualFc2Bias(CudaMlpShape shape, std::uint32_t block) { return ResidualFc2Weight(shape, block) + static_cast<std::uint64_t>(shape.hd2) * shape.hd2; }

__device__ void ForwardSample(CudaMlpShape shape,
                              const float* weights,
                              const mgt::TrainState80& state,
                              float* z1,
                              float* a1,
                              float* z2,
                              float* a2,
                              float* rz1,
                              float* ra1,
                              float* rz2,
                              float* block_inputs,
                              float* cur,
                              float* output) {
    const std::uint64_t input_bias = InputBias(shape);
    const std::uint64_t hidden_weight = HiddenWeight(shape);
    const std::uint64_t hidden_bias = HiddenBias(shape);
    for (std::uint32_t h = 0; h < shape.hd1; ++h) {
        float sum = weights[input_bias + h];
        for (std::uint32_t pos = 0; pos < shape.state_len; ++pos) {
            const std::uint32_t value = state.v[pos];
            const std::uint64_t row = static_cast<std::uint64_t>(pos) * shape.state_value_pad + value;
            sum += weights[row * shape.hd1 + h];
        }
        z1[h] = sum;
        a1[h] = Relu(sum);
    }
    for (std::uint32_t j = 0; j < shape.hd2; ++j) {
        float sum = weights[hidden_bias + j];
        for (std::uint32_t h = 0; h < shape.hd1; ++h) sum += a1[h] * weights[hidden_weight + static_cast<std::uint64_t>(h) * shape.hd2 + j];
        z2[j] = sum;
        a2[j] = Relu(sum);
        cur[j] = a2[j];
        block_inputs[j] = cur[j];
    }
    for (std::uint32_t block = 0; block < shape.residual_blocks; ++block) {
        const std::uint64_t fc1w = ResidualFc1Weight(shape, block);
        const std::uint64_t fc1b = ResidualFc1Bias(shape, block);
        const std::uint64_t fc2w = ResidualFc2Weight(shape, block);
        const std::uint64_t fc2b = ResidualFc2Bias(shape, block);
        const std::uint64_t base = static_cast<std::uint64_t>(block) * shape.hd2;
        for (std::uint32_t j = 0; j < shape.hd2; ++j) {
            float sum = weights[fc1b + j];
            for (std::uint32_t i = 0; i < shape.hd2; ++i) sum += cur[i] * weights[fc1w + static_cast<std::uint64_t>(i) * shape.hd2 + j];
            rz1[base + j] = sum;
            ra1[base + j] = Relu(sum);
        }
        for (std::uint32_t j = 0; j < shape.hd2; ++j) {
            float sum = weights[fc2b + j];
            for (std::uint32_t i = 0; i < shape.hd2; ++i) sum += ra1[base + i] * weights[fc2w + static_cast<std::uint64_t>(i) * shape.hd2 + j];
            rz2[base + j] = sum;
            cur[j] = Relu(cur[j] + sum);
            block_inputs[static_cast<std::uint64_t>(block + 1U) * shape.hd2 + j] = cur[j];
        }
    }
    float y = weights[OutputBias(shape)];
    const std::uint64_t output_weight = OutputWeight(shape);
    for (std::uint32_t j = 0; j < shape.hd2; ++j) y += cur[j] * weights[output_weight + j];
    *output = y;
}

__global__ void MlpLossGradKernel(CudaMlpShape shape,
                                  const float* weights,
                                  const mgt::TrainState80* states,
                                  const float* labels,
                                  std::uint32_t sample_count,
                                  float* loss,
                                  float* grad) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    if (shape.hd1 > kMaxHd1 || shape.hd2 > kMaxHd2 || shape.residual_blocks > kMaxResidualBlocks) return;
    const std::uint64_t param_count = ParamCount(shape);
    for (std::uint64_t i = 0; i < param_count; ++i) grad[i] = 0.0f;
    *loss = 0.0f;

    float z1[kMaxHd1], a1[kMaxHd1], dz1[kMaxHd1], da1[kMaxHd1];
    float z2[kMaxHd2], a2[kMaxHd2], dz2[kMaxHd2], da2[kMaxHd2];
    float rz1[kMaxResidualBlocks * kMaxHd2], ra1[kMaxResidualBlocks * kMaxHd2], rz2[kMaxResidualBlocks * kMaxHd2];
    float block_inputs[(kMaxResidualBlocks + 1U) * kMaxHd2];
    float cur[kMaxHd2], dcur[kMaxHd2], dprev[kMaxHd2], dfc1[kMaxHd2], dzfc2[kMaxHd2], dzfc1[kMaxHd2];
    const float inv_n = 1.0f / static_cast<float>(sample_count);
    const std::uint64_t hidden_weight = HiddenWeight(shape);
    const std::uint64_t hidden_bias = HiddenBias(shape);
    const std::uint64_t output_weight = OutputWeight(shape);
    const std::uint64_t output_bias = OutputBias(shape);
    const std::uint64_t input_bias = InputBias(shape);

    for (std::uint32_t sample = 0; sample < sample_count; ++sample) {
        float output = 0.0f;
        ForwardSample(shape, weights, states[sample], z1, a1, z2, a2, rz1, ra1, rz2, block_inputs, cur, &output);
        const float diff = output - labels[sample];
        *loss += diff * diff * inv_n;
        const float dy = 2.0f * diff * inv_n;
        grad[output_bias] += dy;
        for (std::uint32_t j = 0; j < shape.hd2; ++j) {
            grad[output_weight + j] += cur[j] * dy;
            dcur[j] = weights[output_weight + j] * dy;
        }
        for (std::uint32_t rblock = shape.residual_blocks; rblock > 0; --rblock) {
            const std::uint32_t block = rblock - 1U;
            const std::uint64_t base = static_cast<std::uint64_t>(block) * shape.hd2;
            const float* input = block_inputs + base;
            const std::uint64_t fc1w = ResidualFc1Weight(shape, block);
            const std::uint64_t fc1b = ResidualFc1Bias(shape, block);
            const std::uint64_t fc2w = ResidualFc2Weight(shape, block);
            const std::uint64_t fc2b = ResidualFc2Bias(shape, block);
            for (std::uint32_t i = 0; i < shape.hd2; ++i) { dprev[i] = 0.0f; dfc1[i] = 0.0f; }
            for (std::uint32_t j = 0; j < shape.hd2; ++j) dzfc2[j] = dcur[j] * ReluGrad(input[j] + rz2[base + j]);
            for (std::uint32_t j = 0; j < shape.hd2; ++j) grad[fc2b + j] += dzfc2[j];
            for (std::uint32_t i = 0; i < shape.hd2; ++i) {
                for (std::uint32_t j = 0; j < shape.hd2; ++j) {
                    grad[fc2w + static_cast<std::uint64_t>(i) * shape.hd2 + j] += ra1[base + i] * dzfc2[j];
                    dfc1[i] += weights[fc2w + static_cast<std::uint64_t>(i) * shape.hd2 + j] * dzfc2[j];
                }
            }
            for (std::uint32_t j = 0; j < shape.hd2; ++j) dzfc1[j] = dfc1[j] * ReluGrad(rz1[base + j]);
            for (std::uint32_t j = 0; j < shape.hd2; ++j) grad[fc1b + j] += dzfc1[j];
            for (std::uint32_t i = 0; i < shape.hd2; ++i) {
                dprev[i] += dzfc2[i];
                for (std::uint32_t j = 0; j < shape.hd2; ++j) {
                    grad[fc1w + static_cast<std::uint64_t>(i) * shape.hd2 + j] += input[i] * dzfc1[j];
                    dprev[i] += weights[fc1w + static_cast<std::uint64_t>(i) * shape.hd2 + j] * dzfc1[j];
                }
            }
            for (std::uint32_t i = 0; i < shape.hd2; ++i) dcur[i] = dprev[i];
        }
        for (std::uint32_t j = 0; j < shape.hd2; ++j) {
            da2[j] = dcur[j];
            dz2[j] = da2[j] * ReluGrad(z2[j]);
            grad[hidden_bias + j] += dz2[j];
        }
        for (std::uint32_t h = 0; h < shape.hd1; ++h) da1[h] = 0.0f;
        for (std::uint32_t h = 0; h < shape.hd1; ++h) {
            for (std::uint32_t j = 0; j < shape.hd2; ++j) {
                grad[hidden_weight + static_cast<std::uint64_t>(h) * shape.hd2 + j] += a1[h] * dz2[j];
                da1[h] += weights[hidden_weight + static_cast<std::uint64_t>(h) * shape.hd2 + j] * dz2[j];
            }
        }
        for (std::uint32_t h = 0; h < shape.hd1; ++h) {
            dz1[h] = da1[h] * ReluGrad(z1[h]);
            grad[input_bias + h] += dz1[h];
        }
        for (std::uint32_t pos = 0; pos < shape.state_len; ++pos) {
            const std::uint32_t value = states[sample].v[pos];
            const std::uint64_t row = static_cast<std::uint64_t>(pos) * shape.state_value_pad + value;
            const std::uint64_t base = row * shape.hd1;
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
    if (ValidateCudaMlpShape(shape) != mgt::Status::kOk || device_weights == nullptr || device_states == nullptr || device_labels == nullptr || device_loss == nullptr || device_grad == nullptr || sample_count == 0) return mgt::Status::kInvalidConfig;
    MlpLossGradKernel<<<1, 1, 0, stream>>>(shape, device_weights, device_states, device_labels, sample_count, device_loss, device_grad);
    return cudaGetLastError() == cudaSuccess ? mgt::Status::kOk : mgt::Status::kCudaFailure;
}

}  // namespace mgt_cuda