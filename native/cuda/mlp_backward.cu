#include "mgt_cuda/mlp_backward.cuh"
#include "mgt_cuda/device_context.cuh"
#include <cuda_runtime.h>

namespace mgt_cuda {
namespace {

__device__ float Relu(float x) { return x > 0.0f ? x : 0.0f; }
__device__ float ReluGradFromActivation(float x) { return x > 0.0f ? 1.0f : 0.0f; }
__device__ float ReluGradFromPreactivation(float x) { return x > 0.0f ? 1.0f : 0.0f; }

__host__ __device__ std::uint64_t ResidualBlockParams(CudaMlpShape shape) { return 2ULL * (static_cast<std::uint64_t>(shape.hd2) * shape.hd2 + shape.hd2); }
__host__ __device__ std::uint64_t InputBias(CudaMlpShape shape) { return static_cast<std::uint64_t>(shape.state_len) * shape.state_value_pad * shape.hd1; }
__host__ __device__ std::uint64_t HiddenWeight(CudaMlpShape shape) { return InputBias(shape) + shape.hd1; }
__host__ __device__ std::uint64_t HiddenBias(CudaMlpShape shape) { return HiddenWeight(shape) + static_cast<std::uint64_t>(shape.hd1) * shape.hd2; }
__host__ __device__ std::uint64_t ResidualBase(CudaMlpShape shape) { return HiddenBias(shape) + shape.hd2; }
__host__ __device__ std::uint64_t OutputWeight(CudaMlpShape shape) { return ResidualBase(shape) + static_cast<std::uint64_t>(shape.residual_blocks) * ResidualBlockParams(shape); }
__host__ __device__ std::uint64_t OutputBias(CudaMlpShape shape) { return OutputWeight(shape) + shape.hd2; }
__host__ __device__ std::uint64_t ParamCount(CudaMlpShape shape) { return OutputBias(shape) + 1; }
__host__ __device__ std::uint64_t ResidualFc1Weight(CudaMlpShape shape, std::uint32_t block) { return ResidualBase(shape) + static_cast<std::uint64_t>(block) * ResidualBlockParams(shape); }
__host__ __device__ std::uint64_t ResidualFc1Bias(CudaMlpShape shape, std::uint32_t block) { return ResidualFc1Weight(shape, block) + static_cast<std::uint64_t>(shape.hd2) * shape.hd2; }
__host__ __device__ std::uint64_t ResidualFc2Weight(CudaMlpShape shape, std::uint32_t block) { return ResidualFc1Bias(shape, block) + shape.hd2; }
__host__ __device__ std::uint64_t ResidualFc2Bias(CudaMlpShape shape, std::uint32_t block) { return ResidualFc2Weight(shape, block) + static_cast<std::uint64_t>(shape.hd2) * shape.hd2; }

struct Workspace {
    float* a1;
    float* z2;
    float* block_inputs;
    float* rz1;
    float* ra1;
    float* rz2;
    float* dy;
    float* dcur;
    float* dprev;
    float* dz2;
    float* da1;
    float* dz1;
    float* dzfc2;
    float* dra1;
    float* dzfc1;
};

std::uint64_t WorkspaceFloats(CudaMlpShape shape, std::uint32_t samples) {
    const std::uint64_t b = samples;
    const std::uint64_t h1 = shape.hd1;
    const std::uint64_t h2 = shape.hd2;
    const std::uint64_t r = shape.residual_blocks;
    return b * h1 +                 // a1
           b * h2 +                 // z2
           b * (r + 1ULL) * h2 +    // block_inputs
           b * r * h2 +             // rz1
           b * r * h2 +             // ra1
           b * r * h2 +             // rz2
           b +                      // dy
           b * h2 +                 // dcur
           b * h2 +                 // dprev
           b * h2 +                 // dz2
           b * h1 +                 // da1
           b * h1 +                 // dz1
           b * h2 +                 // dzfc2
           b * h2 +                 // dra1
           b * h2;                  // dzfc1
}

Workspace MakeWorkspace(float* base, CudaMlpShape shape, std::uint32_t samples) {
    Workspace w{};
    const std::uint64_t b = samples;
    const std::uint64_t h1 = shape.hd1;
    const std::uint64_t h2 = shape.hd2;
    const std::uint64_t r = shape.residual_blocks;
    w.a1 = base; base += b * h1;
    w.z2 = base; base += b * h2;
    w.block_inputs = base; base += b * (r + 1ULL) * h2;
    w.rz1 = base; base += b * r * h2;
    w.ra1 = base; base += b * r * h2;
    w.rz2 = base; base += b * r * h2;
    w.dy = base; base += b;
    w.dcur = base; base += b * h2;
    w.dprev = base; base += b * h2;
    w.dz2 = base; base += b * h2;
    w.da1 = base; base += b * h1;
    w.dz1 = base; base += b * h1;
    w.dzfc2 = base; base += b * h2;
    w.dra1 = base; base += b * h2;
    w.dzfc1 = base;
    return w;
}

__global__ void InputForwardKernel(CudaMlpShape shape, const float* weights, const mgt::TrainState80* states, std::uint32_t samples, float* a1) {
    const std::uint64_t item = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::uint64_t total = static_cast<std::uint64_t>(samples) * shape.hd1;
    if (item >= total) return;
    const std::uint32_t b = static_cast<std::uint32_t>(item / shape.hd1);
    const std::uint32_t h = static_cast<std::uint32_t>(item - static_cast<std::uint64_t>(b) * shape.hd1);
    float sum = weights[InputBias(shape) + h];
    const mgt::TrainState80 state = states[b];
    for (std::uint32_t pos = 0; pos < shape.state_len; ++pos) {
        const std::uint32_t value = state.v[pos];
        const std::uint64_t row = static_cast<std::uint64_t>(pos) * shape.state_value_pad + value;
        sum += weights[row * shape.hd1 + h];
    }
    a1[item] = Relu(sum);
}

__global__ void HiddenForwardKernel(CudaMlpShape shape, const float* weights, const float* a1, std::uint32_t samples, float* z2, float* block0) {
    const std::uint64_t item = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::uint64_t total = static_cast<std::uint64_t>(samples) * shape.hd2;
    if (item >= total) return;
    const std::uint32_t b = static_cast<std::uint32_t>(item / shape.hd2);
    const std::uint32_t j = static_cast<std::uint32_t>(item - static_cast<std::uint64_t>(b) * shape.hd2);
    float sum = weights[HiddenBias(shape) + j];
    const std::uint64_t hidden_weight = HiddenWeight(shape);
    for (std::uint32_t h = 0; h < shape.hd1; ++h) sum += a1[static_cast<std::uint64_t>(b) * shape.hd1 + h] * weights[hidden_weight + static_cast<std::uint64_t>(h) * shape.hd2 + j];
    z2[item] = sum;
    block0[item] = Relu(sum);
}

__global__ void ResidualFc1ForwardKernel(CudaMlpShape shape, const float* weights, std::uint32_t samples, std::uint32_t block, const float* block_inputs, float* rz1, float* ra1) {
    const std::uint64_t item = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::uint64_t total = static_cast<std::uint64_t>(samples) * shape.hd2;
    if (item >= total) return;
    const std::uint32_t b = static_cast<std::uint32_t>(item / shape.hd2);
    const std::uint32_t j = static_cast<std::uint32_t>(item - static_cast<std::uint64_t>(b) * shape.hd2);
    const std::uint64_t batch_h2 = static_cast<std::uint64_t>(samples) * shape.hd2;
    const float* input = block_inputs + static_cast<std::uint64_t>(block) * batch_h2 + static_cast<std::uint64_t>(b) * shape.hd2;
    const std::uint64_t fc1w = ResidualFc1Weight(shape, block);
    const std::uint64_t fc1b = ResidualFc1Bias(shape, block);
    float sum = weights[fc1b + j];
    for (std::uint32_t i = 0; i < shape.hd2; ++i) sum += input[i] * weights[fc1w + static_cast<std::uint64_t>(i) * shape.hd2 + j];
    const std::uint64_t off = static_cast<std::uint64_t>(block) * batch_h2 + item;
    rz1[off] = sum;
    ra1[off] = Relu(sum);
}

__global__ void ResidualFc2ForwardKernel(CudaMlpShape shape, const float* weights, std::uint32_t samples, std::uint32_t block, float* block_inputs, const float* ra1, float* rz2) {
    const std::uint64_t item = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::uint64_t total = static_cast<std::uint64_t>(samples) * shape.hd2;
    if (item >= total) return;
    const std::uint32_t b = static_cast<std::uint32_t>(item / shape.hd2);
    const std::uint32_t j = static_cast<std::uint32_t>(item - static_cast<std::uint64_t>(b) * shape.hd2);
    const std::uint64_t batch_h2 = static_cast<std::uint64_t>(samples) * shape.hd2;
    const float* input = block_inputs + static_cast<std::uint64_t>(block) * batch_h2 + static_cast<std::uint64_t>(b) * shape.hd2;
    const float* block_ra1 = ra1 + static_cast<std::uint64_t>(block) * batch_h2 + static_cast<std::uint64_t>(b) * shape.hd2;
    const std::uint64_t fc2w = ResidualFc2Weight(shape, block);
    const std::uint64_t fc2b = ResidualFc2Bias(shape, block);
    float sum = weights[fc2b + j];
    for (std::uint32_t i = 0; i < shape.hd2; ++i) sum += block_ra1[i] * weights[fc2w + static_cast<std::uint64_t>(i) * shape.hd2 + j];
    const std::uint64_t off = static_cast<std::uint64_t>(block) * batch_h2 + item;
    rz2[off] = sum;
    block_inputs[static_cast<std::uint64_t>(block + 1U) * batch_h2 + item] = Relu(input[j] + sum);
}
__global__ void OutputBackwardInitKernel(CudaMlpShape shape, const float* weights, const float* labels, std::uint32_t samples, const float* final_act, float* loss, float* dy, float* dcur) {
    const std::uint32_t b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= samples) return;
    const std::uint64_t output_weight = OutputWeight(shape);
    float y = weights[OutputBias(shape)];
    const float* act = final_act + static_cast<std::uint64_t>(b) * shape.hd2;
    for (std::uint32_t j = 0; j < shape.hd2; ++j) y += act[j] * weights[output_weight + j];
    const float diff = y - labels[b];
    const float inv_n = 1.0f / static_cast<float>(samples);
    const float local_dy = 2.0f * diff * inv_n;
    dy[b] = local_dy;
    atomicAdd(loss, diff * diff * inv_n);
    for (std::uint32_t j = 0; j < shape.hd2; ++j) dcur[static_cast<std::uint64_t>(b) * shape.hd2 + j] = weights[output_weight + j] * local_dy;
}

__global__ void OutputGradKernel(CudaMlpShape shape, const float* final_act, const float* dy, std::uint32_t samples, float* grad) {
    const std::uint32_t j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j > shape.hd2) return;
    if (j == shape.hd2) {
        float sum = 0.0f;
        for (std::uint32_t b = 0; b < samples; ++b) sum += dy[b];
        grad[OutputBias(shape)] = sum;
        return;
    }
    float sum = 0.0f;
    for (std::uint32_t b = 0; b < samples; ++b) sum += final_act[static_cast<std::uint64_t>(b) * shape.hd2 + j] * dy[b];
    grad[OutputWeight(shape) + j] = sum;
}

__global__ void ResidualDzFc2Kernel(CudaMlpShape shape, std::uint32_t samples, std::uint32_t block, const float* block_inputs, const float* rz2, const float* dcur, float* dzfc2) {
    const std::uint64_t item = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::uint64_t total = static_cast<std::uint64_t>(samples) * shape.hd2;
    if (item >= total) return;
    const std::uint32_t b = static_cast<std::uint32_t>(item / shape.hd2);
    const std::uint32_t i = static_cast<std::uint32_t>(item - static_cast<std::uint64_t>(b) * shape.hd2);
    const std::uint64_t batch_h2 = static_cast<std::uint64_t>(samples) * shape.hd2;
    const std::uint64_t off = static_cast<std::uint64_t>(block) * batch_h2 + item;
    const float* input = block_inputs + static_cast<std::uint64_t>(block) * batch_h2 + static_cast<std::uint64_t>(b) * shape.hd2;
    dzfc2[item] = dcur[item] * ReluGradFromPreactivation(input[i] + rz2[off]);
}

__global__ void ResidualDzFc1Kernel(CudaMlpShape shape, const float* weights, std::uint32_t samples, std::uint32_t block, const float* rz1, const float* dzfc2, float* dra1, float* dzfc1) {
    const std::uint64_t item = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::uint64_t total = static_cast<std::uint64_t>(samples) * shape.hd2;
    if (item >= total) return;
    const std::uint32_t b = static_cast<std::uint32_t>(item / shape.hd2);
    const std::uint32_t i = static_cast<std::uint32_t>(item - static_cast<std::uint64_t>(b) * shape.hd2);
    const std::uint64_t batch_h2 = static_cast<std::uint64_t>(samples) * shape.hd2;
    const std::uint64_t off = static_cast<std::uint64_t>(block) * batch_h2 + item;
    const std::uint64_t fc2w = ResidualFc2Weight(shape, block);
    float back_ra1 = 0.0f;
    for (std::uint32_t j = 0; j < shape.hd2; ++j) back_ra1 += dzfc2[static_cast<std::uint64_t>(b) * shape.hd2 + j] * weights[fc2w + static_cast<std::uint64_t>(i) * shape.hd2 + j];
    dra1[item] = back_ra1;
    dzfc1[item] = back_ra1 * ReluGradFromPreactivation(rz1[off]);
}

__global__ void ResidualDPrevKernel(CudaMlpShape shape, const float* weights, std::uint32_t samples, std::uint32_t block, const float* dzfc2, const float* dzfc1, const float* dcur, float* dprev) {
    const std::uint64_t item = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::uint64_t total = static_cast<std::uint64_t>(samples) * shape.hd2;
    if (item >= total) return;
    const std::uint32_t b = static_cast<std::uint32_t>(item / shape.hd2);
    const std::uint32_t i = static_cast<std::uint32_t>(item - static_cast<std::uint64_t>(b) * shape.hd2);
    const std::uint64_t fc1w = ResidualFc1Weight(shape, block);
    float prev = dzfc2[item];
    for (std::uint32_t j = 0; j < shape.hd2; ++j) prev += dzfc1[static_cast<std::uint64_t>(b) * shape.hd2 + j] * weights[fc1w + static_cast<std::uint64_t>(i) * shape.hd2 + j];
    dprev[item] = prev;
}
__global__ void ResidualGradKernel(CudaMlpShape shape, std::uint32_t samples, std::uint32_t block, const float* block_inputs, const float* ra1, const float* dzfc2, const float* dzfc1, float* grad) {
    const std::uint64_t fc_elems = static_cast<std::uint64_t>(shape.hd2) * shape.hd2;
    const std::uint64_t item = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (item >= 2ULL * (fc_elems + shape.hd2)) return;
    const std::uint64_t batch_h2 = static_cast<std::uint64_t>(samples) * shape.hd2;
    const bool second = item >= fc_elems + shape.hd2;
    const std::uint64_t local = second ? item - (fc_elems + shape.hd2) : item;
    const std::uint64_t weight_base = second ? ResidualFc2Weight(shape, block) : ResidualFc1Weight(shape, block);
    const std::uint64_t bias_base = second ? ResidualFc2Bias(shape, block) : ResidualFc1Bias(shape, block);
    const float* lhs = second ? (ra1 + static_cast<std::uint64_t>(block) * batch_h2) : (block_inputs + static_cast<std::uint64_t>(block) * batch_h2);
    const float* dz = second ? dzfc2 : dzfc1;
    if (local >= fc_elems) {
        const std::uint32_t j = static_cast<std::uint32_t>(local - fc_elems);
        float sum = 0.0f;
        for (std::uint32_t b = 0; b < samples; ++b) sum += dz[static_cast<std::uint64_t>(b) * shape.hd2 + j];
        grad[bias_base + j] = sum;
        return;
    }
    const std::uint32_t i = static_cast<std::uint32_t>(local / shape.hd2);
    const std::uint32_t j = static_cast<std::uint32_t>(local - static_cast<std::uint64_t>(i) * shape.hd2);
    float sum = 0.0f;
    for (std::uint32_t b = 0; b < samples; ++b) sum += lhs[static_cast<std::uint64_t>(b) * shape.hd2 + i] * dz[static_cast<std::uint64_t>(b) * shape.hd2 + j];
    grad[weight_base + static_cast<std::uint64_t>(i) * shape.hd2 + j] = sum;
}

__global__ void HiddenDz2Kernel(CudaMlpShape shape, const float* z2, const float* dcur, std::uint32_t samples, float* dz2) {
    const std::uint64_t item = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::uint64_t total = static_cast<std::uint64_t>(samples) * shape.hd2;
    if (item >= total) return;
    dz2[item] = dcur[item] * ReluGradFromPreactivation(z2[item]);
}

__global__ void HiddenBackwardPointwiseKernel(CudaMlpShape shape, const float* weights, const float* a1, const float* dz2, std::uint32_t samples, float* da1, float* dz1) {
    const std::uint64_t item = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::uint64_t total = static_cast<std::uint64_t>(samples) * shape.hd1;
    if (item >= total) return;
    const std::uint32_t b = static_cast<std::uint32_t>(item / shape.hd1);
    const std::uint32_t h = static_cast<std::uint32_t>(item - static_cast<std::uint64_t>(b) * shape.hd1);
    float sum = 0.0f;
    const std::uint64_t hidden_weight = HiddenWeight(shape);
    for (std::uint32_t j = 0; j < shape.hd2; ++j) sum += dz2[static_cast<std::uint64_t>(b) * shape.hd2 + j] * weights[hidden_weight + static_cast<std::uint64_t>(h) * shape.hd2 + j];
    da1[item] = sum;
    dz1[item] = sum * ReluGradFromActivation(a1[item]);
}
__global__ void HiddenGradKernel(CudaMlpShape shape, const float* a1, const float* dz2, std::uint32_t samples, float* grad) {
    const std::uint64_t weight_elems = static_cast<std::uint64_t>(shape.hd1) * shape.hd2;
    const std::uint64_t item = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (item >= weight_elems + shape.hd2) return;
    if (item >= weight_elems) {
        const std::uint32_t j = static_cast<std::uint32_t>(item - weight_elems);
        float sum = 0.0f;
        for (std::uint32_t b = 0; b < samples; ++b) sum += dz2[static_cast<std::uint64_t>(b) * shape.hd2 + j];
        grad[HiddenBias(shape) + j] = sum;
        return;
    }
    const std::uint32_t h = static_cast<std::uint32_t>(item / shape.hd2);
    const std::uint32_t j = static_cast<std::uint32_t>(item - static_cast<std::uint64_t>(h) * shape.hd2);
    float sum = 0.0f;
    for (std::uint32_t b = 0; b < samples; ++b) sum += a1[static_cast<std::uint64_t>(b) * shape.hd1 + h] * dz2[static_cast<std::uint64_t>(b) * shape.hd2 + j];
    grad[HiddenWeight(shape) + static_cast<std::uint64_t>(h) * shape.hd2 + j] = sum;
}

__global__ void InputGradKernel(CudaMlpShape shape, const mgt::TrainState80* states, const float* dz1, std::uint32_t samples, float* grad) {
    const std::uint64_t item = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::uint64_t total = static_cast<std::uint64_t>(samples) * shape.hd1;
    if (item >= total) return;
    const std::uint32_t b = static_cast<std::uint32_t>(item / shape.hd1);
    const std::uint32_t h = static_cast<std::uint32_t>(item - static_cast<std::uint64_t>(b) * shape.hd1);
    const float v = dz1[item];
    atomicAdd(grad + InputBias(shape) + h, v);
    const mgt::TrainState80 state = states[b];
    for (std::uint32_t pos = 0; pos < shape.state_len; ++pos) {
        const std::uint32_t value = state.v[pos];
        const std::uint64_t row = static_cast<std::uint64_t>(pos) * shape.state_value_pad + value;
        atomicAdd(grad + row * shape.hd1 + h, v);
    }
}

bool LaunchOk() { return cudaGetLastError() == cudaSuccess; }

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
    const std::uint64_t param_count = ParamCount(shape);
    const std::uint64_t workspace_floats = WorkspaceFloats(shape, sample_count);
    float* workspace_base = nullptr;
    if (cudaMalloc(&workspace_base, workspace_floats * sizeof(float)) != cudaSuccess) return mgt::Status::kCudaFailure;
    Workspace w = MakeWorkspace(workspace_base, shape, sample_count);
    if (cudaMemsetAsync(device_loss, 0, sizeof(float), stream) != cudaSuccess) { cudaFree(workspace_base); return mgt::Status::kCudaFailure; }
    if (cudaMemsetAsync(device_grad, 0, param_count * sizeof(float), stream) != cudaSuccess) { cudaFree(workspace_base); return mgt::Status::kCudaFailure; }

    const DeviceLaunchConfig h1_launch = Build1DLaunchConfig(static_cast<std::uint64_t>(sample_count) * shape.hd1, 128);
    const DeviceLaunchConfig h2_launch = Build1DLaunchConfig(static_cast<std::uint64_t>(sample_count) * shape.hd2, 128);
    InputForwardKernel<<<h1_launch.blocks, h1_launch.threads, 0, stream>>>(shape, device_weights, device_states, sample_count, w.a1);
    if (!LaunchOk()) { cudaFree(workspace_base); return mgt::Status::kCudaFailure; }
    HiddenForwardKernel<<<h2_launch.blocks, h2_launch.threads, 0, stream>>>(shape, device_weights, w.a1, sample_count, w.z2, w.block_inputs);
    if (!LaunchOk()) { cudaFree(workspace_base); return mgt::Status::kCudaFailure; }
    for (std::uint32_t block = 0; block < shape.residual_blocks; ++block) {
        ResidualFc1ForwardKernel<<<h2_launch.blocks, h2_launch.threads, 0, stream>>>(shape, device_weights, sample_count, block, w.block_inputs, w.rz1, w.ra1);
        if (!LaunchOk()) { cudaFree(workspace_base); return mgt::Status::kCudaFailure; }
        ResidualFc2ForwardKernel<<<h2_launch.blocks, h2_launch.threads, 0, stream>>>(shape, device_weights, sample_count, block, w.block_inputs, w.ra1, w.rz2);
        if (!LaunchOk()) { cudaFree(workspace_base); return mgt::Status::kCudaFailure; }
    }

    const float* final_act = w.block_inputs + static_cast<std::uint64_t>(shape.residual_blocks) * sample_count * shape.hd2;
    const DeviceLaunchConfig sample_launch = Build1DLaunchConfig(sample_count, 128);
    OutputBackwardInitKernel<<<sample_launch.blocks, sample_launch.threads, 0, stream>>>(shape, device_weights, device_labels, sample_count, final_act, device_loss, w.dy, w.dcur);
    if (!LaunchOk()) { cudaFree(workspace_base); return mgt::Status::kCudaFailure; }
    const DeviceLaunchConfig output_grad_launch = Build1DLaunchConfig(shape.hd2 + 1ULL, 128);
    OutputGradKernel<<<output_grad_launch.blocks, output_grad_launch.threads, 0, stream>>>(shape, final_act, w.dy, sample_count, device_grad);
    if (!LaunchOk()) { cudaFree(workspace_base); return mgt::Status::kCudaFailure; }

    const DeviceLaunchConfig residual_grad_launch = Build1DLaunchConfig(2ULL * (static_cast<std::uint64_t>(shape.hd2) * shape.hd2 + shape.hd2), 256);
    for (std::uint32_t rblock = shape.residual_blocks; rblock > 0; --rblock) {
        const std::uint32_t block = rblock - 1U;
        ResidualDzFc2Kernel<<<h2_launch.blocks, h2_launch.threads, 0, stream>>>(shape, sample_count, block, w.block_inputs, w.rz2, w.dcur, w.dzfc2);
        if (!LaunchOk()) { cudaFree(workspace_base); return mgt::Status::kCudaFailure; }
        ResidualDzFc1Kernel<<<h2_launch.blocks, h2_launch.threads, 0, stream>>>(shape, device_weights, sample_count, block, w.rz1, w.dzfc2, w.dra1, w.dzfc1);
        if (!LaunchOk()) { cudaFree(workspace_base); return mgt::Status::kCudaFailure; }
        ResidualDPrevKernel<<<h2_launch.blocks, h2_launch.threads, 0, stream>>>(shape, device_weights, sample_count, block, w.dzfc2, w.dzfc1, w.dcur, w.dprev);
        if (!LaunchOk()) { cudaFree(workspace_base); return mgt::Status::kCudaFailure; }
        ResidualGradKernel<<<residual_grad_launch.blocks, residual_grad_launch.threads, 0, stream>>>(shape, sample_count, block, w.block_inputs, w.ra1, w.dzfc2, w.dzfc1, device_grad);
        if (!LaunchOk()) { cudaFree(workspace_base); return mgt::Status::kCudaFailure; }
        float* tmp = w.dcur;
        w.dcur = w.dprev;
        w.dprev = tmp;
    }

    HiddenDz2Kernel<<<h2_launch.blocks, h2_launch.threads, 0, stream>>>(shape, w.z2, w.dcur, sample_count, w.dz2);
    if (!LaunchOk()) { cudaFree(workspace_base); return mgt::Status::kCudaFailure; }
    HiddenBackwardPointwiseKernel<<<h1_launch.blocks, h1_launch.threads, 0, stream>>>(shape, device_weights, w.a1, w.dz2, sample_count, w.da1, w.dz1);
    if (!LaunchOk()) { cudaFree(workspace_base); return mgt::Status::kCudaFailure; }
    const DeviceLaunchConfig hidden_grad_launch = Build1DLaunchConfig(static_cast<std::uint64_t>(shape.hd1) * shape.hd2 + shape.hd2, 256);
    HiddenGradKernel<<<hidden_grad_launch.blocks, hidden_grad_launch.threads, 0, stream>>>(shape, w.a1, w.dz2, sample_count, device_grad);
    if (!LaunchOk()) { cudaFree(workspace_base); return mgt::Status::kCudaFailure; }
    InputGradKernel<<<h1_launch.blocks, h1_launch.threads, 0, stream>>>(shape, device_states, w.dz1, sample_count, device_grad);
    if (!LaunchOk()) { cudaFree(workspace_base); return mgt::Status::kCudaFailure; }

    if (cudaFree(workspace_base) != cudaSuccess) return mgt::Status::kCudaFailure;
    return mgt::Status::kOk;
}

}  // namespace mgt_cuda
