#include "mgt_cuda/mlp_backward.cuh"
#include "mgt_cuda/device_context.cuh"
#include <cublas_v2.h>
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

bool LaunchOk() { return cudaGetLastError() == cudaSuccess; }

mgt::Status GemmRowMajor(cublasHandle_t handle, const float* a, const float* b, float* c, std::uint32_t m, std::uint32_t n, std::uint32_t k, float beta = 0.0f) {
    const float alpha = 1.0f;
    const cublasStatus_t status = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                             static_cast<int>(n), static_cast<int>(m), static_cast<int>(k),
                                             &alpha, b, static_cast<int>(n), a, static_cast<int>(k),
                                             &beta, c, static_cast<int>(n));
    return status == CUBLAS_STATUS_SUCCESS ? mgt::Status::kOk : mgt::Status::kCudaFailure;
}

mgt::Status GemmGradWeights(cublasHandle_t handle, const float* x, const float* dz, float* grad_w, std::uint32_t samples, std::uint32_t in_dim, std::uint32_t out_dim) {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    const cublasStatus_t status = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T,
                                             static_cast<int>(out_dim), static_cast<int>(in_dim), static_cast<int>(samples),
                                             &alpha, dz, static_cast<int>(out_dim), x, static_cast<int>(in_dim),
                                             &beta, grad_w, static_cast<int>(out_dim));
    return status == CUBLAS_STATUS_SUCCESS ? mgt::Status::kOk : mgt::Status::kCudaFailure;
}

mgt::Status GemmBackpropInput(cublasHandle_t handle, const float* dz, const float* weights, float* dx, std::uint32_t samples, std::uint32_t in_dim, std::uint32_t out_dim, float beta = 0.0f) {
    const float alpha = 1.0f;
    const cublasStatus_t status = cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                                             static_cast<int>(in_dim), static_cast<int>(samples), static_cast<int>(out_dim),
                                             &alpha, weights, static_cast<int>(out_dim), dz, static_cast<int>(out_dim),
                                             &beta, dx, static_cast<int>(in_dim));
    return status == CUBLAS_STATUS_SUCCESS ? mgt::Status::kOk : mgt::Status::kCudaFailure;
}

__global__ void InputForwardKernel(CudaMlpShape shape, const float* weights, const mgt::TrainState80* states, std::uint32_t samples, float* a1) {
    const std::uint32_t b = blockIdx.x;
    if (b >= samples) return;
    const mgt::TrainState80 state = states[b];
    const std::uint64_t out_base = static_cast<std::uint64_t>(b) * shape.hd1;
    const std::uint64_t input_bias = InputBias(shape);
    for (std::uint32_t h = threadIdx.x; h < shape.hd1; h += blockDim.x) {
        float sum = weights[input_bias + h];
        for (std::uint32_t pos = 0; pos < shape.state_len; ++pos) {
            const std::uint32_t value = state.v[pos];
            const std::uint64_t row = static_cast<std::uint64_t>(pos) * shape.state_value_pad + value;
            sum += weights[row * shape.hd1 + h];
        }
        a1[out_base + h] = Relu(sum);
    }
}


__global__ void AddBiasReluKernel(float* x, const float* bias, std::uint32_t rows, std::uint32_t cols) {
    const std::uint64_t item = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::uint64_t total = static_cast<std::uint64_t>(rows) * cols;
    if (item >= total) return;
    const std::uint32_t col = static_cast<std::uint32_t>(item % cols);
    x[item] = Relu(x[item] + bias[col]);
}

__global__ void AddBiasReluCopyKernel(float* x, const float* bias, std::uint32_t rows, std::uint32_t cols, float* out) {
    const std::uint64_t item = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::uint64_t total = static_cast<std::uint64_t>(rows) * cols;
    if (item >= total) return;
    const std::uint32_t col = static_cast<std::uint32_t>(item % cols);
    const float v = Relu(x[item] + bias[col]);
    x[item] = v;
    out[item] = v;
}

__global__ void AddBiasResidualReluKernel(float* z, const float* bias, const float* residual, std::uint32_t rows, std::uint32_t cols, float* out) {
    const std::uint64_t item = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::uint64_t total = static_cast<std::uint64_t>(rows) * cols;
    if (item >= total) return;
    const std::uint32_t col = static_cast<std::uint32_t>(item % cols);
    z[item] += bias[col];
    out[item] = Relu(residual[item] + z[item]);
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

__global__ void BiasGradKernel(const float* dz, std::uint32_t rows, std::uint32_t cols, float* bias_grad) {
    const std::uint32_t col = blockIdx.x;
    if (col >= cols) return;
    float sum = 0.0f;
    for (std::uint32_t row = threadIdx.x; row < rows; row += blockDim.x) sum += dz[static_cast<std::uint64_t>(row) * cols + col];
    __shared__ float scratch[256];
    scratch[threadIdx.x] = sum;
    __syncthreads();
    for (std::uint32_t stride = blockDim.x >> 1U; stride > 0; stride >>= 1U) {
        if (threadIdx.x < stride) scratch[threadIdx.x] += scratch[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) bias_grad[col] = scratch[0];
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

__global__ void ResidualDzFc1Kernel(CudaMlpShape shape, std::uint32_t samples, std::uint32_t block, const float* rz1, const float* dra1, float* dzfc1) {
    const std::uint64_t item = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::uint64_t total = static_cast<std::uint64_t>(samples) * shape.hd2;
    if (item >= total) return;
    const std::uint64_t batch_h2 = static_cast<std::uint64_t>(samples) * shape.hd2;
    const std::uint64_t off = static_cast<std::uint64_t>(block) * batch_h2 + item;
    dzfc1[item] = dra1[item] * ReluGradFromPreactivation(rz1[off]);
}

__global__ void AddInPlaceKernel(float* dst, const float* src, std::uint64_t count) {
    const std::uint64_t item = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (item >= count) return;
    dst[item] += src[item];
}

__global__ void HiddenDz2Kernel(CudaMlpShape shape, const float* z2, const float* dcur, std::uint32_t samples, float* dz2) {
    const std::uint64_t item = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::uint64_t total = static_cast<std::uint64_t>(samples) * shape.hd2;
    if (item >= total) return;
    dz2[item] = dcur[item] * ReluGradFromPreactivation(z2[item]);
}

__global__ void HiddenDz1Kernel(const float* a1, const float* da1, std::uint32_t samples, std::uint32_t hd1, float* dz1) {
    const std::uint64_t item = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::uint64_t total = static_cast<std::uint64_t>(samples) * hd1;
    if (item >= total) return;
    dz1[item] = da1[item] * ReluGradFromActivation(a1[item]);
}

__global__ void InputGradKernel(CudaMlpShape shape, const mgt::TrainState80* states, const float* dz1, std::uint32_t samples, float* grad) {
    const std::uint32_t b = blockIdx.x;
    if (b >= samples) return;
    const mgt::TrainState80 state = states[b];
    const std::uint64_t dz_base = static_cast<std::uint64_t>(b) * shape.hd1;
    const std::uint64_t input_bias = InputBias(shape);
    for (std::uint32_t h = threadIdx.x; h < shape.hd1; h += blockDim.x) {
        const float v = dz1[dz_base + h];
        atomicAdd(grad + input_bias + h, v);
        for (std::uint32_t pos = 0; pos < shape.state_len; ++pos) {
            const std::uint32_t value = state.v[pos];
            const std::uint64_t row = static_cast<std::uint64_t>(pos) * shape.state_value_pad + value;
            atomicAdd(grad + row * shape.hd1 + h, v);
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
    const std::uint64_t param_count = ParamCount(shape);
    const std::uint64_t workspace_floats = WorkspaceFloats(shape, sample_count);
    float* workspace_base = nullptr;
    if (cudaMalloc(&workspace_base, workspace_floats * sizeof(float)) != cudaSuccess) return mgt::Status::kCudaFailure;

    cublasHandle_t blas = nullptr;
    if (cublasCreate(&blas) != CUBLAS_STATUS_SUCCESS) { cudaFree(workspace_base); return mgt::Status::kCudaFailure; }
    if (cublasSetStream(blas, stream) != CUBLAS_STATUS_SUCCESS) { cublasDestroy(blas); cudaFree(workspace_base); return mgt::Status::kCudaFailure; }

    Workspace w = MakeWorkspace(workspace_base, shape, sample_count);
    if (cudaMemsetAsync(device_loss, 0, sizeof(float), stream) != cudaSuccess) { cublasDestroy(blas); cudaFree(workspace_base); return mgt::Status::kCudaFailure; }
    if (cudaMemsetAsync(device_grad, 0, param_count * sizeof(float), stream) != cudaSuccess) { cublasDestroy(blas); cudaFree(workspace_base); return mgt::Status::kCudaFailure; }

    const DeviceLaunchConfig h1_launch = Build1DLaunchConfig(static_cast<std::uint64_t>(sample_count) * shape.hd1, 128);
    const DeviceLaunchConfig h2_launch = Build1DLaunchConfig(static_cast<std::uint64_t>(sample_count) * shape.hd2, 128);
    InputForwardKernel<<<sample_count, 256, 0, stream>>>(shape, device_weights, device_states, sample_count, w.a1);
    if (!LaunchOk()) { cublasDestroy(blas); cudaFree(workspace_base); return mgt::Status::kCudaFailure; }

    if (GemmRowMajor(blas, w.a1, device_weights + HiddenWeight(shape), w.z2, sample_count, shape.hd2, shape.hd1) != mgt::Status::kOk) { cublasDestroy(blas); cudaFree(workspace_base); return mgt::Status::kCudaFailure; }
    AddBiasReluCopyKernel<<<h2_launch.blocks, h2_launch.threads, 0, stream>>>(w.z2, device_weights + HiddenBias(shape), sample_count, shape.hd2, w.block_inputs);
    if (!LaunchOk()) { cublasDestroy(blas); cudaFree(workspace_base); return mgt::Status::kCudaFailure; }

    for (std::uint32_t block = 0; block < shape.residual_blocks; ++block) {
        const std::uint64_t batch_h2 = static_cast<std::uint64_t>(sample_count) * shape.hd2;
        const float* block_in = w.block_inputs + static_cast<std::uint64_t>(block) * batch_h2;
        float* block_rz1 = w.rz1 + static_cast<std::uint64_t>(block) * batch_h2;
        float* block_ra1 = w.ra1 + static_cast<std::uint64_t>(block) * batch_h2;
        float* block_rz2 = w.rz2 + static_cast<std::uint64_t>(block) * batch_h2;
        float* block_out = w.block_inputs + static_cast<std::uint64_t>(block + 1U) * batch_h2;
        if (GemmRowMajor(blas, block_in, device_weights + ResidualFc1Weight(shape, block), block_rz1, sample_count, shape.hd2, shape.hd2) != mgt::Status::kOk) { cublasDestroy(blas); cudaFree(workspace_base); return mgt::Status::kCudaFailure; }
        AddBiasReluKernel<<<h2_launch.blocks, h2_launch.threads, 0, stream>>>(block_rz1, device_weights + ResidualFc1Bias(shape, block), sample_count, shape.hd2);
        if (!LaunchOk()) { cublasDestroy(blas); cudaFree(workspace_base); return mgt::Status::kCudaFailure; }
        cudaMemcpyAsync(block_ra1, block_rz1, batch_h2 * sizeof(float), cudaMemcpyDeviceToDevice, stream);
        if (GemmRowMajor(blas, block_ra1, device_weights + ResidualFc2Weight(shape, block), block_rz2, sample_count, shape.hd2, shape.hd2) != mgt::Status::kOk) { cublasDestroy(blas); cudaFree(workspace_base); return mgt::Status::kCudaFailure; }
        AddBiasResidualReluKernel<<<h2_launch.blocks, h2_launch.threads, 0, stream>>>(block_rz2, device_weights + ResidualFc2Bias(shape, block), block_in, sample_count, shape.hd2, block_out);
        if (!LaunchOk()) { cublasDestroy(blas); cudaFree(workspace_base); return mgt::Status::kCudaFailure; }
    }

    const float* final_act = w.block_inputs + static_cast<std::uint64_t>(shape.residual_blocks) * sample_count * shape.hd2;
    const DeviceLaunchConfig sample_launch = Build1DLaunchConfig(sample_count, 128);
    OutputBackwardInitKernel<<<sample_launch.blocks, sample_launch.threads, 0, stream>>>(shape, device_weights, device_labels, sample_count, final_act, device_loss, w.dy, w.dcur);
    if (!LaunchOk()) { cublasDestroy(blas); cudaFree(workspace_base); return mgt::Status::kCudaFailure; }
    const DeviceLaunchConfig output_grad_launch = Build1DLaunchConfig(shape.hd2 + 1ULL, 128);
    OutputGradKernel<<<output_grad_launch.blocks, output_grad_launch.threads, 0, stream>>>(shape, final_act, w.dy, sample_count, device_grad);
    if (!LaunchOk()) { cublasDestroy(blas); cudaFree(workspace_base); return mgt::Status::kCudaFailure; }

    for (std::uint32_t rblock = shape.residual_blocks; rblock > 0; --rblock) {
        const std::uint32_t block = rblock - 1U;
        const std::uint64_t batch_h2 = static_cast<std::uint64_t>(sample_count) * shape.hd2;
        const float* block_in = w.block_inputs + static_cast<std::uint64_t>(block) * batch_h2;
        const float* block_ra1 = w.ra1 + static_cast<std::uint64_t>(block) * batch_h2;

        ResidualDzFc2Kernel<<<h2_launch.blocks, h2_launch.threads, 0, stream>>>(shape, sample_count, block, w.block_inputs, w.rz2, w.dcur, w.dzfc2);
        if (!LaunchOk()) { cublasDestroy(blas); cudaFree(workspace_base); return mgt::Status::kCudaFailure; }
        if (GemmGradWeights(blas, block_ra1, w.dzfc2, device_grad + ResidualFc2Weight(shape, block), sample_count, shape.hd2, shape.hd2) != mgt::Status::kOk) { cublasDestroy(blas); cudaFree(workspace_base); return mgt::Status::kCudaFailure; }
        BiasGradKernel<<<shape.hd2, 256, 0, stream>>>(w.dzfc2, sample_count, shape.hd2, device_grad + ResidualFc2Bias(shape, block));
        if (!LaunchOk()) { cublasDestroy(blas); cudaFree(workspace_base); return mgt::Status::kCudaFailure; }
        if (GemmBackpropInput(blas, w.dzfc2, device_weights + ResidualFc2Weight(shape, block), w.dra1, sample_count, shape.hd2, shape.hd2) != mgt::Status::kOk) { cublasDestroy(blas); cudaFree(workspace_base); return mgt::Status::kCudaFailure; }

        ResidualDzFc1Kernel<<<h2_launch.blocks, h2_launch.threads, 0, stream>>>(shape, sample_count, block, w.rz1, w.dra1, w.dzfc1);
        if (!LaunchOk()) { cublasDestroy(blas); cudaFree(workspace_base); return mgt::Status::kCudaFailure; }
        if (GemmGradWeights(blas, block_in, w.dzfc1, device_grad + ResidualFc1Weight(shape, block), sample_count, shape.hd2, shape.hd2) != mgt::Status::kOk) { cublasDestroy(blas); cudaFree(workspace_base); return mgt::Status::kCudaFailure; }
        BiasGradKernel<<<shape.hd2, 256, 0, stream>>>(w.dzfc1, sample_count, shape.hd2, device_grad + ResidualFc1Bias(shape, block));
        if (!LaunchOk()) { cublasDestroy(blas); cudaFree(workspace_base); return mgt::Status::kCudaFailure; }
        if (GemmBackpropInput(blas, w.dzfc1, device_weights + ResidualFc1Weight(shape, block), w.dprev, sample_count, shape.hd2, shape.hd2) != mgt::Status::kOk) { cublasDestroy(blas); cudaFree(workspace_base); return mgt::Status::kCudaFailure; }
        AddInPlaceKernel<<<h2_launch.blocks, h2_launch.threads, 0, stream>>>(w.dprev, w.dzfc2, batch_h2);
        if (!LaunchOk()) { cublasDestroy(blas); cudaFree(workspace_base); return mgt::Status::kCudaFailure; }
        float* tmp = w.dcur;
        w.dcur = w.dprev;
        w.dprev = tmp;
    }

    HiddenDz2Kernel<<<h2_launch.blocks, h2_launch.threads, 0, stream>>>(shape, w.z2, w.dcur, sample_count, w.dz2);
    if (!LaunchOk()) { cublasDestroy(blas); cudaFree(workspace_base); return mgt::Status::kCudaFailure; }
    if (GemmGradWeights(blas, w.a1, w.dz2, device_grad + HiddenWeight(shape), sample_count, shape.hd1, shape.hd2) != mgt::Status::kOk) { cublasDestroy(blas); cudaFree(workspace_base); return mgt::Status::kCudaFailure; }
    BiasGradKernel<<<shape.hd2, 256, 0, stream>>>(w.dz2, sample_count, shape.hd2, device_grad + HiddenBias(shape));
    if (!LaunchOk()) { cublasDestroy(blas); cudaFree(workspace_base); return mgt::Status::kCudaFailure; }
    if (GemmBackpropInput(blas, w.dz2, device_weights + HiddenWeight(shape), w.da1, sample_count, shape.hd1, shape.hd2) != mgt::Status::kOk) { cublasDestroy(blas); cudaFree(workspace_base); return mgt::Status::kCudaFailure; }
    HiddenDz1Kernel<<<h1_launch.blocks, h1_launch.threads, 0, stream>>>(w.a1, w.da1, sample_count, shape.hd1, w.dz1);
    if (!LaunchOk()) { cublasDestroy(blas); cudaFree(workspace_base); return mgt::Status::kCudaFailure; }
    InputGradKernel<<<sample_count, 256, 0, stream>>>(shape, device_states, w.dz1, sample_count, device_grad);
    if (!LaunchOk()) { cublasDestroy(blas); cudaFree(workspace_base); return mgt::Status::kCudaFailure; }

    if (cublasDestroy(blas) != CUBLAS_STATUS_SUCCESS) { cudaFree(workspace_base); return mgt::Status::kCudaFailure; }
    if (cudaFree(workspace_base) != cudaSuccess) return mgt::Status::kCudaFailure;
    return mgt::Status::kOk;
}

}  // namespace mgt_cuda
