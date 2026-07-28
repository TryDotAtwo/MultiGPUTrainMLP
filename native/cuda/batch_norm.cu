#include "batch_norm.cuh"

#include <cuda_runtime.h>

namespace mgt_cuda {
namespace {

constexpr int kThreads = 256;

__global__ void BatchNormStatsKernel(
    const float* x, int rows, int cols,
    float* running_mean, float* running_var,
    float momentum, float epsilon,
    float* mean, float* inv_std) {
    const int col = blockIdx.x;
    if (col >= cols) return;
    __shared__ float sums[kThreads];
    __shared__ float square_sums[kThreads];
    float local_sum = 0.0f;
    float local_square_sum = 0.0f;
    for (int row = threadIdx.x; row < rows; row += blockDim.x) {
        const float value = x[row * cols + col];
        local_sum += value;
        local_square_sum += value * value;
    }
    sums[threadIdx.x] = local_sum;
    square_sums[threadIdx.x] = local_square_sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            sums[threadIdx.x] += sums[threadIdx.x + stride];
            square_sums[threadIdx.x] += square_sums[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        const float inv_rows = 1.0f / static_cast<float>(rows);
        const float batch_mean = sums[0] * inv_rows;
        const float second_moment = square_sums[0] * inv_rows;
        const float batch_var = fmaxf(second_moment - batch_mean * batch_mean, 0.0f);
        mean[col] = batch_mean;
        inv_std[col] = rsqrtf(batch_var + epsilon);
        running_mean[col] = (1.0f - momentum) * running_mean[col] + momentum * batch_mean;
        const float unbiased_var = rows > 1
            ? batch_var * static_cast<float>(rows) / static_cast<float>(rows - 1)
            : 0.0f;
        running_var[col] = (1.0f - momentum) * running_var[col] + momentum * unbiased_var;
    }
}

__global__ void BatchNormNormalizeKernel(
    const float* x, int count, int cols,
    const float* gamma, const float* beta,
    const float* mean, const float* inv_std,
    float* y, float* normalized) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += blockDim.x * gridDim.x) {
        const int col = i % cols;
        const float xhat = (x[i] - mean[col]) * inv_std[col];
        normalized[i] = xhat;
        y[i] = xhat * gamma[col] + beta[col];
    }
}

__global__ void BatchNormBackwardStatsKernel(
    const float* dy, const float* normalized,
    int rows, int cols, float* dgamma, float* dbeta) {
    const int col = blockIdx.x;
    if (col >= cols) return;
    __shared__ float gamma_sums[kThreads];
    __shared__ float beta_sums[kThreads];
    float local_gamma = 0.0f;
    float local_beta = 0.0f;
    for (int row = threadIdx.x; row < rows; row += blockDim.x) {
        const int i = row * cols + col;
        local_beta += dy[i];
        local_gamma += dy[i] * normalized[i];
    }
    gamma_sums[threadIdx.x] = local_gamma;
    beta_sums[threadIdx.x] = local_beta;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            gamma_sums[threadIdx.x] += gamma_sums[threadIdx.x + stride];
            beta_sums[threadIdx.x] += beta_sums[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        dgamma[col] = gamma_sums[0];
        dbeta[col] = beta_sums[0];
    }
}

__global__ void BatchNormDxKernel(
    const float* dy, int count, int rows, int cols,
    const float* gamma, const float* inv_std,
    const float* normalized,
    const float* dgamma, const float* dbeta,
    float* dx) {
    const float inv_rows = 1.0f / static_cast<float>(rows);
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += blockDim.x * gridDim.x) {
        const int col = i % cols;
        dx[i] = gamma[col] * inv_std[col] * inv_rows *
                (static_cast<float>(rows) * dy[i] - dbeta[col] - normalized[i] * dgamma[col]);
    }
}

bool Valid(const float* input, int rows, int cols) {
    return input != nullptr && rows > 0 && cols > 0;
}

}  // namespace

mgt::Status LaunchBatchNormForward(
    const float* x, int rows, int cols,
    const float* gamma, const float* beta,
    float* running_mean, float* running_var,
    float momentum, float epsilon,
    float* y, float* mean, float* inv_std,
    float* normalized, cudaStream_t stream) {
    if (!Valid(x, rows, cols) || gamma == nullptr || beta == nullptr ||
        running_mean == nullptr || running_var == nullptr || y == nullptr ||
        mean == nullptr || inv_std == nullptr || normalized == nullptr ||
        momentum < 0.0f || momentum > 1.0f || epsilon <= 0.0f) {
        return mgt::Status::kInvalidConfig;
    }
    BatchNormStatsKernel<<<cols, kThreads, 0, stream>>>(
        x, rows, cols, running_mean, running_var, momentum, epsilon, mean, inv_std);
    const int count = rows * cols;
    const int blocks = (count + kThreads - 1) / kThreads;
    BatchNormNormalizeKernel<<<blocks, kThreads, 0, stream>>>(
        x, count, cols, gamma, beta, mean, inv_std, y, normalized);
    return cudaPeekAtLastError() == cudaSuccess ? mgt::Status::kOk : mgt::Status::kCudaFailure;
}

mgt::Status LaunchBatchNormBackward(
    const float* dy, int rows, int cols,
    const float* gamma, const float* inv_std,
    const float* normalized,
    float* dx, float* dgamma, float* dbeta,
    cudaStream_t stream) {
    if (!Valid(dy, rows, cols) || gamma == nullptr || inv_std == nullptr ||
        normalized == nullptr || dx == nullptr || dgamma == nullptr || dbeta == nullptr) {
        return mgt::Status::kInvalidConfig;
    }
    BatchNormBackwardStatsKernel<<<cols, kThreads, 0, stream>>>(
        dy, normalized, rows, cols, dgamma, dbeta);
    const int count = rows * cols;
    const int blocks = (count + kThreads - 1) / kThreads;
    BatchNormDxKernel<<<blocks, kThreads, 0, stream>>>(
        dy, count, rows, cols, gamma, inv_std, normalized, dgamma, dbeta, dx);
    return cudaPeekAtLastError() == cudaSuccess ? mgt::Status::kOk : mgt::Status::kCudaFailure;
}

}  // namespace mgt_cuda