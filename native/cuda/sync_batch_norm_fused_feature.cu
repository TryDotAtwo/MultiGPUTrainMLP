#include "sync_batch_norm.cuh"
#include "mgt_cuda/allreduce_nccl.cuh"

namespace mgt_cuda {
namespace {
constexpr int kThreads = 256;

__global__ void LocalMomentsByFeature(
    const float* x, int rows, int cols, int stride, float* workspace) {
    const int col = blockIdx.x;
    __shared__ float sums[kThreads];
    __shared__ float squares[kThreads];
    float sum = 0.0f;
    float square = 0.0f;
    for (int row = threadIdx.x; row < rows; row += kThreads) {
        const float value = x[row * stride + col];
        sum += value;
        square += value * value;
    }
    sums[threadIdx.x] = sum;
    squares[threadIdx.x] = square;
    __syncthreads();
    for (int delta = kThreads / 2; delta != 0; delta >>= 1) {
        if (threadIdx.x < delta) {
            sums[threadIdx.x] += sums[threadIdx.x + delta];
            squares[threadIdx.x] += squares[threadIdx.x + delta];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        workspace[col] = sums[0];
        workspace[cols + col] = squares[0];
    }
}

__global__ void ForwardEpilogueByFeature(
    const float* x, int rows, int global_rows, int cols, int stride,
    const float* gamma, const float* beta, float momentum, float epsilon,
    float* running_mean, float* running_var, float* y, float* mean,
    float* inv_std, float* normalized, const float* workspace) {
    const int col = blockIdx.x;
    __shared__ float block_mean;
    __shared__ float block_inv_std;
    if (threadIdx.x == 0) {
        const float current_mean = workspace[col] / global_rows;
        const float variance = fmaxf(
            workspace[cols + col] / global_rows - current_mean * current_mean,
            0.0f);
        block_mean = current_mean;
        block_inv_std = rsqrtf(variance + epsilon);
        mean[col] = current_mean;
        inv_std[col] = block_inv_std;
        running_mean[col] =
            (1.0f - momentum) * running_mean[col] + momentum * current_mean;
        const float unbiased =
            global_rows > 1 ? variance * global_rows / (global_rows - 1) : 0.0f;
        running_var[col] =
            (1.0f - momentum) * running_var[col] + momentum * unbiased;
    }
    __syncthreads();
    for (int row = threadIdx.x; row < rows; row += kThreads) {
        const int index = row * stride + col;
        const float value = (x[index] - block_mean) * block_inv_std;
        normalized[index] = value;
        y[index] = value * gamma[col] + beta[col];
    }
}

__global__ void LocalGradsByFeature(
    const float* dy, const float* normalized, int rows, int cols, int stride,
    float* workspace) {
    const int col = blockIdx.x;
    __shared__ float gamma_sums[kThreads];
    __shared__ float beta_sums[kThreads];
    float dgamma = 0.0f;
    float dbeta = 0.0f;
    for (int row = threadIdx.x; row < rows; row += kThreads) {
        const int index = row * stride + col;
        dbeta += dy[index];
        dgamma += dy[index] * normalized[index];
    }
    gamma_sums[threadIdx.x] = dgamma;
    beta_sums[threadIdx.x] = dbeta;
    __syncthreads();
    for (int delta = kThreads / 2; delta != 0; delta >>= 1) {
        if (threadIdx.x < delta) {
            gamma_sums[threadIdx.x] += gamma_sums[threadIdx.x + delta];
            beta_sums[threadIdx.x] += beta_sums[threadIdx.x + delta];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        workspace[col] = gamma_sums[0];
        workspace[cols + col] = beta_sums[0];
    }
}

__global__ void BackwardEpilogueByFeature(
    const float* dy, int rows, int global_rows, int cols, int stride,
    const float* gamma, const float* inv_std, const float* normalized,
    float* dx, float* dgamma, float* dbeta, const float* workspace) {
    const int col = blockIdx.x;
    const float gamma_grad = workspace[col];
    const float beta_grad = workspace[cols + col];
    if (threadIdx.x == 0) {
        dgamma[col] = gamma_grad;
        dbeta[col] = beta_grad;
    }
    const float scale = gamma[col] * inv_std[col] / global_rows;
    for (int row = threadIdx.x; row < rows; row += kThreads) {
        const int index = row * stride + col;
        dx[index] = scale *
            (global_rows * dy[index] - beta_grad -
             normalized[index] * gamma_grad);
    }
}
}  // namespace

mgt::Status LaunchStridedSyncBatchNormForwardFused(
    const float* x, int local_rows, int global_rows, int cols, int row_stride,
    const float* gamma, const float* beta, float* running_mean, float* running_var,
    float momentum, float epsilon, float* y, float* mean, float* inv_std,
    float* normalized, float* stats_workspace, NcclRankContext* context,
    cudaStream_t stream) {
    if (!x || local_rows <= 0 || global_rows < local_rows || cols <= 0 ||
        row_stride < cols || !gamma || !beta || !running_mean || !running_var ||
        !y || !mean || !inv_std || !normalized || !stats_workspace || !context ||
        epsilon <= 0.0f) return mgt::Status::kInvalidConfig;
    LocalMomentsByFeature<<<cols, kThreads, 0, stream>>>(
        x, local_rows, cols, row_stride, stats_workspace);
    auto status = NcclAllreduceSumFloat(
        stats_workspace, 2 * cols, context, stream);
    if (status != mgt::Status::kOk) return status;
    ForwardEpilogueByFeature<<<cols, kThreads, 0, stream>>>(
        x, local_rows, global_rows, cols, row_stride, gamma, beta, momentum,
        epsilon, running_mean, running_var, y, mean, inv_std, normalized,
        stats_workspace);
    return cudaPeekAtLastError() == cudaSuccess
        ? mgt::Status::kOk : mgt::Status::kCudaFailure;
}

mgt::Status LaunchStridedSyncBatchNormBackwardFused(
    const float* dy, int local_rows, int global_rows, int cols, int row_stride,
    const float* gamma, const float* inv_std, const float* normalized, float* dx,
    float* dgamma, float* dbeta, float* stats_workspace,
    NcclRankContext* context, cudaStream_t stream) {
    if (!dy || local_rows <= 0 || global_rows < local_rows || cols <= 0 ||
        row_stride < cols || !gamma || !inv_std || !normalized || !dx ||
        !dgamma || !dbeta || !stats_workspace || !context)
        return mgt::Status::kInvalidConfig;
    LocalGradsByFeature<<<cols, kThreads, 0, stream>>>(
        dy, normalized, local_rows, cols, row_stride, stats_workspace);
    auto status = NcclAllreduceSumFloat(
        stats_workspace, 2 * cols, context, stream);
    if (status != mgt::Status::kOk) return status;
    BackwardEpilogueByFeature<<<cols, kThreads, 0, stream>>>(
        dy, local_rows, global_rows, cols, row_stride, gamma, inv_std,
        normalized, dx, dgamma, dbeta, stats_workspace);
    return cudaPeekAtLastError() == cudaSuccess
        ? mgt::Status::kOk : mgt::Status::kCudaFailure;
}

}  // namespace mgt_cuda
