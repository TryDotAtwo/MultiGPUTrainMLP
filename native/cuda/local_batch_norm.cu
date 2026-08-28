#include "mgt_cuda/local_batch_norm.cuh"

namespace mgt_cuda {
namespace {

constexpr int kThreads = 256;
constexpr int kTileCols = 32;
constexpr int kTileRows = 8;
constexpr int kRowsPerBlock = 256;

__global__ void ForwardByFeature(
    const float* x, int rows, int cols, int stride,
    const float* gamma, const float* beta,
    float* running_mean, float* running_var,
    float momentum, float epsilon,
    float* y, float* mean, float* inv_std, float* normalized,
    float* workspace) {
    const int col = blockIdx.x;
    __shared__ float sums[kThreads];
    __shared__ float squares[kThreads];
    __shared__ float block_mean;
    __shared__ float block_inv_std;
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
        const float current_mean = sums[0] / rows;
        const float variance = fmaxf(squares[0] / rows - current_mean * current_mean, 0.0f);
        const float current_inv_std = rsqrtf(variance + epsilon);
        block_mean = current_mean;
        block_inv_std = current_inv_std;
        mean[col] = current_mean;
        inv_std[col] = current_inv_std;
        workspace[col] = sums[0];
        workspace[cols + col] = squares[0];
        running_mean[col] =
            (1.0f - momentum) * running_mean[col] + momentum * current_mean;
        const float unbiased = rows > 1 ? variance * rows / (rows - 1) : 0.0f;
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

__global__ void ForwardPartialCoalesced(
    const float* x, int rows, int cols, int stride, float* sums, float* squares) {
    const int col = blockIdx.x * kTileCols + threadIdx.x;
    const int row_lane = threadIdx.y;
    const int row_begin = blockIdx.y * kRowsPerBlock;
    const int row_end = min(row_begin + kRowsPerBlock, rows);
    float sum = 0.0f;
    float square = 0.0f;
    if (col < cols) {
        for (int row = row_begin + row_lane; row < row_end; row += kTileRows) {
            const float value = x[row * stride + col];
            sum += value;
            square += value * value;
        }
    }
    __shared__ float sum_tile[kTileRows][kTileCols];
    __shared__ float square_tile[kTileRows][kTileCols];
    sum_tile[row_lane][threadIdx.x] = sum;
    square_tile[row_lane][threadIdx.x] = square;
    __syncthreads();
    if (row_lane == 0 && col < cols) {
        for (int lane = 1; lane < kTileRows; ++lane) {
            sum += sum_tile[lane][threadIdx.x];
            square += square_tile[lane][threadIdx.x];
        }
        atomicAdd(sums + col, sum);
        atomicAdd(squares + col, square);
    }
}

__global__ void ForwardFinalize(
    int rows, int cols, const float* gamma, const float* beta,
    float* running_mean, float* running_var, float momentum, float epsilon,
    float* mean, float* inv_std, const float* sums, const float* squares) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= cols) return;
    const float current_mean = sums[col] / rows;
    const float variance = fmaxf(squares[col] / rows - current_mean * current_mean, 0.0f);
    mean[col] = current_mean;
    inv_std[col] = rsqrtf(variance + epsilon);
    running_mean[col] = (1.0f - momentum) * running_mean[col] + momentum * current_mean;
    const float unbiased = rows > 1 ? variance * rows / (rows - 1) : 0.0f;
    running_var[col] = (1.0f - momentum) * running_var[col] + momentum * unbiased;
}

__global__ void ForwardApplyCoalesced(
    const float* x, int rows, int cols, int stride, const float* gamma,
    const float* beta, const float* mean, const float* inv_std,
    float* y, float* normalized) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= rows * stride) return;
    const int col = index % stride;
    if (col < cols) {
        const float value = (x[index] - mean[col]) * inv_std[col];
        normalized[index] = value;
        y[index] = value * gamma[col] + beta[col];
    } else {
        normalized[index] = 0.0f;
        y[index] = 0.0f;
    }
}

__global__ void BackwardByFeature(
    const float* dy, int rows, int cols, int stride,
    const float* gamma, const float* inv_std, const float* normalized,
    float* dx, float* dgamma, float* dbeta, float* workspace) {
    const int col = blockIdx.x;
    __shared__ float gamma_sums[kThreads];
    __shared__ float beta_sums[kThreads];
    __shared__ float gamma_grad;
    __shared__ float beta_grad;
    float local_gamma = 0.0f;
    float local_beta = 0.0f;
    for (int row = threadIdx.x; row < rows; row += kThreads) {
        const int index = row * stride + col;
        local_beta += dy[index];
        local_gamma += dy[index] * normalized[index];
    }
    gamma_sums[threadIdx.x] = local_gamma;
    beta_sums[threadIdx.x] = local_beta;
    __syncthreads();
    for (int delta = kThreads / 2; delta != 0; delta >>= 1) {
        if (threadIdx.x < delta) {
            gamma_sums[threadIdx.x] += gamma_sums[threadIdx.x + delta];
            beta_sums[threadIdx.x] += beta_sums[threadIdx.x + delta];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        gamma_grad = gamma_sums[0];
        beta_grad = beta_sums[0];
        dgamma[col] = gamma_grad;
        dbeta[col] = beta_grad;
        workspace[col] = gamma_grad;
        workspace[cols + col] = beta_grad;
    }
    __syncthreads();
    const float scale = gamma[col] * inv_std[col] / rows;
    for (int row = threadIdx.x; row < rows; row += kThreads) {
        const int index = row * stride + col;
        dx[index] = scale *
            (rows * dy[index] - beta_grad - normalized[index] * gamma_grad);
    }
}

__global__ void BackwardPartialCoalesced(
    const float* dy, int rows, int cols, int stride, const float* normalized,
    float* dgamma, float* dbeta) {
    const int col = blockIdx.x * kTileCols + threadIdx.x;
    const int row_lane = threadIdx.y;
    const int row_begin = blockIdx.y * kRowsPerBlock;
    const int row_end = min(row_begin + kRowsPerBlock, rows);
    float gamma_sum = 0.0f;
    float beta_sum = 0.0f;
    if (col < cols) {
        for (int row = row_begin + row_lane; row < row_end; row += kTileRows) {
            const int index = row * stride + col;
            beta_sum += dy[index];
            gamma_sum += dy[index] * normalized[index];
        }
    }
    __shared__ float gamma_tile[kTileRows][kTileCols];
    __shared__ float beta_tile[kTileRows][kTileCols];
    gamma_tile[row_lane][threadIdx.x] = gamma_sum;
    beta_tile[row_lane][threadIdx.x] = beta_sum;
    __syncthreads();
    if (row_lane == 0 && col < cols) {
        for (int lane = 1; lane < kTileRows; ++lane) {
            gamma_sum += gamma_tile[lane][threadIdx.x];
            beta_sum += beta_tile[lane][threadIdx.x];
        }
        atomicAdd(dgamma + col, gamma_sum);
        atomicAdd(dbeta + col, beta_sum);
    }
}

__global__ void BackwardApplyCoalesced(
    const float* dy, int rows, int cols, int stride, const float* gamma,
    const float* inv_std, const float* normalized, const float* dgamma,
    const float* dbeta, float* dx) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= rows * stride) return;
    const int col = index % stride;
    if (col < cols) {
        const float scale = gamma[col] * inv_std[col] / rows;
        dx[index] = scale *
            (rows * dy[index] - dbeta[col] - normalized[index] * dgamma[col]);
    } else {
        dx[index] = 0.0f;
    }
}

__global__ void ZeroPadding(float* values, int rows, int cols, int stride) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int padding = stride - cols;
    if (index < rows * padding) {
        const int row = index / padding;
        const int lane = index - row * padding;
        values[row * stride + cols + lane] = 0.0f;
    }
}

bool ValidCommon(int rows, int cols, int stride) {
    return rows > 0 && cols > 0 && stride >= cols;
}

}  // namespace

mgt::Status LaunchLocalStridedBatchNormForward(
    const float* x, int rows, int cols, int row_stride,
    const float* gamma, const float* beta,
    float* running_mean, float* running_var,
    float momentum, float epsilon,
    float* y, float* mean, float* inv_std, float* normalized,
    float* stats_workspace, cudaStream_t stream) {
    if (!ValidCommon(rows, cols, row_stride) || !x || !gamma || !beta ||
        !running_mean || !running_var || !y || !mean || !inv_std ||
        !normalized || !stats_workspace || momentum < 0.0f || momentum > 1.0f ||
        epsilon <= 0.0f) return mgt::Status::kInvalidConfig;
    const auto stats_bytes = static_cast<std::size_t>(2) * cols * sizeof(float);
    if (cudaMemsetAsync(stats_workspace, 0, stats_bytes, stream) != cudaSuccess)
        return mgt::Status::kCudaFailure;
    const dim3 block(kTileCols, kTileRows);
    const dim3 grid((cols + kTileCols - 1) / kTileCols,
                    (rows + kRowsPerBlock - 1) / kRowsPerBlock);
    ForwardPartialCoalesced<<<grid, block, 0, stream>>>(
        x, rows, cols, row_stride, stats_workspace, stats_workspace + cols);
    ForwardFinalize<<<(cols + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
        rows, cols, gamma, beta, running_mean, running_var, momentum, epsilon,
        mean, inv_std, stats_workspace, stats_workspace + cols);
    const int count = rows * row_stride;
    ForwardApplyCoalesced<<<(count + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
        x, rows, cols, row_stride, gamma, beta, mean, inv_std, y, normalized);
    return cudaPeekAtLastError() == cudaSuccess
        ? mgt::Status::kOk : mgt::Status::kCudaFailure;
}

mgt::Status LaunchLocalStridedBatchNormBackward(
    const float* dy, int rows, int cols, int row_stride,
    const float* gamma, const float* inv_std, const float* normalized,
    float* dx, float* dgamma, float* dbeta,
    float* stats_workspace, cudaStream_t stream) {
    if (!ValidCommon(rows, cols, row_stride) || !dy || !gamma || !inv_std ||
        !normalized || !dx || !dgamma || !dbeta || !stats_workspace)
        return mgt::Status::kInvalidConfig;
    const auto stats_bytes = static_cast<std::size_t>(2) * cols * sizeof(float);
    if (cudaMemsetAsync(stats_workspace, 0, stats_bytes, stream) != cudaSuccess)
        return mgt::Status::kCudaFailure;
    const dim3 block(kTileCols, kTileRows);
    const dim3 grid((cols + kTileCols - 1) / kTileCols,
                    (rows + kRowsPerBlock - 1) / kRowsPerBlock);
    BackwardPartialCoalesced<<<grid, block, 0, stream>>>(
        dy, rows, cols, row_stride, normalized, stats_workspace,
        stats_workspace + cols);
    const int count = rows * row_stride;
    BackwardApplyCoalesced<<<(count + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
        dy, rows, cols, row_stride, gamma, inv_std, normalized,
        stats_workspace, stats_workspace + cols, dx);
    if (cudaMemcpyAsync(dgamma, stats_workspace, cols * sizeof(float),
                        cudaMemcpyDeviceToDevice, stream) != cudaSuccess ||
        cudaMemcpyAsync(dbeta, stats_workspace + cols, cols * sizeof(float),
                        cudaMemcpyDeviceToDevice, stream) != cudaSuccess)
        return mgt::Status::kCudaFailure;
    return cudaPeekAtLastError() == cudaSuccess
        ? mgt::Status::kOk : mgt::Status::kCudaFailure;
}

}  // namespace mgt_cuda
