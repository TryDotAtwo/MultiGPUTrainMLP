#include "mgt_cuda/local_batch_norm.cuh"

#include <climits>
#include <cstdint>
#include <initializer_list>

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
    const int row_end = row_begin + min(kRowsPerBlock, rows - row_begin);
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

template <bool Relu, bool Residual, bool HalfMirror>
__global__ void ForwardApplyCoalesced(
    const float* x, int rows, int cols, int stride, const float* gamma,
    const float* beta, const float* mean, const float* inv_std,
    float* y, float* normalized, const float* residual, __half* half_output) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= rows * stride) return;
    const int col = index % stride;
    float affine = 0.0f;
    if (col < cols) {
        const float value = (x[index] - mean[col]) * inv_std[col];
        normalized[index] = value;
        affine = value * gamma[col] + beta[col];
    } else {
        normalized[index] = 0.0f;
    }
    // Match the old affine FP32 store followed by residual + affine. Explicit
    // rounding prevents contraction/reassociation across that former store.
    if constexpr (Residual) affine = __fadd_rn(residual[index], affine);
    if constexpr (Relu) affine = affine > 0.0f ? affine : 0.0f;
    y[index] = affine;
    if constexpr (HalfMirror) half_output[index] = __float2half_rn(affine);
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

template <bool Relu>
__device__ __forceinline__ float BackwardGradient(
    const float* dy, const float* activated, int index) {
    // Preserve the old select, including NaN activations and signed zero.
    // Multiplication by a boolean mask would not have the same semantics.
    if constexpr (Relu) return activated[index] > 0.0f ? dy[index] : 0.0f;
    return dy[index];
}

template <bool Relu>
__global__ void BackwardPartialCoalesced(
    const float* dy, int rows, int cols, int stride, const float* normalized,
    float* dgamma, float* dbeta, const float* activated) {
    const int col = blockIdx.x * kTileCols + threadIdx.x;
    const int row_lane = threadIdx.y;
    const int row_begin = blockIdx.y * kRowsPerBlock;
    const int row_end = row_begin + min(kRowsPerBlock, rows - row_begin);
    float gamma_sum = 0.0f;
    float beta_sum = 0.0f;
    if (col < cols) {
        for (int row = row_begin + row_lane; row < row_end; row += kTileRows) {
            const int index = row * stride + col;
            const float gradient = BackwardGradient<Relu>(dy, activated, index);
            beta_sum += gradient;
            gamma_sum += gradient * normalized[index];
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

template <bool Relu, bool Residual, bool HalfMirror>
__global__ void BackwardApplyCoalesced(
    const float* dy, int rows, int cols, int stride, const float* gamma,
    const float* inv_std, const float* normalized, const float* dgamma,
    const float* dbeta, float* dx, __half* half_output,
    const float* activated, float* residual_grad) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= rows * stride) return;
    const int col = index % stride;
    float gradient = 0.0f;
    if (col < cols || Residual)
        gradient = BackwardGradient<Relu>(dy, activated, index);
    // The residual branch needs incoming masked dY, before the BN derivative,
    // on every physical lane. Capture it before an exact in-place dX store.
    if constexpr (Residual) residual_grad[index] = gradient;
    float value = 0.0f;
    if (col < cols) {
        const float scale = gamma[col] * inv_std[col] / rows;
        value = scale *
            (rows * gradient - dbeta[col] - normalized[index] * dgamma[col]);
    }
    // Both destinations consume the same FP32 result. Keep the original
    // derivative expression and contraction policy; do not accumulate in half.
    dx[index] = value;
    if constexpr (HalfMirror) half_output[index] = __float2half_rn(value);
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
    return rows > 0 && cols > 0 && stride >= cols && rows <= INT_MAX / stride;
}

bool Overlap(const void* a, std::size_t a_bytes, const void* b, std::size_t b_bytes) {
    if (!a || !b) return false;
    const auto first = reinterpret_cast<std::uintptr_t>(a);
    const auto second = reinterpret_cast<std::uintptr_t>(b);
    return first <= second ? second - first < a_bytes : first - second < b_bytes;
}

bool ValidApply(
    const float* x, int rows, int cols, int stride, const float* gamma,
    const float* beta, const float* mean, const float* inv_std,
    float* y, float* normalized, LocalBatchNormForwardEpilogue epilogue) {
    if (!ValidCommon(rows, cols, stride) || !x || !gamma || !beta || !mean ||
        !inv_std || !y || !normalized ||
        (!epilogue.relu && (epilogue.residual || epilogue.half_output))) return false;
    const auto bytes = static_cast<std::size_t>(rows) * stride * sizeof(float);
    const auto half_bytes = bytes / sizeof(float) * sizeof(__half);
    const auto feature_bytes = static_cast<std::size_t>(cols) * sizeof(float);
    if ((x != y && Overlap(x, bytes, y, bytes)) ||
        Overlap(normalized, bytes, x, bytes) || Overlap(normalized, bytes, y, bytes) ||
        Overlap(epilogue.residual, bytes, x, bytes) ||
        Overlap(epilogue.residual, bytes, y, bytes) ||
        Overlap(epilogue.residual, bytes, normalized, bytes)) return false;
    for (const float* tensor : {x, static_cast<const float*>(y),
                               static_cast<const float*>(normalized), epilogue.residual}) {
        if (Overlap(epilogue.half_output, half_bytes, tensor, bytes)) return false;
    }
    for (const float* feature : {gamma, beta, mean, inv_std}) {
        if (Overlap(y, bytes, feature, feature_bytes) ||
            Overlap(normalized, bytes, feature, feature_bytes) ||
            Overlap(epilogue.half_output, half_bytes, feature, feature_bytes)) return false;
    }
    return true;
}

bool ValidBackwardApply(
    const float* dy, int rows, int cols, int stride, const float* gamma,
    const float* inv_std, const float* normalized, const float* dgamma,
    const float* dbeta, float* dx, LocalBatchNormBackwardEpilogue epilogue) {
    if (!ValidCommon(rows, cols, stride) || !dy || !gamma || !inv_std ||
        !normalized || !dx || !dgamma || !dbeta ||
        (epilogue.residual_grad && !epilogue.activated)) return false;
    const auto bytes = static_cast<std::size_t>(rows) * stride * sizeof(float);
    const auto half_bytes = bytes / sizeof(float) * sizeof(__half);
    const auto feature_bytes = static_cast<std::size_t>(cols) * sizeof(float);
    if ((dy != dx && Overlap(dy, bytes, dx, bytes)) ||
        Overlap(dx, bytes, normalized, bytes) ||
        Overlap(epilogue.activated, bytes, dx, bytes)) return false;
    for (const float* tensor : {dy, static_cast<const float*>(dx), normalized,
                               epilogue.activated}) {
        if (Overlap(epilogue.half_output, half_bytes, tensor, bytes) ||
            Overlap(epilogue.residual_grad, bytes, tensor, bytes)) return false;
    }
    if (Overlap(epilogue.half_output, half_bytes, epilogue.residual_grad, bytes))
        return false;
    for (const float* feature : {gamma, inv_std, dgamma, dbeta}) {
        if (Overlap(dx, bytes, feature, feature_bytes) ||
            Overlap(epilogue.half_output, half_bytes, feature, feature_bytes) ||
            Overlap(epilogue.residual_grad, bytes, feature, feature_bytes)) return false;
    }
    return true;
}

}  // namespace

mgt::Status LaunchLocalStridedBatchNormApply(
    const float* x, int rows, int cols, int row_stride,
    const float* gamma, const float* beta, const float* mean, const float* inv_std,
    float* y, float* normalized,
    LocalBatchNormForwardEpilogue epilogue, cudaStream_t stream) {
    if (!ValidApply(x, rows, cols, row_stride, gamma, beta, mean, inv_std,
                    y, normalized, epilogue)) return mgt::Status::kInvalidConfig;
    const int count = rows * row_stride;
    const int blocks = 1 + (count - 1) / kThreads;
    if (!epilogue.relu) {
        ForwardApplyCoalesced<false, false, false><<<blocks, kThreads, 0, stream>>>(
            x, rows, cols, row_stride, gamma, beta, mean, inv_std, y, normalized,
            nullptr, nullptr);
    } else if (epilogue.residual && epilogue.half_output) {
        ForwardApplyCoalesced<true, true, true><<<blocks, kThreads, 0, stream>>>(
            x, rows, cols, row_stride, gamma, beta, mean, inv_std, y, normalized,
            epilogue.residual, epilogue.half_output);
    } else if (epilogue.residual) {
        ForwardApplyCoalesced<true, true, false><<<blocks, kThreads, 0, stream>>>(
            x, rows, cols, row_stride, gamma, beta, mean, inv_std, y, normalized,
            epilogue.residual, nullptr);
    } else if (epilogue.half_output) {
        ForwardApplyCoalesced<true, false, true><<<blocks, kThreads, 0, stream>>>(
            x, rows, cols, row_stride, gamma, beta, mean, inv_std, y, normalized,
            nullptr, epilogue.half_output);
    } else {
        ForwardApplyCoalesced<true, false, false><<<blocks, kThreads, 0, stream>>>(
            x, rows, cols, row_stride, gamma, beta, mean, inv_std, y, normalized,
            nullptr, nullptr);
    }
    return cudaPeekAtLastError() == cudaSuccess
        ? mgt::Status::kOk : mgt::Status::kCudaFailure;
}

mgt::Status LaunchLocalStridedBatchNormForward(
    const float* x, int rows, int cols, int row_stride,
    const float* gamma, const float* beta,
    float* running_mean, float* running_var,
    float momentum, float epsilon,
    float* y, float* mean, float* inv_std, float* normalized,
    float* stats_workspace, cudaStream_t stream,
    LocalBatchNormForwardEpilogue epilogue) {
    if (!ValidApply(x, rows, cols, row_stride, gamma, beta, mean, inv_std,
                    y, normalized, epilogue) ||
        !running_mean || !running_var || !stats_workspace ||
        momentum < 0.0f || momentum > 1.0f ||
        epsilon <= 0.0f) return mgt::Status::kInvalidConfig;
    const auto stats_bytes = static_cast<std::size_t>(2) * cols * sizeof(float);
    const auto half_bytes = static_cast<std::size_t>(rows) * row_stride * sizeof(__half);
    if (Overlap(epilogue.half_output, half_bytes, running_mean, stats_bytes / 2) ||
        Overlap(epilogue.half_output, half_bytes, running_var, stats_bytes / 2) ||
        Overlap(epilogue.half_output, half_bytes, stats_workspace, stats_bytes))
        return mgt::Status::kInvalidConfig;
    if (cudaMemsetAsync(stats_workspace, 0, stats_bytes, stream) != cudaSuccess)
        return mgt::Status::kCudaFailure;
    const dim3 block(kTileCols, kTileRows);
    const dim3 grid(1 + (cols - 1) / kTileCols, 1 + (rows - 1) / kRowsPerBlock);
    ForwardPartialCoalesced<<<grid, block, 0, stream>>>(
        x, rows, cols, row_stride, stats_workspace, stats_workspace + cols);
    ForwardFinalize<<<1 + (cols - 1) / kThreads, kThreads, 0, stream>>>(
        rows, cols, gamma, beta, running_mean, running_var, momentum, epsilon,
        mean, inv_std, stats_workspace, stats_workspace + cols);
    return LaunchLocalStridedBatchNormApply(x, rows, cols, row_stride, gamma, beta,
        mean, inv_std, y, normalized, epilogue, stream);
}

mgt::Status LaunchLocalStridedBatchNormBackwardApply(
    const float* dy, int rows, int cols, int row_stride,
    const float* gamma, const float* inv_std, const float* normalized,
    const float* dgamma, const float* dbeta, float* dx,
    LocalBatchNormBackwardEpilogue epilogue, cudaStream_t stream) {
    if (!ValidBackwardApply(dy, rows, cols, row_stride, gamma, inv_std,
                            normalized, dgamma, dbeta, dx, epilogue))
        return mgt::Status::kInvalidConfig;
    const int count = rows * row_stride;
    if (epilogue.residual_grad && epilogue.half_output) {
        BackwardApplyCoalesced<true, true, true><<<1 + (count - 1) / kThreads, kThreads, 0, stream>>>(
            dy, rows, cols, row_stride, gamma, inv_std, normalized, dgamma, dbeta,
            dx, epilogue.half_output, epilogue.activated, epilogue.residual_grad);
    } else if (epilogue.residual_grad) {
        BackwardApplyCoalesced<true, true, false><<<1 + (count - 1) / kThreads, kThreads, 0, stream>>>(
            dy, rows, cols, row_stride, gamma, inv_std, normalized, dgamma, dbeta,
            dx, nullptr, epilogue.activated, epilogue.residual_grad);
    } else if (epilogue.activated && epilogue.half_output) {
        BackwardApplyCoalesced<true, false, true><<<1 + (count - 1) / kThreads, kThreads, 0, stream>>>(
            dy, rows, cols, row_stride, gamma, inv_std, normalized, dgamma, dbeta,
            dx, epilogue.half_output, epilogue.activated, nullptr);
    } else if (epilogue.activated) {
        BackwardApplyCoalesced<true, false, false><<<1 + (count - 1) / kThreads, kThreads, 0, stream>>>(
            dy, rows, cols, row_stride, gamma, inv_std, normalized, dgamma, dbeta,
            dx, nullptr, epilogue.activated, nullptr);
    } else if (epilogue.half_output) {
        BackwardApplyCoalesced<false, false, true><<<1 + (count - 1) / kThreads, kThreads, 0, stream>>>(
            dy, rows, cols, row_stride, gamma, inv_std, normalized, dgamma, dbeta,
            dx, epilogue.half_output, nullptr, nullptr);
    } else {
        BackwardApplyCoalesced<false, false, false><<<1 + (count - 1) / kThreads, kThreads, 0, stream>>>(
            dy, rows, cols, row_stride, gamma, inv_std, normalized, dgamma, dbeta,
            dx, nullptr, nullptr, nullptr);
    }
    return cudaPeekAtLastError() == cudaSuccess
        ? mgt::Status::kOk : mgt::Status::kCudaFailure;
}

mgt::Status LaunchLocalStridedBatchNormBackward(
    const float* dy, int rows, int cols, int row_stride,
    const float* gamma, const float* inv_std, const float* normalized,
    float* dx, float* dgamma, float* dbeta,
    float* stats_workspace, cudaStream_t stream,
    LocalBatchNormBackwardEpilogue epilogue) {
    if (!ValidBackwardApply(dy, rows, cols, row_stride, gamma, inv_std,
                            normalized, dgamma, dbeta, dx, epilogue) || !stats_workspace)
        return mgt::Status::kInvalidConfig;
    const auto stats_bytes = static_cast<std::size_t>(2) * cols * sizeof(float);
    const auto half_bytes = static_cast<std::size_t>(rows) * row_stride * sizeof(__half);
    const auto bytes = static_cast<std::size_t>(rows) * row_stride * sizeof(float);
    if (Overlap(epilogue.half_output, half_bytes, stats_workspace, stats_bytes) ||
        Overlap(epilogue.activated, bytes, stats_workspace, stats_bytes) ||
        Overlap(epilogue.activated, bytes, dgamma, stats_bytes / 2) ||
        Overlap(epilogue.activated, bytes, dbeta, stats_bytes / 2) ||
        Overlap(epilogue.residual_grad, bytes, stats_workspace, stats_bytes))
        return mgt::Status::kInvalidConfig;
    if (cudaMemsetAsync(stats_workspace, 0, stats_bytes, stream) != cudaSuccess)
        return mgt::Status::kCudaFailure;
    const dim3 block(kTileCols, kTileRows);
    const dim3 grid(1 + (cols - 1) / kTileCols, 1 + (rows - 1) / kRowsPerBlock);
    if (epilogue.activated) {
        BackwardPartialCoalesced<true><<<grid, block, 0, stream>>>(
            dy, rows, cols, row_stride, normalized, stats_workspace,
            stats_workspace + cols, epilogue.activated);
    } else {
        BackwardPartialCoalesced<false><<<grid, block, 0, stream>>>(
            dy, rows, cols, row_stride, normalized, stats_workspace,
            stats_workspace + cols, nullptr);
    }
    const auto apply_status = LaunchLocalStridedBatchNormBackwardApply(
        dy, rows, cols, row_stride, gamma, inv_std, normalized,
        stats_workspace, stats_workspace + cols, dx, epilogue, stream);
    if (apply_status != mgt::Status::kOk) return apply_status;
    if (cudaMemcpyAsync(dgamma, stats_workspace, cols * sizeof(float),
                        cudaMemcpyDeviceToDevice, stream) != cudaSuccess ||
        cudaMemcpyAsync(dbeta, stats_workspace + cols, cols * sizeof(float),
                        cudaMemcpyDeviceToDevice, stream) != cudaSuccess)
        return mgt::Status::kCudaFailure;
    return cudaPeekAtLastError() == cudaSuccess
        ? mgt::Status::kOk : mgt::Status::kCudaFailure;
}

}  // namespace mgt_cuda
