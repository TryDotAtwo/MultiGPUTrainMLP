#include "sync_batch_norm.cuh"
#include "mgt/batch_norm.hpp"

#include <cmath>
#include <vector>

namespace {
bool Near(float actual, float expected) {
    return std::fabs(actual - expected) <= 2.0e-4f;
}
}

int main() {
    constexpr int kRows = 3;
    constexpr int kCols = 2;
    constexpr int kStride = 4;
    float x[kRows * kStride] = {1, 2, 99, 99, 3, 6, 99, 99, 8, 4, 99, 99};
    float dy[kRows * kStride] = {1, -1, 77, 77, 2, .5f, 77, 77, -.5f, 3, 77, 77};
    float gamma[kCols] = {.7f, 1.2f};
    float beta[kCols] = {.1f, -.3f};

    std::vector<float> packed_x;
    std::vector<float> packed_dy;
    for (int row = 0; row < kRows; ++row) {
        for (int col = 0; col < kCols; ++col) {
            packed_x.push_back(x[row * kStride + col]);
            packed_dy.push_back(dy[row * kStride + col]);
        }
    }
    float cpu_running_mean[kCols] = {};
    float cpu_running_var[kCols] = {1, 1};
    float cpu_y[kRows * kCols] = {};
    float cpu_dx[kRows * kCols] = {};
    float cpu_dgamma[kCols] = {};
    float cpu_dbeta[kCols] = {};
    mgt::BatchNormCache cache;
    mgt::batch_norm_forward_cpu(
        packed_x.data(), kRows, kCols, gamma, beta, cpu_running_mean, cpu_running_var,
        .1f, 1.0e-5f, true, cpu_y, &cache);
    mgt::batch_norm_backward_cpu(
        packed_dy.data(), cache, gamma, cpu_dx, cpu_dgamma, cpu_dbeta);

    float *device_x, *device_dy, *device_gamma, *device_beta;
    float *device_running_mean, *device_running_var, *device_y;
    float *device_mean, *device_inv_std, *device_normalized;
    float *device_dx, *device_dgamma, *device_dbeta, *device_workspace;
    cudaMalloc(&device_x, sizeof(x));
    cudaMalloc(&device_dy, sizeof(dy));
    cudaMalloc(&device_y, sizeof(x));
    cudaMalloc(&device_normalized, sizeof(x));
    cudaMalloc(&device_dx, sizeof(x));
    float** col_buffers[] = {
        &device_gamma, &device_beta, &device_running_mean, &device_running_var,
        &device_mean, &device_inv_std, &device_dgamma, &device_dbeta,
    };
    for (float** buffer : col_buffers) cudaMalloc(buffer, kCols * sizeof(float));
    cudaMalloc(&device_workspace, 2 * kCols * sizeof(float));
    cudaMemcpy(device_x, x, sizeof(x), cudaMemcpyHostToDevice);
    cudaMemcpy(device_dy, dy, sizeof(dy), cudaMemcpyHostToDevice);
    cudaMemcpy(device_gamma, gamma, sizeof(gamma), cudaMemcpyHostToDevice);
    cudaMemcpy(device_beta, beta, sizeof(beta), cudaMemcpyHostToDevice);
    cudaMemset(device_running_mean, 0, kCols * sizeof(float));
    float initial_running_var[kCols] = {1, 1};
    cudaMemcpy(device_running_var, initial_running_var, sizeof(initial_running_var), cudaMemcpyHostToDevice);

    mgt_cuda::NcclRankContext* context = nullptr;
    if (mgt_cuda::CreateNcclSingleRankContext(0, &context) != mgt::Status::kOk) return 1;
    if (mgt_cuda::LaunchStridedSyncBatchNormForwardFused(
            device_x, kRows, kRows, kCols, kStride, device_gamma, device_beta,
            device_running_mean, device_running_var, .1f, 1.0e-5f, device_y,
            device_mean, device_inv_std, device_normalized, device_workspace, context, 0) != mgt::Status::kOk) return 1;
    if (mgt_cuda::LaunchStridedSyncBatchNormBackwardFused(
            device_dy, kRows, kRows, kCols, kStride, device_gamma, device_inv_std,
            device_normalized, device_dx, device_dgamma, device_dbeta,
            device_workspace, context, 0) != mgt::Status::kOk) return 1;
    if (cudaDeviceSynchronize() != cudaSuccess) return 1;

    float gpu_y[kRows * kStride] = {};
    float gpu_dx[kRows * kStride] = {};
    float gpu_dgamma[kCols] = {};
    float gpu_dbeta[kCols] = {};
    float gpu_running_mean[kCols] = {};
    float gpu_running_var[kCols] = {};
    cudaMemcpy(gpu_y, device_y, sizeof(gpu_y), cudaMemcpyDeviceToHost);
    cudaMemcpy(gpu_dx, device_dx, sizeof(gpu_dx), cudaMemcpyDeviceToHost);
    cudaMemcpy(gpu_dgamma, device_dgamma, sizeof(gpu_dgamma), cudaMemcpyDeviceToHost);
    cudaMemcpy(gpu_dbeta, device_dbeta, sizeof(gpu_dbeta), cudaMemcpyDeviceToHost);
    cudaMemcpy(gpu_running_mean, device_running_mean, sizeof(gpu_running_mean), cudaMemcpyDeviceToHost);
    cudaMemcpy(gpu_running_var, device_running_var, sizeof(gpu_running_var), cudaMemcpyDeviceToHost);

    for (int row = 0; row < kRows; ++row) {
        for (int col = 0; col < kCols; ++col) {
            const int strided = row * kStride + col;
            const int packed = row * kCols + col;
            if (!Near(gpu_y[strided], cpu_y[packed]) ||
                !Near(gpu_dx[strided], cpu_dx[packed])) return 1;
        }
    }
    for (int col = 0; col < kCols; ++col) {
        if (!Near(gpu_dgamma[col], cpu_dgamma[col]) ||
            !Near(gpu_dbeta[col], cpu_dbeta[col]) ||
            !Near(gpu_running_mean[col], cpu_running_mean[col]) ||
            !Near(gpu_running_var[col], cpu_running_var[col])) return 1;
    }
    return mgt_cuda::DestroyNcclRankContext(context) == mgt::Status::kOk ? 0 : 1;
}
