#include "mgt/batch_norm.hpp"
#include "mgt_cuda/local_batch_norm.cuh"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdlib>
#include <vector>

namespace {
bool Near(float actual, float expected, float tolerance = 2.0e-4f) {
    return std::fabs(actual - expected) <= tolerance;
}
}  // namespace

int main() {
    constexpr int kRows = 4;
    constexpr int kCols = 3;
    constexpr int kStride = 4;
    constexpr float kPadding = 91.0f;
    const float x[kRows * kStride] = {
        1.0f, 2.0f, -1.0f, kPadding,
        3.0f, 6.0f,  2.0f, kPadding,
        8.0f, 4.0f,  0.0f, kPadding,
       -2.0f, 5.0f,  7.0f, kPadding,
    };
    const float dy[kRows * kStride] = {
         1.0f, -1.0f,  0.25f, kPadding,
         2.0f,  0.5f, -0.75f, kPadding,
        -0.5f,  3.0f,  1.25f, kPadding,
         0.2f, -2.0f,  0.5f, kPadding,
    };
    const float gamma[kCols] = {0.7f, 1.2f, -0.4f};
    const float beta[kCols] = {0.1f, -0.3f, 0.2f};

    std::vector<float> packed_x;
    std::vector<float> packed_dy;
    for (int row = 0; row < kRows; ++row) {
        for (int col = 0; col < kCols; ++col) {
            packed_x.push_back(x[row * kStride + col]);
            packed_dy.push_back(dy[row * kStride + col]);
        }
    }
    float expected_running_mean[kCols] = {};
    float expected_running_var[kCols] = {1.0f, 1.0f, 1.0f};
    const float initial_running_var[kCols] = {1.0f, 1.0f, 1.0f};
    float expected_y[kRows * kCols] = {};
    float expected_dx[kRows * kCols] = {};
    float expected_dgamma[kCols] = {};
    float expected_dbeta[kCols] = {};
    mgt::BatchNormCache cache;
    mgt::batch_norm_forward_cpu(
        packed_x.data(), kRows, kCols, gamma, beta, expected_running_mean,
        expected_running_var, 0.1f, 1.0e-5f, true, expected_y, &cache);
    mgt::batch_norm_backward_cpu(
        packed_dy.data(), cache, gamma, expected_dx, expected_dgamma,
        expected_dbeta);

    float *d_x = nullptr, *d_dy = nullptr, *d_gamma = nullptr, *d_beta = nullptr;
    float *d_running_mean = nullptr, *d_running_var = nullptr, *d_y = nullptr;
    float *d_mean = nullptr, *d_inv_std = nullptr, *d_normalized = nullptr;
    float *d_dx = nullptr, *d_dgamma = nullptr, *d_dbeta = nullptr;
    float* d_workspace = nullptr;
    if (cudaMalloc(&d_x, sizeof(x)) != cudaSuccess ||
        cudaMalloc(&d_dy, sizeof(dy)) != cudaSuccess ||
        cudaMalloc(&d_gamma, sizeof(gamma)) != cudaSuccess ||
        cudaMalloc(&d_beta, sizeof(beta)) != cudaSuccess ||
        cudaMalloc(&d_running_mean, kCols * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&d_running_var, kCols * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&d_y, sizeof(x)) != cudaSuccess ||
        cudaMalloc(&d_mean, kCols * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&d_inv_std, kCols * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&d_normalized, sizeof(x)) != cudaSuccess ||
        cudaMalloc(&d_dx, sizeof(x)) != cudaSuccess ||
        cudaMalloc(&d_dgamma, kCols * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&d_dbeta, kCols * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&d_workspace, 2 * kCols * sizeof(float)) != cudaSuccess) return 1;

    cudaMemcpy(d_x, x, sizeof(x), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dy, dy, sizeof(dy), cudaMemcpyHostToDevice);
    cudaMemcpy(d_gamma, gamma, sizeof(gamma), cudaMemcpyHostToDevice);
    cudaMemcpy(d_beta, beta, sizeof(beta), cudaMemcpyHostToDevice);
    cudaMemset(d_running_mean, 0, kCols * sizeof(float));
    cudaMemcpy(d_running_var, initial_running_var, sizeof(initial_running_var), cudaMemcpyHostToDevice);
    cudaMemset(d_y, 0x7f, sizeof(x));
    cudaMemset(d_normalized, 0x7f, sizeof(x));
    cudaMemset(d_dx, 0x7f, sizeof(x));

    if (mgt_cuda::LaunchLocalStridedBatchNormForward(
            d_x, kRows, kCols, kStride, d_gamma, d_beta, d_running_mean,
            d_running_var, 0.1f, 1.0e-5f, d_y, d_mean, d_inv_std,
            d_normalized, d_workspace, nullptr) != mgt::Status::kOk ||
        mgt_cuda::LaunchLocalStridedBatchNormBackward(
            d_dy, kRows, kCols, kStride, d_gamma, d_inv_std, d_normalized,
            d_dx, d_dgamma, d_dbeta, d_workspace, nullptr) != mgt::Status::kOk ||
        cudaDeviceSynchronize() != cudaSuccess) return 2;

    float y[kRows * kStride], dx[kRows * kStride], running_mean[kCols];
    float running_var[kCols], dgamma[kCols], dbeta[kCols];
    cudaMemcpy(y, d_y, sizeof(y), cudaMemcpyDeviceToHost);
    cudaMemcpy(dx, d_dx, sizeof(dx), cudaMemcpyDeviceToHost);
    cudaMemcpy(running_mean, d_running_mean, sizeof(running_mean), cudaMemcpyDeviceToHost);
    cudaMemcpy(running_var, d_running_var, sizeof(running_var), cudaMemcpyDeviceToHost);
    cudaMemcpy(dgamma, d_dgamma, sizeof(dgamma), cudaMemcpyDeviceToHost);
    cudaMemcpy(dbeta, d_dbeta, sizeof(dbeta), cudaMemcpyDeviceToHost);
    for (int row = 0; row < kRows; ++row) {
        for (int col = 0; col < kCols; ++col) {
            const int strided = row * kStride + col;
            const int packed = row * kCols + col;
            if (!Near(y[strided], expected_y[packed]) ||
                !Near(dx[strided], expected_dx[packed])) return 3;
        }
        if (y[row * kStride + kCols] != 0.0f ||
            dx[row * kStride + kCols] != 0.0f) return 4;
    }
    for (int col = 0; col < kCols; ++col) {
        if (!Near(running_mean[col], expected_running_mean[col]) ||
            !Near(running_var[col], expected_running_var[col]) ||
            !Near(dgamma[col], expected_dgamma[col]) ||
            !Near(dbeta[col], expected_dbeta[col])) return 5;
    }
    return 0;
}
