#include "batch_norm.cuh"
#include "mgt/batch_norm.hpp"

#include <cuda_runtime.h>
#include <cmath>
#include <cstdlib>
#include <vector>

namespace {
bool Ok(cudaError_t status) { return status == cudaSuccess; }
bool Near(float a, float b, float tol = 4.0e-5f) { return std::fabs(a - b) <= tol; }
}

int main() {
    constexpr int rows = 5;
    constexpr int cols = 3;
    std::vector<float> x(rows * cols), dy(rows * cols);
    for (int i = 0; i < rows * cols; ++i) {
        x[i] = static_cast<float>((i % 7) - 3) * 0.4f;
        dy[i] = static_cast<float>((i % 5) - 2) * 0.25f;
    }
    std::vector<float> gamma = {0.7f, 1.1f, -0.4f};
    std::vector<float> beta = {0.2f, -0.3f, 0.5f};
    std::vector<float> running_mean(cols, 0.0f), running_var(cols, 1.0f);
    std::vector<float> cpu_y(rows * cols), cpu_dx(rows * cols), cpu_dgamma(cols), cpu_dbeta(cols);
    mgt::BatchNormCache cache;
    mgt::batch_norm_forward_cpu(x.data(), rows, cols, gamma.data(), beta.data(), running_mean.data(), running_var.data(), 0.1f, 1.0e-5f, true, cpu_y.data(), &cache);
    mgt::batch_norm_backward_cpu(dy.data(), cache, gamma.data(), cpu_dx.data(), cpu_dgamma.data(), cpu_dbeta.data());

    float *d_x, *d_dy, *d_gamma, *d_beta, *d_running_mean, *d_running_var;
    float *d_y, *d_mean, *d_inv_std, *d_normalized, *d_dx, *d_dgamma, *d_dbeta;
    const std::size_t matrix_bytes = x.size() * sizeof(float);
    const std::size_t vector_bytes = cols * sizeof(float);
    if (!Ok(cudaMalloc(&d_x, matrix_bytes)) || !Ok(cudaMalloc(&d_dy, matrix_bytes)) ||
        !Ok(cudaMalloc(&d_gamma, vector_bytes)) || !Ok(cudaMalloc(&d_beta, vector_bytes)) ||
        !Ok(cudaMalloc(&d_running_mean, vector_bytes)) || !Ok(cudaMalloc(&d_running_var, vector_bytes)) ||
        !Ok(cudaMalloc(&d_y, matrix_bytes)) || !Ok(cudaMalloc(&d_mean, vector_bytes)) ||
        !Ok(cudaMalloc(&d_inv_std, vector_bytes)) || !Ok(cudaMalloc(&d_normalized, matrix_bytes)) ||
        !Ok(cudaMalloc(&d_dx, matrix_bytes)) || !Ok(cudaMalloc(&d_dgamma, vector_bytes)) ||
        !Ok(cudaMalloc(&d_dbeta, vector_bytes))) return EXIT_FAILURE;
    std::vector<float> initial_mean(cols, 0.0f), initial_var(cols, 1.0f);
    cudaMemcpy(d_x, x.data(), matrix_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_dy, dy.data(), matrix_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_gamma, gamma.data(), vector_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_beta, beta.data(), vector_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_running_mean, initial_mean.data(), vector_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_running_var, initial_var.data(), vector_bytes, cudaMemcpyHostToDevice);

    if (mgt_cuda::LaunchBatchNormForward(d_x, rows, cols, d_gamma, d_beta, d_running_mean,
        d_running_var, 0.1f, 1.0e-5f, d_y, d_mean, d_inv_std, d_normalized, 0) != mgt::Status::kOk) return EXIT_FAILURE;
    if (mgt_cuda::LaunchBatchNormBackward(d_dy, rows, cols, d_gamma, d_inv_std,
        d_normalized, d_dx, d_dgamma, d_dbeta, 0) != mgt::Status::kOk) return EXIT_FAILURE;
    if (!Ok(cudaDeviceSynchronize())) return EXIT_FAILURE;

    std::vector<float> gpu_y(x.size()), gpu_dx(x.size()), gpu_dgamma(cols), gpu_dbeta(cols);
    cudaMemcpy(gpu_y.data(), d_y, matrix_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(gpu_dx.data(), d_dx, matrix_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(gpu_dgamma.data(), d_dgamma, vector_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(gpu_dbeta.data(), d_dbeta, vector_bytes, cudaMemcpyDeviceToHost);
    for (std::size_t i = 0; i < x.size(); ++i) {
        if (!Near(gpu_y[i], cpu_y[i]) || !Near(gpu_dx[i], cpu_dx[i])) return EXIT_FAILURE;
    }
    for (int c = 0; c < cols; ++c) {
        if (!Near(gpu_dgamma[c], cpu_dgamma[c]) || !Near(gpu_dbeta[c], cpu_dbeta[c])) return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}