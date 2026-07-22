#include "mgt_cuda/adamw.cuh"

#include <cuda_runtime.h>
#include <cmath>
#include <cstdlib>
#include <vector>

namespace {

int Check(cudaError_t status) {
    return status == cudaSuccess ? 0 : 1;
}

void CpuAdamWStepLocal(float* weights,
                       const float* grad,
                       float* m,
                       float* v,
                       std::uint64_t param_count,
                       std::uint64_t step,
                       float lr,
                       float beta1,
                       float beta2,
                       float eps,
                       float weight_decay) {
    const float bias1 = 1.0f - std::pow(beta1, static_cast<float>(step));
    const float bias2 = 1.0f - std::pow(beta2, static_cast<float>(step));
    for (std::uint64_t i = 0; i < param_count; ++i) {
        m[i] = beta1 * m[i] + (1.0f - beta1) * grad[i];
        v[i] = beta2 * v[i] + (1.0f - beta2) * grad[i] * grad[i];
        const float m_hat = m[i] / bias1;
        const float v_hat = v[i] / bias2;
        weights[i] -= lr * (m_hat / (std::sqrt(v_hat) + eps) + weight_decay * weights[i]);
    }
}
}  // namespace

int main() {
    int device_count = 0;
    if (Check(cudaGetDeviceCount(&device_count)) != 0 || device_count <= 0) return EXIT_FAILURE;
    if (Check(cudaSetDevice(0)) != 0) return EXIT_FAILURE;

    constexpr std::uint64_t kParams = 1024;
    std::vector<float> weights(kParams), grad(kParams), cpu_weights(kParams), cpu_m(kParams), cpu_v(kParams);
    for (std::uint64_t i = 0; i < kParams; ++i) {
        weights[i] = static_cast<float>((static_cast<int>(i % 17) - 8) * 0.01);
        grad[i] = static_cast<float>((static_cast<int>(i % 13) - 6) * 0.001);
    }
    cpu_weights = weights;
    CpuAdamWStepLocal(cpu_weights.data(), grad.data(), cpu_m.data(), cpu_v.data(), kParams,
                       1, 0.001f, 0.9f, 0.999f, 1.0e-8f, 0.01f);

    float* d_weights = nullptr;
    float* d_grad = nullptr;
    float* d_m = nullptr;
    float* d_v = nullptr;
    if (Check(cudaMalloc(&d_weights, kParams * sizeof(float))) != 0) return EXIT_FAILURE;
    if (Check(cudaMalloc(&d_grad, kParams * sizeof(float))) != 0) return EXIT_FAILURE;
    if (Check(cudaMalloc(&d_m, kParams * sizeof(float))) != 0) return EXIT_FAILURE;
    if (Check(cudaMalloc(&d_v, kParams * sizeof(float))) != 0) return EXIT_FAILURE;
    if (Check(cudaMemcpy(d_weights, weights.data(), kParams * sizeof(float), cudaMemcpyHostToDevice)) != 0) return EXIT_FAILURE;
    if (Check(cudaMemcpy(d_grad, grad.data(), kParams * sizeof(float), cudaMemcpyHostToDevice)) != 0) return EXIT_FAILURE;
    if (Check(cudaMemset(d_m, 0, kParams * sizeof(float))) != 0) return EXIT_FAILURE;
    if (Check(cudaMemset(d_v, 0, kParams * sizeof(float))) != 0) return EXIT_FAILURE;

    const mgt_cuda::AdamWKernelConfig config{kParams, 1, 0.001f, 0.9f, 0.999f, 1.0e-8f, 0.01f};
    if (mgt_cuda::LaunchAdamWKernel(config, d_weights, d_grad, d_m, d_v, 0) != mgt::Status::kOk) {
        return EXIT_FAILURE;
    }
    if (Check(cudaDeviceSynchronize()) != 0) return EXIT_FAILURE;

    std::vector<float> gpu_weights(kParams), gpu_m(kParams), gpu_v(kParams);
    if (Check(cudaMemcpy(gpu_weights.data(), d_weights, kParams * sizeof(float), cudaMemcpyDeviceToHost)) != 0) return EXIT_FAILURE;
    if (Check(cudaMemcpy(gpu_m.data(), d_m, kParams * sizeof(float), cudaMemcpyDeviceToHost)) != 0) return EXIT_FAILURE;
    if (Check(cudaMemcpy(gpu_v.data(), d_v, kParams * sizeof(float), cudaMemcpyDeviceToHost)) != 0) return EXIT_FAILURE;

    for (std::uint64_t i = 0; i < kParams; ++i) {
        if (std::fabs(gpu_weights[i] - cpu_weights[i]) > 2.0e-6f) return EXIT_FAILURE;
        if (std::fabs(gpu_m[i] - cpu_m[i]) > 2.0e-8f) return EXIT_FAILURE;
        if (std::fabs(gpu_v[i] - cpu_v[i]) > 2.0e-10f) return EXIT_FAILURE;
    }

    cudaFree(d_v);
    cudaFree(d_m);
    cudaFree(d_grad);
    cudaFree(d_weights);
    return EXIT_SUCCESS;
}