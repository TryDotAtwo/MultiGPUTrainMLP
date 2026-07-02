#include "mgt_cuda/adamw.cuh"
#include "mgt_cuda/device_context.cuh"
#include <cuda_runtime.h>

namespace mgt_cuda {
namespace {

__global__ void AdamWKernel(AdamWKernelConfig config,
                            float* weights,
                            const float* grad,
                            float* m,
                            float* v) {
    const std::uint64_t i = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= config.param_count) return;

    const float decayed_grad = grad[i] + config.weight_decay * weights[i];
    m[i] = config.beta1 * m[i] + (1.0f - config.beta1) * decayed_grad;
    v[i] = config.beta2 * v[i] + (1.0f - config.beta2) * decayed_grad * decayed_grad;
    const float bias1 = 1.0f - powf(config.beta1, static_cast<float>(config.step));
    const float bias2 = 1.0f - powf(config.beta2, static_cast<float>(config.step));
    const float m_hat = m[i] / bias1;
    const float v_hat = v[i] / bias2;
    weights[i] -= config.learning_rate * m_hat / (sqrtf(v_hat) + config.eps);
}

}  // namespace

__host__ mgt::Status ValidateAdamWKernelConfig(const AdamWKernelConfig& config) {
    if (config.param_count == 0 || config.step == 0 || config.learning_rate <= 0.0f ||
        config.beta1 < 0.0f || config.beta1 >= 1.0f ||
        config.beta2 < 0.0f || config.beta2 >= 1.0f ||
        config.eps <= 0.0f || config.weight_decay < 0.0f) {
        return mgt::Status::kInvalidConfig;
    }
    return mgt::Status::kOk;
}

__host__ mgt::Status LaunchAdamWKernel(const AdamWKernelConfig& config,
                                       float* device_weights,
                                       const float* device_grad,
                                       float* device_m,
                                       float* device_v,
                                       cudaStream_t stream) {
    if (ValidateAdamWKernelConfig(config) != mgt::Status::kOk ||
        device_weights == nullptr || device_grad == nullptr || device_m == nullptr || device_v == nullptr) {
        return mgt::Status::kInvalidConfig;
    }
    const DeviceLaunchConfig launch = Build1DLaunchConfig(config.param_count, 256);
    AdamWKernel<<<launch.blocks, launch.threads, 0, stream>>>(config, device_weights, device_grad, device_m, device_v);
    return cudaGetLastError() == cudaSuccess ? mgt::Status::kOk : mgt::Status::kCudaFailure;
}

}  // namespace mgt_cuda