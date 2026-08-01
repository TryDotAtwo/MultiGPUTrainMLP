#include "mgt_cuda/adamw.cuh"
#include "mgt_cuda/device_context.cuh"
#include <cuda_runtime.h>

namespace mgt_cuda {
namespace {

__device__ float AdamWUpdateOne(AdamWKernelConfig config,
                               float weight,
                               float grad_value,
                               float* m_value,
                               float* v_value) {
    const float next_m = config.beta1 * *m_value + (1.0f - config.beta1) * grad_value;
    const float next_v = config.beta2 * *v_value + (1.0f - config.beta2) * grad_value * grad_value;
    *m_value = next_m;
    *v_value = next_v;
    const float bias1 = 1.0f - powf(config.beta1, static_cast<float>(config.step));
    const float bias2 = 1.0f - powf(config.beta2, static_cast<float>(config.step));
    const float m_hat = next_m / bias1;
    const float v_hat = next_v / bias2;
    return weight - config.learning_rate * (m_hat / (sqrtf(v_hat) + config.eps) + config.weight_decay * weight);
}

__global__ void AdamWKernel(AdamWKernelConfig config,
                            float* weights,
                            const float* grad,
                            float* m,
                            float* v) {
    const std::uint64_t i = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= config.param_count) return;
    weights[i] = AdamWUpdateOne(config, weights[i], grad[i], m + i, v + i);
}

__global__ void AdamWWithHalfMirrorKernel(AdamWKernelConfig config,
                                          float* weights,
                                          __half* weights_half,
                                          const float* grad,
                                          float* m,
                                          float* v) {
    const std::uint64_t i = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= config.param_count) return;
    const float next_weight = AdamWUpdateOne(config, weights[i], grad[i], m + i, v + i);
    weights[i] = next_weight;
    weights_half[i] = __float2half_rn(next_weight);
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


__host__ mgt::Status LaunchAdamWKernelWithHalfMirror(const AdamWKernelConfig& config,
                                                     float* device_weights,
                                                     __half* device_weights_half,
                                                     const float* device_grad,
                                                     float* device_m,
                                                     float* device_v,
                                                     cudaStream_t stream) {
    if (ValidateAdamWKernelConfig(config) != mgt::Status::kOk ||
        device_weights == nullptr || device_weights_half == nullptr || device_grad == nullptr || device_m == nullptr || device_v == nullptr) {
        return mgt::Status::kInvalidConfig;
    }
    const DeviceLaunchConfig launch = Build1DLaunchConfig(config.param_count, 256);
    AdamWWithHalfMirrorKernel<<<launch.blocks, launch.threads, 0, stream>>>(config, device_weights, device_weights_half, device_grad, device_m, device_v);
    return cudaGetLastError() == cudaSuccess ? mgt::Status::kOk : mgt::Status::kCudaFailure;
}
}  // namespace mgt_cuda
namespace mgt_cuda {
namespace {
__global__ void FloatToBfloat16MirrorKernel(const float* master,__nv_bfloat16* mirror,std::uint64_t count){const std::uint64_t i=static_cast<std::uint64_t>(blockIdx.x)*blockDim.x+threadIdx.x;if(i<count)mirror[i]=__float2bfloat16_rn(master[i]);}
__global__ void AdamWWithBfloat16MirrorKernel(AdamWKernelConfig config,float* master,__nv_bfloat16* mirror,const float* grad,float* moment1,float* moment2){const std::uint64_t i=static_cast<std::uint64_t>(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=config.param_count)return;const float next=AdamWUpdateOne(config,master[i],grad[i],moment1+i,moment2+i);master[i]=next;mirror[i]=__float2bfloat16_rn(next);}
}
mgt::Status LaunchFloatToBfloat16Mirror(const float* master,__nv_bfloat16* mirror,std::uint64_t count,cudaStream_t stream){if(!master||!mirror||!count)return mgt::Status::kInvalidConfig;const auto launch=Build1DLaunchConfig(count,256);FloatToBfloat16MirrorKernel<<<launch.blocks,launch.threads,0,stream>>>(master,mirror,count);return cudaPeekAtLastError()==cudaSuccess?mgt::Status::kOk:mgt::Status::kCudaFailure;}
mgt::Status LaunchAdamWKernelWithBfloat16Mirror(const AdamWKernelConfig& config,float* master,__nv_bfloat16* mirror,const float* grad,float* moment1,float* moment2,cudaStream_t stream){if(ValidateAdamWKernelConfig(config)!=mgt::Status::kOk||!master||!mirror||!grad||!moment1||!moment2)return mgt::Status::kInvalidConfig;const auto launch=Build1DLaunchConfig(config.param_count,256);AdamWWithBfloat16MirrorKernel<<<launch.blocks,launch.threads,0,stream>>>(config,master,mirror,grad,moment1,moment2);return cudaPeekAtLastError()==cudaSuccess?mgt::Status::kOk:mgt::Status::kCudaFailure;}
}  // namespace mgt_cuda
