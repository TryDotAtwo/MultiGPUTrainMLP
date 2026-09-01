#include "mgt_cuda/adamw.cuh"
#include "mgt_cuda/device_context.cuh"
#include <cuda_runtime.h>

#include <limits>

namespace mgt_cuda {
namespace {

__device__ __forceinline__ std::uint64_t AdamWPhysicalIndex(
    const AdamWKernelConfig& config, std::uint64_t logical) {
    if (!config.sparse_active_bins) return logical;
    const std::uint64_t active_elements =
        static_cast<std::uint64_t>(config.sparse_active_bin_count) *
        config.sparse_row_width;
    if (logical >= active_elements)
        return config.sparse_full_prefix_count + logical - active_elements;
    const unsigned active_index =
        static_cast<unsigned>(logical / config.sparse_row_width);
    const unsigned column = static_cast<unsigned>(
        logical - static_cast<std::uint64_t>(active_index) *
                      config.sparse_row_width);
    return static_cast<std::uint64_t>(
               config.sparse_active_bins[active_index]) *
               config.sparse_row_width +
           column;
}

__device__ float AdamWUpdateOneWithBias(AdamWKernelConfig config,
                                       float bias1,
                                       float bias2,
                                       float weight,
                                       float grad_value,
                                       float* m_value,
                                       float* v_value) {
    const float next_m = config.beta1 * *m_value + (1.0f - config.beta1) * grad_value;
    const float next_v = config.beta2 * *v_value + (1.0f - config.beta2) * grad_value * grad_value;
    *m_value = next_m;
    *v_value = next_v;
    const float m_hat = next_m / bias1;
    const float v_hat = next_v / bias2;
    return weight - config.learning_rate * (m_hat / (sqrtf(v_hat) + config.eps) + config.weight_decay * weight);
}

__device__ float AdamWUpdateOne(AdamWKernelConfig config,
                               float weight,
                               float grad_value,
                               float* m_value,
                               float* v_value) {
    const float bias1 = 1.0f - powf(config.beta1, static_cast<float>(config.step));
    const float bias2 = 1.0f - powf(config.beta2, static_cast<float>(config.step));
    return AdamWUpdateOneWithBias(
        config, bias1, bias2, weight, grad_value, m_value, v_value);
}

__global__ void AdamWKernel(AdamWKernelConfig config,
                            float* weights,
                            const float* grad,
                            float* m,
                            float* v) {
    const std::uint64_t i = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= config.param_count) return;
    const std::uint64_t physical = AdamWPhysicalIndex(config, i);
    weights[physical] = AdamWUpdateOne(
        config, weights[physical], grad[physical], m + physical, v + physical);
}

__global__ void AdamWWithHalfMirrorKernel(AdamWKernelConfig config,
                                          float* weights,
                                          __half* weights_half,
                                          const float* grad,
                                          float* m,
                                          float* v) {
    __shared__ float bias1;
    __shared__ float bias2;
    if (threadIdx.x == 0) {
        bias1 = 1.0f - powf(config.beta1, static_cast<float>(config.step));
        bias2 = 1.0f - powf(config.beta2, static_cast<float>(config.step));
    }
    __syncthreads();
    const std::uint64_t i = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= config.param_count) return;
    const std::uint64_t physical = AdamWPhysicalIndex(config, i);
    const float next_weight = AdamWUpdateOneWithBias(
        config, bias1, bias2, weights[physical], grad[physical], m + physical,
        v + physical);
    weights[physical] = next_weight;
    weights_half[physical] = __float2half_rn(next_weight);
}

}  // namespace

namespace detail {
const void* WeightAdamTrainingKernel() {
    return reinterpret_cast<const void*>(AdamWWithHalfMirrorKernel);
}
const void* AffineAdamTrainingKernel() {
    return reinterpret_cast<const void*>(AdamWKernel);
}
}  // namespace detail

__host__ mgt::Status ValidateAdamWKernelConfig(const AdamWKernelConfig& config) {
    if (config.param_count == 0 || config.step == 0 || config.learning_rate <= 0.0f ||
        config.beta1 < 0.0f || config.beta1 >= 1.0f ||
        config.beta2 < 0.0f || config.beta2 >= 1.0f ||
        config.eps <= 0.0f || config.weight_decay < 0.0f) {
        return mgt::Status::kInvalidConfig;
    }
    const bool any_sparse = config.sparse_active_bins ||
        config.sparse_full_prefix_count || config.sparse_row_width ||
        config.sparse_active_bin_count;
    if (any_sparse) {
        if (!config.sparse_active_bins || !config.sparse_full_prefix_count ||
            !config.sparse_row_width || !config.sparse_active_bin_count ||
            config.weight_decay != 0.0f ||
            config.sparse_full_prefix_count % config.sparse_row_width != 0)
            return mgt::Status::kInvalidConfig;
        const std::uint64_t full_rows =
            config.sparse_full_prefix_count / config.sparse_row_width;
        if (config.sparse_active_bin_count > full_rows ||
            config.sparse_active_bin_count >
                std::numeric_limits<std::uint64_t>::max() /
                    config.sparse_row_width)
            return mgt::Status::kInvalidConfig;
        const std::uint64_t active_elements =
            static_cast<std::uint64_t>(config.sparse_active_bin_count) *
            config.sparse_row_width;
        if (active_elements > config.param_count ||
            config.sparse_full_prefix_count >
                std::numeric_limits<std::uint64_t>::max() -
                    (config.param_count - active_elements))
            return mgt::Status::kInvalidConfig;
    }
    return mgt::Status::kOk;
}

__host__ mgt::Status QueryAdamWPhysicalParameterCount(
    const AdamWKernelConfig& config, std::uint64_t* count) {
    if (!count || ValidateAdamWKernelConfig(config) != mgt::Status::kOk)
        return mgt::Status::kInvalidConfig;
    if (!config.sparse_active_bins) {
        *count = config.param_count;
        return mgt::Status::kOk;
    }
    const std::uint64_t active_elements =
        static_cast<std::uint64_t>(config.sparse_active_bin_count) *
        config.sparse_row_width;
    *count = config.sparse_full_prefix_count +
        (config.param_count - active_elements);
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
    const DeviceLaunchConfig launch = Build1DLaunchConfig(config.param_count, 128);
    AdamWWithHalfMirrorKernel<<<launch.blocks, launch.threads, 0, stream>>>(config, device_weights, device_weights_half, device_grad, device_m, device_v);
    return cudaGetLastError() == cudaSuccess ? mgt::Status::kOk : mgt::Status::kCudaFailure;
}
}  // namespace mgt_cuda
namespace mgt_cuda {
namespace {
__global__ void FloatToBfloat16MirrorKernel(const float* master,__nv_bfloat16* mirror,std::uint64_t count){const std::uint64_t i=static_cast<std::uint64_t>(blockIdx.x)*blockDim.x+threadIdx.x;if(i<count)mirror[i]=__float2bfloat16_rn(master[i]);}
__global__ void AdamWWithBfloat16MirrorKernel(AdamWKernelConfig config,float* master,__nv_bfloat16* mirror,const float* grad,float* moment1,float* moment2){const std::uint64_t i=static_cast<std::uint64_t>(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=config.param_count)return;const std::uint64_t physical=AdamWPhysicalIndex(config,i);const float next=AdamWUpdateOne(config,master[physical],grad[physical],moment1+physical,moment2+physical);master[physical]=next;mirror[physical]=__float2bfloat16_rn(next);}
}
mgt::Status LaunchFloatToBfloat16Mirror(const float* master,__nv_bfloat16* mirror,std::uint64_t count,cudaStream_t stream){if(!master||!mirror||!count)return mgt::Status::kInvalidConfig;const auto launch=Build1DLaunchConfig(count,256);FloatToBfloat16MirrorKernel<<<launch.blocks,launch.threads,0,stream>>>(master,mirror,count);return cudaPeekAtLastError()==cudaSuccess?mgt::Status::kOk:mgt::Status::kCudaFailure;}
mgt::Status LaunchAdamWKernelWithBfloat16Mirror(const AdamWKernelConfig& config,float* master,__nv_bfloat16* mirror,const float* grad,float* moment1,float* moment2,cudaStream_t stream){if(ValidateAdamWKernelConfig(config)!=mgt::Status::kOk||!master||!mirror||!grad||!moment1||!moment2)return mgt::Status::kInvalidConfig;const auto launch=Build1DLaunchConfig(config.param_count,256);AdamWWithBfloat16MirrorKernel<<<launch.blocks,launch.threads,0,stream>>>(config,master,mirror,grad,moment1,moment2);return cudaPeekAtLastError()==cudaSuccess?mgt::Status::kOk:mgt::Status::kCudaFailure;}
}  // namespace mgt_cuda
