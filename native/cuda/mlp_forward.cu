#include "mgt_cuda/mlp_forward.cuh"
#include "mgt_cuda/device_context.cuh"

namespace mgt_cuda {
namespace {

__device__ float Relu(float x) {
    return x > 0.0f ? x : 0.0f;
}

__global__ void MlpForwardKernel(CudaMlpShape shape,
                                 const float* weights,
                                 const mgt::TrainState80* states,
                                 std::uint32_t sample_count,
                                 float* outputs) {
    const std::uint32_t sample = blockIdx.x * blockDim.x + threadIdx.x;
    if (sample >= sample_count) return;

    const std::uint64_t input_table = 0;
    const std::uint64_t input_bias = input_table + static_cast<std::uint64_t>(shape.state_len) * shape.state_value_pad * shape.hd1;
    const std::uint64_t hidden_weight = input_bias + shape.hd1;
    const std::uint64_t hidden_bias = hidden_weight + static_cast<std::uint64_t>(shape.hd1) * shape.hd2;
    const std::uint64_t output_weight = hidden_bias + shape.hd2;
    const std::uint64_t output_bias = output_weight + shape.hd2;

    float a1[64];
    float a2[64];
    if (shape.hd1 > 64 || shape.hd2 > 64) return;

    const mgt::TrainState80 state = states[sample];
    for (std::uint32_t h = 0; h < shape.hd1; ++h) {
        float sum = weights[input_bias + h];
        for (std::uint32_t pos = 0; pos < shape.state_len; ++pos) {
            const std::uint32_t value = state.v[pos];
            const std::uint64_t row = static_cast<std::uint64_t>(pos) * shape.state_value_pad + value;
            sum += weights[input_table + row * shape.hd1 + h];
        }
        a1[h] = Relu(sum);
    }
    for (std::uint32_t j = 0; j < shape.hd2; ++j) {
        float sum = weights[hidden_bias + j];
        for (std::uint32_t h = 0; h < shape.hd1; ++h) {
            sum += a1[h] * weights[hidden_weight + static_cast<std::uint64_t>(h) * shape.hd2 + j];
        }
        a2[j] = Relu(sum);
    }
    float y = weights[output_bias];
    for (std::uint32_t j = 0; j < shape.hd2; ++j) {
        y += a2[j] * weights[output_weight + j];
    }
    outputs[sample] = y;
}

}  // namespace

__host__ mgt::Status ValidateCudaMlpShape(const CudaMlpShape& shape) {
    if (shape.state_len == 0 || shape.state_len > mgt::kStateLen ||
        shape.state_value_pad == 0 || shape.state_value_pad > mgt::kStateValuePad ||
        shape.hd1 == 0 || shape.hd1 > 64 || shape.hd2 == 0 || shape.hd2 > 64) {
        return mgt::Status::kInvalidConfig;
    }
    return mgt::Status::kOk;
}

__host__ mgt::Status LaunchMlpForwardKernel(const CudaMlpShape& shape,
                                            const float* device_weights,
                                            const mgt::TrainState80* device_states,
                                            std::uint32_t sample_count,
                                            float* device_outputs,
                                            cudaStream_t stream) {
    if (ValidateCudaMlpShape(shape) != mgt::Status::kOk || device_weights == nullptr ||
        device_states == nullptr || device_outputs == nullptr || sample_count == 0) {
        return mgt::Status::kInvalidConfig;
    }
    const DeviceLaunchConfig launch = Build1DLaunchConfig(sample_count, 128);
    MlpForwardKernel<<<launch.blocks, launch.threads, 0, stream>>>(shape, device_weights, device_states, sample_count, device_outputs);
    return cudaGetLastError() == cudaSuccess ? mgt::Status::kOk : mgt::Status::kCudaFailure;
}

}  // namespace mgt_cuda