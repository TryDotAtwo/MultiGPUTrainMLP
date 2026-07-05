#include "mgt_cuda/mlp_forward.cuh"
#include "mgt_cuda/device_context.cuh"

namespace mgt_cuda {
namespace {

constexpr std::uint32_t kMaxHd1 = 4096;
constexpr std::uint32_t kMaxHd2 = 512;
constexpr std::uint32_t kMaxResidualBlocks = 32;

__device__ float Relu(float x) { return x > 0.0f ? x : 0.0f; }

__device__ std::uint64_t ResidualBlockParams(CudaMlpShape shape) {
    return 2ULL * (static_cast<std::uint64_t>(shape.hd2) * shape.hd2 + shape.hd2);
}

__device__ std::uint64_t InputBias(CudaMlpShape shape) {
    return static_cast<std::uint64_t>(shape.state_len) * shape.state_value_pad * shape.hd1;
}

__device__ std::uint64_t HiddenWeight(CudaMlpShape shape) { return InputBias(shape) + shape.hd1; }
__device__ std::uint64_t HiddenBias(CudaMlpShape shape) { return HiddenWeight(shape) + static_cast<std::uint64_t>(shape.hd1) * shape.hd2; }
__device__ std::uint64_t ResidualBase(CudaMlpShape shape) { return HiddenBias(shape) + shape.hd2; }
__device__ std::uint64_t OutputWeight(CudaMlpShape shape) { return ResidualBase(shape) + static_cast<std::uint64_t>(shape.residual_blocks) * ResidualBlockParams(shape); }
__device__ std::uint64_t OutputBias(CudaMlpShape shape) { return OutputWeight(shape) + static_cast<std::uint64_t>(shape.hd2) * shape.output_dim; }
__device__ std::uint64_t ResidualFc1Weight(CudaMlpShape shape, std::uint32_t block) { return ResidualBase(shape) + static_cast<std::uint64_t>(block) * ResidualBlockParams(shape); }
__device__ std::uint64_t ResidualFc1Bias(CudaMlpShape shape, std::uint32_t block) { return ResidualFc1Weight(shape, block) + static_cast<std::uint64_t>(shape.hd2) * shape.hd2; }
__device__ std::uint64_t ResidualFc2Weight(CudaMlpShape shape, std::uint32_t block) { return ResidualFc1Bias(shape, block) + shape.hd2; }
__device__ std::uint64_t ResidualFc2Bias(CudaMlpShape shape, std::uint32_t block) { return ResidualFc2Weight(shape, block) + static_cast<std::uint64_t>(shape.hd2) * shape.hd2; }

__global__ void MlpForwardKernel(CudaMlpShape shape,
                                 const float* weights,
                                 const mgt::TrainStateStorage* states,
                                 std::uint32_t sample_count,
                                 float* outputs) {
    const std::uint32_t sample = blockIdx.x * blockDim.x + threadIdx.x;
    if (sample >= sample_count) return;
    if (shape.hd1 > kMaxHd1 || shape.hd2 > kMaxHd2 || shape.residual_blocks > kMaxResidualBlocks) return;

    float a1[kMaxHd1];
    float cur[kMaxHd2];
    float tmp[kMaxHd2];
    const mgt::TrainStateStorage state = states[sample];
    const std::uint64_t input_bias = InputBias(shape);
    const std::uint64_t hidden_weight = HiddenWeight(shape);
    const std::uint64_t hidden_bias = HiddenBias(shape);

    for (std::uint32_t h = 0; h < shape.hd1; ++h) {
        float sum = weights[input_bias + h];
        for (std::uint32_t pos = 0; pos < shape.state_len; ++pos) {
            const std::uint32_t value = state.v[pos];
            const std::uint64_t row = static_cast<std::uint64_t>(pos) * shape.state_value_pad + value;
            sum += weights[row * shape.hd1 + h];
        }
        a1[h] = Relu(sum);
    }
    for (std::uint32_t j = 0; j < shape.hd2; ++j) {
        float sum = weights[hidden_bias + j];
        for (std::uint32_t h = 0; h < shape.hd1; ++h) sum += a1[h] * weights[hidden_weight + static_cast<std::uint64_t>(h) * shape.hd2 + j];
        cur[j] = Relu(sum);
    }
    for (std::uint32_t block = 0; block < shape.residual_blocks; ++block) {
        const std::uint64_t fc1w = ResidualFc1Weight(shape, block);
        const std::uint64_t fc1b = ResidualFc1Bias(shape, block);
        const std::uint64_t fc2w = ResidualFc2Weight(shape, block);
        const std::uint64_t fc2b = ResidualFc2Bias(shape, block);
        for (std::uint32_t j = 0; j < shape.hd2; ++j) {
            float sum = weights[fc1b + j];
            for (std::uint32_t i = 0; i < shape.hd2; ++i) sum += cur[i] * weights[fc1w + static_cast<std::uint64_t>(i) * shape.hd2 + j];
            tmp[j] = Relu(sum);
        }
        for (std::uint32_t j = 0; j < shape.hd2; ++j) {
            float sum = weights[fc2b + j];
            for (std::uint32_t i = 0; i < shape.hd2; ++i) sum += tmp[i] * weights[fc2w + static_cast<std::uint64_t>(i) * shape.hd2 + j];
            cur[j] = Relu(cur[j] + sum);
        }
    }
    const std::uint64_t output_weight = OutputWeight(shape);
    const std::uint64_t output_bias = OutputBias(shape);
    for (std::uint32_t out = 0; out < shape.output_dim; ++out) {
        float y = weights[output_bias + out];
        for (std::uint32_t j = 0; j < shape.hd2; ++j) y += cur[j] * weights[output_weight + static_cast<std::uint64_t>(j) * shape.output_dim + out];
        outputs[static_cast<std::uint64_t>(sample) * shape.output_dim + out] = y;
    }
}

}  // namespace

__host__ mgt::Status ValidateCudaMlpShape(const CudaMlpShape& shape) {
    if (shape.state_len == 0 || shape.state_len > mgt::kStateLen ||
        shape.state_value_pad == 0 || shape.state_value_pad > mgt::kStateValuePad ||
        shape.hd1 == 0 || shape.hd1 > kMaxHd1 || shape.hd2 == 0 || shape.hd2 > kMaxHd2 ||
        shape.residual_blocks > kMaxResidualBlocks || shape.output_dim == 0) {
        return mgt::Status::kInvalidConfig;
    }
    return mgt::Status::kOk;
}

__host__ mgt::Status LaunchMlpForwardKernel(const CudaMlpShape& shape,
                                            const float* device_weights,
                                            const mgt::TrainStateStorage* device_states,
                                            std::uint32_t sample_count,
                                            float* device_outputs,
                                            cudaStream_t stream) {
    if (ValidateCudaMlpShape(shape) != mgt::Status::kOk || device_weights == nullptr || device_states == nullptr || device_outputs == nullptr || sample_count == 0) return mgt::Status::kInvalidConfig;
    const DeviceLaunchConfig launch = Build1DLaunchConfig(sample_count, 128);
    MlpForwardKernel<<<launch.blocks, launch.threads, 0, stream>>>(shape, device_weights, device_states, sample_count, device_outputs);
    return cudaGetLastError() == cudaSuccess ? mgt::Status::kOk : mgt::Status::kCudaFailure;
}

}  // namespace mgt_cuda