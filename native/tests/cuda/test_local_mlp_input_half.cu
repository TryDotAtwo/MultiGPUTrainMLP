#include "mgt_cuda/local_mlp_batch_norm.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdlib>
#include <vector>

namespace {

__global__ void ScalarReference(
    mgt_cuda::CudaMlpShape shape, std::uint32_t logical_hd1,
    const __half* weights, const mgt::TrainStateStorage* states,
    std::uint32_t rows, float* output) {
    const std::uint64_t q = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (q >= static_cast<std::uint64_t>(rows) * shape.hd1) return;
    const std::uint32_t row = static_cast<std::uint32_t>(q / shape.hd1);
    const std::uint32_t h = static_cast<std::uint32_t>(q - static_cast<std::uint64_t>(row) * shape.hd1);
    if (h >= logical_hd1) { output[q] = 0.0f; return; }
    const std::uint64_t bias =
        static_cast<std::uint64_t>(shape.state_len) * shape.state_value_pad * shape.hd1;
    float value = __half2float(weights[bias + h]);
    for (std::uint32_t p = 0; p < shape.state_len; ++p)
        value += __half2float(weights[
            (static_cast<std::uint64_t>(p) * shape.state_value_pad + states[row].v[p]) *
            shape.hd1 + h]);
    output[q] = value;
}

bool RunCase(std::uint32_t physical_hd1, std::uint32_t logical_hd1) {
    constexpr std::uint32_t kRows = 3;
    mgt_cuda::CudaMlpShape shape{2, 4, physical_hd1, 2, 1, 1};
    const std::uint64_t embedding_count =
        static_cast<std::uint64_t>(shape.state_len) * shape.state_value_pad * shape.hd1;
    const std::uint64_t weight_count = embedding_count + shape.hd1;
    std::vector<__half> weights(weight_count);
    for (std::uint64_t i = 0; i < weight_count; ++i)
        weights[i] = __float2half(static_cast<float>(static_cast<int>(i % 17) - 8) / 16.0f);

    std::vector<mgt::TrainStateStorage> states(kRows);
    states[0].v[0] = 0; states[0].v[1] = 3;
    states[1].v[0] = 1; states[1].v[1] = 2;
    states[2].v[0] = 3; states[2].v[1] = 0;

    __half* d_weights = nullptr;
    mgt::TrainStateStorage* d_states = nullptr;
    float* d_output = nullptr;
    cudaMalloc(&d_weights, weight_count * sizeof(__half));
    cudaMalloc(&d_states, states.size() * sizeof(states[0]));
    cudaMalloc(&d_output, static_cast<std::uint64_t>(kRows) * physical_hd1 * sizeof(float));
    cudaMemcpy(d_weights, weights.data(), weight_count * sizeof(__half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_states, states.data(), states.size() * sizeof(states[0]), cudaMemcpyHostToDevice);

    const auto status = mgt_cuda::LaunchLocalMlpInputHalf(
        shape, logical_hd1, d_weights, d_states, kRows, d_output, nullptr);
    std::vector<float> output(static_cast<std::uint64_t>(kRows) * physical_hd1);
    cudaMemcpy(output.data(), d_output, output.size() * sizeof(float), cudaMemcpyDeviceToHost);
    const bool cuda_ok = cudaGetLastError() == cudaSuccess;
    cudaFree(d_output); cudaFree(d_states); cudaFree(d_weights);
    if (status != mgt::Status::kOk || !cuda_ok) return false;

    for (std::uint32_t row = 0; row < kRows; ++row) {
        for (std::uint32_t h = 0; h < physical_hd1; ++h) {
            float expected = 0.0f;
            if (h < logical_hd1) {
                expected = __half2float(weights[embedding_count + h]);
                for (std::uint32_t p = 0; p < shape.state_len; ++p) {
                    const std::uint64_t index =
                        (static_cast<std::uint64_t>(p) * shape.state_value_pad + states[row].v[p]) *
                        shape.hd1 + h;
                    expected += __half2float(weights[index]);
                }
            }
            if (output[static_cast<std::uint64_t>(row) * physical_hd1 + h] != expected)
                return false;
        }
    }
    return true;
}

bool RunProductionCase(std::uint32_t rows) {
    constexpr std::uint32_t kLogical = 2556;
    mgt_cuda::CudaMlpShape shape{72, 72, 2560, 2, 1, 1};
    const std::uint64_t embedding_count =
        static_cast<std::uint64_t>(shape.state_len) * shape.state_value_pad * shape.hd1;
    const std::uint64_t weight_count = embedding_count + shape.hd1;
    std::vector<__half> weights(weight_count);
    for (std::uint64_t i = 0; i < weight_count; ++i)
        weights[i] = __float2half(static_cast<float>(static_cast<int>((i * 13) % 31) - 15) / 32.0f);
    std::vector<mgt::TrainStateStorage> states(rows);
    for (std::uint32_t row = 0; row < rows; ++row)
        for (std::uint32_t p = 0; p < shape.state_len; ++p)
            states[row].v[p] = static_cast<mgt::StateValue>((row * 17 + p * 71) % 72);

    __half* d_weights = nullptr;
    mgt::TrainStateStorage* d_states = nullptr;
    float *d_fast = nullptr, *d_reference = nullptr;
    const std::uint64_t output_count = static_cast<std::uint64_t>(rows) * shape.hd1;
    cudaMalloc(&d_weights, weight_count * sizeof(__half));
    cudaMalloc(&d_states, states.size() * sizeof(states[0]));
    cudaMalloc(&d_fast, output_count * sizeof(float));
    cudaMalloc(&d_reference, output_count * sizeof(float));
    cudaMemcpy(d_weights, weights.data(), weight_count * sizeof(__half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_states, states.data(), states.size() * sizeof(states[0]), cudaMemcpyHostToDevice);
    const auto status = mgt_cuda::LaunchLocalMlpInputHalf(
        shape, kLogical, d_weights, d_states, rows, d_fast, nullptr);
    const std::uint32_t blocks = static_cast<std::uint32_t>((output_count + 255) / 256);
    ScalarReference<<<blocks, 256>>>(shape, kLogical, d_weights, d_states, rows, d_reference);
    std::vector<float> fast(output_count), reference(output_count);
    cudaMemcpy(fast.data(), d_fast, output_count * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(reference.data(), d_reference, output_count * sizeof(float), cudaMemcpyDeviceToHost);
    const bool ok = status == mgt::Status::kOk && cudaGetLastError() == cudaSuccess && fast == reference;
    cudaFree(d_reference); cudaFree(d_fast); cudaFree(d_states); cudaFree(d_weights);
    return ok;
}

}  // namespace

int main() {
    if (!RunCase(8, 8)) return EXIT_FAILURE;
    if (!RunCase(8, 7)) return EXIT_FAILURE;
    if (!RunCase(7, 7)) return EXIT_FAILURE;
    if (!RunCase(7, 6)) return EXIT_FAILURE;
    if (!RunProductionCase(1)) return EXIT_FAILURE;
    if (!RunProductionCase(17)) return EXIT_FAILURE;
    if (!RunProductionCase(4096)) return EXIT_FAILURE;
    return EXIT_SUCCESS;
}
