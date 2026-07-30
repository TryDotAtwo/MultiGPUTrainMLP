#include "mgt_cuda/allreduce_nccl.cuh"
#include "mgt_cuda/mlp_batch_norm_forward.cuh"

#include "mgt/config.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

namespace {

enum class StatePattern { kPermutation, kZero };

template <class T>
bool Alloc(T** pointer, std::uint64_t count) {
    return cudaMalloc(reinterpret_cast<void**>(pointer), count * sizeof(T)) == cudaSuccess;
}

bool ParseStatePattern(StatePattern* pattern) {
    const char* value = std::getenv("MGT_BENCH_STATE_PATTERN");
    if (value == nullptr || std::strcmp(value, "permutation") == 0) {
        *pattern = StatePattern::kPermutation;
        return true;
    }
    if (std::strcmp(value, "zero") == 0) {
        *pattern = StatePattern::kZero;
        return true;
    }
    std::cerr << "MGT_BENCH_STATE_PATTERN must be permutation or zero\n";
    return false;
}

const char* StatePatternName(StatePattern pattern) {
    return pattern == StatePattern::kPermutation ? "permutation" : "zero";
}

std::string EnvironmentValue(const char* name, const char* fallback) {
    const char* value = std::getenv(name);
    return value == nullptr || *value == '\0' ? fallback : value;
}

std::string JsonEscape(const std::string& value) {
    std::string escaped;
    escaped.reserve(value.size());
    for (const char c : value) {
        if (c == '"' || c == '\\') escaped.push_back('\\');
        if (c == '\n') {
            escaped += "\\n";
        } else if (c == '\r') {
            escaped += "\\r";
        } else if (c == '\t') {
            escaped += "\\t";
        } else {
            escaped.push_back(c);
        }
    }
    return escaped;
}

std::uint64_t InputParameterCount(const mgt_cuda::CudaMlpShape& shape) {
    return static_cast<std::uint64_t>(shape.state_len) * shape.state_value_pad *
               shape.hd1 +
           shape.hd1;
}

std::uint64_t Fnv1a64(const std::vector<float>& values) {
    constexpr std::uint64_t offset = 14695981039346656037ULL;
    constexpr std::uint64_t prime = 1099511628211ULL;
    std::uint64_t hash = offset;
    const auto* bytes = reinterpret_cast<const unsigned char*>(values.data());
    for (std::size_t i = 0; i < values.size() * sizeof(float); ++i) {
        hash ^= bytes[i];
        hash *= prime;
    }
    return hash;
}

bool Check(cudaError_t status, const char* operation) {
    if (status == cudaSuccess) return true;
    std::cerr << operation << ": " << cudaGetErrorString(status) << '\n';
    return false;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 9) {
        std::cerr << "usage: device rank world id_file local_rows global_rows warmup steps\n";
        return 2;
    }
    const auto device = static_cast<std::uint32_t>(std::strtoul(argv[1], nullptr, 10));
    const auto rank = static_cast<std::uint32_t>(std::strtoul(argv[2], nullptr, 10));
    const auto world = static_cast<std::uint32_t>(std::strtoul(argv[3], nullptr, 10));
    const auto local_rows = static_cast<std::uint32_t>(std::strtoul(argv[5], nullptr, 10));
    const auto global_rows = static_cast<std::uint32_t>(std::strtoul(argv[6], nullptr, 10));
    const auto warmup = static_cast<std::uint32_t>(std::strtoul(argv[7], nullptr, 10));
    const auto steps = static_cast<std::uint32_t>(std::strtoul(argv[8], nullptr, 10));
    if (world == 0 || local_rows == 0 || global_rows < local_rows || steps == 0 ||
        static_cast<std::uint64_t>(world) * local_rows != global_rows || device >= world) {
        return 2;
    }

    StatePattern state_pattern{};
    if (!ParseStatePattern(&state_pattern) || cudaSetDevice(device) != cudaSuccess) return 3;
    const mgt_cuda::CudaMlpShape shape{
        mgt::kStateLen,
        mgt::kStateValuePad,
        mgt::RoundUp(mgt::kHd1, mgt::kHiddenAlignment),
        mgt::RoundUp(mgt::kHd2, mgt::kHiddenAlignment),
        mgt::kResidualBlocks,
        mgt::kOutputDim,
    };
    mgt::BatchNormTrainingPlan plan;
    if (mgt::BuildBatchNormTrainingPlan(
            mgt::kHd1,
            mgt::kHd2,
            shape.hd1,
            shape.hd2,
            shape.residual_blocks,
            local_rows,
            &plan) != mgt::Status::kOk) {
        return 3;
    }

    const std::uint64_t workspace_count =
        mgt_cuda::MlpBatchNormForwardWorkspaceFloats(shape, plan, local_rows);
    const std::uint64_t input_count =
        static_cast<std::uint64_t>(local_rows) * shape.hd1;
    const std::uint64_t parameter_count = InputParameterCount(shape);
    std::vector<mgt::TrainStateStorage> host_states(local_rows);
    std::memset(
        host_states.data(), 0xA5, host_states.size() * sizeof(mgt::TrainStateStorage));
    for (std::uint32_t row = 0; row < local_rows; ++row) {
        for (std::uint32_t position = 0; position < shape.state_len; ++position) {
            host_states[row].v[position] =
                state_pattern == StatePattern::kZero
                    ? 0
                    : static_cast<mgt::StateValue>(
                          (static_cast<std::uint64_t>(rank) * local_rows + row + position) %
                          shape.state_value_pad);
        }
    }

    std::vector<float> host_affine(plan.trainable_count, 0.0f);
    const auto& input_site = plan.sites[0];
    for (std::uint32_t h = 0; h < mgt::kHd1; ++h) {
        host_affine[input_site.affine_offset + h] = 1.0f;
    }
    std::vector<float> host_workspace(workspace_count, 0.0f);
    for (std::uint64_t i = 0; i < input_count; ++i) host_workspace[i] = 1.0f;
    const std::uint64_t batch_norm_base = input_count +
        static_cast<std::uint64_t>(2U * shape.residual_blocks + 1U) * local_rows * shape.hd2;
    for (std::uint32_t h = 0; h < mgt::kHd1; ++h) {
        host_workspace[batch_norm_base + input_site.inv_std_offset + h] = 1.0f;
    }
    std::vector<float> host_input(input_count, 0.0f);
    for (std::uint32_t row = 0; row < local_rows; ++row) {
        for (std::uint32_t h = 0; h < mgt::kHd1; ++h) {
            const float magnitude =
                static_cast<float>(((row / 2U + h * 3U) % 7U) + 1U) / 8.0f;
            if ((row & 1U) == 0U && row + 1U < local_rows) {
                host_input[static_cast<std::uint64_t>(row) * shape.hd1 + h] = magnitude;
            } else if ((row & 1U) != 0U) {
                host_input[static_cast<std::uint64_t>(row) * shape.hd1 + h] = -magnitude;
            }
        }
    }

    mgt::TrainStateStorage* states = nullptr;
    float *affine = nullptr, *workspace = nullptr, *input = nullptr;
    float *weight_grad = nullptr, *affine_grad = nullptr;
    if (!Alloc(&states, local_rows) || !Alloc(&affine, host_affine.size()) ||
        !Alloc(&workspace, workspace_count) || !Alloc(&input, input_count) ||
        !Alloc(&weight_grad, parameter_count) || !Alloc(&affine_grad, plan.trainable_count)) {
        return 4;
    }
    if (!Check(cudaMemcpy(states, host_states.data(), host_states.size() * sizeof(host_states[0]),
                          cudaMemcpyHostToDevice), "copy states") ||
        !Check(cudaMemcpy(affine, host_affine.data(), host_affine.size() * sizeof(float),
                          cudaMemcpyHostToDevice), "copy affine") ||
        !Check(cudaMemcpy(workspace, host_workspace.data(), workspace_count * sizeof(float),
                          cudaMemcpyHostToDevice), "copy workspace") ||
        !Check(cudaMemcpy(input, host_input.data(), input_count * sizeof(float),
                          cudaMemcpyHostToDevice), "copy input") ||
        !Check(cudaMemset(weight_grad, 0, parameter_count * sizeof(float)), "clear weight grad") ||
        !Check(cudaMemset(affine_grad, 0, plan.trainable_count * sizeof(float)),
               "clear affine grad")) {
        return 4;
    }

    mgt_cuda::NcclRankContext* context = nullptr;
    const auto context_status = world == 1
        ? mgt_cuda::CreateNcclSingleRankContext(device, &context)
        : mgt_cuda::CreateNcclRankContext(
              device, world, rank, std::filesystem::path(argv[4]), &context);
    if (context_status != mgt::Status::kOk) return 5;
    cudaStream_t stream = nullptr;
    cudaEvent_t start = nullptr, stop = nullptr;
    if (cudaStreamCreate(&stream) != cudaSuccess || cudaEventCreate(&start) != cudaSuccess ||
        cudaEventCreate(&stop) != cudaSuccess) return 6;

    const auto run = [&] {
        return mgt_cuda::LaunchMlpBatchNormInputBackward(
            shape, affine, states, local_rows, global_rows, workspace, plan, input,
            weight_grad, affine_grad, context, stream);
    };
    for (std::uint32_t i = 0; i < warmup; ++i) {
        if (run() != mgt::Status::kOk) return 7;
    }
    if (cudaStreamSynchronize(stream) != cudaSuccess ||
        cudaEventRecord(start, stream) != cudaSuccess) return 8;
    for (std::uint32_t i = 0; i < steps; ++i) {
        if (run() != mgt::Status::kOk) return 9;
    }
    if (cudaEventRecord(stop, stream) != cudaSuccess ||
        cudaEventSynchronize(stop) != cudaSuccess) return 10;
    float total_ms = 0.0f;
    if (cudaEventElapsedTime(&total_ms, start, stop) != cudaSuccess) return 11;

    std::vector<float> host_result(parameter_count);
    if (cudaMemcpy(host_result.data(), weight_grad, parameter_count * sizeof(float),
                   cudaMemcpyDeviceToHost) != cudaSuccess) return 12;
    double sum_abs = 0.0;
    for (const float value : host_result) sum_abs += std::fabs(value);
    const std::uint64_t checksum = Fnv1a64(host_result);
    const bool expected_nonzero = state_pattern != StatePattern::kZero;
    if (!std::isfinite(sum_abs) || (expected_nonzero && !(sum_abs > 0.0)) ||
        (!expected_nonzero && sum_abs != 0.0)) {
        std::cerr << "unexpected gradient sum for state pattern "
                  << StatePatternName(state_pattern) << ": " << sum_abs << '\n';
        return 13;
    }

    if (rank == 0) {
        const double step_ms = total_ms / steps;
        const double samples_per_second =
            static_cast<double>(global_rows) * steps / (total_ms / 1000.0);
        std::cout << "{\"world\":" << world << ",\"local_rows\":" << local_rows
                  << ",\"global_rows\":" << global_rows << ",\"warmup\":" << warmup
                  << ",\"steps\":" << steps << ",\"state_pattern\":\""
                  << StatePatternName(state_pattern) << "\",\"input_grad_kernel\":\""
                  << JsonEscape(EnvironmentValue("MGT_BN_INPUT_GRAD_KERNEL", "auto"))
                  << "\",\"input_grad_positions_per_block\":\""
                  << JsonEscape(EnvironmentValue(
                         "MGT_BN_INPUT_GRAD_POSITIONS_PER_BLOCK", "0"))
                  << "\",\"input_backward_ms\":" << step_ms
                  << ",\"samples_per_second\":" << samples_per_second
                  << ",\"gradient_checksum\":" << checksum
                  << ",\"gradient_sum_abs\":" << sum_abs
                  << ",\"workspace_floats_per_rank\":" << workspace_count << "}\n";
    }
    cudaEventDestroy(stop);
    cudaEventDestroy(start);
    cudaStreamDestroy(stream);
    return mgt_cuda::DestroyNcclRankContext(context) == mgt::Status::kOk ? 0 : 14;
}
