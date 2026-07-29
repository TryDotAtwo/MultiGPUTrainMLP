#include "mgt/batch_norm_training.hpp"
#include "mgt_cuda/allreduce_nccl.cuh"
#include "mgt_cuda/mlp_batch_norm_forward.cuh"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>

namespace {

constexpr std::size_t kGuard = 31;
constexpr float kGuardValue = -12345.25f;

struct Case {
    const char* name;
    mgt_cuda::CudaMlpShape shape;
    std::uint32_t logical_hd1;
    std::uint32_t logical_hd2;
    std::uint32_t rows;
};

std::uint64_t InputTableFloats(const mgt_cuda::CudaMlpShape& shape) {
    return static_cast<std::uint64_t>(shape.state_len) * shape.state_value_pad * shape.hd1;
}

bool IsPositiveZero(float value) {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits == 0U;
}

bool CheckCuda(cudaError_t status, const char* where) {
    if (status == cudaSuccess) return true;
    std::cerr << where << ": " << cudaGetErrorString(status) << '\n';
    return false;
}

bool CheckGuards(const std::vector<float>& values, std::size_t payload, const char* name) {
    for (std::size_t i = 0; i < kGuard; ++i) {
        if (values[i] != kGuardValue || values[kGuard + payload + i] != kGuardValue) {
            std::cerr << name << " guard modified at " << i << '\n';
            return false;
        }
    }
    return true;
}

bool RunCase(const Case& test, mgt_cuda::NcclRankContext* context) {
    const auto& shape = test.shape;
    mgt::BatchNormTrainingPlan plan;
    if (mgt::BuildBatchNormTrainingPlan(test.logical_hd1, test.logical_hd2, shape.hd1, shape.hd2,
                                        shape.residual_blocks, test.rows, &plan) != mgt::Status::kOk) {
        std::cerr << test.name << ": unable to build BN plan\n";
        return false;
    }
    const std::uint64_t workspace_floats = mgt_cuda::MlpBatchNormForwardWorkspaceFloats(shape, plan, test.rows);
    const std::uint64_t table_floats = InputTableFloats(shape);
    const std::uint64_t grad_floats = table_floats + shape.hd1;
    std::vector<mgt::TrainStateStorage> states(test.rows);
    for (auto& state : states) std::memset(state.v, 0xA5, sizeof(state.v));
    for (std::uint32_t row = 0; row < test.rows; ++row) {
        for (std::uint32_t pos = 0; pos < shape.state_len; ++pos) {
            // Deliberately skewed: exercise empty bins, repeated bins, and every position.
            states[row].v[pos] = static_cast<mgt::StateValue>((row * (pos + 3U) + pos * 5U + 1U) % shape.state_value_pad);
        }
    }

    std::vector<float> affine(2U * plan.logical_feature_count, 0.0f);
    const auto& input_site = plan.sites[0];
    for (std::uint32_t h = 0; h < test.logical_hd1; ++h) {
        affine[input_site.affine_offset + h] = static_cast<float>(test.rows);
    }
    std::vector<float> workspace(workspace_floats, 0.0f);
    for (std::uint64_t i = 0; i < static_cast<std::uint64_t>(test.rows) * shape.hd1; ++i) workspace[i] = 1.0f;
    const std::uint64_t batch_norm_base =
        static_cast<std::uint64_t>(test.rows) * shape.hd1 +
        static_cast<std::uint64_t>(2U * shape.residual_blocks + 1U) * test.rows * shape.hd2;
    for (std::uint32_t h = 0; h < test.logical_hd1; ++h) {
        workspace[batch_norm_base + input_site.inv_std_offset + h] = 1.0f;
    }

    std::vector<float> input_grad(static_cast<std::uint64_t>(test.rows) * shape.hd1, 0.0f);
    std::vector<float> expected(grad_floats, 0.0f);
    for (std::uint32_t row = 0; row < test.rows; ++row) {
        for (std::uint32_t h = 0; h < test.logical_hd1; ++h) {
            float dy = 0.0f;
            if ((row & 1U) == 0U && row + 1U < test.rows) {
                dy = static_cast<float>(((row / 2U + h * 3U) % 7U) + 1U) / 8.0f;
            } else if ((row & 1U) != 0U) {
                dy = -static_cast<float>((((row - 1U) / 2U + h * 3U) % 7U) + 1U) / 8.0f;
            }
            input_grad[static_cast<std::uint64_t>(row) * shape.hd1 + h] = dy;
            const float dz = static_cast<float>(test.rows) * dy;
            for (std::uint32_t pos = 0; pos < shape.state_len; ++pos) {
                const std::uint64_t row_index = static_cast<std::uint64_t>(pos) * shape.state_value_pad + states[row].v[pos];
                expected[row_index * shape.hd1 + h] += dz;
            }
            expected[table_floats + h] += dz;
        }
    }

    mgt::TrainStateStorage* device_states = nullptr;
    float *device_affine = nullptr, *device_workspace = nullptr, *device_input_grad = nullptr,
          *device_weight_storage = nullptr, *device_affine_grad_storage = nullptr;
    const std::size_t weight_storage_floats = static_cast<std::size_t>(grad_floats) + 2U * kGuard;
    const std::size_t affine_grad_floats = 2U * plan.logical_feature_count;
    const std::size_t affine_storage_floats = affine_grad_floats + 2U * kGuard;
    bool ok = CheckCuda(cudaMalloc(&device_states, states.size() * sizeof(states[0])), "cudaMalloc states") &&
              CheckCuda(cudaMalloc(&device_affine, affine.size() * sizeof(float)), "cudaMalloc affine") &&
              CheckCuda(cudaMalloc(&device_workspace, workspace.size() * sizeof(float)), "cudaMalloc workspace") &&
              CheckCuda(cudaMalloc(&device_input_grad, input_grad.size() * sizeof(float)), "cudaMalloc input_grad") &&
              CheckCuda(cudaMalloc(&device_weight_storage, weight_storage_floats * sizeof(float)), "cudaMalloc weight grad") &&
              CheckCuda(cudaMalloc(&device_affine_grad_storage, affine_storage_floats * sizeof(float)), "cudaMalloc affine grad");
    std::vector<float> guarded_weight(weight_storage_floats, kGuardValue);
    std::vector<float> guarded_affine(affine_storage_floats, kGuardValue);
    for (std::size_t i = kGuard; i < kGuard + affine_grad_floats; ++i) {
        guarded_affine[i] = 0.0f;
    }
    std::vector<float> result(grad_floats), repeat(grad_floats), affine_result(affine_storage_floats);
    if (ok) {
        ok = CheckCuda(cudaMemcpy(device_states, states.data(), states.size() * sizeof(states[0]), cudaMemcpyHostToDevice), "copy states") &&
             CheckCuda(cudaMemcpy(device_affine, affine.data(), affine.size() * sizeof(float), cudaMemcpyHostToDevice), "copy affine") &&
             CheckCuda(cudaMemcpy(device_workspace, workspace.data(), workspace.size() * sizeof(float), cudaMemcpyHostToDevice), "copy workspace");
    }
    auto invoke = [&]() -> bool {
        if (!CheckCuda(cudaMemcpy(device_input_grad, input_grad.data(), input_grad.size() * sizeof(float), cudaMemcpyHostToDevice), "reset input grad") ||
            !CheckCuda(cudaMemcpy(device_weight_storage, guarded_weight.data(), guarded_weight.size() * sizeof(float), cudaMemcpyHostToDevice), "reset weight guards") ||
            !CheckCuda(cudaMemcpy(device_affine_grad_storage, guarded_affine.data(), guarded_affine.size() * sizeof(float), cudaMemcpyHostToDevice), "reset affine guards")) return false;
        const auto status = mgt_cuda::LaunchMlpBatchNormInputBackward(shape, device_affine, device_states, test.rows, test.rows,
            device_workspace, plan, device_input_grad, device_weight_storage + kGuard,
            device_affine_grad_storage + kGuard, context, 0);
        return status == mgt::Status::kOk && CheckCuda(cudaDeviceSynchronize(), "input backward synchronize");
    };
    if (ok) ok = invoke();
    if (ok) ok = CheckCuda(cudaMemcpy(result.data(), device_weight_storage + kGuard, grad_floats * sizeof(float), cudaMemcpyDeviceToHost), "copy result") &&
                 CheckCuda(cudaMemcpy(affine_result.data(), device_affine_grad_storage, affine_result.size() * sizeof(float), cudaMemcpyDeviceToHost), "copy affine result");
    if (ok) {
        for (std::size_t i = 0; i < result.size(); ++i) {
            if (expected[i] == 0.0f ? !IsPositiveZero(result[i]) : result[i] != expected[i]) {
                std::cerr << test.name << ": gradient mismatch at " << i << ", got " << result[i] << ", expected " << expected[i] << '\n';
                ok = false;
                break;
            }
        }
        ok = ok && CheckGuards(affine_result, affine_grad_floats, "affine") && CheckCuda(cudaMemcpy(guarded_weight.data(), device_weight_storage, guarded_weight.size() * sizeof(float), cudaMemcpyDeviceToHost), "copy weight guards") && CheckGuards(guarded_weight, grad_floats, "weight");
        for (std::size_t i = kGuard; ok && i < kGuard + affine_grad_floats; ++i) {
            if (!IsPositiveZero(affine_result[i])) { std::cerr << test.name << ": affine gradient not +0 at " << (i - kGuard) << '\n'; ok = false; }
        }
    }
    if (ok) ok = invoke();
    if (ok) ok = CheckCuda(cudaMemcpy(repeat.data(), device_weight_storage + kGuard, grad_floats * sizeof(float), cudaMemcpyDeviceToHost), "copy repeated result");
    if (ok && std::memcmp(result.data(), repeat.data(), result.size() * sizeof(float)) != 0) {
        std::cerr << test.name << ": repeated output is not byte-identical\n";
        ok = false;
    }
    cudaFree(device_affine_grad_storage); cudaFree(device_weight_storage); cudaFree(device_input_grad);
    cudaFree(device_workspace); cudaFree(device_affine); cudaFree(device_states);
    return ok;
}

}  // namespace

int main() {
    mgt_cuda::NcclRankContext* context = nullptr;
    if (mgt_cuda::CreateNcclSingleRankContext(0, &context) != mgt::Status::kOk) return 1;
    const Case cases[] = {
        {"small_irregular", {3, 4, 8, 8, 1, 1}, 5, 3, 7},
        {"row_tail", {5, 8, 104, 8, 1, 1}, 101, 3, 257},
        {"p888_geometry", {72, 72, 2560, 224, 16, 1}, 2556, 218, 17},
    };
    bool ok = true;
    for (const auto& test : cases) ok = RunCase(test, context) && ok;
    if (mgt_cuda::DestroyNcclRankContext(context) != mgt::Status::kOk) ok = false;
    return ok ? 0 : 1;
}
