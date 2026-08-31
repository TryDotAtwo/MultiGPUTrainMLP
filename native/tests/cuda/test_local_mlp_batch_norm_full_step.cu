#include "mgt_cuda/local_mlp_batch_norm.cuh"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdint>
#include <vector>

namespace {
template <class T> T* Device(std::size_t count) {
    T* pointer = nullptr;
    return cudaMalloc(&pointer, count * sizeof(T)) == cudaSuccess ? pointer : nullptr;
}
bool Near(float actual, float expected,
#ifdef MGT_TEST_LOCAL_FP16
          float tolerance = 2e-3f) {
#else
          float tolerance = 6e-5f) {
#endif
    return std::fabs(actual - expected) <= tolerance;
}
}

int main() {
    const mgt_cuda::CudaMlpShape shape{2, 4, 3, 2, 1, 1};
    mgt::BatchNormTrainingPlan plan;
    if (mgt::BuildBatchNormTrainingPlan(3, 2, 3, 2, 1, 4, &plan) != mgt::Status::kOk)
        return 1;
    auto bn = mgt::InitializeBatchNormTrainingState(plan);
    constexpr std::uint64_t parameter_count = 50;
    std::vector<float> weights(parameter_count);
    for (std::size_t i = 0; i < weights.size(); ++i)
        weights[i] = .03f * static_cast<float>(static_cast<int>(i % 9) - 4);
    mgt::TrainStateStorage states[4]{};
    for (unsigned row = 0; row < 4; ++row) {
        states[row].v[0] = row;
        states[row].v[1] = (row + 1) % 4;
    }
    const float labels[4]{1.f, 3.f, -1.f, 2.f};
    const auto workspace_count = mgt_cuda::LocalMlpBatchNormForwardWorkspaceFloats(shape, plan, 4);
    auto* d_weights = Device<float>(parameter_count);
    auto* d_weight_grad = Device<float>(parameter_count);
    auto* d_weight_m = Device<float>(parameter_count);
    auto* d_weight_v = Device<float>(parameter_count);
    auto* d_affine = Device<float>(bn.affine.size());
    auto* d_affine_grad = Device<float>(bn.affine.size());
    auto* d_affine_m = Device<float>(bn.affine.size());
    auto* d_affine_v = Device<float>(bn.affine.size());
    auto* d_running = Device<float>(bn.running.size());
    auto* d_outputs = Device<float>(4);
    auto* d_workspace = Device<float>(workspace_count);
    auto* d_loss = Device<float>(1);
    auto* d_output_dy = Device<float>(4);
    auto* d_block_grad = Device<float>(8);
    auto* d_fc1_grad = Device<float>(8);
    auto* d_residual_grad = Device<float>(8);
    auto* d_input_grad = Device<float>(12);
    auto* d_states = Device<mgt::TrainStateStorage>(4);
    auto* d_labels = Device<float>(4);
#ifdef MGT_TEST_LOCAL_FP16
    auto* d_weight_half = Device<__half>(parameter_count);
    const std::uint64_t activation_tape_count =
        4 * shape.hd1 + 2ULL * shape.residual_blocks * 4 * shape.hd2;
    auto* d_operand_a = Device<__half>(activation_tape_count);
    auto* d_operand_b = Device<__half>(8);
#endif
    if (!d_weights || !d_weight_grad || !d_weight_m || !d_weight_v || !d_affine ||
        !d_affine_grad || !d_affine_m || !d_affine_v || !d_running || !d_outputs ||
        !d_workspace || !d_loss || !d_output_dy || !d_block_grad || !d_fc1_grad ||
        !d_residual_grad || !d_input_grad || !d_states || !d_labels
#ifdef MGT_TEST_LOCAL_FP16
        || !d_weight_half || !d_operand_a || !d_operand_b
#endif
        ) return 1;
    cudaMemcpy(d_weights, weights.data(), weights.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_affine, bn.affine.data(), bn.affine.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_running, bn.running.data(), bn.running.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_states, states, sizeof(states), cudaMemcpyHostToDevice);
    cudaMemcpy(d_labels, labels, sizeof(labels), cudaMemcpyHostToDevice);
    cudaMemset(d_weight_grad, 0, parameter_count * sizeof(float));
    cudaMemset(d_weight_m, 0, parameter_count * sizeof(float));
    cudaMemset(d_weight_v, 0, parameter_count * sizeof(float));
    cudaMemset(d_affine_grad, 0, bn.affine.size() * sizeof(float));
    cudaMemset(d_affine_m, 0, bn.affine.size() * sizeof(float));
    cudaMemset(d_affine_v, 0, bn.affine.size() * sizeof(float));
#ifdef MGT_TEST_LOCAL_FP16
    if (mgt_cuda::LaunchFloatToHalf(
            d_weights, d_weight_half, parameter_count, nullptr) != mgt::Status::kOk)
        return 1;
    cudaMemset(d_operand_a, 0xff, activation_tape_count * sizeof(__half));
#endif
    cublasHandle_t blas = nullptr;
    if (cublasCreate(&blas) != CUBLAS_STATUS_SUCCESS) return 1;
    const mgt_cuda::AdamWKernelConfig adam{0, 1, .001f, .9f, .999f, 1e-8f, 0.f};
    const mgt_cuda::MlpBatchNormStepBuffers buffers{
        d_weights, d_weight_grad, d_weight_m, d_weight_v, d_affine,
        d_affine_grad, d_affine_m, d_affine_v, d_running, d_outputs,
        d_workspace, d_loss, d_output_dy, d_block_grad, d_fc1_grad,
        d_residual_grad, d_input_grad};
#ifdef MGT_TEST_LOCAL_FP16
    mgt_cuda::LocalMlpFp16Context fp16{
        d_weights, d_weight_half, d_operand_a, d_operand_b, activation_tape_count, 8};
    const auto launch_status = mgt_cuda::LaunchLocalMlpBatchNormTrainStepFp16(
            shape, 3, 2, d_states, d_labels, 4, plan, workspace_count,
            adam, buffers, &fp16, blas, nullptr);
#else
    const auto launch_status = mgt_cuda::LaunchLocalMlpBatchNormTrainStep(
            shape, 3, 2, d_states, d_labels, 4, plan, workspace_count,
            adam, buffers, blas, nullptr);
#endif
    if (launch_status != mgt::Status::kOk || cudaDeviceSynchronize() != cudaSuccess) return 1;
#ifdef MGT_TEST_LOCAL_FP16
    std::vector<float> saved(workspace_count);
    std::vector<__half> activation_tape(activation_tape_count);
    cudaMemcpy(saved.data(), d_workspace, saved.size() * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(activation_tape.data(), d_operand_a,
               activation_tape.size() * sizeof(__half), cudaMemcpyDeviceToHost);
    const std::uint64_t batch_hd1 = 4 * shape.hd1;
    const std::uint64_t batch_hd2 = 4 * shape.hd2;
    const std::uint64_t block_inputs = batch_hd1;
    const std::uint64_t fc1_activations =
        block_inputs + (shape.residual_blocks + 1ULL) * batch_hd2;
    for (std::uint64_t i = 0; i < batch_hd1; ++i) {
        if (__half2float(activation_tape[i]) != __half2float(__float2half_rn(saved[i])))
            return 1;
    }
    for (std::uint64_t block = 0; block < shape.residual_blocks; ++block) {
        const std::uint64_t tape_block = batch_hd1 + 2 * block * batch_hd2;
        for (std::uint64_t i = 0; i < batch_hd2; ++i) {
            if (__half2float(activation_tape[tape_block + i]) !=
                    __half2float(__float2half_rn(saved[block_inputs + block * batch_hd2 + i])) ||
                __half2float(activation_tape[tape_block + batch_hd2 + i]) !=
                    __half2float(__float2half_rn(saved[fc1_activations + block * batch_hd2 + i])))
                return 1;
        }
    }
#endif
    float gradient[27]{}, affine_gradient[6]{};
    float loss = 0.0f;
    cudaMemcpy(&loss, d_loss, sizeof(loss), cudaMemcpyDeviceToHost);
    cudaMemcpy(gradient, d_weight_grad, sizeof(gradient), cudaMemcpyDeviceToHost);
    cudaMemcpy(affine_gradient, d_affine_grad, 3 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(affine_gradient + 3, d_affine_grad + 9, 3 * sizeof(float), cudaMemcpyDeviceToHost);
    if (!Near(loss, 3.6869926453f,
#ifdef MGT_TEST_LOCAL_FP16
              2e-3f
#else
              3e-5f
#endif
              )) {
        std::fprintf(stderr, "loss=%.9g expected %.9g\n", loss, 3.6869926453f);
        return 1;
    }
    const float expected_gradient[27]{
        .3715080619f,.1338969618f,-.1037141159f,-.0928778797f,-.0334745273f,.0259287991f,
        -.1857523322f,-.0669478998f,.0518565141f,-.0928778797f,-.0334745273f,.0259287991f,
        -.0928778797f,-.0334745273f,.0259287991f,.3715080619f,.1338969618f,-.1037141159f,
        -.0928778797f,-.0334745273f,.0259287991f,-.1857523322f,-.0669478998f,.0518565141f,
        -1.4901161e-8f,7.4505806e-9f,0.f};
    const float expected_affine[6]{
        .0001709695f,.0000579668f,-.0000550380f,.0279030837f,.0100533739f,-.0077963332f};
    for (int i = 0; i < 27; ++i) if (!Near(gradient[i], expected_gradient[i])) {
        std::fprintf(stderr, "weight_grad[%d]=%.9g expected %.9g\n", i, gradient[i], expected_gradient[i]);
        return 1;
    }
    for (int i = 0; i < 6; ++i) if (!Near(affine_gradient[i], expected_affine[i])) {
        std::fprintf(stderr, "affine_grad[%d]=%.9g expected %.9g\n", i, affine_gradient[i], expected_affine[i]);
        return 1;
    }
    float updated_weight = 0, updated_affine = 0;
    cudaMemcpy(&updated_weight, d_weights, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&updated_affine, d_affine, sizeof(float), cudaMemcpyDeviceToHost);
    cublasDestroy(blas);
    if (!Near(updated_weight, -.1209999999f,
#ifdef MGT_TEST_LOCAL_FP16
              2e-3f
#else
              2e-6f
#endif
              ) || !Near(updated_affine, .9990000725f,
#ifdef MGT_TEST_LOCAL_FP16
              2e-3f
#else
              2e-6f
#endif
              )) {
        std::fprintf(stderr, "updated weight=%.9g affine=%.9g\n", updated_weight, updated_affine);
        return 1;
    }
    return 0;
}
