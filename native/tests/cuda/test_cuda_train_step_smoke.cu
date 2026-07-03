#include "mgt_cuda/adamw.cuh"
#include "mgt_cuda/mlp_backward.cuh"
#include "mgt_cuda/mlp_forward.cuh"
#include "mgt_cuda/random_walk_kernel.cuh"
#include "mgt/puzzle_io.hpp"
#include <cuda_runtime.h>
#include <cmath>
#include <cstdlib>
#include <vector>

namespace {

int Check(cudaError_t status) { return status == cudaSuccess ? 0 : 1; }

std::uint64_t ResidualBlockParams(const mgt_cuda::CudaMlpShape& shape) {
    return 2ULL * (static_cast<std::uint64_t>(shape.hd2) * shape.hd2 + shape.hd2);
}

std::uint64_t ParamCount(const mgt_cuda::CudaMlpShape& shape) {
    return static_cast<std::uint64_t>(shape.state_len) * shape.state_value_pad * shape.hd1 + shape.hd1 +
           static_cast<std::uint64_t>(shape.hd1) * shape.hd2 + shape.hd2 +
           static_cast<std::uint64_t>(shape.residual_blocks) * ResidualBlockParams(shape) + shape.hd2 + 1ULL;
}

mgt::PuzzleDefinition BuildPuzzle() {
    mgt::PuzzleDefinition puzzle{};
    for (std::uint32_t move = 0; move < mgt::kMoveCount; ++move) {
        for (std::uint32_t i = 0; i < mgt::kStateLen; ++i) {
            puzzle.moves[move].v[i] = static_cast<mgt::StateValue>((i + move + 1) % mgt::kStateLen);
        }
        for (std::uint32_t i = mgt::kStateLen; i < mgt::kStateStorageLen; ++i) {
            puzzle.moves[move].v[i] = 0;
        }
    }
    for (std::uint32_t i = 0; i < mgt::kStateLen; ++i) {
        puzzle.target.v[i] = static_cast<mgt::StateValue>(i);
    }
    for (std::uint32_t i = mgt::kStateLen; i < mgt::kStateStorageLen; ++i) {
        puzzle.target.v[i] = 0;
    }
    return puzzle;
}

}  // namespace

int main() {
    int device_count = 0;
    if (Check(cudaGetDeviceCount(&device_count)) != 0 || device_count <= 0) return EXIT_FAILURE;
    if (Check(cudaSetDevice(0)) != 0) return EXIT_FAILURE;

    const mgt_cuda::CudaMlpShape shape{mgt::kStateLen, mgt::kStateLen, 5, 3, 1};
    constexpr std::uint32_t kSamples = 32;
    const std::uint64_t params = ParamCount(shape);
    std::vector<float> weights(params);
    for (std::uint64_t i = 0; i < params; ++i) {
        weights[i] = static_cast<float>((static_cast<int>(i % 29) - 14) * 0.0001);
    }

    const mgt::PuzzleDefinition puzzle = BuildPuzzle();
    float* d_weights = nullptr;
    float* d_labels = nullptr;
    float* d_loss = nullptr;
    float* d_grad = nullptr;
    float* d_m = nullptr;
    float* d_v = nullptr;
    mgt::TrainState80* d_states = nullptr;
    mgt::WalkMeta* d_meta = nullptr;
    mgt::TrainState80* d_moves = nullptr;
    mgt::TrainState80* d_target = nullptr;
    if (Check(cudaMalloc(&d_weights, params * sizeof(float))) != 0) return EXIT_FAILURE;
    if (Check(cudaMalloc(&d_labels, kSamples * sizeof(float))) != 0) return EXIT_FAILURE;
    if (Check(cudaMalloc(&d_loss, sizeof(float))) != 0) return EXIT_FAILURE;
    if (Check(cudaMalloc(&d_grad, params * sizeof(float))) != 0) return EXIT_FAILURE;
    if (Check(cudaMalloc(&d_m, params * sizeof(float))) != 0) return EXIT_FAILURE;
    if (Check(cudaMalloc(&d_v, params * sizeof(float))) != 0) return EXIT_FAILURE;
    if (Check(cudaMalloc(&d_states, kSamples * sizeof(mgt::TrainState80))) != 0) return EXIT_FAILURE;
    if (Check(cudaMalloc(&d_meta, kSamples * sizeof(mgt::WalkMeta))) != 0) return EXIT_FAILURE;
    if (Check(cudaMalloc(&d_moves, mgt::kMoveCount * sizeof(mgt::TrainState80))) != 0) return EXIT_FAILURE;
    if (Check(cudaMalloc(&d_target, sizeof(mgt::TrainState80))) != 0) return EXIT_FAILURE;
    if (Check(cudaMemcpy(d_weights, weights.data(), params * sizeof(float), cudaMemcpyHostToDevice)) != 0) return EXIT_FAILURE;
    if (Check(cudaMemcpy(d_moves, puzzle.moves.data(), mgt::kMoveCount * sizeof(mgt::TrainState80), cudaMemcpyHostToDevice)) != 0) return EXIT_FAILURE;
    if (Check(cudaMemcpy(d_target, &puzzle.target, sizeof(mgt::TrainState80), cudaMemcpyHostToDevice)) != 0) return EXIT_FAILURE;
    if (Check(cudaMemset(d_m, 0, params * sizeof(float))) != 0) return EXIT_FAILURE;
    if (Check(cudaMemset(d_v, 0, params * sizeof(float))) != 0) return EXIT_FAILURE;

    const mgt_cuda::RandomWalkKernelConfig walks{kSamples, 1, 9};
    if (mgt_cuda::LaunchRandomWalkKernel(walks, 1234, 5, 7, 0, d_moves, d_target, d_states, d_labels, d_meta, 0) != mgt::Status::kOk) return EXIT_FAILURE;
    if (Check(cudaDeviceSynchronize()) != 0) return EXIT_FAILURE;

    if (mgt_cuda::LaunchMlpLossGradKernel(shape, d_weights, d_states, d_labels, kSamples, d_loss, d_grad, 0) != mgt::Status::kOk) return EXIT_FAILURE;
    if (Check(cudaDeviceSynchronize()) != 0) return EXIT_FAILURE;
    float loss_before = 0.0f;
    if (Check(cudaMemcpy(&loss_before, d_loss, sizeof(float), cudaMemcpyDeviceToHost)) != 0) return EXIT_FAILURE;
    if (!std::isfinite(loss_before) || loss_before <= 0.0f) return EXIT_FAILURE;

    const mgt_cuda::AdamWKernelConfig adam{params, 1, 0.0001f, 0.9f, 0.999f, 1.0e-8f, 0.0f};
    if (mgt_cuda::LaunchAdamWKernel(adam, d_weights, d_grad, d_m, d_v, 0) != mgt::Status::kOk) return EXIT_FAILURE;
    if (Check(cudaDeviceSynchronize()) != 0) return EXIT_FAILURE;
    if (mgt_cuda::LaunchMlpLossGradKernel(shape, d_weights, d_states, d_labels, kSamples, d_loss, d_grad, 0) != mgt::Status::kOk) return EXIT_FAILURE;
    if (Check(cudaDeviceSynchronize()) != 0) return EXIT_FAILURE;
    float loss_after = 0.0f;
    if (Check(cudaMemcpy(&loss_after, d_loss, sizeof(float), cudaMemcpyDeviceToHost)) != 0) return EXIT_FAILURE;
    if (!std::isfinite(loss_after) || loss_after > loss_before) return EXIT_FAILURE;

    cudaFree(d_target);
    cudaFree(d_moves);
    cudaFree(d_meta);
    cudaFree(d_states);
    cudaFree(d_v);
    cudaFree(d_m);
    cudaFree(d_grad);
    cudaFree(d_loss);
    cudaFree(d_labels);
    cudaFree(d_weights);
    return EXIT_SUCCESS;
}