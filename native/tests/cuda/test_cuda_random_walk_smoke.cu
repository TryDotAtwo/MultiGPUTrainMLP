#include "mgt_cuda/random_walk_kernel.cuh"
#include "mgt/puzzle_io.hpp"
#include <cuda_runtime.h>
#include <cstdlib>
#include <vector>

namespace {

int Check(cudaError_t status) {
    return status == cudaSuccess ? 0 : 1;
}

}  // namespace

int main() {
    int device_count = 0;
    if (Check(cudaGetDeviceCount(&device_count)) != 0 || device_count <= 0) return EXIT_FAILURE;
    if (Check(cudaSetDevice(0)) != 0) return EXIT_FAILURE;

    mgt::PuzzleDefinition puzzle{};
    for (std::uint32_t move = 0; move < mgt::kMoveCount; ++move) {
        for (std::uint32_t i = 0; i < mgt::kStateLen; ++i) {
            puzzle.moves[move].v[i] = static_cast<mgt::StateValue>((i + move + 1) % mgt::kStateLen);
        }
        for (std::uint32_t i = mgt::kStateLen; i < mgt::kStateStorageLen; ++i) {
            puzzle.moves[move].v[i] = static_cast<mgt::StateValue>(i);
        }
    }
    for (std::uint32_t i = 0; i < mgt::kStateLen; ++i) {
        puzzle.target.v[i] = static_cast<mgt::StateValue>(i);
    }
    for (std::uint32_t i = mgt::kStateLen; i < mgt::kStateStorageLen; ++i) {
        puzzle.target.v[i] = 0;
    }

    constexpr std::uint32_t kSamples = 256;
    mgt::TrainStateStorage* d_states = nullptr;
    float* d_labels = nullptr;
    mgt::WalkMeta* d_meta = nullptr;
    mgt::TrainStateStorage* d_moves = nullptr;
    mgt::TrainStateStorage* d_target = nullptr;
    if (Check(cudaMalloc(&d_states, kSamples * sizeof(mgt::TrainStateStorage))) != 0) return EXIT_FAILURE;
    if (Check(cudaMalloc(&d_labels, kSamples * sizeof(float))) != 0) return EXIT_FAILURE;
    if (Check(cudaMalloc(&d_meta, kSamples * sizeof(mgt::WalkMeta))) != 0) return EXIT_FAILURE;
    if (Check(cudaMalloc(&d_moves, mgt::kMoveCount * sizeof(mgt::TrainStateStorage))) != 0) return EXIT_FAILURE;
    if (Check(cudaMalloc(&d_target, sizeof(mgt::TrainStateStorage))) != 0) return EXIT_FAILURE;
    if (Check(cudaMemcpy(d_moves, puzzle.moves.data(), mgt::kMoveCount * sizeof(mgt::TrainStateStorage), cudaMemcpyHostToDevice)) != 0) return EXIT_FAILURE;
    if (Check(cudaMemcpy(d_target, &puzzle.target, sizeof(mgt::TrainStateStorage), cudaMemcpyHostToDevice)) != 0) return EXIT_FAILURE;

    const mgt_cuda::RandomWalkKernelConfig config{kSamples, 1, 9};
    const auto launch_status = mgt_cuda::LaunchRandomWalkKernel(
        config, 1234, 5, 7, 2, d_moves, d_target, d_states, d_labels, d_meta, 0);
    if (launch_status != mgt::Status::kOk) return EXIT_FAILURE;
    if (Check(cudaDeviceSynchronize()) != 0) return EXIT_FAILURE;

    std::vector<mgt::TrainStateStorage> states(kSamples);
    std::vector<float> labels(kSamples);
    std::vector<mgt::WalkMeta> meta(kSamples);
    if (Check(cudaMemcpy(states.data(), d_states, kSamples * sizeof(mgt::TrainStateStorage), cudaMemcpyDeviceToHost)) != 0) return EXIT_FAILURE;
    if (Check(cudaMemcpy(labels.data(), d_labels, kSamples * sizeof(float), cudaMemcpyDeviceToHost)) != 0) return EXIT_FAILURE;
    if (Check(cudaMemcpy(meta.data(), d_meta, kSamples * sizeof(mgt::WalkMeta), cudaMemcpyDeviceToHost)) != 0) return EXIT_FAILURE;

    bool saw_non_target = false;
    for (std::uint32_t sample = 0; sample < kSamples; ++sample) {
        if (meta[sample].depth < config.k_min || meta[sample].depth > config.k_max) return EXIT_FAILURE;
        if (labels[sample] != static_cast<float>(meta[sample].depth)) return EXIT_FAILURE;
        if (meta[sample].last_move >= mgt::kMoveCount) return EXIT_FAILURE;
        if (meta[sample].rng_counter == 0) return EXIT_FAILURE;
        for (std::uint32_t i = mgt::kStateLen; i < mgt::kStateStorageLen; ++i) {
            if (states[sample].v[i] != 0) return EXIT_FAILURE;
        }
        for (std::uint32_t i = 0; i < mgt::kStateLen; ++i) {
            if (states[sample].v[i] != puzzle.target.v[i]) saw_non_target = true;
        }
    }
    if (!saw_non_target) return EXIT_FAILURE;

    cudaFree(d_target);
    cudaFree(d_moves);
    cudaFree(d_meta);
    cudaFree(d_labels);
    cudaFree(d_states);
    return EXIT_SUCCESS;
}