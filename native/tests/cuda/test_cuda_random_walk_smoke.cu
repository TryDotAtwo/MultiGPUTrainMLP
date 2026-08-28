#include "mgt_cuda/random_walk_kernel.cuh"
#include "mgt/puzzle_io.hpp"
#include "mgt/random_walk.hpp"
#include <cuda_runtime.h>
#include <cstring>
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
    if (mgt::LoadPuzzleDefinition(
            "native/production_inputs/p888.json",
            "native/tests/fixtures/p888-target.bin",
            &puzzle) != mgt::Status::kOk) return EXIT_FAILURE;

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

    mgt_cuda::RandomWalkKernelConfig config{kSamples, 1, 29};
    config.epoch_sample_offset = 123456;
    config.original_p888_schedule = 1;
    const auto launch_status = mgt_cuda::LaunchRandomWalkKernel(
        config, 1234, 5, 7, 2, d_moves, d_target, d_states, d_labels, d_meta, 0);
    if (launch_status != mgt::Status::kOk) return EXIT_FAILURE;
    if (Check(cudaDeviceSynchronize()) != 0) return EXIT_FAILURE;

    std::vector<mgt::TrainStateStorage> states(kSamples);
    std::vector<float> labels(kSamples);
    std::vector<mgt::WalkMeta> meta(kSamples);
    std::vector<mgt::TrainStateStorage> cpu_states(kSamples);
    std::vector<float> cpu_labels(kSamples);
    std::vector<mgt::WalkMeta> cpu_meta(kSamples);
    if (Check(cudaMemcpy(states.data(), d_states, kSamples * sizeof(mgt::TrainStateStorage), cudaMemcpyDeviceToHost)) != 0) return EXIT_FAILURE;
    if (Check(cudaMemcpy(labels.data(), d_labels, kSamples * sizeof(float), cudaMemcpyDeviceToHost)) != 0) return EXIT_FAILURE;
    if (Check(cudaMemcpy(meta.data(), d_meta, kSamples * sizeof(mgt::WalkMeta), cudaMemcpyDeviceToHost)) != 0) return EXIT_FAILURE;
    const mgt::WalkRequest cpu_request{
        1234, 5, 7, 2, 1, 29, kSamples, 123456, true};
    if (mgt::GenerateRandomWalksCpu(
            puzzle, cpu_request, cpu_states.data(), cpu_labels.data(),
            cpu_meta.data()) != mgt::Status::kOk) return EXIT_FAILURE;
    if (std::memcmp(states.data(), cpu_states.data(),
                    kSamples * sizeof(mgt::TrainStateStorage)) != 0 ||
        std::memcmp(labels.data(), cpu_labels.data(),
                    kSamples * sizeof(float)) != 0 ||
        std::memcmp(meta.data(), cpu_meta.data(),
                    kSamples * sizeof(mgt::WalkMeta)) != 0) return EXIT_FAILURE;

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
