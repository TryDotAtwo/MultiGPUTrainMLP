#include "mgt/puzzle_io.hpp"
#include "mgt_cuda/adamw.cuh"
#include "mgt_cuda/mlp_backward.cuh"
#include "mgt_cuda/random_walk_kernel.cuh"
#ifdef MGT_HAS_NCCL
#include "mgt_cuda/allreduce_nccl.cuh"
#endif

#include <cuda_runtime.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace {

struct Args {
    std::filesystem::path output_dir = "runs/native-smoke";
    std::uint32_t steps = 4;
    std::uint32_t device_id = 0;
    std::uint32_t world_size = 1;
    std::uint32_t global_rank = 0;
    std::uint32_t local_rank = 0;
    std::uint32_t batch_size = 64;
    std::uint32_t k_min = 1;
    std::uint32_t k_max = 9;
    std::uint32_t hd1 = 5;
    std::uint32_t hd2 = 3;
    std::filesystem::path nccl_id_file;
};

int Check(cudaError_t status) { return status == cudaSuccess ? 0 : 1; }

std::uint64_t ParamCount(const mgt_cuda::CudaMlpShape& shape) {
    return static_cast<std::uint64_t>(shape.state_len) * shape.state_value_pad * shape.hd1 +
           shape.hd1 + static_cast<std::uint64_t>(shape.hd1) * shape.hd2 + shape.hd2 + shape.hd2 + 1;
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
    return puzzle;
}

bool WriteBinaryWeights(const std::filesystem::path& output_dir,
                        const std::vector<float>& weights,
                        const mgt_cuda::CudaMlpShape& shape) {
    std::filesystem::create_directories(output_dir / "weights");
    std::ofstream data(output_dir / "weights" / "weights.f32.bin", std::ios::binary);
    if (!data) return false;
    data.write(reinterpret_cast<const char*>(weights.data()), static_cast<std::streamsize>(weights.size() * sizeof(float)));
    if (!data) return false;
    std::ofstream manifest(output_dir / "weights" / "manifest.json", std::ios::binary);
    if (!manifest) return false;
    manifest << "{\n"
             << "  \"format\": \"stream1_weights\",\n"
             << "  \"version\": 1,\n"
             << "  \"model_mode\": \"MLP2RB\",\n"
             << "  \"output_dim\": 1,\n"
             << "  \"state_len\": " << shape.state_len << ",\n"
             << "  \"state_value_pad\": " << shape.state_value_pad << ",\n"
             << "  \"hd1\": " << shape.hd1 << ",\n"
             << "  \"hd2\": " << shape.hd2 << ",\n"
             << "  \"dtype\": \"float32\",\n"
             << "  \"data\": \"weights.f32.bin\",\n"
             << "  \"total_params\": " << weights.size() << "\n"
             << "}\n";
    return static_cast<bool>(manifest);
}


bool WriteCheckpoint(const std::filesystem::path& output_dir,
                     const std::vector<float>& weights,
                     const std::vector<float>& adam_m,
                     const std::vector<float>& adam_v,
                     std::uint32_t steps,
                     const mgt_cuda::CudaMlpShape& shape) {
    if (weights.size() != adam_m.size() || weights.size() != adam_v.size()) return false;
    std::filesystem::create_directories(output_dir / "checkpoint");
    std::ofstream data(output_dir / "checkpoint" / "state.f32.bin", std::ios::binary);
    if (!data) return false;
    data.write(reinterpret_cast<const char*>(weights.data()), static_cast<std::streamsize>(weights.size() * sizeof(float)));
    data.write(reinterpret_cast<const char*>(adam_m.data()), static_cast<std::streamsize>(adam_m.size() * sizeof(float)));
    data.write(reinterpret_cast<const char*>(adam_v.data()), static_cast<std::streamsize>(adam_v.size() * sizeof(float)));
    if (!data) return false;
    std::ofstream manifest(output_dir / "checkpoint" / "manifest.json", std::ios::binary);
    if (!manifest) return false;
    manifest << "{\n"
             << "  \"format\": \"mgt_train_checkpoint\",\n"
             << "  \"version\": 1,\n"
             << "  \"step\": " << steps << ",\n"
             << "  \"model_mode\": \"MLP2RB\",\n"
             << "  \"output_dim\": 1,\n"
             << "  \"state_len\": " << shape.state_len << ",\n"
             << "  \"state_value_pad\": " << shape.state_value_pad << ",\n"
             << "  \"hd1\": " << shape.hd1 << ",\n"
             << "  \"hd2\": " << shape.hd2 << ",\n"
             << "  \"optimizer\": \"AdamW\",\n"
             << "  \"weight_decay\": 0,\n"
             << "  \"dtype\": \"float32\",\n"
             << "  \"data\": \"state.f32.bin\",\n"
             << "  \"sections\": [\"weights\", \"adam_m\", \"adam_v\"],\n"
             << "  \"params_per_section\": " << weights.size() << "\n"
             << "}\n";
    return static_cast<bool>(manifest);
}
bool ParseUint(const char* text, std::uint32_t* out) {
    try {
        const unsigned long value = std::stoul(text);
        *out = static_cast<std::uint32_t>(value);
        return value <= 0xffffffffUL;
    } catch (...) {
        return false;
    }
}

bool ParseArgs(int argc, char** argv, Args* args) {
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        if (i + 1 >= argc) return false;
        const char* value = argv[++i];
        if (key == "--output-dir") {
            args->output_dir = value;
        } else if (key == "--steps") {
            if (!ParseUint(value, &args->steps)) return false;
        } else if (key == "--device-id") {
            if (!ParseUint(value, &args->device_id)) return false;
        } else if (key == "--world-size") {
            if (!ParseUint(value, &args->world_size)) return false;
        } else if (key == "--global-rank") {
            if (!ParseUint(value, &args->global_rank)) return false;
        } else if (key == "--local-rank") {
            if (!ParseUint(value, &args->local_rank)) return false;
        } else if (key == "--batch-size") {
            if (!ParseUint(value, &args->batch_size)) return false;
        } else if (key == "--k-min") {
            if (!ParseUint(value, &args->k_min)) return false;
        } else if (key == "--k-max") {
            if (!ParseUint(value, &args->k_max)) return false;
        } else if (key == "--hd1") {
            if (!ParseUint(value, &args->hd1)) return false;
        } else if (key == "--hd2") {
            if (!ParseUint(value, &args->hd2)) return false;
        } else if (key == "--nccl-id-file") {
            args->nccl_id_file = value;
        } else {
            return false;
        }
    }
    return args->steps > 0 && args->world_size > 0 && args->global_rank < args->world_size &&
           args->batch_size > 0 && args->k_min > 0 && args->k_min <= args->k_max &&
           args->hd1 > 0 && args->hd1 <= 64 && args->hd2 > 0 && args->hd2 <= 64;
}

}  // namespace

int main(int argc, char** argv) {
    Args args{};
    if (!ParseArgs(argc, argv, &args)) {
        std::cerr << "usage: mgt_native_train_smoke --output-dir DIR --steps N --device-id ID --world-size N --global-rank R --local-rank R [--batch-size N --k-min N --k-max N --hd1 N --hd2 N --nccl-id-file PATH]\n";
        return EXIT_FAILURE;
    }
    int device_count = 0;
    if (Check(cudaGetDeviceCount(&device_count)) != 0 || device_count <= static_cast<int>(args.device_id)) return EXIT_FAILURE;
    if (Check(cudaSetDevice(static_cast<int>(args.device_id))) != 0) return EXIT_FAILURE;

    const mgt_cuda::CudaMlpShape shape{mgt::kStateLen, 128, args.hd1, args.hd2};
    const std::uint32_t kSamples = args.batch_size;
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

    bool nccl_enabled = false;
#ifdef MGT_HAS_NCCL
    mgt_cuda::NcclRankContext* nccl_context = nullptr;
    if (mgt_cuda::CreateNcclRankContext(args.device_id, args.world_size, args.global_rank, args.nccl_id_file, &nccl_context) != mgt::Status::kOk) {
        return EXIT_FAILURE;
    }
    nccl_enabled = true;
#else
    if (args.world_size != 1) {
        std::cerr << "NCCL is required when world_size > 1\n";
        return EXIT_FAILURE;
    }
#endif

    std::filesystem::create_directories(args.output_dir);
    std::ofstream log(args.output_dir / "train.log");
    log << "rank=" << args.global_rank << " local_rank=" << args.local_rank << " device=" << args.device_id
        << " world_size=" << args.world_size << " phase=start batch_states=" << kSamples
        << " hd1=" << shape.hd1 << " hd2=" << shape.hd2 << " k_min=" << args.k_min << " k_max=" << args.k_max
        << " nccl=" << (nccl_enabled ? 1 : 0) << "\n";

    const mgt_cuda::AdamWKernelConfig adam{params, 1, 0.0001f, 0.9f, 0.999f, 1.0e-8f, 0.0f};
    float last_loss = 0.0f;
    for (std::uint32_t step = 0; step < args.steps; ++step) {
        const mgt_cuda::RandomWalkKernelConfig walks{kSamples, args.k_min, args.k_max};
        if (mgt_cuda::LaunchRandomWalkKernel(walks, 1234, 0, step, args.global_rank, d_moves, d_target, d_states, d_labels, d_meta, 0) != mgt::Status::kOk) return EXIT_FAILURE;
        if (mgt_cuda::LaunchMlpLossGradKernel(shape, d_weights, d_states, d_labels, kSamples, d_loss, d_grad, 0) != mgt::Status::kOk) return EXIT_FAILURE;
#ifdef MGT_HAS_NCCL
        if (nccl_enabled) {
            const mgt::AllreduceConfig allreduce{args.world_size, args.global_rank, step, static_cast<std::size_t>(params)};
            if (mgt_cuda::NcclAllreduceAverageFloat(allreduce, d_grad, nccl_context, 0) != mgt::Status::kOk) return EXIT_FAILURE;
        }
#endif
        if (mgt_cuda::LaunchAdamWKernel(adam, d_weights, d_grad, d_m, d_v, 0) != mgt::Status::kOk) return EXIT_FAILURE;
        if (Check(cudaDeviceSynchronize()) != 0) return EXIT_FAILURE;
        if (Check(cudaMemcpy(&last_loss, d_loss, sizeof(float), cudaMemcpyDeviceToHost)) != 0) return EXIT_FAILURE;
        log << "rank=" << args.global_rank << " step=" << step << " phase=train loss=" << last_loss << "\n";
    }

    std::vector<float> adam_m(params);
    std::vector<float> adam_v(params);
    if (Check(cudaMemcpy(weights.data(), d_weights, params * sizeof(float), cudaMemcpyDeviceToHost)) != 0) return EXIT_FAILURE;
    if (Check(cudaMemcpy(adam_m.data(), d_m, params * sizeof(float), cudaMemcpyDeviceToHost)) != 0) return EXIT_FAILURE;
    if (Check(cudaMemcpy(adam_v.data(), d_v, params * sizeof(float), cudaMemcpyDeviceToHost)) != 0) return EXIT_FAILURE;
    if (!WriteBinaryWeights(args.output_dir, weights, shape)) return EXIT_FAILURE;
    if (!WriteCheckpoint(args.output_dir, weights, adam_m, adam_v, args.steps, shape)) return EXIT_FAILURE;

    std::ofstream meta(args.output_dir / "metadata.env");
    meta << "MODEL_MODE=MLP2RB\nOUTPUT_DIM=1\nWORLD_SIZE=" << args.world_size << "\nGLOBAL_RANK=" << args.global_rank
         << "\nLOCAL_RANK=" << args.local_rank << "\nDEVICE_ID=" << args.device_id << "\nHD1=" << shape.hd1 << "\nHD2=" << shape.hd2
         << "\nK_MIN=" << args.k_min << "\nK_MAX=" << args.k_max << "\nBATCH_SIZE=" << kSamples
         << "\nNCCL_ENABLED=" << (nccl_enabled ? 1 : 0) << "\nNUM_PARAMETERS=" << params << "\n";
    std::ofstream layers(args.output_dir / "layers.json");
    layers << "{\n  \"model_mode\": \"MLP2RB\",\n  \"output_dim\": 1,\n  \"state_len\": 72,\n  \"state_value_pad\": 128,\n  \"hd1\": " << shape.hd1 << ",\n  \"hd2\": " << shape.hd2 << ",\n  \"num_parameters\": " << params << "\n}\n";

#ifdef MGT_HAS_NCCL
    if (nccl_enabled && mgt_cuda::DestroyNcclRankContext(nccl_context) != mgt::Status::kOk) return EXIT_FAILURE;
#endif

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
    std::cout << "native_train_smoke_ok loss=" << last_loss << "\n";
    return EXIT_SUCCESS;
}