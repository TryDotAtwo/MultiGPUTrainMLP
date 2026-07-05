#include "mgt/puzzle_io.hpp"
#include "mgt/train_plan.hpp"
#include "mgt_cuda/adamw.cuh"
#include "mgt_cuda/mlp_backward.cuh"
#include "mgt_cuda/random_walk_kernel.cuh"
#ifdef MGT_HAS_NCCL
#include "mgt_cuda/allreduce_nccl.cuh"
#endif

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#ifdef MGT_HAS_NVTX
#include <nvtx3/nvToolsExt.h>
#endif
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <cstring>
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
    std::uint32_t group_id = mgt::PuzzleSpec{}.group_id;
    std::uint32_t target_id = mgt::PuzzleSpec{}.target_id;
    std::uint32_t state_len = mgt::PuzzleSpec{}.raw_state_dim;
    std::uint32_t state_value_count = mgt::PuzzleSpec{}.state_value_count;
    std::uint32_t move_count = mgt::PuzzleSpec{}.move_count;
    std::uint32_t state_alignment = mgt::PuzzleSpec{}.state_alignment;
    std::uint32_t batch_size = 64;
    std::uint32_t k_min = 1;
    std::uint32_t k_max = 9;
    std::uint32_t walkers = 0;
    std::uint32_t hd1 = 5;
    std::uint32_t hd2 = 3;
    std::uint32_t nrd = 1;
    std::uint32_t output_dim = 1;
    std::uint32_t hidden_alignment = mgt::kHiddenAlignment;
    std::uint32_t gradient_carousel_slots = mgt::kGradientCarouselSlots;
    std::uint32_t input_grad_partial_chunks = mgt::kInputGradPartialChunks;
    std::uint32_t input_grad_positions_per_block = mgt::kInputGradPositionsPerBlock;
    std::uint32_t input_grad_position_tile = 0;
    bool input_grad_fp16 = true;
    bool linear_fp16 = true;
    bool persistent_half_weights = true;
    std::uint64_t allreduce_bucket_bytes = mgt::kDefaultAllreduceBucketBytes;
    std::uint64_t lt_workspace_bytes = mgt::kDefaultLtWorkspaceBytes;
    std::uint64_t seed = 1234;
    float lr = 0.0001f;
    float weight_decay = 0.0f;
    bool write_artifacts = true;
    bool backward_profile = false;
    bool overlap_allreduce = true;
    std::filesystem::path nccl_id_file;
    std::filesystem::path resume_checkpoint;
};

int Check(cudaError_t status) { return status == cudaSuccess ? 0 : 1; }
__global__ void FloatToHalfMirrorKernel(const float* values, std::uint64_t count, __half* values_half) {
    const std::uint64_t item = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (item >= count) return;
    values_half[item] = __float2half_rn(values[item]);
}

__global__ void ExpandScalarLabelsKernel(const float* scalar_labels, std::uint32_t samples, std::uint32_t output_dim, float* labels) {
    const std::uint64_t item = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::uint64_t total = static_cast<std::uint64_t>(samples) * output_dim;
    if (item >= total) return;
    const std::uint32_t sample = static_cast<std::uint32_t>(item / output_dim);
    labels[item] = scalar_labels[sample];
}

std::uint64_t ResidualBlockParams(const mgt_cuda::CudaMlpShape& shape) {
    return 2ULL * (static_cast<std::uint64_t>(shape.hd2) * shape.hd2 + shape.hd2);
}

std::uint64_t ParamCount(const mgt_cuda::CudaMlpShape& shape) {
    const std::uint64_t input = static_cast<std::uint64_t>(shape.state_len) * shape.state_value_pad * shape.hd1 + shape.hd1;
    const std::uint64_t hidden = static_cast<std::uint64_t>(shape.hd1) * shape.hd2 + shape.hd2;
    const std::uint64_t residual = static_cast<std::uint64_t>(shape.residual_blocks) * ResidualBlockParams(shape);
    return input + hidden + residual + static_cast<std::uint64_t>(shape.hd2) * shape.output_dim + shape.output_dim;
}

const char* InputGradBackendName(const Args& args, const mgt::TrainPlan& train_plan) {
    const std::uint32_t chunks = train_plan.config.runtime.input_grad_partial_chunks;
    const std::uint32_t positions = train_plan.config.runtime.input_grad_positions_per_block;
    if (train_plan.config.runtime.input_grad_position_tile > 0U) {
        return args.input_grad_fp16 ? "position_tiled_one_hot_half_gemm" : "position_tiled_one_hot_float_gemm";
    }
    if (chunks > 1U) return "position_chunked_owner_write";
    if (positions > 1U) return "position_group_owner_write";
    return args.input_grad_fp16 ? "dense_one_hot_half_gemm" : "dense_one_hot_float_gemm";
}

const char* StatusName(mgt::Status status) {
    switch (status) {
        case mgt::Status::kOk: return "ok";
        case mgt::Status::kInvalidConfig: return "invalid_config";
        case mgt::Status::kInvalidPuzzle: return "invalid_puzzle";
        case mgt::Status::kCapacityExceeded: return "capacity_exceeded";
        case mgt::Status::kCudaFailure: return "cuda_failure";
        case mgt::Status::kNcclFailure: return "nccl_failure";
        case mgt::Status::kIoFailure: return "io_failure";
    }
    return "unknown";
}

mgt::PuzzleDefinition BuildPuzzle(const mgt::TrainPlan& plan) {
    mgt::PuzzleDefinition puzzle{};
    const std::uint32_t state_len = plan.config.puzzle.raw_state_dim;
    const std::uint32_t move_count = plan.config.puzzle.move_count;
    const std::uint32_t storage_len = plan.padded_state_dim <= mgt::kStateStorageLen ? plan.padded_state_dim : mgt::kStateStorageLen;
    for (std::uint32_t move = 0; move < mgt::kMoveCount; ++move) {
        for (std::uint32_t i = 0; i < state_len; ++i) {
            puzzle.moves[move].v[i] = static_cast<mgt::StateValue>(move < move_count ? ((i + move + 1) % state_len) : i);
        }
        for (std::uint32_t i = state_len; i < storage_len; ++i) {
            puzzle.moves[move].v[i] = 0;
        }
        for (std::uint32_t i = storage_len; i < mgt::kStateStorageLen; ++i) {
            puzzle.moves[move].v[i] = 0;
        }
    }
    for (std::uint32_t i = 0; i < state_len; ++i) {
        puzzle.target.v[i] = static_cast<mgt::StateValue>(i);
    }
    for (std::uint32_t i = state_len; i < mgt::kStateStorageLen; ++i) {
        puzzle.target.v[i] = 0;
    }
    return puzzle;
}

std::uint32_t RoundUp(std::uint32_t value, std::uint32_t alignment) {
    return ((value + alignment - 1U) / alignment) * alignment;
}

const char* LinearPhaseName(mgt::LinearOpPhase phase) {
    switch (phase) {
        case mgt::LinearOpPhase::kForward: return "forward";
        case mgt::LinearOpPhase::kBackwardWeights: return "backward_weights";
        case mgt::LinearOpPhase::kBackwardInput: return "backward_input";
    }
    return "unknown";
}

const char* MatrixTransposeName(mgt::MatrixTranspose transpose) {
    return transpose == mgt::MatrixTranspose::kYes ? "T" : "N";
}

const char* LinearBackendName(mgt::LinearBackendClass backend) {
    switch (backend) {
        case mgt::LinearBackendClass::kCustomKernel: return "custom_kernel";
        case mgt::LinearBackendClass::kLibraryTensorOp: return "library_tensorop";
        case mgt::LinearBackendClass::kCutlassLargeKReductionCandidate: return "cutlass_large_k_reduction_candidate";
    }
    return "unknown";
}

const char* ParamRoleName(mgt::ParamBlockRole role) {
    switch (role) {
        case mgt::ParamBlockRole::kInputEmbedding: return "input_embedding";
        case mgt::ParamBlockRole::kInputBias: return "input_bias";
        case mgt::ParamBlockRole::kHiddenWeight: return "hidden_weight";
        case mgt::ParamBlockRole::kHiddenBias: return "hidden_bias";
        case mgt::ParamBlockRole::kResidualFc1Weight: return "residual_fc1_weight";
        case mgt::ParamBlockRole::kResidualFc1Bias: return "residual_fc1_bias";
        case mgt::ParamBlockRole::kResidualFc2Weight: return "residual_fc2_weight";
        case mgt::ParamBlockRole::kResidualFc2Bias: return "residual_fc2_bias";
        case mgt::ParamBlockRole::kOutputWeight: return "output_weight";
        case mgt::ParamBlockRole::kOutputBias: return "output_bias";
    }
    return "unknown";
}

bool WriteLinearOpsManifest(const std::filesystem::path& output_dir, const mgt::TrainPlan& train_plan) {
    std::ofstream out(output_dir / "linear_ops.json");
    if (!out) return false;
    out << "{\n  \"format\": \"mgt_linear_ops_v1\",\n";
    out << "  \"op_count\": " << train_plan.linear_ops.size() << ",\n";
    out << "  \"linear_ops\": [\n";
    for (std::size_t i = 0; i < train_plan.linear_ops.size(); ++i) {
        const mgt::LinearOpPlan& op = train_plan.linear_ops[i];
        out << "    {\"op_id\": " << op.op_id
            << ", \"phase\": \"" << LinearPhaseName(op.phase) << "\""
            << ", \"role\": \"" << ParamRoleName(op.role) << "\""
            << ", \"block_index\": " << op.block_index
            << ", \"transpose_a\": \"" << MatrixTransposeName(op.transpose_a) << "\""
            << ", \"transpose_b\": \"" << MatrixTransposeName(op.transpose_b) << "\""
            << ", \"m\": " << op.m
            << ", \"n\": " << op.n
            << ", \"k\": " << op.k
            << ", \"lhs_rows\": " << op.lhs_rows
            << ", \"lhs_cols\": " << op.lhs_cols
            << ", \"rhs_rows\": " << op.rhs_rows
            << ", \"rhs_cols\": " << op.rhs_cols
            << ", \"output_rows\": " << op.output_rows
            << ", \"output_cols\": " << op.output_cols
            << ", \"backend\": \"" << LinearBackendName(op.backend) << "\""
            << ", \"tensor_core_eligible\": " << (op.tensor_core_eligible ? "true" : "false")
            << ", \"logical_tail_masked\": " << (op.logical_tail_masked ? "true" : "false") << "}";
        out << (i + 1U == train_plan.linear_ops.size() ? "\n" : ",\n");
    }
    out << "  ]\n}\n";
    return static_cast<bool>(out);
}
std::uint16_t FloatToFp16(float value) {
    std::uint32_t bits = 0;
    static_assert(sizeof(bits) == sizeof(value));
    std::memcpy(&bits, &value, sizeof(bits));
    const std::uint32_t sign = (bits >> 16U) & 0x8000U;
    std::int32_t exp = static_cast<std::int32_t>((bits >> 23U) & 0xffU) - 127 + 15;
    std::uint32_t mant = bits & 0x7fffffU;
    if (exp <= 0) {
        if (exp < -10) return static_cast<std::uint16_t>(sign);
        mant = (mant | 0x800000U) >> static_cast<std::uint32_t>(1 - exp);
        return static_cast<std::uint16_t>(sign | ((mant + 0x1000U) >> 13U));
    }
    if (exp >= 31) return static_cast<std::uint16_t>(sign | 0x7c00U);
    return static_cast<std::uint16_t>(sign | (static_cast<std::uint32_t>(exp) << 10U) | ((mant + 0x1000U) >> 13U));
}

bool WriteFp16File(const std::filesystem::path& path, const std::vector<float>& values) {
    std::ofstream data(path, std::ios::binary);
    if (!data) return false;
    for (float value : values) {
        const std::uint16_t half = FloatToFp16(value);
        data.write(reinterpret_cast<const char*>(&half), sizeof(half));
    }
    return static_cast<bool>(data);
}

bool WriteStream1Weights(const std::filesystem::path& output_dir,
                         const std::vector<float>& weights,
                         const mgt_cuda::CudaMlpShape& shape,
                         std::uint32_t requested_hd1,
                         std::uint32_t requested_hd2) {
    std::filesystem::create_directories(output_dir / "weights");
    std::ofstream flat(output_dir / "weights" / "weights.f32.bin", std::ios::binary);
    if (!flat) return false;
    flat.write(reinterpret_cast<const char*>(weights.data()), static_cast<std::streamsize>(weights.size() * sizeof(float)));
    if (!flat) return false;

    const std::uint64_t input_table = 0;
    const std::uint64_t input_bias = input_table + static_cast<std::uint64_t>(shape.state_len) * shape.state_value_pad * shape.hd1;
    const std::uint64_t hidden_weight = input_bias + shape.hd1;
    const std::uint64_t hidden_bias = hidden_weight + static_cast<std::uint64_t>(shape.hd1) * shape.hd2;
    const std::uint64_t residual_base = hidden_bias + shape.hd2;
    const std::uint64_t output_weight = residual_base + static_cast<std::uint64_t>(shape.residual_blocks) * ResidualBlockParams(shape);
    const std::uint64_t output_bias = output_weight + static_cast<std::uint64_t>(shape.hd2) * shape.output_dim;
    const std::uint32_t hidden1 = RoundUp(shape.hd1, 8);
    const std::uint32_t hidden2 = RoundUp(shape.hd2, 8);
    const std::uint32_t residual_count = shape.residual_blocks;

    const std::uint32_t export_value_pad = shape.state_value_pad;
    std::vector<float> buffer;
    buffer.assign(static_cast<std::uint64_t>(shape.state_len) * export_value_pad * hidden1, 0.0f);
    for (std::uint32_t pos = 0; pos < shape.state_len; ++pos) {
        for (std::uint32_t val = 0; val < shape.state_value_pad; ++val) {
            const std::uint64_t src_row = input_table + (static_cast<std::uint64_t>(pos) * shape.state_value_pad + val) * shape.hd1;
            const std::uint64_t dst_row = (static_cast<std::uint64_t>(pos) * export_value_pad + val) * hidden1;
            for (std::uint32_t h = 0; h < shape.hd1; ++h) buffer[dst_row + h] = weights[src_row + h];
        }
    }
    if (!WriteFp16File(output_dir / "weights" / "input_weight_hxk.fp16", buffer)) return false;

    buffer.assign(hidden1, 0.0f);
    for (std::uint32_t h = 0; h < shape.hd1; ++h) buffer[h] = weights[input_bias + h];
    if (!WriteFp16File(output_dir / "weights" / "input_bias.fp16", buffer)) return false;

    buffer.assign(static_cast<std::uint64_t>(hidden1) * hidden2, 0.0f);
    for (std::uint32_t h = 0; h < shape.hd1; ++h) {
        for (std::uint32_t j = 0; j < shape.hd2; ++j) {
            buffer[static_cast<std::uint64_t>(h) * hidden2 + j] = weights[hidden_weight + static_cast<std::uint64_t>(h) * shape.hd2 + j];
        }
    }
    if (!WriteFp16File(output_dir / "weights" / "hidden_weight_hxk.fp16", buffer)) return false;

    buffer.assign(hidden2, 0.0f);
    for (std::uint32_t j = 0; j < shape.hd2; ++j) buffer[j] = weights[hidden_bias + j];
    if (!WriteFp16File(output_dir / "weights" / "hidden_bias.fp16", buffer)) return false;

    for (std::uint32_t block = 0; block < residual_count; ++block) {
        const std::uint64_t block_base = residual_base + static_cast<std::uint64_t>(block) * ResidualBlockParams(shape);
        const std::uint64_t fc1w = block_base;
        const std::uint64_t fc1b = fc1w + static_cast<std::uint64_t>(shape.hd2) * shape.hd2;
        const std::uint64_t fc2w = fc1b + shape.hd2;
        const std::uint64_t fc2b = fc2w + static_cast<std::uint64_t>(shape.hd2) * shape.hd2;
        buffer.assign(static_cast<std::uint64_t>(hidden2) * hidden2, 0.0f);
        for (std::uint32_t i = 0; i < shape.hd2; ++i) {
            for (std::uint32_t j = 0; j < shape.hd2; ++j) buffer[static_cast<std::uint64_t>(i) * hidden2 + j] = weights[fc1w + static_cast<std::uint64_t>(i) * shape.hd2 + j];
        }
        if (!WriteFp16File(output_dir / "weights" / ("residual" + std::to_string(block) + "_fc1_weight_hxk.fp16"), buffer)) return false;
        buffer.assign(hidden2, 0.0f);
        for (std::uint32_t j = 0; j < shape.hd2; ++j) buffer[j] = weights[fc1b + j];
        if (!WriteFp16File(output_dir / "weights" / ("residual" + std::to_string(block) + "_fc1_bias.fp16"), buffer)) return false;
        buffer.assign(static_cast<std::uint64_t>(hidden2) * hidden2, 0.0f);
        for (std::uint32_t i = 0; i < shape.hd2; ++i) {
            for (std::uint32_t j = 0; j < shape.hd2; ++j) buffer[static_cast<std::uint64_t>(i) * hidden2 + j] = weights[fc2w + static_cast<std::uint64_t>(i) * shape.hd2 + j];
        }
        if (!WriteFp16File(output_dir / "weights" / ("residual" + std::to_string(block) + "_fc2_weight_hxk.fp16"), buffer)) return false;
        buffer.assign(hidden2, 0.0f);
        for (std::uint32_t j = 0; j < shape.hd2; ++j) buffer[j] = weights[fc2b + j];
        if (!WriteFp16File(output_dir / "weights" / ("residual" + std::to_string(block) + "_fc2_bias.fp16"), buffer)) return false;
    }

    buffer.assign(static_cast<std::uint64_t>(hidden2) * shape.output_dim, 0.0f);
    for (std::uint32_t j = 0; j < shape.hd2; ++j) {
        for (std::uint32_t out = 0; out < shape.output_dim; ++out) {
            buffer[static_cast<std::uint64_t>(j) * shape.output_dim + out] = weights[output_weight + static_cast<std::uint64_t>(j) * shape.output_dim + out];
        }
    }
    if (!WriteFp16File(output_dir / "weights" / "output_weight_hxk.fp16", buffer)) return false;

    buffer.assign(shape.output_dim, 0.0f);
    for (std::uint32_t out = 0; out < shape.output_dim; ++out) buffer[out] = weights[output_bias + out];
    if (!WriteFp16File(output_dir / "weights" / "output_bias.fp16", buffer)) return false;

    std::ofstream manifest(output_dir / "weights" / "manifest.json", std::ios::binary);
    if (!manifest) return false;
    manifest << "{\n"
             << "  \"format\": \"stream1_weights\",\n"
             << "  \"version\": 1,\n"
             << "  \"state_len\": " << shape.state_len << ",\n"
             << "  \"num_classes\": " << shape.state_value_pad << ",\n"
             << "  \"logical_num_classes\": " << shape.state_value_pad << ",\n"
             << "  \"hd1\": " << hidden1 << ",\n"
             << "  \"hd2\": " << hidden2 << ",\n"
             << "  \"original_hd1\": " << requested_hd1 << ",\n"
             << "  \"original_hd2\": " << requested_hd2 << ",\n"
             << "  \"hidden_alignment\": 8,\n"
             << "  \"nrd\": " << residual_count << ",\n"
             << "  \"output_dim\": " << shape.output_dim << ",\n"
             << "  \"dtype\": \"fp16\",\n"
             << "  \"normalization\": \"none\",\n"
             << "  \"layout\": \"row-major input activations times weight_hxk\",\n"
             << "  \"flat_debug_weights\": \"weights.f32.bin\"\n"
             << "}\n";
    return static_cast<bool>(manifest);
}
bool WriteCheckpoint(const std::filesystem::path& output_dir,
                     const std::vector<float>& weights,
                     const std::vector<float>& adam_m,
                     const std::vector<float>& adam_v,
                     std::uint32_t steps,
                     const mgt_cuda::CudaMlpShape& shape,
                     std::uint32_t requested_hd1,
                     std::uint32_t requested_hd2,
                     const mgt::TrainPlan& train_plan) {
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
             << "  \"output_dim\": " << shape.output_dim << ",\n"
             << "  \"state_len\": " << shape.state_len << ",\n"
             << "  \"state_storage_len\": " << train_plan.padded_state_dim << ",\n"
             << "  \"num_classes\": " << shape.state_value_pad << ",\n"
             << "  \"hd1\": " << requested_hd1 << ",\n"
             << "  \"physical_hd1\": " << shape.hd1 << ",\n"
             << "  \"hd2\": " << requested_hd2 << ",\n"
             << "  \"physical_hd2\": " << shape.hd2 << ",\n"
             << "  \"nrd\": " << shape.residual_blocks << ",\n"
             << "  \"optimizer\": \"AdamW\",\n"
             << "  \"weight_decay\": " << train_plan.config.train.weight_decay << ",\n"
             << "  \"dtype\": \"float32\",\n"
             << "  \"data\": \"state.f32.bin\",\n"
             << "  \"sections\": [\"weights\", \"adam_m\", \"adam_v\"],\n"
             << "  \"params_per_section\": " << weights.size() << "\n"
             << "}\n";
    return static_cast<bool>(manifest);
}

bool ReadCheckpoint(const std::filesystem::path& checkpoint_dir,
                    std::vector<float>* weights,
                    std::vector<float>* adam_m,
                    std::vector<float>* adam_v) {
    if (weights == nullptr || adam_m == nullptr || adam_v == nullptr) return false;
    if (weights->size() != adam_m->size() || weights->size() != adam_v->size()) return false;
    std::ifstream data(checkpoint_dir / "state.f32.bin", std::ios::binary);
    if (!data) return false;
    data.read(reinterpret_cast<char*>(weights->data()), static_cast<std::streamsize>(weights->size() * sizeof(float)));
    data.read(reinterpret_cast<char*>(adam_m->data()), static_cast<std::streamsize>(adam_m->size() * sizeof(float)));
    data.read(reinterpret_cast<char*>(adam_v->data()), static_cast<std::streamsize>(adam_v->size() * sizeof(float)));
    return static_cast<bool>(data);
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

bool ParseUint64(const char* text, std::uint64_t* out) {
    try {
        const unsigned long long value = std::stoull(text);
        *out = static_cast<std::uint64_t>(value);
        return true;
    } catch (...) {
        return false;
    }
}

bool ParseFloat(const char* text, float* out) {
    try {
        *out = std::stof(text);
        return true;
    } catch (...) {
        return false;
    }
}

bool ParseBool(const char* text, bool* out) {
    const std::string value = text;
    if (value == "1" || value == "true" || value == "yes" || value == "on") {
        *out = true;
        return true;
    }
    if (value == "0" || value == "false" || value == "no" || value == "off") {
        *out = false;
        return true;
    }
    return false;
}

mgt::TrainConfig BuildTrainConfig(const Args& args) {
    mgt::TrainConfig config = mgt::MakeArchiveP888TrainConfig();
    config.puzzle.group_id = args.group_id;
    config.puzzle.target_id = args.target_id;
    config.puzzle.raw_state_dim = args.state_len;
    config.puzzle.state_value_count = args.state_value_count;
    config.puzzle.move_count = args.move_count;
    config.puzzle.state_alignment = args.state_alignment;
    config.model.hd1 = args.hd1;
    config.model.hd2 = args.hd2;
    config.model.residual_blocks = args.nrd;
    config.model.output_dim = args.output_dim;
    config.model.hidden_alignment = args.hidden_alignment;
    config.train.batch_size = args.batch_size;
    config.train.k_min = args.k_min;
    config.train.k_max = args.k_max;
    config.train.walkers = args.walkers == 0 ? (args.k_max == 0 ? 0 : args.batch_size / args.k_max) : args.walkers;
    config.train.lr = args.lr;
    config.train.weight_decay = args.weight_decay;
    config.train.seed = args.seed;
    config.runtime.device_count = args.world_size;
    config.runtime.devices[0] = args.device_id;
    config.runtime.gradient_carousel_slots = args.gradient_carousel_slots;
    config.runtime.input_grad_partial_chunks = args.input_grad_partial_chunks;
    config.runtime.input_grad_positions_per_block = args.input_grad_positions_per_block;
    config.runtime.input_grad_position_tile = args.input_grad_position_tile;
    config.runtime.lt_workspace_bytes = args.lt_workspace_bytes;
    config.runtime.allreduce_bucket_bytes = args.allreduce_bucket_bytes;
    config.runtime.mode = args.world_size == 1 ? mgt::RuntimeMode::kSingleGpu : mgt::RuntimeMode::kMultiGpu;
    return config;
}

#ifdef MGT_HAS_NCCL
struct GradientCommContext {
    bool enabled = false;
    std::uint32_t world_size = 1;
    std::uint32_t global_rank = 0;
    std::uint32_t step = 0;
    std::uint64_t bucket_params = 0;
    float* gradients = nullptr;
    mgt_cuda::NcclRankContext* nccl_context = nullptr;
    cudaStream_t comm_stream = nullptr;
    std::vector<cudaEvent_t>* ready_events = nullptr;
    std::uint32_t scheduled_ranges = 0;
    std::uint32_t scheduled_chunks = 0;
};

mgt::Status ScheduleGradientAllreduce(void* user,
                                      std::uint32_t ready_id,
                                      std::uint64_t param_offset,
                                      std::uint64_t param_count,
                                      cudaStream_t producer_stream) {
    auto* context = static_cast<GradientCommContext*>(user);
    if (context == nullptr || !context->enabled) return mgt::Status::kOk;
    if (context->gradients == nullptr || context->nccl_context == nullptr || context->comm_stream == nullptr ||
        context->ready_events == nullptr || context->bucket_params == 0 || ready_id >= context->ready_events->size()) {
        return mgt::Status::kInvalidConfig;
    }
    cudaEvent_t ready = (*context->ready_events)[ready_id];
    if (cudaEventRecord(ready, producer_stream) != cudaSuccess) return mgt::Status::kCudaFailure;
    if (cudaStreamWaitEvent(context->comm_stream, ready, 0) != cudaSuccess) return mgt::Status::kCudaFailure;

    std::uint64_t offset = param_offset;
    std::uint64_t remaining = param_count;
    ++context->scheduled_ranges;
    while (remaining > 0) {
        const std::uint64_t chunk = std::min<std::uint64_t>(remaining, context->bucket_params);
        const mgt::AllreduceConfig allreduce{context->world_size, context->global_rank, context->step, static_cast<std::size_t>(chunk)};
        if (mgt_cuda::NcclAllreduceAverageFloat(allreduce, context->gradients + offset, context->nccl_context, context->comm_stream) != mgt::Status::kOk) {
            return mgt::Status::kNcclFailure;
        }
        offset += chunk;
        remaining -= chunk;
        ++context->scheduled_chunks;
    }
    return mgt::Status::kOk;
}
#endif
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
        } else if (key == "--group-id") {
            if (!ParseUint(value, &args->group_id)) return false;
        } else if (key == "--target-id") {
            if (!ParseUint(value, &args->target_id)) return false;
        } else if (key == "--state-len") {
            if (!ParseUint(value, &args->state_len)) return false;
        } else if (key == "--state-value-count") {
            if (!ParseUint(value, &args->state_value_count)) return false;
        } else if (key == "--move-count") {
            if (!ParseUint(value, &args->move_count)) return false;
        } else if (key == "--state-alignment") {
            if (!ParseUint(value, &args->state_alignment)) return false;
        } else if (key == "--batch-size") {
            if (!ParseUint(value, &args->batch_size)) return false;
        } else if (key == "--k-min") {
            if (!ParseUint(value, &args->k_min)) return false;
        } else if (key == "--k-max") {
            if (!ParseUint(value, &args->k_max)) return false;
        } else if (key == "--walkers") {
            if (!ParseUint(value, &args->walkers)) return false;
        } else if (key == "--hd1") {
            if (!ParseUint(value, &args->hd1)) return false;
        } else if (key == "--hd2") {
            if (!ParseUint(value, &args->hd2)) return false;
        } else if (key == "--nrd") {
            if (!ParseUint(value, &args->nrd)) return false;
        } else if (key == "--output-dim") {
            if (!ParseUint(value, &args->output_dim)) return false;
        } else if (key == "--hidden-alignment") {
            if (!ParseUint(value, &args->hidden_alignment)) return false;
        } else if (key == "--gradient-carousel-slots") {
            if (!ParseUint(value, &args->gradient_carousel_slots)) return false;
        } else if (key == "--input-grad-partial-chunks") {
            if (!ParseUint(value, &args->input_grad_partial_chunks)) return false;
        } else if (key == "--input-grad-positions-per-block") {
            if (!ParseUint(value, &args->input_grad_positions_per_block)) return false;
        } else if (key == "--input-grad-position-tile") {
            if (!ParseUint(value, &args->input_grad_position_tile)) return false;
        } else if (key == "--input-grad-fp16") {
            if (!ParseBool(value, &args->input_grad_fp16)) return false;
        } else if (key == "--linear-fp16") {
            if (!ParseBool(value, &args->linear_fp16)) return false;
        } else if (key == "--persistent-half-weights") {
            if (!ParseBool(value, &args->persistent_half_weights)) return false;
        } else if (key == "--allreduce-bucket-bytes") {
            if (!ParseUint64(value, &args->allreduce_bucket_bytes)) return false;
        } else if (key == "--lt-workspace-bytes") {
            if (!ParseUint64(value, &args->lt_workspace_bytes)) return false;
        } else if (key == "--seed") {
            if (!ParseUint64(value, &args->seed)) return false;
        } else if (key == "--lr") {
            if (!ParseFloat(value, &args->lr)) return false;
        } else if (key == "--weight-decay") {
            if (!ParseFloat(value, &args->weight_decay)) return false;
        } else if (key == "--write-artifacts") {
            if (!ParseBool(value, &args->write_artifacts)) return false;
        } else if (key == "--backward-profile") {
            if (!ParseBool(value, &args->backward_profile)) return false;
        } else if (key == "--overlap-allreduce") {
            if (!ParseBool(value, &args->overlap_allreduce)) return false;
        } else if (key == "--nccl-id-file") {
            args->nccl_id_file = value;
        } else if (key == "--resume-checkpoint") {
            args->resume_checkpoint = value;
        } else {
            return false;
        }
    }
    const mgt::TrainConfig config = BuildTrainConfig(*args);
    return args->steps > 0 && args->world_size > 0 && args->global_rank < args->world_size &&
           args->batch_size > 0 && mgt::ValidateTrainConfig(config) == mgt::Status::kOk;
}



}  // namespace

int main(int argc, char** argv) {
    Args args{};
    if (!ParseArgs(argc, argv, &args)) {
        std::cerr << "usage: mgt_native_train_smoke --output-dir DIR --steps N --device-id ID --world-size N --global-rank R --local-rank R [--batch-size N --k-min N --k-max N --hd1 N --hd2 N --nrd N --write-artifacts 0|1 --backward-profile 0|1 --overlap-allreduce 0|1 --input-grad-partial-chunks N --input-grad-positions-per-block N --input-grad-position-tile N --input-grad-fp16 0|1 --linear-fp16 0|1 --persistent-half-weights 0|1 --lt-workspace-bytes N --nccl-id-file PATH --resume-checkpoint DIR]\n";
        return EXIT_FAILURE;
    }
    int device_count = 0;
    if (Check(cudaGetDeviceCount(&device_count)) != 0 || device_count <= static_cast<int>(args.device_id)) return EXIT_FAILURE;
    if (Check(cudaSetDevice(static_cast<int>(args.device_id))) != 0) return EXIT_FAILURE;

    const mgt::TrainConfig train_config = BuildTrainConfig(args);
    mgt::TrainPlan train_plan{};
    if (mgt::BuildTrainPlan(train_config, &train_plan) != mgt::Status::kOk) return EXIT_FAILURE;
    if (!train_plan.compatible_with_legacy_state_storage || train_plan.config.puzzle.raw_state_dim > mgt::kStateLen ||
        train_plan.config.puzzle.state_value_count > mgt::kStateValuePad ||
        train_plan.config.puzzle.move_count > mgt::kMoveCount) {
        std::cerr << "current CUDA smoke path supports puzzles fitting TrainStateStorage only\n";
        return EXIT_FAILURE;
    }

    const std::uint32_t requested_hd1 = train_plan.layout.logical_hd1;
    const std::uint32_t requested_hd2 = train_plan.layout.logical_hd2;
    const mgt_cuda::CudaMlpShape shape{train_plan.layout.state_dim, train_plan.layout.state_value_count, train_plan.layout.physical_hd1, train_plan.layout.physical_hd2, train_plan.layout.residual_blocks, train_plan.layout.output_dim};
    const std::uint32_t kSamples = train_plan.config.train.batch_size;
    const std::uint64_t params = train_plan.layout.total_params;
    if (ParamCount(shape) != params) return EXIT_FAILURE;
    std::vector<float> weights(params);
    std::vector<float> host_adam_m(params, 0.0f);
    std::vector<float> host_adam_v(params, 0.0f);
    for (std::uint64_t i = 0; i < params; ++i) {
        weights[i] = static_cast<float>((static_cast<int>(i % 29) - 14) * 0.0001);
    }
    const bool resumed = !args.resume_checkpoint.empty();
    if (resumed && !ReadCheckpoint(args.resume_checkpoint, &weights, &host_adam_m, &host_adam_v)) return EXIT_FAILURE;

    const mgt::PuzzleDefinition puzzle = BuildPuzzle(train_plan);
    float* d_weights = nullptr;
    __half* d_weights_half = nullptr;
    float* d_labels = nullptr;
    float* d_walk_labels = nullptr;
    float* d_loss = nullptr;
    float* d_grad_carousel = nullptr;
    float* d_grad = nullptr;
    float* d_backward_workspace = nullptr;
    void* d_lt_workspace = nullptr;
    cublasHandle_t backward_blas = nullptr;
    cublasLtHandle_t backward_blas_lt = nullptr;
    cudaStream_t compute_stream = nullptr;
    cudaStream_t comm_stream = nullptr;
    float* d_m = nullptr;
    float* d_v = nullptr;
    mgt::TrainStateStorage* d_states = nullptr;
    mgt::WalkMeta* d_meta = nullptr;
    mgt::TrainStateStorage* d_moves = nullptr;
    mgt::TrainStateStorage* d_target = nullptr;
    if (Check(cudaMalloc(&d_weights, params * sizeof(float))) != 0) return EXIT_FAILURE;
    if (args.linear_fp16 && args.persistent_half_weights && Check(cudaMalloc(&d_weights_half, params * sizeof(__half))) != 0) return EXIT_FAILURE;
    if (Check(cudaMalloc(&d_labels, static_cast<std::uint64_t>(kSamples) * shape.output_dim * sizeof(float))) != 0) return EXIT_FAILURE;
    if (shape.output_dim > 1U && Check(cudaMalloc(&d_walk_labels, kSamples * sizeof(float))) != 0) return EXIT_FAILURE;
    if (Check(cudaMalloc(&d_loss, sizeof(float))) != 0) return EXIT_FAILURE;
    if (Check(cudaMalloc(&d_grad_carousel, params * train_plan.gradient_carousel_slots * sizeof(float))) != 0) return EXIT_FAILURE;
    d_grad = d_grad_carousel;
    const std::uint64_t backward_workspace_floats = mgt_cuda::MlpLossGradWorkspaceFloats(shape, kSamples, train_plan.config.runtime.input_grad_partial_chunks, args.input_grad_fp16, args.linear_fp16);
    if (backward_workspace_floats == 0) return EXIT_FAILURE;
    if (Check(cudaMalloc(&d_backward_workspace, backward_workspace_floats * sizeof(float))) != 0) return EXIT_FAILURE;
    if (args.lt_workspace_bytes > 0 && Check(cudaMalloc(&d_lt_workspace, args.lt_workspace_bytes)) != 0) return EXIT_FAILURE;
    if (cublasCreate(&backward_blas) != CUBLAS_STATUS_SUCCESS) return EXIT_FAILURE;
    if ((args.input_grad_fp16 || args.linear_fp16) && cublasLtCreate(&backward_blas_lt) != CUBLAS_STATUS_SUCCESS) return EXIT_FAILURE;
    if (Check(cudaStreamCreateWithFlags(&compute_stream, cudaStreamNonBlocking)) != 0) return EXIT_FAILURE;
    if (Check(cudaStreamCreateWithFlags(&comm_stream, cudaStreamNonBlocking)) != 0) return EXIT_FAILURE;
    if (Check(cudaMalloc(&d_m, params * sizeof(float))) != 0) return EXIT_FAILURE;
    if (Check(cudaMalloc(&d_v, params * sizeof(float))) != 0) return EXIT_FAILURE;
    if (Check(cudaMalloc(&d_states, kSamples * sizeof(mgt::TrainStateStorage))) != 0) return EXIT_FAILURE;
    if (Check(cudaMalloc(&d_meta, kSamples * sizeof(mgt::WalkMeta))) != 0) return EXIT_FAILURE;
    if (Check(cudaMalloc(&d_moves, mgt::kMoveCount * sizeof(mgt::TrainStateStorage))) != 0) return EXIT_FAILURE;
    if (Check(cudaMalloc(&d_target, sizeof(mgt::TrainStateStorage))) != 0) return EXIT_FAILURE;
    if (Check(cudaMemcpy(d_weights, weights.data(), params * sizeof(float), cudaMemcpyHostToDevice)) != 0) return EXIT_FAILURE;
    if (d_weights_half != nullptr) {
        constexpr std::uint32_t init_threads = 256;
        const std::uint32_t init_blocks = static_cast<std::uint32_t>((params + init_threads - 1ULL) / init_threads);
        FloatToHalfMirrorKernel<<<init_blocks, init_threads, 0, compute_stream>>>(d_weights, params, d_weights_half);
        if (Check(cudaGetLastError()) != 0) return EXIT_FAILURE;
    }
    if (Check(cudaMemcpy(d_moves, puzzle.moves.data(), mgt::kMoveCount * sizeof(mgt::TrainStateStorage), cudaMemcpyHostToDevice)) != 0) return EXIT_FAILURE;
    if (Check(cudaMemcpy(d_target, &puzzle.target, sizeof(mgt::TrainStateStorage), cudaMemcpyHostToDevice)) != 0) return EXIT_FAILURE;
    if (Check(cudaMemcpy(d_m, host_adam_m.data(), params * sizeof(float), cudaMemcpyHostToDevice)) != 0) return EXIT_FAILURE;
    if (Check(cudaMemcpy(d_v, host_adam_v.data(), params * sizeof(float), cudaMemcpyHostToDevice)) != 0) return EXIT_FAILURE;

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
    std::ofstream profile(args.output_dir / "profile.jsonl");
    cudaEvent_t step_start = nullptr;
    cudaEvent_t walk_stop = nullptr;
    cudaEvent_t backward_stop = nullptr;
    cudaEvent_t allreduce_stop = nullptr;
    cudaEvent_t step_stop = nullptr;
    if (Check(cudaEventCreate(&step_start)) != 0) return EXIT_FAILURE;
    if (Check(cudaEventCreate(&walk_stop)) != 0) return EXIT_FAILURE;
    if (Check(cudaEventCreate(&backward_stop)) != 0) return EXIT_FAILURE;
    if (Check(cudaEventCreate(&allreduce_stop)) != 0) return EXIT_FAILURE;
    if (Check(cudaEventCreate(&step_stop)) != 0) return EXIT_FAILURE;
    std::vector<cudaEvent_t> gradient_ready_events;
    cudaEvent_t comm_done_event = nullptr;
#ifdef MGT_HAS_NCCL
    const bool overlap_allreduce = nccl_enabled && args.overlap_allreduce && args.world_size > 1;
    const std::uint32_t expected_gradient_ready_ranges = 2U * shape.residual_blocks + 3U;
    if (overlap_allreduce) {
        gradient_ready_events.resize(expected_gradient_ready_ranges, nullptr);
        for (cudaEvent_t& event : gradient_ready_events) {
            if (Check(cudaEventCreateWithFlags(&event, cudaEventDisableTiming)) != 0) return EXIT_FAILURE;
        }
        if (Check(cudaEventCreateWithFlags(&comm_done_event, cudaEventDisableTiming)) != 0) return EXIT_FAILURE;
    }
#else
    const bool overlap_allreduce = false;
#endif
    const char* input_grad_backend = InputGradBackendName(args, train_plan);
    log << "rank=" << args.global_rank << " local_rank=" << args.local_rank << " device=" << args.device_id
        << " world_size=" << args.world_size << " phase=start batch_states=" << kSamples
        << " hd1=" << requested_hd1 << " physical_hd1=" << shape.hd1 << " hd2=" << requested_hd2 << " physical_hd2=" << shape.hd2 << " nrd=" << shape.residual_blocks << " k_min=" << args.k_min << " k_max=" << args.k_max
        << " input_grad_partial_chunks=" << train_plan.config.runtime.input_grad_partial_chunks
        << " input_grad_positions_per_block=" << train_plan.config.runtime.input_grad_positions_per_block
        << " input_grad_position_tile=" << train_plan.config.runtime.input_grad_position_tile
        << " input_grad_fp16=" << (args.input_grad_fp16 ? 1 : 0)
        << " input_grad_backend=" << input_grad_backend
        << " linear_fp16=" << (args.linear_fp16 ? 1 : 0)
        << " persistent_half_weights=" << (args.persistent_half_weights ? 1 : 0)
        << " lt_workspace_bytes=" << args.lt_workspace_bytes
        << " overlap_allreduce=" << (overlap_allreduce ? 1 : 0)
        << " nccl=" << (nccl_enabled ? 1 : 0) << " resumed=" << (resumed ? 1 : 0) << "\n";

    float last_loss = 0.0f;
    for (std::uint32_t step = 0; step < args.steps; ++step) {
#ifdef MGT_HAS_NVTX
        nvtxRangePushA("mgt_train_step");
#endif
        if (Check(cudaEventRecord(step_start, compute_stream)) != 0) return EXIT_FAILURE;
        const mgt_cuda::RandomWalkKernelConfig walks{kSamples, train_plan.config.train.k_min, train_plan.config.train.k_max, train_plan.config.puzzle.move_count, train_plan.config.puzzle.raw_state_dim, train_plan.padded_state_dim};
        float* walk_labels = shape.output_dim == 1U ? d_labels : d_walk_labels;
        if (mgt_cuda::LaunchRandomWalkKernel(walks, train_plan.config.train.seed, 0, step, args.global_rank, d_moves, d_target, d_states, walk_labels, d_meta, compute_stream) != mgt::Status::kOk) return EXIT_FAILURE;
        if (shape.output_dim > 1U) {
            const std::uint64_t label_items = static_cast<std::uint64_t>(kSamples) * shape.output_dim;
            const std::uint32_t threads = 128;
            const std::uint32_t blocks = static_cast<std::uint32_t>((label_items + threads - 1ULL) / threads);
            ExpandScalarLabelsKernel<<<blocks, threads, 0, compute_stream>>>(d_walk_labels, kSamples, shape.output_dim, d_labels);
            if (Check(cudaGetLastError()) != 0) return EXIT_FAILURE;
        }
        if (Check(cudaEventRecord(walk_stop, compute_stream)) != 0) return EXIT_FAILURE;
        mgt_cuda::MlpBackwardProfile backward_profile{};
        std::uint32_t overlap_ranges = 0;
        std::uint32_t overlap_chunks = 0;
#ifdef MGT_HAS_NCCL
        GradientCommContext comm_context{};
        if (overlap_allreduce) {
            comm_context.enabled = true;
            comm_context.world_size = args.world_size;
            comm_context.global_rank = args.global_rank;
            comm_context.step = step;
            comm_context.bucket_params = std::max<std::uint64_t>(1ULL, train_plan.config.runtime.allreduce_bucket_bytes / sizeof(float));
            comm_context.gradients = d_grad;
            comm_context.nccl_context = nccl_context;
            comm_context.comm_stream = comm_stream;
            comm_context.ready_events = &gradient_ready_events;
        }
#endif
        const mgt::Status backward_status =
#ifdef MGT_HAS_NCCL
            overlap_allreduce
                ? mgt_cuda::LaunchMlpLossGradKernelProfiledWithWorkspaceLtAndCallbackExternalHalf(shape, d_weights, d_weights_half, d_states, d_labels, kSamples, d_loss, d_grad, d_backward_workspace, backward_workspace_floats, backward_blas, backward_blas_lt, train_plan.config.runtime.input_grad_partial_chunks, train_plan.config.runtime.input_grad_positions_per_block, args.input_grad_fp16, args.linear_fp16, args.backward_profile ? &backward_profile : nullptr, ScheduleGradientAllreduce, &comm_context, compute_stream, d_lt_workspace, args.lt_workspace_bytes, train_plan.config.runtime.input_grad_position_tile)
                :
#endif
            (args.backward_profile
                ? mgt_cuda::LaunchMlpLossGradKernelProfiledWithWorkspaceLtExternalHalf(shape, d_weights, d_weights_half, d_states, d_labels, kSamples, d_loss, d_grad, d_backward_workspace, backward_workspace_floats, backward_blas, backward_blas_lt, train_plan.config.runtime.input_grad_partial_chunks, train_plan.config.runtime.input_grad_positions_per_block, args.input_grad_fp16, args.linear_fp16, &backward_profile, compute_stream, d_lt_workspace, args.lt_workspace_bytes, train_plan.config.runtime.input_grad_position_tile)
                : mgt_cuda::LaunchMlpLossGradKernelWithWorkspaceLtExternalHalf(shape, d_weights, d_weights_half, d_states, d_labels, kSamples, d_loss, d_grad, d_backward_workspace, backward_workspace_floats, backward_blas, backward_blas_lt, train_plan.config.runtime.input_grad_partial_chunks, train_plan.config.runtime.input_grad_positions_per_block, args.input_grad_fp16, args.linear_fp16, compute_stream, d_lt_workspace, args.lt_workspace_bytes, train_plan.config.runtime.input_grad_position_tile));
        if (backward_status != mgt::Status::kOk) {
            log << "rank=" << args.global_rank << " step=" << step << " phase=backward status=" << StatusName(backward_status)
                << " status_code=" << static_cast<int>(backward_status) << "\n";
            std::cerr << "backward failed: " << StatusName(backward_status) << " (" << static_cast<int>(backward_status) << ")\n";
            return EXIT_FAILURE;
        }
        if (Check(cudaEventRecord(backward_stop, compute_stream)) != 0) return EXIT_FAILURE;
#ifdef MGT_HAS_NCCL
        if (nccl_enabled) {
            if (overlap_allreduce) {
                if (comm_context.scheduled_ranges != expected_gradient_ready_ranges) return EXIT_FAILURE;
                overlap_ranges = comm_context.scheduled_ranges;
                overlap_chunks = comm_context.scheduled_chunks;
                if (Check(cudaEventRecord(comm_done_event, comm_stream)) != 0) return EXIT_FAILURE;
                if (Check(cudaStreamWaitEvent(compute_stream, comm_done_event, 0)) != 0) return EXIT_FAILURE;
            } else {
                const mgt::AllreduceConfig allreduce{args.world_size, args.global_rank, step, static_cast<std::size_t>(params)};
                if (mgt_cuda::NcclAllreduceAverageFloat(allreduce, d_grad, nccl_context, compute_stream) != mgt::Status::kOk) return EXIT_FAILURE;
                overlap_chunks = args.world_size > 1 ? 1U : 0U;
            }
        }
#endif
        if (Check(cudaEventRecord(allreduce_stop, compute_stream)) != 0) return EXIT_FAILURE;
        const mgt_cuda::AdamWKernelConfig adam{params, static_cast<std::uint64_t>(step) + 1ULL, train_plan.config.train.lr, 0.9f, 0.999f, 1.0e-8f, train_plan.config.train.weight_decay};
        if (d_weights_half != nullptr) {
            if (mgt_cuda::LaunchAdamWKernelWithHalfMirror(adam, d_weights, d_weights_half, d_grad, d_m, d_v, compute_stream) != mgt::Status::kOk) return EXIT_FAILURE;
        } else {
            if (mgt_cuda::LaunchAdamWKernel(adam, d_weights, d_grad, d_m, d_v, compute_stream) != mgt::Status::kOk) return EXIT_FAILURE;
        }
        if (Check(cudaEventRecord(step_stop, compute_stream)) != 0) return EXIT_FAILURE;
        if (Check(cudaEventSynchronize(step_stop)) != 0) return EXIT_FAILURE;
#ifdef MGT_HAS_NVTX
        nvtxRangePop();
#endif
        float step_ms = 0.0f;
        float walk_ms = 0.0f;
        float backward_ms = 0.0f;
        float allreduce_ms = 0.0f;
        float adam_ms = 0.0f;
        if (Check(cudaEventElapsedTime(&step_ms, step_start, step_stop)) != 0) return EXIT_FAILURE;
        if (Check(cudaEventElapsedTime(&walk_ms, step_start, walk_stop)) != 0) return EXIT_FAILURE;
        if (Check(cudaEventElapsedTime(&backward_ms, walk_stop, backward_stop)) != 0) return EXIT_FAILURE;
        if (Check(cudaEventElapsedTime(&allreduce_ms, backward_stop, allreduce_stop)) != 0) return EXIT_FAILURE;
        if (Check(cudaEventElapsedTime(&adam_ms, allreduce_stop, step_stop)) != 0) return EXIT_FAILURE;
        if (Check(cudaMemcpy(&last_loss, d_loss, sizeof(float), cudaMemcpyDeviceToHost)) != 0) return EXIT_FAILURE;
        std::size_t free_bytes = 0;
        std::size_t total_bytes = 0;
        if (Check(cudaMemGetInfo(&free_bytes, &total_bytes)) != 0) return EXIT_FAILURE;
        const std::size_t used_bytes = total_bytes - free_bytes;
        log << "rank=" << args.global_rank << " step=" << step << " phase=train loss=" << last_loss << " step_ms=" << step_ms
            << " walk_ms=" << walk_ms << " backward_ms=" << backward_ms << " allreduce_ms=" << allreduce_ms
            << " adam_ms=" << adam_ms << " memory_bytes=" << used_bytes << "\n";
        profile << "{\"rank\":" << args.global_rank << ",\"device\":" << args.device_id << ",\"step\":" << step
                << ",\"phase\":\"train\",\"milliseconds\":" << std::fixed << std::setprecision(3) << step_ms
                << ",\"walk_ms\":" << walk_ms << ",\"backward_ms\":" << backward_ms
                << ",\"allreduce_ms\":" << allreduce_ms << ",\"adam_ms\":" << adam_ms
                << ",\"batch_states\":" << kSamples << ",\"loss\":" << std::setprecision(6) << last_loss
                << ",\"memory_bytes\":" << used_bytes << ",\"gradient_slots\":" << train_plan.gradient_carousel_slots << ",\"gradient_buckets\":" << train_plan.gradient_buckets.size()
                << ",\"input_grad_fp16\":" << (args.input_grad_fp16 ? 1 : 0) << ",\"input_grad_position_tile\":" << train_plan.config.runtime.input_grad_position_tile << ",\"input_grad_backend\":\"" << input_grad_backend << "\",\"linear_fp16\":" << (args.linear_fp16 ? 1 : 0)
                << ",\"persistent_half_weights\":" << (args.persistent_half_weights ? 1 : 0)
                << ",\"lt_workspace_bytes\":" << args.lt_workspace_bytes
                << ",\"overlap_allreduce\":" << (overlap_allreduce ? 1 : 0) << ",\"overlap_ranges\":" << overlap_ranges << ",\"overlap_chunks\":" << overlap_chunks
                << ",\"bw_input_forward_ms\":" << backward_profile.input_forward_ms
                << ",\"bw_hidden_forward_ms\":" << backward_profile.hidden_forward_ms
                << ",\"bw_residual_forward_ms\":" << backward_profile.residual_forward_ms
                << ",\"bw_output_ms\":" << backward_profile.output_ms
                << ",\"bw_residual_backward_ms\":" << backward_profile.residual_backward_ms
                << ",\"bw_hidden_backward_ms\":" << backward_profile.hidden_backward_ms
                << ",\"bw_input_grad_ms\":" << backward_profile.input_grad_ms << ",\"status\":\"ok\"}\n";
    }

    if (args.write_artifacts) {
        std::vector<float> adam_m(params);
        std::vector<float> adam_v(params);
        if (Check(cudaMemcpy(weights.data(), d_weights, params * sizeof(float), cudaMemcpyDeviceToHost)) != 0) return EXIT_FAILURE;
        if (Check(cudaMemcpy(adam_m.data(), d_m, params * sizeof(float), cudaMemcpyDeviceToHost)) != 0) return EXIT_FAILURE;
        if (Check(cudaMemcpy(adam_v.data(), d_v, params * sizeof(float), cudaMemcpyDeviceToHost)) != 0) return EXIT_FAILURE;
        if (!WriteStream1Weights(args.output_dir, weights, shape, requested_hd1, requested_hd2)) return EXIT_FAILURE;
        if (!WriteCheckpoint(args.output_dir, weights, adam_m, adam_v, args.steps, shape, requested_hd1, requested_hd2, train_plan)) return EXIT_FAILURE;
    }

    std::ofstream meta(args.output_dir / "metadata.env");
    meta << "MODEL_MODE=MLP2RB\nOUTPUT_DIM=" << shape.output_dim << "\nWORLD_SIZE=" << args.world_size << "\nGLOBAL_RANK=" << args.global_rank
         << "\nLOCAL_RANK=" << args.local_rank << "\nDEVICE_ID=" << args.device_id << "\nHD1=" << requested_hd1 << "\nPHYSICAL_HD1=" << shape.hd1 << "\nHD2=" << requested_hd2 << "\nPHYSICAL_HD2=" << shape.hd2 << "\nNUM_CLASSES=" << shape.state_value_pad << "\nNRD=" << shape.residual_blocks
         << "\nK_MIN=" << args.k_min << "\nK_MAX=" << args.k_max << "\nBATCH_SIZE=" << kSamples
         << "\nGRADIENT_CAROUSEL_SLOTS=" << train_plan.gradient_carousel_slots << "\nLT_WORKSPACE_BYTES=" << args.lt_workspace_bytes << "\nINPUT_GRAD_PARTIAL_CHUNKS=" << train_plan.config.runtime.input_grad_partial_chunks << "\nINPUT_GRAD_POSITIONS_PER_BLOCK=" << train_plan.config.runtime.input_grad_positions_per_block << "\nINPUT_GRAD_POSITION_TILE=" << train_plan.config.runtime.input_grad_position_tile << "\nINPUT_GRAD_FP16=" << (args.input_grad_fp16 ? 1 : 0) << "\nINPUT_GRAD_BACKEND=" << input_grad_backend << "\nLINEAR_FP16=" << (args.linear_fp16 ? 1 : 0) << "\nPERSISTENT_HALF_WEIGHTS=" << (args.persistent_half_weights ? 1 : 0) << "\nOVERLAP_ALLREDUCE=" << (overlap_allreduce ? 1 : 0) << "\nGRADIENT_BUCKETS=" << train_plan.gradient_buckets.size() << "\nNCCL_ENABLED=" << (nccl_enabled ? 1 : 0) << "\nRESUME_CHECKPOINT=" << (resumed ? 1 : 0) << "\nARTIFACTS_WRITTEN=" << (args.write_artifacts ? 1 : 0) << "\nNUM_PARAMETERS=" << params << "\n";
    std::ofstream layers(args.output_dir / "layers.json");
    layers << "{\n  \"model_mode\": \"MLP2RB\",\n  \"output_dim\": " << train_plan.layout.output_dim << ",\n  \"state_len\": " << shape.state_len << ",\n  \"state_storage_len\": " << train_plan.padded_state_dim << ",\n  \"num_classes\": " << shape.state_value_pad << ",\n  \"hd1\": " << requested_hd1 << ",\n  \"physical_hd1\": " << shape.hd1 << ",\n  \"hd2\": " << requested_hd2 << ",\n  \"physical_hd2\": " << shape.hd2 << ",\n  \"nrd\": " << shape.residual_blocks << ",\n  \"artifacts_written\": " << (args.write_artifacts ? "true" : "false") << ",\n  \"num_parameters\": " << params << "\n}\n";
    if (!WriteLinearOpsManifest(args.output_dir, train_plan)) return EXIT_FAILURE;

#ifdef MGT_HAS_NCCL
    if (nccl_enabled && mgt_cuda::DestroyNcclRankContext(nccl_context) != mgt::Status::kOk) return EXIT_FAILURE;
#endif

    if (comm_done_event != nullptr) cudaEventDestroy(comm_done_event);
    for (cudaEvent_t event : gradient_ready_events) {
        if (event != nullptr) cudaEventDestroy(event);
    }
    cudaEventDestroy(step_stop);
    cudaEventDestroy(allreduce_stop);
    cudaEventDestroy(backward_stop);
    cudaEventDestroy(walk_stop);
    cudaEventDestroy(step_start);
    cudaFree(d_target);
    cudaFree(d_moves);
    cudaFree(d_meta);
    cudaFree(d_states);
    if (d_walk_labels != nullptr) cudaFree(d_walk_labels);
    cudaFree(d_v);
    cudaFree(d_m);
    if (comm_stream != nullptr) cudaStreamDestroy(comm_stream);
    if (compute_stream != nullptr) cudaStreamDestroy(compute_stream);
    if (backward_blas_lt != nullptr) cublasLtDestroy(backward_blas_lt);
    if (backward_blas != nullptr) cublasDestroy(backward_blas);
    if (d_lt_workspace != nullptr) cudaFree(d_lt_workspace);
    cudaFree(d_backward_workspace);
    cudaFree(d_grad_carousel);
    cudaFree(d_loss);
    cudaFree(d_labels);
    if (d_weights_half != nullptr) cudaFree(d_weights_half);
    cudaFree(d_weights);
    std::cout << "native_train_smoke_ok loss=" << last_loss << "\n";
    return EXIT_SUCCESS;
}
