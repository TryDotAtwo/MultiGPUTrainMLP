#include "mgt_cuda/input_embedding_bf16.cuh"

#include <array>
#include <new>
#include <vector>

namespace mgt_cuda {
namespace {
constexpr std::uint32_t kTileCount = 9;
constexpr std::uint32_t kTilePositions = 8;
constexpr std::uint32_t kTileColumns = 576;

__global__ void MaterializeOneHotTile(
    const mgt::TrainStateStorage* states, std::uint32_t rows,
    std::uint32_t first_position, __nv_bfloat16* one_hot) {
    const std::uint64_t q = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::uint64_t count = static_cast<std::uint64_t>(rows) * kTileColumns;
    if (q >= count) return;
    const std::uint32_t row = static_cast<std::uint32_t>(q / kTileColumns);
    const std::uint32_t col = static_cast<std::uint32_t>(q - static_cast<std::uint64_t>(row) * kTileColumns);
    const std::uint32_t tile_position = col / 72U;
    const std::uint32_t value = col - tile_position * 72U;
    const std::uint32_t state_value = states[row].v[first_position + tile_position];
    one_hot[q] = __float2bfloat16(state_value < 72U && state_value == value ? 1.0f : 0.0f);
}

const mgt::A100ArenaSliceV1* FindWorkspace(const A100StaticArenaView& arena) {
    if (!arena.plan) return nullptr;
    for (std::uint32_t i = 0; i < arena.plan->slice_count; ++i)
        if (arena.plan->slices[i].kind == mgt::A100ArenaSliceKind::kLtWorkspace)
            return &arena.plan->slices[i];
    return nullptr;
}

mgt::Status PrepareFixed(cublasLtHandle_t handle,
                         const mgt::Bf16AlgorithmTable& algorithms,
                         const Bf16LinearProblem& problem,
                         mgt::Bf16GemmRole role, float beta,
                         void* workspace, std::uint64_t workspace_bytes,
                         FixedBf16GemmPlan** out) {
    mgt::Bf16GemmKeyV1 key{};
    if (BuildBf16LinearGemmKey(problem, role, beta, &key) != mgt::Status::kOk)
        return mgt::Status::kInvalidConfig;
    const mgt::Bf16GemmChoiceV1* choice = nullptr;
    if (mgt::LookupBf16GemmChoice(algorithms, key, &choice) != mgt::Status::kOk || !choice)
        return mgt::Status::kInvalidConfig;
    return CreateFixedBf16GemmPlan(handle, key, *choice, workspace, workspace_bytes, out);
}
}  // namespace

struct InputEmbeddingBf16Plan {
    struct RowPlan {
        std::uint32_t active_rows = 0;
        std::array<FixedBf16GemmPlan*, kTileCount> forward{};
        std::array<FixedBf16GemmPlan*, kTileCount> table_grad{};
    };
    InputEmbeddingBf16Config config{};
    __nv_bfloat16* one_hot = nullptr;
    std::vector<RowPlan> rows;
};

mgt::Status CreateInputEmbeddingBf16Plan(
    InputEmbeddingBf16Config config, const std::uint32_t* supported_rows,
    std::uint32_t row_count, cublasLtHandle_t handle,
    const mgt::Bf16AlgorithmTable& algorithms, const A100StaticArenaView& arena,
    __nv_bfloat16* one_hot, std::uint64_t one_hot_elements,
    InputEmbeddingBf16Plan** out) {
    if (!out) return mgt::Status::kInvalidConfig;
    *out = nullptr;
    if (!supported_rows || !row_count || !handle || !one_hot ||
        config.state_len != 72 || config.state_value_pad != 72 ||
        config.output_features != 2560 || config.positions_per_tile != 8 ||
        !config.capacity_rows || one_hot_elements < static_cast<std::uint64_t>(config.capacity_rows) * kTileColumns ||
        mgt::ValidateBf16AlgorithmTable(algorithms) != mgt::Status::kOk)
        return mgt::Status::kInvalidConfig;
    const auto* ws = FindWorkspace(arena);
    if (!ws || !arena.ordinary_base || ws->offset > arena.ordinary_bytes ||
        ws->bytes > arena.ordinary_bytes - ws->offset)
        return mgt::Status::kInvalidConfig;
    void* workspace = static_cast<std::uint8_t*>(arena.ordinary_base) + ws->offset;
    auto* plan = new (std::nothrow) InputEmbeddingBf16Plan;
    if (!plan) return mgt::Status::kCapacityExceeded;
    plan->config = config;
    plan->one_hot = one_hot;
    plan->rows.reserve(row_count);
    for (std::uint32_t r = 0; r < row_count; ++r) {
        if (!supported_rows[r] || supported_rows[r] > config.capacity_rows) {
            DestroyInputEmbeddingBf16Plan(plan);
            return mgt::Status::kInvalidConfig;
        }
        InputEmbeddingBf16Plan::RowPlan row{};
        row.active_rows = supported_rows[r];
        for (std::uint32_t tile = 0; tile < kTileCount; ++tile) {
            const Bf16LinearProblem p{row.active_rows, 100U + tile, row.active_rows,
                                     kTileColumns, config.output_features};
            auto status = PrepareFixed(handle, algorithms, p, mgt::Bf16GemmRole::kInputForward,
                                       tile == 0 ? 0.0f : 1.0f, workspace, ws->bytes,
                                       &row.forward[tile]);
            if (status == mgt::Status::kOk)
                status = PrepareFixed(handle, algorithms, p, mgt::Bf16GemmRole::kInputTableGrad,
                                      0.0f, workspace, ws->bytes, &row.table_grad[tile]);
            if (status != mgt::Status::kOk) {
                plan->rows.push_back(row);
                DestroyInputEmbeddingBf16Plan(plan);
                return status;
            }
        }
        plan->rows.push_back(row);
    }
    *out = plan;
    return mgt::Status::kOk;
}

mgt::Status DestroyInputEmbeddingBf16Plan(InputEmbeddingBf16Plan* plan) {
    if (!plan) return mgt::Status::kInvalidConfig;
    mgt::Status result = mgt::Status::kOk;
    for (auto& row : plan->rows) for (std::uint32_t tile = 0; tile < kTileCount; ++tile) {
        if (row.table_grad[tile] && DestroyFixedBf16GemmPlan(row.table_grad[tile]) != mgt::Status::kOk)
            result = mgt::Status::kCudaFailure;
        if (row.forward[tile] && DestroyFixedBf16GemmPlan(row.forward[tile]) != mgt::Status::kOk)
            result = mgt::Status::kCudaFailure;
    }
    delete plan;
    return result;
}

namespace {
const InputEmbeddingBf16Plan::RowPlan* FindRow(const InputEmbeddingBf16Plan* plan,
                                                std::uint32_t active_rows) {
    if (!plan) return nullptr;
    for (const auto& row : plan->rows) if (row.active_rows == active_rows) return &row;
    return nullptr;
}

mgt::Status Materialize(const InputEmbeddingBf16Plan* plan,
                        const mgt::TrainStateStorage* states, std::uint32_t rows,
                        std::uint32_t tile, cudaStream_t stream) {
    const std::uint64_t count = static_cast<std::uint64_t>(rows) * kTileColumns;
    MaterializeOneHotTile<<<static_cast<unsigned>((count + 255) / 256), 256, 0, stream>>>(
        states, rows, tile * kTilePositions, plan->one_hot);
    return cudaPeekAtLastError() == cudaSuccess ? mgt::Status::kOk : mgt::Status::kCudaFailure;
}
}  // namespace

mgt::Status LaunchInputEmbeddingForwardBf16(
    const InputEmbeddingBf16Plan* plan, const mgt::TrainStateStorage* states,
    const __nv_bfloat16* table, std::uint32_t active_rows, float* output,
    cudaStream_t stream) {
    const auto* row = FindRow(plan, active_rows);
    if (!row || !states || !table || !output || !stream) return mgt::Status::kInvalidConfig;
    for (std::uint32_t tile = 0; tile < kTileCount; ++tile) {
        auto status = Materialize(plan, states, active_rows, tile, stream);
        if (status != mgt::Status::kOk) return status;
        status = LaunchFixedBf16Gemm(row->forward[tile], plan->one_hot,
                                    table + static_cast<std::uint64_t>(tile) * kTileColumns * 2560ULL,
                                    output, stream);
        if (status != mgt::Status::kOk) return status;
    }
    return mgt::Status::kOk;
}

mgt::Status LaunchInputEmbeddingTableGradBf16(
    const InputEmbeddingBf16Plan* plan, const mgt::TrainStateStorage* states,
    const __nv_bfloat16* dz, std::uint32_t active_rows, float* table_grad,
    cudaStream_t stream) {
    const auto* row = FindRow(plan, active_rows);
    if (!row || !states || !dz || !table_grad || !stream) return mgt::Status::kInvalidConfig;
    for (std::uint32_t tile = 0; tile < kTileCount; ++tile) {
        auto status = Materialize(plan, states, active_rows, tile, stream);
        if (status != mgt::Status::kOk) return status;
        status = LaunchFixedBf16Gemm(row->table_grad[tile], plan->one_hot, dz,
                                     table_grad + static_cast<std::uint64_t>(tile) * kTileColumns * 2560ULL,
                                     stream);
        if (status != mgt::Status::kOk) return status;
    }
    return mgt::Status::kOk;
}

}  // namespace mgt_cuda