#include "mgt_cuda/sync_batch_norm_tiled.cuh"
#include "mgt_cuda/allreduce_nccl.cuh"
#include "mgt_cuda/bf16_batch_norm_epilogue.cuh"
#include <limits>

namespace mgt_cuda { namespace {
constexpr unsigned kWarp = 32, kWarps = 8, kThreads = kWarp * kWarps;

bool ValidConfig(const TiledSyncBatchNormConfig& c) {
    return (c.row_chunk == 256 || c.row_chunk == 512 || c.row_chunk == 1024) &&
           c.feature_tile == 32 &&
           (c.xhat_storage == mgt::A100XhatStorage::kFp32 ||
            c.xhat_storage == mgt::A100XhatStorage::kBf16);
}
std::uint64_t Chunks(unsigned rows, unsigned chunk) {
    return (static_cast<std::uint64_t>(rows) + chunk - 1) / chunk;
}
template<class X> __device__ float ReadXhat(const X* x, std::uint64_t q) {
    return static_cast<float>(x[q]);
}
template<> __device__ float ReadXhat<float>(const float* x, std::uint64_t q) {
    return x[q];
}

__global__ void ForwardPartials(const float* x, unsigned rows, unsigned chunk,
                                unsigned logical, unsigned physical, float* partials) {
    const unsigned lane = threadIdx.x & 31U, warp = threadIdx.x >> 5U;
    const unsigned col = blockIdx.x * kWarp + lane;
    const unsigned begin = blockIdx.y * chunk, end = min(begin + chunk, rows);
    float sum = 0.0f, square = 0.0f;
    if (col < logical) for (unsigned row = begin + warp; row < end; row += kWarps) {
        const float value = x[static_cast<std::uint64_t>(row) * physical + col];
        sum += value;
        square = fmaf(value, value, square);
    }
    __shared__ float sums[kWarps][kWarp], squares[kWarps][kWarp];
    sums[warp][lane] = sum; squares[warp][lane] = square;
    __syncthreads();
    if (warp == 0 && col < logical) {
        sum = square = 0.0f;
#pragma unroll
        for (unsigned w = 0; w < kWarps; ++w) {
            sum += sums[w][lane]; square += squares[w][lane];
        }
        const std::uint64_t base = static_cast<std::uint64_t>(blockIdx.y) * 2 * logical;
        partials[base + col] = sum;
        partials[base + logical + col] = square;
    }
}

template<class X>
__global__ void BackwardPartials(const float* upstream, const unsigned* mask,
                                 const X* xhat, unsigned rows, unsigned chunk,
                                 unsigned logical, unsigned physical, float* partials) {
    const unsigned lane = threadIdx.x & 31U, warp = threadIdx.x >> 5U;
    const unsigned col = blockIdx.x * kWarp + lane;
    const unsigned begin = blockIdx.y * chunk, end = min(begin + chunk, rows);
    const unsigned words = (physical + 31U) / 32U;
    float dgamma = 0.0f, dbeta = 0.0f;
    if (col < logical) for (unsigned row = begin + warp; row < end; row += kWarps) {
        const std::uint64_t q = static_cast<std::uint64_t>(row) * physical + col;
        const bool enabled = ((mask[static_cast<std::uint64_t>(row) * words + col / 32U] >>
                               (col & 31U)) & 1U) != 0;
        const float dy = enabled ? upstream[q] : 0.0f;
        dbeta += dy;
        dgamma = fmaf(dy, ReadXhat(xhat, q), dgamma);
    }
    __shared__ float gamma_sums[kWarps][kWarp], beta_sums[kWarps][kWarp];
    gamma_sums[warp][lane] = dgamma; beta_sums[warp][lane] = dbeta;
    __syncthreads();
    if (warp == 0 && col < logical) {
        dgamma = dbeta = 0.0f;
#pragma unroll
        for (unsigned w = 0; w < kWarps; ++w) {
            dgamma += gamma_sums[w][lane]; dbeta += beta_sums[w][lane];
        }
        const std::uint64_t base = static_cast<std::uint64_t>(blockIdx.y) * 2 * logical;
        partials[base + col] = dgamma;
        partials[base + logical + col] = dbeta;
    }
}

__global__ void ReducePartials(const float* partials, unsigned chunks,
                               unsigned features, float* reduced) {
    const unsigned col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= features) return;
    float first = 0.0f, second = 0.0f;
    for (unsigned chunk = 0; chunk < chunks; ++chunk) {
        const std::uint64_t base = static_cast<std::uint64_t>(chunk) * 2 * features;
        first += partials[base + col]; second += partials[base + features + col];
    }
    reduced[col] = first; reduced[features + col] = second;
}

__global__ void FinalizeMoments(const float* stats, unsigned global_rows,
                                unsigned features, float momentum, float epsilon,
                                float* running_mean, float* running_variance,
                                float* mean, float* inv_std) {
    const unsigned col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= features) return;
    const float m = stats[col] / global_rows;
    const float variance = fmaxf(stats[features + col] / global_rows - m * m, 0.0f);
    mean[col] = m; inv_std[col] = rsqrtf(variance + epsilon);
    running_mean[col] = (1.0f - momentum) * running_mean[col] + momentum * m;
    const float unbiased = global_rows > 1 ? variance * global_rows / (global_rows - 1) : 0.0f;
    running_variance[col] = (1.0f - momentum) * running_variance[col] + momentum * unbiased;
}

__global__ void CopyAffineGrads(const float* stats, unsigned features,
                                float* dgamma, float* dbeta) {
    const unsigned col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < features) { dgamma[col] = stats[col]; dbeta[col] = stats[features + col]; }
}

bool ValidWorkspace(const TiledSyncBatchNormWorkspace& w,
                    std::uint64_t required, unsigned features) {
    return w.partials && w.reduced && w.partials != w.reduced &&
           w.partial_count >= required && w.reduced_count >= 2ULL * features;
}
}  // namespace

std::uint64_t TiledSyncBatchNormPartialFloats(
    unsigned rows, unsigned features, const TiledSyncBatchNormConfig& config) {
    if (!ValidConfig(config) || !rows || !features) return 0;
    const auto chunks = Chunks(rows, config.row_chunk);
    return chunks > std::numeric_limits<std::uint64_t>::max() / (2ULL * features)
        ? 0 : chunks * 2ULL * features;
}

mgt::Status LaunchTiledSyncBatchNormForward(
    const TiledSyncBatchNormConfig& c, const float* x, const __nv_bfloat16* residual,
    unsigned rows, unsigned capacity, unsigned global_rows, unsigned logical,
    unsigned physical, const float* gamma, const float* beta, float* running_mean,
    float* running_variance, float momentum, float epsilon, float* mean, float* inv,
    void* xhat, __nv_bfloat16* activation, unsigned* mask,
    TiledSyncBatchNormWorkspace w, NcclRankContext* context, cudaStream_t stream) {
    const auto required = TiledSyncBatchNormPartialFloats(capacity, logical, c);
    if (!x || !gamma || !beta || !running_mean || !running_variance || !mean || !inv ||
        !xhat || !activation || !mask || !context || !rows || rows > capacity ||
        global_rows < rows || !logical || logical > physical || momentum < 0.0f ||
        momentum > 1.0f || epsilon <= 0.0f || !required || !ValidWorkspace(w, required, logical))
        return mgt::Status::kInvalidConfig;
    const unsigned chunks = static_cast<unsigned>(Chunks(rows, c.row_chunk));
    ForwardPartials<<<dim3((logical + 31) / 32, chunks), kThreads, 0, stream>>>(
        x, rows, c.row_chunk, logical, physical, w.partials);
    ReducePartials<<<(logical + 255) / 256, 256, 0, stream>>>(w.partials, chunks, logical, w.reduced);
    auto status = NcclAllreduceSumFloat(w.reduced, 2ULL * logical, context, stream);
    if (status != mgt::Status::kOk) return status;
    FinalizeMoments<<<(logical + 255) / 256, 256, 0, stream>>>(w.reduced, global_rows,
        logical, momentum, epsilon, running_mean, running_variance, mean, inv);
    if (cudaPeekAtLastError() != cudaSuccess) return mgt::Status::kCudaFailure;
    return LaunchBf16BatchNormForwardEpilogue(x, mean, inv, gamma, beta, residual,
        rows, capacity, logical, physical, c.xhat_storage, xhat, activation, mask, stream);
}

mgt::Status LaunchTiledSyncBatchNormBackward(
    const TiledSyncBatchNormConfig& c, const float* upstream, const unsigned* mask,
    const void* xhat, unsigned rows, unsigned capacity, unsigned global_rows,
    unsigned logical, unsigned physical, const float* gamma, const float* inv,
    float* dgamma, float* dbeta, __nv_bfloat16* dz, float* residual_grad,
    TiledSyncBatchNormWorkspace w, NcclRankContext* context, cudaStream_t stream) {
    const auto required = TiledSyncBatchNormPartialFloats(capacity, logical, c);
    if (!upstream || !mask || !xhat || !gamma || !inv || !dgamma || !dbeta || !dz ||
        !context || !rows || rows > capacity || global_rows < rows || !logical ||
        logical > physical || !required || !ValidWorkspace(w, required, logical))
        return mgt::Status::kInvalidConfig;
    const unsigned chunks = static_cast<unsigned>(Chunks(rows, c.row_chunk));
    const dim3 grid((logical + 31) / 32, chunks);
    if (c.xhat_storage == mgt::A100XhatStorage::kFp32)
        BackwardPartials<<<grid, kThreads, 0, stream>>>(upstream, mask,
            static_cast<const float*>(xhat), rows, c.row_chunk, logical, physical, w.partials);
    else if (c.xhat_storage == mgt::A100XhatStorage::kBf16)
        BackwardPartials<<<grid, kThreads, 0, stream>>>(upstream, mask,
            static_cast<const __nv_bfloat16*>(xhat), rows, c.row_chunk, logical, physical, w.partials);
    else return mgt::Status::kInvalidConfig;
    ReducePartials<<<(logical + 255) / 256, 256, 0, stream>>>(w.partials, chunks, logical, w.reduced);
    auto status = NcclAllreduceSumFloat(w.reduced, 2ULL * logical, context, stream);
    if (status != mgt::Status::kOk) return status;
    CopyAffineGrads<<<(logical + 255) / 256, 256, 0, stream>>>(w.reduced, logical, dgamma, dbeta);
    if (cudaPeekAtLastError() != cudaSuccess) return mgt::Status::kCudaFailure;
    return LaunchBf16BatchNormBackwardEpilogue(upstream, mask, xhat, inv, gamma,
        w.reduced, w.reduced + logical, rows, capacity, global_rows, logical, physical,
        c.xhat_storage, dz, residual_grad, stream);
}
}  // namespace mgt_cuda
