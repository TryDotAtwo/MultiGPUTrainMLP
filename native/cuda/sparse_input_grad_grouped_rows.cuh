#pragma once

#include "mgt_cuda/mlp_forward.cuh"

namespace mgt_cuda::detail {
namespace {

// Private kernel shared with the isolated numerical regression. Row IDs must
// be in ascending order; each thread owns one output and its complete sum.
__global__ void SparseInputGradGroupedRows(
    CudaMlpShape s, const float* dz, unsigned rows, const unsigned* counts,
    const unsigned* row_ids, float* grad) {
    // Keep bins adjacent for a fixed feature tile to favor dZ reuse in L2.
    // This is a locality hint, never a dependency on CUDA block scheduling.
    const unsigned h = blockIdx.y * blockDim.x + threadIdx.x;
    const std::uint64_t bin = blockIdx.x;
    if (h >= s.hd1) return;
    float sum = 0.0f;
    const unsigned count = counts[bin];
    for (unsigned i = 0; i < count; ++i)
        sum += dz[static_cast<std::uint64_t>(row_ids[bin * rows + i]) * s.hd1 + h];
    grad[bin * s.hd1 + h] = sum;
}

// Two adjacent features keep independent serial FP32 sums while sharing the
// row-ID lookup and address arithmetic. The even-width path uses aligned
// float2 transactions; the odd-width path is a complete, tail-safe fallback
// for the isolated exact regression.
__global__ void SparseInputGradGroupedRowsAdjacent2(
    CudaMlpShape s, const float* dz, unsigned rows, const unsigned* counts,
    const unsigned* row_ids, float* grad) {
    const unsigned h0 = 2U * (blockIdx.y * blockDim.x + threadIdx.x);
    const std::uint64_t bin = blockIdx.x;
    if (h0 >= s.hd1) return;
    const std::uint64_t list_base = bin * rows;
    const std::uint64_t output_base = bin * s.hd1;
    const unsigned count = counts[bin];
    float sum0 = 0.0f;
    float sum1 = 0.0f;
    if ((s.hd1 & 1U) == 0) {
        for (unsigned i = 0; i < count; ++i) {
            const std::uint64_t row_base =
                static_cast<std::uint64_t>(row_ids[list_base + i]) * s.hd1;
            const float2 value = *reinterpret_cast<const float2*>(dz + row_base + h0);
            sum0 += value.x;
            sum1 += value.y;
        }
        *reinterpret_cast<float2*>(grad + output_base + h0) = make_float2(sum0, sum1);
    } else {
        const bool has_h1 = h0 + 1U < s.hd1;
        for (unsigned i = 0; i < count; ++i) {
            const std::uint64_t row_base =
                static_cast<std::uint64_t>(row_ids[list_base + i]) * s.hd1;
            sum0 += dz[row_base + h0];
            if (has_h1) sum1 += dz[row_base + h0 + 1U];
        }
        grad[output_base + h0] = sum0;
        if (has_h1) grad[output_base + h0 + 1U] = sum1;
    }
}

constexpr unsigned kSparseAdjacent2PackedThreadsPerBin = 32;
constexpr unsigned kSparseAdjacent2PackedBinsPerBlock = 4;
constexpr unsigned kSparseAdjacent2PackedThreads =
    kSparseAdjacent2PackedThreadsPerBin * kSparseAdjacent2PackedBinsPerBlock;

// Four independent bins share one CTA. Each warp owns one bin and each lane
// owns two adjacent features. uint16_t row IDs are exact while rows <= 65535;
// the host dispatcher enforces that bound before selecting this kernel.
__global__ void SparseInputGradGroupedRowsAdjacent2PackedU16(
    CudaMlpShape s, const float* dz, unsigned rows, const unsigned* counts,
    const std::uint16_t* row_ids, float* grad) {
    const unsigned bin_in_block =
        threadIdx.x / kSparseAdjacent2PackedThreadsPerBin;
    const unsigned lane = threadIdx.x % kSparseAdjacent2PackedThreadsPerBin;
    const std::uint64_t bin =
        static_cast<std::uint64_t>(blockIdx.x) * kSparseAdjacent2PackedBinsPerBlock +
        bin_in_block;
    const unsigned h0 = 2U *
        (blockIdx.y * kSparseAdjacent2PackedThreadsPerBin + lane);
    const std::uint64_t bins =
        static_cast<std::uint64_t>(s.state_len) * s.state_value_pad;
    if (bin >= bins || h0 >= s.hd1) return;
    const std::uint64_t list_base = bin * rows;
    const std::uint64_t output_base = bin * s.hd1;
    const unsigned count = counts[bin];
    float sum0 = 0.0f;
    float sum1 = 0.0f;
    if ((s.hd1 & 1U) == 0) {
        for (unsigned i = 0; i < count; ++i) {
            const std::uint64_t row_base =
                static_cast<std::uint64_t>(row_ids[list_base + i]) * s.hd1;
            const float2 value = *reinterpret_cast<const float2*>(dz + row_base + h0);
            sum0 += value.x;
            sum1 += value.y;
        }
        *reinterpret_cast<float2*>(grad + output_base + h0) = make_float2(sum0, sum1);
    } else {
        const bool has_h1 = h0 + 1U < s.hd1;
        for (unsigned i = 0; i < count; ++i) {
            const std::uint64_t row_base =
                static_cast<std::uint64_t>(row_ids[list_base + i]) * s.hd1;
            sum0 += dz[row_base + h0];
            if (has_h1) sum1 += dz[row_base + h0 + 1U];
        }
        grad[output_base + h0] = sum0;
        if (has_h1) grad[output_base + h0 + 1U] = sum1;
    }
}

}  // namespace
}  // namespace mgt_cuda::detail
