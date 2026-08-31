#pragma once

// Rejected production-layout experiments retained only as regression coverage.
// See docs/audits/2026-08-31-sparse-gradient-warp-bins.md for measured results.
#include "../../cuda/sparse_input_grad_grouped_rows.cuh"

namespace mgt_cuda::detail {
namespace {

// Same 256-feature tile and bin-major traversal, with two independent columns
// per thread. Launch with 128 threads; never split or reassociate a column sum.
__global__ void SparseInputGradGroupedRowsX2(
    CudaMlpShape s, const float* dz, unsigned rows, const unsigned* counts,
    const unsigned* row_ids, float* grad) {
    const unsigned h0 = blockIdx.y * 256U + threadIdx.x;
    if (h0 >= s.hd1) return;
    const unsigned h1 = h0 + 128U;
    const bool has_h1 = h1 < s.hd1;
    const std::uint64_t bin = blockIdx.x;
    const std::uint64_t list_base = bin * rows;
    const std::uint64_t output_base = bin * s.hd1;
    const unsigned count = counts[bin];
    float sum0 = 0.0f;
    float sum1 = 0.0f;
    for (unsigned i = 0; i < count; ++i) {
        const std::uint64_t row_base =
            static_cast<std::uint64_t>(row_ids[list_base + i]) * s.hd1;
        sum0 += dz[row_base + h0];
        if (has_h1) sum1 += dz[row_base + h1];
    }
    // Empty bins and physical padding are fully overwritten, just as before.
    grad[output_base + h0] = sum0;
    if (has_h1) grad[output_base + h1] = sum1;
}

// Narrower feature stripes, with independent bin groups sharing a 256-thread
// CTA. Each thread still owns one complete serial FP32 sum. No cross-warp
// communication is needed, so partial bins/features can return independently.
template <unsigned FeaturesPerBin>
__global__ void SparseInputGradGroupedRowsWarpBins(
    CudaMlpShape s, const float* dz, unsigned rows, const unsigned* counts,
    const unsigned* row_ids, float* grad) {
    static_assert(FeaturesPerBin == 32 || FeaturesPerBin == 64);
    constexpr unsigned bins_per_block = 256 / FeaturesPerBin;
    const std::uint64_t bin = static_cast<std::uint64_t>(blockIdx.x) * bins_per_block +
                              threadIdx.x / FeaturesPerBin;
    const unsigned h = blockIdx.y * FeaturesPerBin + threadIdx.x % FeaturesPerBin;
    const std::uint64_t bins = static_cast<std::uint64_t>(s.state_len) * s.state_value_pad;
    if (bin >= bins || h >= s.hd1) return;
    float sum = 0.0f;
    const unsigned count = counts[bin];
    for (unsigned i = 0; i < count; ++i)
        sum += dz[static_cast<std::uint64_t>(row_ids[bin * rows + i]) * s.hd1 + h];
    grad[bin * s.hd1 + h] = sum;
}

}  // namespace
}  // namespace mgt_cuda::detail
