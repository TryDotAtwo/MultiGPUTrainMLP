#pragma once

#include <cuda_runtime.h>
#include <cstdint>

namespace mgt_cuda::detail {
namespace {

constexpr unsigned kColumnSumThreads = 256;

// Fixed 256-thread CTA. Adjacent lanes read adjacent columns, while each
// thread owns Columns of the original 256 virtual row-lane sums. This keeps
// both the per-lane accumulation and every edge of the old FP32 tree intact.
template <unsigned Columns>
__global__ void ColumnSum(const float* x, unsigned rows, unsigned logical,
                          unsigned stride, float* out) {
    static_assert(Columns && Columns <= 16 && (Columns & (Columns - 1)) == 0,
                  "column tiles must be powers of two in [1,16]");
    constexpr unsigned RowLanes = kColumnSumThreads / Columns;
    const unsigned column_lane = threadIdx.x % Columns;
    const unsigned row_lane = threadIdx.x / Columns;
    const std::uint64_t c = static_cast<std::uint64_t>(blockIdx.x) * Columns + column_lane;
    float z[Columns]{};
    if (c < logical && c < stride) {
        for (std::uint64_t base = 0; base < rows; base += kColumnSumThreads) {
#pragma unroll
            for (unsigned q = 0; q < Columns; ++q) {
                const std::uint64_t r = base + row_lane + q * RowLanes;
                if (r < rows) z[q] = __fadd_rn(z[q], x[r * stride + c]);
            }
        }
    }
    // Old tree offsets >= RowLanes join virtual sums owned by this same
    // thread. Fold those exact pairs in registers, not in a reordered sum.
#pragma unroll
    for (unsigned d = Columns / 2; d; d >>= 1) {
#pragma unroll
        for (unsigned q = 0; q < d; ++q) z[q] = __fadd_rn(z[q], z[q + d]);
    }
    __shared__ float v[kColumnSumThreads];
    v[threadIdx.x] = z[0];
    // Padded column lanes still initialize shared memory and reach every
    // barrier; they never read NaN-poisoned input padding or write past out.
    __syncthreads();
    for (unsigned d = RowLanes / 2; d; d >>= 1) {
        if (row_lane < d)
            v[threadIdx.x] = __fadd_rn(v[threadIdx.x], v[threadIdx.x + d * Columns]);
        __syncthreads();
    }
    if (row_lane == 0 && c < stride) out[c] = v[column_lane];
}

}  // namespace
}  // namespace mgt_cuda::detail
