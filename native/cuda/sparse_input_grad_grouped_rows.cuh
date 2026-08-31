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

}  // namespace
}  // namespace mgt_cuda::detail
