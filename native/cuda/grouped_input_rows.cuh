#pragma once

#include "mgt_cuda/mlp_forward.cuh"

namespace mgt_cuda::detail {
namespace {

constexpr unsigned kGroupedInputRowsThreads = 256;
constexpr unsigned kGroupedInputRowsWarps = kGroupedInputRowsThreads / 32;

// Private builder shared with the isolated exact grouping regression.
// Launch a 1D grid of ceil(bins / kGroupedInputRowsWarps) CTAs, each with
// kGroupedInputRowsThreads threads. Each whole warp owns one bin and emits
// ascending row IDs; the consumer can keep its original FP32 addition order.
__global__ void BuildGroupedInputRows(
    CudaMlpShape s, const mgt::TrainStateStorage* states, unsigned rows,
    unsigned* counts, unsigned* row_ids) {
    const unsigned lane = threadIdx.x & 31U;
    const std::uint64_t bin =
        static_cast<std::uint64_t>(blockIdx.x) * kGroupedInputRowsWarps +
        (threadIdx.x >> 5);
    const std::uint64_t bins =
        static_cast<std::uint64_t>(s.state_len) * s.state_value_pad;
    // This return is warp-uniform, including unused warps in the final CTA.
    if (bin >= bins) return;
    const unsigned position = static_cast<unsigned>(bin / s.state_value_pad);
    const unsigned value = static_cast<unsigned>(bin % s.state_value_pad);
    const unsigned lower_lanes = (1U << lane) - 1U;
    unsigned count = 0;
    for (std::uint64_t base = 0; base < rows; base += 32) {
        const std::uint64_t row = base + lane;
        const bool match = row < rows && states[row].v[position] == value;
        // Tail lanes must participate too. Ballot supplies lane-ordered rank,
        // not a memory barrier; all stores are disjoint and the consumer is
        // launched later in the same stream, after this kernel completes.
        const unsigned mask = __ballot_sync(0xffffffffU, match);
        if (match)
            row_ids[bin * rows + count + __popc(mask & lower_lanes)] =
                static_cast<unsigned>(row);
        count += __popc(mask);
    }
    if (lane == 0) counts[bin] = count;
}

}  // namespace
}  // namespace mgt_cuda::detail
