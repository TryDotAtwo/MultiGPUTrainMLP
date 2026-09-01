#pragma once

#include "mgt_cuda/mlp_forward.cuh"
#include <cuda_fp16.h>

namespace mgt_cuda::detail {

constexpr bool UseInputHalfU32Indexing(int major, int minor) {
    return major == 8 && minor == 6;
}

namespace {

// One CTA owns one (row, feature-tile). Row-major grid.x within a fixed
// grid.y tile favors reusing a weight slice; correctness does not depend on
// CTA scheduling order. Only the even-physical/even-logical path uses this.
template <unsigned Threads>
__global__ void InputHalf2Row(
    CudaMlpShape s, unsigned logical, const __half* w,
    const mgt::TrainStateStorage* states, unsigned rows, float* out) {
    static_assert(Threads == 128 || Threads == 256, "supported gather block sizes");
    __shared__ std::uint64_t offsets[mgt::kStateStorageLen];
    const unsigned row = blockIdx.x;
    if (row >= rows) return;
    if (threadIdx.x < s.state_len)
        offsets[threadIdx.x] =
            (static_cast<std::uint64_t>(threadIdx.x) * s.state_value_pad +
             states[row].v[threadIdx.x]) * s.hd1;
    __syncthreads();
    // Feature-tail lanes may leave only after shared offsets are initialized.
    const std::uint64_t h =
        2ULL * (static_cast<std::uint64_t>(blockIdx.y) * Threads + threadIdx.x);
    if (h >= s.hd1) return;
    const std::uint64_t bias =
        static_cast<std::uint64_t>(s.state_len) * s.state_value_pad * s.hd1;
    float2 value{};
    if (h < logical) {
        value = __half22float2(*reinterpret_cast<const __half2*>(w + bias + h));
        for (unsigned p = 0; p < s.state_len; ++p) {
            const float2 add = __half22float2(
                *reinterpret_cast<const __half2*>(w + offsets[p] + h));
            value.x = __fadd_rn(value.x, add.x);
            value.y = __fadd_rn(value.y, add.y);
        }
    }
    *reinterpret_cast<float2*>(out + static_cast<std::uint64_t>(row) * s.hd1 + h) = value;
}

// Same ownership and arithmetic as InputHalf2Row, for shapes whose complete
// weight/output element ranges fit in u32. The dispatcher must prove the bound.
template <unsigned Threads>
__global__ void InputHalf2Row32(
    CudaMlpShape s, unsigned logical, const __half* w,
    const mgt::TrainStateStorage* states, unsigned rows, float* out) {
    static_assert(Threads == 128, "supported u32 gather block size");
    __shared__ unsigned offsets[mgt::kStateStorageLen];
    const unsigned row = blockIdx.x;
    if (row >= rows) return;
    if (threadIdx.x < s.state_len)
        offsets[threadIdx.x] =
            (threadIdx.x * s.state_value_pad + states[row].v[threadIdx.x]) * s.hd1;
    __syncthreads();
    const unsigned h = 2U * (blockIdx.y * Threads + threadIdx.x);
    if (h >= s.hd1) return;
    const unsigned bias = s.state_len * s.state_value_pad * s.hd1;
    float2 value{};
    if (h < logical) {
        value = __half22float2(*reinterpret_cast<const __half2*>(w + bias + h));
        for (unsigned p = 0; p < s.state_len; ++p) {
            const float2 add = __half22float2(
                *reinterpret_cast<const __half2*>(w + offsets[p] + h));
            value.x = __fadd_rn(value.x, add.x);
            value.y = __fadd_rn(value.y, add.y);
        }
    }
    *reinterpret_cast<float2*>(out + row * s.hd1 + h) = value;
}

}  // namespace
}  // namespace mgt_cuda::detail
