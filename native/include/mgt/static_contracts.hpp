#pragma once

#include "mgt/config.hpp"
#include "mgt/status.hpp"
#include <cstdint>

namespace mgt {

using StateValue = std::uint8_t;

struct alignas(16) TrainState80 {
    StateValue v[kStateStorageLen];
};

struct alignas(16) WalkMeta {
    std::uint32_t depth;
    std::uint32_t last_move;
    std::uint64_t rng_counter;
};

struct alignas(16) LossStats {
    float loss_sum;
    float loss_max;
    std::uint32_t sample_count;
    std::uint32_t overflow;
};

struct alignas(32) TensorBlockHeader {
    std::uint64_t offset_bytes;
    std::uint64_t size_bytes;
    std::uint32_t rows;
    std::uint32_t cols;
    std::uint32_t dtype;
    std::uint32_t reserved;
};

Status ValidateStaticContracts();

}  // namespace mgt