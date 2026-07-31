#pragma once

#include "mgt/static_contracts.hpp"

#include <cstdint>
#include <vector>

namespace mgt {

struct BenchmarkSnapshotShape {
    std::uint32_t state_len = 0;
    std::uint32_t state_value_pad = 0;
    std::uint32_t output_dim = 0;
    std::uint64_t linear_parameter_count = 0;
    std::uint64_t batch_norm_feature_count = 0;
};

struct BenchmarkMutableState {
    std::vector<float> weights;
    std::vector<float> weight_grad;
    std::vector<float> weight_m;
    std::vector<float> weight_v;
    std::vector<float> affine;
    std::vector<float> affine_grad;
    std::vector<float> affine_m;
    std::vector<float> affine_v;
    std::vector<float> running;
};

std::uint64_t SplitMix64(std::uint64_t value);
float SignedUnit(std::uint64_t value);

Status FillBenchmarkSnapshot(
    std::uint64_t seed,
    std::uint64_t global_sample_offset,
    std::uint32_t active_rows,
    const BenchmarkSnapshotShape& shape,
    BenchmarkMutableState* mutable_state,
    std::vector<TrainStateStorage>* states,
    std::vector<float>* labels);

}  // namespace mgt
