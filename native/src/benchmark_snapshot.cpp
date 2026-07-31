#include "mgt/benchmark_snapshot.hpp"

#include <cmath>
#include <limits>

namespace mgt {
namespace {

constexpr std::uint64_t kGolden = UINT64_C(0x9e3779b97f4a7c15);
constexpr std::uint64_t kStateDomain = UINT64_C(0x243f6a8885a308d3);
constexpr std::uint64_t kLabelDomain = UINT64_C(0x13198a2e03707344);
constexpr std::uint64_t kWeightDomain = UINT64_C(0xa4093822299f31d0);
constexpr std::uint64_t kWeightMDomain = UINT64_C(0x082efa98ec4e6c89);
constexpr std::uint64_t kWeightVDomain = UINT64_C(0x452821e638d01377);
constexpr std::uint64_t kAffineDomain = UINT64_C(0xbe5466cf34e90c6c);
constexpr std::uint64_t kAffineMDomain = UINT64_C(0xc0ac29b7c97c50dd);
constexpr std::uint64_t kAffineVDomain = UINT64_C(0x3f84d5b5b5470917);
constexpr std::uint64_t kRunningDomain = UINT64_C(0x9216d5d98979fb1b);

std::uint64_t Key(
    std::uint64_t seed,
    std::uint64_t domain,
    std::uint64_t index) {
    return seed ^ (domain + kGolden * index);
}

void FillSigned(
    std::vector<float>* values,
    std::uint64_t seed,
    std::uint64_t domain,
    float scale) {
    for (std::uint64_t i = 0; i < values->size(); ++i) {
        (*values)[i] = scale * SignedUnit(Key(seed, domain, i));
    }
}

}  // namespace

std::uint64_t SplitMix64(std::uint64_t value) {
    value += kGolden;
    value = (value ^ (value >> 30U)) * UINT64_C(0xbf58476d1ce4e5b9);
    value = (value ^ (value >> 27U)) * UINT64_C(0x94d049bb133111eb);
    return value ^ (value >> 31U);
}

float SignedUnit(std::uint64_t value) {
    const std::uint32_t bits =
        static_cast<std::uint32_t>(SplitMix64(value) >> 40U);
    return static_cast<float>(bits) * (1.0f / 8388608.0f) - 1.0f;
}

Status FillBenchmarkSnapshot(
    std::uint64_t seed,
    std::uint64_t global_sample_offset,
    std::uint32_t active_rows,
    const BenchmarkSnapshotShape& shape,
    BenchmarkMutableState* mutable_state,
    std::vector<TrainStateStorage>* states,
    std::vector<float>* labels) {
    if (mutable_state == nullptr || states == nullptr || labels == nullptr ||
        active_rows == 0 || shape.state_len == 0 ||
        shape.state_len > kStateLen || shape.state_value_pad == 0 ||
        shape.state_value_pad > 256 || shape.output_dim == 0 ||
        shape.linear_parameter_count == 0 ||
        shape.batch_norm_feature_count == 0) {
        return Status::kInvalidConfig;
    }
    if (global_sample_offset >
        std::numeric_limits<std::uint64_t>::max() - (active_rows - 1ULL)) {
        return Status::kCapacityExceeded;
    }
    if (shape.batch_norm_feature_count >
            std::numeric_limits<std::size_t>::max() / 2ULL ||
        static_cast<std::uint64_t>(active_rows) >
            std::numeric_limits<std::size_t>::max() / shape.output_dim) {
        return Status::kCapacityExceeded;
    }

    const auto linear_count =
        static_cast<std::size_t>(shape.linear_parameter_count);
    const auto bn_features =
        static_cast<std::size_t>(shape.batch_norm_feature_count);
    const auto affine_count = 2ULL * bn_features;
    mutable_state->weights.resize(linear_count);
    mutable_state->weight_grad.assign(linear_count, 0.0f);
    mutable_state->weight_m.resize(linear_count);
    mutable_state->weight_v.resize(linear_count);
    mutable_state->affine.resize(affine_count);
    mutable_state->affine_grad.assign(affine_count, 0.0f);
    mutable_state->affine_m.resize(affine_count);
    mutable_state->affine_v.resize(affine_count);
    mutable_state->running.resize(affine_count);

    FillSigned(&mutable_state->weights, seed, kWeightDomain, 0.02f);
    FillSigned(&mutable_state->weight_m, seed, kWeightMDomain, 0.001f);
    FillSigned(&mutable_state->affine_m, seed, kAffineMDomain, 0.001f);
    for (std::size_t i = 0; i < linear_count; ++i) {
        mutable_state->weight_v[i] =
            0.0005f + 0.0005f *
                std::fabs(SignedUnit(Key(seed, kWeightVDomain, i)));
    }
    for (std::size_t i = 0; i < affine_count; ++i) {
        mutable_state->affine_v[i] =
            0.0005f + 0.0005f *
                std::fabs(SignedUnit(Key(seed, kAffineVDomain, i)));
        const float affine_noise =
            0.02f * SignedUnit(Key(seed, kAffineDomain, i));
        const float running_noise =
            0.05f * SignedUnit(Key(seed, kRunningDomain, i));
        mutable_state->affine[i] =
            i < bn_features ? 1.0f + affine_noise : affine_noise;
        mutable_state->running[i] =
            i < bn_features ? running_noise : 1.0f + running_noise;
    }

    states->assign(active_rows, TrainStateStorage{});
    labels->resize(static_cast<std::size_t>(active_rows) * shape.output_dim);
    for (std::uint32_t row = 0; row < active_rows; ++row) {
        const std::uint64_t global_row = global_sample_offset + row;
        for (std::uint32_t position = 0; position < shape.state_len; ++position) {
            const std::uint64_t index =
                global_row * kStateStorageLen + position;
            (*states)[row].v[position] = static_cast<StateValue>(
                SplitMix64(Key(seed, kStateDomain, index)) %
                shape.state_value_pad);
        }
        for (std::uint32_t output = 0; output < shape.output_dim; ++output) {
            const std::uint64_t index =
                global_row * shape.output_dim + output;
            (*labels)[static_cast<std::size_t>(row) * shape.output_dim + output] =
                SignedUnit(Key(seed, kLabelDomain, index));
        }
    }
    return Status::kOk;
}

}  // namespace mgt
