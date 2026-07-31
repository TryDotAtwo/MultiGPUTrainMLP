#include "mgt/benchmark_snapshot.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

namespace {

bool SameStates(
    const std::vector<mgt::TrainStateStorage>& a,
    const std::vector<mgt::TrainStateStorage>& b) {
    return a.size() == b.size() &&
           std::memcmp(a.data(), b.data(), a.size() * sizeof(a[0])) == 0;
}

bool SameMutableState(
    const mgt::BenchmarkMutableState& a,
    const mgt::BenchmarkMutableState& b) {
    return a.weights == b.weights && a.weight_grad == b.weight_grad &&
           a.weight_m == b.weight_m && a.weight_v == b.weight_v &&
           a.affine == b.affine && a.affine_grad == b.affine_grad &&
           a.affine_m == b.affine_m && a.affine_v == b.affine_v &&
           a.running == b.running;
}

}  // namespace

int main() {
    if (mgt::SplitMix64(0) != UINT64_C(0xe220a8397b1dcdaf) ||
        mgt::SplitMix64(1) != UINT64_C(0x910a2dec89025cc1)) {
        return EXIT_FAILURE;
    }

    const mgt::BenchmarkSnapshotShape shape{
        72, 72, 1, 64, 5};
    mgt::BenchmarkMutableState whole_mutable{};
    std::vector<mgt::TrainStateStorage> whole_states;
    std::vector<float> whole_labels;
    if (mgt::FillBenchmarkSnapshot(
            888, 0, 10, shape, &whole_mutable, &whole_states,
            &whole_labels) != mgt::Status::kOk) {
        return EXIT_FAILURE;
    }

    if (whole_states.size() != 10 || whole_labels.size() != 10 ||
        whole_mutable.weights.size() != 64 ||
        whole_mutable.affine.size() != 10 ||
        whole_mutable.running.size() != 10) {
        return EXIT_FAILURE;
    }
    if (std::all_of(
            whole_mutable.weights.begin(), whole_mutable.weights.end(),
            [](float value) { return value == 0.0f; }) ||
        std::all_of(
            whole_labels.begin(), whole_labels.end(),
            [](float value) { return value == 0.0f; }) ||
        !std::all_of(
            whole_mutable.weight_grad.begin(),
            whole_mutable.weight_grad.end(),
            [](float value) { return value == 0.0f; }) ||
        !std::all_of(
            whole_mutable.affine_grad.begin(),
            whole_mutable.affine_grad.end(),
            [](float value) { return value == 0.0f; })) {
        return EXIT_FAILURE;
    }
    for (const auto& state : whole_states) {
        for (std::uint32_t i = 0; i < shape.state_len; ++i) {
            if (state.v[i] >= shape.state_value_pad) return EXIT_FAILURE;
        }
        for (std::uint32_t i = shape.state_len; i < mgt::kStateStorageLen; ++i) {
            if (state.v[i] != 0) return EXIT_FAILURE;
        }
    }
    for (std::size_t i = shape.batch_norm_feature_count;
         i < whole_mutable.running.size(); ++i) {
        if (!(whole_mutable.running[i] > 0.0f)) return EXIT_FAILURE;
    }

    std::vector<mgt::TrainStateStorage> joined_states;
    std::vector<float> joined_labels;
    constexpr std::array<std::uint32_t, 3> kRows{4, 3, 3};
    std::uint64_t offset = 0;
    for (std::uint32_t rows : kRows) {
        mgt::BenchmarkMutableState part_mutable{};
        std::vector<mgt::TrainStateStorage> part_states;
        std::vector<float> part_labels;
        if (mgt::FillBenchmarkSnapshot(
                888, offset, rows, shape, &part_mutable, &part_states,
                &part_labels) != mgt::Status::kOk ||
            !SameMutableState(whole_mutable, part_mutable)) {
            return EXIT_FAILURE;
        }
        joined_states.insert(
            joined_states.end(), part_states.begin(), part_states.end());
        joined_labels.insert(
            joined_labels.end(), part_labels.begin(), part_labels.end());
        offset += rows;
    }
    if (!SameStates(whole_states, joined_states) ||
        whole_labels != joined_labels) {
        return EXIT_FAILURE;
    }

    mgt::BenchmarkMutableState repeat_mutable{};
    std::vector<mgt::TrainStateStorage> repeat_states;
    std::vector<float> repeat_labels;
    if (mgt::FillBenchmarkSnapshot(
            888, 0, 10, shape, &repeat_mutable, &repeat_states,
            &repeat_labels) != mgt::Status::kOk ||
        !SameMutableState(whole_mutable, repeat_mutable) ||
        !SameStates(whole_states, repeat_states) ||
        whole_labels != repeat_labels) {
        return EXIT_FAILURE;
    }

    mgt::BenchmarkMutableState changed_mutable{};
    std::vector<mgt::TrainStateStorage> changed_states;
    std::vector<float> changed_labels;
    if (mgt::FillBenchmarkSnapshot(
            889, 0, 10, shape, &changed_mutable, &changed_states,
            &changed_labels) != mgt::Status::kOk ||
        SameMutableState(whole_mutable, changed_mutable) ||
        SameStates(whole_states, changed_states) ||
        whole_labels == changed_labels) {
        return EXIT_FAILURE;
    }

    mgt::BenchmarkSnapshotShape invalid = shape;
    invalid.state_value_pad = 0;
    if (mgt::FillBenchmarkSnapshot(
            888, 0, 10, invalid, &changed_mutable, &changed_states,
            &changed_labels) != mgt::Status::kInvalidConfig ||
        mgt::FillBenchmarkSnapshot(
            888, std::numeric_limits<std::uint64_t>::max(), 2, shape,
            &changed_mutable, &changed_states, &changed_labels) !=
            mgt::Status::kCapacityExceeded) {
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
