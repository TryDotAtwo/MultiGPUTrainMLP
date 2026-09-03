#include "../../cuda/grouped_input_rows.cuh"
#include "../../cuda/sparse_input_grad_grouped_rows.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr std::size_t kGuard = 128;
constexpr unsigned char kPoison = 0xa5;
bool cleanup_failed = false;

__global__ void SparseInputGradCompactHalfSerialOracle(
    mgt_cuda::CudaMlpShape shape, const mgt::TrainStateStorage* states,
    const __half* dz, unsigned rows, const std::uint16_t* active_bins,
    unsigned active_count, float* grad) {
    const std::uint64_t index =
        static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::uint64_t count =
        static_cast<std::uint64_t>(active_count) * shape.hd1;
    if (index >= count) return;
    const unsigned active_index = static_cast<unsigned>(index / shape.hd1);
    const unsigned h = static_cast<unsigned>(index -
        static_cast<std::uint64_t>(active_index) * shape.hd1);
    const unsigned bin = active_bins[active_index];
    const unsigned position = bin / shape.state_value_pad;
    const unsigned value = bin - position * shape.state_value_pad;
    float sum = 0.0f;
    for (unsigned row = 0; row < rows; ++row) {
        if (states[row].v[position] == value) {
            sum = __fadd_rn(sum, __half2float(
                dz[static_cast<std::uint64_t>(row) * shape.hd1 + h]));
        }
    }
    grad[static_cast<std::uint64_t>(bin) * shape.hd1 + h] = sum;
}

void Require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

void Cuda(cudaError_t status, const std::string& operation) {
    if (status != cudaSuccess)
        throw std::runtime_error(operation + ": " + cudaGetErrorString(status));
}

template <class T> class DeviceBuffer {
public:
    DeviceBuffer(std::size_t count, const char* name) : count_(count), name_(name) {
        Cuda(cudaMalloc(&allocation_, (count_ + 2 * kGuard) * sizeof(T)),
             name_ + " cudaMalloc");
        Cuda(cudaMemset(allocation_, kPoison, (count_ + 2 * kGuard) * sizeof(T)),
             name_ + " initialize");
    }
    ~DeviceBuffer() {
        if (!allocation_) return;
        const cudaError_t status = cudaFree(allocation_);
        if (status != cudaSuccess) {
            cleanup_failed = true;
            std::fprintf(stderr, "%s cudaFree: %s\n", name_.c_str(),
                         cudaGetErrorString(status));
        }
    }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    T* get() const { return allocation_ + kGuard; }
    void Fill(unsigned char byte) {
        Cuda(cudaMemset(get(), byte, count_ * sizeof(T)), name_ + " fill");
    }
    void Put(const std::vector<T>& values) {
        Require(values.size() == count_, name_ + " upload size");
        Cuda(cudaMemcpy(get(), values.data(), count_ * sizeof(T), cudaMemcpyHostToDevice),
             name_ + " upload");
    }
    std::vector<T> Read() const {
        CheckGuards();
        std::vector<T> values(count_);
        Cuda(cudaMemcpy(values.data(), get(), count_ * sizeof(T), cudaMemcpyDeviceToHost),
             name_ + " download");
        return values;
    }
    void CheckGuards() const {
        std::vector<unsigned char> bytes(2 * kGuard * sizeof(T));
        Cuda(cudaMemcpy(bytes.data(), allocation_, kGuard * sizeof(T),
                        cudaMemcpyDeviceToHost), name_ + " leading guard");
        Cuda(cudaMemcpy(bytes.data() + kGuard * sizeof(T), get() + count_,
                        kGuard * sizeof(T), cudaMemcpyDeviceToHost),
             name_ + " trailing guard");
        for (unsigned char byte : bytes)
            Require(byte == kPoison, name_ + " canary changed");
    }

private:
    std::size_t count_;
    std::string name_;
    T* allocation_ = nullptr;
};

class Harness {
public:
    Harness()
        : bins_(shape_.state_len * shape_.state_value_pad),
          elements_(static_cast<std::size_t>(bins_) * shape_.hd1),
          states_(capacity_rows_, "states"),
          dz_(static_cast<std::size_t>(capacity_rows_) * shape_.hd1, "dz"),
          dz_half_(static_cast<std::size_t>(capacity_rows_) * shape_.hd1, "dz half"),
          active_(active_bins_.size(), "active bins"),
          full_counts_(bins_, "full counts"),
          full_rows_(static_cast<std::size_t>(bins_) * capacity_rows_, "full rows"),
          compact_counts_(active_bins_.size(), "compact counts"),
          compact_rows_(active_bins_.size() * capacity_rows_, "compact rows"),
          baseline_(elements_, "baseline"), candidate_(elements_, "candidate") {
        active_.Put(active_bins_);
    }

    void RunOwnershipWitness() {
        const unsigned rows = 33;
        const auto states = MakeStates(5);
        PutInputs(states, MakeDz(5));
        LaunchFull(rows);
        candidate_.Fill(kPoison);
        LaunchCompact(rows);
        Cuda(cudaDeviceSynchronize(), "ownership synchronize");
        const auto expected = baseline_.Read();
        const auto actual = candidate_.Read();
        std::vector<bool> active_mask(bins_, false);
        for (std::uint16_t bin : active_bins_) active_mask[bin] = true;
        for (unsigned bin = 0; bin < bins_; ++bin) {
            for (unsigned h = 0; h < shape_.hd1; ++h) {
                const std::size_t index = static_cast<std::size_t>(bin) * shape_.hd1 + h;
                if (active_mask[bin]) {
                    Require(Bits(actual[index]) == Bits(expected[index]),
                            "active output mismatch in ownership witness");
                } else {
                    Require(Bits(actual[index]) == 0xa5a5a5a5U,
                            "compact consumer touched inactive output");
                }
            }
        }
        candidate_.Fill(0);
        LaunchCompact(rows);
        Cuda(cudaDeviceSynchronize(), "zero invariant synchronize");
        EqualOutputs(expected, candidate_.Read(), "initial persistent-zero result");
        CheckRows(states, rows);
    }

    void RunHalfMirrorWitness() {
        const unsigned rows = 33;
        const auto states = MakeStates(29);
        auto dz = MakeDz(31);
        for (std::size_t index = 0; index < dz.size(); ++index) {
            const int numerator = static_cast<int>((index * 23U + 7U) % 29U) - 14;
            dz[index] += static_cast<float>(numerator) * 0.1f;
        }
        std::vector<__half> expected_half(dz.size());
        for (std::size_t index = 0; index < dz.size(); ++index)
            expected_half[index] = __float2half_rn(dz[index]);
        PutInputs(states, dz);
        baseline_.Fill(kPoison);
        candidate_.Fill(kPoison);
        LaunchHalfOracle(rows);
        LaunchCompactHalf(rows);
        Cuda(cudaDeviceSynchronize(), "half mirror witness synchronize");
        EqualOutputs(baseline_.Read(), candidate_.Read(),
                     "RN-half serial sparse result");
        candidate_.Fill(kPoison);
        LaunchCompactHalfU32(rows);
        Cuda(cudaDeviceSynchronize(), "half u32 witness synchronize");
        EqualOutputs(baseline_.Read(), candidate_.Read(),
                     "RN-half u32 serial sparse result");
        const auto actual_half = dz_half_.Read();
        Require(std::memcmp(expected_half.data(), actual_half.data(),
                            expected_half.size() * sizeof(__half)) == 0,
                "half mirror input mutated");
        CheckRows(states, rows);
        candidate_.Fill(0);
    }

    void RunReuse(unsigned rows, unsigned salt) {
        const auto states = MakeStates(salt);
        PutInputs(states, MakeDz(salt));
        LaunchFull(rows);
        // Deliberately no memset: active bins must be owner-written and every
        // impossible bin must retain the trainer's initial zero forever.
        LaunchCompact(rows);
        Cuda(cudaDeviceSynchronize(), "reuse synchronize");
        EqualOutputs(baseline_.Read(), candidate_.Read(),
                     "persistent-zero changed-row reuse");
        CheckRows(states, rows);
    }

    void RunHalfU32Reuse(unsigned rows, unsigned salt) {
        const auto states = MakeStates(salt);
        PutInputs(states, MakeDz(salt));
        baseline_.Fill(kPoison);
        candidate_.Fill(kPoison);
        LaunchHalfOracle(rows);
        LaunchCompactHalfU32(rows);
        Cuda(cudaDeviceSynchronize(), "half u32 reuse synchronize");
        EqualOutputs(baseline_.Read(), candidate_.Read(),
                     "half u32 changed-row reuse");
        CheckRows(states, rows);
        candidate_.Fill(0);
    }

    void RunFusedAdamWitness(unsigned rows, unsigned salt) {
        constexpr std::uint64_t tail_count = 37;
        const std::uint64_t total_count = elements_ + tail_count;
        const std::uint64_t active_elements =
            active_bins_.size() * shape_.hd1;
        const std::uint64_t live_count = active_elements + tail_count;
        const auto states = MakeStates(salt);
        PutInputs(states, MakeDz(salt + 41U));
        BuildCompact(rows);

        std::vector<float> weights(total_count), grad(total_count, 0.0f);
        std::vector<float> moment1(total_count, 0.0f);
        std::vector<float> moment2(total_count, 0.0f);
        std::vector<__half> half(total_count);
        std::vector<bool> active_mask(bins_, false);
        for (std::uint16_t bin : active_bins_) active_mask[bin] = true;
        for (std::uint64_t i = 0; i < total_count; ++i) {
            weights[i] = static_cast<float>(
                static_cast<int>((i * 17U + salt) % 43U) - 21) * .0013f;
            half[i] = __float2half_rn(weights[i]);
            const bool active_input = i < elements_ &&
                active_mask[i / shape_.hd1];
            const bool tail = i >= elements_;
            if (active_input || tail) {
                grad[i] = static_cast<float>(
                    static_cast<int>((i * 19U + salt) % 31U) - 15) * .00017f;
                moment1[i] = static_cast<float>(
                    static_cast<int>((i * 23U + salt) % 29U) - 14) * .00011f;
                moment2[i] = static_cast<float>((i * 13U + salt) % 37U) *
                    .000006f;
            }
        }

        DeviceBuffer<float> reference_weights(total_count, "reference weights");
        DeviceBuffer<__half> reference_half(total_count, "reference half");
        DeviceBuffer<float> reference_grad(total_count, "reference grad");
        DeviceBuffer<float> reference_m(total_count, "reference m");
        DeviceBuffer<float> reference_v(total_count, "reference v");
        DeviceBuffer<float> fused_weights(total_count, "fused weights");
        DeviceBuffer<__half> fused_half(total_count, "fused half");
        DeviceBuffer<float> fused_grad(total_count, "fused grad");
        DeviceBuffer<float> fused_m(total_count, "fused m");
        DeviceBuffer<float> fused_v(total_count, "fused v");
        for (auto* buffer : {&reference_weights, &fused_weights})
            buffer->Put(weights);
        for (auto* buffer : {&reference_half, &fused_half}) buffer->Put(half);
        for (auto* buffer : {&reference_grad, &fused_grad}) buffer->Put(grad);
        for (auto* buffer : {&reference_m, &fused_m}) buffer->Put(moment1);
        for (auto* buffer : {&reference_v, &fused_v}) buffer->Put(moment2);

        mgt_cuda::AdamWKernelConfig sparse{
            live_count, 1, .0001f, .9f, .999f, 1e-8f, 0.0f};
        sparse.sparse_active_bins = active_.get();
        sparse.sparse_full_prefix_count = elements_;
        sparse.sparse_row_width = shape_.hd1;
        sparse.sparse_active_bin_count =
            static_cast<std::uint32_t>(active_bins_.size());
        auto fused = sparse;
        fused.param_count = active_elements;
        fused.sparse_active_bins = nullptr;
        fused.sparse_full_prefix_count = 0;
        fused.sparse_row_width = 0;
        fused.sparse_active_bin_count = 0;
        auto tail = fused;
        tail.param_count = tail_count;
        tail.dense_physical_offset = elements_;
        std::uint64_t physical_count = 0;
        Require(mgt_cuda::QueryAdamWPhysicalParameterCount(
                    sparse, &physical_count) == mgt::Status::kOk &&
                    physical_count == total_count,
                "sparse Adam physical extent");
        Require(mgt_cuda::QueryAdamWPhysicalParameterCount(
                    tail, &physical_count) == mgt::Status::kOk &&
                    physical_count == total_count,
                "tail Adam physical extent");

        const unsigned active_count =
            static_cast<unsigned>(active_bins_.size());
        const dim3 grid(
            (active_count +
             mgt_cuda::detail::kSparseCompactActiveHalfU32BinsPerBlock - 1U) /
                mgt_cuda::detail::kSparseCompactActiveHalfU32BinsPerBlock,
            (shape_.hd1 / 2U +
             mgt_cuda::detail::kSparseCompactActiveThreadsPerBin - 1U) /
                mgt_cuda::detail::kSparseCompactActiveThreadsPerBin);
        for (std::uint64_t step : {1ULL, 2ULL, 997ULL, 65535ULL}) {
            sparse.step = step;
            fused.step = step;
            tail.step = step;
            mgt_cuda::detail::
                SparseInputGradCompactActiveAdjacent2PackedHalfU16U32
                <<<grid,
                   mgt_cuda::detail::kSparseCompactActiveHalfU32Threads>>>(
                    shape_, dz_half_.get(), rows, active_.get(), active_count,
                    compact_counts_.get(), compact_rows_.get(),
                    reference_grad.get());
            Require(mgt_cuda::LaunchAdamWKernelWithHalfMirror(
                        sparse, reference_weights.get(), reference_half.get(),
                        reference_grad.get(), reference_m.get(),
                        reference_v.get(), nullptr) == mgt::Status::kOk,
                    "reference sparse Adam launch");
            mgt_cuda::detail::
                SparseInputGradCompactActiveAdjacent2PackedHalfU16U32AdamW
                <<<grid,
                   mgt_cuda::detail::kSparseCompactActiveHalfU32Threads>>>(
                    fused, shape_, dz_half_.get(), rows, active_.get(),
                    active_count, compact_counts_.get(), compact_rows_.get(),
                    fused_grad.get(), fused_weights.get(), fused_half.get(),
                    fused_m.get(), fused_v.get());
            Require(mgt_cuda::LaunchAdamWKernelWithHalfMirror(
                        tail, fused_weights.get(), fused_half.get(),
                        fused_grad.get(), fused_m.get(), fused_v.get(),
                        nullptr) == mgt::Status::kOk,
                    "dense tail Adam launch");
            Cuda(cudaGetLastError(), "fused Adam launches");
        }
        Cuda(cudaDeviceSynchronize(), "fused Adam synchronize");
        const auto same = [](const auto& left, const auto& right,
                             const char* field) {
            const auto a = left.Read();
            const auto b = right.Read();
            Require(a.size() == b.size() &&
                        std::memcmp(a.data(), b.data(),
                                    a.size() * sizeof(a[0])) == 0,
                    std::string("fused Adam byte mismatch: ") + field);
        };
        same(reference_weights, fused_weights, "weights");
        same(reference_half, fused_half, "half mirror");
        same(reference_grad, fused_grad, "gradient");
        same(reference_m, fused_m, "moment1");
        same(reference_v, fused_v, "moment2");
        const auto final_grad = fused_grad.Read();
        const auto final_m = fused_m.Read();
        const auto final_v = fused_v.Read();
        const auto final_weights = fused_weights.Read();
        const auto final_half = fused_half.Read();
        for (unsigned bin = 0; bin < bins_; ++bin) {
            if (active_mask[bin]) continue;
            for (unsigned h = 0; h < shape_.hd1; ++h) {
                const std::size_t index =
                    static_cast<std::size_t>(bin) * shape_.hd1 + h;
                Require(Bits(final_grad[index]) == 0 &&
                            Bits(final_m[index]) == 0 &&
                            Bits(final_v[index]) == 0 &&
                            Bits(final_weights[index]) == Bits(weights[index]) &&
                            std::memcmp(&final_half[index], &half[index],
                                        sizeof(__half)) == 0,
                        "fused Adam touched pristine inactive state");
            }
        }
        CheckRows(states, rows);
    }

    void Finish() {
        Require(active_.Read() == active_bins_, "active-bin map mutated");
        states_.CheckGuards();
        dz_.CheckGuards();
        dz_half_.CheckGuards();
        full_counts_.CheckGuards();
        full_rows_.CheckGuards();
        compact_counts_.CheckGuards();
        compact_rows_.CheckGuards();
        baseline_.CheckGuards();
        candidate_.CheckGuards();
    }

private:
    static std::uint32_t Bits(float value) {
        std::uint32_t bits = 0;
        std::memcpy(&bits, &value, sizeof(bits));
        return bits;
    }

    std::vector<mgt::TrainStateStorage> MakeStates(unsigned salt) const {
        std::vector<mgt::TrainStateStorage> states(capacity_rows_);
        constexpr unsigned values[3][3] = {{0, 2, 0}, {1, 4, 1}, {0, 3, 4}};
        constexpr unsigned sizes[3] = {2, 2, 3};
        for (unsigned row = 0; row < capacity_rows_; ++row) {
            std::memset(states[row].v, 0xff, sizeof(states[row].v));
            for (unsigned position = 0; position < shape_.state_len; ++position)
                states[row].v[position] = static_cast<mgt::StateValue>(
                    values[position][(row * 5U + position + salt) % sizes[position]]);
        }
        return states;
    }

    std::vector<float> MakeDz(unsigned salt) const {
        std::vector<float> dz(static_cast<std::size_t>(capacity_rows_) * shape_.hd1);
        for (unsigned row = 0; row < capacity_rows_; ++row) {
            for (unsigned h = 0; h < shape_.hd1; ++h) {
                const int magnitude = static_cast<int>((row * 17U + h * 13U + salt) % 19U) - 9;
                const float scale = (row + h + salt) % 7U == 0 ? 4096.0f : 0.000244140625f;
                dz[static_cast<std::size_t>(row) * shape_.hd1 + h] = magnitude * scale;
            }
        }
        return dz;
    }

    void PutInputs(const std::vector<mgt::TrainStateStorage>& states,
                   const std::vector<float>& dz) {
        states_.Put(states);
        dz_.Put(dz);
        std::vector<__half> dz_half(dz.size());
        for (std::size_t index = 0; index < dz.size(); ++index)
            dz_half[index] = __float2half_rn(dz[index]);
        dz_half_.Put(dz_half);
        full_counts_.Fill(kPoison);
        full_rows_.Fill(0xff);
        compact_counts_.Fill(kPoison);
        compact_rows_.Fill(0xff);
        baseline_.Fill(0);
    }

    void LaunchFull(unsigned rows) {
        const unsigned builder_blocks =
            (bins_ + mgt_cuda::detail::kGroupedInputRowsWarps - 1U) /
            mgt_cuda::detail::kGroupedInputRowsWarps;
        mgt_cuda::detail::BuildGroupedInputRows16
            <<<builder_blocks, mgt_cuda::detail::kGroupedInputRowsThreads>>>(
                shape_, states_.get(), rows, full_counts_.get(), full_rows_.get());
        const dim3 grid(
            (bins_ + mgt_cuda::detail::kSparseAdjacent2PackedBinsPerBlock - 1U) /
                mgt_cuda::detail::kSparseAdjacent2PackedBinsPerBlock,
            (shape_.hd1 / 2U +
             mgt_cuda::detail::kSparseAdjacent2PackedThreadsPerBin - 1U) /
                mgt_cuda::detail::kSparseAdjacent2PackedThreadsPerBin);
        mgt_cuda::detail::SparseInputGradGroupedRowsAdjacent2PackedU16
            <<<grid, mgt_cuda::detail::kSparseAdjacent2PackedThreads>>>(
                shape_, dz_.get(), rows, full_counts_.get(), full_rows_.get(),
                baseline_.get());
        Cuda(cudaGetLastError(), "full launch");
    }

    void BuildCompact(unsigned rows) {
        const unsigned active_count = static_cast<unsigned>(active_bins_.size());
        const unsigned builder_blocks =
            (active_count + mgt_cuda::detail::kCompactActiveRowsWarps - 1U) /
            mgt_cuda::detail::kCompactActiveRowsWarps;
        mgt_cuda::detail::BuildCompactActiveRows16
            <<<builder_blocks, mgt_cuda::detail::kCompactActiveRowsThreads>>>(
                shape_, states_.get(), rows, active_.get(), active_count,
                compact_counts_.get(), compact_rows_.get());
    }

    void LaunchCompact(unsigned rows) {
        const unsigned active_count = static_cast<unsigned>(active_bins_.size());
        BuildCompact(rows);
        const dim3 grid(
            (active_count + mgt_cuda::detail::kSparseCompactActiveBinsPerBlock - 1U) /
                mgt_cuda::detail::kSparseCompactActiveBinsPerBlock,
            (shape_.hd1 / 2U +
             mgt_cuda::detail::kSparseCompactActiveThreadsPerBin - 1U) /
                mgt_cuda::detail::kSparseCompactActiveThreadsPerBin);
        mgt_cuda::detail::SparseInputGradCompactActiveAdjacent2PackedU16
            <<<grid, mgt_cuda::detail::kSparseCompactActiveThreads>>>(
                shape_, dz_.get(), rows, active_.get(), active_count,
                compact_counts_.get(), compact_rows_.get(), candidate_.get());
        Cuda(cudaGetLastError(), "compact launch");
    }

    void LaunchCompactHalf(unsigned rows) {
        const unsigned active_count = static_cast<unsigned>(active_bins_.size());
        BuildCompact(rows);
        const dim3 grid(
            (active_count + mgt_cuda::detail::kSparseCompactActiveBinsPerBlock - 1U) /
                mgt_cuda::detail::kSparseCompactActiveBinsPerBlock,
            (shape_.hd1 / 2U +
             mgt_cuda::detail::kSparseCompactActiveThreadsPerBin - 1U) /
                mgt_cuda::detail::kSparseCompactActiveThreadsPerBin);
        mgt_cuda::detail::SparseInputGradCompactActiveAdjacent2PackedHalfU16
            <<<grid, mgt_cuda::detail::kSparseCompactActiveThreads>>>(
                shape_, dz_half_.get(), rows, active_.get(), active_count,
                compact_counts_.get(), compact_rows_.get(), candidate_.get());
        Cuda(cudaGetLastError(), "compact half launch");
    }

    void LaunchCompactHalfU32(unsigned rows) {
        const unsigned active_count = static_cast<unsigned>(active_bins_.size());
        BuildCompact(rows);
        const dim3 grid(
            (active_count +
             mgt_cuda::detail::kSparseCompactActiveHalfU32BinsPerBlock - 1U) /
                mgt_cuda::detail::kSparseCompactActiveHalfU32BinsPerBlock,
            (shape_.hd1 / 2U +
             mgt_cuda::detail::kSparseCompactActiveThreadsPerBin - 1U) /
                mgt_cuda::detail::kSparseCompactActiveThreadsPerBin);
        mgt_cuda::detail::SparseInputGradCompactActiveAdjacent2PackedHalfU16U32
            <<<grid, mgt_cuda::detail::kSparseCompactActiveHalfU32Threads>>>(
                shape_, dz_half_.get(), rows, active_.get(), active_count,
                compact_counts_.get(), compact_rows_.get(), candidate_.get());
        Cuda(cudaGetLastError(), "compact half u32 launch");
    }

    void LaunchHalfOracle(unsigned rows) {
        const unsigned active_count = static_cast<unsigned>(active_bins_.size());
        const std::uint64_t count =
            static_cast<std::uint64_t>(active_count) * shape_.hd1;
        SparseInputGradCompactHalfSerialOracle<<<
            static_cast<unsigned>((count + 255U) / 256U), 256>>>(
                shape_, states_.get(), dz_half_.get(), rows, active_.get(),
                active_count, baseline_.get());
        Cuda(cudaGetLastError(), "half serial oracle launch");
    }

    void CheckRows(const std::vector<mgt::TrainStateStorage>& states, unsigned rows) {
        const auto counts = compact_counts_.Read();
        const auto ids = compact_rows_.Read();
        for (std::size_t active_index = 0; active_index < active_bins_.size(); ++active_index) {
            const unsigned bin = active_bins_[active_index];
            const unsigned position = bin / shape_.state_value_pad;
            const unsigned value = bin % shape_.state_value_pad;
            std::vector<std::uint16_t> expected;
            for (unsigned row = 0; row < rows; ++row)
                if (states[row].v[position] == value)
                    expected.push_back(static_cast<std::uint16_t>(row));
            Require(counts[active_index] == expected.size(), "compact count mismatch");
            for (std::size_t i = 0; i < expected.size(); ++i)
                Require(ids[active_index * rows + i] == expected[i],
                        "compact ordered row IDs mismatch");
        }
    }

    static void EqualOutputs(const std::vector<float>& expected,
                             const std::vector<float>& actual,
                             const char* context) {
        Require(expected.size() == actual.size(), std::string(context) + " size");
        if (std::memcmp(expected.data(), actual.data(), expected.size() * sizeof(float)) == 0)
            return;
        for (std::size_t i = 0; i < expected.size(); ++i)
            if (Bits(expected[i]) != Bits(actual[i]))
                throw std::runtime_error(std::string(context) +
                                         " bitwise mismatch index=" + std::to_string(i));
    }

    const mgt_cuda::CudaMlpShape shape_{3, 5, 66, 1, 0, 1};
    const unsigned capacity_rows_ = 257;
    const std::vector<std::uint16_t> active_bins_{0, 2, 6, 9, 10, 13, 14};
    unsigned bins_;
    std::size_t elements_;
    DeviceBuffer<mgt::TrainStateStorage> states_;
    DeviceBuffer<float> dz_;
    DeviceBuffer<__half> dz_half_;
    DeviceBuffer<std::uint16_t> active_;
    DeviceBuffer<unsigned> full_counts_;
    DeviceBuffer<std::uint16_t> full_rows_;
    DeviceBuffer<unsigned> compact_counts_;
    DeviceBuffer<std::uint16_t> compact_rows_;
    DeviceBuffer<float> baseline_;
    DeviceBuffer<float> candidate_;
};

}  // namespace

int main() {
    try {
        int count = 0;
        Cuda(cudaGetDeviceCount(&count), "cudaGetDeviceCount");
        Require(count > 0, "CUDA device required");
        Harness harness;
        harness.RunOwnershipWitness();
        harness.RunHalfMirrorWitness();
        for (const auto [rows, salt] :
             {std::pair{1U, 11U}, std::pair{31U, 13U}, std::pair{32U, 17U},
              std::pair{33U, 19U}, std::pair{257U, 23U}}) {
            harness.RunReuse(rows, salt);
            harness.RunHalfU32Reuse(rows, salt + 37U);
            harness.RunFusedAdamWitness(rows, salt + 73U);
        }
        harness.Finish();
        Require(!cleanup_failed, "CUDA cleanup failed");
        std::printf("PASS compact active input gradient: bitwise FP32/RN-half serial order, fused active Adam+dense tail, owner writes, persistent zero, immutable inputs/map and canaries\n");
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL compact active input gradient: %s\n", error.what());
        return EXIT_FAILURE;
    }
}
