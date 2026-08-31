#include "../../cuda/grouped_input_rows.cuh"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr unsigned kThreads = 256;
constexpr unsigned kBinsPerBlock = 8;
constexpr std::size_t kGuardElements = 256;
constexpr unsigned char kGuardByte = 0xa5;
constexpr unsigned kUnusedRow = std::numeric_limits<unsigned>::max();
bool cleanup_failed = false;

void Require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

void Cuda(cudaError_t status, const std::string& operation) {
    if (status != cudaSuccess)
        throw std::runtime_error(operation + ": " + cudaGetErrorString(status));
}

// Test-owned storage. Every allocation has initialized guards on both sides.
template <class T> class DeviceBuffer {
public:
    DeviceBuffer(std::size_t count, const char* name) : count_(count), name_(name) {
        Cuda(cudaMalloc(&allocation_, (count_ + 2 * kGuardElements) * sizeof(T)),
             name_ + " cudaMalloc");
        try {
            Cuda(cudaMemset(allocation_, kGuardByte,
                            (count_ + 2 * kGuardElements) * sizeof(T)),
                 name_ + " initialize guards");
        } catch (...) {
            Release();
            throw;
        }
    }

    ~DeviceBuffer() { Release(); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    T* get() const { return allocation_ + kGuardElements; }

    void Put(const std::vector<T>& values) {
        Require(values.size() == count_, name_ + " upload size mismatch");
        if (count_ != 0)
            Cuda(cudaMemcpy(get(), values.data(), count_ * sizeof(T), cudaMemcpyHostToDevice),
                 name_ + " upload");
    }

    void FillBytes(unsigned char byte) {
        if (count_ != 0)
            Cuda(cudaMemset(get(), byte, count_ * sizeof(T)), name_ + " reset payload");
    }

    std::vector<T> Read() const {
        CheckGuards();
        std::vector<T> values(count_);
        if (count_ != 0)
            Cuda(cudaMemcpy(values.data(), get(), count_ * sizeof(T), cudaMemcpyDeviceToHost),
                 name_ + " download");
        return values;
    }

    void CheckGuards() const {
        unsigned char guards[2 * kGuardElements * sizeof(T)];
        Cuda(cudaMemcpy(guards, allocation_, kGuardElements * sizeof(T),
                        cudaMemcpyDeviceToHost), name_ + " leading guard");
        Cuda(cudaMemcpy(guards + kGuardElements * sizeof(T), get() + count_,
                        kGuardElements * sizeof(T), cudaMemcpyDeviceToHost),
             name_ + " trailing guard");
        for (unsigned char byte : guards)
            Require(byte == kGuardByte, name_ + " allocation canary changed");
    }

private:
    void Release() noexcept {
        if (!allocation_) return;
        const cudaError_t status = cudaFree(allocation_);
        allocation_ = nullptr;
        if (status != cudaSuccess) {
            cleanup_failed = true;
            std::fprintf(stderr, "%s cudaFree: %s\n", name_.c_str(), cudaGetErrorString(status));
        }
    }

    std::size_t count_;
    std::string name_;
    T* allocation_ = nullptr;
};

enum class Distribution { kBalanced, kAllInOne, kSkewed, kWithInvalid, kAllInvalid };

struct Fixture {
    std::string name;
    unsigned rows;
    std::vector<mgt::TrainStateStorage> states;
};

struct Expected {
    std::vector<unsigned> counts;
    std::vector<unsigned> row_ids;
};

Fixture MakeFixture(const mgt_cuda::CudaMlpShape& shape, unsigned capacity_rows,
                    unsigned rows, Distribution distribution, unsigned salt,
                    const std::string& name) {
    Require(rows <= capacity_rows, name + " invalid row count");
    Require(shape.state_len > 0 && shape.state_len <= 72 &&
                shape.state_value_pad > 0 && shape.state_value_pad <= 72,
            name + " invalid test shape");
    Fixture f{name, rows, std::vector<mgt::TrainStateStorage>(capacity_rows)};
    for (unsigned row = 0; row < capacity_rows; ++row) {
        // Poison all physical u8[80] padding with alternating valid/invalid
        // values. Inactive capacity rows remain valid, to expose over-reading.
        for (unsigned position = 0; position < sizeof(f.states[row].v); ++position)
            f.states[row].v[position] = static_cast<mgt::StateValue>(
                (row + position + salt) % 2 == 0 ? 0 : 255);
        for (unsigned position = 0; position < shape.state_len; ++position) {
            unsigned value = (row + 5 * position + salt) % shape.state_value_pad;
            if (row < rows) {
                if (distribution == Distribution::kAllInOne)
                    value = (7 * position + salt) % shape.state_value_pad;
                else if (distribution == Distribution::kSkewed)
                    value = row % 17 != 16 ? (11 * position + salt) % shape.state_value_pad
                                          : (row / 17 + position + salt) % shape.state_value_pad;
                else if (distribution == Distribution::kWithInvalid) {
                    const unsigned selector = (row + position + salt) % 7;
                    if (selector == 0) value = shape.state_value_pad;
                    if (selector == 1) value = 255;
                    if (selector == 2) value = shape.state_value_pad + 1;
                } else if (distribution == Distribution::kAllInvalid) {
                    value = (row + position) % 2 == 0 ? shape.state_value_pad : 255;
                }
            }
            f.states[row].v[position] = static_cast<mgt::StateValue>(value);
        }
    }
    return f;
}

// Independent CPU oracle visits each input row once and appends it to the
// corresponding position/value list. It uses neither GPU ballot/prefix logic
// nor production helpers. The physical output stride is the current row count,
// not the allocation capacity, and every unused element remains UINT_MAX.
Expected CpuOracle(const mgt_cuda::CudaMlpShape& shape, const Fixture& f) {
    const std::size_t bins = static_cast<std::size_t>(shape.state_len) * shape.state_value_pad;
    Expected expected{std::vector<unsigned>(bins, 0),
                      std::vector<unsigned>(bins * f.states.size(), kUnusedRow)};
    for (unsigned row = 0; row < f.rows; ++row) {
        for (unsigned position = 0; position < shape.state_len; ++position) {
            const unsigned value = f.states[row].v[position];
            if (value >= shape.state_value_pad) continue;
            const std::size_t bin = static_cast<std::size_t>(position) * shape.state_value_pad + value;
            expected.row_ids[bin * f.rows + expected.counts[bin]++] = row;
        }
    }
    return expected;
}

void Equal(const std::vector<unsigned>& actual, const std::vector<unsigned>& expected,
           unsigned rows, const std::string& context) {
    Require(actual.size() == expected.size(), context + " output size mismatch");
    if (actual.empty() ||
        std::memcmp(actual.data(), expected.data(), actual.size() * sizeof(unsigned)) == 0)
        return;
    for (std::size_t i = 0; i < actual.size(); ++i) {
        if (actual[i] == expected[i]) continue;
        char detail[256];
        std::snprintf(detail, sizeof(detail),
                      " index=%zu bin=%zu entry=%zu got=%u expected=%u rows=%u",
                      i, rows == 0 ? 0 : i / rows, rows == 0 ? i : i % rows,
                      actual[i], expected[i], rows);
        throw std::runtime_error(context + detail);
    }
}

class Harness {
public:
    Harness(mgt_cuda::CudaMlpShape shape, unsigned capacity_rows)
        : shape_(shape), capacity_rows_(capacity_rows),
          bins_(shape.state_len * shape.state_value_pad),
          states_(capacity_rows, "states"), counts_(bins_, "counts"),
          row_ids_(static_cast<std::size_t>(bins_) * capacity_rows, "row_ids") {}

    void Run(const Fixture& f, const std::vector<std::vector<unsigned>>* literal = nullptr) {
        Require(f.rows <= capacity_rows_ && f.states.size() == capacity_rows_,
                f.name + " allocation capacity mismatch");
        const auto expected = CpuOracle(shape_, f);
        if (literal) {
            Require(literal->size() == bins_, f.name + " literal bin count mismatch");
            for (std::size_t bin = 0; bin < bins_; ++bin) {
                Require(expected.counts[bin] == (*literal)[bin].size(),
                        f.name + " CPU oracle disagrees with hand-derived counts");
                for (std::size_t entry = 0; entry < (*literal)[bin].size(); ++entry)
                    Require(expected.row_ids[bin * f.rows + entry] == (*literal)[bin][entry],
                            f.name + " CPU oracle disagrees with hand-derived row IDs");
            }
        }

        states_.Put(f.states);
        counts_.FillBytes(0xcd);
        row_ids_.FillBytes(0xff);
        // Regression contract: eight whole warps per CTA, one warp per bin,
        // including a partially occupied last CTA and a partial final row tile.
        const unsigned blocks = (bins_ + kBinsPerBlock - 1) / kBinsPerBlock;
        mgt_cuda::detail::BuildGroupedInputRows<<<blocks, kThreads>>>(
            shape_, states_.get(), f.rows, counts_.get(), row_ids_.get());
        Cuda(cudaGetLastError(), f.name + " production launch");
        Cuda(cudaDeviceSynchronize(), f.name + " synchronize");

        const auto actual_counts = counts_.Read();
        Equal(actual_counts, expected.counts, 1, f.name + " exact counts");
        const auto actual_rows = row_ids_.Read();
        // Full-capacity comparison also checks every unused per-bin tail and
        // the suffix beyond bins*rows after shrinking the live row count.
        Equal(actual_rows, expected.row_ids, f.rows, f.name + " exact row IDs and untouched tails");
        for (std::size_t bin = 0; bin < bins_; ++bin) {
            Require(actual_counts[bin] <= f.rows, f.name + " count exceeds rows");
            for (unsigned entry = 0; entry < actual_counts[bin]; ++entry) {
                const std::size_t index = bin * f.rows + entry;
                Require(actual_rows[index] < f.rows, f.name + " row ID exceeds rows");
                Require(entry == 0 || actual_rows[index - 1] < actual_rows[index],
                        f.name + " row IDs are not strictly ascending");
            }
        }
        const auto after_states = states_.Read();
        Require(after_states.empty() ||
                    std::memcmp(after_states.data(), f.states.data(),
                                f.states.size() * sizeof(mgt::TrainStateStorage)) == 0,
                f.name + " input states or physical padding mutated");
        std::printf("PASS %-36s rows=%u capacity=%u shape=%ux%u bins=%u\n",
                    f.name.c_str(), f.rows, capacity_rows_, shape_.state_len,
                    shape_.state_value_pad, bins_);
    }

private:
    mgt_cuda::CudaMlpShape shape_;
    unsigned capacity_rows_;
    unsigned bins_;
    DeviceBuffer<mgt::TrainStateStorage> states_;
    DeviceBuffer<unsigned> counts_;
    DeviceBuffer<unsigned> row_ids_;
};

void RunQuickCases() {
    const mgt_cuda::CudaMlpShape shape{3, 3, 1, 1, 0, 1};
    Harness harness(shape, 33);
    // The old one-thread/bin kernel under this flat grid writes positions 0
    // and 1 only. Position 2 fails by output mismatch, without an illegal read.
    harness.Run(MakeFixture(shape, 33, 33, Distribution::kBalanced, 0,
                            "flat-grid-partial-CTA-and-warp"));

    auto f = MakeFixture(shape, 33, 5, Distribution::kBalanced, 13,
                         "literal-invalid-empty-and-stable");
    constexpr unsigned char values[5][3] = {
        {0, 1, 2}, {2, 1, 0}, {0, 255, 2}, {3, 0, 2}, {2, 1, 255}};
    for (unsigned row = 0; row < 5; ++row)
        for (unsigned position = 0; position < 3; ++position)
            f.states[row].v[position] = values[row][position];
    const std::vector<std::vector<unsigned>> literal = {
        {0, 2}, {}, {1, 4}, {3}, {0, 1, 4}, {}, {1}, {}, {0, 2, 3}};
    harness.Run(f, &literal);
    harness.Run(MakeFixture(shape, 33, 0, Distribution::kBalanced, 1,
                            "zero-rows-overwrites-all-counts"));
}

void RunAllCases(bool quick) {
    RunQuickCases();
    if (quick) return;

    {
        const mgt_cuda::CudaMlpShape shape{5, 7, 1, 1, 0, 1};
        Harness harness(shape, 4097);
        // Exercise all warp/CTA/4096 boundaries with changing live stride in
        // the same allocations. Later launches must not inherit old counts.
        const unsigned row_counts[] = {4097, 1, 31, 32, 33, 255, 256, 257, 4095, 4096, 0, 4097};
        const Distribution distributions[] = {Distribution::kBalanced, Distribution::kAllInOne,
                                               Distribution::kSkewed, Distribution::kWithInvalid};
        for (unsigned i = 0; i < sizeof(row_counts) / sizeof(row_counts[0]); ++i)
            harness.Run(MakeFixture(shape, 4097, row_counts[i], distributions[i % 4], 17 + i,
                                    "changed-rows-" + std::to_string(row_counts[i])));
        harness.Run(MakeFixture(shape, 4097, 257, Distribution::kAllInvalid, 5,
                                "invalid-values-empty-every-bin"));
    }

    {
        // One valid bin exercises a CTA with seven idle warps. Mixed invalid
        // values include exactly state_value_pad, not only a distant sentinel.
        const mgt_cuda::CudaMlpShape shape{1, 1, 1, 1, 0, 1};
        Harness harness(shape, 257);
        harness.Run(MakeFixture(shape, 257, 257, Distribution::kAllInOne, 0,
                                "single-bin-all-rows"));
        harness.Run(MakeFixture(shape, 257, 33, Distribution::kWithInvalid, 0,
                                "single-bin-invalid-values"));
    }

    {
        // Largest device row-ID allocation is 5184*4097*4 = 84,955,392 bytes.
        const mgt_cuda::CudaMlpShape shape{72, 72, 1, 1, 0, 1};
        Harness harness(shape, 4097);
        harness.Run(MakeFixture(shape, 4097, 4097, Distribution::kBalanced, 19,
                                "production-72x72-balanced"));
        harness.Run(MakeFixture(shape, 4097, 4096, Distribution::kAllInOne, 23,
                                "production-72x72-all-in-one"));
        harness.Run(MakeFixture(shape, 4097, 4095, Distribution::kSkewed, 29,
                                "production-72x72-skewed"));
        harness.Run(MakeFixture(shape, 4097, 257, Distribution::kWithInvalid, 31,
                                "production-reused-smaller-stride"));
    }
}

}  // namespace

int main(int argc, char** argv) {
    const bool quick = argc == 2 && std::strcmp(argv[1], "--quick") == 0;
    if (argc != 1 && !quick) {
        std::fprintf(stderr, "usage: %s [--quick]\n", argv[0]);
        return EXIT_FAILURE;
    }
    try {
        int device_count = 0;
        Cuda(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");
        Require(device_count > 0, "a CUDA device is required");
        Cuda(cudaSetDevice(0), "cudaSetDevice");
        cudaDeviceProp properties{};
        Cuda(cudaGetDeviceProperties(&properties, 0), "cudaGetDeviceProperties");
        std::printf("BuildGroupedInputRows device=%s sm=%d%d mode=%s\n",
                    properties.name, properties.major, properties.minor, quick ? "quick" : "all");
        RunAllCases(quick);
        Require(!cleanup_failed, "CUDA allocation cleanup failed");
        std::printf("PASS stable grouped input rows: exact counts, ordered IDs, tails, input bytes and canaries\n");
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL stable grouped input rows: %s\n", error.what());
        return EXIT_FAILURE;
    }
}
