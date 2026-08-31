#include "../../cuda/sparse_input_grad_grouped_rows.cuh"

#include <cuda_runtime.h>

#include <algorithm>
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
constexpr std::size_t kGuardElements = 256;
constexpr unsigned char kGuardByte = 0xa5;
bool cleanup_failed = false;

void Require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

void Cuda(cudaError_t status, const std::string& operation) {
    if (status != cudaSuccess)
        throw std::runtime_error(operation + ": " + cudaGetErrorString(status));
}

// Test-owned storage. Guards cover a complete CTA of accidental tail writes.
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
        Cuda(cudaMemcpy(get(), values.data(), count_ * sizeof(T), cudaMemcpyHostToDevice),
             name_ + " upload");
    }

    void FillBytes(unsigned char byte) {
        Cuda(cudaMemset(get(), byte, count_ * sizeof(T)), name_ + " reset payload");
    }

    std::vector<T> Read() const {
        CheckGuards();
        std::vector<T> values(count_);
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

std::uint32_t Bits(float value) {
    std::uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

void Equal(const std::vector<float>& actual, const std::vector<float>& expected,
           unsigned hd1, const std::string& context) {
    Require(actual.size() == expected.size(), context + " output size mismatch");
    if (std::memcmp(actual.data(), expected.data(), actual.size() * sizeof(float)) == 0)
        return;
    for (std::size_t i = 0; i < actual.size(); ++i) {
        if (Bits(actual[i]) == Bits(expected[i])) continue;
        char detail[256];
        std::snprintf(detail, sizeof(detail),
                      " bin=%zu h=%zu got=%g [0x%08x] expected=%g [0x%08x]",
                      i / hd1, i % hd1, static_cast<double>(actual[i]), Bits(actual[i]),
                      static_cast<double>(expected[i]), Bits(expected[i]));
        throw std::runtime_error(context + detail);
    }
}

// Independent old-grid oracle: one thread serially accumulates one output.
// Explicit FP32 round-to-nearest additions make order/reassociation observable;
// neither the production kernel nor any production indexing helper is reused.
__global__ void OldGridSerialOracle(const float* dz, unsigned rows, unsigned hd1,
                                   const unsigned* counts, const unsigned* row_ids,
                                   float* expected) {
    const unsigned feature = blockIdx.x * blockDim.x + threadIdx.x;
    const std::size_t bin = blockIdx.y;
    if (feature >= hd1) return;
    float result = 0.0f;
    for (unsigned entry = 0; entry < counts[bin]; ++entry) {
        const unsigned row = row_ids[bin * rows + entry];
        result = __fadd_rn(result, dz[static_cast<std::size_t>(row) * hd1 + feature]);
    }
    expected[bin * hd1 + feature] = result;
}

enum class Distribution { kBalanced, kAllInOne, kSkewed };

struct Fixture {
    std::string name;
    unsigned rows;
    unsigned nonzero_rows;
    unsigned logical_hd1;
    std::vector<float> dz;
    std::vector<unsigned> counts;
    std::vector<unsigned> row_ids;
};

float GradientValue(unsigned row, unsigned feature, unsigned salt) {
    // Large cancellation and half-ULP terms catch reversed or tree-reduced sums.
    constexpr float values[] = {
        16777216.0f, 1.0f, -16777216.0f, 3.0f,
        1.0f, 0x1p-24f, -1.0f, -0x1p-24f,
        0.5f, 0x1p-25f, -0.5f, 5.0f,
        -16777216.0f, 3.0f, 16777216.0f, -7.0f};
    constexpr float scales[] = {1.0f, 0.5f, 2.0f, -1.0f, 0.25f};
    return values[(row + 3 * feature + salt) % 16] * scales[feature % 5];
}

Fixture MakeFixture(const mgt_cuda::CudaMlpShape& shape, unsigned capacity_rows,
                    unsigned rows, unsigned nonzero_rows, unsigned logical_hd1,
                    Distribution distribution, unsigned salt, const std::string& name) {
    Require(rows > 0 && rows <= capacity_rows, name + " invalid row count");
    Require(nonzero_rows <= rows && logical_hd1 <= shape.hd1, name + " invalid padding");
    const std::size_t bins = static_cast<std::size_t>(shape.state_len) * shape.state_value_pad;
    Fixture f{name, rows, nonzero_rows, logical_hd1,
              std::vector<float>(static_cast<std::size_t>(capacity_rows) * shape.hd1, 0.0f),
              std::vector<unsigned>(bins, 0),
              std::vector<unsigned>(bins * capacity_rows, std::numeric_limits<unsigned>::max())};

    // The consumer gets host-authored ordered lists, not BuildGroupedInputRows.
    // Row-major append independently enforces ascending order. Layout uses the
    // current rows, NOT the allocation capacity, including after a row-count change.
    for (unsigned row = 0; row < rows; ++row) {
        for (unsigned position = 0; position < shape.state_len; ++position) {
            unsigned value = (row + 3 * position + salt) % shape.state_value_pad;
            if (distribution == Distribution::kAllInOne)
                value = (7 * position + salt) % shape.state_value_pad;
            else if (distribution == Distribution::kSkewed)
                value = row % 8 != 7 ? (11 * position + salt) % shape.state_value_pad
                                    : (row / 8 + position + salt) % shape.state_value_pad;
            const std::size_t bin = static_cast<std::size_t>(position) * shape.state_value_pad + value;
            f.row_ids[bin * rows + f.counts[bin]++] = row;
        }
        if (row < nonzero_rows) {
            for (unsigned h = 0; h < logical_hd1; ++h)
                f.dz[static_cast<std::size_t>(row) * shape.hd1 + h] = GradientValue(row, h, salt);
        }
    }
    return f;
}

float SerialFloatAdd(float lhs, float rhs) {
    // Force each small-case CPU oracle addition through an FP32 store.
    volatile float rounded = lhs + rhs;
    return rounded;
}

std::vector<float> CpuSerialOracle(const Fixture& f, unsigned hd1) {
    std::vector<float> expected(f.counts.size() * hd1, 0.0f);
    for (std::size_t bin = 0; bin < f.counts.size(); ++bin) {
        for (unsigned h = 0; h < hd1; ++h) {
            float result = 0.0f;
            for (unsigned entry = 0; entry < f.counts[bin]; ++entry) {
                const unsigned row = f.row_ids[bin * f.rows + entry];
                result = SerialFloatAdd(result, f.dz[static_cast<std::size_t>(row) * hd1 + h]);
            }
            expected[bin * hd1 + h] = result;
        }
    }
    return expected;
}

class Harness {
public:
    Harness(mgt_cuda::CudaMlpShape shape, unsigned capacity_rows)
        : shape_(shape), capacity_rows_(capacity_rows),
          bins_(static_cast<unsigned>(shape.state_len * shape.state_value_pad)),
          dz_(static_cast<std::size_t>(capacity_rows) * shape.hd1, "dz"),
          counts_(bins_, "counts"),
          row_ids_(static_cast<std::size_t>(bins_) * capacity_rows, "row_ids"),
          actual_(static_cast<std::size_t>(bins_) * shape.hd1, "actual gradient"),
          expected_(static_cast<std::size_t>(bins_) * shape.hd1, "oracle gradient") {}

    void Run(const Fixture& f, const std::vector<float>* literal = nullptr) {
        Require(f.rows <= capacity_rows_, f.name + " exceeds allocation capacity");
        for (std::size_t bin = 0; bin < f.counts.size(); ++bin) {
            Require(f.counts[bin] <= f.rows, f.name + " invalid list length");
            for (unsigned entry = 0; entry < f.counts[bin]; ++entry) {
                const std::size_t index = bin * f.rows + entry;
                Require(f.row_ids[index] < f.rows, f.name + " invalid row ID");
                Require(entry == 0 || f.row_ids[index - 1] < f.row_ids[index],
                        f.name + " unordered row-ID fixture");
            }
        }
        dz_.Put(f.dz);
        counts_.Put(f.counts);
        row_ids_.Put(f.row_ids);
        // Different nonzero sentinels also catch unwritten empty bins/features.
        actual_.FillBytes(0xcd);
        expected_.FillBytes(0x5a);
        const unsigned h_tiles = (shape_.hd1 + kThreads - 1) / kThreads;
        OldGridSerialOracle<<<dim3(h_tiles, bins_), kThreads>>>(
            dz_.get(), f.rows, shape_.hd1, counts_.get(), row_ids_.get(), expected_.get());
        Cuda(cudaGetLastError(), f.name + " old-grid oracle launch");

        // Regression contract: the real kernel must consume the transposed CTA
        // grid. Leaving its old blockIdx coordinates drops bins when bins != h_tiles.
        mgt_cuda::detail::SparseInputGradGroupedRows<<<dim3(bins_, h_tiles), kThreads>>>(
            shape_, dz_.get(), f.rows, counts_.get(), row_ids_.get(), actual_.get());
        Cuda(cudaGetLastError(), f.name + " production launch");
        Cuda(cudaDeviceSynchronize(), f.name + " synchronize");

        dz_.CheckGuards();
        counts_.CheckGuards();
        row_ids_.CheckGuards();
        const auto actual = actual_.Read();
        const auto expected = expected_.Read();
        Equal(actual, expected, shape_.hd1, f.name + " old-grid bitwise comparison");
        // Large cases stay on GPU: no production-sized CPU bin/feature/row loop.
        if (static_cast<std::uint64_t>(f.rows) * shape_.state_len * shape_.hd1 <= 1000000)
            Equal(expected, CpuSerialOracle(f, shape_.hd1), shape_.hd1,
                  f.name + " independent CPU oracle");
        if (literal) Equal(actual, *literal, shape_.hd1, f.name + " hand-derived result");
        for (std::size_t bin = 0; bin < bins_; ++bin) {
            for (unsigned h = f.logical_hd1; h < shape_.hd1; ++h)
                Require(Bits(actual[bin * shape_.hd1 + h]) == 0,
                        f.name + " padded dz feature did not produce positive zero");
        }
        std::printf("PASS %-32s rows=%u nonzero_rows=%u hd1=%u logical_hd1=%u bins=%u\n",
                    f.name.c_str(), f.rows, f.nonzero_rows, shape_.hd1, f.logical_hd1, bins_);
    }

private:
    mgt_cuda::CudaMlpShape shape_;
    unsigned capacity_rows_;
    unsigned bins_;
    DeviceBuffer<float> dz_;
    DeviceBuffer<unsigned> counts_;
    DeviceBuffer<unsigned> row_ids_;
    DeviceBuffer<float> actual_;
    DeviceBuffer<float> expected_;
};

void RunCancellationWitness() {
    const mgt_cuda::CudaMlpShape shape{2, 3, 257, 1, 0, 1};
    Fixture f{"ordered-cancellation-literal", 4, 4, 255,
              std::vector<float>(4 * shape.hd1, 0.0f), {0, 0, 4, 0, 0, 4},
              std::vector<unsigned>(6 * 4, std::numeric_limits<unsigned>::max())};
    const float columns[6][4] = {
        {16777216.0f, 1.0f, -16777216.0f, 0.0f},
        {16777216.0f, -16777216.0f, 1.0f, 0.0f},
        {1.0f, 0x1p-24f, -1.0f, 0.0f},
        {-1.0f, 0x1p-24f, 1.0f, 0.0f},
        {16777216.0f, 3.0f, -16777216.0f, 0.0f},
        {0.5f, 0x1p-25f, -0.5f, 0.0f}};
    const float sums[6] = {0.0f, 1.0f, 0.0f, 0x1p-24f, 4.0f, 0.0f};
    std::vector<float> literal(6 * shape.hd1, 0.0f);
    for (unsigned h = 0; h < 6; ++h) {
        for (unsigned row = 0; row < 4; ++row) f.dz[row * shape.hd1 + h] = columns[h][row];
        literal[2 * shape.hd1 + h] = sums[h];
        literal[5 * shape.hd1 + h] = sums[h];
    }
    for (unsigned row = 0; row < 4; ++row) {
        f.row_ids[2 * 4 + row] = row;
        f.row_ids[5 * 4 + row] = row;
    }
    float reversed = 0.0f;
    for (unsigned row = 4; row > 0; --row) reversed = SerialFloatAdd(reversed, columns[0][row - 1]);
    Require(Bits(reversed) == Bits(1.0f) && Bits(sums[0]) == Bits(0.0f),
            "cancellation witness must distinguish forward from reversed order");
    Harness harness(shape, 4);
    harness.Run(f, &literal);
}

void RunAllCases(bool quick) {
    // First case is small, has empty bins and a partial CTA, and fails by output
    // mismatch (not an illegal read) if the production coordinates remain old.
    RunCancellationWitness();
    if (quick) return;

    const unsigned widths[] = {1, 31, 32, 33, 255, 256, 257, 511, 513};
    for (unsigned hd1 : widths) {
        const mgt_cuda::CudaMlpShape shape{3, 7, hd1, 1, 0, 1};
        Harness harness(shape, 33);
        harness.Run(MakeFixture(shape, 33, 33, 31, hd1 > 1 ? hd1 - 1 : hd1,
                                Distribution::kBalanced, hd1, "feature-tail-" + std::to_string(hd1)));
    }

    {
        const mgt_cuda::CudaMlpShape shape{4, 9, 257, 1, 0, 1};
        Harness harness(shape, 4097);
        // Reuse the same allocations while both counts and the per-bin row-ID
        // stride shrink/grow. UINT_MAX poisons every unused row-ID slot.
        const unsigned row_counts[] = {4097, 33, 4096, 31, 4095, 32, 1, 4097};
        for (unsigned i = 0; i < sizeof(row_counts) / sizeof(row_counts[0]); ++i) {
            const unsigned rows = row_counts[i];
            const auto distribution = i % 2 == 0 ? Distribution::kAllInOne : Distribution::kSkewed;
            harness.Run(MakeFixture(shape, 4097, rows, rows > 2 ? rows - 2 : rows, 253,
                                    distribution, 17 + i, "changed-rows-" + std::to_string(rows)));
        }
        harness.Run(MakeFixture(shape, 4097, 33, 0, 253, Distribution::kAllInOne, 7,
                                "all-zero-dz-overwrites-output"));
    }

    {
        const mgt_cuda::CudaMlpShape shape{72, 72, 2560, 1, 0, 1};
        Harness harness(shape, 4096);
        harness.Run(MakeFixture(shape, 4096, 4096, 4096, 2560,
                                Distribution::kBalanced, 19, "production-72x72-4096x2560"));
        harness.Run(MakeFixture(shape, 4096, 4096, 4093, 2557,
                                Distribution::kSkewed, 23, "production-skew-and-padding"));
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
        std::printf("SparseInputGradGroupedRows device=%s sm=%d%d mode=%s\n",
                    properties.name, properties.major, properties.minor, quick ? "quick" : "all");
        RunAllCases(quick);
        Require(!cleanup_failed, "CUDA allocation cleanup failed");
        std::printf("PASS sparse grouped-row CTA mapping: exact bytes and canaries\n");
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL sparse grouped-row CTA mapping: %s\n", error.what());
        return EXIT_FAILURE;
    }
}
