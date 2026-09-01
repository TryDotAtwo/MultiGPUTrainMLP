#include "sparse_input_grad_layout_candidates.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
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

// The oracle always owns a 256-feature tile, independent of candidate mapping.
constexpr unsigned kOracleThreads = 256;
#ifdef MGT_TEST_SPARSE_WARP_BINS
constexpr unsigned kFeatureTile = MGT_TEST_SPARSE_WARP_BINS;
static_assert(kFeatureTile == 32 || kFeatureTile == 64);
constexpr unsigned kBinsPerBlock = 256 / kFeatureTile;
constexpr unsigned kCandidateThreads = 256;
constexpr const char* kCandidateName = "warp-bins";
#elif defined(MGT_TEST_SPARSE_ADJACENT2)
constexpr unsigned kFeatureTile = 128;
constexpr unsigned kCandidateThreads = 64;
constexpr const char* kCandidateName = "adjacent2";
#elif defined(MGT_TEST_SPARSE_COLS2)
constexpr unsigned kFeatureTile = 256;
constexpr unsigned kBinsPerBlock = 1;
constexpr unsigned kCandidateThreads = 128;
constexpr const char* kCandidateName = "cols2";
#else
constexpr unsigned kFeatureTile = 256;
constexpr unsigned kBinsPerBlock = 1;
constexpr unsigned kCandidateThreads = 256;
constexpr const char* kCandidateName = "baseline";
#endif
constexpr std::size_t kGuardElements = 256;
constexpr unsigned char kGuardByte = 0xa5;
bool cleanup_failed = false;
unsigned cases_run = 0;

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
    DeviceBuffer(std::size_t count, const char* name, bool tightly_sized = false)
        : count_(count), name_(name), tightly_sized_(tightly_sized) {
        const std::size_t allocated_count = count_ + (tightly_sized_ ? 0 : 2 * kGuardElements);
        Cuda(cudaMalloc(&allocation_, allocated_count * sizeof(T)),
             name_ + " cudaMalloc");
        try {
            Cuda(cudaMemset(allocation_, kGuardByte, allocated_count * sizeof(T)),
                 name_ + " initialize guards");
        } catch (...) {
            Release();
            throw;
        }
    }

    ~DeviceBuffer() { Release(); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    T* get() const { return allocation_ + (tightly_sized_ ? 0 : kGuardElements); }

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

    void RequireUnchanged(const std::vector<T>& expected) const {
        Require(expected.size() == count_, name_ + " immutable input size mismatch");
        const auto actual = Read();
        Require(std::memcmp(actual.data(), expected.data(), count_ * sizeof(T)) == 0,
                name_ + " input payload changed");
    }

    void CheckGuards() const {
        if (tightly_sized_) return;
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
    bool tightly_sized_;
    T* allocation_ = nullptr;
};

std::uint32_t Bits(float value) {
    std::uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

float FromBits(std::uint32_t bits) {
    float value;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

float QuietNaN() { return FromBits(0x7fc00000U); }

void Equal(const std::vector<float>& actual, const std::vector<float>& expected,
           unsigned hd1, const std::string& context, bool compare_nan_bits = true) {
    Require(actual.size() == expected.size(), context + " output size mismatch");
    if (std::memcmp(actual.data(), expected.data(), actual.size() * sizeof(float)) == 0)
        return;
    for (std::size_t i = 0; i < actual.size(); ++i) {
        if (Bits(actual[i]) == Bits(expected[i])) continue;
        // CPU and CUDA may canonicalize NaN payloads differently. The GPU-to-GPU
        // comparison remains bitwise, including NaNs; CPU finite values do too.
        if (!compare_nan_bits && std::isnan(actual[i]) && std::isnan(expected[i])) continue;
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

enum class Distribution { kBalanced, kAllInOne, kSkewed, kThreeOrbits };

struct Fixture {
    std::string name;
    unsigned rows;
    unsigned nonzero_rows;
    unsigned logical_hd1;
    std::vector<float> dz;
    std::vector<unsigned> counts;
    std::vector<unsigned> row_ids;
    unsigned char output_poison = 0xcd;
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
    Require(distribution != Distribution::kThreeOrbits ||
                (shape.state_len == 72 && shape.state_value_pad == 72),
            name + " three-orbit fixture requires a 72x72 shape");
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
            else if (distribution == Distribution::kThreeOrbits) {
                // Synthetic relabeling of three disjoint 24-position/value
                // orbits, not an actual permutation walk or measured histogram.
                value = (position / 24) * 24 + (row + position + salt) % 24;
            }
            const std::size_t bin = static_cast<std::size_t>(position) * shape.state_value_pad + value;
            f.row_ids[bin * rows + f.counts[bin]++] = row;
        }
        if (row < nonzero_rows) {
            for (unsigned h = 0; h < logical_hd1; ++h)
                f.dz[static_cast<std::size_t>(row) * shape.hd1 + h] = GradientValue(row, h, salt);
        }
    }
    // Allocated but inactive rows must never enter a current-row-stride list.
    std::fill(f.dz.begin() + static_cast<std::size_t>(rows) * shape.hd1,
              f.dz.end(), QuietNaN());
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
    Harness(mgt_cuda::CudaMlpShape shape, unsigned capacity_rows, bool tightly_sized = false)
        : shape_(shape), capacity_rows_(capacity_rows), tightly_sized_(tightly_sized),
          bins_(static_cast<unsigned>(shape.state_len * shape.state_value_pad)),
          dz_(static_cast<std::size_t>(capacity_rows) * shape.hd1, "dz", tightly_sized),
          counts_(bins_, "counts", tightly_sized),
          row_ids_(static_cast<std::size_t>(bins_) * capacity_rows, "row_ids", tightly_sized),
          actual_(static_cast<std::size_t>(bins_) * shape.hd1, "actual gradient", tightly_sized),
          expected_(static_cast<std::size_t>(bins_) * shape.hd1, "oracle gradient", tightly_sized) {}

    void Run(const Fixture& f, const std::vector<float>* literal = nullptr) {
        Require(f.rows > 0 && f.rows <= capacity_rows_, f.name + " invalid allocation extent");
        Require(f.counts.size() == bins_ && f.logical_hd1 <= shape_.hd1,
                f.name + " invalid fixture shape");
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
        // Both finite and NaN output poison are used by fixtures. Every physical
        // feature, including empty bins and zero-padding, must be overwritten.
        actual_.FillBytes(f.output_poison);
        expected_.FillBytes(0x5a);
        const unsigned h_tiles = (shape_.hd1 + kFeatureTile - 1) / kFeatureTile;
        const unsigned oracle_tiles = (shape_.hd1 + kOracleThreads - 1) / kOracleThreads;
        OldGridSerialOracle<<<dim3(oracle_tiles, bins_), kOracleThreads>>>(
            dz_.get(), f.rows, shape_.hd1, counts_.get(), row_ids_.get(), expected_.get());
        Cuda(cudaGetLastError(), f.name + " old-grid oracle launch");

        // Regression contract: the real kernel must consume the transposed CTA
        // grid. Leaving its old blockIdx coordinates drops bins when bins != h_tiles.
#ifdef MGT_TEST_SPARSE_WARP_BINS
        // Missing a bin group or using blockDim as the feature tile must fail
        // against the independently fixed old-grid oracle, including bin tails.
        mgt_cuda::detail::SparseInputGradGroupedRowsWarpBins<kFeatureTile>
            <<<dim3((bins_ + kBinsPerBlock - 1) / kBinsPerBlock, h_tiles), kCandidateThreads>>>(
#elif defined(MGT_TEST_SPARSE_ADJACENT2)
        mgt_cuda::detail::SparseInputGradGroupedRowsAdjacent2
            <<<dim3(bins_, h_tiles), kCandidateThreads>>>(
#elif defined(MGT_TEST_SPARSE_COLS2)
        // The declaration intentionally exists only in the future candidate
        // build. The existing target does not require the X2 kernel to exist.
        mgt_cuda::detail::SparseInputGradGroupedRowsX2<<<dim3(bins_, h_tiles), kCandidateThreads>>>(
#else
        mgt_cuda::detail::SparseInputGradGroupedRows<<<dim3(bins_, h_tiles), kCandidateThreads>>>(
#endif
            shape_, dz_.get(), f.rows, counts_.get(), row_ids_.get(), actual_.get());
        Cuda(cudaGetLastError(), f.name + " production launch");
        Cuda(cudaDeviceSynchronize(), f.name + " synchronize");

        dz_.RequireUnchanged(f.dz);
        counts_.RequireUnchanged(f.counts);
        row_ids_.RequireUnchanged(f.row_ids);
        const auto actual = actual_.Read();
        const auto expected = expected_.Read();
        Equal(actual, expected, shape_.hd1, f.name + " old-grid bitwise comparison");
        // Large cases stay on GPU: no production-sized CPU bin/feature/row loop.
        if (static_cast<std::uint64_t>(f.rows) * shape_.state_len * shape_.hd1 <= 1000000)
            Equal(expected, CpuSerialOracle(f, shape_.hd1), shape_.hd1,
                  f.name + " independent CPU oracle", false);
        if (literal)
            Equal(actual, *literal, shape_.hd1, f.name + " hand-derived result", false);
        for (std::size_t bin = 0; bin < bins_; ++bin) {
            for (unsigned h = f.logical_hd1; h < shape_.hd1; ++h)
                Require(Bits(actual[bin * shape_.hd1 + h]) == 0,
                        f.name + " padded dz feature did not produce positive zero");
        }
        ++cases_run;
        std::printf("PASS %-32s rows=%u nonzero_rows=%u hd1=%u logical_hd1=%u bins=%u poison=0x%02x allocation=%s\n",
                    f.name.c_str(), f.rows, f.nonzero_rows, shape_.hd1, f.logical_hd1, bins_,
                    static_cast<unsigned>(f.output_poison), tightly_sized_ ? "tight" : "guarded");
    }

private:
    mgt_cuda::CudaMlpShape shape_;
    unsigned capacity_rows_;
    bool tightly_sized_;
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
        // Both X2 accumulators must retain the serial rounding order.
        for (unsigned row = 0; row < 4; ++row) {
            f.dz[row * shape.hd1 + h] = columns[h][row];
            f.dz[row * shape.hd1 + 128 + h] = columns[h][row];
        }
        literal[2 * shape.hd1 + h] = sums[h];
        literal[5 * shape.hd1 + h] = sums[h];
        literal[2 * shape.hd1 + 128 + h] = sums[h];
        literal[5 * shape.hd1 + 128 + h] = sums[h];
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

void RunCountBoundaryCases(bool quick) {
    constexpr unsigned counts[] = {0, 1, 31, 32, 33, 63, 64, 65};
    constexpr unsigned rows = 289;  // Literal sum of the eight bin counts.
    constexpr unsigned capacity_rows = 321;
    const unsigned widths[] = {127, 128, 129, 255, 256, 257};
    for (unsigned hd1 : widths) {
        if (quick && hd1 != 129) continue;
        const mgt_cuda::CudaMlpShape shape{1, 8, hd1, 1, 0, 1};
        Fixture f{"literal-count-boundaries-" + std::to_string(hd1), rows, rows, hd1,
                  std::vector<float>(static_cast<std::size_t>(capacity_rows) * hd1, QuietNaN()),
                  std::vector<unsigned>(counts, counts + 8),
                  std::vector<unsigned>(8 * capacity_rows, std::numeric_limits<unsigned>::max())};
        f.output_poison = 0xff;
        std::vector<float> literal(8 * hd1, 0.0f);
        unsigned next_row = 0;
        for (unsigned bin = 0; bin < 8; ++bin) {
            for (unsigned i = 0; i < counts[bin]; ++i) {
                const unsigned row = next_row++;
                f.row_ids[bin * rows + i] = row;
                for (unsigned h = 0; h < hd1; ++h)
                    f.dz[static_cast<std::size_t>(row) * hd1 + h] =
                        static_cast<float>(1 + h % 7) * 0.25f;
            }
            for (unsigned h = 0; h < hd1; ++h) {
                // Constant quarter-integers sum exactly at these counts. This
                // literal-count oracle catches omitted second columns, wrong
                // column offsets, row-loop tails and unwritten empty bins.
                literal[bin * hd1 + h] =
                    static_cast<float>(counts[bin] * (1 + h % 7)) * 0.25f;
            }
        }
        Require(next_row == rows, f.name + " literal bin counts do not cover rows");
        Harness harness(shape, capacity_rows);
        harness.Run(f, &literal);
    }
}

void RunNaNInputWitness() {
    const mgt_cuda::CudaMlpShape shape{2, 3, 257, 1, 0, 1};
    auto f = MakeFixture(shape, 5, 3, 3, 257, Distribution::kAllInOne, 0,
                         "active-nan-input-and-empty-bins");
    // NaNs in both X2-owned column groups must propagate for occupied bins;
    // empty bins must still become +0 without reading any poisoned row IDs.
    const unsigned features[] = {0, 127, 128, 256};
    for (unsigned h : features) f.dz[shape.hd1 + h] = QuietNaN();
    Harness harness(shape, 5);
    harness.Run(f);
}

void RunTightAllocationCases() {
    const unsigned widths[] = {129, 257};
    for (unsigned hd1 : widths) {
        const unsigned rows = hd1 == 129 ? 1 : 3;
        const mgt_cuda::CudaMlpShape shape{1, 1, hd1, 1, 0, 1};
        const auto f = MakeFixture(shape, rows, rows, rows, hd1,
                                   Distribution::kAllInOne, 5,
                                   "tight-allocation-tail-" + std::to_string(hd1));
        // cudaMalloc requests exactly the active extent, including dZ's last
        // row. Unlike centered guards, this lets memcheck catch an unguarded
        // second-column load even if its value is discarded before the store.
        Harness harness(shape, rows, true);
        harness.Run(f);
    }
}

void RunArithmeticWitness() {
    const mgt_cuda::CudaMlpShape shape{1, 2, 257, 1, 0, 1};
    Fixture f{"signed-zero-subnormal-infinity", 3, 3, 257,
              std::vector<float>(3 * shape.hd1, 0.0f), {3, 0},
              {0, 1, 2, std::numeric_limits<unsigned>::max(),
               std::numeric_limits<unsigned>::max(), std::numeric_limits<unsigned>::max()}};
    // The checked baseline NVCC command has neither --ftz nor --use_fast_math;
    // NVCC defaults to --ftz=false. Host compilation also has no fast-math.
    // These independent literals deliberately require that unchanged FP32 mode.
    const std::uint32_t input_bits[][3] = {
        {0x80000000U, 0x80000000U, 0x80000000U},  // +0 accumulator, three -0 inputs
        {0x00000000U, 0x80000000U, 0x00000000U},
        {0x00000001U, 0x00000001U, 0x00000001U},  // three smallest subnormals
        {0x80000001U, 0x80000001U, 0x80000001U},
        {0x00800000U, 0x807fffffU, 0x00000000U},  // normal - largest subnormal
        {0x007fffffU, 0x00000001U, 0x00000000U},  // cross into normal range
        {0x7f800000U, 0x3f800000U, 0xc0000000U},  // +Inf + 1 - 2
        {0xff800000U, 0x40400000U, 0x3f800000U},  // -Inf + 3 + 1
        {0x7f800000U, 0xff800000U, 0x40000000U},  // opposite infinities -> NaN
        {0x7f7fffffU, 0x7f7fffffU, 0xff7fffffU},  // FP32 overflow remains +Inf
        {0x3f800000U, 0xbf800000U, 0x80000000U},  // cancellation then -0
        {0x80800000U, 0x007fffffU, 0x80000000U}};
    const std::uint32_t result_bits[] = {
        0x00000000U, 0x00000000U, 0x00000003U, 0x80000003U,
        0x00000001U, 0x00800000U, 0x7f800000U, 0xff800000U,
        0x7fc00000U, 0x7f800000U, 0x00000000U, 0x80000001U};
    std::vector<float> literal(2 * shape.hd1, 0.0f);
    for (unsigned entry = 0; entry < sizeof(result_bits) / sizeof(result_bits[0]); ++entry) {
        // Repeat in the first and second accumulator groups; h=256 separately
        // exercises subnormal arithmetic in the final one-column tile.
        const unsigned features[] = {entry, 128 + entry};
        for (unsigned h : features) {
            for (unsigned row = 0; row < 3; ++row)
                f.dz[row * shape.hd1 + h] = FromBits(input_bits[entry][row]);
            literal[h] = FromBits(result_bits[entry]);
        }
    }
    for (unsigned row = 0; row < 3; ++row) f.dz[row * shape.hd1 + 256] = FromBits(1U);
    literal[256] = FromBits(3U);
    volatile float host_zero = 0.0f;
    volatile float host_subnormal = FromBits(1U);
    Require(Bits(SerialFloatAdd(host_zero, host_subnormal)) == 1U,
            f.name + " host FP32 mode unexpectedly flushes subnormals");
    Harness harness(shape, 3);
    harness.Run(f, &literal);
}

void RunAllCases(bool quick) {
    // First case is small, has empty bins and a partial CTA, and fails by output
    // mismatch (not an illegal read) if the production coordinates remain old.
    RunCancellationWitness();
    RunCountBoundaryCases(quick);
    RunNaNInputWitness();
    RunTightAllocationCases();
    RunArithmeticWitness();
    if (quick) return;

    const unsigned widths[] = {1, 31, 32, 33, 127, 128, 129, 255, 256, 257,
                               511, 513, 2556, 2560};
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
        auto orbit = MakeFixture(shape, 4096, 4096, 4096, 2556,
                                 Distribution::kThreeOrbits, 29,
                                 "production-three-orbits-padding");
        orbit.output_poison = 0xff;
        unsigned empty_bins = 0;
        for (unsigned count : orbit.counts) {
            if (count == 0) ++empty_bins;
            else Require(count == 170 || count == 171,
                         orbit.name + " expected floor/ceil(4096/24) rows per occupied bin");
        }
        Require(empty_bins == 3456, orbit.name + " expected two-thirds empty bins");
        harness.Run(orbit);
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
        std::printf("SparseInputGradGroupedRows candidate=%s threads=%u feature_tile=%u device=%s sm=%d%d mode=%s\n",
                    kCandidateName, kCandidateThreads, kFeatureTile, properties.name,
                    properties.major, properties.minor, quick ? "quick" : "all");
        RunAllCases(quick);
        Require(!cleanup_failed, "CUDA allocation cleanup failed");
        std::printf("PASS sparse grouped-row CTA mapping: candidate=%s cases=%u exact GPU bytes, finite CPU bytes, NaN classification, canaries and immutable inputs\n",
                    kCandidateName, cases_run);
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL sparse grouped-row CTA mapping: %s\n", error.what());
        return EXIT_FAILURE;
    }
}
