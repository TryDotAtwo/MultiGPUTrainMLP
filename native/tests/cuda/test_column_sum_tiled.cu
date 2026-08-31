#include "../../cuda/column_sum.cuh"

#include <cuda_runtime.h>

#include <cfenv>
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
unsigned passed_cases = 0;
unsigned passed_variants = 0;

void Require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

void Cuda(cudaError_t status, const std::string& operation) {
    if (status != cudaSuccess)
        throw std::runtime_error(operation + ": " + cudaGetErrorString(status));
}

// Test-owned storage is reused across tile widths and changing live row counts.
// Both guards are initialized, so partial-CTA writes are observable without
// depending on undefined/uninitialized memory or compiled-out assertions.
class DeviceBuffer {
public:
    DeviceBuffer(std::size_t count, const char* name) : count_(count), name_(name) {
        Cuda(cudaMalloc(&allocation_, (count_ + 2 * kGuardElements) * sizeof(float)),
             name_ + " cudaMalloc");
        try {
            Cuda(cudaMemset(allocation_, kGuardByte,
                            (count_ + 2 * kGuardElements) * sizeof(float)),
                 name_ + " initialize guards");
        } catch (...) {
            Release();
            throw;
        }
    }

    ~DeviceBuffer() { Release(); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    float* get() const { return allocation_ + kGuardElements; }

    void Put(const std::vector<float>& values) {
        Require(values.size() == count_, name_ + " upload size mismatch");
        if (count_)
            Cuda(cudaMemcpy(get(), values.data(), count_ * sizeof(float),
                            cudaMemcpyHostToDevice), name_ + " upload");
    }

    void FillBytes(unsigned char byte) {
        if (count_) Cuda(cudaMemset(get(), byte, count_ * sizeof(float)), name_ + " reset");
    }

    std::vector<float> Read() const {
        CheckGuards();
        std::vector<float> values(count_);
        if (count_)
            Cuda(cudaMemcpy(values.data(), get(), count_ * sizeof(float),
                            cudaMemcpyDeviceToHost), name_ + " download");
        return values;
    }

    void CheckGuards() const {
        unsigned char guards[2 * kGuardElements * sizeof(float)];
        Cuda(cudaMemcpy(guards, allocation_, kGuardElements * sizeof(float),
                        cudaMemcpyDeviceToHost), name_ + " leading guard");
        Cuda(cudaMemcpy(guards + kGuardElements * sizeof(float), get() + count_,
                        kGuardElements * sizeof(float), cudaMemcpyDeviceToHost),
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
    float* allocation_ = nullptr;
};

class Event {
public:
    Event() { Cuda(cudaEventCreate(&event_), "cudaEventCreate"); }
    ~Event() {
        const cudaError_t status = cudaEventDestroy(event_);
        if (status != cudaSuccess) {
            cleanup_failed = true;
            std::fprintf(stderr, "cudaEventDestroy: %s\n", cudaGetErrorString(status));
        }
    }
    Event(const Event&) = delete;
    Event& operator=(const Event&) = delete;
    cudaEvent_t get() const { return event_; }

private:
    cudaEvent_t event_{};
};

std::uint32_t Bits(float value) {
    std::uint32_t bits;
    static_assert(sizeof(bits) == sizeof(value), "FP32 storage required");
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

float FromBits(std::uint32_t bits) {
    float value;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

void Equal(const std::vector<float>& actual, const std::vector<float>& expected,
           const std::string& context) {
    Require(actual.size() == expected.size(), context + " output size mismatch");
    if (actual.empty() ||
        std::memcmp(actual.data(), expected.data(), actual.size() * sizeof(float)) == 0)
        return;
    for (std::size_t column = 0; column < actual.size(); ++column) {
        if (Bits(actual[column]) == Bits(expected[column])) continue;
        char detail[256];
        std::snprintf(detail, sizeof(detail),
                      " column=%zu got=%g [0x%08x] expected=%g [0x%08x]",
                      column, static_cast<double>(actual[column]), Bits(actual[column]),
                      static_cast<double>(expected[column]), Bits(expected[column]));
        throw std::runtime_error(context + detail);
    }
}

// Independent GPU oracle keeps the old grid: one CTA and 256 row lanes per
// column. Explicit RN additions prohibit compiler reassociation from hiding a
// changed accumulation order. It does not call production indexing/helpers.
__global__ void OldGridOracle(const float* input, unsigned rows, unsigned logical,
                              unsigned stride, float* output) {
    const unsigned column = blockIdx.x;
    const unsigned lane = threadIdx.x;
    __shared__ float sums[kThreads];
    float partial = 0.0f;
    if (column < logical) {
        for (unsigned row = lane; row < rows; row += kThreads)
            partial = __fadd_rn(partial, input[static_cast<std::size_t>(row) * stride + column]);
    }
    sums[lane] = partial;
    __syncthreads();
    for (unsigned offset = 128; offset != 0; offset /= 2) {
        if (lane < offset) sums[lane] = __fadd_rn(sums[lane], sums[lane + offset]);
        __syncthreads();
    }
    if (lane == 0) output[column] = sums[0];
}

float RoundedAdd(float lhs, float rhs) {
    // Every addition is stored as FP32, including virtual-lane accumulation
    // and every edge of the specified 128,64,...,1 binary reduction tree.
    volatile float rounded = lhs + rhs;
    return rounded;
}

struct Fixture {
    std::string name;
    unsigned rows;
    unsigned logical;
    std::vector<float> input;
};

// CPU oracle follows the mathematical contract: visit rows in increasing
// order, assign each to residue class row mod 256, then reduce those 256 sums.
// No production kernel or host helper participates in expected-value creation.
std::vector<float> CpuOracle(const Fixture& fixture, unsigned stride) {
    std::vector<float> expected(stride, 0.0f);
    for (unsigned column = 0; column < fixture.logical; ++column) {
        float lanes[256]{};
        for (unsigned row = 0; row < fixture.rows; ++row) {
            const unsigned lane = row % 256;
            lanes[lane] = RoundedAdd(lanes[lane],
                                     fixture.input[static_cast<std::size_t>(row) * stride + column]);
        }
        for (unsigned active = 128; active != 0; active /= 2)
            for (unsigned lane = 0; lane < active; ++lane)
                lanes[lane] = RoundedAdd(lanes[lane], lanes[lane + active]);
        expected[column] = lanes[0];
    }
    return expected;
}

float InputValue(unsigned row, unsigned column, unsigned salt) {
    // Finite mixed-scale values expose reassociation, cancellation, half-ULP
    // rounding and signed zeros. A 20-value period differs from 256 row lanes.
    constexpr float values[] = {
        16777216.0f, 1.0f, -16777216.0f, 3.0f, 1.0f,
        0x1p-24f, -1.0f, -0x1p-24f, 0.5f, 0x1p-25f,
        -0.5f, 5.0f, -16777216.0f, 3.0f, 16777216.0f,
        -7.0f, 0.0f, -0.0f, 0.125f, -0.25f};
    constexpr float scales[] = {1.0f, 0.5f, 2.0f, -1.0f, 0.25f};
    return values[(13 * row + 3 * column + salt) % 20] * scales[column % 5];
}

Fixture MakeFixture(unsigned stride, unsigned capacity_rows, unsigned rows,
                    unsigned logical, unsigned salt, const std::string& name) {
    Require(rows <= capacity_rows && logical <= stride && stride > 0,
            name + " invalid fixture dimensions");
    // Padded columns AND inactive rows retain quiet NaNs. Reading either into
    // the reduction must fail the finite/positive-zero output comparison.
    Fixture fixture{name, rows, logical,
                    std::vector<float>(static_cast<std::size_t>(capacity_rows) * stride,
                                       FromBits(0x7fc13579U))};
    for (unsigned row = 0; row < rows; ++row)
        for (unsigned column = 0; column < logical; ++column)
            fixture.input[static_cast<std::size_t>(row) * stride + column] =
                InputValue(row, column, salt);
    return fixture;
}

class Harness {
public:
    Harness(unsigned stride, unsigned capacity_rows)
        : stride_(stride), capacity_rows_(capacity_rows),
          input_(static_cast<std::size_t>(capacity_rows) * stride, "input"),
          actual_(stride, "production output"), oracle_(stride, "old-grid output") {}

    void Run(const Fixture& fixture, const std::vector<float>* literal = nullptr) {
        Validate(fixture);
        const auto expected = CpuOracle(fixture, stride_);
        if (literal) Equal(expected, *literal, fixture.name + " CPU oracle vs hand-derived result");
        input_.Put(fixture.input);
        oracle_.FillBytes(0x5a);
        OldGridOracle<<<stride_, kThreads>>>(input_.get(), fixture.rows, fixture.logical,
                                           stride_, oracle_.get());
        Cuda(cudaGetLastError(), fixture.name + " old-grid oracle launch");
        Cuda(cudaDeviceSynchronize(), fixture.name + " oracle synchronize");
        Equal(oracle_.Read(), expected, fixture.name + " independent GPU vs CPU oracle");

        // Width four goes first: the extracted old kernel ignores Columns and
        // misses column 3 for stride 9/grid 3, a deterministic output-only RED.
        RunVariant<4>(fixture, expected);
        RunVariant<1>(fixture, expected);
        RunVariant<2>(fixture, expected);
        RunVariant<8>(fixture, expected);
        RunVariant<16>(fixture, expected);
        CheckInput(fixture);
        ++passed_cases;
        std::printf("PASS %-39s rows=%u logical=%u stride=%u capacity=%u widths=1,2,4,8,16\n",
                    fixture.name.c_str(), fixture.rows, fixture.logical, stride_, capacity_rows_);
    }

    void Benchmark(const Fixture& fixture) {
        Validate(fixture);
        const auto expected = CpuOracle(fixture, stride_);
        input_.Put(fixture.input);
        Event begin;
        Event end;
        BenchmarkVariant<1>(fixture, expected, begin.get(), end.get());
        BenchmarkVariant<2>(fixture, expected, begin.get(), end.get());
        BenchmarkVariant<4>(fixture, expected, begin.get(), end.get());
        BenchmarkVariant<8>(fixture, expected, begin.get(), end.get());
        BenchmarkVariant<16>(fixture, expected, begin.get(), end.get());
        CheckInput(fixture);
    }

private:
    void Validate(const Fixture& fixture) const {
        Require(fixture.rows <= capacity_rows_ && fixture.logical <= stride_ &&
                    fixture.input.size() == static_cast<std::size_t>(capacity_rows_) * stride_,
                fixture.name + " allocation capacity mismatch");
    }

    void CheckInput(const Fixture& fixture) const {
        const auto after = input_.Read();
        Require(after.empty() ||
                    std::memcmp(after.data(), fixture.input.data(), after.size() * sizeof(float)) == 0,
                fixture.name + " input, row tail or physical padding mutated");
    }

    template <unsigned Columns> void Launch(const Fixture& fixture) {
        const unsigned blocks = (stride_ + Columns - 1) / Columns;
        mgt_cuda::detail::ColumnSum<Columns><<<blocks, kThreads>>>(
            input_.get(), fixture.rows, fixture.logical, stride_, actual_.get());
    }

    template <unsigned Columns>
    void RunVariant(const Fixture& fixture, const std::vector<float>& expected) {
        const std::string context = fixture.name + " Columns=" + std::to_string(Columns);
        actual_.FillBytes(0xcd);
        Launch<Columns>(fixture);
        Cuda(cudaGetLastError(), context + " production launch");
        Cuda(cudaDeviceSynchronize(), context + " synchronize");
        const auto actual = actual_.Read();
        Equal(actual, expected, context + " exact 256-lane tree");
        for (unsigned column = fixture.logical; column < stride_; ++column)
            Require(Bits(actual[column]) == 0, context + " padded column is not positive zero");
        input_.CheckGuards();
        ++passed_variants;
    }

    template <unsigned Columns>
    void BenchmarkVariant(const Fixture& fixture, const std::vector<float>& expected,
                          cudaEvent_t begin, cudaEvent_t end) {
        constexpr unsigned warmup = 100;
        constexpr unsigned iterations = 300;
        actual_.FillBytes(0xcd);
        for (unsigned i = 0; i < warmup; ++i) Launch<Columns>(fixture);
        Cuda(cudaGetLastError(), "benchmark warmup launches");
        Cuda(cudaDeviceSynchronize(), "benchmark warmup synchronize");
        Cuda(cudaEventRecord(begin), "benchmark begin record");
        for (unsigned i = 0; i < iterations; ++i) Launch<Columns>(fixture);
        Cuda(cudaGetLastError(), "benchmark timed launches");
        Cuda(cudaEventRecord(end), "benchmark end record");
        Cuda(cudaEventSynchronize(end), "benchmark end synchronize");
        float elapsed_ms = 0.0f;
        Cuda(cudaEventElapsedTime(&elapsed_ms, begin, end), "benchmark elapsed time");
        Require(elapsed_ms > 0.0f, "benchmark elapsed time must be positive");
        Equal(actual_.Read(), expected, fixture.name + " benchmark output Columns=" +
              std::to_string(Columns));
        input_.CheckGuards();
        std::printf("COLUMN_SUM_BENCH {\"rows\":%u,\"logical\":%u,\"stride\":%u,"
                    "\"columns\":%u,\"warmup\":%u,\"iterations\":%u,\"kernel_us\":%.6f}\n",
                    fixture.rows, fixture.logical, stride_, Columns, warmup, iterations,
                    static_cast<double>(elapsed_ms) * 1000.0 / iterations);
    }

    unsigned stride_;
    unsigned capacity_rows_;
    DeviceBuffer input_;
    DeviceBuffer actual_;
    DeviceBuffer oracle_;
};

void RunQuickCases() {
    Harness harness(9, 257);
    auto first = MakeFixture(9, 257, 33, 8, 0, "partial-column-CTA-old-grid-RED");
    for (unsigned row = 0; row < first.rows; ++row)
        for (unsigned column = 0; column < first.logical; ++column)
            first.input[row * 9 + column] = static_cast<float>(column + 1);
    const std::vector<float> first_literal{33, 66, 99, 132, 165, 198, 231, 264, 0};
    harness.Run(first, &first_literal);

    // The r256 term must be added to lane 0 BEFORE its r128 tree partner.
    // With 64/128 physical row lanes or serial row reduction, column 0 becomes
    // 1 instead of 0. Column 1 separately fixes the 128-before-64 tree ordering.
    auto witness = MakeFixture(9, 257, 257, 6, 0, "literal-257-row-cancellation");
    for (unsigned row = 0; row < witness.rows; ++row)
        for (unsigned column = 0; column < witness.logical; ++column)
            witness.input[row * 9 + column] = 0.0f;
    const float at_zero[] = {16777216.0f, 16777216.0f, 1.0f, -1.0f, 16777216.0f, 0.5f};
    const float at_128[] = {-16777216.0f, -16777216.0f, -1.0f, 1.0f, -16777216.0f, -0.5f};
    const float at_256[] = {1.0f, 0.0f, 0x1p-24f, 0x1p-24f, 3.0f, 0x1p-25f};
    for (unsigned column = 0; column < 6; ++column) {
        witness.input[column] = at_zero[column];
        witness.input[128 * 9 + column] = at_128[column];
        witness.input[256 * 9 + column] = at_256[column];
    }
    witness.input[64 * 9 + 1] = 1.0f;
    const std::vector<float> literal{0.0f, 1.0f, 0.0f, 0x1p-24f, 4.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    Require(Bits(RoundedAdd(RoundedAdd(16777216.0f, -16777216.0f), 1.0f)) == Bits(1.0f),
            "cancellation fixture must distinguish the reduced-lane/serial result");
    harness.Run(witness, &literal);

    const std::vector<float> zeros(9, 0.0f);
    harness.Run(MakeFixture(9, 257, 0, 8, 1, "zero-rows-ignore-poison-capacity"), &zeros);
    harness.Run(MakeFixture(9, 257, 1, 0, 2, "zero-logical-columns-ignore-poison"), &zeros);
    harness.Run(MakeFixture(9, 257, 31, 7, 3, "quick-partial-row-warp-and-padding"));
    harness.Run(MakeFixture(9, 257, 32, 9, 4, "quick-full-warp-no-column-padding"));
    harness.Run(MakeFixture(9, 257, 33, 1, 5, "quick-one-live-column-partial-warp"));
}

void RunAllCases() {
    RunQuickCases();
    const unsigned strides[] = {1, 7, 9, 31, 32, 33, 224, 257, 2560};
    for (unsigned stride : strides) {
        Harness harness(stride, 33);
        const unsigned logical = stride > 1 ? stride - 1 : 1;
        harness.Run(MakeFixture(stride, 33, 33, logical, stride,
                                "column-tail-" + std::to_string(stride)));
    }

    {
        Harness harness(257, 4097);
        // Same allocation is repeatedly shrunk/grown across every row tail.
        // Inactive rows are NaN poison, exposing capacity-for-live-row mistakes.
        const unsigned rows[] = {4097, 0, 1, 4096, 31, 4095, 32, 33, 255, 256, 257};
        for (unsigned i = 0; i < sizeof(rows) / sizeof(rows[0]); ++i)
            harness.Run(MakeFixture(257, 4097, rows[i], 253, 17 + i,
                                    "changed-live-rows-" + std::to_string(rows[i])));
    }

    {
        Harness harness(224, 4097);
        harness.Run(MakeFixture(224, 4097, 4097, 218, 23, "residual-logical218-tail4097"));
        harness.Run(MakeFixture(224, 4097, 4096, 224, 29, "residual-unpadded4096"));
        harness.Run(MakeFixture(224, 4097, 4095, 218, 31, "residual-logical218-tail4095"));
        harness.Run(MakeFixture(224, 4097, 1, 218, 37, "residual-shrink-to-one-row"));
    }

    {
        // Largest device input payload is 4096*2560*4 = 40 MiB, allocated once
        // for all five widths and all live-row changes (guards add 2 KiB).
        Harness harness(2560, 4096);
        harness.Run(MakeFixture(2560, 4096, 4096, 2556, 41, "input-projection-logical2556"));
        harness.Run(MakeFixture(2560, 4096, 257, 2556, 43, "input-projection-shrink257"));
        harness.Run(MakeFixture(2560, 4096, 0, 2556, 47, "input-projection-empty-reuse"));
        harness.Run(MakeFixture(2560, 4096, 4096, 2560, 53, "input-projection-unpadded-regrow"));
    }
}

void RunBenchmarks() {
    RunQuickCases();
    {
        Harness harness(224, 4096);
        harness.Benchmark(MakeFixture(224, 4096, 4096, 218, 61, "benchmark-residual"));
    }
    {
        Harness harness(2560, 4096);
        harness.Benchmark(MakeFixture(2560, 4096, 4096, 2556, 67, "benchmark-input-projection"));
    }
}

}  // namespace

int main(int argc, char** argv) {
    const bool quick = argc == 2 && std::strcmp(argv[1], "--quick") == 0;
    const bool benchmark = argc == 2 && std::strcmp(argv[1], "--benchmark") == 0;
    if (argc != 1 && !quick && !benchmark) {
        std::fprintf(stderr, "usage: %s [--quick|--benchmark]\n", argv[0]);
        return EXIT_FAILURE;
    }
    try {
        Require(std::numeric_limits<float>::is_iec559 && sizeof(float) == 4,
                "IEEE-754 FP32 host arithmetic is required");
        Require(std::fegetround() == FE_TONEAREST, "CPU oracle requires round-to-nearest");
        int device_count = 0;
        Cuda(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");
        Require(device_count > 0, "a CUDA device is required");
        Cuda(cudaSetDevice(0), "cudaSetDevice");
        cudaDeviceProp properties{};
        Cuda(cudaGetDeviceProperties(&properties, 0), "cudaGetDeviceProperties");
        std::printf("ColumnSum device=%s sm=%d%d mode=%s\n", properties.name,
                    properties.major, properties.minor, benchmark ? "benchmark" : quick ? "quick" : "all");
        if (benchmark) RunBenchmarks();
        else if (quick) RunQuickCases();
        else RunAllCases();
        Require(!cleanup_failed, "CUDA resource cleanup failed");
        std::printf("PASS ColumnSum: cases=%u variants=%u exact-256-lane-tree positive-zero-padding "
                    "input-immutability canaries\n", passed_cases, passed_variants);
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL ColumnSum: %s\n", error.what());
        return EXIT_FAILURE;
    }
}
