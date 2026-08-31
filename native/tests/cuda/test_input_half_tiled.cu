#include "../../cuda/input_half_tiled.cuh"

#include <cuda_fp16.h>
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

constexpr std::size_t kGuardElements = 256;
constexpr unsigned char kGuardByte = 0xa5;
constexpr unsigned char kOutputByte = 0xcd;
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

// Test-owned allocations retain initialized guards and are reused by both
// tile widths, partial-grid launches, and shrinking/growing live row counts.
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
        if (count_)
            Cuda(cudaMemcpy(get(), values.data(), count_ * sizeof(T),
                            cudaMemcpyHostToDevice), name_ + " upload");
    }

    void FillBytes(unsigned char byte) {
        if (count_) Cuda(cudaMemset(get(), byte, count_ * sizeof(T)), name_ + " reset");
    }

    std::vector<T> Read() const {
        CheckGuards();
        std::vector<T> values(count_);
        if (count_)
            Cuda(cudaMemcpy(values.data(), get(), count_ * sizeof(T),
                            cudaMemcpyDeviceToHost), name_ + " download");
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

__half HalfBits(std::uint16_t bits) {
    __half value;
    static_assert(sizeof(value) == sizeof(bits), "FP16 storage required");
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

float DecodeHalf(__half value) {
    // Host oracle decodes IEEE binary16 independently of CUDA half helpers.
    std::uint16_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    const unsigned exponent = (bits >> 10) & 31U;
    const unsigned fraction = bits & 1023U;
    const std::uint32_t sign = static_cast<std::uint32_t>(bits & 0x8000U) << 16;
    if (exponent == 0) {
        if (fraction == 0) return FromBits(sign);
        const float magnitude = static_cast<float>(fraction) * 0x1p-24f;
        return sign ? -magnitude : magnitude;
    }
    if (exponent == 31) return FromBits(sign | 0x7f800000U | (fraction << 13));
    return FromBits(sign | ((exponent + 112U) << 23) | (fraction << 13));
}

float RoundedAdd(float lhs, float rhs) {
    volatile float rounded = lhs + rhs;
    return rounded;
}

void Equal(const std::vector<float>& actual, const std::vector<float>& expected,
           unsigned stride, const std::string& context) {
    Require(actual.size() == expected.size(), context + " output size mismatch");
    if (actual.empty() ||
        std::memcmp(actual.data(), expected.data(), actual.size() * sizeof(float)) == 0)
        return;
    for (std::size_t index = 0; index < actual.size(); ++index) {
        if (Bits(actual[index]) == Bits(expected[index])) continue;
        char detail[256];
        std::snprintf(detail, sizeof(detail),
                      " row=%zu h=%zu got=%g [0x%08x] expected=%g [0x%08x]",
                      index / stride, index % stride, static_cast<double>(actual[index]),
                      Bits(actual[index]), static_cast<double>(expected[index]), Bits(expected[index]));
        throw std::runtime_error(context + detail);
    }
}

// Independent old-row-grid oracle: one CTA owns every feature of a row via
// the original feature loop. Explicit FP32 RN additions retain bias-first,
// increasing-position semantics; no production helper or kernel is reused.
__global__ void OldRowOracle(mgt_cuda::CudaMlpShape shape, unsigned logical,
                             const __half* weights, const mgt::TrainStateStorage* states,
                             unsigned rows, float* output) {
    const unsigned row = blockIdx.x;
    if (row >= rows) return;
    __shared__ std::uint64_t offsets[mgt::kStateStorageLen];
    if (threadIdx.x < shape.state_len) {
        const unsigned position = threadIdx.x;
        offsets[position] = (static_cast<std::uint64_t>(position) * shape.state_value_pad +
                             states[row].v[position]) * shape.hd1;
    }
    __syncthreads();
    const std::uint64_t bias =
        static_cast<std::uint64_t>(shape.state_len) * shape.state_value_pad * shape.hd1;
    for (unsigned h = 2 * threadIdx.x; h < shape.hd1; h += 2 * blockDim.x) {
        float2 result = make_float2(0.0f, 0.0f);
        if (h < logical) {
            result = __half22float2(*reinterpret_cast<const __half2*>(weights + bias + h));
            for (unsigned position = 0; position < shape.state_len; ++position) {
                const float2 value = __half22float2(
                    *reinterpret_cast<const __half2*>(weights + offsets[position] + h));
                result.x = __fadd_rn(result.x, value.x);
                result.y = __fadd_rn(result.y, value.y);
            }
        }
        *reinterpret_cast<float2*>(output + static_cast<std::uint64_t>(row) * shape.hd1 + h) = result;
    }
}

struct Fixture {
    std::string name;
    unsigned rows;
    unsigned logical;
    std::vector<__half> weights;
    std::vector<mgt::TrainStateStorage> states;
};

std::size_t WeightCount(const mgt_cuda::CudaMlpShape& shape) {
    return (static_cast<std::size_t>(shape.state_len) * shape.state_value_pad + 1) * shape.hd1;
}

Fixture MakeFixture(const mgt_cuda::CudaMlpShape& shape, unsigned capacity_rows,
                    unsigned rows, unsigned logical, unsigned salt, const std::string& name) {
    Require(rows > 0 && rows <= capacity_rows && logical <= shape.hd1 &&
                shape.hd1 > 0 && (shape.hd1 % 2) == 0 && (logical % 2) == 0 &&
                shape.state_len > 0 && shape.state_len <= 72 &&
                shape.state_value_pad > 0 && shape.state_value_pad <= 72,
            name + " invalid even-path fixture dimensions");
    Fixture fixture{name, rows, logical,
                    std::vector<__half>(WeightCount(shape), HalfBits(0x7e35U)),
                    std::vector<mgt::TrainStateStorage>(capacity_rows)};
    // Every padded embedding/bias feature remains quiet NaN. Active values
    // cover signed zero, binary16 extrema, normal and subnormal half operands.
    constexpr std::uint16_t values[] = {
        0x7bff, 0x3c00, 0xfbff, 0x4200, 0x0001, 0x8001, 0x3800, 0xb800,
        0x1400, 0x9400, 0x0000, 0x8000, 0x0400, 0x8400, 0x4400, 0xc700};
    const std::size_t tables = static_cast<std::size_t>(shape.state_len) * shape.state_value_pad + 1;
    for (std::size_t table = 0; table < tables; ++table)
        for (unsigned h = 0; h < logical; ++h)
            fixture.weights[table * shape.hd1 + h] = HalfBits(values[(17 * table + 13 * h + salt) % 16]);
    for (unsigned row = 0; row < capacity_rows; ++row) {
        // Physical state padding and inactive rows contain invalid values;
        // active positions always stay within [0, state_value_pad).
        for (unsigned position = 0; position < sizeof(fixture.states[row].v); ++position)
            fixture.states[row].v[position] = static_cast<mgt::StateValue>(
                (row + position + salt) % 2 == 0 ? 255 : shape.state_value_pad);
        if (row < rows)
            for (unsigned position = 0; position < shape.state_len; ++position)
                fixture.states[row].v[position] = static_cast<mgt::StateValue>(
                    (row + 17 * position + salt) % shape.state_value_pad);
    }
    return fixture;
}

std::vector<float> CpuOracle(const mgt_cuda::CudaMlpShape& shape, const Fixture& fixture) {
    std::vector<float> expected(fixture.states.size() * shape.hd1, FromBits(0xcdcdcdcdU));
    const std::size_t bias = static_cast<std::size_t>(shape.state_len) * shape.state_value_pad * shape.hd1;
    for (unsigned row = 0; row < fixture.rows; ++row) {
        for (unsigned h = 0; h < shape.hd1; ++h) {
            float value = 0.0f;
            if (h < fixture.logical) {
                value = DecodeHalf(fixture.weights[bias + h]);
                for (unsigned position = 0; position < shape.state_len; ++position) {
                    const std::size_t table = static_cast<std::size_t>(position) * shape.state_value_pad +
                                              fixture.states[row].v[position];
                    value = RoundedAdd(value, DecodeHalf(fixture.weights[table * shape.hd1 + h]));
                }
            }
            expected[static_cast<std::size_t>(row) * shape.hd1 + h] = value;
        }
    }
    return expected;
}

class Harness {
public:
    Harness(mgt_cuda::CudaMlpShape shape, unsigned capacity_rows)
        : shape_(shape), capacity_rows_(capacity_rows),
          weights_(WeightCount(shape), "weights"), states_(capacity_rows, "states"),
          actual_(static_cast<std::size_t>(capacity_rows) * shape.hd1, "production output"),
          oracle_(static_cast<std::size_t>(capacity_rows) * shape.hd1, "old-row output") {}

    void RunPartialTile(const Fixture& fixture) {
        Upload(fixture);
        const auto expected = Reference(fixture);
        // This is an observable tile-ownership contract, not a source check:
        // deliberately launch only tile y=0 and require later tiles untouched.
        RunVariant<128>(fixture, expected, true);
        RunVariant<256>(fixture, expected, true);
        CheckInputs(fixture);
        Finish(fixture, "single-tile ownership");
    }

    void Run(const Fixture& fixture, const std::vector<float>* literal_columns = nullptr) {
        Upload(fixture);
        const auto expected = Reference(fixture);
        if (literal_columns) {
            Require(literal_columns->size() == shape_.hd1, fixture.name + " literal width mismatch");
            std::vector<float> literal(expected.size(), FromBits(0xcdcdcdcdU));
            for (unsigned row = 0; row < fixture.rows; ++row)
                for (unsigned h = 0; h < shape_.hd1; ++h)
                    literal[static_cast<std::size_t>(row) * shape_.hd1 + h] = (*literal_columns)[h];
            Equal(expected, literal, shape_.hd1, fixture.name + " hand-derived results");
        }
        RunVariant<128>(fixture, expected, false);
        RunVariant<256>(fixture, expected, false);
        CheckInputs(fixture);
        Finish(fixture, "complete tiled grid");
    }

    void Benchmark(const Fixture& fixture) {
        Upload(fixture);
        const auto expected = Reference(fixture);
        Event begin;
        Event end;
        BenchmarkVariant<0>(fixture, expected, begin.get(), end.get());
        BenchmarkVariant<128>(fixture, expected, begin.get(), end.get());
        BenchmarkVariant<256>(fixture, expected, begin.get(), end.get());
        CheckInputs(fixture);
    }

private:
    void Upload(const Fixture& fixture) {
        Require(fixture.rows > 0 && fixture.rows <= capacity_rows_ && fixture.logical <= shape_.hd1 &&
                    fixture.logical % 2 == 0 && fixture.weights.size() == WeightCount(shape_) &&
                    fixture.states.size() == capacity_rows_, fixture.name + " allocation mismatch");
        weights_.Put(fixture.weights);
        states_.Put(fixture.states);
    }

    std::vector<float> Reference(const Fixture& fixture) {
        oracle_.FillBytes(kOutputByte);
        OldRowOracle<<<fixture.rows, 256>>>(shape_, fixture.logical, weights_.get(), states_.get(),
                                          fixture.rows, oracle_.get());
        Cuda(cudaGetLastError(), fixture.name + " old-row oracle launch");
        Cuda(cudaDeviceSynchronize(), fixture.name + " oracle synchronize");
        auto expected = oracle_.Read();
        // Avoid a 4096*2560*72 host reduction in the production-size test.
        if (static_cast<std::uint64_t>(fixture.rows) * fixture.logical * shape_.state_len <= 4000000)
            Equal(expected, CpuOracle(shape_, fixture), shape_.hd1,
                  fixture.name + " independent CPU vs old-row GPU oracle");
        return expected;
    }

    void CheckInputs(const Fixture& fixture) const {
        const auto after_weights = weights_.Read();
        Require(std::memcmp(after_weights.data(), fixture.weights.data(),
                            after_weights.size() * sizeof(__half)) == 0,
                fixture.name + " embedding, bias or NaN weight padding mutated");
        const auto after_states = states_.Read();
        Require(std::memcmp(after_states.data(), fixture.states.data(),
                            after_states.size() * sizeof(mgt::TrainStateStorage)) == 0,
                fixture.name + " input states, inactive rows or state padding mutated");
    }

    template <unsigned Threads> void Launch(const Fixture& fixture, bool partial = false) {
        if constexpr (Threads == 0) {
            OldRowOracle<<<fixture.rows, 256>>>(shape_, fixture.logical, weights_.get(), states_.get(),
                                              fixture.rows, actual_.get());
        } else {
            const unsigned tiles = partial ? 1 : (shape_.hd1 + 2 * Threads - 1) / (2 * Threads);
            mgt_cuda::detail::InputHalf2Row<Threads><<<dim3(fixture.rows, tiles), Threads>>>(
                shape_, fixture.logical, weights_.get(), states_.get(), fixture.rows, actual_.get());
        }
    }

    template <unsigned Threads>
    void RunVariant(const Fixture& fixture, const std::vector<float>& full_expected, bool partial) {
        const std::string context = fixture.name + " Threads=" + std::to_string(Threads);
        auto expected = full_expected;
        if (partial)
            for (unsigned row = 0; row < fixture.rows; ++row)
                for (unsigned h = 2 * Threads; h < shape_.hd1; ++h)
                    expected[static_cast<std::size_t>(row) * shape_.hd1 + h] = FromBits(0xcdcdcdcdU);
        actual_.FillBytes(kOutputByte);
        Launch<Threads>(fixture, partial);
        Cuda(cudaGetLastError(), context + " production launch");
        Cuda(cudaDeviceSynchronize(), context + " synchronize");
        const auto actual = actual_.Read();
        Equal(actual, expected, shape_.hd1, context + (partial ? " untouched unlaunched tiles" : " exact sums"));
        for (unsigned row = 0; row < fixture.rows; ++row)
            for (unsigned h = fixture.logical; h < shape_.hd1; ++h)
                if (!partial || h < 2 * Threads)
                    Require(Bits(actual[static_cast<std::size_t>(row) * shape_.hd1 + h]) == 0,
                            context + " padded output must be positive zero");
        weights_.CheckGuards();
        states_.CheckGuards();
        ++passed_variants;
    }

    template <unsigned Threads>
    void BenchmarkVariant(const Fixture& fixture, const std::vector<float>& expected,
                          cudaEvent_t begin, cudaEvent_t end) {
        constexpr unsigned warmup = 100;
        constexpr unsigned iterations = 300;
        actual_.FillBytes(kOutputByte);
        for (unsigned i = 0; i < warmup; ++i) Launch<Threads>(fixture);
        Cuda(cudaGetLastError(), "benchmark warmup launches");
        Cuda(cudaDeviceSynchronize(), "benchmark warmup synchronize");
        Cuda(cudaEventRecord(begin), "benchmark begin record");
        for (unsigned i = 0; i < iterations; ++i) Launch<Threads>(fixture);
        Cuda(cudaGetLastError(), "benchmark timed launches");
        Cuda(cudaEventRecord(end), "benchmark end record");
        Cuda(cudaEventSynchronize(end), "benchmark end synchronize");
        float elapsed_ms = 0.0f;
        Cuda(cudaEventElapsedTime(&elapsed_ms, begin, end), "benchmark elapsed time");
        Require(elapsed_ms > 0.0f, "benchmark elapsed time must be positive");
        Equal(actual_.Read(), expected, shape_.hd1, fixture.name + " benchmark output");
        weights_.CheckGuards();
        states_.CheckGuards();
        std::printf("INPUT_HALF_BENCH {\"kind\":\"%s\",\"threads\":%u,\"rows\":%u,"
                    "\"state_len\":%u,\"state_value_pad\":%u,\"logical\":%u,\"hd1\":%u,"
                    "\"warmup\":%u,\"iterations\":%u,\"kernel_us\":%.6f}\n",
                    Threads == 0 ? "old_row_reference" : "production_tiled",
                    Threads == 0 ? 256 : Threads, fixture.rows, shape_.state_len,
                    shape_.state_value_pad, fixture.logical, shape_.hd1, warmup, iterations,
                    static_cast<double>(elapsed_ms) * 1000.0 / iterations);
    }

    void Finish(const Fixture& fixture, const char* coverage) {
        ++passed_cases;
        std::printf("PASS %-37s rows=%u capacity=%u logical=%u hd1=%u shape=%ux%u T=128,256 %s\n",
                    fixture.name.c_str(), fixture.rows, capacity_rows_, fixture.logical,
                    shape_.hd1, shape_.state_len, shape_.state_value_pad, coverage);
    }

    mgt_cuda::CudaMlpShape shape_;
    unsigned capacity_rows_;
    DeviceBuffer<__half> weights_;
    DeviceBuffer<mgt::TrainStateStorage> states_;
    DeviceBuffer<float> actual_;
    DeviceBuffer<float> oracle_;
};

void RunLiteralCancellation() {
    const mgt_cuda::CudaMlpShape shape{2, 4, 18, 2, 1, 1};
    auto fixture = MakeFixture(shape, 3, 3, 10, 0, "bias-position-order-half-extremes");
    const float bias[] = {65504.0f, 65504.0f, 0x1p-10f, -65504.0f, 1.0f,
                          -1.0f, 65504.0f, -0.0f, 65504.0f, 0.0f};
    const float p0[] = {0x1p-10f, -65504.0f, 65504.0f, 0x1p-10f, 0x1p-24f,
                        0x1p-24f, 65504.0f, -0.0f, 32.0f, 0x1p-14f};
    const float p1[] = {-65504.0f, 0x1p-10f, -65504.0f, 65504.0f, -1.0f,
                        1.0f, -65504.0f, -0.0f, -65504.0f, -0x1p-24f};
    for (unsigned h = 0; h < 10; ++h) {
        fixture.weights[8 * shape.hd1 + h] = __float2half_rn(bias[h]);
        for (unsigned value = 0; value < 4; ++value) {
            fixture.weights[value * shape.hd1 + h] = __float2half_rn(p0[h]);
            fixture.weights[(4 + value) * shape.hd1 + h] = __float2half_rn(p1[h]);
        }
    }
    const std::vector<float> literal{
        0.0f, 0x1p-10f, 0.0f, 0.0f, 0.0f, 0x1p-24f, 65504.0f, -0.0f,
        32.0f, 0x1.ff8p-15f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    // Swapping p0/p1 or adding bias last changes at least one literal column;
    // FP16 accumulation overflows column 6, whereas FP32 returns finite 65504.
    Require(Bits(RoundedAdd(RoundedAdd(65504.0f, -65504.0f), 0x1p-10f)) == Bits(0x1p-10f),
            "literal fixture must distinguish changed position order");
    Harness harness(shape, 3);
    harness.Run(fixture, &literal);
}

void RunQuickCases() {
    const mgt_cuda::CudaMlpShape shape{2, 4, 514, 2, 1, 1};
    Harness harness(shape, 3);
    auto fixture = MakeFixture(shape, 3, 3, 512, 7, "single-tile-old-row-loop-RED");
    // Old row-loop kernel writes h=256 despite a T128 single-tile launch.
    // Output remains allocated; RED is an exact sentinel mismatch, not OOB.
    harness.RunPartialTile(fixture);
    fixture.name = "full-grid-partial-final-feature-tile";
    harness.Run(fixture);
    harness.Run(MakeFixture(shape, 3, 1, 514, 11, "one-row-unpadded-live-suffix"));
    harness.Run(MakeFixture(shape, 3, 3, 0, 13, "all-padding-NaNs-produce-positive-zero"));
    RunLiteralCancellation();
}

void RunAllCases() {
    RunQuickCases();
    const unsigned widths[] = {254, 256, 258, 510, 512, 514, 1022, 1024, 1026, 2560};
    for (unsigned width : widths) {
        const mgt_cuda::CudaMlpShape shape{3, 7, width, 2, 1, 1};
        Harness harness(shape, 17);
        harness.Run(MakeFixture(shape, 17, 17, width - 2, width,
                                "even-feature-tail-" + std::to_string(width)));
    }
    {
        // Maximum input weights: (72*72+1)*2560*2 = 26,547,200 bytes.
        // Each full-capacity output: 4096*2560*4 = 40 MiB. Reused for both T's.
        const mgt_cuda::CudaMlpShape shape{72, 72, 2560, 2, 1, 1};
        Harness harness(shape, 4096);
        const unsigned rows[] = {1, 17, 4096, 3};
        for (unsigned i = 0; i < sizeof(rows) / sizeof(rows[0]); ++i)
            harness.Run(MakeFixture(shape, 4096, rows[i], 2556, 19 + i,
                                    "production72x72-rows-" + std::to_string(rows[i])));
    }
}

void RunBenchmarks() {
    RunQuickCases();
    const mgt_cuda::CudaMlpShape shape{72, 72, 2560, 2, 1, 1};
    Harness harness(shape, 4096);
    const auto fixture = MakeFixture(shape, 4096, 4096, 2556, 31, "benchmark-production72x72");
    harness.Run(fixture);
    harness.Benchmark(fixture);
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
        std::printf("InputHalf2Row device=%s sm=%d%d mode=%s\n", properties.name,
                    properties.major, properties.minor, benchmark ? "benchmark" : quick ? "quick" : "all");
        if (benchmark) RunBenchmarks();
        else if (quick) RunQuickCases();
        else RunAllCases();
        Require(!cleanup_failed, "CUDA resource cleanup failed");
        std::printf("PASS InputHalf2Row: cases=%u variants=%u tile-ownership exact-position-order "
                    "positive-zero-padding input-immutability canaries\n", passed_cases, passed_variants);
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL InputHalf2Row: %s\n", error.what());
        return EXIT_FAILURE;
    }
}
