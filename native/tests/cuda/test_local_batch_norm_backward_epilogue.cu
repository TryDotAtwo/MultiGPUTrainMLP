#include "mgt_cuda/local_batch_norm.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cfenv>
#include <climits>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {
constexpr std::size_t kGuard = 256;
constexpr unsigned char kGuardByte = 0xa5;
bool cleanup_failed = false;
unsigned apply_cases = 0, integration_cases = 0, rejected_cases = 0;

void Require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}
void Cuda(cudaError_t status, const std::string& operation) {
    if (status != cudaSuccess)
        throw std::runtime_error(operation + ": " + cudaGetErrorString(status));
}
void Cleanup(cudaError_t status, const char* operation) noexcept {
    if (status == cudaSuccess) return;
    cleanup_failed = true;
    std::fprintf(stderr, "%s: %s\n", operation, cudaGetErrorString(status));
}

class Stream {
public:
    Stream() { Cuda(cudaStreamCreateWithFlags(&value_, cudaStreamNonBlocking), "create stream"); }
    ~Stream() { Cleanup(cudaStreamDestroy(value_), "destroy stream"); }
    Stream(const Stream&) = delete;
    Stream& operator=(const Stream&) = delete;
    cudaStream_t get() const { return value_; }
    void Sync() const { Cuda(cudaStreamSynchronize(value_), "synchronize test stream"); }
private:
    cudaStream_t value_ = nullptr;
};

// Test-owned initialized payloads and two full-CTA canaries. Allocations are
// reused across calls with shrinking/growing live rows, not made by production.
template <class T> class Device {
public:
    Device(std::size_t count, const char* name) : count_(count), name_(name) {
        Cuda(cudaMalloc(&allocation_, (count_ + 2 * kGuard) * sizeof(T)), name_ + " allocate");
        try {
            Cuda(cudaMemset(allocation_, kGuardByte, (count_ + 2 * kGuard) * sizeof(T)),
                 name_ + " initialize canaries");
        } catch (...) {
            Cleanup(cudaFree(allocation_), "free failed allocation");
            allocation_ = nullptr;
            throw;
        }
    }
    ~Device() { if (allocation_) Cleanup(cudaFree(allocation_), "free test allocation"); }
    Device(const Device&) = delete;
    Device& operator=(const Device&) = delete;
    T* get() const { return allocation_ + kGuard; }
    std::size_t size() const { return count_; }
    void Put(const std::vector<T>& values) {
        Require(values.size() == count_, name_ + " upload size");
        Cuda(cudaMemcpy(get(), values.data(), count_ * sizeof(T), cudaMemcpyHostToDevice),
             name_ + " upload");
    }
    void Poison() { Cuda(cudaMemset(get(), 0xcd, count_ * sizeof(T)), name_ + " poison"); }
    std::vector<T> Read() const {
        CheckGuards();
        std::vector<T> result(count_);
        Cuda(cudaMemcpy(result.data(), get(), count_ * sizeof(T), cudaMemcpyDeviceToHost),
             name_ + " download");
        return result;
    }
    void CheckGuards() const {
        unsigned char guards[2 * kGuard * sizeof(T)];
        Cuda(cudaMemcpy(guards, allocation_, kGuard * sizeof(T), cudaMemcpyDeviceToHost),
             name_ + " leading canary");
        Cuda(cudaMemcpy(guards + kGuard * sizeof(T), get() + count_, kGuard * sizeof(T),
                        cudaMemcpyDeviceToHost), name_ + " trailing canary");
        for (unsigned char byte : guards)
            if (byte != kGuardByte) throw std::runtime_error(name_ + " allocation canary changed");
    }
private:
    std::size_t count_;
    std::string name_;
    T* allocation_ = nullptr;
};

std::uint32_t Bits(float value) {
    std::uint32_t result;
    static_assert(sizeof(result) == sizeof(value), "FP32 storage");
    std::memcpy(&result, &value, sizeof(result));
    return result;
}
std::uint16_t Bits(__half value) {
    std::uint16_t result;
    static_assert(sizeof(result) == sizeof(value), "FP16 storage");
    std::memcpy(&result, &value, sizeof(result));
    return result;
}
template <class T> void Equal(const std::vector<T>& actual, const std::vector<T>& expected,
                              const std::string& name) {
    Require(actual.size() == expected.size(), name + " size mismatch");
    if (std::memcmp(actual.data(), expected.data(), actual.size() * sizeof(T)) == 0) return;
    for (std::size_t i = 0; i < actual.size(); ++i) {
        if (std::memcmp(&actual[i], &expected[i], sizeof(T)) != 0)
            throw std::runtime_error(name + " bit mismatch at index " + std::to_string(i));
    }
}

// Independent old apply expression, not a call to the new API or its helpers.
// Ordinary expressions intentionally retain the old NVCC contraction policy.
// Fixed supplied statistics make exact comparison independent of BN atomics.
__global__ void OldBackwardApply(const float* dy, int rows, int cols, int stride,
    const float* gamma, const float* inv, const float* normalized,
    const float* dgamma, const float* dbeta, float* dx) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= rows * stride) return;
    const int col = index % stride;
    if (col < cols) {
        const float scale = gamma[col] * inv[col] / rows;
        dx[index] = scale *
            (rows * dy[index] - dbeta[col] - normalized[index] * dgamma[col]);
    } else {
        dx[index] = 0.0f;
    }
}

struct Fixture {
    int rows, cols, stride;
    std::vector<float> dy, normalized, gamma, inv, dgamma, dbeta;
    Fixture(int capacity_rows, int live_rows, int logical, int physical)
        : rows(live_rows), cols(logical), stride(physical),
          dy(static_cast<std::size_t>(capacity_rows) * physical,
             std::numeric_limits<float>::quiet_NaN()), normalized(dy),
          gamma(logical), inv(logical), dgamma(logical), dbeta(logical) {
        for (int c = 0; c < cols; ++c) {
            gamma[c] = static_cast<float>(c % 9 - 4) / 4.0f;
            inv[c] = static_cast<float>(c % 7 + 1) / 8.0f;
            dgamma[c] = static_cast<float>(c % 11 - 5) / 32.0f;
            dbeta[c] = static_cast<float>(c % 13 - 6) / 16.0f;
        }
        for (int r = 0; r < rows; ++r) for (int c = 0; c < cols; ++c) {
            const std::size_t i = static_cast<std::size_t>(r) * stride + c;
            dy[i] = static_cast<float>((r * 13 + c * 7) % 65 - 32) / 8.0f;
            normalized[i] = static_cast<float>((r * 5 + c * 11) % 33 - 16) / 16.0f;
        }
    }
};

void CheckCpu(const Fixture& f, const std::vector<float>& actual, const std::string& name) {
    // Independent double-precision mathematical evaluation, with a tolerance
    // for the old FP32 intermediates/FMA. Bit exactness has a separate GPU gate.
    for (int r = 0; r < f.rows; ++r) for (int c = 0; c < f.stride; ++c) {
        const std::size_t i = static_cast<std::size_t>(r) * f.stride + c;
        if (c >= f.cols) {
            if (Bits(actual[i]) != 0)
                throw std::runtime_error(name + " padding is not positive zero");
            continue;
        }
        const double scale = static_cast<double>(f.gamma[c]) * f.inv[c] / f.rows;
        const double expected = scale * (static_cast<double>(f.rows) * f.dy[i] -
            f.dbeta[c] - static_cast<double>(f.normalized[i]) * f.dgamma[c]);
        if (!std::isfinite(actual[i]) || !std::isfinite(expected) ||
            std::fabs(static_cast<double>(actual[i]) - expected) > 2e-6 + 2e-5 * std::fabs(expected))
            throw std::runtime_error(name + " CPU formula mismatch at " + std::to_string(i));
    }
}
void CheckHalf(const std::vector<float>& authoritative, const std::vector<__half>& half,
               std::size_t live, const std::string& name) {
    for (std::size_t i = 0; i < live; ++i)
        if (Bits(half[i]) != Bits(__float2half_rn(authoritative[i])))
            throw std::runtime_error(name + " RN half mirror mismatch at " + std::to_string(i));
}

class Harness {
public:
    Harness(int capacity_rows, int cols, int stride)
        : capacity_rows_(capacity_rows), cols_(cols), stride_(stride),
          dy_(static_cast<std::size_t>(capacity_rows) * stride, "dy"),
          normalized_(dy_.size(), "normalized"), dx_(dy_.size(), "dx"),
          oracle_(dy_.size(), "old dx"), half_(dy_.size(), "half mirror"),
          gamma_(cols, "gamma"), inv_(cols, "inverse std"),
          dgamma_(cols, "dgamma"), dbeta_(cols, "dbeta") {}

    // Detects missing mirrors, FP16 replacing FP32, changed arithmetic, early
    // tail returns, padding reads, stale live-row extents, and broken inplace.
    void Run(const Fixture& f, bool mirror, bool inplace, const std::string& name,
             const std::vector<float>* literal = nullptr,
             const std::vector<std::uint16_t>* literal_half = nullptr) {
        Require(f.rows > 0 && f.rows <= capacity_rows_ && f.cols == cols_ &&
            f.stride == stride_ && f.dy.size() == dy_.size(), name + " fixture extent");
        dy_.Put(f.dy); normalized_.Put(f.normalized); gamma_.Put(f.gamma);
        inv_.Put(f.inv); dgamma_.Put(f.dgamma); dbeta_.Put(f.dbeta);
        dx_.Poison(); oracle_.Poison(); half_.Poison();
        Cuda(cudaStreamSynchronize(nullptr), "finish apply fixture uploads");
        const auto before_half = half_.Read();
        const auto before_dx = dx_.Read();
        const std::size_t live = static_cast<std::size_t>(f.rows) * f.stride;
        OldBackwardApply<<<static_cast<unsigned>((live + 255) / 256), 256, 0, stream_.get()>>>(
            dy_.get(), f.rows, f.cols, f.stride, gamma_.get(), inv_.get(), normalized_.get(),
            dgamma_.get(), dbeta_.get(), oracle_.get());
        Cuda(cudaGetLastError(), "old apply oracle launch");
        mgt_cuda::LocalBatchNormBackwardEpilogue ep;
        ep.half_output = mirror ? half_.get() : nullptr;
        const auto status = mgt_cuda::LaunchLocalStridedBatchNormBackwardApply(
            dy_.get(), f.rows, f.cols, f.stride, gamma_.get(), inv_.get(), normalized_.get(),
            dgamma_.get(), dbeta_.get(), inplace ? dy_.get() : dx_.get(), ep, stream_.get());
        stream_.Sync();
        Require(status == mgt::Status::kOk, name + " valid apply rejected");
        const auto actual = inplace ? dy_.Read() : dx_.Read();
        auto expected = oracle_.Read();
        if (inplace) std::copy(f.dy.begin() + live, f.dy.end(), expected.begin() + live);
        Equal(actual, expected, name + " FP32 vs old GPU + inactive capacity");
        CheckCpu(f, actual, name);
        const auto actual_half = half_.Read();
        if (mirror) {
            CheckHalf(actual, actual_half, live, name);
            for (std::size_t i = live; i < actual_half.size(); ++i)
                if (Bits(actual_half[i]) != Bits(before_half[i]))
                    throw std::runtime_error(name + " half capacity overwritten");
        } else Equal(actual_half, before_half, name + " disabled mirror changed");
        if (literal) {
            Require(literal->size() == live, name + " literal extent");
            Equal(std::vector<float>(actual.begin(), actual.begin() + live), *literal, name + " literal FP32");
        }
        if (literal_half) {
            Require(mirror && literal_half->size() == live, name + " half literal extent");
            for (std::size_t i = 0; i < live; ++i)
                Require(Bits(actual_half[i]) == (*literal_half)[i], name + " literal half bits");
        }
        if (!inplace) Equal(dy_.Read(), f.dy, name + " dy modified");
        else Equal(dx_.Read(), before_dx, name + " unused out-of-place buffer modified");
        Equal(normalized_.Read(), f.normalized, name + " normalized modified");
        Equal(gamma_.Read(), f.gamma, name + " gamma modified");
        Equal(inv_.Read(), f.inv, name + " inverse modified");
        Equal(dgamma_.Read(), f.dgamma, name + " dgamma modified");
        Equal(dbeta_.Read(), f.dbeta, name + " dbeta modified");
        ++apply_cases;
        std::printf("PASS apply %-28s rows=%d cols=%d stride=%d mirror=%d inplace=%d\n",
            name.c_str(), f.rows, f.cols, f.stride, mirror, inplace);
    }
private:
    int capacity_rows_, cols_, stride_;
    Stream stream_;
    Device<float> dy_, normalized_, dx_, oracle_;
    Device<__half> half_;
    Device<float> gamma_, inv_, dgamma_, dbeta_;
};

void LiteralRounding() {
    Fixture f(1, 1, 11, 13);
    std::fill(f.gamma.begin(), f.gamma.end(), 1.0f);
    std::fill(f.inv.begin(), f.inv.end(), 1.0f);
    std::fill(f.dgamma.begin(), f.dgamma.end(), 0.0f);
    std::fill(f.dbeta.begin(), f.dbeta.end(), 0.0f);
    const std::vector<float> literal{0.0f, -0.0f, 1.0f + 0x1p-11f, 1.0f + 3 * 0x1p-11f,
        0x1p-25f, -0x1p-25f, 65504.0f, 65520.0f, -65520.0f, 32768.0f, -32768.0f, 0.0f, 0.0f};
    for (int c = 0; c < f.cols; ++c) { f.dy[c] = literal[c]; f.normalized[c] = 0.0f; }
    const std::vector<std::uint16_t> half_bits{0x0000, 0x8000, 0x3c00, 0x3c02,
        0x0000, 0x8000, 0x7bff, 0x7c00, 0xfc00, 0x7800, 0xf800, 0x0000, 0x0000};
    Harness harness(1, f.cols, f.stride);
    harness.Run(f, true, false, "literal-RN-ties-overflow", &literal, &half_bits);
    harness.Run(f, true, true, "literal-RN-inplace", &literal, &half_bits);
}

// Detects a public full-backward wrapper which ignores the new descriptor or
// changes old statistics/copies. These small shapes have one row-partial CTA
// per column tile, so two launches do not compare different atomic sum orders.
void FullBackward(bool inplace) {
    Fixture f(17, 17, 5, 7);
    Stream stream;
    Device<float> dy(f.dy.size(), "full dy"), second_dy(f.dy.size(), "full second dy"),
        normalized(f.dy.size(), "full normalized"), old_dx(f.dy.size(), "full old dx"),
        dx(f.dy.size(), "full dx"), gamma(f.cols, "full gamma"), inv(f.cols, "full inv"),
        old_dg(f.cols, "full old dg"), old_db(f.cols, "full old db"),
        dg(f.cols, "full dg"), db(f.cols, "full db"),
        old_stats(2 * f.cols, "full old stats"), stats(2 * f.cols, "full stats");
    Device<__half> half(f.dy.size(), "full mirror");
    dy.Put(f.dy); second_dy.Put(f.dy); normalized.Put(f.normalized);
    gamma.Put(f.gamma); inv.Put(f.inv); old_dx.Poison(); dx.Poison(); half.Poison();
    Cuda(cudaStreamSynchronize(nullptr), "finish full backward fixture uploads");
    const auto old_status = mgt_cuda::LaunchLocalStridedBatchNormBackward(dy.get(),
        f.rows, f.cols, f.stride, gamma.get(), inv.get(), normalized.get(), old_dx.get(),
        old_dg.get(), old_db.get(), old_stats.get(), stream.get());
    mgt_cuda::LocalBatchNormBackwardEpilogue ep;
    ep.half_output = half.get();
    const auto status = mgt_cuda::LaunchLocalStridedBatchNormBackward(second_dy.get(),
        f.rows, f.cols, f.stride, gamma.get(), inv.get(), normalized.get(),
        inplace ? second_dy.get() : dx.get(), dg.get(), db.get(), stats.get(), stream.get(), ep);
    stream.Sync();
    Require(old_status == mgt::Status::kOk && status == mgt::Status::kOk, "full backward rejected");
    const auto actual = inplace ? second_dy.Read() : dx.Read();
    Equal(actual, old_dx.Read(), "full backward FP32 unchanged");
    CheckHalf(actual, half.Read(), actual.size(), "full backward mirror");
    Equal(dg.Read(), old_dg.Read(), "full backward dgamma unchanged");
    Equal(db.Read(), old_db.Read(), "full backward dbeta unchanged");
    Equal(stats.Read(), old_stats.Read(), "full backward statistics unchanged");
    Equal(dy.Read(), f.dy, "full backward original dy immutable");
    if (!inplace) Equal(second_dy.Read(), f.dy, "full backward second dy immutable");
    Equal(normalized.Read(), f.normalized, "full backward normalized immutable");
    Equal(gamma.Read(), f.gamma, "full backward gamma immutable");
    Equal(inv.Read(), f.inv, "full backward inverse immutable");
    dx.CheckGuards(); second_dy.CheckGuards();
    ++integration_cases;
}

// Detects missing validation and rejection after work has already been queued.
// Oversized physical allocations keep deliberately overlapping aliases inside
// test-owned memory; no test needs an invalid device address to establish RED.
void InvalidCases() {
    Stream stream;
    Device<float> dy(32, "invalid dy"), normalized(32, "invalid normalized"),
        gamma(32, "invalid gamma"), inv(32, "invalid inv"), dg(32, "invalid dg"),
        db(32, "invalid db"), dx(32, "invalid dx"), stats(32, "invalid stats");
    Device<__half> half(32, "invalid mirror");
    Device<float>* arrays[]{&dy, &normalized, &gamma, &inv, &dg, &db, &dx, &stats};
    std::vector<std::vector<float>> before;
    for (auto* array : arrays) before.push_back(array->Read());
    const auto before_half = half.Read();
    auto unchanged = [&](const std::string& name) {
        stream.Sync();
        for (std::size_t i = 0; i < before.size(); ++i)
            Equal(arrays[i]->Read(), before[i], name + " float buffer " + std::to_string(i));
        Equal(half.Read(), before_half, name + " mirror");
    };
    struct Args {
        const float *dy, *gamma, *inv, *normalized, *dg, *db;
        float* dx;
        __half* half;
        int rows = 2, cols = 3, stride = 4;
    };
    const Args good{dy.get(), gamma.get(), inv.get(), normalized.get(), dg.get(), db.get(),
                    dx.get(), half.get()};
    auto reject = [&](const Args& a, const std::string& name) {
        mgt_cuda::LocalBatchNormBackwardEpilogue ep;
        ep.half_output = a.half;
        const auto status = mgt_cuda::LaunchLocalStridedBatchNormBackwardApply(a.dy,
            a.rows, a.cols, a.stride, a.gamma, a.inv, a.normalized, a.dg, a.db,
            a.dx, ep, stream.get());
        unchanged(name);
        Require(status == mgt::Status::kInvalidConfig, name + " accepted invalid apply");
        ++rejected_cases;
    };
    for (int field = 0; field < 7; ++field) {
        auto a = good;
        switch (field) {
            case 0: a.dy = nullptr; break;
            case 1: a.gamma = nullptr; break;
            case 2: a.inv = nullptr; break;
            case 3: a.normalized = nullptr; break;
            case 4: a.dg = nullptr; break;
            case 5: a.db = nullptr; break;
            case 6: a.dx = nullptr; break;
        }
        reject(a, "null-field-" + std::to_string(field));
    }
    for (int invalid : {0, -1}) {
        auto a = good; a.rows = invalid; reject(a, "invalid rows");
        a = good; a.cols = invalid; reject(a, "invalid cols");
        a = good; a.stride = invalid; reject(a, "invalid stride");
    }
    { auto a = good; a.stride = 2; reject(a, "stride below cols"); }
    { auto a = good; a.rows = INT_MAX; reject(a, "row-stride integer overflow"); }
    { auto a = good; a.dx = dy.get() + 1; reject(a, "partial dy/dx overlap"); }
    { auto a = good; a.dy = dx.get() + 1; reject(a, "reverse partial dy/dx overlap"); }
    for (auto* target : {&normalized, &gamma, &inv, &dg, &db}) {
        auto a = good; a.dx = target->get(); reject(a, "dx aliases read-only input");
        a.dx = target->get() + 1; reject(a, "partial dx/read-only overlap");
    }
    for (auto* target : {&dy, &normalized, &gamma, &inv, &dg, &db, &dx}) {
        auto a = good;
        a.half = reinterpret_cast<__half*>(target->get());
        reject(a, "half aliases float input/output");
        // Start at the final half-word of the live float extent, not merely at
        // equal pointers, to exercise interval-overlap checks in both layouts.
        a.half += (target == &dy || target == &normalized || target == &dx) ? 15 : 5;
        reject(a, "half partially overlaps float input/output");
    }
    {
        auto a = good;
        a.half = reinterpret_cast<__half*>(dy.get());
        a.dy = dy.get() + 3;
        reject(a, "reverse half overlap uses physical stride extent");
    }
    for (auto* target : {&dy, &normalized, &gamma, &inv, &dg, &db, &dx, &stats}) {
        mgt_cuda::LocalBatchNormBackwardEpilogue ep;
        ep.half_output = reinterpret_cast<__half*>(target->get());
        const auto status = mgt_cuda::LaunchLocalStridedBatchNormBackward(dy.get(),
            2, 3, 4, gamma.get(), inv.get(), normalized.get(), dx.get(), dg.get(), db.get(),
            stats.get(), stream.get(), ep);
        unchanged("full-backward invalid half alias");
        Require(status == mgt::Status::kInvalidConfig, "full backward accepted half alias");
        ++rejected_cases;
    }
    {
        mgt_cuda::LocalBatchNormBackwardEpilogue ep;
        ep.half_output = reinterpret_cast<__half*>(stats.get()) + 11;
        const auto status = mgt_cuda::LaunchLocalStridedBatchNormBackward(dy.get(),
            2, 3, 4, gamma.get(), inv.get(), normalized.get(), dx.get(), dg.get(), db.get(),
            stats.get(), stream.get(), ep);
        unchanged("full-backward partial statistics overlap");
        Require(status == mgt::Status::kInvalidConfig, "full backward accepted partial statistics overlap");
        ++rejected_cases;
    }
}

void QuickCases() {
    LiteralRounding();
    Harness odd(17, 5, 7);
    odd.Run(Fixture(17, 17, 5, 7), true, true, "odd-tail-inplace");
    odd.Run(Fixture(17, 1, 5, 7), false, false, "odd-tail-shrink-half-off");
    Harness narrow(17, 218, 224);
    narrow.Run(Fixture(17, 17, 218, 224), true, false, "narrow-production-stride");
    narrow.Run(Fixture(17, 17, 218, 224), false, true, "narrow-half-off-inplace");
}
void AllCases() {
    QuickCases();
    {
        Harness narrow(4097, 218, 224);
        const int rows[]{1, 17, 255, 256, 257, 4095, 4096, 4097};
        for (unsigned i = 0; i < sizeof(rows) / sizeof(rows[0]); ++i)
            narrow.Run(Fixture(4097, rows[i], 218, 224), i != 3, (i & 1) != 0,
                       "narrow-row-boundary-" + std::to_string(rows[i]));
    }
    {
        Harness wide(4096, 2556, 2560);
        wide.Run(Fixture(4096, 17, 2556, 2560), true, false, "wide-generic-mirror");
        wide.Run(Fixture(4096, 257, 2556, 2560), false, false, "wide-half-off");
        wide.Run(Fixture(4096, 4096, 2556, 2560), true, true, "wide-generic-inplace");
        wide.Run(Fixture(4096, 3, 2556, 2560), true, false, "wide-shrink-capacity");
    }
    const std::pair<int, int> shapes[] = {{1, 1}, {255, 255}, {256, 257}};
    for (const auto& shape : shapes) {
        Harness odd(1, shape.first, shape.second);
        odd.Run(Fixture(1, 1, shape.first, shape.second), true, false, "odd-or-unpadded-width");
    }
}
}  // namespace

int main(int argc, char** argv) {
    const bool quick = argc == 2 && std::strcmp(argv[1], "--quick") == 0;
    if (argc != 1 && !quick) {
        std::fprintf(stderr, "usage: %s [--quick]\n", argv[0]);
        return 1;
    }
    try {
        Require(std::fegetround() == FE_TONEAREST, "CPU half/reference requires round-to-nearest");
        if (quick) QuickCases(); else AllCases();
        FullBackward(false);
        FullBackward(true);
        InvalidCases();
        Require(!cleanup_failed, "CUDA cleanup failed");
        std::printf("PASS BN backward epilogue: apply=%u integration=%u invalid=%u "
                    "old-FP32-exact RN-half padding inplace immutability canaries\n",
                    apply_cases, integration_cases, rejected_cases);
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL BN backward epilogue: %s\n", error.what());
        return 1;
    }
}
