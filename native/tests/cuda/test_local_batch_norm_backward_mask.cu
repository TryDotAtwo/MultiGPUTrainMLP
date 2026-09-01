#include "mgt_cuda/local_batch_norm.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cfenv>
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
unsigned apply_cases = 0, full_cases = 0, rejected_cases = 0;
using Values = std::vector<float>;

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
    Stream() { Cuda(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking), "create stream"); }
    ~Stream() { Cleanup(cudaStreamDestroy(stream_), "destroy stream"); }
    Stream(const Stream&) = delete;
    Stream& operator=(const Stream&) = delete;
    cudaStream_t get() const { return stream_; }
    void Sync() const { Cuda(cudaStreamSynchronize(stream_), "synchronize test stream"); }
private:
    cudaStream_t stream_ = nullptr;
};
template <class T> class Device {
public:
    Device(std::size_t size, const char* name) : size_(size), name_(name) {
        Cuda(cudaMalloc(&allocation_, (size + 2 * kGuard) * sizeof(T)), name_ + " allocate");
        try {
            Cuda(cudaMemset(allocation_, kGuardByte, (size + 2 * kGuard) * sizeof(T)),
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
    std::size_t size() const { return size_; }
    void Put(const std::vector<T>& values) {
        Require(values.size() == size_, name_ + " upload size");
        Cuda(cudaMemcpy(get(), values.data(), size_ * sizeof(T), cudaMemcpyHostToDevice), name_ + " upload");
    }
    void Poison() { Cuda(cudaMemset(get(), 0xcd, size_ * sizeof(T)), name_ + " poison"); }
    void Guards() const {
        unsigned char bytes[2 * kGuard * sizeof(T)];
        Cuda(cudaMemcpy(bytes, allocation_, kGuard * sizeof(T), cudaMemcpyDeviceToHost), name_ + " leading guard");
        Cuda(cudaMemcpy(bytes + kGuard * sizeof(T), get() + size_, kGuard * sizeof(T),
                        cudaMemcpyDeviceToHost), name_ + " trailing guard");
        for (unsigned char byte : bytes)
            if (byte != kGuardByte) throw std::runtime_error(name_ + " allocation guard changed");
    }
    std::vector<T> Read() const {
        Guards();
        std::vector<T> values(size_);
        Cuda(cudaMemcpy(values.data(), get(), size_ * sizeof(T), cudaMemcpyDeviceToHost), name_ + " download");
        return values;
    }
private:
    std::size_t size_;
    std::string name_;
    T* allocation_ = nullptr;
};

std::uint32_t Bits(float value) {
    std::uint32_t bits;
    static_assert(sizeof(bits) == sizeof(value), "FP32 storage");
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}
std::uint16_t Bits(__half value) {
    std::uint16_t bits;
    static_assert(sizeof(bits) == sizeof(value), "FP16 storage");
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}
bool HalfNaN(__half value) { return (Bits(value) & 0x7fff) > 0x7c00; }
template <class T> void SameBytes(const std::vector<T>& a, const std::vector<T>& b,
                                 const std::string& name) {
    Require(a.size() == b.size(), name + " size");
    if (std::memcmp(a.data(), b.data(), a.size() * sizeof(T)) == 0) return;
    for (std::size_t i = 0; i < a.size(); ++i)
        if (std::memcmp(&a[i], &b[i], sizeof(T)) != 0)
            throw std::runtime_error(name + " changed bytes at " + std::to_string(i));
}
void SameFloat(float actual, float expected, bool exact, const std::string& name,
               std::size_t index) {
    const bool equal = std::isnan(expected) ? std::isnan(actual) :
        (exact || !std::isfinite(expected)) ? Bits(actual) == Bits(expected) :
        std::isfinite(actual) && std::fabs(static_cast<double>(actual) - expected) <=
            2e-5 + 2e-4 * std::fabs(static_cast<double>(expected));
    if (!equal) {
        std::fprintf(stderr, "%s[%zu] actual=%.9g/0x%08x expected=%.9g/0x%08x exact=%d\n",
            name.c_str(), index, actual, Bits(actual), expected, Bits(expected), exact);
        throw std::runtime_error(name + " value mismatch");
    }
}
void Compare(const Values& a, const Values& b, std::size_t count, bool exact,
             const std::string& name) {
    Require(a.size() >= count && b.size() >= count, name + " extent");
    for (std::size_t i = 0; i < count; ++i) SameFloat(a[i], b[i], exact, name, i);
}

// Independent pre-fusion composition. Masking is a select, not multiplication:
// NaN/negative/negative-zero activations choose literal positive zero. Padding
// is masked too, before BN drops its columns; residual keeps that masked value.
__global__ void OldMask(const float* activated, const float* dy, float* masked, int count) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) masked[i] = !activated || activated[i] > 0.0f ? dy[i] : 0.0f;
}
__global__ void OldApply(const float* masked, int rows, int cols, int stride,
    const float* gamma, const float* inv, const float* normalized,
    const float* dg, const float* db, float* dx) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * stride) return;
    const int c = i % stride;
    if (c < cols) {
        const float scale = gamma[c] * inv[c] / rows;
        dx[i] = scale * (rows * masked[i] - db[c] - normalized[i] * dg[c]);
    } else dx[i] = 0.0f;
}
// The original 32-column x 8-row ownership, 256-row partial, row-lane order,
// and final lane-fold order are explicit here, independent of production APIs.
__global__ void OldPartial(const float* masked, int rows, int cols, int stride,
                           const float* normalized, float* dg, float* db) {
    const int c = blockIdx.x * 32 + threadIdx.x;
    const int lane = threadIdx.y, begin = blockIdx.y * 256;
    const int end = min(begin + 256, rows);
    float g = 0.0f, b = 0.0f;
    if (c < cols) for (int r = begin + lane; r < end; r += 8) {
        const int i = r * stride + c;
        b += masked[i];
        g += masked[i] * normalized[i];
    }
    __shared__ float gs[8][32], bs[8][32];
    gs[lane][threadIdx.x] = g; bs[lane][threadIdx.x] = b;
    __syncthreads();
    if (lane == 0 && c < cols) {
        for (int other = 1; other < 8; ++other) {
            g += gs[other][threadIdx.x]; b += bs[other][threadIdx.x];
        }
        atomicAdd(dg + c, g); atomicAdd(db + c, b);
    }
}

struct Fixture {
    int rows, cols, stride;
    bool dyadic;
    Values dy, activated, normalized, gamma, beta, inv, dg, db;
    Fixture(int capacity, int live_rows, int logical, int physical, bool exact_sums = true)
        : rows(live_rows), cols(logical), stride(physical), dyadic(exact_sums),
          dy(static_cast<std::size_t>(capacity) * physical, std::numeric_limits<float>::quiet_NaN()),
          activated(dy), normalized(dy), gamma(logical), beta(logical), inv(logical),
          dg(logical), db(logical) {
        for (int c = 0; c < cols; ++c) {
            gamma[c] = static_cast<float>(c % 7 - 3) / 4.0f;
            beta[c] = static_cast<float>(c % 5 - 2) / 8.0f;
            inv[c] = static_cast<float>(c % 5 + 1) / 8.0f;
            dg[c] = static_cast<float>(c % 9 - 4) / 16.0f;
            db[c] = static_cast<float>(c % 11 - 5) / 8.0f;
        }
        for (int r = 0; r < rows; ++r) for (int c = 0; c < stride; ++c) {
            const std::size_t i = static_cast<std::size_t>(r) * stride + c;
            dy[i] = static_cast<float>((r * 13 + c * 7) % 33 - 16) / (dyadic ? 8.0f : 7.0f);
            if (c < cols) normalized[i] = static_cast<float>((r * 5 + c * 11) % 17 - 8) /
                (dyadic ? 16.0f : 13.0f);
            switch ((r + c) % 6) {
                case 0: activated[i] = 1.0f; break;
                case 1: activated[i] = 0.0f; break;
                case 2: activated[i] = -0.0f; break;
                case 3: activated[i] = -2.0f; break;
                case 4: activated[i] = std::numeric_limits<float>::quiet_NaN(); break;
                case 5: activated[i] = 2.0f; break;
            }
        }
    }
};
enum class Mask { Separate, DyAlias, NormalizedAlias, None };

class Harness {
public:
    Harness(int capacity, int cols, int stride)
        : capacity_(capacity), cols_(cols), stride_(stride),
          dy_(static_cast<std::size_t>(capacity) * stride, "dy"),
          activated_(dy_.size(), "activated"), normalized_(dy_.size(), "normalized"),
          dx_(dy_.size(), "dx"), masked_(dy_.size(), "old masked"),
          oracle_(dy_.size(), "old dx"), residual_(dy_.size(), "residual"), half_(dy_.size(), "half"),
          gamma_(cols, "gamma"), beta_(cols, "beta"), inv_(cols, "inv"),
          dg_(cols, "dgamma"), db_(cols, "dbeta"),
          stats_(2 * cols, "stats"), old_stats_(2 * cols, "old stats") {}

    // Missing mask in either partial/apply, masking dx after BN, zeroing the
    // residual padding, reading updated inplace dy, or mirroring residual
    // instead of dx each breaks an independent observable result below.
    void Run(Fixture f, bool full, bool mirror, bool inplace, bool residual,
             const std::string& name, Mask mask = Mask::Separate, bool literal = false,
             bool recompute_mask = false) {
        Require(f.rows > 0 && f.rows <= capacity_ && f.cols == cols_ && f.stride == stride_ &&
                f.dy.size() == dy_.size(), name + " fixture extent");
        Require(!residual || mask != Mask::None, name + " residual requires mask");
        Require(!inplace || mask != Mask::DyAlias, name + " writable activation alias");
        if (mask == Mask::DyAlias) f.activated = f.dy;
        if (mask == Mask::NormalizedAlias) f.activated = f.normalized;
        dy_.Put(f.dy); activated_.Put(f.activated); normalized_.Put(f.normalized);
        gamma_.Put(f.gamma); beta_.Put(f.beta); inv_.Put(f.inv); dg_.Put(f.dg); db_.Put(f.db);
        dx_.Poison(); masked_.Poison(); oracle_.Poison(); residual_.Poison(); half_.Poison();
        stats_.Poison(); old_stats_.Poison();
        Cuda(cudaStreamSynchronize(nullptr), "finish fixture uploads");
        const auto before_dx = dx_.Read(), before_residual = residual_.Read();
        const auto before_half = half_.Read();
        const auto before_stats = stats_.Read();
        const std::size_t live = static_cast<std::size_t>(f.rows) * f.stride;
        const unsigned blocks = static_cast<unsigned>((live + 255) / 256);
        const float* a = mask == Mask::None ? nullptr : mask == Mask::DyAlias ? dy_.get() :
            mask == Mask::NormalizedAlias ? normalized_.get() : activated_.get();
        OldMask<<<blocks, 256, 0, stream_.get()>>>(a, dy_.get(), masked_.get(), static_cast<int>(live));
        Cuda(cudaGetLastError(), "old mask launch");
        if (full) {
            Cuda(cudaMemsetAsync(old_stats_.get(), 0, 2 * f.cols * sizeof(float), stream_.get()), "zero old stats");
            OldPartial<<<dim3((f.cols + 31) / 32, (f.rows + 255) / 256), dim3(32, 8), 0, stream_.get()>>>(
                masked_.get(), f.rows, f.cols, f.stride, normalized_.get(), old_stats_.get(), old_stats_.get() + f.cols);
            Cuda(cudaGetLastError(), "old partial launch");
        }
        OldApply<<<blocks, 256, 0, stream_.get()>>>(masked_.get(), f.rows, f.cols, f.stride,
            gamma_.get(), inv_.get(), normalized_.get(), full ? old_stats_.get() : dg_.get(),
            full ? old_stats_.get() + f.cols : db_.get(), oracle_.get());
        Cuda(cudaGetLastError(), "old apply launch");
        mgt_cuda::LocalBatchNormBackwardEpilogue ep;
        ep.half_output = mirror ? half_.get() : nullptr;
        ep.activated = recompute_mask ? nullptr : a;
        ep.relu_beta = recompute_mask ? beta_.get() : nullptr;
        ep.residual_grad = residual ? residual_.get() : nullptr;
        float* destination = inplace ? dy_.get() : dx_.get();
        const auto status = full ? mgt_cuda::LaunchLocalStridedBatchNormBackward(dy_.get(),
            f.rows, f.cols, f.stride, gamma_.get(), inv_.get(), normalized_.get(), destination,
            dg_.get(), db_.get(), stats_.get(), stream_.get(), ep) :
            mgt_cuda::LaunchLocalStridedBatchNormBackwardApply(dy_.get(), f.rows, f.cols, f.stride,
                gamma_.get(), inv_.get(), normalized_.get(), dg_.get(), db_.get(), destination, ep, stream_.get());
        stream_.Sync();
        Require(status == mgt::Status::kOk, name + " valid call rejected");
        const auto actual = inplace ? dy_.Read() : dx_.Read();
        const auto expected = oracle_.Read(), masked = masked_.Read();
        // Products are integer multiples of 1/128, magnitude <=1, and rows<=4096:
        // every dyadic partial/global sum is exactly representable, in any CTA
        // arrival order. General multi-CTA values use tolerances, not bit claims.
        const bool exact = !full || f.rows <= 256 || f.dyadic;
        Compare(actual, expected, live, exact, name + " old GPU dx");
        for (std::size_t i = live; i < actual.size(); ++i)
            if (Bits(actual[i]) != Bits(inplace ? f.dy[i] : before_dx[i]))
                throw std::runtime_error(name + " dx inactive capacity changed at " + std::to_string(i));
        const auto got_residual = residual_.Read();
        if (residual) {
            Compare(got_residual, masked, live, true, name + " old masked residual");
            for (std::size_t i = live; i < got_residual.size(); ++i)
                if (Bits(got_residual[i]) != Bits(before_residual[i]))
                    throw std::runtime_error(name + " residual inactive capacity changed at " + std::to_string(i));
        } else SameBytes(got_residual, before_residual, name + " disabled residual");
        const auto got_half = half_.Read();
        if (mirror) for (std::size_t i = 0; i < live; ++i) {
            const bool good = std::isnan(actual[i]) ? HalfNaN(got_half[i]) :
                Bits(got_half[i]) == Bits(__float2half_rn(actual[i]));
            if (!good) throw std::runtime_error(name + " RN dx mirror at " + std::to_string(i));
        }
        for (std::size_t i = mirror ? live : 0; i < got_half.size(); ++i)
            if (Bits(got_half[i]) != Bits(before_half[i]))
                throw std::runtime_error(name + " untouched half capacity at " + std::to_string(i));
        std::vector<double> cpu_g(f.cols), cpu_b(f.cols);
        for (int c = 0; c < f.cols; ++c) { cpu_g[c] = full ? 0.0 : f.dg[c]; cpu_b[c] = full ? 0.0 : f.db[c]; }
        if (full) for (int r = 0; r < f.rows; ++r) for (int c = 0; c < f.cols; ++c) {
            const std::size_t i = static_cast<std::size_t>(r) * f.stride + c;
            const float d = mask == Mask::None || f.activated[i] > 0.0f ? f.dy[i] : 0.0f;
            cpu_g[c] += static_cast<double>(d) * f.normalized[i]; cpu_b[c] += d;
        }
        auto cpu_close = [&](float got, double want, const std::string& field, std::size_t i) {
            const bool good = std::isnan(want) ? std::isnan(got) : std::isinf(want) ?
                std::isinf(got) && std::signbit(got) == std::signbit(want) :
                std::isfinite(got) && std::fabs(static_cast<double>(got) - want) <= 2e-5 + 2e-4 * std::fabs(want);
            if (!good) throw std::runtime_error(name + " CPU " + field + " at " + std::to_string(i));
        };
        for (int r = 0; r < f.rows; ++r) for (int c = 0; c < f.stride; ++c) {
            const std::size_t i = static_cast<std::size_t>(r) * f.stride + c;
            if (c >= f.cols) {
                Require(Bits(actual[i]) == 0, name + " dx padding is not positive zero");
                if (mirror) Require(Bits(got_half[i]) == 0, name + " half padding is not positive zero");
            } else {
                const float d = mask == Mask::None || f.activated[i] > 0.0f ? f.dy[i] : 0.0f;
                const double want = static_cast<double>(f.gamma[c]) * f.inv[c] / f.rows *
                    (static_cast<double>(f.rows) * d - cpu_b[c] - static_cast<double>(f.normalized[i]) * cpu_g[c]);
                cpu_close(actual[i], want, "dx", i);
            }
        }
        if (full) {
            const auto actual_g = dg_.Read(), actual_b = db_.Read(), stats = stats_.Read(), old = old_stats_.Read();
            Compare(stats, old, stats.size(), exact, name + " old GPU stats");
            for (int c = 0; c < f.cols; ++c) {
                SameFloat(actual_g[c], stats[c], true, name + " dgamma copy", c);
                SameFloat(actual_b[c], stats[f.cols + c], true, name + " dbeta copy", c);
                cpu_close(actual_g[c], cpu_g[c], "dgamma", c); cpu_close(actual_b[c], cpu_b[c], "dbeta", c);
            }
        } else {
            SameBytes(dg_.Read(), f.dg, name + " supplied dgamma immutable");
            SameBytes(db_.Read(), f.db, name + " supplied dbeta immutable");
            SameBytes(stats_.Read(), before_stats, name + " unused stats immutable");
        }
        if (!inplace) SameBytes(dy_.Read(), f.dy, name + " incoming dy immutable");
        else SameBytes(dx_.Read(), before_dx, name + " unused dx immutable");
        SameBytes(activated_.Read(), f.activated, name + " activation immutable");
        SameBytes(normalized_.Read(), f.normalized, name + " normalized immutable");
        SameBytes(gamma_.Read(), f.gamma, name + " gamma immutable");
        SameBytes(beta_.Read(), f.beta, name + " beta immutable");
        SameBytes(inv_.Read(), f.inv, name + " inverse immutable");
        if (literal) {
            const std::uint16_t half_bits[]{0x0000, 0x8000, 0x3c00, 0x3c02, 0x0000, 0x8000,
                0x7bff, 0x7c00, 0xfc00, 0x7800, 0xf800, 0xc000, 0x0000};
            for (std::size_t i = 0; i < sizeof(half_bits) / sizeof(half_bits[0]); ++i)
                Require(Bits(got_half[i]) == half_bits[i], name + " literal half at " + std::to_string(i));
            Require(Bits(actual[11]) == Bits(-2.0f), name + " masked-off BN dx must be nonzero");
            Require(std::isnan(actual[13]) && HalfNaN(got_half[13]), name + " masked 0*NaN must propagate");
            Require(Bits(got_residual[14]) == Bits(-0.0f) && got_residual[15] == -7.0f &&
                    Bits(got_residual[16]) == 0, name + " padding residual mask literals");
        }
        if (full) ++full_cases; else ++apply_cases;
        std::printf("PASS %-5s %-31s rows=%d cols=%d stride=%d mirror=%d inplace=%d residual=%d exact=%d\n",
            full ? "full" : "apply", name.c_str(), f.rows, f.cols, f.stride, mirror, inplace, residual, exact);
    }
private:
    int capacity_, cols_, stride_;
    Stream stream_;
    Device<float> dy_, activated_, normalized_, dx_, masked_, oracle_, residual_;
    Device<__half> half_;
    Device<float> gamma_, beta_, inv_, dg_, db_, stats_, old_stats_;
};

void RecomputedMaskCase() {
    Fixture f(257, 257, 5, 7);
    for (int row = 0; row < f.rows; ++row) for (int col = 0; col < f.stride; ++col) {
        const std::size_t index = static_cast<std::size_t>(row) * f.stride + col;
        f.activated[index] = col < f.cols
            ? std::fma(f.normalized[index], f.gamma[col], f.beta[col]) : 0.0f;
    }
    Harness h(257, 5, 7);
    h.Run(f, true, true, false, false, "recomputed-affine-mask",
          Mask::Separate, false, true);
}

void LiteralCases() {
    Fixture f(1, 1, 14, 17);
    std::fill(f.gamma.begin(), f.gamma.end(), 1.0f); std::fill(f.inv.begin(), f.inv.end(), 1.0f);
    std::fill(f.dg.begin(), f.dg.end(), 0.0f); std::fill(f.db.begin(), f.db.end(), 0.0f);
    std::fill(f.normalized.begin(), f.normalized.end(), 0.0f);
    std::fill(f.activated.begin(), f.activated.end(), 1.0f);
    f.dy = {0.0f, -0.0f, 1.0f + 0x1p-11f, 1.0f + 3 * 0x1p-11f,
        0x1p-25f, -0x1p-25f, 65504.0f, 65520.0f, -65520.0f, 32768.0f, -32768.0f,
        7.0f, std::numeric_limits<float>::quiet_NaN(), 9.0f, -0.0f, -7.0f, 8.0f};
    f.activated[11] = -0.0f; f.db[11] = 2.0f;
    f.activated[12] = -1.0f;
    f.activated[13] = f.normalized[13] = std::numeric_limits<float>::quiet_NaN();
    f.activated[16] = std::numeric_limits<float>::quiet_NaN();
    Harness h(1, 14, 17);
    h.Run(f, false, true, false, true, "literal-RN-mask-padding", Mask::Separate, true);
    h.Run(f, false, true, true, true, "literal-RN-inplace", Mask::Separate, true);
}

void QuickCases() {
    LiteralCases();
    RecomputedMaskCase();
    {
        Harness h(257, 5, 7);
        const int rows[]{17, 1, 255, 256, 257, 3};
        for (unsigned i = 0; i < sizeof(rows) / sizeof(rows[0]); ++i)
            h.Run(Fixture(257, rows[i], 5, 7), false, i != 0 && i != 3, (i & 1) != 0,
                  i != 0 && i != 2, "odd-reuse-" + std::to_string(rows[i]));
        h.Run(Fixture(257, 17, 5, 7), false, true, false, true, "activation-alias-dy", Mask::DyAlias);
        h.Run(Fixture(257, 17, 5, 7), false, true, false, true, "activation-alias-normalized", Mask::NormalizedAlias);
        h.Run(Fixture(257, 17, 5, 7), true, true, false, false, "one-partial-row17");
        h.Run(Fixture(257, 255, 5, 7, false), true, false, true, true, "one-partial-decimal-row255");
        h.Run(Fixture(257, 257, 5, 7), true, true, true, true, "two-partial-dyadic-row257");
        Fixture nan(257, 17, 5, 7);
        nan.activated[1] = 0.0f; nan.normalized[1] = std::numeric_limits<float>::quiet_NaN();
        h.Run(nan, true, true, false, true, "masked-normalized-NaN-partial");
    }
    {
        Harness h(17, 3, 5);
        h.Run(Fixture(17, 17, 3, 5), false, true, false, false, "unmasked-half-parity", Mask::None);
    }
    {
        Harness h(3, 1, 65);
        h.Run(Fixture(3, 3, 1, 65), true, true, false, true,
              "residual-padding-multiple-tiles");
    }
    {
        Harness h(17, 218, 224);
        h.Run(Fixture(17, 17, 218, 224), false, true, false, true, "narrow-small");
    }
    {
        Harness h(17, 2556, 2560);
        h.Run(Fixture(17, 17, 2556, 2560), false, false, true, false, "wide-small-half-off");
    }
}

void OutputAliasFallback() {
    constexpr int rows = 17, cols = 3, stride = 5;
    Fixture f(rows, rows, cols, stride);
    Device<float> dy(f.dy.size(), "alias dy"), normalized(f.normalized.size(), "alias normalized"),
        gamma(cols, "alias gamma/dgamma"), inv(cols, "alias inverse/dbeta"),
        dx_reference(f.dy.size(), "alias reference dx"), dx_alias(f.dy.size(), "alias output dx"),
        dgamma(cols, "alias reference dgamma"), dbeta(cols, "alias reference dbeta"),
        stats_reference(2 * cols, "alias reference stats"), stats_alias(2 * cols, "alias output stats");
    Stream stream;
    dy.Put(f.dy); normalized.Put(f.normalized); gamma.Put(f.gamma); inv.Put(f.inv);
    dx_reference.Poison(); dgamma.Poison(); dbeta.Poison(); stats_reference.Poison();
    auto status = mgt_cuda::LaunchLocalStridedBatchNormBackward(
        dy.get(), rows, cols, stride, gamma.get(), inv.get(), normalized.get(),
        dx_reference.get(), dgamma.get(), dbeta.get(), stats_reference.get(), stream.get());
    stream.Sync();
    Require(status == mgt::Status::kOk, "output-alias reference rejected");
    const auto expected_dx = dx_reference.Read();
    const auto expected_dgamma = dgamma.Read(), expected_dbeta = dbeta.Read();
    const auto expected_stats = stats_reference.Read();

    gamma.Put(f.gamma); inv.Put(f.inv); dx_alias.Poison(); stats_alias.Poison();
    status = mgt_cuda::LaunchLocalStridedBatchNormBackward(
        dy.get(), rows, cols, stride, gamma.get(), inv.get(), normalized.get(),
        dx_alias.get(), gamma.get(), inv.get(), stats_alias.get(), stream.get());
    stream.Sync();
    Require(status == mgt::Status::kOk, "output aliases of gamma/inverse rejected");
    Compare(dx_alias.Read(), expected_dx, expected_dx.size(), true, "output-alias fallback dx");
    Compare(gamma.Read(), expected_dgamma, cols, true, "output-alias fallback dgamma");
    Compare(inv.Read(), expected_dbeta, cols, true, "output-alias fallback dbeta");
    Compare(stats_alias.Read(), expected_stats, expected_stats.size(), true,
            "output-alias fallback stats");
    SameBytes(dy.Read(), f.dy, "output-alias fallback dy immutable");
    SameBytes(normalized.Read(), f.normalized, "output-alias fallback normalized immutable");
    ++full_cases;
    std::puts("PASS full  output aliases preserve post-apply copy semantics");
}

void AllCases() {
    QuickCases();
    {
        Harness h(4096, 218, 224);
        h.Run(Fixture(4096, 4096, 218, 224), false, true, true, true, "narrow-production-apply");
        h.Run(Fixture(4096, 4096, 218, 224), true, true, true, true, "narrow-production-full");
        h.Run(Fixture(4096, 17, 218, 224), false, true, false, false, "narrow-shrink-capacity");
    }
    {
        Harness h(4096, 2556, 2560);
        h.Run(Fixture(4096, 4096, 2556, 2560), false, false, true, false, "wide-large-half-off");
        h.Run(Fixture(4096, 4096, 2556, 2560), true, true, false, true, "wide-large-full");
        h.Run(Fixture(4096, 257, 2556, 2560), false, true, false, true, "wide-shrink-capacity");
    }
    {
        Harness h(513, 33, 35);
        h.Run(Fixture(513, 513, 33, 35, false), true, true, true, true, "multi-partial-decimal-CPU-tol");
    }
    const std::pair<int, int> shapes[]{{1, 1}, {255, 255}, {256, 257}};
    for (const auto& shape : shapes) {
        Harness h(1, shape.first, shape.second);
        h.Run(Fixture(1, 1, shape.first, shape.second), false, true, false, true, "width-boundary");
    }
}

// No invalid address is needed. Oversized, guarded fixture allocations keep
// even deliberately overlapping views valid while asserting no writes at all.
void InvalidCases() {
    Stream stream;
    Device<float> dy(64, "invalid dy"), a(64, "invalid activated"), n(64, "invalid normalized"),
        gamma(64, "invalid gamma"), inv(64, "invalid inverse"), dg(64, "invalid dg"),
        db(64, "invalid db"), dx(64, "invalid dx"), residual(64, "invalid residual"), stats(64, "invalid stats");
    Device<__half> half(64, "invalid half");
    Device<float>* arrays[]{&dy, &a, &n, &gamma, &inv, &dg, &db, &dx, &residual, &stats};
    std::vector<Values> before;
    for (auto* array : arrays) before.push_back(array->Read());
    const auto before_half = half.Read();
    struct Args {
        float *dy, *a, *beta, *n, *gamma, *inv, *dg, *db, *dx, *residual, *stats;
        __half* half;
    };
    const Args good{dy.get(), a.get(), nullptr, n.get(), gamma.get(), inv.get(), dg.get(), db.get(),
                    dx.get(), residual.get(), stats.get(), half.get()};
    auto reject = [&](const Args& args, const std::string& name, bool full) {
        mgt_cuda::LocalBatchNormBackwardEpilogue ep;
        ep.half_output = args.half; ep.activated = args.a; ep.residual_grad = args.residual;
        ep.relu_beta = args.beta;
        const auto status = full ? mgt_cuda::LaunchLocalStridedBatchNormBackward(args.dy,
            2, 3, 5, args.gamma, args.inv, args.n, args.dx, args.dg, args.db, args.stats, stream.get(), ep) :
            mgt_cuda::LaunchLocalStridedBatchNormBackwardApply(args.dy, 2, 3, 5, args.gamma,
                args.inv, args.n, args.dg, args.db, args.dx, ep, stream.get());
        stream.Sync();
        for (std::size_t i = 0; i < before.size(); ++i)
            SameBytes(arrays[i]->Read(), before[i], name + " float allocation " + std::to_string(i));
        SameBytes(half.Read(), before_half, name + " half allocation");
        Require(status == mgt::Status::kInvalidConfig, name + " invalid alias accepted");
        ++rejected_cases;
    };
    auto both = [&](const Args& args, const std::string& name) {
        reject(args, name + " apply", false); reject(args, name + " full", true);
    };
    { auto x = good; x.a = nullptr; both(x, "residual without activated"); }
    { auto x = good; x.residual = nullptr; x.beta = gamma.get();
      both(x, "activated and recomputed masks both selected"); }
    { auto x = good; x.a = nullptr; x.residual = nullptr; x.beta = dx.get();
      both(x, "recomputed beta overlaps writable dx"); }
    for (auto* target : {&dg, &db}) {
        auto x = good; x.a = nullptr; x.residual = nullptr; x.beta = target->get();
        reject(x, "full recomputed beta overlaps writable feature output", true);
    }
    { auto x = good; x.a = nullptr; x.residual = nullptr; x.beta = stats.get();
      reject(x, "full recomputed beta overlaps statistics", true); }
    for (int offset : {0, 9}) {
        auto x = good; x.a = dx.get() + offset; both(x, "activated overlaps dx");
    }
    { auto x = good; x.a = dy.get(); x.dx = dy.get(); both(x, "inplace activation is writable"); }
    for (auto* target : {&dy, &a, &n, &dx, &gamma, &inv, &dg, &db}) {
        const bool matrix = target == &dy || target == &a || target == &n || target == &dx;
        for (int offset : {0, matrix ? 9 : 2}) {
            auto x = good; x.residual = target->get() + offset;
            both(x, "residual overlaps live float view");
        }
    }
    for (auto* target : {&a, &residual}) for (int offset : {0, 19}) {
        auto x = good; x.half = reinterpret_cast<__half*>(target->get()) + offset;
        both(x, "half overlaps activated/residual");
    }
    { auto x = good; x.residual = reinterpret_cast<float*>(half.get()) + 4;
      both(x, "reverse residual overlaps half suffix"); }
    { auto x = good; x.residual = residual.get(); x.a = residual.get() + 9;
      both(x, "reverse activation overlaps residual suffix"); }
    for (auto* target : {&dg, &db, &stats}) for (int offset : {0, target == &stats ? 5 : 2}) {
        auto x = good; x.a = target->get() + offset;
        reject(x, "full activated overlaps writable feature/statistics", true);
    }
    for (int offset : {0, 5}) {
        auto x = good; x.residual = stats.get() + offset;
        reject(x, "full residual overlaps statistics", true);
    }
    { auto x = good; x.stats = a.get() + 9;
      reject(x, "reverse full statistics overlaps activated suffix", true); }
    { auto x = good; x.stats = residual.get() + 9;
      reject(x, "reverse full statistics overlaps residual suffix", true); }
    { auto x = good; x.half = reinterpret_cast<__half*>(stats.get()) + 11;
      reject(x, "full half overlaps statistics with masking", true); }
}
}  // namespace

int main(int argc, char** argv) {
    const bool quick = argc == 2 && std::strcmp(argv[1], "--quick") == 0;
    if (argc != 1 && !quick) { std::fprintf(stderr, "usage: %s [--quick]\n", argv[0]); return 1; }
    try {
        Require(std::fegetround() == FE_TONEAREST, "CPU RN oracle requires round-to-nearest");
        if (quick) QuickCases(); else AllCases();
        OutputAliasFallback();
        InvalidCases();
        Require(!cleanup_failed, "CUDA cleanup failed");
        std::printf("PASS BN backward mask: apply=%u full=%u invalid=%u old-mask/FP32/RN-half "
                    "partial-statistics padding residual NaN inplace aliases capacity canaries\n",
                    apply_cases, full_cases, rejected_cases);
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL BN backward mask: %s\n", error.what());
        return 1;
    }
}
