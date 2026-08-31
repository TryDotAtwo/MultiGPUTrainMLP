#include "mgt_cuda/local_batch_norm.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
void Require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}
void Cuda(cudaError_t status) {
    Require(status == cudaSuccess, cudaGetErrorString(status));
}

// Test-owned storage, including aligned leading/trailing canaries for every array.
template <class T> class Device {
public:
    explicit Device(std::size_t count) : count_(count) {
        Cuda(cudaMalloc(&allocation_, (count_ + 2 * kGuard) * sizeof(T)));
        const auto status = cudaMemset(allocation_, 0xa5, (count_ + 2 * kGuard) * sizeof(T));
        if (status != cudaSuccess) { cudaFree(allocation_); Cuda(status); }
    }
    ~Device() { cudaFree(allocation_); }
    Device(const Device&) = delete;
    Device& operator=(const Device&) = delete;
    T* get() const { return allocation_ + kGuard; }
    void Put(const std::vector<T>& values) {
        Require(values.size() == count_, "upload size");
        Cuda(cudaMemcpy(get(), values.data(), count_ * sizeof(T), cudaMemcpyHostToDevice));
    }
    std::vector<T> Read() const {
        CheckGuards();
        std::vector<T> result(count_);
        Cuda(cudaMemcpy(result.data(), get(), count_ * sizeof(T), cudaMemcpyDeviceToHost));
        return result;
    }
    void CheckGuards() const {
        unsigned char guards[2 * kGuard * sizeof(T)];
        Cuda(cudaMemcpy(guards, allocation_, kGuard * sizeof(T), cudaMemcpyDeviceToHost));
        Cuda(cudaMemcpy(guards + kGuard * sizeof(T), get() + count_,
                        kGuard * sizeof(T), cudaMemcpyDeviceToHost));
        for (auto byte : guards) Require(byte == 0xa5, "device allocation canary changed");
    }
private:
    static constexpr std::size_t kGuard = 8;
    std::size_t count_;
    T* allocation_ = nullptr;
};

template <class T> void Equal(const std::vector<T>& actual,
                              const std::vector<T>& expected, const char* field) {
    Require(actual.size() == expected.size(), std::string(field) + " size");
    for (std::size_t i = 0; i < actual.size(); ++i) {
        if (std::memcmp(&actual[i], &expected[i], sizeof(T)) != 0) {
            throw std::runtime_error(std::string(field) + " bit mismatch at " + std::to_string(i));
        }
    }
}

// Independent old composition: affine is rounded into a global FP32 array before
// another kernel reads it. Ordinary expressions retain the baseline NVCC policy.
__global__ void OldAffine(const float* x, int count, int cols, int stride,
                          const float* gamma, const float* beta, const float* mean,
                          const float* inv, float* affine, float* normalized) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const int c = i % stride;
    float value = 0.0f;
    float output = 0.0f;
    if (c < cols) {
        value = (x[i] - mean[c]) * inv[c];
        output = value * gamma[c] + beta[c];
    }
    normalized[i] = value;
    affine[i] = output;
}

__global__ void OldActivation(float* values, const float* residual,
                              __half* half_output, int count) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    float value = values[i];
    if (residual) value = residual[i] + value;
    value = value > 0.0f ? value : 0.0f;
    values[i] = value;
    if (half_output) half_output[i] = __float2half_rn(value);
}

struct Fixture {
    int rows, cols, stride;
    std::vector<float> x, residual, gamma, beta, mean, inv;
    Fixture(int r, int c, int s) : rows(r), cols(c), stride(s), x(r * s),
        residual(r * s), gamma(c), beta(c), mean(c), inv(c) {
        for (int i = 0; i < r * s; ++i) {
            x[i] = i % s < c ? static_cast<float>((i * 17) % 33 - 16) / 16.0f : NAN;
            // Positive padding proves BN-pad-zero then residual/ReLU is preserved.
            residual[i] = static_cast<float>((i * 7) % 19 - 7) / 8.0f;
        }
        for (int i = 0; i < c; ++i) {
            gamma[i] = static_cast<float>(i % 7 - 3) / 4.0f;
            beta[i] = static_cast<float>(i % 5 - 2) / 8.0f;
            mean[i] = static_cast<float>(i % 3 - 1) / 4.0f;
            inv[i] = static_cast<float>(i % 4 + 1) / 2.0f;
        }
    }
};

// Catches missing epilogues, FP16 replacing FP32 tape, reassociation across the
// old store, dropped/incorrect padding, tail OOB, and destructive inplace reads.
void ApplyCase(const Fixture& f, bool relu, bool residual, bool half, bool inplace,
               const std::vector<float>* literal_norm = nullptr,
               const std::vector<float>* literal_output = nullptr) {
    const int count = f.rows * f.stride;
    Device<float> x(count), res(count), gamma(f.cols), beta(f.cols), mean(f.cols), inv(f.cols);
    Device<float> y(count), norm(count), old_y(count), old_norm(count);
    Device<__half> h(count), old_h(count);
    x.Put(f.x); res.Put(f.residual); gamma.Put(f.gamma); beta.Put(f.beta);
    mean.Put(f.mean); inv.Put(f.inv);
    OldAffine<<<(count + 255) / 256, 256>>>(x.get(), count, f.cols, f.stride,
        gamma.get(), beta.get(), mean.get(), inv.get(), old_y.get(), old_norm.get());
    if (relu) OldActivation<<<(count + 255) / 256, 256>>>(old_y.get(),
        residual ? res.get() : nullptr, half ? old_h.get() : nullptr, count);
    Cuda(cudaGetLastError());
    mgt_cuda::LocalBatchNormForwardEpilogue epilogue;
    epilogue.relu = relu;
    epilogue.residual = residual ? res.get() : nullptr;
    epilogue.half_output = half ? h.get() : nullptr;
    Require(mgt_cuda::LaunchLocalStridedBatchNormApply(x.get(), f.rows, f.cols,
        f.stride, gamma.get(), beta.get(), mean.get(), inv.get(),
        inplace ? x.get() : y.get(), norm.get(), epilogue, nullptr) == mgt::Status::kOk,
        "LaunchLocalStridedBatchNormApply rejected valid case");
    Cuda(cudaDeviceSynchronize());
    const auto actual = inplace ? x.Read() : y.Read();
    const auto normalized = norm.Read();
    Equal(actual, old_y.Read(), "FP32 output");
    Equal(normalized, old_norm.Read(), "FP32 normalized");
    if (half) Equal(h.Read(), old_h.Read(), "FP16 output");
    if (literal_norm) Equal(normalized, *literal_norm, "hand-derived normalized");
    if (literal_output) {
        Equal(actual, *literal_output, "hand-derived output");
        if (half) {
            const auto actual_half = h.Read();
            // Hand-derived +2^-26 and +2^-46 must stay positive in FP32, half +0.
            for (int i : {1, 5}) {
                const std::uint16_t zero = 0;
                Require(actual[i] > 0.0f &&
                    std::memcmp(&actual_half[i], &zero, sizeof(zero)) == 0,
                    "FP32 positive activation lost before FP16 conversion");
            }
        }
    }
    if (!inplace) Equal(x.Read(), f.x, "input modified");
    Equal(res.Read(), f.residual, "residual modified");
    Equal(gamma.Read(), f.gamma, "gamma modified"); Equal(beta.Read(), f.beta, "beta modified");
    Equal(mean.Read(), f.mean, "mean modified"); Equal(inv.Read(), f.inv, "inv_std modified");
    x.CheckGuards(); y.CheckGuards(); h.CheckGuards(); old_h.CheckGuards();
}

void HandDerivedCases() {
    Fixture f(1, 8, 10);
    const float above_one = 1.0f + std::ldexp(1.0f, -23);
    const float below_one = 1.0f - std::ldexp(1.0f, -23);
    f.x = {1, 2, 1, NAN, -0.0f, above_one, 1, 1, NAN, NAN};
    f.mean = {0.5f, 0, 0, 0, 0, 0, 0, 0};
    f.inv = {2, 1, 1, 1, 1, 1, 1, 1};
    f.gamma = {2, 0, -2, 1, 1, below_one, 1, 1};
    f.beta = {-0.5f, std::ldexp(1.0f, -26), 0.5f, 0, -0.0f,
              -1, std::ldexp(1.0f, -24), std::ldexp(1.0f, -24)};
    f.residual = {-0.25f, 0, 2, 0, -0.0f, std::ldexp(1.0f, -45),
                  -1, std::ldexp(1.0f, -24), 0.75f, -0.75f};
    // (1+2^-23)*(1-2^-23)-1 = -2^-46 with baseline affine FFMA;
    // adding 2^-45 then gives +2^-46. Col6/7 protect the affine-store boundary.
    const std::vector<float> expected = {1.25f, std::ldexp(1.0f, -26), 0.5f,
        0, 0, std::ldexp(1.0f, -46), 0, 1, 0.75f, 0};
    for (bool inplace : {false, true}) ApplyCase(f, true, true, true, inplace, nullptr, &expected);
    // Finite dyadic fixture independently checks normalized, zero gamma, negative gamma.
    Fixture dyadic(2, 3, 4);
    dyadic.x = {1, 2, -1, NAN, -1, 0, 3, NAN};
    dyadic.mean = {0, 1, 1}; dyadic.inv = {0.5f, 2, 0.25f};
    dyadic.gamma = {2, 0, -2}; dyadic.beta = {0.25f, -1, 0.5f};
    const std::vector<float> normalized = {0.5f, 2, -0.5f, 0, -0.5f, -2, 0.5f, 0};
    const std::vector<float> output = {1.25f, -1, 1.5f, 0, -0.75f, -1, -0.5f, 0};
    ApplyCase(dyadic, false, false, false, true, &normalized, &output);
    // Without residual, both NaN and -0 must map to the old comparison-policy +0.
    ApplyCase(f, true, false, false, false);
}

// Catches bypassing the fused consumer or changing statistics/running-state
// updates. Dyadic inputs make both partial-sum orders exact across two row tiles.
void FullForwardCase() {
    Fixture f(257, 218, 224);
    const int count = f.rows * f.stride;
    Device<float> x(count), res(count), gamma(f.cols), beta(f.cols);
    Device<float> y(count), norm(count), mean(f.cols), inv(f.cols), rm(f.cols), rv(f.cols), ws(2 * f.cols);
    Device<float> old_y(count), old_norm(count), old_mean(f.cols), old_inv(f.cols);
    Device<float> old_rm(f.cols), old_rv(f.cols), old_ws(2 * f.cols);
    Device<__half> h(count), old_h(count);
    x.Put(f.x); res.Put(f.residual); gamma.Put(f.gamma); beta.Put(f.beta);
    rm.Put(f.mean); old_rm.Put(f.mean); rv.Put(f.inv); old_rv.Put(f.inv);
    Require(mgt_cuda::LaunchLocalStridedBatchNormForward(x.get(), f.rows, f.cols,
        f.stride, gamma.get(), beta.get(), old_rm.get(), old_rv.get(), 0.125f, 1e-5f,
        old_y.get(), old_mean.get(), old_inv.get(), old_norm.get(), old_ws.get(),
        nullptr) == mgt::Status::kOk, "old full forward launch");
    OldActivation<<<(count + 255) / 256, 256>>>(old_y.get(), res.get(), old_h.get(), count);
    Cuda(cudaGetLastError());
    mgt_cuda::LocalBatchNormForwardEpilogue epilogue;
    epilogue.relu = true; epilogue.residual = res.get(); epilogue.half_output = h.get();
    Require(mgt_cuda::LaunchLocalStridedBatchNormForward(x.get(), f.rows, f.cols,
        f.stride, gamma.get(), beta.get(), rm.get(), rv.get(), 0.125f, 1e-5f,
        y.get(), mean.get(), inv.get(), norm.get(), ws.get(), nullptr, epilogue)
        == mgt::Status::kOk, "fused full forward launch");
    Cuda(cudaDeviceSynchronize());
    Equal(y.Read(), old_y.Read(), "full forward output");
    Equal(norm.Read(), old_norm.Read(), "full forward normalized");
    Equal(h.Read(), old_h.Read(), "full forward half");
    Equal(mean.Read(), old_mean.Read(), "full forward mean");
    Equal(inv.Read(), old_inv.Read(), "full forward inv_std");
    Equal(rm.Read(), old_rm.Read(), "running mean");
    Equal(rv.Read(), old_rv.Read(), "running variance");
    Equal(ws.Read(), old_ws.Read(), "partial statistics");
    Equal(x.Read(), f.x, "full forward input"); Equal(res.Read(), f.residual, "full forward residual");
    Equal(gamma.Read(), f.gamma, "full forward gamma"); Equal(beta.Read(), f.beta, "full forward beta");
}

// Catches half mirrors overwriting forward-only state that Apply does not see.
void InvalidForwardHalfAliases() {
    Device<float> x(16), y(16), norm(16), res(16), gamma(4), beta(4), mean(4), inv(4);
    Device<float> rm(4), rv(4), ws(8);
    Device<float>* arrays[] = {&x, &y, &norm, &res, &gamma, &beta, &mean, &inv, &rm, &rv, &ws};
    const char* names[] = {"input", "output", "normalized", "residual", "gamma", "beta",
        "mean", "inv_std", "running_mean", "running_var", "stats_workspace"};
    std::vector<std::vector<float>> before;
    for (const auto* array : arrays) before.push_back(array->Read());
    Device<float>* alias_targets[] = {&rm, &rv, &ws};
    const char* alias_names[] = {"running_mean", "running_var", "stats_workspace"};
    for (int target = 0; target < 3; ++target) {
        mgt_cuda::LocalBatchNormForwardEpilogue e;
        e.relu = true;
        e.residual = res.get();
        // Eight half elements fit even the four-float running-state allocations.
        e.half_output = reinterpret_cast<__half*>(alias_targets[target]->get());
        const auto status = mgt_cuda::LaunchLocalStridedBatchNormForward(x.get(), 2, 3, 4,
            gamma.get(), beta.get(), rm.get(), rv.get(), 0.1f, 1e-5f, y.get(),
            mean.get(), inv.get(), norm.get(), ws.get(), nullptr, e);
        Cuda(cudaDeviceSynchronize());
        Require(status == mgt::Status::kInvalidConfig,
            std::string("full-forward half_output alias accepted: ") + alias_names[target]);
        for (std::size_t i = 0; i < before.size(); ++i)
            Equal(arrays[i]->Read(), before[i], names[i]);
    }
}

// Catches validation removed or delayed until after asynchronous writes.
void InvalidCases() {
    Device<float> x(16), y(16), norm(16), res(16), gamma(4), beta(4), mean(4), inv(4);
    Device<__half> half(16);
    const auto before_x = x.Read(), before_y = y.Read(), before_norm = norm.Read();
    const auto before_res = res.Read();
    const auto before_half = half.Read();
    mgt_cuda::LocalBatchNormForwardEpilogue e;
    e.relu = true; e.residual = res.get(); e.half_output = half.get();
    auto reject = [&](int rows, int cols, int stride, const float* input,
                      float* output, float* normalized,
                      mgt_cuda::LocalBatchNormForwardEpilogue epilogue, int null_field = -1) {
        Require(mgt_cuda::LaunchLocalStridedBatchNormApply(input, rows, cols, stride,
            null_field == 0 ? nullptr : gamma.get(), null_field == 1 ? nullptr : beta.get(),
            null_field == 2 ? nullptr : mean.get(), null_field == 3 ? nullptr : inv.get(),
            output, normalized, epilogue, nullptr) == mgt::Status::kInvalidConfig,
            "invalid apply configuration accepted");
    };
    for (int field = 0; field < 4; ++field) reject(2, 3, 4, x.get(), y.get(), norm.get(), e, field);
    reject(2, 3, 4, nullptr, y.get(), norm.get(), e);
    reject(2, 3, 4, x.get(), nullptr, norm.get(), e);
    reject(2, 3, 4, x.get(), y.get(), nullptr, e);
    for (int bad : {0, -1}) {
        reject(bad, 3, 4, x.get(), y.get(), norm.get(), e);
        reject(2, bad, 4, x.get(), y.get(), norm.get(), e);
    }
    reject(2, 3, 2, x.get(), y.get(), norm.get(), e);
    reject(2, 3, -1, x.get(), y.get(), norm.get(), e);
    reject(INT_MAX, 2, 2, x.get(), y.get(), norm.get(), e);
    for (float* alias : {x.get(), y.get(), res.get(), y.get() + 1})
        reject(2, 3, 4, x.get(), y.get(), alias, e);
    for (float* alias : {x.get(), y.get(), x.get() + 1}) {
        auto bad = e; bad.residual = alias;
        reject(2, 3, 4, x.get(), y.get(), norm.get(), bad);
    }
    reject(2, 3, 4, x.get(), x.get() + 1, norm.get(), e);
    for (int mode : {1, 2, 3}) {
        mgt_cuda::LocalBatchNormForwardEpilogue bad;
        bad.residual = mode & 1 ? res.get() : nullptr;
        bad.half_output = mode & 2 ? half.get() : nullptr;
        reject(2, 3, 4, x.get(), y.get(), norm.get(), bad);
    }
    auto half_alias = e; half_alias.half_output = reinterpret_cast<__half*>(y.get());
    reject(2, 3, 4, x.get(), y.get(), norm.get(), half_alias);
    Device<float> rm(4), rv(4), ws(8);
    const auto before_rm = rm.Read(), before_rv = rv.Read(), before_ws = ws.Read();
    auto bad = e; bad.relu = false;
    Require(mgt_cuda::LaunchLocalStridedBatchNormForward(x.get(), 2, 3, 4,
        gamma.get(), beta.get(), rm.get(), rv.get(), 0.1f, 1e-5f, y.get(),
        mean.get(), inv.get(), norm.get(), ws.get(), nullptr, bad) == mgt::Status::kInvalidConfig,
        "invalid full-forward epilogue accepted");
    Cuda(cudaDeviceSynchronize());
    Equal(rm.Read(), before_rm, "invalid call changed running mean");
    Equal(rv.Read(), before_rv, "invalid call changed running variance");
    Equal(ws.Read(), before_ws, "invalid call changed workspace");
    Equal(x.Read(), before_x, "invalid call changed input payload");
    Equal(y.Read(), before_y, "invalid call changed output payload");
    Equal(norm.Read(), before_norm, "invalid call changed normalized payload");
    Equal(res.Read(), before_res, "invalid call changed residual payload");
    Equal(half.Read(), before_half, "invalid call changed half payload");
    x.CheckGuards(); y.CheckGuards(); norm.CheckGuards(); res.CheckGuards(); half.CheckGuards();
    gamma.CheckGuards(); beta.CheckGuards(); mean.CheckGuards(); inv.CheckGuards();
}
}  // namespace

int main() {
    try {
        HandDerivedCases();
        const int shapes[][3] = {{1, 255, 255}, {1, 256, 256}, {1, 257, 257},
            {257, 1, 1}, {3, 5, 7}, {17, 2556, 2560}, {4096, 218, 224}};
        for (const auto& shape : shapes) {
            const Fixture fixture(shape[0], shape[1], shape[2]);
            for (bool inplace : {false, true}) {
                ApplyCase(fixture, false, false, false, inplace);
                for (bool residual : {false, true}) for (bool half : {false, true})
                    ApplyCase(fixture, true, residual, half, inplace);
            }
        }
        FullForwardCase();
        InvalidForwardHalfAliases();
        InvalidCases();
        std::puts("local batch norm epilogue: PASS");
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "local batch norm epilogue: %s\n", error.what());
        return 1;
    }
}
