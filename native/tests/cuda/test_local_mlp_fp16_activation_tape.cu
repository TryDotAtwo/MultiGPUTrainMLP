#include "mgt_cuda/local_mlp_batch_norm.cuh"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using Values = std::vector<float>;
constexpr std::uint32_t kCapacityRows = 4;
constexpr std::size_t kGuardHalfs = 64;
constexpr std::size_t kDeviceGuardElements = 64;
constexpr std::uint16_t kPoison = 0xffff;

void Cuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
}

template <class T> struct Device {
    T* allocation = nullptr;
    T* pointer = nullptr;
    std::size_t count;
    explicit Device(std::size_t size) : count(size) {
        Cuda(cudaMalloc(&allocation, (count + 2 * kDeviceGuardElements) * sizeof(T)), "cudaMalloc");
        pointer = allocation + kDeviceGuardElements;
        try {
            Cuda(cudaMemset(allocation, 0xa5, (count + 2 * kDeviceGuardElements) * sizeof(T)),
                 "initialize allocation canaries");
            Cuda(cudaMemset(pointer, 0, count * sizeof(T)), "cudaMemset");
        } catch (...) {
            cudaFree(allocation);
            allocation = pointer = nullptr;
            throw;
        }
    }
    ~Device() { if (allocation) cudaFree(allocation); }
    Device(const Device&) = delete;
    Device& operator=(const Device&) = delete;
    void Upload(const std::vector<T>& source) {
        if (source.size() > count) throw std::runtime_error("fixture exceeds device capacity");
        Cuda(cudaMemcpy(pointer, source.data(), source.size() * sizeof(T), cudaMemcpyHostToDevice),
             "upload");
    }
    void CheckGuards() const {
        unsigned char guards[2 * kDeviceGuardElements * sizeof(T)];
        Cuda(cudaMemcpy(guards, allocation, kDeviceGuardElements * sizeof(T), cudaMemcpyDeviceToHost),
             "read leading allocation canary");
        Cuda(cudaMemcpy(guards + kDeviceGuardElements * sizeof(T), pointer + count,
                        kDeviceGuardElements * sizeof(T), cudaMemcpyDeviceToHost),
             "read trailing allocation canary");
        for (unsigned char value : guards)
            if (value != 0xa5) throw std::runtime_error("allocation canary changed");
    }
    std::vector<T> Read() const {
        CheckGuards();
        std::vector<T> result(count);
        Cuda(cudaMemcpy(result.data(), pointer, count * sizeof(T), cudaMemcpyDeviceToHost), "read");
        return result;
    }
};

template <class T> void RequireSameBytes(const std::vector<T>& actual,
                                        const std::vector<T>& expected,
                                        const std::string& context) {
    if (actual.size() != expected.size() ||
        std::memcmp(actual.data(), expected.data(), actual.size() * sizeof(T)) != 0)
        throw std::runtime_error(context + " changed bytes");
}

struct Blas {
    cublasHandle_t handle = nullptr;
    Blas() {
        if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS)
            throw std::runtime_error("cublasCreate");
    }
    ~Blas() { if (handle) cublasDestroy(handle); }
};

// This independent layout describes the public model, not the production tape helpers.
struct Layout {
    std::size_t input_bias, hidden_weight, hidden_bias, residual, output_weight, output_bias, count;
    explicit Layout(const mgt_cuda::CudaMlpShape& s) {
        input_bias = static_cast<std::size_t>(s.state_len) * s.state_value_pad * s.hd1;
        hidden_weight = input_bias + s.hd1;
        hidden_bias = hidden_weight + s.hd1 * s.hd2;
        residual = hidden_bias + s.hd2;
        output_weight = residual + s.residual_blocks * 2 * (s.hd2 * s.hd2 + s.hd2);
        output_bias = output_weight + s.hd2 * s.output_dim;
        count = output_bias + s.output_dim;
    }
};

float Half(float x) { return __half2float(__float2half_rn(x)); }
std::uint16_t HalfBits(float x) {
    const __half value = __float2half_rn(x);
    std::uint16_t bits;
    static_assert(sizeof(bits) == sizeof(value), "half storage size");
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

struct BnCache { Values normalized, inverse; };
struct Reference {
    float loss = 0;
    Values outputs, weight_grad, affine_grad, running, input_activation;
    std::vector<Values> block_inputs, fc1_activations;
};

// Plain host loops are the oracle. The FP32 mode measures numerical agreement;
// the rounded-operand mode additionally checks the actual mixed-precision contract.
Reference CpuReference(const mgt_cuda::CudaMlpShape& s, const Values& w, const Values& affine,
                       const Values& running,
                       const std::vector<mgt::TrainStateStorage>& states, const Values& labels,
                       std::uint32_t rows, bool round_operands) {
    const Layout layout(s);
    const std::size_t features = s.hd1 + (1 + 2 * s.residual_blocks) * s.hd2;
    Reference result;
    result.running = running;
    result.weight_grad.assign(layout.count, 0);
    result.affine_grad.assign(2 * features, 0);
    std::vector<BnCache> caches(2 + 2 * s.residual_blocks);
    auto operand = [](float value, bool rounded) { return rounded ? Half(value) : value; };
    auto dense = [&](const Values& x, unsigned in, unsigned out, std::size_t weight,
                     std::size_t bias, bool rounded) {
        Values y(rows * out);
        for (unsigned r = 0; r < rows; ++r) for (unsigned o = 0; o < out; ++o) {
            float sum = 0;
            for (unsigned i = 0; i < in; ++i)
                sum += operand(x[r * in + i], rounded) * operand(w[weight + i * out + o], rounded);
            y[r * out + o] = sum + w[bias + o];
        }
        return y;
    };
    auto dense_backward = [&](const Values& x, const Values& dy, unsigned in, unsigned out,
                              std::size_t weight, std::size_t bias, bool rounded) {
        Values dx(rows * in, 0);
        for (unsigned r = 0; r < rows; ++r) for (unsigned o = 0; o < out; ++o) {
            const float grad = operand(dy[r * out + o], rounded);
            result.weight_grad[bias + o] += dy[r * out + o];
            for (unsigned i = 0; i < in; ++i) {
                result.weight_grad[weight + i * out + o] += grad * operand(x[r * in + i], rounded);
                dx[r * in + i] += grad * operand(w[weight + i * out + o], rounded);
            }
        }
        return dx;
    };
    auto bn_forward = [&](Values x, unsigned width, unsigned site) {
        auto& cache = caches[site];
        cache.normalized.resize(x.size());
        cache.inverse.resize(width);
        const std::size_t offset = site == 0 ? 0 : s.hd1 + (site - 1) * s.hd2;
        for (unsigned c = 0; c < width; ++c) {
            float sum = 0, squares = 0;
            for (unsigned r = 0; r < rows; ++r) {
                const float value = x[r * width + c];
                sum += value;
                squares += value * value;
            }
            const float mean = sum / rows;
            const float variance = std::max(squares / rows - mean * mean, 0.0f);
            const float inverse = 1.0f / std::sqrt(variance + 1e-5f);
            result.running[offset + c] = 0.9f * running[offset + c] + 0.1f * mean;
            const float unbiased = rows > 1 ? variance * rows / (rows - 1) : 0.0f;
            result.running[features + offset + c] =
                0.9f * running[features + offset + c] + 0.1f * unbiased;
            cache.inverse[c] = inverse;
            for (unsigned r = 0; r < rows; ++r) {
                const unsigned index = r * width + c;
                cache.normalized[index] = (x[index] - mean) * inverse;
                x[index] = cache.normalized[index] * affine[offset + c] + affine[features + offset + c];
            }
        }
        return x;
    };
    auto bn_backward = [&](Values dy, unsigned width, unsigned site) {
        const auto& cache = caches[site];
        const std::size_t offset = site == 0 ? 0 : s.hd1 + (site - 1) * s.hd2;
        for (unsigned c = 0; c < width; ++c) {
            float beta_grad = 0, gamma_grad = 0;
            for (unsigned r = 0; r < rows; ++r) {
                const unsigned index = r * width + c;
                beta_grad += dy[index];
                gamma_grad += dy[index] * cache.normalized[index];
            }
            result.affine_grad[offset + c] = gamma_grad;
            result.affine_grad[features + offset + c] = beta_grad;
            const float scale = affine[offset + c] * cache.inverse[c] / rows;
            for (unsigned r = 0; r < rows; ++r) {
                const unsigned index = r * width + c;
                dy[index] = scale * (rows * dy[index] - beta_grad - cache.normalized[index] * gamma_grad);
            }
        }
        return dy;
    };
    auto relu = [](Values x) { for (float& value : x) value = value > 0 ? value : 0; return x; };
    auto relu_backward = [](Values grad, const Values& activated) {
        for (std::size_t i = 0; i < grad.size(); ++i) if (!(activated[i] > 0)) grad[i] = 0;
        return grad;
    };
    Values input(rows * s.hd1);
    for (unsigned r = 0; r < rows; ++r) for (unsigned h = 0; h < s.hd1; ++h) {
        float value = operand(w[layout.input_bias + h], round_operands);
        for (unsigned p = 0; p < s.state_len; ++p)
            value += operand(w[(p * s.state_value_pad + states[r].v[p]) * s.hd1 + h], round_operands);
        input[r * s.hd1 + h] = value;
    }
    result.input_activation = relu(bn_forward(input, s.hd1, 0));
    result.block_inputs.push_back(relu(bn_forward(dense(result.input_activation, s.hd1, s.hd2,
        layout.hidden_weight, layout.hidden_bias, round_operands && s.hd2 > 1), s.hd2, 1)));
    for (unsigned b = 0; b < s.residual_blocks; ++b) {
        const std::size_t first = layout.residual + b * 2 * (s.hd2 * s.hd2 + s.hd2);
        const std::size_t second = first + s.hd2 * s.hd2 + s.hd2;
        result.fc1_activations.push_back(relu(bn_forward(dense(result.block_inputs.back(), s.hd2, s.hd2,
            first, first + s.hd2 * s.hd2, round_operands && s.hd2 > 1), s.hd2, 2 + 2 * b)));
        Values next = bn_forward(dense(result.fc1_activations.back(), s.hd2, s.hd2,
            second, second + s.hd2 * s.hd2, round_operands && s.hd2 > 1), s.hd2, 3 + 2 * b);
        for (std::size_t i = 0; i < next.size(); ++i) next[i] += result.block_inputs.back()[i];
        result.block_inputs.push_back(relu(next));
    }
    result.outputs = dense(result.block_inputs.back(), s.hd2, s.output_dim,
        layout.output_weight, layout.output_bias, round_operands && s.output_dim > 1);
    Values dy(result.outputs.size());
    const float norm = 1.0f / (rows * s.output_dim);
    for (std::size_t i = 0; i < dy.size(); ++i) {
        const float error = result.outputs[i] - labels[i];
        result.loss += error * error * norm;
        dy[i] = 2 * error * norm;
    }
    Values grad = dense_backward(result.block_inputs.back(), dy, s.hd2, s.output_dim,
        layout.output_weight, layout.output_bias, round_operands && s.output_dim > 1);
    for (unsigned b = s.residual_blocks; b-- > 0;) {
        const std::size_t first = layout.residual + b * 2 * (s.hd2 * s.hd2 + s.hd2);
        const std::size_t second = first + s.hd2 * s.hd2 + s.hd2;
        Values residual = relu_backward(grad, result.block_inputs[b + 1]);
        grad = bn_backward(residual, s.hd2, 3 + 2 * b);
        grad = dense_backward(result.fc1_activations[b], grad, s.hd2, s.hd2,
            second, second + s.hd2 * s.hd2, round_operands);
        grad = bn_backward(relu_backward(grad, result.fc1_activations[b]), s.hd2, 2 + 2 * b);
        grad = dense_backward(result.block_inputs[b], grad, s.hd2, s.hd2,
            first, first + s.hd2 * s.hd2, round_operands);
        for (std::size_t i = 0; i < grad.size(); ++i) grad[i] += residual[i];
    }
    grad = bn_backward(relu_backward(grad, result.block_inputs[0]), s.hd2, 1);
    grad = dense_backward(result.input_activation, grad, s.hd1, s.hd2,
        layout.hidden_weight, layout.hidden_bias, round_operands);
    grad = bn_backward(relu_backward(grad, result.input_activation), s.hd1, 0);
    for (unsigned r = 0; r < rows; ++r) for (unsigned h = 0; h < s.hd1; ++h) {
        result.weight_grad[layout.input_bias + h] += grad[r * s.hd1 + h];
        for (unsigned p = 0; p < s.state_len; ++p)
            result.weight_grad[(p * s.state_value_pad + states[r].v[p]) * s.hd1 + h] += grad[r * s.hd1 + h];
    }
    return result;
}

bool Compare(const char* name, const Values& actual, const Values& expected,
             float absolute, float relative) {
    if (actual.size() != expected.size()) {
        std::fprintf(stderr, "%s size=%zu expected=%zu\n", name, actual.size(), expected.size());
        return false;
    }
    float max_absolute = 0, max_ratio = 0;
    std::size_t worst = 0;
    for (std::size_t i = 0; i < actual.size(); ++i) {
        const float tolerance = absolute + relative * std::fabs(expected[i]);
        if (!std::isfinite(actual[i]) || !std::isfinite(expected[i])) {
            std::fprintf(stderr, "%s[%zu]=%.9g expected=%.9g tolerance=%.9g\n",
                         name, i, actual[i], expected[i], tolerance);
            return false;
        }
        const float difference = std::fabs(actual[i] - expected[i]);
        max_absolute = std::max(max_absolute, difference);
        if (difference / tolerance > max_ratio) {
            max_ratio = difference / tolerance;
            worst = i;
        }
    }
    std::fprintf(stdout, "%s max_abs=%.9g max_tolerance_ratio=%.9g worst_index=%zu\n",
                 name, max_absolute, max_ratio, worst);
    if (max_ratio > 1) {
        std::fprintf(stderr, "%s[%zu]=%.9g expected=%.9g absolute_tolerance=%.9g relative_tolerance=%.9g\n",
                     name, worst, actual[worst], expected[worst], absolute, relative);
        return false;
    }
    return true;
}

bool CheckWeightMirror(const Values& master, const std::vector<__half>& mirror) {
    for (std::size_t i = 0; i < master.size(); ++i) {
        std::uint16_t actual;
        std::memcpy(&actual, &mirror[i], sizeof(actual));
        const auto expected = HalfBits(master[i]);
        if (actual != expected) {
            std::fprintf(stderr, "updated weight_mirror[%zu]=0x%04x expected=0x%04x master=%.9g\n",
                         i, static_cast<unsigned>(actual), static_cast<unsigned>(expected), master[i]);
            return false;
        }
    }
    return true;
}

void RequireEmptyGradientCache(const mgt_cuda::LocalMlpFp16Context& fp16,
                               const char* phase) {
    if (fp16.cached_operand_b_source != nullptr || fp16.cached_operand_b_count != 0) {
        std::fprintf(stderr, "gradient cache not empty after %s: source=%p count=%llu\n",
            phase, static_cast<const void*>(fp16.cached_operand_b_source),
            static_cast<unsigned long long>(fp16.cached_operand_b_count));
        throw std::runtime_error("gradient cache escaped its training-step lifetime");
    }
}

void PoisonGradientCache(mgt_cuda::LocalMlpFp16Context& fp16,
                         Device<__half>& operand_b, const float* stale_source,
                         std::uint64_t stale_count) {
    Cuda(cudaMemset(operand_b.pointer, 0xff, operand_b.count * sizeof(__half)),
         "poison gradient operand and canary");
    fp16.cached_operand_b_source = stale_source;
    fp16.cached_operand_b_count = stale_count;
}

bool CheckLastDenseGradientMirror(const Values& block_gradient,
                                 const std::vector<__half>& operand_b,
                                 std::size_t hidden_elements,
                                 std::size_t active_dense_capacity) {
    // Hidden BN is the last dense-gradient producer. The later input BN must
    // not overwrite this scratch, even when hd1==hd2 makes its extent fit.
    if (block_gradient.size() < hidden_elements ||
        operand_b.size() < active_dense_capacity || hidden_elements > active_dense_capacity)
        throw std::runtime_error("gradient mirror test extent");
    for (std::size_t i = 0; i < hidden_elements; ++i) {
        std::uint16_t actual;
        std::memcpy(&actual, &operand_b[i], sizeof(actual));
        const auto expected = HalfBits(block_gradient[i]);
        if (!std::isfinite(block_gradient[i]) || actual != expected) {
            std::fprintf(stderr, "last dense gradient mirror[%zu]=0x%04x expected=0x%04x "
                         "block_grad=%.9g\n", i, static_cast<unsigned>(actual),
                         static_cast<unsigned>(expected), block_gradient[i]);
            return false;
        }
    }
    // A vector head wider than hd2 may have used the middle suffix legitimately.
    // Only the area beyond every active dense operand is an untouched canary.
    for (std::size_t i = active_dense_capacity; i < operand_b.size(); ++i) {
        std::uint16_t actual;
        std::memcpy(&actual, &operand_b[i], sizeof(actual));
        if (actual != kPoison) {
            std::fprintf(stderr, "gradient operand canary[%zu]=0x%04x expected=0xffff\n",
                         i, static_cast<unsigned>(actual));
            return false;
        }
    }
    return true;
}

std::uint64_t TapeCount(const mgt_cuda::CudaMlpShape& s, std::uint32_t rows) {
    // No final half slot for the FP32 scalar head; vector-head dW needs it.
    return static_cast<std::uint64_t>(rows) *
        (s.hd1 + 2ULL * s.residual_blocks * s.hd2 + (s.output_dim > 1 ? s.hd2 : 0));
}

bool CheckTape(const mgt_cuda::CudaMlpShape& s, unsigned rows, const Values& saved,
               const std::vector<std::uint16_t>& tape) {
    const std::size_t bh1 = rows * s.hd1, bh2 = rows * s.hd2;
    const std::size_t fc1 = bh1 + (s.residual_blocks + 1) * bh2;
    auto mirror = [&](const char* site, unsigned block, std::size_t source,
                      std::size_t target, std::size_t count) {
        for (std::size_t i = 0; i < count; ++i) {
            const auto expected = HalfBits(saved[source + i]);
            if (tape[target + i] != expected) {
                std::fprintf(stderr, "tape site=%s block=%u slot_offset=%zu element=%zu bits=0x%04x "
                             "expected=0x%04x source=%.9g\n", site, block, target, i,
                             static_cast<unsigned>(tape[target + i]), static_cast<unsigned>(expected),
                             saved[source + i]);
                return false;
            }
        }
        return true;
    };
    bool ok = mirror("input", 0, 0, 0, bh1);
    for (unsigned b = 0; b < s.residual_blocks; ++b) {
        ok = mirror(b == 0 ? "hidden" : "intermediate_residual", b,
                    bh1 + b * bh2, bh1 + 2 * b * bh2, bh2) && ok;
        ok = mirror("fc1", b, fc1 + b * bh2, bh1 + (2 * b + 1) * bh2, bh2) && ok;
    }
    if (s.output_dim > 1)
        ok = mirror("final_vector_head", s.residual_blocks, bh1 + s.residual_blocks * bh2,
                    bh1 + 2 * s.residual_blocks * bh2, bh2) && ok;
    for (std::size_t i = TapeCount(s, rows); i < tape.size(); ++i) {
        if (tape[i] != kPoison) {
            std::fprintf(stderr, "tape canary overwritten: offset=%zu logical_capacity=%llu "
                         "bits=0x%04x expected=0xffff\n", i,
                         static_cast<unsigned long long>(TapeCount(s, rows)),
                         static_cast<unsigned>(tape[i]));
            ok = false;
            break;
        }
    }
    return ok;
}

// Catches zero-block overflow, missing vector-head activation, wrong residual
// slots, stale cross-step rows/gradient-cache tags, omitted capacity checks,
// overwritten final dense mirrors, and corrupted dense dW.
bool RunCase(unsigned blocks, unsigned output_dim, unsigned hd1 = 3, unsigned hd2 = 2) {
    const mgt_cuda::CudaMlpShape shape{2, 4, hd1, hd2, blocks, output_dim};
    const Layout layout(shape);
    unsigned active_rows = 0;
    bool equal_width_input_is_distinct = hd1 != hd2;
    try {
        mgt::BatchNormTrainingPlan plan;
        if (mgt::BuildBatchNormTrainingPlan(hd1, hd2, hd1, hd2, blocks, kCapacityRows, &plan)
            != mgt::Status::kOk) throw std::runtime_error("BuildBatchNormTrainingPlan");
        auto bn = mgt::InitializeBatchNormTrainingState(plan);
        // Fixed dyadic weights and non-correlated input pairs keep every BN
        // variance away from zero on both three and four rows. A periodic
        // arithmetic-progression fixture can hide a nearly constant channel.
        const float fixture_weights[]{
            -.8125f, -.75f, .3125f, .875f, .125f, 1.f, .375f, .5625f,
            .875f, .0625f, .3125f, -.25f, -.8125f, -.6875f, -.9375f, .1875f,
            -.6875f, .9375f, .5625f, .5625f, -.3125f, -.125f, -.8125f, -.625f,
            .4375f, -1.f, .875f, .625f, .8125f, -.5f, -.25f, -.5625f,
            .6875f, -.5f, -.25f, -.8125f, 1.f, -.9375f, .25f, .0625f,
            -.875f, -.6875f, -.125f, -1.f, .4375f, .25f, .5625f, -.75f,
            .1875f, -.0625f, .4375f, .1875f, .1875f, .5f, -.25f, .375f,
            -.5f, -.3125f, -.125f, .8125f, .5625f, .5625f, -.0625f, -.375f, -.5625f};
        if (layout.count > sizeof(fixture_weights) / sizeof(fixture_weights[0]))
            throw std::runtime_error("shape exceeds hand-seeded fixture");
        const Values initial_weights(fixture_weights, fixture_weights + layout.count);
        for (std::size_t i = 0; i < plan.logical_feature_count; ++i) {
            bn.affine[i] = 0.75f + 0.125f * (i % 3);
            bn.affine[plan.logical_feature_count + i] = 0.125f * (static_cast<int>(i % 3) - 1);
        }
        const auto workspace_count = mgt_cuda::LocalMlpBatchNormForwardWorkspaceFloats(shape, plan, kCapacityRows);
        Device<float> weights(layout.count), weight_grad(layout.count), weight_m(layout.count), weight_v(layout.count);
        Device<float> affine(bn.affine.size()), affine_grad(bn.affine.size()), affine_m(bn.affine.size()),
            affine_v(bn.affine.size()), running(bn.running.size());
        Device<float> outputs(kCapacityRows * output_dim), workspace(workspace_count), loss(1),
            output_dy(kCapacityRows * output_dim), block_grad(kCapacityRows * hd2),
            fc1_grad(kCapacityRows * hd2), residual_grad(kCapacityRows * hd2), input_grad(kCapacityRows * hd1),
            labels(kCapacityRows * output_dim);
        Device<mgt::TrainStateStorage> states(kCapacityRows);
        const std::uint64_t gradient_capacity = kCapacityRows * std::max(hd2, output_dim);
        Device<__half> weight_half(layout.count), tape(TapeCount(shape, kCapacityRows) + kGuardHalfs),
            operand_b(gradient_capacity + kGuardHalfs);
        weights.Upload(initial_weights);
        affine.Upload(bn.affine);
        running.Upload(bn.running);
        if (mgt_cuda::LaunchFloatToHalf(weights.pointer, weight_half.pointer, layout.count, nullptr)
            != mgt::Status::kOk) throw std::runtime_error("initialize weight mirror");
        Blas blas;
        const mgt_cuda::MlpBatchNormStepBuffers buffers{
            weights.pointer, weight_grad.pointer, weight_m.pointer, weight_v.pointer,
            affine.pointer, affine_grad.pointer, affine_m.pointer, affine_v.pointer,
            running.pointer, outputs.pointer, workspace.pointer, loss.pointer, output_dy.pointer,
            block_grad.pointer, fc1_grad.pointer, residual_grad.pointer, input_grad.pointer};
        mgt_cuda::LocalMlpFp16Context fp16{weights.pointer, weight_half.pointer, tape.pointer,
            operand_b.pointer, TapeCount(shape, kCapacityRows), gradient_capacity};
        auto alias_preflight = [&](unsigned rows) {
            const std::size_t live_b = static_cast<std::size_t>(rows) * std::max(hd2, output_dim);
            const std::size_t live_labels = static_cast<std::size_t>(rows) * output_dim;
            Cuda(cudaMemset(operand_b.pointer, 0xff, operand_b.count * sizeof(__half)),
                 "poison alias-test gradient scratch");
            Device<float>* floats[]{&weights, &weight_grad, &weight_m, &weight_v,
                &affine, &affine_grad, &affine_m, &affine_v, &running, &outputs,
                &workspace, &loss, &output_dy, &block_grad, &fc1_grad, &residual_grad,
                &input_grad, &labels};
            const char* names[]{"weights", "weight_grad", "weight_m", "weight_v",
                "affine", "affine_grad", "affine_m", "affine_v", "running", "outputs",
                "forward_workspace", "loss", "output_dy", "block_grad", "fc1_grad",
                "residual_grad", "input_grad", "labels"};
            std::vector<Values> before;
            for (auto* array : floats) before.push_back(array->Read());
            const auto before_states = states.Read();
            const auto before_half = weight_half.Read(), before_tape = tape.Read(),
                       before_b = operand_b.Read();
            const mgt_cuda::AdamWKernelConfig probe_adam{0, 1, .0001f, .9f, .999f, 1e-8f, 0.f};
            unsigned rejected = 0, accepted = 0;
            auto trial_context = [&] {
                auto trial = fp16;
                trial.operand_a_capacity = TapeCount(shape, rows);
                trial.operand_b_capacity = live_b;
                trial.cached_operand_b_source = buffers.output_dy;
                trial.cached_operand_b_count = live_labels;
                return trial;
            };
            auto unchanged = [&](const std::string& name) {
                for (std::size_t i = 0; i < before.size(); ++i)
                    RequireSameBytes(floats[i]->Read(), before[i], name + ": " + names[i]);
                RequireSameBytes(states.Read(), before_states, name + ": states");
                RequireSameBytes(weight_half.Read(), before_half, name + ": weight half");
                RequireSameBytes(tape.Read(), before_tape, name + ": activation tape");
                RequireSameBytes(operand_b.Read(), before_b, name + ": gradient scratch");
            };
            auto reject = [&](const char* name, mgt_cuda::LocalMlpFp16Context trial,
                              mgt_cuda::MlpBatchNormStepBuffers views,
                              mgt::Status expected = mgt::Status::kInvalidConfig,
                              const mgt::BatchNormTrainingPlan* alternate_plan = nullptr,
                              unsigned logical_hd2 = 0) {
                const auto status = mgt_cuda::LaunchLocalMlpBatchNormTrainStepFp16(shape,
                    hd1, logical_hd2 ? logical_hd2 : hd2, states.pointer, labels.pointer,
                    rows, alternate_plan ? *alternate_plan : plan, workspace_count,
                    probe_adam, views, &trial, blas.handle, nullptr);
                Cuda(cudaDeviceSynchronize(), "synchronize alias-preflight rejection");
                unchanged(name);
                RequireEmptyGradientCache(trial, name);
                if (status != expected) {
                    std::fprintf(stderr, "alias preflight %s status=%d expected=%d\n", name,
                                 static_cast<int>(status), static_cast<int>(expected));
                    throw std::runtime_error("alias preflight accepted invalid live views");
                }
                ++rejected;
            };
            // Equal-pointer coverage names every protected category. Buffer
            // canaries make even the loss-scalar alias safe to diagnose when a
            // broken preflight lets the larger half write cross its payload.
            for (std::size_t i = 0; i < before.size(); ++i) {
                if (blocks == 0 && (floats[i] == &fc1_grad || floats[i] == &residual_grad)) continue;
                auto trial = trial_context();
                trial.operand_b = reinterpret_cast<__half*>(floats[i]->pointer);
                reject(names[i], trial, buffers);
            }
            for (auto* target : {&weight_half, &tape}) {
                auto trial = trial_context(); trial.operand_b = target->pointer;
                reject(target == &tape ? "activation tape" : "weight half", trial, buffers);
            }
            { auto trial = trial_context(); trial.operand_b = reinterpret_cast<__half*>(states.pointer);
              reject("states", trial, buffers); }
            { auto trial = trial_context();
              trial.operand_b = reinterpret_cast<__half*>(output_dy.pointer) + 2 * live_labels - 1;
              reject("B begins at output_dy end-minus-one-half", trial, buffers); }
            { auto trial = trial_context(); auto views = buffers;
              views.output_dy = reinterpret_cast<float*>(operand_b.pointer + live_b - 2);
              reject("output_dy begins inside B suffix", trial, views); }
            { auto trial = trial_context(); auto views = buffers;
              views.loss = reinterpret_cast<float*>(operand_b.pointer + live_b - 2);
              reject("smaller loss view inside B suffix", trial, views); }
            { auto trial = trial_context(); trial.operand_b = nullptr;
              reject("null B", trial, buffers, mgt::Status::kCapacityExceeded); }
            { auto trial = trial_context(); trial.operand_b_capacity = live_b - 1;
              reject("short B", trial, buffers, mgt::Status::kCapacityExceeded); }
            // These deliberately non-dereferenceable ranges must be rejected
            // entirely on the host, before any kernel can observe the pointer.
            { auto trial = trial_context();
              trial.operand_b = reinterpret_cast<__half*>(
                  std::numeric_limits<std::uintptr_t>::max() - 1);
              reject("B byte extent overflows address space", trial, buffers); }
            { auto views = buffers;
              views.weight_grad = reinterpret_cast<float*>(
                  std::numeric_limits<std::uintptr_t>::max() - 3);
              reject("protected byte extent overflows address space", trial_context(), views); }
            if (hd2 > output_dim) {
                // Physical padding still belongs to the dense mirror. A
                // logical-width-only B range would miss this loss overlap.
                mgt::BatchNormTrainingPlan padded;
                if (mgt::BuildBatchNormTrainingPlan(hd1, hd2 - 1, hd1, hd2, blocks,
                                                   kCapacityRows, &padded) != mgt::Status::kOk)
                    throw std::runtime_error("build padded alias fixture");
                auto trial = trial_context(); auto views = buffers;
                views.loss = reinterpret_cast<float*>(operand_b.pointer + live_b - 2);
                reject("physical-only B tail overlaps loss", trial, views,
                       mgt::Status::kInvalidConfig, &padded, hd2 - 1);
            }
            { auto broken = plan; broken.sites[0].normalized_offset = broken.workspace_floats;
              reject("normalized slice outside BN workspace", trial_context(), buffers,
                     mgt::Status::kInvalidConfig, &broken); }
            { auto broken = plan; broken.sites[1].physical_stride = hd2 + 1;
              reject("BN site stride differs from physical shape", trial_context(), buffers,
                     mgt::Status::kInvalidConfig, &broken); }
            { auto broken = plan; broken.sites[1].logical_features = hd2 + 1;
              reject("BN site logical extent exceeds physical shape", trial_context(), buffers,
                     mgt::Status::kInvalidConfig, &broken); }

            auto restore = [&] {
                for (std::size_t i = 0; i < before.size(); ++i) {
                    floats[i]->CheckGuards(); floats[i]->Upload(before[i]);
                }
                states.CheckGuards(); weight_half.CheckGuards(); tape.CheckGuards(); operand_b.CheckGuards();
                states.Upload(before_states); weight_half.Upload(before_half);
                tape.Upload(before_tape); operand_b.Upload(before_b);
                Cuda(cudaDeviceSynchronize(), "restore isolated alias-probe state");
            };
            auto accept = [&](const char* name, mgt_cuda::LocalMlpFp16Context trial,
                              const float* label_view) {
                const auto status = mgt_cuda::LaunchLocalMlpBatchNormTrainStepFp16(shape,
                    hd1, hd2, states.pointer, label_view, rows, plan, workspace_count,
                    probe_adam, buffers, &trial, blas.handle, nullptr);
                Cuda(cudaDeviceSynchronize(), "synchronize legal alias boundary");
                RequireEmptyGradientCache(trial, name);
                if (status != mgt::Status::kOk)
                    throw std::runtime_error(std::string(name) + " rejected legal disjoint live views");
                std::vector<__half> mirror(live_b);
                Cuda(cudaMemcpy(mirror.data(), trial.operand_b, live_b * sizeof(__half),
                                cudaMemcpyDeviceToHost), "read legal-boundary dense mirror");
                if (!CheckLastDenseGradientMirror(block_grad.Read(), mirror,
                        static_cast<std::size_t>(rows) * hd2, live_b))
                    throw std::runtime_error(std::string(name) + " wrong dense mirror");
                RequireSameBytes(states.Read(), before_states, "legal alias states");
                RequireSameBytes(labels.Read(), before.back(), "legal alias original labels");
                // Legal probes are full steps, but are isolated from the three
                // oracle-checked steps by restoring all test-owned model state.
                restore();
                ++accepted;
            };
            // The backing allocation includes both views. Only active B bytes
            // count: its advertised capacity may extend through the label view.
            Device<__half> packed(live_b + 2 * live_labels);
            for (bool b_first : {true, false}) {
                auto trial = trial_context();
                trial.operand_b = packed.pointer + (b_first ? 0 : 2 * live_labels);
                trial.operand_b_capacity = b_first ? packed.count : live_b;
                float* label_view = reinterpret_cast<float*>(packed.pointer + (b_first ? live_b : 0));
                Cuda(cudaMemset(trial.operand_b, 0xff, live_b * sizeof(__half)), "poison packed B view");
                Cuda(cudaMemcpy(label_view, labels.pointer, live_labels * sizeof(float),
                                cudaMemcpyDeviceToDevice), "prepare adjacent label view");
                accept(b_first ? "B ends exactly at labels" : "labels end exactly at B", trial, label_view);
                std::vector<float> actual_labels(live_labels);
                Cuda(cudaMemcpy(actual_labels.data(), label_view, live_labels * sizeof(float),
                                cudaMemcpyDeviceToHost), "read adjacent labels");
                RequireSameBytes(actual_labels, Values(before.back().begin(), before.back().begin() + live_labels),
                                 "adjacent label payload");
                packed.CheckGuards();
            }
            if (blocks == 0) for (auto* target : {&fc1_grad, &residual_grad}) {
                auto trial = trial_context();
                trial.operand_b = reinterpret_cast<__half*>(target->pointer);
                trial.operand_b_capacity = target->count * 2;
                accept(target == &fc1_grad ? "unused FC1 storage" : "unused residual storage",
                       trial, labels.pointer);
            }
            unchanged("alias-probe restoration");
            std::fprintf(stdout, "PASS alias-preflight hd1=%u hd2=%u nrd=%u out=%u rejects=%u legal=%u\n",
                         hd1, hd2, blocks, output_dim, rejected, accepted);
        };
        const unsigned row_sequence[]{4, 3, 4};
        const unsigned state_pairs[3][4][2]{
            {{0, 1}, {1, 3}, {2, 0}, {3, 2}},
            {{1, 0}, {3, 2}, {0, 3}, {2, 1}},
            {{2, 3}, {0, 2}, {3, 1}, {1, 0}}};
        for (unsigned step = 0; step < 3; ++step) {
            active_rows = row_sequence[step];
            std::vector<mgt::TrainStateStorage> host_states(kCapacityRows);
            Values host_labels(kCapacityRows * output_dim);
            const float targets[]{1.0f, 3.0f, -1.0f, 2.0f};
            for (unsigned r = 0; r < kCapacityRows; ++r) {
                host_states[r].v[0] = static_cast<mgt::StateValue>(state_pairs[step][r][0]);
                host_states[r].v[1] = static_cast<mgt::StateValue>(state_pairs[step][r][1]);
                for (unsigned o = 0; o < output_dim; ++o)
                    host_labels[r * output_dim + o] = targets[r] + 0.25f * o - 0.125f * step;
            }
            states.Upload(host_states);
            labels.Upload(host_labels);
            if (step == 0) alias_preflight(active_rows);
            const auto host_weights = weights.Read(), host_affine = affine.Read(), host_running = running.Read();
            const auto fp32 = CpuReference(shape, host_weights, host_affine, host_running, host_states,
                                           host_labels, active_rows, false);
            const auto mixed = CpuReference(shape, host_weights, host_affine, host_running, host_states,
                                            host_labels, active_rows, true);
            Cuda(cudaMemset(tape.pointer, 0xff, tape.count * sizeof(__half)), "poison tape");
            // Characterize overwrite independence before changing production clears.
            // A missing write or beta=1 accumulation must not inherit either NaN
            // or a finite nonzero sentinel from any preceding training step.
            const unsigned char gradient_poison = step % 2 == 0 ? 0xff : 0x5a;
            Cuda(cudaMemset(weight_grad.pointer, gradient_poison, weight_grad.count * sizeof(float)),
                 "poison complete weight gradient before step");
            Cuda(cudaMemset(affine_grad.pointer, gradient_poison, affine_grad.count * sizeof(float)),
                 "poison complete affine gradient before step");
            Cuda(cudaMemset(loss.pointer, gradient_poison, loss.count * sizeof(float)),
                 "poison loss before step");
            fp16.operand_a_capacity = TapeCount(shape, active_rows);
            const std::uint64_t active_gradient_capacity =
                static_cast<std::uint64_t>(active_rows) * std::max(hd2, output_dim);
            fp16.operand_b_capacity = active_gradient_capacity;
            // The first vector-head dW precedes every BN mirror producer. A
            // matching stale output_dy tag must never authorize these NaNs.
            // Subsequent steps exercise a different source with matching extent,
            // then the matching output source with the wrong element count.
            const float* stale_source = step == 1 ? buffers.block_grad : buffers.output_dy;
            const std::uint64_t stale_count = static_cast<std::uint64_t>(active_rows) *
                (step == 1 ? hd2 : output_dim) + (step == 2 ? 1ULL : 0ULL);
            PoisonGradientCache(fp16, operand_b, stale_source, stale_count);
            const mgt_cuda::AdamWKernelConfig adam{0, step + 1ULL, .0001f, .9f, .999f, 1e-8f, 0.f};
            const auto status = mgt_cuda::LaunchLocalMlpBatchNormTrainStepFp16(shape, hd1, hd2,
                states.pointer, labels.pointer, active_rows, plan, workspace_count, adam, buffers,
                &fp16, blas.handle, nullptr);
            Cuda(cudaDeviceSynchronize(), "synchronize training step");
            std::vector<std::uint16_t> tape_bits(tape.count);
            Cuda(cudaMemcpy(tape_bits.data(), tape.pointer, tape_bits.size() * sizeof(std::uint16_t),
                            cudaMemcpyDeviceToHost), "read tape bits");
            if (status != mgt::Status::kOk) {
                std::fprintf(stderr, "train_step status=%d expected=kOk (final head must have a valid operand)\n",
                             static_cast<int>(status));
                throw std::runtime_error("training launch failed");
            }
            RequireEmptyGradientCache(fp16, "successful full step");
            bool ok = CheckTape(shape, active_rows, workspace.Read(), tape_bits);
            const auto final_block_gradient = block_grad.Read();
            ok = CheckLastDenseGradientMirror(final_block_gradient, operand_b.Read(),
                static_cast<std::size_t>(active_rows) * hd2, active_gradient_capacity) && ok;
            if (hd1 == hd2) {
                const auto final_input_gradient = input_grad.Read();
                for (std::size_t i = 0; i < static_cast<std::size_t>(active_rows) * hd2; ++i)
                    equal_width_input_is_distinct = equal_width_input_is_distinct ||
                        HalfBits(final_input_gradient[i]) != HalfBits(final_block_gradient[i]);
            }
            std::fprintf(stdout, "CHECK hd1=%u hd2=%u nrd=%u output_dim=%u rows=%u step=%u\n",
                         hd1, hd2, blocks, output_dim, active_rows, step + 1);
            const auto actual_loss = loss.Read(), actual_weight_grad = weight_grad.Read(),
                       actual_affine_grad = affine_grad.Read();
            Values actual_outputs = outputs.Read();
            actual_outputs.resize(active_rows * output_dim);
            // Two-block FP16 arithmetic may deviate from unrounded FP32; the
            // independently rounded oracle below has a much tighter gate.
            ok = Compare("fp32.loss", actual_loss, {fp32.loss}, 5e-3f, 1e-2f) && ok;
            ok = Compare("fp32.weight_grad", actual_weight_grad, fp32.weight_grad, 8e-3f, 3e-2f) && ok;
            ok = Compare("fp32.affine_grad", actual_affine_grad, fp32.affine_grad, 8e-3f, 3e-2f) && ok;
            ok = Compare("mixed.outputs", actual_outputs, mixed.outputs, 3e-4f, 2e-3f) && ok;
            ok = Compare("mixed.loss", actual_loss, {mixed.loss}, 3e-4f, 2e-3f) && ok;
            ok = Compare("mixed.weight_grad", actual_weight_grad, mixed.weight_grad, 5e-4f, 3e-3f) && ok;
            ok = Compare("mixed.affine_grad", actual_affine_grad, mixed.affine_grad, 5e-4f, 3e-3f) && ok;
            ok = Compare("mixed.running", running.Read(), mixed.running, 2e-5f, 1e-4f) && ok;
            ok = CheckWeightMirror(weights.Read(), weight_half.Read()) && ok;
            if (!ok) throw std::runtime_error("activation/gradient mismatch");

            // Rejection must happen before any kernel changes the tape or parameters.
            const auto before_weights = weights.Read(), before_affine = affine.Read(), before_running = running.Read();
            Cuda(cudaMemset(tape.pointer, 0xff, tape.count * sizeof(__half)), "poison capacity check");
            fp16.operand_a_capacity = TapeCount(shape, active_rows) - 1;
            PoisonGradientCache(fp16, operand_b, buffers.output_dy,
                                static_cast<std::uint64_t>(active_rows) * output_dim);
            const auto short_status = mgt_cuda::LaunchLocalMlpBatchNormTrainStepFp16(shape, hd1, hd2,
                states.pointer, labels.pointer, active_rows, plan, workspace_count, adam, buffers,
                &fp16, blas.handle, nullptr);
            Cuda(cudaDeviceSynchronize(), "synchronize capacity rejection");
            if (short_status != mgt::Status::kCapacityExceeded) {
                std::fprintf(stderr, "capacity=%llu status=%d expected=kCapacityExceeded required=%llu\n",
                    static_cast<unsigned long long>(fp16.operand_a_capacity), static_cast<int>(short_status),
                    static_cast<unsigned long long>(TapeCount(shape, active_rows)));
                throw std::runtime_error("capacity-1 accepted");
            }
            RequireEmptyGradientCache(fp16, "early tape-capacity rejection");
            Cuda(cudaMemcpy(tape_bits.data(), tape.pointer, tape_bits.size() * sizeof(std::uint16_t),
                            cudaMemcpyDeviceToHost), "read rejected tape");
            if (!std::all_of(tape_bits.begin(), tape_bits.end(), [](std::uint16_t v) { return v == kPoison; }) ||
                before_weights != weights.Read() || before_affine != affine.Read() || before_running != running.Read())
                throw std::runtime_error("capacity rejection changed tape/weights/BN state");

            // A scratch-capacity error must retire stale tags regardless of
            // whether production detects it during preflight or later work.
            fp16.operand_a_capacity = TapeCount(shape, active_rows);
            fp16.operand_b_capacity = active_gradient_capacity - 1;
            PoisonGradientCache(fp16, operand_b, buffers.block_grad,
                                static_cast<std::uint64_t>(active_rows) * hd2);
            const auto short_gradient_status = mgt_cuda::LaunchLocalMlpBatchNormTrainStepFp16(
                shape, hd1, hd2, states.pointer, labels.pointer, active_rows, plan,
                workspace_count, adam, buffers, &fp16, blas.handle, nullptr);
            Cuda(cudaDeviceSynchronize(), "synchronize gradient-capacity rejection");
            if (short_gradient_status == mgt::Status::kOk)
                throw std::runtime_error("gradient scratch capacity-1 accepted");
            RequireEmptyGradientCache(fp16, "gradient scratch-capacity rejection");

            // With valid scratch, invalid Adam epsilon exercises a late failure
            // in the existing optimizer gate. Running state/tape/gradients may
            // already be updated: this test intentionally requires no rollback.
            fp16.operand_b_capacity = active_gradient_capacity;
            PoisonGradientCache(fp16, operand_b, buffers.output_dy,
                                static_cast<std::uint64_t>(active_rows) * output_dim);
            auto invalid_adam = adam;
            invalid_adam.eps = 0.0f;
            const auto late_status = mgt_cuda::LaunchLocalMlpBatchNormTrainStepFp16(
                shape, hd1, hd2, states.pointer, labels.pointer, active_rows, plan,
                workspace_count, invalid_adam, buffers, &fp16, blas.handle, nullptr);
            Cuda(cudaDeviceSynchronize(), "synchronize invalid Adam rejection");
            if (late_status == mgt::Status::kOk)
                throw std::runtime_error("invalid Adam epsilon accepted");
            RequireEmptyGradientCache(fp16, "invalid Adam failure exit");
        }
        if (!equal_width_input_is_distinct)
            throw std::runtime_error("equal-width fixture cannot distinguish input vs hidden gradient mirrors");
        std::fprintf(stdout, "PASS hd1=%u hd2=%u nrd=%u output_dim=%u rows=4,3,4 tape/guards/all-gradients/poison-overwrite/capacity/cache-lifetime/last-dense-mirror\n",
                     hd1, hd2, blocks, output_dim);
        return true;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL hd1=%u hd2=%u nrd=%u output_dim=%u rows=%u: %s\n",
                     hd1, hd2, blocks, output_dim, active_rows, error.what());
        return false;
    }
}
}  // namespace

int main() {
    bool ok = true;
    ok = RunCase(0, 1) && ok;
    ok = RunCase(0, 2) && ok;
    ok = RunCase(2, 1) && ok;
    ok = RunCase(2, 2) && ok;
    // The one-feature hidden path uses FP32 forward but still needs half dW operands.
    ok = RunCase(2, 2, 3, 1) && ok;
    // Equal widths must not make input BN eligible to overwrite the dense cache.
    // Zero residual blocks keep this fixture within the hand-seeded 65 weights.
    ok = RunCase(0, 2, 3, 3) && ok;
    return ok ? 0 : 1;
}
