#include "mgt_cuda/local_mlp_batch_norm.cuh"
#include "../../cuda/column_sum.cuh"
#ifdef MGT_TEST_NCCL
#include "mgt_cuda/allreduce_nccl.cuh"
#endif

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using Values = std::vector<float>;
constexpr unsigned kRows = 4;
constexpr unsigned kLogical1 = 3;
constexpr unsigned kLogical2 = 2;
constexpr std::size_t kGuard = 256;
constexpr unsigned char kGuardByte = 0xa5;
bool cleanup_failed = false;

void Require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

void Cuda(cudaError_t status, const std::string& operation) {
    if (status != cudaSuccess)
        throw std::runtime_error(operation + ": " + cudaGetErrorString(status));
}

template <class T> class Device {
public:
    Device(std::size_t count, const char* name) : count_(count), name_(name) {
        Cuda(cudaMalloc(&allocation_, (count_ + 2 * kGuard) * sizeof(T)), name_ + " allocate");
        try {
            Cuda(cudaMemset(allocation_, kGuardByte, (count_ + 2 * kGuard) * sizeof(T)),
                 name_ + " initialize guards");
            Fill(0);
        } catch (...) {
            Release();
            throw;
        }
    }
    ~Device() { Release(); }
    Device(const Device&) = delete;
    Device& operator=(const Device&) = delete;
    T* get() const { return allocation_ + kGuard; }
    std::size_t size() const { return count_; }
    void Fill(unsigned char byte) {
        Cuda(cudaMemset(get(), byte, count_ * sizeof(T)), name_ + " fill payload");
    }
    void Upload(const std::vector<T>& values) {
        Require(values.size() == count_, name_ + " upload size mismatch");
        Cuda(cudaMemcpy(get(), values.data(), count_ * sizeof(T), cudaMemcpyHostToDevice),
             name_ + " upload");
    }
    std::vector<T> Read() const {
        Guards();
        std::vector<T> values(count_);
        Cuda(cudaMemcpy(values.data(), get(), count_ * sizeof(T), cudaMemcpyDeviceToHost),
             name_ + " download");
        return values;
    }
    void Guards() const {
        unsigned char bytes[2 * kGuard * sizeof(T)];
        Cuda(cudaMemcpy(bytes, allocation_, kGuard * sizeof(T), cudaMemcpyDeviceToHost),
             name_ + " leading guard");
        Cuda(cudaMemcpy(bytes + kGuard * sizeof(T), get() + count_, kGuard * sizeof(T),
                        cudaMemcpyDeviceToHost), name_ + " trailing guard");
        for (unsigned char byte : bytes)
            Require(byte == kGuardByte, name_ + " allocation guard changed");
    }
private:
    void Release() noexcept {
        if (!allocation_) return;
        const auto status = cudaFree(allocation_);
        allocation_ = nullptr;
        if (status != cudaSuccess) {
            cleanup_failed = true;
            std::fprintf(stderr, "%s cudaFree: %s\n", name_.c_str(), cudaGetErrorString(status));
        }
    }
    T* allocation_ = nullptr;
    std::size_t count_;
    std::string name_;
};

struct Blas {
    cublasHandle_t handle = nullptr;
    Blas() { Require(cublasCreate(&handle) == CUBLAS_STATUS_SUCCESS, "cublasCreate"); }
    ~Blas() {
        if (handle && cublasDestroy(handle) != CUBLAS_STATUS_SUCCESS) cleanup_failed = true;
    }
};

#ifdef MGT_TEST_NCCL
struct Nccl {
    mgt_cuda::NcclRankContext* context = nullptr;
    Nccl() {
        int device = 0;
        Cuda(cudaGetDevice(&device), "query NCCL device");
        Require(mgt_cuda::CreateNcclSingleRankContext(static_cast<std::uint32_t>(device), &context)
                    == mgt::Status::kOk, "CreateNcclSingleRankContext");
    }
    ~Nccl() {
        if (context && mgt_cuda::DestroyNcclRankContext(context) != mgt::Status::kOk) {
            cleanup_failed = true;
            std::fprintf(stderr, "DestroyNcclRankContext failed\n");
        }
    }
    Nccl(const Nccl&) = delete;
    Nccl& operator=(const Nccl&) = delete;
};
#endif

std::uint64_t WorkspaceCount(const mgt_cuda::CudaMlpShape& shape,
                             const mgt::BatchNormTrainingPlan& plan) {
#ifdef MGT_TEST_NCCL
    return mgt_cuda::MlpBatchNormForwardWorkspaceFloats(shape, plan, kRows);
#else
    return mgt_cuda::LocalMlpBatchNormForwardWorkspaceFloats(shape, plan, kRows);
#endif
}

// Independent public parameter layout. Padding tests below use only the
// mathematical support of x^T*dY, never production gradient helpers.
struct Layout {
    std::size_t ib, hw, hb, residual, ow, ob, count;
    explicit Layout(const mgt_cuda::CudaMlpShape& s) {
        ib = static_cast<std::size_t>(s.state_len) * s.state_value_pad * s.hd1;
        hw = ib + s.hd1;
        hb = hw + s.hd1 * s.hd2;
        residual = hb + s.hd2;
        ow = residual + s.residual_blocks * 2 * (s.hd2 * s.hd2 + s.hd2);
        ob = ow + s.hd2 * s.output_dim;
        count = ob + s.output_dim;
    }
};

std::uint64_t TapeCount(const mgt_cuda::CudaMlpShape& s, unsigned rows) {
    return static_cast<std::uint64_t>(rows) *
        (s.hd1 + 2ULL * s.residual_blocks * s.hd2 + (s.output_dim > 1 ? s.hd2 : 0));
}

void Close(const char* field, const Values& zero, const Values& poison) {
    Require(zero.size() == poison.size(), std::string(field) + " size mismatch");
    for (std::size_t i = 0; i < zero.size(); ++i) {
        // Vector-head loss and bias gradients contain small unordered atomic
        // reductions. Do not impose bitwise equality on independent executions.
        const float tolerance = 3e-6f + 3e-5f * std::max(std::fabs(zero[i]), std::fabs(poison[i]));
        if (!std::isfinite(zero[i]) || !std::isfinite(poison[i]) ||
            std::fabs(zero[i] - poison[i]) > tolerance) {
            std::fprintf(stderr, "%s[%zu] zero=%.9g poison=%.9g tolerance=%.9g\n",
                         field, i, zero[i], poison[i], tolerance);
            throw std::runtime_error(std::string(field) + " depends on initial gradient storage");
        }
    }
}

void BitExactFinite(const char* field, const Values& zero, const Values& poison) {
    Require(zero.size() == poison.size(), std::string(field) + " size mismatch");
    for (std::size_t i = 0; i < zero.size(); ++i) {
        std::uint32_t a = 0, b = 0;
        std::memcpy(&a, &zero[i], sizeof(a));
        std::memcpy(&b, &poison[i], sizeof(b));
        if (!std::isfinite(zero[i]) || !std::isfinite(poison[i]) || a != b) {
            std::fprintf(stderr, "%s[%zu] zero=%.9g/0x%08x poison=%.9g/0x%08x\n",
                         field, i, zero[i], static_cast<unsigned>(a),
                         poison[i], static_cast<unsigned>(b));
            throw std::runtime_error(std::string(field) + " failed finite bit-exact overwrite");
        }
    }
}

template <class T>
void Unchanged(const char* field, const std::vector<T>& expected, const Device<T>& device) {
    const auto actual = device.Read();  // Also checks both allocation guards.
    Require(actual.size() == expected.size() &&
                std::memcmp(actual.data(), expected.data(), expected.size() * sizeof(T)) == 0,
            std::string(field) + " input bytes changed");
}

void FixedOperandOverwrite(unsigned rows, unsigned in, unsigned out,
                           unsigned logical_in, unsigned logical_out) {
    Values x(static_cast<std::size_t>(rows) * in), dy(static_cast<std::size_t>(rows) * out);
    for (unsigned r = 0; r < rows; ++r) {
        for (unsigned i = 0; i < logical_in; ++i)
            x[static_cast<std::size_t>(r) * in + i] = (1 + (7 * r + 3 * i) % 15) / 16.0f;
        for (unsigned o = 0; o < logical_out; ++o) {
            const int numerator = static_cast<int>((11 * r + 5 * o) % 31) - 15;
            dy[static_cast<std::size_t>(r) * out + o] = (numerator == 0 ? 7 : numerator) / 32.0f;
        }
    }
    // Fixed post-BN operands, including exact zero logical padding. These
    // dyadic values are half-exact, so the small host oracle serves both paths.
    std::vector<__half> xh(x.size()), dyh(dy.size());
    for (std::size_t i = 0; i < x.size(); ++i) xh[i] = __float2half_rn(x[i]);
    for (std::size_t i = 0; i < dy.size(); ++i) dyh[i] = __float2half_rn(dy[i]);
    Device<float> dx(x.size(), "fixed x"), ddy(dy.size(), "fixed dY");
    Device<__half> dxh(xh.size(), "fixed half x"), ddyh(dyh.size(), "fixed half dY");
    Device<float> dw(static_cast<std::size_t>(in) * out, "fixed dW");
    Device<float> bias4(out, "fixed ColumnSum4 bias"), bias8(out, "fixed ColumnSum8 bias");
    dx.Upload(x); ddy.Upload(dy); dxh.Upload(xh); ddyh.Upload(dyh);
    Blas blas;
    struct FixedResult { Values weight, bias4, bias8; };
    auto run = [&](bool mixed, unsigned char initial) {
        dw.Fill(initial); bias4.Fill(initial); bias8.Fill(initial);
        if (mixed) {
            Require(mgt_cuda::LaunchFp16LinearGradWeight(blas.handle, dxh.get(), ddyh.get(),
                        dw.get(), rows, in, out, nullptr) == mgt::Status::kOk,
                    "fixed FP16 GradWeight");
        } else {
            Require(cublasSetStream(blas.handle, nullptr) == CUBLAS_STATUS_SUCCESS,
                    "fixed SGEMM stream");
            const float alpha = 1.0f, beta = 0.0f;
            Require(cublasSgemm(blas.handle, CUBLAS_OP_N, CUBLAS_OP_T, out, in, rows,
                        &alpha, ddy.get(), out, dx.get(), in, &beta, dw.get(), out)
                        == CUBLAS_STATUS_SUCCESS, "fixed SGEMM beta=0");
        }
        mgt_cuda::detail::ColumnSum<4><<<(out + 3U) / 4U, 256>>>(
            ddy.get(), rows, logical_out, out, bias4.get());
        Cuda(cudaGetLastError(), "fixed ColumnSum4 launch");
        mgt_cuda::detail::ColumnSum<8><<<(out + 7U) / 8U, 256>>>(
            ddy.get(), rows, logical_out, out, bias8.get());
        Cuda(cudaGetLastError(), "fixed ColumnSum8 launch");
        Cuda(cudaDeviceSynchronize(), "fixed operand gradients synchronize");
        Unchanged("fixed x", x, dx); Unchanged("fixed dY", dy, ddy);
        Unchanged("fixed half x", xh, dxh); Unchanged("fixed half dY", dyh, ddyh);
        FixedResult result{dw.Read(), bias4.Read(), bias8.Read()};
        for (unsigned i = 0; i < in; ++i) for (unsigned o = 0; o < out; ++o)
            if (i >= logical_in || o >= logical_out)
                Require(result.weight[static_cast<std::size_t>(i) * out + o] == 0.0f,
                        "fixed padded dW is not exactly zero");
        for (unsigned o = logical_out; o < out; ++o)
            Require(result.bias4[o] == 0.0f && result.bias8[o] == 0.0f,
                    "fixed padded bias is not exactly zero");
        return result;
    };
    Values expected_bias(out), expected_weight;
    for (unsigned o = 0; o < out; ++o) {
        double sum = 0;
        for (unsigned r = 0; r < rows; ++r) sum += dy[static_cast<std::size_t>(r) * out + o];
        expected_bias[o] = static_cast<float>(sum);
    }
    if (in <= 8) {
        expected_weight.resize(static_cast<std::size_t>(in) * out);
        for (unsigned i = 0; i < in; ++i) for (unsigned o = 0; o < out; ++o) {
            double sum = 0;
            for (unsigned r = 0; r < rows; ++r)
                sum += static_cast<double>(x[static_cast<std::size_t>(r) * in + i]) *
                       dy[static_cast<std::size_t>(r) * out + o];
            expected_weight[static_cast<std::size_t>(i) * out + o] = static_cast<float>(sum);
        }
    }
    for (bool mixed : {false, true}) {
        // Reuse identical buffers and the same BLAS handle. Only initial C
        // changes: no BN/atomic reduction or address-dependent plan difference.
        const auto zero = run(mixed, 0);
        Close("fixed CPU bias4", expected_bias, zero.bias4);
        Close("fixed CPU bias8", expected_bias, zero.bias8);
        if (!expected_weight.empty()) Close("fixed CPU dW", expected_weight, zero.weight);
        for (unsigned char initial : {static_cast<unsigned char>(0xff), static_cast<unsigned char>(0x5a)}) {
            // 0xffffffff is a quiet NaN; 0x5a5a5a5a is finite and nonzero.
            const auto poison = run(mixed, initial);
            BitExactFinite("fixed dW", zero.weight, poison.weight);
            BitExactFinite("fixed ColumnSum4", zero.bias4, poison.bias4);
            BitExactFinite("fixed ColumnSum8", zero.bias8, poison.bias8);
        }
        std::printf("PASS fixed-overwrite mode=%s rows=%u physical=%u/%u logical=%u/%u "
                    "qNaN+finite bit-exact dW/ColumnSum4/ColumnSum8/guards/input-bytes\n",
                    mixed ? "fp16" : "fp32", rows, in, out, logical_in, logical_out);
    }
}

struct Snapshot {
    Values weights, weight_grad, weight_m, weight_v, affine, affine_grad, affine_m, affine_v;
    Values running, outputs, loss;
};

class Model {
public:
    Model(mgt_cuda::CudaMlpShape shape, const mgt::BatchNormTrainingPlan& plan, bool fp16)
        : shape_(shape), plan_(plan), mixed_(fp16), layout_(shape),
          workspace_count_(WorkspaceCount(shape, plan)),
          weights_(layout_.count, "weights"), wg_(layout_.count, "weight_grad"),
          wm_(layout_.count, "weight_m"), wv_(layout_.count, "weight_v"),
          affine_(plan.trainable_count, "affine"), ag_(plan.trainable_count, "affine_grad"),
          am_(plan.trainable_count, "affine_m"), av_(plan.trainable_count, "affine_v"),
          running_(plan.running_count, "running"), outputs_(kRows * shape.output_dim, "outputs"),
          workspace_(workspace_count_, "workspace"), loss_(1, "loss"),
          dy_(kRows * shape.output_dim, "output_dy"), block_(kRows * shape.hd2, "block_grad"),
          fc1_(kRows * shape.hd2, "fc1_grad"), residual_(kRows * shape.hd2, "residual_grad"),
          input_(kRows * shape.hd1, "input_grad"), labels_(kRows * shape.output_dim, "labels"),
          states_(kRows, "states"), mirror_(layout_.count, "weight_half"),
          tape_(TapeCount(shape, kRows), "activation_tape"),
          operand_b_(kRows * std::max(shape.hd2, shape.output_dim), "operand_b") {
#ifdef MGT_TEST_NCCL
        Require(!mixed_, "NCCL world-size-one harness supports FP32 only");
#endif
        Values weights(layout_.count), wm(layout_.count), wv(layout_.count);
        for (std::size_t i = 0; i < weights.size(); ++i) {
            const int numerator = static_cast<int>((37 * i + 11) % 101) - 50;
            weights[i] = static_cast<float>(numerator == 0 ? 7 : numerator) / 64.0f;
            wm[i] = (i % 2 ? -1.0f : 1.0f) * (1 + i % 7) / 2048.0f;
            wv[i] = (1 + i % 11) / 1024.0f;
        }
        auto bn = mgt::InitializeBatchNormTrainingState(plan);
        for (std::size_t i = 0; i < plan.logical_feature_count; ++i) {
            bn.affine[i] = .75f + .125f * (i % 3);
            bn.affine[plan.logical_feature_count + i] = .125f * (static_cast<int>(i % 3) - 1);
        }
        Values am(bn.affine.size()), av(bn.affine.size());
        for (std::size_t i = 0; i < am.size(); ++i) {
            am[i] = (i % 2 ? -.001f : .001f) * (1 + i % 5);
            av[i] = .002f * (1 + i % 7);
        }
        weights_.Upload(weights); wm_.Upload(wm); wv_.Upload(wv);
        affine_.Upload(bn.affine); am_.Upload(am); av_.Upload(av); running_.Upload(bn.running);
        Require(mgt_cuda::LaunchFloatToHalf(weights_.get(), mirror_.get(), layout_.count, nullptr)
                    == mgt::Status::kOk, "initialize weight mirror");
        fp16_ = {weights_.get(), mirror_.get(), tape_.get(), operand_b_.get(),
                 TapeCount(shape, kRows), operand_b_.size()};
    }

    Snapshot Step(unsigned rows, unsigned step, bool poison,
                  const std::vector<mgt::TrainStateStorage>& states, const Values& labels) {
        states_.Upload(states);
        labels_.Upload(labels);
        // NaNs on alternating steps catch beta=1/partial overwrite. A finite
        // nonzero sentinel catches missing writes without relying only on NaNs.
        const unsigned char byte = poison ? (step % 2 == 0 ? 0xff : 0x5a) : 0;
        wg_.Fill(byte); ag_.Fill(byte); loss_.Fill(byte);
        const mgt_cuda::MlpBatchNormStepBuffers buffers{
            weights_.get(), wg_.get(), wm_.get(), wv_.get(), affine_.get(), ag_.get(), am_.get(), av_.get(),
            running_.get(), outputs_.get(), workspace_.get(), loss_.get(), dy_.get(), block_.get(),
            fc1_.get(), residual_.get(), input_.get()};
        const mgt_cuda::AdamWKernelConfig adam{0, step + 1ULL, .0001f, .9f, .999f, 1e-8f, .01f};
        fp16_.operand_a_capacity = TapeCount(shape_, rows);
#ifdef MGT_TEST_NCCL
        // Exercise the public nonlocal compilation with local_rows == global_rows.
        const auto status = mgt_cuda::LaunchMlpBatchNormTrainStep(shape_, kLogical1, kLogical2,
            states_.get(), labels_.get(), rows, rows, plan_, workspace_count_, adam, buffers,
            nccl_.context, blas_.handle, nullptr);
#else
        const auto status = mixed_
            ? mgt_cuda::LaunchLocalMlpBatchNormTrainStepFp16(shape_, kLogical1, kLogical2,
                states_.get(), labels_.get(), rows, plan_, workspace_count_, adam, buffers,
                &fp16_, blas_.handle, nullptr)
            : mgt_cuda::LaunchLocalMlpBatchNormTrainStep(shape_, kLogical1, kLogical2,
                states_.get(), labels_.get(), rows, plan_, workspace_count_, adam, buffers,
                blas_.handle, nullptr);
#endif
        Require(status == mgt::Status::kOk, "training step status=" + std::to_string(static_cast<int>(status)));
        Cuda(cudaDeviceSynchronize(), "synchronize full training step");
        workspace_.Guards(); dy_.Guards(); block_.Guards(); fc1_.Guards(); residual_.Guards();
        input_.Guards(); tape_.Guards(); operand_b_.Guards(); mirror_.Guards();
        const auto after_states = states_.Read();
        Require(std::memcmp(after_states.data(), states.data(), states.size() * sizeof(states[0])) == 0,
                "training mutated input states");
        Require(labels_.Read() == labels, "training mutated labels");
        Snapshot snapshot{weights_.Read(), wg_.Read(), wm_.Read(), wv_.Read(), affine_.Read(), ag_.Read(),
                          am_.Read(), av_.Read(), running_.Read(), outputs_.Read(), loss_.Read()};
        if (mixed_) {
            const auto half = mirror_.Read();
            for (std::size_t i = 0; i < half.size(); ++i) {
                const __half expected = __float2half_rn(snapshot.weights[i]);
                Require(std::memcmp(&half[i], &expected, sizeof(expected)) == 0,
                        "updated weight mirror does not match master");
            }
        }
        return snapshot;
    }
private:
    mgt_cuda::CudaMlpShape shape_;
    mgt::BatchNormTrainingPlan plan_;
    bool mixed_;
    Layout layout_;
    std::uint64_t workspace_count_;
#ifdef MGT_TEST_NCCL
    Nccl nccl_;
#endif
    Device<float> weights_, wg_, wm_, wv_, affine_, ag_, am_, av_, running_, outputs_, workspace_, loss_;
    Device<float> dy_, block_, fc1_, residual_, input_, labels_;
    Device<mgt::TrainStateStorage> states_;
    Device<__half> mirror_, tape_, operand_b_;
    mgt_cuda::LocalMlpFp16Context fp16_{};
    Blas blas_;
};

void PaddingZeros(const mgt_cuda::CudaMlpShape& s, const Values& grad) {
    const Layout layout(s);
    auto zero = [&](std::size_t index) {
        Require(std::isfinite(grad[index]) && grad[index] == 0.0f,
                "padded weight gradient must be exactly zero: index=" + std::to_string(index));
    };
    for (std::size_t bin = 0; bin < static_cast<std::size_t>(s.state_len) * s.state_value_pad; ++bin)
        for (unsigned h = kLogical1; h < s.hd1; ++h) zero(bin * s.hd1 + h);
    for (unsigned h = kLogical1; h < s.hd1; ++h) zero(layout.ib + h);
    auto dense = [&](std::size_t offset, unsigned in, unsigned out, unsigned live_in, unsigned live_out) {
        for (unsigned i = 0; i < in; ++i) for (unsigned o = 0; o < out; ++o)
            if (i >= live_in || o >= live_out) zero(offset + static_cast<std::size_t>(i) * out + o);
    };
    // BN/ReLU makes padded activations and post-BN dY exactly zero. Nonzero
    // padded weights/momenta are legal and must NOT themselves be zeroed.
    dense(layout.hw, s.hd1, s.hd2, kLogical1, kLogical2);
    for (unsigned h = kLogical2; h < s.hd2; ++h) zero(layout.hb + h);
    for (unsigned b = 0; b < s.residual_blocks; ++b) for (unsigned fc = 0; fc < 2; ++fc) {
        const std::size_t offset = layout.residual + (2 * b + fc) * (s.hd2 * s.hd2 + s.hd2);
        dense(offset, s.hd2, s.hd2, kLogical2, kLogical2);
        for (unsigned h = kLogical2; h < s.hd2; ++h) zero(offset + s.hd2 * s.hd2 + h);
    }
    dense(layout.ow, s.hd2, s.output_dim, kLogical2, s.output_dim);
}

void CheckLoss(const Snapshot& snapshot, const Values& labels, unsigned rows, unsigned outputs) {
    double sum = 0;
    for (unsigned i = 0; i < rows * outputs; ++i) {
        const double error = static_cast<double>(snapshot.outputs[i]) - labels[i];
        sum += error * error;
    }
    Close("independent MSE", {static_cast<float>(sum / (rows * outputs))}, snapshot.loss);
}

void RunCase(bool mixed, unsigned hd1, unsigned hd2, unsigned output_dim) {
    const mgt_cuda::CudaMlpShape shape{2, 4, hd1, hd2, 1, output_dim};
    mgt::BatchNormTrainingPlan plan;
    Require(mgt::BuildBatchNormTrainingPlan(kLogical1, kLogical2, hd1, hd2, 1, kRows, &plan)
                == mgt::Status::kOk, "BuildBatchNormTrainingPlan");
    Model zero(shape, plan, mixed), poison(shape, plan, mixed);
    const unsigned rows[] = {4, 2, 1, 4};
    const unsigned pairs[kRows][2] = {{0, 1}, {1, 3}, {2, 0}, {3, 2}};
    for (unsigned step = 0; step < 4; ++step) {
        std::vector<mgt::TrainStateStorage> states(kRows);
        Values labels(kRows * output_dim);
        for (unsigned row = 0; row < kRows; ++row) {
            for (unsigned p = 0; p < 2; ++p)
                states[row].v[p] = static_cast<mgt::StateValue>(pairs[(row + step) % kRows][p]);
            for (unsigned o = 0; o < output_dim; ++o)
                labels[row * output_dim + o] = .5f * (static_cast<int>((row * 3 + step) % 7) - 3) + .25f * o;
        }
        const auto a = zero.Step(rows[step], step, false, states, labels);
        const auto b = poison.Step(rows[step], step, true, states, labels);
        Close("weight_grad", a.weight_grad, b.weight_grad);
        Close("affine_grad", a.affine_grad, b.affine_grad);
        Close("loss", a.loss, b.loss);
        Close("weights", a.weights, b.weights);
        Close("weight_m", a.weight_m, b.weight_m);
        Close("weight_v", a.weight_v, b.weight_v);
        Close("affine", a.affine, b.affine);
        Close("affine_m", a.affine_m, b.affine_m);
        Close("affine_v", a.affine_v, b.affine_v);
        Close("running", a.running, b.running);
        Close("outputs", a.outputs, b.outputs);
        PaddingZeros(shape, a.weight_grad); PaddingZeros(shape, b.weight_grad);
        CheckLoss(a, labels, rows[step], output_dim); CheckLoss(b, labels, rows[step], output_dim);
#ifdef MGT_TEST_NCCL
        const char* mode = "nccl-world1-fp32";
#else
        const char* mode = mixed ? "local-fp16" : "local-fp32";
#endif
        std::printf("PASS gradient-overwrite mode=%s physical=%u/%u logical=3/2 output=%u rows=%u step=%u poison=%s\n",
                    mode, hd1, hd2, output_dim, rows[step], step + 1,
                    step % 2 == 0 ? "NaN" : "finite-sentinel");
    }
}
}  // namespace

int main() {
    try {
        // Same fixed-operand checks in local and NCCL binaries; no BN or
        // collectives participate in this direct overwrite-contract test.
        FixedOperandOverwrite(17, 5, 3, 3, 2);
        FixedOperandOverwrite(257, 8, 4, 3, 2);
        FixedOperandOverwrite(257, 224, 224, 219, 221);
        std::printf("PASS fixed-overwrite: 3 fixtures, 6 precision variants, 12 poison comparisons\n");
        // Paired executions characterize independence, not numerical truth.
        // The separate activation-tape test retains its independent CPU oracle.
#ifdef MGT_TEST_NCCL
        constexpr bool modes[] = {false};
#else
        constexpr bool modes[] = {false, true};
#endif
        for (bool mixed : modes) for (unsigned outputs : {1U, 2U}) {
            RunCase(mixed, 5, 3, outputs);  // Odd physical shape / scalar gather fallback.
            RunCase(mixed, 8, 4, outputs);  // Aligned dense GEMM with logical padding.
        }
        Require(!cleanup_failed, "CUDA/BLAS/NCCL resource cleanup failed");
#ifdef MGT_TEST_NCCL
        std::printf("PASS gradient-overwrite NCCL world1: 4 paired configurations, 16 changed-row steps, full physical gradients/Adam/guards\n");
#else
        std::printf("PASS gradient-overwrite: 8 paired configurations, 32 changed-row steps, full physical gradients/Adam/guards\n");
#endif
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL gradient-overwrite: %s\n", error.what());
        return EXIT_FAILURE;
    }
}
