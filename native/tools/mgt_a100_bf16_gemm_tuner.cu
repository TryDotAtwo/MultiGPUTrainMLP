#include "mgt_cuda/bf16_linear_train_ops.cuh"

#include <cublasLt.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {
struct Descriptors {
    cublasLtMatmulDesc_t operation = nullptr;
    cublasLtMatrixLayout_t a = nullptr;
    cublasLtMatrixLayout_t b = nullptr;
    cublasLtMatrixLayout_t c = nullptr;
};

void Destroy(Descriptors* d) {
    if (d->c) cublasLtMatrixLayoutDestroy(d->c);
    if (d->b) cublasLtMatrixLayoutDestroy(d->b);
    if (d->a) cublasLtMatrixLayoutDestroy(d->a);
    if (d->operation) cublasLtMatmulDescDestroy(d->operation);
}

bool Create(const mgt::Bf16GemmKeyV1& key, Descriptors* d) {
    const auto op_a = static_cast<cublasOperation_t>(key.op_a);
    const auto op_b = static_cast<cublasOperation_t>(key.op_b);
    const auto order = CUBLASLT_ORDER_ROW;
    const std::uint64_t ar = op_a == CUBLAS_OP_N ? key.m : key.k;
    const std::uint64_t ac = op_a == CUBLAS_OP_N ? key.k : key.m;
    const std::uint64_t br = op_b == CUBLAS_OP_N ? key.k : key.n;
    const std::uint64_t bc = op_b == CUBLAS_OP_N ? key.n : key.k;
    cublasStatus_t s = cublasLtMatmulDescCreate(&d->operation, CUBLAS_COMPUTE_32F, CUDA_R_32F);
    if (s == CUBLAS_STATUS_SUCCESS) s = cublasLtMatmulDescSetAttribute(d->operation, CUBLASLT_MATMUL_DESC_TRANSA, &op_a, sizeof(op_a));
    if (s == CUBLAS_STATUS_SUCCESS) s = cublasLtMatmulDescSetAttribute(d->operation, CUBLASLT_MATMUL_DESC_TRANSB, &op_b, sizeof(op_b));
    if (s == CUBLAS_STATUS_SUCCESS) s = cublasLtMatrixLayoutCreate(&d->a, CUDA_R_16BF, ar, ac, key.lda);
    if (s == CUBLAS_STATUS_SUCCESS) s = cublasLtMatrixLayoutSetAttribute(d->a, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order));
    if (s == CUBLAS_STATUS_SUCCESS) s = cublasLtMatrixLayoutCreate(&d->b, CUDA_R_16BF, br, bc, key.ldb);
    if (s == CUBLAS_STATUS_SUCCESS) s = cublasLtMatrixLayoutSetAttribute(d->b, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order));
    if (s == CUBLAS_STATUS_SUCCESS) s = cublasLtMatrixLayoutCreate(&d->c, CUDA_R_32F, key.m, key.n, key.ldc);
    if (s == CUBLAS_STATUS_SUCCESS) s = cublasLtMatrixLayoutSetAttribute(d->c, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order));
    return s == CUBLAS_STATUS_SUCCESS;
}

template <typename T>
bool Get(const cublasLtMatmulAlgo_t& algo, cublasLtMatmulAlgoConfigAttributes_t attr, T* value) {
    std::size_t written = 0;
    return cublasLtMatmulAlgoConfigGetAttribute(&algo, attr, value, sizeof(*value), &written) == CUBLAS_STATUS_SUCCESS && written == sizeof(*value);
}

struct Result {
    int index = -1;
    float median_ms = 0.0f;
    float min_ms = 0.0f;
};

Result TimeCandidate(cublasLtHandle_t handle, const Descriptors& d,
                     const cublasLtMatmulHeuristicResult_t& candidate,
                     const __nv_bfloat16* a, const __nv_bfloat16* b, float* c,
                     void* workspace, cudaStream_t stream, std::uint32_t warmups,
                     std::uint32_t repeats) {
    const float alpha = 1.0f, beta = 0.0f;
    for (std::uint32_t i = 0; i < warmups; ++i) {
        if (cublasLtMatmul(handle, d.operation, &alpha, a, d.a, b, d.b, &beta,
                           c, d.c, c, d.c, &candidate.algo, workspace,
                           candidate.workspaceSize, stream) != CUBLAS_STATUS_SUCCESS) return {};
    }
    std::vector<float> samples;
    samples.reserve(repeats);
    cudaEvent_t start = nullptr, stop = nullptr;
    if (cudaEventCreate(&start) != cudaSuccess || cudaEventCreate(&stop) != cudaSuccess) return {};
    for (std::uint32_t i = 0; i < repeats; ++i) {
        if (cudaEventRecord(start, stream) != cudaSuccess ||
            cublasLtMatmul(handle, d.operation, &alpha, a, d.a, b, d.b, &beta,
                           c, d.c, c, d.c, &candidate.algo, workspace,
                           candidate.workspaceSize, stream) != CUBLAS_STATUS_SUCCESS ||
            cudaEventRecord(stop, stream) != cudaSuccess ||
            cudaEventSynchronize(stop) != cudaSuccess) {
            samples.clear();
            break;
        }
        float ms = 0.0f;
        if (cudaEventElapsedTime(&ms, start, stop) != cudaSuccess) { samples.clear(); break; }
        samples.push_back(ms);
    }
    cudaEventDestroy(stop);
    cudaEventDestroy(start);
    if (samples.empty()) return {};
    std::sort(samples.begin(), samples.end());
    return {0, samples[samples.size() / 2], samples.front()};
}

const char* RoleName(mgt::Bf16GemmRole role) {
    if (role == mgt::Bf16GemmRole::kGradWeight) return "dw";
    if (role == mgt::Bf16GemmRole::kGradInput || role == mgt::Bf16GemmRole::kInputTableGrad) return "dx";
    return "forward";
}

bool Tune(cublasLtHandle_t handle, cudaStream_t stream, void* workspace,
          std::uint64_t workspace_bytes, const mgt_cuda::Bf16LinearProblem& problem,
          mgt::Bf16GemmRole role, std::uint32_t warmups, std::uint32_t repeats) {
    mgt::Bf16GemmKeyV1 key{};
    if (mgt_cuda::BuildBf16LinearGemmKey(problem, role, 0.0f, &key) != mgt::Status::kOk) return false;
    Descriptors d{};
    if (!Create(key, &d)) return false;
    cublasLtMatmulPreference_t preference = nullptr;
    if (cublasLtMatmulPreferenceCreate(&preference) != CUBLAS_STATUS_SUCCESS) { Destroy(&d); return false; }
    const std::size_t capacity = workspace_bytes;
    cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &capacity, sizeof(capacity));
    constexpr int kMaxCandidates = 64;
    cublasLtMatmulHeuristicResult_t candidates[kMaxCandidates]{};
    int count = 0;
    const auto status = cublasLtMatmulAlgoGetHeuristic(handle, d.operation, d.a, d.b, d.c, d.c,
                                                        preference, kMaxCandidates, candidates, &count);
    const std::size_t a_count = static_cast<std::size_t>(key.op_a == CUBLAS_OP_N ? key.m : key.k) * key.lda;
    const std::size_t b_count = static_cast<std::size_t>(key.op_b == CUBLAS_OP_N ? key.k : key.n) * key.ldb;
    const std::size_t c_count = static_cast<std::size_t>(key.m) * key.ldc;
    __nv_bfloat16 *a = nullptr, *b = nullptr;
    float* c = nullptr;
    bool ok = status == CUBLAS_STATUS_SUCCESS && count > 0 &&
              cudaMalloc(&a, a_count * sizeof(*a)) == cudaSuccess &&
              cudaMalloc(&b, b_count * sizeof(*b)) == cudaSuccess &&
              cudaMalloc(&c, c_count * sizeof(*c)) == cudaSuccess;
    if (ok) {
        cudaMemsetAsync(a, 0x3c, a_count * sizeof(*a), stream);
        cudaMemsetAsync(b, 0x3d, b_count * sizeof(*b), stream);
        cudaMemsetAsync(c, 0, c_count * sizeof(*c), stream);
    }
    static bool device_warmed = false;
    if (ok && !device_warmed) {
        const float alpha = 1.0f, beta = 0.0f;
        constexpr std::uint32_t kBurnInIterations = 12000;
        for (std::uint32_t i = 0; i < kBurnInIterations; ++i) {
            if (cublasLtMatmul(handle, d.operation, &alpha, a, d.a, b, d.b, &beta,
                               c, d.c, c, d.c, &candidates[0].algo, workspace,
                               candidates[0].workspaceSize, stream) != CUBLAS_STATUS_SUCCESS) {
                ok = false;
                break;
            }
        }
        if (ok && cudaStreamSynchronize(stream) != cudaSuccess) ok = false;
        device_warmed = ok;
        std::fprintf(stderr, "bf16_tuner_burn_in iterations=%u status=%s\n",
                     kBurnInIterations, ok ? "ok" : "failed");
    }
    Result best{};
    best.median_ms = 1.0e30f;
    int valid = 0;
    for (int i = 0; ok && i < count; ++i) {
        if (candidates[i].state != CUBLAS_STATUS_SUCCESS || candidates[i].workspaceSize > capacity) continue;
        Result measured = TimeCandidate(handle, d, candidates[i], a, b, c, workspace, stream, warmups, repeats);
        if (measured.median_ms <= 0.0f) continue;
        ++valid;
        measured.index = i;
        if (measured.median_ms < best.median_ms) best = measured;
    }
    if (valid == 0 || best.index < 0) ok = false;
    if (ok) {
        const auto& winner = candidates[best.index];
        int algo = -1;
        std::uint32_t tile = 0, stages = 0, split_k = 0, reduction = 0, swizzle = 0, custom = 0;
        ok = Get(winner.algo, CUBLASLT_ALGO_CONFIG_ID, &algo) &&
             Get(winner.algo, CUBLASLT_ALGO_CONFIG_TILE_ID, &tile) &&
             Get(winner.algo, CUBLASLT_ALGO_CONFIG_STAGES_ID, &stages) &&
             Get(winner.algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &split_k) &&
             Get(winner.algo, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, &reduction) &&
             Get(winner.algo, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING, &swizzle) &&
             Get(winner.algo, CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION, &custom);
        const double tflops = 2.0 * static_cast<double>(key.m) * key.n * key.k / (best.median_ms * 1.0e9);
        std::printf("{\"rows\":%u,\"input\":%u,\"output\":%u,\"role\":\"%s\",\"m\":%llu,\"n\":%llu,\"k\":%llu,\"candidates\":%d,\"valid\":%d,\"algo\":%d,\"tile\":%u,\"stages\":%u,\"split_k\":%u,\"reduction\":%u,\"swizzle\":%u,\"custom\":%u,\"workspace\":%llu,\"median_ms\":%.6f,\"min_ms\":%.6f,\"tflops\":%.3f}\n",
                    problem.compute_rows, problem.input_features, problem.output_features, RoleName(role),
                    static_cast<unsigned long long>(key.m), static_cast<unsigned long long>(key.n),
                    static_cast<unsigned long long>(key.k), count, valid, algo, tile, stages, split_k,
                    reduction, swizzle, custom, static_cast<unsigned long long>(winner.workspaceSize),
                    best.median_ms, best.min_ms, tflops);
    }
    if (c) cudaFree(c);
    if (b) cudaFree(b);
    if (a) cudaFree(a);
    cublasLtMatmulPreferenceDestroy(preference);
    Destroy(&d);
    return ok;
}
}

int main(int argc, char** argv) {
    const std::uint32_t warmups = argc > 1 ? static_cast<std::uint32_t>(std::strtoul(argv[1], nullptr, 10)) : 5;
    const std::uint32_t repeats = argc > 2 ? static_cast<std::uint32_t>(std::strtoul(argv[2], nullptr, 10)) : 31;
    if (warmups == 0 || repeats < 3 || cudaSetDevice(0) != cudaSuccess) return 2;
    cublasLtHandle_t handle = nullptr;
    cudaStream_t stream = nullptr;
    constexpr std::uint64_t workspace_bytes = 256ULL << 20;
    void* workspace = nullptr;
    if (cublasLtCreate(&handle) != CUBLAS_STATUS_SUCCESS ||
        cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) != cudaSuccess ||
        cudaMalloc(&workspace, workspace_bytes) != cudaSuccess) return 3;
    const std::uint32_t rows[] = {12500, 12498, 12497};
    bool ok = true;
    std::uint32_t site = 1;
    for (std::uint32_t r : rows) {
        const mgt_cuda::Bf16LinearProblem input{r, site++, r, 2560, 224};
        const mgt_cuda::Bf16LinearProblem hidden{r, site++, r, 224, 224};
        ok = Tune(handle, stream, workspace, workspace_bytes, input, mgt::Bf16GemmRole::kInputForward, warmups, repeats) && ok;
        ok = Tune(handle, stream, workspace, workspace_bytes, hidden, mgt::Bf16GemmRole::kHiddenForward, warmups, repeats) && ok;
        ok = Tune(handle, stream, workspace, workspace_bytes, input, mgt::Bf16GemmRole::kGradWeight, warmups, repeats) && ok;
        ok = Tune(handle, stream, workspace, workspace_bytes, hidden, mgt::Bf16GemmRole::kGradWeight, warmups, repeats) && ok;
        ok = Tune(handle, stream, workspace, workspace_bytes, input, mgt::Bf16GemmRole::kInputTableGrad, warmups, repeats) && ok;
        ok = Tune(handle, stream, workspace, workspace_bytes, hidden, mgt::Bf16GemmRole::kGradInput, warmups, repeats) && ok;
    }
    cudaFree(workspace);
    cudaStreamDestroy(stream);
    cublasLtDestroy(handle);
    return ok ? 0 : 4;
}