#include "mgt_cuda/mlp_backward.cuh"
#include "mgt_cuda/device_context.cuh"
#include <cublas_v2.h>
#include <cublasLt.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cstddef>
#include <limits>
#ifdef MGT_HAS_CUTLASS_HALF_GEMM
#include <cutlass/arch/arch.h>
#include <cutlass/cutlass.h>
#include <cutlass/epilogue/thread/linear_combination.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/gemm/threadblock/threadblock_swizzle.h>
#include <cutlass/layout/matrix.h>
#include <cutlass/numeric_types.h>
#endif

namespace mgt_cuda {
namespace {

__device__ float Relu(float x) { return x > 0.0f ? x : 0.0f; }
__device__ float ReluGradFromActivation(float x) { return x > 0.0f ? 1.0f : 0.0f; }
__device__ float ReluGradFromPreactivation(float x) { return x > 0.0f ? 1.0f : 0.0f; }

__host__ __device__ std::uint64_t ResidualBlockParams(CudaMlpShape shape) { return 2ULL * (static_cast<std::uint64_t>(shape.hd2) * shape.hd2 + shape.hd2); }
__host__ __device__ std::uint64_t InputBias(CudaMlpShape shape) { return static_cast<std::uint64_t>(shape.state_len) * shape.state_value_pad * shape.hd1; }
__host__ __device__ std::uint64_t HiddenWeight(CudaMlpShape shape) { return InputBias(shape) + shape.hd1; }
__host__ __device__ std::uint64_t HiddenBias(CudaMlpShape shape) { return HiddenWeight(shape) + static_cast<std::uint64_t>(shape.hd1) * shape.hd2; }
__host__ __device__ std::uint64_t ResidualBase(CudaMlpShape shape) { return HiddenBias(shape) + shape.hd2; }
__host__ __device__ std::uint64_t OutputWeight(CudaMlpShape shape) { return ResidualBase(shape) + static_cast<std::uint64_t>(shape.residual_blocks) * ResidualBlockParams(shape); }
__host__ __device__ std::uint64_t OutputBias(CudaMlpShape shape) { return OutputWeight(shape) + static_cast<std::uint64_t>(shape.hd2) * shape.output_dim; }
__host__ __device__ std::uint64_t ParamCount(CudaMlpShape shape) { return OutputBias(shape) + shape.output_dim; }
__host__ __device__ std::uint64_t ResidualFc1Weight(CudaMlpShape shape, std::uint32_t block) { return ResidualBase(shape) + static_cast<std::uint64_t>(block) * ResidualBlockParams(shape); }
__host__ __device__ std::uint64_t ResidualFc1Bias(CudaMlpShape shape, std::uint32_t block) { return ResidualFc1Weight(shape, block) + static_cast<std::uint64_t>(shape.hd2) * shape.hd2; }
__host__ __device__ std::uint64_t ResidualFc2Weight(CudaMlpShape shape, std::uint32_t block) { return ResidualFc1Bias(shape, block) + shape.hd2; }
__host__ __device__ std::uint64_t ResidualFc2Bias(CudaMlpShape shape, std::uint32_t block) { return ResidualFc2Weight(shape, block) + static_cast<std::uint64_t>(shape.hd2) * shape.hd2; }

std::uint64_t HalfStorageFloats(std::uint64_t half_count);

inline constexpr std::uint64_t kHalfWorkspaceAlignmentFloats = 8ULL;

std::uint64_t AlignUp(std::uint64_t value, std::uint64_t alignment) {
    return ((value + alignment - 1ULL) / alignment) * alignment;
}

std::uint64_t AddAlignedHalfStorage(std::uint64_t offset, std::uint64_t half_count) {
    return AlignUp(offset, kHalfWorkspaceAlignmentFloats) + HalfStorageFloats(half_count);
}

float* AlignHalfWorkspaceCursor(float* root, float* cursor) {
    const std::uint64_t offset = static_cast<std::uint64_t>(cursor - root);
    return root + AlignUp(offset, kHalfWorkspaceAlignmentFloats);
}

struct Workspace {
    float* a1;
    float* z2;
    float* block_inputs;
    float* rz1;
    float* rz2;
    float* output;
    float* dy;
    float* loss_terms;
    float* ones;
    float* dcur;
    float* dprev;
    float* dz2;
    float* da1;
    float* dz1;
    float* dzfc2;
    float* dra1;
    float* dzfc1;
    float* input_one_hot;
    __half* dz1_half;
    __half* weights_half;
    __half* a1_half;
    __half* block_inputs_half;
    __half* ra1_half;
    __half* h2_right_half;
    __half* output_half;
    float* input_grad_partials;
};

std::uint64_t WorkspaceFloats(CudaMlpShape shape, std::uint32_t samples, std::uint32_t input_grad_partial_chunks, bool use_half_input_grad, bool use_half_linear) {
    const std::uint64_t b = samples;
    const std::uint64_t h1 = shape.hd1;
    const std::uint64_t h2 = shape.hd2;
    const std::uint64_t out = shape.output_dim;
    const std::uint64_t r = shape.residual_blocks;
    const std::uint64_t partial_chunks = input_grad_partial_chunks <= 1 ? 0ULL : input_grad_partial_chunks;
    std::uint64_t offset = 0;
    offset += b * h1;
    offset += b * h2;
    offset += b * (r + 1ULL) * h2;
    offset += b * r * h2;
    offset += b * r * h2;
    offset += b * out;
    offset += b * out;
    offset += b * out;
    offset += b;
    offset += b * h2;
    offset += b * h2;
    offset += b * h2;
    offset += b * h1;
    offset += b * h1;
    offset += b * h2;
    offset += b * h2;
    offset += b * h2;
    offset += b * shape.state_len * shape.state_value_pad;
    if (use_half_input_grad) offset = AddAlignedHalfStorage(offset, b * h1);
    if (use_half_linear) {
        offset = AddAlignedHalfStorage(offset, ParamCount(shape));
        offset = AddAlignedHalfStorage(offset, b * h1);
        offset = AddAlignedHalfStorage(offset, b * (r + 1ULL) * h2);
        offset = AddAlignedHalfStorage(offset, b * r * h2);
        offset = AddAlignedHalfStorage(offset, b * h2);
        offset = AddAlignedHalfStorage(offset, b * out);
    }
    offset += partial_chunks * shape.state_len * shape.state_value_pad * h1;
    return offset;
}

Workspace MakeWorkspace(float* base, CudaMlpShape shape, std::uint32_t samples, std::uint32_t input_grad_partial_chunks, bool use_half_input_grad, bool use_half_linear) {
    Workspace w{};
    float* root = base;
    const std::uint64_t b = samples;
    const std::uint64_t h1 = shape.hd1;
    const std::uint64_t h2 = shape.hd2;
    const std::uint64_t out = shape.output_dim;
    const std::uint64_t r = shape.residual_blocks;
    const std::uint64_t partial_chunks = input_grad_partial_chunks <= 1 ? 0ULL : input_grad_partial_chunks;
    w.a1 = base; base += b * h1;
    w.z2 = base; base += b * h2;
    w.block_inputs = base; base += b * (r + 1ULL) * h2;
    w.rz1 = base; base += b * r * h2;
    w.rz2 = base; base += b * r * h2;
    w.output = base; base += b * out;
    w.dy = base; base += b * out;
    w.loss_terms = base; base += b * out;
    w.ones = base; base += b;
    w.dcur = base; base += b * h2;
    w.dprev = base; base += b * h2;
    w.dz2 = base; base += b * h2;
    w.da1 = base; base += b * h1;
    w.dz1 = base; base += b * h1;
    w.dzfc2 = base; base += b * h2;
    w.dra1 = base; base += b * h2;
    w.dzfc1 = base; base += b * h2;
    w.input_one_hot = base; base += b * shape.state_len * shape.state_value_pad;
    if (use_half_input_grad) {
        base = AlignHalfWorkspaceCursor(root, base);
        w.dz1_half = reinterpret_cast<__half*>(base); base += HalfStorageFloats(b * h1);
    }
    if (use_half_linear) {
        base = AlignHalfWorkspaceCursor(root, base);
        w.weights_half = reinterpret_cast<__half*>(base); base += HalfStorageFloats(ParamCount(shape));
        base = AlignHalfWorkspaceCursor(root, base);
        w.a1_half = reinterpret_cast<__half*>(base); base += HalfStorageFloats(b * h1);
        base = AlignHalfWorkspaceCursor(root, base);
        w.block_inputs_half = reinterpret_cast<__half*>(base); base += HalfStorageFloats(b * (r + 1ULL) * h2);
        base = AlignHalfWorkspaceCursor(root, base);
        w.ra1_half = reinterpret_cast<__half*>(base); base += HalfStorageFloats(b * r * h2);
        base = AlignHalfWorkspaceCursor(root, base);
        w.h2_right_half = reinterpret_cast<__half*>(base); base += HalfStorageFloats(b * h2);
        base = AlignHalfWorkspaceCursor(root, base);
        w.output_half = reinterpret_cast<__half*>(base); base += HalfStorageFloats(b * out);
    }
    w.input_grad_partials = base; base += partial_chunks * shape.state_len * shape.state_value_pad * h1;
    return w;
}

std::uint64_t HalfStorageFloats(std::uint64_t half_count) { return (half_count + 1ULL) / 2ULL; }

bool LaunchOk() { return cudaGetLastError() == cudaSuccess; }

mgt::Status NotifyGradientReady(MlpGradientReadyCallback callback,
                                void* user,
                                std::uint32_t* ready_id,
                                std::uint64_t param_offset,
                                std::uint64_t param_count,
                                cudaStream_t producer_stream) {
    if (callback == nullptr) return mgt::Status::kOk;
    if (ready_id == nullptr || param_count == 0) return mgt::Status::kInvalidConfig;
    const std::uint32_t id = *ready_id;
    *ready_id = id + 1U;
    return callback(user, id, param_offset, param_count, producer_stream);
}

struct BackwardStageTimer {
    cudaEvent_t last = nullptr;
    cudaEvent_t now = nullptr;
    cudaStream_t stream = nullptr;
    MlpBackwardProfile* profile = nullptr;

    ~BackwardStageTimer() {
        if (now != nullptr) cudaEventDestroy(now);
        if (last != nullptr) cudaEventDestroy(last);
    }

    mgt::Status Start(cudaStream_t target_stream, MlpBackwardProfile* target_profile) {
        stream = target_stream;
        profile = target_profile;
        if (profile == nullptr) return mgt::Status::kOk;
        *profile = MlpBackwardProfile{};
        if (cudaEventCreate(&last) != cudaSuccess) return mgt::Status::kCudaFailure;
        if (cudaEventCreate(&now) != cudaSuccess) return mgt::Status::kCudaFailure;
        if (cudaEventRecord(last, stream) != cudaSuccess) return mgt::Status::kCudaFailure;
        return mgt::Status::kOk;
    }

    mgt::Status Mark(float* dst) {
        if (profile == nullptr) return mgt::Status::kOk;
        if (dst == nullptr) return mgt::Status::kInvalidConfig;
        if (cudaEventRecord(now, stream) != cudaSuccess) return mgt::Status::kCudaFailure;
        if (cudaEventSynchronize(now) != cudaSuccess) return mgt::Status::kCudaFailure;
        if (cudaEventElapsedTime(dst, last, now) != cudaSuccess) return mgt::Status::kCudaFailure;
        cudaEvent_t tmp = last;
        last = now;
        now = tmp;
        return mgt::Status::kOk;
    }
};
mgt::Status GemmRowMajor(cublasHandle_t handle, const float* a, const float* b, float* c, std::uint32_t m, std::uint32_t n, std::uint32_t k, float beta = 0.0f) {
    const float alpha = 1.0f;
    const cublasStatus_t status = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                             static_cast<int>(n), static_cast<int>(m), static_cast<int>(k),
                                             &alpha, b, static_cast<int>(n), a, static_cast<int>(k),
                                             &beta, c, static_cast<int>(n));
    return status == CUBLAS_STATUS_SUCCESS ? mgt::Status::kOk : mgt::Status::kCudaFailure;
}

#ifdef MGT_HAS_CUTLASS_HALF_GEMM
template <typename LayoutA, typename LayoutB>
mgt::Status CutlassGemmHalfToFloatSimt(cudaStream_t stream,
                                       const __half* a,
                                       int lda,
                                       const __half* b,
                                       int ldb,
                                       float* c,
                                       int ldc,
                                       int m,
                                       int n,
                                       int k,
                                       float beta) {
    using ElementInput = cutlass::half_t;
    using ElementOutput = float;
    using ElementAccumulator = float;
    using ElementCompute = float;
    using LayoutOutput = cutlass::layout::RowMajor;
    using ThreadblockShape = cutlass::gemm::GemmShape<64, 64, 8>;
    using WarpShape = cutlass::gemm::GemmShape<32, 32, 8>;
    using InstructionShape = cutlass::gemm::GemmShape<1, 1, 1>;
    using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
        ElementOutput,
        1,
        ElementAccumulator,
        ElementCompute>;
    using Gemm = cutlass::gemm::device::Gemm<
        ElementInput,
        LayoutA,
        ElementInput,
        LayoutB,
        ElementOutput,
        LayoutOutput,
        ElementAccumulator,
        cutlass::arch::OpClassSimt,
        cutlass::arch::Sm50,
        ThreadblockShape,
        WarpShape,
        InstructionShape,
        EpilogueOp,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
        2>;

    typename Gemm::Arguments args(
        {m, n, k},
        {reinterpret_cast<ElementInput const*>(a), lda},
        {reinterpret_cast<ElementInput const*>(b), ldb},
        {c, ldc},
        {c, ldc},
        {1.0f, beta});
    if (Gemm::can_implement(args) != cutlass::Status::kSuccess) return mgt::Status::kInvalidConfig;
    Gemm gemm;
    const cutlass::Status status = gemm(args, nullptr, stream);
    return status == cutlass::Status::kSuccess ? mgt::Status::kOk : mgt::Status::kCudaFailure;
}
template <typename LayoutA, typename LayoutB>
mgt::Status CutlassGemmHalfToFloat(cudaStream_t stream,
                                   const __half* a,
                                   int lda,
                                   const __half* b,
                                   int ldb,
                                   float* c,
                                   int ldc,
                                   int m,
                                   int n,
                                   int k,
                                   float beta) {
    if (a == nullptr || b == nullptr || c == nullptr || m <= 0 || n <= 0 || k <= 0 || lda <= 0 || ldb <= 0 || ldc <= 0) {
        return mgt::Status::kInvalidConfig;
    }
    using ElementInput = cutlass::half_t;
    using ElementOutput = float;
    using ElementAccumulator = float;
    using ElementCompute = float;
    using LayoutOutput = cutlass::layout::RowMajor;
#ifdef MGT_CUTLASS_ARCH_SM80
    using CutlassArch = cutlass::arch::Sm80;
#else
    using CutlassArch = cutlass::arch::Sm75;
#endif
    static_assert(sizeof(ElementInput) == sizeof(__half));

    using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 32>;
    using WarpShape = cutlass::gemm::GemmShape<64, 64, 32>;
    using InstructionShape = cutlass::gemm::GemmShape<16, 8, 8>;
    using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
        ElementOutput,
        128 / cutlass::sizeof_bits<ElementOutput>::value,
        ElementAccumulator,
        ElementCompute>;
    using Gemm = cutlass::gemm::device::Gemm<
        ElementInput,
        LayoutA,
        ElementInput,
        LayoutB,
        ElementOutput,
        LayoutOutput,
        ElementAccumulator,
        cutlass::arch::OpClassTensorOp,
        CutlassArch,
        ThreadblockShape,
        WarpShape,
        InstructionShape,
        EpilogueOp,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
        2>;

    typename Gemm::Arguments args(
        {m, n, k},
        {reinterpret_cast<ElementInput const*>(a), lda},
        {reinterpret_cast<ElementInput const*>(b), ldb},
        {c, ldc},
        {c, ldc},
        {1.0f, beta});
    const cutlass::Status can_implement = Gemm::can_implement(args);
    if (can_implement != cutlass::Status::kSuccess) {
        return CutlassGemmHalfToFloatSimt<LayoutA, LayoutB>(stream, a, lda, b, ldb, c, ldc, m, n, k, beta);
    }
    Gemm gemm;
    const cutlass::Status status = gemm(args, nullptr, stream);
    return status == cutlass::Status::kSuccess ? mgt::Status::kOk : mgt::Status::kCudaFailure;
}

mgt::Status CutlassMatmulHalfToFloat(cudaStream_t stream,
                                     const __half* a,
                                     const __half* b,
                                     float* c,
                                     std::uint32_t a_rows,
                                     std::uint32_t a_cols,
                                     std::uint32_t b_rows,
                                     std::uint32_t b_cols,
                                     std::uint32_t c_rows,
                                     std::uint32_t c_cols,
                                     cublasOperation_t op_a,
                                     cublasOperation_t op_b,
                                     float beta) {
    if (op_a == CUBLAS_OP_N && op_b == CUBLAS_OP_N) {
        if (c_rows != a_rows || c_cols != b_cols || a_cols != b_rows) return mgt::Status::kInvalidConfig;
        return CutlassGemmHalfToFloat<cutlass::layout::RowMajor, cutlass::layout::RowMajor>(
            stream, a, static_cast<int>(a_cols), b, static_cast<int>(b_cols), c, static_cast<int>(c_cols),
            static_cast<int>(c_rows), static_cast<int>(c_cols), static_cast<int>(a_cols), beta);
    }
    if (op_a == CUBLAS_OP_T && op_b == CUBLAS_OP_N) {
        if (c_rows != a_cols || c_cols != b_cols || a_rows != b_rows) return mgt::Status::kInvalidConfig;
        return CutlassGemmHalfToFloat<cutlass::layout::ColumnMajor, cutlass::layout::RowMajor>(
            stream, a, static_cast<int>(a_cols), b, static_cast<int>(b_cols), c, static_cast<int>(c_cols),
            static_cast<int>(c_rows), static_cast<int>(c_cols), static_cast<int>(a_rows), beta);
    }
    if (op_a == CUBLAS_OP_N && op_b == CUBLAS_OP_T) {
        if (c_rows != a_rows || c_cols != b_rows || a_cols != b_cols) return mgt::Status::kInvalidConfig;
        return CutlassGemmHalfToFloat<cutlass::layout::RowMajor, cutlass::layout::ColumnMajor>(
            stream, a, static_cast<int>(a_cols), b, static_cast<int>(b_cols), c, static_cast<int>(c_cols),
            static_cast<int>(c_rows), static_cast<int>(c_cols), static_cast<int>(a_cols), beta);
    }
    return mgt::Status::kInvalidConfig;
}
#endif
enum class HalfGemmOpKind : std::uint32_t {
    kForward = 1,
    kGradWeights = 2,
    kBackpropInput = 3,
    kInputEmbeddingGrad = 4
};

bool ShouldUseCutlassHalfGemm(HalfGemmOpKind kind,
                              std::uint32_t c_rows,
                              std::uint32_t c_cols,
                              std::uint32_t k,
                              cublasOperation_t op_a,
                              cublasOperation_t op_b) {
#ifdef MGT_USE_CUTLASS_HALF_GEMM
    (void)kind;
    (void)c_rows;
    (void)c_cols;
    (void)k;
    (void)op_a;
    (void)op_b;
    return true;
#elif defined(MGT_AUTO_CUTLASS_HALF_GEMM)
    return kind == HalfGemmOpKind::kInputEmbeddingGrad &&
           op_a == CUBLAS_OP_T && op_b == CUBLAS_OP_N &&
           c_rows >= 4096U && c_cols >= 4096U && k >= 8192U;
#else
    (void)kind;
    (void)c_rows;
    (void)c_cols;
    (void)k;
    (void)op_a;
    (void)op_b;
    return false;
#endif
}
struct LtMatmulPlanKey {
    std::uint32_t a_rows;
    std::uint32_t a_cols;
    std::uint32_t b_rows;
    std::uint32_t b_cols;
    std::uint32_t c_rows;
    std::uint32_t c_cols;
    cublasOperation_t op_a;
    cublasOperation_t op_b;
    std::uint64_t workspace_bytes;
};

inline constexpr std::uint32_t kLtMatmulMaxHeuristicCandidates = 16;
inline constexpr std::uint32_t kLtMatmulMaxWarmupIterations = 8;
inline constexpr std::uint32_t kLtMatmulMaxTimingIterations = 16;

LtMatmulAutotuneConfig NormalizeLtMatmulAutotuneConfig(LtMatmulAutotuneConfig config) {
    if (config.max_candidates == 0U) config.max_candidates = 1U;
    config.max_candidates = std::min<std::uint32_t>(config.max_candidates, kLtMatmulMaxHeuristicCandidates);
    config.warmup_iterations = std::min<std::uint32_t>(config.warmup_iterations, kLtMatmulMaxWarmupIterations);
    if (config.timing_iterations == 0U) config.timing_iterations = 1U;
    config.timing_iterations = std::min<std::uint32_t>(config.timing_iterations, kLtMatmulMaxTimingIterations);
    return config;
}

struct LtMatmulAlgoCandidate {
    cublasLtMatmulAlgo_t algo{};
    std::size_t workspace_bytes = 0;
};

struct LtMatmulPlanCacheEntry {
    bool initialized = false;
    LtMatmulPlanKey key{};
    cublasLtMatmulDesc_t op_desc = nullptr;
    cublasLtMatrixLayout_t a_desc = nullptr;
    cublasLtMatrixLayout_t b_desc = nullptr;
    cublasLtMatrixLayout_t c_desc = nullptr;
    cublasLtMatmulAlgo_t algo{};
    std::size_t algo_workspace_bytes = 0;
    bool has_algo = false;
    bool autotuned = false;
    std::uint32_t candidate_count = 0;
    LtMatmulAlgoCandidate candidates[kLtMatmulMaxHeuristicCandidates]{};
};

inline constexpr std::uint32_t kLtMatmulPlanCacheCapacity = 256;
LtMatmulAutotuneConfig g_lt_matmul_autotune_config{};
LtMatmulPlanCacheEntry g_lt_matmul_plan_cache[kLtMatmulPlanCacheCapacity]{};

bool SameLtMatmulPlanKey(const LtMatmulPlanKey& lhs, const LtMatmulPlanKey& rhs) {
    return lhs.a_rows == rhs.a_rows && lhs.a_cols == rhs.a_cols && lhs.b_rows == rhs.b_rows && lhs.b_cols == rhs.b_cols &&
           lhs.c_rows == rhs.c_rows && lhs.c_cols == rhs.c_cols && lhs.op_a == rhs.op_a && lhs.op_b == rhs.op_b &&
           lhs.workspace_bytes == rhs.workspace_bytes;
}


void DestroyLtMatmulPlan(cublasLtMatmulDesc_t op_desc,
                         cublasLtMatrixLayout_t a_desc,
                         cublasLtMatrixLayout_t b_desc,
                         cublasLtMatrixLayout_t c_desc) {
    if (c_desc != nullptr) cublasLtMatrixLayoutDestroy(c_desc);
    if (b_desc != nullptr) cublasLtMatrixLayoutDestroy(b_desc);
    if (a_desc != nullptr) cublasLtMatrixLayoutDestroy(a_desc);
    if (op_desc != nullptr) cublasLtMatmulDescDestroy(op_desc);
}

mgt::Status InitLtMatmulPlan(cublasLtHandle_t handle, LtMatmulPlanCacheEntry* entry, const LtMatmulPlanKey& key) {
    if (entry == nullptr) return mgt::Status::kInvalidConfig;
    cublasLtMatmulDesc_t op_desc = nullptr;
    cublasLtMatrixLayout_t a_desc = nullptr;
    cublasLtMatrixLayout_t b_desc = nullptr;
    cublasLtMatrixLayout_t c_desc = nullptr;
    const cublasLtOrder_t order = CUBLASLT_ORDER_ROW;
    cublasStatus_t status = cublasLtMatmulDescCreate(&op_desc, CUBLAS_COMPUTE_32F, CUDA_R_32F);
    if (status == CUBLAS_STATUS_SUCCESS) status = cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_TRANSA, &key.op_a, sizeof(key.op_a));
    if (status == CUBLAS_STATUS_SUCCESS) status = cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_TRANSB, &key.op_b, sizeof(key.op_b));
    if (status == CUBLAS_STATUS_SUCCESS) status = cublasLtMatrixLayoutCreate(&a_desc, CUDA_R_16F, static_cast<std::uint64_t>(key.a_rows), static_cast<std::uint64_t>(key.a_cols), static_cast<std::int64_t>(key.a_cols));
    if (status == CUBLAS_STATUS_SUCCESS) status = cublasLtMatrixLayoutSetAttribute(a_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order));
    if (status == CUBLAS_STATUS_SUCCESS) status = cublasLtMatrixLayoutCreate(&b_desc, CUDA_R_16F, static_cast<std::uint64_t>(key.b_rows), static_cast<std::uint64_t>(key.b_cols), static_cast<std::int64_t>(key.b_cols));
    if (status == CUBLAS_STATUS_SUCCESS) status = cublasLtMatrixLayoutSetAttribute(b_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order));
    if (status == CUBLAS_STATUS_SUCCESS) status = cublasLtMatrixLayoutCreate(&c_desc, CUDA_R_32F, static_cast<std::uint64_t>(key.c_rows), static_cast<std::uint64_t>(key.c_cols), static_cast<std::int64_t>(key.c_cols));
    if (status == CUBLAS_STATUS_SUCCESS) status = cublasLtMatrixLayoutSetAttribute(c_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order));
    if (status != CUBLAS_STATUS_SUCCESS) {
        DestroyLtMatmulPlan(op_desc, a_desc, b_desc, c_desc);
        return mgt::Status::kCudaFailure;
    }

    entry->key = key;
    entry->op_desc = op_desc;
    entry->a_desc = a_desc;
    entry->b_desc = b_desc;
    entry->c_desc = c_desc;
    entry->algo_workspace_bytes = 0;
    entry->has_algo = false;
    entry->autotuned = false;
    entry->candidate_count = 0;

    if (key.workspace_bytes > 0) {
        if (handle == nullptr) {
            DestroyLtMatmulPlan(op_desc, a_desc, b_desc, c_desc);
            return mgt::Status::kInvalidConfig;
        }
        cublasLtMatmulPreference_t preference = nullptr;
        if (cublasLtMatmulPreferenceCreate(&preference) != CUBLAS_STATUS_SUCCESS) {
            DestroyLtMatmulPlan(op_desc, a_desc, b_desc, c_desc);
            return mgt::Status::kCudaFailure;
        }
        const std::size_t max_workspace_bytes = static_cast<std::size_t>(key.workspace_bytes);
        cublasStatus_t heuristic_status = cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &max_workspace_bytes, sizeof(max_workspace_bytes));
        const LtMatmulAutotuneConfig autotune_config = NormalizeLtMatmulAutotuneConfig(g_lt_matmul_autotune_config);
        cublasLtMatmulHeuristicResult_t heuristics[kLtMatmulMaxHeuristicCandidates]{};
        int returned_algorithms = 0;
        if (heuristic_status == CUBLAS_STATUS_SUCCESS) {
            heuristic_status = cublasLtMatmulAlgoGetHeuristic(handle, op_desc, a_desc, b_desc, c_desc, c_desc, preference, static_cast<int>(autotune_config.max_candidates), heuristics, &returned_algorithms);
        }
        if (heuristic_status == CUBLAS_STATUS_SUCCESS) {
            const std::uint32_t checked_algorithms = std::min<std::uint32_t>(static_cast<std::uint32_t>(returned_algorithms), autotune_config.max_candidates);
            for (std::uint32_t i = 0; i < checked_algorithms; ++i) {
                cublasLtMatmulHeuristicResult_t checked{};
                if (heuristics[i].workspaceSize <= max_workspace_bytes &&
                    cublasLtMatmulAlgoCheck(handle, op_desc, a_desc, b_desc, c_desc, c_desc, &heuristics[i].algo, &checked) == CUBLAS_STATUS_SUCCESS &&
                    checked.workspaceSize <= max_workspace_bytes && entry->candidate_count < kLtMatmulMaxHeuristicCandidates) {
                    LtMatmulAlgoCandidate& candidate = entry->candidates[entry->candidate_count++];
                    candidate.algo = heuristics[i].algo;
                    candidate.workspace_bytes = checked.workspaceSize > heuristics[i].workspaceSize ? checked.workspaceSize : heuristics[i].workspaceSize;
                }
            }
            if (entry->candidate_count > 0U) {
                entry->algo = entry->candidates[0].algo;
                entry->algo_workspace_bytes = entry->candidates[0].workspace_bytes;
                entry->has_algo = true;
            }
        }
        cublasLtMatmulPreferenceDestroy(preference);
        // Keep the descriptors even without a workspace algorithm; cuBLASLt can still use its default algorithm without workspace.
    }

    entry->initialized = true;
    return mgt::Status::kOk;
}
mgt::Status GetLtMatmulPlan(cublasLtHandle_t handle, const LtMatmulPlanKey& key, LtMatmulPlanCacheEntry** out) {
    if (out == nullptr) return mgt::Status::kInvalidConfig;
    for (std::uint32_t i = 0; i < kLtMatmulPlanCacheCapacity; ++i) {
        LtMatmulPlanCacheEntry& entry = g_lt_matmul_plan_cache[i];
        if (entry.initialized && SameLtMatmulPlanKey(entry.key, key)) {
            *out = &entry;
            return mgt::Status::kOk;
        }
    }
    for (std::uint32_t i = 0; i < kLtMatmulPlanCacheCapacity; ++i) {
        LtMatmulPlanCacheEntry& entry = g_lt_matmul_plan_cache[i];
        if (!entry.initialized) {
            const mgt::Status status = InitLtMatmulPlan(handle, &entry, key);
            if (status != mgt::Status::kOk) return status;
            *out = &entry;
            return mgt::Status::kOk;
        }
    }
    return mgt::Status::kInvalidConfig;
}
mgt::Status AutotuneLtMatmulPlan(cublasLtHandle_t handle,
                                  LtMatmulPlanCacheEntry* plan,
                                  cudaStream_t stream,
                                  void* lt_workspace,
                                  std::uint64_t lt_workspace_bytes,
                                  const __half* a,
                                  const __half* b,
                                  float* c,
                                  float beta) {
    LtMatmulAutotuneConfig config = NormalizeLtMatmulAutotuneConfig(g_lt_matmul_autotune_config);
    if (!config.enabled || plan == nullptr || plan->autotuned || plan->candidate_count <= 1U || beta != 0.0f) return mgt::Status::kOk;
    plan->autotuned = true;
    if (handle == nullptr || lt_workspace == nullptr || lt_workspace_bytes == 0U) return mgt::Status::kOk;

    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    if (cudaEventCreate(&start) != cudaSuccess) return mgt::Status::kCudaFailure;
    if (cudaEventCreate(&stop) != cudaSuccess) {
        cudaEventDestroy(start);
        return mgt::Status::kCudaFailure;
    }

    const float alpha = 1.0f;
    const std::size_t workspace_bytes = static_cast<std::size_t>(lt_workspace_bytes);
    float best_ms = std::numeric_limits<float>::infinity();
    std::uint32_t best_index = 0;
    bool selected = false;
    const std::uint32_t candidate_count = std::min<std::uint32_t>(plan->candidate_count, config.max_candidates);
    for (std::uint32_t candidate_index = 0; candidate_index < candidate_count; ++candidate_index) {
        const LtMatmulAlgoCandidate& candidate = plan->candidates[candidate_index];
        if (candidate.workspace_bytes > workspace_bytes) continue;
        bool failed = false;
        for (std::uint32_t iter = 0; iter < config.warmup_iterations; ++iter) {
            const cublasStatus_t status = cublasLtMatmul(handle, plan->op_desc, &alpha, a, plan->a_desc, b, plan->b_desc, &beta, c, plan->c_desc, c, plan->c_desc, &candidate.algo, lt_workspace, workspace_bytes, stream);
            if (status != CUBLAS_STATUS_SUCCESS) {
                failed = true;
                break;
            }
        }
        if (failed) continue;
        if (cudaEventRecord(start, stream) != cudaSuccess) {
            failed = true;
        }
        for (std::uint32_t iter = 0; !failed && iter < config.timing_iterations; ++iter) {
            const cublasStatus_t status = cublasLtMatmul(handle, plan->op_desc, &alpha, a, plan->a_desc, b, plan->b_desc, &beta, c, plan->c_desc, c, plan->c_desc, &candidate.algo, lt_workspace, workspace_bytes, stream);
            if (status != CUBLAS_STATUS_SUCCESS) failed = true;
        }
        if (failed || cudaEventRecord(stop, stream) != cudaSuccess || cudaEventSynchronize(stop) != cudaSuccess) continue;
        float elapsed_ms = 0.0f;
        if (cudaEventElapsedTime(&elapsed_ms, start, stop) != cudaSuccess) continue;
        const float avg_ms = elapsed_ms / static_cast<float>(config.timing_iterations);
        if (avg_ms < best_ms) {
            best_ms = avg_ms;
            best_index = candidate_index;
            selected = true;
        }
    }

    cudaEventDestroy(stop);
    cudaEventDestroy(start);
    if (selected) {
        plan->algo = plan->candidates[best_index].algo;
        plan->algo_workspace_bytes = plan->candidates[best_index].workspace_bytes;
        plan->has_algo = true;
    } else {
        plan->algo_workspace_bytes = 0;
        plan->has_algo = false;
    }
    return mgt::Status::kOk;
}

mgt::Status LtMatmulHalfToFloat(cublasLtHandle_t handle,
                                 cudaStream_t stream,
                                 void* lt_workspace,
                                 std::uint64_t lt_workspace_bytes,
                                 const __half* a,
                                 const __half* b,
                                 float* c,
                                 std::uint32_t a_rows,
                                 std::uint32_t a_cols,
                                 std::uint32_t b_rows,
                                 std::uint32_t b_cols,
                                 std::uint32_t c_rows,
                                 std::uint32_t c_cols,
                                 cublasOperation_t op_a,
                                 cublasOperation_t op_b,
                                 float beta,
                                 HalfGemmOpKind kind) {
    if (handle == nullptr) return mgt::Status::kInvalidConfig;
    if (lt_workspace_bytes > 0 && lt_workspace == nullptr) return mgt::Status::kInvalidConfig;
#ifdef MGT_HAS_CUTLASS_HALF_GEMM
    if (ShouldUseCutlassHalfGemm(kind, c_rows, c_cols, op_a == CUBLAS_OP_T ? a_rows : a_cols, op_a, op_b)) {
        return CutlassMatmulHalfToFloat(stream, a, b, c, a_rows, a_cols, b_rows, b_cols, c_rows, c_cols, op_a, op_b, beta);
    }
#else
    (void)kind;
#endif
    const float alpha = 1.0f;
    const LtMatmulPlanKey key{a_rows, a_cols, b_rows, b_cols, c_rows, c_cols, op_a, op_b, lt_workspace_bytes};
    LtMatmulPlanCacheEntry* plan = nullptr;
    const mgt::Status plan_status = GetLtMatmulPlan(handle, key, &plan);
    if (plan_status != mgt::Status::kOk) return plan_status;
    const mgt::Status autotune_status = AutotuneLtMatmulPlan(handle, plan, stream, lt_workspace, lt_workspace_bytes, a, b, c, beta);
    if (autotune_status != mgt::Status::kOk) return autotune_status;
    const cublasLtMatmulAlgo_t* algo = plan->has_algo ? &plan->algo : nullptr;
    void* workspace = plan->has_algo ? lt_workspace : nullptr;
    const std::size_t workspace_bytes = plan->has_algo ? static_cast<std::size_t>(lt_workspace_bytes) : 0;
    const cublasStatus_t status = cublasLtMatmul(handle, plan->op_desc, &alpha, a, plan->a_desc, b, plan->b_desc, &beta, c, plan->c_desc, c, plan->c_desc, algo, workspace, workspace_bytes, stream);
    return status == CUBLAS_STATUS_SUCCESS ? mgt::Status::kOk : mgt::Status::kCudaFailure;
}

mgt::Status GemmRowMajorHalfToFloat(cublasLtHandle_t lt, cudaStream_t stream, void* lt_workspace, std::uint64_t lt_workspace_bytes, const __half* a, const __half* b, float* c, std::uint32_t m, std::uint32_t n, std::uint32_t k, float beta = 0.0f) {
    return LtMatmulHalfToFloat(lt, stream, lt_workspace, lt_workspace_bytes, a, b, c, m, k, k, n, m, n, CUBLAS_OP_N, CUBLAS_OP_N, beta, HalfGemmOpKind::kForward);
}
mgt::Status GemmGradWeights(cublasHandle_t handle, const float* x, const float* dz, float* grad_w, std::uint32_t samples, std::uint32_t in_dim, std::uint32_t out_dim) {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    const cublasStatus_t status = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T,
                                             static_cast<int>(out_dim), static_cast<int>(in_dim), static_cast<int>(samples),
                                             &alpha, dz, static_cast<int>(out_dim), x, static_cast<int>(in_dim),
                                             &beta, grad_w, static_cast<int>(out_dim));
    return status == CUBLAS_STATUS_SUCCESS ? mgt::Status::kOk : mgt::Status::kCudaFailure;
}
mgt::Status GemmGradWeightsHalf(cublasLtHandle_t lt, cudaStream_t stream, void* lt_workspace, std::uint64_t lt_workspace_bytes, const __half* x, const __half* dz, float* grad_w, std::uint32_t samples, std::uint32_t in_dim, std::uint32_t out_dim, HalfGemmOpKind kind = HalfGemmOpKind::kGradWeights) {
    return LtMatmulHalfToFloat(lt, stream, lt_workspace, lt_workspace_bytes, x, dz, grad_w, samples, in_dim, samples, out_dim, in_dim, out_dim, CUBLAS_OP_T, CUBLAS_OP_N, 0.0f, kind);
}

mgt::Status GemmBackpropInput(cublasHandle_t handle, const float* dz, const float* weights, float* dx, std::uint32_t samples, std::uint32_t in_dim, std::uint32_t out_dim, float beta = 0.0f) {
    const float alpha = 1.0f;
    const cublasStatus_t status = cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                                             static_cast<int>(in_dim), static_cast<int>(samples), static_cast<int>(out_dim),
                                             &alpha, weights, static_cast<int>(out_dim), dz, static_cast<int>(out_dim),
                                             &beta, dx, static_cast<int>(in_dim));
    return status == CUBLAS_STATUS_SUCCESS ? mgt::Status::kOk : mgt::Status::kCudaFailure;
}

mgt::Status GemmBackpropInputHalf(cublasLtHandle_t lt, cudaStream_t stream, void* lt_workspace, std::uint64_t lt_workspace_bytes, const __half* dz, const __half* weights, float* dx, std::uint32_t samples, std::uint32_t in_dim, std::uint32_t out_dim, float beta = 0.0f) {
    return LtMatmulHalfToFloat(lt, stream, lt_workspace, lt_workspace_bytes, dz, weights, dx, samples, out_dim, in_dim, out_dim, samples, in_dim, CUBLAS_OP_N, CUBLAS_OP_T, beta, HalfGemmOpKind::kBackpropInput);
}
mgt::Status GemvBiasGrad(cublasHandle_t handle, const float* dz, const float* ones, float* bias_grad, std::uint32_t samples, std::uint32_t cols) {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    const cublasStatus_t status = cublasSgemv(handle, CUBLAS_OP_N,
                                             static_cast<int>(cols), static_cast<int>(samples),
                                             &alpha, dz, static_cast<int>(cols), ones, 1,
                                             &beta, bias_grad, 1);
    return status == CUBLAS_STATUS_SUCCESS ? mgt::Status::kOk : mgt::Status::kCudaFailure;
}

__global__ void FillOnesKernel(float* values, std::uint32_t count) {
    const std::uint32_t item = blockIdx.x * blockDim.x + threadIdx.x;
    if (item >= count) return;
    values[item] = 1.0f;
}

__global__ void BuildInputOneHotKernel(CudaMlpShape shape, const mgt::TrainStateStorage* states, std::uint32_t samples, float* input_one_hot) {
    const std::uint64_t item = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::uint64_t total = static_cast<std::uint64_t>(samples) * shape.state_len;
    if (item >= total) return;
    const std::uint32_t pos = static_cast<std::uint32_t>(item % shape.state_len);
    const std::uint32_t b = static_cast<std::uint32_t>(item / shape.state_len);
    const std::uint32_t value = states[b].v[pos];
    if (value >= shape.state_value_pad) return;
    const std::uint32_t input_dim = shape.state_len * shape.state_value_pad;
    const std::uint32_t col = pos * shape.state_value_pad + value;
    input_one_hot[static_cast<std::uint64_t>(b) * input_dim + col] = 1.0f;
}
__global__ void BuildInputOneHotHalfKernel(CudaMlpShape shape, const mgt::TrainStateStorage* states, std::uint32_t samples, __half* input_one_hot) {
    const std::uint64_t item = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::uint64_t total = static_cast<std::uint64_t>(samples) * shape.state_len;
    if (item >= total) return;
    const std::uint32_t pos = static_cast<std::uint32_t>(item % shape.state_len);
    const std::uint32_t b = static_cast<std::uint32_t>(item / shape.state_len);
    const std::uint32_t value = states[b].v[pos];
    if (value >= shape.state_value_pad) return;
    const std::uint32_t input_dim = shape.state_len * shape.state_value_pad;
    const std::uint32_t col = pos * shape.state_value_pad + value;
    input_one_hot[static_cast<std::uint64_t>(b) * input_dim + col] = __float2half(1.0f);
}

__global__ void BuildInputPositionTileOneHotKernel(CudaMlpShape shape, const mgt::TrainStateStorage* states, std::uint32_t samples, std::uint32_t first_pos, std::uint32_t tile_positions, float* input_one_hot) {
    const std::uint32_t tile_cols = tile_positions * shape.state_value_pad;
    const std::uint64_t total = static_cast<std::uint64_t>(samples) * tile_cols;
    const std::uint64_t item = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (item >= total) return;
    const std::uint32_t tile_col = static_cast<std::uint32_t>(item % tile_cols);
    const std::uint32_t b = static_cast<std::uint32_t>(item / tile_cols);
    const std::uint32_t tile_pos = tile_col / shape.state_value_pad;
    const std::uint32_t value = tile_col - tile_pos * shape.state_value_pad;
    const std::uint32_t pos = first_pos + tile_pos;
    input_one_hot[item] = (pos < shape.state_len && static_cast<std::uint32_t>(states[b].v[pos]) == value) ? 1.0f : 0.0f;
}

__global__ void BuildInputPositionTileOneHotHalfKernel(CudaMlpShape shape, const mgt::TrainStateStorage* states, std::uint32_t samples, std::uint32_t first_pos, std::uint32_t tile_positions, __half* input_one_hot) {
    const std::uint32_t tile_cols = tile_positions * shape.state_value_pad;
    const std::uint64_t total = static_cast<std::uint64_t>(samples) * tile_cols;
    const std::uint64_t item = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (item >= total) return;
    const std::uint32_t tile_col = static_cast<std::uint32_t>(item % tile_cols);
    const std::uint32_t b = static_cast<std::uint32_t>(item / tile_cols);
    const std::uint32_t tile_pos = tile_col / shape.state_value_pad;
    const std::uint32_t value = tile_col - tile_pos * shape.state_value_pad;
    const std::uint32_t pos = first_pos + tile_pos;
    input_one_hot[item] = (pos < shape.state_len && static_cast<std::uint32_t>(states[b].v[pos]) == value) ? __float2half(1.0f) : __float2half(0.0f);
}

__global__ void FloatToHalfKernel(const float* src, __half* dst, std::uint64_t count) {
    const std::uint64_t item = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (item >= count) return;
    dst[item] = __float2half(src[item]);
}
inline constexpr std::uint32_t kInputForwardThreads = 128;
inline constexpr std::uint32_t kInputForwardVec = 4;

__global__ void InputForwardKernel(CudaMlpShape shape, const float* __restrict__ weights, const mgt::TrainStateStorage* __restrict__ states, std::uint32_t samples, float* __restrict__ a1) {
    const std::uint32_t b = blockIdx.x;
    const std::uint32_t h = (blockIdx.y * blockDim.x + threadIdx.x) * kInputForwardVec;
    if (b >= samples) return;

    __shared__ std::uint64_t input_row_offsets[mgt::kStateStorageLen];
    if (threadIdx.x < shape.state_len) {
        const std::uint32_t value = static_cast<std::uint32_t>(states[b].v[threadIdx.x]);
        input_row_offsets[threadIdx.x] = (static_cast<std::uint64_t>(threadIdx.x) * shape.state_value_pad + value) * shape.hd1;
    }
    __syncthreads();
    if (h >= shape.hd1) return;

    const std::uint64_t input_bias = InputBias(shape);
    const bool vector4_aligned = (shape.hd1 & 3U) == 0U && h + 3U < shape.hd1;
    if (vector4_aligned) {
        const float4 bias = *reinterpret_cast<const float4*>(weights + input_bias + h);
        float4 acc = bias;
        for (std::uint32_t pos = 0; pos < shape.state_len; ++pos) {
            const float4 w = *reinterpret_cast<const float4*>(weights + input_row_offsets[pos] + h);
            acc.x += w.x;
            acc.y += w.y;
            acc.z += w.z;
            acc.w += w.w;
        }
        const std::uint64_t out = static_cast<std::uint64_t>(b) * shape.hd1 + h;
        float4 y;
        y.x = Relu(acc.x);
        y.y = Relu(acc.y);
        y.z = Relu(acc.z);
        y.w = Relu(acc.w);
        *reinterpret_cast<float4*>(a1 + out) = y;
        return;
    }

    float acc0 = weights[input_bias + h];
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    float acc3 = 0.0f;
    const bool has1 = h + 1U < shape.hd1;
    const bool has2 = h + 2U < shape.hd1;
    const bool has3 = h + 3U < shape.hd1;
    if (has1) acc1 = weights[input_bias + h + 1U];
    if (has2) acc2 = weights[input_bias + h + 2U];
    if (has3) acc3 = weights[input_bias + h + 3U];
    for (std::uint32_t pos = 0; pos < shape.state_len; ++pos) {
        const std::uint64_t idx = input_row_offsets[pos] + h;
        acc0 += weights[idx];
        if (has1) acc1 += weights[idx + 1U];
        if (has2) acc2 += weights[idx + 2U];
        if (has3) acc3 += weights[idx + 3U];
    }
    const std::uint64_t out = static_cast<std::uint64_t>(b) * shape.hd1 + h;
    a1[out] = Relu(acc0);
    if (has1) a1[out + 1U] = Relu(acc1);
    if (has2) a1[out + 2U] = Relu(acc2);
    if (has3) a1[out + 3U] = Relu(acc3);
}

__global__ void InputForwardHalfKernel(CudaMlpShape shape, const __half* __restrict__ weights, const mgt::TrainStateStorage* __restrict__ states, std::uint32_t samples, float* __restrict__ a1, __half* __restrict__ a1_half) {
    const std::uint32_t b = blockIdx.x;
    const std::uint32_t h = (blockIdx.y * blockDim.x + threadIdx.x) * kInputForwardVec;
    if (b >= samples) return;

    __shared__ std::uint64_t input_row_offsets[mgt::kStateStorageLen];
    if (threadIdx.x < shape.state_len) {
        const std::uint32_t value = static_cast<std::uint32_t>(states[b].v[threadIdx.x]);
        input_row_offsets[threadIdx.x] = (static_cast<std::uint64_t>(threadIdx.x) * shape.state_value_pad + value) * shape.hd1;
    }
    __syncthreads();
    if (h >= shape.hd1) return;

    const std::uint64_t input_bias = InputBias(shape);
    const bool vector4_aligned = (shape.hd1 & 3U) == 0U && h + 3U < shape.hd1;
    if (vector4_aligned) {
        float2 acc01 = __half22float2(*reinterpret_cast<const __half2*>(weights + input_bias + h));
        float2 acc23 = __half22float2(*reinterpret_cast<const __half2*>(weights + input_bias + h + 2U));
        for (std::uint32_t pos = 0; pos < shape.state_len; ++pos) {
            const std::uint64_t idx = input_row_offsets[pos] + h;
            const float2 w01 = __half22float2(*reinterpret_cast<const __half2*>(weights + idx));
            const float2 w23 = __half22float2(*reinterpret_cast<const __half2*>(weights + idx + 2U));
            acc01.x += w01.x;
            acc01.y += w01.y;
            acc23.x += w23.x;
            acc23.y += w23.y;
        }
        const std::uint64_t out = static_cast<std::uint64_t>(b) * shape.hd1 + h;
        const float y0 = Relu(acc01.x);
        const float y1 = Relu(acc01.y);
        const float y2 = Relu(acc23.x);
        const float y3 = Relu(acc23.y);
        a1[out] = y0;
        a1[out + 1U] = y1;
        a1[out + 2U] = y2;
        a1[out + 3U] = y3;
        *reinterpret_cast<__half2*>(a1_half + out) = __floats2half2_rn(y0, y1);
        *reinterpret_cast<__half2*>(a1_half + out + 2U) = __floats2half2_rn(y2, y3);
        return;
    }

    float acc0 = __half2float(weights[input_bias + h]);
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    float acc3 = 0.0f;
    const bool has1 = h + 1U < shape.hd1;
    const bool has2 = h + 2U < shape.hd1;
    const bool has3 = h + 3U < shape.hd1;
    if (has1) acc1 = __half2float(weights[input_bias + h + 1U]);
    if (has2) acc2 = __half2float(weights[input_bias + h + 2U]);
    if (has3) acc3 = __half2float(weights[input_bias + h + 3U]);
    for (std::uint32_t pos = 0; pos < shape.state_len; ++pos) {
        const std::uint64_t idx = input_row_offsets[pos] + h;
        acc0 += __half2float(weights[idx]);
        if (has1) acc1 += __half2float(weights[idx + 1U]);
        if (has2) acc2 += __half2float(weights[idx + 2U]);
        if (has3) acc3 += __half2float(weights[idx + 3U]);
    }
    const std::uint64_t out = static_cast<std::uint64_t>(b) * shape.hd1 + h;
    const float y0 = Relu(acc0);
    a1[out] = y0;
    a1_half[out] = __float2half(y0);
    if (has1) {
        const float y1 = Relu(acc1);
        a1[out + 1U] = y1;
        a1_half[out + 1U] = __float2half(y1);
    }
    if (has2) {
        const float y2 = Relu(acc2);
        a1[out + 2U] = y2;
        a1_half[out + 2U] = __float2half(y2);
    }
    if (has3) {
        const float y3 = Relu(acc3);
        a1[out + 3U] = y3;
        a1_half[out + 3U] = __float2half(y3);
    }
}
__global__ void AddBiasReluKernel(float* x, const float* bias, std::uint32_t rows, std::uint32_t cols) {
    const std::uint64_t item = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::uint64_t total = static_cast<std::uint64_t>(rows) * cols;
    if (item >= total) return;
    const std::uint32_t col = static_cast<std::uint32_t>(item % cols);
    x[item] = Relu(x[item] + bias[col]);
}

__global__ void AddBiasReluCopyKernel(float* x, const float* bias, std::uint32_t rows, std::uint32_t cols, float* out) {
    const std::uint64_t item = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::uint64_t total = static_cast<std::uint64_t>(rows) * cols;
    if (item >= total) return;
    const std::uint32_t col = static_cast<std::uint32_t>(item % cols);
    const float v = Relu(x[item] + bias[col]);
    x[item] = v;
    out[item] = v;
}

__global__ void AddBiasResidualReluKernel(float* z, const float* bias, const float* residual, std::uint32_t rows, std::uint32_t cols, float* out) {
    const std::uint64_t item = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::uint64_t total = static_cast<std::uint64_t>(rows) * cols;
    if (item >= total) return;
    const std::uint32_t col = static_cast<std::uint32_t>(item % cols);
    z[item] += bias[col];
    out[item] = Relu(residual[item] + z[item]);
}

__global__ void AddBiasReluHalfKernel(float* x, const float* bias, std::uint32_t rows, std::uint32_t cols, __half* x_half) {
    const std::uint64_t item = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::uint64_t total = static_cast<std::uint64_t>(rows) * cols;
    if (item >= total) return;
    const std::uint32_t col = static_cast<std::uint32_t>(item % cols);
    const float v = Relu(x[item] + bias[col]);
    x[item] = v;
    x_half[item] = __float2half(v);
}

__global__ void AddBiasReluCopyHalfKernel(float* x, const float* bias, std::uint32_t rows, std::uint32_t cols, float* out, __half* out_half) {
    const std::uint64_t item = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::uint64_t total = static_cast<std::uint64_t>(rows) * cols;
    if (item >= total) return;
    const std::uint32_t col = static_cast<std::uint32_t>(item % cols);
    const float v = Relu(x[item] + bias[col]);
    x[item] = v;
    out[item] = v;
    out_half[item] = __float2half(v);
}

__global__ void AddBiasResidualReluHalfKernel(float* z, const float* bias, const float* residual, std::uint32_t rows, std::uint32_t cols, float* out, __half* out_half) {
    const std::uint64_t item = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::uint64_t total = static_cast<std::uint64_t>(rows) * cols;
    if (item >= total) return;
    const std::uint32_t col = static_cast<std::uint32_t>(item % cols);
    z[item] += bias[col];
    const float v = Relu(residual[item] + z[item]);
    out[item] = v;
    out_half[item] = __float2half(v);
}
__global__ void OutputBackwardInitKernel(CudaMlpShape shape, const float* weights, const float* labels, std::uint32_t samples, const float* final_act, float* loss_terms, float* dy, float* dcur) {
    const std::uint32_t b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= samples) return;
    const std::uint64_t output_weight = OutputWeight(shape);
    float y = weights[OutputBias(shape)];
    const float* act = final_act + static_cast<std::uint64_t>(b) * shape.hd2;
    for (std::uint32_t j = 0; j < shape.hd2; ++j) y += act[j] * weights[output_weight + j];
    const float diff = y - labels[b];
    const float inv_n = 1.0f / static_cast<float>(samples);
    const float local_dy = 2.0f * diff * inv_n;
    dy[b] = local_dy;
    loss_terms[b] = diff * diff * inv_n;
    for (std::uint32_t j = 0; j < shape.hd2; ++j) dcur[static_cast<std::uint64_t>(b) * shape.hd2 + j] = weights[output_weight + j] * local_dy;
}

__global__ void ReduceLossKernel(const float* loss_terms, std::uint32_t samples, float* loss) {
    __shared__ float partial[256];
    const std::uint32_t tid = threadIdx.x;
    float sum = 0.0f;
    for (std::uint32_t b = tid; b < samples; b += blockDim.x) sum += loss_terms[b];
    partial[tid] = sum;
    __syncthreads();
    for (std::uint32_t stride = blockDim.x >> 1U; stride > 0; stride >>= 1U) {
        if (tid < stride) partial[tid] += partial[tid + stride];
        __syncthreads();
    }
    if (tid == 0) *loss = partial[0];
}

__global__ void OutputGradKernel(CudaMlpShape shape, const float* final_act, const float* dy, std::uint32_t samples, float* grad) {
    const std::uint32_t j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j > shape.hd2) return;
    if (j == shape.hd2) {
        float sum = 0.0f;
        for (std::uint32_t b = 0; b < samples; ++b) sum += dy[b];
        grad[OutputBias(shape)] = sum;
        return;
    }
    float sum = 0.0f;
    for (std::uint32_t b = 0; b < samples; ++b) sum += final_act[static_cast<std::uint64_t>(b) * shape.hd2 + j] * dy[b];
    grad[OutputWeight(shape) + j] = sum;
}__global__ void AddOutputBiasLossDyKernel(CudaMlpShape shape, const float* bias, const float* labels, std::uint32_t samples, float* output, float* loss_terms, float* dy) {
    const std::uint64_t item = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::uint64_t total = static_cast<std::uint64_t>(samples) * shape.output_dim;
    if (item >= total) return;
    const std::uint32_t out = static_cast<std::uint32_t>(item % shape.output_dim);
    const float y = output[item] + bias[out];
    output[item] = y;
    const float diff = y - labels[item];
    const float inv_n = 1.0f / static_cast<float>(total);
    dy[item] = 2.0f * diff * inv_n;
    loss_terms[item] = diff * diff * inv_n;
}
__global__ void ResidualDzFc2Kernel(CudaMlpShape shape, std::uint32_t samples, std::uint32_t block, const float* block_inputs, const float* rz2, const float* dcur, float* dzfc2) {
    const std::uint64_t item = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::uint64_t total = static_cast<std::uint64_t>(samples) * shape.hd2;
    if (item >= total) return;
    const std::uint32_t b = static_cast<std::uint32_t>(item / shape.hd2);
    const std::uint32_t i = static_cast<std::uint32_t>(item - static_cast<std::uint64_t>(b) * shape.hd2);
    const std::uint64_t batch_h2 = static_cast<std::uint64_t>(samples) * shape.hd2;
    const std::uint64_t off = static_cast<std::uint64_t>(block) * batch_h2 + item;
    const float* input = block_inputs + static_cast<std::uint64_t>(block) * batch_h2 + static_cast<std::uint64_t>(b) * shape.hd2;
    dzfc2[item] = dcur[item] * ReluGradFromPreactivation(input[i] + rz2[off]);
}

__global__ void ResidualDzFc1Kernel(CudaMlpShape shape, std::uint32_t samples, std::uint32_t block, const float* rz1, const float* dra1, float* dzfc1) {
    const std::uint64_t item = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::uint64_t total = static_cast<std::uint64_t>(samples) * shape.hd2;
    if (item >= total) return;
    const std::uint64_t batch_h2 = static_cast<std::uint64_t>(samples) * shape.hd2;
    const std::uint64_t off = static_cast<std::uint64_t>(block) * batch_h2 + item;
    dzfc1[item] = dra1[item] * ReluGradFromPreactivation(rz1[off]);
}

__global__ void AddInPlaceKernel(float* dst, const float* src, std::uint64_t count) {
    const std::uint64_t item = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (item >= count) return;
    dst[item] += src[item];
}

__global__ void ResidualDzFc2HalfKernel(CudaMlpShape shape, std::uint32_t samples, std::uint32_t block, const float* block_inputs, const float* rz2, const float* dcur, float* dzfc2, __half* dzfc2_half) {
    const std::uint64_t item = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::uint64_t total = static_cast<std::uint64_t>(samples) * shape.hd2;
    if (item >= total) return;
    const std::uint32_t b = static_cast<std::uint32_t>(item / shape.hd2);
    const std::uint32_t i = static_cast<std::uint32_t>(item - static_cast<std::uint64_t>(b) * shape.hd2);
    const std::uint64_t batch_h2 = static_cast<std::uint64_t>(samples) * shape.hd2;
    const std::uint64_t off = static_cast<std::uint64_t>(block) * batch_h2 + item;
    const float* input = block_inputs + static_cast<std::uint64_t>(block) * batch_h2 + static_cast<std::uint64_t>(b) * shape.hd2;
    const float v = dcur[item] * ReluGradFromPreactivation(input[i] + rz2[off]);
    dzfc2[item] = v;
    dzfc2_half[item] = __float2half(v);
}

__global__ void ResidualDzFc1HalfKernel(CudaMlpShape shape, std::uint32_t samples, std::uint32_t block, const float* rz1, const float* dra1, float* dzfc1, __half* dzfc1_half) {
    const std::uint64_t item = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::uint64_t total = static_cast<std::uint64_t>(samples) * shape.hd2;
    if (item >= total) return;
    const std::uint64_t batch_h2 = static_cast<std::uint64_t>(samples) * shape.hd2;
    const std::uint64_t off = static_cast<std::uint64_t>(block) * batch_h2 + item;
    const float v = dra1[item] * ReluGradFromPreactivation(rz1[off]);
    dzfc1[item] = v;
    dzfc1_half[item] = __float2half(v);
}
__global__ void HiddenDz2Kernel(CudaMlpShape shape, const float* z2, const float* dcur, std::uint32_t samples, float* dz2) {
    const std::uint64_t item = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::uint64_t total = static_cast<std::uint64_t>(samples) * shape.hd2;
    if (item >= total) return;
    dz2[item] = dcur[item] * ReluGradFromPreactivation(z2[item]);
}

__global__ void HiddenDz1Kernel(const float* a1, const float* da1, std::uint32_t samples, std::uint32_t hd1, float* dz1) {
    const std::uint64_t item = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::uint64_t total = static_cast<std::uint64_t>(samples) * hd1;
    if (item >= total) return;
    dz1[item] = da1[item] * ReluGradFromActivation(a1[item]);
}

__global__ void HiddenDz2HalfKernel(CudaMlpShape shape, const float* z2, const float* dcur, std::uint32_t samples, float* dz2, __half* dz2_half) {
    const std::uint64_t item = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::uint64_t total = static_cast<std::uint64_t>(samples) * shape.hd2;
    if (item >= total) return;
    const float v = dcur[item] * ReluGradFromPreactivation(z2[item]);
    dz2[item] = v;
    dz2_half[item] = __float2half(v);
}

__global__ void HiddenDz1HalfKernel(const float* a1, const float* da1, std::uint32_t samples, std::uint32_t hd1, float* dz1, __half* dz1_half) {
    const std::uint64_t item = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::uint64_t total = static_cast<std::uint64_t>(samples) * hd1;
    if (item >= total) return;
    const float v = da1[item] * ReluGradFromActivation(a1[item]);
    dz1[item] = v;
    dz1_half[item] = __float2half(v);
}
inline constexpr std::uint32_t kInputGradThreads = 96;
inline constexpr std::uint32_t kInputGradSharedStride = mgt::kStateValuePad + 1;

__global__ void InputGradByPositionHiddenTileKernel(CudaMlpShape shape, const mgt::TrainStateStorage* states, const float* dz1, std::uint32_t samples, float* grad) {
    extern __shared__ float shared_sums[];
    const std::uint32_t h = blockIdx.x * blockDim.x + threadIdx.x;
    const std::uint32_t pos = blockIdx.y;
    if (pos >= shape.state_len) return;

    float* value_sums = shared_sums + static_cast<std::uint32_t>(threadIdx.x) * kInputGradSharedStride;
    const bool active_h = h < shape.hd1;
    const std::uint32_t lane = threadIdx.x & 31U;
    for (std::uint32_t value = 0; value < shape.state_value_pad; ++value) value_sums[value] = 0.0f;
    for (std::uint32_t b = 0; b < samples; ++b) {
        std::uint32_t value = 0;
        if (lane == 0) value = states[b].v[pos];
        value = __shfl_sync(0xffffffffU, value, 0);
        if (active_h && value < shape.state_value_pad) {
            value_sums[value] += dz1[static_cast<std::uint64_t>(b) * shape.hd1 + h];
        }
    }
    if (!active_h) return;

    for (std::uint32_t value = 0; value < shape.state_value_pad; ++value) {
        const std::uint64_t row = static_cast<std::uint64_t>(pos) * shape.state_value_pad + value;
        grad[row * shape.hd1 + h] = value_sums[value];
    }
}
__global__ void InputGradByPositionGroupHiddenTileKernel(CudaMlpShape shape, const mgt::TrainStateStorage* states, const float* dz1, std::uint32_t samples, std::uint32_t positions_per_block, float* grad) {
    extern __shared__ float shared_sums[];
    const std::uint32_t h = blockIdx.x * blockDim.x + threadIdx.x;
    const std::uint32_t first_pos = blockIdx.y * positions_per_block;
    const bool active_h = h < shape.hd1;
    const std::uint32_t lane = threadIdx.x & 31U;

    for (std::uint32_t group = 0; group < positions_per_block; ++group) {
        const std::uint32_t pos = first_pos + group;
        if (pos >= shape.state_len) continue;
        float* value_sums = shared_sums + (static_cast<std::uint32_t>(group) * blockDim.x + threadIdx.x) * kInputGradSharedStride;
        for (std::uint32_t value = 0; value < shape.state_value_pad; ++value) value_sums[value] = 0.0f;
    }

    for (std::uint32_t b = 0; b < samples; ++b) {
        const float dz = active_h ? dz1[static_cast<std::uint64_t>(b) * shape.hd1 + h] : 0.0f;
        for (std::uint32_t group = 0; group < positions_per_block; ++group) {
            const std::uint32_t pos = first_pos + group;
            if (pos >= shape.state_len) continue;
            std::uint32_t value = 0;
            if (lane == 0) value = states[b].v[pos];
            value = __shfl_sync(0xffffffffU, value, 0);
            if (active_h && value < shape.state_value_pad) {
                float* value_sums = shared_sums + (static_cast<std::uint32_t>(group) * blockDim.x + threadIdx.x) * kInputGradSharedStride;
                value_sums[value] += dz;
            }
        }
    }
    if (!active_h) return;

    for (std::uint32_t group = 0; group < positions_per_block; ++group) {
        const std::uint32_t pos = first_pos + group;
        if (pos >= shape.state_len) continue;
        const float* value_sums = shared_sums + (static_cast<std::uint32_t>(group) * blockDim.x + threadIdx.x) * kInputGradSharedStride;
        for (std::uint32_t value = 0; value < shape.state_value_pad; ++value) {
            const std::uint64_t row = static_cast<std::uint64_t>(pos) * shape.state_value_pad + value;
            grad[row * shape.hd1 + h] = value_sums[value];
        }
    }
}

inline constexpr std::uint32_t kInputGradHalf2Threads = 64;

__global__ void InputGradByPositionHiddenPairHalfKernel(CudaMlpShape shape, const mgt::TrainStateStorage* states, const __half* dz1, std::uint32_t samples, float* grad) {
    extern __shared__ float shared_sums[];
    const std::uint32_t h0 = (blockIdx.x * blockDim.x + threadIdx.x) * 2U;
    const std::uint32_t pos = blockIdx.y;
    if (pos >= shape.state_len) return;

    float* value_sums0 = shared_sums + static_cast<std::uint32_t>(threadIdx.x) * 2U * kInputGradSharedStride;
    float* value_sums1 = value_sums0 + kInputGradSharedStride;
    const bool active_h0 = h0 < shape.hd1;
    const bool active_h1 = h0 + 1U < shape.hd1;
    const std::uint32_t lane = threadIdx.x & 31U;
    for (std::uint32_t value = 0; value < shape.state_value_pad; ++value) {
        value_sums0[value] = 0.0f;
        value_sums1[value] = 0.0f;
    }

    for (std::uint32_t b = 0; b < samples; ++b) {
        std::uint32_t value = 0;
        if (lane == 0) value = states[b].v[pos];
        value = __shfl_sync(0xffffffffU, value, 0);
        if (value >= shape.state_value_pad || !active_h0) continue;
        const __half* row = dz1 + static_cast<std::uint64_t>(b) * shape.hd1 + h0;
        if (active_h1) {
            const __half2 pair = __halves2half2(row[0], row[1]);
            value_sums0[value] += __half2float(__low2half(pair));
            value_sums1[value] += __half2float(__high2half(pair));
        } else {
            value_sums0[value] += __half2float(row[0]);
        }
    }

    if (!active_h0) return;
    for (std::uint32_t value = 0; value < shape.state_value_pad; ++value) {
        const std::uint64_t row = static_cast<std::uint64_t>(pos) * shape.state_value_pad + value;
        grad[row * shape.hd1 + h0] = value_sums0[value];
        if (active_h1) grad[row * shape.hd1 + h0 + 1U] = value_sums1[value];
    }
}

std::size_t InputGradHalf2SharedBytes(std::uint32_t threads) {
    return static_cast<std::size_t>(threads) * 2U * kInputGradSharedStride * sizeof(float);
}
std::size_t InputGradGroupSharedBytes(std::uint32_t positions_per_block) {
    return static_cast<std::size_t>(positions_per_block) * kInputGradThreads * kInputGradSharedStride * sizeof(float);
}

std::uint32_t ResolveInputGradPositionsPerBlock(CudaMlpShape shape, std::uint32_t requested_positions_per_block, std::size_t* shared_bytes) {
    if (shared_bytes == nullptr || requested_positions_per_block <= 1U || shape.state_len <= 1U) return 1U;
    std::uint32_t positions_per_block = requested_positions_per_block < shape.state_len ? requested_positions_per_block : shape.state_len;
    int device = 0;
    if (cudaGetDevice(&device) != cudaSuccess) return 1U;
    int max_shared = 0;
    if (cudaDeviceGetAttribute(&max_shared, cudaDevAttrMaxSharedMemoryPerBlockOptin, device) != cudaSuccess || max_shared <= 0) {
        if (cudaDeviceGetAttribute(&max_shared, cudaDevAttrMaxSharedMemoryPerBlock, device) != cudaSuccess || max_shared <= 0) return 1U;
    }
    while (positions_per_block > 1U) {
        const std::size_t bytes = InputGradGroupSharedBytes(positions_per_block);
        if (bytes <= static_cast<std::size_t>(max_shared) &&
            cudaFuncSetAttribute(InputGradByPositionGroupHiddenTileKernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(bytes)) == cudaSuccess) {
            *shared_bytes = bytes;
            return positions_per_block;
        }
        cudaGetLastError();
        --positions_per_block;
    }
    return 1U;
}
__global__ void InputGradPartialByChunkKernel(CudaMlpShape shape, const mgt::TrainStateStorage* states, const float* dz1, std::uint32_t samples, std::uint32_t chunk_size, float* partials) {
    extern __shared__ float shared_sums[];
    const std::uint32_t h = blockIdx.x * blockDim.x + threadIdx.x;
    const std::uint32_t pos = blockIdx.y;
    const std::uint32_t chunk = blockIdx.z;
    if (pos >= shape.state_len) return;

    float* value_sums = shared_sums + static_cast<std::uint32_t>(threadIdx.x) * kInputGradSharedStride;
    const bool active_h = h < shape.hd1;
    const std::uint32_t lane = threadIdx.x & 31U;
    const std::uint32_t begin = chunk * chunk_size;
    const std::uint32_t end = begin + chunk_size < samples ? begin + chunk_size : samples;
    for (std::uint32_t value = 0; value < shape.state_value_pad; ++value) value_sums[value] = 0.0f;
    for (std::uint32_t b = begin; b < end; ++b) {
        std::uint32_t value = 0;
        if (lane == 0) value = states[b].v[pos];
        value = __shfl_sync(0xffffffffU, value, 0);
        if (active_h && value < shape.state_value_pad) {
            value_sums[value] += dz1[static_cast<std::uint64_t>(b) * shape.hd1 + h];
        }
    }
    if (!active_h) return;

    const std::uint64_t chunk_base = static_cast<std::uint64_t>(chunk) * shape.state_len * shape.state_value_pad * shape.hd1;
    for (std::uint32_t value = 0; value < shape.state_value_pad; ++value) {
        const std::uint64_t row = static_cast<std::uint64_t>(pos) * shape.state_value_pad + value;
        partials[chunk_base + row * shape.hd1 + h] = value_sums[value];
    }
}

__global__ void InputGradPartialByChunkHalfKernel(CudaMlpShape shape, const mgt::TrainStateStorage* states, const __half* dz1, std::uint32_t samples, std::uint32_t chunk_size, float* partials) {
    extern __shared__ float shared_sums[];
    const std::uint32_t h = blockIdx.x * blockDim.x + threadIdx.x;
    const std::uint32_t pos = blockIdx.y;
    const std::uint32_t chunk = blockIdx.z;
    if (pos >= shape.state_len) return;

    float* value_sums = shared_sums + static_cast<std::uint32_t>(threadIdx.x) * kInputGradSharedStride;
    const bool active_h = h < shape.hd1;
    const std::uint32_t lane = threadIdx.x & 31U;
    const std::uint32_t begin = chunk * chunk_size;
    const std::uint32_t end = begin + chunk_size < samples ? begin + chunk_size : samples;
    for (std::uint32_t value = 0; value < shape.state_value_pad; ++value) value_sums[value] = 0.0f;
    for (std::uint32_t b = begin; b < end; ++b) {
        std::uint32_t value = 0;
        if (lane == 0) value = states[b].v[pos];
        value = __shfl_sync(0xffffffffU, value, 0);
        if (active_h && value < shape.state_value_pad) {
            value_sums[value] += __half2float(dz1[static_cast<std::uint64_t>(b) * shape.hd1 + h]);
        }
    }
    if (!active_h) return;

    const std::uint64_t chunk_base = static_cast<std::uint64_t>(chunk) * shape.state_len * shape.state_value_pad * shape.hd1;
    for (std::uint32_t value = 0; value < shape.state_value_pad; ++value) {
        const std::uint64_t row = static_cast<std::uint64_t>(pos) * shape.state_value_pad + value;
        partials[chunk_base + row * shape.hd1 + h] = value_sums[value];
    }
}

std::size_t InputGradPartialSharedBytes(std::uint32_t threads) {
    return static_cast<std::size_t>(threads) * kInputGradSharedStride * sizeof(float);
}

__global__ void InputGradReducePartialsKernel(CudaMlpShape shape, const float* partials, std::uint32_t chunk_count, float* grad) {
    const std::uint32_t h = blockIdx.x * blockDim.x + threadIdx.x;
    const std::uint32_t row = blockIdx.y;
    const std::uint32_t row_count = shape.state_len * shape.state_value_pad;
    if (h >= shape.hd1 || row >= row_count) return;
    float sum = 0.0f;
    const std::uint64_t chunk_stride = static_cast<std::uint64_t>(row_count) * shape.hd1;
    for (std::uint32_t chunk = 0; chunk < chunk_count; ++chunk) {
        sum += partials[static_cast<std::uint64_t>(chunk) * chunk_stride + static_cast<std::uint64_t>(row) * shape.hd1 + h];
    }
    grad[static_cast<std::uint64_t>(row) * shape.hd1 + h] = sum;
}


}  // namespace

__host__ void ConfigureLtMatmulAutotune(const LtMatmulAutotuneConfig& config) {
    g_lt_matmul_autotune_config = NormalizeLtMatmulAutotuneConfig(config);
}

__host__ LtMatmulAutotuneConfig CurrentLtMatmulAutotuneConfig() {
    return NormalizeLtMatmulAutotuneConfig(g_lt_matmul_autotune_config);
}

__host__ std::uint64_t MlpLossGradWorkspaceFloats(const CudaMlpShape& shape,
                                                  std::uint32_t sample_count) {
    return MlpLossGradWorkspaceFloats(shape, sample_count, mgt::kInputGradPartialChunks);
}

__host__ std::uint64_t MlpLossGradWorkspaceFloats(const CudaMlpShape& shape,
                                                  std::uint32_t sample_count,
                                                  std::uint32_t input_grad_partial_chunks) {
    if (ValidateCudaMlpShape(shape) != mgt::Status::kOk || sample_count == 0 || input_grad_partial_chunks == 0) return 0;
    return WorkspaceFloats(shape, sample_count, input_grad_partial_chunks, false, false);
}

__host__ std::uint64_t MlpLossGradWorkspaceFloats(const CudaMlpShape& shape,
                                                  std::uint32_t sample_count,
                                                  std::uint32_t input_grad_partial_chunks,
                                                  bool use_half_input_grad) {
    if (ValidateCudaMlpShape(shape) != mgt::Status::kOk || sample_count == 0 || input_grad_partial_chunks == 0) return 0;
    return WorkspaceFloats(shape, sample_count, input_grad_partial_chunks, use_half_input_grad, false);
}

__host__ std::uint64_t MlpLossGradWorkspaceFloats(const CudaMlpShape& shape,
                                                  std::uint32_t sample_count,
                                                  std::uint32_t input_grad_partial_chunks,
                                                  bool use_half_input_grad,
                                                  bool use_half_linear) {
    if (ValidateCudaMlpShape(shape) != mgt::Status::kOk || sample_count == 0 || input_grad_partial_chunks == 0) return 0;
    return WorkspaceFloats(shape, sample_count, input_grad_partial_chunks, use_half_input_grad, use_half_linear);
}
mgt::Status LaunchMlpLossGradKernelWithWorkspaceInternal(const CudaMlpShape& shape,
                                                          const float* device_weights,
                                                          const mgt::TrainStateStorage* device_states,
                                                          const float* device_labels,
                                                          std::uint32_t sample_count,
                                                          float* device_loss,
                                                          float* device_grad,
                                                          float* workspace_base,
                                                          std::uint64_t workspace_floats,
                                                          cublasHandle_t blas,
                                                         cublasLtHandle_t blas_lt,
                                                         const __half* external_weights_half,
                                                         std::uint32_t input_grad_partial_chunks,
                                                         std::uint32_t input_grad_positions_per_block,
                                                         bool use_half_input_grad,
                                                         bool use_half_linear,
                                                         MlpBackwardProfile* profile,
                                                         MlpGradientReadyCallback gradient_ready,
                                                         void* gradient_ready_user,
                                                         cudaStream_t stream,
                                                         void* lt_workspace_base = nullptr,
                                                         std::uint64_t lt_workspace_bytes = 0,
                                                         std::uint32_t input_grad_position_tile = 0, bool input_grad_sparse = false) {
    if (ValidateCudaMlpShape(shape) != mgt::Status::kOk || device_weights == nullptr || device_states == nullptr || device_labels == nullptr || device_loss == nullptr || device_grad == nullptr || sample_count == 0 || workspace_base == nullptr || blas == nullptr || input_grad_partial_chunks == 0 || input_grad_positions_per_block == 0) return mgt::Status::kInvalidConfig;
    if ((use_half_input_grad || use_half_linear) && blas_lt == nullptr) return mgt::Status::kInvalidConfig;
    if (lt_workspace_bytes > 0 && lt_workspace_base == nullptr) return mgt::Status::kInvalidConfig;
    const std::uint64_t param_count = ParamCount(shape);
    const std::uint64_t required_workspace_floats = WorkspaceFloats(shape, sample_count, input_grad_partial_chunks, use_half_input_grad, use_half_linear);
    if (workspace_floats < required_workspace_floats) return mgt::Status::kCapacityExceeded;
    if (cublasSetStream(blas, stream) != CUBLAS_STATUS_SUCCESS) return mgt::Status::kCudaFailure;
    if (cublasSetMathMode(blas, CUBLAS_TF32_TENSOR_OP_MATH) != CUBLAS_STATUS_SUCCESS) return mgt::Status::kCudaFailure;
    Workspace w = MakeWorkspace(workspace_base, shape, sample_count, input_grad_partial_chunks, use_half_input_grad, use_half_linear);
    if (use_half_linear && external_weights_half != nullptr) w.weights_half = const_cast<__half*>(external_weights_half);
    BackwardStageTimer stage_timer;
    if (stage_timer.Start(stream, profile) != mgt::Status::kOk) return mgt::Status::kCudaFailure;
    std::uint32_t gradient_ready_id = 0;

    const DeviceLaunchConfig h1_launch = Build1DLaunchConfig(static_cast<std::uint64_t>(sample_count) * shape.hd1, 128);
    const DeviceLaunchConfig h2_launch = Build1DLaunchConfig(static_cast<std::uint64_t>(sample_count) * shape.hd2, 128);
    if (use_half_linear && external_weights_half == nullptr) {
        const DeviceLaunchConfig weight_half_launch = Build1DLaunchConfig(param_count, 256);
        FloatToHalfKernel<<<weight_half_launch.blocks, weight_half_launch.threads, 0, stream>>>(device_weights, w.weights_half, param_count);
        if (!LaunchOk()) return mgt::Status::kCudaFailure;
    }
    const dim3 input_forward_grid(sample_count, (shape.hd1 + kInputForwardThreads * kInputForwardVec - 1U) / (kInputForwardThreads * kInputForwardVec));
    if (use_half_linear) {
        InputForwardHalfKernel<<<input_forward_grid, kInputForwardThreads, 0, stream>>>(shape, w.weights_half, device_states, sample_count, w.a1, w.a1_half);
    } else {
        InputForwardKernel<<<input_forward_grid, kInputForwardThreads, 0, stream>>>(shape, device_weights, device_states, sample_count, w.a1);
    }
    if (!LaunchOk()) return mgt::Status::kCudaFailure;
    if (stage_timer.Mark(profile == nullptr ? nullptr : &profile->input_forward_ms) != mgt::Status::kOk) return mgt::Status::kCudaFailure;

    if (use_half_linear) {
        if (GemmRowMajorHalfToFloat(blas_lt, stream, lt_workspace_base, lt_workspace_bytes, w.a1_half, w.weights_half + HiddenWeight(shape), w.z2, sample_count, shape.hd2, shape.hd1) != mgt::Status::kOk) return mgt::Status::kCudaFailure;
    } else {
        if (GemmRowMajor(blas, w.a1, device_weights + HiddenWeight(shape), w.z2, sample_count, shape.hd2, shape.hd1) != mgt::Status::kOk) return mgt::Status::kCudaFailure;
    }
    if (use_half_linear) {
        AddBiasReluCopyHalfKernel<<<h2_launch.blocks, h2_launch.threads, 0, stream>>>(w.z2, device_weights + HiddenBias(shape), sample_count, shape.hd2, w.block_inputs, w.block_inputs_half);
    } else {
        AddBiasReluCopyKernel<<<h2_launch.blocks, h2_launch.threads, 0, stream>>>(w.z2, device_weights + HiddenBias(shape), sample_count, shape.hd2, w.block_inputs);
    }
    if (!LaunchOk()) return mgt::Status::kCudaFailure;
    if (stage_timer.Mark(profile == nullptr ? nullptr : &profile->hidden_forward_ms) != mgt::Status::kOk) return mgt::Status::kCudaFailure;

    for (std::uint32_t block = 0; block < shape.residual_blocks; ++block) {
        const std::uint64_t batch_h2 = static_cast<std::uint64_t>(sample_count) * shape.hd2;
        const float* block_in = w.block_inputs + static_cast<std::uint64_t>(block) * batch_h2;
        float* block_rz1 = w.rz1 + static_cast<std::uint64_t>(block) * batch_h2;
        float* block_rz2 = w.rz2 + static_cast<std::uint64_t>(block) * batch_h2;
        float* block_out = w.block_inputs + static_cast<std::uint64_t>(block + 1U) * batch_h2;
        __half* block_in_half = use_half_linear ? w.block_inputs_half + static_cast<std::uint64_t>(block) * batch_h2 : nullptr;
        __half* block_ra1_half = use_half_linear ? w.ra1_half + static_cast<std::uint64_t>(block) * batch_h2 : nullptr;
        __half* block_out_half = use_half_linear ? w.block_inputs_half + static_cast<std::uint64_t>(block + 1U) * batch_h2 : nullptr;
        if (use_half_linear) {
            if (GemmRowMajorHalfToFloat(blas_lt, stream, lt_workspace_base, lt_workspace_bytes, block_in_half, w.weights_half + ResidualFc1Weight(shape, block), block_rz1, sample_count, shape.hd2, shape.hd2) != mgt::Status::kOk) return mgt::Status::kCudaFailure;
        } else {
            if (GemmRowMajor(blas, block_in, device_weights + ResidualFc1Weight(shape, block), block_rz1, sample_count, shape.hd2, shape.hd2) != mgt::Status::kOk) return mgt::Status::kCudaFailure;
        }
        if (use_half_linear) {
            AddBiasReluHalfKernel<<<h2_launch.blocks, h2_launch.threads, 0, stream>>>(block_rz1, device_weights + ResidualFc1Bias(shape, block), sample_count, shape.hd2, block_ra1_half);
            if (!LaunchOk()) return mgt::Status::kCudaFailure;
            if (GemmRowMajorHalfToFloat(blas_lt, stream, lt_workspace_base, lt_workspace_bytes, block_ra1_half, w.weights_half + ResidualFc2Weight(shape, block), block_rz2, sample_count, shape.hd2, shape.hd2) != mgt::Status::kOk) return mgt::Status::kCudaFailure;
        } else {
            AddBiasReluKernel<<<h2_launch.blocks, h2_launch.threads, 0, stream>>>(block_rz1, device_weights + ResidualFc1Bias(shape, block), sample_count, shape.hd2);
            if (!LaunchOk()) return mgt::Status::kCudaFailure;
            if (GemmRowMajor(blas, block_rz1, device_weights + ResidualFc2Weight(shape, block), block_rz2, sample_count, shape.hd2, shape.hd2) != mgt::Status::kOk) return mgt::Status::kCudaFailure;
        }
        if (use_half_linear) {
            AddBiasResidualReluHalfKernel<<<h2_launch.blocks, h2_launch.threads, 0, stream>>>(block_rz2, device_weights + ResidualFc2Bias(shape, block), block_in, sample_count, shape.hd2, block_out, block_out_half);
        } else {
            AddBiasResidualReluKernel<<<h2_launch.blocks, h2_launch.threads, 0, stream>>>(block_rz2, device_weights + ResidualFc2Bias(shape, block), block_in, sample_count, shape.hd2, block_out);
        }
        if (!LaunchOk()) return mgt::Status::kCudaFailure;
    }

    if (stage_timer.Mark(profile == nullptr ? nullptr : &profile->residual_forward_ms) != mgt::Status::kOk) return mgt::Status::kCudaFailure;

    const std::uint64_t final_batch_h2 = static_cast<std::uint64_t>(sample_count) * shape.hd2;
    const float* final_act = w.block_inputs + static_cast<std::uint64_t>(shape.residual_blocks) * final_batch_h2;
    const __half* final_act_half = use_half_linear ? w.block_inputs_half + static_cast<std::uint64_t>(shape.residual_blocks) * final_batch_h2 : nullptr;
    const DeviceLaunchConfig sample_launch = Build1DLaunchConfig(sample_count, 128);
    FillOnesKernel<<<sample_launch.blocks, sample_launch.threads, 0, stream>>>(w.ones, sample_count);
    if (!LaunchOk()) return mgt::Status::kCudaFailure;
    if (shape.output_dim == 1U) {
        OutputBackwardInitKernel<<<sample_launch.blocks, sample_launch.threads, 0, stream>>>(shape, device_weights, device_labels, sample_count, final_act, w.loss_terms, w.dy, w.dcur);
        if (!LaunchOk()) return mgt::Status::kCudaFailure;
        ReduceLossKernel<<<1, 256, 0, stream>>>(w.loss_terms, sample_count, device_loss);
        if (!LaunchOk()) return mgt::Status::kCudaFailure;
        const DeviceLaunchConfig output_grad_launch = Build1DLaunchConfig(shape.hd2 + 1ULL, 128);
        OutputGradKernel<<<output_grad_launch.blocks, output_grad_launch.threads, 0, stream>>>(shape, final_act, w.dy, sample_count, device_grad);
        if (!LaunchOk()) return mgt::Status::kCudaFailure;
    } else {
        const std::uint64_t output_items = static_cast<std::uint64_t>(sample_count) * shape.output_dim;
        const DeviceLaunchConfig output_items_launch = Build1DLaunchConfig(output_items, 128);
        if (use_half_linear) {
            if (GemmRowMajorHalfToFloat(blas_lt, stream, lt_workspace_base, lt_workspace_bytes, final_act_half, w.weights_half + OutputWeight(shape), w.output, sample_count, shape.output_dim, shape.hd2) != mgt::Status::kOk) return mgt::Status::kCudaFailure;
        } else {
            if (GemmRowMajor(blas, final_act, device_weights + OutputWeight(shape), w.output, sample_count, shape.output_dim, shape.hd2) != mgt::Status::kOk) return mgt::Status::kCudaFailure;
        }
        AddOutputBiasLossDyKernel<<<output_items_launch.blocks, output_items_launch.threads, 0, stream>>>(shape, device_weights + OutputBias(shape), device_labels, sample_count, w.output, w.loss_terms, w.dy);
        if (!LaunchOk()) return mgt::Status::kCudaFailure;
        ReduceLossKernel<<<1, 256, 0, stream>>>(w.loss_terms, static_cast<std::uint32_t>(output_items), device_loss);
        if (!LaunchOk()) return mgt::Status::kCudaFailure;
        if (use_half_linear) {
            FloatToHalfKernel<<<output_items_launch.blocks, output_items_launch.threads, 0, stream>>>(w.dy, w.output_half, output_items);
            if (!LaunchOk()) return mgt::Status::kCudaFailure;
            if (GemmGradWeightsHalf(blas_lt, stream, lt_workspace_base, lt_workspace_bytes, final_act_half, w.output_half, device_grad + OutputWeight(shape), sample_count, shape.hd2, shape.output_dim) != mgt::Status::kOk) return mgt::Status::kCudaFailure;
            if (GemmBackpropInputHalf(blas_lt, stream, lt_workspace_base, lt_workspace_bytes, w.output_half, w.weights_half + OutputWeight(shape), w.dcur, sample_count, shape.hd2, shape.output_dim) != mgt::Status::kOk) return mgt::Status::kCudaFailure;
        } else {
            if (GemmGradWeights(blas, final_act, w.dy, device_grad + OutputWeight(shape), sample_count, shape.hd2, shape.output_dim) != mgt::Status::kOk) return mgt::Status::kCudaFailure;
            if (GemmBackpropInput(blas, w.dy, device_weights + OutputWeight(shape), w.dcur, sample_count, shape.hd2, shape.output_dim) != mgt::Status::kOk) return mgt::Status::kCudaFailure;
        }
        if (GemvBiasGrad(blas, w.dy, w.ones, device_grad + OutputBias(shape), sample_count, shape.output_dim) != mgt::Status::kOk) return mgt::Status::kCudaFailure;
    }
    if (NotifyGradientReady(gradient_ready, gradient_ready_user, &gradient_ready_id, OutputWeight(shape), static_cast<std::uint64_t>(shape.hd2) * shape.output_dim + shape.output_dim, stream) != mgt::Status::kOk) return mgt::Status::kCudaFailure;
    if (stage_timer.Mark(profile == nullptr ? nullptr : &profile->output_ms) != mgt::Status::kOk) return mgt::Status::kCudaFailure;

    for (std::uint32_t rblock = shape.residual_blocks; rblock > 0; --rblock) {
        const std::uint32_t block = rblock - 1U;
        const std::uint64_t batch_h2 = static_cast<std::uint64_t>(sample_count) * shape.hd2;
        const float* block_in = w.block_inputs + static_cast<std::uint64_t>(block) * batch_h2;
        const float* block_ra1 = w.rz1 + static_cast<std::uint64_t>(block) * batch_h2;
        const __half* block_in_half = use_half_linear ? w.block_inputs_half + static_cast<std::uint64_t>(block) * batch_h2 : nullptr;
        const __half* block_ra1_half = use_half_linear ? w.ra1_half + static_cast<std::uint64_t>(block) * batch_h2 : nullptr;

        if (use_half_linear) {
            ResidualDzFc2HalfKernel<<<h2_launch.blocks, h2_launch.threads, 0, stream>>>(shape, sample_count, block, w.block_inputs, w.rz2, w.dcur, w.dzfc2, w.h2_right_half);
            if (!LaunchOk()) return mgt::Status::kCudaFailure;
            if (GemmGradWeightsHalf(blas_lt, stream, lt_workspace_base, lt_workspace_bytes, block_ra1_half, w.h2_right_half, device_grad + ResidualFc2Weight(shape, block), sample_count, shape.hd2, shape.hd2) != mgt::Status::kOk) return mgt::Status::kCudaFailure;
        } else {
            ResidualDzFc2Kernel<<<h2_launch.blocks, h2_launch.threads, 0, stream>>>(shape, sample_count, block, w.block_inputs, w.rz2, w.dcur, w.dzfc2);
            if (!LaunchOk()) return mgt::Status::kCudaFailure;
            if (GemmGradWeights(blas, block_ra1, w.dzfc2, device_grad + ResidualFc2Weight(shape, block), sample_count, shape.hd2, shape.hd2) != mgt::Status::kOk) return mgt::Status::kCudaFailure;
        }
        if (GemvBiasGrad(blas, w.dzfc2, w.ones, device_grad + ResidualFc2Bias(shape, block), sample_count, shape.hd2) != mgt::Status::kOk) return mgt::Status::kCudaFailure;
        if (NotifyGradientReady(gradient_ready, gradient_ready_user, &gradient_ready_id, ResidualFc2Weight(shape, block), static_cast<std::uint64_t>(shape.hd2) * shape.hd2 + shape.hd2, stream) != mgt::Status::kOk) return mgt::Status::kCudaFailure;
        if (use_half_linear) {
            if (GemmBackpropInputHalf(blas_lt, stream, lt_workspace_base, lt_workspace_bytes, w.h2_right_half, w.weights_half + ResidualFc2Weight(shape, block), w.dra1, sample_count, shape.hd2, shape.hd2) != mgt::Status::kOk) return mgt::Status::kCudaFailure;
        } else {
            if (GemmBackpropInput(blas, w.dzfc2, device_weights + ResidualFc2Weight(shape, block), w.dra1, sample_count, shape.hd2, shape.hd2) != mgt::Status::kOk) return mgt::Status::kCudaFailure;
        }

        if (use_half_linear) {
            ResidualDzFc1HalfKernel<<<h2_launch.blocks, h2_launch.threads, 0, stream>>>(shape, sample_count, block, w.rz1, w.dra1, w.dzfc1, w.h2_right_half);
            if (!LaunchOk()) return mgt::Status::kCudaFailure;
            if (GemmGradWeightsHalf(blas_lt, stream, lt_workspace_base, lt_workspace_bytes, block_in_half, w.h2_right_half, device_grad + ResidualFc1Weight(shape, block), sample_count, shape.hd2, shape.hd2) != mgt::Status::kOk) return mgt::Status::kCudaFailure;
        } else {
            ResidualDzFc1Kernel<<<h2_launch.blocks, h2_launch.threads, 0, stream>>>(shape, sample_count, block, w.rz1, w.dra1, w.dzfc1);
            if (!LaunchOk()) return mgt::Status::kCudaFailure;
            if (GemmGradWeights(blas, block_in, w.dzfc1, device_grad + ResidualFc1Weight(shape, block), sample_count, shape.hd2, shape.hd2) != mgt::Status::kOk) return mgt::Status::kCudaFailure;
        }
        if (GemvBiasGrad(blas, w.dzfc1, w.ones, device_grad + ResidualFc1Bias(shape, block), sample_count, shape.hd2) != mgt::Status::kOk) return mgt::Status::kCudaFailure;
        if (NotifyGradientReady(gradient_ready, gradient_ready_user, &gradient_ready_id, ResidualFc1Weight(shape, block), static_cast<std::uint64_t>(shape.hd2) * shape.hd2 + shape.hd2, stream) != mgt::Status::kOk) return mgt::Status::kCudaFailure;
        if (use_half_linear) {
            if (GemmBackpropInputHalf(blas_lt, stream, lt_workspace_base, lt_workspace_bytes, w.h2_right_half, w.weights_half + ResidualFc1Weight(shape, block), w.dprev, sample_count, shape.hd2, shape.hd2) != mgt::Status::kOk) return mgt::Status::kCudaFailure;
        } else {
            if (GemmBackpropInput(blas, w.dzfc1, device_weights + ResidualFc1Weight(shape, block), w.dprev, sample_count, shape.hd2, shape.hd2) != mgt::Status::kOk) return mgt::Status::kCudaFailure;
        }
        AddInPlaceKernel<<<h2_launch.blocks, h2_launch.threads, 0, stream>>>(w.dprev, w.dzfc2, batch_h2);
        if (!LaunchOk()) return mgt::Status::kCudaFailure;
        float* tmp = w.dcur;
        w.dcur = w.dprev;
        w.dprev = tmp;
    }

    if (stage_timer.Mark(profile == nullptr ? nullptr : &profile->residual_backward_ms) != mgt::Status::kOk) return mgt::Status::kCudaFailure;

    if (use_half_linear) {
        HiddenDz2HalfKernel<<<h2_launch.blocks, h2_launch.threads, 0, stream>>>(shape, w.z2, w.dcur, sample_count, w.dz2, w.h2_right_half);
        if (!LaunchOk()) return mgt::Status::kCudaFailure;
        if (GemmGradWeightsHalf(blas_lt, stream, lt_workspace_base, lt_workspace_bytes, w.a1_half, w.h2_right_half, device_grad + HiddenWeight(shape), sample_count, shape.hd1, shape.hd2) != mgt::Status::kOk) return mgt::Status::kCudaFailure;
    } else {
        HiddenDz2Kernel<<<h2_launch.blocks, h2_launch.threads, 0, stream>>>(shape, w.z2, w.dcur, sample_count, w.dz2);
        if (!LaunchOk()) return mgt::Status::kCudaFailure;
        if (GemmGradWeights(blas, w.a1, w.dz2, device_grad + HiddenWeight(shape), sample_count, shape.hd1, shape.hd2) != mgt::Status::kOk) return mgt::Status::kCudaFailure;
    }
    if (GemvBiasGrad(blas, w.dz2, w.ones, device_grad + HiddenBias(shape), sample_count, shape.hd2) != mgt::Status::kOk) return mgt::Status::kCudaFailure;
    if (NotifyGradientReady(gradient_ready, gradient_ready_user, &gradient_ready_id, HiddenWeight(shape), static_cast<std::uint64_t>(shape.hd1) * shape.hd2 + shape.hd2, stream) != mgt::Status::kOk) return mgt::Status::kCudaFailure;
    if (use_half_linear) {
        if (GemmBackpropInputHalf(blas_lt, stream, lt_workspace_base, lt_workspace_bytes, w.h2_right_half, w.weights_half + HiddenWeight(shape), w.da1, sample_count, shape.hd1, shape.hd2) != mgt::Status::kOk) return mgt::Status::kCudaFailure;
    } else {
        if (GemmBackpropInput(blas, w.dz2, device_weights + HiddenWeight(shape), w.da1, sample_count, shape.hd1, shape.hd2) != mgt::Status::kOk) return mgt::Status::kCudaFailure;
    }
    if (use_half_input_grad) {
        HiddenDz1HalfKernel<<<h1_launch.blocks, h1_launch.threads, 0, stream>>>(w.a1, w.da1, sample_count, shape.hd1, w.dz1, w.dz1_half);
    } else {
        HiddenDz1Kernel<<<h1_launch.blocks, h1_launch.threads, 0, stream>>>(w.a1, w.da1, sample_count, shape.hd1, w.dz1);
    }
    if (!LaunchOk()) return mgt::Status::kCudaFailure;
    if (stage_timer.Mark(profile == nullptr ? nullptr : &profile->hidden_backward_ms) != mgt::Status::kOk) return mgt::Status::kCudaFailure;
    if (input_grad_sparse) {
        if (use_half_input_grad) {
            const std::size_t input_grad_sparse_shared_bytes = InputGradHalf2SharedBytes(kInputGradHalf2Threads);
            const dim3 input_grad_grid((shape.hd1 + 2U * kInputGradHalf2Threads - 1U) / (2U * kInputGradHalf2Threads), shape.state_len);
            InputGradByPositionHiddenPairHalfKernel<<<input_grad_grid, kInputGradHalf2Threads, input_grad_sparse_shared_bytes, stream>>>(shape, device_states, w.dz1_half, sample_count, device_grad);
        } else {
            const std::size_t input_grad_shared_bytes = kInputGradThreads * kInputGradSharedStride * sizeof(float);
            const dim3 input_grad_grid((shape.hd1 + kInputGradThreads - 1U) / kInputGradThreads, shape.state_len);
            InputGradByPositionHiddenTileKernel<<<input_grad_grid, kInputGradThreads, input_grad_shared_bytes, stream>>>(shape, device_states, w.dz1, sample_count, device_grad);
        }
        if (!LaunchOk()) return mgt::Status::kCudaFailure;
    } else if (input_grad_position_tile > 0U) {
        const std::uint32_t tile_positions_requested = input_grad_position_tile < shape.state_len ? input_grad_position_tile : shape.state_len;
        const std::uint32_t tile_positions = tile_positions_requested == 0U ? 1U : tile_positions_requested;
        for (std::uint32_t first_pos = 0; first_pos < shape.state_len; first_pos += tile_positions) {
            const std::uint32_t active_positions = first_pos + tile_positions <= shape.state_len ? tile_positions : shape.state_len - first_pos;
            const std::uint32_t tile_cols = active_positions * shape.state_value_pad;
            const std::uint64_t tile_items = static_cast<std::uint64_t>(sample_count) * tile_cols;
            const DeviceLaunchConfig tile_launch = Build1DLaunchConfig(tile_items, 256);
            float* grad_tile = device_grad + static_cast<std::uint64_t>(first_pos) * shape.state_value_pad * shape.hd1;
            if (use_half_input_grad) {
                __half* input_one_hot_half = reinterpret_cast<__half*>(w.input_one_hot);
                BuildInputPositionTileOneHotHalfKernel<<<tile_launch.blocks, tile_launch.threads, 0, stream>>>(shape, device_states, sample_count, first_pos, active_positions, input_one_hot_half);
                if (!LaunchOk()) return mgt::Status::kCudaFailure;
                if (GemmGradWeightsHalf(blas_lt, stream, lt_workspace_base, lt_workspace_bytes, input_one_hot_half, w.dz1_half, grad_tile, sample_count, tile_cols, shape.hd1, HalfGemmOpKind::kInputEmbeddingGrad) != mgt::Status::kOk) return mgt::Status::kCudaFailure;
            } else {
                BuildInputPositionTileOneHotKernel<<<tile_launch.blocks, tile_launch.threads, 0, stream>>>(shape, device_states, sample_count, first_pos, active_positions, w.input_one_hot);
                if (!LaunchOk()) return mgt::Status::kCudaFailure;
                if (GemmGradWeights(blas, w.input_one_hot, w.dz1, grad_tile, sample_count, tile_cols, shape.hd1) != mgt::Status::kOk) return mgt::Status::kCudaFailure;
            }
        }
    } else if (input_grad_partial_chunks <= 1U) {
        if (input_grad_positions_per_block <= 1U) {
            const std::uint32_t input_dim = shape.state_len * shape.state_value_pad;
            const std::uint64_t one_hot_items = static_cast<std::uint64_t>(sample_count) * input_dim;
            const DeviceLaunchConfig one_hot_launch = Build1DLaunchConfig(static_cast<std::uint64_t>(sample_count) * shape.state_len, 256);
            if (use_half_input_grad) {
                __half* input_one_hot_half = reinterpret_cast<__half*>(w.input_one_hot);
                if (cudaMemsetAsync(input_one_hot_half, 0, one_hot_items * sizeof(__half), stream) != cudaSuccess) return mgt::Status::kCudaFailure;
                BuildInputOneHotHalfKernel<<<one_hot_launch.blocks, one_hot_launch.threads, 0, stream>>>(shape, device_states, sample_count, input_one_hot_half);
                if (!LaunchOk()) return mgt::Status::kCudaFailure;
                if (GemmGradWeightsHalf(blas_lt, stream, lt_workspace_base, lt_workspace_bytes, input_one_hot_half, w.dz1_half, device_grad, sample_count, input_dim, shape.hd1, HalfGemmOpKind::kInputEmbeddingGrad) != mgt::Status::kOk) return mgt::Status::kCudaFailure;
            } else {
                if (cudaMemsetAsync(w.input_one_hot, 0, one_hot_items * sizeof(float), stream) != cudaSuccess) return mgt::Status::kCudaFailure;
                BuildInputOneHotKernel<<<one_hot_launch.blocks, one_hot_launch.threads, 0, stream>>>(shape, device_states, sample_count, w.input_one_hot);
                if (!LaunchOk()) return mgt::Status::kCudaFailure;
                if (GemmGradWeights(blas, w.input_one_hot, w.dz1, device_grad, sample_count, input_dim, shape.hd1) != mgt::Status::kOk) return mgt::Status::kCudaFailure;
            }
        } else {
            std::size_t input_grad_shared_bytes = kInputGradThreads * kInputGradSharedStride * sizeof(float);
            const std::uint32_t resolved_positions_per_block = ResolveInputGradPositionsPerBlock(shape, input_grad_positions_per_block, &input_grad_shared_bytes);
            if (resolved_positions_per_block <= 1U) {
                const dim3 input_grad_grid((shape.hd1 + kInputGradThreads - 1U) / kInputGradThreads, shape.state_len);
                InputGradByPositionHiddenTileKernel<<<input_grad_grid, kInputGradThreads, input_grad_shared_bytes, stream>>>(shape, device_states, w.dz1, sample_count, device_grad);
            } else {
                const dim3 input_grad_grid((shape.hd1 + kInputGradThreads - 1U) / kInputGradThreads, (shape.state_len + resolved_positions_per_block - 1U) / resolved_positions_per_block);
                InputGradByPositionGroupHiddenTileKernel<<<input_grad_grid, kInputGradThreads, input_grad_shared_bytes, stream>>>(shape, device_states, w.dz1, sample_count, resolved_positions_per_block, device_grad);
            }
            if (!LaunchOk()) return mgt::Status::kCudaFailure;
        }
    } else {
        const std::uint32_t input_grad_chunk_size = (sample_count + input_grad_partial_chunks - 1U) / input_grad_partial_chunks;
        const std::uint32_t input_grad_chunk_count = (sample_count + input_grad_chunk_size - 1U) / input_grad_chunk_size;
        constexpr std::uint32_t input_grad_chunk_threads = 128;
        const dim3 input_grad_h_grid((shape.hd1 + input_grad_chunk_threads - 1U) / input_grad_chunk_threads, shape.state_len, input_grad_chunk_count);
        const std::size_t input_grad_partial_shared_bytes = InputGradPartialSharedBytes(input_grad_chunk_threads);
        if (use_half_input_grad) {
            InputGradPartialByChunkHalfKernel<<<input_grad_h_grid, input_grad_chunk_threads, input_grad_partial_shared_bytes, stream>>>(shape, device_states, w.dz1_half, sample_count, input_grad_chunk_size, w.input_grad_partials);
        } else {
            InputGradPartialByChunkKernel<<<input_grad_h_grid, input_grad_chunk_threads, input_grad_partial_shared_bytes, stream>>>(shape, device_states, w.dz1, sample_count, input_grad_chunk_size, w.input_grad_partials);
        }
        if (!LaunchOk()) return mgt::Status::kCudaFailure;
        const dim3 input_grad_reduce_grid((shape.hd1 + input_grad_chunk_threads - 1U) / input_grad_chunk_threads, shape.state_len * shape.state_value_pad);
        InputGradReducePartialsKernel<<<input_grad_reduce_grid, input_grad_chunk_threads, 0, stream>>>(shape, w.input_grad_partials, input_grad_chunk_count, device_grad);
        if (!LaunchOk()) return mgt::Status::kCudaFailure;
    }
    if (GemvBiasGrad(blas, w.dz1, w.ones, device_grad + InputBias(shape), sample_count, shape.hd1) != mgt::Status::kOk) return mgt::Status::kCudaFailure;
    if (NotifyGradientReady(gradient_ready, gradient_ready_user, &gradient_ready_id, 0ULL, InputBias(shape) + shape.hd1, stream) != mgt::Status::kOk) return mgt::Status::kCudaFailure;
    if (stage_timer.Mark(profile == nullptr ? nullptr : &profile->input_grad_ms) != mgt::Status::kOk) return mgt::Status::kCudaFailure;

    return mgt::Status::kOk;
}

__host__ mgt::Status LaunchMlpLossGradKernelWithWorkspace(const CudaMlpShape& shape,
                                                          const float* device_weights,
                                                          const mgt::TrainStateStorage* device_states,
                                                          const float* device_labels,
                                                          std::uint32_t sample_count,
                                                          float* device_loss,
                                                          float* device_grad,
                                                          float* workspace_base,
                                                          std::uint64_t workspace_floats,
                                                          cublasHandle_t blas,
                                                          std::uint32_t input_grad_partial_chunks,
                                                          cudaStream_t stream) {
    return LaunchMlpLossGradKernelWithWorkspaceInternal(shape, device_weights, device_states, device_labels, sample_count, device_loss, device_grad, workspace_base, workspace_floats, blas, nullptr, nullptr, input_grad_partial_chunks, mgt::kInputGradPositionsPerBlock, false, false, nullptr, nullptr, nullptr, stream);
}

__host__ mgt::Status LaunchMlpLossGradKernelWithWorkspace(const CudaMlpShape& shape,
                                                          const float* device_weights,
                                                          const mgt::TrainStateStorage* device_states,
                                                          const float* device_labels,
                                                          std::uint32_t sample_count,
                                                          float* device_loss,
                                                          float* device_grad,
                                                          float* workspace_base,
                                                          std::uint64_t workspace_floats,
                                                          cublasHandle_t blas,
                                                          std::uint32_t input_grad_partial_chunks,
                                                          std::uint32_t input_grad_positions_per_block,
                                                          cudaStream_t stream) {
    return LaunchMlpLossGradKernelWithWorkspaceInternal(shape, device_weights, device_states, device_labels, sample_count, device_loss, device_grad, workspace_base, workspace_floats, blas, nullptr, nullptr, input_grad_partial_chunks, input_grad_positions_per_block, false, false, nullptr, nullptr, nullptr, stream);
}

__host__ mgt::Status LaunchMlpLossGradKernelWithWorkspace(const CudaMlpShape& shape,
                                                          const float* device_weights,
                                                          const mgt::TrainStateStorage* device_states,
                                                          const float* device_labels,
                                                          std::uint32_t sample_count,
                                                          float* device_loss,
                                                          float* device_grad,
                                                          float* workspace_base,
                                                          std::uint64_t workspace_floats,
                                                          cublasHandle_t blas,
                                                          std::uint32_t input_grad_partial_chunks,
                                                          std::uint32_t input_grad_positions_per_block,
                                                          bool use_half_input_grad,
                                                          cudaStream_t stream) {
    return LaunchMlpLossGradKernelWithWorkspaceInternal(shape, device_weights, device_states, device_labels, sample_count, device_loss, device_grad, workspace_base, workspace_floats, blas, nullptr, nullptr, input_grad_partial_chunks, input_grad_positions_per_block, use_half_input_grad, false, nullptr, nullptr, nullptr, stream);
}

__host__ mgt::Status LaunchMlpLossGradKernelWithWorkspace(const CudaMlpShape& shape,
                                                          const float* device_weights,
                                                          const mgt::TrainStateStorage* device_states,
                                                          const float* device_labels,
                                                          std::uint32_t sample_count,
                                                          float* device_loss,
                                                          float* device_grad,
                                                          float* workspace_base,
                                                          std::uint64_t workspace_floats,
                                                          cublasHandle_t blas,
                                                          std::uint32_t input_grad_partial_chunks,
                                                          std::uint32_t input_grad_positions_per_block,
                                                          bool use_half_input_grad,
                                                          bool use_half_linear,
                                                          cudaStream_t stream) {
    return LaunchMlpLossGradKernelWithWorkspaceInternal(shape, device_weights, device_states, device_labels, sample_count, device_loss, device_grad, workspace_base, workspace_floats, blas, nullptr, nullptr, input_grad_partial_chunks, input_grad_positions_per_block, use_half_input_grad, use_half_linear, nullptr, nullptr, nullptr, stream);
}

__host__ mgt::Status LaunchMlpLossGradKernelWithWorkspace(const CudaMlpShape& shape,
                                                          const float* device_weights,
                                                          const mgt::TrainStateStorage* device_states,
                                                          const float* device_labels,
                                                          std::uint32_t sample_count,
                                                          float* device_loss,
                                                          float* device_grad,
                                                          float* workspace_base,
                                                          std::uint64_t workspace_floats,
                                                          cublasHandle_t blas,
                                                          cudaStream_t stream) {
    return LaunchMlpLossGradKernelWithWorkspace(shape, device_weights, device_states, device_labels, sample_count, device_loss, device_grad, workspace_base, workspace_floats, blas, mgt::kInputGradPartialChunks, stream);
}

__host__ mgt::Status LaunchMlpLossGradKernelWithWorkspaceLt(const CudaMlpShape& shape,
                                                            const float* device_weights,
                                                            const mgt::TrainStateStorage* device_states,
                                                            const float* device_labels,
                                                            std::uint32_t sample_count,
                                                            float* device_loss,
                                                            float* device_grad,
                                                            float* workspace_base,
                                                            std::uint64_t workspace_floats,
                                                            cublasHandle_t blas,
                                                            cublasLtHandle_t blas_lt,
                                                            std::uint32_t input_grad_partial_chunks,
                                                            std::uint32_t input_grad_positions_per_block,
                                                            bool use_half_input_grad,
                                                            bool use_half_linear,
                                                            cudaStream_t stream) {
    return LaunchMlpLossGradKernelWithWorkspaceInternal(shape, device_weights, device_states, device_labels, sample_count, device_loss, device_grad, workspace_base, workspace_floats, blas, blas_lt, nullptr, input_grad_partial_chunks, input_grad_positions_per_block, use_half_input_grad, use_half_linear, nullptr, nullptr, nullptr, stream);
}
__host__ mgt::Status LaunchMlpLossGradKernelProfiledWithWorkspace(const CudaMlpShape& shape,
                                                                  const float* device_weights,
                                                                  const mgt::TrainStateStorage* device_states,
                                                                  const float* device_labels,
                                                                  std::uint32_t sample_count,
                                                                  float* device_loss,
                                                                  float* device_grad,
                                                                  float* workspace_base,
                                                                  std::uint64_t workspace_floats,
                                                                  cublasHandle_t blas,
                                                                  std::uint32_t input_grad_partial_chunks,
                                                                  MlpBackwardProfile* profile,
                                                                  cudaStream_t stream) {
    if (profile == nullptr) return mgt::Status::kInvalidConfig;
    return LaunchMlpLossGradKernelWithWorkspaceInternal(shape, device_weights, device_states, device_labels, sample_count, device_loss, device_grad, workspace_base, workspace_floats, blas, nullptr, nullptr, input_grad_partial_chunks, mgt::kInputGradPositionsPerBlock, false, false, profile, nullptr, nullptr, stream);
}

__host__ mgt::Status LaunchMlpLossGradKernelProfiledWithWorkspace(const CudaMlpShape& shape,
                                                                  const float* device_weights,
                                                                  const mgt::TrainStateStorage* device_states,
                                                                  const float* device_labels,
                                                                  std::uint32_t sample_count,
                                                                  float* device_loss,
                                                                  float* device_grad,
                                                                  float* workspace_base,
                                                                  std::uint64_t workspace_floats,
                                                                  cublasHandle_t blas,
                                                                  std::uint32_t input_grad_partial_chunks,
                                                                  std::uint32_t input_grad_positions_per_block,
                                                                  MlpBackwardProfile* profile,
                                                                  cudaStream_t stream) {
    if (profile == nullptr) return mgt::Status::kInvalidConfig;
    return LaunchMlpLossGradKernelWithWorkspaceInternal(shape, device_weights, device_states, device_labels, sample_count, device_loss, device_grad, workspace_base, workspace_floats, blas, nullptr, nullptr, input_grad_partial_chunks, input_grad_positions_per_block, false, false, profile, nullptr, nullptr, stream);
}

__host__ mgt::Status LaunchMlpLossGradKernelProfiledWithWorkspace(const CudaMlpShape& shape,
                                                                  const float* device_weights,
                                                                  const mgt::TrainStateStorage* device_states,
                                                                  const float* device_labels,
                                                                  std::uint32_t sample_count,
                                                                  float* device_loss,
                                                                  float* device_grad,
                                                                  float* workspace_base,
                                                                  std::uint64_t workspace_floats,
                                                                  cublasHandle_t blas,
                                                                  std::uint32_t input_grad_partial_chunks,
                                                                  std::uint32_t input_grad_positions_per_block,
                                                                  bool use_half_input_grad,
                                                                  MlpBackwardProfile* profile,
                                                                  cudaStream_t stream) {
    if (profile == nullptr) return mgt::Status::kInvalidConfig;
    return LaunchMlpLossGradKernelWithWorkspaceInternal(shape, device_weights, device_states, device_labels, sample_count, device_loss, device_grad, workspace_base, workspace_floats, blas, nullptr, nullptr, input_grad_partial_chunks, input_grad_positions_per_block, use_half_input_grad, false, profile, nullptr, nullptr, stream);
}

__host__ mgt::Status LaunchMlpLossGradKernelProfiledWithWorkspace(const CudaMlpShape& shape,
                                                                  const float* device_weights,
                                                                  const mgt::TrainStateStorage* device_states,
                                                                  const float* device_labels,
                                                                  std::uint32_t sample_count,
                                                                  float* device_loss,
                                                                  float* device_grad,
                                                                  float* workspace_base,
                                                                  std::uint64_t workspace_floats,
                                                                  cublasHandle_t blas,
                                                         std::uint32_t input_grad_partial_chunks,
                                                         std::uint32_t input_grad_positions_per_block,
                                                         bool use_half_input_grad,
                                                         bool use_half_linear,
                                                         MlpBackwardProfile* profile,
                                                                  cudaStream_t stream) {
    if (profile == nullptr) return mgt::Status::kInvalidConfig;
    return LaunchMlpLossGradKernelWithWorkspaceInternal(shape, device_weights, device_states, device_labels, sample_count, device_loss, device_grad, workspace_base, workspace_floats, blas, nullptr, nullptr, input_grad_partial_chunks, input_grad_positions_per_block, use_half_input_grad, use_half_linear, profile, nullptr, nullptr, stream);
}

__host__ mgt::Status LaunchMlpLossGradKernelProfiledWithWorkspace(const CudaMlpShape& shape,
                                                                  const float* device_weights,
                                                                  const mgt::TrainStateStorage* device_states,
                                                                  const float* device_labels,
                                                                  std::uint32_t sample_count,
                                                                  float* device_loss,
                                                                  float* device_grad,
                                                                  float* workspace_base,
                                                                  std::uint64_t workspace_floats,
                                                                  cublasHandle_t blas,
                                                                  MlpBackwardProfile* profile,
                                                                  cudaStream_t stream) {
    return LaunchMlpLossGradKernelProfiledWithWorkspace(shape, device_weights, device_states, device_labels, sample_count, device_loss, device_grad, workspace_base, workspace_floats, blas, mgt::kInputGradPartialChunks, profile, stream);
}

__host__ mgt::Status LaunchMlpLossGradKernelProfiledWithWorkspaceLt(const CudaMlpShape& shape,
                                                                    const float* device_weights,
                                                                    const mgt::TrainStateStorage* device_states,
                                                                    const float* device_labels,
                                                                    std::uint32_t sample_count,
                                                                    float* device_loss,
                                                                    float* device_grad,
                                                                    float* workspace_base,
                                                                    std::uint64_t workspace_floats,
                                                                    cublasHandle_t blas,
                                                                    cublasLtHandle_t blas_lt,
                                                                    std::uint32_t input_grad_partial_chunks,
                                                                    std::uint32_t input_grad_positions_per_block,
                                                                    bool use_half_input_grad,
                                                                    bool use_half_linear,
                                                                    MlpBackwardProfile* profile,
                                                                    cudaStream_t stream) {
    if (profile == nullptr) return mgt::Status::kInvalidConfig;
    return LaunchMlpLossGradKernelWithWorkspaceInternal(shape, device_weights, device_states, device_labels, sample_count, device_loss, device_grad, workspace_base, workspace_floats, blas, blas_lt, nullptr, input_grad_partial_chunks, input_grad_positions_per_block, use_half_input_grad, use_half_linear, profile, nullptr, nullptr, stream);
}

__host__ mgt::Status LaunchMlpLossGradKernelWithWorkspaceLtExternalHalf(const CudaMlpShape& shape,
                                                                        const float* device_weights,
                                                                        const __half* device_weights_half,
                                                                        const mgt::TrainStateStorage* device_states,
                                                                        const float* device_labels,
                                                                        std::uint32_t sample_count,
                                                                        float* device_loss,
                                                                        float* device_grad,
                                                                        float* workspace_base,
                                                                        std::uint64_t workspace_floats,
                                                                        cublasHandle_t blas,
                                                                        cublasLtHandle_t blas_lt,
                                                                        std::uint32_t input_grad_partial_chunks,
                                                                        std::uint32_t input_grad_positions_per_block,
                                                                        bool use_half_input_grad,
                                                                        bool use_half_linear,
                                                                        cudaStream_t stream,
                                                                        void* lt_workspace_base,
                                                                        std::uint64_t lt_workspace_bytes,
                                                                        std::uint32_t input_grad_position_tile, bool input_grad_sparse) {
    return LaunchMlpLossGradKernelWithWorkspaceInternal(shape, device_weights, device_states, device_labels, sample_count, device_loss, device_grad, workspace_base, workspace_floats, blas, blas_lt, device_weights_half, input_grad_partial_chunks, input_grad_positions_per_block, use_half_input_grad, use_half_linear, nullptr, nullptr, nullptr, stream, lt_workspace_base, lt_workspace_bytes, input_grad_position_tile, input_grad_sparse);
}

__host__ mgt::Status LaunchMlpLossGradKernelProfiledWithWorkspaceLtExternalHalf(const CudaMlpShape& shape,
                                                                                const float* device_weights,
                                                                                const __half* device_weights_half,
                                                                                const mgt::TrainStateStorage* device_states,
                                                                                const float* device_labels,
                                                                                std::uint32_t sample_count,
                                                                                float* device_loss,
                                                                                float* device_grad,
                                                                                float* workspace_base,
                                                                                std::uint64_t workspace_floats,
                                                                                cublasHandle_t blas,
                                                                                cublasLtHandle_t blas_lt,
                                                                                std::uint32_t input_grad_partial_chunks,
                                                                                std::uint32_t input_grad_positions_per_block,
                                                                                bool use_half_input_grad,
                                                                                bool use_half_linear,
                                                                                MlpBackwardProfile* profile,
                                                                                cudaStream_t stream,
                                                                                void* lt_workspace_base,
                                                                                std::uint64_t lt_workspace_bytes,
                                                                                std::uint32_t input_grad_position_tile, bool input_grad_sparse) {
    if (profile == nullptr) return mgt::Status::kInvalidConfig;
    return LaunchMlpLossGradKernelWithWorkspaceInternal(shape, device_weights, device_states, device_labels, sample_count, device_loss, device_grad, workspace_base, workspace_floats, blas, blas_lt, device_weights_half, input_grad_partial_chunks, input_grad_positions_per_block, use_half_input_grad, use_half_linear, profile, nullptr, nullptr, stream, lt_workspace_base, lt_workspace_bytes, input_grad_position_tile, input_grad_sparse);
}

__host__ mgt::Status LaunchMlpLossGradKernelProfiledWithWorkspaceLtAndCallbackExternalHalf(const CudaMlpShape& shape,
                                                                                           const float* device_weights,
                                                                                           const __half* device_weights_half,
                                                                                           const mgt::TrainStateStorage* device_states,
                                                                                           const float* device_labels,
                                                                                           std::uint32_t sample_count,
                                                                                           float* device_loss,
                                                                                           float* device_grad,
                                                                                           float* workspace_base,
                                                                                           std::uint64_t workspace_floats,
                                                                                           cublasHandle_t blas,
                                                                                           cublasLtHandle_t blas_lt,
                                                                                           std::uint32_t input_grad_partial_chunks,
                                                                                           std::uint32_t input_grad_positions_per_block,
                                                                                           bool use_half_input_grad,
                                                                                           bool use_half_linear,
                                                                                           MlpBackwardProfile* profile,
                                                                                           MlpGradientReadyCallback gradient_ready,
                                                                                           void* gradient_ready_user,
                                                                                           cudaStream_t stream,
                                                                                           void* lt_workspace_base,
                                                                                           std::uint64_t lt_workspace_bytes,
                                                                                           std::uint32_t input_grad_position_tile, bool input_grad_sparse) {
    return LaunchMlpLossGradKernelWithWorkspaceInternal(shape, device_weights, device_states, device_labels, sample_count, device_loss, device_grad, workspace_base, workspace_floats, blas, blas_lt, device_weights_half, input_grad_partial_chunks, input_grad_positions_per_block, use_half_input_grad, use_half_linear, profile, gradient_ready, gradient_ready_user, stream, lt_workspace_base, lt_workspace_bytes, input_grad_position_tile, input_grad_sparse);
}
__host__ mgt::Status LaunchMlpLossGradKernelProfiledWithWorkspaceLtAndCallback(const CudaMlpShape& shape,
                                                                               const float* device_weights,
                                                                               const mgt::TrainStateStorage* device_states,
                                                                               const float* device_labels,
                                                                               std::uint32_t sample_count,
                                                                               float* device_loss,
                                                                               float* device_grad,
                                                                               float* workspace_base,
                                                                               std::uint64_t workspace_floats,
                                                                               cublasHandle_t blas,
                                                                               cublasLtHandle_t blas_lt,
                                                                               std::uint32_t input_grad_partial_chunks,
                                                                               std::uint32_t input_grad_positions_per_block,
                                                                               bool use_half_input_grad,
                                                                               bool use_half_linear,
                                                                               MlpBackwardProfile* profile,
                                                                               MlpGradientReadyCallback gradient_ready,
                                                                               void* gradient_ready_user,
                                                                               cudaStream_t stream) {
    return LaunchMlpLossGradKernelWithWorkspaceInternal(shape, device_weights, device_states, device_labels, sample_count, device_loss, device_grad, workspace_base, workspace_floats, blas, blas_lt, nullptr, input_grad_partial_chunks, input_grad_positions_per_block, use_half_input_grad, use_half_linear, profile, gradient_ready, gradient_ready_user, stream);
}

__host__ mgt::Status LaunchMlpLossGradKernel(const CudaMlpShape& shape,
                                             const float* device_weights,
                                             const mgt::TrainStateStorage* device_states,
                                             const float* device_labels,
                                             std::uint32_t sample_count,
                                             float* device_loss,
                                             float* device_grad,
                                             cudaStream_t stream) {
    const std::uint64_t workspace_floats = MlpLossGradWorkspaceFloats(shape, sample_count, mgt::kInputGradPartialChunks);
    if (workspace_floats == 0) return mgt::Status::kInvalidConfig;
    float* workspace_base = nullptr;
    if (cudaMalloc(&workspace_base, workspace_floats * sizeof(float)) != cudaSuccess) return mgt::Status::kCudaFailure;
    cublasHandle_t blas = nullptr;
    if (cublasCreate(&blas) != CUBLAS_STATUS_SUCCESS) {
        cudaFree(workspace_base);
        return mgt::Status::kCudaFailure;
    }
    const mgt::Status status = LaunchMlpLossGradKernelWithWorkspace(shape, device_weights, device_states, device_labels, sample_count, device_loss, device_grad, workspace_base, workspace_floats, blas, stream);
    const cublasStatus_t destroy_status = cublasDestroy(blas);
    const cudaError_t free_status = cudaFree(workspace_base);
    if (status != mgt::Status::kOk) return status;
    if (destroy_status != CUBLAS_STATUS_SUCCESS || free_status != cudaSuccess) return mgt::Status::kCudaFailure;
    return mgt::Status::kOk;
}
}  // namespace mgt_cuda
