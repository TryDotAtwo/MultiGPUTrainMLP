#include "mgt_cuda/bf16_linear_train_ops.cuh"

#include <cublasLt.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

namespace {
struct Descriptors {
    cublasLtMatmulDesc_t operation = nullptr;
    cublasLtMatrixLayout_t a = nullptr, b = nullptr, c = nullptr;
};

void Destroy(Descriptors* d) {
    if (d->c) cublasLtMatrixLayoutDestroy(d->c);
    if (d->b) cublasLtMatrixLayoutDestroy(d->b);
    if (d->a) cublasLtMatrixLayoutDestroy(d->a);
    if (d->operation) cublasLtMatmulDescDestroy(d->operation);
}

bool CreateDescriptors(const mgt::Bf16GemmKeyV1& key, Descriptors* d) {
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

template <class T>
bool Get(const cublasLtMatmulAlgo_t& algo, cublasLtMatmulAlgoConfigAttributes_t attribute, T* out) {
    std::size_t written = 0;
    return cublasLtMatmulAlgoConfigGetAttribute(&algo, attribute, out, sizeof(*out), &written) == CUBLAS_STATUS_SUCCESS && written == sizeof(*out);
}

bool Discover(cublasLtHandle_t handle, const mgt::Bf16GemmKeyV1& key,
              std::uint64_t workspace_bytes, mgt::Bf16GemmChoiceV1* choice) {
    Descriptors d{};
    if (!CreateDescriptors(key, &d)) return false;
    cublasLtMatmulPreference_t preference = nullptr;
    if (cublasLtMatmulPreferenceCreate(&preference) != CUBLAS_STATUS_SUCCESS) { Destroy(&d); return false; }
    const std::size_t capacity = static_cast<std::size_t>(workspace_bytes);
    cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &capacity, sizeof(capacity));
    cublasLtMatmulHeuristicResult_t results[32]{};
    int count = 0;
    const auto status = cublasLtMatmulAlgoGetHeuristic(handle, d.operation, d.a, d.b, d.c, d.c,
                                                        preference, 32, results, &count);
    bool found = false;
    for (int i = 0; status == CUBLAS_STATUS_SUCCESS && i < count; ++i) {
        if (results[i].state != CUBLAS_STATUS_SUCCESS || results[i].workspaceSize > capacity) continue;
        choice->backend = mgt::Bf16GemmBackend::kCublasLt;
        found = Get(results[i].algo, CUBLASLT_ALGO_CONFIG_ID, &choice->cublaslt_algo_id) &&
                Get(results[i].algo, CUBLASLT_ALGO_CONFIG_TILE_ID, &choice->tile_id) &&
                Get(results[i].algo, CUBLASLT_ALGO_CONFIG_STAGES_ID, &choice->stages_id) &&
                Get(results[i].algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &choice->split_k) &&
                Get(results[i].algo, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, &choice->reduction_scheme) &&
                Get(results[i].algo, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING, &choice->cta_swizzle) &&
                Get(results[i].algo, CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION, &choice->custom_option);
        if (found) {
            choice->workspace_bytes = results[i].workspaceSize;
            choice->workspace_alignment = 256;
            choice->split_k_contract.split_count = choice->split_k == 0 ? 1 : choice->split_k;
            choice->split_k_contract.scratch_alignment = 256;
            choice->split_k_contract.slot_count = 1;
            break;
        }
    }
    cublasLtMatmulPreferenceDestroy(preference);
    Destroy(&d);
    return found;
}

float Pattern(std::size_t index, int period, float scale) {
    return static_cast<float>(static_cast<int>(index % period) - period / 2) * scale;
}
}

int RunBackwardCase(cublasLtHandle_t handle, void* workspace, std::uint64_t workspace_bytes, const mgt_cuda::Bf16LinearProblem& p, mgt::Bf16GemmRole role, float beta, const std::vector<__nv_bfloat16>& dy, const std::vector<__nv_bfloat16>& rhs) {
    mgt::Bf16GemmKeyV1 key{}; mgt::Bf16GemmChoiceV1 choice{};
    if (mgt_cuda::BuildBf16LinearGemmKey(p, role, beta, &key) != mgt::Status::kOk || !Discover(handle, key, workspace_bytes, &choice)) return 20;
    mgt_cuda::FixedBf16GemmPlan* plan=nullptr; if (mgt_cuda::CreateFixedBf16GemmPlan(handle,key,choice,workspace,workspace_bytes,&plan)!=mgt::Status::kOk) return 21;
    std::vector<float> initial(key.m*key.n), expected(key.m*key.n), got(key.m*key.n); for(std::size_t i=0;i<initial.size();++i) initial[i]=Pattern(i*11+5,23,0.017f); expected=initial; if(beta==0.0f) std::fill(expected.begin(),expected.end(),0.0f);
    if(role==mgt::Bf16GemmRole::kGradWeight) for(std::uint32_t o=0;o<p.output_features;++o) for(std::uint32_t i=0;i<p.input_features;++i) for(std::uint32_t r=0;r<p.compute_rows;++r) expected[o*p.input_features+i]+=static_cast<float>(dy[r*p.output_features+o])*static_cast<float>(rhs[r*p.input_features+i]);
    else for(std::uint32_t r=0;r<p.compute_rows;++r) for(std::uint32_t i=0;i<p.input_features;++i) for(std::uint32_t o=0;o<p.output_features;++o) expected[r*p.input_features+i]+=static_cast<float>(dy[r*p.output_features+o])*static_cast<float>(rhs[o*p.input_features+i]);
    __nv_bfloat16 *da=nullptr,*db=nullptr; float* dc=nullptr; if(cudaMalloc(&da,dy.size()*sizeof(*da))!=cudaSuccess||cudaMalloc(&db,rhs.size()*sizeof(*db))!=cudaSuccess||cudaMalloc(&dc,got.size()*sizeof(*dc))!=cudaSuccess) return 22;
    cudaMemcpy(da,dy.data(),dy.size()*sizeof(*da),cudaMemcpyHostToDevice); cudaMemcpy(db,rhs.data(),rhs.size()*sizeof(*db),cudaMemcpyHostToDevice); cudaMemcpy(dc,initial.data(),initial.size()*sizeof(*dc),cudaMemcpyHostToDevice);
    if(mgt_cuda::LaunchFixedBf16Gemm(plan,da,db,dc,nullptr)!=mgt::Status::kOk||cudaDeviceSynchronize()!=cudaSuccess) return 23; cudaMemcpy(got.data(),dc,got.size()*sizeof(*dc),cudaMemcpyDeviceToHost);
    double e2=0.0,r2=0.0,maxe=0.0; for(std::size_t i=0;i<got.size();++i){if(!std::isfinite(got[i])) return 24; const double e=std::abs(static_cast<double>(got[i])-expected[i]); e2+=e*e; r2+=static_cast<double>(expected[i])*expected[i]; maxe=std::max(maxe,e/std::max(1.0,std::abs(static_cast<double>(expected[i]))));} const double l2=std::sqrt(e2/std::max(r2,1e-30));
    std::printf("bf16_fixed_%s_beta%.0f algo=%d tile=%u stages=%u workspace=%llu relative_l2=%.9g scaled_max=%.9g\n",role==mgt::Bf16GemmRole::kGradWeight?"dw":"dx",beta,choice.cublaslt_algo_id,choice.tile_id,choice.stages_id,static_cast<unsigned long long>(choice.workspace_bytes),l2,maxe);
    int result=(l2>1e-4||maxe>2e-3)?25:0; if(result==0&&role==mgt::Bf16GemmRole::kGradInput) for(std::size_t i=static_cast<std::size_t>(p.active_rows)*p.input_features;i<got.size();++i) if(got[i]!=(beta==0.0f?0.0f:initial[i])){result=26;break;} mgt_cuda::DestroyFixedBf16GemmPlan(plan); cudaFree(dc); cudaFree(db); cudaFree(da); return result;
}
int main() {
    constexpr std::uint32_t active = 127, rows = 128, input = 64, output = 48;
    const mgt_cuda::Bf16LinearProblem problem{active, 9, rows, input, output};
    mgt::Bf16GemmKeyV1 key{};
    if (mgt_cuda::BuildBf16LinearGemmKey(problem, mgt::Bf16GemmRole::kHiddenForward, 0.0f, &key) != mgt::Status::kOk) return 1;
    cublasLtHandle_t handle = nullptr;
    if (cublasLtCreate(&handle) != CUBLAS_STATUS_SUCCESS) return 2;
    constexpr std::uint64_t workspace_bytes = 32ULL << 20;
    void* workspace = nullptr;
    if (cudaMalloc(&workspace, workspace_bytes) != cudaSuccess) return 3;
    mgt::Bf16GemmChoiceV1 choice{};
    if (!Discover(handle, key, workspace_bytes, &choice)) return 4;
    mgt_cuda::FixedBf16GemmPlan* plan = nullptr;
    if (mgt_cuda::CreateFixedBf16GemmPlan(handle, key, choice, workspace, workspace_bytes, &plan) != mgt::Status::kOk) return 5;

    std::vector<__nv_bfloat16> a(rows * input), b(output * input);
    for (std::size_t i = 0; i < a.size(); ++i) a[i] = __float2bfloat16(Pattern(i, 29, 0.031f));
    for (std::size_t i = active * input; i < a.size(); ++i) a[i] = __float2bfloat16(0.0f);
    for (std::size_t i = 0; i < b.size(); ++i) b[i] = __float2bfloat16(Pattern(i * 7 + 3, 31, 0.023f));
    std::vector<float> expected(rows * output, 0.0f), got(rows * output, 0.0f);
    for (std::uint32_t r = 0; r < rows; ++r)
        for (std::uint32_t o = 0; o < output; ++o)
            for (std::uint32_t k = 0; k < input; ++k)
                expected[r * output + o] += static_cast<float>(a[r * input + k]) * static_cast<float>(b[o * input + k]);
    __nv_bfloat16 *da = nullptr, *db = nullptr; float* dc = nullptr;
    if (cudaMalloc(&da, a.size() * sizeof(*da)) != cudaSuccess ||
        cudaMalloc(&db, b.size() * sizeof(*db)) != cudaSuccess ||
        cudaMalloc(&dc, got.size() * sizeof(*dc)) != cudaSuccess) return 6;
    cudaMemcpy(da, a.data(), a.size() * sizeof(*da), cudaMemcpyHostToDevice);
    cudaMemcpy(db, b.data(), b.size() * sizeof(*db), cudaMemcpyHostToDevice);
    cudaMemset(dc, 0, got.size() * sizeof(*dc));
    if (mgt_cuda::LaunchFixedBf16Gemm(plan, da, db, dc, nullptr) != mgt::Status::kOk ||
        cudaDeviceSynchronize() != cudaSuccess) return 7;
    cudaMemcpy(got.data(), dc, got.size() * sizeof(*dc), cudaMemcpyDeviceToHost);
    double error2 = 0.0, reference2 = 0.0, scaled_max = 0.0;
    for (std::size_t i = 0; i < got.size(); ++i) {
        if (!std::isfinite(got[i])) return 8;
        const double error = std::abs(static_cast<double>(got[i]) - expected[i]);
        error2 += error * error;
        reference2 += static_cast<double>(expected[i]) * expected[i];
        scaled_max = std::max(scaled_max, error / std::max(1.0, std::abs(static_cast<double>(expected[i]))));
    }
    const double relative_l2 = std::sqrt(error2 / std::max(reference2, 1e-30));
    std::printf("bf16_fixed_forward algo=%d tile=%u stages=%u workspace=%llu relative_l2=%.9g scaled_max=%.9g\n",
                choice.cublaslt_algo_id, choice.tile_id, choice.stages_id,
                static_cast<unsigned long long>(choice.workspace_bytes), relative_l2, scaled_max);
    if (relative_l2 > 1e-4 || scaled_max > 2e-3) return 9;
    for (std::size_t i = active * output; i < got.size(); ++i) if (got[i] != 0.0f) return 10;
    mgt_cuda::DestroyFixedBf16GemmPlan(plan);
    cudaFree(dc); cudaFree(db); cudaFree(da);
    std::vector<__nv_bfloat16> dy(rows * output), x(rows * input), weights(output * input);
    for(std::size_t i=0;i<dy.size();++i) dy[i]=__float2bfloat16(Pattern(i*5+1,37,0.019f)); for(std::size_t i=active*output;i<dy.size();++i) dy[i]=__float2bfloat16(0.0f);
    for(std::size_t i=0;i<x.size();++i) x[i]=__float2bfloat16(Pattern(i*3+2,41,0.013f)); for(std::size_t i=active*input;i<x.size();++i) x[i]=__float2bfloat16(0.0f); for(std::size_t i=0;i<weights.size();++i) weights[i]=__float2bfloat16(Pattern(i*7+4,43,0.011f));
    const int dw0=RunBackwardCase(handle,workspace,workspace_bytes,problem,mgt::Bf16GemmRole::kGradWeight,0.0f,dy,x); const int dw1=RunBackwardCase(handle,workspace,workspace_bytes,problem,mgt::Bf16GemmRole::kGradWeight,1.0f,dy,x); const int dx0=RunBackwardCase(handle,workspace,workspace_bytes,problem,mgt::Bf16GemmRole::kGradInput,0.0f,dy,weights); const int dx1=RunBackwardCase(handle,workspace,workspace_bytes,problem,mgt::Bf16GemmRole::kGradInput,1.0f,dy,weights);
    cudaFree(workspace); cublasLtDestroy(handle); if(dw0!=0)return dw0; if(dw1!=0)return dw1; if(dx0!=0)return dx0; return dx1;
}
