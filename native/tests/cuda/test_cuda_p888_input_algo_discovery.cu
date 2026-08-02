#include "mgt_cuda/bf16_linear_train_ops.cuh"

#include <cublasLt.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>

namespace {
struct Desc {
    cublasLtMatmulDesc_t op = nullptr;
    cublasLtMatrixLayout_t a = nullptr, b = nullptr, c = nullptr;
    ~Desc() { if(c)cublasLtMatrixLayoutDestroy(c); if(b)cublasLtMatrixLayoutDestroy(b); if(a)cublasLtMatrixLayoutDestroy(a); if(op)cublasLtMatmulDescDestroy(op); }
};

template<class T> bool Get(const cublasLtMatmulAlgo_t& algo,
                           cublasLtMatmulAlgoConfigAttributes_t attr, T* out) {
    std::size_t n=0;
    return cublasLtMatmulAlgoConfigGetAttribute(&algo,attr,out,sizeof(*out),&n)==CUBLAS_STATUS_SUCCESS && n==sizeof(*out);
}

bool Create(const mgt::Bf16GemmKeyV1& k, Desc* d) {
    const auto oa=static_cast<cublasOperation_t>(k.op_a), ob=static_cast<cublasOperation_t>(k.op_b);
    const auto order=CUBLASLT_ORDER_ROW;
    const std::uint64_t ar=oa==CUBLAS_OP_N?k.m:k.k, ac=oa==CUBLAS_OP_N?k.k:k.m;
    const std::uint64_t br=ob==CUBLAS_OP_N?k.k:k.n, bc=ob==CUBLAS_OP_N?k.n:k.k;
    auto s=cublasLtMatmulDescCreate(&d->op,CUBLAS_COMPUTE_32F,CUDA_R_32F);
    if(s==CUBLAS_STATUS_SUCCESS)s=cublasLtMatmulDescSetAttribute(d->op,CUBLASLT_MATMUL_DESC_TRANSA,&oa,sizeof(oa));
    if(s==CUBLAS_STATUS_SUCCESS)s=cublasLtMatmulDescSetAttribute(d->op,CUBLASLT_MATMUL_DESC_TRANSB,&ob,sizeof(ob));
    if(s==CUBLAS_STATUS_SUCCESS)s=cublasLtMatrixLayoutCreate(&d->a,CUDA_R_16BF,ar,ac,k.lda);
    if(s==CUBLAS_STATUS_SUCCESS)s=cublasLtMatrixLayoutSetAttribute(d->a,CUBLASLT_MATRIX_LAYOUT_ORDER,&order,sizeof(order));
    if(s==CUBLAS_STATUS_SUCCESS)s=cublasLtMatrixLayoutCreate(&d->b,CUDA_R_16BF,br,bc,k.ldb);
    if(s==CUBLAS_STATUS_SUCCESS)s=cublasLtMatrixLayoutSetAttribute(d->b,CUBLASLT_MATRIX_LAYOUT_ORDER,&order,sizeof(order));
    if(s==CUBLAS_STATUS_SUCCESS)s=cublasLtMatrixLayoutCreate(&d->c,CUDA_R_32F,k.m,k.n,k.ldc);
    if(s==CUBLAS_STATUS_SUCCESS)s=cublasLtMatrixLayoutSetAttribute(d->c,CUBLASLT_MATRIX_LAYOUT_ORDER,&order,sizeof(order));
    return s==CUBLAS_STATUS_SUCCESS;
}

bool Discover(cublasLtHandle_t handle,const mgt::Bf16GemmKeyV1& key,std::uint64_t cap) {
    Desc d{}; if(!Create(key,&d))return false;
    cublasLtMatmulPreference_t pref=nullptr;
    if(cublasLtMatmulPreferenceCreate(&pref)!=CUBLAS_STATUS_SUCCESS)return false;
    const std::size_t bytes=static_cast<std::size_t>(cap);
    cublasLtMatmulPreferenceSetAttribute(pref,CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,&bytes,sizeof(bytes));
    cublasLtMatmulHeuristicResult_t result[32]{}; int count=0;
    const auto s=cublasLtMatmulAlgoGetHeuristic(handle,d.op,d.a,d.b,d.c,d.c,pref,32,result,&count);
    cublasLtMatmulPreferenceDestroy(pref);
    if(s!=CUBLAS_STATUS_SUCCESS||count==0)return false;
    for(int i=0;i<count;++i){
        if(result[i].state!=CUBLAS_STATUS_SUCCESS||result[i].workspaceSize>cap)continue;
        int algo=0; std::uint32_t tile=0,stages=0,split=0,reduction=0,swizzle=0,custom=0;
        if(!Get(result[i].algo,CUBLASLT_ALGO_CONFIG_ID,&algo)||
           !Get(result[i].algo,CUBLASLT_ALGO_CONFIG_TILE_ID,&tile)||
           !Get(result[i].algo,CUBLASLT_ALGO_CONFIG_STAGES_ID,&stages)||
           !Get(result[i].algo,CUBLASLT_ALGO_CONFIG_SPLITK_NUM,&split)||
           !Get(result[i].algo,CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME,&reduction)||
           !Get(result[i].algo,CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING,&swizzle)||
           !Get(result[i].algo,CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION,&custom))continue;
        std::printf("rows=%u role=%u beta_bits=%u algo=%d tile=%u stages=%u split=%u reduction=%u swizzle=%u custom=%u workspace=%llu waves=%.3f\n",
                    key.active_rows,static_cast<unsigned>(key.role),key.beta_bits,algo,tile,stages,split,reduction,swizzle,custom,
                    static_cast<unsigned long long>(result[i].workspaceSize),result[i].wavesCount);
        return true;
    }
    return false;
}
}

int main(){
    cublasLtHandle_t handle=nullptr; if(cublasLtCreate(&handle)!=CUBLAS_STATUS_SUCCESS)return 1;
    constexpr std::uint64_t cap=256ULL<<20;
    for(std::uint32_t rows:{12497U,12498U,12500U}){
        const mgt_cuda::Bf16LinearProblem p{rows,100,rows,576,2560};
        for(const auto spec:{std::pair{mgt::Bf16GemmRole::kInputForward,0.0f},
                             std::pair{mgt::Bf16GemmRole::kInputForward,1.0f},
                             std::pair{mgt::Bf16GemmRole::kInputTableGrad,0.0f}}){
            mgt::Bf16GemmKeyV1 key{};
            if(mgt_cuda::BuildBf16LinearGemmKey(p,spec.first,spec.second,&key)!=mgt::Status::kOk||!Discover(handle,key,cap))return 2;
        }
    }
    cublasLtDestroy(handle); return 0;
}