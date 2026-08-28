#ifdef MGT_LOCAL_MLP_IMPLEMENTATION
#define MlpBatchNormForwardWorkspaceFloats LocalMlpBatchNormForwardWorkspaceFloatsImpl
#define LaunchMlpBatchNormForward LaunchLocalMlpBatchNormForwardImpl
#define LaunchMlpBatchNormOutputLossGrad LaunchLocalMlpBatchNormOutputLossGradImpl
#define LaunchMlpBatchNormResidualFc2Backward LaunchLocalMlpBatchNormResidualFc2BackwardImpl
#define LaunchMlpBatchNormResidualStackBackward LaunchLocalMlpBatchNormResidualStackBackwardImpl
#define LaunchMlpBatchNormHiddenBackward LaunchLocalMlpBatchNormHiddenBackwardImpl
#define LaunchMlpBatchNormInputBackward LaunchLocalMlpBatchNormInputBackwardImpl
#define LaunchMlpBatchNormAdamStep LaunchLocalMlpBatchNormAdamStepImpl
#define LaunchMlpBatchNormTrainStep LaunchLocalMlpBatchNormTrainStepImpl
#define MGT_STRICT_INPUT_BACKWARD LaunchLocalMlpBatchNormInputBackwardStrictLegacy
#else
#define MGT_STRICT_INPUT_BACKWARD LaunchMlpBatchNormInputBackwardStrictLegacy
#endif
#include "mgt_cuda/mlp_batch_norm_forward.cuh"
#ifdef MGT_LOCAL_MLP_IMPLEMENTATION
#include "mgt_cuda/local_batch_norm.cuh"
#include "mgt_cuda/fp16_linear_train_ops.cuh"
#else
#include "mgt_cuda/allreduce_nccl.cuh"
#endif
#include "batch_norm_activation.cuh"
#include "mgt_cuda/sync_batch_norm_selector.cuh"
#include <algorithm>
#include "mgt/input_grad_grouping.hpp"
#include <cstdio>
#include <cstdlib>
#include <cstring>
namespace mgt_cuda { namespace {
#ifdef MGT_LOCAL_MLP_IMPLEMENTATION
NcclRankContext* LocalBackendContext(LocalMlpFp16Context* fp16 = nullptr) {
    return reinterpret_cast<NcclRankContext*>(fp16 ? fp16 : reinterpret_cast<LocalMlpFp16Context*>(1));
}
LocalMlpFp16Context* LocalFp16(NcclRankContext* context) {
    return reinterpret_cast<std::uintptr_t>(context) == 1
        ? nullptr : reinterpret_cast<LocalMlpFp16Context*>(context);
}
mgt::Status LocalNoopAllreduce(float*, std::uint64_t, NcclRankContext*, cudaStream_t) {
    return mgt::Status::kOk;
}
mgt::Status LocalBnForward(
    const float* x, int rows, int global_rows, int cols, int stride,
    const float* gamma, const float* beta, float* running_mean,
    float* running_var, float momentum, float epsilon, float* y, float* mean,
    float* inv_std, float* normalized, float* scratch, NcclRankContext*,
    cudaStream_t stream) {
    if (rows != global_rows) return mgt::Status::kInvalidConfig;
    return LaunchLocalStridedBatchNormForward(
        x, rows, cols, stride, gamma, beta, running_mean, running_var,
        momentum, epsilon, y, mean, inv_std, normalized, scratch, stream);
}
mgt::Status LocalBnBackward(
    const float* dy, int rows, int global_rows, int cols, int stride,
    const float* gamma, const float* inv_std, const float* normalized,
    float* dx, float* dgamma, float* dbeta, float* scratch,
    NcclRankContext*, cudaStream_t stream) {
    if (rows != global_rows) return mgt::Status::kInvalidConfig;
    return LaunchLocalStridedBatchNormBackward(
        dy, rows, cols, stride, gamma, inv_std, normalized, dx, dgamma,
        dbeta, scratch, stream);
}
#define NcclAllreduceSumFloat LocalNoopAllreduce
#define LaunchSelectedStridedSyncBatchNormForward LocalBnForward
#define LaunchSelectedStridedSyncBatchNormBackward LocalBnBackward
#endif
constexpr unsigned T=256;
__host__ __device__ std::uint64_t RB(CudaMlpShape s){return 2ULL*(static_cast<std::uint64_t>(s.hd2)*s.hd2+s.hd2);}__host__ __device__ std::uint64_t IB(CudaMlpShape s){return static_cast<std::uint64_t>(s.state_len)*s.state_value_pad*s.hd1;}__host__ __device__ std::uint64_t HW(CudaMlpShape s){return IB(s)+s.hd1;}__host__ __device__ std::uint64_t HB(CudaMlpShape s){return HW(s)+static_cast<std::uint64_t>(s.hd1)*s.hd2;}__host__ __device__ std::uint64_t R0(CudaMlpShape s){return HB(s)+s.hd2;}__host__ __device__ std::uint64_t OW(CudaMlpShape s){return R0(s)+static_cast<std::uint64_t>(s.residual_blocks)*RB(s);}__host__ __device__ std::uint64_t OB(CudaMlpShape s){return OW(s)+static_cast<std::uint64_t>(s.hd2)*s.output_dim;}__host__ __device__ std::uint64_t F1W(CudaMlpShape s,unsigned b){return R0(s)+static_cast<std::uint64_t>(b)*RB(s);}__host__ __device__ std::uint64_t F1B(CudaMlpShape s,unsigned b){return F1W(s,b)+static_cast<std::uint64_t>(s.hd2)*s.hd2;}__host__ __device__ std::uint64_t F2W(CudaMlpShape s,unsigned b){return F1B(s,b)+s.hd2;}__host__ __device__ std::uint64_t F2B(CudaMlpShape s,unsigned b){return F2W(s,b)+static_cast<std::uint64_t>(s.hd2)*s.hd2;}
__global__ void Input(CudaMlpShape s,unsigned logical,const float*w,const mgt::TrainStateStorage*states,unsigned rows,float*out){unsigned q=blockIdx.x*blockDim.x+threadIdx.x;if(q>=rows*s.hd1)return;unsigned r=q/s.hd1,h=q-r*s.hd1;if(h>=logical){out[q]=0;return;}float v=w[IB(s)+h];for(unsigned p=0;p<s.state_len;p++)v+=w[(static_cast<std::uint64_t>(p)*s.state_value_pad+states[r].v[p])*s.hd1+h];out[q]=v;}
#ifdef MGT_LOCAL_MLP_IMPLEMENTATION
__global__ void InputHalf(CudaMlpShape s,unsigned logical,const __half*w,const mgt::TrainStateStorage*states,unsigned rows,float*out){unsigned q=blockIdx.x*blockDim.x+threadIdx.x;if(q>=rows*s.hd1)return;unsigned r=q/s.hd1,h=q-r*s.hd1;if(h>=logical){out[q]=0;return;}float v=__half2float(w[IB(s)+h]);for(unsigned p=0;p<s.state_len;p++)v+=__half2float(w[(static_cast<std::uint64_t>(p)*s.state_value_pad+states[r].v[p])*s.hd1+h]);out[q]=v;}

__global__ void InputHalf2Row(CudaMlpShape s,unsigned logical,const __half*w,const mgt::TrainStateStorage*states,unsigned rows,float*out){
    __shared__ std::uint64_t offsets[mgt::kStateStorageLen];
    const unsigned row=blockIdx.x;
    if(row>=rows)return;
    if(threadIdx.x<s.state_len)offsets[threadIdx.x]=(static_cast<std::uint64_t>(threadIdx.x)*s.state_value_pad+states[row].v[threadIdx.x])*s.hd1;
    __syncthreads();
    for(unsigned h=2U*threadIdx.x;h<s.hd1;h+=2U*blockDim.x){
        float2 value{};
        if(h<logical){
            value=__half22float2(*reinterpret_cast<const __half2*>(w+IB(s)+h));
            for(unsigned p=0;p<s.state_len;++p){
                const float2 add=__half22float2(*reinterpret_cast<const __half2*>(w+offsets[p]+h));
                value.x+=add.x; value.y+=add.y;
            }
        }
        *reinterpret_cast<float2*>(out+static_cast<std::uint64_t>(row)*s.hd1+h)=value;
    }
}

mgt::Status LaunchInputHalfInternal(CudaMlpShape s,unsigned logical,const __half*w,const mgt::TrainStateStorage*states,unsigned rows,float*out,cudaStream_t st){
    if(!w||!states||!out||rows==0||logical>s.hd1)return mgt::Status::kInvalidConfig;
    if((s.hd1&1U)==0&&(logical&1U)==0)InputHalf2Row<<<rows,T,0,st>>>(s,logical,w,states,rows,out);
    else { const unsigned blocks=(rows*s.hd1+T-1)/T; InputHalf<<<blocks,T,0,st>>>(s,logical,w,states,rows,out); }
    return cudaPeekAtLastError()==cudaSuccess?mgt::Status::kOk:mgt::Status::kCudaFailure;
}
#endif
mgt::Status LaunchInputBackend(CudaMlpShape s,unsigned logical,const float*w,const mgt::TrainStateStorage*states,unsigned rows,float*out,NcclRankContext*ctx,cudaStream_t st){unsigned blocks=(rows*s.hd1+T-1)/T;
#ifdef MGT_LOCAL_MLP_IMPLEMENTATION
if(auto*fp=LocalFp16(ctx))return LaunchInputHalfInternal(s,logical,fp->weight_mirror,states,rows,out,st);else
#endif
Input<<<blocks,T,0,st>>>(s,logical,w,states,rows,out);return cudaPeekAtLastError()==cudaSuccess?mgt::Status::kOk:mgt::Status::kCudaFailure;}
__global__ void Bias(float*x,const float*b,unsigned rows,unsigned stride,unsigned logical){unsigned q=blockIdx.x*blockDim.x+threadIdx.x;if(q>=rows*stride)return;unsigned c=q%stride;x[q]=c<logical?x[q]+b[c]:0;}
__global__ void OutputBias(float*x,const float*b,unsigned rows,unsigned cols){unsigned q=blockIdx.x*blockDim.x+threadIdx.x;if(q<rows*cols)x[q]+=b[q%cols];}
__global__ void OutputLossGrad(const float*outputs,const float*labels,unsigned count,unsigned cols,float scale,float*loss,float*dy,float*bias_grad){unsigned q=blockIdx.x*blockDim.x+threadIdx.x;if(q>=count)return;float d=outputs[q]-labels[q];dy[q]=2.0f*d*scale;atomicAdd(loss,d*d*scale);atomicAdd(bias_grad+q%cols,dy[q]);}
__global__ void ScalarOutputLossGrad(const float*outputs,const float*labels,unsigned rows,float scale,float*loss,float*dy,float*bias_grad){__shared__ float sl[T],sb[T];unsigned r=blockIdx.x*blockDim.x+threadIdx.x;float l=0,g=0;if(r<rows){float d=outputs[r]-labels[r];g=2.0f*d*scale;l=d*d*scale;dy[r]=g;}sl[threadIdx.x]=l;sb[threadIdx.x]=g;__syncthreads();for(unsigned d=blockDim.x/2;d;d>>=1){if(threadIdx.x<d){sl[threadIdx.x]+=sl[threadIdx.x+d];sb[threadIdx.x]+=sb[threadIdx.x+d];}__syncthreads();}if(threadIdx.x==0){atomicAdd(loss,sl[0]);atomicAdd(bias_grad,sb[0]);}}
__global__ void ScalarOutputInputGrad(const float*dy,const float*w,unsigned rows,unsigned hd2,float*d_final){unsigned q=blockIdx.x*blockDim.x+threadIdx.x;if(q<rows*hd2)d_final[q]=dy[q/hd2]*w[q%hd2];}
__global__ void ColumnSum(const float*x,unsigned rows,unsigned logical,unsigned stride,float*out){unsigned c=blockIdx.x;if(c>=stride)return;if(c>=logical){if(threadIdx.x==0)out[c]=0;return;}__shared__ float v[T];float z=0;for(unsigned r=threadIdx.x;r<rows;r+=T)z+=x[static_cast<std::uint64_t>(r)*stride+c];v[threadIdx.x]=z;__syncthreads();for(unsigned d=T/2;d;d>>=1){if(threadIdx.x<d)v[threadIdx.x]+=v[threadIdx.x+d];__syncthreads();}if(threadIdx.x==0)out[c]=v[0];}
__global__ void AddInPlace(float*x,const float*y,std::uint64_t n){std::uint64_t q=static_cast<std::uint64_t>(blockIdx.x)*blockDim.x+threadIdx.x;if(q<n)x[q]+=y[q];}
constexpr unsigned INPUT_T=96;
__global__ void SparseInputGrad(CudaMlpShape s,const mgt::TrainStateStorage*states,const float*dz,unsigned rows,float*grad){extern __shared__ float sums[];unsigned h=blockIdx.x*blockDim.x+threadIdx.x,pos=blockIdx.y,stride=s.state_value_pad+1;float*v=sums+threadIdx.x*stride;bool active=h<s.hd1;unsigned lane=threadIdx.x&31U;for(unsigned x=0;x<s.state_value_pad;x++)v[x]=0;for(unsigned r=0;r<rows;r++){unsigned value=0;if(lane==0)value=states[r].v[pos];value=__shfl_sync(0xffffffffU,value,0);if(active&&value<s.state_value_pad)v[value]+=dz[static_cast<std::uint64_t>(r)*s.hd1+h];}if(!active)return;for(unsigned value=0;value<s.state_value_pad;value++){std::uint64_t row=static_cast<std::uint64_t>(pos)*s.state_value_pad+value;grad[row*s.hd1+h]=v[value];}}
__global__ void SparseInputGradCoalesced96(
    CudaMlpShape s,
    const mgt::TrainStateStorage* states,
    const float* dz,
    unsigned rows,
    float* grad) {
    extern __shared__ float sums[];
    const unsigned h = blockIdx.x * INPUT_T + threadIdx.x;
    const unsigned position = blockIdx.y;
    const bool active = h < s.hd1;
    const unsigned lane = threadIdx.x & 31U;
    for (unsigned value = 0; value < s.state_value_pad; ++value) {
        sums[value * INPUT_T + threadIdx.x] = 0.0f;
    }
    for (unsigned row = 0; row < rows; ++row) {
        unsigned value = 0;
        if (lane == 0) value = states[row].v[position];
        value = __shfl_sync(0xffffffffU, value, 0);
        if (active && value < s.state_value_pad) {
            sums[value * INPUT_T + threadIdx.x] +=
                dz[static_cast<std::uint64_t>(row) * s.hd1 + h];
        }
    }
    if (!active) return;
    for (unsigned value = 0; value < s.state_value_pad; ++value) {
        const std::uint64_t output_row =
            static_cast<std::uint64_t>(position) * s.state_value_pad + value;
        grad[output_row * s.hd1 + h] = sums[value * INPUT_T + threadIdx.x];
    }
}

constexpr unsigned GROUPED_INPUT_MAX_POSITIONS = 4;

__global__ void SparseInputGradExactGrouped(
    CudaMlpShape s,
    const mgt::TrainStateStorage* states,
    const float* dz,
    unsigned rows,
    unsigned positions,
    float* grad) {
    extern __shared__ float sums[];
    const unsigned h = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned first_position = blockIdx.y * positions;
    const bool active = h < s.hd1;
    for (unsigned p = 0; p < positions; ++p) {
        for (unsigned value = 0; value < s.state_value_pad; ++value) {
            sums[(p * s.state_value_pad + value) * blockDim.x + threadIdx.x] = 0.0f;
        }
    }
    const unsigned lane = threadIdx.x & 31U;
    for (unsigned row = 0; row < rows; ++row) {
        const float value_grad =
            active ? dz[static_cast<std::uint64_t>(row) * s.hd1 + h] : 0.0f;
        for (unsigned p = 0; p < positions; ++p) {
            const unsigned position = first_position + p;
            if (position >= s.state_len) break;
            unsigned value = 0;
            if (lane == 0) value = states[row].v[position];
            value = __shfl_sync(0xffffffffU, value, 0);
            if (active && value < s.state_value_pad) {
                sums[(p * s.state_value_pad + value) * blockDim.x + threadIdx.x] +=
                    value_grad;
            }
        }
    }
    if (!active) return;
    for (unsigned p = 0; p < positions; ++p) {
        const unsigned position = first_position + p;
        if (position >= s.state_len) break;
        for (unsigned value = 0; value < s.state_value_pad; ++value) {
            grad[(static_cast<std::uint64_t>(position) * s.state_value_pad + value) *
                     s.hd1 +
                 h] = sums[(p * s.state_value_pad + value) * blockDim.x +
                           threadIdx.x];
        }
    }
}

unsigned GroupedInputThreads(unsigned positions) {
    if (positions <= 1U) return 256U;
    if (positions == 2U) return 128U;
    return 64U;
}

struct InputGradLaunchConfig {
    mgt::Status status;
    bool exact;
    unsigned positions;
    std::size_t shared;
};

InputGradLaunchConfig ResolveInputGradLaunch(CudaMlpShape shape) {
    static thread_local bool initialized = false;
    static thread_local int cached_device = -1;
    static thread_local unsigned cached_state_len = 0;
    static thread_local unsigned cached_value_pad = 0;
    static thread_local unsigned cached_hd1 = 0;
    static thread_local InputGradLaunchConfig cached{};
    int device = 0;
    if (cudaGetDevice(&device) != cudaSuccess) {
        return {mgt::Status::kCudaFailure, false, 1, 0};
    }
    if (initialized && device == cached_device && shape.state_len == cached_state_len &&
        shape.state_value_pad == cached_value_pad && shape.hd1 == cached_hd1) {
        return cached;
    }

    cached = {mgt::Status::kInvalidConfig, false, 1, 0};
    const char* mode = std::getenv("MGT_BN_INPUT_GRAD_KERNEL");
    const char* requested_text =
        std::getenv("MGT_BN_INPUT_GRAD_POSITIONS_PER_BLOCK");
    const bool strict = mode != nullptr && std::strcmp(mode, "strict") == 0;
    const bool automatic = mode == nullptr || std::strcmp(mode, "auto") == 0;
    const bool coalesced = mode != nullptr && std::strcmp(mode, "coalesced") == 0;
    const bool exact = mode != nullptr && std::strcmp(mode, "exact") == 0;
    unsigned requested = 0;
    bool requested_valid = true;
    if (requested_text != nullptr) {
        char* end = nullptr;
        const unsigned long parsed = std::strtoul(requested_text, &end, 10);
        requested_valid = *requested_text != '\0' && end != requested_text &&
                          *end == '\0' && parsed <= GROUPED_INPUT_MAX_POSITIONS;
        if (requested_valid) requested = static_cast<unsigned>(parsed);
    }

    if ((strict || automatic || exact || coalesced) && requested_valid &&
        (!(strict || coalesced) || requested == 0)) {
        if (coalesced) {
            cached = {mgt::Status::kOk, true, 0,
                      static_cast<std::size_t>(INPUT_T) * shape.state_value_pad * sizeof(float)};
        } else if (strict) {
            cached = {mgt::Status::kOk, false, 1, 0};
        } else if (automatic && requested == 0) {
            cached = {mgt::Status::kOk, false, 1, 0};
        } else {
            cudaDeviceProp properties{};
            if (cudaGetDeviceProperties(&properties, device) != cudaSuccess) {
                cached.status = mgt::Status::kCudaFailure;
            } else if (automatic && requested == 0 && properties.major >= 8) {
                // The owner-write kernel is ~2.6x faster than grouped exact on A100.
                // Keep grouped exact available explicitly and as the T4 auto policy.
                cached = {mgt::Status::kOk, false, 1, 0};
            } else {
                const mgt::InputGradGroupingLimits limits{
                    static_cast<std::uint64_t>(properties.sharedMemPerBlock),
                    static_cast<std::uint64_t>(properties.sharedMemPerMultiprocessor),
                    2,
                    GROUPED_INPUT_MAX_POSITIONS,
                };
                const std::uint64_t shared_per_position =
                    static_cast<std::uint64_t>(shape.state_value_pad) *
                    32U * sizeof(float);
                mgt::InputGradGroupingConfig grouping{};
                cached.status = mgt::ResolveInputGradGrouping(
                    shape.state_len,
                    requested,
                    shared_per_position,
                    limits,
                    &grouping);
                if (cached.status == mgt::Status::kOk) {
                    cached.exact = true;
                    cached.positions = grouping.positions_per_block;
                    const unsigned threads = GroupedInputThreads(cached.positions);
                    cached.shared = static_cast<std::size_t>(cached.positions) *
                                    shape.state_value_pad * threads * sizeof(float);
                    if (cached.shared > static_cast<std::size_t>(properties.sharedMemPerBlockOptin)) {
                        cached.status = mgt::Status::kInvalidConfig;
                    }
                }
            }
        }
    }

    initialized = true;
    cached_device = device;
    cached_state_len = shape.state_len;
    cached_value_pad = shape.state_value_pad;
    cached_hd1 = shape.hd1;
    std::fprintf(
        stderr,
        "mgt input_grad kernel=%s positions=%u shared=%zu status=%u\n",
        cached.status == mgt::Status::kOk
            ? (cached.exact ? (cached.positions == 0 ? "coalesced" : "exact") : "strict")
            : "invalid",
        cached.positions,
        cached.shared,
        static_cast<unsigned>(cached.status));
    return cached;
}
mgt::Status Gemm(cublasHandle_t h,const float*a,const float*b,float*c,unsigned m,unsigned n,unsigned k,NcclRankContext*ctx,cudaStream_t st){
#ifdef MGT_LOCAL_MLP_IMPLEMENTATION
    if(auto*fp=LocalFp16(ctx);fp&&n>1){
        const auto ac=static_cast<std::uint64_t>(m)*k;
        if(ac>fp->operand_a_capacity||!fp->master_weights||!fp->weight_mirror||!fp->operand_a)return mgt::Status::kInvalidConfig;
        const auto offset=b-fp->master_weights;
        if(offset<0||LaunchFloatToHalf(a,fp->operand_a,ac,st)!=mgt::Status::kOk)return mgt::Status::kCudaFailure;
        return LaunchFp16LinearForward(h,fp->operand_a,fp->weight_mirror+offset,c,m,k,n,st);
    }
#endif
    float alpha=1,beta=0;return cublasSgemm(h,CUBLAS_OP_N,CUBLAS_OP_N,n,m,k,&alpha,b,n,a,k,&beta,c,n)==CUBLAS_STATUS_SUCCESS?mgt::Status::kOk:mgt::Status::kCudaFailure;
}
#define Gemm(h,a,b,c,m,n,k) Gemm(h,a,b,c,m,n,k,ctx,st)
#ifdef MGT_LOCAL_MLP_IMPLEMENTATION
cublasStatus_t LocalSgemm(cublasHandle_t h,cublasOperation_t opa,cublasOperation_t opb,int m,int n,int k,const float*alpha,const float*a,int lda,const float*b,int ldb,const float*beta,float*c,int ldc,NcclRankContext*ctx,cudaStream_t st){
    auto*fp=LocalFp16(ctx);
    if(fp&&opa==CUBLAS_OP_N&&opb==CUBLAS_OP_T){
        const auto xc=static_cast<std::uint64_t>(k)*n,dyc=static_cast<std::uint64_t>(k)*m;
        if(xc>fp->operand_a_capacity||dyc>fp->operand_b_capacity||LaunchFloatToHalf(b,fp->operand_a,xc,st)!=mgt::Status::kOk||LaunchFloatToHalf(a,fp->operand_b,dyc,st)!=mgt::Status::kOk)return CUBLAS_STATUS_EXECUTION_FAILED;
        return LaunchFp16LinearGradWeight(h,fp->operand_a,fp->operand_b,c,k,n,m,st)==mgt::Status::kOk?CUBLAS_STATUS_SUCCESS:CUBLAS_STATUS_EXECUTION_FAILED;
    }
    if(fp&&opa==CUBLAS_OP_T&&opb==CUBLAS_OP_N){
        const std::uint64_t dyc=static_cast<std::uint64_t>(n)*k;
        const auto offset=a-fp->master_weights;
        if(offset<0||dyc>fp->operand_b_capacity||LaunchFloatToHalf(b,fp->operand_b,dyc,st)!=mgt::Status::kOk)return CUBLAS_STATUS_EXECUTION_FAILED;
        return LaunchFp16LinearGradInput(h,fp->operand_b,fp->weight_mirror+offset,c,n,m,k,*beta,st)==mgt::Status::kOk?CUBLAS_STATUS_SUCCESS:CUBLAS_STATUS_EXECUTION_FAILED;
    }
    return cublasSgemm(h,opa,opb,m,n,k,alpha,a,lda,b,ldb,beta,c,ldc);
}
#undef cublasSgemm
#define cublasSgemm(h,opa,opb,m,n,k,alpha,a,lda,b,ldb,beta,c,ldc) LocalSgemm(h,opa,opb,m,n,k,alpha,a,lda,b,ldb,beta,c,ldc,ctx,st)
#endif
bool launch(){return cudaPeekAtLastError()==cudaSuccess;}
}
std::uint64_t MlpBatchNormForwardWorkspaceFloats(const CudaMlpShape&s,const mgt::BatchNormTrainingPlan&p,std::uint32_t rows){if(!rows||p.workspace_floats==0)return 0;return static_cast<std::uint64_t>(rows)*(s.hd1+(2ULL*s.residual_blocks+1ULL)*s.hd2)+p.workspace_floats;}
mgt::Status LaunchMlpBatchNormForward(const CudaMlpShape&s,std::uint32_t lh1,std::uint32_t lh2,const float*w,float*aff,float*run,const mgt::TrainStateStorage*states,std::uint32_t lr,std::uint32_t gr,float*out,float*base,std::uint64_t count,const mgt::BatchNormTrainingPlan&p,NcclRankContext*ctx,cublasHandle_t blas,cudaStream_t st){if(ValidateCudaMlpShape(s)!=mgt::Status::kOk||!lh1||!lh2||lh1>s.hd1||lh2>s.hd2||!w||!aff||!run||!states||!lr||gr<lr||!out||!base||!ctx||!blas||p.sites.size()!=2+2*s.residual_blocks||count<MlpBatchNormForwardWorkspaceFloats(s,p,lr))return mgt::Status::kInvalidConfig;if(cublasSetStream(blas,st)!=CUBLAS_STATUS_SUCCESS)return mgt::Status::kCudaFailure;const std::uint64_t bh2=static_cast<std::uint64_t>(lr)*s.hd2;float*a1=base;base+=static_cast<std::uint64_t>(lr)*s.hd1;float*block_inputs=base;base+=static_cast<std::uint64_t>(s.residual_blocks+1U)*bh2;float*fc1_activations=base;base+=static_cast<std::uint64_t>(s.residual_blocks)*bh2;float*cur=block_inputs;float*f1=nullptr;float*f2=nullptr;float*bn=base;float*scratch=bn+p.reduction_offset;auto site=[&](unsigned i,float*x){const auto&q=p.sites[i];return LaunchSelectedStridedSyncBatchNormForward(x,lr,gr,q.logical_features,q.physical_stride,aff+q.affine_offset,aff+p.logical_feature_count+q.affine_offset,run+q.running_offset,run+p.logical_feature_count+q.running_offset,.1f,1e-5f,x,bn+q.mean_offset,bn+q.inv_std_offset,bn+q.normalized_offset,scratch,ctx,st);};unsigned blocks2=(lr*s.hd2+T-1)/T;if(LaunchInputBackend(s,lh1,w,states,lr,a1,ctx,st)!=mgt::Status::kOk)return mgt::Status::kCudaFailure;if(site(0,a1)!=mgt::Status::kOk||LaunchBatchNormReluForward(a1,static_cast<std::uint64_t>(lr)*s.hd1,st)!=mgt::Status::kOk)return mgt::Status::kCudaFailure;if(Gemm(blas,a1,w+HW(s),cur,lr,s.hd2,s.hd1)!=mgt::Status::kOk)return mgt::Status::kCudaFailure;Bias<<<blocks2,T,0,st>>>(cur,w+HB(s),lr,s.hd2,lh2);if(!launch()||site(1,cur)!=mgt::Status::kOk||LaunchBatchNormReluForward(cur,static_cast<std::uint64_t>(lr)*s.hd2,st)!=mgt::Status::kOk)return mgt::Status::kCudaFailure;for(unsigned b=0;b<s.residual_blocks;b++){cur=block_inputs+static_cast<std::uint64_t>(b)*bh2;f1=fc1_activations+static_cast<std::uint64_t>(b)*bh2;f2=block_inputs+static_cast<std::uint64_t>(b+1U)*bh2;if(Gemm(blas,cur,w+F1W(s,b),f1,lr,s.hd2,s.hd2)!=mgt::Status::kOk)return mgt::Status::kCudaFailure;Bias<<<blocks2,T,0,st>>>(f1,w+F1B(s,b),lr,s.hd2,lh2);if(!launch()||site(2+2*b,f1)!=mgt::Status::kOk||LaunchBatchNormReluForward(f1,static_cast<std::uint64_t>(lr)*s.hd2,st)!=mgt::Status::kOk)return mgt::Status::kCudaFailure;if(Gemm(blas,f1,w+F2W(s,b),f2,lr,s.hd2,s.hd2)!=mgt::Status::kOk)return mgt::Status::kCudaFailure;Bias<<<blocks2,T,0,st>>>(f2,w+F2B(s,b),lr,s.hd2,lh2);if(!launch()||site(3+2*b,f2)!=mgt::Status::kOk||LaunchBatchNormResidualReluForward(cur,f2,static_cast<std::uint64_t>(lr)*s.hd2,st)!=mgt::Status::kOk)return mgt::Status::kCudaFailure;}cur=block_inputs+static_cast<std::uint64_t>(s.residual_blocks)*bh2;if(Gemm(blas,cur,w+OW(s),out,lr,s.output_dim,s.hd2)!=mgt::Status::kOk)return mgt::Status::kCudaFailure;unsigned bo=(lr*s.output_dim+T-1)/T;OutputBias<<<bo,T,0,st>>>(out,w+OB(s),lr,s.output_dim);return launch()?mgt::Status::kOk:mgt::Status::kCudaFailure;}
mgt::Status LaunchMlpBatchNormOutputLossGrad(const CudaMlpShape&s,const float*w,const float*labels,const float*outputs,const float*final_activation,std::uint32_t lr,std::uint32_t gr,float*loss,float*weight_grad,float*dy,float*d_final,NcclRankContext*ctx,cublasHandle_t blas,cudaStream_t st){if(ValidateCudaMlpShape(s)!=mgt::Status::kOk||!w||!labels||!outputs||!final_activation||!lr||gr<lr||!loss||!weight_grad||!dy||!d_final||!ctx||!blas)return mgt::Status::kInvalidConfig;if(cublasSetStream(blas,st)!=CUBLAS_STATUS_SUCCESS)return mgt::Status::kCudaFailure;const std::uint64_t wc=static_cast<std::uint64_t>(s.hd2)*s.output_dim,pc=wc+s.output_dim;if(cudaMemsetAsync(loss,0,sizeof(float),st)!=cudaSuccess||cudaMemsetAsync(weight_grad+OW(s),0,pc*sizeof(float),st)!=cudaSuccess)return mgt::Status::kCudaFailure;unsigned count=lr*s.output_dim,blocks=(count+T-1)/T;float alpha=1,beta=0;if(s.output_dim==1){ScalarOutputLossGrad<<<(lr+T-1)/T,T,0,st>>>(outputs,labels,lr,1.0f/static_cast<float>(gr),loss,dy,weight_grad+OB(s));if(!launch()||cublasSgemv(blas,CUBLAS_OP_N,s.hd2,lr,&alpha,final_activation,s.hd2,dy,1,&beta,weight_grad+OW(s),1)!=CUBLAS_STATUS_SUCCESS)return mgt::Status::kCudaFailure;ScalarOutputInputGrad<<<(lr*s.hd2+T-1)/T,T,0,st>>>(dy,w+OW(s),lr,s.hd2,d_final);if(!launch())return mgt::Status::kCudaFailure;}else{OutputLossGrad<<<blocks,T,0,st>>>(outputs,labels,count,s.output_dim,1.0f/(static_cast<float>(gr)*s.output_dim),loss,dy,weight_grad+OB(s));if(!launch()||cublasSgemm(blas,CUBLAS_OP_N,CUBLAS_OP_T,s.output_dim,s.hd2,lr,&alpha,dy,s.output_dim,final_activation,s.hd2,&beta,weight_grad+OW(s),s.output_dim)!=CUBLAS_STATUS_SUCCESS||cublasSgemm(blas,CUBLAS_OP_T,CUBLAS_OP_N,s.hd2,lr,s.output_dim,&alpha,w+OW(s),s.output_dim,dy,s.output_dim,&beta,d_final,s.hd2)!=CUBLAS_STATUS_SUCCESS)return mgt::Status::kCudaFailure;}if(NcclAllreduceSumFloat(weight_grad+OW(s),pc,ctx,st)!=mgt::Status::kOk||NcclAllreduceSumFloat(loss,1,ctx,st)!=mgt::Status::kOk)return mgt::Status::kCudaFailure;return mgt::Status::kOk;}
static mgt::Status ResidualFc2Impl(const CudaMlpShape&s,const float*w,const float*aff,std::uint32_t b,std::uint32_t lr,std::uint32_t gr,float*fw,const mgt::BatchNormTrainingPlan&p,float*block_grad,float*weight_grad,float*affine_grad,float*d_fc1,float*d_residual,NcclRankContext*ctx,cublasHandle_t blas,cudaStream_t st,bool reduce){if(ValidateCudaMlpShape(s)!=mgt::Status::kOk||!w||!aff||b>=s.residual_blocks||!lr||gr<lr||!fw||p.sites.size()!=2+2*s.residual_blocks||!block_grad||!weight_grad||!affine_grad||!d_fc1||!d_residual||!ctx||!blas)return mgt::Status::kInvalidConfig;if(cublasSetStream(blas,st)!=CUBLAS_STATUS_SUCCESS)return mgt::Status::kCudaFailure;const std::uint64_t bh2=static_cast<std::uint64_t>(lr)*s.hd2;float*block_inputs=fw+static_cast<std::uint64_t>(lr)*s.hd1;float*fc1_activations=block_inputs+static_cast<std::uint64_t>(s.residual_blocks+1U)*bh2;float*bn=fc1_activations+static_cast<std::uint64_t>(s.residual_blocks)*bh2;float*activated=block_inputs+static_cast<std::uint64_t>(b+1U)*bh2;float*fc1=fc1_activations+static_cast<std::uint64_t>(b)*bh2;const auto&q=p.sites[3+2*b];if(LaunchBatchNormResidualReluBackward(activated,block_grad,block_grad,d_residual,bh2,st)!=mgt::Status::kOk||LaunchSelectedStridedSyncBatchNormBackward(block_grad,lr,gr,q.logical_features,q.physical_stride,aff+q.affine_offset,bn+q.inv_std_offset,bn+q.normalized_offset,block_grad,affine_grad+q.affine_offset,affine_grad+p.logical_feature_count+q.affine_offset,bn+p.reduction_offset,ctx,st)!=mgt::Status::kOk)return mgt::Status::kCudaFailure;const std::uint64_t wc=static_cast<std::uint64_t>(s.hd2)*s.hd2,pc=wc+s.hd2;if(cudaMemsetAsync(weight_grad+F2W(s,b),0,pc*sizeof(float),st)!=cudaSuccess)return mgt::Status::kCudaFailure;float alpha=1,beta=0;if(cublasSgemm(blas,CUBLAS_OP_N,CUBLAS_OP_T,s.hd2,s.hd2,lr,&alpha,block_grad,s.hd2,fc1,s.hd2,&beta,weight_grad+F2W(s,b),s.hd2)!=CUBLAS_STATUS_SUCCESS||cublasSgemm(blas,CUBLAS_OP_T,CUBLAS_OP_N,s.hd2,lr,s.hd2,&alpha,w+F2W(s,b),s.hd2,block_grad,s.hd2,&beta,d_fc1,s.hd2)!=CUBLAS_STATUS_SUCCESS)return mgt::Status::kCudaFailure;ColumnSum<<<s.hd2,T,0,st>>>(block_grad,lr,q.logical_features,s.hd2,weight_grad+F2B(s,b));if(!launch()||(reduce&&NcclAllreduceSumFloat(weight_grad+F2W(s,b),pc,ctx,st)!=mgt::Status::kOk))return mgt::Status::kCudaFailure;return mgt::Status::kOk;}
mgt::Status LaunchMlpBatchNormResidualFc2Backward(const CudaMlpShape&s,const float*w,const float*aff,std::uint32_t b,std::uint32_t lr,std::uint32_t gr,float*fw,const mgt::BatchNormTrainingPlan&p,float*block_grad,float*weight_grad,float*affine_grad,float*d_fc1,float*d_residual,NcclRankContext*ctx,cublasHandle_t blas,cudaStream_t st){return ResidualFc2Impl(s,w,aff,b,lr,gr,fw,p,block_grad,weight_grad,affine_grad,d_fc1,d_residual,ctx,blas,st,true);}
static mgt::Status ResidualFc1Impl(const CudaMlpShape&s,const float*w,const float*aff,std::uint32_t b,std::uint32_t lr,std::uint32_t gr,float*fw,const mgt::BatchNormTrainingPlan&p,float*d_fc1,float*weight_grad,float*affine_grad,const float*d_residual,float*block_grad,NcclRankContext*ctx,cublasHandle_t blas,cudaStream_t st){const std::uint64_t bh2=static_cast<std::uint64_t>(lr)*s.hd2;float*block_inputs=fw+static_cast<std::uint64_t>(lr)*s.hd1;float*fc1_activations=block_inputs+static_cast<std::uint64_t>(s.residual_blocks+1U)*bh2;float*bn=fc1_activations+static_cast<std::uint64_t>(s.residual_blocks)*bh2;float*activated=fc1_activations+static_cast<std::uint64_t>(b)*bh2;float*input=block_inputs+static_cast<std::uint64_t>(b)*bh2;const auto&q=p.sites[2+2*b];if(LaunchBatchNormReluBackward(activated,d_fc1,d_fc1,bh2,st)!=mgt::Status::kOk||LaunchSelectedStridedSyncBatchNormBackward(d_fc1,lr,gr,q.logical_features,q.physical_stride,aff+q.affine_offset,bn+q.inv_std_offset,bn+q.normalized_offset,d_fc1,affine_grad+q.affine_offset,affine_grad+p.logical_feature_count+q.affine_offset,bn+p.reduction_offset,ctx,st)!=mgt::Status::kOk)return mgt::Status::kCudaFailure;const std::uint64_t wc=static_cast<std::uint64_t>(s.hd2)*s.hd2,pc=wc+s.hd2;if(cudaMemsetAsync(weight_grad+F1W(s,b),0,pc*sizeof(float),st)!=cudaSuccess)return mgt::Status::kCudaFailure;float alpha=1,beta=0;if(cublasSgemm(blas,CUBLAS_OP_N,CUBLAS_OP_T,s.hd2,s.hd2,lr,&alpha,d_fc1,s.hd2,input,s.hd2,&beta,weight_grad+F1W(s,b),s.hd2)!=CUBLAS_STATUS_SUCCESS||cublasSgemm(blas,CUBLAS_OP_T,CUBLAS_OP_N,s.hd2,lr,s.hd2,&alpha,w+F1W(s,b),s.hd2,d_fc1,s.hd2,&beta,block_grad,s.hd2)!=CUBLAS_STATUS_SUCCESS)return mgt::Status::kCudaFailure;ColumnSum<<<s.hd2,T,0,st>>>(d_fc1,lr,q.logical_features,s.hd2,weight_grad+F1B(s,b));AddInPlace<<<static_cast<unsigned>((bh2+T-1)/T),T,0,st>>>(block_grad,d_residual,bh2);return launch()?mgt::Status::kOk:mgt::Status::kCudaFailure;}
mgt::Status LaunchMlpBatchNormResidualStackBackward(const CudaMlpShape&s,const float*w,const float*aff,std::uint32_t lr,std::uint32_t gr,float*fw,const mgt::BatchNormTrainingPlan&p,float*block_grad,float*weight_grad,float*affine_grad,float*d_fc1,float*d_residual,NcclRankContext*ctx,cublasHandle_t blas,cudaStream_t st){if(ValidateCudaMlpShape(s)!=mgt::Status::kOk||!w||!aff||!lr||gr<lr||!fw||p.sites.size()!=2+2*s.residual_blocks||!block_grad||!weight_grad||!affine_grad||!d_fc1||!d_residual||!ctx||!blas)return mgt::Status::kInvalidConfig;for(std::uint32_t b=s.residual_blocks;b-->0;){auto z=ResidualFc2Impl(s,w,aff,b,lr,gr,fw,p,block_grad,weight_grad,affine_grad,d_fc1,d_residual,ctx,blas,st,false);if(z!=mgt::Status::kOk)return z;z=ResidualFc1Impl(s,w,aff,b,lr,gr,fw,p,d_fc1,weight_grad,affine_grad,d_residual,block_grad,ctx,blas,st);if(z!=mgt::Status::kOk)return z;}return NcclAllreduceSumFloat(weight_grad+R0(s),OW(s)-R0(s),ctx,st);}mgt::Status LaunchMlpBatchNormHiddenBackward(const CudaMlpShape&s,const float*w,const float*aff,std::uint32_t lr,std::uint32_t gr,float*fw,const mgt::BatchNormTrainingPlan&p,float*hidden_grad,float*weight_grad,float*affine_grad,float*input_grad,NcclRankContext*ctx,cublasHandle_t blas,cudaStream_t st){if(ValidateCudaMlpShape(s)!=mgt::Status::kOk||!w||!aff||!lr||gr<lr||!fw||p.sites.size()!=2+2*s.residual_blocks||!hidden_grad||!weight_grad||!affine_grad||!input_grad||!ctx||!blas)return mgt::Status::kInvalidConfig;if(cublasSetStream(blas,st)!=CUBLAS_STATUS_SUCCESS)return mgt::Status::kCudaFailure;const std::uint64_t bh2=static_cast<std::uint64_t>(lr)*s.hd2;float*a1=fw;float*hidden=fw+static_cast<std::uint64_t>(lr)*s.hd1;float*fc1=hidden+static_cast<std::uint64_t>(s.residual_blocks+1U)*bh2;float*bn=fc1+static_cast<std::uint64_t>(s.residual_blocks)*bh2;const auto&q=p.sites[1];if(LaunchBatchNormReluBackward(hidden,hidden_grad,hidden_grad,bh2,st)!=mgt::Status::kOk||LaunchSelectedStridedSyncBatchNormBackward(hidden_grad,lr,gr,q.logical_features,q.physical_stride,aff+q.affine_offset,bn+q.inv_std_offset,bn+q.normalized_offset,hidden_grad,affine_grad+q.affine_offset,affine_grad+p.logical_feature_count+q.affine_offset,bn+p.reduction_offset,ctx,st)!=mgt::Status::kOk)return mgt::Status::kCudaFailure;const std::uint64_t pc=static_cast<std::uint64_t>(s.hd1)*s.hd2+s.hd2;if(cudaMemsetAsync(weight_grad+HW(s),0,pc*sizeof(float),st)!=cudaSuccess)return mgt::Status::kCudaFailure;float alpha=1,beta=0;if(cublasSgemm(blas,CUBLAS_OP_N,CUBLAS_OP_T,s.hd2,s.hd1,lr,&alpha,hidden_grad,s.hd2,a1,s.hd1,&beta,weight_grad+HW(s),s.hd2)!=CUBLAS_STATUS_SUCCESS||cublasSgemm(blas,CUBLAS_OP_T,CUBLAS_OP_N,s.hd1,lr,s.hd2,&alpha,w+HW(s),s.hd2,hidden_grad,s.hd2,&beta,input_grad,s.hd1)!=CUBLAS_STATUS_SUCCESS)return mgt::Status::kCudaFailure;ColumnSum<<<s.hd2,T,0,st>>>(hidden_grad,lr,q.logical_features,s.hd2,weight_grad+HB(s));if(!launch()||NcclAllreduceSumFloat(weight_grad+HW(s),pc,ctx,st)!=mgt::Status::kOk)return mgt::Status::kCudaFailure;return mgt::Status::kOk;}
#ifdef MGT_LOCAL_MLP_IMPLEMENTATION
#undef LaunchMlpBatchNormInputBackward
#define LaunchMlpBatchNormInputBackward LaunchLocalMlpBatchNormInputBackwardStrictLegacy
#else
#define LaunchMlpBatchNormInputBackward LaunchMlpBatchNormInputBackwardStrictLegacy
#endif
mgt::Status LaunchMlpBatchNormInputBackward(const CudaMlpShape&s,const float*aff,const mgt::TrainStateStorage*states,std::uint32_t lr,std::uint32_t gr,float*fw,const mgt::BatchNormTrainingPlan&p,float*input_grad,float*weight_grad,float*affine_grad,NcclRankContext*ctx,cudaStream_t st){if(ValidateCudaMlpShape(s)!=mgt::Status::kOk||!aff||!states||!lr||gr<lr||!fw||p.sites.size()!=2+2*s.residual_blocks||!input_grad||!weight_grad||!affine_grad||!ctx)return mgt::Status::kInvalidConfig;float*a1=fw;float*fc1=fw+static_cast<std::uint64_t>(lr)*s.hd1+static_cast<std::uint64_t>(s.residual_blocks+1U)*lr*s.hd2;float*bn=fc1+static_cast<std::uint64_t>(s.residual_blocks)*lr*s.hd2;const auto&q=p.sites[0];const std::uint64_t n=static_cast<std::uint64_t>(lr)*s.hd1;if(LaunchBatchNormReluBackward(a1,input_grad,input_grad,n,st)!=mgt::Status::kOk||LaunchSelectedStridedSyncBatchNormBackward(input_grad,lr,gr,q.logical_features,q.physical_stride,aff+q.affine_offset,bn+q.inv_std_offset,bn+q.normalized_offset,input_grad,affine_grad+q.affine_offset,affine_grad+p.logical_feature_count+q.affine_offset,bn+p.reduction_offset,ctx,st)!=mgt::Status::kOk)return mgt::Status::kCudaFailure;const std::uint64_t pc=HW(s);if(cudaMemsetAsync(weight_grad,0,pc*sizeof(float),st)!=cudaSuccess)return mgt::Status::kCudaFailure;dim3 grid((s.hd1+INPUT_T-1)/INPUT_T,s.state_len);std::size_t shared=static_cast<std::size_t>(INPUT_T)*(s.state_value_pad+1U)*sizeof(float);SparseInputGrad<<<grid,INPUT_T,shared,st>>>(s,states,input_grad,lr,weight_grad);ColumnSum<<<s.hd1,T,0,st>>>(input_grad,lr,q.logical_features,s.hd1,weight_grad+IB(s));if(!launch()||NcclAllreduceSumFloat(weight_grad,pc,ctx,st)!=mgt::Status::kOk)return mgt::Status::kCudaFailure;return mgt::Status::kOk;}
#undef LaunchMlpBatchNormInputBackward
#ifdef MGT_LOCAL_MLP_IMPLEMENTATION
#define LaunchMlpBatchNormInputBackward LaunchLocalMlpBatchNormInputBackwardImpl
#endif
mgt::Status LaunchMlpBatchNormInputBackward(
    const CudaMlpShape& s, const float* aff, const mgt::TrainStateStorage* states,
    std::uint32_t lr, std::uint32_t gr, float* fw,
    const mgt::BatchNormTrainingPlan& p, float* input_grad, float* weight_grad,
    float* affine_grad, NcclRankContext* ctx, cudaStream_t st) {
    if (ValidateCudaMlpShape(s) != mgt::Status::kOk || !aff || !states || !lr || gr < lr ||
        !fw || p.sites.size() != 2 + 2 * s.residual_blocks || !input_grad ||
        !weight_grad || !affine_grad || !ctx) return mgt::Status::kInvalidConfig;
    const InputGradLaunchConfig input_launch = ResolveInputGradLaunch(s);
    if (input_launch.status != mgt::Status::kOk) return input_launch.status;
    if (!input_launch.exact) {
        return MGT_STRICT_INPUT_BACKWARD(
            s, aff, states, lr, gr, fw, p, input_grad, weight_grad, affine_grad, ctx, st);
    }
    float* a1 = fw;
    float* fc1 = fw + static_cast<std::uint64_t>(lr) * s.hd1 +
                 static_cast<std::uint64_t>(s.residual_blocks + 1U) * lr * s.hd2;
    float* bn = fc1 + static_cast<std::uint64_t>(s.residual_blocks) * lr * s.hd2;
    const auto& q = p.sites[0];
    const std::uint64_t n = static_cast<std::uint64_t>(lr) * s.hd1;
    if (LaunchBatchNormReluBackward(a1, input_grad, input_grad, n, st) != mgt::Status::kOk ||
        LaunchSelectedStridedSyncBatchNormBackward(
            input_grad, lr, gr, q.logical_features, q.physical_stride,
            aff + q.affine_offset, bn + q.inv_std_offset, bn + q.normalized_offset,
            input_grad, affine_grad + q.affine_offset,
            affine_grad + p.logical_feature_count + q.affine_offset,
            bn + p.reduction_offset, ctx, st) != mgt::Status::kOk) {
        return mgt::Status::kCudaFailure;
    }
    const std::uint64_t parameter_count = HW(s);
    if (cudaMemsetAsync(weight_grad, 0, parameter_count * sizeof(float), st) != cudaSuccess)
        return mgt::Status::kCudaFailure;
    if (input_launch.positions == 0) {
        const dim3 grid((s.hd1 + INPUT_T - 1U) / INPUT_T, s.state_len);
        SparseInputGradCoalesced96<<<grid, INPUT_T, input_launch.shared, st>>>(
            s, states, input_grad, lr, weight_grad);
    } else {
        const unsigned threads = GroupedInputThreads(input_launch.positions);
        if (cudaFuncSetAttribute(SparseInputGradExactGrouped,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 static_cast<int>(input_launch.shared)) != cudaSuccess) {
            return mgt::Status::kCudaFailure;
        }
        const dim3 grid((s.hd1 + threads - 1U) / threads,
                        (s.state_len + input_launch.positions - 1U) / input_launch.positions);
        SparseInputGradExactGrouped<<<grid, threads, input_launch.shared, st>>>(
            s, states, input_grad, lr, input_launch.positions, weight_grad);
    }
    if (!launch()) return mgt::Status::kCudaFailure;
    ColumnSum<<<s.hd1, T, 0, st>>>(
        input_grad, lr, q.logical_features, s.hd1, weight_grad + IB(s));
    if (!launch() || NcclAllreduceSumFloat(weight_grad, parameter_count, ctx, st) != mgt::Status::kOk)
        return mgt::Status::kCudaFailure;
    return mgt::Status::kOk;
}
mgt::Status LaunchMlpBatchNormAdamStep(const CudaMlpShape&s,const mgt::BatchNormTrainingPlan&p,const AdamWKernelConfig&base,float*w,const float*wg,float*wm,float*wv,float*aff,const float*ag,float*am,float*av,cudaStream_t st){if(ValidateCudaMlpShape(s)!=mgt::Status::kOk||p.logical_feature_count==0||!w||!wg||!wm||!wv||!aff||!ag||!am||!av)return mgt::Status::kInvalidConfig;AdamWKernelConfig wc=base;wc.param_count=OB(s)+s.output_dim;AdamWKernelConfig ac=base;ac.param_count=2ULL*p.logical_feature_count;if(LaunchAdamWKernel(wc,w,wg,wm,wv,st)!=mgt::Status::kOk||LaunchAdamWKernel(ac,aff,ag,am,av,st)!=mgt::Status::kOk)return mgt::Status::kCudaFailure;return mgt::Status::kOk;}
static mgt::Status LaunchAdamBackend(const CudaMlpShape&s,const mgt::BatchNormTrainingPlan&p,const AdamWKernelConfig&base,MlpBatchNormStepBuffers b,NcclRankContext*ctx,cudaStream_t st){
#ifdef MGT_LOCAL_MLP_IMPLEMENTATION
if(auto*fp=LocalFp16(ctx)){AdamWKernelConfig wc=base;wc.param_count=OB(s)+s.output_dim;AdamWKernelConfig ac=base;ac.param_count=2ULL*p.logical_feature_count;if(LaunchAdamWKernelWithHalfMirror(wc,b.weights,fp->weight_mirror,b.weight_grad,b.weight_m,b.weight_v,st)!=mgt::Status::kOk||LaunchAdamWKernel(ac,b.affine,b.affine_grad,b.affine_m,b.affine_v,st)!=mgt::Status::kOk)return mgt::Status::kCudaFailure;return mgt::Status::kOk;}
#endif
return LaunchMlpBatchNormAdamStep(s,p,base,b.weights,b.weight_grad,b.weight_m,b.weight_v,b.affine,b.affine_grad,b.affine_m,b.affine_v,st);}
mgt::Status LaunchMlpBatchNormTrainStep(const CudaMlpShape&s,std::uint32_t lh1,std::uint32_t lh2,const mgt::TrainStateStorage*states,const float*labels,std::uint32_t lr,std::uint32_t gr,const mgt::BatchNormTrainingPlan&p,std::uint64_t fw_count,const AdamWKernelConfig&adam,MlpBatchNormStepBuffers b,NcclRankContext*ctx,cublasHandle_t blas,cudaStream_t st){if(!states||!labels||!b.weights||!b.weight_grad||!b.weight_m||!b.weight_v||!b.affine||!b.affine_grad||!b.affine_m||!b.affine_v||!b.running||!b.outputs||!b.forward_workspace||!b.loss||!b.output_dy||!b.block_grad||!b.fc1_grad||!b.residual_grad||!b.input_grad)return mgt::Status::kInvalidConfig;auto z=LaunchMlpBatchNormForward(s,lh1,lh2,b.weights,b.affine,b.running,states,lr,gr,b.outputs,b.forward_workspace,fw_count,p,ctx,blas,st);if(z!=mgt::Status::kOk)return z;float*final_activation=b.forward_workspace+static_cast<std::uint64_t>(lr)*s.hd1+static_cast<std::uint64_t>(s.residual_blocks)*lr*s.hd2;z=LaunchMlpBatchNormOutputLossGrad(s,b.weights,labels,b.outputs,final_activation,lr,gr,b.loss,b.weight_grad,b.output_dy,b.block_grad,ctx,blas,st);if(z!=mgt::Status::kOk)return z;z=LaunchMlpBatchNormResidualStackBackward(s,b.weights,b.affine,lr,gr,b.forward_workspace,p,b.block_grad,b.weight_grad,b.affine_grad,b.fc1_grad,b.residual_grad,ctx,blas,st);if(z!=mgt::Status::kOk)return z;z=LaunchMlpBatchNormHiddenBackward(s,b.weights,b.affine,lr,gr,b.forward_workspace,p,b.block_grad,b.weight_grad,b.affine_grad,b.input_grad,ctx,blas,st);if(z!=mgt::Status::kOk)return z;z=LaunchMlpBatchNormInputBackward(s,b.affine,states,lr,gr,b.forward_workspace,p,b.input_grad,b.weight_grad,b.affine_grad,ctx,st);if(z!=mgt::Status::kOk)return z;return LaunchAdamBackend(s,p,adam,b,ctx,st);}
}
