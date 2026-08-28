#include "mgt_cuda/fp16_linear_train_ops.cuh"

namespace mgt_cuda {
namespace {
constexpr unsigned kThreads = 256;

__global__ void FloatToHalf(const float* input, __half* output, std::uint64_t count) {
    const auto index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) output[index] = __float2half_rn(input[index]);
}

bool Valid(cublasHandle_t blas, const void* a, const void* b, float* c,
           std::uint32_t rows, std::uint32_t input_features,
           std::uint32_t output_features) {
    return blas && a && b && c && rows && input_features && output_features;
}
}

mgt::Status LaunchFloatToHalf(
    const float* input, __half* output, std::uint64_t count, cudaStream_t stream) {
    if (!input || !output || !count) return mgt::Status::kInvalidConfig;
    FloatToHalf<<<static_cast<unsigned>((count + kThreads - 1) / kThreads),
                  kThreads, 0, stream>>>(input, output, count);
    return cudaPeekAtLastError() == cudaSuccess ? mgt::Status::kOk : mgt::Status::kCudaFailure;
}

mgt::Status LaunchFp16LinearForward(
    cublasHandle_t blas, const __half* input, const __half* weight,
    float* output, std::uint32_t rows, std::uint32_t input_features,
    std::uint32_t output_features, cudaStream_t stream) {
    if (!Valid(blas, input, weight, output, rows, input_features, output_features) ||
        cublasSetStream(blas, stream) != CUBLAS_STATUS_SUCCESS)
        return mgt::Status::kInvalidConfig;
    const float alpha = 1.0f, beta = 0.0f;
    const auto status = cublasGemmEx(
        blas, CUBLAS_OP_N, CUBLAS_OP_N, output_features, rows, input_features,
        &alpha, weight, CUDA_R_16F, output_features,
        input, CUDA_R_16F, input_features, &beta,
        output, CUDA_R_32F, output_features, CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    return status == CUBLAS_STATUS_SUCCESS ? mgt::Status::kOk : mgt::Status::kCudaFailure;
}

mgt::Status LaunchFp16LinearGradWeight(
    cublasHandle_t blas, const __half* input, const __half* grad_output,
    float* grad_weight, std::uint32_t rows, std::uint32_t input_features,
    std::uint32_t output_features, cudaStream_t stream) {
    if (!Valid(blas, input, grad_output, grad_weight, rows, input_features, output_features) ||
        cublasSetStream(blas, stream) != CUBLAS_STATUS_SUCCESS)
        return mgt::Status::kInvalidConfig;
    const float alpha = 1.0f, beta = 0.0f;
    const auto status = cublasGemmEx(
        blas, CUBLAS_OP_N, CUBLAS_OP_T, output_features, input_features, rows,
        &alpha, grad_output, CUDA_R_16F, output_features,
        input, CUDA_R_16F, input_features, &beta,
        grad_weight, CUDA_R_32F, output_features, CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    return status == CUBLAS_STATUS_SUCCESS ? mgt::Status::kOk : mgt::Status::kCudaFailure;
}

mgt::Status LaunchFp16LinearGradInput(
    cublasHandle_t blas, const __half* grad_output, const __half* weight,
    float* grad_input, std::uint32_t rows, std::uint32_t input_features,
    std::uint32_t output_features, float beta, cudaStream_t stream) {
    if (!Valid(blas, grad_output, weight, grad_input, rows, input_features, output_features) ||
        cublasSetStream(blas, stream) != CUBLAS_STATUS_SUCCESS)
        return mgt::Status::kInvalidConfig;
    const float alpha = 1.0f;
    const auto status = cublasGemmEx(
        blas, CUBLAS_OP_T, CUBLAS_OP_N, input_features, rows, output_features,
        &alpha, weight, CUDA_R_16F, output_features,
        grad_output, CUDA_R_16F, output_features, &beta,
        grad_input, CUDA_R_32F, input_features, CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    return status == CUBLAS_STATUS_SUCCESS ? mgt::Status::kOk : mgt::Status::kCudaFailure;
}

}  // namespace mgt_cuda
