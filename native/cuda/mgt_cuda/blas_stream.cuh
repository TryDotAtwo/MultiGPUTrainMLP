#pragma once

#include <cublas_v2.h>

namespace mgt_cuda::detail {

// cuBLAS unconditionally discards a user workspace on cublasSetStream, even
// when the stream is unchanged. A prepared handle belongs to one stream: keep
// its workspace in that case. A real stream change retains cuBLAS reset semantics.
inline cublasStatus_t BindBlasStream(cublasHandle_t blas, cudaStream_t stream) {
    if (!blas) return CUBLAS_STATUS_NOT_INITIALIZED;
    cudaStream_t current = nullptr;
    const auto status = cublasGetStream(blas, &current);
    if (status != CUBLAS_STATUS_SUCCESS || current == stream) return status;
    return cublasSetStream(blas, stream);
}

}  // namespace mgt_cuda::detail
