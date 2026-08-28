#include "mgt_cuda/fp16_linear_train_ops.cuh"

#include <cmath>

int main() {
    constexpr unsigned rows = 3, in = 8, out = 8;
    float h_input[rows * in], h_weight[in * out], h_dy[rows * out];
    for (unsigned i = 0; i < rows * in; ++i) h_input[i] = .03f * (int(i % 11) - 5);
    for (unsigned i = 0; i < in * out; ++i) h_weight[i] = .02f * (int(i % 13) - 6);
    for (unsigned i = 0; i < rows * out; ++i) h_dy[i] = .01f * (int(i % 7) - 3);
    float *d_input_f, *d_weight_f, *d_dy_f, *d_y, *d_dw, *d_dx;
    __half *d_input, *d_weight, *d_dy;
    cudaMalloc(&d_input_f, sizeof(h_input)); cudaMalloc(&d_weight_f, sizeof(h_weight));
    cudaMalloc(&d_dy_f, sizeof(h_dy)); cudaMalloc(&d_input, sizeof(__half) * rows * in);
    cudaMalloc(&d_weight, sizeof(__half) * in * out); cudaMalloc(&d_dy, sizeof(__half) * rows * out);
    cudaMalloc(&d_y, sizeof(float) * rows * out); cudaMalloc(&d_dw, sizeof(float) * in * out);
    cudaMalloc(&d_dx, sizeof(float) * rows * in);
    cudaMemcpy(d_input_f, h_input, sizeof(h_input), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weight_f, h_weight, sizeof(h_weight), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dy_f, h_dy, sizeof(h_dy), cudaMemcpyHostToDevice);
    cublasHandle_t blas{}; cublasCreate(&blas);
    if (mgt_cuda::LaunchFloatToHalf(d_input_f, d_input, rows * in, 0) != mgt::Status::kOk ||
        mgt_cuda::LaunchFloatToHalf(d_weight_f, d_weight, in * out, 0) != mgt::Status::kOk ||
        mgt_cuda::LaunchFloatToHalf(d_dy_f, d_dy, rows * out, 0) != mgt::Status::kOk ||
        mgt_cuda::LaunchFp16LinearForward(blas, d_input, d_weight, d_y, rows, in, out, 0) != mgt::Status::kOk ||
        mgt_cuda::LaunchFp16LinearGradWeight(blas, d_input, d_dy, d_dw, rows, in, out, 0) != mgt::Status::kOk ||
        mgt_cuda::LaunchFp16LinearGradInput(blas, d_dy, d_weight, d_dx, rows, in, out, 0, 0) != mgt::Status::kOk ||
        cudaDeviceSynchronize() != cudaSuccess) return 1;
    float y[rows * out], dw[in * out], dx[rows * in];
    cudaMemcpy(y, d_y, sizeof(y), cudaMemcpyDeviceToHost);
    cudaMemcpy(dw, d_dw, sizeof(dw), cudaMemcpyDeviceToHost);
    cudaMemcpy(dx, d_dx, sizeof(dx), cudaMemcpyDeviceToHost);
    for (unsigned r = 0; r < rows; ++r) for (unsigned o = 0; o < out; ++o) {
        float expected = 0; for (unsigned k = 0; k < in; ++k) expected += h_input[r*in+k] * h_weight[k*out+o];
        if (std::fabs(y[r*out+o] - expected) > 2e-4f) return 2;
    }
    for (unsigned k = 0; k < in; ++k) for (unsigned o = 0; o < out; ++o) {
        float expected = 0; for (unsigned r = 0; r < rows; ++r) expected += h_input[r*in+k] * h_dy[r*out+o];
        if (std::fabs(dw[k*out+o] - expected) > 2e-4f) return 3;
    }
    for (unsigned r = 0; r < rows; ++r) for (unsigned k = 0; k < in; ++k) {
        float expected = 0; for (unsigned o = 0; o < out; ++o) expected += h_dy[r*out+o] * h_weight[k*out+o];
        if (std::fabs(dx[r*in+k] - expected) > 2e-4f) return 4;
    }
    cublasDestroy(blas);
    return 0;
}
