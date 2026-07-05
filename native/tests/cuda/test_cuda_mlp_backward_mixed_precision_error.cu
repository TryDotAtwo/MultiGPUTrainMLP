#include "mgt_cuda/mlp_backward.cuh"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <cublasLt.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

int Check(cudaError_t status) { return status == cudaSuccess ? 0 : 1; }

std::uint64_t ResidualBlockParams(const mgt_cuda::CudaMlpShape& shape) {
    return 2ULL * (static_cast<std::uint64_t>(shape.hd2) * shape.hd2 + shape.hd2);
}

std::uint64_t ParamCount(const mgt_cuda::CudaMlpShape& shape) {
    return static_cast<std::uint64_t>(shape.state_len) * shape.state_value_pad * shape.hd1 + shape.hd1 +
           static_cast<std::uint64_t>(shape.hd1) * shape.hd2 + shape.hd2 +
           static_cast<std::uint64_t>(shape.residual_blocks) * ResidualBlockParams(shape) + static_cast<std::uint64_t>(shape.hd2) * shape.output_dim + shape.output_dim;
}

bool CloseLoss(float lhs, float rhs) {
    const float scale = std::fmax(std::fabs(lhs), std::fabs(rhs));
    return std::fabs(lhs - rhs) <= 2.0e-3f + 2.0e-2f * scale;
}

int RunCase(const mgt_cuda::CudaMlpShape& shape, std::uint32_t samples) {
    const std::uint64_t params = ParamCount(shape);

    std::vector<float> weights(params), labels(static_cast<std::uint64_t>(samples) * shape.output_dim), grad_ref(params), grad_mixed(params), grad_tile(params);
    std::vector<mgt::TrainStateStorage> states(samples);
    for (std::uint64_t i = 0; i < params; ++i) {
        weights[i] = static_cast<float>((static_cast<int>((i * 5 + shape.output_dim) % 31) - 15) * 0.0015);
    }
    for (std::uint32_t sample = 0; sample < samples; ++sample) {
        for (std::uint32_t out = 0; out < shape.output_dim; ++out) {
            labels[static_cast<std::uint64_t>(sample) * shape.output_dim + out] = static_cast<float>(static_cast<int>((sample + 3U * out) % 7) - 3) * 0.125f;
        }
        for (std::uint32_t pos = 0; pos < shape.state_len; ++pos) {
            states[sample].v[pos] = static_cast<mgt::StateValue>((sample + 3U * pos + shape.output_dim) % shape.state_value_pad);
        }
        for (std::uint32_t pos = shape.state_len; pos < mgt::kStateStorageLen; ++pos) {
            states[sample].v[pos] = 0;
        }
    }

    float* d_weights = nullptr;
    __half* d_weights_half = nullptr;
    float* d_labels = nullptr;
    float* d_loss_ref = nullptr;
    float* d_loss_mixed = nullptr;
    float* d_loss_tile = nullptr;
    float* d_grad_ref = nullptr;
    float* d_grad_mixed = nullptr;
    float* d_grad_tile = nullptr;
    float* d_workspace_ref = nullptr;
    float* d_workspace_mixed = nullptr;
    mgt::TrainStateStorage* d_states = nullptr;
    cublasHandle_t blas_ref = nullptr;
    cublasHandle_t blas_mixed = nullptr;
    cublasLtHandle_t blas_lt = nullptr;

    const std::uint64_t workspace_ref = mgt_cuda::MlpLossGradWorkspaceFloats(shape, samples, 1, false, false);
    const std::uint64_t workspace_mixed = mgt_cuda::MlpLossGradWorkspaceFloats(shape, samples, 1, true, true);
    if (workspace_ref == 0 || workspace_mixed == 0) return EXIT_FAILURE;

    if (Check(cudaMalloc(&d_weights, params * sizeof(float))) != 0) return EXIT_FAILURE;
    if (Check(cudaMalloc(&d_weights_half, params * sizeof(__half))) != 0) return EXIT_FAILURE;
    if (Check(cudaMalloc(&d_labels, labels.size() * sizeof(float))) != 0) return EXIT_FAILURE;
    if (Check(cudaMalloc(&d_loss_ref, sizeof(float))) != 0) return EXIT_FAILURE;
    if (Check(cudaMalloc(&d_loss_mixed, sizeof(float))) != 0) return EXIT_FAILURE;
    if (Check(cudaMalloc(&d_loss_tile, sizeof(float))) != 0) return EXIT_FAILURE;
    if (Check(cudaMalloc(&d_grad_ref, params * sizeof(float))) != 0) return EXIT_FAILURE;
    if (Check(cudaMalloc(&d_grad_mixed, params * sizeof(float))) != 0) return EXIT_FAILURE;
    if (Check(cudaMalloc(&d_grad_tile, params * sizeof(float))) != 0) return EXIT_FAILURE;
    if (Check(cudaMalloc(&d_workspace_ref, workspace_ref * sizeof(float))) != 0) return EXIT_FAILURE;
    if (Check(cudaMalloc(&d_workspace_mixed, workspace_mixed * sizeof(float))) != 0) return EXIT_FAILURE;
    if (Check(cudaMalloc(&d_states, samples * sizeof(mgt::TrainStateStorage))) != 0) return EXIT_FAILURE;
    if (cublasCreate(&blas_ref) != CUBLAS_STATUS_SUCCESS) return EXIT_FAILURE;
    if (cublasCreate(&blas_mixed) != CUBLAS_STATUS_SUCCESS) return EXIT_FAILURE;
    if (cublasLtCreate(&blas_lt) != CUBLAS_STATUS_SUCCESS) return EXIT_FAILURE;

    if (Check(cudaMemcpy(d_weights, weights.data(), params * sizeof(float), cudaMemcpyHostToDevice)) != 0) return EXIT_FAILURE;
    std::vector<__half> weights_half(params);
    for (std::uint64_t i = 0; i < params; ++i) weights_half[i] = __float2half_rn(weights[i]);
    if (Check(cudaMemcpy(d_weights_half, weights_half.data(), params * sizeof(__half), cudaMemcpyHostToDevice)) != 0) return EXIT_FAILURE;
    if (Check(cudaMemcpy(d_labels, labels.data(), labels.size() * sizeof(float), cudaMemcpyHostToDevice)) != 0) return EXIT_FAILURE;
    if (Check(cudaMemcpy(d_states, states.data(), samples * sizeof(mgt::TrainStateStorage), cudaMemcpyHostToDevice)) != 0) return EXIT_FAILURE;

    mgt::Status status = mgt_cuda::LaunchMlpLossGradKernelWithWorkspace(shape, d_weights, d_states, d_labels, samples, d_loss_ref, d_grad_ref,
                                                                          d_workspace_ref, workspace_ref, blas_ref, 1, 1, false, false, 0);
    if (status != mgt::Status::kOk) {
        std::fprintf(stderr, "float32 path output_dim=%u status=%d\n", shape.output_dim, static_cast<int>(status));
        return EXIT_FAILURE;
    }
    status = mgt_cuda::LaunchMlpLossGradKernelWithWorkspace(shape, d_weights, d_states, d_labels, samples, d_loss_mixed, d_grad_mixed,
                                                           d_workspace_mixed, workspace_mixed, blas_mixed, 1, 1, true, true, 0);
    if (status != mgt::Status::kInvalidConfig) {
        std::fprintf(stderr, "mixed without Lt output_dim=%u status=%d\n", shape.output_dim, static_cast<int>(status));
        return EXIT_FAILURE;
    }
    status = mgt_cuda::LaunchMlpLossGradKernelWithWorkspaceLt(shape, d_weights, d_states, d_labels, samples, d_loss_mixed, d_grad_mixed,
                                                             d_workspace_mixed, workspace_mixed, blas_mixed, blas_lt, 1, 1, true, true, 0);
    if (status != mgt::Status::kOk) {
        std::fprintf(stderr, "mixed Lt/CUTLASS output_dim=%u status=%d\n", shape.output_dim, static_cast<int>(status));
        return EXIT_FAILURE;
    }
    status = mgt_cuda::LaunchMlpLossGradKernelWithWorkspaceLtExternalHalf(shape, d_weights, d_weights_half, d_states, d_labels, samples, d_loss_tile, d_grad_tile,
                                                                            d_workspace_mixed, workspace_mixed, blas_mixed, blas_lt, 1, 1, true, true, 0, nullptr, 0, 4);
    if (status != mgt::Status::kOk) {
        std::fprintf(stderr, "position-tiled mixed Lt/CUTLASS output_dim=%u status=%d\n", shape.output_dim, static_cast<int>(status));
        return EXIT_FAILURE;
    }
    if (Check(cudaDeviceSynchronize()) != 0) return EXIT_FAILURE;

    float loss_ref = 0.0f;
    float loss_mixed = 0.0f;
    float loss_tile = 0.0f;
    if (Check(cudaMemcpy(&loss_ref, d_loss_ref, sizeof(float), cudaMemcpyDeviceToHost)) != 0) return EXIT_FAILURE;
    if (Check(cudaMemcpy(&loss_mixed, d_loss_mixed, sizeof(float), cudaMemcpyDeviceToHost)) != 0) return EXIT_FAILURE;
    if (Check(cudaMemcpy(&loss_tile, d_loss_tile, sizeof(float), cudaMemcpyDeviceToHost)) != 0) return EXIT_FAILURE;
    if (!std::isfinite(loss_ref) || !std::isfinite(loss_mixed) || !CloseLoss(loss_ref, loss_mixed)) {
        std::fprintf(stderr, "mixed loss mismatch output_dim=%u ref=%0.9g mixed=%0.9g\n", shape.output_dim, loss_ref, loss_mixed);
        return EXIT_FAILURE;
    }
    if (!std::isfinite(loss_tile) || !CloseLoss(loss_ref, loss_tile)) {
        std::fprintf(stderr, "position-tiled loss mismatch output_dim=%u ref=%0.9g tile=%0.9g\n", shape.output_dim, loss_ref, loss_tile);
        return EXIT_FAILURE;
    }

    if (Check(cudaMemcpy(grad_ref.data(), d_grad_ref, params * sizeof(float), cudaMemcpyDeviceToHost)) != 0) return EXIT_FAILURE;
    if (Check(cudaMemcpy(grad_mixed.data(), d_grad_mixed, params * sizeof(float), cudaMemcpyDeviceToHost)) != 0) return EXIT_FAILURE;
    if (Check(cudaMemcpy(grad_tile.data(), d_grad_tile, params * sizeof(float), cudaMemcpyDeviceToHost)) != 0) return EXIT_FAILURE;

    double diff2 = 0.0;
    double ref2 = 0.0;
    double max_abs = 0.0;
    for (std::uint64_t i = 0; i < params; ++i) {
        if (!std::isfinite(grad_ref[i]) || !std::isfinite(grad_mixed[i])) return EXIT_FAILURE;
        const double diff = static_cast<double>(grad_mixed[i]) - static_cast<double>(grad_ref[i]);
        diff2 += diff * diff;
        ref2 += static_cast<double>(grad_ref[i]) * static_cast<double>(grad_ref[i]);
        max_abs = std::fmax(max_abs, std::fabs(diff));
    }
    const double rel_l2 = std::sqrt(diff2 / std::fmax(ref2, 1.0e-30));
    if (rel_l2 > 8.0e-2 || max_abs > 2.0e-2) {
        std::fprintf(stderr, "mixed grad error output_dim=%u rel_l2=%0.9g max_abs=%0.9g loss_ref=%0.9g loss_mixed=%0.9g\n", shape.output_dim, rel_l2, max_abs, loss_ref, loss_mixed);
        return EXIT_FAILURE;
    }

    double tile_diff2 = 0.0;
    double tile_ref2 = 0.0;
    double tile_max_abs = 0.0;
    for (std::uint64_t i = 0; i < params; ++i) {
        if (!std::isfinite(grad_tile[i])) return EXIT_FAILURE;
        const double diff = static_cast<double>(grad_tile[i]) - static_cast<double>(grad_ref[i]);
        tile_diff2 += diff * diff;
        tile_ref2 += static_cast<double>(grad_ref[i]) * static_cast<double>(grad_ref[i]);
        tile_max_abs = std::fmax(tile_max_abs, std::fabs(diff));
    }
    const double rel_l2_tile = std::sqrt(tile_diff2 / std::fmax(tile_ref2, 1.0e-30));
    if (rel_l2_tile > 8.0e-2 || tile_max_abs > 2.0e-2) {
        std::fprintf(stderr, "position-tiled grad error output_dim=%u rel_l2=%0.9g max_abs=%0.9g loss_ref=%0.9g loss_tile=%0.9g\n", shape.output_dim, rel_l2_tile, tile_max_abs, loss_ref, loss_tile);
        return EXIT_FAILURE;
    }

    cublasDestroy(blas_mixed);
    cublasLtDestroy(blas_lt);
    cublasDestroy(blas_ref);
    cudaFree(d_states);
    cudaFree(d_workspace_mixed);
    cudaFree(d_workspace_ref);
    cudaFree(d_grad_mixed);
    cudaFree(d_grad_tile);
    cudaFree(d_grad_ref);
    cudaFree(d_loss_mixed);
    cudaFree(d_loss_tile);
    cudaFree(d_loss_ref);
    cudaFree(d_labels);
    cudaFree(d_weights);
    cudaFree(d_weights_half);
    return EXIT_SUCCESS;
}

}  // namespace

int main() {
    int device_count = 0;
    if (Check(cudaGetDeviceCount(&device_count)) != 0 || device_count <= 0) return EXIT_FAILURE;
    if (Check(cudaSetDevice(0)) != 0) return EXIT_FAILURE;

    const struct TestCase {
        mgt_cuda::CudaMlpShape shape;
        std::uint32_t samples;
    } cases[] = {
        {{8, 8, 16, 16, 2, 1}, 32},
        {{8, 8, 16, 16, 2, 3}, 32},
        {{8, 8, 16, 16, 2, 17}, 32},
        {{5, 8, 8, 8, 1, 128}, 16},
    };

    for (const TestCase& test_case : cases) {
        if (RunCase(test_case.shape, test_case.samples) != EXIT_SUCCESS) return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
