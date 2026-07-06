#pragma once

#include "mgt_cuda/mlp_forward.cuh"

#include <cublas_v2.h>
#include <cublasLt.h>
#include <cuda_fp16.h>

namespace mgt_cuda {

struct MlpBackwardProfile {
    float input_forward_ms = 0.0f;
    float hidden_forward_ms = 0.0f;
    float residual_forward_ms = 0.0f;
    float output_ms = 0.0f;
    float residual_backward_ms = 0.0f;
    float hidden_backward_ms = 0.0f;
    float input_grad_ms = 0.0f;
};

struct LtMatmulAutotuneConfig {
    bool enabled = false;
    std::uint32_t max_candidates = 8;
    std::uint32_t warmup_iterations = 1;
    std::uint32_t timing_iterations = 2;
};

__host__ void ConfigureLtMatmulAutotune(const LtMatmulAutotuneConfig& config);
__host__ LtMatmulAutotuneConfig CurrentLtMatmulAutotuneConfig();

using MlpGradientReadyCallback = mgt::Status (*)(void* user,
                                                 std::uint32_t ready_id,
                                                 std::uint64_t param_offset,
                                                 std::uint64_t param_count,
                                                 cudaStream_t producer_stream);

__host__ std::uint64_t MlpLossGradWorkspaceFloats(const CudaMlpShape& shape,
                                                  std::uint32_t sample_count);

__host__ std::uint64_t MlpLossGradWorkspaceFloats(const CudaMlpShape& shape,
                                                  std::uint32_t sample_count,
                                                  std::uint32_t input_grad_partial_chunks);

__host__ std::uint64_t MlpLossGradWorkspaceFloats(const CudaMlpShape& shape,
                                                  std::uint32_t sample_count,
                                                  std::uint32_t input_grad_partial_chunks,
                                                  bool use_half_input_grad);

__host__ std::uint64_t MlpLossGradWorkspaceFloats(const CudaMlpShape& shape,
                                                  std::uint32_t sample_count,
                                                  std::uint32_t input_grad_partial_chunks,
                                                  bool use_half_input_grad,
                                                  bool use_half_linear);

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
                                                          cudaStream_t stream);

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
                                                          cudaStream_t stream);

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
                                                          cudaStream_t stream);

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
                                                          cudaStream_t stream);

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
                                                          cudaStream_t stream);

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
                                                            cudaStream_t stream);

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
                                                                  cudaStream_t stream);

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
                                                                  cudaStream_t stream);

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
                                                                  cudaStream_t stream);

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
                                                                  cudaStream_t stream);

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
                                                                  cudaStream_t stream);

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
                                                                    cudaStream_t stream);
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
                                                                        void* lt_workspace_base = nullptr,
                                                                        std::uint64_t lt_workspace_bytes = 0,
                                                                        std::uint32_t input_grad_position_tile = 0,
                                                                        bool input_grad_sparse = false);
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
                                                                                void* lt_workspace_base = nullptr,
                                                                                std::uint64_t lt_workspace_bytes = 0,
                                                                                std::uint32_t input_grad_position_tile = 0,
                                                                                bool input_grad_sparse = false);
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
                                                                                           void* lt_workspace_base = nullptr,
                                                                                           std::uint64_t lt_workspace_bytes = 0,
                                                                                           std::uint32_t input_grad_position_tile = 0,
                                                                                           bool input_grad_sparse = false);
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
                                                                               cudaStream_t stream);
__host__ mgt::Status LaunchMlpLossGradKernel(const CudaMlpShape& shape,
                                             const float* device_weights,
                                             const mgt::TrainStateStorage* device_states,
                                             const float* device_labels,
                                             std::uint32_t sample_count,
                                             float* device_loss,
                                             float* device_grad,
                                             cudaStream_t stream);

}  // namespace mgt_cuda