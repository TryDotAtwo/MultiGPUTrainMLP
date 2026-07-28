#pragma once
#include "mgt/batch_norm_training.hpp"
#include "mgt_cuda/mlp_forward.cuh"
#include <cublas_v2.h>
namespace mgt_cuda {
struct NcclRankContext;
std::uint64_t MlpBatchNormForwardWorkspaceFloats(const CudaMlpShape& physical_shape,const mgt::BatchNormTrainingPlan& batch_norm,std::uint32_t local_rows);
mgt::Status LaunchMlpBatchNormForward(const CudaMlpShape& physical_shape,std::uint32_t logical_hd1,std::uint32_t logical_hd2,const float* weights,float* batch_norm_affine,float* batch_norm_running,const mgt::TrainStateStorage* states,std::uint32_t local_rows,std::uint32_t global_rows,float* outputs,float* workspace,std::uint64_t workspace_floats,const mgt::BatchNormTrainingPlan& plan,NcclRankContext* context,cublasHandle_t blas,cudaStream_t stream);
mgt::Status LaunchMlpBatchNormOutputLossGrad(const CudaMlpShape& physical_shape,const float* weights,const float* labels,const float* outputs,const float* final_activation,std::uint32_t local_rows,std::uint32_t global_rows,float* loss,float* weight_grad,float* output_dy,float* final_activation_grad,NcclRankContext* context,cublasHandle_t blas,cudaStream_t stream);
}