#pragma once

#include "mgt_cuda/single_gpu_trainer.cuh"
#include <array>

namespace mgt_cuda::detail {

inline constexpr std::uint64_t kSingleGpuBlasWorkspaceBytes = 4ULL * 1024 * 1024;
inline constexpr bool kSingleGpuTrainGraphSupported = CUDART_VERSION >= 12080;

// Owned by SingleGpuTrainer. Source-node argument storage must outlive replay.
struct SingleGpuTrainGraph {
    cudaGraph_t source = nullptr;
    cudaGraphExec_t executable = nullptr;
    std::array<cudaGraphNode_t, 3> nodes{};
    std::array<cudaKernelNodeParams, 3> parameters{};
    std::uint32_t rows = 0;
};

const void* RandomWalkTrainingKernel();
const void* WeightAdamTrainingKernel();
const void* AffineAdamTrainingKernel();

// Takes ownership of source once arguments are accepted, including on failure.
mgt::Status InstantiateSingleGpuTrainGraph(
    cudaGraph_t source, std::uint32_t rows, std::uint64_t weight_count,
    std::uint64_t affine_count, SingleGpuTrainGraph* graph);
mgt::Status LaunchSingleGpuTrainGraph(
    SingleGpuTrainGraph* graph, const SingleGpuTrainStepRequest& request,
    cudaStream_t stream);
// Caller drains the stream first. Safe for partial initialization.
mgt::Status DestroySingleGpuTrainGraph(SingleGpuTrainGraph* graph);

}  // namespace mgt_cuda::detail
