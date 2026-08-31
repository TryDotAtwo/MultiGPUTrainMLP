#include "mgt_cuda/single_gpu_train_graph.cuh"
#include "mgt_cuda/random_walk_kernel.cuh"

#include <algorithm>
#include <new>
#include <vector>
#if CUDART_VERSION >= 12080
#include <cuda.h>
#endif

namespace mgt_cuda::detail {

mgt::Status InstantiateSingleGpuTrainGraph(
    cudaGraph_t source, std::uint32_t rows, std::uint64_t weight_count,
    std::uint64_t affine_count, SingleGpuTrainGraph* graph) {
    if (!source || !graph || graph->source || graph->executable || !rows ||
        !weight_count || !affine_count) return mgt::Status::kInvalidConfig;
    graph->source = source;
#if CUDART_VERSION >= 12080
    try {
        const std::array<const void*, 3> symbols{
            RandomWalkTrainingKernel(), WeightAdamTrainingKernel(), AffineAdamTrainingKernel()};
        std::array<CUfunction, 3> functions{};
        for (unsigned i = 0; i < symbols.size(); ++i) {
            cudaFunction_t function = nullptr;
            if (cudaGetFuncBySymbol(&function, symbols[i]) != cudaSuccess)
                return mgt::Status::kCudaFailure;
            functions[i] = reinterpret_cast<CUfunction>(function);
        }
        std::size_t count = 0;
        if (cudaGraphGetNodes(source, nullptr, &count) != cudaSuccess)
            return mgt::Status::kCudaFailure;
        std::vector<cudaGraphNode_t> nodes(count);
        if (cudaGraphGetNodes(source, nodes.data(), &count) != cudaSuccess)
            return mgt::Status::kCudaFailure;
        for (auto node : nodes) {
            cudaGraphNodeType type;
            if (cudaGraphNodeGetType(node, &type) != cudaSuccess)
                return mgt::Status::kCudaFailure;
            if (type == cudaGraphNodeTypeMemAlloc || type == cudaGraphNodeTypeMemFree)
                return mgt::Status::kInvalidConfig;
            if (type != cudaGraphNodeTypeKernel) continue;
            // cuBLAS nodes may not have a runtime host-function registration.
            CUDA_KERNEL_NODE_PARAMS driver_params{};
            if (cuGraphKernelNodeGetParams(reinterpret_cast<CUgraphNode>(node),
                                           &driver_params) != CUDA_SUCCESS)
                return mgt::Status::kCudaFailure;
            for (unsigned i = 0; i < functions.size(); ++i) {
                if (driver_params.func != functions[i]) continue;
                if (graph->nodes[i]) return mgt::Status::kInvalidConfig;
                auto& params = graph->parameters[i];
                if (cudaGraphKernelNodeGetParams(node, &params) != cudaSuccess)
                    return mgt::Status::kCudaFailure;
                if (params.func != symbols[i] || !params.kernelParams || params.extra)
                    return mgt::Status::kInvalidConfig;
                graph->nodes[i] = node;
            }
        }
        for (auto node : graph->nodes) if (!node) return mgt::Status::kInvalidConfig;
        const auto& walk = *static_cast<const RandomWalkKernelConfig*>(graph->parameters[0].kernelParams[0]);
        const auto& weight = *static_cast<const AdamWKernelConfig*>(graph->parameters[1].kernelParams[0]);
        const auto& affine = *static_cast<const AdamWKernelConfig*>(graph->parameters[2].kernelParams[0]);
        if (walk.sample_count != rows || !walk.original_p888_schedule ||
            weight.param_count != weight_count || affine.param_count != affine_count)
            return mgt::Status::kInvalidConfig;
        if (cudaGraphInstantiate(&graph->executable, source, 0) != cudaSuccess)
            return mgt::Status::kCudaFailure;
        graph->rows = rows;
        return mgt::Status::kOk;
    } catch (const std::bad_alloc&) {
        return mgt::Status::kCapacityExceeded;
    }
#else
    return mgt::Status::kInvalidConfig;
#endif
}

mgt::Status LaunchSingleGpuTrainGraph(
    SingleGpuTrainGraph* graph, const SingleGpuTrainStepRequest& request,
    cudaStream_t stream) {
    if (!graph || !graph->executable || request.active_rows != graph->rows ||
        !request.optimizer_step) return mgt::Status::kInvalidConfig;
    auto walk_params = graph->parameters[0];
    std::array<void*, 10> walk_args{};
    std::copy_n(walk_params.kernelParams, walk_args.size(), walk_args.begin());
    auto walk = *static_cast<const RandomWalkKernelConfig*>(walk_args[0]);
    auto epoch = request.epoch;
    auto step = request.optimizer_step;
    walk.epoch_sample_offset = request.epoch_sample_offset;
    if (ValidateRandomWalkKernelConfig(walk) != mgt::Status::kOk)
        return mgt::Status::kInvalidConfig;
    walk_args[0] = &walk;
    walk_args[2] = &epoch;
    walk_args[3] = &step;
    walk_params.kernelParams = walk_args.data();
    if (cudaGraphExecKernelNodeSetParams(graph->executable, graph->nodes[0], &walk_params) != cudaSuccess)
        return mgt::Status::kCudaFailure;
    for (unsigned i = 1; i < graph->nodes.size(); ++i) {
        auto params = graph->parameters[i];
        std::array<void*, 6> args{};
        std::copy_n(params.kernelParams, i == 1 ? 6 : 5, args.begin());
        auto adam = *static_cast<const AdamWKernelConfig*>(args[0]);
        adam.step = request.optimizer_step;
        args[0] = &adam;
        params.kernelParams = args.data();
        if (cudaGraphExecKernelNodeSetParams(graph->executable, graph->nodes[i], &params) != cudaSuccess)
            return mgt::Status::kCudaFailure;
    }
    return cudaGraphLaunch(graph->executable, stream) == cudaSuccess
        ? mgt::Status::kOk : mgt::Status::kCudaFailure;
}

mgt::Status DestroySingleGpuTrainGraph(SingleGpuTrainGraph* graph) {
    if (!graph) return mgt::Status::kInvalidConfig;
    auto status = mgt::Status::kOk;
    if (graph->executable && cudaGraphExecDestroy(graph->executable) != cudaSuccess)
        status = mgt::Status::kCudaFailure;
    if (graph->source && cudaGraphDestroy(graph->source) != cudaSuccess)
        status = mgt::Status::kCudaFailure;
    *graph = {};
    return status;
}

}  // namespace mgt_cuda::detail
