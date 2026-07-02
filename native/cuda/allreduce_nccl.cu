#include "mgt_cuda/allreduce_nccl.cuh"

#include <nccl.h>

namespace mgt_cuda {

struct NcclSingleRankContext {
    ncclComm_t comm;
    std::uint32_t device_id;
};

namespace {

mgt::Status NcclStatus(ncclResult_t result) {
    return result == ncclSuccess ? mgt::Status::kOk : mgt::Status::kNcclFailure;
}

}  // namespace

mgt::Status CreateNcclSingleRankContext(std::uint32_t device_id, NcclSingleRankContext** context) {
    if (context == nullptr) {
        return mgt::Status::kInvalidConfig;
    }
    *context = nullptr;
    auto* created = new NcclSingleRankContext{};
    created->device_id = device_id;
    cudaError_t cuda_status = cudaSetDevice(static_cast<int>(device_id));
    if (cuda_status != cudaSuccess) {
        delete created;
        return mgt::Status::kCudaFailure;
    }
    ncclUniqueId id{};
    ncclResult_t nccl_status = ncclGetUniqueId(&id);
    if (nccl_status == ncclSuccess) {
        nccl_status = ncclCommInitRank(&created->comm, 1, id, 0);
    }
    if (nccl_status != ncclSuccess) {
        delete created;
        return mgt::Status::kNcclFailure;
    }
    *context = created;
    return mgt::Status::kOk;
}

mgt::Status DestroyNcclSingleRankContext(NcclSingleRankContext* context) {
    if (context == nullptr) {
        return mgt::Status::kOk;
    }
    const ncclResult_t status = ncclCommDestroy(context->comm);
    delete context;
    return NcclStatus(status);
}

mgt::Status NcclAllreduceAverageFloat(const mgt::AllreduceConfig& config,
                                      float* device_gradients,
                                      NcclSingleRankContext* context,
                                      cudaStream_t stream) {
    if (mgt::ValidateAllreduceConfig(config) != mgt::Status::kOk || device_gradients == nullptr || context == nullptr) {
        return mgt::Status::kInvalidConfig;
    }
    if (config.world_size != 1 || config.global_rank != 0) {
        return mgt::Status::kInvalidConfig;
    }
    const ncclResult_t status = ncclAllReduce(device_gradients,
                                              device_gradients,
                                              config.element_count,
                                              ncclFloat32,
                                              ncclSum,
                                              context->comm,
                                              stream);
    if (status != ncclSuccess) {
        return mgt::Status::kNcclFailure;
    }
    return mgt::Status::kOk;
}

}  // namespace mgt_cuda