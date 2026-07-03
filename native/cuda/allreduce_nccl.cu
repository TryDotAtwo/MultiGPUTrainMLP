#include "mgt_cuda/allreduce_nccl.cuh"

#include <nccl.h>
#include <chrono>
#include <fstream>
#include <thread>

namespace mgt_cuda {

struct NcclRankContext {
    ncclComm_t comm;
    std::uint32_t device_id;
    std::uint32_t world_size;
    std::uint32_t global_rank;
};

namespace {

__global__ void ScaleKernel(float* values, std::size_t count, float scale) {
    const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < count) {
        values[idx] *= scale;
    }
}

mgt::Status NcclStatus(ncclResult_t result) {
    return result == ncclSuccess ? mgt::Status::kOk : mgt::Status::kNcclFailure;
}

mgt::Status WriteUniqueId(const std::filesystem::path& id_file, const ncclUniqueId& id) {
    std::filesystem::create_directories(id_file.parent_path());
    std::ofstream out(id_file, std::ios::binary | std::ios::trunc);
    if (!out) return mgt::Status::kIoFailure;
    out.write(reinterpret_cast<const char*>(&id), sizeof(id));
    return out ? mgt::Status::kOk : mgt::Status::kIoFailure;
}

mgt::Status ReadUniqueId(const std::filesystem::path& id_file, ncclUniqueId* id) {
    for (int attempt = 0; attempt < 300; ++attempt) {
        std::ifstream in(id_file, std::ios::binary);
        if (in) {
            in.read(reinterpret_cast<char*>(id), sizeof(*id));
            if (in.gcount() == static_cast<std::streamsize>(sizeof(*id))) {
                return mgt::Status::kOk;
            }
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    return mgt::Status::kIoFailure;
}

}  // namespace

mgt::Status CreateNcclSingleRankContext(std::uint32_t device_id, NcclRankContext** context) {
    return CreateNcclRankContext(device_id, 1, 0, {}, context);
}

mgt::Status CreateNcclRankContext(std::uint32_t device_id,
                                  std::uint32_t world_size,
                                  std::uint32_t global_rank,
                                  const std::filesystem::path& id_file,
                                  NcclRankContext** context) {
    if (context == nullptr || world_size == 0 || global_rank >= world_size) {
        return mgt::Status::kInvalidConfig;
    }
    if (world_size > 1 && id_file.empty()) {
        return mgt::Status::kInvalidConfig;
    }
    *context = nullptr;
    auto* created = new NcclRankContext{};
    created->device_id = device_id;
    created->world_size = world_size;
    created->global_rank = global_rank;
    cudaError_t cuda_status = cudaSetDevice(static_cast<int>(device_id));
    if (cuda_status != cudaSuccess) {
        delete created;
        return mgt::Status::kCudaFailure;
    }

    ncclUniqueId id{};
    ncclResult_t nccl_status = ncclSuccess;
    if (world_size == 1) {
        nccl_status = ncclGetUniqueId(&id);
    } else if (global_rank == 0) {
        nccl_status = ncclGetUniqueId(&id);
        if (nccl_status == ncclSuccess) {
            const mgt::Status write_status = WriteUniqueId(id_file, id);
            if (write_status != mgt::Status::kOk) {
                delete created;
                return write_status;
            }
        }
    } else {
        const mgt::Status read_status = ReadUniqueId(id_file, &id);
        if (read_status != mgt::Status::kOk) {
            delete created;
            return read_status;
        }
    }
    if (nccl_status == ncclSuccess) {
        nccl_status = ncclCommInitRank(&created->comm, static_cast<int>(world_size), id, static_cast<int>(global_rank));
    }
    if (nccl_status != ncclSuccess) {
        delete created;
        return mgt::Status::kNcclFailure;
    }
    *context = created;
    return mgt::Status::kOk;
}

mgt::Status DestroyNcclRankContext(NcclRankContext* context) {
    if (context == nullptr) {
        return mgt::Status::kOk;
    }
    const ncclResult_t status = ncclCommDestroy(context->comm);
    delete context;
    return NcclStatus(status);
}

mgt::Status NcclAllreduceAverageFloat(const mgt::AllreduceConfig& config,
                                      float* device_gradients,
                                      NcclRankContext* context,
                                      cudaStream_t stream) {
    if (mgt::ValidateAllreduceConfig(config) != mgt::Status::kOk || device_gradients == nullptr || context == nullptr) {
        return mgt::Status::kInvalidConfig;
    }
    if (config.world_size != context->world_size || config.global_rank != context->global_rank) {
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
    if (config.world_size > 1) {
        const std::size_t threads = 256;
        const std::size_t blocks = (config.element_count + threads - 1) / threads;
        ScaleKernel<<<static_cast<unsigned int>(blocks), static_cast<unsigned int>(threads), 0, stream>>>(
            device_gradients, config.element_count, 1.0f / static_cast<float>(config.world_size));
        if (cudaGetLastError() != cudaSuccess) {
            return mgt::Status::kCudaFailure;
        }
    }
    return mgt::Status::kOk;
}

}  // namespace mgt_cuda