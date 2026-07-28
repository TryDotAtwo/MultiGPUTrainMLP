#pragma once

#include "mgt/allreduce.hpp"
#include "mgt/status.hpp"
#include <cuda_runtime.h>
#include <filesystem>
#include <cstddef>
#include <cstdint>

namespace mgt_cuda {

struct NcclRankContext;

mgt::Status CreateNcclSingleRankContext(std::uint32_t device_id, NcclRankContext** context);
mgt::Status CreateNcclRankContext(std::uint32_t device_id,
                                  std::uint32_t world_size,
                                  std::uint32_t global_rank,
                                  const std::filesystem::path& id_file,
                                  NcclRankContext** context);
mgt::Status DestroyNcclRankContext(NcclRankContext* context);
mgt::Status NcclAllreduceAverageFloat(const mgt::AllreduceConfig& config,
                                      float* device_gradients,
                                      NcclRankContext* context,
                                      cudaStream_t stream);
mgt::Status NcclAllreduceSumFloat(float* device_values,
                                  std::size_t element_count,
                                  NcclRankContext* context,
                                  cudaStream_t stream);

using NcclSingleRankContext = NcclRankContext;
inline mgt::Status DestroyNcclSingleRankContext(NcclSingleRankContext* context) {
    return DestroyNcclRankContext(context);
}

}  // namespace mgt_cuda