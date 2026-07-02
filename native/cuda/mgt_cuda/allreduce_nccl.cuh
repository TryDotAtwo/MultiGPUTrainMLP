#pragma once

#include "mgt/allreduce.hpp"
#include "mgt/status.hpp"
#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>

namespace mgt_cuda {

struct NcclSingleRankContext;

mgt::Status CreateNcclSingleRankContext(std::uint32_t device_id, NcclSingleRankContext** context);
mgt::Status DestroyNcclSingleRankContext(NcclSingleRankContext* context);
mgt::Status NcclAllreduceAverageFloat(const mgt::AllreduceConfig& config,
                                      float* device_gradients,
                                      NcclSingleRankContext* context,
                                      cudaStream_t stream);

}  // namespace mgt_cuda