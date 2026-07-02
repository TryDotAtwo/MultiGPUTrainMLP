#include "mgt_cuda/random_walk_kernel.cuh"

namespace mgt_cuda {

__host__ mgt::Status ValidateRandomWalkKernelConfig(const RandomWalkKernelConfig& config) {
    if (config.sample_count == 0 || config.k_min == 0 || config.k_min > config.k_max) {
        return mgt::Status::kInvalidConfig;
    }
    return mgt::Status::kOk;
}

}  // namespace mgt_cuda