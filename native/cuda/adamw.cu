#include "mgt_cuda/adamw.cuh"

namespace mgt_cuda {

__host__ mgt::Status ValidateAdamWKernelConfig(const AdamWKernelConfig& config) {
    if (config.param_count == 0 || config.step == 0 || config.learning_rate <= 0.0f ||
        config.beta1 < 0.0f || config.beta1 >= 1.0f ||
        config.beta2 < 0.0f || config.beta2 >= 1.0f ||
        config.eps <= 0.0f || config.weight_decay < 0.0f) {
        return mgt::Status::kInvalidConfig;
    }
    return mgt::Status::kOk;
}

}  // namespace mgt_cuda