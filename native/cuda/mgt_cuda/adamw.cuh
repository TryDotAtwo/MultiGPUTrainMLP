#pragma once

#include "mgt/status.hpp"
#include <cstdint>

namespace mgt_cuda {

struct AdamWKernelConfig {
    std::uint64_t param_count;
    std::uint64_t step;
    float learning_rate;
    float beta1;
    float beta2;
    float eps;
    float weight_decay;
};

__host__ mgt::Status ValidateAdamWKernelConfig(const AdamWKernelConfig& config);

}  // namespace mgt_cuda