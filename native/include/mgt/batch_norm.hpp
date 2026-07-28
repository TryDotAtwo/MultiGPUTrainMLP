#pragma once

#include <vector>

namespace mgt {

struct BatchNormCache {
    int rows = 0;
    int cols = 0;
    std::vector<float> mean;
    std::vector<float> variance;
    std::vector<float> inv_std;
    std::vector<float> normalized;
};

void batch_norm_forward_cpu(
    const float* x, int rows, int cols,
    const float* gamma, const float* beta,
    float* running_mean, float* running_var,
    float momentum, float epsilon, bool training,
    float* y, BatchNormCache* cache);

void batch_norm_backward_cpu(
    const float* dy, const BatchNormCache& cache,
    const float* gamma, float* dx,
    float* dgamma, float* dbeta);

}  // namespace mgt