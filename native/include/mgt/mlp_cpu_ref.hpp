#pragma once

#include "mgt/static_contracts.hpp"
#include <cstdint>
#include <span>
#include <vector>

namespace mgt {

struct CpuMlpShape {
    std::uint32_t state_len;
    std::uint32_t state_value_pad;
    std::uint32_t hd1;
    std::uint32_t hd2;
    std::uint32_t residual_blocks;
    std::uint32_t output_dim;
};

std::uint64_t CpuMlpParamCount(const CpuMlpShape& shape);

struct CpuMlpBatchNormState {
    std::vector<float> affine;
    std::vector<float> running;
};

std::uint64_t CpuMlpBatchNormFeatureCount(const CpuMlpShape& shape);
CpuMlpBatchNormState InitializeCpuMlpBatchNormState(const CpuMlpShape& shape);

Status CpuMlpBatchNormForward(const CpuMlpShape& shape,
                              std::span<const float> weights,
                              CpuMlpBatchNormState* batch_norm,
                              const TrainStateStorage* states,
                              std::uint32_t sample_count,
                              float momentum,
                              float epsilon,
                              bool training,
                              float* outputs);

Status CpuMlpForward(const CpuMlpShape& shape,
                     std::span<const float> weights,
                     const TrainStateStorage* states,
                     std::uint32_t sample_count,
                     float* outputs);

Status CpuMlpLossAndGrad(const CpuMlpShape& shape,
                         std::span<const float> weights,
                         const TrainStateStorage* states,
                         const float* labels,
                         std::uint32_t sample_count,
                         float* loss,
                         float* grad);

Status CpuAdamWStep(float* weights,
                    const float* grad,
                    float* m,
                    float* v,
                    std::uint64_t param_count,
                    std::uint64_t step,
                    float lr,
                    float beta1,
                    float beta2,
                    float eps,
                    float weight_decay);

}  // namespace mgt