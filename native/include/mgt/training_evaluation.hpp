#pragma once

#include "mgt/static_contracts.hpp"

#include <cstddef>
#include <cstdint>

namespace mgt {

struct RegressionMetrics {
    double mse = 0.0;
    double mae = 0.0;
    std::uint64_t sample_count = 0;
};

std::uint64_t HeldoutSeed(std::uint64_t training_seed);

Status ValidateEvaluationSchedule(std::uint32_t eval_samples,
                                  std::uint32_t batch_samples,
                                  std::uint64_t period_steps);

Status ComputeRegressionMetrics(const float* predictions,
                                const float* labels,
                                std::size_t count,
                                RegressionMetrics* metrics);

}  // namespace mgt
