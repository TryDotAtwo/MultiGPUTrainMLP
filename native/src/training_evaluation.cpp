#include "mgt/training_evaluation.hpp"

#include <cmath>

namespace mgt {

std::uint64_t HeldoutSeed(std::uint64_t training_seed) {
    std::uint64_t x = training_seed ^ 0xd1b54a32d192ed03ULL;
    x ^= x >> 30;
    x *= 0xbf58476d1ce4e5b9ULL;
    x ^= x >> 27;
    x *= 0x94d049bb133111ebULL;
    x ^= x >> 31;
    return x == training_seed ? x ^ 0x9e3779b97f4a7c15ULL : x;
}

Status ValidateEvaluationSchedule(std::uint32_t eval_samples,
                                  std::uint32_t batch_samples,
                                  std::uint64_t period_steps) {
    if (eval_samples == 0 && period_steps == 0) return Status::kOk;
    if (eval_samples == 0 || period_steps == 0 || batch_samples == 0 || eval_samples > batch_samples) {
        return Status::kInvalidConfig;
    }
    return Status::kOk;
}

Status ComputeRegressionMetrics(const float* predictions,
                                const float* labels,
                                std::size_t count,
                                RegressionMetrics* metrics) {
    if (predictions == nullptr || labels == nullptr || metrics == nullptr || count == 0) {
        return Status::kInvalidConfig;
    }
    double squared_error = 0.0;
    double absolute_error = 0.0;
    for (std::size_t i = 0; i < count; ++i) {
        if (!std::isfinite(predictions[i]) || !std::isfinite(labels[i])) {
            return Status::kInvalidConfig;
        }
        const double error = static_cast<double>(predictions[i]) - labels[i];
        squared_error += error * error;
        absolute_error += std::abs(error);
    }
    metrics->mse = squared_error / static_cast<double>(count);
    metrics->mae = absolute_error / static_cast<double>(count);
    metrics->sample_count = count;
    return Status::kOk;
}

}  // namespace mgt
