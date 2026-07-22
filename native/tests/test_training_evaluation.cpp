#include "mgt/training_evaluation.hpp"

#include <cmath>
#include <cstdlib>

int main() {
    const std::uint64_t training_seed = 1234;
    const std::uint64_t heldout = mgt::HeldoutSeed(training_seed);
    if (heldout == training_seed || heldout != mgt::HeldoutSeed(training_seed)) return EXIT_FAILURE;
    if (heldout == mgt::HeldoutSeed(training_seed + 1)) return EXIT_FAILURE;

    if (mgt::ValidateEvaluationSchedule(4096, 57344, 100) != mgt::Status::kOk ||
        mgt::ValidateEvaluationSchedule(0, 57344, 0) != mgt::Status::kOk ||
        mgt::ValidateEvaluationSchedule(4096, 57344, 0) != mgt::Status::kInvalidConfig ||
        mgt::ValidateEvaluationSchedule(60000, 57344, 100) != mgt::Status::kInvalidConfig) {
        return EXIT_FAILURE;
    }

    const float predictions[] = {1.0f, 4.0f, 8.0f};
    const float labels[] = {2.0f, 4.0f, 5.0f};
    mgt::RegressionMetrics metrics{};
    if (mgt::ComputeRegressionMetrics(predictions, labels, 3, &metrics) != mgt::Status::kOk) {
        return EXIT_FAILURE;
    }
    if (std::abs(metrics.mse - (10.0 / 3.0)) > 1.0e-6 ||
        std::abs(metrics.mae - (4.0 / 3.0)) > 1.0e-6 || metrics.sample_count != 3) {
        return EXIT_FAILURE;
    }
    if (mgt::ComputeRegressionMetrics(nullptr, labels, 3, &metrics) != mgt::Status::kInvalidConfig ||
        mgt::ComputeRegressionMetrics(predictions, labels, 0, &metrics) != mgt::Status::kInvalidConfig) {
        return EXIT_FAILURE;
    }
    const float bad[] = {NAN};
    if (mgt::ComputeRegressionMetrics(bad, labels, 1, &metrics) != mgt::Status::kInvalidConfig) {
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
