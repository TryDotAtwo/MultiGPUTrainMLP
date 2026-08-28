#include "mgt/config.hpp"

#include <cmath>
#include <cstdlib>

int main() {
    using Contract = mgt::P888TrainingContract;
    if (Contract::kStateValueCount != 72) return EXIT_FAILURE;
    if (Contract::kInputFeatures != 5184) return EXIT_FAILURE;
    if (Contract::kHidden1 != 2556) return EXIT_FAILURE;
    if (Contract::kHidden2 != 218) return EXIT_FAILURE;
    if (Contract::kResidualBlocks != 16) return EXIT_FAILURE;
    if (Contract::kBatchNormSites != 34) return EXIT_FAILURE;
    if (Contract::kMinDepth != 1 || Contract::kMaxDepth != 29) return EXIT_FAILURE;
    if (Contract::kWalkersPerDepth != 34482) return EXIT_FAILURE;
    if (Contract::kSamplesPerEpoch != 999978) return EXIT_FAILURE;
    if (Contract::kGlobalBatch != 100000) return EXIT_FAILURE;
    if (Contract::kFinalGlobalBatch != 99978) return EXIT_FAILURE;
    if (Contract::kOptimizerStepsPerEpoch != 10) return EXIT_FAILURE;
    if (Contract::kEpochs != 32692) return EXIT_FAILURE;
    if (Contract::kOptimizerSteps != 326920) return EXIT_FAILURE;
    if (std::fabs(Contract::kLearningRate - 1.0e-4f) > 1.0e-12f) return EXIT_FAILURE;
    if (Contract::kWeightDecay != 0.0f) return EXIT_FAILURE;
    if (std::fabs(Contract::kBatchNormEpsilon - 1.0e-5f) > 1.0e-12f) return EXIT_FAILURE;
    if (std::fabs(Contract::kBatchNormMomentum - 0.1f) > 1.0e-7f) return EXIT_FAILURE;
    return EXIT_SUCCESS;
}
