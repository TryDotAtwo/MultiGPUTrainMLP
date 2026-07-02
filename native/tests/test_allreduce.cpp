#include "mgt/allreduce.hpp"

#include <cmath>
#include <cstdlib>

int main() {
    const mgt::AllreduceConfig one_rank{1, 0, 7, 4};
    if (mgt::ValidateAllreduceConfig(one_rank) != mgt::Status::kOk) return EXIT_FAILURE;

    const mgt::AllreduceConfig bad_rank{2, 2, 7, 4};
    if (mgt::ValidateAllreduceConfig(bad_rank) != mgt::Status::kInvalidConfig) return EXIT_FAILURE;

    float gradients[4] = {2.0f, -4.0f, 8.0f, 1.0f};
    if (mgt::AverageReducedGradients(gradients, 4, 2) != mgt::Status::kOk) return EXIT_FAILURE;
    if (std::fabs(gradients[0] - 1.0f) > 1.0e-6f) return EXIT_FAILURE;
    if (std::fabs(gradients[1] + 2.0f) > 1.0e-6f) return EXIT_FAILURE;
    if (std::fabs(gradients[2] - 4.0f) > 1.0e-6f) return EXIT_FAILURE;
    if (std::fabs(gradients[3] - 0.5f) > 1.0e-6f) return EXIT_FAILURE;

    const mgt::AllreduceLogEntry ordered[4] = {
        mgt::MakeAllreduceLogEntry({2, 0, 10, 256}, true),
        mgt::MakeAllreduceLogEntry({2, 1, 10, 256}, true),
        mgt::MakeAllreduceLogEntry({2, 0, 10, 256}, false),
        mgt::MakeAllreduceLogEntry({2, 1, 10, 256}, false),
    };
    if (!mgt::SameCollectiveOrder(ordered, 4, 2)) return EXIT_FAILURE;

    const mgt::AllreduceLogEntry broken[2] = {
        mgt::MakeAllreduceLogEntry({2, 0, 10, 256}, true),
        mgt::MakeAllreduceLogEntry({2, 1, 11, 256}, true),
    };
    if (mgt::SameCollectiveOrder(broken, 2, 2)) return EXIT_FAILURE;

    return EXIT_SUCCESS;
}