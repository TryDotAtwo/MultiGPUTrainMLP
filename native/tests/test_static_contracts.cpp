#include "mgt/static_contracts.hpp"
#include <cstdlib>

int main() {
    if (mgt::kStateLen != 72) return EXIT_FAILURE;
    if (mgt::kStateStorageLen != 80) return EXIT_FAILURE;
    if (mgt::kMoveCount != 18) return EXIT_FAILURE;
    if (mgt::kOutputDim != 1) return EXIT_FAILURE;
    if (sizeof(mgt::TrainStateStorage) != mgt::kStateStorageLen) return EXIT_FAILURE;
    if (alignof(mgt::TrainStateStorage) != 16) return EXIT_FAILURE;
    if (sizeof(mgt::WalkMeta) != 16) return EXIT_FAILURE;
    if (sizeof(mgt::LossStats) != 16) return EXIT_FAILURE;
    if (sizeof(mgt::TensorBlockHeader) != 32) return EXIT_FAILURE;
    if (mgt::ValidateStaticContracts() != mgt::Status::kOk) return EXIT_FAILURE;
    return EXIT_SUCCESS;
}
