#include "mgt/static_contracts.hpp"

namespace mgt {

static_assert(sizeof(TrainStateStorage) == kStateStorageLen);
static_assert(alignof(TrainStateStorage) == 16);
static_assert(sizeof(WalkMeta) == 16);
static_assert(alignof(WalkMeta) == 16);
static_assert(sizeof(LossStats) == 16);
static_assert(alignof(LossStats) == 16);
static_assert(sizeof(TensorBlockHeader) == 32);
static_assert(alignof(TensorBlockHeader) == 32);

Status ValidateStaticContracts() {
    if (sizeof(TrainStateStorage) != kStateStorageLen || alignof(TrainStateStorage) != 16) {
        return Status::kInvalidConfig;
    }
    if (sizeof(WalkMeta) != 16 || alignof(WalkMeta) != 16) {
        return Status::kInvalidConfig;
    }
    if (sizeof(LossStats) != 16 || alignof(LossStats) != 16) {
        return Status::kInvalidConfig;
    }
    if (sizeof(TensorBlockHeader) != 32 || alignof(TensorBlockHeader) != 32) {
        return Status::kInvalidConfig;
    }
    return Status::kOk;
}

}  // namespace mgt