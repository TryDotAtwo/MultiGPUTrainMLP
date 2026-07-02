#pragma once

namespace mgt {

enum class Status {
    kOk = 0,
    kInvalidConfig = 1,
    kInvalidPuzzle = 2,
    kCapacityExceeded = 3,
    kCudaFailure = 4,
    kNcclFailure = 5,
    kIoFailure = 6
};

}  // namespace mgt