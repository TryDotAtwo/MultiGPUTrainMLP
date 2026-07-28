#include "sync_batch_norm.cuh"

int main() {
    const auto status = mgt_cuda::LaunchSyncBatchNormForward(
        nullptr, 1, 1, 1, nullptr, nullptr, nullptr, nullptr, 0.1f, 1.0e-5f,
        nullptr, nullptr, nullptr, nullptr, nullptr,
        static_cast<mgt_cuda::NcclRankContext*>(nullptr), nullptr);
    return status == mgt::Status::kInvalidConfig ? 0 : 1;
}