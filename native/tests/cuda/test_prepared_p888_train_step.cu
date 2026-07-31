#include "mgt_cuda/prepared_p888_train_step.cuh"

#include <cstdint>

namespace {

template <class T>
T* Fake(std::uintptr_t address) {
    return reinterpret_cast<T*>(address);
}

mgt_cuda::PreparedP888StrictRuntimeCreateInfo ValidInfo() {
    mgt_cuda::PreparedP888StrictRuntimeCreateInfo info{};
    info.shape = {2, 4, 3, 2, 1, 1};
    info.capacity_rows = 4;
    static const std::uint32_t rows[] = {4, 3};
    info.supported_active_rows = rows;
    info.supported_active_row_count = 2;
    info.world = 1;
    info.rank = 0;
    info.buffers = {
        Fake<float>(0x1000), Fake<float>(0x2000), Fake<float>(0x3000),
        Fake<float>(0x4000), Fake<float>(0x5000), Fake<float>(0x6000),
        Fake<float>(0x7000), Fake<float>(0x8000), Fake<float>(0x9000),
        Fake<float>(0xa000), Fake<float>(0xb000), Fake<float>(0xc000),
        Fake<float>(0xd000), Fake<float>(0xe000), Fake<float>(0xf000),
        Fake<float>(0x10000), Fake<float>(0x11000)};
    if (mgt::BuildBatchNormTrainingPlan(3, 2, 3, 2, 1, 4,
                                        &info.batch_norm_plan) != mgt::Status::kOk) {
        info.capacity_rows = 0;
    }
    info.adam = {1, 1, 0.001f, 0.9f, 0.999f, 1e-8f, 0.0f};
    info.state_slots = {Fake<mgt::TrainStateStorage>(0x12000),
                        Fake<mgt::TrainStateStorage>(0x13000)};
    info.label_slots = {Fake<float>(0x14000), Fake<float>(0x15000)};
    return info;
}

}  // namespace

int main() {
    auto info = ValidInfo();
    std::uint64_t bytes = 0;
    if (mgt_cuda::QueryPreparedP888StrictRuntimeBytes(info, &bytes) != mgt::Status::kOk ||
        bytes == 0) return 1;

    auto duplicate = info;
    static const std::uint32_t duplicate_rows[] = {4, 4};
    duplicate.supported_active_rows = duplicate_rows;
    if (mgt_cuda::QueryPreparedP888StrictRuntimeBytes(duplicate, &bytes) !=
        mgt::Status::kInvalidConfig) return 2;

    auto excessive = info;
    static const std::uint32_t excessive_rows[] = {5};
    excessive.supported_active_rows = excessive_rows;
    excessive.supported_active_row_count = 1;
    if (mgt_cuda::QueryPreparedP888StrictRuntimeBytes(excessive, &bytes) !=
        mgt::Status::kInvalidConfig) return 3;

    auto multirank = info;
    multirank.world = 8;
    if (mgt_cuda::QueryPreparedP888StrictRuntimeBytes(multirank, &bytes) !=
        mgt::Status::kInvalidConfig) return 4;
    return 0;
}
