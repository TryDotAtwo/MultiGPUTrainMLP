#include "mgt/distributed_batch.hpp"

#include <array>
#include <cstdint>
#include <cstdlib>

namespace {

bool ExpectPartition(
    std::uint32_t global_rows,
    const std::uint32_t* expected_rows,
    std::uint32_t world) {
    std::uint64_t next_offset = 0;
    std::uint64_t covered = 0;
    for (std::uint32_t rank = 0; rank < world; ++rank) {
        mgt::DistributedBatchSlice slice{};
        if (mgt::PartitionGlobalBatch(global_rows, world, rank, &slice) !=
            mgt::Status::kOk) {
            return false;
        }
        if (slice.active_rows != expected_rows[rank] ||
            slice.global_offset != next_offset) {
            return false;
        }
        next_offset += expected_rows[rank];
        covered += slice.active_rows;
    }
    return covered == global_rows && next_offset == global_rows;
}

}  // namespace

int main() {
    constexpr std::array<std::uint32_t, 8> kFull8{
        12500, 12500, 12500, 12500, 12500, 12500, 12500, 12500};
    constexpr std::array<std::uint32_t, 8> kTail8{
        12498, 12498, 12497, 12497, 12497, 12497, 12497, 12497};
    constexpr std::array<std::uint32_t, 4> kTail4{
        24995, 24995, 24994, 24994};

    if (!ExpectPartition(100000, kFull8.data(), 8) ||
        !ExpectPartition(99978, kTail8.data(), 8) ||
        !ExpectPartition(99978, kTail4.data(), 4)) {
        return EXIT_FAILURE;
    }

    mgt::DistributedBatchSlice slice{};
    if (mgt::PartitionGlobalBatch(99978, 8, 2, &slice) != mgt::Status::kOk ||
        slice.active_rows != 12497 || slice.global_offset != 24996) {
        return EXIT_FAILURE;
    }

    if (mgt::PartitionGlobalBatch(0, 8, 0, &slice) !=
            mgt::Status::kInvalidConfig ||
        mgt::PartitionGlobalBatch(8, 0, 0, &slice) !=
            mgt::Status::kInvalidConfig ||
        mgt::PartitionGlobalBatch(8, 9, 0, &slice) !=
            mgt::Status::kInvalidConfig ||
        mgt::PartitionGlobalBatch(8, 8, 8, &slice) !=
            mgt::Status::kInvalidConfig ||
        mgt::PartitionGlobalBatch(8, 8, 0, nullptr) !=
            mgt::Status::kInvalidConfig) {
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
