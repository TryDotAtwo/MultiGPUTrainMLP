#include "mgt/distributed_batch.hpp"

namespace mgt {

Status PartitionGlobalBatch(
    std::uint32_t global_rows,
    std::uint32_t world,
    std::uint32_t rank,
    DistributedBatchSlice* out) {
    if (out == nullptr || global_rows == 0 || world == 0 ||
        world > global_rows || rank >= world) {
        return Status::kInvalidConfig;
    }

    const std::uint32_t base = global_rows / world;
    const std::uint32_t remainder = global_rows % world;
    out->active_rows = base + (rank < remainder ? 1U : 0U);
    out->global_offset = static_cast<std::uint64_t>(rank) * base +
                         (rank < remainder ? rank : remainder);
    return Status::kOk;
}

}  // namespace mgt
