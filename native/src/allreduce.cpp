#include "mgt/allreduce.hpp"

namespace mgt {

Status ValidateAllreduceConfig(const AllreduceConfig& config) {
    if (config.world_size == 0 || config.element_count == 0) {
        return Status::kInvalidConfig;
    }
    if (config.global_rank >= config.world_size) {
        return Status::kInvalidConfig;
    }
    return Status::kOk;
}

Status AverageReducedGradients(float* gradients, std::size_t element_count, std::uint32_t world_size) {
    if (gradients == nullptr || element_count == 0 || world_size == 0) {
        return Status::kInvalidConfig;
    }
    const float inv_world = 1.0f / static_cast<float>(world_size);
    for (std::size_t i = 0; i < element_count; ++i) {
        gradients[i] *= inv_world;
    }
    return Status::kOk;
}

AllreduceLogEntry MakeAllreduceLogEntry(const AllreduceConfig& config, bool begin) {
    return AllreduceLogEntry{
        config.global_rank,
        config.collective_seq,
        config.element_count * sizeof(float),
        begin,
    };
}

bool SameCollectiveOrder(const AllreduceLogEntry* entries, std::size_t entry_count, std::uint32_t world_size) {
    if (entries == nullptr || world_size == 0 || entry_count == 0 || entry_count % world_size != 0) {
        return false;
    }
    for (std::size_t offset = 0; offset < entry_count; offset += world_size) {
        const std::uint64_t seq = entries[offset].collective_seq;
        const std::size_t bytes = entries[offset].bytes;
        const bool begin = entries[offset].begin;
        std::uint32_t seen_mask = 0;
        for (std::uint32_t rank_slot = 0; rank_slot < world_size; ++rank_slot) {
            const AllreduceLogEntry& entry = entries[offset + rank_slot];
            if (entry.collective_seq != seq || entry.bytes != bytes || entry.begin != begin) {
                return false;
            }
            if (entry.global_rank >= world_size || entry.global_rank >= 32) {
                return false;
            }
            const std::uint32_t rank_bit = 1u << entry.global_rank;
            if ((seen_mask & rank_bit) != 0) {
                return false;
            }
            seen_mask |= rank_bit;
        }
    }
    return true;
}

}  // namespace mgt