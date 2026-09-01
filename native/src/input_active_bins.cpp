#include "mgt/input_active_bins.hpp"

#include "mgt/static_contracts.hpp"

#include <array>
#include <cstdint>
#include <new>
#include <utility>
#include <vector>

namespace mgt {
namespace {

class PositionComponents {
public:
    explicit PositionComponents(std::uint32_t size) : size_(size) {
        for (std::uint32_t i = 0; i < size_; ++i) parent_[i] = i;
    }

    std::uint32_t Find(std::uint32_t position) {
        while (parent_[position] != position) {
            parent_[position] = parent_[parent_[position]];
            position = parent_[position];
        }
        return position;
    }

    void Join(std::uint32_t lhs, std::uint32_t rhs) {
        lhs = Find(lhs);
        rhs = Find(rhs);
        if (lhs == rhs) return;
        if (rank_[lhs] < rank_[rhs]) std::swap(lhs, rhs);
        parent_[rhs] = lhs;
        if (rank_[lhs] == rank_[rhs]) ++rank_[lhs];
    }

private:
    std::uint32_t size_;
    std::array<std::uint32_t, kStateLen> parent_{};
    std::array<std::uint8_t, kStateLen> rank_{};
};

}  // namespace

Status BuildInputActiveBins(
    const PuzzleDefinition& puzzle,
    std::uint32_t state_len,
    std::uint32_t state_value_pad,
    std::vector<std::uint16_t>* bins) {
    if (bins == nullptr || state_len == 0 || state_len > kStateLen ||
        state_value_pad == 0 || state_value_pad > kStateLen) {
        return Status::kInvalidConfig;
    }

    PositionComponents components(state_len);
    for (const auto& move : puzzle.moves) {
        for (std::uint32_t position = 0; position < state_len; ++position) {
            const std::uint32_t source = move.v[position];
            if (source >= state_len) return Status::kInvalidPuzzle;
            components.Join(position, source);
        }
    }
    for (std::uint32_t source = 0; source < state_len; ++source) {
        if (puzzle.target.v[source] >= state_value_pad) {
            return Status::kInvalidPuzzle;
        }
    }

    try {
        std::vector<std::uint16_t> result;
        result.reserve(static_cast<std::size_t>(state_len) * state_value_pad);
        for (std::uint32_t position = 0; position < state_len; ++position) {
            std::array<bool, kStateLen> values{};
            const std::uint32_t root = components.Find(position);
            for (std::uint32_t source = 0; source < state_len; ++source) {
                if (components.Find(source) == root) {
                    values[puzzle.target.v[source]] = true;
                }
            }
            for (std::uint32_t value = 0; value < state_value_pad; ++value) {
                if (values[value]) {
                    result.push_back(static_cast<std::uint16_t>(
                        position * state_value_pad + value));
                }
            }
        }
        *bins = std::move(result);
    } catch (const std::bad_alloc&) {
        return Status::kCapacityExceeded;
    }
    return Status::kOk;
}

}  // namespace mgt
