#pragma once

#include "mgt/puzzle_io.hpp"
#include "mgt/status.hpp"

#include <cstdint>
#include <vector>

namespace mgt {

// Returns a sorted, position-major superset of every categorical input bin
// reachable from target through any sequence of puzzle moves. A superset is
// intentional when generators are not supplied with explicit inverses.
Status BuildInputActiveBins(
    const PuzzleDefinition& puzzle,
    std::uint32_t state_len,
    std::uint32_t state_value_pad,
    std::vector<std::uint16_t>* bins);

}  // namespace mgt
