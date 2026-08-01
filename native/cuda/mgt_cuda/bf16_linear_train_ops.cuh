#pragma once

#include "mgt/a100_bf16_algorithm_table.hpp"
#include "mgt/status.hpp"

#include <cstdint>

namespace mgt_cuda {

struct Bf16LinearProblem {
    std::uint32_t active_rows;
    std::uint32_t site_id;
    std::uint32_t compute_rows;
    std::uint32_t input_features;
    std::uint32_t output_features;
};

mgt::Status BuildBf16LinearGemmKey(const Bf16LinearProblem& problem,
                                    mgt::Bf16GemmRole role,
                                    float beta,
                                    mgt::Bf16GemmKeyV1* out);

}  // namespace mgt_cuda
