#include "mgt/mlp_cpu_ref.hpp"

#include <cmath>
#include <cstdlib>

int main() {
    float weight = 2.0f;
    const float grad = 3.0f;
    float m = 0.0f;
    float v = 0.0f;
    if (mgt::CpuAdamWStep(
            &weight, &grad, &m, &v, 1, 1,
            0.1f, 0.0f, 0.0f, 1.0f, 0.5f) != mgt::Status::kOk) {
        return EXIT_FAILURE;
    }
    const float expected_m = 3.0f;
    const float expected_v = 9.0f;
    const float expected_weight =
        2.0f - 0.1f * (3.0f / (3.0f + 1.0f) + 0.5f * 2.0f);
    if (std::fabs(m - expected_m) > 1.0e-6f ||
        std::fabs(v - expected_v) > 1.0e-6f ||
        std::fabs(weight - expected_weight) > 1.0e-6f) {
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
