#include "mgt/mlp_cpu_ref.hpp"
#include <cmath>
#include <cstdlib>
#include <vector>

int main() {
    const mgt::CpuMlpShape shape{4, 8, 5, 3, 1, 1};
    const std::uint64_t params = mgt::CpuMlpParamCount(shape);
    if (params == 0) return EXIT_FAILURE;

    std::vector<float> weights(params);
    for (std::uint64_t i = 0; i < params; ++i) {
        weights[i] = static_cast<float>((static_cast<int>(i % 11) - 5) * 0.01);
    }

    mgt::TrainStateStorage states[2]{};
    for (std::uint32_t sample = 0; sample < 2; ++sample) {
        for (std::uint32_t i = 0; i < shape.state_len; ++i) {
            states[sample].v[i] = static_cast<mgt::StateValue>((i + sample) % shape.state_value_pad);
        }
    }
    const float labels[2] = {1.0f, 3.0f};

    std::vector<float> outputs(2, 0.0f);
    if (mgt::CpuMlpForward(shape, weights, states, 2, outputs.data()) != mgt::Status::kOk) {
        return EXIT_FAILURE;
    }
    if (!std::isfinite(outputs[0]) || !std::isfinite(outputs[1])) return EXIT_FAILURE;

    std::vector<float> grad(params, 0.0f);
    float loss = 0.0f;
    if (mgt::CpuMlpLossAndGrad(shape, weights, states, labels, 2, &loss, grad.data()) != mgt::Status::kOk) {
        return EXIT_FAILURE;
    }
    if (!std::isfinite(loss) || loss <= 0.0f) return EXIT_FAILURE;

    const float eps = 1.0e-3f;
    const std::uint64_t check_indices[] = {0, 7, 31, params - 2};
    for (std::uint64_t idx : check_indices) {
        std::vector<float> plus = weights;
        std::vector<float> minus = weights;
        plus[idx] += eps;
        minus[idx] -= eps;
        float loss_plus = 0.0f;
        float loss_minus = 0.0f;
        std::vector<float> scratch(params, 0.0f);
        if (mgt::CpuMlpLossAndGrad(shape, plus, states, labels, 2, &loss_plus, scratch.data()) != mgt::Status::kOk) return EXIT_FAILURE;
        if (mgt::CpuMlpLossAndGrad(shape, minus, states, labels, 2, &loss_minus, scratch.data()) != mgt::Status::kOk) return EXIT_FAILURE;
        const float numeric = (loss_plus - loss_minus) / (2.0f * eps);
        if (std::fabs(numeric - grad[idx]) > 5.0e-3f) return EXIT_FAILURE;
    }

    std::vector<float> adam_weights = weights;
    std::vector<float> m(params, 0.0f);
    std::vector<float> v(params, 0.0f);
    if (mgt::CpuAdamWStep(adam_weights.data(), grad.data(), m.data(), v.data(), params,
                          1, 0.001f, 0.9f, 0.999f, 1.0e-8f, 0.0f) != mgt::Status::kOk) {
        return EXIT_FAILURE;
    }
    if (adam_weights == weights) return EXIT_FAILURE;

    return EXIT_SUCCESS;
}