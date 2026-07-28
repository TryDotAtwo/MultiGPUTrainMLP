#include "mgt/mlp_cpu_ref.hpp"
#include <cmath>
#include <cstdlib>
#include <vector>

int main() {
    const mgt::CpuMlpShape shape{2, 4, 3, 2, 1, 1};
    std::vector<float> weights(mgt::CpuMlpParamCount(shape));
    for (std::size_t i = 0; i < weights.size(); ++i) weights[i] = 0.03f * static_cast<float>(static_cast<int>(i % 9) - 4);
    mgt::TrainStateStorage states[4]{};
    for (std::uint32_t sample = 0; sample < 4; ++sample) {
        states[sample].v[0] = static_cast<mgt::StateValue>(sample);
        states[sample].v[1] = static_cast<mgt::StateValue>((sample + 1) % 4);
    }
    auto batch_norm = mgt::InitializeCpuMlpBatchNormState(shape);
    const auto features = mgt::CpuMlpBatchNormFeatureCount(shape);
    if (features != shape.hd1 + 3U * shape.hd2) return EXIT_FAILURE;
    if (batch_norm.affine.size() != 2U * features || batch_norm.running.size() != 2U * features) return EXIT_FAILURE;
    std::vector<float> train_outputs(4);
    if (mgt::CpuMlpBatchNormForward(shape, weights, &batch_norm, states, 4, 0.1f, 1.0e-5f, true, train_outputs.data()) != mgt::Status::kOk) return EXIT_FAILURE;
    for (float value : train_outputs) if (!std::isfinite(value)) return EXIT_FAILURE;
    const float expected_train[4] = {-0.0066793207f, -0.0014381879f, -0.1513643116f, -0.0014381879f};
    for (std::size_t i = 0; i < train_outputs.size(); ++i) if (std::fabs(train_outputs[i] - expected_train[i]) > 2.0e-5f) return EXIT_FAILURE;
    bool mean_changed = false, var_changed = false;
    for (std::uint64_t i = 0; i < features; ++i) {
        mean_changed |= std::fabs(batch_norm.running[i]) > 1.0e-7f;
        var_changed |= std::fabs(batch_norm.running[features + i] - 1.0f) > 1.0e-7f;
    }
    if (!mean_changed || !var_changed) return EXIT_FAILURE;
    const float expected_running[18] = {
        -0.0022499997f, 0.0067500002f, 0.0157499984f, -0.0021335066f, 0.0049332469f,
        -0.0017226263f, -0.0076337680f, -0.0063073630f, -0.0004969927f,
        0.9007424712f, 0.9007424712f, 0.9007424712f, 0.9016661048f, 0.9004164934f,
        0.9000119567f, 0.9004281163f, 0.9001960754f, 0.9002638459f};
    for (std::size_t i = 0; i < batch_norm.running.size(); ++i) if (std::fabs(batch_norm.running[i] - expected_running[i]) > 2.0e-5f) return EXIT_FAILURE;
    const auto running_after_train = batch_norm.running;
    std::vector<float> eval_outputs(4);
    if (mgt::CpuMlpBatchNormForward(shape, weights, &batch_norm, states, 4, 0.1f, 1.0e-5f, false, eval_outputs.data()) != mgt::Status::kOk) return EXIT_FAILURE;
    if (batch_norm.running != running_after_train || eval_outputs == train_outputs) return EXIT_FAILURE;
    return EXIT_SUCCESS;
}