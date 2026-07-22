#include "mgt/mlp_cpu_ref.hpp"
#include <algorithm>
#include <cmath>
#include <vector>

namespace mgt {
namespace {

struct Offsets {
    std::uint64_t input_table;
    std::uint64_t input_bias;
    std::uint64_t hidden_weight;
    std::uint64_t hidden_bias;
    std::uint64_t residual_base;
    std::uint64_t output_weight;
    std::uint64_t output_bias;
    std::uint64_t total;
};

std::uint64_t ResidualBlockParams(const CpuMlpShape& shape) {
    return 2ULL * (static_cast<std::uint64_t>(shape.hd2) * shape.hd2 + shape.hd2);
}

std::uint64_t ResidualFc1Weight(const CpuMlpShape& shape, const Offsets& offsets, std::uint32_t block) {
    return offsets.residual_base + static_cast<std::uint64_t>(block) * ResidualBlockParams(shape);
}

std::uint64_t ResidualFc1Bias(const CpuMlpShape& shape, const Offsets& offsets, std::uint32_t block) {
    return ResidualFc1Weight(shape, offsets, block) + static_cast<std::uint64_t>(shape.hd2) * shape.hd2;
}

std::uint64_t ResidualFc2Weight(const CpuMlpShape& shape, const Offsets& offsets, std::uint32_t block) {
    return ResidualFc1Bias(shape, offsets, block) + shape.hd2;
}

std::uint64_t ResidualFc2Bias(const CpuMlpShape& shape, const Offsets& offsets, std::uint32_t block) {
    return ResidualFc2Weight(shape, offsets, block) + static_cast<std::uint64_t>(shape.hd2) * shape.hd2;
}

Offsets BuildOffsets(const CpuMlpShape& shape) {
    Offsets o{};
    o.input_table = 0;
    o.input_bias = o.input_table + static_cast<std::uint64_t>(shape.state_len) * shape.state_value_pad * shape.hd1;
    o.hidden_weight = o.input_bias + shape.hd1;
    o.hidden_bias = o.hidden_weight + static_cast<std::uint64_t>(shape.hd1) * shape.hd2;
    o.residual_base = o.hidden_bias + shape.hd2;
    o.output_weight = o.residual_base + static_cast<std::uint64_t>(shape.residual_blocks) * ResidualBlockParams(shape);
    o.output_bias = o.output_weight + shape.hd2 * shape.output_dim;
    o.total = o.output_bias + shape.output_dim;
    return o;
}

bool ValidShape(const CpuMlpShape& shape) {
    return shape.state_len > 0 && shape.state_len <= kStateLen &&
           shape.state_value_pad > 0 && shape.state_value_pad <= kStateValuePad &&
           shape.hd1 > 0 && shape.hd2 > 0 && shape.residual_blocks <= 64 && shape.output_dim > 0;
}

float Relu(float x) { return x > 0.0f ? x : 0.0f; }
float ReluGrad(float x) { return x > 0.0f ? 1.0f : 0.0f; }

Status ForwardOne(const CpuMlpShape& shape,
                  const Offsets& offsets,
                  std::span<const float> weights,
                  const TrainStateStorage& state,
                  std::vector<float>* z1,
                  std::vector<float>* a1,
                  std::vector<float>* z2,
                  std::vector<float>* a2,
                  std::vector<float>* residual_z1,
                  std::vector<float>* residual_a1,
                  std::vector<float>* residual_z2,
                  std::vector<float>* residual_out,
                  float* output) {
    std::fill(z1->begin(), z1->end(), 0.0f);
    for (std::uint32_t h = 0; h < shape.hd1; ++h) (*z1)[h] = weights[offsets.input_bias + h];
    for (std::uint32_t pos = 0; pos < shape.state_len; ++pos) {
        const std::uint32_t value = state.v[pos];
        if (value >= shape.state_value_pad) return Status::kInvalidConfig;
        const std::uint64_t row = static_cast<std::uint64_t>(pos) * shape.state_value_pad + value;
        const std::uint64_t base = offsets.input_table + row * shape.hd1;
        for (std::uint32_t h = 0; h < shape.hd1; ++h) (*z1)[h] += weights[base + h];
    }
    for (std::uint32_t h = 0; h < shape.hd1; ++h) (*a1)[h] = Relu((*z1)[h]);

    for (std::uint32_t j = 0; j < shape.hd2; ++j) {
        float sum = weights[offsets.hidden_bias + j];
        for (std::uint32_t h = 0; h < shape.hd1; ++h) sum += (*a1)[h] * weights[offsets.hidden_weight + static_cast<std::uint64_t>(h) * shape.hd2 + j];
        (*z2)[j] = sum;
        (*a2)[j] = Relu(sum);
        (*residual_out)[j] = (*a2)[j];
    }

    for (std::uint32_t block = 0; block < shape.residual_blocks; ++block) {
        const std::uint64_t fc1w = ResidualFc1Weight(shape, offsets, block);
        const std::uint64_t fc1b = ResidualFc1Bias(shape, offsets, block);
        const std::uint64_t fc2w = ResidualFc2Weight(shape, offsets, block);
        const std::uint64_t fc2b = ResidualFc2Bias(shape, offsets, block);
        const std::uint64_t base = static_cast<std::uint64_t>(block) * shape.hd2;
        for (std::uint32_t j = 0; j < shape.hd2; ++j) {
            float sum = weights[fc1b + j];
            for (std::uint32_t i = 0; i < shape.hd2; ++i) sum += (*residual_out)[i] * weights[fc1w + static_cast<std::uint64_t>(i) * shape.hd2 + j];
            (*residual_z1)[base + j] = sum;
            (*residual_a1)[base + j] = Relu(sum);
        }
        for (std::uint32_t j = 0; j < shape.hd2; ++j) {
            float sum = weights[fc2b + j];
            for (std::uint32_t i = 0; i < shape.hd2; ++i) sum += (*residual_a1)[base + i] * weights[fc2w + static_cast<std::uint64_t>(i) * shape.hd2 + j];
            (*residual_z2)[base + j] = sum;
            (*residual_out)[j] = Relu((*residual_out)[j] + sum);
        }
    }

    for (std::uint32_t out = 0; out < shape.output_dim; ++out) {
        float y = weights[offsets.output_bias + out];
        for (std::uint32_t j = 0; j < shape.hd2; ++j) y += (*residual_out)[j] * weights[offsets.output_weight + static_cast<std::uint64_t>(j) * shape.output_dim + out];
        output[out] = y;
    }
    return Status::kOk;
}

}  // namespace

std::uint64_t CpuMlpParamCount(const CpuMlpShape& shape) {
    if (!ValidShape(shape)) return 0;
    return BuildOffsets(shape).total;
}

Status CpuMlpForward(const CpuMlpShape& shape,
                     std::span<const float> weights,
                     const TrainStateStorage* states,
                     std::uint32_t sample_count,
                     float* outputs) {
    if (!ValidShape(shape) || states == nullptr || outputs == nullptr || sample_count == 0) return Status::kInvalidConfig;
    const Offsets offsets = BuildOffsets(shape);
    if (weights.size() != offsets.total) return Status::kInvalidConfig;
    std::vector<float> z1(shape.hd1), a1(shape.hd1), z2(shape.hd2), a2(shape.hd2);
    std::vector<float> rz1(static_cast<std::uint64_t>(shape.residual_blocks) * shape.hd2);
    std::vector<float> ra1(static_cast<std::uint64_t>(shape.residual_blocks) * shape.hd2);
    std::vector<float> rz2(static_cast<std::uint64_t>(shape.residual_blocks) * shape.hd2);
    std::vector<float> rout(shape.hd2);
    for (std::uint32_t sample = 0; sample < sample_count; ++sample) {
        const Status status = ForwardOne(shape, offsets, weights, states[sample], &z1, &a1, &z2, &a2, &rz1, &ra1, &rz2, &rout, outputs + static_cast<std::uint64_t>(sample) * shape.output_dim);
        if (status != Status::kOk) return status;
    }
    return Status::kOk;
}

Status CpuMlpLossAndGrad(const CpuMlpShape& shape,
                         std::span<const float> weights,
                         const TrainStateStorage* states,
                         const float* labels,
                         std::uint32_t sample_count,
                         float* loss,
                         float* grad) {
    if (!ValidShape(shape) || states == nullptr || labels == nullptr || loss == nullptr || grad == nullptr || sample_count == 0) return Status::kInvalidConfig;
    const Offsets offsets = BuildOffsets(shape);
    if (weights.size() != offsets.total) return Status::kInvalidConfig;
    std::fill(grad, grad + offsets.total, 0.0f);
    *loss = 0.0f;

    std::vector<float> z1(shape.hd1), a1(shape.hd1), z2(shape.hd2), a2(shape.hd2);
    std::vector<float> rz1(static_cast<std::uint64_t>(shape.residual_blocks) * shape.hd2);
    std::vector<float> ra1(static_cast<std::uint64_t>(shape.residual_blocks) * shape.hd2);
    std::vector<float> rz2(static_cast<std::uint64_t>(shape.residual_blocks) * shape.hd2);
    std::vector<float> block_inputs(static_cast<std::uint64_t>(shape.residual_blocks + 1U) * shape.hd2);
    std::vector<float> rout(shape.hd2), output(shape.output_dim), dcur(shape.hd2), dprev(shape.hd2), dfc1(shape.hd2), dzfc2(shape.hd2), dzfc1(shape.hd2);
    std::vector<float> dz1(shape.hd1), da1(shape.hd1), dz2(shape.hd2), da2(shape.hd2);
    const float inv_n = 1.0f / static_cast<float>(static_cast<std::uint64_t>(sample_count) * shape.output_dim);

    for (std::uint32_t sample = 0; sample < sample_count; ++sample) {
        std::fill(output.begin(), output.end(), 0.0f);
        const Status status = ForwardOne(shape, offsets, weights, states[sample], &z1, &a1, &z2, &a2, &rz1, &ra1, &rz2, &rout, output.data());
        if (status != Status::kOk) return status;
        std::copy(a2.begin(), a2.end(), block_inputs.begin());
        std::vector<float> replay = a2;
        for (std::uint32_t block = 0; block < shape.residual_blocks; ++block) {
            const std::uint64_t base = static_cast<std::uint64_t>(block) * shape.hd2;
            const std::uint64_t next = static_cast<std::uint64_t>(block + 1U) * shape.hd2;
            for (std::uint32_t j = 0; j < shape.hd2; ++j) replay[j] = Relu(replay[j] + rz2[base + j]);
            std::copy(replay.begin(), replay.end(), block_inputs.begin() + next);
        }

        std::fill(dcur.begin(), dcur.end(), 0.0f);
        for (std::uint32_t out = 0; out < shape.output_dim; ++out) {
            const float diff = output[out] - labels[static_cast<std::uint64_t>(sample) * shape.output_dim + out];
            *loss += diff * diff * inv_n;
            const float dy = 2.0f * diff * inv_n;
            grad[offsets.output_bias + out] += dy;
            for (std::uint32_t j = 0; j < shape.hd2; ++j) {
                const std::uint64_t weight_idx = offsets.output_weight + static_cast<std::uint64_t>(j) * shape.output_dim + out;
                grad[weight_idx] += rout[j] * dy;
                dcur[j] += weights[weight_idx] * dy;
            }
        }

        for (std::uint32_t rblock = shape.residual_blocks; rblock > 0; --rblock) {
            const std::uint32_t block = rblock - 1U;
            const std::uint64_t base = static_cast<std::uint64_t>(block) * shape.hd2;
            const float* input = block_inputs.data() + base;
            const std::uint64_t fc1w = ResidualFc1Weight(shape, offsets, block);
            const std::uint64_t fc1b = ResidualFc1Bias(shape, offsets, block);
            const std::uint64_t fc2w = ResidualFc2Weight(shape, offsets, block);
            const std::uint64_t fc2b = ResidualFc2Bias(shape, offsets, block);
            std::fill(dprev.begin(), dprev.end(), 0.0f);
            std::fill(dfc1.begin(), dfc1.end(), 0.0f);
            for (std::uint32_t j = 0; j < shape.hd2; ++j) dzfc2[j] = dcur[j] * ReluGrad(input[j] + rz2[base + j]);
            for (std::uint32_t j = 0; j < shape.hd2; ++j) grad[fc2b + j] += dzfc2[j];
            for (std::uint32_t i = 0; i < shape.hd2; ++i) {
                for (std::uint32_t j = 0; j < shape.hd2; ++j) {
                    grad[fc2w + static_cast<std::uint64_t>(i) * shape.hd2 + j] += ra1[base + i] * dzfc2[j];
                    dfc1[i] += weights[fc2w + static_cast<std::uint64_t>(i) * shape.hd2 + j] * dzfc2[j];
                }
            }
            for (std::uint32_t j = 0; j < shape.hd2; ++j) dzfc1[j] = dfc1[j] * ReluGrad(rz1[base + j]);
            for (std::uint32_t j = 0; j < shape.hd2; ++j) grad[fc1b + j] += dzfc1[j];
            for (std::uint32_t i = 0; i < shape.hd2; ++i) {
                dprev[i] += dzfc2[i];
                for (std::uint32_t j = 0; j < shape.hd2; ++j) {
                    grad[fc1w + static_cast<std::uint64_t>(i) * shape.hd2 + j] += input[i] * dzfc1[j];
                    dprev[i] += weights[fc1w + static_cast<std::uint64_t>(i) * shape.hd2 + j] * dzfc1[j];
                }
            }
            dcur.swap(dprev);
        }

        for (std::uint32_t j = 0; j < shape.hd2; ++j) {
            da2[j] = dcur[j];
            dz2[j] = da2[j] * ReluGrad(z2[j]);
            grad[offsets.hidden_bias + j] += dz2[j];
        }
        std::fill(da1.begin(), da1.end(), 0.0f);
        for (std::uint32_t h = 0; h < shape.hd1; ++h) {
            for (std::uint32_t j = 0; j < shape.hd2; ++j) {
                grad[offsets.hidden_weight + static_cast<std::uint64_t>(h) * shape.hd2 + j] += a1[h] * dz2[j];
                da1[h] += weights[offsets.hidden_weight + static_cast<std::uint64_t>(h) * shape.hd2 + j] * dz2[j];
            }
        }
        for (std::uint32_t h = 0; h < shape.hd1; ++h) {
            dz1[h] = da1[h] * ReluGrad(z1[h]);
            grad[offsets.input_bias + h] += dz1[h];
        }
        for (std::uint32_t pos = 0; pos < shape.state_len; ++pos) {
            const std::uint32_t value = states[sample].v[pos];
            const std::uint64_t row = static_cast<std::uint64_t>(pos) * shape.state_value_pad + value;
            const std::uint64_t base = offsets.input_table + row * shape.hd1;
            for (std::uint32_t h = 0; h < shape.hd1; ++h) grad[base + h] += dz1[h];
        }
    }
    return Status::kOk;
}

Status CpuAdamWStep(float* weights,
                    const float* grad,
                    float* m,
                    float* v,
                    std::uint64_t param_count,
                    std::uint64_t step,
                    float lr,
                    float beta1,
                    float beta2,
                    float eps,
                    float weight_decay) {
    if (weights == nullptr || grad == nullptr || m == nullptr || v == nullptr || param_count == 0 || step == 0 || lr <= 0.0f || eps <= 0.0f || beta1 < 0.0f || beta1 >= 1.0f || beta2 < 0.0f || beta2 >= 1.0f || weight_decay < 0.0f) return Status::kInvalidConfig;
    const float bias1 = 1.0f - std::pow(beta1, static_cast<float>(step));
    const float bias2 = 1.0f - std::pow(beta2, static_cast<float>(step));
    for (std::uint64_t i = 0; i < param_count; ++i) {
        m[i] = beta1 * m[i] + (1.0f - beta1) * grad[i];
        v[i] = beta2 * v[i] + (1.0f - beta2) * grad[i] * grad[i];
        const float m_hat = m[i] / bias1;
        const float v_hat = v[i] / bias2;
        weights[i] -= lr * (m_hat / (std::sqrt(v_hat) + eps) + weight_decay * weights[i]);
    }
    return Status::kOk;
}

}  // namespace mgt