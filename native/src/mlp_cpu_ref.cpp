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
    std::uint64_t output_weight;
    std::uint64_t output_bias;
    std::uint64_t total;
};

Offsets BuildOffsets(const CpuMlpShape& shape) {
    Offsets o{};
    o.input_table = 0;
    o.input_bias = o.input_table + static_cast<std::uint64_t>(shape.state_len) * shape.state_value_pad * shape.hd1;
    o.hidden_weight = o.input_bias + shape.hd1;
    o.hidden_bias = o.hidden_weight + static_cast<std::uint64_t>(shape.hd1) * shape.hd2;
    o.output_weight = o.hidden_bias + shape.hd2;
    o.output_bias = o.output_weight + shape.hd2 * shape.output_dim;
    o.total = o.output_bias + shape.output_dim;
    return o;
}

bool ValidShape(const CpuMlpShape& shape) {
    return shape.state_len > 0 && shape.state_len <= kStateLen &&
           shape.state_value_pad > 0 && shape.state_value_pad <= kStateValuePad &&
           shape.hd1 > 0 && shape.hd2 > 0 && shape.output_dim == 1;
}

float Relu(float x) {
    return x > 0.0f ? x : 0.0f;
}

float ReluGrad(float x) {
    return x > 0.0f ? 1.0f : 0.0f;
}

Status ForwardOne(const CpuMlpShape& shape,
                  const Offsets& offsets,
                  std::span<const float> weights,
                  const TrainState80& state,
                  std::vector<float>* z1,
                  std::vector<float>* a1,
                  std::vector<float>* z2,
                  std::vector<float>* a2,
                  float* output) {
    std::fill(z1->begin(), z1->end(), 0.0f);
    for (std::uint32_t h = 0; h < shape.hd1; ++h) {
        (*z1)[h] = weights[offsets.input_bias + h];
    }
    for (std::uint32_t pos = 0; pos < shape.state_len; ++pos) {
        const std::uint32_t value = state.v[pos];
        if (value >= shape.state_value_pad) return Status::kInvalidConfig;
        const std::uint64_t row = static_cast<std::uint64_t>(pos) * shape.state_value_pad + value;
        const std::uint64_t base = offsets.input_table + row * shape.hd1;
        for (std::uint32_t h = 0; h < shape.hd1; ++h) {
            (*z1)[h] += weights[base + h];
        }
    }
    for (std::uint32_t h = 0; h < shape.hd1; ++h) {
        (*a1)[h] = Relu((*z1)[h]);
    }

    for (std::uint32_t j = 0; j < shape.hd2; ++j) {
        float sum = weights[offsets.hidden_bias + j];
        for (std::uint32_t h = 0; h < shape.hd1; ++h) {
            sum += (*a1)[h] * weights[offsets.hidden_weight + static_cast<std::uint64_t>(h) * shape.hd2 + j];
        }
        (*z2)[j] = sum;
        (*a2)[j] = Relu(sum);
    }

    float y = weights[offsets.output_bias];
    for (std::uint32_t j = 0; j < shape.hd2; ++j) {
        y += (*a2)[j] * weights[offsets.output_weight + j];
    }
    *output = y;
    return Status::kOk;
}

}  // namespace

std::uint64_t CpuMlpParamCount(const CpuMlpShape& shape) {
    if (!ValidShape(shape)) return 0;
    return BuildOffsets(shape).total;
}

Status CpuMlpForward(const CpuMlpShape& shape,
                     std::span<const float> weights,
                     const TrainState80* states,
                     std::uint32_t sample_count,
                     float* outputs) {
    if (!ValidShape(shape) || states == nullptr || outputs == nullptr || sample_count == 0) {
        return Status::kInvalidConfig;
    }
    const Offsets offsets = BuildOffsets(shape);
    if (weights.size() != offsets.total) return Status::kInvalidConfig;

    std::vector<float> z1(shape.hd1), a1(shape.hd1), z2(shape.hd2), a2(shape.hd2);
    for (std::uint32_t sample = 0; sample < sample_count; ++sample) {
        const Status status = ForwardOne(shape, offsets, weights, states[sample], &z1, &a1, &z2, &a2, &outputs[sample]);
        if (status != Status::kOk) return status;
    }
    return Status::kOk;
}

Status CpuMlpLossAndGrad(const CpuMlpShape& shape,
                         std::span<const float> weights,
                         const TrainState80* states,
                         const float* labels,
                         std::uint32_t sample_count,
                         float* loss,
                         float* grad) {
    if (!ValidShape(shape) || states == nullptr || labels == nullptr || loss == nullptr ||
        grad == nullptr || sample_count == 0) {
        return Status::kInvalidConfig;
    }
    const Offsets offsets = BuildOffsets(shape);
    if (weights.size() != offsets.total) return Status::kInvalidConfig;
    std::fill(grad, grad + offsets.total, 0.0f);
    *loss = 0.0f;

    std::vector<float> z1(shape.hd1), a1(shape.hd1), z2(shape.hd2), a2(shape.hd2);
    std::vector<float> dz1(shape.hd1), da1(shape.hd1), dz2(shape.hd2), da2(shape.hd2);
    for (std::uint32_t sample = 0; sample < sample_count; ++sample) {
        float output = 0.0f;
        const Status status = ForwardOne(shape, offsets, weights, states[sample], &z1, &a1, &z2, &a2, &output);
        if (status != Status::kOk) return status;
        const float diff = output - labels[sample];
        *loss += diff * diff / static_cast<float>(sample_count);
        const float dy = 2.0f * diff / static_cast<float>(sample_count);

        grad[offsets.output_bias] += dy;
        for (std::uint32_t j = 0; j < shape.hd2; ++j) {
            grad[offsets.output_weight + j] += a2[j] * dy;
            da2[j] = weights[offsets.output_weight + j] * dy;
            dz2[j] = da2[j] * ReluGrad(z2[j]);
        }

        std::fill(da1.begin(), da1.end(), 0.0f);
        for (std::uint32_t h = 0; h < shape.hd1; ++h) {
            for (std::uint32_t j = 0; j < shape.hd2; ++j) {
                grad[offsets.hidden_weight + static_cast<std::uint64_t>(h) * shape.hd2 + j] += a1[h] * dz2[j];
                da1[h] += weights[offsets.hidden_weight + static_cast<std::uint64_t>(h) * shape.hd2 + j] * dz2[j];
            }
        }
        for (std::uint32_t j = 0; j < shape.hd2; ++j) {
            grad[offsets.hidden_bias + j] += dz2[j];
        }
        for (std::uint32_t h = 0; h < shape.hd1; ++h) {
            dz1[h] = da1[h] * ReluGrad(z1[h]);
            grad[offsets.input_bias + h] += dz1[h];
        }
        for (std::uint32_t pos = 0; pos < shape.state_len; ++pos) {
            const std::uint32_t value = states[sample].v[pos];
            const std::uint64_t row = static_cast<std::uint64_t>(pos) * shape.state_value_pad + value;
            const std::uint64_t base = offsets.input_table + row * shape.hd1;
            for (std::uint32_t h = 0; h < shape.hd1; ++h) {
                grad[base + h] += dz1[h];
            }
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
    if (weights == nullptr || grad == nullptr || m == nullptr || v == nullptr ||
        param_count == 0 || step == 0 || lr <= 0.0f || eps <= 0.0f ||
        beta1 < 0.0f || beta1 >= 1.0f || beta2 < 0.0f || beta2 >= 1.0f || weight_decay < 0.0f) {
        return Status::kInvalidConfig;
    }
    const float bias1 = 1.0f - std::pow(beta1, static_cast<float>(step));
    const float bias2 = 1.0f - std::pow(beta2, static_cast<float>(step));
    for (std::uint64_t i = 0; i < param_count; ++i) {
        const float decayed_grad = grad[i] + weight_decay * weights[i];
        m[i] = beta1 * m[i] + (1.0f - beta1) * decayed_grad;
        v[i] = beta2 * v[i] + (1.0f - beta2) * decayed_grad * decayed_grad;
        const float m_hat = m[i] / bias1;
        const float v_hat = v[i] / bias2;
        weights[i] -= lr * m_hat / (std::sqrt(v_hat) + eps);
    }
    return Status::kOk;
}

}  // namespace mgt