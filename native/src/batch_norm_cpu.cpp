#include "mgt/batch_norm.hpp"

#include <algorithm>
#include <cmath>

namespace mgt {

void batch_norm_forward_cpu(
    const float* x, int rows, int cols,
    const float* gamma, const float* beta,
    float* running_mean, float* running_var,
    float momentum, float epsilon, bool training,
    float* y, BatchNormCache* cache) {
    if (training) {
        cache->rows = rows;
        cache->cols = cols;
        cache->mean.assign(cols, 0.0f);
        cache->variance.assign(cols, 0.0f);
        cache->inv_std.assign(cols, 0.0f);
        cache->normalized.assign(static_cast<std::size_t>(rows) * cols, 0.0f);
        for (int r = 0; r < rows; ++r) {
            for (int c = 0; c < cols; ++c) cache->mean[c] += x[r * cols + c];
        }
        const float inv_rows = 1.0f / static_cast<float>(rows);
        for (int c = 0; c < cols; ++c) cache->mean[c] *= inv_rows;
        for (int r = 0; r < rows; ++r) {
            for (int c = 0; c < cols; ++c) {
                const float centered = x[r * cols + c] - cache->mean[c];
                cache->variance[c] += centered * centered;
            }
        }
        for (int c = 0; c < cols; ++c) {
            cache->variance[c] *= inv_rows;
            cache->inv_std[c] = 1.0f / std::sqrt(cache->variance[c] + epsilon);
            running_mean[c] = (1.0f - momentum) * running_mean[c] + momentum * cache->mean[c];
            const float unbiased = rows > 1
                ? cache->variance[c] * static_cast<float>(rows) / static_cast<float>(rows - 1)
                : 0.0f;
            running_var[c] = (1.0f - momentum) * running_var[c] + momentum * unbiased;
        }
        for (int r = 0; r < rows; ++r) {
            for (int c = 0; c < cols; ++c) {
                const int i = r * cols + c;
                cache->normalized[i] = (x[i] - cache->mean[c]) * cache->inv_std[c];
                y[i] = cache->normalized[i] * gamma[c] + beta[c];
            }
        }
        return;
    }

    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            const int i = r * cols + c;
            const float normalized = (x[i] - running_mean[c]) / std::sqrt(running_var[c] + epsilon);
            y[i] = normalized * gamma[c] + beta[c];
        }
    }
}

void batch_norm_backward_cpu(
    const float* dy, const BatchNormCache& cache,
    const float* gamma, float* dx,
    float* dgamma, float* dbeta) {
    std::fill(dgamma, dgamma + cache.cols, 0.0f);
    std::fill(dbeta, dbeta + cache.cols, 0.0f);
    for (int r = 0; r < cache.rows; ++r) {
        for (int c = 0; c < cache.cols; ++c) {
            const int i = r * cache.cols + c;
            dbeta[c] += dy[i];
            dgamma[c] += dy[i] * cache.normalized[i];
        }
    }
    const float inv_rows = 1.0f / static_cast<float>(cache.rows);
    for (int r = 0; r < cache.rows; ++r) {
        for (int c = 0; c < cache.cols; ++c) {
            const int i = r * cache.cols + c;
            dx[i] = gamma[c] * cache.inv_std[c] * inv_rows *
                    (static_cast<float>(cache.rows) * dy[i] - dbeta[c] -
                     cache.normalized[i] * dgamma[c]);
        }
    }
}

}  // namespace mgt