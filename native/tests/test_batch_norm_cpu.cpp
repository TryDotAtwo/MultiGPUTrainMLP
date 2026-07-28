#include "mgt/batch_norm.hpp"

#include <cmath>
#include <cstdlib>
#include <vector>

namespace {
bool Near(float a, float b, float tol = 2.0e-5f) { return std::fabs(a - b) <= tol; }
}

int main() {
    constexpr int rows = 3;
    constexpr int cols = 2;
    const float x[rows * cols] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 8.0f};
    const float gamma[cols] = {1.5f, 0.5f};
    const float beta[cols] = {-0.25f, 0.75f};
    float running_mean[cols] = {0.0f, 0.0f};
    float running_var[cols] = {1.0f, 1.0f};
    float y[rows * cols]{};
    mgt::BatchNormCache cache;
    mgt::batch_norm_forward_cpu(x, rows, cols, gamma, beta, running_mean, running_var,
                                0.1f, 1.0e-5f, true, y, &cache);

    if (!Near(cache.mean[0], 3.0f) || !Near(cache.mean[1], 14.0f / 3.0f)) return EXIT_FAILURE;
    if (!Near(cache.variance[0], 8.0f / 3.0f) || !Near(cache.variance[1], 56.0f / 9.0f)) return EXIT_FAILURE;
    if (!Near(running_mean[0], 0.3f) || !Near(running_mean[1], 14.0f / 30.0f)) return EXIT_FAILURE;
    if (!Near(running_var[0], 1.3f) || !Near(running_var[1], 11.0f / 6.0f)) return EXIT_FAILURE;
    for (float value : y) if (!std::isfinite(value)) return EXIT_FAILURE;

    const float dy[rows * cols] = {0.5f, -1.0f, 2.0f, 0.25f, -0.75f, 1.5f};
    float dx[rows * cols]{};
    float dgamma[cols]{};
    float dbeta[cols]{};
    mgt::batch_norm_backward_cpu(dy, cache, gamma, dx, dgamma, dbeta);
    if (!Near(dbeta[0], 1.75f) || !Near(dbeta[1], 0.75f)) return EXIT_FAILURE;
    float sum_dx0 = 0.0f;
    float sum_dx1 = 0.0f;
    for (int r = 0; r < rows; ++r) {
        sum_dx0 += dx[r * cols];
        sum_dx1 += dx[r * cols + 1];
    }
    if (!Near(sum_dx0, 0.0f) || !Near(sum_dx1, 0.0f)) return EXIT_FAILURE;

    float eval_y[rows * cols]{};
    mgt::batch_norm_forward_cpu(x, rows, cols, gamma, beta, running_mean, running_var,
                                0.1f, 1.0e-5f, false, eval_y, nullptr);
    if (Near(eval_y[0], y[0], 1.0e-3f)) return EXIT_FAILURE;
    return EXIT_SUCCESS;
}