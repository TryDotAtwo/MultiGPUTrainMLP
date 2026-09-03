#include "mgt/single_gpu_benchmark.hpp"

#include <cmath>
#include <cstdlib>
#include <iostream>

namespace {

void Require(bool value, const char* message) {
    if (!value) {
        std::cerr << "FAIL single GPU benchmark: " << message << '\n';
        std::exit(EXIT_FAILURE);
    }
}

}  // namespace

int main() {
    using mgt::SingleGpuBenchmarkPosition;
    constexpr std::uint32_t samples = 999978;
    constexpr std::uint32_t batch = 61440;
    constexpr std::uint64_t steps_per_epoch = samples / batch;
    static_assert(steps_per_epoch == 16);

    const auto first = SingleGpuBenchmarkPosition(1, batch, samples);
    Require(first.epoch == 0 && first.epoch_sample_offset == 0,
            "first step position");
    const auto last = SingleGpuBenchmarkPosition(16, batch, samples);
    Require(last.epoch == 0 && last.epoch_sample_offset == 15 * batch,
            "last full batch in first epoch");
    const auto wrapped = SingleGpuBenchmarkPosition(17, batch, samples);
    Require(wrapped.epoch == 1 && wrapped.epoch_sample_offset == 0,
            "epoch wrap");
    const auto late = SingleGpuBenchmarkPosition(240, batch, samples);
    Require(late.epoch == 14 && late.epoch_sample_offset == 15 * batch,
            "240-step schedule");

    constexpr double step_ms = 8.0;
    const auto metrics = mgt::SingleGpuBenchmarkThroughput(
        batch, step_ms, 12469164ULL, 13075776ULL, 15.9744);
    Require(std::abs(metrics.samples_per_second - 7680000.0) < 1e-6,
            "samples per second");
    Require(std::abs(metrics.useful_tflops - 95.76317952) < 1e-7,
            "useful TFLOPS");
    Require(std::abs(metrics.issued_tflops - 100.42195968) < 1e-7,
            "issued TFLOPS");
    Require(std::abs(metrics.useful_peak_utilization_percent -
                     599.479038461538) < 1e-6,
            "fixed-peak utilization");

    std::cout << "PASS single GPU benchmark epoch cycling and throughput\n";
    return EXIT_SUCCESS;
}
