#pragma once

#include <cstdint>

namespace mgt {

struct SingleGpuBenchmarkStepPosition {
    std::uint64_t epoch = 0;
    std::uint64_t epoch_sample_offset = 0;
};

constexpr SingleGpuBenchmarkStepPosition SingleGpuBenchmarkPosition(
    std::uint64_t optimizer_step, std::uint32_t batch,
    std::uint32_t samples_per_epoch) {
    const std::uint64_t steps_per_epoch = samples_per_epoch / batch;
    const std::uint64_t zero_based_step = optimizer_step - 1;
    return {zero_based_step / steps_per_epoch,
            (zero_based_step % steps_per_epoch) * batch};
}

struct SingleGpuBenchmarkThroughputMetrics {
    double samples_per_second = 0.0;
    double useful_tflops = 0.0;
    double issued_tflops = 0.0;
    double useful_peak_utilization_percent = 0.0;
};

constexpr SingleGpuBenchmarkThroughputMetrics SingleGpuBenchmarkThroughput(
    std::uint32_t batch, double step_ms,
    std::uint64_t useful_flops_per_sample,
    std::uint64_t issued_flops_per_sample, double dense_fp16_peak_tflops) {
    const double samples_per_second = batch * 1000.0 / step_ms;
    const double useful_tflops =
        samples_per_second * useful_flops_per_sample / 1.0e12;
    const double issued_tflops =
        samples_per_second * issued_flops_per_sample / 1.0e12;
    return {samples_per_second, useful_tflops, issued_tflops,
            useful_tflops * 100.0 / dense_fp16_peak_tflops};
}

}  // namespace mgt
