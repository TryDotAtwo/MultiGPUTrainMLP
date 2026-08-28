#include "mgt/single_gpu_trainer_ffi.h"

#include <cstring>

#ifndef MGT_TEST_SOURCE_DIR
#define MGT_TEST_SOURCE_DIR "."
#endif

int main() {
    if (mgt_single_gpu_v1_abi_version() != MGT_SINGLE_GPU_ABI_V1) return 1;
    MgtSingleGpuConfigV1 config{};
    config.struct_size = sizeof(config);
    config.abi_version = MGT_SINGLE_GPU_ABI_V1;
    config.device_id = 0;
    config.capacity_rows = 4;
    config.learning_rate = .001f;
    config.beta1 = .9f;
    config.beta2 = .999f;
    config.epsilon = 1e-8f;
    config.group_json_utf8 = MGT_TEST_SOURCE_DIR "/tests/fixtures/p888.json";
    config.target_bin_utf8 = MGT_TEST_SOURCE_DIR "/tests/fixtures/p888-target.bin";
    config.base_seed = 0x8881;
    config.k_min = 1;
    config.k_max = 29;
    MgtSingleGpuHandle* handle = nullptr;
    auto bad = config;
    bad.struct_size -= 1;
    if (mgt_single_gpu_v1_create(&bad, &handle) != MGT_STATUS_INVALID_CONFIG || handle)
        return 2;
    bad = config;
    bad.reserved_u32[1] = 1;
    if (mgt_single_gpu_v1_create(&bad, &handle) != MGT_STATUS_INVALID_CONFIG || handle)
        return 3;
    if (mgt_single_gpu_v1_create(&config, &handle) != MGT_STATUS_OK || !handle)
        return 4;
    MgtSingleGpuStepV1 step{};
    step.struct_size = sizeof(step);
    step.active_rows = 4;
    step.optimizer_step = 1;
    MgtSingleGpuMetricsV1 metrics{};
    metrics.struct_size = sizeof(metrics);
    if (mgt_single_gpu_v1_train_step(handle, &step, &metrics) != MGT_STATUS_INVALID_CONFIG)
        return 5;
    if (mgt_single_gpu_v1_prepare(handle) != MGT_STATUS_OK ||
        mgt_single_gpu_v1_train_step(handle, &step, &metrics) != MGT_STATUS_OK ||
        metrics.completed_sequence != 1 || metrics.optimizer_step != 1)
        return 6;
    char error[8]{};
    if (mgt_single_gpu_v1_checkpoint(handle, "") != MGT_STATUS_INVALID_CONFIG ||
        mgt_single_gpu_v1_last_error(handle, error, sizeof(error)) < sizeof(error) ||
        error[sizeof(error) - 1] != '\0') return 7;
    if (mgt_single_gpu_v1_destroy(&handle) != MGT_STATUS_OK || handle ||
        mgt_single_gpu_v1_destroy(&handle) != MGT_STATUS_INVALID_CONFIG)
        return 8;
    return 0;
}
