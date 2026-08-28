#ifndef MGT_SINGLE_GPU_TRAINER_FFI_H
#define MGT_SINGLE_GPU_TRAINER_FFI_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define MGT_SINGLE_GPU_ABI_V1 1u

typedef enum MgtStatus {
    MGT_STATUS_OK = 0,
    MGT_STATUS_INVALID_CONFIG = 1,
    MGT_STATUS_INVALID_PUZZLE = 2,
    MGT_STATUS_CAPACITY_EXCEEDED = 3,
    MGT_STATUS_CUDA_FAILURE = 4,
    MGT_STATUS_NCCL_FAILURE = 5,
    MGT_STATUS_IO_FAILURE = 6
} MgtStatus;

typedef struct MgtSingleGpuHandle MgtSingleGpuHandle;

typedef struct MgtSingleGpuConfigV1 {
    uint32_t struct_size;
    uint32_t abi_version;
    uint32_t device_id;
    uint32_t capacity_rows;
    float learning_rate;
    float beta1;
    float beta2;
    float epsilon;
    float weight_decay;
    uint32_t reserved_u32[3];
    const char* group_json_utf8;
    const char* target_bin_utf8;
    uint64_t base_seed;
    uint32_t k_min;
    uint32_t k_max;
} MgtSingleGpuConfigV1;

typedef struct MgtSingleGpuStepV1 {
    uint32_t struct_size;
    uint32_t active_rows;
    uint64_t optimizer_step;
    uint64_t semantic_epoch;
    uint64_t epoch_sample_offset;
} MgtSingleGpuStepV1;

typedef struct MgtSingleGpuMetricsV1 {
    uint32_t struct_size;
    uint32_t reserved_u32;
    uint64_t completed_sequence;
    uint64_t optimizer_step;
    float loss;
    float reserved_f32;
} MgtSingleGpuMetricsV1;

uint32_t mgt_single_gpu_v1_abi_version(void);
MgtStatus mgt_single_gpu_v1_create(
    const MgtSingleGpuConfigV1* config, MgtSingleGpuHandle** out);
MgtStatus mgt_single_gpu_v1_prepare(MgtSingleGpuHandle* handle);
MgtStatus mgt_single_gpu_v1_train_step(
    MgtSingleGpuHandle* handle, const MgtSingleGpuStepV1* step,
    MgtSingleGpuMetricsV1* metrics);
MgtStatus mgt_single_gpu_v1_checkpoint(
    MgtSingleGpuHandle* handle, const char* directory_utf8);
MgtStatus mgt_single_gpu_v1_destroy(MgtSingleGpuHandle** handle);
size_t mgt_single_gpu_v1_last_error(
    MgtSingleGpuHandle* handle, char* destination, size_t capacity);

#ifdef __cplusplus
}
#endif
#endif
