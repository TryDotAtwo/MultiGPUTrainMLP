#include "mgt_cuda/device_context.cuh"
#include "mgt_cuda/random_walk_kernel.cuh"
#include "mgt_cuda/adamw.cuh"

int main() {
    const mgt_cuda::DeviceLaunchConfig cfg = mgt_cuda::Build1DLaunchConfig(1024, 256);
    if (cfg.blocks == 0 || cfg.threads == 0) return 1;
    return 0;
}