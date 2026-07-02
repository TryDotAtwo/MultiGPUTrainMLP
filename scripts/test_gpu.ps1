$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Image = if ($env:MGT_DOCKER_IMAGE) { $env:MGT_DOCKER_IMAGE } else { "cmz-native-dev:2026-05-26" }
$Arch = if ($env:MGT_CUDA_ARCH) { $env:MGT_CUDA_ARCH } else { "86" }
$Command = "cmake -S native -B build-gpu-smoke -DMGT_ENABLE_CUDA=ON -DMGT_ENABLE_NCCL=OFF -DCMAKE_CUDA_ARCHITECTURES=$Arch && cmake --build build-gpu-smoke --config Release && ctest --test-dir build-gpu-smoke -R 'cuda_compile|cuda_random_walk_smoke|cuda_adamw_smoke|cuda_mlp_forward_smoke' --output-on-failure -C Release"
docker run --rm --gpus all -v "${Root}:/workspace" -w /workspace $Image bash -lc $Command