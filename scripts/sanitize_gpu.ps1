$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Image = if ($env:MGT_DOCKER_IMAGE) { $env:MGT_DOCKER_IMAGE } else { "cmz-native-dev:2026-05-26" }
$Arch = if ($env:MGT_CUDA_ARCH) { $env:MGT_CUDA_ARCH } else { "86" }
$Build = "cmake -S native -B build-gpu-smoke -DMGT_ENABLE_CUDA=ON -DMGT_ENABLE_NCCL=OFF -DCMAKE_CUDA_ARCHITECTURES=$Arch && cmake --build build-gpu-smoke --config Release"
$Sanitize = "compute-sanitizer --tool memcheck --leak-check full ./build-gpu-smoke/test_cuda_random_walk_smoke && compute-sanitizer --tool memcheck --leak-check full ./build-gpu-smoke/test_cuda_adamw_smoke && compute-sanitizer --tool memcheck --leak-check full ./build-gpu-smoke/test_cuda_mlp_forward_smoke && compute-sanitizer --tool memcheck --leak-check full ./build-gpu-smoke/test_cuda_mlp_backward_smoke && compute-sanitizer --tool memcheck --leak-check full ./build-gpu-smoke/test_cuda_train_step_smoke"
docker run --rm --gpus all -v "${Root}:/workspace" -w /workspace $Image bash -lc "$Build && $Sanitize"