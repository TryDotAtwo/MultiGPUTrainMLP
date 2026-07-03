$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Image = if ($env:MGT_DOCKER_IMAGE) { $env:MGT_DOCKER_IMAGE } else { "cmz-native-dev:2026-05-26" }
$Arch = if ($env:MGT_CUDA_ARCH) { $env:MGT_CUDA_ARCH } else { "86" }
$Steps = if ($env:MGT_PROFILE_STEPS) { $env:MGT_PROFILE_STEPS } else { "4" }
$Output = if ($env:MGT_PROFILE_OUTPUT) { $env:MGT_PROFILE_OUTPUT } else { "test_results/native_profile" }
$Ncu = if ($env:MGT_PROFILE_NCU) { $env:MGT_PROFILE_NCU } else { "0" }
$Command = @"
set -euo pipefail
mkdir -p '$Output'
cmake -S native -B build-gpu-smoke -DMGT_ENABLE_CUDA=ON -DMGT_ENABLE_NCCL=OFF -DCMAKE_CUDA_ARCHITECTURES=$Arch
cmake --build build-gpu-smoke --config Release --target mgt_native_train
rm -rf '$Output/run'
nsys profile --trace=cuda,nvtx,osrt --force-overwrite=true --stats=true -o '$Output/native_train' ./build-gpu-smoke/mgt_native_train --output-dir '$Output/run' --steps '$Steps' --device-id 0 --world-size 1 --global-rank 0 --local-rank 0 --batch-size 4096 --k-min 1 --k-max 9 --hd1 32 --hd2 16
if [ '$Ncu' = '1' ]; then
  ncu --target-processes all --set roofline --export '$Output/native_train_ncu' --force-overwrite ./build-gpu-smoke/mgt_native_train --output-dir '$Output/run_ncu' --steps 1 --device-id 0 --world-size 1 --global-rank 0 --local-rank 0 --batch-size 1024 --k-min 1 --k-max 5 --hd1 16 --hd2 8
fi
test -s '$Output/run/profile.jsonl'
test -s '$Output/run/train.log'
ls -la '$Output'
"@
docker run --rm --gpus all -v "${Root}:/workspace" -w /workspace $Image bash -lc $Command
