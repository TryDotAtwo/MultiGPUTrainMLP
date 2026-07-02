$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Build = Join-Path $Root "build-native"
cmake -S (Join-Path $Root "native") -B $Build -DMGT_ENABLE_CUDA=OFF -DMGT_ENABLE_NCCL=OFF
cmake --build $Build --config Release