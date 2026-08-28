param(
  [string]$Image = "mgt-single-gpu-dev:2026-08-28",
  [string]$Name = "mgt-gpu-queue",
  [string]$Root = "D:/MultiGPUTrainMLP",
  [int]$CooldownSec = 10
)

$ErrorActionPreference = "Stop"

$running = docker ps --filter "name=^/$Name$" --format "{{.Names}}"
if ($LASTEXITCODE -ne 0) { throw "docker ps failed with exit code $LASTEXITCODE" }
if ($running -eq $Name) {
  Write-Host "gpu_queue_container=running name=$Name"
  exit 0
}

$existing = docker ps -a --filter "name=^/$Name$" --format "{{.Names}}"
if ($LASTEXITCODE -ne 0) { throw "docker ps -a failed with exit code $LASTEXITCODE" }
if ($existing -eq $Name) {
  docker start $Name
  if ($LASTEXITCODE -ne 0) { throw "docker start failed with exit code $LASTEXITCODE" }
  Write-Host "gpu_queue_container=started name=$Name"
  exit 0
}

docker run -d --name $Name --gpus all `
  -v "$Root`:/work" `
  -w /work `
  -e MGT_GPU_QUEUE_ROOT=/work/.gpu_queue `
  -e MGT_GPU_QUEUE_COOLDOWN_SEC=$CooldownSec `
  $Image python3 scripts/gpu_queue_worker.py
if ($LASTEXITCODE -ne 0) { throw "docker run failed with exit code $LASTEXITCODE" }
Write-Host "gpu_queue_container=created name=$Name image=$Image cooldown_sec=$CooldownSec"
