$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Image = if ($env:MGT_DOCKER_IMAGE) { $env:MGT_DOCKER_IMAGE } else { "cmz-native-dev:2026-05-26" }

if (Get-Command cargo -ErrorAction SilentlyContinue) {
  cargo test --workspace
} else {
  docker run --rm -v "${Root}:/workspace" -w /workspace $Image bash -lc "cargo test --workspace"
}

$Build = Join-Path $Root "build-native"
ctest --test-dir $Build --output-on-failure -C Release
