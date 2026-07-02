$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
cargo test --workspace
$Build = Join-Path $Root "build-native"
ctest --test-dir $Build --output-on-failure -C Release