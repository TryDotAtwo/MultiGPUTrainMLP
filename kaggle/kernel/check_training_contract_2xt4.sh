#!/usr/bin/env bash
set -euo pipefail

: "${MGT_GROUP_JSON:?set MGT_GROUP_JSON to the real p888 group JSON}"
: "${MGT_TARGET_BIN:?set MGT_TARGET_BIN to the real p888 target binary}"
test -s "$MGT_GROUP_JSON"
test -s "$MGT_TARGET_BIN"

if [ "$(nvidia-smi -L | wc -l)" -ne 2 ]; then
  echo "training contract requires exactly 2 visible T4 GPUs" >&2
  exit 2
fi
if [ "$(nvidia-smi --query-gpu=name --format=csv,noheader | grep -c 'Tesla T4')" -ne 2 ]; then
  echo "training contract requires exactly 2 Tesla T4 GPUs" >&2
  exit 2
fi

root="${MGT_CONTRACT_ROOT:-runs/training-contract-2xt4}"
build_dir="${MGT_BUILD_DIR:-build-kaggle-2xt4}"
common_env=(
  MGT_GROUP_JSON="$MGT_GROUP_JSON"
  MGT_TARGET_BIN="$MGT_TARGET_BIN"
  MGT_FORCE_WORLD_SIZE=2
  MGT_WRITE_ARTIFACTS=1
  MGT_CHECKPOINT_PERIOD_STEPS=2
  MGT_WEIGHT_EXPORT_PERIOD_STEPS=2
  MGT_WAIT_GPU_IDLE=0
  MGT_BUILD_DIR="$build_dir"
)
rm -rf "$root/continuous" "$root/split-a" "$root/split-b"

env "${common_env[@]}" MGT_RUN_ROOT="$root/continuous" MGT_STEPS=4 \
  bash kaggle/kernel/run_ranks_2xt4.sh
env "${common_env[@]}" MGT_SKIP_BUILD=1 MGT_RUN_ROOT="$root/split-a" MGT_STEPS=2 \
  bash kaggle/kernel/run_ranks_2xt4.sh
env "${common_env[@]}" MGT_SKIP_BUILD=1 MGT_RUN_ROOT="$root/split-b" \
  MGT_RESUME_ROOT="$root/split-a" MGT_STEPS=2 \
  bash kaggle/kernel/run_ranks_2xt4.sh

for rank in 0 1; do
  grep -q '^completed_steps=4$' "$root/continuous/rank${rank}/checkpoint/manifest.env"
  grep -q '^completed_steps=4$' "$root/split-b/rank${rank}/checkpoint/manifest.env"
  cmp "$root/continuous/rank${rank}/checkpoint/state.f32.bin" \
      "$root/split-b/rank${rank}/checkpoint/state.f32.bin"
  cmp "$root/continuous/rank${rank}/weights/weights.f32.bin" \
      "$root/split-b/rank${rank}/weights/weights.f32.bin"
  test -s "$root/continuous/rank${rank}/checkpoints/step-2/state.f32.bin"
  test -s "$root/continuous/rank${rank}/weight_exports/step-2/weights.f32.bin"
done
cmp "$root/continuous/rank0/checkpoint/state.f32.bin" \
    "$root/continuous/rank1/checkpoint/state.f32.bin"
cmp "$root/split-b/rank0/checkpoint/state.f32.bin" \
    "$root/split-b/rank1/checkpoint/state.f32.bin"

echo "training_contract_2xt4_ok root=$root"
