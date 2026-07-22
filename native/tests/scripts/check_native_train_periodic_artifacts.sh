#!/usr/bin/env bash
set -euo pipefail
run_dir="${1:?run dir required}"
for step in 1 2; do
  test -s "$run_dir/checkpoints/step-${step}/manifest.env"
  test -s "$run_dir/checkpoints/step-${step}/state.f32.bin"
  grep -q "completed_steps=${step}" "$run_dir/checkpoints/step-${step}/manifest.env"
  test -s "$run_dir/weight_exports/step-${step}/manifest.json"
  test -s "$run_dir/weight_exports/step-${step}/weights.f32.bin"
done
test -s "$run_dir/checkpoint/manifest.env"
test -s "$run_dir/checkpoint/state.f32.bin"
test -s "$run_dir/weights/manifest.json"
test ! -e "$run_dir/checkpoint.tmp"
test ! -e "$run_dir/weights.tmp"
