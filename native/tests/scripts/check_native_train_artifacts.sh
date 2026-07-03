#!/usr/bin/env bash
set -euo pipefail
run_dir="${1:?run dir required}"
test -s "$run_dir/train.log"
test -s "$run_dir/metadata.env"
test -s "$run_dir/layers.json"
test -s "$run_dir/weights/manifest.json"
test -s "$run_dir/weights/weights.f32.bin"
grep -q '"format": "stream1_weights"' "$run_dir/weights/manifest.json"
grep -q 'OUTPUT_DIM=1' "$run_dir/metadata.env"