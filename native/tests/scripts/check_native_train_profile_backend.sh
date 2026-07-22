#!/usr/bin/env bash
set -euo pipefail
binary="${1:?binary required}"
run_dir="${2:?run dir required}"
expected_output_dim="${3:?expected output dim required}"
expected_backend="${4:?expected backend required}"
shift 4
"$binary" --synthetic-benchmark 1 --output-dir "$run_dir" --steps 1 --device-id 0 --world-size 1 --global-rank 0 --local-rank 0 --batch-size 8 --k-min 2 --k-max 5 --hd1 7 --hd2 4 --write-artifacts 1 --backward-profile 1 "$@"
test -s "$run_dir/metadata.env"
test -s "$run_dir/layers.json"
test -s "$run_dir/profile.jsonl"
test -s "$run_dir/weights/manifest.json"
grep -q "OUTPUT_DIM=$expected_output_dim" "$run_dir/metadata.env"
grep -q "INPUT_GRAD_BACKEND=$expected_backend" "$run_dir/metadata.env"
grep -q "\"output_dim\": $expected_output_dim" "$run_dir/layers.json"
grep -q "\"output_dim\": $expected_output_dim" "$run_dir/weights/manifest.json"
grep -q "\"input_grad_backend\":\"$expected_backend\"" "$run_dir/profile.jsonl"