#!/usr/bin/env bash
set -euo pipefail
root="${1:?run root required}"
world_size="${2:?world size required}"
if [ "$world_size" -lt 1 ]; then
  echo "world_size must be positive" >&2
  exit 1
fi
seen=""
for rank in $(seq 0 $((world_size - 1))); do
  run_dir="$root/rank${rank}"
  test -d "$run_dir"
  test -s "$run_dir/metadata.env"
  test -s "$run_dir/train.log"
  test -s "$run_dir/layers.json"
  test -s "$run_dir/weights/manifest.json"
  test -s "$run_dir/weights/weights.f32.bin"
  test -s "$run_dir/weights/input_weight_hxk.fp16"
  test -s "$run_dir/weights/hidden_weight_hxk.fp16"
  test -s "$run_dir/weights/output_weight_hxk.fp16"
  test -s "$run_dir/checkpoint/manifest.json"
  test -s "$run_dir/checkpoint/state.f32.bin"
  grep -q "WORLD_SIZE=${world_size}" "$run_dir/metadata.env"
  grep -q "GLOBAL_RANK=${rank}" "$run_dir/metadata.env"
  grep -q "LOCAL_RANK=${rank}" "$run_dir/metadata.env"
  grep -q "DEVICE_ID=${rank}" "$run_dir/metadata.env"
  grep -q 'MODEL_MODE=MLP2RB' "$run_dir/metadata.env"
  grep -q 'OUTPUT_DIM=1' "$run_dir/metadata.env"
  grep -q 'NCCL_ENABLED=1' "$run_dir/metadata.env"
  grep -q 'phase=train' "$run_dir/train.log"
  grep -q 'nccl=1' "$run_dir/train.log"
  grep -q '"format": "stream1_weights"' "$run_dir/weights/manifest.json"
  grep -q '"dtype": "fp16"' "$run_dir/weights/manifest.json"
  grep -q '"num_classes": 72' "$run_dir/weights/manifest.json"
  grep -q "\"nrd\": ${MGT_NRD:-1}" "$run_dir/weights/manifest.json"
  grep -q "\"original_hd1\": ${MGT_HD1:-5}" "$run_dir/weights/manifest.json"
  grep -q "\"original_hd2\": ${MGT_HD2:-3}" "$run_dir/weights/manifest.json"
  grep -q '"format": "mgt_train_checkpoint"' "$run_dir/checkpoint/manifest.json"
  current=$(grep '^GLOBAL_RANK=' "$run_dir/metadata.env" | cut -d= -f2)
  case " $seen " in
    *" $current "*) echo "duplicate rank ${current}" >&2; exit 1 ;;
  esac
  seen="$seen $current"
done
count=$(find "$root" -maxdepth 1 -type d -name 'rank*' | wc -l)
if [ "$count" -ne "$world_size" ]; then
  echo "rank directory count ${count} != world_size ${world_size}" >&2
  exit 1
fi
echo "rank_outputs_ok world_size=${world_size}"