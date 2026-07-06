#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
export MGT_CUDA_ARCH=${MGT_CUDA_ARCH:-75}
if [ -f scripts/ensure_cutlass.sh ]; then
  source scripts/ensure_cutlass.sh
fi
build_dir="${MGT_BUILD_DIR:-build-kaggle-2xt4}"
run_root="${MGT_RUN_ROOT:-runs/kaggle-2xt4}"
if [ "${MGT_SKIP_BUILD:-0}" != "1" ]; then
  cmake -S native -B "$build_dir" -DMGT_ENABLE_CUDA=ON -DMGT_ENABLE_NCCL=ON -DCMAKE_CUDA_ARCHITECTURES=${MGT_CUDA_ARCH} -DMGT_CUTLASS_ROOT="${CUTLASS_ROOT:-/opt/cutlass}" -DMGT_AUTO_CUTLASS_HALF_GEMM=${MGT_AUTO_CUTLASS_HALF_GEMM:-ON}
  cmake --build "$build_dir" --config Release --target mgt_native_train
fi
if command -v nvidia-smi >/dev/null 2>&1; then
  visible_count=$(nvidia-smi -L | wc -l)
else
  visible_count=$(python3 - <<'PY'
import os
value = os.environ.get('CUDA_VISIBLE_DEVICES', '')
print(len([part for part in value.split(',') if part.strip()]) if value else 1)
PY
)
fi
if [ "${MGT_FORCE_WORLD_SIZE:-}" != "" ]; then
  world_size="${MGT_FORCE_WORLD_SIZE}"
elif [ "$visible_count" -ge 2 ]; then
  world_size=2
else
  world_size=1
fi
mkdir -p "$run_root"
rm -f "$run_root/nccl.id"
if [ "${MGT_PERF_RUN:-0}" = "1" ]; then
  export MGT_FULL_MODEL="${MGT_FULL_MODEL:-1}"
  export MGT_STEPS="${MGT_STEPS:-8}"
  export MGT_BATCH_SIZE="${MGT_BATCH_SIZE:-53248}"
  export MGT_WRITE_ARTIFACTS="${MGT_WRITE_ARTIFACTS:-0}"
  export MGT_BACKWARD_PROFILE="${MGT_BACKWARD_PROFILE:-0}"
  export MGT_INPUT_GRAD_FP16="${MGT_INPUT_GRAD_FP16:-1}"
  export MGT_INPUT_GRAD_POSITION_TILE="${MGT_INPUT_GRAD_POSITION_TILE:-48}"
  export MGT_INPUT_GRAD_SPARSE="${MGT_INPUT_GRAD_SPARSE:-0}"
  export MGT_LINEAR_FP16="${MGT_LINEAR_FP16:-1}"
  export MGT_OVERLAP_ALLREDUCE="${MGT_OVERLAP_ALLREDUCE:-1}"
  export MGT_ALLREDUCE_BUCKET_BYTES="${MGT_ALLREDUCE_BUCKET_BYTES:-4194304}"
  export MGT_LT_WORKSPACE_BYTES="${MGT_LT_WORKSPACE_BYTES:-16777216}"
  export MGT_LT_AUTOTUNE="${MGT_LT_AUTOTUNE:-0}"
  export MGT_LT_AUTOTUNE_CANDIDATES="${MGT_LT_AUTOTUNE_CANDIDATES:-8}"
  export MGT_LT_AUTOTUNE_WARMUPS="${MGT_LT_AUTOTUNE_WARMUPS:-1}"
  export MGT_LT_AUTOTUNE_ITERS="${MGT_LT_AUTOTUNE_ITERS:-2}"
fi
if [ "${MGT_FULL_MODEL:-0}" = "1" ]; then
  export MGT_STEPS="${MGT_STEPS:-1}"
  export MGT_BATCH_SIZE="${MGT_BATCH_SIZE:-1}"
  export MGT_K_MIN="${MGT_K_MIN:-1}"
  export MGT_K_MAX="${MGT_K_MAX:-29}"
  export MGT_HD1="${MGT_HD1:-2556}"
  export MGT_HD2="${MGT_HD2:-218}"
  export MGT_NRD="${MGT_NRD:-16}"
fi
pids=()
for rank in $(seq 0 $((world_size - 1))); do
  "$build_dir/mgt_native_train" \
    --output-dir "$run_root/rank${rank}" \
    --steps "${MGT_STEPS:-3}" \
    --device-id "$rank" \
    --world-size "$world_size" \
    --global-rank "$rank" \
    --local-rank "$rank" \
    --batch-size "${MGT_BATCH_SIZE:-64}" \
    --k-min "${MGT_K_MIN:-1}" \
    --k-max "${MGT_K_MAX:-9}" \
    --hd1 "${MGT_HD1:-5}" \
    --hd2 "${MGT_HD2:-3}" \
    --nrd "${MGT_NRD:-1}" \
    --output-dim "${MGT_OUTPUT_DIM:-1}" \
    --write-artifacts "${MGT_WRITE_ARTIFACTS:-1}" \
    --backward-profile "${MGT_BACKWARD_PROFILE:-0}" \
    --input-grad-fp16 "${MGT_INPUT_GRAD_FP16:-0}" \
    --input-grad-position-tile "${MGT_INPUT_GRAD_POSITION_TILE:-0}" \
    --input-grad-sparse "${MGT_INPUT_GRAD_SPARSE:-0}" \
    --linear-fp16 "${MGT_LINEAR_FP16:-0}" \
    --lt-workspace-bytes "${MGT_LT_WORKSPACE_BYTES:-0}" \
    --lt-autotune "${MGT_LT_AUTOTUNE:-0}" \
    --lt-autotune-candidates "${MGT_LT_AUTOTUNE_CANDIDATES:-8}" \
    --lt-autotune-warmups "${MGT_LT_AUTOTUNE_WARMUPS:-1}" \
    --lt-autotune-iters "${MGT_LT_AUTOTUNE_ITERS:-2}" \
    --overlap-allreduce "${MGT_OVERLAP_ALLREDUCE:-1}" \
    --allreduce-bucket-bytes "${MGT_ALLREDUCE_BUCKET_BYTES:-4194304}" \
    --nccl-id-file "$run_root/nccl.id" > "$run_root/rank${rank}.stdout" 2>&1 &
  pids+=("$!")
done
for pid in "${pids[@]}"; do
  wait "$pid"
done
bash kaggle/kernel/check_rank_outputs.sh "$run_root" "$world_size"
echo "rank_launch_ok world_size=${world_size}"
MGT_SUMMARY_ROOT="$run_root" MGT_CONFIG_ID="${MGT_CONFIG_ID:-$(basename "$run_root")}" python3 - <<'PY'
import json
import os
import statistics
from pathlib import Path
root = Path(os.environ["MGT_SUMMARY_ROOT"])
rank_rows = []
for path in sorted(root.glob("rank*/profile.jsonl")):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    rank_rows.append((path.parent.name, rows))
summary = {
    "config_id": os.environ.get("MGT_CONFIG_ID", root.name),
    "world_size": len(rank_rows),
    "steady_steps": [],
    "status": "ok",
    "config": {
        "batch_size": int(os.environ.get("MGT_BATCH_SIZE", "64")),
        "steps": int(os.environ.get("MGT_STEPS", "3")),
        "output_dim": int(os.environ.get("MGT_OUTPUT_DIM", "1")),
        "input_grad_position_tile": int(os.environ.get("MGT_INPUT_GRAD_POSITION_TILE", "0")),
        "input_grad_sparse": int(os.environ.get("MGT_INPUT_GRAD_SPARSE", "0")),
        "input_grad_fp16": int(os.environ.get("MGT_INPUT_GRAD_FP16", "0")),
        "linear_fp16": int(os.environ.get("MGT_LINEAR_FP16", "0")),
        "overlap_allreduce": int(os.environ.get("MGT_OVERLAP_ALLREDUCE", "1")),
        "allreduce_bucket_bytes": int(os.environ.get("MGT_ALLREDUCE_BUCKET_BYTES", "4194304")),
        "lt_workspace_bytes": int(os.environ.get("MGT_LT_WORKSPACE_BYTES", "0")),
        "lt_autotune": int(os.environ.get("MGT_LT_AUTOTUNE", "0")),
        "lt_autotune_candidates": int(os.environ.get("MGT_LT_AUTOTUNE_CANDIDATES", "8")),
        "lt_autotune_warmups": int(os.environ.get("MGT_LT_AUTOTUNE_WARMUPS", "1")),
        "lt_autotune_iters": int(os.environ.get("MGT_LT_AUTOTUNE_ITERS", "2")),
        "backward_profile": int(os.environ.get("MGT_BACKWARD_PROFILE", "0")),
    },
}
if rank_rows and rank_rows[0][1]:
    first = rank_rows[0][1][0]
    summary["input_grad_backend"] = first.get("input_grad_backend")
    summary["input_grad_position_tile"] = first.get("input_grad_position_tile")
    summary["batch_states_per_rank"] = first.get("batch_states")
if rank_rows:
    common_steps = sorted(set.intersection(*(set(row["step"] for row in rows) for _, rows in rank_rows)))
    stage_keys = [
        "walk_ms",
        "backward_ms",
        "allreduce_ms",
        "adam_ms",
        "bw_input_forward_ms",
        "bw_hidden_forward_ms",
        "bw_residual_forward_ms",
        "bw_output_ms",
        "bw_residual_backward_ms",
        "bw_residual_fc2_dz_ms",
        "bw_residual_fc2_grad_weight_ms",
        "bw_residual_fc2_bias_ms",
        "bw_residual_fc2_backprop_ms",
        "bw_residual_fc1_dz_ms",
        "bw_residual_fc1_grad_weight_ms",
        "bw_residual_fc1_bias_ms",
        "bw_residual_fc1_backprop_ms",
        "bw_residual_skip_add_ms",
        "bw_hidden_backward_ms",
        "bw_input_grad_ms",
    ]
    for step in common_steps:
        if step == 0:
            continue
        per_rank = [next(row for row in rows if row["step"] == step) for _, rows in rank_rows]
        batch_states = sum(int(row["batch_states"]) for row in per_rank)
        step_ms = max(float(row["milliseconds"]) for row in per_rank)
        step_row = {
            "step": step,
            "global_batch_states": batch_states,
            "max_rank_ms": step_ms,
            "throughput_states_s": batch_states * 1000.0 / step_ms,
            "rank_ms": [float(row["milliseconds"]) for row in per_rank],
            "rank_losses": [float(row["loss"]) for row in per_rank],
            "overlap_chunks": [int(row.get("overlap_chunks", 0)) for row in per_rank],
        }
        for key in stage_keys:
            values = [float(row.get(key, 0.0)) for row in per_rank]
            step_row[f"max_{key}"] = max(values)
        summary["steady_steps"].append(step_row)
if summary["steady_steps"]:
    values = [row["throughput_states_s"] for row in summary["steady_steps"]]
    summary["avg_throughput_states_s"] = sum(values) / len(values)
    summary["min_throughput_states_s"] = min(values)
    summary["max_throughput_states_s"] = max(values)
    summary["avg_step_ms"] = statistics.mean(row["max_rank_ms"] for row in summary["steady_steps"])
    for key in [
        "walk_ms",
        "backward_ms",
        "allreduce_ms",
        "adam_ms",
        "bw_input_forward_ms",
        "bw_hidden_forward_ms",
        "bw_residual_forward_ms",
        "bw_output_ms",
        "bw_residual_backward_ms",
        "bw_residual_fc2_dz_ms",
        "bw_residual_fc2_grad_weight_ms",
        "bw_residual_fc2_bias_ms",
        "bw_residual_fc2_backprop_ms",
        "bw_residual_fc1_dz_ms",
        "bw_residual_fc1_grad_weight_ms",
        "bw_residual_fc1_bias_ms",
        "bw_residual_fc1_backprop_ms",
        "bw_residual_skip_add_ms",
        "bw_hidden_backward_ms",
        "bw_input_grad_ms",
    ]:
        summary[f"avg_{key}"] = statistics.mean(row[f"max_{key}"] for row in summary["steady_steps"])
else:
    summary["status"] = "no_steady_steps"
(root / "throughput_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print("throughput_summary=", json.dumps(summary, sort_keys=True), flush=True)
PY
