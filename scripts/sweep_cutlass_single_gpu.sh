#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export MGT_CUDA_ARCH=${MGT_CUDA_ARCH:-75}

if [ -f scripts/ensure_cutlass.sh ]; then
  source scripts/ensure_cutlass.sh
fi

timestamp=$(date +%Y%m%d_%H%M%S)
build_root="${MGT_BUILD_DIR:-build-cutlass-single-gpu}"
sweep_root="${MGT_SWEEP_ROOT:-runs/local-cutlass-single-gpu-${timestamp}}"
mkdir -p "$sweep_root"

echo "preflight_cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
if command -v nvidia-smi >/dev/null 2>&1; then
  echo "preflight_nvidia_smi_l"
  nvidia-smi -L || true
else
  echo "preflight_nvidia_smi_missing=1"
fi
if [ "${MGT_WAIT_GPU_IDLE:-1}" = "1" ]; then
  python3 scripts/wait_gpu_idle.py --devices "${CUDA_VISIBLE_DEVICES:-}" \
    --max-util "${MGT_GPU_IDLE_MAX_UTIL:-15}" \
    --max-memory-mb "${MGT_GPU_IDLE_MAX_MEMORY_MB:-1536}" \
    --timeout-sec "${MGT_GPU_IDLE_TIMEOUT_SEC:-900}" \
    --poll-sec "${MGT_GPU_IDLE_POLL_SEC:-5}"
fi

cat > "$sweep_root/sweep_manifest.tsv" <<'EOF'
config_id	cutlass_half_gemm_kinds	batch_size	steps	input_grad_position_tile	lt_workspace_bytes	backward_profile	input_grad_sparse
EOF

default_configs=$(cat <<'EOF'
input input_embedding_grad
input_forward input_embedding_grad,forward
input_backprop input_embedding_grad,backprop_input
input_grad_weights input_embedding_grad,grad_weights
all all
EOF
)

configs_text="${MGT_CUTLASS_SELECTOR_CONFIGS:-$default_configs}"
configs_text="${configs_text//\\n/$'\n'}"

build_for_selector() {
  local selector="$1"
  local safe="$selector"
  safe="${safe//,/__}"
  safe="${safe//;/__}"
  safe="${safe// /}"
  local build_dir="${build_root}-${safe}"
  if [ ! -x "$build_dir/mgt_native_train" ]; then
    cmake -S native -B "$build_dir" \
      -DMGT_ENABLE_CUDA=ON \
      -DMGT_ENABLE_NCCL=OFF \
      -DCMAKE_CUDA_ARCHITECTURES="${MGT_CUDA_ARCH}" \
      -DMGT_CUTLASS_ROOT="${CUTLASS_ROOT:-/opt/cutlass}" \
      -DMGT_AUTO_CUTLASS_HALF_GEMM="${MGT_AUTO_CUTLASS_HALF_GEMM:-ON}" \
      -DMGT_CUTLASS_HALF_GEMM_KINDS="$selector" >&2
    cmake --build "$build_dir" --config Release --target mgt_native_train >&2
  fi
  printf '%s\n' "$build_dir"
}

while read -r config_id cutlass_kinds config_batch_size config_position_tile config_lt_workspace config_backward_profile config_input_grad_sparse; do
  if [ -z "${config_id:-}" ]; then
    continue
  fi
  case "$config_id" in
    \#*) continue ;;
  esac
  cutlass_kinds="${cutlass_kinds:-input_embedding_grad,forward}"
  run_dir="$sweep_root/$config_id"
  mkdir -p "$run_dir"
  build_dir="$(build_for_selector "$cutlass_kinds")"
  batch_size="${config_batch_size:-${MGT_BATCH_SIZE:-24576}}"
  steps="${MGT_SWEEP_STEPS:-8}"
  position_tile="${config_position_tile:-${MGT_INPUT_GRAD_POSITION_TILE:-32}}"
  lt_workspace="${config_lt_workspace:-${MGT_LT_WORKSPACE_BYTES:-16777216}}"
  backward_profile="${config_backward_profile:-${MGT_BACKWARD_PROFILE:-0}}"
  input_grad_sparse="${config_input_grad_sparse:-${MGT_INPUT_GRAD_SPARSE:-0}}"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$config_id" "$cutlass_kinds" "$batch_size" "$steps" "$position_tile" "$lt_workspace" "$backward_profile" "$input_grad_sparse" >> "$sweep_root/sweep_manifest.tsv"
  echo "single_cutlass_config_start id=${config_id} cutlass=${cutlass_kinds} batch=${batch_size} steps=${steps} tile=${position_tile} workspace=${lt_workspace} profile=${backward_profile} sparse=${input_grad_sparse}"
  if [ "${MGT_WAIT_GPU_IDLE:-1}" = "1" ]; then
    python3 scripts/wait_gpu_idle.py --devices "${CUDA_VISIBLE_DEVICES:-}" \
      --max-util "${MGT_GPU_IDLE_MAX_UTIL:-15}" \
      --max-memory-mb "${MGT_GPU_IDLE_MAX_MEMORY_MB:-1536}" \
      --timeout-sec "${MGT_GPU_IDLE_TIMEOUT_SEC:-900}" \
      --poll-sec "${MGT_GPU_IDLE_POLL_SEC:-5}"
  fi
  set +e
  "$build_dir/mgt_native_train" \
    --synthetic-benchmark 1 \
    --output-dir "$run_dir" \
    --steps "$steps" \
    --device-id "${MGT_DEVICE_ID:-0}" \
    --world-size 1 \
    --global-rank 0 \
    --local-rank 0 \
    --batch-size "$batch_size" \
    --k-min "${MGT_K_MIN:-1}" \
    --k-max "${MGT_K_MAX:-29}" \
    --hd1 "${MGT_HD1:-2556}" \
    --hd2 "${MGT_HD2:-218}" \
    --nrd "${MGT_NRD:-16}" \
    --output-dim "${MGT_OUTPUT_DIM:-1}" \
    --write-artifacts "${MGT_WRITE_ARTIFACTS:-0}" \
    --backward-profile "$backward_profile" \
    --input-grad-fp16 "${MGT_INPUT_GRAD_FP16:-1}" \
    --input-grad-position-tile "$position_tile" \
    --input-grad-sparse "$input_grad_sparse" \
    --linear-fp16 "${MGT_LINEAR_FP16:-1}" \
    --lt-workspace-bytes "$lt_workspace" \
    --lt-autotune "${MGT_LT_AUTOTUNE:-1}" \
    --lt-autotune-candidates "${MGT_LT_AUTOTUNE_CANDIDATES:-8}" \
    --lt-autotune-warmups "${MGT_LT_AUTOTUNE_WARMUPS:-1}" \
    --lt-autotune-iters "${MGT_LT_AUTOTUNE_ITERS:-2}" \
    --overlap-allreduce 0 \
    --allreduce-bucket-bytes "${MGT_ALLREDUCE_BUCKET_BYTES:-4194304}" > "$run_dir/launch.log" 2>&1
  code=$?
  set -e
  echo "$code" > "$run_dir/runner_return_code.txt"
  echo "$cutlass_kinds" > "$run_dir/cutlass_half_gemm_kinds.txt"
  echo "single_cutlass_config_done id=${config_id} return_code=${code}"
done <<< "$configs_text"

MGT_SWEEP_SUMMARY_ROOT="$sweep_root" python3 - <<'PY'
import json
import os
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path("scripts").resolve()))
from estimate_train_flops import estimate_train_flops

root = Path(os.environ["MGT_SWEEP_SUMMARY_ROOT"])
rows = []
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
for run_dir in sorted(path for path in root.iterdir() if path.is_dir()):
    code_path = run_dir / "runner_return_code.txt"
    code = int(code_path.read_text().strip()) if code_path.exists() else None
    selector_path = run_dir / "cutlass_half_gemm_kinds.txt"
    row = {
        "config_id": run_dir.name,
        "return_code": code,
        "status": "missing_profile",
        "cutlass_half_gemm_kinds": selector_path.read_text().strip() if selector_path.exists() else "",
    }
    profile_path = run_dir / "profile.jsonl"
    if profile_path.exists():
        profile_rows = [json.loads(line) for line in profile_path.read_text().splitlines() if line.strip()]
        steady = [item for item in profile_rows if int(item.get("step", 0)) > 0]
        if steady:
            row["status"] = "ok" if code == 0 else "failed"
            step_ms_values = [float(item["milliseconds"]) for item in steady]
            throughput_values = [float(item["batch_states"]) * 1000.0 / float(item["milliseconds"]) for item in steady]
            row["avg_step_ms"] = statistics.mean(step_ms_values)
            row["median_step_ms"] = statistics.median(step_ms_values)
            row["min_step_ms"] = min(step_ms_values)
            row["max_step_ms"] = max(step_ms_values)
            row["avg_throughput_states_s"] = statistics.mean(throughput_values)
            row["median_throughput_states_s"] = statistics.median(throughput_values)
            row["min_throughput_states_s"] = min(throughput_values)
            row["max_throughput_states_s"] = max(throughput_values)
            row["input_grad_backend"] = steady[0].get("input_grad_backend")
            row["batch_states"] = int(steady[0].get("batch_states", 0))
            row["steady_steps"] = len(steady)
            flop_estimate = estimate_train_flops(
                state_len=int(os.environ.get("MGT_STATE_LEN", "80")),
                state_value_pad=int(os.environ.get("MGT_STATE_VALUE_PAD", "2")),
                hd1=int(os.environ.get("MGT_PHYSICAL_HD1", os.environ.get("MGT_HD1", "2556"))),
                hd2=((int(os.environ.get("MGT_HD2", "218")) + 7) // 8) * 8,
                residual_blocks=int(os.environ.get("MGT_NRD", "16")),
                output_dim=int(os.environ.get("MGT_OUTPUT_DIM", "1")),
                batch_size=row["batch_states"],
                throughput_states_s=row["median_throughput_states_s"],
                peak_tflops=float(os.environ["MGT_GPU_PEAK_TFLOPS"]) if os.environ.get("MGT_GPU_PEAK_TFLOPS") else None,
            )
            row["flop_estimate"] = flop_estimate.__dict__
            for key in stage_keys:
                row[f"avg_{key}"] = statistics.mean(float(item.get(key, 0.0)) for item in steady)
        else:
            row["status"] = "no_steady_steps"
    else:
        launch = run_dir / "launch.log"
        if launch.exists():
            row["launch_tail"] = launch.read_text(errors="replace").splitlines()[-20:]
        if code not in (None, 0):
            row["status"] = "failed"
    rows.append(row)

ok_rows = [row for row in rows if row.get("status") == "ok" and "median_throughput_states_s" in row]
ok_rows.sort(key=lambda row: row["median_throughput_states_s"], reverse=True)
summary = {
    "format": "mgt_single_gpu_cutlass_selector_sweep_v1",
    "run_root": str(root),
    "rows": rows,
    "best": ok_rows[0] if ok_rows else None,
    "ok_count": len(ok_rows),
    "failed_count": len(rows) - len(ok_rows),
}
(root / "sweep_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print("single_cutlass_summary=", json.dumps({
    "ok_count": summary["ok_count"],
    "failed_count": summary["failed_count"],
    "best_config_id": summary["best"].get("config_id") if summary["best"] else None,
    "best_avg_throughput_states_s": summary["best"].get("avg_throughput_states_s") if summary["best"] else None,
    "best_median_throughput_states_s": summary["best"].get("median_throughput_states_s") if summary["best"] else None,
    "best_cutlass_half_gemm_kinds": summary["best"].get("cutlass_half_gemm_kinds") if summary["best"] else None,
}, sort_keys=True), flush=True)
for row in ok_rows:
    print("single_cutlass_row\t{config_id}\t{cutlass}\tmedian={median:.2f}\tavg={avg:.2f}\tstep_med={step_median:.3f}\tstep_avg={step:.3f}\twalk={walk:.3f}\tbackward={backward:.3f}\tinput_grad={input_grad:.3f}\tadam={adam:.3f}".format(
        config_id=row["config_id"],
        cutlass=row.get("cutlass_half_gemm_kinds", ""),
        median=row.get("median_throughput_states_s", 0.0),
        avg=row.get("avg_throughput_states_s", 0.0),
        step_median=row.get("median_step_ms", 0.0),
        step=row.get("avg_step_ms", 0.0),
        walk=row.get("avg_walk_ms", 0.0),
        backward=row.get("avg_backward_ms", 0.0),
        input_grad=row.get("avg_bw_input_grad_ms", 0.0),
        adam=row.get("avg_adam_ms", 0.0),
    ), flush=True)
    flop = row.get("flop_estimate", {})
    if flop:
        print("single_cutlass_flops\t{config_id}\ttotal_per_sample={per_sample}\tachieved_tflops={achieved:.3f}\tpeak_util={util}".format(
            config_id=row["config_id"],
            per_sample=flop.get("total_flops_per_sample", 0),
            achieved=float(flop.get("achieved_tflops") or 0.0),
            util=("n/a" if flop.get("peak_utilization_percent") is None else f"{float(flop['peak_utilization_percent']):.2f}%"),
        ), flush=True)
PY
