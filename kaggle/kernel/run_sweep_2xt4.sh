#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
export MGT_CUDA_ARCH=${MGT_CUDA_ARCH:-75}
if [ -f scripts/ensure_cutlass.sh ]; then
  source scripts/ensure_cutlass.sh
fi
build_dir="${MGT_BUILD_DIR:-build-kaggle-2xt4}"
timestamp=$(date +%Y%m%d_%H%M%S)
sweep_root="${MGT_SWEEP_ROOT:-runs/kaggle-2xt4-sweep-${timestamp}}"
mkdir -p "$sweep_root"
echo "preflight_cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
if command -v nvidia-smi >/dev/null 2>&1; then
  echo "preflight_nvidia_smi_l"
  nvidia-smi -L || true
  echo "preflight_nvidia_smi"
  nvidia-smi || true
else
  echo "preflight_nvidia_smi_missing=1"
fi
cmake -S native -B "$build_dir" -DMGT_ENABLE_CUDA=ON -DMGT_ENABLE_NCCL=ON -DCMAKE_CUDA_ARCHITECTURES=${MGT_CUDA_ARCH} -DMGT_CUTLASS_ROOT="${CUTLASS_ROOT:-/opt/cutlass}" -DMGT_AUTO_CUTLASS_HALF_GEMM=${MGT_AUTO_CUTLASS_HALF_GEMM:-ON}
if [ "${MGT_SWEEP_RUN_CTEST:-1}" = "1" ]; then
  cmake --build "$build_dir" --config Release
  ctest --test-dir "$build_dir" -R 'cuda_compile|cuda_random_walk_smoke|cuda_adamw_smoke|cuda_mlp_forward_smoke|cuda_mlp_backward_smoke|cuda_train_step_smoke|nccl_single_rank_smoke|nccl_two_device_smoke|native_train_smoke|native_train_profile_smoke|native_train_input_grad_.*backend|native_train_resume_smoke|native_train_artifacts' --output-on-failure -C Release
else
  cmake --build "$build_dir" --config Release --target mgt_native_train
fi
cat > "$sweep_root/sweep_manifest.tsv" <<'EOF'
config_id	batch_size	input_grad_position_tile	lt_workspace_bytes	allreduce_bucket_bytes	backward_profile	overlap_allreduce	input_grad_sparse
EOF
default_configs=$(cat <<'EOF'
b49152_t36_ws0_bucket4m 49152 36 0 4194304 0 1 0
b53248_t36_ws0_bucket4m 53248 36 0 4194304 0 1 0
b57344_t36_ws0_bucket4m 57344 36 0 4194304 0 1 0
b61440_t36_ws0_bucket4m 61440 36 0 4194304 0 1 0
b53248_t16_ws0_bucket4m 53248 16 0 4194304 0 1 0
b53248_t24_ws0_bucket4m 53248 24 0 4194304 0 1 0
b53248_t32_ws0_bucket4m 53248 32 0 4194304 0 1 0
b53248_t40_ws0_bucket4m 53248 40 0 4194304 0 1 0
b53248_t36_ws16m_bucket4m 53248 36 16777216 4194304 0 1 0
b53248_t48_ws16m_bucket4m 53248 48 16777216 4194304 0 1 0
b53248_t56_ws16m_bucket4m 53248 56 16777216 4194304 0 1 0
b53248_t64_ws16m_bucket4m 53248 64 16777216 4194304 0 1 0
b53248_t72_ws16m_bucket4m 53248 72 16777216 4194304 0 1 0
b53248_t36_ws24m_bucket4m 53248 36 25165824 4194304 0 1 0
b53248_t36_ws32m_bucket4m 53248 36 33554432 4194304 0 1 0
b53248_t36_ws40m_bucket4m 53248 36 41943040 4194304 0 1 0
b53248_t36_ws48m_bucket4m 53248 36 50331648 4194304 0 1 0
b53248_t36_ws32m_bucket2m 53248 36 33554432 2097152 0 1 0
b53248_t36_ws32m_bucket8m 53248 36 33554432 8388608 0 1 0
b51200_t36_ws32m_bucket4m 51200 36 33554432 4194304 0 1 0
b55296_t36_ws32m_bucket4m 55296 36 33554432 4194304 0 1 0
b53248_sparse_ws32m_bucket4m 53248 36 33554432 4194304 0 1 1
b53248_t36_ws64m_bucket4m 53248 36 67108864 4194304 0 1 0
b53248_t36_ws0_nooverlap 53248 36 0 4194304 0 0 0
b53248_t36_ws0_profile 53248 36 0 4194304 1 1 0
b53248_t36_ws32m_profile 53248 36 33554432 4194304 1 1 0
EOF
)
configs_text="${MGT_SWEEP_CONFIGS:-$default_configs}"
while read -r config_id batch_size position_tile lt_workspace bucket_bytes backward_profile overlap_allreduce input_grad_sparse; do
  if [ -z "${config_id:-}" ]; then
    continue
  fi
  case "$config_id" in
    \#*) continue ;;
  esac
  input_grad_sparse="${input_grad_sparse:-0}"
  run_dir="$sweep_root/$config_id"
  mkdir -p "$run_dir"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$config_id" "$batch_size" "$position_tile" "$lt_workspace" "$bucket_bytes" "$backward_profile" "$overlap_allreduce" "$input_grad_sparse" >> "$sweep_root/sweep_manifest.tsv"
  echo "sweep_config_start id=${config_id} batch=${batch_size} tile=${position_tile} workspace=${lt_workspace} bucket=${bucket_bytes} profile=${backward_profile} overlap=${overlap_allreduce} sparse=${input_grad_sparse}"
  set +e
  MGT_CONFIG_ID="$config_id" \
  MGT_BUILD_DIR="$build_dir" \
  MGT_RUN_ROOT="$run_dir" \
  MGT_SKIP_BUILD=1 \
  MGT_PERF_RUN=1 \
  MGT_FULL_MODEL=1 \
  MGT_STEPS="${MGT_SWEEP_STEPS:-8}" \
  MGT_BATCH_SIZE="$batch_size" \
  MGT_INPUT_GRAD_POSITION_TILE="$position_tile" \
  MGT_INPUT_GRAD_SPARSE="$input_grad_sparse" \
  MGT_LT_WORKSPACE_BYTES="$lt_workspace" \
  MGT_ALLREDUCE_BUCKET_BYTES="$bucket_bytes" \
  MGT_BACKWARD_PROFILE="$backward_profile" \
  MGT_OVERLAP_ALLREDUCE="$overlap_allreduce" \
  MGT_WRITE_ARTIFACTS=0 \
  MGT_INPUT_GRAD_FP16="${MGT_INPUT_GRAD_FP16:-1}" \
  MGT_LINEAR_FP16="${MGT_LINEAR_FP16:-1}" \
  MGT_OUTPUT_DIM="${MGT_OUTPUT_DIM:-1}" \
  bash kaggle/kernel/run_ranks_2xt4.sh > "$run_dir/launch.log" 2>&1
  code=$?
  set -e
  echo "$code" > "$run_dir/runner_return_code.txt"
  echo "sweep_config_done id=${config_id} return_code=${code}"
done <<< "$configs_text"
MGT_SWEEP_SUMMARY_ROOT="$sweep_root" python3 - <<'PY'
import json
import os
from pathlib import Path
root = Path(os.environ["MGT_SWEEP_SUMMARY_ROOT"])
rows = []
for run_dir in sorted(path for path in root.iterdir() if path.is_dir()):
    code_path = run_dir / "runner_return_code.txt"
    code = int(code_path.read_text().strip()) if code_path.exists() else None
    summary_path = run_dir / "throughput_summary.json"
    row = {"config_id": run_dir.name, "return_code": code, "status": "missing_summary"}
    if summary_path.exists():
        data = json.loads(summary_path.read_text())
        row.update(data)
        row["status"] = "ok" if code == 0 and data.get("status") == "ok" else data.get("status", "failed")
    else:
        launch = run_dir / "launch.log"
        if launch.exists():
            lines = launch.read_text(errors="replace").splitlines()
            row["launch_tail"] = lines[-20:]
        if code not in (None, 0):
            row["status"] = "failed"
    rows.append(row)
ok_rows = [row for row in rows if row.get("status") == "ok" and "avg_throughput_states_s" in row]
ok_rows.sort(key=lambda row: row["avg_throughput_states_s"], reverse=True)
summary = {
    "format": "mgt_2xt4_sweep_v1",
    "run_root": str(root),
    "rows": rows,
    "best": ok_rows[0] if ok_rows else None,
    "ok_count": len(ok_rows),
    "failed_count": len(rows) - len(ok_rows),
}
(root / "sweep_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print("sweep_summary=", json.dumps({
    "ok_count": summary["ok_count"],
    "failed_count": summary["failed_count"],
    "best_config_id": summary["best"].get("config_id") if summary["best"] else None,
    "best_avg_throughput_states_s": summary["best"].get("avg_throughput_states_s") if summary["best"] else None,
}, sort_keys=True), flush=True)
for row in ok_rows:
    print("sweep_row\t{config_id}\t{avg:.2f}\t{step:.3f}\t{walk:.3f}\t{backward:.3f}\t{allreduce:.3f}\t{adam:.3f}".format(
        config_id=row["config_id"],
        avg=row.get("avg_throughput_states_s", 0.0),
        step=row.get("avg_step_ms", 0.0),
        walk=row.get("avg_walk_ms", 0.0),
        backward=row.get("avg_backward_ms", 0.0),
        allreduce=row.get("avg_allreduce_ms", 0.0),
        adam=row.get("avg_adam_ms", 0.0),
    ), flush=True)
PY