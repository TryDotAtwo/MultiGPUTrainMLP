#!/usr/bin/env bash
#SBATCH --job-name=mgt-nccl-sweep
#SBATCH --partition=kaf12
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:8
#SBATCH --mem=128G
#SBATCH --time=00:30:00

set -euo pipefail

world="${SLURM_NTASKS:-8}"
global_rows="${MGT_GLOBAL_ROWS:-100000}"
warmup="${MGT_WARMUP:-20}"
steps="${MGT_STEPS:-100}"
repeats="${MGT_REPEATS:-3}"
build_dir="${MGT_BUILD_DIR:-build-a100-production}"
benchmark="${MGT_BENCHMARK:-${build_dir}/mgt_bn_step_benchmark}"
results_root="${MGT_RESULTS_ROOT:-results/${SLURM_JOB_ID:-manual}-nccl-sweep}"

if [[ "$world" -ne 8 ]]; then
  echo "error: expected 8 SLURM tasks, got ${world}" >&2
  exit 2
fi
if (( global_rows % world != 0 )); then
  echo "error: global rows ${global_rows} are not divisible by world ${world}" >&2
  exit 2
fi
if [[ ! -x "$benchmark" ]]; then
  echo "error: benchmark is not executable: ${benchmark}" >&2
  exit 2
fi

local_rows=$((global_rows / world))
mkdir -p "$results_root/raw"

echo "revision=$(git rev-parse HEAD)"
echo "benchmark=${benchmark}"
echo "results=${results_root}"
echo "world=${world} global_rows=${global_rows} local_rows=${local_rows}"
nvidia-smi -L

# Keep the baseline at both ends to expose thermal or cluster drift. Unsupported
# NCCL combinations are recorded as failed candidates instead of aborting the sweep.
configs=(
  "baseline||||"
  "ring|Ring|||"
  "tree|Tree|||"
  "simple||Simple||"
  "ll||LL||"
  "ll128||LL128||"
  "ring_simple|Ring|Simple||"
  "ring_ll|Ring|LL||"
  "ring_ll128|Ring|LL128||"
  "tree_simple|Tree|Simple||"
  "tree_ll|Tree|LL||"
  "channels4||||4"
  "channels8||||8"
  "channels16||||16"
)

run_one() {
  local spec="$1"
  local repeat="$2"
  local name algo proto min_channels max_channels
  IFS='|' read -r name algo proto min_channels max_channels <<< "$spec"

  local tag="${name}-r${repeat}"
  local log="${results_root}/raw/${tag}.log"
  local id_file="${results_root}/raw/${tag}.nccl-id"
  rm -f -- "$id_file"

  echo "RUN_BEGIN config=${name} repeat=${repeat} algo=${algo:-default} proto=${proto:-default} max_channels=${max_channels:-default}"
  set +e
  NCCL_DEBUG=WARN \
  NCCL_ALGO="$algo" \
  NCCL_PROTO="$proto" \
  NCCL_MIN_NCHANNELS="$min_channels" \
  NCCL_MAX_NCHANNELS="$max_channels" \
  MGT_BENCHMARK_ABS="$(realpath "$benchmark")" \
  MGT_ID_FILE_ABS="$(realpath -m "$id_file")" \
  MGT_LOCAL_ROWS="$local_rows" \
  MGT_GLOBAL_ROWS="$global_rows" \
  MGT_WARMUP="$warmup" \
  MGT_STEPS="$steps" \
  srun --ntasks="$world" --ntasks-per-node="$world" --kill-on-bad-exit=1 \
    bash -lc 'exec "$MGT_BENCHMARK_ABS" "$SLURM_LOCALID" "$SLURM_PROCID" "$SLURM_NTASKS" "$MGT_ID_FILE_ABS" "$MGT_LOCAL_ROWS" "$MGT_GLOBAL_ROWS" "$MGT_WARMUP" "$MGT_STEPS"' \
    >"$log" 2>&1
  local code=$?
  set -e
  printf '%s\n' "$code" > "${results_root}/raw/${tag}.exit-code"
  if [[ "$code" -eq 0 ]]; then
    grep -E '^\{"world":' "$log" | tail -n 1
  else
    echo "RUN_FAILED config=${name} repeat=${repeat} exit_code=${code}"
    tail -n 20 "$log"
  fi
  echo "RUN_END config=${name} repeat=${repeat} exit_code=${code}"
}

for ((repeat = 1; repeat <= repeats; ++repeat)); do
  if (( repeat % 2 == 1 )); then
    for spec in "${configs[@]}"; do run_one "$spec" "$repeat"; done
  else
    for ((index = ${#configs[@]} - 1; index >= 0; --index)); do
      run_one "${configs[$index]}" "$repeat"
    done
  fi
done

# One final baseline makes a large end-of-job drift visible in the summary.
run_one "${configs[0]}" "final"

MGT_NCCL_SWEEP_RESULTS="$results_root" python3 - <<'PY'
import json
import os
import statistics
from pathlib import Path

root = Path(os.environ["MGT_NCCL_SWEEP_RESULTS"])
grouped = {}
failures = []
for log_path in sorted((root / "raw").glob("*.log")):
    tag = log_path.stem
    config = tag.rsplit("-r", 1)[0]
    code_path = log_path.with_suffix(".exit-code")
    code = int(code_path.read_text().strip()) if code_path.exists() else None
    json_rows = []
    for line in log_path.read_text(errors="replace").splitlines():
        if line.startswith('{"world":'):
            json_rows.append(json.loads(line))
    if code != 0 or len(json_rows) != 1:
        failures.append({"tag": tag, "exit_code": code, "json_rows": len(json_rows)})
        continue
    row = json_rows[0]
    grouped.setdefault(config, []).append({
        "tag": tag,
        "step_ms": float(row["step_ms"]),
        "samples_per_second": float(row["samples_per_second"]),
        "stage_input_backward_ms": float(row["stage_input_backward_ms"]),
    })

summary_rows = []
for config, rows in grouped.items():
    step_values = [row["step_ms"] for row in rows]
    summary_rows.append({
        "config": config,
        "successful_runs": len(rows),
        "median_step_ms": statistics.median(step_values),
        "min_step_ms": min(step_values),
        "max_step_ms": max(step_values),
        "median_samples_per_second": statistics.median(row["samples_per_second"] for row in rows),
        "median_input_backward_ms": statistics.median(row["stage_input_backward_ms"] for row in rows),
        "runs": rows,
    })

summary_rows.sort(key=lambda row: row["median_step_ms"])
baseline = next((row for row in summary_rows if row["config"] == "baseline"), None)
for row in summary_rows:
    if baseline:
        row["improvement_vs_baseline_percent"] = (
            baseline["median_step_ms"] / row["median_step_ms"] - 1.0
        ) * 100.0

summary = {
    "format": "mgt_nccl_8xa100_sweep_v1",
    "baseline": baseline,
    "winner": summary_rows[0] if summary_rows else None,
    "candidates": summary_rows,
    "failures": failures,
}
(root / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
print("NCCL_SWEEP_SUMMARY")
print(json.dumps(summary, indent=2, sort_keys=True))

if baseline is None or baseline["successful_runs"] < 3:
    raise SystemExit("baseline gate failed: fewer than three successful runs")
PY

echo "RESULTS=${results_root}"
