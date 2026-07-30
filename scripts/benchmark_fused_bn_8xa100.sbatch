#!/usr/bin/env bash
#SBATCH --job-name=mgt-fused-bn
#SBATCH --partition=kaf12
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:8
#SBATCH --mem=128G
#SBATCH --time=00:30:00

set -euo pipefail

expected_sha="${MGT_EXPECTED_SHA:?set MGT_EXPECTED_SHA to the exact revision}"
actual_sha="$(git rev-parse HEAD)"
if [[ "$actual_sha" != "$expected_sha" ]]; then
  echo "error: revision mismatch expected=${expected_sha} actual=${actual_sha}" >&2
  exit 2
fi

source scripts/ensure_cutlass.sh
build_dir="${MGT_BUILD_DIR:-build-a100-fused-bn}"
results_root="${MGT_RESULTS_ROOT:-results/${SLURM_JOB_ID:-manual}-fused-bn}"
mkdir -p "$results_root/raw"

cmake_args=(
  -S native
  -B "$build_dir"
  -DMGT_ENABLE_CUDA=ON
  -DMGT_ENABLE_NCCL=ON
  -DCMAKE_BUILD_TYPE=Release
  -DCMAKE_CUDA_ARCHITECTURES=80
  -DMGT_CUTLASS_ROOT="${CUTLASS_ROOT:-/opt/cutlass}"
)
if [[ -n "${MGT_NCCL_INCLUDE_DIR:-}" ]]; then
  cmake_args+=("-DNCCL_INCLUDE_DIR=${MGT_NCCL_INCLUDE_DIR}")
fi
if [[ -n "${MGT_NCCL_LIB:-}" ]]; then
  cmake_args+=("-DNCCL_LIB=${MGT_NCCL_LIB}")
fi
cmake "${cmake_args[@]}"
cmake --build "$build_dir" --config Release --parallel "${SLURM_CPUS_ON_NODE:-32}" \
  --target test_sync_batch_norm_fused_strided test_mlp_batch_norm_forward \
           test_mlp_batch_norm_full_backward mgt_bn_step_benchmark

MGT_BN_FUSED_EPILOGUE=1 ctest --test-dir "$build_dir" -C Release \
  -R '^(sync_batch_norm_fused_strided|mlp_batch_norm_forward|mlp_batch_norm_full_backward)$' \
  --output-on-failure

world="${SLURM_NTASKS:-8}"
global_rows="${MGT_GLOBAL_ROWS:-100000}"
local_rows=$((global_rows / world))
warmup="${MGT_WARMUP:-20}"
steps="${MGT_STEPS:-100}"
repeats="${MGT_REPEATS:-4}"
benchmark="$(realpath "$build_dir/mgt_bn_step_benchmark")"

run_one() {
  local mode="$1"
  local repeat="$2"
  local tag="${mode}-r${repeat}"
  local id_file="$(realpath -m "$results_root/raw/${tag}.nccl-id")"
  local log="$results_root/raw/${tag}.log"
  rm -f -- "$id_file"
  echo "RUN_BEGIN mode=${mode} repeat=${repeat}"
  MGT_BN_INPUT_GRAD_KERNEL=strict \
  MGT_BN_FUSED_EPILOGUE="$([[ "$mode" == fused ]] && echo 1 || echo 0)" \
  MGT_BENCHMARK="$benchmark" MGT_ID_FILE="$id_file" \
  MGT_LOCAL_ROWS="$local_rows" MGT_GLOBAL_ROWS="$global_rows" \
  MGT_WARMUP="$warmup" MGT_STEPS="$steps" \
  srun --ntasks="$world" --ntasks-per-node="$world" --kill-on-bad-exit=1 \
    bash -lc 'exec "$MGT_BENCHMARK" "$SLURM_LOCALID" "$SLURM_PROCID" "$SLURM_NTASKS" "$MGT_ID_FILE" "$MGT_LOCAL_ROWS" "$MGT_GLOBAL_ROWS" "$MGT_WARMUP" "$MGT_STEPS"' \
    >"$log" 2>&1
  grep -E '^\{"world":' "$log" | tail -n 1
  echo "RUN_END mode=${mode} repeat=${repeat}"
}

for ((repeat = 1; repeat <= repeats; ++repeat)); do
  if (( repeat % 2 == 1 )); then
    run_one baseline "$repeat"
    run_one fused "$repeat"
  else
    run_one fused "$repeat"
    run_one baseline "$repeat"
  fi
done

MGT_FUSED_BN_RESULTS="$results_root" python3 - <<'PY'
import json
import os
import statistics
from pathlib import Path

root = Path(os.environ["MGT_FUSED_BN_RESULTS"])
rows = {"baseline": [], "fused": []}
for path in sorted((root / "raw").glob("*.log")):
    mode = path.stem.split("-r", 1)[0]
    payloads = [
        json.loads(line) for line in path.read_text(errors="replace").splitlines()
        if line.startswith('{"world":')
    ]
    if len(payloads) != 1:
        raise SystemExit(f"expected one JSON row in {path}, got {len(payloads)}")
    rows[mode].append(payloads[0])

if len(rows["baseline"]) != 4 or len(rows["fused"]) != 4:
    raise SystemExit(
        f"expected 4+4 rows, got {len(rows['baseline'])}+{len(rows['fused'])}")

baseline_ms = statistics.median(float(row["step_ms"]) for row in rows["baseline"])
fused_ms = statistics.median(float(row["step_ms"]) for row in rows["fused"])
summary = {
    "format": "mgt_fused_bn_8xa100_ab_v1",
    "baseline_step_ms": baseline_ms,
    "fused_step_ms": fused_ms,
    "speedup": baseline_ms / fused_ms,
    "improvement_percent": (baseline_ms / fused_ms - 1.0) * 100.0,
    "baseline_samples_per_second": statistics.median(
        float(row["samples_per_second"]) for row in rows["baseline"]),
    "fused_samples_per_second": statistics.median(
        float(row["samples_per_second"]) for row in rows["fused"]),
}
(root / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
print("FUSED_BN_SUMMARY")
print(json.dumps(summary, indent=2, sort_keys=True))
PY

echo "RESULTS=${results_root}"
