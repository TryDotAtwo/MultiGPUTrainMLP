#!/usr/bin/env bash
set -euo pipefail

: "${MGT_GROUP_JSON:?real group JSON is required}"
: "${MGT_TARGET_BIN:?real target binary is required}"
test -s "$MGT_GROUP_JSON"
test -s "$MGT_TARGET_BIN"

if [ "$(nvidia-smi -L | wc -l)" -ne 2 ] || \
   [ "$(nvidia-smi --query-gpu=name --format=csv,noheader | grep -c 'Tesla T4')" -ne 2 ]; then
  echo "autotune requires exactly 2 Tesla T4 GPUs" >&2
  exit 2
fi

root="${MGT_AUTOTUNE_ROOT:-runs/autotune-2xt4}"
cache_root="${MGT_AUTOTUNE_CACHE_ROOT:-runs/autotune-cache}"
build_dir="${MGT_BUILD_DIR:-build-kaggle-2xt4}"
mkdir -p "$root" "$cache_root"
context="$root/fingerprint-context.json"
python3 - "$context" <<'PY'
import json, os, subprocess, sys

def output(args):
    return subprocess.check_output(args, text=True).strip().splitlines()
context = {
    "schema": 1,
    "gpu_names": output(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"]),
    "compute_caps": output(["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"]),
    "driver": output(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"])[0],
    "cuda_arch": os.environ.get("MGT_CUDA_ARCH", "75"),
    "git_rev": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    "workload": {key: os.environ.get(key, default) for key, default in {
        "MGT_HD1": "2556", "MGT_HD2": "218", "MGT_NRD": "16",
        "MGT_OUTPUT_DIM": "1", "MGT_K_MIN": "1", "MGT_K_MAX": "29",
        "MGT_INPUT_GRAD_FP16": "1", "MGT_LINEAR_FP16": "1",
        "MGT_OVERLAP_ALLREDUCE": "1", "MGT_CUTLASS_HALF_GEMM_KINDS": "input_embedding_grad,forward",
    }.items()},
}
with open(sys.argv[1], "w", encoding="utf-8") as handle:
    json.dump(context, handle, indent=2, sort_keys=True)
PY
fingerprint=$(PYTHONPATH=scripts python3 - "$context" <<'PY'
import json, sys
from autotune_2xt4 import build_fingerprint
print(build_fingerprint(json.load(open(sys.argv[1], encoding="utf-8"))))
PY
)
cache="$cache_root/${fingerprint}.json"
selected_env="$root/selected.env"

decision=$(PYTHONPATH=scripts python3 - "$cache" "$fingerprint" <<'PY'
import json, pathlib, sys
from autotune_2xt4 import reuse_decision
path = pathlib.Path(sys.argv[1])
cache = json.loads(path.read_text()) if path.is_file() else {}
print(reuse_decision(cache, sys.argv[2], None))
PY
)

run_stage() {
  local stage="$1"
  local configs="$2"
  local run_ctest="$3"
  MGT_SWEEP_ROOT="$root/$stage" \
  MGT_SWEEP_CONFIGS="$configs" \
  MGT_SWEEP_RUN_CTEST="$run_ctest" \
  MGT_SWEEP_STEPS="${MGT_AUTOTUNE_STEPS:-12}" \
  MGT_BUILD_DIR="$build_dir" \
  MGT_WAIT_GPU_IDLE="${MGT_WAIT_GPU_IDLE:-1}" \
  bash kaggle/kernel/run_sweep_2xt4.sh
}

select_field() {
  PYTHONPATH=scripts python3 - "$1" "$2" <<'PY'
import json, sys
from autotune_2xt4 import select_stable_candidate
summary = json.load(open(sys.argv[1], encoding="utf-8"))
row = select_stable_candidate(summary["rows"])
print(row["config"][sys.argv[2]])
PY
}

full_tune() {
  run_stage tile $'tile36 53248 36 16777216 4194304 0 1 0 input_embedding_grad,forward 1\ntile48 53248 48 16777216 4194304 0 1 0 input_embedding_grad,forward 1\ntile56 53248 56 16777216 4194304 0 1 0 input_embedding_grad,forward 1\ntile64 53248 64 16777216 4194304 0 1 0 input_embedding_grad,forward 1' 1
  tile=$(select_field "$root/tile/sweep_summary.json" input_grad_position_tile)
  run_stage batch "batch49152 49152 $tile 16777216 4194304 0 1 0 input_embedding_grad,forward 1
batch53248 53248 $tile 16777216 4194304 0 1 0 input_embedding_grad,forward 1
batch57344 57344 $tile 16777216 4194304 0 1 0 input_embedding_grad,forward 1
batch61440 61440 $tile 16777216 4194304 0 1 0 input_embedding_grad,forward 1" 0
  batch=$(select_field "$root/batch/sweep_summary.json" batch_size)
  run_stage transport "ws0_b4m $batch $tile 0 4194304 0 1 0 input_embedding_grad,forward 1
ws16m_b2m $batch $tile 16777216 2097152 0 1 0 input_embedding_grad,forward 1
ws16m_b4m $batch $tile 16777216 4194304 0 1 0 input_embedding_grad,forward 1
ws16m_b8m $batch $tile 16777216 8388608 0 1 0 input_embedding_grad,forward 1
ws32m_b4m $batch $tile 33554432 4194304 0 1 0 input_embedding_grad,forward 1" 0
  PYTHONPATH=scripts python3 - "$fingerprint" "$context" "$cache" "$selected_env" \
      "$root/tile/sweep_summary.json" "$root/batch/sweep_summary.json" "$root/transport/sweep_summary.json" <<'PY'
import json, os, sys, tempfile
from autotune_2xt4 import CACHE_SCHEMA, render_selected_env, select_final_stage_candidate
fingerprint, context_path, cache_path, env_path, *summaries = sys.argv[1:]
stages = [json.load(open(path, encoding="utf-8"))["rows"] for path in summaries]
winner = select_final_stage_candidate(stages)
payload = {
    "schema": CACHE_SCHEMA,
    "fingerprint": fingerprint,
    "context": json.load(open(context_path, encoding="utf-8")),
    "baseline_throughput_states_s": winner["avg_throughput_states_s"],
    "conservative_throughput_states_s": winner["min_throughput_states_s"],
    "selected": winner,
}
for path, text in ((cache_path, json.dumps(payload, indent=2, sort_keys=True) + "\n"),
                   (env_path, render_selected_env(winner))):
    temp = path + ".tmp"
    with open(temp, "w", encoding="utf-8") as handle:
        handle.write(text)
    os.replace(temp, path)
print("autotune_selected=" + winner["config_id"])
PY
}

if [ "$decision" = "quick_check" ]; then
  PYTHONPATH=scripts python3 - "$cache" "$selected_env" <<'PY'
import json, sys
from autotune_2xt4 import render_selected_env
cache = json.load(open(sys.argv[1], encoding="utf-8"))
open(sys.argv[2], "w", encoding="utf-8").write(render_selected_env(cache["selected"]))
PY
  # shellcheck disable=SC1090
  source "$selected_env"
  quick_config="quick $MGT_BATCH_SIZE $MGT_INPUT_GRAD_POSITION_TILE $MGT_LT_WORKSPACE_BYTES $MGT_ALLREDUCE_BUCKET_BYTES 0 1 $MGT_INPUT_GRAD_SPARSE $MGT_CUTLASS_HALF_GEMM_KINDS $MGT_LT_AUTOTUNE"
  run_stage quick "$quick_config" 0
  measured=$(select_field "$root/quick/sweep_summary.json" avg_throughput_states_s 2>/dev/null || \
    python3 -c "import json; print(json.load(open('$root/quick/sweep_summary.json'))['best']['avg_throughput_states_s'])")
  decision=$(PYTHONPATH=scripts python3 - "$cache" "$fingerprint" "$measured" <<'PY'
import json, sys
from autotune_2xt4 import reuse_decision
print(reuse_decision(json.load(open(sys.argv[1], encoding="utf-8")), sys.argv[2], float(sys.argv[3])))
PY
)
fi

if [ "$decision" != "reuse" ]; then
  full_tune
fi

echo "autotune_2xt4_ok fingerprint=$fingerprint decision=$decision cache=$cache selected_env=$selected_env"
