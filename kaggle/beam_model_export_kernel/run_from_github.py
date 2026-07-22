import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

repo = "https://github.com/TryDotAtwo/MultiGPUTrainMLP.git"
ref = "codex-native-trainer-implementation"
required_rev = "f8b5ec4"
work = Path("/tmp/MultiGPUTrainMLP")
run_root = Path("/tmp/beam-model-export-2xt4")
out = Path("/kaggle/working/beam-model-export-2xt4")
for path in (work, run_root, out):
    shutil.rmtree(path, ignore_errors=True)
out.mkdir(parents=True)

subprocess.run(["git", "clone", "--depth", "50", "--branch", ref, repo, str(work)], check=True)
rev = subprocess.check_output(["git", "-C", str(work), "rev-parse", "--short", "HEAD"], text=True).strip()
subprocess.run(["git", "-C", str(work), "merge-base", "--is-ancestor", required_rev, "HEAD"], check=True)
print(f"git_rev={rev}", flush=True)

target = Path("/tmp/p888-target.bin")
target.write_bytes(bytes(range(72)))
env = os.environ.copy()
env.update({
    "CUDA_VISIBLE_DEVICES": "0,1",
    "MGT_GROUP_JSON": str(work / "native/production_inputs/p888.json"),
    "MGT_TARGET_BIN": str(target),
    "MGT_RUN_ROOT": str(run_root),
    "MGT_BUILD_DIR": "/tmp/build-kaggle-2xt4",
    "MGT_CUDA_ARCH": "75",
    "MGT_FORCE_WORLD_SIZE": "2",
    "MGT_PERF_RUN": "1",
    "MGT_FULL_MODEL": "1",
    "MGT_STEPS": "600",
    "MGT_BATCH_SIZE": "57344",
    "MGT_INPUT_GRAD_POSITION_TILE": "48",
    "MGT_LT_WORKSPACE_BYTES": "16777216",
    "MGT_ALLREDUCE_BUCKET_BYTES": "2097152",
    "MGT_CUTLASS_HALF_GEMM_KINDS": "input_embedding_grad,forward",
    "MGT_LT_AUTOTUNE": "1",
    "MGT_WRITE_ARTIFACTS": "1",
    "MGT_EVAL_SAMPLES": "4096",
    "MGT_EVAL_PERIOD_STEPS": "100",
    "MGT_WAIT_GPU_IDLE": "1",
})
result = subprocess.run(["bash", "kaggle/kernel/run_ranks_2xt4.sh"], cwd=str(work), env=env)
summary = {"runner_return_code": result.returncode, "git_rev": rev, "steps": 600}
if result.returncode == 0:
    rank0 = run_root / "rank0"
    shutil.copytree(rank0 / "weights", out / "weights")
    shutil.copy2(rank0 / "evaluation.jsonl", out / "evaluation.jsonl")
    shutil.copy2(run_root / "rank0.stdout", out / "rank0.stdout")
    shutil.copy2(run_root / "rank1.stdout", out / "rank1.stdout")
    summary["status"] = "ok"
else:
    summary["status"] = "runner_failed"
(out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
print("beam_export_summary=" + json.dumps(summary, sort_keys=True), flush=True)
sys.exit(result.returncode)
