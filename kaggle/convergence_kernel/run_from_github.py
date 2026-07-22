import json
import math
import os
import subprocess
import sys
from pathlib import Path

repo = "https://github.com/TryDotAtwo/MultiGPUTrainMLP.git"
ref = "codex-native-trainer-implementation"
required_rev = "92bb7fa"
work = Path("/tmp/MultiGPUTrainMLP")
out = Path("/kaggle/working/production-convergence-2xt4")
subprocess.run(["rm", "-rf", str(work)], check=True)
subprocess.run(["git", "clone", "--depth", "50", "--branch", ref, repo, str(work)], check=True)
rev = subprocess.check_output(["git", "-C", str(work), "rev-parse", "--short", "HEAD"], text=True).strip()
print(f"git_rev={rev}", flush=True)
subprocess.run(["git", "-C", str(work), "merge-base", "--is-ancestor", required_rev, "HEAD"], check=True)
target = Path("/tmp/p888-target.bin")
target.write_bytes(bytes(range(72)))
env = os.environ.copy()
env.update({
    "CUDA_VISIBLE_DEVICES": "0,1",
    "MGT_GROUP_JSON": str(work / "native/production_inputs/p888.json"),
    "MGT_TARGET_BIN": str(target),
    "MGT_RUN_ROOT": str(out),
    "MGT_BUILD_DIR": "/tmp/build-kaggle-2xt4",
    "MGT_CUDA_ARCH": "75",
    "MGT_FORCE_WORLD_SIZE": "2",
    "MGT_PERF_RUN": "1",
    "MGT_FULL_MODEL": "1",
    "MGT_STEPS": "1000",
    "MGT_BATCH_SIZE": "57344",
    "MGT_INPUT_GRAD_POSITION_TILE": "48",
    "MGT_LT_WORKSPACE_BYTES": "16777216",
    "MGT_ALLREDUCE_BUCKET_BYTES": "2097152",
    "MGT_CUTLASS_HALF_GEMM_KINDS": "input_embedding_grad,forward",
    "MGT_LT_AUTOTUNE": "1",
    "MGT_WRITE_ARTIFACTS": "0",
    "MGT_EVAL_SAMPLES": "4096",
    "MGT_EVAL_PERIOD_STEPS": "100",
    "MGT_WAIT_GPU_IDLE": "1",
})
result = subprocess.run(["bash", "kaggle/kernel/run_ranks_2xt4.sh"], cwd=str(work), env=env)
summary = {"runner_return_code": result.returncode, "git_rev": rev}
if result.returncode == 0:
    rank_rows = []
    for rank in (0, 1):
        path = out / f"rank{rank}" / "evaluation.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        if len(rows) < 11 or any(not math.isfinite(float(row["mse"])) for row in rows):
            raise RuntimeError(f"invalid evaluation series for rank {rank}")
        rank_rows.append(rows)
    series0 = [(int(row["completed_steps"]), float(row["mse"])) for row in rank_rows[0]]
    series1 = [(int(row["completed_steps"]), float(row["mse"])) for row in rank_rows[1]]
    if series0 != series1:
        raise RuntimeError("evaluation differs between ranks")
    baseline, final = series0[0][1], series0[-1][1]
    if final >= baseline:
        raise RuntimeError(f"held-out MSE did not improve: {baseline} -> {final}")
    summary.update({
        "status": "ok", "series": series0, "baseline_mse": baseline,
        "final_mse": final, "relative_improvement": (baseline - final) / baseline,
    })
else:
    summary["status"] = "runner_failed"
Path("/kaggle/working/convergence_summary.json").write_text(
    json.dumps(summary, indent=2) + "\n", encoding="utf-8")
print("convergence_summary=" + json.dumps(summary, sort_keys=True), flush=True)
sys.exit(result.returncode)
