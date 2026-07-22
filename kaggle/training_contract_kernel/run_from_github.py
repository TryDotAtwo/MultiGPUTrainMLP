import os
import subprocess
import sys
from pathlib import Path

repo = "https://github.com/TryDotAtwo/MultiGPUTrainMLP.git"
ref = "codex-native-trainer-implementation"
expected_rev = "e049b46"
work = Path("/tmp/MultiGPUTrainMLP")
out = Path("/kaggle/working/training-contract-2xt4")
subprocess.run(["rm", "-rf", str(work)], check=True)
subprocess.run(["git", "clone", "--depth", "10", "--branch", ref, repo, str(work)], check=True)
rev = subprocess.check_output(["git", "-C", str(work), "rev-parse", "--short", "HEAD"], text=True).strip()
print(f"git_rev={rev}", flush=True)
subprocess.run(
    ["git", "-C", str(work), "merge-base", "--is-ancestor", expected_rev, "HEAD"], check=True)
target = Path("/tmp/p888-target.bin")
target.write_bytes(bytes(range(72)))
env = os.environ.copy()
env.update({
    "CUDA_VISIBLE_DEVICES": "0,1",
    "MGT_GROUP_JSON": str(work / "native/production_inputs/p888.json"),
    "MGT_TARGET_BIN": str(target),
    "MGT_CONTRACT_ROOT": str(out),
    "MGT_BUILD_DIR": "/tmp/build-kaggle-2xt4",
    "MGT_WAIT_GPU_IDLE": "1",
    "MGT_EVAL_SAMPLES": "32",
    "MGT_EVAL_PERIOD_STEPS": "2",
})
result = subprocess.run(
    ["bash", "kaggle/kernel/check_training_contract_2xt4.sh"],
    cwd=str(work), env=env)
Path("/kaggle/working/runner_return_code.txt").write_text(
    str(result.returncode) + "\n", encoding="utf-8")
sys.exit(result.returncode)
