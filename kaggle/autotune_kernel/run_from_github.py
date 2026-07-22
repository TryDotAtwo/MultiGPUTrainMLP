import os
import subprocess
import sys
from pathlib import Path

repo = "https://github.com/TryDotAtwo/MultiGPUTrainMLP.git"
ref = "codex-native-trainer-implementation"
required_rev = "b65c8c2"
work = Path("/tmp/MultiGPUTrainMLP")
subprocess.run(["rm", "-rf", str(work)], check=True)
subprocess.run(["git", "clone", "--depth", "10", "--branch", ref, repo, str(work)], check=True)
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
    "MGT_AUTOTUNE_ROOT": "/kaggle/working/autotune-2xt4",
    "MGT_AUTOTUNE_CACHE_ROOT": "/kaggle/working/autotune-cache",
    "MGT_BUILD_DIR": "/tmp/build-kaggle-2xt4",
    "MGT_CUDA_ARCH": "75",
    "MGT_WAIT_GPU_IDLE": "1",
})
result = subprocess.run(["bash", "kaggle/kernel/run_autotune_2xt4.sh"], cwd=str(work), env=env)
Path("/kaggle/working/runner_return_code.txt").write_text(str(result.returncode) + "\n", encoding="utf-8")
sys.exit(result.returncode)
