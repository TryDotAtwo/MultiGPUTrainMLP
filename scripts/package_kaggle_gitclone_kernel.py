#!/usr/bin/env python3
"""Create a tiny Kaggle kernel package that runs this repo from GitHub.

The package intentionally does not embed local sources. Kaggle clones the selected
Git ref into /tmp, keeps dependencies/build trees outside /kaggle/working, and
exports only compact result files.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


RUNNER_TEMPLATE = r'''import os
import shutil
import subprocess
import sys
from pathlib import Path

repo = os.environ.get("MGT_GITHUB_REPO", "{repo}")
ref = os.environ.get("MGT_GITHUB_REF", "{ref}")
entry = os.environ.get("MGT_GITHUB_ENTRY", "{entry}")
root = Path("/kaggle/working")
work = Path(os.environ.get("MGT_WORKDIR", "/tmp/MultiGPUTrainMLP"))
out = root / os.environ.get("MGT_OUTPUT_SUBDIR", "{output_subdir}")

if work.exists():
    shutil.rmtree(work)
if out.exists():
    shutil.rmtree(out)
out.mkdir(parents=True, exist_ok=True)

print(f"clone_repo={{repo}}", flush=True)
print(f"clone_ref={{ref}}", flush=True)
subprocess.run(["git", "clone", "--depth", "1", "--branch", ref, repo, str(work)], check=True)
rev = subprocess.check_output(["git", "-C", str(work), "rev-parse", "HEAD"], text=True).strip()
(out / "git_rev.txt").write_text(rev + "\n", encoding="utf-8")
print(f"git_rev={{rev}}", flush=True)

env = os.environ.copy()
env.setdefault("CUDA_VISIBLE_DEVICES", "0,1")
env.setdefault("MGT_CUDA_ARCH", "75")
env.setdefault("MGT_DEPS_DIR", "/tmp/mgt_deps")
env.setdefault("MGT_WRITE_ARTIFACTS", "0")
env.setdefault("MGT_INPUT_GRAD_FP16", "1")
env.setdefault("MGT_INPUT_GRAD_SPARSE", "0")
env.setdefault("MGT_LINEAR_FP16", "1")
env.setdefault("MGT_OUTPUT_DIM", "1")
extra_env_defaults = {extra_env_defaults}
for key, value in extra_env_defaults.items():
    env.setdefault(key, value)
if entry.endswith("run_sweep_2xt4.sh"):
    env.setdefault("MGT_SWEEP_ROOT", "/tmp/mgt_2xt4_sweep")
    env.setdefault("MGT_SWEEP_STEPS", "8")
    env.setdefault("MGT_SWEEP_RUN_CTEST", "1")
else:
    env.setdefault("MGT_PERF_RUN", "1")
    env.setdefault("MGT_FULL_MODEL", "1")
    env.setdefault("MGT_STEPS", "8")
    env.setdefault("MGT_BATCH_SIZE", "53248")
    env.setdefault("MGT_INPUT_GRAD_POSITION_TILE", "48")
    env.setdefault("MGT_OVERLAP_ALLREDUCE", "1")
    env.setdefault("MGT_BACKWARD_PROFILE", "0")
    env.setdefault("MGT_LT_WORKSPACE_BYTES", "16777216")
    env.setdefault("MGT_CLEAN_BUILD_OUTPUT", "1")

runner = work / entry
print(f"runner={{runner}}", flush=True)
result = subprocess.run(["bash", str(runner)], cwd=str(work), env=env)

copy_roots = []
sweep_root = Path(env.get("MGT_SWEEP_ROOT", "/tmp/mgt_2xt4_sweep"))
if sweep_root.exists():
    copy_roots.append((sweep_root, out / "sweep"))
single_root = work / "runs" / "kaggle-2xt4"
if single_root.exists():
    copy_roots.append((single_root, out / "single"))
for src_root, dst_root in copy_roots:
    if dst_root.exists():
        shutil.rmtree(dst_root)
    shutil.copytree(src_root, dst_root)

(out / "runner_return_code.txt").write_text(str(result.returncode) + "\n", encoding="utf-8")
print(f"runner_return_code={{result.returncode}}", flush=True)
sys.exit(result.returncode)
'''


def write_metadata(output_dir: Path, kernel_id: str, title: str, code_file: str, machine_shape: str) -> None:
    metadata = {
        "id": kernel_id,
        "title": title,
        "code_file": code_file,
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": True,
        "machine_shape": machine_shape,
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }
    (output_dir / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="test_results/mgt_kaggle_gitclone_sweep")
    parser.add_argument("--kernel-id", default="trydotatwo/native-multigpu-mlp-trainer-gitclone-sweep")
    parser.add_argument("--title", default="native-multigpu-mlp-trainer-gitclone-sweep")
    parser.add_argument("--repo", default="https://github.com/TryDotAtwo/MultiGPUTrainMLP.git")
    parser.add_argument("--ref", default="main")
    parser.add_argument("--entry", default="kaggle/kernel/run_sweep_2xt4.sh")
    parser.add_argument("--output-subdir", default="mgt_results")
    parser.add_argument("--machine-shape", default="NvidiaTeslaT4")
    parser.add_argument("--set-env", action="append", default=[], metavar="KEY=VALUE", help="Embed an environment default in the generated Kaggle runner. Use \\n for multiline values.")
    args = parser.parse_args()

    extra_env_defaults: dict[str, str] = {}
    for item in args.set_env:
        if "=" not in item:
            raise SystemExit(f"--set-env expects KEY=VALUE, got {item!r}")
        key, value = item.split("=", 1)
        if not key:
            raise SystemExit("--set-env key must not be empty")
        extra_env_defaults[key] = value.replace("\\n", "\n")

    root = Path(__file__).resolve().parents[1]
    output_dir = (root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    code_file = "run_from_github.py"
    runner = RUNNER_TEMPLATE.format(
        repo=args.repo,
        ref=args.ref,
        entry=args.entry,
        output_subdir=args.output_subdir,
        extra_env_defaults=json.dumps(extra_env_defaults, sort_keys=True),
    )
    (output_dir / code_file).write_text(runner, encoding="utf-8")
    write_metadata(output_dir, args.kernel_id, args.title, code_file, args.machine_shape)
    print(output_dir)


if __name__ == "__main__":
    main()