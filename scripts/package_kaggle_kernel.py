#!/usr/bin/env python3
"""Build a self-contained Kaggle script kernel package.

Kaggle script kernels execute only the declared code_file. This packer embeds the
project sources into run_2xt4.py as a base64 zip payload, then the script extracts
that payload on Kaggle and runs kaggle/kernel/run_2xt4.sh.
"""

from __future__ import annotations

import argparse
import base64
import json
import zipfile
from pathlib import Path


def add_path(zf: zipfile.ZipFile, root: Path, path: Path) -> None:
    if path.is_dir():
        for child in sorted(path.rglob("*")):
            if child.is_file():
                zf.write(child, child.relative_to(root).as_posix())
    else:
        zf.write(path, path.relative_to(root).as_posix())


def build_payload(root: Path, output: Path) -> bytes:
    include = [root / "native", root / "crates", root / "kaggle", root / "Cargo.toml", root / "Cargo.lock"]
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in include:
            add_path(zf, root, path)
    return output.read_bytes()


def write_kernel_script(output_dir: Path, payload: bytes, full_model: bool) -> None:
    encoded = base64.b64encode(payload).decode("ascii")
    full_model_default = "1" if full_model else "0"
    script = f'''import base64
import os
import subprocess
import sys
import zipfile
from pathlib import Path

PAYLOAD_B64 = {encoded!r}

root = Path.cwd()
payload = root / "payload.zip"
payload.write_bytes(base64.b64decode(PAYLOAD_B64))
with zipfile.ZipFile(payload, "r") as zf:
    zf.extractall(root)
runner = root / "kaggle" / "kernel" / "run_2xt4.sh"
print("runner_exists=", runner.exists(), "cwd=", root, flush=True)
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0,1")
os.environ.setdefault("MGT_CUDA_ARCH", "75")
os.environ.setdefault("MGT_FULL_MODEL", {full_model_default!r})
result = subprocess.run(["bash", str(runner)])
sys.exit(result.returncode)
'''
    (output_dir / "run_2xt4.py").write_text(script, encoding="utf-8")


def write_metadata(output_dir: Path, kernel_id: str, title: str) -> None:
    metadata = {
        "id": kernel_id,
        "title": title,
        "code_file": "run_2xt4.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": True,
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }
    (output_dir / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="test_results/mgt_kaggle_kernel")
    parser.add_argument("--kernel-id", default="trydotatwo/native-multigpu-mlp-trainer")
    parser.add_argument("--title", default="native-multigpu-mlp-trainer")
    parser.add_argument("--full-model", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    output_dir = (root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    payload_path = output_dir / "payload.zip"
    payload = build_payload(root, payload_path)
    write_kernel_script(output_dir, payload, args.full_model)
    write_metadata(output_dir, args.kernel_id, args.title)
    print(output_dir)
    print(f"payload_bytes={len(payload)}")


if __name__ == "__main__":
    main()