#!/usr/bin/env python3
"""Wait until visible NVIDIA GPUs look idle enough for a benchmark run."""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
import time
from dataclasses import dataclass


@dataclass
class GpuState:
    index: str
    name: str
    util_percent: int
    memory_used_mb: int


def _run_nvidia_smi(args: list[str]) -> str | None:
    try:
        return subprocess.check_output(["nvidia-smi", *args], text=True, stderr=subprocess.DEVNULL)
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None


def _parse_visible_devices(value: str | None) -> set[str] | None:
    if not value or value.strip() in {"", "all"}:
        return None
    devices = {part.strip() for part in value.split(",") if part.strip()}
    return devices or None


def query_gpus() -> list[GpuState] | None:
    output = _run_nvidia_smi([
        "--query-gpu=index,name,utilization.gpu,memory.used",
        "--format=csv,noheader,nounits",
    ])
    if output is None:
        return None
    rows: list[GpuState] = []
    for row in csv.reader(output.splitlines()):
        if len(row) < 4:
            continue
        rows.append(GpuState(
            index=row[0].strip(),
            name=row[1].strip(),
            util_percent=int(row[2].strip()),
            memory_used_mb=int(row[3].strip()),
        ))
    return rows


def is_idle(gpus: list[GpuState], visible: set[str] | None, max_util: int, max_memory_mb: int) -> tuple[bool, str]:
    selected = [gpu for gpu in gpus if visible is None or gpu.index in visible]
    if not selected:
        selected = gpus
    busy = [
        gpu for gpu in selected
        if gpu.util_percent > max_util or gpu.memory_used_mb > max_memory_mb
    ]
    details = "; ".join(
        f"gpu{gpu.index} {gpu.name}: util={gpu.util_percent}% mem={gpu.memory_used_mb}MiB"
        for gpu in selected
    )
    return not busy, details


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--devices", default=os.environ.get("CUDA_VISIBLE_DEVICES", ""))
    parser.add_argument("--max-util", type=int, default=int(os.environ.get("MGT_GPU_IDLE_MAX_UTIL", "15")))
    parser.add_argument("--max-memory-mb", type=int, default=int(os.environ.get("MGT_GPU_IDLE_MAX_MEMORY_MB", "1536")))
    parser.add_argument("--timeout-sec", type=float, default=float(os.environ.get("MGT_GPU_IDLE_TIMEOUT_SEC", "900")))
    parser.add_argument("--poll-sec", type=float, default=float(os.environ.get("MGT_GPU_IDLE_POLL_SEC", "5")))
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()

    visible = _parse_visible_devices(args.devices)
    start = time.monotonic()
    last_details = ""
    while True:
        gpus = query_gpus()
        if gpus is None:
            print("gpu_idle_check=skipped reason=nvidia-smi-unavailable", flush=True)
            return 0
        idle, details = is_idle(gpus, visible, args.max_util, args.max_memory_mb)
        last_details = details
        if idle:
            print(f"gpu_idle_check=ok {details}", flush=True)
            return 0
        elapsed = time.monotonic() - start
        print(f"gpu_idle_check=busy elapsed_sec={elapsed:.1f} {details}", flush=True)
        if args.once:
            return 1
        if elapsed >= args.timeout_sec:
            print(f"gpu_idle_check=timeout timeout_sec={args.timeout_sec} last='{last_details}'", file=sys.stderr, flush=True)
            return 2
        time.sleep(args.poll_sec)


if __name__ == "__main__":
    raise SystemExit(main())