#!/usr/bin/env python3
"""Single-container GPU job queue worker.

The worker lives inside one long-running GPU-enabled container. Other agents submit
jobs with docker exec into the same container; this process executes them one at a
time and keeps a cooldown between jobs.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def ensure_dirs(root: Path) -> dict[str, Path]:
    paths = {
        "pending": root / "pending",
        "running": root / "running",
        "done": root / "done",
        "failed": root / "failed",
        "logs": root / "logs",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def write_status(root: Path, status: dict[str, Any]) -> None:
    write_json(root / "status.json", status)


def pick_job(pending: Path) -> Path | None:
    jobs = sorted(pending.glob("*.json"), key=lambda p: p.name)
    return jobs[0] if jobs else None


def finish_job(src: Path, dst_dir: Path, job: dict[str, Any]) -> Path:
    dst = dst_dir / src.name
    write_json(src, job)
    src.replace(dst)
    return dst


def run_job(job_path: Path, paths: dict[str, Path], root: Path, cooldown_sec: float) -> None:
    running_path = paths["running"] / job_path.name
    job_path.replace(running_path)
    job = load_json(running_path)
    job_id = str(job.get("id", running_path.stem))
    label = str(job.get("label", job_id))
    command = job.get("command")
    if not isinstance(command, list) or not command:
        job["status"] = "failed"
        job["error"] = "command must be a non-empty list"
        job["started_at"] = utc_now()
        job["finished_at"] = utc_now()
        finish_job(running_path, paths["failed"], job)
        return

    cwd = str(job.get("cwd") or "/work")
    env = os.environ.copy()
    for item in job.get("env", []) or []:
        if isinstance(item, str) and "=" in item:
            key, value = item.split("=", 1)
            env[key] = value

    log_path = paths["logs"] / f"{job_id}.log"
    job.update({
        "status": "running",
        "started_at": utc_now(),
        "log_path": str(log_path),
    })
    write_json(running_path, job)
    write_status(root, {
        "status": "running",
        "job_id": job_id,
        "label": label,
        "command": command,
        "started_at": job["started_at"],
        "log_path": str(log_path),
    })

    started = time.monotonic()
    with log_path.open("w", encoding="utf-8", errors="replace") as log:
        log.write(f"gpu_queue_job_start id={job_id} label={label} at={job['started_at']}\n")
        log.write(f"cwd={cwd}\n")
        log.write("command=" + json.dumps(command, ensure_ascii=False) + "\n")
        log.flush()
        try:
            proc = subprocess.Popen(
                command,
                cwd=cwd,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
            )
            return_code = proc.wait()
        except Exception as exc:  # keep queue alive after a bad submit
            return_code = 127
            log.write(f"gpu_queue_job_exception {type(exc).__name__}: {exc}\n")
        finished_at = utc_now()
        elapsed = time.monotonic() - started
        log.write(f"gpu_queue_job_finish id={job_id} return_code={return_code} elapsed_sec={elapsed:.3f} at={finished_at}\n")
        log.flush()

    job.update({
        "status": "done" if return_code == 0 else "failed",
        "return_code": return_code,
        "finished_at": finished_at,
        "elapsed_sec": elapsed,
    })
    dst_dir = paths["done"] if return_code == 0 else paths["failed"]
    result_path = finish_job(running_path, dst_dir, job)
    write_status(root, {
        "status": "cooldown",
        "last_job_id": job_id,
        "last_label": label,
        "last_return_code": return_code,
        "last_result_path": str(result_path),
        "cooldown_sec": cooldown_sec,
        "started_at": utc_now(),
    })
    if cooldown_sec > 0:
        time.sleep(cooldown_sec)


def recover_running(paths: dict[str, Path]) -> None:
    for path in sorted(paths["running"].glob("*.json")):
        try:
            job = load_json(path)
        except Exception:
            job = {"id": path.stem}
        job["status"] = "failed"
        job["error"] = "worker restarted while job was running"
        job["finished_at"] = utc_now()
        finish_job(path, paths["failed"], job)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue-root", default=os.environ.get("MGT_GPU_QUEUE_ROOT", "/work/.gpu_queue"))
    parser.add_argument("--poll-sec", type=float, default=float(os.environ.get("MGT_GPU_QUEUE_POLL_SEC", "1")))
    parser.add_argument("--cooldown-sec", type=float, default=float(os.environ.get("MGT_GPU_QUEUE_COOLDOWN_SEC", "10")))
    args = parser.parse_args()

    root = Path(args.queue_root)
    paths = ensure_dirs(root)
    recover_running(paths)
    write_status(root, {"status": "idle", "started_at": utc_now(), "queue_root": str(root)})
    print(f"gpu_queue_worker_ready root={root} cooldown_sec={args.cooldown_sec:g}", flush=True)

    while True:
        job_path = pick_job(paths["pending"])
        if job_path is None:
            write_status(root, {"status": "idle", "updated_at": utc_now(), "queue_root": str(root)})
            time.sleep(args.poll_sec)
            continue
        try:
            run_job(job_path, paths, root, args.cooldown_sec)
        except Exception as exc:
            print(f"gpu_queue_worker_error {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
            time.sleep(args.cooldown_sec)


if __name__ == "__main__":
    raise SystemExit(main())