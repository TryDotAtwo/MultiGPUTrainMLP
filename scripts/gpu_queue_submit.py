#!/usr/bin/env python3
"""Submit commands to the shared in-container GPU queue."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sanitize(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value)[:80] or "job"


def write_json_atomic(path: Path, data: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def load_result(root: Path, job_id: str) -> tuple[str, dict[str, Any] | None]:
    for state in ("done", "failed", "running", "pending"):
        matches = list((root / state).glob(f"*_{job_id}_*.json"))
        if matches:
            try:
                return state, json.loads(matches[0].read_text(encoding="utf-8"))
            except Exception:
                return state, None
    return "missing", None


def tail_log(path: str, lines: int) -> None:
    if not path:
        return
    log_path = Path(path)
    if not log_path.exists():
        return
    content = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    for line in content[-lines:]:
        print(line)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue-root", default=os.environ.get("MGT_GPU_QUEUE_ROOT", "/work/.gpu_queue"))
    parser.add_argument("--label", default=os.environ.get("MGT_GPU_QUEUE_LABEL", "gpu-job"))
    parser.add_argument("--cwd", default=os.environ.get("MGT_GPU_QUEUE_CWD", "/work"))
    parser.add_argument("--env", action="append", default=[])
    parser.add_argument("--wait", action="store_true")
    parser.add_argument("--timeout-sec", type=float, default=float(os.environ.get("MGT_GPU_QUEUE_WAIT_TIMEOUT_SEC", "86400")))
    parser.add_argument("--poll-sec", type=float, default=float(os.environ.get("MGT_GPU_QUEUE_WAIT_POLL_SEC", "2")))
    parser.add_argument("--tail", type=int, default=40)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    command = args.command
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        parser.error("command is required after --")

    root = Path(args.queue_root)
    pending = root / "pending"
    pending.mkdir(parents=True, exist_ok=True)
    for dirname in ("running", "done", "failed", "logs"):
        (root / dirname).mkdir(parents=True, exist_ok=True)

    job_id = uuid.uuid4().hex[:12]
    label = sanitize(args.label)
    filename = f"{utc_stamp()}_{job_id}_{label}.json"
    job = {
        "id": job_id,
        "label": args.label,
        "status": "pending",
        "submitted_at": utc_now(),
        "cwd": args.cwd,
        "env": args.env,
        "command": command,
    }
    write_json_atomic(pending / filename, job)
    print(f"gpu_queue_submitted id={job_id} label={args.label} file={pending / filename}", flush=True)

    if not args.wait:
        return 0

    start = time.monotonic()
    last_state = ""
    while True:
        state, result = load_result(root, job_id)
        if state != last_state:
            print(f"gpu_queue_state id={job_id} state={state}", flush=True)
            last_state = state
        if state in {"done", "failed"}:
            result = result or {}
            log_path = str(result.get("log_path", ""))
            if args.tail > 0:
                tail_log(log_path, args.tail)
            return int(result.get("return_code", 1 if state == "failed" else 0))
        if time.monotonic() - start >= args.timeout_sec:
            print(f"gpu_queue_wait_timeout id={job_id} state={state}", file=sys.stderr, flush=True)
            return 124
        time.sleep(args.poll_sec)


if __name__ == "__main__":
    raise SystemExit(main())