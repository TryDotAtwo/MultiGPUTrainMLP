import json
import os
import statistics
import subprocess
import sys
from pathlib import Path

REPO = "https://github.com/TryDotAtwo/MultiGPUTrainMLP.git"
REF = "codex/sparse-input-grad-v3"
REV = "2a555ccdf6c5f41e612cdb1cfaafd6eb87e535b9"
WORK = Path("/tmp/MultiGPUTrainMLP")
BUILD = Path("/tmp/build-input-grad-sm75")
OUT = Path("/kaggle/working/input-grad-2xt4")


def run(command, **kwargs):
    print("+", " ".join(map(str, command)), flush=True)
    return subprocess.run(command, check=True, **kwargs)


def launch(mode, group, pattern, repeat, warmup, steps):
    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": "0,1",
            "MGT_BN_INPUT_GRAD_KERNEL": mode,
            "MGT_BN_INPUT_GRAD_POSITIONS_PER_BLOCK": str(group),
            "MGT_BENCH_STATE_PATTERN": pattern,
            "NCCL_DEBUG": "WARN",
        }
    )
    tag = f"{pattern}-{mode}-g{group}-r{repeat}"
    id_file = OUT / f"nccl-{tag}.id"
    id_file.unlink(missing_ok=True)
    processes = []
    for rank in range(2):
        command = [
            str(BUILD / "mgt_bn_input_grad_benchmark"),
            str(rank),
            str(rank),
            "2",
            str(id_file),
            "25000",
            "50000",
            str(warmup),
            str(steps),
        ]
        processes.append(
            subprocess.Popen(
                command,
                cwd=WORK,
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
        )
    rows = []
    log = []
    for rank, process in enumerate(processes):
        output, _ = process.communicate()
        log.append(f"=== rank {rank} ===\n{output}")
        if process.returncode:
            raise RuntimeError(f"{tag} rank {rank} exited {process.returncode}")
        for line in output.splitlines():
            if line.startswith("{") and '"input_backward_ms"' in line:
                rows.append(json.loads(line))
    (OUT / f"{tag}.log").write_text("\n".join(log), encoding="utf-8")
    if len(rows) != 1:
        raise RuntimeError(f"{tag}: expected one JSON row, got {len(rows)}")
    row = rows[0]
    row.update({"tag": tag, "mode": mode, "group": group, "repeat": repeat})
    print(json.dumps(row, sort_keys=True), flush=True)
    return row


OUT.mkdir(parents=True, exist_ok=True)
run(["rm", "-rf", str(WORK), str(BUILD)])
run(["git", "clone", "--filter=blob:none", "--no-checkout", REPO, str(WORK)])
run(["git", "-C", str(WORK), "checkout", "--detach", REV])
actual = subprocess.check_output(["git", "-C", str(WORK), "rev-parse", "HEAD"], text=True).strip()
if actual != REV:
    raise RuntimeError(f"revision mismatch: {actual}")

gpu_names = subprocess.check_output(
    ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True
).splitlines()
if len(gpu_names) != 2 or any("T4" not in name for name in gpu_names):
    raise RuntimeError(f"expected exactly 2 Tesla T4 GPUs, got {gpu_names}")
run(["nvidia-smi", "-L"])

env = os.environ.copy()
env["CUDA_VISIBLE_DEVICES"] = "0,1"
env["MGT_CUDA_ARCH"] = "75"
run(["bash", "scripts/ensure_cutlass.sh"], cwd=WORK, env=env)
run(
    [
        "cmake",
        "-S",
        "native",
        "-B",
        str(BUILD),
        "-DMGT_ENABLE_CUDA=ON",
        "-DMGT_ENABLE_NCCL=ON",
        "-DCMAKE_BUILD_TYPE=Release",
        "-DCMAKE_CUDA_ARCHITECTURES=75",
        f"-DMGT_CUTLASS_ROOT={env.get('CUTLASS_ROOT', '/opt/cutlass')}",
    ],
    cwd=WORK,
    env=env,
)
run(
    [
        "cmake",
        "--build",
        str(BUILD),
        "--parallel",
        "2",
        "--target",
        "test_input_grad_grouping",
        "test_mlp_batch_norm_input_backward",
        "mgt_bn_input_grad_benchmark",
    ],
    cwd=WORK,
    env=env,
)
run([str(BUILD / "test_input_grad_grouping")], cwd=WORK, env=env)

records = []
configs = [("strict", 0), ("auto", 0)]
for repeat in range(1, 4):
    for mode, group in configs:
        records.append(launch(mode, group, "permutation", repeat, 10, 50))
for mode, group in configs:
    records.append(launch(mode, group, "zero", "stress", 3, 10))

permutation = [row for row in records if row["state_pattern"] == "permutation"]
checksums = {row["gradient_checksum"] for row in permutation}
if len(checksums) != 1:
    raise RuntimeError(f"checksum mismatch: {sorted(checksums)}")
zero = [row for row in records if row["state_pattern"] == "zero"]
if any(row["gradient_sum_abs"] != 0 for row in zero):
    raise RuntimeError("zero-pattern gradient was not zero")

medians = {}
for mode, group in configs:
    values = [
        row["input_backward_ms"]
        for row in permutation
        if row["mode"] == mode and row["group"] == group
    ]
    medians[f"{mode}:g{group}"] = statistics.median(values)
winner = min(medians, key=medians.get)
summary = {
    "revision": REV,
    "gpu_names": gpu_names,
    "gradient_checksum": next(iter(checksums)),
    "median_input_backward_ms": medians,
    "winner": winner,
    "winner_ms": medians[winner],
    "auto_ms": medians["auto:g0"],
    "auto_over_winner_percent": (medians["auto:g0"] / medians[winner] - 1.0) * 100.0,
}
(OUT / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
(OUT / "records.json").write_text(json.dumps(records, indent=2, sort_keys=True) + "\n")
print("FINAL_SUMMARY=" + json.dumps(summary, sort_keys=True), flush=True)
