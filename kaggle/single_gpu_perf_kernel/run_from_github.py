import hashlib
import json
import os
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

REPO = "https://github.com/TryDotAtwo/MultiGPUTrainMLP.git"
REV = "07f4c51cc196a4564c3458f8750f737b51078aa0"
WORK = Path("/tmp/MultiGPUTrainMLP")
BUILD = Path("/tmp/build-single-t4")
OUT = Path("/kaggle/working/single-t4-sweep")
BATCHES = [4096, 8192, 16384, 24576, 32768, 40960, 49152, 53248, 57344, 61440]


def run(command, **kwargs):
    print("+", " ".join(map(str, command)), flush=True)
    return subprocess.run(command, check=True, **kwargs)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def gpu_sample():
    fields = "clocks.current.sm,temperature.gpu,power.draw,memory.used,memory.total"
    text = subprocess.check_output(
        ["nvidia-smi", "--id=0", f"--query-gpu={fields}",
         "--format=csv,noheader,nounits"], text=True).strip()
    values = [float(value.strip()) for value in text.split(",")]
    return dict(zip(["clock_mhz", "temperature_c", "power_w",
                     "memory_used_mib", "memory_total_mib"], values))


def benchmark(batch, binary, group_json, target_bin):
    samples = []
    stop = threading.Event()

    def poll():
        while not stop.is_set():
            try:
                samples.append(gpu_sample())
            except Exception as error:
                samples.append({"sample_error": str(error)})
            stop.wait(0.2)

    env = os.environ.copy()
    env.update({
        "CUDA_VISIBLE_DEVICES": "0",
        "MGT_DENSE_FP16_PEAK_TFLOPS": "65.1264",
    })
    thread = threading.Thread(target=poll, daemon=True)
    thread.start()
    begin = time.monotonic()
    process = subprocess.run(
        [str(binary), str(batch), "140", "100", str(group_json),
         str(target_bin), "graph", "fp16"],
        cwd=WORK, env=env, text=True, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT)
    elapsed = time.monotonic() - begin
    stop.set()
    thread.join(timeout=2)
    (OUT / f"batch-{batch}.log").write_text(process.stdout, encoding="utf-8")
    valid_samples = [row for row in samples if "clock_mhz" in row]
    result = {
        "batch": batch,
        "return_code": process.returncode,
        "wall_s": elapsed,
        "telemetry": samples,
        "status": "failed" if process.returncode else "ok",
    }
    if process.returncode == 0:
        rows = [json.loads(line) for line in process.stdout.splitlines()
                if line.startswith("{") and '"step_ms"' in line]
        if len(rows) != 1:
            raise RuntimeError(f"batch {batch}: expected one result, got {len(rows)}")
        result.update(rows[0])
    if valid_samples:
        for field in ["clock_mhz", "temperature_c", "power_w",
                      "memory_used_mib", "memory_total_mib"]:
            values = [row[field] for row in valid_samples]
            result[field + "_min"] = min(values)
            result[field + "_median"] = statistics.median(values)
            result[field + "_max"] = max(values)
        median_clock = result["clock_mhz_median"]
        result["active_clock_dense_fp16_peak_tflops"] = 0.04096 * median_clock
        if result.get("useful_tflops") is not None:
            result["active_clock_useful_utilization_percent"] = (
                result["useful_tflops"] * 100.0 /
                result["active_clock_dense_fp16_peak_tflops"])
        result["free_vram_fraction_at_peak"] = 1.0 - (
            result["memory_used_mib_max"] / result["memory_total_mib_max"])
        if result["status"] == "ok" and result["free_vram_fraction_at_peak"] < 0.10:
            result["status"] = "rejected_vram_margin"
    print(json.dumps(result, sort_keys=True), flush=True)
    return result


OUT.mkdir(parents=True, exist_ok=True)
run(["rm", "-rf", str(WORK), str(BUILD)])
run(["git", "clone", "--filter=blob:none", "--no-checkout", REPO, str(WORK)])
run(["git", "-C", str(WORK), "checkout", "--detach", REV])
actual = subprocess.check_output(
    ["git", "-C", str(WORK), "rev-parse", "HEAD"], text=True).strip()
if actual != REV:
    raise RuntimeError(f"revision mismatch: {actual}")

gpu_names = subprocess.check_output(
    ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
    text=True).splitlines()
if len(gpu_names) != 2 or any("T4" not in name for name in gpu_names):
    raise RuntimeError(f"expected exactly 2 Tesla T4 GPUs, got {gpu_names}")
run(["nvidia-smi", "-L"])

env = os.environ.copy()
env["CUDA_VISIBLE_DEVICES"] = "0"
env["MGT_CUDA_ARCH"] = "75"
cutlass_output = subprocess.check_output(
    ["bash", "scripts/ensure_cutlass.sh"], cwd=WORK, env=env, text=True)
print(cutlass_output, end="", flush=True)
cutlass_lines = [line.split("=", 1)[1] for line in cutlass_output.splitlines()
                 if line.startswith("cutlass_root=")]
if len(cutlass_lines) != 1:
    raise RuntimeError(f"cannot resolve CUTLASS root from: {cutlass_output!r}")
cutlass_root = cutlass_lines[0]

ldconfig = subprocess.check_output(["ldconfig", "-p"], text=True)
driver_candidates = [Path(line.split("=>", 1)[1].strip())
                     for line in ldconfig.splitlines()
                     if "libcuda.so.1" in line and "=>" in line]
for root in [Path("/usr/local/nvidia"), Path("/usr/local/cuda/compat"),
             Path("/usr/lib/x86_64-linux-gnu"), Path("/lib/x86_64-linux-gnu")]:
    if root.exists():
        driver_candidates.extend(root.glob("**/libcuda.so*"))
if not driver_candidates:
    raise RuntimeError(f"Kaggle loader exposes no libcuda.so.1: {ldconfig}")
driver_dir = Path("/tmp/cuda-driver")
driver_dir.mkdir(exist_ok=True)
(driver_dir / "libcuda.so").symlink_to(driver_candidates[0])
env["CMAKE_LIBRARY_PATH"] = str(driver_dir)
env["LIBRARY_PATH"] = str(driver_dir)
run([
    "cmake", "-S", "native", "-B", str(BUILD),
    "-DMGT_ENABLE_CUDA=ON", "-DMGT_ENABLE_NCCL=OFF",
    "-DCMAKE_BUILD_TYPE=Release", "-DCMAKE_CUDA_ARCHITECTURES=75",
    f"-DMGT_CUTLASS_ROOT={cutlass_root}",
], cwd=WORK, env=env)
targets = [
    "test_single_gpu_benchmark", "test_cuda_adamw_half_mirror_exact",
    "test_sparse_input_grad_compact_active", "test_input_half_tiled",
    "test_single_gpu_trainer_lifecycle", "test_fp16_linear_train_ops",
    "mgt_single_gpu_benchmark",
]
run(["cmake", "--build", str(BUILD), "--parallel", "2", "--target", *targets],
    cwd=WORK, env=env)
run(["ctest", "--test-dir", str(BUILD), "--output-on-failure",
     "-R", "^(single_gpu_benchmark|cuda_adamw_half_mirror_exact|"
     "sparse_input_grad_compact_active|input_half_tiled|"
     "single_gpu_trainer_lifecycle|fp16_linear_train_ops)$"], cwd=WORK, env=env)

binary = BUILD / "mgt_single_gpu_benchmark"
group_json = WORK / "native/production_inputs/p888.json"
target_bin = WORK / "native/tests/fixtures/p888-target.bin"
manifest = {
    "revision": REV,
    "gpu_names": gpu_names,
    "binary_sha256": sha256(binary),
    "group_json_sha256": sha256(group_json),
    "target_bin_sha256": sha256(target_bin),
    "warmup": 140,
    "steps": 100,
    "mode": "graph",
    "input_gradient_precision": "fp16",
    "nominal_dense_fp16_peak_tflops": 65.1264,
}
records = []
for batch in BATCHES:
    try:
        records.append(benchmark(batch, binary, group_json, target_bin))
    except Exception as error:
        record = {"batch": batch, "status": "exception", "error": str(error)}
        records.append(record)
        print(json.dumps(record, sort_keys=True), flush=True)

accepted = [row for row in records if row.get("status") == "ok"]
if not accepted:
    raise RuntimeError("no accepted batch result")
winner = max(accepted, key=lambda row: row["useful_tflops"])
summary = {**manifest, "records": records, "winner": winner}
(OUT / "summary.json").write_text(
    json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
Path("/kaggle/working/single_t4_summary.json").write_text(
    json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print("FINAL_SUMMARY=" + json.dumps(summary, sort_keys=True), flush=True)
sys.exit(0)
