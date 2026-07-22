import csv
import json
import os
import re
import shutil
import subprocess
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path("/kaggle/working")
REPO = Path("/tmp/MultiGPUBeamSearch")
BUILD = Path("/tmp/beam-build-t4")
DATA = Path("/tmp/ihes-data")
MODELS = Path("/tmp/models")
OUT = ROOT / "beam-model-comparison"
PUZZLES = [7]
DEPTH = 12
BEAM = 262_144

for path in (REPO, BUILD, DATA, MODELS, OUT):
    shutil.rmtree(path, ignore_errors=True)
MODELS.mkdir(parents=True)
OUT.mkdir(parents=True)

subprocess.run(["git", "clone", "--depth", "1", "https://github.com/TryDotAtwo/MultiGPUBeamSearch.git", str(REPO)], check=True)
beam_rev = subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"], text=True).strip()
subprocess.run(["git", "clone", "--depth", "1", "--branch", "v3.5.1", "https://github.com/NVIDIA/cutlass.git", "/tmp/cutlass"], check=True)

release = json.load(urllib.request.urlopen("https://api.github.com/repos/TryDotAtwo/MultiGPUBeamSearch/releases/tags/ihes-p888-model"))
asset = next(a for a in release["assets"] if a["name"] == "cayleypy-ihes-cube.zip")
archive = Path("/tmp/cayleypy-ihes-cube.zip")
urllib.request.urlretrieve(asset["browser_download_url"], archive)
DATA.mkdir()
with zipfile.ZipFile(archive) as zf:
    zf.extractall(DATA)

def find_one(pattern):
    matches = list(Path("/kaggle/input").rglob(pattern))
    if len(matches) != 1:
        raise RuntimeError(f"expected one {pattern}, got {matches}")
    return matches[0]

old_checkpoints = {
    "ihes_e32692": find_one("p888-t000_1778521793_e32692.pth"),
    "ihes_e40960": find_one("p888-t000_1780290207_e40960.pth"),
}
for name, checkpoint in old_checkpoints.items():
    subprocess.run([
        "python", str(REPO / "tools/export_stream1_mlp.py"),
        "--weights", str(checkpoint), "--out", str(MODELS / name),
        "--format", "batchnorm-folded", "--dtype", "fp16", "--num-classes", "72",
    ], check=True)

new_dir = MODELS / "native_step600"
native_manifests = []
for candidate in Path("/kaggle/input").rglob("manifest.json"):
    manifest = json.loads(candidate.read_text())
    if manifest.get("flat_debug_weights") == "weights.f32.bin" and manifest.get("normalization") == "none":
        native_manifests.append(candidate)
if len(native_manifests) != 1:
    raise RuntimeError(f"expected one native inference manifest, got {native_manifests}")
shutil.copytree(native_manifests[0].parent, new_dir)

subprocess.run([
    "cmake", "-S", str(REPO), "-B", str(BUILD), "-G", "Ninja",
    "-DCMAKE_BUILD_TYPE=Release", "-DCMAKE_CUDA_ARCHITECTURES=75", "-DCUTLASS_DIR=/tmp/cutlass",
    f"-DBEAM_PUZZLE_INFO_JSON={DATA / 'puzzle_info.json'}",
], check=True)
subprocess.run(["cmake", "--build", str(BUILD), "--target", "production_runner", "-j", "2"], check=True)

rows = []
for model_name in ("native_step600", "ihes_e32692", "ihes_e40960"):
    for puzzle_id in PUZZLES:
        log = OUT / f"{model_name}_p{puzzle_id}.log"
        env = os.environ.copy()
        env.update({
            "CUDA_VISIBLE_DEVICES": "0,1", "WORLD_SIZE": "2",
            "MASTER_ADDR": "127.0.0.1", "MASTER_PORT": "29631",
            "BEAM_WEIGHT_DIR": str(MODELS / model_name),
            "BEAM_PUZZLE_INFO_JSON": str(DATA / "puzzle_info.json"),
            "BEAM_GENERATOR_PATH": str(DATA / "puzzle_info.json"),
            "BEAM_TEST_CSV": str(DATA / "test.csv"),
            "BEAM_HISTORY_MODE": "ram", "BEAM_DEPTH_LOG_EVERY": "1",
        })
        procs = []
        handles = []
        for rank in (0, 1):
            rank_env = env.copy()
            rank_env.update({"RANK": str(rank), "LOCAL_RANK": str(rank)})
            handle = open(OUT / f"{model_name}_p{puzzle_id}_rank{rank}.log", "w")
            handles.append(handle)
            procs.append(subprocess.Popen([str(BUILD / "production_runner"), str(puzzle_id), str(DEPTH), str(BEAM)], env=rank_env, stdout=handle, stderr=subprocess.STDOUT, text=True))
        codes = [p.wait(timeout=180) for p in procs]
        for handle in handles:
            handle.close()
        text = (OUT / f"{model_name}_p{puzzle_id}_rank0.log").read_text(errors="replace")
        log.write_text(text)
        solved = re.findall(r"puzzle_solved=(\d).*?seconds=([0-9.eE+-]+).*?solution_length=(-?\d+)", text)
        if any(code != 0 for code in codes) or not solved:
            raise RuntimeError(f"beam run failed {model_name} p{puzzle_id}: codes={codes}")
        flag, seconds, length = solved[-1]
        rows.append({"model": model_name, "puzzle_id": puzzle_id, "solved": int(flag), "seconds": float(seconds), "solution_length": int(length), "depth_limit": DEPTH, "beam_width": BEAM})
        print("beam_result=" + json.dumps(rows[-1], sort_keys=True), flush=True)

with (OUT / "comparison.csv").open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
    writer.writeheader(); writer.writerows(rows)
(OUT / "summary.json").write_text(json.dumps({"beam_repo_rev": beam_rev, "rows": rows}, indent=2) + "\n")
