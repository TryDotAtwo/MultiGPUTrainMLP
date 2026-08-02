#!/usr/bin/env bash
set -euo pipefail

repo="${MGT_REPO:-$PWD}"
build_dir="${MGT_BUILD_DIR:-$repo/build-nccl-v3}"
results_dir="${MGT_RESULTS_DIR:-$repo/results/prepared-bf16-nsys-${SLURM_JOB_ID:-manual}}"
expected_sha="${MGT_EXPECTED_SHA:?set MGT_EXPECTED_SHA to the exact revision}"

cd "$repo"
actual_sha="$(git rev-parse HEAD)"
if [[ "$actual_sha" != "$expected_sha" ]]; then
  echo "error: revision mismatch expected=$expected_sha actual=$actual_sha" >&2
  exit 2
fi
if [[ "${SLURM_NTASKS:-0}" -ne 8 ]]; then
  echo "error: run inside an allocation with exactly 8 tasks" >&2
  exit 2
fi
command -v nsys >/dev/null || { echo "error: nsys is not on PATH" >&2; exit 2; }

mkdir -p "$results_dir"
source scripts/ensure_cutlass.sh
cmake -S native -B "$build_dir" \
  -DMGT_ENABLE_CUDA=ON -DMGT_ENABLE_NCCL=ON \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=80 \
  -DMGT_CUTLASS_ROOT="${CUTLASS_ROOT:-/opt/cutlass}"
cmake --build "$build_dir" --parallel "${SLURM_CPUS_ON_NODE:-32}" \
  --target test_prepared_p888_train_step test_mlp_batch_norm_full_backward \
           test_bf16_residual_stack_8rank
ctest --test-dir "$build_dir" \
  -R '^(prepared_p888_train_step|mlp_batch_norm_full_backward)$' \
  --output-on-failure

benchmark="$(realpath "$build_dir/test_bf16_residual_stack_8rank")"

run_world() {
  local tag="$1"
  local profile_rank0="$2"
  local id_root="$results_dir/$tag"
  rm -f -- "$id_root.bn.residual" "$id_root.weight.residual" \
            "$id_root.metrics.residual"
  MGT_BN_ID="$id_root.bn" MGT_WEIGHT_ID="$id_root.weight" \
  MGT_METRICS_ID="$id_root.metrics" MGT_BENCHMARK="$benchmark" \
  MGT_PROFILE_RANK0="$profile_rank0" MGT_NSYS_OUT="$results_dir/$tag-rank0" \
  srun --ntasks=8 --gpus-per-task=1 --gpu-bind=single:1 \
    --kill-on-bad-exit=1 bash -lc '
      if [[ "$MGT_PROFILE_RANK0" == 1 && "$SLURM_PROCID" == 0 ]]; then
        exec nsys profile --trace=cuda,nvtx,osrt,cublas,nccl \
          --sample=none --cpuctxsw=none --stats=false --force-overwrite=true \
          -o "$MGT_NSYS_OUT" "$MGT_BENCHMARK"
      fi
      exec "$MGT_BENCHMARK"
    ' 2>&1 | tee "$results_dir/$tag.log"
}

echo "PROFILE_BEGIN revision=$actual_sha results=$results_dir"
run_world baseline 0
run_world nsys 1
nsys stats --report cuda_gpu_kern_sum,cuda_api_sum,nvtx_sum,nccl_sum \
  --format csv "$results_dir/nsys-rank0.nsys-rep" \
  > "$results_dir/nsys-stats.csv"

grep -E 'bf16 input phases|bf16 prepared input_hidden_residual' \
  "$results_dir/baseline.log" "$results_dir/nsys.log" || true
echo "PROFILE_OK results=$results_dir report=$results_dir/nsys-rank0.nsys-rep stats=$results_dir/nsys-stats.csv"
