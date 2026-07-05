#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
export MGT_CUDA_ARCH=${MGT_CUDA_ARCH:-75}
if [ "${MGT_PERF_RUN:-0}" = "1" ]; then
  export MGT_CLEAN_BUILD_OUTPUT="${MGT_CLEAN_BUILD_OUTPUT:-1}"
fi
if [ -f scripts/ensure_cutlass.sh ]; then
  source scripts/ensure_cutlass.sh
fi
cmake -S native -B build-kaggle-2xt4 -DMGT_ENABLE_CUDA=ON -DMGT_ENABLE_NCCL=ON -DCMAKE_CUDA_ARCHITECTURES=${MGT_CUDA_ARCH} -DMGT_CUTLASS_ROOT="${CUTLASS_ROOT:-/opt/cutlass}" -DMGT_AUTO_CUTLASS_HALF_GEMM=${MGT_AUTO_CUTLASS_HALF_GEMM:-ON}
cmake --build build-kaggle-2xt4 --config Release
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi || true
  visible_count=$(nvidia-smi -L 2>/dev/null | wc -l)
else
  visible_count=0
fi
echo "kaggle_cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
echo "kaggle_nvidia_smi_device_count=${visible_count}"
if [ "$visible_count" -lt 2 ]; then
  echo "NO_2XT4: Kaggle did not expose two GPUs to this run; skipping runtime CUDA/NCCL tests."
  exit 0
fi
ctest --test-dir build-kaggle-2xt4 -R 'cuda_compile|cuda_random_walk_smoke|cuda_adamw_smoke|cuda_mlp_forward_smoke|cuda_mlp_backward_smoke|cuda_train_step_smoke|nccl_single_rank_smoke|nccl_two_device_smoke|native_train_smoke|native_train_profile_smoke|native_train_resume_smoke|native_train_artifacts' --output-on-failure -C Release
bash kaggle/kernel/run_ranks_2xt4.sh
if [ "${MGT_CLEAN_BUILD_OUTPUT:-0}" = "1" ]; then
  rm -rf build-kaggle-2xt4 payload.zip
fi
