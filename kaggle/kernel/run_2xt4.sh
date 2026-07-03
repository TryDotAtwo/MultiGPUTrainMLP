#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
export MGT_CUDA_ARCH=${MGT_CUDA_ARCH:-75}
cmake -S native -B build-kaggle-2xt4 -DMGT_ENABLE_CUDA=ON -DMGT_ENABLE_NCCL=ON -DCMAKE_CUDA_ARCHITECTURES=${MGT_CUDA_ARCH}
cmake --build build-kaggle-2xt4 --config Release
ctest --test-dir build-kaggle-2xt4 -R 'cuda_compile|cuda_random_walk_smoke|cuda_adamw_smoke|cuda_mlp_forward_smoke|cuda_mlp_backward_smoke|cuda_train_step_smoke|nccl_single_rank_smoke|nccl_two_device_smoke|native_train_smoke|native_train_profile_smoke|native_train_resume_smoke|native_train_artifacts' --output-on-failure -C Release
bash kaggle/kernel/run_ranks_2xt4.sh
