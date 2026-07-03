#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
export MGT_CUDA_ARCH=${MGT_CUDA_ARCH:-75}
cmake -S native -B build-kaggle-2xt4 -DMGT_ENABLE_CUDA=ON -DMGT_ENABLE_NCCL=ON -DCMAKE_CUDA_ARCHITECTURES=${MGT_CUDA_ARCH}
cmake --build build-kaggle-2xt4 --config Release --target mgt_native_train_smoke
if command -v nvidia-smi >/dev/null 2>&1; then
  visible_count=$(nvidia-smi -L | wc -l)
else
  visible_count=$(python3 - <<'PY'
import os
value = os.environ.get('CUDA_VISIBLE_DEVICES', '')
print(len([part for part in value.split(',') if part.strip()]) if value else 1)
PY
)
fi
if [ "${MGT_FORCE_WORLD_SIZE:-}" != "" ]; then
  world_size="${MGT_FORCE_WORLD_SIZE}"
elif [ "$visible_count" -ge 2 ]; then
  world_size=2
else
  world_size=1
fi
mkdir -p runs/kaggle-2xt4
rm -f runs/kaggle-2xt4/nccl.id
pids=()
for rank in $(seq 0 $((world_size - 1))); do
  ./build-kaggle-2xt4/mgt_native_train_smoke \
    --output-dir "runs/kaggle-2xt4/rank${rank}" \
    --steps "${MGT_STEPS:-3}" \
    --device-id "$rank" \
    --world-size "$world_size" \
    --global-rank "$rank" \
    --local-rank "$rank" \
    --batch-size "${MGT_BATCH_SIZE:-64}" \
    --k-min "${MGT_K_MIN:-1}" \
    --k-max "${MGT_K_MAX:-9}" \
    --hd1 "${MGT_HD1:-5}" \
    --hd2 "${MGT_HD2:-3}" \
    --nccl-id-file "runs/kaggle-2xt4/nccl.id" > "runs/kaggle-2xt4/rank${rank}.stdout" 2>&1 &
  pids+=("$!")
done
for pid in "${pids[@]}"; do
  wait "$pid"
done
bash kaggle/kernel/check_rank_outputs.sh runs/kaggle-2xt4 "$world_size"
echo "rank_launch_ok world_size=${world_size}"