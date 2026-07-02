# Kaggle 2xT4 Smoke Run

Цель этого шага - проверить сборку и CUDA-smoke контур на двух T4 до добавления NCCL-сведения.

## Команда

```bash
bash kaggle/kernel/run_2xt4.sh
```

## Ожидаемый результат

- сборка с `CMAKE_CUDA_ARCHITECTURES=75`;
- `cuda_compile` проходит;
- `cuda_random_walk_smoke` проходит;
- `cuda_adamw_smoke` проходит;
- `cuda_mlp_forward_smoke` проходит;
- `cuda_mlp_backward_smoke` проходит;
- `cuda_train_step_smoke` проходит.

## Следующий контракт

После этого добавляется отдельный NCCL smoke: два ранга, одинаковый порядок collective sequence, один плоский буфер градиентов, деление результата на `WORLD_SIZE` после сведения.