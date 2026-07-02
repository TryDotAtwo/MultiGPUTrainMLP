# Kaggle 2xT4 Smoke Run

Цель этого шага - проверить сборку, CUDA-smoke контур и NCCL-сведение на двух T4 перед длинным обучением.

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
- `cuda_train_step_smoke` проходит;
- `nccl_single_rank_smoke` проходит;
- `nccl_two_device_smoke` на Kaggle видит две T4 и проверяет `allreduce(sum) / WORLD_SIZE`.

## Контракт

Двухкарточный smoke использует один плоский буфер градиентов на каждом устройстве, одинаковый порядок collective sequence и делит результат на `WORLD_SIZE` после сведения. На однокарточной локальной машине этот тест завершается как пропуск-успех, чтобы локальный Docker-контур не блокировал разработку.