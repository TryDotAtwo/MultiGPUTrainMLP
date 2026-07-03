# Kaggle 2xT4 Smoke Run

Цель этого шага - проверить сборку, CUDA smoke-контур, NCCL-сведение и ранговый запуск перед длинным обучением.

## Команды

```bash
bash kaggle/kernel/run_2xt4.sh
```

Этот скрипт делает две вещи:

- собирает `build-kaggle-2xt4` с `MGT_ENABLE_CUDA=ON`, `MGT_ENABLE_NCCL=ON`, `CMAKE_CUDA_ARCHITECTURES=75`;
- запускает CTest smoke-набор и затем `kaggle/kernel/run_ranks_2xt4.sh`.

Для отдельной проверки рангового запуска:

```bash
bash kaggle/kernel/run_ranks_2xt4.sh
```

## Ожидаемый результат

CTest должен пройти:

- `cuda_compile`;
- `cuda_random_walk_smoke`;
- `cuda_adamw_smoke`;
- `cuda_mlp_forward_smoke`;
- `cuda_mlp_backward_smoke`;
- `cuda_train_step_smoke`;
- `nccl_single_rank_smoke`;
- `nccl_two_device_smoke`;
- `native_train_smoke`;
- `native_train_profile_smoke`;
- `native_train_artifacts`.

Ранговый запуск пишет:

- `runs/kaggle-2xt4/rank0/metadata.env`;
- `runs/kaggle-2xt4/rank0/train.log`;
- `runs/kaggle-2xt4/rank0/weights/weights.f32.bin`;
- на двух T4 также `rank1/...` с `WORLD_SIZE=2` и `GLOBAL_RANK=1`.

На локальной однокарточной машине `run_ranks_2xt4.sh` запускает `WORLD_SIZE=1`, чтобы Docker-контур разработки не блокировался отсутствием второй карты. На Kaggle при двух видимых T4 запускаются два процесса, по одному на устройство.

## Контракт

Онлайн данные генерируются внутри каждого ранга через random walks. Общий датасет на диске не используется. Каждый ранг пишет отдельный каталог артефактов, а smoke-профиль фиксирует `output_dim=1`, `AdamW weight_decay=0` по умолчанию и экспортирует плоский `float32` файл весов для будущего инференс-контура.