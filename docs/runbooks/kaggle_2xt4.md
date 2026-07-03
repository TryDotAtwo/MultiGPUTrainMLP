# Kaggle 2xT4 Smoke Run

Цель этого шага - проверить сборку, CUDA smoke-контур, NCCL-сведение, ранговый запуск и контракт артефактов перед длинным обучением.

## Команды

```bash
bash kaggle/kernel/run_2xt4.sh
```

Скрипт собирает `build-kaggle-2xt4` с `MGT_ENABLE_CUDA=ON`, `MGT_ENABLE_NCCL=ON`, `CMAKE_CUDA_ARCHITECTURES=75`, запускает CTest smoke-набор, затем запускает ранги через `kaggle/kernel/run_ranks_2xt4.sh`.

Для отдельной проверки рангового запуска:

```bash
bash kaggle/kernel/run_ranks_2xt4.sh
```

Для отдельной проверки уже созданных rank-артефактов:

```bash
bash kaggle/kernel/check_rank_outputs.sh runs/kaggle-2xt4 2
```

## CTest Smoke

Должны пройти:

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

## Rank Outputs

`run_ranks_2xt4.sh` пишет отдельный каталог на каждый ранг:

- `runs/kaggle-2xt4/rank0/metadata.env`;
- `runs/kaggle-2xt4/rank0/train.log`;
- `runs/kaggle-2xt4/rank0/layers.json`;
- `runs/kaggle-2xt4/rank0/weights/manifest.json`;
- `runs/kaggle-2xt4/rank0/weights/weights.f32.bin`.

На Kaggle при двух видимых T4 также создается `rank1/...`. Проверка `check_rank_outputs.sh` требует:

- число каталогов `rank*` равно `WORLD_SIZE`;
- каждый ранг имеет уникальный `GLOBAL_RANK`;
- `WORLD_SIZE`, `GLOBAL_RANK`, `LOCAL_RANK`, `DEVICE_ID` согласованы с запуском;
- `MODEL_MODE=MLP2RB`, `OUTPUT_DIM=1` присутствуют в metadata;
- `train.log` содержит хотя бы один `phase=train`;
- `weights/manifest.json` объявляет `format=stream1_weights`;
- файл весов непустой.

На локальной однокарточной машине launcher использует `WORLD_SIZE=1`, чтобы Docker-контур разработки не блокировался отсутствием второй карты. На Kaggle при двух видимых T4 запускаются два процесса, по одному на устройство.

## Контракт

Онлайн данные генерируются внутри каждого ранга через random walks. Общий датасет на диске не используется. Каждый ранг пишет отдельный каталог артефактов, а smoke-профиль фиксирует `output_dim=1`, `AdamW weight_decay=0` по умолчанию и экспортирует плоский `float32` файл весов для будущего инференс-контура.