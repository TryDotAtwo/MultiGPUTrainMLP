# Kaggle 2xT4 smoke/perf run

Цель: проверить сборку, CUDA smoke-контур, NCCL, двухранговый запуск и контракт артефактов перед длинным обучением.

## Команда

```bash
bash kaggle/kernel/run_2xt4.sh
```

Скрипт:

- находит или подтягивает CUTLASS через `scripts/ensure_cutlass.sh`;
- собирает `build-kaggle-2xt4` с CUDA, NCCL и `CMAKE_CUDA_ARCHITECTURES=75`;
- запускает CTest smoke-набор;
- запускает ранги через `kaggle/kernel/run_ranks_2xt4.sh`.

Для perf-режима полной p888 модели:

```bash
MGT_PERF_RUN=1 bash kaggle/kernel/run_2xt4.sh
```

## Отдельный запуск рангов

```bash
bash kaggle/kernel/run_ranks_2xt4.sh
```

Проверка уже созданных rank-артефактов:

```bash
bash kaggle/kernel/check_rank_outputs.sh runs/kaggle-2xt4 2
```

## CTest smoke

Должны пройти CUDA/NCCL smoke-тесты, native train smoke, resume smoke и artifact contract.

## Rank outputs

`run_ranks_2xt4.sh` пишет отдельный каталог на каждый ранг:

- `runs/kaggle-2xt4/rank0/metadata.env`;
- `runs/kaggle-2xt4/rank0/train.log`;
- `runs/kaggle-2xt4/rank0/layers.json`;
- `runs/kaggle-2xt4/rank0/profile.jsonl`;
- `runs/kaggle-2xt4/throughput_summary.json`.

При `MGT_WRITE_ARTIFACTS=1` также пишутся `weights/manifest.json` и flat-файлы весов.

## Контракт

Online-данные генерируются внутри каждого ранга через random walks. Общий датасет на диске не используется. `AdamW` по умолчанию работает с `weight_decay=0`. Экспорт весов остается совместимым с быстрым inference-контуром beam search.