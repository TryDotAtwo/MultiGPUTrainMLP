# MultiGPUTrainMLP

Нативный CUDA-тренер для конфигурируемых MLP value-моделей, совместимых с быстрым инференсом MultiGPU beam search.

Тренер генерирует online random walks на каждом ранге, не использует общий датасет на диске, экспортирует веса в beam-search-совместимом формате и работает на одной GPU, Kaggle 2xT4 и дальше в NCCL multi-GPU запуске.

## Текущий p888 baseline

Полная модель по умолчанию:

- длина состояния: `80`, паддинг считается из конфига;
- слои: `2556 -> 218`, физический размер `hd2 = 224`;
- residual-блоки: `16`;
- `output_dim`: конфигурируемый, сейчас по умолчанию `1`;
- оптимизатор: `AdamW`, `weight_decay=0` по умолчанию.

Последние измерения:

- локально, полная p888, одна GPU, `batch=53248`, tiled input gradient: около `256k` кандидатов/с;
- Kaggle 2xT4, полная p888, `batch=53248` на ранг, tile 56: устойчивые повторы около `536k` кандидатов/с; input-gradient-only CUTLASS повторы около `556k` кандидатов/с.

Оценка старого ориентира `30000` эпох за `3` суток при `batch_size=100000`:

- если эпоха равна одному batch: `30000 * 100000 / (3 * 86400) = 11574` кандидатов/с;
- если эпоха покрывает все `K=1..29`: около `335648` кандидатов/с.

## Сборка после git clone

```bash
git clone https://github.com/TryDotAtwo/MultiGPUTrainMLP.git
cd MultiGPUTrainMLP
source scripts/ensure_cutlass.sh
cmake -S native -B build-native -DMGT_ENABLE_CUDA=ON -DMGT_ENABLE_NCCL=ON -DCMAKE_CUDA_ARCHITECTURES=75 -DMGT_CUTLASS_ROOT="$CUTLASS_ROOT"
cmake --build build-native --config Release --target mgt_native_train
```

`scripts/ensure_cutlass.sh` сначала использует готовый `CUTLASS_ROOT`, `/opt/cutlass` или `MGT_CUTLASS_ROOT`. Если CUTLASS не найден, скрипт клонирует NVIDIA CUTLASS в `.deps/cutlass`.

## Локальная GPU-проверка

```bash
ctest --test-dir build-native --output-on-failure
```

Одноранговый p888 smoke:

```bash
./build-native/mgt_native_train \
  --synthetic-benchmark 1 \
  --output-dir runs/local-p888 \
  --steps 8 \
  --device-id 0 \
  --world-size 1 \
  --global-rank 0 \
  --local-rank 0 \
  --batch-size 53248 \
  --k-min 1 \
  --k-max 29 \
  --hd1 2556 \
  --hd2 218 \
  --nrd 16 \
  --input-grad-fp16 1 \
  --input-grad-position-tile 56 \
  --linear-fp16 1 \
  --write-artifacts 0
```

## Kaggle 2xT4

В Kaggle notebook/script с двумя T4 и включенным internet:

```bash
git clone https://github.com/TryDotAtwo/MultiGPUTrainMLP.git
cd MultiGPUTrainMLP
MGT_PERF_RUN=1 bash kaggle/kernel/run_2xt4.sh
cat runs/kaggle-2xt4/throughput_summary.json
```

Упаковка Kaggle script kernel с локальной машины:

```bash
python scripts/package_kaggle_kernel.py --full-model --perf-run --output-dir test_results/mgt_kaggle_kernel
kaggle kernels push -p test_results/mgt_kaggle_kernel --accelerator NvidiaTeslaT4
```

## Выходные файлы

Каждый запуск пишет:

- `metadata.env` с точными размерами puzzle/model/runtime;
- `layers.json` с раскладкой слоев;
- `profile.jsonl` с таймингами шагов;
- `weights/manifest.json` и файлы весов при `--write-artifacts 1`;
- `checkpoint/manifest.json` и состояние оптимизатора при включенном checkpoint.

Быстрый путь не делает динамических аллокаций внутри training step после построения плана. Паддинг, размеры слоев, gradient buckets, carousel slots и `output_dim` берутся из конфига и расчетной раскладки, а не из hard-coded чисел в ядрах.