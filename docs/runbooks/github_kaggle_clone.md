# GitHub и Kaggle через git clone

## Публикация репозитория

Целевой remote:

```bash
https://github.com/TryDotAtwo/MultiGPUTrainMLP.git
```

Когда GitHub auth станет валидным:

```bash
gh repo create TryDotAtwo/MultiGPUTrainMLP --private --source . --remote origin --push
```

Если репозиторий уже создан:

```bash
git remote add origin https://github.com/TryDotAtwo/MultiGPUTrainMLP.git
git push -u origin codex-native-trainer-implementation
```

## Kaggle notebook

Нужны две T4 и включенный internet:

```bash
git clone https://github.com/TryDotAtwo/MultiGPUTrainMLP.git
cd MultiGPUTrainMLP
MGT_PERF_RUN=1 bash kaggle/kernel/run_2xt4.sh
```

Launcher вызывает `scripts/ensure_cutlass.sh`: если Kaggle не дает `/opt/cutlass`, CUTLASS подтягивается в `.deps/cutlass`.

## Ожидаемые признаки успешного запуска

В логе должны быть:

- две видимые NVIDIA T4;
- сообщение CMake `CUTLASS found`;
- `100% tests passed`;
- `rank_outputs_ok world_size=2`;
- `rank_launch_ok world_size=2`.

Сводка скорости:

```bash
cat runs/kaggle-2xt4/throughput_summary.json
```

Последний полный p888 ориентир на 2xT4:

```text
avg_throughput_states_s = 462623
input_grad_backend = position_tiled_one_hot_half_gemm
input_grad_position_tile = 36
```