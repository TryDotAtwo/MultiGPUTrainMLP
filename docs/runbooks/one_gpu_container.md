# Однокарточный контейнерный профиль

Цель: проверить нативный тренер в Docker с GPU и сохранить профильные артефакты перед запуском на Kaggle 2xT4.

Запуск с Windows-хоста:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/profile_single_gpu.ps1
```

По умолчанию используется образ `cmz-native-dev:2026-05-26`, архитектура `86`, несколько шагов обучения и каталог `test_results/native_profile`.

Настройки через переменные окружения:

```powershell
$env:MGT_CUDA_ARCH = "75"
$env:MGT_PROFILE_STEPS = "8"
$env:MGT_PROFILE_OUTPUT = "test_results/native_profile_t4"
$env:MGT_PROFILE_NCU = "1"
powershell -ExecutionPolicy Bypass -File scripts/profile_single_gpu.ps1
```

Артефакты:

- `profile.jsonl` - время шага, batch size, loss и память;
- `train.log` - журнал обучения;
- `weights/weights.f32.bin` - веса для inference-контура;
- `checkpoint/state.f32.bin` - состояние возобновления обучения;
- Nsight Systems/Compute отчеты, если включены соответствующие флаги.

`MGT_PROFILE_NCU=1` тяжелее обычного smoke-режима и нужен для разбора конкретных CUDA-ядер.