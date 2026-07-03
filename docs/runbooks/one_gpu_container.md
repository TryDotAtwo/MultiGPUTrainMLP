# Однокарточный профильный прогон

Цель: проверить нативный ранговый тренер в Docker с GPU и сохранить профильные артефакты до запуска на Kaggle 2xT4.

Запуск с хоста Windows:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/profile_single_gpu.ps1
```

По умолчанию используется образ `cmz-native-dev:2026-05-26`, архитектура `86`, четыре шага обучения и каталог `test_results/native_profile`.

Настройки через переменные окружения:

```powershell
$env:MGT_CUDA_ARCH = "75"
$env:MGT_PROFILE_STEPS = "8"
$env:MGT_PROFILE_OUTPUT = "test_results/native_profile_t4"
$env:MGT_PROFILE_NCU = "1"
powershell -ExecutionPolicy Bypass -File scripts/profile_single_gpu.ps1
```

Артефакты:

- `test_results/native_profile/native_train.nsys-rep` - трасса Nsight Systems;
- `test_results/native_profile/run/profile.jsonl` - время шага, размер пачки, loss и память из самого ранга;
- `test_results/native_profile/run/train.log` - обычный журнал обучения;
- `test_results/native_profile/run/weights/weights.f32.bin` - веса для инференс-контура;
- `test_results/native_profile/run/checkpoint/state.f32.bin` - состояние возобновления обучения.

Если `MGT_PROFILE_NCU=1`, дополнительно создается отчет Nsight Compute. Этот режим тяжелее и нужен для разбора конкретных ядер, а не для каждого smoke-прогона.