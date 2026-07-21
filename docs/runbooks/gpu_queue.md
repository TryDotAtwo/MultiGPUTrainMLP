# Общая очередь GPU

Для локальных GPU-прогонов используется один долгоживущий контейнер `mgt-gpu-queue`.
Все агенты отправляют задания в него через `docker exec`, а не запускают свои `docker run --gpus` напрямую.

## Старт

```powershell
powershell -ExecutionPolicy Bypass -File scripts/start_gpu_queue.ps1
```

Контейнер монтирует репозиторий как `/work`, запускает `scripts/gpu_queue_worker.py` и читает очередь из `/work/.gpu_queue`.
После каждого задания воркер ждет 10 секунд перед следующим.

## Отправка задания

```powershell
docker exec mgt-gpu-queue python3 scripts/gpu_queue_submit.py --label smoke --wait -- bash -lc "python3 --version"
```

Для профиля или обучения команда такая же, только после `--` передается реальная команда:

```powershell
docker exec mgt-gpu-queue python3 scripts/gpu_queue_submit.py --label nsys-p888 --wait -- bash -lc "python3 scripts/wait_gpu_idle.py && nsys profile --trace=cuda,nvtx,osrt,cublas --stats=true --force-overwrite=true -o test_results/nsys/p888 ./build/mgt_native_train ..."
```

## Где смотреть состояние

- `.gpu_queue/status.json` - текущее состояние воркера.
- `.gpu_queue/pending` - ожидающие задания.
- `.gpu_queue/running` - текущее задание.
- `.gpu_queue/done` - успешные задания.
- `.gpu_queue/failed` - упавшие задания.
- `.gpu_queue/logs/*.log` - полный вывод команд.

Правило: если задача может занять GPU, она должна идти через `mgt-gpu-queue`.