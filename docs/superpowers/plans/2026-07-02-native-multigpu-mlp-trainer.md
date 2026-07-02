# План Реализации Нативного Многокарточного Тренера МЛП

> **Для агентных исполнителей:** обязательный поднавык: используйте `superpowers:subagent-driven-development` или `superpowers:executing-plans`, выполняйте план по задачам и отмечайте чекбоксы. Каждый этап должен завершаться проверкой и отдельным коммитом.

**Цель:** собрать самостоятельный нативный тренер МЛП, который обучается на онлайн случайных блужданиях, пишет веса в формат быстрого поиска и работает от одной видеокарты до многосерверного запуска.

**Архитектура:** Раст отвечает за командную строку, конфигурацию, журналы и запуск нативного ранга. Си++ задает контракты данных, загрузку головоломки, план памяти, артефакты и вызов КУДА-кода. КУДА, КАТЛАСС и NCCL выполняют генерацию пачки, прямой проход, обратный проход, сведение градиентов и шаг АдамВ без выделения памяти внутри цикла обучения.

**Стек:** Раст, Си++20, КУДА, КАТЛАСС, NCCL, CMake, Docker, Кагл, Nsight Systems, Nsight Compute.

---

## Исходные Документы

- Архитектура: `D:\MultiGPUTrainMLP\docs\superpowers\specs\2026-07-01-native-multigpu-mlp-trainer-design.md`
- Контракт поиска: `D:\100XH100\ARCHITECTURE_NEED.md`
- Экспорт весов поиска: `D:\100XH100\tools\export_stream1_mlp.py`
- Чтение весов поиска: `D:\100XH100\tools\stream1_weight_io.hpp`
- Конфиг поиска: `D:\100XH100\src\config.hpp`
- Пример обучения: `C:\Users\Иван Литвак\Downloads\Telegram Desktop\archive.zip`

## Файловая Структура

Создать:

- `Cargo.toml` - корневой Раст-проект.
- `crates/trainer-cli/Cargo.toml` - пакет командной строки.
- `crates/trainer-cli/src/main.rs` - разбор команд и запуск нативного тренера.
- `crates/trainer-cli/src/config.rs` - типы конфигурации и проверки.
- `crates/trainer-cli/src/native.rs` - граница вызова нативной библиотеки.
- `crates/trainer-cli/tests/config_contract.rs` - контрактные тесты конфигурации.
- `native/CMakeLists.txt` - сборка нативной библиотеки, тестов и исполняемого ранга.
- `native/include/mgt/config.hpp` - статическая конфигурация и производные размеры.
- `native/include/mgt/status.hpp` - коды ошибок без исключений в горячем пути.
- `native/include/mgt/static_contracts.hpp` - структуры `TrainState80`, `WalkMeta`, `LossStats`, `TensorBlockHeader`.
- `native/src/static_contracts.cpp` - проверки контракта при запуске.
- `native/tests/test_static_contracts.cpp` - тест размеров и выравнивания.
- `native/include/mgt/puzzle_io.hpp` - загрузка группы и цели.
- `native/src/puzzle_io.cpp` - строгий разбор `p888.json` и `p888-t000.pt`.
- `native/tests/test_puzzle_io.cpp` - тест загрузки p888 из фикстур.
- `native/tests/fixtures/p888.json` - минимальная фикстура таблицы ходов.
- `native/tests/fixtures/p888-target.bin` - целевое состояние 72 байта.
- `native/include/mgt/model_layout.hpp` - расчет блоков весов и градиентов.
- `native/src/model_layout.cpp` - смещения, размеры, выравнивание и число параметров.
- `native/tests/test_model_layout.cpp` - тест числа параметров и смещений.
- `native/include/mgt/weight_export.hpp` - запись `manifest.json` и бинарных блоков весов.
- `native/src/weight_export.cpp` - совместимый экспорт версии один.
- `native/tests/test_weight_export.cpp` - тест манифеста и бинарных размеров.
- `native/include/mgt/random_walk.hpp` - интерфейс генератора блужданий.
- `native/src/random_walk_cpu.cpp` - эталонный генератор на процессоре.
- `native/tests/test_random_walk_cpu.cpp` - детерминированные тесты блужданий.
- `native/include/mgt/mlp_cpu_ref.hpp` - малая эталонная МЛП для численных проверок.
- `native/src/mlp_cpu_ref.cpp` - прямой проход, обратный проход, АдамВ на процессоре.
- `native/tests/test_mlp_cpu_ref.cpp` - проверка градиента и АдамВ.
- `native/include/mgt/train_loop.hpp` - публичный интерфейс запуска тренировки.
- `native/src/train_loop.cpp` - подготовка ранга, память, фазы, журналы.
- `native/src/main_native.cpp` - исполняемый нативный вход без Раста.
- `native/cuda/device_context.cuh` - владение устройством, потоками, событиями.
- `native/cuda/device_context.cu` - инициализация устройства и проверка доступности.
- `native/cuda/random_walk_kernel.cuh` - сигнатуры генерации блужданий.
- `native/cuda/random_walk_kernel.cu` - генерация онлайн пачки.
- `native/cuda/mlp_forward.cuh` - сигнатуры прямого прохода.
- `native/cuda/mlp_forward.cu` - прямой проход версии один.
- `native/cuda/mlp_backward.cuh` - сигнатуры обратного прохода.
- `native/cuda/mlp_backward.cu` - градиенты версии один.
- `native/cuda/adamw.cuh` - сигнатуры шага АдамВ.
- `native/cuda/adamw.cu` - обновление весов на видеокарте.
- `native/cuda/allreduce_nccl.cuh` - граница сведения градиентов.
- `native/cuda/allreduce_nccl.cu` - NCCL-сведение с одинаковым порядком операций.
- `native/cuda/memory_plan.cuh` - расчет и проверка видеопамяти ранга.
- `native/cuda/memory_plan.cu` - предвыделение и фазовые раскладки.
- `native/tests/test_cuda_smoke.cpp` - короткая однокарточная проверка при наличии КУДА.
- `scripts/build_native.ps1` - сборка нативного кода на Windows.
- `scripts/test_cpu.ps1` - запуск процессорных контрактных тестов.
- `scripts/test_single_gpu.ps1` - короткая однокарточная КУДА-проверка.
- `scripts/profile_single_gpu.ps1` - запуск профиля одного устройства.
- `docker/Dockerfile.cuda` - контейнер для однокарточного прогона и профилировщика.
- `docker/run_single_gpu.sh` - запуск внутри контейнера.
- `kaggle/kernel/run_2xt4.sh` - запуск двух рангов на Кагле.
- `kaggle/kernel/kernel.json` - описание Кагл-ядра.
- `hpc/run_8xa100.sh` - запуск восьми А100.
- `hpc/run_multinode.sh` - шаблон многосерверного запуска.
- `docs/runbooks/one_gpu_container.md` - инструкция однокарточного контейнерного прогона.
- `docs/runbooks/kaggle_2xt4.md` - инструкция Кагл-прогона.
- `docs/runbooks/hpc_a100.md` - инструкция кластерного прогона.

Изменять:

- `D:\MultiGPUTrainMLP\docs\superpowers\specs\2026-07-01-native-multigpu-mlp-trainer-design.md` - только если проверка реализации выявит неточный контракт.

## Инварианты Первой Версии

- `STATE_LEN = 72`
- `STATE_STORAGE_LEN = 80`
- `STATE_ALIGNMENT = 16`
- `STATE_VALUE_PAD = 128`
- `MOVE_COUNT = 18`
- `OUTPUT_DIM = 1`
- `weight_decay = 0` по умолчанию
- общий набор данных на диске отсутствует
- каждый ранг генерирует свою пачку от цели
- внутри шага обучения нет `cudaMalloc`, `cudaFree`, `new`, `delete`, `malloc`, `free`
- если `WORLD_SIZE == 1`, сведение градиентов не вызывается
- если `WORLD_SIZE > 1`, порядок NCCL-вызовов одинаков на всех рангах
- быстрый экспорт весов версии один читается проектом поиска без изменения его контракта
- двухкарточная внешняя проверка начинается на Кагле

## Задача 1: Каркас Репозитория И Команды Сборки

**Файлы:**

- Создать: `D:\MultiGPUTrainMLP\Cargo.toml`
- Создать: `D:\MultiGPUTrainMLP\crates\trainer-cli\Cargo.toml`
- Создать: `D:\MultiGPUTrainMLP\crates\trainer-cli\src\main.rs`
- Создать: `D:\MultiGPUTrainMLP\crates\trainer-cli\src\config.rs`
- Создать: `D:\MultiGPUTrainMLP\crates\trainer-cli\src\native.rs`
- Создать: `D:\MultiGPUTrainMLP\crates\trainer-cli\tests\config_contract.rs`
- Создать: `D:\MultiGPUTrainMLP\native\CMakeLists.txt`
- Создать: `D:\MultiGPUTrainMLP\scripts\build_native.ps1`
- Создать: `D:\MultiGPUTrainMLP\scripts\test_cpu.ps1`

- [ ] **Шаг 1: добавить корневой Раст-проект**

`D:\MultiGPUTrainMLP\Cargo.toml`:

```toml
[workspace]
resolver = "2"
members = ["crates/trainer-cli"]

[workspace.package]
edition = "2021"
license = "MIT"
```

- [ ] **Шаг 2: добавить пакет командной строки**

`D:\MultiGPUTrainMLP\crates\trainer-cli\Cargo.toml`:

```toml
[package]
name = "trainer-cli"
version = "0.1.0"
edition.workspace = true
license.workspace = true

[dependencies]
anyhow = "1.0"
clap = { version = "4.5", features = ["derive"] }
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
toml = "0.8"
```

- [ ] **Шаг 3: написать тип конфигурации**

`D:\MultiGPUTrainMLP\crates\trainer-cli\src\config.rs`:

```rust
use anyhow::{bail, Result};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TrainerConfig {
    pub group_id: u32,
    pub target_id: u32,
    pub state_len: u32,
    pub state_alignment: u32,
    pub state_value_pad: u32,
    pub move_count: u32,
    pub output_dim: u32,
    pub hd1: u32,
    pub hd2: u32,
    pub residual_blocks: u32,
    pub walkers_per_depth: u32,
    pub k_min: u32,
    pub k_max: u32,
    pub epochs: u64,
    pub learning_rate: f32,
    pub weight_decay: f32,
    pub adam_beta1: f32,
    pub adam_beta2: f32,
    pub adam_eps: f32,
    pub base_seed: u64,
    pub checkpoint_period_steps: u64,
    pub weight_export_period_steps: u64,
}

impl TrainerConfig {
    pub fn p888_default() -> Self {
        Self {
            group_id: 888,
            target_id: 0,
            state_len: 72,
            state_alignment: 16,
            state_value_pad: 128,
            move_count: 18,
            output_dim: 1,
            hd1: 2556,
            hd2: 218,
            residual_blocks: 16,
            walkers_per_depth: 3449,
            k_min: 1,
            k_max: 29,
            epochs: 32692,
            learning_rate: 0.0001,
            weight_decay: 0.0,
            adam_beta1: 0.9,
            adam_beta2: 0.999,
            adam_eps: 1.0e-8,
            base_seed: 0x8880_0000_0000_0001,
            checkpoint_period_steps: 4096,
            weight_export_period_steps: 4096,
        }
    }

    pub fn state_storage_len(&self) -> u32 {
        let min_len = self.state_len.max(self.state_len + 4);
        round_up(min_len, self.state_alignment)
    }

    pub fn batch_states_per_rank(&self) -> u64 {
        self.walkers_per_depth as u64 * (self.k_max - self.k_min + 1) as u64
    }

    pub fn validate(&self) -> Result<()> {
        if self.state_len != 72 {
            bail!("state_len must be 72 for version one");
        }
        if self.state_storage_len() != 80 {
            bail!("state_storage_len must be 80 for p888");
        }
        if self.state_alignment != 16 {
            bail!("state_alignment must be 16");
        }
        if self.state_value_pad != 128 {
            bail!("state_value_pad must be 128");
        }
        if self.move_count != 18 {
            bail!("move_count must be 18");
        }
        if self.output_dim != 1 {
            bail!("output_dim must be 1");
        }
        if self.k_min == 0 || self.k_min > self.k_max {
            bail!("invalid walk depth range");
        }
        if self.learning_rate <= 0.0 {
            bail!("learning_rate must be positive");
        }
        if self.weight_decay < 0.0 {
            bail!("weight_decay must be non-negative");
        }
        Ok(())
    }
}

fn round_up(value: u32, alignment: u32) -> u32 {
    ((value + alignment - 1) / alignment) * alignment
}
```

- [ ] **Шаг 4: написать вход командной строки**

`D:\MultiGPUTrainMLP\crates\trainer-cli\src\main.rs`:

```rust
mod config;
mod native;

use anyhow::Result;
use clap::{Parser, Subcommand};
use config::TrainerConfig;
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "mgt")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    PrintDefaultConfig,
    ValidateConfig { path: PathBuf },
    Train { config: PathBuf, output_dir: PathBuf },
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::PrintDefaultConfig => {
            let text = toml::to_string_pretty(&TrainerConfig::p888_default())?;
            println!("{text}");
        }
        Command::ValidateConfig { path } => {
            let text = std::fs::read_to_string(path)?;
            let cfg: TrainerConfig = toml::from_str(&text)?;
            cfg.validate()?;
            println!("config_ok");
        }
        Command::Train { config, output_dir } => {
            let text = std::fs::read_to_string(config)?;
            let cfg: TrainerConfig = toml::from_str(&text)?;
            cfg.validate()?;
            native::run_training(&cfg, &output_dir)?;
        }
    }
    Ok(())
}
```

- [ ] **Шаг 5: добавить временную нативную границу**

`D:\MultiGPUTrainMLP\crates\trainer-cli\src\native.rs`:

```rust
use crate::config::TrainerConfig;
use anyhow::Result;
use std::path::Path;

pub fn run_training(cfg: &TrainerConfig, output_dir: &Path) -> Result<()> {
    std::fs::create_dir_all(output_dir)?;
    let snapshot = toml::to_string_pretty(cfg)?;
    std::fs::write(output_dir.join("config.snapshot.toml"), snapshot)?;
    println!("native_training_entry_ready");
    Ok(())
}
```

- [ ] **Шаг 6: добавить тесты конфигурации**

`D:\MultiGPUTrainMLP\crates\trainer-cli\tests\config_contract.rs`:

```rust
#[path = "../src/config.rs"]
mod config;

use config::TrainerConfig;

#[test]
fn p888_default_matches_static_contract() {
    let cfg = TrainerConfig::p888_default();
    assert_eq!(cfg.state_len, 72);
    assert_eq!(cfg.state_storage_len(), 80);
    assert_eq!(cfg.state_alignment, 16);
    assert_eq!(cfg.state_value_pad, 128);
    assert_eq!(cfg.move_count, 18);
    assert_eq!(cfg.output_dim, 1);
    assert_eq!(cfg.batch_states_per_rank(), 100_021);
    assert!(cfg.validate().is_ok());
}

#[test]
fn rejects_wrong_output_dim() {
    let mut cfg = TrainerConfig::p888_default();
    cfg.output_dim = 2;
    let err = cfg.validate().unwrap_err().to_string();
    assert!(err.contains("output_dim"));
}
```

- [ ] **Шаг 7: добавить нативную сборку**

`D:\MultiGPUTrainMLP\native\CMakeLists.txt`:

```cmake
cmake_minimum_required(VERSION 3.24)
project(multigpu_train_mlp LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

option(MGT_ENABLE_CUDA "Build CUDA trainer code" ON)
option(MGT_ENABLE_NCCL "Build NCCL allreduce code" ON)

add_library(mgt_core STATIC
    src/static_contracts.cpp
)

target_include_directories(mgt_core PUBLIC include)

add_executable(test_static_contracts tests/test_static_contracts.cpp)
target_link_libraries(test_static_contracts PRIVATE mgt_core)

enable_testing()
add_test(NAME static_contracts COMMAND test_static_contracts)
```

- [ ] **Шаг 8: добавить команды сборки**

`D:\MultiGPUTrainMLP\scripts\build_native.ps1`:

```powershell
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Build = Join-Path $Root "build-native"
cmake -S (Join-Path $Root "native") -B $Build -DMGT_ENABLE_CUDA=ON
cmake --build $Build --config Release
```

`D:\MultiGPUTrainMLP\scripts\test_cpu.ps1`:

```powershell
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
cargo test --workspace
$Build = Join-Path $Root "build-native"
ctest --test-dir $Build --output-on-failure -C Release
```

- [ ] **Шаг 9: выполнить проверку**

Команды:

```powershell
cargo test --workspace
powershell -ExecutionPolicy Bypass -File D:\MultiGPUTrainMLP\scripts\build_native.ps1
powershell -ExecutionPolicy Bypass -File D:\MultiGPUTrainMLP\scripts\test_cpu.ps1
```

Ожидаемый результат:

```text
test result: ok
100% tests passed
```

- [ ] **Шаг 10: коммит**

```powershell
git add Cargo.toml crates native scripts
git commit -m "feat: add trainer project scaffold"
```

## Задача 2: Статические Контракты Данных

**Файлы:**

- Создать: `D:\MultiGPUTrainMLP\native\include\mgt\config.hpp`
- Создать: `D:\MultiGPUTrainMLP\native\include\mgt\status.hpp`
- Создать: `D:\MultiGPUTrainMLP\native\include\mgt\static_contracts.hpp`
- Создать: `D:\MultiGPUTrainMLP\native\src\static_contracts.cpp`
- Создать: `D:\MultiGPUTrainMLP\native\tests\test_static_contracts.cpp`

- [ ] **Шаг 1: задать статические размеры**

`D:\MultiGPUTrainMLP\native\include\mgt\config.hpp`:

```cpp
#pragma once

#include <cstddef>
#include <cstdint>

namespace mgt {

inline constexpr std::uint32_t kStateLen = 72;
inline constexpr std::uint32_t kStateAlignment = 16;
inline constexpr std::uint32_t kStateValuePad = 128;
inline constexpr std::uint32_t kMoveCount = 18;
inline constexpr std::uint32_t kOutputDim = 1;
inline constexpr std::uint32_t kHd1 = 2556;
inline constexpr std::uint32_t kHd2 = 218;
inline constexpr std::uint32_t kResidualBlocks = 16;

constexpr std::uint32_t RoundUp(std::uint32_t value, std::uint32_t alignment) {
    return ((value + alignment - 1U) / alignment) * alignment;
}

inline constexpr std::uint32_t kStateStorageLen =
    RoundUp(kStateLen + 4U, kStateAlignment);

static_assert(kStateStorageLen == 80);

}  // namespace mgt
```

- [ ] **Шаг 2: добавить коды состояния**

`D:\MultiGPUTrainMLP\native\include\mgt\status.hpp`:

```cpp
#pragma once

namespace mgt {

enum class Status {
    kOk = 0,
    kInvalidConfig = 1,
    kInvalidPuzzle = 2,
    kCapacityExceeded = 3,
    kCudaFailure = 4,
    kNcclFailure = 5,
    kIoFailure = 6
};

}  // namespace mgt
```

- [ ] **Шаг 3: добавить структуры данных**

`D:\MultiGPUTrainMLP\native\include\mgt\static_contracts.hpp`:

```cpp
#pragma once

#include "mgt/config.hpp"
#include "mgt/status.hpp"
#include <cstdint>

namespace mgt {

using StateValue = std::uint8_t;

struct alignas(16) TrainState80 {
    StateValue v[kStateStorageLen];
};

struct alignas(16) WalkMeta {
    std::uint32_t depth;
    std::uint32_t last_move;
    std::uint64_t rng_counter;
};

struct alignas(16) LossStats {
    float loss_sum;
    float loss_max;
    std::uint32_t sample_count;
    std::uint32_t overflow;
};

struct alignas(32) TensorBlockHeader {
    std::uint64_t offset_bytes;
    std::uint64_t size_bytes;
    std::uint32_t rows;
    std::uint32_t cols;
    std::uint32_t dtype;
    std::uint32_t reserved;
};

Status ValidateStaticContracts();

}  // namespace mgt
```

- [ ] **Шаг 4: добавить проверки**

`D:\MultiGPUTrainMLP\native\src\static_contracts.cpp`:

```cpp
#include "mgt/static_contracts.hpp"

namespace mgt {

static_assert(sizeof(TrainState80) == 80);
static_assert(alignof(TrainState80) == 16);
static_assert(sizeof(WalkMeta) == 16);
static_assert(alignof(WalkMeta) == 16);
static_assert(sizeof(LossStats) == 16);
static_assert(alignof(LossStats) == 16);
static_assert(sizeof(TensorBlockHeader) == 32);
static_assert(alignof(TensorBlockHeader) == 32);

Status ValidateStaticContracts() {
    if (sizeof(TrainState80) != 80 || alignof(TrainState80) != 16) {
        return Status::kInvalidConfig;
    }
    if (sizeof(WalkMeta) != 16 || alignof(WalkMeta) != 16) {
        return Status::kInvalidConfig;
    }
    if (sizeof(LossStats) != 16 || alignof(LossStats) != 16) {
        return Status::kInvalidConfig;
    }
    if (sizeof(TensorBlockHeader) != 32 || alignof(TensorBlockHeader) != 32) {
        return Status::kInvalidConfig;
    }
    return Status::kOk;
}

}  // namespace mgt
```

- [ ] **Шаг 5: добавить тест**

`D:\MultiGPUTrainMLP\native\tests\test_static_contracts.cpp`:

```cpp
#include "mgt/static_contracts.hpp"
#include <cstdlib>

int main() {
    if (mgt::kStateLen != 72) return EXIT_FAILURE;
    if (mgt::kStateStorageLen != 80) return EXIT_FAILURE;
    if (mgt::kMoveCount != 18) return EXIT_FAILURE;
    if (mgt::kOutputDim != 1) return EXIT_FAILURE;
    if (sizeof(mgt::TrainState80) != 80) return EXIT_FAILURE;
    if (alignof(mgt::TrainState80) != 16) return EXIT_FAILURE;
    if (sizeof(mgt::WalkMeta) != 16) return EXIT_FAILURE;
    if (sizeof(mgt::LossStats) != 16) return EXIT_FAILURE;
    if (sizeof(mgt::TensorBlockHeader) != 32) return EXIT_FAILURE;
    if (mgt::ValidateStaticContracts() != mgt::Status::kOk) return EXIT_FAILURE;
    return EXIT_SUCCESS;
}
```

- [ ] **Шаг 6: выполнить проверку и коммит**

```powershell
powershell -ExecutionPolicy Bypass -File D:\MultiGPUTrainMLP\scripts\build_native.ps1
powershell -ExecutionPolicy Bypass -File D:\MultiGPUTrainMLP\scripts\test_cpu.ps1
git add native
git commit -m "feat: add static data contracts"
```

## Задача 3: Загрузка Головоломки И Цели

**Файлы:**

- Создать: `D:\MultiGPUTrainMLP\native\include\mgt\puzzle_io.hpp`
- Создать: `D:\MultiGPUTrainMLP\native\src\puzzle_io.cpp`
- Создать: `D:\MultiGPUTrainMLP\native\tests\test_puzzle_io.cpp`
- Создать: `D:\MultiGPUTrainMLP\native\tests\fixtures\p888.json`
- Создать: `D:\MultiGPUTrainMLP\native\tests\fixtures\p888-target.bin`
- Изменить: `D:\MultiGPUTrainMLP\native\CMakeLists.txt`

- [ ] **Шаг 1: добавить фикстуру таблицы ходов**

Фикстура должна содержать 18 перестановок длины 72. Для первого теста достаточно таблицы, где все ходы являются тождественными перестановками, потому что этот тест проверяет формат, ширину 80 и хвост `72..79`.

`D:\MultiGPUTrainMLP\native\tests\fixtures\p888.json`:

```json
{
  "group_id": 888,
  "state_len": 72,
  "move_count": 18,
  "moves": [
    [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71],
    [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71],
    [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71],
    [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71],
    [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71],
    [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71],
    [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71],
    [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71],
    [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71],
    [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71],
    [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71],
    [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71],
    [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71],
    [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71],
    [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71],
    [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71],
    [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71],
    [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71]
  ]
}
```

`D:\MultiGPUTrainMLP\native\tests\fixtures\p888-target.bin` записать 72 байта со значениями `0..71`.

- [ ] **Шаг 2: добавить интерфейс загрузки**

`D:\MultiGPUTrainMLP\native\include\mgt\puzzle_io.hpp`:

```cpp
#pragma once

#include "mgt/static_contracts.hpp"
#include <array>
#include <filesystem>

namespace mgt {

struct PuzzleDefinition {
    std::array<TrainState80, kMoveCount> moves;
    TrainState80 target;
};

Status LoadPuzzleDefinition(const std::filesystem::path& group_json,
                            const std::filesystem::path& target_bin,
                            PuzzleDefinition* out);

}  // namespace mgt
```

- [ ] **Шаг 3: реализовать строгую загрузку**

`D:\MultiGPUTrainMLP\native\src\puzzle_io.cpp`:

```cpp
#include "mgt/puzzle_io.hpp"
#include <cctype>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

namespace mgt {
namespace {

std::vector<int> ExtractIntegers(const std::string& text) {
    std::vector<int> values;
    std::size_t i = 0;
    while (i < text.size()) {
        while (i < text.size() && !std::isdigit(static_cast<unsigned char>(text[i])) && text[i] != '-') {
            ++i;
        }
        if (i == text.size()) break;
        int sign = 1;
        if (text[i] == '-') {
            sign = -1;
            ++i;
        }
        int value = 0;
        while (i < text.size() && std::isdigit(static_cast<unsigned char>(text[i]))) {
            value = value * 10 + (text[i] - '0');
            ++i;
        }
        values.push_back(sign * value);
    }
    return values;
}

}  // namespace

Status LoadPuzzleDefinition(const std::filesystem::path& group_json,
                            const std::filesystem::path& target_bin,
                            PuzzleDefinition* out) {
    if (out == nullptr) return Status::kInvalidPuzzle;

    std::ifstream json_file(group_json, std::ios::binary);
    if (!json_file) return Status::kIoFailure;
    const std::string text((std::istreambuf_iterator<char>(json_file)),
                           std::istreambuf_iterator<char>());

    const std::vector<int> values = ExtractIntegers(text);
    const std::size_t required = 3 + static_cast<std::size_t>(kMoveCount) * kStateLen;
    if (values.size() < required) return Status::kInvalidPuzzle;
    if (values[0] != 888 || values[1] != static_cast<int>(kStateLen) ||
        values[2] != static_cast<int>(kMoveCount)) {
        return Status::kInvalidPuzzle;
    }

    std::size_t pos = 3;
    for (std::uint32_t move = 0; move < kMoveCount; ++move) {
        for (std::uint32_t i = 0; i < kStateLen; ++i) {
            const int v = values[pos++];
            if (v < 0 || v >= static_cast<int>(kStateLen)) return Status::kInvalidPuzzle;
            out->moves[move].v[i] = static_cast<StateValue>(v);
        }
        for (std::uint32_t i = kStateLen; i < kStateStorageLen; ++i) {
            out->moves[move].v[i] = static_cast<StateValue>(i);
        }
    }

    std::ifstream target_file(target_bin, std::ios::binary);
    if (!target_file) return Status::kIoFailure;
    target_file.read(reinterpret_cast<char*>(out->target.v), kStateLen);
    if (target_file.gcount() != static_cast<std::streamsize>(kStateLen)) {
        return Status::kInvalidPuzzle;
    }
    for (std::uint32_t i = kStateLen; i < kStateStorageLen; ++i) {
        out->target.v[i] = 0;
    }

    return Status::kOk;
}

}  // namespace mgt
```

- [ ] **Шаг 4: добавить тест загрузки**

`D:\MultiGPUTrainMLP\native\tests\test_puzzle_io.cpp`:

```cpp
#include "mgt/puzzle_io.hpp"
#include <cstdlib>
#include <filesystem>

int main() {
    const std::filesystem::path fixture_dir = "native/tests/fixtures";
    mgt::PuzzleDefinition puzzle{};
    const auto status = mgt::LoadPuzzleDefinition(
        fixture_dir / "p888.json",
        fixture_dir / "p888-target.bin",
        &puzzle);
    if (status != mgt::Status::kOk) return EXIT_FAILURE;
    for (std::uint32_t i = 0; i < mgt::kStateLen; ++i) {
        if (puzzle.target.v[i] != static_cast<mgt::StateValue>(i)) return EXIT_FAILURE;
    }
    for (std::uint32_t i = mgt::kStateLen; i < mgt::kStateStorageLen; ++i) {
        if (puzzle.target.v[i] != 0) return EXIT_FAILURE;
        if (puzzle.moves[0].v[i] != static_cast<mgt::StateValue>(i)) return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
```

- [ ] **Шаг 5: подключить тесты**

В `D:\MultiGPUTrainMLP\native\CMakeLists.txt` добавить:

```cmake
target_sources(mgt_core PRIVATE src/puzzle_io.cpp)

add_executable(test_puzzle_io tests/test_puzzle_io.cpp)
target_link_libraries(test_puzzle_io PRIVATE mgt_core)
add_test(NAME puzzle_io COMMAND test_puzzle_io)
```

- [ ] **Шаг 6: выполнить проверку и коммит**

```powershell
powershell -ExecutionPolicy Bypass -File D:\MultiGPUTrainMLP\scripts\build_native.ps1
powershell -ExecutionPolicy Bypass -File D:\MultiGPUTrainMLP\scripts\test_cpu.ps1
git add native
git commit -m "feat: load puzzle and target contracts"
```

## Задача 4: Раскладка Модели И Экспорт Весов

**Файлы:**

- Создать: `D:\MultiGPUTrainMLP\native\include\mgt\model_layout.hpp`
- Создать: `D:\MultiGPUTrainMLP\native\src\model_layout.cpp`
- Создать: `D:\MultiGPUTrainMLP\native\tests\test_model_layout.cpp`
- Создать: `D:\MultiGPUTrainMLP\native\include\mgt\weight_export.hpp`
- Создать: `D:\MultiGPUTrainMLP\native\src\weight_export.cpp`
- Создать: `D:\MultiGPUTrainMLP\native\tests\test_weight_export.cpp`
- Изменить: `D:\MultiGPUTrainMLP\native\CMakeLists.txt`

- [ ] **Шаг 1: задать блоки параметров**

`D:\MultiGPUTrainMLP\native\include\mgt\model_layout.hpp`:

```cpp
#pragma once

#include "mgt/static_contracts.hpp"
#include <array>
#include <cstddef>

namespace mgt {

enum class DType : std::uint32_t {
    kFloat32 = 1
};

enum class ParamBlock : std::uint32_t {
    kInputTable = 0,
    kInputBias = 1,
    kHiddenWeight = 2,
    kHiddenBias = 3,
    kResidualStart = 4
};

inline constexpr std::uint32_t kResidualLinearBlocks = kResidualBlocks * 2U;
inline constexpr std::uint32_t kParamBlockCount = 4U + kResidualLinearBlocks * 2U + 2U;

struct ModelLayout {
    std::array<TensorBlockHeader, kParamBlockCount> blocks;
    std::uint64_t total_bytes;
    std::uint64_t total_params;
};

ModelLayout BuildModelLayout();

}  // namespace mgt
```

- [ ] **Шаг 2: реализовать расчет смещений**

`D:\MultiGPUTrainMLP\native\src\model_layout.cpp`:

```cpp
#include "mgt/model_layout.hpp"

namespace mgt {
namespace {

std::uint64_t Align64(std::uint64_t value) {
    return ((value + 63ULL) / 64ULL) * 64ULL;
}

void AddBlock(ModelLayout* layout, std::uint32_t* index, std::uint32_t rows, std::uint32_t cols) {
    const std::uint64_t size = static_cast<std::uint64_t>(rows) * cols * sizeof(float);
    layout->blocks[*index] = TensorBlockHeader{
        layout->total_bytes,
        size,
        rows,
        cols,
        static_cast<std::uint32_t>(DType::kFloat32),
        0};
    layout->total_bytes = Align64(layout->total_bytes + size);
    layout->total_params += static_cast<std::uint64_t>(rows) * cols;
    ++(*index);
}

}  // namespace

ModelLayout BuildModelLayout() {
    ModelLayout layout{};
    std::uint32_t index = 0;
    AddBlock(&layout, &index, kStateLen * kStateValuePad, kHd1);
    AddBlock(&layout, &index, 1, kHd1);
    AddBlock(&layout, &index, kHd1, kHd2);
    AddBlock(&layout, &index, 1, kHd2);
    for (std::uint32_t block = 0; block < kResidualBlocks; ++block) {
        AddBlock(&layout, &index, kHd2, kHd2);
        AddBlock(&layout, &index, 1, kHd2);
        AddBlock(&layout, &index, kHd2, kHd2);
        AddBlock(&layout, &index, 1, kHd2);
    }
    AddBlock(&layout, &index, kHd2, kOutputDim);
    AddBlock(&layout, &index, 1, kOutputDim);
    return layout;
}

}  // namespace mgt
```

- [ ] **Шаг 3: проверить число параметров**

`D:\MultiGPUTrainMLP\native\tests\test_model_layout.cpp`:

```cpp
#include "mgt/model_layout.hpp"
#include <cstdlib>

int main() {
    const auto layout = mgt::BuildModelLayout();
    if (layout.blocks[0].rows != mgt::kStateLen * mgt::kStateValuePad) return EXIT_FAILURE;
    if (layout.blocks[0].cols != mgt::kHd1) return EXIT_FAILURE;
    if (layout.blocks[mgt::kParamBlockCount - 2].cols != mgt::kOutputDim) return EXIT_FAILURE;
    if (layout.total_params == 0) return EXIT_FAILURE;
    if (layout.total_bytes % 64 != 0) return EXIT_FAILURE;
    for (std::uint32_t i = 1; i < mgt::kParamBlockCount; ++i) {
        if (layout.blocks[i].offset_bytes <= layout.blocks[i - 1].offset_bytes) return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
```

- [ ] **Шаг 4: добавить экспорт весов**

`D:\MultiGPUTrainMLP\native\include\mgt\weight_export.hpp`:

```cpp
#pragma once

#include "mgt/model_layout.hpp"
#include <filesystem>
#include <span>

namespace mgt {

Status ExportInferenceWeights(const std::filesystem::path& output_dir,
                              const ModelLayout& layout,
                              std::span<const float> weights);

}  // namespace mgt
```

`D:\MultiGPUTrainMLP\native\src\weight_export.cpp`:

```cpp
#include "mgt/weight_export.hpp"
#include <filesystem>
#include <fstream>

namespace mgt {

Status ExportInferenceWeights(const std::filesystem::path& output_dir,
                              const ModelLayout& layout,
                              std::span<const float> weights) {
    const std::uint64_t expected = layout.total_params;
    if (weights.size() != expected) return Status::kInvalidConfig;
    std::error_code ec;
    std::filesystem::create_directories(output_dir, ec);
    if (ec) return Status::kIoFailure;

    std::ofstream data(output_dir / "weights.f32.bin", std::ios::binary);
    if (!data) return Status::kIoFailure;
    data.write(reinterpret_cast<const char*>(weights.data()),
               static_cast<std::streamsize>(weights.size() * sizeof(float)));
    if (!data) return Status::kIoFailure;

    std::ofstream manifest(output_dir / "manifest.json", std::ios::binary);
    if (!manifest) return Status::kIoFailure;
    manifest
        << "{\n"
        << "  \"format\": \"stream1_weights\",\n"
        << "  \"version\": 1,\n"
        << "  \"group_id\": 888,\n"
        << "  \"target_id\": 0,\n"
        << "  \"state_len\": " << kStateLen << ",\n"
        << "  \"state_storage_len\": " << kStateStorageLen << ",\n"
        << "  \"state_value_pad\": " << kStateValuePad << ",\n"
        << "  \"move_count\": " << kMoveCount << ",\n"
        << "  \"output_dim\": " << kOutputDim << ",\n"
        << "  \"hd1\": " << kHd1 << ",\n"
        << "  \"hd2\": " << kHd2 << ",\n"
        << "  \"residual_blocks\": " << kResidualBlocks << ",\n"
        << "  \"dtype\": \"float32\",\n"
        << "  \"data\": \"weights.f32.bin\",\n"
        << "  \"total_params\": " << layout.total_params << "\n"
        << "}\n";
    return manifest ? Status::kOk : Status::kIoFailure;
}

}  // namespace mgt
```

- [ ] **Шаг 5: добавить тест экспорта**

`D:\MultiGPUTrainMLP\native\tests\test_weight_export.cpp`:

```cpp
#include "mgt/weight_export.hpp"
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <vector>

int main() {
    const auto layout = mgt::BuildModelLayout();
    std::vector<float> weights(layout.total_params, 0.125f);
    const std::filesystem::path out = "build-native/test_weight_export";
    const auto status = mgt::ExportInferenceWeights(out, layout, weights);
    if (status != mgt::Status::kOk) return EXIT_FAILURE;
    if (!std::filesystem::exists(out / "manifest.json")) return EXIT_FAILURE;
    if (!std::filesystem::exists(out / "weights.f32.bin")) return EXIT_FAILURE;
    if (std::filesystem::file_size(out / "weights.f32.bin") != weights.size() * sizeof(float)) {
        return EXIT_FAILURE;
    }
    std::ifstream manifest(out / "manifest.json", std::ios::binary);
    const std::string text((std::istreambuf_iterator<char>(manifest)),
                           std::istreambuf_iterator<char>());
    if (text.find("\"output_dim\": 1") == std::string::npos) return EXIT_FAILURE;
    return EXIT_SUCCESS;
}
```

- [ ] **Шаг 6: подключить и проверить**

В `D:\MultiGPUTrainMLP\native\CMakeLists.txt` добавить:

```cmake
target_sources(mgt_core PRIVATE src/model_layout.cpp src/weight_export.cpp)

add_executable(test_model_layout tests/test_model_layout.cpp)
target_link_libraries(test_model_layout PRIVATE mgt_core)
add_test(NAME model_layout COMMAND test_model_layout)

add_executable(test_weight_export tests/test_weight_export.cpp)
target_link_libraries(test_weight_export PRIVATE mgt_core)
add_test(NAME weight_export COMMAND test_weight_export)
```

Команды:

```powershell
powershell -ExecutionPolicy Bypass -File D:\MultiGPUTrainMLP\scripts\build_native.ps1
powershell -ExecutionPolicy Bypass -File D:\MultiGPUTrainMLP\scripts\test_cpu.ps1
git add native
git commit -m "feat: add model layout and weight export"
```

## Задача 5: Эталон На Процессоре

**Файлы:**

- Создать: `D:\MultiGPUTrainMLP\native\include\mgt\random_walk.hpp`
- Создать: `D:\MultiGPUTrainMLP\native\src\random_walk_cpu.cpp`
- Создать: `D:\MultiGPUTrainMLP\native\tests\test_random_walk_cpu.cpp`
- Создать: `D:\MultiGPUTrainMLP\native\include\mgt\mlp_cpu_ref.hpp`
- Создать: `D:\MultiGPUTrainMLP\native\src\mlp_cpu_ref.cpp`
- Создать: `D:\MultiGPUTrainMLP\native\tests\test_mlp_cpu_ref.cpp`
- Изменить: `D:\MultiGPUTrainMLP\native\CMakeLists.txt`

- [ ] **Шаг 1: реализовать детерминированные блуждания**

Код генератора должен использовать `seed = hash(base_seed, epoch, step, global_rank, sample_index)`, запрещать немедленный обратный ход и обнулять хвост состояния `72..79`.

Минимальный интерфейс:

```cpp
namespace mgt {

struct WalkRequest {
    std::uint64_t base_seed;
    std::uint64_t epoch;
    std::uint64_t step;
    std::uint32_t global_rank;
    std::uint32_t k_min;
    std::uint32_t k_max;
    std::uint32_t sample_count;
};

Status GenerateRandomWalksCpu(const PuzzleDefinition& puzzle,
                              const WalkRequest& request,
                              TrainState80* states,
                              float* labels,
                              WalkMeta* meta);

}
```

Проверка:

```powershell
ctest --test-dir D:\MultiGPUTrainMLP\build-native -R random_walk_cpu --output-on-failure -C Release
```

Ожидаемый результат:

```text
100% tests passed
```

- [ ] **Шаг 2: реализовать малую МЛП на процессоре**

Эталон должен поддерживать размеры, передаваемые в тесте: `state_len=4`, `state_value_pad=8`, `hd1=5`, `hd2=3`, `residual_blocks=1`, `output_dim=1`. Проверяются прямой проход, обратный проход по конечной разности и АдамВ при `weight_decay=0`.

Проверка:

```powershell
ctest --test-dir D:\MultiGPUTrainMLP\build-native -R mlp_cpu_ref --output-on-failure -C Release
```

Ожидаемый результат:

```text
100% tests passed
```

- [ ] **Шаг 3: общий прогон и коммит**

```powershell
powershell -ExecutionPolicy Bypass -File D:\MultiGPUTrainMLP\scripts\build_native.ps1
powershell -ExecutionPolicy Bypass -File D:\MultiGPUTrainMLP\scripts\test_cpu.ps1
git add native
git commit -m "feat: add cpu reference training checks"
```

## Задача 6: Однокарточный КУДА-Путь

**Файлы:**

- Создать: `D:\MultiGPUTrainMLP\native\cuda\device_context.cuh`
- Создать: `D:\MultiGPUTrainMLP\native\cuda\device_context.cu`
- Создать: `D:\MultiGPUTrainMLP\native\cuda\memory_plan.cuh`
- Создать: `D:\MultiGPUTrainMLP\native\cuda\memory_plan.cu`
- Создать: `D:\MultiGPUTrainMLP\native\cuda\random_walk_kernel.cuh`
- Создать: `D:\MultiGPUTrainMLP\native\cuda\random_walk_kernel.cu`
- Создать: `D:\MultiGPUTrainMLP\native\cuda\mlp_forward.cuh`
- Создать: `D:\MultiGPUTrainMLP\native\cuda\mlp_forward.cu`
- Создать: `D:\MultiGPUTrainMLP\native\cuda\mlp_backward.cuh`
- Создать: `D:\MultiGPUTrainMLP\native\cuda\mlp_backward.cu`
- Создать: `D:\MultiGPUTrainMLP\native\cuda\adamw.cuh`
- Создать: `D:\MultiGPUTrainMLP\native\cuda\adamw.cu`
- Создать: `D:\MultiGPUTrainMLP\native\tests\test_cuda_smoke.cpp`
- Создать: `D:\MultiGPUTrainMLP\scripts\test_single_gpu.ps1`
- Изменить: `D:\MultiGPUTrainMLP\native\CMakeLists.txt`

- [ ] **Шаг 1: добавить владение устройством**

Контракт `DeviceContext`: один владелец устройства, один основной поток, события фаз, все выделения до цикла обучения.

Проверка в `test_cuda_smoke`:

```text
device_count >= 1
context.device_id == requested_device_id
stream != nullptr
```

- [ ] **Шаг 2: добавить план памяти**

План должен считать:

```text
weights_bytes
optimizer_m_bytes
optimizer_v_bytes
gradient_bytes
states_bytes
labels_bytes
walk_meta_bytes
forward_scratch_bytes
backward_scratch_bytes
allreduce_scratch_bytes
checkpoint_scratch_bytes
training_scratch_bytes = max(phase_bytes)
```

Проверка:

```text
никакая емкость не равна нулю
training_scratch_bytes >= states_bytes + labels_bytes + walk_meta_bytes
```

- [ ] **Шаг 3: реализовать КУДА-генерацию блужданий**

Ядро версии один:

```text
один поток генерирует одно состояние
глубина берется из диапазона K_MIN..K_MAX
немедленный обратный ход исключается через inverse_move[18]
хвост v[72..79] обнуляется
label = depth
```

Сравнение:

```text
для batch=256 CPU и GPU совпадают по глубинам и хвосту состояния
```

- [ ] **Шаг 4: реализовать прямой проход**

Первый вариант допускает простые КУДА-ядра без КАТЛАСС для малой проверки, но интерфейс должен принимать уже рассчитанный `ModelLayout`, чтобы замена матричных блоков на КАТЛАСС не меняла внешний контракт.

Проверка:

```text
на малой модели max_abs(cpu - gpu) <= 1e-4
```

- [ ] **Шаг 5: реализовать обратный проход и АдамВ**

Первый вариант должен пройти малую численную проверку:

```text
max_abs(cpu_grad - gpu_grad) <= 1e-3
max_abs(cpu_weight_after_adamw - gpu_weight_after_adamw) <= 1e-5
```

`weight_decay` в тесте равен `0`.

- [ ] **Шаг 6: запретить выделения в цикле**

В `train_loop` все указатели передаются из `MemoryPlan`. Внутри функций шага запрещены вызовы:

```text
cudaMalloc
cudaFree
cudaMallocAsync
cudaFreeAsync
new
delete
malloc
free
```

Проверка:

```powershell
rg "cudaMalloc|cudaFree|cudaMallocAsync|cudaFreeAsync|new |delete |malloc|free" D:\MultiGPUTrainMLP\native\cuda D:\MultiGPUTrainMLP\native\src
```

Ожидаемый результат: совпадения есть только в файлах подготовки памяти и тестах.

- [ ] **Шаг 7: прогон и коммит**

```powershell
powershell -ExecutionPolicy Bypass -File D:\MultiGPUTrainMLP\scripts\build_native.ps1
powershell -ExecutionPolicy Bypass -File D:\MultiGPUTrainMLP\scripts\test_single_gpu.ps1
git add native scripts
git commit -m "feat: add single gpu cuda training path"
```

## Задача 7: Цикл Тренировки И Журналы

**Файлы:**

- Создать: `D:\MultiGPUTrainMLP\native\include\mgt\train_loop.hpp`
- Создать: `D:\MultiGPUTrainMLP\native\src\train_loop.cpp`
- Создать: `D:\MultiGPUTrainMLP\native\src\main_native.cpp`
- Изменить: `D:\MultiGPUTrainMLP\crates\trainer-cli\src\native.rs`
- Изменить: `D:\MultiGPUTrainMLP\native\CMakeLists.txt`

- [ ] **Шаг 1: задать публичный интерфейс**

```cpp
namespace mgt {

struct TrainRunConfig {
    std::uint32_t device_id;
    std::uint32_t world_size;
    std::uint32_t global_rank;
    std::uint32_t local_rank;
    std::uint64_t epochs;
    std::uint64_t steps_per_epoch;
    std::uint64_t base_seed;
    std::uint32_t walkers_per_depth;
    std::uint32_t k_min;
    std::uint32_t k_max;
    float learning_rate;
    float weight_decay;
    const char* group_json;
    const char* target_bin;
    const char* output_dir;
};

Status RunTraining(const TrainRunConfig& config);

}
```

- [ ] **Шаг 2: реализовать порядок фаз**

Порядок на каждом шаге:

```text
generate
forward
loss
backward
allreduce_or_skip
adamw
export_if_due
log_step
```

Лог строки:

```text
run_id rank device epoch step phase milliseconds batch_states loss memory_bytes status notes
```

- [ ] **Шаг 3: связать Раст с нативным входом**

Для первой версии Раст запускает `mgt_native_train.exe` как дочерний процесс и передает путь к файлу конфигурации. Это сохраняет простую сборку на Windows и в контейнере; прямая связь через библиотеку добавляется только после стабильности нативного входа.

Проверка:

```powershell
cargo run -p trainer-cli -- print-default-config
cargo run -p trainer-cli -- train D:\MultiGPUTrainMLP\run.toml D:\MultiGPUTrainMLP\runs\smoke
```

Ожидаемый результат:

```text
config_ok
native rank starts
```

- [ ] **Шаг 4: коммит**

```powershell
git add crates native
git commit -m "feat: add native training loop"
```

## Задача 8: NCCL-Сведение И Ранговый Запуск

**Файлы:**

- Создать: `D:\MultiGPUTrainMLP\native\cuda\allreduce_nccl.cuh`
- Создать: `D:\MultiGPUTrainMLP\native\cuda\allreduce_nccl.cu`
- Создать: `D:\MultiGPUTrainMLP\native\tests\test_nccl_order.cpp`
- Изменить: `D:\MultiGPUTrainMLP\native\src\train_loop.cpp`
- Изменить: `D:\MultiGPUTrainMLP\native\CMakeLists.txt`

- [ ] **Шаг 1: задать контракт сведения**

```text
collective_seq начинается с 0
каждый шаг увеличивает collective_seq на 1
сводится один плоский буфер градиентов
после сведения буфер делится на WORLD_SIZE
при WORLD_SIZE == 1 функция возвращает kOk без NCCL
```

- [ ] **Шаг 2: добавить проверку порядка**

Каждый ранг пишет:

```text
rank=<global_rank> collective_seq=<n> bytes=<gradient_bytes> status=begin
rank=<global_rank> collective_seq=<n> bytes=<gradient_bytes> status=end
```

Тест `test_nccl_order` проверяет однокарточный путь без NCCL и синтетический журнал для двух рангов.

- [ ] **Шаг 3: подключить NCCL в сборке**

В CMake добавить условное подключение:

```cmake
if(MGT_ENABLE_NCCL)
    find_library(NCCL_LIB nccl)
    if(NCCL_LIB)
        target_link_libraries(mgt_core PUBLIC ${NCCL_LIB})
        target_compile_definitions(mgt_core PUBLIC MGT_ENABLE_NCCL=1)
    endif()
endif()
```

- [ ] **Шаг 4: коммит**

```powershell
powershell -ExecutionPolicy Bypass -File D:\MultiGPUTrainMLP\scripts\build_native.ps1
powershell -ExecutionPolicy Bypass -File D:\MultiGPUTrainMLP\scripts\test_cpu.ps1
git add native
git commit -m "feat: add nccl gradient allreduce"
```

## Задача 9: Контейнер Одной Видеокарты И Профиль

**Файлы:**

- Создать: `D:\MultiGPUTrainMLP\docker\Dockerfile.cuda`
- Создать: `D:\MultiGPUTrainMLP\docker\run_single_gpu.sh`
- Создать: `D:\MultiGPUTrainMLP\scripts\profile_single_gpu.ps1`
- Создать: `D:\MultiGPUTrainMLP\docs\runbooks\one_gpu_container.md`

- [ ] **Шаг 1: добавить контейнер**

`D:\MultiGPUTrainMLP\docker\Dockerfile.cuda`:

```dockerfile
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake ninja-build curl git ca-certificates pkg-config \
    && rm -rf /var/lib/apt/lists/*

RUN curl https://sh.rustup.rs -sSf | sh -s -- -y --profile minimal
ENV PATH="/root/.cargo/bin:${PATH}"

WORKDIR /workspace
COPY . /workspace
RUN cmake -S native -B build-native -G Ninja -DMGT_ENABLE_CUDA=ON -DMGT_ENABLE_NCCL=OFF
RUN cmake --build build-native --config Release
RUN cargo test --workspace
```

- [ ] **Шаг 2: добавить запуск**

`D:\MultiGPUTrainMLP\docker\run_single_gpu.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail
./build-native/mgt_native_train \
  --device-id 0 \
  --world-size 1 \
  --global-rank 0 \
  --local-rank 0 \
  --steps 32 \
  --output-dir runs/docker-one-gpu
```

- [ ] **Шаг 3: добавить профиль**

`D:\MultiGPUTrainMLP\scripts\profile_single_gpu.ps1`:

```powershell
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
docker build -f (Join-Path $Root "docker/Dockerfile.cuda") -t mgt-native-trainer:cuda $Root
docker run --rm --gpus all -v "${Root}:/workspace/out" mgt-native-trainer:cuda bash docker/run_single_gpu.sh
```

- [ ] **Шаг 4: проверить и коммит**

```powershell
docker build -f D:\MultiGPUTrainMLP\docker\Dockerfile.cuda -t mgt-native-trainer:cuda D:\MultiGPUTrainMLP
docker run --rm --gpus all mgt-native-trainer:cuda bash docker/run_single_gpu.sh
git add docker scripts docs
git commit -m "feat: add one gpu container profile path"
```

## Задача 10: Кагл На Двух Т4

**Файлы:**

- Создать: `D:\MultiGPUTrainMLP\kaggle\kernel\run_2xt4.sh`
- Создать: `D:\MultiGPUTrainMLP\kaggle\kernel\kernel.json`
- Создать: `D:\MultiGPUTrainMLP\docs\runbooks\kaggle_2xt4.md`

- [ ] **Шаг 1: добавить запуск двух рангов**

`D:\MultiGPUTrainMLP\kaggle\kernel\run_2xt4.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail
export NCCL_DEBUG=INFO
export CUDA_VISIBLE_DEVICES=0,1
cmake -S native -B build-native -DMGT_ENABLE_CUDA=ON -DMGT_ENABLE_NCCL=ON
cmake --build build-native --config Release
./build-native/mgt_native_train --device-id 0 --world-size 2 --global-rank 0 --local-rank 0 --steps 64 --output-dir runs/kaggle-rank0 &
pid0=$!
./build-native/mgt_native_train --device-id 1 --world-size 2 --global-rank 1 --local-rank 1 --steps 64 --output-dir runs/kaggle-rank1 &
pid1=$!
wait "$pid0"
wait "$pid1"
grep "collective_seq" runs/kaggle-rank0/train.log
grep "collective_seq" runs/kaggle-rank1/train.log
```

- [ ] **Шаг 2: добавить описание ядра**

`D:\MultiGPUTrainMLP\kaggle\kernel\kernel.json`:

```json
{
  "id": "local/native-multigpu-mlp-trainer",
  "title": "native-multigpu-mlp-trainer",
  "code_file": "run_2xt4.sh",
  "language": "bash",
  "kernel_type": "script",
  "is_private": true,
  "enable_gpu": true
}
```

- [ ] **Шаг 3: критерии успешного Кагл-прогона**

В `D:\MultiGPUTrainMLP\docs\runbooks\kaggle_2xt4.md` записать:

```text
Успешный прогон содержит:
1. оба ранга стартовали на разных устройствах;
2. оба ранга записали одинаковые номера collective_seq;
3. loss конечного шага не больше loss первого шага на коротком запуске;
4. главный ранг записал manifest.json и weights.f32.bin;
5. веса читаются проверкой совместимости быстрого поиска.
```

- [ ] **Шаг 4: коммит**

```powershell
git add kaggle docs
git commit -m "feat: add kaggle two t4 run path"
```

## Задача 11: Кластерные Запуски А100 И Многосерверный Режим

**Файлы:**

- Создать: `D:\MultiGPUTrainMLP\hpc\run_8xa100.sh`
- Создать: `D:\MultiGPUTrainMLP\hpc\run_multinode.sh`
- Создать: `D:\MultiGPUTrainMLP\docs\runbooks\hpc_a100.md`

- [ ] **Шаг 1: добавить запуск восьми А100**

`D:\MultiGPUTrainMLP\hpc\run_8xa100.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail
export NCCL_DEBUG=INFO
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
for rank in 0 1 2 3 4 5 6 7; do
  ./build-native/mgt_native_train \
    --device-id "$rank" \
    --world-size 8 \
    --global-rank "$rank" \
    --local-rank "$rank" \
    --steps 256 \
    --output-dir "runs/a100-rank${rank}" &
done
wait
```

- [ ] **Шаг 2: добавить многосерверный шаблон**

`D:\MultiGPUTrainMLP\hpc\run_multinode.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail
: "${MASTER_ADDR:?MASTER_ADDR is required}"
: "${MASTER_PORT:?MASTER_PORT is required}"
: "${NODE_RANK:?NODE_RANK is required}"
: "${NODES:?NODES is required}"
: "${GPUS_PER_NODE:?GPUS_PER_NODE is required}"
export NCCL_DEBUG=INFO
world_size=$((NODES * GPUS_PER_NODE))
for local_rank in $(seq 0 $((GPUS_PER_NODE - 1))); do
  global_rank=$((NODE_RANK * GPUS_PER_NODE + local_rank))
  ./build-native/mgt_native_train \
    --master-addr "$MASTER_ADDR" \
    --master-port "$MASTER_PORT" \
    --device-id "$local_rank" \
    --world-size "$world_size" \
    --global-rank "$global_rank" \
    --local-rank "$local_rank" \
    --steps 256 \
    --output-dir "runs/node${NODE_RANK}-rank${global_rank}" &
done
wait
```

- [ ] **Шаг 3: коммит**

```powershell
git add hpc docs
git commit -m "feat: add hpc launch scripts"
```

## Задача 12: Финальная Проверка Версии Один

**Файлы:**

- Изменить: `D:\MultiGPUTrainMLP\docs\superpowers\specs\2026-07-01-native-multigpu-mlp-trainer-design.md`, если фактическая реализация уточнила контракт.
- Создать: `D:\MultiGPUTrainMLP\docs\runbooks\v1_acceptance.md`

- [ ] **Шаг 1: выполнить локальные проверки**

```powershell
cargo test --workspace
powershell -ExecutionPolicy Bypass -File D:\MultiGPUTrainMLP\scripts\build_native.ps1
powershell -ExecutionPolicy Bypass -File D:\MultiGPUTrainMLP\scripts\test_cpu.ps1
rg "cudaMalloc|cudaFree|cudaMallocAsync|cudaFreeAsync" D:\MultiGPUTrainMLP\native
```

Ожидаемый результат:

```text
Раст-тесты прошли.
Си++-тесты прошли.
Выделения памяти найдены только в подготовке памяти.
```

- [ ] **Шаг 2: выполнить однокарточный контейнерный прогон**

```powershell
powershell -ExecutionPolicy Bypass -File D:\MultiGPUTrainMLP\scripts\profile_single_gpu.ps1
```

Ожидаемый результат:

```text
ошибка не растет на коротком запуске
manifest.json создан
weights.f32.bin создан
профиль содержит фазы generate, forward, backward, adamw
```

- [ ] **Шаг 3: выполнить Кагл-прогон на двух Т4**

Команда выполняется в Кагл-ядре:

```bash
bash kaggle/kernel/run_2xt4.sh
```

Ожидаемый результат:

```text
оба ранга завершились с кодом 0
номера collective_seq совпадают
главный ранг записал веса
```

- [ ] **Шаг 4: выполнить прогон на восьми А100**

Команда выполняется на кластере:

```bash
bash hpc/run_8xa100.sh
```

Ожидаемый результат:

```text
8 рангов завершились с кодом 0
collective_seq одинаков на всех рангах
контрольная точка записана только главным рангом
```

- [ ] **Шаг 5: записать приемочный отчет**

`D:\MultiGPUTrainMLP\docs\runbooks\v1_acceptance.md` должен содержать:

```markdown
# Приемка Версии Один

## Локальные Тесты

- cargo test: пройдено
- ctest: пройдено
- проверка выделений памяти: пройдено

## Однокарточный Контейнер

- дата запуска:
- устройство:
- шагов:
- начальная ошибка:
- конечная ошибка:
- путь профиля:

## Кагл Две Т4

- дата запуска:
- ранги:
- collective_seq:
- экспорт весов:

## Восемь А100

- дата запуска:
- ранги:
- collective_seq:
- контрольная точка:
```

- [ ] **Шаг 6: финальный коммит**

```powershell
git add docs
git commit -m "docs: record v1 trainer acceptance"
```

## Самопроверка Плана

Покрытие требований:

- обычная модель с `output_dim=1`: задачи 1, 2, 4, 6;
- онлайн случайные блуждания на каждом ранге: задачи 5, 6, 7;
- отсутствие общего датасета на диске: инварианты, задачи 5, 7;
- самостоятельный нативный тренер: задачи 1, 6, 7;
- кастомный обратный проход: задачи 5, 6;
- АдамВ с `weight_decay=0`: задачи 1, 5, 6;
- работа на одной карте: задачи 6, 7, 9;
- Кагл на двух Т4: задача 10;
- восемь А100 и многосерверность: задача 11;
- статические массивы и фиксированные форматы: задачи 2, 3, 4, 6;
- совместимость с быстрым инференсом поиска: задачи 4, 12;
- отсутствие двухкарточного контейнерного этапа: задачи 9 и 10 разделены явно.

Проверка на запрещенные пустые места:

- В плане нет маркеров неопределенной реализации.
- Каждый внешний формат имеет файл, тест и команду проверки.
- Ранговый порядок NCCL задан отдельным контрактом.
- Все команды запуска имеют ожидаемый результат.

Рискованные места, которые нужно проверять раньше остальных:

- точное соответствие `manifest.json` текущему читателю весов в `D:\100XH100`;
- реальная структура `p888.json` из архива, потому что фикстура задачи 3 проверяет формат, а не полную семантику группы;
- численное совпадение КУДА-обратного прохода с процессорным эталоном;
- отсутствие выделений в горячем цикле после добавления КАТЛАСС;
- одинаковый порядок NCCL-вызовов на двух Т4.
