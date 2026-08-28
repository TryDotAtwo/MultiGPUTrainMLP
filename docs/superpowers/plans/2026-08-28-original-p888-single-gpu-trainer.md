# Original p888 Single-GPU Trainer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a prepared, allocation-free, Rust-controlled CUDA trainer for one correct original-p888 step, then establish an unprofiled and Nsight-attributed RTX 3070 baseline.

**Architecture:** Rust owns configuration and artifacts and calls a versioned C ABI. A C++20 opaque trainer owns one CUDA device, cuBLASLt/CUTLASS plans, streams, events, and a persistent arena; CUDA C++ executes random-walk generation and the complete BatchNorm MLP step without NCCL. Existing correctness-proven primitives are reused behind a local single-rank reduction backend before optimization.

**Tech Stack:** Rust 2021, C ABI, C++20, CUDA C++17, cuBLAS/cuBLASLt, CUTLASS, CUB/CCCL, CMake/CTest, Docker, Nsight Systems, Nsight Compute.

**Spec:** `docs/superpowers/specs/2026-08-28-original-p888-single-gpu-trainer-design.md`

## Global Constraints

- Model: logical `5184 -> 2556 -> 16 * (218 -> 218) -> 1`; physical hidden strides `2560` and `224`.
- BatchNorm follows the source-backed placement and PyTorch training semantics; padding lanes are inert.
- Mixed precision: FP16 linear operands, FP32 accumulation, master state, BatchNorm, loss, gradients at optimizer boundaries, and AdamW state.
- Development target: RTX 3070 Laptop GPU, SM86, inside `mgt-gpu-queue`; every GPU command uses `gpu_queue_submit.py`.
- Acceptance target remains Tesla T4 SM75; no SM86 result selects an SM75 policy.
- No NCCL communicator, Rust callback, allocation, file I/O, CPU readback, plan construction, or device-wide synchronization inside the steady-state step.
- Correctness and performance runs produce separate artifacts; profiled timing is never headline throughput.
- Never read or modify `kaggle/kernel/run_ranks_2xt4.sh~31640`; verify only length `5258` and `LastWriteTimeUtc.Ticks=639188887422510152` before commits.

## File Structure

- `Dockerfile.single-gpu-rust`: pinned minimal Rust layer over the existing CUDA/CUTLASS/Nsight image.
- `native/include/mgt/single_gpu_contract.hpp`, `native/src/single_gpu_contract.cpp`: fixed original-p888 logical/physical contract and validation.
- `native/cuda/mgt_cuda/local_batch_norm.cuh`, `native/cuda/local_batch_norm.cu`: collective-free local BatchNorm forward/backward.
- `native/cuda/mgt_cuda/local_mlp_batch_norm.cuh`, `native/cuda/local_mlp_batch_norm.cu`: one-GPU original-model orchestration with no reference to the NCCL implementation unit.
- `native/cuda/mgt_cuda/single_gpu_trainer.cuh`, `native/cuda/single_gpu_trainer.cu`: opaque prepared CUDA runtime and arena ownership.
- `native/include/mgt/single_gpu_trainer_ffi.h`, `native/src/single_gpu_trainer_ffi.cpp`: stable C ABI only.
- `crates/trainer-cli/src/single_gpu_ffi.rs`: safe Rust owner and ABI conversion.
- `scripts/profile_original_p888_single_gpu.ps1`: host entrypoint that submits build, correctness, baseline, and profile jobs to the shared queue.
- `scripts/summarize_single_gpu_profile.py`: deterministic JSON summary from benchmark output.

---

### Task 0: Add Rust to the shared CUDA development image

**Files:**
- Create: `Dockerfile.single-gpu-rust`
- Modify: `scripts/start_gpu_queue.ps1`

- [x] **Step 1: Verify RED** — the base image reports `cargo: command not found`.
- [x] **Step 2: Build `mgt-single-gpu-dev:2026-08-28`** — install pinned Rust 1.89 with the minimal rustup profile and remove installer packages/lists in the same layer.
- [x] **Step 3: Recreate the empty queue container** — preserve `.gpu_queue` and workspace through the existing bind mount.
- [x] **Step 4: Verify GREEN** — Rust/Cargo, Nsight, RTX 3070 SM86, and all six `trainer-cli` tests pass inside the queue.

---

### Task 1: Freeze the aligned original-p888 contract

**Files:**
- Create: `native/include/mgt/single_gpu_contract.hpp`
- Create: `native/src/single_gpu_contract.cpp`
- Create: `native/tests/test_single_gpu_contract.cpp`
- Modify: `native/CMakeLists.txt`

**Interfaces:**
- Produces: `mgt::OriginalP888SingleGpuContract() -> SingleGpuModelContract`.
- Produces: `mgt::ValidateSingleGpuModelContract(const SingleGpuModelContract&) -> Status`.

- [x] **Step 1: Write the failing contract test**

```cpp
const auto c = mgt::OriginalP888SingleGpuContract();
CHECK(c.state_len == 72);
CHECK(c.state_value_count == 72);
CHECK(c.input_features == 5184);
CHECK(c.logical_hd1 == 2556 && c.physical_hd1 == 2560);
CHECK(c.logical_hd2 == 218 && c.physical_hd2 == 224);
CHECK(c.residual_blocks == 16 && c.batch_norm_sites == 34);
CHECK(mgt::ValidateSingleGpuModelContract(c) == mgt::Status::kOk);
auto bad = c;
bad.input_features = 6336;
CHECK(mgt::ValidateSingleGpuModelContract(bad) == mgt::Status::kInvalidConfig);
```

- [x] **Step 2: Run RED**

Run: `cmake --build native/build-cpu-codex --config Release --target test_single_gpu_contract`

Expected: FAIL because the target/header does not exist.

- [x] **Step 3: Implement the immutable contract**

```cpp
struct SingleGpuModelContract {
    std::uint32_t schema_version;
    std::uint32_t state_len;
    std::uint32_t state_value_count;
    std::uint32_t input_features;
    std::uint32_t logical_hd1;
    std::uint32_t physical_hd1;
    std::uint32_t logical_hd2;
    std::uint32_t physical_hd2;
    std::uint32_t residual_blocks;
    std::uint32_t batch_norm_sites;
    std::uint32_t output_dim;
};
```

Validation requires schema 1, `input_features == state_len * state_value_count`, logical widths not larger than physical widths, physical widths divisible by 8, 16 blocks, 34 BN sites, and scalar output.

- [x] **Step 4: Run GREEN**

Run: `ctest --test-dir native/build-cpu-codex -C Release -R '^single_gpu_contract$' --output-on-failure --no-tests=error`

Expected: one passing test.

- [x] **Step 5: Commit**

```bash
git add native/include/mgt/single_gpu_contract.hpp native/src/single_gpu_contract.cpp native/tests/test_single_gpu_contract.cpp native/CMakeLists.txt
git commit -m "test: freeze original p888 single gpu contract"
```

### Task 2: Remove NCCL from the one-GPU BatchNorm path

**Files:**
- Create: `native/cuda/mgt_cuda/local_batch_norm.cuh`
- Create: `native/cuda/local_batch_norm.cu`
- Create: `native/cuda/mgt_cuda/local_mlp_batch_norm.cuh`
- Create: `native/cuda/local_mlp_batch_norm.cu`
- Create: `native/tests/cuda/test_local_batch_norm.cu`
- Modify: `native/CMakeLists.txt`

**Interfaces:**
- Produces: `LaunchLocalStridedBatchNormForward(...) -> mgt::Status`.
- Produces: `LaunchLocalStridedBatchNormBackward(...) -> mgt::Status`.
- Produces: `LaunchLocalMlpBatchNormTrainStep(...) -> mgt::Status`, implemented in an object file with no `NcclRankContext` or NCCL reference.

- [x] **Step 1: Write a failing local-BN parity test**

Use four rows, three logical columns, stride four, nontrivial gamma/beta, and the existing CPU fixture. Assert forward output, mean, inverse standard deviation, `dx`, `dgamma`, `dbeta`, running mean/variance, and a zero fourth padding lane.

```cpp
CHECK(mgt_cuda::LaunchLocalStridedBatchNormForward(
    x, 4, 3, 4, gamma, beta, running_mean, running_var,
    0.1f, 1.0e-5f, y, mean, inv_std, normalized, workspace, stream)
    == mgt::Status::kOk);
```

- [x] **Step 2: Run RED through the GPU queue**

Run:

```powershell
docker exec mgt-gpu-queue python3 scripts/gpu_queue_submit.py --label single-local-bn-red --wait -- bash -lc "cmake -S native -B /tmp/mgt-single-sm86 -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86 -DMGT_ENABLE_CUDA=ON -DMGT_ENABLE_NCCL=ON -DMGT_CUTLASS_ROOT=/opt/cutlass && cmake --build /tmp/mgt-single-sm86 --target test_local_batch_norm"
```

Expected: FAIL because the target/API is absent.

- [x] **Step 3: Implement local reductions**

Reuse the existing strided local-moment and local-gradient math, but finalize device-local workspace directly. Do not create a communicator and do not call `ncclAllReduce`. Implement the local original-model orchestration in `local_mlp_batch_norm.cu`; route every BN site locally and omit parameter-gradient collectives entirely.

- [x] **Step 4: Prove parity and absence of NCCL calls**

Run through the queue:

```bash
cmake --build /tmp/mgt-single-sm86 --target test_local_batch_norm test_mlp_batch_norm_full_backward
ctest --test-dir /tmp/mgt-single-sm86 -R '^(local_batch_norm|mlp_batch_norm_full_backward)$' --output-on-failure --no-tests=error
```

Expected: both tests pass. Then run `rg -n 'Nccl|nccl' native/cuda/local_batch_norm.cu native/cuda/local_mlp_batch_norm.cu`; expected: no output.

- [x] **Step 5: Commit**

```bash
git add native/cuda/mgt_cuda/local_batch_norm.cuh native/cuda/local_batch_norm.cu native/cuda/mgt_cuda/local_mlp_batch_norm.cuh native/cuda/local_mlp_batch_norm.cu native/tests/cuda/test_local_batch_norm.cu native/CMakeLists.txt
git commit -m "perf: add collective free single gpu batchnorm"
```

### Task 3: Add a prepared allocation-free native runtime

**Files:**
- Create: `native/cuda/mgt_cuda/single_gpu_trainer.cuh`
- Create: `native/cuda/single_gpu_trainer.cu`
- Create: `native/tests/cuda/test_single_gpu_trainer_lifecycle.cu`
- Modify: `native/CMakeLists.txt`

**Interfaces:**
- Produces opaque `mgt_cuda::SingleGpuTrainer`.
- Produces `QuerySingleGpuTrainerBytes`, `CreateSingleGpuTrainer`, `PrepareSingleGpuTrainer`, `LaunchSingleGpuTrainStep`, `ReadSingleGpuMetrics`, and `DestroySingleGpuTrainer`.

- [x] **Step 1: Write failing lifecycle tests**

Test invalid device, zero capacity, wrong contract schema, physical-width mismatch, double prepare, step before prepare, capacity overflow, monotonically increasing optimizer step, and double destroy through a pointer-to-pointer helper.

```cpp
mgt_cuda::SingleGpuTrainer* trainer = nullptr;
CHECK(mgt_cuda::CreateSingleGpuTrainer(info, &trainer) == mgt::Status::kOk);
CHECK(mgt_cuda::LaunchSingleGpuTrainStep(trainer, request, &ticket)
      == mgt::Status::kInvalidConfig);
CHECK(mgt_cuda::PrepareSingleGpuTrainer(trainer) == mgt::Status::kOk);
```

- [x] **Step 2: Run RED through the queue**

Run: build `test_single_gpu_trainer_lifecycle` in `/tmp/mgt-single-sm86`.

Expected: FAIL because the runtime API is absent.

- [x] **Step 3: Implement one persistent arena**

`CreateSingleGpuTrainer` computes checked byte offsets for weights, gradients, Adam moments, BN affine/running state, states, labels, forward workspace, backward workspace, loss, and metrics. It performs one `cudaMalloc` for the arena and creates streams/events/cuBLAS handles. `PrepareSingleGpuTrainer` uploads immutable state, initializes selected library plans, and runs warmups. `LaunchSingleGpuTrainStep` performs no allocation and accepts only `active_rows <= capacity_rows`.

- [x] **Step 4: Instrument allocation prohibition**

Expose a test-only allocation counter incremented by the runtime allocation wrapper. Record its value after prepare; assert it is unchanged after three steps. Assert no `cudaDeviceSynchronize` exists in `single_gpu_trainer.cu` with:

```bash
! rg 'cudaDeviceSynchronize' native/cuda/single_gpu_trainer.cu
```

- [x] **Step 5: Run GREEN**

Run through the queue:

```bash
cmake --build /tmp/mgt-single-sm86 --target test_single_gpu_trainer_lifecycle
ctest --test-dir /tmp/mgt-single-sm86 -R '^single_gpu_trainer_lifecycle$' --output-on-failure --no-tests=error
```

Expected: pass.

- [ ] **Step 6: Commit**

```bash
git add native/cuda/mgt_cuda/single_gpu_trainer.cuh native/cuda/single_gpu_trainer.cu native/tests/cuda/test_single_gpu_trainer_lifecycle.cu native/CMakeLists.txt
git commit -m "feat: add prepared single gpu trainer runtime"
```

### Task 4: Expose the versioned C ABI

**Files:**
- Create: `native/include/mgt/single_gpu_trainer_ffi.h`
- Create: `native/src/single_gpu_trainer_ffi.cpp`
- Create: `native/tests/test_single_gpu_trainer_ffi.cpp`
- Modify: `native/CMakeLists.txt`

**Interfaces:**
- Produces shared library `mgt_single_gpu_trainer`.
- Produces ABI functions prefixed `mgt_single_gpu_v1_`.

- [ ] **Step 1: Write a failing ABI test**

The test includes only the C header. It asserts ABI version, struct-size rejection, null-handle rejection, stable status values, bounded error text, and create/destroy ownership.

```c
MgtSingleGpuConfigV1 cfg = {0};
cfg.struct_size = sizeof(cfg);
cfg.abi_version = MGT_SINGLE_GPU_ABI_V1;
MgtSingleGpuHandle* handle = NULL;
CHECK(mgt_single_gpu_v1_create(&cfg, &handle) == MGT_STATUS_INVALID_CONFIG);
```

- [ ] **Step 2: Run RED**

Run: build `test_single_gpu_trainer_ffi`.

Expected: FAIL because the header/library is absent.

- [ ] **Step 3: Implement the ABI**

Define plain fixed-width structs only:

```c
uint32_t mgt_single_gpu_v1_abi_version(void);
MgtStatus mgt_single_gpu_v1_create(const MgtSingleGpuConfigV1*, MgtSingleGpuHandle**);
MgtStatus mgt_single_gpu_v1_prepare(MgtSingleGpuHandle*);
MgtStatus mgt_single_gpu_v1_train_step(MgtSingleGpuHandle*, const MgtSingleGpuStepV1*, MgtSingleGpuMetricsV1*);
MgtStatus mgt_single_gpu_v1_checkpoint(MgtSingleGpuHandle*, const char* directory_utf8);
MgtStatus mgt_single_gpu_v1_destroy(MgtSingleGpuHandle**);
size_t mgt_single_gpu_v1_last_error(MgtSingleGpuHandle*, char* dst, size_t capacity);
```

Catch every C++ exception before it crosses the ABI. Validate `struct_size`, `abi_version`, reserved-zero fields, UTF-8 path presence, and pointer ownership.

- [ ] **Step 4: Run GREEN on CPU and GPU builds**

Run the ABI test in the CPU build for validation-only cases and through the queue for create/prepare/destroy on device 0.

Expected: both pass.

- [ ] **Step 5: Commit**

```bash
git add native/include/mgt/single_gpu_trainer_ffi.h native/src/single_gpu_trainer_ffi.cpp native/tests/test_single_gpu_trainer_ffi.cpp native/CMakeLists.txt
git commit -m "feat: expose single gpu trainer c abi"
```

### Task 5: Add the safe Rust owner

**Files:**
- Create: `crates/trainer-cli/build.rs`
- Create: `crates/trainer-cli/src/single_gpu_ffi.rs`
- Create: `crates/trainer-cli/tests/single_gpu_ffi.rs`
- Modify: `crates/trainer-cli/Cargo.toml`
- Modify: `crates/trainer-cli/src/main.rs`

**Interfaces:**
- Produces `SingleGpuTrainer::create`, `prepare`, `train_step`, `checkpoint`, and RAII `Drop`.
- Adds CLI command `train-single-gpu --config <path> --output-dir <path>`.

- [ ] **Step 1: Write failing Rust layout and validation tests**

```rust
assert_eq!(std::mem::size_of::<MgtSingleGpuConfigV1>(), EXPECTED_CONFIG_BYTES);
let err = SingleGpuTrainer::create(invalid_config()).unwrap_err();
assert!(err.to_string().contains("input_features"));
```

Add a compile-time assertion that the handle wrapper is `Send` only if ownership is moved, and is not `Clone`.

- [ ] **Step 2: Run RED**

Run: `cargo test -p trainer-cli --test single_gpu_ffi`

Expected: FAIL because the module and native link configuration are absent.

- [ ] **Step 3: Implement explicit FFI declarations and RAII**

`build.rs` reads `MGT_SINGLE_GPU_NATIVE_LIB_DIR`, emits the exact native search path, and links `mgt_single_gpu_trainer`. The Rust wrapper converts every nonzero status into `anyhow::Error` using a fixed 4096-byte error buffer. `Drop` calls destroy once and never panics.

- [ ] **Step 4: Run GREEN against the Docker-built library**

Run through the queue with the library directory exported:

```bash
MGT_SINGLE_GPU_NATIVE_LIB_DIR=/tmp/mgt-single-sm86 cargo test -p trainer-cli --test single_gpu_ffi -- --nocapture
```

Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add crates/trainer-cli/build.rs crates/trainer-cli/src/single_gpu_ffi.rs crates/trainer-cli/tests/single_gpu_ffi.rs crates/trainer-cli/Cargo.toml crates/trainer-cli/src/main.rs
git commit -m "feat: control single gpu trainer from rust"
```

### Task 6: Prove the complete original-model step

**Files:**
- Create: `native/tests/cuda/test_original_p888_single_gpu_step.cu`
- Create: `native/tests/fixtures/p888_single_gpu_step.json`
- Modify: `native/tests/reference/generate_p888_bn_fixture.py`
- Modify: `native/CMakeLists.txt`

**Interfaces:**
- Consumes the prepared runtime and original-p888 contract.
- Produces the authoritative deterministic mixed-precision correctness gate.

- [ ] **Step 1: Extend the fixture generator first**

Generate deterministic source-backed data for all 34 BN sites, all 16 residual blocks, negative/zero/positive gamma, ReLU-at-zero cases, logical widths, padding sentinels, loss, every parameter-gradient family, running state, and one AdamW update.

- [ ] **Step 2: Write the failing full-step test**

Load the fixture, execute one prepared step, and compare outputs/gradients/state with per-field tolerances. Assert every physical padding lane is exactly zero and all values are finite.

- [ ] **Step 3: Run RED through the queue**

Run: build and execute `test_original_p888_single_gpu_step`.

Expected: FAIL at the first unsupported production-shape or mixed-precision path, with the failing field printed.

- [ ] **Step 4: Implement the minimum path to pass**

Route the runtime through local BN and existing linear/backward/AdamW primitives. Enable FP16 operands with FP32 accumulation only after strict FP32 passes. Keep padding masks at initialization, BN boundaries, residual additions, gradients, and optimizer updates.

- [ ] **Step 5: Run GREEN plus regressions**

Run through the queue:

```bash
ctest --test-dir /tmp/mgt-single-sm86 -R '^(original_p888_single_gpu_step|local_batch_norm|mlp_batch_norm_full_backward|cuda_batch_norm_reference|cuda_adamw_smoke)$' --output-on-failure --no-tests=error
```

Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add native/tests/cuda/test_original_p888_single_gpu_step.cu native/tests/fixtures/p888_single_gpu_step.json native/tests/reference/generate_p888_bn_fixture.py native/CMakeLists.txt native/cuda/single_gpu_trainer.cu
git commit -m "test: prove original p888 single gpu step"
```

### Task 7: Establish the RTX 3070 baseline and Nsight attribution

**Files:**
- Create: `native/tools/mgt_single_gpu_benchmark.cu`
- Create: `scripts/profile_original_p888_single_gpu.ps1`
- Create: `scripts/summarize_single_gpu_profile.py`
- Create: `scripts/tests/test_summarize_single_gpu_profile.py`
- Modify: `native/CMakeLists.txt`
- Create: `test_results/original_p888_single_gpu_3070_baseline.md`

**Interfaces:**
- Produces JSONL rows with `gpu`, `arch`, `batch`, `warmup`, `step`, `step_ms`, `samples_s`, `memory_bytes`, `flops`, and `status`.
- Produces one summary JSON with Q50/Q95 and one Markdown evidence report.

- [ ] **Step 1: Write failing summarizer tests**

Cover unordered rows, warmup exclusion, failed-row rejection, nearest-rank Q50/Q95, mismatched GPU/arch/config rejection, and stable JSON key ordering.

- [ ] **Step 2: Run RED**

Run: `py -m unittest scripts.tests.test_summarize_single_gpu_profile -v`

Expected: FAIL because the summarizer is absent.

- [ ] **Step 3: Implement benchmark and summarizer**

Benchmark exact original p888 batches `16384,24576,32768,40960,49152`, with 10 warmups and 100 measured steps per fresh process. Record OOM/failure rows. Select the fastest stable batch with at least 10% free VRAM margin; do not select a profiled row.

- [ ] **Step 4: Run correctness and unprofiled sweep through the queue**

`profile_original_p888_single_gpu.ps1` first verifies idle GPU and correctness, then submits the sweep. Store raw outputs under `test_results/original_p888_single_gpu_3070_<timestamp>/`.

- [ ] **Step 5: Run Nsight Systems on the selected batch**

Capture CUDA, NVTX, cuBLAS, and OS runtime traces for 5 warmups plus 20 measured steps. Report launch gaps, host synchronization, top CUDA kernels, cuBLAS/CUTLASS time, BN time, optimizer time, and memory-copy time.

- [ ] **Step 6: Run Nsight Compute only on the dominant kernel family**

Collect `sm__throughput`, Tensor Core pipe activity, DRAM throughput, L2 hit rate, occupancy, and eligible/issued warps for the top one or two kernels identified by Systems. Do not profile the entire training process with NCU.

- [ ] **Step 7: Verify and commit evidence**

Run CPU/Rust tests, targeted CUDA gates, `git diff --check`, and verify the protected backup metadata. The report states RTX 3070 results are diagnostic and names the next optimization by measured critical-path share.

```bash
git add native/tools/mgt_single_gpu_benchmark.cu scripts/profile_original_p888_single_gpu.ps1 scripts/summarize_single_gpu_profile.py scripts/tests/test_summarize_single_gpu_profile.py native/CMakeLists.txt test_results/original_p888_single_gpu_3070_baseline.md
git commit -m "bench: baseline original p888 on one rtx3070"
```

## Completion Gate

- Rust executes one complete original-p888 CUDA step through ABI v1.
- The single-GPU path has no NCCL symbol/call and no steady-state allocation or device-wide sync.
- Full-step mixed-precision correctness, padding, finite-state, and one-step AdamW gates pass.
- Unprofiled RTX 3070 Q50/Q95, samples/s, VRAM, and FLOP estimate are recorded.
- Nsight evidence identifies the next optimization; no speed claim is made from profiled timing.
