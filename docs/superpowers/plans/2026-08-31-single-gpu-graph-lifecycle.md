# Single GPU Graph Lifecycle Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans inline. No subagents per user instruction. Track steps with checkboxes.

**Goal:** Integrate the verified fixed-batch graph path into the native trainer and explicit C ABI/Rust creation APIs.

**Architecture:** An owned internal graph object captures the unchanged enqueue-only body at prepare time. Three typed updates feed full-batch replay; smaller valid requests use eager. Explicit workspace belongs to the queried arena and failure poisons the trainer until destruction.

**Tech Stack:** C++17/CUDA12.8/cuBLAS, versioned C ABI, Rust RAII, Docker GPU queue, Nsight Systems.

**Spec:** `docs/superpowers/specs/2026-08-31-single-gpu-graph-lifecycle.md`

## Global Constraints

- RTX3070 Laptop / SM86 inside `mgt-gpu-queue`; no subagents or environment changes.
- Eager default, original p888 precision/data/optimizer unchanged.
- No mutation of V1 layout or reserved-zero validation; graph opt-in is explicit.
- No graph retry/fallback after runtime failure. Valid smaller batches use the declared eager tail path.
- Preserve previous binaries/artifacts and unrelated worktree files; normal scoped pushes only.

### Task 1: Native owned graph and lifecycle

**Files:** `native/cuda/mgt_cuda/single_gpu_trainer.cuh`, `native/cuda/single_gpu_trainer.cu`, `native/cuda/mgt_cuda/single_gpu_train_graph.cuh`, `native/cuda/single_gpu_train_graph.cu`, `native/cuda/random_walk_kernel.cu`, `native/cuda/adamw.cu`, `native/CMakeLists.txt`, `native/tests/cuda/test_single_gpu_native_graph.cu`.

**Interfaces:** enum `SingleGpuExecutionMode{kEager,kFixedBatchGraph}` and `SingleGpuTrainerCreateInfo::execution_mode`; internal graph accepts a stream capture body, kernel identity functions and typed step request. Public create/prepare/launch/metrics signatures remain unchanged.

- [x] Add explicit execution mode with eager default and compile a failing behavioral test: graph query must add4194304bytes; actual prepare must create a graph and leave sequence0.

```cpp
auto graph_info=eager_info;
graph_info.execution_mode=SingleGpuExecutionMode::kFixedBatchGraph;
Check(QuerySingleGpuTrainerBytes(graph_info,&graph_bytes)==mgt::Status::kOk,"query");
Check(graph_bytes==eager_bytes+4194304,"workspace arena budget");
```

- [x] Implement owned graph source/executable and function-identity matching using the proven diagnostic algorithm. Retain graph argument storage, reject allocation/free nodes, update only the three dynamic nodes, and destroy after stream drain.
- [x] Refactor the old device-work body into `EnqueueSingleGpuTrainStep`; keep validation/event/sequence handling in `LaunchSingleGpuTrainStep`. Prepare captures without calling the public bookkeeping path.

```cpp
const auto status=request.active_rows==trainer->info.capacity_rows&&graph_mode
    ? LaunchPreparedSingleGpuGraph(trainer,request)
    : EnqueueSingleGpuTrainStep(trainer,request);
// On success, record completion then advance sequence. Any runtime failure
// poisons the trainer; invalid requests are rejected before this block.
```

- [x] Test exact states/model bytes at capacity4/256, full→tail→full, epoch rollover, queued requests, invalid requests, workspace accounting, and fail-stop rejection after injected failed state. Run mem/init/race/sync and existing lifecycle/overwrite/BN gates in Docker.

### Task 2: Explicit C ABI and Rust creation mode

**Files:** `native/include/mgt/single_gpu_trainer_ffi.h`, `native/src/single_gpu_trainer_ffi.cpp`, `native/tests/test_single_gpu_trainer_ffi.cpp`, `crates/trainer-cli/src/single_gpu_ffi.rs`, `crates/trainer-cli/tests/single_gpu_ffi.rs`.

**Interfaces:** `MgtSingleGpuExecutionOptionsV1 {uint32_t struct_size, abi_version, execution_mode, reserved_u32;}` and `mgt_single_gpu_v1_create_with_options(const MgtSingleGpuConfigV1*,const MgtSingleGpuExecutionOptionsV1*,MgtSingleGpuHandle**)`; Rust `SingleGpuTrainer::create_with_mode(config, SingleGpuExecutionMode)`.

- [x] Add ABI declarations and a failing behavior test requiring explicit graph options to create/prepare/train a full+tail sequence while invalid size/version/mode/reserved fields leave output null.
- [x] Route legacy create through eager and new create through validated options; no reserved-field reinterpretation. Use scoped ownership during puzzle load/create to avoid leaks on exceptions.
- [x] Add Rust mode enum/options representation and creation method. Keep `create(config)` eager and existing `SingleGpuConfig` fields unchanged.
- [x] Split optional metrics reads from the Rust hot path: `enqueue_step` calls
  the existing C ABI with null metrics; `read_metrics` uses the additive
  `mgt_single_gpu_v1_read_metrics` API. Verify repeated reads do not advance
  sequence, queued full/tail runs complete, and Drop drains outstanding work.

```rust
let mut trainer=SingleGpuTrainer::create_with_mode(config,SingleGpuExecutionMode::FixedBatchGraph)?;
trainer.prepare()?;
let first=trainer.train_step(4,1,0,0)?;
let tail=trainer.train_step(3,2,0,4)?;
assert_eq!((first.optimizer_step,tail.optimizer_step),(1,2));
```

- [x] Run C ABI and Rust tests sequentially inside the GPU queue, including unchanged80-byte config,16-byte options, failure cleanup and Drop after queued work.

### Task 3: Integrated performance and publication

**Files:** `native/tools/mgt_single_gpu_benchmark.cu`, `docs/audits/2026-08-31-single-gpu-native-graph.md`, `docs/audits/original_p888_single_gpu_audit.md`.

- [x] Add optional `eager|graph` benchmark argument while preserving the original5-argument invocation. Report mode and queried arena bytes; reject invalid mode and illegal epoch extent.
- [x] Run the native graph benchmark under memcheck at4096, freeze binaries, then ABBAAB140warmup/100timed and node-level Nsight100warmup/4timed. Compare identical work and count captures/node updates.
- [x] Run all relevant native/C ABI/Rust regression gates on the final tree, document exact evidence and limits, check scoped diff/remote, commit/push and verify remote SHA. Published as `c00e7edbdab8303c3cb9abf85822389d8b2417b8`; remote SHA matched.

## Self-review

Tasks1–3 cover every spec requirement. Capture/graph internals remain separate
from C ABI ownership; C++ mode and C ABI options share explicit numeric values.
No API will automatically enable graphs or treat a failed capture as eager.
