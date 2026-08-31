# Single GPU Graph Feasibility Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. No subagents per user instruction. Steps use checkbox syntax for tracking.

**Goal:** Validate workspace-preserving stream binding and measure fixed-shape graph replay without changing public trainer APIs.

**Architecture:** Keep eager production behavior except redundant stream resets. A private-runtime diagnostic captures once and updates three typed kernel argument blocks. Graph performance is compared against an eager path using the same explicit workspace.

**Tech Stack:** C++17, CUDA 12.8, cuBLAS, existing Rust/C ABI regression suite, Docker queue, Nsight Systems.

**Spec:** `docs/superpowers/specs/2026-08-31-single-gpu-graph-feasibility.md`

## Global Constraints

- Docker `mgt-gpu-queue`, CUDA 12.8, RTX3070 Laptop / SM86 only.
- No subagents, public API change, approximation, clock change, or T4 claim.
- Preserve the C ABI, Rust API, arena query, default eager path, and nonlocal cuBLAS stream-reset behavior.
- All new GPU runs are serialized; frozen binaries and prior results are not overwritten.

### Task 1: Workspace-preserving stream binding

**Files:** `native/cuda/mgt_cuda/blas_stream.cuh`, `native/cuda/fp16_linear_train_ops.cu`, `native/cuda/mlp_batch_norm_forward.cu`, `native/tests/cuda/test_single_gpu_graph_capture.cu`, `native/CMakeLists.txt`.

**Interfaces:** `mgt_cuda::detail::BindBlasStream(cublasHandle_t, cudaStream_t) -> cublasStatus_t`; existing trainer APIs unchanged.

- [x] Reproduce meaningful RED with an explicit workspace: capture must fail the assertion `alloc_nodes == 0 && free_nodes == 0` before fixing redundant `cublasSetStream`.
- [x] Implement the helper:

```cpp
cudaStream_t current = nullptr;
auto status = cublasGetStream(blas, &current);
if (status != CUBLAS_STATUS_SUCCESS || current == stream) return status;
return cublasSetStream(blas, stream);
```

- [x] Run cold/warm rows4 and warm rows4096 captures, graph update, and small memcheck. Expected zero allocation/free nodes and exact small-model state.
- [x] Promote the diagnostic to a portable CMake test with full cleanup and real-stream-change checks. Run `ctest -R '^single_gpu_graph_capture$' --output-on-failure` through the queue.
- [x] Run existing CUDA/C ABI/Rust regression gates and sanitizer coverage; review the scoped diff before commit.

### Task 2: Capture once, update three kernels

**Files:** `native/tests/cuda/single_gpu_graph_test_support.cuh`, `native/tests/cuda/test_single_gpu_graph_replay.cu`, `native/tools/mgt_single_gpu_graph_probe.cu`, `native/CMakeLists.txt`.

**Interfaces:** test-only `TrainGraph(trainer, rows)`, `Launch(request, ticket)`; no exported ABI additions.

- [x] Write a stale-parameter negative control. The implemented capture uses `{4, 1, 0, 0}`; three omission controls agree at step1 and fail against eager at `{4, 2, 1, 12}` in states, weight or affine state respectively.
- [x] Match kernel addresses, require one of each, and update typed configs:

```cpp
walk.epoch_sample_offset = request.epoch_sample_offset;
adam.step = request.optimizer_step;
// Local kernelParams arrays retain captured pointer arguments; replace only
// the copied config, epoch, and step entries before cudaGraphExecKernelNodeSetParams.
```

- [x] Validate multiple offsets/epochs, enqueued replays, invalid sequence/shape/epoch range, completion tickets, complete small-model bytes, and production data/meta. Run memcheck/initcheck and bounded racecheck/synccheck.
- [x] Add a diagnostic benchmark with the same arguments as the eager benchmark plus an explicit mode (`eager-default`, `eager-workspace`, `graph`). Capture/setup excluded from timing; optimizer/data sequence identical across modes. Fail on any invalid request or nonfinite metric.

### Task 3: Evidence and publication

**Files:** `docs/audits/2026-08-31-single-gpu-graph-feasibility.md`, `docs/audits/original_p888_single_gpu_audit.md`.

- [x] Freeze binaries with SHA256; run ABBAAB at batch4096, warmup140, timed100. Compare eager-default against eager-workspace, then eager-workspace against graph. Retain every run and telemetry.
- [x] Capture fresh Nsight traces only after correctness and unprofiled gates. Verify graph kernel/memcpy/memset counts and dynamic input/Adam kernels, not merely graph launch count.
- [x] Record timings, workspace bytes, setup cost, failure attempts, sources, and explicit limits in the audit. Promote only measured, validated changes.
- [ ] Run final tests, `git diff --check`, check branch/remote and protected backup metadata, commit exact scoped files, push the existing branch, verify remote SHA.

## Self-review

Every spec requirement maps to Tasks 1–3. Test-only inclusion of implementation
translation units deliberately exposes private state without widening the
production API; it must link without duplicate definitions. CUDA Graph API and
Rust integration remain outside this feasibility deliverable.
