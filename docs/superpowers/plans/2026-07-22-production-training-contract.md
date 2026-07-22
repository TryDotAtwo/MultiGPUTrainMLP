# Production Training Contract Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `mgt_native_train` consume real puzzle data, resume deterministically, and emit safe periodic artifacts before autotuning the 2xT4 workload.

**Architecture:** Move checkpoint parsing, validation, scheduling, and atomic publication into CPU-testable `mgt_core` helpers. Keep CUDA orchestration in the trainer, but make it consume the validated state and cumulative step supplied by those helpers. Preserve synthetic data only behind an explicit benchmark flag.

**Tech Stack:** C++20, CUDA 12, NCCL, CMake/CTest, Rust trainer CLI, Kaggle 2xT4.

## Global Constraints

- Run no local GPU workload; all CUDA/NCCL execution and performance gates run only on Kaggle 2xT4.
- Preserve the current validated cuBLASLt input-gradient default and exclude measured losing CUTLASS variants.
- Use test-first red/green cycles for every behavior change.
- Do not touch unrelated `kaggle/kernel/run_ranks_2xt4.sh~31640`.

---

### Task 1: CPU-testable checkpoint metadata and scheduling

**Files:**
- Create: `native/include/mgt/training_artifacts.hpp`
- Create: `native/src/training_artifacts.cpp`
- Create: `native/tests/test_training_artifacts.cpp`
- Modify: `native/CMakeLists.txt`

**Interfaces:**
- Produces: `CheckpointMetadata`, `ReadCheckpointMetadata`, `WriteCheckpointMetadata`, `ValidateCheckpointMetadata`, `ShouldWritePeriodicArtifact`, and `PublishDirectoryAtomically` in namespace `mgt`.
- Consumes: plain paths, model/puzzle fingerprint strings, optimizer values, and cumulative step values.

- [ ] **Step 1: Write failing metadata round-trip, mismatch, checksum, schedule, and atomic-publication tests**

Create a CPU test that writes version-2 metadata, reads it back, rejects a changed fingerprint/parameter count, verifies period boundaries use cumulative steps, and verifies publication never exposes the temporary directory as `latest`.

- [ ] **Step 2: Run the focused test and verify RED**

Run: `cmake --build build-native-cpu --target test_training_artifacts && ctest --test-dir build-native-cpu -R training_artifacts --output-on-failure`

Expected: build failure because `mgt/training_artifacts.hpp` does not exist.

- [ ] **Step 3: Implement minimal artifact helpers**

Use an explicit line-oriented version-2 manifest with strict required keys. Compute a deterministic 64-bit FNV-1a checksum over payload bytes. Write files to `<name>.tmp`, close and verify them, remove only the previous `<name>.old`, rename current to old, rename temp to current, then remove old.

- [ ] **Step 4: Run the focused test and verify GREEN**

Run the Task 1 command again. Expected: one focused CTest passes.

- [ ] **Step 5: Commit**

Commit message: `feat: add validated training artifact metadata`

### Task 2: Real puzzle input contract

**Files:**
- Create: `native/include/mgt/trainer_inputs.hpp`
- Create: `native/src/trainer_inputs.cpp`
- Create: `native/tests/test_trainer_inputs.cpp`
- Modify: `native/tools/mgt_native_train_smoke.cu`
- Modify: `native/CMakeLists.txt`

**Interfaces:**
- Produces: `TrainerPuzzleInputs { group_json, target_bin, synthetic_benchmark }` and `LoadTrainerPuzzle(inputs, config, out, fingerprint)`.
- Consumes: existing `LoadPuzzleDefinition` and `TrainConfig`.

- [ ] **Step 1: Write failing production/synthetic-mode tests**

Cover missing paired inputs, mutually exclusive synthetic and real inputs, valid fixture loading, configured shape mismatch, and a stable two-file fingerprint.

- [ ] **Step 2: Verify RED with CPU-only CTest**

Run: `cmake --build build-native-cpu --target test_trainer_inputs && ctest --test-dir build-native-cpu -R trainer_inputs --output-on-failure`

Expected: build failure because the interface is absent.

- [ ] **Step 3: Implement input validation and route the trainer through it**

Add `--group-json`, `--target-bin`, and `--synthetic-benchmark`. Remove unconditional `BuildPuzzle`; invoke synthetic construction only when explicitly requested. Update all smoke invocations to pass `--synthetic-benchmark 1`.

- [ ] **Step 4: Verify GREEN and existing CPU suite**

Run: `ctest --test-dir build-native-cpu --output-on-failure`.

- [ ] **Step 5: Commit**

Commit message: `feat: require real puzzle inputs for native training`

### Task 3: Deterministic cumulative-step resume

**Files:**
- Create: `native/tests/test_resume_contract.cpp`
- Modify: `native/tools/mgt_native_train_smoke.cu`
- Modify: `native/include/mgt/training_artifacts.hpp`
- Modify: `native/src/training_artifacts.cpp`
- Modify: `native/CMakeLists.txt`

**Interfaces:**
- Produces: `ResumeState { completed_steps, seed, optimizer, fingerprint }` and `GlobalStep(completed_steps, local_step)`.
- Consumes: version-2 checkpoint metadata and payload checksum.

- [ ] **Step 1: Write failing resume-contract tests**

Assert global steps for fresh and resumed runs, rejection of truncated/trailing state payloads, wrong checksum, incompatible fingerprint, and changed optimizer configuration.

- [ ] **Step 2: Verify RED**

Run the focused CPU target and confirm the missing behavior causes the failure.

- [ ] **Step 3: Implement strict checkpoint read and cumulative step use**

Load and validate the manifest before copying state. Use `completed_steps + local_step` for random-walk epoch/step and `completed_steps + local_step + 1` for Adam bias correction. Write cumulative completed steps after updates.

- [ ] **Step 4: Verify GREEN**

Run the focused test and full CPU CTest suite.

- [ ] **Step 5: Commit**

Commit message: `fix: resume native training from cumulative step`

### Task 4: Periodic artifacts and optimizer plumbing

**Files:**
- Create: `native/tests/test_optimizer_contract.cpp`
- Modify: `native/tools/mgt_native_train_smoke.cu`
- Modify: `native/cuda/adamw.cu`
- Modify: `native/src/mlp_cpu_ref.cpp`
- Modify: `native/tests/cuda/test_cuda_adamw_smoke.cu`
- Modify: `crates/trainer-cli/src/native.rs`

**Interfaces:**
- Produces CLI flags `--checkpoint-period-steps`, `--weight-export-period-steps`, `--adam-beta1`, `--adam-beta2`, and `--adam-eps`.
- Consumes `ShouldWritePeriodicArtifact` and atomic publication from Task 1.

- [ ] **Step 1: Write failing CPU AdamW and CLI command tests**

The optimizer reference test distinguishes decoupled AdamW from L2-in-gradient for non-zero decay. Rust tests assert all five configuration values appear in the native command.

- [ ] **Step 2: Verify RED**

Run focused CTest and `cargo test -p trainer-cli native` and confirm expected failures.

- [ ] **Step 3: Implement minimal optimizer and periodic-write changes**

Compute Adam moments from the raw gradient, then update `weight -= lr * (m_hat / (sqrt(v_hat)+eps) + weight_decay * weight)`. At cumulative period boundaries, synchronize ranks, copy rank-0 state, and publish step-qualified artifacts atomically. Always publish the final enabled artifact.

- [ ] **Step 4: Verify GREEN**

Run CPU CTest, Rust tests, and compile-only CUDA targets. Do not execute CUDA locally.

- [ ] **Step 5: Commit**

Commit message: `feat: add periodic artifacts and native optimizer config`

### Task 5: Numerical and 2xT4 completion gates

**Files:**
- Create: `kaggle/kernel/check_training_contract_2xt4.sh`
- Modify: `native/tools/mgt_native_train_smoke.cu`
- Modify: `kaggle/kernel/run_ranks_2xt4.sh`
- Create: `test_results/training_contract_2xt4_<date>.md`

**Interfaces:**
- Produces a byte/checksum comparison for uninterrupted versus resumed training and both ranks.
- Consumes the production CLI and artifacts from Tasks 1-4.

- [ ] **Step 1: Add a failing finite-value gate test and the 2xT4 validation script**

The native loop must terminate on non-finite loss, gradient, or parameter state and report the cumulative step. The Kaggle script runs real inputs, split/resume equivalence, rank checksum equality, and periodic artifact checks.

- [ ] **Step 2: Verify the local RED test**

Run CPU/compile-only checks locally; do not execute the GPU script locally.

- [ ] **Step 3: Implement finite-value checks**

Use a device reduction flag copied once per step alongside the loss. On failure, synchronize NCCL ranks and exit non-zero with the cumulative step.

- [ ] **Step 4: Run the GPU gate only on Kaggle 2xT4**

Expected: both ranks agree, continuous/resume checksums match, all scheduled artifacts exist, and the final exported weights load successfully.

- [ ] **Step 5: Record evidence and commit**

Commit message: `test: validate production training contract on 2xT4`

## Self-review

- Spec coverage: real inputs, fail-closed mode selection, deterministic resume, periodic atomic artifacts, optimizer plumbing, finite gates, and 2xT4 validation each map to a task.
- Placeholder scan: no TBD/TODO or unspecified implementation step remains.
- Type consistency: Tasks 2-5 consume the metadata, schedule, and publication interfaces produced by Task 1; cumulative-step semantics are defined once in Task 3.
