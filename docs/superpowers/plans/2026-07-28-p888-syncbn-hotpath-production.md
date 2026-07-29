# P888 SyncBN Production Hot-Path and Original-Parity Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Starting from `c770e20`, turn the correct but diagnostic FP32 SyncBatchNorm train step into a reproducible, hardware-tuned production trainer that preserves the original P888 model/training contract, is maximally efficient on 2x Tesla T4, completes all 32,692 semantic epochs, and passes the puzzle-0 depth-100/beam-10M quality gate.

**Architecture:** Keep an immutable strict-FP32 oracle path. First repair and prove the original contract and benchmark methodology, then expose the already-proven tiled input-embedding-gradient primitive to the BatchNorm path, port the existing mixed-precision linear primitives, fuse only correctness-proven BatchNorm memory passes, and optimize communication behind explicit policies. Develop and profile on 8x A100, but select and cache the production policy only from authoritative 2x T4 measurements. Finally integrate the optimized step into the real epoch/checkpoint/export pipeline and run full training plus beam-search acceptance.

**Tech Stack:** C++20 host code, CUDA C++17, cuBLAS/cuBLASLt, optional CUTLASS experiments, NCCL, CMake/CTest, Python 3/PyTorch reference fixtures, SLURM on MEPhI 8x A100, Kaggle 2x Tesla T4.

## Global Constraints

- Base all work on branch `codex-native-trainer-implementation` at or after commit `c770e20`.
- This is a delta plan. Do not replay completed Tasks 1-6 from `2026-07-28-p888-original-parity-optimized-training.md`.
- This plan document is committed and pushed before Task 1 begins; numbered task commits stage only their listed implementation payload and do not re-stage this document.
- The final performance target is exactly 2x Tesla T4. An 8x A100 result is diagnostic evidence only and never authorizes a T4 default.
- Do not execute SSH commands. The user enters cluster commands in the built-in Codex terminal. The implementing agent may only provide exact commands and read the resulting terminal output.
- Run all MEPhI computation through `sbatch -p kaf12`; never compute on a login node.
- Route every local CUDA test, benchmark, sanitizer, Nsight, or training command through the existing long-lived `mgt-gpu-queue` with its 10-second cooldown. Use one concrete task-specific queue label and place build plus test/benchmark in that queued inner command; for example, `docker exec mgt-gpu-queue python3 scripts/gpu_queue_submit.py --label task7-linear-ctest --wait -- bash -lc "cmake --build native/build --config Release --parallel && ctest --test-dir native/build -C Release -R 'cuda_linear_train_ops|mlp_batch_norm_mixed_precision' --output-on-failure --no-tests=error"`. Do not execute a displayed CUDA inner command directly, start an independent GPU container, or replace the queue with a file lock. CPU-only configure/tests may run directly.
- On the MEPhI 8x A100 job, use 8 ranks, 4 CPUs per rank, and `--gres=gpu:8`. Do not use `--gpus-per-task=1` or `--gpu-bind=single:1`; that topology already failed with an NVML/NCCL PCI-bus lookup error.
- Never open, modify, stage, rename, or delete the user backup `kaggle/kernel/run_ranks_2xt4.sh~31640`. The plan-frozen metadata baseline is exactly length `5258` and `LastWriteTimeUtc.Ticks=639188887422510152`. At the start and end of every task/session, require it still exists as a leaf and compare only those two metadata fields to the frozen literals; any difference stops work. Never recapture a new baseline, hash, or read its contents.
- Before every commit, inspect `git status --short`; stage only the paths listed in that task.
- Keep the strict FP32 path available and green throughout the work. Never weaken its tolerances to make TF32 or FP16 pass.
- BatchNorm statistics, affine parameters, affine gradients, running state, and Adam state remain FP32 in every policy.
- The hot loop must allocate no host or device memory, create no streams/events/handles, perform no synchronous CPU readback, and perform no global device synchronization. The only host observation allowed during training is Task 15's preallocated asynchronous telemetry ring with nonblocking `cudaEventQuery`; it never stalls the compute stream.
- BatchNorm collectives and linear-gradient collectives use `SUM`. Do not reuse the old no-BN trainer's `AVERAGE` allreduce callback without an explicit reduction-op parameter and tests.
- Unknown enum values, incompatible cached policies, size overflow, invalid categorical values, and insufficient workspace fail closed.
- Preserve the inference model layout, weight ordering, scoring, and multi-GPU beam-search contract.
- Preserve these exact training values:

  ```text
  state_len                     72
  state_value_count             72
  logical input features      5184
  logical hd1                 2556
  physical hd1                2560
  logical hd2                  218
  physical hd2                 224
  residual blocks               16
  BatchNorm sites                34
  output dimension                1
  BatchNorm epsilon           1e-5
  BatchNorm momentum            0.1
  Adam learning rate           1e-4
  Adam beta1                    0.9
  Adam beta2                  0.999
  Adam epsilon                 1e-8
  weight decay                    0
  depths                       1..29 inclusive
  walkers per depth            34482
  samples per semantic epoch  999978
  full global batch           100000
  final global batch           99978
  optimizer steps per epoch        10
  semantic epochs              32692
  optimizer steps             326920
  ```

---

## Current Baseline and Why This Order Is Mandatory

The implementation worker must treat the following as evidence, not assumptions:

| Item | Current evidence | Consequence |
| --- | --- | --- |
| Full strict SyncBN backward | `test_mlp_batch_norm_full_backward` passes | Preserve it as the oracle path. |
| A100 TF32 smoke | roughly 50.97-54.00 ms/step, 1.85-1.96M samples/s | Re-measure; the Linux build was not proved to have `CMAKE_BUILD_TYPE=Release`. |
| Stage sample | forward 8.23 ms, residual backward 12.98 ms, input backward 20.01 ms | Do not call 20.01 ms the sparse-kernel time; it also contains BN and a 50.63 MiB allreduce. |
| Benchmark data | weights, states, and labels are all zero | Replace before accepting any optimization. |
| Workspace | 248,824,612 FP32/rank, about 949.2 MiB | Track peak memory for every candidate. |
| Current collective graph | 34 BN forward + 34 BN backward + 4 weight + 1 loss = 73 calls/step | Optimize weight reductions separately; BN calls are layer-dependent. |
| Old 2x T4 winner | tiled FP16 one-hot cuBLASLt, tile 48, Lt workspace 16 MiB, allreduce bucket 4 MiB, about 511,028 states/s | Use it as the first T4 candidate, not as a universal constant. |
| Old sparse owner-write | about 72-73k states/s and 1.46 s/step on 2x T4 | Never make sparse owner-write the production default. |
| Contract test | asserts `kInputFeatures=6336` | Repair before optimizing; 72x72 and original parameter count prove 5184. |

The exact original logical parameter arithmetic is:

```text
linear:
  5184*2556 + 2556
  + 2556*218 + 218
  + 16*2*(218*218 + 218)
  + 218 + 1
  = 15,338,249

BatchNorm affine:
  2*(2556 + 33*218)
  = 19,500

original logical total:
  15,338,249 + 19,500
  = 15,357,749
```

The original metadata reports exactly `15,357,749`. The padded native linear storage is separately `15,460,289` FP32 values and must never be presented as the original logical parameter count.

## Stop-the-Line Conditions

The worker must stop the current task, preserve the last green commit, and report the exact failing command if any of these occurs:

- the archived original residual activation order cannot be proved from source or a full state-dict/module manifest;
- a supposedly semantic-free refactor changes strict FP32 outputs, gradients, running statistics, or Adam updates;
- a two-rank test silently skips instead of proving that two GPUs participated;
- the build manifest does not prove `Release` and the expected CUDA architecture;
- an FP16 policy passes only after loosening the strict FP32 oracle;
- rank collective order differs, a job hangs, or NCCL reports an asynchronous error;
- a candidate requires more than 14.5 GiB peak memory per T4;
- a cached policy fingerprint differs from the running hardware/software/workload fingerprint;
- a full or final batch uses padding, duplicates, or dropped samples.

## Execution Protocol for a Low-Reasoning Worker

At the beginning of every new PowerShell session, define the following helpers before running any native command:

```powershell
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
function Invoke-NativeChecked([string]$Label, [scriptblock]$Command) {
    & $Command
    $code = $LASTEXITCODE
    if ($code -ne 0) { throw "$Label failed with exit code $code" }
}
function Get-NativeChecked([string]$Label, [scriptblock]$Command) {
    $lines = @(& $Command)
    $code = $LASTEXITCODE
    if ($code -ne 0) { throw "$Label failed with exit code $code" }
    return $lines
}
function Assert-StagedExactly([string[]]$Expected) {
    $actual = @(Get-NativeChecked 'read staged paths' { git diff --cached --name-only })
    $delta = @(Compare-Object ($Expected | Sort-Object) ($actual | Sort-Object))
    if ($delta.Count -ne 0) { throw "unexpected staged payload: $($delta | Out-String)" }
    Invoke-NativeChecked 'cached diff check' { git diff --cached --check }
}
$script:ProtectedBackupPath = 'kaggle/kernel/run_ranks_2xt4.sh~31640'
function Assert-ProtectedBackupUnchanged {
    if (-not (Test-Path -LiteralPath $script:ProtectedBackupPath -PathType Leaf)) {
        throw 'protected backup is missing or is not a leaf'
    }
    $info = Get-Item -LiteralPath $script:ProtectedBackupPath
    if ($info.Length -ne 5258 -or $info.LastWriteTimeUtc.Ticks -ne 639188887422510152) {
        throw 'protected backup metadata differs from the plan-frozen baseline; do not inspect its contents'
    }
}
function Invoke-ExactCommit([string]$Message, [string[]]$Paths, [string[]]$ForcePaths = @()) {
    Assert-ProtectedBackupUnchanged
    $already = @(Get-NativeChecked 'read pre-existing index' { git diff --cached --name-only })
    if ($already.Count -ne 0) { throw "index must be empty before staging: $already" }
    if ($Paths.Count -gt 0) { Invoke-NativeChecked 'stage task paths' { git add -- $Paths } }
    if ($ForcePaths.Count -gt 0) { Invoke-NativeChecked 'force-stage evidence paths' { git add -f -- $ForcePaths } }
    $expected = @($Paths) + @($ForcePaths)
    Assert-StagedExactly $expected
    Invoke-NativeChecked 'inspect status before commit' { git status --short }
    Invoke-NativeChecked 'commit exact task payload' { git commit -m $Message }
    Assert-ProtectedBackupUnchanged
}
Assert-ProtectedBackupUnchanged
```

PowerShell 5.1 does not throw for a native nonzero exit. Therefore every bare native line in later snippets is a readable command manifest, not permission to run an unchecked multiline block: execute it through `Invoke-NativeChecked`; capture output only through `Get-NativeChecked`. Treat each task's `git add` paths as the exact `$Paths`/`$ForcePaths` arguments to `Invoke-ExactCommit`. Never call `git commit` with a pre-populated index, and wrap every push before reading or using the resulting SHA. Run `Assert-ProtectedBackupUnchanged` as the first operation of every new shell/task, immediately before every commit, and immediately after every push/final exit path. The literal length/tick pair above is the metadata-only baseline frozen while creating this already-committed plan; never recapture it from a possibly changed file. This global rule applies to Tasks 1-17 and supersedes shorter snippets. The protected backup is never an allowed task path.

For every task below:

1. Read only the listed implementation and test files first.
2. Run the listed red test before changing production code.
3. Make the smallest change that satisfies that task; do not pull forward later optimizations.
4. Run the focused green tests, then the broader gate.
5. For a performance task, run correctness before timing and preserve the baseline artifact.
6. Record accepted and rejected candidates; never silently replace a previous measurement.
7. Commit only after `git diff --check` succeeds.
8. Before every `ctest`, build the current tree with `cmake --build native/build --config Release --parallel` (or the task's explicit external build directory) and pass `--no-tests=error`; zero discovered tests is a failure, never a green result.
9. Push only a green commit. Do not combine two numbered tasks in one commit.

---

## Task 1: Repair and Freeze the Exact Original P888 Contract

**Files:**

- Create: `native/tests/fixtures/p888_original_contract.json`
- Create: `native/tests/reference/extract_p888_original_contract.py`
- Create: `test_results/original_p888_metadata/original_checkpoint_manifest.json`
- Create: `scripts/acquire_p888_original_checkpoints.py`
- Create: `scripts/tests/test_acquire_p888_original_checkpoints.py`
- Modify: `native/include/mgt/config.hpp`
- Modify: `native/tests/test_p888_training_contract.cpp`
- Modify: `docs/superpowers/specs/2026-07-28-p888-original-parity-optimized-training-design.md`
- Modify: `docs/superpowers/plans/2026-07-28-p888-original-parity-optimized-training.md`
- Modify: `native/CMakeLists.txt`

- [ ] Add a superseded notice to the old plan that links to this delta plan and says its unchecked implementation steps must not be replayed.

- [ ] Treat original-model acquisition below as Task 1's first mutation and stop-line. Do not edit this test or any production/design file until acquisition has produced at least one `available=true` model. Then add failing runtime checks to `test_p888_training_contract.cpp` so the red build succeeds and CTest cannot accidentally execute a stale binary:

  ```cpp
  if (mgt::P888TrainingContract::kInputFeatures !=
      mgt::kStateLen * mgt::kStateValuePad) return EXIT_FAILURE;
  if (mgt::P888TrainingContract::kInputFeatures != 5184) return EXIT_FAILURE;
  if (mgt::P888TrainingContract::kOriginalLogicalLinearParameters !=
      15338249ULL) return EXIT_FAILURE;
  if (mgt::P888TrainingContract::kOriginalBatchNormAffineParameters !=
      19500ULL) return EXIT_FAILURE;
  if (mgt::P888TrainingContract::kOriginalLogicalParameters !=
      15357749ULL) return EXIT_FAILURE;
  ```

  After the constants are corrected and the red/green cycle is observed, add equivalent `static_assert` statements as compile-time guards.

- [ ] Run:

  ```powershell
  cmake --build native/build --config Release --target test_p888_training_contract
  cmake --build native/build --config Release --parallel
  ctest --test-dir native/build -C Release -R "^p888_training_contract$" --output-on-failure --no-tests=error
  ```

  Expected before the fix: `p888_training_contract` fails on `6336`.

- [ ] Implement `extract_p888_original_contract.py` so it reads:

  - both run metadata files `test_results/original_p888_metadata/model_p888-t000_1778521793.json` and `test_results/original_p888_metadata/model_p888-t000_1780290207.json`;
  - canonical trainer sources `test_results/original_p888_code/train.txt` and `test_results/original_p888_code/trainer.txt`;
  - mirrors `test_results/archive_inspect_20260704/train.txt` and `trainer.txt`, which must SHA256-match the canonical copies;
  - `test_results/archive_inspect_20260704/888_lern.ipynb`, `utils.txt`, `p888.json`, and `native/production_inputs/p888.json`;
  - one acquired original `state_dict` checkpoint plus either the original `Pilgrim` model source or a complete module/tensor-key manifest sufficient to prove residual order.

  Treat both 817-byte `p888-t000.pt` files as target tensors, not model checkpoints. Hash-check that the two target copies match, but never add them to the comparison-checkpoint list.

- [ ] Acquire original model artifacts read-only before the extractor stop-line. `acquire_p888_original_checkpoints.py` accepts repeatable `--dataset` and exact `--output-root test_results/original_p888_checkpoints`; invoke it for `arabidopsisthalian/ihes-model-1778521793` and `arabidopsisthalian/model-ihes-1780290207-e40960`. It calls only checked read-only Kaggle list/download operations, downloads into a fresh temporary directory, rejects auth/network/partial downloads, securely extracts without absolute/backslash/`..`/symlink/duplicate paths, and atomically publishes under an ignored dataset-slug directory. Existing bytes are reused only when every path/size/SHA matches; a mismatch is never overwritten. Load candidate tensors on CPU with `torch.load(..., map_location='cpu', weights_only=True)` or an explicitly safe tensor format, reject executable/custom pickle objects, and emit canonical dataset/file/SHA256/size/tensor-key/shape/dtype manifests. Unit tests use a fake Kaggle CLI and malicious archives; no GPU or dataset mutation is allowed.

  ```powershell
  Invoke-NativeChecked 'original checkpoint acquisition tests' { py -m unittest scripts.tests.test_acquire_p888_original_checkpoints -v }
  Invoke-NativeChecked 'download original checkpoints read-only' { py scripts/acquire_p888_original_checkpoints.py --dataset arabidopsisthalian/ihes-model-1778521793 --dataset arabidopsisthalian/model-ihes-1780290207-e40960 --output-root test_results/original_p888_checkpoints }
  ```

  This read-only acquisition is the only Kaggle action allowed before Task 16. It must not push a kernel, create/version a dataset, or submit GPU work.

  `acquire_p888_original_checkpoints.py` emits only the canonical `test_results/original_p888_metadata/original_checkpoint_manifest.json`: a closed ordered list of every comparison checkpoint with exact path, SHA256, model ID, tensor-key manifest, and `available: true|false`. After acquisition, `extract_p888_original_contract.py` consumes that manifest and emits `native/tests/fixtures/p888_original_contract.json` with source paths/hashes, layer order, residual activation order, BN site order, input dimension, logical parameter formulas, optimizer/data schedule, and the source-backed random-walk start state, inverse-move exclusion, transition, and target-label definition. No later task may interpret “relevant checkpoint” outside the checkpoint manifest.

- [ ] Freeze the original data semantics from `trainer.txt`: move IDs use the source `all_moves` order; inverse IDs use the exact `generate_inverse_moves(move_names)` result; the initial `last_moves=-1` deliberately excludes `inverse_moves[-1]` on move zero; depths are shuffled by one global `torch.randperm(total)` before batching. Store these facts, the ordered move-name/inverse arrays, and the original RNG/shuffle behavior in the fixture. The later counter-based generator may change random bits, as initialization/sequence parity is not required, but it must preserve this move-zero quirk, legal-move distribution, and one global epoch permutation.

- [ ] Make checkpoint availability a Task 1 stop-line, not a final surprise. `original_checkpoint_manifest.json` may mark known historical checkpoints unavailable, but it must contain at least one `available=true` real model state_dict with verified SHA256, nonempty tensor-key manifest, and original model ID before Task 2 starts. If no checkpoint has been acquired, do not touch the later Task-1 files. Commit only the already-tested acquisition tooling plus the unavailable manifest with this exact blocked payload, then stop and report the missing expected artifact names; Task 1 remains incomplete and Task 2 must not start:

  ```powershell
  $blockedTask1Paths = @('scripts/acquire_p888_original_checkpoints.py','scripts/tests/test_acquire_p888_original_checkpoints.py')
  $blockedTask1Evidence = @('test_results/original_p888_metadata/original_checkpoint_manifest.json')
  Invoke-ExactCommit 'test: record unavailable original p888 checkpoints' $blockedTask1Paths $blockedTask1Evidence
  ```

  Before that blocked commit, require the working/index payload after excluding the protected backup and ignored download cache to contain exactly those three paths; otherwise stop without committing. When an original model later becomes available, resume Task 1 after this stop-line and make the normal full Task-1 commit from the remaining listed paths.
- [ ] Make the extractor fail if the original residual order is not explicitly proved. The accepted order must state whether the block is:

  ```text
  relu(skip + BN(fc2(relu(BN(fc1(skip)))))))
  ```

  or a different source-backed order. Do not infer it from the current native code.

- [ ] Set in `P888TrainingContract`:

  ```cpp
  static constexpr std::uint32_t kInputFeatures =
      kStateLen * kStateValuePad;
  static constexpr std::uint64_t kOriginalLogicalLinearParameters = 15338249ULL;
  static constexpr std::uint64_t kOriginalBatchNormAffineParameters = 19500ULL;
  static constexpr std::uint64_t kOriginalLogicalParameters = 15357749ULL;
  static constexpr std::uint64_t kNativePhysicalLinearParameters = 15460289ULL;
  ```

- [ ] Extend the test to assert all 34 BN site names in forward order, no output BN, the exact Adam values, and the full semantic-epoch schedule.

- [ ] Keep Task 1 padding checks static: physical dimensions are implementation storage, logical fingerprints use logical shapes, and export omits padded lanes. Runtime zero-after-update checks belong to the CUDA full-step fixture task and must not be claimed by this CPU contract test.

- [ ] Regenerate the fixture twice and compare hashes:

  ```powershell
  py native/tests/reference/extract_p888_original_contract.py
  $first = (Get-FileHash native/tests/fixtures/p888_original_contract.json -Algorithm SHA256).Hash
  py native/tests/reference/extract_p888_original_contract.py
  $second = (Get-FileHash native/tests/fixtures/p888_original_contract.json -Algorithm SHA256).Hash
  if ($first -ne $second) { throw "contract fixture is not deterministic" }
  ```

  Expected: identical hashes.

- [ ] Run:

  ```powershell
  cmake --build native/build --config Release --parallel
  ctest --test-dir native/build -C Release -R "p888_training_contract|model_layout|weight_export" --output-on-failure --no-tests=error
  git diff --check
  ```

  Expected: all selected tests pass.

- [ ] Commit through exactly one of the normal/resumed paths; never expect already-committed unchanged acquisition files in the index:

  ```powershell
  $task1AcquisitionPaths = @('scripts/acquire_p888_original_checkpoints.py','scripts/tests/test_acquire_p888_original_checkpoints.py')
  $task1RemainingPaths = @('native/tests/fixtures/p888_original_contract.json','native/tests/reference/extract_p888_original_contract.py','native/include/mgt/config.hpp','native/tests/test_p888_training_contract.cpp','docs/superpowers/specs/2026-07-28-p888-original-parity-optimized-training-design.md','docs/superpowers/plans/2026-07-28-p888-original-parity-optimized-training.md','native/CMakeLists.txt')
  $task1Evidence = @('test_results/original_p888_metadata/original_checkpoint_manifest.json')
  $blockedRows = @(Get-NativeChecked 'inspect blocked Task-1 history' { git log --format='%H%x09%s' c770e20..HEAD } | Where-Object { $_ -match '^(?<sha>[0-9a-f]{40})\ttest: record unavailable original p888 checkpoints$' })
  if ($blockedRows.Count -gt 1) { throw 'multiple blocked Task-1 commits found' }
  if ($blockedRows.Count -eq 0) {
      $task1CommitPaths = @($task1RemainingPaths + $task1AcquisitionPaths)
  } else {
      $blockedSha = ([regex]::Match($blockedRows[0], '^[0-9a-f]{40}')).Value
      $blockedExpected = @($task1AcquisitionPaths + $task1Evidence | Sort-Object)
      $blockedActual = @(Get-NativeChecked 'verify blocked Task-1 payload' { git show --format= --name-only $blockedSha } | Where-Object { $_ } | Sort-Object)
      if ((Compare-Object $blockedExpected $blockedActual).Count -ne 0) { throw 'blocked Task-1 commit payload is not exact' }
      foreach ($path in $task1AcquisitionPaths) { if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { throw "blocked acquisition file disappeared: $path" } }
      $changedAcquisition = @(Get-NativeChecked 'inspect resumed acquisition changes' { git diff --name-only HEAD -- $task1AcquisitionPaths })
      $task1CommitPaths = @($task1RemainingPaths + $changedAcquisition | Sort-Object -Unique)
  }
  Invoke-ExactCommit 'fix: prove exact original p888 training contract' $task1CommitPaths $task1Evidence
  ```

---

## Task 2: Expand the Minified CUDA Hot Path Without Semantic Changes

**Files:**

- Modify formatting only: `native/cuda/sync_batch_norm.cu`
- Modify formatting only: `native/cuda/mlp_batch_norm_forward.cu`
- Modify formatting only: `native/cuda/batch_norm_activation.cu`

- [ ] Save the strict test output and SHA256 hashes of the three files before editing.

- [ ] Reflow the existing code using whitespace and line breaks only. Do not rename identifiers, add comments, change launch geometry, change operation order, change precision, or change any API.

- [ ] Verify the patch is whitespace/comment-only:

  ```powershell
  git diff --ignore-all-space --exit-code -- native/cuda/sync_batch_norm.cu native/cuda/mlp_batch_norm_forward.cu native/cuda/batch_norm_activation.cu
  ```

  Expected: exit code 0.

- [ ] Run:

  ```powershell
  docker exec mgt-gpu-queue python3 scripts/gpu_queue_submit.py --label task2-format-regression --wait -- bash -lc "cmake --build native/build --config Release --parallel && ctest --test-dir native/build -C Release -R 'batch_norm_activation|sync_batch_norm_strided|sync_batch_norm_2rank|mlp_batch_norm_forward|mlp_batch_norm_output_backward|mlp_batch_norm_residual_fc2_backward|mlp_batch_norm_residual_stack_backward|mlp_batch_norm_hidden_backward|mlp_batch_norm_full_backward' --output-on-failure --no-tests=error"
  git diff --check
  ```

  Expected: every listed test passes with unchanged strict results.

- [ ] Commit:

  ```powershell
  git add native/cuda/sync_batch_norm.cu native/cuda/mlp_batch_norm_forward.cu native/cuda/batch_norm_activation.cu
  git commit -m "style: expand sync batchnorm cuda hot path"
  ```

---

## Task 3: Generate a Full-Step Numerical Oracle and Unify Parameter Offsets

**Files:**

- Create: `native/cuda/mgt_cuda/mlp_parameter_layout.cuh`
- Modify: `native/cuda/mlp_forward.cu`
- Modify: `native/cuda/mlp_backward.cu`
- Modify: `native/cuda/mlp_batch_norm_forward.cu`
- Create: `native/tests/reference/generate_p888_full_step_fixture.py`
- Create: `native/tests/fixtures/p888_full_step_fixture.hpp`
- Create: `native/tests/fixtures/p888_full_step_fixture_manifest.json`
- Create: `native/tests/cuda/test_mlp_batch_norm_pytorch_fixture.cu`
- Modify: `native/tests/test_model_layout.cpp`
- Modify: `native/CMakeLists.txt`

- [ ] Add a failing layout test for one shared contiguous CUDA parameter map. Define:

  ```cpp
  struct CudaMlpParameterLayout {
      std::uint64_t input_weight = 0;
      std::uint64_t input_bias = 0;
      std::uint64_t hidden_weight = 0;
      std::uint64_t hidden_bias = 0;
      std::uint64_t residual_base = 0;
      std::uint64_t residual_block_stride = 0;
      std::uint64_t output_weight = 0;
      std::uint64_t output_bias = 0;
      std::uint64_t total = 0;

      __host__ __device__ std::uint64_t ResidualFc1Weight(
          std::uint32_t block) const;
      __host__ __device__ std::uint64_t ResidualFc1Bias(
          std::uint32_t block) const;
      __host__ __device__ std::uint64_t ResidualFc2Weight(
          std::uint32_t block) const;
      __host__ __device__ std::uint64_t ResidualFc2Bias(
          std::uint32_t block) const;
  };

  __host__ __device__ constexpr CudaMlpParameterLayout
  BuildCudaMlpParameterLayout(const CudaMlpShape& shape);

  enum class CudaMlpLogicalTensor : std::uint32_t {
      kInputWeight = 1,
      kInputBias = 2,
      kHiddenWeight = 3,
      kHiddenBias = 4,
      kResidualFc1Weight = 5,
      kResidualFc1Bias = 6,
      kResidualFc2Weight = 7,
      kResidualFc2Bias = 8,
      kOutputWeight = 9,
      kOutputBias = 10,
  };

  struct CudaMlpLogicalProjection {
      std::uint64_t physical_offset = 0;
      std::uint32_t logical_rows = 0;
      std::uint32_t logical_cols = 0;
      std::uint32_t physical_row_stride = 0;
  };

  __host__ __device__ constexpr CudaMlpLogicalProjection
  BuildCudaMlpLogicalProjection(
      const CudaMlpShape& shape,
      CudaMlpLogicalTensor tensor,
      std::uint32_t residual_block);
  ```

- [ ] Make the map describe the actual contiguous physical CUDA training buffer with no hidden 64-byte gaps and assert P888 `total == 15,460,289`. Add a logical projection record `{physical_offset, logical_rows, logical_cols, physical_row_stride}` for each original tensor. CUDA forward/backward, the BatchNorm step, and schema-v3 checkpoint must agree on physical slices; CPU logical layout and export must agree with the projection that selects logical rows/columns and omits padding. Never assert equality between the `15,338,249` logical linear count and `15,460,289` physical count.

- [ ] Replace private duplicate offset arithmetic in the three listed CUDA files with the common layout. This is a semantic-preservation refactor; run existing forward/backward/full-BN tests before generating a new fixture.

- [ ] Implement `generate_p888_full_step_fixture.py` with:

  ```text
  torch.manual_seed(888)
  state_len = 3
  value_pad = 4
  logical/physical hd1 = 5/8
  logical/physical hd2 = 3/8
  residual_blocks = 16
  output_dim = 1
  rows = 7
  rank split = 4 + 3
  epsilon = 1e-5
  momentum = 0.1
  Adam = (lr 1e-4, beta1 0.9, beta2 0.999, eps 1e-8, wd 0)
  ```

- [ ] Build the network in the exact source-backed residual order from Task 1. Use nonsymmetric deterministic weights, negative and positive gamma, nonzero beta/running state, nonuniform states, and nonzero labels.

- [ ] Emit every value as an exact IEEE-754 bit pattern in `p888_full_step_fixture.hpp`. The header must include:

  ```text
  states and labels
  initial linear weights and Adam m/v
  initial BN affine/running state and Adam m/v
  forward output and global MSE
  every linear gradient slice
  all 34 dgamma/dbeta slices
  updated running mean/variance for all 34 sites
  one-step updated linear/affine weights and Adam m/v
  folded-evaluation output
  two-rank local inputs and global expected results
  ```

- [ ] Emit `p888_full_step_fixture_manifest.json` with schema version, generator SHA256, PyTorch version, seed, residual order, tensor names, shapes, dtypes, element counts, byte order, and SHA256 for each generated array.

- [ ] Regenerate twice and require byte-identical header and manifest. Do not require PyTorch at normal CTest time; CTest consumes checked-in fixture data.

- [ ] In `test_mlp_batch_norm_pytorch_fixture.cu`, compare all listed tensors. Use:

  ```text
  strict FP32 kernel atol = 6e-5, rtol = 1e-4
  global SyncBN reduction atol = 2e-4, rtol = 2e-4
  Adam one-step atol = 2e-6
  padding exact zero
  ```

- [ ] Make the fixture nonsymmetric across input/output axes so a transposed parameter map or wrong-axis BN fold fails clearly.

- [ ] Run:

  ```powershell
  py native/tests/reference/generate_p888_full_step_fixture.py
  cmake --build native/build --config Release --target test_mlp_batch_norm_pytorch_fixture
  docker exec mgt-gpu-queue python3 scripts/gpu_queue_submit.py --label task3-oracle-gates --wait -- bash -lc "cmake --build native/build --config Release --parallel && ctest --test-dir native/build -C Release -R 'model_layout|cuda_mlp_forward_smoke|cuda_mlp_backward_cpu_compare|mlp_batch_norm_full_backward|mlp_batch_norm_pytorch_fixture' --output-on-failure --no-tests=error"
  git diff --check
  ```

- [ ] Commit:

  ```powershell
  git add native/cuda/mgt_cuda/mlp_parameter_layout.cuh native/cuda/mlp_forward.cu native/cuda/mlp_backward.cu native/cuda/mlp_batch_norm_forward.cu native/tests/reference/generate_p888_full_step_fixture.py native/tests/fixtures/p888_full_step_fixture.hpp native/tests/fixtures/p888_full_step_fixture_manifest.json native/tests/cuda/test_mlp_batch_norm_pytorch_fixture.cu native/tests/test_model_layout.cpp native/CMakeLists.txt
  git commit -m "test: add full sync batchnorm step oracle"
  ```

---
## Task 4: Make Distributed Batch Partitioning and the Benchmark Trustworthy

**Files:**

- Create: `native/include/mgt/distributed_batch.hpp`
- Create: `native/src/distributed_batch.cpp`
- Create: `native/tests/test_distributed_batch.cpp`
- Create: `native/include/mgt/benchmark_data.hpp`
- Create: `native/src/benchmark_data.cpp`
- Create: `native/tests/test_benchmark_data.cpp`
- Create: `native/cuda/mgt_cuda/mlp_batch_norm_profile.cuh`
- Create: `scripts/cluster/run_p888_bn_a100_release.sbatch`
- Create: `scripts/summarize_bn_benchmark.py`
- Create: `scripts/tests/test_summarize_bn_benchmark.py`
- Modify: `native/cuda/mgt_cuda/mlp_batch_norm_forward.cuh`
- Modify: `native/cuda/mlp_batch_norm_forward.cu`
- Modify: `native/cuda/sync_batch_norm.cuh`
- Modify: `native/cuda/sync_batch_norm.cu`
- Rewrite for readability: `native/tools/mgt_bn_step_benchmark.cu`
- Modify: `native/CMakeLists.txt`

- [ ] Add a failing CPU test for deterministic quotient/remainder partitioning:

  ```cpp
  struct RankBatchSlice {
      std::uint32_t rank = 0;
      std::uint32_t local_rows = 0;
      std::uint64_t global_offset = 0;
  };

  mgt::Status PartitionGlobalBatch(
      std::uint64_t global_rows,
      std::uint32_t world_size,
      std::uint32_t rank,
      RankBatchSlice* out);
  ```

  Required results:

  ```text
  100000 / 2 -> 50000, 50000
   99978 / 2 -> 49989, 49989
  100000 / 8 -> 8 x 12500
   99978 / 8 -> 12498, 12498, then 6 x 12497
  ```

- [ ] Implement `PartitionGlobalBatch` with checked arithmetic and assert that all slices are contiguous, non-overlapping, and sum to `global_rows`. Return `kInvalidConfig` for null output, zero world, or `rank>=world`; return `kCapacityExceeded` if a local row count cannot fit uint32. Test all failures plus `global_rows=0`.

- [ ] Build max-rank summaries offline from per-rank JSONL after the timed region. Do not insert a timing allreduce into the measured collective graph.

- [ ] Define a benchmark-only profile structure:

  ```cpp
  struct MlpBatchNormStageProfile {
      float forward_ms = 0.0f;
      float output_loss_ms = 0.0f;
      float residual_backward_ms = 0.0f;
      float hidden_backward_ms = 0.0f;
      float input_relu_ms = 0.0f;
      float input_bn_local_ms = 0.0f;
      float input_bn_collective_ms = 0.0f;
      float input_bn_dx_ms = 0.0f;
      float input_table_grad_ms = 0.0f;
      float input_bias_grad_ms = 0.0f;
      float input_weight_collective_ms = 0.0f;
      float linear_adam_ms = 0.0f;
      float affine_adam_ms = 0.0f;
      float end_to_end_ms = 0.0f;
  };
  ```

- [ ] Add a `SyncBatchNormStageProfile` with local-reduction, collective, finalize, and dX fields. Add an explicitly named profiled overload in `sync_batch_norm.cuh`; keep the existing unprofiled wrapper and route both through one internal implementation.

- [ ] Add NVTX ranges with the same stable names. Only the profiled overload may use caller-owned timing events. When no profile pointer is supplied, the production path must create no timing events and perform no synchronization.

- [ ] Replace the benchmark positional CLI with explicit flags:

  ```text
  --device
  --rank
  --world
  --nccl-id-file
  --global-rows
  --warmup
  --steps
  --repeats
  --pair-index
  --pair-attempt-nonce
  --pair-role standalone|baseline|candidate
  --pair-order
  --seed
  --math fp32|tf32|fp16_candidate
  --profile-stages 0|1
  --allow-smoke 0|1
  --jsonl-dir
  ```

- [ ] Remove the invalid condition `world * local_rows == global_rows`. Compute `local_rows` and `global_offset` from `PartitionGlobalBatch`.

- [ ] Replace zero initialization with a reusable, deterministic nonzero fixture in `benchmark_data.cpp`. All unsigned operations below wrap modulo `2^32`:

  ```cpp
  std::uint32_t XorShiftWord(std::uint32_t seed,
                             std::uint32_t stream_tag,
                             std::uint64_t index) {
      std::uint32_t x = seed ^ stream_tag ^
          static_cast<std::uint32_t>(index * UINT64_C(0x9E3779B9));
      x ^= x << 13;
      x ^= x >> 17;
      x ^= x << 5;
      return x;
  }
  double Unit24(std::uint32_t x) {
      return static_cast<double>(x >> 8) * 0x1p-24;
  }
  ```

  Use stream tags `0xA341316C` for linear weights, `0xC8013EA4` for gamma, `0xAD90777D` for beta, `0x7E95761E` for running mean, and `0x6C8E9CF5` for running variance. Convert only once at the end of each double-precision affine expression:

  ```text
  state[row,pos] = (17*global_sample_id + 29*pos + seed) mod 72
  label[row]     = float(0.01 * (int64(global_sample_id mod 101) - 50))
  weight[i]      = float(-0.02 + 0.04*Unit24(XorShiftWord(seed, weight_tag, i)))
  gamma[i]       = float( 0.90 + 0.20*Unit24(XorShiftWord(seed, gamma_tag, i)))
  beta[i]        = float(-0.05 + 0.10*Unit24(XorShiftWord(seed, beta_tag, i)))
  running_mean[i]= float(-0.10 + 0.20*Unit24(XorShiftWord(seed, mean_tag, i)))
  running_var[i] = float( 0.80 + 0.40*Unit24(XorShiftWord(seed, var_tag, i)))
  ```

  Every value depends only on `(seed, global_sample_id, logical_parameter_index)`, never rank or world size. The same logical sample must be bit-identical under 1, 2, and 8-rank partitions. Every categorical value must occur at every position in a production-size run. The first red test independently constructs seed 888, samples 0..15 with all 72 positions, then 32 values from each parameter stream, serializes state bytes followed by little-endian FP32 labels/streams, and requires SHA256 `2cc9366a3bc284fd82d6e45fe2e34413175a5caa3ba70f3da559c61afc8b81d6`. Keep the literal digest in the test; do not obtain the expected value by calling the C++ generator under test.

- [ ] Define repeat identity before timing. Acceptance requires `--pair-index 0|1|2`, a controller-generated 32-lowercase-hex `--pair-attempt-nonce`, `--pair-role standalone|baseline|candidate`, and `--pair-order 0|1`. Each fresh process uses `--repeats 1`; JSONL stores all four fields plus in-process `repeat=0`. The logical repeat ID is `pair_index`; reject another repeat, a reused nonce outside its declared pair, or role/order inconsistent with the pair manifest. Task-4 standalone A100 baselines use one unique nonce per case/pair with role `standalone`, order 0. Smoke/diagnostic runs may omit pair identity, but their rows are ineligible.

- [ ] Separate measurement modes without creating misleading percentiles:

  - acceptance (`--profile-stages 0`) precreates `2*steps` CUDA events per repeat, records a start/stop pair around every complete step, performs no event synchronization/readback until all steps in that repeat have been enqueued, then synchronizes the final stop event and emits one JSONL row per `(case,rank,repeat,step)`; it may set `acceptance_eligible=true`;
  - diagnostic profiling (`--profile-stages 1`) records per-step/per-stage rows, may synchronize caller-owned events after the diagnostic region, and always sets `acceptance_eligible=false`.

  Event creation/destruction, JSON serialization, loss copies, and NVML polling remain outside the timed loop. The event array is benchmark-owned; the production trainer creates no timing events.

- [ ] Implement `summarize_bn_benchmark.py` to join rank rows by `(case,policy_checksum,pair_index,pair_attempt_nonce,pair_role,pair_order,repeat,step)`, reject missing/duplicate ranks or mismatched row vectors, and require `pair_index={0,1,2}`, `repeat=0`, and 100 steps per acceptance process. Standalone mode requires role/order `standalone/0` and three distinct nonces. Paired mode additionally consumes canonical Task-14 pair manifests and rejects any cross-policy mismatch. For each logical step take the maximum elapsed time and corresponding slow-rank ID, then compute statistics over 300 max-rank samples per policy/case. Sort ascending and define nearest-rank quantile as element `max(0, ceil(q*n)-1)` for `q in {0.05,0.50,0.95}`. Emit min, p5, median, p95, max, median throughput, and each pair-index median. Test missing/duplicate pair indices/nonces, wrong role/order, duplicate rank rows, mismatched policies/vectors, slow-rank changes, and literal `n=300` quantile indices.

- [ ] Require at least 20 warmup and 100 measured steps for an acceptance run. Permit smaller counts only with `--allow-smoke 1`. Any invocation with `--allow-smoke 1`, regardless of its counts, must emit `"acceptance_eligible": false`; this is also how external-profiler runs are excluded.

- [ ] Map math policies exactly and record both requested and effective modes:

  ```text
  fp32: cublasSetMathMode(CUBLAS_DEFAULT_MATH), Lt compute CUBLAS_COMPUTE_32F
  tf32: cublasSetMathMode(CUBLAS_TF32_TENSOR_OP_MATH), Lt compute CUBLAS_COMPUTE_32F_FAST_TF32
  fp16_candidate: FP16 A/B with CUBLAS_COMPUTE_32F and FP32 output
  ```

  Query `cublasGetMathMode` after configuration and fail if it does not match the requested policy.

- [ ] Add a benchmark self-check that compares a small nonzero strict-FP32 step against the CPU/PyTorch fixture before any timing row is eligible.

- [ ] Create the exact MEPhI job script with two uninstrumented acceptance cases and one separate diagnostic case:

  ```bash
  #!/bin/bash
  #SBATCH --nodes=1
  #SBATCH --ntasks-per-node=8
  #SBATCH --cpus-per-task=4
  #SBATCH --gres=gpu:8
  #SBATCH --time=00:45:00

  set -euo pipefail
  : "${MGT_EXPECTED_TASK4_SHA:?MGT_EXPECTED_TASK4_SHA is required}"
  cd /mnt/pool/6/vokirova/p888-a100-smoke
  test "$(git -C source rev-parse HEAD)" = "$MGT_EXPECTED_TASK4_SHA"
  test -z "$(git -C source status --porcelain=v1 --untracked-files=all)"

  export CUTLASS_ROOT=/mnt/pool/3/vokirova/cutlass
  export NCCL_ROOT=/mnt/pool/3/vokirova/venvs/qwen3vl-full-sft-py311/lib/python3.11/site-packages/nvidia/nccl
  export LD_LIBRARY_PATH="$NCCL_ROOT/lib:${LD_LIBRARY_PATH:-}"
  export MGT_REQUIRE_TWO_GPUS=1

  build=source/build-a100-sm80-bn-release
  results="results/${SLURM_JOB_ID}"
  mkdir -p "$results"

  cmake -S source/native -B "$build" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
    -DMGT_ENABLE_CUDA=ON \
    -DMGT_ENABLE_NCCL=ON \
    -DCMAKE_CUDA_ARCHITECTURES=80 \
    -DMGT_CUTLASS_ROOT="$CUTLASS_ROOT" \
    -DNCCL_INCLUDE_DIR:PATH="$NCCL_ROOT/include" \
    -DNCCL_LIB:FILEPATH="$NCCL_ROOT/lib/libnccl.so.2"

  grep '^CMAKE_BUILD_TYPE:STRING=Release$' "$build/CMakeCache.txt"
  cmake --build "$build" --parallel 32 \
    --target test_mlp_batch_norm_pytorch_fixture test_mlp_batch_norm_full_backward mgt_bn_step_benchmark
  ctest --test-dir "$build" \
    -R 'mlp_batch_norm_pytorch_fixture|mlp_batch_norm_full_backward' \
    --output-on-failure --no-tests=error

  run_case() {
    export MGT_BENCH_CASE="$1"
    export MGT_BENCH_GLOBAL_ROWS="$2"
    export MGT_BENCH_WARMUP="$3"
    export MGT_BENCH_STEPS="$4"
    export MGT_BENCH_PAIR_INDEX="$5"
    export MGT_BENCH_PROFILE="$6"
    export MGT_BENCH_PAIR_NONCE="$(printf '%s' "${SLURM_JOB_ID}:${MGT_BENCH_CASE}:${MGT_BENCH_PAIR_INDEX}" | sha256sum | cut -c1-32)"
    export MGT_BENCH_ID_FILE="/mnt/pool/6/vokirova/p888-a100-smoke/nccl-bn-${SLURM_JOB_ID}-${MGT_BENCH_CASE}-p${MGT_BENCH_PAIR_INDEX}.id"
    export MGT_BENCH_OUT="/mnt/pool/6/vokirova/p888-a100-smoke/${results}/${MGT_BENCH_CASE}/pair-${MGT_BENCH_PAIR_INDEX}"
    rm -f "$MGT_BENCH_ID_FILE"
    mkdir -p "$MGT_BENCH_OUT"
    srun --ntasks=8 --cpus-per-task=4 bash -lc '
      exec /mnt/pool/6/vokirova/p888-a100-smoke/source/build-a100-sm80-bn-release/mgt_bn_step_benchmark \
        --device "$SLURM_LOCALID" \
        --rank "$SLURM_PROCID" \
        --world 8 \
        --nccl-id-file "$MGT_BENCH_ID_FILE" \
        --global-rows "$MGT_BENCH_GLOBAL_ROWS" \
        --warmup "$MGT_BENCH_WARMUP" \
        --steps "$MGT_BENCH_STEPS" \
        --repeats 1 \
        --pair-index "$MGT_BENCH_PAIR_INDEX" \
        --pair-attempt-nonce "$MGT_BENCH_PAIR_NONCE" \
        --pair-role standalone \
        --pair-order 0 \
        --seed 888 \
        --math fp32 \
        --profile-stages "$MGT_BENCH_PROFILE" \
        --allow-smoke "$MGT_BENCH_PROFILE" \
        --jsonl-dir "$MGT_BENCH_OUT"
    '
  }

  for pair_index in 0 1 2; do
    run_case full_acceptance 100000 20 100 "$pair_index" 0
    run_case final_acceptance 99978 20 100 "$pair_index" 0
  done
  run_case full_profile 100000 5 20 0 1

  python3 source/scripts/summarize_bn_benchmark.py \
    --input-root "$results" \
    --world-size 8 \
    --output "$results/summary.json" \
    --stage-output "$results/stage_summary.json"

  cp "$build/CMakeCache.txt" "$results/CMakeCache.txt"
  cp "$build/compile_commands.json" "$results/compile_commands.json"
  git -C source rev-parse HEAD > "$results/git-revision.txt"
  git -C source status --short > "$results/git-status.txt"
  nvidia-smi > "$results/nvidia-smi.txt"
  nvidia-smi topo -m > "$results/nvidia-smi-topology.txt"
  ```
- [ ] Before any external run, build the exact local source, run the focused gates, then commit and push the implementation so the cluster can execute an immutable revision:

  ```powershell
  $task4Files = @('native/include/mgt/distributed_batch.hpp','native/src/distributed_batch.cpp','native/tests/test_distributed_batch.cpp','native/include/mgt/benchmark_data.hpp','native/src/benchmark_data.cpp','native/tests/test_benchmark_data.cpp','native/cuda/mgt_cuda/mlp_batch_norm_profile.cuh','scripts/cluster/run_p888_bn_a100_release.sbatch','scripts/summarize_bn_benchmark.py','scripts/tests/test_summarize_bn_benchmark.py','native/cuda/mgt_cuda/mlp_batch_norm_forward.cuh','native/cuda/mlp_batch_norm_forward.cu','native/cuda/sync_batch_norm.cuh','native/cuda/sync_batch_norm.cu','native/tools/mgt_bn_step_benchmark.cu','native/CMakeLists.txt')
  Invoke-NativeChecked 'Task-4 queued gates' { docker exec mgt-gpu-queue python3 scripts/gpu_queue_submit.py --label task4-benchmark-gates --wait -- bash -lc "cmake --build native/build --config Release --parallel && ctest --test-dir native/build -C Release -R 'benchmark_data|distributed_batch|mlp_batch_norm_full_backward' --output-on-failure --no-tests=error" }
  Invoke-NativeChecked 'Task-4 summarizer tests' { py -m unittest scripts.tests.test_summarize_bn_benchmark -v }
  Invoke-NativeChecked 'Task-4 diff check' { git diff --check }
  Invoke-ExactCommit 'test: harden sync batchnorm benchmark baseline' $task4Files
  Invoke-NativeChecked 'push Task-4 commit' { git push origin HEAD:codex-native-trainer-implementation }
  $task4Sha = ((Get-NativeChecked 'read pushed Task-4 SHA' { git rev-parse HEAD })[0]).Trim()
  if ($task4Sha -notmatch '^[0-9a-f]{40}$') { throw 'invalid Task-4 SHA' }
  Write-Host "TASK4_SHA=$task4Sha"
  ```

  The only untracked path allowed after staging is the protected user backup. Stop if `git diff --cached --name-only` contains any path outside the command above.

- [ ] Configure and build on MEPhI only through the committed SLURM job. The user enters these commands in the Codex terminal; the implementing agent reads the output but never runs SSH:

  ```bash
  set -euo pipefail
  export MGT_TASK4_SHA='<literal 40-hex TASK4_SHA printed after the push>'
  cd /mnt/pool/6/vokirova/p888-a100-smoke
  git -C source fetch origin codex-native-trainer-implementation
  git -C source checkout --detach "$MGT_TASK4_SHA"
  test "$(git -C source rev-parse HEAD)" = "$MGT_TASK4_SHA"
  test -z "$(git -C source status --porcelain=v1 --untracked-files=all)"
  job_line=$(sbatch -p kaf12 --export=ALL,MGT_EXPECTED_TASK4_SHA="$MGT_TASK4_SHA" source/scripts/cluster/run_p888_bn_a100_release.sbatch)
  jid=$(printf '%s\n' "$job_line" | awk '/Submitted batch job/{print $4}')
  case "$jid" in (''|*[!0-9]*) echo "invalid sbatch result: $job_line" >&2; exit 1;; esac
  printf 'JOB_ID=%s\n' "$jid"
  scontrol show job "$jid"
  ```

  The implementing agent replaces the placeholder with the exact pushed SHA before handing the block to the user. The SLURM script checks `MGT_EXPECTED_TASK4_SHA` against `git -C source rev-parse HEAD` before configure; a mismatch exits before any benchmark.

- [ ] After completion, preserve:

  ```text
  full_acceptance/pair-0 through pair-2, each with manifest/correctness/config and rank0.jsonl through rank7.jsonl
  final_acceptance/pair-0 through pair-2, each with manifest/correctness/config and rank0.jsonl through rank7.jsonl
  full_profile/pair-0 with manifest/correctness/config and rank0.jsonl through rank7.jsonl
  summary.json covering both acceptance cases
  stage_summary.json covering only full_profile
  CMakeCache.txt
  compile_commands.json
  nvidia-smi.txt
  nvidia-smi-topology.txt
  git-revision.txt
  git-status.txt
  slurm-${jid}.out
  ```

  Require `git-revision.txt` to equal the pushed Task-4 SHA, `git-status.txt` to be empty, every acceptance pair directory to contain all eight rank files with exactly 100 rows/rank and matching `pair_index`, the combined full/final cases to contain exactly 300 max-rank samples each, and every manifest to say `Release`, SM80, world size 8, and `acceptance_eligible` matching its case.

- [ ] Run both global batches, including unequal A100 final-batch rows:

  ```text
  full:  100000
  final:  99978
  ```

- [ ] Explain the former roughly 10.8 ms gap between stage sum and continuous step using max-rank timings and Nsight Systems before attributing the gap to a kernel. This diagnostic evidence does not change the already-pushed Task-4 source commit.
---

## Task 5: Expose One Shared Input-Embedding-Gradient Primitive

**Files:**

- Create: `native/cuda/mgt_cuda/input_embedding_grad.cuh`
- Modify: `native/cuda/mlp_backward.cu`
- Create: `native/tests/cuda/test_cuda_input_embedding_grad.cu`
- Create: `scripts/run_cuda_2rank_test.py`
- Modify: `native/tests/cuda/test_cuda_mlp_backward_mixed_precision_error.cu`
- Modify: `native/CMakeLists.txt`

- [ ] Add a failing dedicated CUDA test with this CPU oracle:

  ```text
  grad[pos,value,h] =
      sum(dz[row,h] for every row where state[row,pos] == value)
  ```

- [ ] Define the exact public API:

  ```cpp
  enum class InputEmbeddingGradBackend : std::uint32_t {
      kOwnerWriteFp32Reference = 1,
      kPositionTiledGemmFp32 = 2,
      kPositionTiledGemmFp16 = 3,
  };

  enum class InputEmbeddingDzF16Source : std::uint32_t {
      kStageInternally = 1,
      kCallerProvided = 2,
  };

  struct InputEmbeddingGradConfig {
      InputEmbeddingGradBackend backend =
          InputEmbeddingGradBackend::kOwnerWriteFp32Reference;
      std::uint32_t positions_per_tile = 0;
      InputEmbeddingDzF16Source dz_f16_source =
          InputEmbeddingDzF16Source::kStageInternally;
  };

  struct InputEmbeddingGradWorkspace {
      void* base = nullptr;
      std::uint64_t bytes = 0;
      void* lt_base = nullptr;
      std::uint64_t lt_bytes = 0;
  };

  struct InputEmbeddingGradPlan;

  mgt::Status QueryInputEmbeddingGradWorkspaceBytes(
      const CudaMlpShape& shape,
      std::uint32_t capacity_rows,
      const InputEmbeddingGradConfig& config,
      std::uint64_t* required_bytes);

  mgt::Status CreateInputEmbeddingGradPlan(
      const CudaMlpShape& shape,
      const std::uint32_t* supported_active_rows,
      std::uint32_t supported_active_row_count,
      const InputEmbeddingGradConfig& config,
      std::uint32_t device_id,
      cublasHandle_t blas,
      cublasLtHandle_t blas_lt,
      std::uint64_t lt_workspace_bytes,
      InputEmbeddingGradPlan** plan);

  mgt::Status DestroyInputEmbeddingGradPlan(
      InputEmbeddingGradPlan* plan);

  mgt::Status LaunchInputEmbeddingGrad(
      const InputEmbeddingGradPlan* plan,
      const mgt::TrainStateStorage* states,
      std::uint32_t active_rows,
      const float* dz_f32,
      const __half* caller_dz_f16,
      float* table_grad_f32,
      std::uint32_t* device_invalid_state_flag,
      InputEmbeddingGradWorkspace workspace,
      cudaStream_t stream);
  ```

- [ ] Make `InputEmbeddingGradPlan` per rank/process and per device. `Create` must bind the exact `lt_workspace_bytes` capacity and build every cuBLAS/cuBLASLt descriptor and algorithm for each supported active-row count and for both full and tail tile widths. An unsupported row count is `kInvalidConfig`; `Launch` must never create a descriptor, choose an algorithm, allocate, autotune, create events, or synchronize.

- [ ] Remove the input-gradient path's dependence on the current global lazy Lt cache. Production uses one process per rank. Create `scripts/run_cuda_2rank_test.py`: accept `--exe`, `--timeout-seconds`, and repeated `--extra-arg`; verify two GPUs; generate a fresh rendezvous path from PID plus a cryptographic nonce; spawn rank 0 and rank 1 with `--device`, `--rank`, `--world 2`, and `--nccl-id-file`; capture separate stdout/stderr; enforce a 120-second default timeout; terminate both workers on either failure; remove the ID file. Return 77 below two GPUs only when `MGT_REQUIRE_TWO_GPUS` is absent, otherwise fail. Every two-rank CUDA CTest in later tasks must use this launcher, never two threads sharing global state.
- [ ] Keep the implementation initially in `mlp_backward.cu` so it reuses the existing one-hot builders and matmul dispatch code. Move the needed descriptor/algorithm state into the new per-rank `InputEmbeddingGradPlan`; do not call the old global lazy cache and do not copy a second independent matmul implementation.

- [ ] Enforce this primitive contract:

  - computes only the local table gradient;
  - does not compute the input bias;
  - does not invoke NCCL;
  - does not allocate or synchronize;
  - does not change a cuBLAS handle's math mode;
  - always outputs FP32 `[state_len * value_pad, physical_hd1]`;
  - uses `beta=0` for each disjoint tile;
  - ignores `positions_per_tile` for `kOwnerWriteFp32Reference`; for either tiled backend, rejects zero and clamps a positive value to `state_len`;
  - handles the final partial tile;
  - bounds-checks every categorical value, performs no out-of-range access, and atomically sets caller-owned `device_invalid_state_flag` on invalid input;
  - requires the trainer to clear that preallocated rank-local flag before generation; Task 6 globalizes it before any stateful update, while this primitive only sets it and performs safe local writes;
  - returns `kInvalidConfig` for an unknown backend;
  - returns `kCapacityExceeded` for insufficient workspace.

- [ ] Implement checked 64-bit workspace arithmetic from `config.dz_f16_source` rather than an independent query boolean:

  ```text
  tile      = min(positions_per_tile, state_len)
  tile_cols = tile * value_pad

  reference owner-write:
      0 bytes

  tiled FP32:
      capacity_rows * tile_cols * sizeof(float)

  tiled FP16, kCallerProvided:
      capacity_rows * tile_cols * sizeof(__half)

  tiled FP16, kStageInternally:
      align_up(capacity_rows * physical_hd1 * sizeof(__half), 32)
      + capacity_rows * tile_cols * sizeof(__half)
  ```

  Require `workspace.base` to be 32-byte aligned when `bytes>0`. Require `lt_base` to be 256-byte aligned when `lt_bytes>0`. Reference requires neither BLAS handle; tiled FP32 requires `blas`; tiled FP16 requires `blas_lt`. `kCallerProvided` requires non-null `caller_dz_f16`; `kStageInternally` requires it to be null. Recompute the ordinary scratch requirement inside `Launch` from the plan/config, require `workspace.lt_bytes == plan.lt_workspace_bytes`, and reject either mismatch before enqueue.
- [ ] Test:

  ```text
  rows=17, state_len=5, value_pad=8, hd1=16, tile=4
  tile=1
  tile=state_len
  tile>state_len
  value_pad=2 binary path
  partial final tile
  physical hd1 > logical hd1
  production geometry 72x72x2560 with moderate rows
  preconverted dz16
  internally converted dz16
  exact workspace
  workspace one byte too small
  null pointers
  tile zero
  invalid enum
  invalid categorical value sets the device flag and performs no out-of-range write
  size overflow
  reference backend with tile zero succeeds
  tiled backend with tile zero fails
  caller-provided policy with null dz16 fails
  internally-staged policy with non-null dz16 fails
  misaligned base and Lt workspaces fail
  launch Lt capacity smaller or larger than the plan-bound capacity fails
  unsupported active-row count fails without lazy plan creation
  ```

- [ ] Use tight CPU tolerance for both FP32 backends. Record per-tensor max absolute error, relative L2, norm ratio, and cosine for FP16; do not yet change a production default.

- [ ] Do not change the old no-BN trainer's public call signatures or runtime policy in this task. Factor only the one-hot builders and matmul launch implementation needed by the new prepared primitive, and keep the legacy lazy-cache call site behavior byte-for-byte covered by `cuda_mlp_backward_cpu_compare`, `cuda_mlp_backward_mixed_precision_error`, and `cuda_train_step_smoke`. A later production task may migrate that unrelated stream only under a separate parity/performance plan.

- [ ] Run:

  ```powershell
  docker exec mgt-gpu-queue python3 scripts/gpu_queue_submit.py --label task5-input-grad-gates --wait -- bash -lc "cmake --build native/build --config Release --parallel && ctest --test-dir native/build -C Release -R 'cuda_input_embedding_grad|cuda_mlp_backward_cpu_compare|cuda_mlp_backward_mixed_precision_error|cuda_train_step_smoke' --output-on-failure --no-tests=error"
  git diff --check
  ```
- [ ] Commit:

  ```powershell
  git add native/cuda/mgt_cuda/input_embedding_grad.cuh native/cuda/mlp_backward.cu native/tests/cuda/test_cuda_input_embedding_grad.cu scripts/run_cuda_2rank_test.py native/tests/cuda/test_cuda_mlp_backward_mixed_precision_error.cu native/CMakeLists.txt
  git commit -m "refactor: share tiled input embedding gradient"
  ```

---

## Task 6: Replace the BatchNorm Sparse Input Gradient Behind a Strict Policy

**Files:**

- Modify: `native/cuda/mgt_cuda/mlp_batch_norm_forward.cuh`
- Modify: `native/cuda/mlp_batch_norm_forward.cu`
- Modify: `native/cuda/sync_batch_norm.cuh`
- Modify: `native/cuda/sync_batch_norm.cu`
- Modify: `native/cuda/mgt_cuda/adamw.cuh`
- Modify: `native/cuda/adamw.cu`
- Modify: `native/tools/mgt_bn_step_benchmark.cu`
- Modify: `native/tests/cuda/test_mlp_batch_norm_full_backward.cu`
- Create: `native/tests/cuda/test_mlp_batch_norm_input_grad_2rank.cu`
- Modify: `native/CMakeLists.txt`

- [ ] Extend `MlpBatchNormStepBuffers` with caller-owned input-gradient scratch/capacity, `std::uint32_t* device_local_health_flag`, and `std::uint32_t* device_global_health_flag`. Extend the train-step call with a prepared `InputEmbeddingGradPlan*`; keep handles and the 256-byte-aligned Lt workspace in the outer per-rank runtime.

- [ ] For this task's independent benchmark/test invocation, clear both flags before generation. Bounds-check every category in the input-forward gather before a table read and in the input-gradient primitive before a table write. An invalid lane writes zero and atomically sets only the local flag; it never indexes the table. Task 15 explicitly replaces this per-invocation clear with sticky cross-step health in the real trainer; do not copy this standalone reset into that loop.

- [ ] Globalize health without adding a collective: reserve one FP32 word immediately after the first input-BN forward `sum/sumsq` payload. Its local-reduction kernel writes `float(local_flag != 0)`, and the existing input-BN NCCL SUM reduces `2*logical_hd1 + 1` floats. The finalize kernel writes `device_global_health_flag = reduced_health != 0`. Every rank executes this collective even when invalid, so all ranks obtain the same flag before the first running-state update. This changes bytes by four on that one call but keeps the 73-call reference graph unchanged.

- [ ] Every BN running-state/activation finalizer, backward producer, and Adam kernel receives the global flag. When nonzero, BN leaves running mean/variance byte-identical, activation/gradient producers write deterministic zeros, and Adam leaves parameters/moments/half mirror byte-identical; all ranks still enqueue the ordinary collective sequence with zero payloads. Read the global flag only at the delayed telemetry/validation boundary and abort all ranks together.

- [ ] Preserve this exact order:

  ```text
  ReLU backward
  -> global SyncBatchNorm backward in FP32
  -> local LaunchInputEmbeddingGrad
  -> existing FP32 ColumnSum for input bias
  -> SUM allreduce of input table plus bias
  ```

- [ ] Make `kOwnerWriteFp32Reference` the API default. Remove no code from the reference path.

- [ ] Parameterize `test_mlp_batch_norm_full_backward` over:

  ```text
  owner-write FP32 reference
  tiled GEMM FP32
  tiled GEMM FP16
  ```

- [ ] For FP32, retain the current approximately `6e-5` gradient tolerance and `2e-6` Adam-update tolerance. For FP16, record error statistics first and set a dedicated mixed gate no weaker than:

  ```text
  gradient relative L2 <= 0.08
  gradient max absolute <= 0.02
  finite loss/running state/moments
  ```

  Give loss, BN running statistics, and optimizer state their own tighter measured tolerances.

- [ ] Add a real two-rank test with deliberately different state/value and activation distributions, plus a case where only rank 1 contains one invalid category. Compare the NCCL-summed table and bias with one concatenated CPU logical batch. Return CTest skip code 77 when two GPUs are absent in an ordinary local run, set `SKIP_RETURN_CODE 77` in CMake, and require `MGT_REQUIRE_TWO_GPUS=1` in every 2x T4/8x A100 acceptance script so an acceptance run fails instead of skipping.

- [ ] Add production-width and padding assertions:

  ```text
  logical/physical hd1 = 2556/2560
  padded dz lanes = 0
  padded table-gradient lanes = 0
  padded parameters and Adam moments remain 0 after one step
  invalid state makes forward/gradient output harmless, sets the flag, and leaves parameters, moments, and running state byte-identical
  ```

- [ ] Run Compute Sanitizer on the partial-tile and exact-workspace cases through one checked PowerShell command:

  ```powershell
  Invoke-NativeChecked 'Task-6 input-gradient memcheck' { docker exec mgt-gpu-queue python3 scripts/gpu_queue_submit.py --label task6-input-grad-memcheck --wait -- bash -lc "compute-sanitizer --tool memcheck --error-exitcode 99 native/build/test_cuda_input_embedding_grad --case tail_exact_workspace" }
  ```

- [ ] Run only a bounded diagnostic smoke here: owner-write FP32, tiled FP32 tile 48, and tiled FP16 tile 48, all with the hardened benchmark, `--allow-smoke 1`, five warmups, twenty steps, and one repeat. This task proves integration and exposes gross regressions; it does not select a default or produce acceptance evidence.

- [ ] Defer the complete 2x T4 tile/Lt-workspace sweep and selection logic to Task 14, and run it authoritatively in Task 16; run any A100 tile matrix only as Task 16 diagnostic evidence. Keep cuBLASLt as the initial backend because P888 `physical_hd1=2560` does not satisfy the existing AUTO CUTLASS threshold. Keep `input_grad_sparse=true` as a named rejected experiment with its historical roughly 72-73k states/s result versus roughly 511k states/s for tiled FP16; never make it a production fallback.

- [ ] Run:

  ```powershell
  docker exec mgt-gpu-queue python3 scripts/gpu_queue_submit.py --label task6-bn-input-gates --wait -- bash -lc "cmake --build native/build --config Release --parallel && ctest --test-dir native/build -C Release -R 'cuda_input_embedding_grad|mlp_batch_norm_full_backward|mlp_batch_norm_input_grad_2rank' --output-on-failure --no-tests=error"
  git diff --check
  ```

- [ ] Commit:

  ```powershell
  git add native/cuda/mgt_cuda/mlp_batch_norm_forward.cuh native/cuda/mlp_batch_norm_forward.cu native/cuda/sync_batch_norm.cuh native/cuda/sync_batch_norm.cu native/cuda/mgt_cuda/adamw.cuh native/cuda/adamw.cu native/tools/mgt_bn_step_benchmark.cu native/tests/cuda/test_mlp_batch_norm_full_backward.cu native/tests/cuda/test_mlp_batch_norm_input_grad_2rank.cu native/CMakeLists.txt
  git commit -m "perf: add tiled input gradient to sync batchnorm step"
  ```

---

## Task 7: Reuse the Proven FP16 Linear Primitives in the BatchNorm Path

**Files:**

- Create: `native/cuda/mgt_cuda/linear_train_ops.cuh`
- Modify: `native/cuda/mlp_backward.cu`
- Modify: `native/cuda/mgt_cuda/mlp_batch_norm_forward.cuh`
- Modify: `native/cuda/mlp_batch_norm_forward.cu`
- Modify: `native/cuda/mgt_cuda/adamw.cuh`
- Modify: `native/tools/mgt_bn_step_benchmark.cu`
- Modify only if required: `native/cuda/adamw.cu`
- Create: `native/tests/cuda/test_cuda_linear_train_ops.cu`
- Create: `native/tests/cuda/test_mlp_batch_norm_mixed_precision.cu`
- Modify: `native/CMakeLists.txt`

- [ ] Expose the existing private operations through one prepared, per-rank plan instead of the global lazy cache:

  ```cpp
  enum class LinearPrecisionPolicy : std::uint32_t {
      kFp32Strict = 1,
      kFp16InputsFp32Accumulate = 2,
  };

  struct LinearTrainOpShape {
      std::uint32_t rows = 0;
      std::uint32_t input_features = 0;
      std::uint32_t output_features = 0;
  };

  struct LinearTrainOpsPlan;

  mgt::Status CreateLinearTrainOpsPlan(
      const LinearTrainOpShape* supported_shapes,
      std::uint32_t supported_shape_count,
      std::uint32_t device_id,
      cublasLtHandle_t blas_lt,
      std::uint64_t lt_workspace_bytes,
      LinearTrainOpsPlan** out);

  mgt::Status DestroyLinearTrainOpsPlan(LinearTrainOpsPlan* plan);

  mgt::Status LaunchFloatToHalf(
      const float* src,
      __half* dst,
      std::uint64_t count,
      cudaStream_t stream);

  mgt::Status LaunchLinearForwardHalfToFloat(
      const LinearTrainOpsPlan* plan,
      cudaStream_t stream,
      void* lt_workspace,
      std::uint64_t lt_workspace_bytes,
      const __half* input,
      const __half* weights,
      float* output,
      LinearTrainOpShape shape);

  mgt::Status LaunchLinearGradWeightsHalfToFloat(
      const LinearTrainOpsPlan* plan,
      cudaStream_t stream,
      void* lt_workspace,
      std::uint64_t lt_workspace_bytes,
      const __half* input,
      const __half* grad_output,
      float* grad_weights,
      LinearTrainOpShape shape);

  mgt::Status LaunchLinearGradInputHalfToFloat(
      const LinearTrainOpsPlan* plan,
      cudaStream_t stream,
      void* lt_workspace,
      std::uint64_t lt_workspace_bytes,
      const __half* grad_output,
      const __half* weights,
      float* grad_input,
      LinearTrainOpShape shape,
      float beta);
  ```

  `CreateLinearTrainOpsPlan` validates unique nonzero shapes and prebuilds forward, dW, and dX descriptors/algorithms for every supported shape plus dX beta values exactly `0.0f` and `1.0f`. It records the device, handle, and required workspace capacity. A launch with an unregistered shape, other beta, current CUDA device different from the plan device, null pointer, or misaligned/wrong-capacity workspace fails before enqueue. The cuBLASLt handle must outlive the plan; the owner must synchronize its compute stream, destroy the plan, and only then destroy the handle. Opaque handle lifetime is a caller invariant and is not probed by Launch. Launch performs no allocation, heuristic lookup, event creation, autotune, or synchronization.

  Register full/final row shapes before warmup for hidden `2560->224`, residual `224->224`, and output `224->1`, plus fixture shapes in tests. Implement the wrappers in `mlp_backward.cu` by factoring the current cuBLASLt code; both trainers use this plan and the old global lazy cache is not reachable from the production BatchNorm step.
- [ ] Add unit tests against FP32 GEMM for forward, dW, dX, beta-zero, beta-one residual accumulation, logical tails, and insufficient Lt workspace.

- [ ] Define the capacity-based mixed workspace before wiring any kernel:

  ```cpp
  struct MlpBatchNormMixedWorkspacePlan {
      std::uint32_t capacity_rows = 0;
      std::uint64_t saved_activations_half_offset_bytes = 0;
      std::uint64_t saved_activations_half_count = 0;
      std::uint64_t grad_half_offset_bytes = 0;
      std::uint64_t grad_half_count = 0;
      std::uint64_t input_grad_offset_bytes = 0;
      std::uint64_t input_grad_bytes = 0;
      std::uint64_t total_bytes = 0;
  };

  mgt::Status BuildMlpBatchNormMixedWorkspacePlan(
      const CudaMlpShape& shape,
      std::uint32_t capacity_rows,
      const InputEmbeddingGradConfig& input_grad_config,
      MlpBatchNormMixedWorkspacePlan* out);
  ```

  Reject null output, zero capacity, arithmetic overflow, unknown policy, or any offset that is not 256-byte aligned. All offsets and slice sizes use `capacity_rows`; every kernel separately receives `active_rows` and must reject `active_rows > capacity_rows`.

- [ ] Use this exact non-overlapping layout:

  ```text
  saved_activations_half_count =
      capacity_rows *
      (physical_hd1 + (2 * residual_blocks + 1) * physical_hd2)

  saved activation order:
      input BN/ReLU output                         [capacity_rows, physical_hd1]
      hidden BN/ReLU output                        [capacity_rows, physical_hd2]
      for block 0..15:
          block.fc1 BN/ReLU output                 [capacity_rows, physical_hd2]
          block residual BN/add/ReLU output        [capacity_rows, physical_hd2]

  grad_half_count =
      capacity_rows * max(physical_hd1, physical_hd2, output_dim)

  input_grad_bytes = QueryInputEmbeddingGradWorkspaceBytes(
      shape, capacity_rows,
      config with dz_f16_source = kCallerProvided)
  ```

  Place each slice at `align_up(previous_end, 256)` and set `total_bytes = align_up(last_end, 256)`. Saved activation slices live from their forward producer until their corresponding backward dW consumes them. The one grad-half slice is reused only in compute-stream order; it must not be overwritten until all cuBLASLt work that reads it has completed. The Task 5 one-hot scratch may reuse only the dedicated `input_grad` slice.

- [ ] Keep the following objects outside that byte workspace and owned by one per-rank runtime created before warmup: `cublasHandle_t`, `cublasLtHandle_t`, the 256-byte-aligned Lt workspace and its capacity, the prepared input-gradient plan, the prepared `LinearTrainOpsPlan`, and the FP16 linear-parameter mirror. No handle, descriptor, event, or allocation may be created lazily in the step.

- [ ] Make the FP16 mirror cover the entire physical contiguous linear parameter array described by `CudaMlpParameterLayout`, including bias slots. This is required because `LaunchAdamWKernelWithHalfMirror` updates a contiguous range. GEMMs read only matrix ranges; half bias slots are initialized and updated but never read. FP32 master biases remain authoritative for forward BatchNorm semantics.

- [ ] Initialize the complete half mirror from the FP32 master exactly once after fresh parameter initialization and exactly once after checkpoint restore, before the first mixed forward. Thereafter update it in the same Adam launch as the FP32 master. Test that a resume cannot execute a mixed forward with an uninitialized or stale mirror.

- [ ] Define input-embedding forward precision explicitly. Strict mode reads the FP32 master table and FP32 bias. Mixed mode reads half-rounded table entries from the full half mirror, converts/accumulates them in FP32, adds the FP32 master bias, and emits FP32 input-BN values plus the requested saved half activation. Half bias slots in the mirror are never read. Add an exhaustive small gather fixture and production-shape random fixture for both modes.

- [ ] In the first mixed implementation, keep every BatchNorm input/output, saved `xhat`, affine/running state, gradient, FP32 master parameter, and Adam moment in FP32. Use `CUBLAS_COMPUTE_32F` and FP32 outputs for forward, dW, and dX. Continue ordinary FP32 Adam for BN affine parameters.

- [ ] Make `kFp32Strict` the API and runtime default until authoritative 2x T4 acceptance. Keep TF32 as a separately named A100 diagnostic policy; Tesla T4 has no TF32 mode.

- [ ] Build a nonzero full-network fixture that checks:

  - output and loss;
  - every linear gradient tensor;
  - all 34 `dgamma` and `dbeta`;
  - every running mean/variance update;
  - every Adam-updated parameter and moment;
  - padded lanes;
  - 1-step and 100-step finite trajectories.

- [ ] Record per tensor class:

  ```text
  max absolute error
  relative L2 error
  cosine similarity
  norm ratio
  ```

- [ ] Make mixed pass/fail automatic. Against the strict fixture require:

  ```text
  output and loss:              max_abs <= 0.01 and relative_L2 <= 0.02
  every linear/affine gradient: max_abs <= 0.02 and relative_L2 <= 0.08
  every BN running tensor:      max_abs <= 0.005 and relative_L2 <= 0.01
  one-step master/m/v tensors:  max_abs <= 0.005 and relative_L2 <= 0.02
  padding and health flags:     exact
  100-step final loss:          relative difference <= 0.10
  100-step parameter cosine:    >= 0.995
  100-step parameter norm ratio in [0.98, 1.02]
  all inspected values finite
  ```

  Reject mixed if any tensor class fails; do not invoke puzzle search in Task 7 because real trainer/export integration begins in Task 15.

- [ ] Run:

  ```powershell
  docker exec mgt-gpu-queue python3 scripts/gpu_queue_submit.py --label task7-linear-gates --wait -- bash -lc "cmake --build native/build --config Release --parallel && ctest --test-dir native/build -C Release -R 'cuda_linear_train_ops|mlp_batch_norm_full_backward|mlp_batch_norm_mixed_precision|cuda_mlp_backward_mixed_precision_error|cuda_train_step_smoke' --output-on-failure --no-tests=error"
  git diff --check
  ```

- [ ] Run only one diagnostic smoke for strict FP32 and mixed FP16 at full and final shapes, with `--allow-smoke 1`; record conversion time separately. Defer the acceptance-sized A100 measurement and every authoritative 2x T4 decision to Tasks 16 and 17.

- [ ] Commit the required payload and include optional `adamw.cu` only when it actually changed:

  ```powershell
  $task7Files = @('native/cuda/mgt_cuda/linear_train_ops.cuh','native/cuda/mlp_backward.cu','native/cuda/mgt_cuda/mlp_batch_norm_forward.cuh','native/cuda/mlp_batch_norm_forward.cu','native/cuda/mgt_cuda/adamw.cuh','native/tools/mgt_bn_step_benchmark.cu','native/tests/cuda/test_cuda_linear_train_ops.cu','native/tests/cuda/test_mlp_batch_norm_mixed_precision.cu','native/CMakeLists.txt')
  $optionalAdamw = @(Get-NativeChecked 'inspect optional adamw implementation' { git diff --name-only -- native/cuda/adamw.cu })
  if ($optionalAdamw.Count -gt 1 -or ($optionalAdamw.Count -eq 1 -and $optionalAdamw[0] -ne 'native/cuda/adamw.cu')) { throw 'unexpected optional Task-7 path state' }
  if ($optionalAdamw.Count -eq 1) { $task7Files += 'native/cuda/adamw.cu' }
  Invoke-ExactCommit 'perf: add mixed precision linear ops to sync batchnorm' $task7Files
  ```

---

## Task 8: Fuse BatchNorm Forward Epilogues Without Changing Reductions

**Files:**

- Modify: `native/cuda/sync_batch_norm.cuh`
- Modify: `native/cuda/sync_batch_norm.cu`
- Modify: `native/cuda/mlp_batch_norm_forward.cu`
- Create: `native/tests/cuda/test_sync_batch_norm_fused_strided.cu`
- Create: `native/tests/cuda/test_sync_batch_norm_fused_2rank.cu`
- Modify: `native/CMakeLists.txt`

- [ ] Preserve the old generic APIs and add a typed fused-forward argument:

  ```cpp
  struct StridedSyncBatchNormFusedForwardArgs {
      const float* x = nullptr;
      const float* bias = nullptr;
      const float* residual = nullptr;
      std::uint32_t local_rows = 0;
      std::uint32_t global_rows = 0;
      std::uint32_t logical_cols = 0;
      std::uint32_t row_stride = 0;
      const float* gamma = nullptr;
      const float* beta = nullptr;
      float* running_mean = nullptr;
      float* running_var = nullptr;
      float momentum = 0.1f;
      float epsilon = 1.0e-5f;
      float* activated = nullptr;
      __half* optional_activated_half = nullptr;
      float* mean = nullptr;
      float* inv_std = nullptr;
      float* normalized = nullptr;
      const std::uint32_t* device_local_health_flag = nullptr;
      std::uint32_t* device_global_health_flag = nullptr;
      bool reduce_health = false;
  };

  struct StridedSyncBatchNormLegacyWorkspace {
      float* data = nullptr;
      std::uint64_t floats = 0;
  };

  mgt::Status LaunchStridedSyncBatchNormReluForward(
      const StridedSyncBatchNormFusedForwardArgs& args,
      StridedSyncBatchNormLegacyWorkspace workspace,
      NcclRankContext* bn_context,
      cudaStream_t stream);

  mgt::Status LaunchStridedSyncBatchNormBiasReluForward(
      const StridedSyncBatchNormFusedForwardArgs& args,
      StridedSyncBatchNormLegacyWorkspace workspace,
      NcclRankContext* bn_context,
      cudaStream_t stream);

  mgt::Status LaunchStridedSyncBatchNormBiasResidualReluForward(
      const StridedSyncBatchNormFusedForwardArgs& args,
      StridedSyncBatchNormLegacyWorkspace workspace,
      NcclRankContext* bn_context,
      cudaStream_t stream);
  ```

  The no-bias wrapper requires `bias == nullptr`; the bias wrapper requires a non-null bias; the residual wrapper requires non-null bias and residual. Legacy workspace requires `2*logical_cols + (reduce_health ? 1 : 0)` FP32 values and 16-byte alignment; one float less returns `kCapacityExceeded`. Only the input BN site sets `reduce_health=true`, requires both health pointers, and all other sites require `reduce_health=false`, a null local pointer, and the already-global non-null flag.

- [ ] Enforce one alias table before any fused launch: `activated` may equal `x`; `normalized` must be distinct from `x`, `activated`, residual, running state, mean/inv-std, and workspace because backward saves it; mean, inv-std, running mean, and running variance are pairwise distinct; bias/residual may not overlap a writable output; workspace may overlap nothing else. A non-null `optional_activated_half` means a dense `[local_rows,row_stride]` half write, must not overlap any FP32 pointer, and must receive exact zero in padded lanes. On a nonzero global health flag, leave running state unchanged and zero activated/optional-half/normalized logical and padded lanes.

- [ ] First fuse only `BN affine + ReLU` and `BN affine + residual + ReLU` into the normalization pass. Leave the separate linear bias kernel in place for this substep.

- [ ] Prove legacy-versus-fused equality for output, saved `xhat`, running mean, and running variance before removing an activation launch.

- [ ] Then fold bias into both local moments and normalization:

  ```text
  x_for_bn = linear_output + FP32 bias
  activated = ReLU(gamma * ((x_for_bn - mean) * inv_std) + beta)
  residual activated = ReLU(residual + BN_affine(x_for_bn))
  ```

  Bias must affect running mean. Do not delete it as an algebraically cancelling term.

- [ ] Optionally write the already-proved FP16 activation mirror from the fused epilogue when the mixed policy requests it. Keep FP32 saved activation for the strict path.

- [ ] Explicitly zero padded output columns in the same epilogue.

- [ ] Test rows `513`, logical/physical widths `2556/2560` and `218/224`, positive/negative gamma, zero gamma, padding sentinels, bias, residual, and non-divisible rows.

- [ ] In the two-rank test, use different rank distributions and compare with one global CPU oracle. Reuse the CTest skip-code-77 convention locally, and make `MGT_REQUIRE_TWO_GPUS=1` turn missing devices into failure in acceptance jobs.

- [ ] Do not repeat the rejected owner-column `fc2 dZ+bias` fusion. Any future backward fused write must keep row/tile-coalesced output.

- [ ] Run:

  ```powershell
  docker exec mgt-gpu-queue python3 scripts/gpu_queue_submit.py --label task8-fused-forward-gates --wait -- bash -lc "cmake --build native/build --config Release --parallel && ctest --test-dir native/build -C Release -R 'sync_batch_norm_fused_strided|sync_batch_norm_fused_2rank|sync_batch_norm_strided|sync_batch_norm_2rank|mlp_batch_norm_forward|mlp_batch_norm_full_backward|mlp_batch_norm_mixed_precision' --output-on-failure --no-tests=error"
  git diff --check
  ```

- [ ] Run separate diagnostic smokes after activation fusion and bias fusion with `--allow-smoke 1`. Keep the two rows separate and revert a substep that is correct but clearly slower. Do not call either row authoritative; Tasks 16 and 17 own acceptance-sized hardware evidence.

- [ ] Commit:

  ```powershell
  git add native/cuda/sync_batch_norm.cuh native/cuda/sync_batch_norm.cu native/cuda/mlp_batch_norm_forward.cu native/tests/cuda/test_sync_batch_norm_fused_strided.cu native/tests/cuda/test_sync_batch_norm_fused_2rank.cu native/CMakeLists.txt
  git commit -m "perf: fuse sync batchnorm forward epilogues"
  ```

---

## Task 9: Add a Capacity-Aware BatchNorm Workspace and Coalesced Forward Moments

**Files:**

- Modify: `native/include/mgt/batch_norm_training.hpp`
- Modify: `native/src/batch_norm_training.cpp`
- Modify: `native/cuda/sync_batch_norm.cuh`
- Modify: `native/cuda/sync_batch_norm.cu`
- Modify: `native/cuda/mlp_batch_norm_forward.cu`
- Create: `native/tests/cuda/test_sync_batch_norm_production_geometry.cu`
- Modify: `native/tests/cuda/test_sync_batch_norm_fused_strided.cu`
- Modify: `native/tests/cuda/test_sync_batch_norm_fused_2rank.cu`
- Modify: `native/CMakeLists.txt`

- [ ] Preserve every legacy generic BN API and its old workspace contract. Add the production query and workspace view without silently reinterpreting old buffers:

  ```cpp
  enum class BatchNormKernelPolicy : std::uint32_t {
      kLegacy = 1,
      kFusedLegacyReduction = 2,
      kFusedCoalescedV2 = 3,
  };

  struct StridedSyncBatchNormWorkspaceV2 {
      float* data = nullptr;
      std::uint64_t floats = 0;
      std::uint32_t capacity_rows = 0;
  };

  mgt::Status QueryStridedSyncBatchNormReductionWorkspaceFloatsV2(
      std::uint32_t capacity_rows,
      std::uint32_t logical_cols,
      std::uint64_t* required_floats);

  mgt::Status LaunchStridedSyncBatchNormReluForwardV2(
      const StridedSyncBatchNormFusedForwardArgs& args,
      StridedSyncBatchNormWorkspaceV2 workspace,
      NcclRankContext* bn_context,
      cudaStream_t stream);

  mgt::Status LaunchStridedSyncBatchNormBiasReluForwardV2(
      const StridedSyncBatchNormFusedForwardArgs& args,
      StridedSyncBatchNormWorkspaceV2 workspace,
      NcclRankContext* bn_context,
      cudaStream_t stream);

  mgt::Status LaunchStridedSyncBatchNormBiasResidualReluForwardV2(
      const StridedSyncBatchNormFusedForwardArgs& args,
      StridedSyncBatchNormWorkspaceV2 workspace,
      NcclRankContext* bn_context,
      cudaStream_t stream);
  ```

- [ ] Validate before launch: the complete Task 8 alias table, non-null outputs, `1 <= local_rows <= capacity_rows`, `global_rows >= local_rows`, positive logical columns, `row_stride >= logical_cols`, checked 64-bit arithmetic, 256-byte workspace alignment, and exact bias/residual/health mode. Workspace may overlap no input, output, saved tensor, running state, or affine vector. Return `kCapacityExceeded` when `workspace.floats` is smaller than the query and `kInvalidConfig` for invalid dimensions/overlap.

- [ ] Redesign `BatchNormTrainingPlan` around capacity, not active rows. Rename the builder's last dimension argument to `capacity_rows`; add fields `capacity_rows`, `legacy_reduction_count`, `v2_reduction_count`, and retain `reduction_offset/reduction_count`. Compute `normalized_count = capacity_rows * storage_feature_count`; align `reduction_offset` up to 64 FP32 values (256 bytes); set `legacy_reduction_count = 2*max(logical_hd1,logical_hd2)+1`, `v2_reduction_count = 1+2*max(logical_hd1,logical_hd2)*ceil(capacity_rows/256)`, `reduction_count=max(legacy_reduction_count,v2_reduction_count)`, and `workspace_floats=reduction_offset+reduction_count`. Every site normalized offset uses capacity rows. The same one scratch slice is reused sequentially by all 34 sites.

- [ ] Use `row_chunk = 256` and this exact layout:

  ```text
  chunk_count_capacity = ceil(capacity_rows / 256)
  required_floats      = 1 + 2 * logical_cols * chunk_count_capacity
  chunk_base(0)        = 0
  chunk_base(chunk>0)  = 1 + chunk * 2 * logical_cols
  partial(chunk, stat, feature) =
      workspace[chunk_base(chunk) + stat * logical_cols + feature]
  health_word          = workspace[2 * logical_cols]
  stat 0 = sum
  stat 1 = sum of squares
  ```

  Launch partials only for `chunk < ceil(local_rows/256)`. A separate finalize kernel reads all active chunks in increasing chunk order, then overwrites chunk 0 with the final local sums: `workspace[feature]` and `workspace[logical_cols + feature]`. For the input site, finalize also writes the local health word and NCCL SUMs the first `2*logical_cols+1` floats; every other forward site SUMs only the first `2*logical_cols`. The normalization epilogue reads globally reduced chunk-0 values and the input-site global health result. No atomic accumulation is allowed.

- [ ] Assert the maximum A100 input-site scratch:

  ```text
  1 + 2 * 2556 * ceil(12500 / 256) = 250489 floats = 1001956 bytes
  ```

  A plan created with capacity 12,500 must run the final local-row counts 12,498 and 12,497 without rebuilding or changing offsets.

- [ ] Keep FP32 `sum/sumsq` semantics in this performance task. Test large mean/small variance, zero variance, unequal rank distributions, negative gamma, non-divisible rows, and padding. If legacy and PyTorch already disagree outside Task 3 tolerances, stop and make a separate numerical-correctness commit; do not hide it by weakening this task's gate.

- [ ] Test rows `1,2,255,256,257,513`, both `2556/2560` and `218/224`, exact workspace, one float too small, misalignment, every forbidden alias, `global_rows<local_rows`, `active_rows < capacity_rows`, input health set on only one rank, and two ranks with unequal local rows. Require legacy-versus-V2 tolerance parity for output, mean, inverse standard deviation, normalized values, running mean, and running variance.

- [ ] Run:

  ```powershell
  docker exec mgt-gpu-queue python3 scripts/gpu_queue_submit.py --label task9-v2-forward-gates --wait -- bash -lc "cmake --build native/build --config Release --parallel && ctest --test-dir native/build -C Release -R 'sync_batch_norm_fused|sync_batch_norm_production_geometry|mlp_batch_norm_forward' --output-on-failure --no-tests=error"
  git diff --check
  ```

- [ ] Run one `--allow-smoke 1` legacy/V2 diagnostic row. Keep V2 disabled if it is slower; authoritative measurement belongs to Tasks 16 and 17.

- [ ] Commit:

  ```powershell
  git add native/include/mgt/batch_norm_training.hpp native/src/batch_norm_training.cpp native/cuda/sync_batch_norm.cuh native/cuda/sync_batch_norm.cu native/cuda/mlp_batch_norm_forward.cu native/tests/cuda/test_sync_batch_norm_production_geometry.cu native/tests/cuda/test_sync_batch_norm_fused_strided.cu native/tests/cuda/test_sync_batch_norm_fused_2rank.cu native/CMakeLists.txt
  git commit -m "perf: coalesce sync batchnorm forward moments"
  ```

---

## Task 10: Coalesce Generic BatchNorm Backward Statistics

**Files:**

- Modify: `native/cuda/sync_batch_norm.cuh`
- Modify: `native/cuda/sync_batch_norm.cu`
- Modify: `native/cuda/mlp_batch_norm_forward.cu`
- Modify: `native/tests/cuda/test_sync_batch_norm_strided.cu`
- Modify: `native/tests/cuda/test_sync_batch_norm_2rank.cu`
- Modify: `native/tests/cuda/test_sync_batch_norm_production_geometry.cu`
- Modify: `native/tests/cuda/test_mlp_batch_norm_pytorch_fixture.cu`
- Create: `native/tests/cuda/test_mlp_batch_norm_full_step_2rank.cu`
- Modify: `native/CMakeLists.txt`

- [ ] Keep the legacy backward wrapper unchanged and add an explicit V2 API:

  ```cpp
  struct StridedSyncBatchNormBackwardV2Args {
      const float* grad_output = nullptr;
      const float* normalized = nullptr;
      const float* gamma = nullptr;
      const float* inv_std = nullptr;
      const std::uint32_t* device_global_health_flag = nullptr;
      std::uint32_t local_rows = 0;
      std::uint32_t global_rows = 0;
      std::uint32_t logical_cols = 0;
      std::uint32_t row_stride = 0;
      float* grad_input = nullptr;
      float* grad_gamma = nullptr;
      float* grad_beta = nullptr;
  };

  mgt::Status LaunchStridedSyncBatchNormBackwardV2(
      const StridedSyncBatchNormBackwardV2Args& args,
      StridedSyncBatchNormWorkspaceV2 workspace,
      NcclRankContext* bn_context,
      cudaStream_t stream);
  ```

- [ ] Reuse Task 9's exact `[chunk][2][feature]` workspace and capacity query. The two statistics are local `sum(dy)` and `sum(dy * xhat)`. Finalize active chunks in deterministic order, overwrite chunk 0, allreduce exactly `2 * logical_cols` FP32 values, then compute dX with the true `global_rows` denominator. `grad_gamma` and `grad_beta` remain explicit outputs in this generic task.

- [ ] Apply one shared alias/health table to this API and Task 11: `grad_input` may equal `grad_output` only because the partial-statistics kernel completes first; `grad_gamma` and `grad_beta` are distinct, non-null, logical-width arrays; neither may overlap each other, workspace, saved normalized/inv-std/gamma, or row tensors; workspace may overlap nothing; `normalized`, gamma, and inv-std remain read-only until dX finishes. On nonzero global health, write zero dX/dgamma/dbeta and preserve the ordinary collective call. Zero padded dX columns in every case.

- [ ] Compare legacy, V2, and the checked-in PyTorch full-step fixture for dX, dgamma, dbeta, every linear gradient downstream, and one Adam update. Use tolerance-based parity because the reduction order changes; exact hashes are required only for repeat runs of the same policy.

- [ ] Add `test_mlp_batch_norm_full_step_2rank.cu` as a rank-worker target launched by `scripts/run_cuda_2rank_test.py`. Use all 16 blocks, unequal rank-local rows, different rank distributions, and one concatenated PyTorch/CPU oracle. Require strict FP32 dX/dgamma/dbeta and linear-gradient `atol<=2e-4, rtol<=2e-4`, one-step Adam `max_abs<=2e-6`, exact padding, and exact collective participation. Configure CTest skip code 77 locally and mandatory two-GPU failure under `MGT_REQUIRE_TWO_GPUS=1`.

- [ ] Test rows `1,2,255,256,257,513`, unequal two-rank local rows, `local_rows < capacity_rows`, logical/physical widths, negative and zero gamma, large mean/small variance, zero variance, in-place dX, exact workspace, one float too small, and padding zero.

- [ ] Run:

  ```powershell
  docker exec mgt-gpu-queue python3 scripts/gpu_queue_submit.py --label task10-v2-backward-gates --wait -- bash -lc "cmake --build native/build --config Release --parallel && ctest --test-dir native/build -C Release -R 'sync_batch_norm_strided|sync_batch_norm_2rank|sync_batch_norm_production_geometry|mlp_batch_norm_pytorch_fixture|mlp_batch_norm_full_backward|mlp_batch_norm_full_step_2rank' --output-on-failure --no-tests=error"
  git diff --check
  ```

- [ ] Run one uninstrumented diagnostic A/B row with `--allow-smoke 1`; do not alter the default on this evidence alone.

- [ ] Commit:

  ```powershell
  git add native/cuda/sync_batch_norm.cuh native/cuda/sync_batch_norm.cu native/cuda/mlp_batch_norm_forward.cu native/tests/cuda/test_sync_batch_norm_strided.cu native/tests/cuda/test_sync_batch_norm_2rank.cu native/tests/cuda/test_sync_batch_norm_production_geometry.cu native/tests/cuda/test_mlp_batch_norm_pytorch_fixture.cu native/tests/cuda/test_mlp_batch_norm_full_step_2rank.cu native/CMakeLists.txt
  git commit -m "perf: coalesce sync batchnorm backward statistics"
  ```

---

## Task 11: Fuse the ReLU/Residual Backward Gate and Remove CopyGrads

**Files:**

- Modify: `native/cuda/sync_batch_norm.cuh`
- Modify: `native/cuda/sync_batch_norm.cu`
- Modify: `native/cuda/mlp_batch_norm_forward.cu`
- Modify: `native/tests/cuda/test_sync_batch_norm_fused_strided.cu`
- Modify: `native/tests/cuda/test_sync_batch_norm_fused_2rank.cu`
- Modify: `native/tests/cuda/test_sync_batch_norm_production_geometry.cu`
- Modify: `native/tests/cuda/test_mlp_batch_norm_full_backward.cu`
- Modify: `native/tests/cuda/test_mlp_batch_norm_full_step_2rank.cu`
- Modify: `native/CMakeLists.txt`

- [ ] Add the exact specialized API:

  ```cpp
  struct StridedSyncBatchNormFusedBackwardArgs {
      const float* activated = nullptr;
      const float* grad_output = nullptr;
      const float* gamma = nullptr;
      const float* inv_std = nullptr;
      const float* normalized = nullptr;
      const std::uint32_t* device_global_health_flag = nullptr;
      std::uint32_t local_rows = 0;
      std::uint32_t global_rows = 0;
      std::uint32_t logical_cols = 0;
      std::uint32_t row_stride = 0;
      float* grad_input = nullptr;
      float* grad_residual = nullptr;
      float* grad_gamma = nullptr;
      float* grad_beta = nullptr;
  };

  mgt::Status LaunchStridedSyncBatchNormReluBackwardV2(
      const StridedSyncBatchNormFusedBackwardArgs& args,
      StridedSyncBatchNormWorkspaceV2 workspace,
      NcclRankContext* bn_context,
      cudaStream_t stream);

  mgt::Status LaunchStridedSyncBatchNormResidualReluBackwardV2(
      const StridedSyncBatchNormFusedBackwardArgs& args,
      StridedSyncBatchNormWorkspaceV2 workspace,
      NcclRankContext* bn_context,
      cudaStream_t stream);
  ```

- [ ] In the partial-statistics pass, compute `dy = activated > 0 ? grad_output : 0`, accumulate `sum(dy)` and `sum(dy*xhat)`, and for the residual variant write exactly that gated `dy` to `grad_residual`. Never reconstruct xhat from activated output. Require `grad_residual != nullptr` only for the residual wrapper and reject `grad_residual == grad_input`.

- [ ] Inherit and enforce the complete Task 10 alias/health table. Permit `grad_input == grad_output` under that ordering rule. After NCCL, let the dX kernel read the first `2*logical_cols` workspace values directly and have one feature lane write `grad_beta[feature]` and `grad_gamma[feature]`; remove the standalone `CopyGrads` launch only after these outputs pass the oracle.

- [ ] Extend both the single-rank full-backward test and `test_mlp_batch_norm_full_step_2rank.cu`. Cover all 16 blocks, distinct residual fanout, negative/zero gamma, ReLU exactly at zero, in-place dX, unequal ranks in the two-process target, both production strides, exact/undersized workspace, invalid health on only rank 1, padding, all affine/linear gradients, running state, parameters, and moments.

- [ ] Run:

  ```powershell
  docker exec mgt-gpu-queue python3 scripts/gpu_queue_submit.py --label task11-fused-backward-gates --wait -- bash -lc "cmake --build native/build --config Release --parallel && ctest --test-dir native/build -C Release -R 'sync_batch_norm_fused|sync_batch_norm_production_geometry|mlp_batch_norm_full_backward|mlp_batch_norm_full_step_2rank|mlp_batch_norm_pytorch_fixture' --output-on-failure --no-tests=error"
  git diff --check
  ```

- [ ] Measure the fused gate and CopyGrads removal as two separate `--allow-smoke 1` rows. Revert only the slower substep, not the verified workspace/backward changes.

- [ ] Commit:

  ```powershell
  git add native/cuda/sync_batch_norm.cuh native/cuda/sync_batch_norm.cu native/cuda/mlp_batch_norm_forward.cu native/tests/cuda/test_sync_batch_norm_fused_strided.cu native/tests/cuda/test_sync_batch_norm_fused_2rank.cu native/tests/cuda/test_sync_batch_norm_production_geometry.cu native/tests/cuda/test_mlp_batch_norm_full_backward.cu native/tests/cuda/test_mlp_batch_norm_full_step_2rank.cu native/CMakeLists.txt
  git commit -m "perf: fuse sync batchnorm backward gate"
  ```

---

## Task 12: Consolidate Linear-Gradient Collectives Without Overlap

**Files:**

- Create: `native/include/mgt/bn_communication_plan.hpp`
- Create: `native/src/bn_communication_plan.cpp`
- Create: `native/tests/test_bn_communication_plan.cpp`
- Create: `native/tests/cuda/test_mlp_batch_norm_communication_2rank.cu`
- Modify: `native/cuda/mgt_cuda/mlp_batch_norm_forward.cuh`
- Modify: `native/cuda/mlp_batch_norm_forward.cu`
- Modify: `native/tools/mgt_bn_step_benchmark.cu`
- Modify: `native/CMakeLists.txt`

- [ ] Define only the reference and first optimized policy in this task:

  ```cpp
  enum class WeightGradientReductionPolicy : std::uint32_t {
      kSectionSumReference = 1,
      kSingleSumAfterBackward = 2,
  };

  struct GradientRange {
      std::uint64_t offset = 0;
      std::uint64_t count = 0;
  };

  struct BatchNormCommunicationPlan {
      WeightGradientReductionPolicy policy =
          WeightGradientReductionPolicy::kSectionSumReference;
      std::array<GradientRange, 4> ranges{};
      std::uint32_t range_count = 0;
      std::uint64_t total_linear_floats = 0;
  };

  mgt::Status BuildBatchNormCommunicationPlan(
      const CudaMlpShape& shape,
      WeightGradientReductionPolicy policy,
      BatchNormCommunicationPlan* out);
  ```

- [ ] Build all ranges from `BuildCudaMlpParameterLayout(shape)`. The reference sequence is output `[output_weight,total)`, residual `[residual_base,output_weight)`, hidden `[hidden_weight,residual_base)`, input `[input_weight,hidden_weight)`. The single-sum sequence is `[input_weight,total)`. Keep ranges in enqueue order. For the reference plan validate reverse adjacency (`ranges[i].offset + ranges[i].count == ranges[i-1].offset`), last offset zero, first end equal total, nonzero counts, and a sorted-copy coverage check with no gap/overlap; for single SUM validate exactly `[0,total)`. Assert P888 total `15,460,289`.

- [ ] Add internal local-only output/residual/hidden/input backward variants. Keep existing public stage wrappers for legacy unit tests, but make the production dispatcher use only local variants followed by exactly one plan-driven reduction schedule.

- [ ] Define the dispatcher/runtime interface now so every later policy is wired through one path:

  ```cpp
  struct MlpBatchNormCollectiveCounters {
      std::uint32_t bn_forward = 0;
      std::uint32_t loss = 0;
      std::uint32_t bn_backward = 0;
      std::uint32_t linear_weight = 0;
      std::uint64_t linear_weight_bytes = 0;
  };

  struct MlpBatchNormPreparedRuntime {
      std::uint32_t device_id = 0;
      std::uint32_t capacity_rows = 0;
      NcclRankContext* bn_context = nullptr;
      cublasHandle_t blas = nullptr;
      cublasLtHandle_t blas_lt = nullptr;
      LinearTrainOpsPlan* linear_plan = nullptr;
      InputEmbeddingGradPlan* input_grad_plan = nullptr;
      void* lt_workspace = nullptr;
      std::uint64_t lt_workspace_bytes = 0;
      cudaStream_t compute_stream = nullptr;
      MlpBatchNormCollectiveCounters* counters = nullptr;
  };

  mgt::Status LaunchMlpBatchNormTrainStepV2(
      const CudaMlpShape& physical_shape,
      std::uint32_t logical_hd1,
      std::uint32_t logical_hd2,
      const mgt::TrainStateStorage* states,
      const float* labels,
      std::uint32_t active_rows,
      std::uint32_t global_rows,
      const mgt::BatchNormTrainingPlan& batch_norm_plan,
      BatchNormKernelPolicy batch_norm_policy,
      LinearPrecisionPolicy linear_policy,
      const InputEmbeddingGradConfig& input_grad_config,
      const BatchNormCommunicationPlan& communication_plan,
      const AdamWKernelConfig& adam,
      MlpBatchNormStepBuffers buffers,
      MlpBatchNormPreparedRuntime* runtime);
  ```

  Validate every policy/capacity/pointer before the first enqueue. In Task 12 all BN, loss, and linear reductions use `runtime->bn_context` and `compute_stream`; there is no overlap. Reference enqueues output, residual, hidden, input weight ranges as each local stage becomes ready; single SUM enqueues `[0,total)` only after input backward. Reset counters before each step and assert exact counts after it. No linear gradient may be reduced twice.

- [ ] Keep all 34 forward BN and 34 backward BN collectives plus loss SUM on the existing BN communicator. Consolidation changes only the four linear-gradient SUMs:

  ```text
  reference: 68 BN + 4 linear + 1 loss = 73 collectives/step
  single:    68 BN + 1 linear + 1 loss = 70 collectives/step
  ```

- [ ] For global mean MSE, divide local derivatives by the true `global_rows`, SUM loss once, and do not divide linear or BN affine gradients after NCCL. Add a structural collective counter in tests and benchmark manifests.

- [ ] Add `test_mlp_batch_norm_communication_2rank.cu`, launched by the shared two-process launcher, and compare section versus single policy on unequal rows. Require reduced linear gradients `max_abs<=2e-5, relative_L2<=2e-5`, loss `max_abs<=2e-6`, one-step parameters/moments `max_abs<=2e-6`, a 100-step final-loss relative difference `<=1e-4`, exact counter values 73/70, and no double reduction. Cross-policy bitwise checksum is not required; same-policy repeated checksum must match.

- [ ] Run:

  ```powershell
  docker exec mgt-gpu-queue python3 scripts/gpu_queue_submit.py --label task12-communication-gates --wait -- bash -lc "cmake --build native/build --config Release --parallel && ctest --test-dir native/build -C Release -R 'bn_communication_plan|mlp_batch_norm_full_backward|mlp_batch_norm_input_grad_2rank|mlp_batch_norm_communication_2rank' --output-on-failure --no-tests=error"
  git diff --check
  ```

- [ ] Run acceptance-shaped uninstrumented reference/single diagnostic rows with `--allow-smoke 1`. Keep the reference default until Tasks 16 and 17 accept a production policy.

- [ ] Commit:

  ```powershell
  git add native/include/mgt/bn_communication_plan.hpp native/src/bn_communication_plan.cpp native/tests/test_bn_communication_plan.cpp native/tests/cuda/test_mlp_batch_norm_communication_2rank.cu native/cuda/mgt_cuda/mlp_batch_norm_forward.cuh native/cuda/mlp_batch_norm_forward.cu native/tools/mgt_bn_step_benchmark.cu native/CMakeLists.txt
  git commit -m "perf: consolidate sync batchnorm weight reductions"
  ```

---

## Task 13: Add Measured Two-Range Overlap on a Separate Communicator

**Files:**

- Modify: `native/include/mgt/bn_communication_plan.hpp`
- Modify: `native/src/bn_communication_plan.cpp`
- Modify: `native/tests/test_bn_communication_plan.cpp`
- Modify: `native/cuda/mgt_cuda/allreduce_nccl.cuh`
- Modify: `native/cuda/allreduce_nccl.cu`
- Modify: `native/cuda/mgt_cuda/mlp_batch_norm_forward.cuh`
- Modify: `native/cuda/mlp_batch_norm_forward.cu`
- Modify: `native/tools/mgt_bn_step_benchmark.cu`
- Create: `native/tests/cuda/test_mlp_batch_norm_overlap_2rank.cu`
- Modify: `native/CMakeLists.txt`

- [ ] Add only the implementation this task can always test:

  ```cpp
  enum class WeightGradientReductionPolicy : std::uint32_t {
      kSectionSumReference = 1,
      kSingleSumAfterBackward = 2,
      kTwoRangeOverlap = 3,
  };
  ```

  Unknown values, including the reserved future tile-bucket value, return `kInvalidConfig`. Do not expose an unimplemented enum to config/autotune.

- [ ] Extend `MlpBatchNormPreparedRuntime` with `weight_context`, `weight_comm_stream`, and precreated `tail_ready`, `prefix_ready`, `weight_done` events. Add the lifecycle API:

  ```cpp
  struct BatchNormDistributedRuntimeConfig {
      std::uint32_t device_id = 0;
      std::uint32_t rank = 0;
      std::uint32_t world = 0;
      std::filesystem::path bn_id_file;
      std::filesystem::path weight_id_file;
      bool create_weight_context = false;
      std::uint32_t rendezvous_timeout_seconds = 120;
  };

  mgt::Status CreateBatchNormDistributedResources(
      const BatchNormDistributedRuntimeConfig& config,
      MlpBatchNormPreparedRuntime* runtime);

  mgt::Status CheckBatchNormDistributedAsyncErrors(
      MlpBatchNormPreparedRuntime* runtime);

  mgt::Status AbortBatchNormDistributedResources(
      MlpBatchNormPreparedRuntime* runtime);

  mgt::Status DestroyBatchNormDistributedResources(
      MlpBatchNormPreparedRuntime* runtime);
  ```

- [ ] The launcher generates a cryptographic per-attempt nonce and places it in both distinct ID filenames, so neither path may exist before rank startup. Rank 0 publishes each NCCL unique ID through a temporary file plus atomic rename; other ranks wait with the configured timeout. Every rank creates `bn_context` first and, only when requested, `weight_context` second, then streams/events before warmup. On partial failure clean up in reverse; normal destruction is events, weight stream, weight context, compute stream, BN context. Create no resource in the step.

- [ ] Use these communicator schedules:

  ```text
  section reference:
      bn_context / compute_stream:
        34 forward BN, loss, 34 backward BN,
        output weight, residual weight, hidden weight, input weight
      weight_context: absent

  single after backward:
      bn_context / compute_stream:
        34 forward BN, loss, 34 backward BN, one full linear-weight SUM
      weight_context: absent

  two-range overlap:
      bn_context / compute_stream:
        34 forward BN, loss, 34 backward BN
      weight_context / weight_comm_stream:
        tail SUM, then prefix SUM
  ```

  Every rank follows the same schedule. Never issue layer BN and linear-weight collectives on different streams of one context.

- [ ] Build the two-range plan exactly from `CudaMlpParameterLayout`:

  ```text
  tail   [residual_base, total)         1,613,025 floats
  prefix [input_weight, residual_base) 13,847,264 floats
  ```

  Record `tail_ready` after residual backward has produced the whole tail; the weight stream waits and enqueues tail SUM. Record `prefix_ready` after input backward has produced the whole prefix; enqueue prefix SUM second. Record `weight_done` after prefix SUM; the compute stream waits before any Adam/half-mirror update. Counters require exactly two weight collectives and full no-gap/no-overlap coverage.

- [ ] `CheckBatchNormDistributedAsyncErrors` calls `ncclCommGetAsyncError` for every existing context only at delayed telemetry boundaries (every 100 steps), before checkpointing, and after stress. Any non-success calls `ncclCommAbort` on both contexts and fails the run. Worker launch/tests use a 120-second wall timeout; timeout kills both rank processes and preserves both logs instead of waiting forever.

- [ ] Add `test_mlp_batch_norm_overlap_2rank.cu`, launched by `run_cuda_2rank_test.py`. Test fresh rendezvous, missing/duplicate path rejection, partial-construction cleanup, delayed producer events, tail-then-prefix ordering, Adam wait, exact coverage/counters, unequal final rows, injected async failure where supported, and 1000 steps. Compare to single SUM with gradient `max_abs<=2e-5, relative_L2<=2e-5`, parameters/moments `max_abs<=2e-6`, final-loss relative difference `<=1e-4`, and exact same-policy repeat checksum.

- [ ] Run:

  ```powershell
  docker exec mgt-gpu-queue python3 scripts/gpu_queue_submit.py --label task13-overlap-gates --wait -- bash -lc "cmake --build native/build --config Release --parallel && ctest --test-dir native/build -C Release -R 'bn_communication_plan|mlp_batch_norm_communication_2rank|mlp_batch_norm_overlap_2rank' --output-on-failure --no-tests=error"
  git diff --check
  ```

- [ ] Keep two-range overlap disabled unless uninstrumented Tasks 16/17 evidence shows at least 3% median full-step improvement, no p95 regression, no final-batch regression beyond 2%, and the 1000-step stress passes. Tile-bucket overlap moves to Optional Experiments after the production gate.

- [ ] Commit:

  ```powershell
  git add native/include/mgt/bn_communication_plan.hpp native/src/bn_communication_plan.cpp native/tests/test_bn_communication_plan.cpp native/cuda/mgt_cuda/allreduce_nccl.cuh native/cuda/allreduce_nccl.cu native/cuda/mgt_cuda/mlp_batch_norm_forward.cuh native/cuda/mlp_batch_norm_forward.cu native/tools/mgt_bn_step_benchmark.cu native/tests/cuda/test_mlp_batch_norm_overlap_2rank.cu native/CMakeLists.txt
  git commit -m "perf: add two-range sync batchnorm reduction overlap"
  ```

---
## Task 14: Make the T4 Autotuner Produce an Executable, Fail-Closed Runtime Policy

**Files:**

- Modify: `scripts/autotune_2xt4.py`
- Modify: `scripts/tests/test_autotune_2xt4.py`
- Create: `scripts/runtime_source_manifest.py`
- Create: `scripts/tests/test_runtime_source_manifest.py`
- Create: `scripts/p888_runtime_source_allowlist.txt`
- Create: `scripts/poll_p888_gpu_memory.py`
- Create: `scripts/tests/test_poll_p888_gpu_memory.py`
- Modify: `scripts/summarize_bn_benchmark.py`
- Modify: `scripts/tests/test_summarize_bn_benchmark.py`
- Modify: `kaggle/kernel/run_autotune_2xt4.sh`
- Modify: `kaggle/kernel/run_sweep_2xt4.sh`
- Create: `kaggle/kernel/run_p888_bn_benchmark_2xt4.sh`
- Modify: `kaggle/kernel/run_ranks_2xt4.sh`
- Create: `native/include/mgt/sha256.hpp`
- Create: `native/src/sha256.cpp`
- Create: `native/tests/test_sha256.cpp`
- Create: `native/include/mgt/p888_runtime_policy.hpp`
- Create: `native/src/p888_runtime_policy.cpp`
- Create: `native/tests/test_p888_runtime_policy.cpp`
- Create: `native/tests/fixtures/p888_runtime_policy_v1.json`
- Create: `native/cuda/mgt_cuda/p888_runtime_policy.cuh`
- Create: `native/cuda/p888_runtime_policy.cu`
- Create: `native/tests/cuda/test_p888_runtime_policy_mapping.cu`
- Create: `native/include/mgt/p888_benchmark_snapshot.hpp`
- Create: `native/src/p888_benchmark_snapshot.cpp`
- Create: `native/tests/test_p888_benchmark_snapshot.cpp`
- Modify: `native/include/mgt/config.hpp`
- Modify: `native/src/train_plan.cpp`
- Modify: `native/tests/test_train_plan.cpp`
- Modify: `native/tools/mgt_bn_step_benchmark.cu`
- Modify: `native/tools/mgt_native_train_smoke.cu`
- Modify: `native/CMakeLists.txt`
- Create: `test_results/p888_bn_autotune_schema.md`

- [ ] Begin with failing SHA256 and cross-language policy tests. Keep the API valid in host C++20 and CUDA C++17 translation units; do not expose `std::span`:

  ```cpp
  using Sha256Digest = std::array<std::uint8_t, 32>;

  Sha256Digest Sha256(const void* data, std::size_t size);
  std::array<char, 65> Sha256LowerHex(const Sha256Digest& digest);
  Status ParseSha256LowerHex(std::string_view text, Sha256Digest* digest);
  ```

  Test NIST empty-string, `abc`, and multi-block vectors, exact lowercase parsing, one uppercase rejection, wrong lengths, and the two 64-byte policy vectors below. `Sha256LowerHex` includes a final NUL; serialization uses only the 64 characters.

- [ ] Keep serialized policy IDs portable in `mgt_core`; never include CUDA, NCCL, or cuBLAS headers from `p888_runtime_policy.hpp`:

  ```cpp
  enum class P888BatchNormPolicyId : std::uint32_t {
      kLegacy = 1,
      kFusedLegacyReduction = 2,
      kFusedCoalescedV2 = 3,
  };
  enum class P888LinearPrecisionId : std::uint32_t {
      kFp32Strict = 1,
      kFp16InputsFp32Accumulate = 2,
  };
  enum class P888MathMode : std::uint32_t {
      kFp32 = 1,
      kTf32Diagnostic = 2,
      kFp16Candidate = 3,
  };
  enum class P888InputGradientId : std::uint32_t {
      kOwnerWriteFp32Reference = 1,
      kPositionTiledGemmFp32 = 2,
      kPositionTiledGemmFp16 = 3,
  };
  enum class P888DzF16SourceId : std::uint32_t {
      kStageInternally = 1,
      kCallerProvided = 2,
  };
  enum class P888WeightReductionId : std::uint32_t {
      kSectionSumReference = 1,
      kSingleSumAfterBackward = 2,
      kTwoRangeOverlap = 3,
  };

  struct P888RuntimePolicy {
      std::uint32_t schema_version = 1;
      P888BatchNormPolicyId batch_norm = P888BatchNormPolicyId::kLegacy;
      P888LinearPrecisionId linear_precision =
          P888LinearPrecisionId::kFp32Strict;
      P888MathMode math_mode = P888MathMode::kFp32;
      P888InputGradientId input_gradient =
          P888InputGradientId::kOwnerWriteFp32Reference;
      std::uint32_t input_tile = 0;
      P888DzF16SourceId dz_f16_source =
          P888DzF16SourceId::kStageInternally;
      std::uint64_t lt_workspace_bytes = 0;
      P888WeightReductionId weight_reduction =
          P888WeightReductionId::kSectionSumReference;
      std::uint64_t allreduce_bucket_bytes = 0;
      std::uint32_t compact_xhat = 0;
      std::uint32_t cuda_graph = 0;
  };
  ```

- [ ] Define candidate-versus-training load semantics explicitly:

  ```cpp
  enum class P888PolicyLoadPurpose : std::uint32_t {
      kBenchmarkCandidateOrAccepted = 1,
      kAcceptedTrainingOnly = 2,
  };
  enum class P888PolicyAcceptance : std::uint32_t {
      kCandidate = 1,
      kAccepted = 2,
  };

  struct P888RuntimePolicyDocument {
      P888RuntimePolicy policy{};
      P888PolicyAcceptance acceptance = P888PolicyAcceptance::kCandidate;
      std::string selection_reason;
      std::string source_git_sha;
      Sha256Digest hardware_compatibility_sha256{};
      std::vector<Sha256Digest> runtime_tree_sha256_allowlist;
      std::array<std::uint8_t, 64> canonical_bytes{};
      Sha256Digest policy_sha256{};
      std::array<char, 17> candidate_id{};
  };
  struct P888RuntimeEnvironment {
      Sha256Digest hardware_compatibility_sha256{};
      Sha256Digest runtime_tree_sha256{};
      std::uint32_t sm_major = 0;
      std::uint32_t sm_minor = 0;
  };

  std::array<std::uint8_t, 64> SerializeP888RuntimePolicy(
      const P888RuntimePolicy& policy);
  Status ValidateP888RuntimePolicy(const P888RuntimePolicy& policy,
                                   std::uint32_t sm_major,
                                   std::uint32_t sm_minor);
  Status LoadP888RuntimePolicyJson(
      const std::filesystem::path& path,
      const P888RuntimeEnvironment& actual,
      P888PolicyLoadPurpose purpose,
      P888RuntimePolicyDocument* document,
      std::string* error);
  ```

  Benchmark purpose accepts `candidate` and `accepted`; training purpose accepts only `accepted`. Candidate reason is exactly `unmeasured_candidate`. Accepted reasons are `measured_winner`, `strict_fallback_no_candidate_cleared_margin`, and `a100_diagnostic`; SM75 production rejects the diagnostic reason.

- [ ] Make serialization byte-exact. The 64 bytes are ASCII `MGTP888P`, then little-endian `u32 schema_version`, `u32 batch_norm`, `u32 linear_precision`, `u32 math_mode`, `u32 input_gradient`, `u32 input_tile`, `u32 dz_f16_source`, `u64 lt_workspace_bytes`, `u32 weight_reduction`, `u64 allreduce_bucket_bytes`, `u32 compact_xhat`, and `u32 cuda_graph`. There is no native-struct padding and no embedded checksum. JSON stores 128 lowercase hex characters as `canonical_hex`, the full digest as `sha256`, and its first 16 characters as `candidate_id`.

  Freeze these literal vectors in C++ and Python:

  ```text
  strict canonical_hex:
  4d475450383838500100000001000000010000000100000001000000000000000100000000000000000000000100000000000000000000000000000000000000
  strict sha256:      2bc050ea14881b5c682ae36d8d01d2cedf1a8ca0431b88400ddda34cffb807b5
  strict candidate:   2bc050ea14881b5c

  mixed canonical_hex:
  4d475450383838500100000003000000020000000300000003000000300000000200000000000001000000000300000000000000000000000000000000000000
  mixed sha256:       ce8ad26fe1f6227c6c710c787d112b76f0b9e59c4479bbd620e293c21ab2b96d
  mixed candidate:    ce8ad26fe1f6227c
  ```

  The mixed vector means coalesced-V2 BN, mixed linear, FP16-candidate math, tiled-FP16 input gradient, tile 48, caller-provided `dz_f16`, 16 MiB Lt workspace, two-range reduction, and all optional fields zero.

- [ ] Add an explicit CUDA adapter rather than casting portable IDs:

  ```cpp
  struct ResolvedP888CudaPolicy {
      BatchNormKernelPolicy batch_norm = BatchNormKernelPolicy::kLegacy;
      LinearPrecisionPolicy linear_precision =
          LinearPrecisionPolicy::kFp32Strict;
      P888MathMode math_mode = P888MathMode::kFp32;
      InputEmbeddingGradConfig input_gradient{};
      std::uint64_t lt_workspace_bytes = 0;
      WeightGradientReductionPolicy weight_reduction =
          WeightGradientReductionPolicy::kSectionSumReference;
  };

  mgt::Status ResolveP888CudaPolicy(
      const mgt::P888RuntimePolicy& serialized,
      std::uint32_t state_len,
      std::uint32_t sm_major,
      std::uint32_t sm_minor,
      ResolvedP888CudaPolicy* resolved);
  ```

  `p888_runtime_policy.cu` contains `static_assert` equality for every portable ID versus the corresponding Tasks 5/7/9/13 CUDA enum, derives `InputEmbeddingGradConfig`, and validates before any resource creation. No `reinterpret_cast` between enums is allowed.

- [ ] Reject all unsupported and noncanonical combinations. Owner input requires tile 0; tiled input requires `36,48,56,64`. Tiled FP16 with mixed linear requires caller-provided `dz_f16`; tiled FP16 with FP32 linear requires internally staged `dz_f16`; owner/tiled-FP32 require internally staged. FP32 linear requires FP32 math; mixed linear requires FP16-candidate math. TF32 is FP32-linear, A100 diagnostic only, and invalid on SM75. Strict owner uses Lt 0; other paths use a 256-byte multiple no greater than 64 MiB. Two-range requires Task-13 distributed resources. `allreduce_bucket_bytes`, `compact_xhat`, and `cuda_graph` are exactly zero. Unknown/reserved values, overflow, and any failed capacity query return `kInvalidConfig`. Tile-bucket overlap has no runtime enum.

- [ ] Parse JSON strictly in Python with `object_pairs_hook=reject_duplicate_pairs` and in C++ with a schema-specific tokenizer that records every key before conversion. Require exact root keys `schema_version`, `acceptance_status`, `selection_reason`, `source_git_sha`, `runtime_tree_sha256_allowlist`, `hardware_compatibility`, `hardware_compatibility_sha256`, `policy`, `canonical_hex`, `sha256`, and `candidate_id`. Require exact policy and compatibility-object keys. Reject duplicate/unknown/missing keys, boolean/float/exponent integers, negative/overflow values, malformed/nonlowercase hex, a `source_git_sha` other than 40 or 64 lowercase hex characters, mismatched canonical bytes/checksum/ID, and noncanonical zeros. The runtime-tree list is nonempty, sorted by raw digest bytes, duplicate-free, and must contain the current tree digest; the loader never appends it automatically.

- [ ] Build hardware compatibility from a sorted multiset of stable GPU descriptors, so rank ordering cannot change the hash. Each descriptor contains normalized model, vendor/device/subsystem IDs, SM major/minor, total memory, and multiprocessor count. Sort descriptors by canonical bytes, then encode GPU count, driver/CUDA/NCCL/cuBLASLt/CUTLASS/compiler/Release/architecture fields and exact P888 workload dimensions/full-final row vectors. Use magic `MGTHW001`, little-endian integers, `u32 length + UTF-8 bytes` strings, and `u32 count + elements` vectors. Freeze a Python/C++ fixture. BDF, UUID, rank mapping, clocks, temperature, utilization, and power are telemetry only; changing them must not alter the hash, while changing any stable/software/workload field must.

- [ ] Make source compatibility content-based. `scripts/p888_runtime_source_allowlist.txt` contains one reviewed Git pathspec per LF line, including `native/CMakeLists.txt`, glob pathspecs for runtime `.hpp/.cpp/.cuh/.cu` under `native/include/mgt`, `native/src`, and `native/cuda`, both native tools, exact `scripts/autotune_2xt4.py`, `scripts/summarize_bn_benchmark.py`, source-manifest and packager scripts, `:(glob)scripts/*p888*.py`, `:(glob)scripts/cluster/run_p888*.sbatch`, `:(glob)kaggle/kernel/run_p888*.sh`, and exact legacy T4 launchers still on the call path. Expand the union through `git ls-files`; a reserved glob may initially match nothing, but the union must be nonempty. Normalize forward slashes; reject NUL, CR, absolute paths, `..`, duplicate expanded paths, and any runtime entry outside reviewed pathspecs; sort UTF-8 path bytes. Hash synthetic `@allowlist` plus its bytes, then `path + NUL + u64_le(size) + bytes` per entry. The result is one `runtime_tree_sha256`; a policy initially allows exactly the measured digest. Future Task-16 files automatically enter through the predeclared globs and change the digest. Git SHA remains provenance only, so evidence-only commits do not change compatibility. Tests cover order, CRLF/LF bytes, one-byte/add/remove runtime changes including a one-byte summarizer drift, a reserved empty glob, an entirely empty union, ignored evidence, untracked files, and traversal.

- [ ] Give every executable path exactly one policy transport: `--policy-json PATH`. Reject it together with `--math`, `--input-grad-fp16`, `--input-grad-position-tile`, `--input-grad-sparse`, `--linear-fp16`, `--lt-workspace-bytes`, any `--lt-autotune` option, `--overlap-allreduce`, or `--allreduce-bucket-bytes`. `run_sweep_2xt4.sh` writes `candidate_id<TAB>policy_json`; it never renders/sources `selected.env`. It invokes `run_p888_bn_benchmark_2xt4.sh`, which directly launches two benchmark ranks. `run_ranks_2xt4.sh` may transport only `MGT_POLICY_JSON` and passes that path unchanged to the trainer; it must not reconstruct policy from environment variables.

- [ ] Add benchmark `--validate-policy-only 0|1`. Value 1 inspects the visible hardware/software environment, loads and CUDA-resolves policy, prints the full identity, creates no NCCL communicator, executes no train step, and exits. Atomic publication re-reads first through Python, then through this mode. A normal benchmark emits loaded policy SHA/effective fields before warmup, queries actual cuBLAS/cuBLASLt math mode and prepared plans, and fails on any mismatch.

- [ ] Define the byte-identical paired-state artifact:

  ```cpp
  struct P888BenchmarkSnapshot {
      std::uint32_t schema_version = 1;
      std::uint64_t seed = 0;
      std::uint64_t optimizer_step = 0;
      std::vector<float> linear_master;
      std::vector<std::uint16_t> linear_half_bits;
      std::vector<float> linear_adam_m;
      std::vector<float> linear_adam_v;
      std::vector<float> bn_affine;
      std::vector<float> bn_running;
      std::vector<float> bn_adam_m;
      std::vector<float> bn_adam_v;
      std::uint32_t device_health_flag = 0;
      std::uint32_t global_health_flag = 0;
  };

  Status WriteP888BenchmarkSnapshotAtomic(
      const std::filesystem::path& path,
      const P888BenchmarkSnapshot& snapshot,
      Sha256Digest* file_sha256);
  Status ReadP888BenchmarkSnapshot(
      const std::filesystem::path& path,
      const Sha256Digest& expected_sha256,
      P888BenchmarkSnapshot* snapshot);
  ```

  Format magic is `MGTP8S01`, followed by fixed-width little-endian vector counts/values and a trailing SHA256 over all preceding bytes. `--snapshot-out PATH --seed 888` is a host-only, single-process create-and-exit mode: it rejects rank/world/NCCL flags, opens with exclusive create, fsyncs, atomically renames, re-reads, and exits before CUDA/NCCL initialization. Only that controller process writes; benchmark ranks are read-only. Load requires exact seed equality, P888 schema/layout and vector sizes, finite FP32 state, zero device/global health, and exact `+0` in every padded lane. Measured processes require `--snapshot-in PATH --snapshot-sha256 HEX`.

- [ ] For every candidate, case, and pair index, create one canonical `pair_manifest.json` before launch with schema, 32-hex pair nonce, candidate/baseline policy SHA256, snapshot SHA256, seed, global/local row vector, warmup/steps/repeats, source/runtime/hardware hashes, and two ordered leg records. Pair 0 launches baseline then candidate, pair 1 candidate then baseline, and pair 2 baseline then candidate; each of the six fresh two-rank processes uses `--repeats 1`, the same pair nonce, its role, and order 0/1. No other benchmark may run between adjacent pair legs. After each leg, atomically add process/rank artifacts and exit codes, then seal the manifest SHA256. The summarizer performs a fail-closed cross-policy join and accepts a pair only when nonce, snapshot, seed/workload, hashes, row vector, counts, adjacency/order, and all rank rows match. Baseline is rerun for every candidate; state is never carried forward. Each policy therefore contributes three pair medians and 300 step samples per case.

- [ ] Implement the peak-memory monitor with no Python dependency beyond the standard library. `poll_p888_gpu_memory.py` loads `libnvidia-ml.so.1` through `ctypes`, starts before child creation, polls every 50 ms, and stops after all rank children exit. Its CLI is `--interval-ms 50 --output memory.jsonl --pid-file rank-pids.txt -- <command and arguments>`. JSONL records timestamp, visible device, BDF telemetry, total/used bytes, and matching child PIDs. Benchmark emits a separate live-allocation ledger. Define `peak_rank_bytes=max(raw NVML used high-water, live owned-allocation high-water)` and candidate peak as the maximum over ranks, full/final cases, pairs, and repeats. Do not subtract idle baseline. T4 limit is exactly `15569256448` bytes.

- [ ] Define controller retune behavior. `--retune allow` is valid only in Task 16 when both `MGT_TASK16_ACCEPTANCE=1` and `MGT_REQUIRE_TWO_GPUS=1`; it generates candidate documents and runs the complete matrix. `--retune never` requires an accepted policy with matching hardware and current runtime-tree membership. Missing/stale/invalid cache is fatal. Quarantine stale files under `rejected-cache/<sha256>-<reason>.json`. Publish `policy.json.tmp`, flush, rename atomically, fsync parent where supported, then Python re-read and native `--validate-policy-only 1`. Task 17 always uses `never`. If no optimization clears promotion, Task 16 explicitly publishes the fully validated strict candidate with reason `strict_fallback_no_candidate_cleared_margin`.

- [ ] Execute this finite search without early stop: strict baseline; all three BN policies retaining two; FP32 versus mixed linear for both retaining two; owner plus tiled-FP32/tiled-FP16 tiles `36,48,56,64` at 16 MiB retaining three; Lt `8,16,24,32 MiB` for nonowner survivors retaining three and explicitly including historical tile48/Lt16; section/single/two-range for all three; finally the deduplicated `2*2*3*3=36` retained interaction combinations, always below cap 64. Keep every OOM/crash/900-second-timeout/correctness/capacity row and prove rank-group/rendezvous cleanup. Bucket/compact/graph remain zero.

- [ ] Delete the old inline average-throughput selector. Require Task-4 summarizer output with exactly 300 acceptance-eligible max-rank step samples per policy/case. Quantiles use sorted element `max(0,ceil(q*n)-1)`. For pair `i` define `candidate_pair_score_i=9*full_pair_median_i+final_pair_median_i` and the same baseline score. Define `epoch_step_score_ms=9*full_Q50_all_300+final_Q50_all_300` and `combined_p95_ms=9*full_Q95_all_300+final_Q95_all_300`. Promotion requires correctness, no failure, peak within limit, all three candidate pair scores lower, candidate full Q50 at most `0.97*baseline`, full Q95 no worse, and final throughput at least `0.98*baseline`. Finish the matrix before selecting. Candidates within `1.005*best_score` form the tie band; choose lower combined p95, then lower peak, then lexicographically lower ID. One policy governs all ten epoch steps.

- [ ] Write `matrix.jsonl`, `summary.json`, and `policy.json` with commands, environment, canonical policy, runtime-tree allowlist, snapshot hash, per-rank artifacts, 300-sample full/final statistics, sealed pair manifests/pair medians/deltas, memory JSONL/ledger, correctness, failure, selection reason, and rejected rows. Tests cover CUDA17 inclusion, every ID mapping, candidate accepted only for benchmark, accepted for training, sorted runtime-tree membership, BDF invariance, stable-field drift, policy-flag conflict, snapshot corruption/A-B equality, native validate-only, exact 300-sample quantiles/pair scores/combined-p95, `15569256448` boundary, tie-breaking, dedup/cap, and absence of `selected.env`/tile-bucket candidates.

- [ ] Run schema/CPU/CUDA-mapping gates; authoritative tuning remains Task 16:

  ```powershell
  py -m unittest scripts.tests.test_autotune_2xt4 scripts.tests.test_runtime_source_manifest scripts.tests.test_poll_p888_gpu_memory scripts.tests.test_summarize_bn_benchmark -v
  cmake --build native/build --config Release --parallel
  ctest --test-dir native/build -C Release -R "^(sha256|p888_runtime_policy|p888_benchmark_snapshot|train_plan|p888_training_contract)$" --output-on-failure --no-tests=error
  docker exec mgt-gpu-queue python3 scripts/gpu_queue_submit.py --label task14-policy-mapping --wait -- bash -lc "cmake --build native/build --config Release --parallel && ctest --test-dir native/build -C Release -R '^p888_runtime_policy_mapping$' --output-on-failure --no-tests=error"
  py scripts/runtime_source_manifest.py --repo-root . --check
  git diff --check
  git status --short
  ```

- [ ] Commit implementation and force-add only the ignored schema evidence:

  ```powershell
  git add scripts/autotune_2xt4.py scripts/tests/test_autotune_2xt4.py scripts/runtime_source_manifest.py scripts/tests/test_runtime_source_manifest.py scripts/p888_runtime_source_allowlist.txt scripts/poll_p888_gpu_memory.py scripts/tests/test_poll_p888_gpu_memory.py scripts/summarize_bn_benchmark.py scripts/tests/test_summarize_bn_benchmark.py kaggle/kernel/run_autotune_2xt4.sh kaggle/kernel/run_sweep_2xt4.sh kaggle/kernel/run_p888_bn_benchmark_2xt4.sh kaggle/kernel/run_ranks_2xt4.sh native/include/mgt/sha256.hpp native/src/sha256.cpp native/tests/test_sha256.cpp native/include/mgt/p888_runtime_policy.hpp native/src/p888_runtime_policy.cpp native/tests/test_p888_runtime_policy.cpp native/tests/fixtures/p888_runtime_policy_v1.json native/cuda/mgt_cuda/p888_runtime_policy.cuh native/cuda/p888_runtime_policy.cu native/tests/cuda/test_p888_runtime_policy_mapping.cu native/include/mgt/p888_benchmark_snapshot.hpp native/src/p888_benchmark_snapshot.cpp native/tests/test_p888_benchmark_snapshot.cpp native/include/mgt/config.hpp native/src/train_plan.cpp native/tests/test_train_plan.cpp native/tools/mgt_bn_step_benchmark.cu native/tools/mgt_native_train_smoke.cu native/CMakeLists.txt
  git add -f test_results/p888_bn_autotune_schema.md
  git commit -m "perf: autotune executable p888 policy per gpu"
  ```
---

## Task 15: Integrate the Accepted-Policy SyncBN Step into a Resumable Real Trainer

**Files:**

- Modify: `native/tools/mgt_native_train_smoke.cu`
- Modify: `native/include/mgt/train_plan.hpp`
- Modify: `native/src/train_plan.cpp`
- Modify: `native/include/mgt/batch_norm_training.hpp`
- Modify: `native/src/batch_norm_training.cpp`
- Modify: `native/tests/test_batch_norm_training_plan.cpp`
- Create: `native/include/mgt/epoch_schedule.hpp`
- Create: `native/src/epoch_schedule.cpp`
- Create: `native/tests/test_epoch_schedule.cpp`
- Create: `native/include/mgt/p888_sample_generator.hpp`
- Create: `native/src/p888_sample_generator.cpp`
- Create: `native/tests/test_p888_sample_generator.cpp`
- Create: `native/tests/fixtures/p888_generator_v1.json`
- Create: `scripts/generate_p888_generator_fixture.py`
- Create: `scripts/generate_p888_checkpoint_v3_fixture.py`
- Create: `native/tests/fixtures/p888_checkpoint_v3_golden.json`
- Modify: `scripts/run_cuda_2rank_test.py`
- Create: `native/cuda/mgt_cuda/p888_sample_generator.cuh`
- Create: `native/cuda/p888_sample_generator.cu`
- Create: `native/tests/cuda/test_p888_sample_generator_cuda.cu`
- Modify: `native/include/mgt/training_artifacts.hpp`
- Modify: `native/src/training_artifacts.cpp`
- Modify: `native/tests/test_training_artifacts.cpp`
- Create: `native/include/mgt/checkpoint_consensus.hpp`
- Create: `native/src/checkpoint_consensus.cpp`
- Create: `native/tests/test_checkpoint_consensus.cpp`
- Create: `native/tests/cuda/test_checkpoint_consensus_2rank.cu`
- Create: `native/cuda/mgt_cuda/checkpoint_snapshot.cuh`
- Create: `native/cuda/checkpoint_snapshot.cu`
- Create: `native/tests/cuda/test_checkpoint_snapshot.cu`
- Create: `native/tests/cuda/test_weight_export_cuda.cu`
- Create: `native/tests/cuda/test_p888_native_train_world2.cu`
- Create: `native/include/mgt/training_telemetry.hpp`
- Create: `native/src/training_telemetry.cpp`
- Create: `native/tests/test_training_telemetry.cpp`
- Create: `native/tests/cuda/test_training_telemetry.cu`
- Create: `native/tests/cuda/test_training_health_2rank.cu`
- Create: `native/cuda/mgt_cuda/training_telemetry.cuh`
- Create: `native/cuda/training_telemetry.cu`
- Modify: `native/cuda/mgt_cuda/mlp_batch_norm_forward.cuh`
- Modify: `native/cuda/mlp_batch_norm_forward.cu`
- Modify: `native/cuda/sync_batch_norm.cuh`
- Modify: `native/cuda/sync_batch_norm.cu`
- Modify: `native/cuda/mgt_cuda/linear_train_ops.cuh`
- Modify: `native/cuda/mlp_backward.cu`
- Modify: `native/cuda/mgt_cuda/adamw.cuh`
- Modify: `native/cuda/adamw.cu`
- Modify: `native/cuda/mgt_cuda/allreduce_nccl.cuh`
- Modify: `native/cuda/allreduce_nccl.cu`
- Modify: `native/include/mgt/weight_export.hpp`
- Modify: `native/src/weight_export.cpp`
- Modify: `native/tests/test_weight_export.cpp`
- Modify: `native/CMakeLists.txt`

- [ ] Add `--batch-norm-training 0|1` and `--policy-json PATH` to the native trainer. Keep the existing no-BN path callable only for regression tests. BN training requires an accepted policy loaded by Task 14 before any CUDA allocation; the trainer has no autotuning code. Print source Git SHA, runtime-tree SHA256, hardware-compatibility SHA256, policy SHA256, contract fingerprint, and model-input hashes before initialization. Refuse obsolete policy-affecting CLI flags when BN training is enabled.

- [ ] Build an executable semantic-epoch schedule and fail on arithmetic overflow:

  ```cpp
  struct EpochBatch {
      std::uint32_t batch_index = 0;
      std::uint64_t epoch_sample_offset = 0;
      std::uint32_t global_rows = 0;
      RankBatchSlice rank_slice{};
  };

  Status BuildP888EpochSchedule(
      std::uint32_t world_size,
      std::uint32_t rank,
      std::array<EpochBatch, 10>* out);
  ```

  Assert nine batches of 100,000 and one batch of 99,978; offsets `0,100000` through `900000`; total 999,978; exact depth counts of 34,482 for each depth 1..29; and no padding, overlap, duplication, or dropped tail. The pure schedule builder supports world 1 only as an explicitly prepared CPU/regression oracle and worlds 2/8 as prepared production/diagnostic plans. Before any CUDA allocation the BN trainer rejects every world other than 2 or 8; Kaggle additionally requires world 2, while the A100 diagnostic requires world 8. Test world sizes 1, 2, and 8 plus rejection of 0, 3, 4, invalid rank, null output, and final A100 slices `12498,12498,12497,12497,12497,12497,12497,12497`.

- [ ] Make sample identity independent of batching, rank, worker scheduling, and resume:

  ```text
  epoch_position in [0,999978)
  source_sample_id = PermuteEpochPosition(epoch_position, semantic_epoch, seed)
  depth            = 1 + source_sample_id / 34482
  walker_index     = source_sample_id % 34482
  sample_key       = (generator_version, seed, semantic_epoch, depth, walker_index)
  random_counter   = (move_index, rejection_retry)
  ```

  Use a four-round balanced Feistel permutation over `[0,1048576)`: split the 20-bit input into 10-bit `L,R`; for round `r=0..3`, compute `F=low10(SplitMix64(epoch_key ^ (uint64(r+1)*0xD1B54A32D192ED03) ^ R))`, then assign `(L,R)=(R,L xor F)`. Return `(L<<10)|R`. Set `epoch_key=SplitMix64(seed) ^ rotl64(SplitMix64(semantic_epoch),17)`. For the restricted domain, repeatedly apply that same permutation while the result is at least 999,978. Cap cycle-walking at 1,048,576 applications and fail if exceeded. Test all 999,978 positions for bijection/no gaps at epochs 0, 1, and 32,691.

- [ ] Specify every generator bit operation, including rejection sampling. Unsigned addition, multiplication, shifts, and rotation wrap modulo `2^64`:

  ```cpp
  std::uint64_t SplitMix64(std::uint64_t x) {
      std::uint64_t z = x + UINT64_C(0x9E3779B97F4A7C15);
      z = (z ^ (z >> 30)) * UINT64_C(0xBF58476D1CE4E5B9);
      z = (z ^ (z >> 27)) * UINT64_C(0x94D049BB133111EB);
      return z ^ (z >> 31);
  }
  ```

  Generator version is 1. Compute `sample_key = SplitMix64(seed) ^ rotl64(SplitMix64(generator_version),5) ^ rotl64(SplitMix64(semantic_epoch),13) ^ rotl64(SplitMix64(depth),29) ^ rotl64(SplitMix64(walker_index),47)`. At move zero exclude exactly `inverse_moves[-1]`; later exclude the inverse of the previous move; enumerate remaining legal move IDs in the canonical Task-1 order. For degree `d`, set `threshold=(uint64(0)-uint64(d)) % uint64(d)`, start `retry=0`, draw `x=SplitMix64(sample_key ^ (uint64(move_index)<<32) ^ uint64(retry))`, retry while `x<threshold`, and select `x%d`. Fail rather than wrap if a rejected draw occurs at `retry=UINT32_MAX` or if `d=0`. The label and random-walk direction follow the archived original source from Task 1 exactly.

- [ ] Generate `p888_generator_v1.json` with an independent Python implementation that imports no native library. Freeze these arithmetic goldens before writing C++:

  ```text
  SplitMix64(0)            = e220a8397b1dcdaf
  SplitMix64(1)            = 910a2dec89025cc1
  SplitMix64(UINT64_MAX)   = e4d971771b652c20
  rotl64(SplitMix64(1),5)  = 2145bd91204b9832

  seed=888, epoch=0, depth=1, walker=0:
  sample_key               = 4735036e19c62b35
  draw(move0,retry0/1/2)   = 5afeb2af0f996c80,
                              1149d5459c4c0e12,
                              0120b97029b95001
  threshold(17)            = 1
  move-zero legal IDs      = 0..15,17
  first selected move ID   = 13
  resulting state FNV      = 926dfa3bef6e5bb1
  label bits               = 3f800000

  seed=888, epoch=32691, depth=29, walker=34481:
  sample_key               = 7f51103af3145311
  first draw               = d4a4afc0e3aff252
  move IDs                 = 7,0,15,8,2,13,4,6,14,16,7,2,5,7,7,10,1,9,15,17,3,0,5,15,7,3,10,10,14
  final state FNV          = badbb7a0bb5c43fb
  label bits               = 41e80000

  epoch0 input12 raw/cycle = 1037444 / 779320
  ```

  Store literal first-32 cycle-walked outputs:

  ```text
  epoch 0:
  83000,172781,62217,720489,974521,552744,547385,184528,
  977733,499328,441032,466676,779320,42144,30808,16834,
  617951,952399,862728,660770,580222,783603,561553,949141,
  203611,608141,44654,274671,118736,722140,874149,595483

  epoch 1:
  114553,630504,912090,772576,986977,246770,120975,411273,
  221056,779587,592715,38841,500123,356299,216840,391895,
  34026,213099,896418,684452,959949,463448,265328,987412,
  19960,667170,702622,113755,376873,379962,153971,142571

  epoch 32691:
  211785,305563,285951,759783,815285,276846,119741,189170,
  769709,955233,804618,539592,912258,207118,344867,850225,
  946525,504712,378010,470234,671992,302618,145653,219520,
  400412,214171,482890,840799,264231,259034,889624,886990
  ```

  Also store state/label bytes for every depth boundary, generator-script and archived-source SHA256. Regenerate twice byte-identically. C++ consumes literals and never invokes Python at CTest time. Compare original/new depth distributions; exact sequence equality is intentionally not required.

- [ ] Keep `p888_sample_generator.cpp` as the independent CPU oracle, but generate production batches on GPU. Expose a prepared CUDA API:

  ```cpp
  struct P888SampleGeneratorPlan;
  struct P888SampleIntegrityDeviceDelta;
  struct P888SampleIntegrityDeviceState;
  struct P888SampleIntegrityFenceWorkspace;

  mgt::Status CreateP888SampleGeneratorPlan(
      const mgt::PuzzleDefinition& puzzle,
      std::uint32_t capacity_rows,
      std::uint32_t device_id,
      P888SampleGeneratorPlan** out);
  mgt::Status DestroyP888SampleGeneratorPlan(P888SampleGeneratorPlan* plan);
  mgt::Status ResetP888SampleIntegrityDelta(
      P888SampleIntegrityDeviceDelta* delta,
      cudaStream_t stream);
  mgt::Status LaunchP888SampleBatch(
      const P888SampleGeneratorPlan* plan,
      std::uint64_t seed,
      std::uint64_t semantic_epoch,
      std::uint64_t epoch_batch_offset,
      const mgt::RankBatchSlice& slice,
      mgt::TrainStateStorage* states,
      float* labels,
      P888SampleIntegrityDeviceDelta* pending_delta,
      cudaStream_t stream);
  mgt::Status CommitP888SampleIntegrityDelta(
      const P888SampleIntegrityDeviceDelta* pending_delta,
      P888SampleIntegrityDeviceState* local_committed_suffix,
      cudaStream_t stream);

  constexpr std::uint32_t kP888IntegrityAccumulatorWords = 38;
  constexpr std::uint32_t kP888IntegrityFenceWords = 86;
  mgt::Status CreateP888SampleIntegrityFenceWorkspace(
      std::uint32_t max_world,
      std::uint32_t device_id,
      P888SampleIntegrityFenceWorkspace** out);
  mgt::Status DestroyP888SampleIntegrityFenceWorkspace(
      P888SampleIntegrityFenceWorkspace* workspace);
  mgt::Status GlobalizeP888SampleIntegrityAtFence(
      P888SampleIntegrityDeviceState* global_base,
      P888SampleIntegrityDeviceState* local_suffix,
      std::uint64_t semantic_epoch,
      std::uint64_t committed_epoch_sample_offset,
      std::uint32_t next_batch_index,
      std::uint64_t* globalized_epoch_offset,
      std::uint64_t completed_epoch_count,
      const std::uint64_t completed_chain_words_le[4],
      std::uint64_t local_last_effective_step_nanoseconds,
      std::uint64_t* canonical_last_effective_step_nanoseconds,
      std::uint32_t rank,
      std::uint32_t world,
      NcclRankContext* bn_context,
      P888SampleIntegrityFenceWorkspace* workspace,
      cudaStream_t stream);

  struct P888TerminalHealthFenceWorkspace;
  struct P888TerminalHealthFenceResult {
      std::uint32_t global_health_mask = 0;
      std::uint32_t validation_code = 0;
  };
  mgt::Status CreateP888TerminalHealthFenceWorkspace(
      std::uint32_t max_world,
      std::uint32_t device_id,
      P888TerminalHealthFenceWorkspace** out);
  mgt::Status DestroyP888TerminalHealthFenceWorkspace(
      P888TerminalHealthFenceWorkspace* workspace);
  mgt::Status AgreeP888TerminalTrainingHealth(
      const std::uint32_t* device_local_health_mask,
      std::uint32_t* device_global_health_mask,
      std::uint32_t rank,
      std::uint32_t world,
      NcclRankContext* bn_context,
      P888TerminalHealthFenceWorkspace* workspace,
      cudaStream_t stream);
  mgt::Status FinishP888TerminalTrainingHealth(
      P888TerminalHealthFenceWorkspace* workspace,
      P888TerminalHealthFenceResult* result);
  ```

  Declare/implement `P888TerminalHealthFenceWorkspace` and its four functions in `native/cuda/mgt_cuda/training_telemetry.cuh` / `native/cuda/training_telemetry.cu`; the sample-integrity files only consume the finished result. It is created with the train plan on every rank and owns one device `u64` send word, `max_world` device receive words, one mapped/pinned `P888TerminalHealthFenceResult`, and one precreated completion event. Pre-collective checks are limited to immutable plan-time pointer/capacity/device/context invariants whose cross-rank fingerprint was already proved equal. For every runtime mask—including an unknown bit on only one rank—`AgreeP888TerminalTrainingHealth` must pack the raw local 32-bit value into the low half of one `u64` with zero high half and execute exactly one fixed-count NCCL AllGather on `runtime.bn_context` before validating mask bits. Its finalize kernel validates every gathered high half and known-bit mask, ORs rank words in ascending order, writes the same raw global mask and validation code (`0=valid`, `1=unknown low-half bit`, `2=nonzero high half`, with code 2 taking precedence) to device/pinned result, and records the event. `FinishP888TerminalTrainingHealth` waits only for that precreated event; every rank therefore returns the same `kOk` or `kInvalidData` and result after peers have completed the collective. It allocates nothing, never uses pageable/stack async storage, and does not clear either mask. Every rank calls it exactly once after the policy-neutral `terminal_health_ready` event on every pause, epoch, final, and drain/failure terminal path. Only after `Finish` succeeds may a pending commit be accepted and terminal telemetry be stamped; nonzero health or validation failure rejects pending/speculative deltas and forbids integrity/checkpoint publication. Tests cover every single category on rank 1, simultaneous categories, an unknown bit only on rank 1, corrupted high-half injection, worlds 2/8, one-word/one-rank-short storage rejected at prepared-plan creation, fixed collective trace, identical OR/status result, and no publication on failure.

  Plan creation copies canonical move permutations, inverse IDs, and target once and validates P888's 18 moves/72 positions. One GPU thread owns one sample and derives `epoch_position=epoch_batch_offset+slice.global_offset+local_row`; it executes the exact Feistel/counter/rejection contract, writes all 72 state bytes, zeros storage padding, writes the original float label, and contributes only to the selected buffer's reset-before-use `pending_delta`. Reduce sums/XOR/histograms in shared memory and issue only one set of global atomics per block, not per field per sample. After global health agreement, merge a committed delta into the rank-local suffix; on health failure or abandoned prefetch discard/reset it. No CPU sample construction or per-batch H2D state/label copy is allowed. CUDA-versus-CPU tests cover noncontiguous slices, partial blocks, every depth boundary, both T4 row counts, literal goldens, exact bytes, discarded speculative deltas, and exactly-once commit.

- [ ] Represent partial-epoch integrity as an identical `global_base` plus one rank-local `local_suffix`. Both start at zero with `globalized_epoch_offset=0`. Normal committed batches update only `local_suffix`. Allocate `P888SampleIntegrityFenceWorkspace` once in the train plan with device send storage for exactly 86 `u64`, device receive storage for `86*max_world`, equally sized pinned mirrors, and one precreated event; reject `max_world` other than 2 or 8, wrong device, null/undersized storage, or communicator mismatch. Do not alias the separate five-word checkpoint-digest workspace.

  At every checkpoint/pause/final/epoch fence, `GlobalizeP888SampleIntegrityAtFence` allgathers this exact 86-word record in rank order: 38 `global_base` words; 38 `local_suffix` words; `semantic_epoch`; `committed_epoch_sample_offset`; zero-extended `next_batch_index`; `globalized_epoch_offset`; `completed_epoch_count`; four little-endian `u64` words of the completed-chain SHA256; and `local_last_effective_step_nanoseconds`. Require the 38 base words, four cursor words, completed count, and all 32 chain bytes to be byte-identical across ranks before any merge; validate each suffix count against the deterministic rank partitions since the prior fence; combine suffixes in ascending rank order; choose the maximum positive timing word; then atomically write the resulting canonical base/timing on every rank, reset every suffix to zero, and set `*globalized_epoch_offset=committed_epoch_sample_offset`. Any validation or collective failure leaves base, suffix, timing, and offset unmodified. A repeated fence with no new batch is idempotent. A checkpoint serializes only that identical base/offset/chain/count and canonical timing word; on resume every rank restores the base and an empty suffix. Thus checkpoint bytes agree while hot steps add no integrity collective. CPU/CUDA tests cover exact 86/172/688-word capacities, world 2/8, mismatched base/cursor/completed-count/chain, bad suffix count, estimator disagreement/max selection, repeated fence, and undersized storage.

- [ ] Compute the partition-independent record checksum without gathering a million records. Encode each sample as `u64_le(epoch_position)`, `u64_le(source_sample_id)`, `u32_le(depth)`, `u32_le(walker_index)`, 72 state bytes, and `u32_le(float_label_bits)`; FNV-1a-64 uses offset `0xCBF29CE484222325` and prime `0x100000001B3`. Set `mixed=SplitMix64(record_fnv ^ SplitMix64(epoch_position))`. A delta contains nine `u64` fields—count, mixed sum/XOR, position sum/square-sum/XOR, source-ID sum/square-sum/XOR—plus 29 `u64` depth counts; arithmetic wraps modulo `2^64`, while counts use checked addition. Combination sums sum-like fields, XORs XOR-like fields, and sums depth counts.

  Canonical epoch-record bytes are exactly: ASCII `MGTPSMP1`; `u32_le(schema=1)`; `u32_le(generator_version)`; `u64_le(seed)`; `u64_le(semantic_epoch)`; `u64_le(count)`; `u32_le(state_len=72)`; `u32_le(depth_count=29)`; then eight `u64_le` fields in order `mixed_sum,mixed_xor,position_sum,position_square_sum,position_xor,source_sum,source_square_sum,source_xor`; then 29 `u64_le` depth counts. The epoch checksum is FNV-1a-64 of those bytes. At the epoch fence require count 999,978, closed-form position/source invariants for `0..999977`, and every depth count 34,482. Cycle-walking's permutation proof establishes no logical duplicates/gaps; the finite checksum is a corruption detector, not a collision-free proof.

  Freeze an aggregator-only two-record fixture: record FNVs `0937139daf56e7d9` and `56a920f909ca2094`, positions `0,1`, source IDs `7,3`, depths `1,2`, generator 1, seed 888, epoch 0. It yields mixed sum `a790166903a8abaf`, mixed XOR `9b20125cf5676278`, position invariants `1,1,1`, source invariants `10,58,4`, depth counts 1 for depths 1/2 and zero otherwise, a 344-byte canonical record, and epoch checksum `edb37fcdfb224b0d`. The independent Python fixture stores the full canonical hex and literal checksum. Test missing, duplicate, reordered-position, altered-state, and altered-label cases.

- [ ] Define the completed-epoch SHA256 chain byte-for-byte. `chain_0 = SHA256(ASCII "MGTP8IC1" || u32_le(1) || u32_le(generator_version) || u64_le(seed) || training_contract_sha256_bytes)`. After verifying epoch `e`, compute `chain_{e+1} = SHA256(ASCII "MGTP8IE1" || chain_e || u64_le(e) || u64_le(epoch_checksum) || canonical_epoch_record_bytes)`. Store completed-epoch count and current 32-byte chain in every checkpoint/bundle; require count `e+1` and exact recurrence on resume. At an epoch boundary extend the chain before resetting base/suffix for the next epoch.

- [ ] Build every stream, event, handle, communicator, descriptor, generator/policy plan, telemetry slot, checkpoint staging buffer, and allocation once. Task 15 creates one timing-disabled, policy-neutral `terminal_health_ready` event on every rank for section, single, and two-range reduction policies. For every ordinary step, finish the policy-specific last weight reduction first (section/single are already ordered on the compute stream; two-range makes compute wait on Task 13's `weight_done`), decode the weight sentinel, run all Adam/moment/FP16-mirror stores and their late-health producers, and record `terminal_health_ready` last. For a health-zero stop decided by the first input-BN control SUM, accept the previous pending record, discard current speculation, and record a fresh `terminal_health_ready` generation immediately after that SUM because no later current-step health producer is admitted. For `drain_only`, record it only after every guarded remaining kernel/collective and possible health producer completes. `AgreeP888TerminalTrainingHealth` must be enqueued on that same compute stream after the path-specific fresh record and, on stop, after the current first-BN control finalize; using another stream with `runtime.bn_context` is forbidden. It never treats `weight_done` as universal step completion. Task 15 creates no third/checkpoint communicator: terminal-health, 86-word integrity, checkpoint digest consensus, and publication-status broadcast all receive and alias the already-created `runtime.bn_context`; `runtime.weight_context` remains exclusive to Task 13 weight-gradient reductions. Maintain one checked `bn_collective_sequence` counter in the prepared runtime and increment it for every BN/control collective so per-rank trace tests prove identical order. Use `capacity_rows=ceil(100000/world_size)`: 50,000 on two ranks and 12,500 on eight. Register active rows `{50000,49989}` for two ranks and `{12500,12498,12497}` for eight. All offsets use capacity; kernels receive active rows. The final batch reuses plans without resize, lazy cache fill, or policy switch. Tests inject a late Adam/moment/mirror bit on rank 1 under each of the three weight policies and prove `terminal_health_ready -> terminal AllGather` plus identical failure.

- [ ] Double-buffer generated states/labels with two pending-integrity deltas, `sample_ready[2]`, and `buffer_free[2]`. Generate batch 0 before timing; for batch `b`, compute waits on its ready event, while the generator waits for the other buffer's prior full backward/input-gradient use, resets its delta, and prepares `b+1`. `PendingStepCommit` has the explicit states `empty|pending|accepted|rejected`; only `pending -> accepted|rejected` is legal and its cursor/delta/telemetry effects execute exactly once. End-of-step `b` creates one pending record. At the first BN collective of step `b+1`, globalize all late health from `b` before any new running-state update and transition that record once. At a pause/epoch/final/drain boundary every rank waits on the policy-neutral `terminal_health_ready` event and calls exactly one `AgreeP888TerminalTrainingHealth`: if the record is still pending because no next-step control consensus ran, the identical OR mask resolves it; if stop/health control already resolved it, the terminal fence only confirms the shared sticky mask and must not transition or emit it twice. A newly accepted result commits its delta/cursor and stamps telemetry before the 86-word integrity gather; an unhealthy result rejects it, discards prefetched deltas, emits failure telemetry, and skips integrity/checkpoint publication identically on all ranks. Prefetched `b+1` never changes committed state. Never overwrite states before input-table gradient finishes.

- [ ] Use one sticky, explicit health mask:

  ```cpp
  enum TrainingHealthBits : std::uint32_t {
      kInvalidCategory     = 1U << 0,
      kNonFiniteLoss       = 1U << 1,
      kNonFiniteBnStats    = 1U << 2,
      kNonFiniteGradient   = 1U << 3,
      kNonFiniteParameter  = 1U << 4,
      kNonFiniteMoment     = 1U << 5,
      kNonFiniteHalfMirror = 1U << 6,
  };
  ```

  Reserve the first padded input-weight gradient `layout.input_weight + logical_hd1` as a nonlogical health sentinel and assert it lies outside every logical projection. Producers OR pre-Adam category bits while producing normal outputs. After input backward and before the section-input/single-sum/two-range-prefix reduction point, write sentinel `1.0f` iff any local bit is set. Existing SUM globalizes it; after the policy-specific final weight-reduction completion (section/single are ordered on compute, two-range waits `weight_done`), decode global health, restore sentinel/padding to exact `+0`, and make all Adam/mirror stores no-op if already unhealthy. Adam/moment/half-mirror validation may set new late bits. Those sticky late bits enter Task-6's first input-BN control payload at the next step before any running-state change and decide the prior `PendingStepCommit`; at a stopping boundary the explicit fence agreement performs the same decision. Task 15 supersedes Task 6's standalone per-generation clear: initialize sticky local/global health only on fresh start or verified resume, clear it only after the corresponding pending step is unanimously healthy at the next consensus/fence, and let generation reset only its selected pending-integrity delta.

  The real trainer extends `MlpBatchNormStepBuffers` with caller-owned `device_local_stop_request`, `device_global_stop_request`, and mapped/pinned `host_local_stop_request`, allocates them once, and prepares the first input-BN reduction with capacity `2*logical_hd1+2`. Task 15 explicitly versions and supersedes Tasks 8-9's one-control-word API: in `StridedSyncBatchNormFusedForwardArgs`, replace `bool reduce_health` with `std::uint32_t control_word_count`, retain the exact local/global health pointers, and add exact `const std::uint32_t* device_local_stop_request` plus `std::uint32_t* device_global_stop_request`. Count 0 requires all four pointers null; count 1 requires only both health pointers; count 2 requires all four; every other/pointer-mismatched mode fails before enqueue. Extend both legacy and V2 workspace queries with that count.

  Freeze the nonoverlapping layout and use it in the query, partial, finalize, and NCCL-count code:

  ```text
  control_words       = 0 | 1 | 2
  chunk_count         = ceil(capacity_rows / 256)
  control_base        = 2 * logical_cols
  chunk_base(0)       = 0
  chunk_base(chunk>0) = control_words + chunk * 2 * logical_cols
  legacy_required     = 2 * logical_cols + control_words
  v2_required         = control_words + 2 * logical_cols * chunk_count
  health_word         = control_base                         when control_words >= 1
  stop_word           = control_base + 1                     when control_words == 2
  ```

  This moves V2 chunk 1 from Task 9's one-control offset to `2*logical_cols+control_words`, so neither control word can alias a partial. Add `input_control_words=2` to the host plan and make its maximum exact workspace queries use `legacy_reduction_count=2*max(logical_hd1,logical_hd2)+2` and `v2_reduction_count=2+2*max(logical_hd1,logical_hd2)*ceil(capacity_rows/256)`. Other BN sites pass zero local control words but share the preallocated maximum. CPU/CUDA tests cover counts 0/1/2, every pointer mismatch, chunk counts 1 and greater than 1, T4/A100 full and final production rows, exact capacity, one-float-short capacity, and sentinel patterns proving control/partial nonaliasing.

  Control word 0 is exactly the Task-6 health SUM; control word 1 is a distinct nonfailure `stop_requested` SUM. Every rank writes each local value as exact `0.0f|1.0f`, uses the same two-word input-site payload length for every policy/world/row shape, and decodes the reduced words separately; a nonzero health word has failure precedence and a stop word never sets health. Before every attempted step, each rank updates only its preallocated stop slot but still enqueues the identical admission prefix through sample generation, the first linear transform, and this first BN reduction. No rank may branch or skip that prefix from its local clock. If health is zero and reduced stop is nonzero, all ranks accept the prior pending commit exactly once, perform no current BN running-state update or later step work, discard the newly admitted current and any prefetched integrity deltas, and enter the common pause/checkpoint fence with no pending record. If health is nonzero, all ranks reject the prior pending record exactly once and take the drain path regardless of stop. If both controls are zero, the ordinary step continues. The later terminal-health fence still runs on stop/drain but confirms rather than repeats that transition. This adds four bytes to an existing collective, not a new collective; collective-byte counters, strict/legacy/V2 workspace queries, exact/undersized capacity, one-rank stop, stop-plus-health precedence, full/final rows, and world-2/8 tests must expect it.

- [ ] Adam reads old parameter/m/v and gradient, computes candidates and FP16 conversion in registers, validates old/input/candidates before store, and suppresses the failing element while setting its sticky bit. BN finalize validates candidate running state before store; gradient producers validate their own outputs. Do not launch a separate model scan. Poll `ncclCommGetAsyncError` on both communicators every 100 steps and at every epoch/checkpoint fence. A rank never exits mid-epoch because its telemetry completed first: all ranks continue identical guarded kernels/collectives, allgather masks at the common fence, make one shared stop decision, and forbid checkpoint publication when any bit is set. Reset masks only after a unanimously healthy fence.
- [ ] Define failure drain as one nonbranching state machine. When a reduced health word becomes nonzero, reject the prior pending record exactly once at that control consensus and transition every rank to `drain_only` for the same physical scheduled batch: do not advance cursor/global step, merge integrity, or create another `PendingStepCommit`; discard current/prefetched deltas; and enqueue the remaining guarded kernels and collectives for that already-admitted batch with deterministic zero payloads and no parameter/running-state stores. At the common terminal health fence, confirm the nonzero mask without a second transition, reject the distinct drain telemetry slot once, drain prior/drain failure telemetry in sequence, publish no checkpoint, and exit nonzero together. If late health is first discovered by an explicit pause/final fence and the record is still pending, no artificial next batch is admitted and that fence performs its sole rejection. Tests inject each late bit on rank 1, verify identical collective traces and physical batch index on both ranks, exactly one pending-state transition and telemetry record, no repeated logical batch/cursor change, all speculative deltas absent, and zero checkpoint publication.

- [ ] Implement a nonblocking, fixed-capacity telemetry pipeline with separate device and completed records:

  ```cpp
  struct DeviceTrainingTelemetryRecord {
      std::uint64_t sequence;
      std::uint64_t semantic_epoch;
      std::uint64_t global_step;
      std::uint32_t batch_index;
      std::uint32_t local_rows;
      std::uint32_t global_rows;
      float loss;
      std::uint32_t local_health_mask;
      std::uint32_t global_health;
      std::uint32_t committed;
  };
  struct CompletedTrainingTelemetryRecord {
      DeviceTrainingTelemetryRecord device;
      float generation_seconds;
      float sample_wait_seconds;
      float compute_seconds;
      float effective_step_seconds;
      TrainingCollectiveCounters collectives;
  };
  ```

  Keep canonical telemetry PODs and `TrainingCollectiveCounters` in `mgt_core`; they contain only fixed-width C++ fields and no CUDA/NCCL types. Each ring slot stores immutable sequence metadata plus host counter snapshots taken immediately before enqueue and immediately after all calls for that step are enqueued; its delta is frozen before the terminal event can be drained. CUDA adapters map existing BN counters into the POD. Preallocate 256 device/pinned records, a telemetry stream, timing-enabled `generation_start/generation_stop`, `sample_wait_start/sample_wait_stop`, `compute_start/compute_stop`, and timing-disabled `commit_decided`, plus `copy_done` per slot. Record wait-start, enqueue `cudaStreamWaitEvent(sample_ready)`, then wait-stop on compute; compute timing starts after it and ends after health/Adam/mirror. `effective_step_seconds=sample_wait_seconds+compute_seconds`; samples/s and ETA use it. Report `hidden_generation_estimate_seconds=max(0,generation_seconds-sample_wait_seconds)` as diagnostic only, never as an exact overlap or promotion metric. A slot remains pending after its timing events: the first-BN consensus of the next step, or the explicit stopping fence, stamps its final global health and `committed=1|0`, records `commit_decided`, and only then may the telemetry stream enqueue the terminal copy and `copy_done`. Thus a late parameter/moment/half-mirror failure can never appear as a healthy committed step. After `copy_done` query succeeds, CPU reads intervals and the already-frozen counter delta, writes per-rank JSONL, and rank 0 prints once in sequence order; a rejected slot is emitted once as terminal failure evidence and is excluded from progress, throughput, and `last_effective_step_nanoseconds`. Tests cover counter mutation after enqueue, 257 records, delayed/out-of-order copies, already-ready/delayed generation, delayed health success/failure, explicit-final-fence decision, ABA, exact order, and no mid-epoch abort.

- [ ] Extend checkpoint schema to version 3 and make it sufficient for bit-identical resume. Store canonical fixed-width metadata, then tensor payloads in this order:

  ```text
  physical FP32 linear master weights
  full FP16 linear mirror bits
  linear Adam m then v
  all 34 BN gamma then beta
  BN-affine Adam m then v
  all 34 BN running mean then variance
  next semantic epoch, next batch index, next epoch-sample offset
  canonical current-epoch global integrity base and globalized epoch offset
  completed-epoch count and integrity-chain SHA256
  global optimizer step, generator version, seed, canonical last-effective-step nanoseconds
  contract and hardware-independent model fingerprints
  exact accepted policy.json bytes and policy SHA256
  source Git SHA, runtime-tree SHA256, hardware-compatibility SHA256
  tensor byte sizes and per-payload SHA256
  trailing whole-checkpoint SHA256 over every preceding checkpoint byte
  ```

- [ ] Freeze one exact checkpoint-v3 wire schema before implementing I/O. The fixed prelude is exactly 616 bytes, contains no native padding or paths, and is serialized in this order:

  ```text
  bytes[8]  ASCII "MGTP8CP3"
  u32_le    schema = 3
  u32_le    endian_marker = 0x01020304
  u32_le    tensor_count = 10
  u32_le    descriptor_bytes = 104
  u32_le    policy_json_bytes
  u32_le    reserved0 = 0
  u64_le    global_optimizer_step
  u64_le    next_semantic_epoch
  u32_le    next_batch_index
  u32_le    generator_version
  u64_le    next_epoch_sample_offset
  u64_le    seed
  u64_le    globalized_epoch_offset
  u64_le    completed_epoch_count
  u64_le    last_effective_step_nanoseconds
  bytes[32] training_contract_sha256
  bytes[32] model_input_sha256
  bytes[32] runtime_tree_sha256
  bytes[32] hardware_compatibility_sha256
  bytes[32] policy_sha256
  bytes[32] completed_integrity_chain_sha256
  bytes[20] runtime_source_git_sha1 decoded from exactly 40 lowercase hex
  u32_le    integrity_word_count = 38
  u64_le[38] canonical current-epoch global integrity base
  ```

  Immediately append the exact accepted canonical `policy.json` bytes, including its single terminal LF, with length equal to `policy_json_bytes`; then append ten 104-byte descriptors; then append payloads contiguously; finally append the 32-byte whole-file digest. Each descriptor is exactly `u32 tensor_id`, `u32 dtype_id`, `u32 rank=1`, `u32 reserved=0`, four `u64` dimensions, `u64 element_count`, `u64 byte_count`, absolute `u64 payload_offset`, and 32 raw SHA256 bytes. Only dimension 0 is nonzero and equals element count; dimensions 1..3 are zero. Dtype IDs are `1=IEEE754_FP32_BITS`, `2=IEEE754_FP16_BITS`. Tensor IDs/order are: `1 linear_master`, `2 linear_half_mirror`, `3 linear_adam_m`, `4 linear_adam_v`, `5 bn_gamma`, `6 bn_beta`, `7 bn_affine_adam_m`, `8 bn_affine_adam_v`, `9 bn_running_mean`, `10 bn_running_variance`. Their counts are derived from the frozen physical layout and 34-site order; the first tensor payload begins at `616 + policy_json_bytes + 10*104`, every later offset equals the prior offset plus byte count, and no gaps/alignment/trailing bytes are legal.

  `generate_p888_checkpoint_v3_fixture.py` is an independent standard-library encoder and writes `p888_checkpoint_v3_golden.json` twice byte-identically. Freeze a serializer-only probe with policy bytes `{}\n`, zero metadata/digests/integrity except `generator_version=1` and `last_effective_step_nanoseconds=1`, ten zero-length rank-1 descriptors with IDs/dtypes above, payload offset 1659, and SHA256(empty) per descriptor. Its canonical prefix is 1659 bytes, total with trailer is 1691 bytes, and trailing SHA256 is exactly `738a1ae71d3570ae4b803eaadb90b373b87a72b7873f489b5035a34c73135d6e`. The probe is intentionally serializer-only; the production loader rejects zero tensor counts. C++ tests consume only the literal JSON/bytes, never call the Python encoder at CTest time, and cover each field offset, swapped descriptor, bad dtype/rank/dimension/count/offset, policy length/LF, reserved bits, source hex, overflow, and trailing data.

  Numeric fields are fixed-width little-endian and tensors are raw IEEE-754/FP16 bits. Canonical `checkpoint-v3.bin` is the exact schema above; the final 32 bytes equal SHA256 of every preceding byte and are not included in their own coverage. Sidecar `manifest.json`, `selected-policy.json`, and `COMPLETE` repeat/describe the digest but are outside it. The serialized global integrity base must describe exactly `[0,next_epoch_sample_offset)` of `next semantic epoch`, `globalized_epoch_offset` must equal that cursor, local suffix must have been reset, and pending prefetch/step deltas are absent. At an epoch boundary the base is zero and the completed epoch has already extended the chain. `last_effective_step_nanoseconds` is one positive little-endian `u64` canonicalized as the all-rank maximum by the 86-word integrity fence; every rank serializes the same value and resume restores it to every local estimator. Fresh training requires external `--policy-json`; resume may use embedded accepted bytes when absent and requires byte equality when present; both re-run Task-14 environment checks. Reject unknown/duplicate fields, extra/missing payloads, bad trailing/per-payload hashes, zero/overflowed timing, nonzero padding, or cursor/integrity/source/runtime/hardware mismatch.

- [ ] Precreate exact checkpoint I/O resources:

  ```cpp
  constexpr std::size_t kCheckpointChunkBytes = 16U * 1024U * 1024U;
  struct CheckpointIoWorkspace {
      std::byte* pinned_chunk;
      std::size_t pinned_chunk_capacity;
      std::uint64_t* pinned_send_words;       // 5
      std::uint64_t* pinned_all_words;        // 5*world
      std::int32_t* pinned_publish_status;
      std::uint64_t* device_send_words;
      std::uint64_t* device_all_words;
      std::int32_t* device_publish_status;
      cudaStream_t stream;
      cudaEvent_t checkpoint_ready;
      cudaEvent_t chunk_ready;
      cudaEvent_t consensus_ready;
      cudaEvent_t publish_done;
  };
  ```

  Allocate/register everything before warmup and verify pinned-memory attributes. On a healthy stopping boundary execute this exact collective/state order on every rank: wait `terminal_health_ready`; run the one-word terminal-health AllGather on `runtime.bn_context` and require `Finish == kOk`, `validation_code == 0`, and `global_health_mask == 0`; resolve/stamp the final `PendingStepCommit` only if it is still pending, otherwise assert its prior accepted/rejected state without a second transition; enqueue and drain every newly decided telemetry slot in sequence order; update the rank-local fixed-point timing estimator from every newly healthy committed slot; run the 86-word integrity AllGather on the same context; freeze all model/state streams and record `checkpoint_ready`; perform descriptor/final-digest consensus AllGathers; then execute the one publication-status broadcast, again on that context. Any terminal `Finish != kOk`, nonzero validation code, or nonzero global mask makes all ranks reject pending state and take the common no-checkpoint failure exit immediately after the health fence; they do not enter later integrity/digest/status calls. This order is mandatory and recorded in `bn_collective_sequence`: a slow final committed step must affect the stored estimator, and no rank-local late failure may be checkpointed. Use two deterministic chunked passes because descriptor hashes precede payloads: pass 1 copies each frozen payload in chunks at most 16 MiB and computes its SHA256 on every rank without writing the final file; all ranks agree the descriptor/hash table. Pass 2 re-copies the unchanged bytes, rank 0 writes the now-complete header/descriptors and payloads while every rank hashes the entire canonical prefix, then rank 0 appends the agreed trailing digest. A changed pass-2 payload hash aborts. Synchronize only the precreated `chunk_ready` per chunk. Test a deliberately slow last step, the exact healthy/failing collective traces, 16 MiB minus/exact/plus one, pass-1/pass-2 mismatch, descriptor corruption, trailing-digest exclusion, and digest independence from chunk boundaries. No checkpoint allocation/I/O occurs in an optimizer step.

- [ ] Use full SHA256 consensus rather than a truncated checksum:

  ```cpp
  struct CheckpointConsensusRecord {
      std::uint64_t payload_bytes = 0;
      std::array<std::uint64_t, 4> sha256_words_le{};
  };

  // mgt_core, no CUDA/NCCL dependency:
  Status VerifyCheckpointConsensusRecords(
      const CheckpointConsensusRecord* records,
      std::uint32_t count);

  // checkpoint_snapshot.cuh, CUDA/NCCL adapter:
  Status NcclAllGatherUint64(const std::uint64_t* send,
                             std::uint64_t* receive,
                             std::uint32_t count,
                             NcclRankContext* context,
                             cudaStream_t stream);
  Status NcclBroadcastInt32(std::int32_t* in_out,
                            std::uint32_t root,
                            NcclRankContext* context,
                            cudaStream_t stream);
  Status AgreeCheckpointDigest(const CheckpointConsensusRecord& local,
                               std::uint32_t rank,
                               std::uint32_t world,
                               CheckpointIoWorkspace* workspace,
                               NcclRankContext* bn_context);
  ```

  Keep `CheckpointConsensusRecord` and byte-equality validation in `mgt_core`; keep `CheckpointIoWorkspace`, streams, events, CUDA pointers, `NcclRankContext`, allgather/broadcast, and orchestration only in `checkpoint_snapshot.cuh/.cu`, preserving a green `MGT_ENABLE_CUDA=OFF` build. Define `sha256_words_le[i]=LoadU64LE(digest_bytes+8*i)`. `AgreeCheckpointDigest` uses `workspace->pinned_send_words`, `pinned_all_words`, and `pinned_publish_status` for every async copy; stack/pageable transfers are forbidden. All ranks allgather exactly payload-size plus four digest words and require byte identity before publication. Rank 0 writes `checkpoint.tmp-<global_step>`, fsyncs data/metadata/`COMPLETE`, fsyncs directory, atomically renames it, atomically replaces and re-reads `latest.json`, validates retention, and attempts required cleanup before setting the final status. All ranks then execute the same `NcclBroadcastInt32` on the aliased `runtime.bn_context`; there is no separately created checkpoint communicator. A write/pointer/retention failure broadcasts failure and every rank stops only after that common fence; no rank exits while peers can enter another collective.

- [ ] Retain only the newest three checksum-valid checkpoint directories in a live session plus the final directory. Rank 0 deletes an older point only after successful new publication, `latest.json` re-read, canonical-child/name validation, and proof at least two other valid resume points remain; it completes and validates retention before the one final all-rank publication-status broadcast. A cleanup failure is fatal but shared: publish a failure status at the common fence and stop all ranks together, leaving every already-valid point intact. A cross-session bundle carries only the checkpoint named by verified `latest.json`; previous Kaggle dataset versions retain earlier recovery points. If `latest.json` is damaged on resume, scan directories by descending step and choose the first `COMPLETE` checkpoint whose entire digest validates. Never remove the sole valid point.

- [ ] Add split/resume tests for uninterrupted 20 steps versus 7+13 and checkpoint after batch 7 with batch 8 prefetched; require identical global base/empty suffix/chain and prove speculative delta absence. Also test CPU-simulated world 2/8 suffix combination, an idempotent repeated fence with no new batch, A100 world-8 fence evidence, prefetch across an epoch boundary; final cursor `next_epoch/0/0`; last-step explicit health agreement; mixed half-mirror restoration; embedded-policy-only load and byte-mismatched external policy; source/runtime/hardware drift; truncated/extra payloads; two-pass/chunk/trailing-digest failures; corrupt `latest.json` fallback; two-rank digest disagreement; rank-0 data/rename/pointer/retention/status failures; retention safety; telemetry drain; and separate rank-1-only post-sentinel injections for parameter, Adam-m, Adam-v, and FP16-mirror failures. Every late failure must reject the pending cursor/integrity commit on all ranks and publish no checkpoint. Require exact counters/cursors/checksums/FP16 bits and Task-3 gates. Two-rank artifacts prove world 2 and both device IDs.

- [ ] Fold the 34 BN-bearing linear sites using the exact values forward reads; the output layer has no BN and is not folded:

  ```text
  W_effective[i,o] = mixed ? float(fp16_mirror_bits[i,o]) : fp32_master[i,o]
  b_effective[o]   = fp32_master_bias[o]
  scale[o]         = gamma[o] / sqrt(running_var[o] + epsilon)
  W_fold[i,o]      = W_effective[i,o] * scale[o]
  b_fold[o]        = beta[o] + (b_effective[o] - running_mean[o]) * scale[o]
  ```

  Half-bias mirror slots are never read. Preserve the checkout's existing row-major `[input,output]` HxK `_weight_hxk.fp16` layout; do not transpose. Fold only logical output columns, repack existing padded HxK shapes, force every padded row/column/bias lane to exact `+0`, and form a pre-cast folded FP32 buffer before casting each exported value to FP16 once. The host conversion must be bit-identical to `__float2half_rn` for ties, subnormals, overflow, infinities, NaNs, and signed zero; freeze literal conversion tests instead of reusing an approximate host converter. Residual fc2 preserves `ReLU(residual+linear)`. Gates: nonsymmetric CPU algebra max-abs/relative-L2 `2e-6`; strict native versus the pre-cast folded FP32 buffer `2e-5`; strict and mixed native versus the actual reloaded structured FP16 export max-abs `0.01`, relative-L2 `0.02`; repeated and pre/post-resume export byte-identical. Include a fixture where FP32 master and widened half mirror differ enough to fail a wrong mixed fold. Inference executes no BatchNorm and keeps the existing manifest/scoring contract.

- [ ] Register an exact CTest named `p888_native_train_world2` that exercises the real production target, never a duplicated mini-trainer. Extend `scripts/run_cuda_2rank_test.py` with a restart-safe two-phase mode and `--post-verify-exe`; CMake invokes it with `--exe $<TARGET_FILE:mgt_native_train>` and a verifier built from `test_p888_native_train_world2.cu`. Phase 1 launches the actual CLI as two ranks on two distinct device IDs, loads an accepted test policy, executes prepared BN training through full/final row shapes, delayed commit, and checkpoint publication; phase 2 relaunches that same production executable from the published checkpoint and proves resume. The companion executable only verifies immutable CLI/artifact/checkpoint/participation outputs and contains no training implementation. `MGT_REQUIRE_TWO_GPUS=1` turns launcher skip or single-device fallback into failure.

- [ ] Run a fresh CUDA-disabled build plus queued CUDA gates. First enumerate every expected GPU test name so one matching alternative cannot hide an unregistered critical test:

  ```powershell
Invoke-NativeChecked 'check generator fixture' { py scripts/generate_p888_generator_fixture.py --check }
Invoke-NativeChecked 'check checkpoint-v3 fixture' { py scripts/generate_p888_checkpoint_v3_fixture.py --check }
$cpuBuild = Join-Path ([IO.Path]::GetTempPath()) ("mgt-p888-cpu-only-" + [Guid]::NewGuid().ToString('N'))
Invoke-NativeChecked 'configure fresh CPU-only build' { cmake -S native -B $cpuBuild -DCMAKE_BUILD_TYPE=Release -DMGT_ENABLE_CUDA=OFF -DMGT_ENABLE_NCCL=OFF }
Invoke-NativeChecked 'build fresh CPU-only tree' { cmake --build $cpuBuild --config Release --parallel }
Invoke-NativeChecked 'run CPU-only Task-15 gates' { ctest --test-dir $cpuBuild -C Release -R "^(epoch_schedule|p888_sample_generator|training_artifacts|training_telemetry|checkpoint_consensus|resume_contract|weight_export|batch_norm_training_plan)$" --output-on-failure --no-tests=error }
$gpuGate = @"
set -euo pipefail
cmake --build native/build --config Release --parallel
listed=`$(ctest --test-dir native/build -C Release -N)
for name in p888_sample_generator_cuda checkpoint_snapshot checkpoint_consensus_2rank training_telemetry_cuda training_health_2rank mlp_batch_norm_full_backward weight_export_cuda p888_native_train_world2; do
  grep -Eq "Test +#[0-9]+: `${name}$" <<<"`$listed"
done
rm -f /tmp/task15-gate.xml
MGT_REQUIRE_TWO_GPUS=1 ctest --test-dir native/build -C Release -R '^(p888_sample_generator_cuda|checkpoint_snapshot|checkpoint_consensus_2rank|training_telemetry_cuda|training_health_2rank|mlp_batch_norm_full_backward|weight_export_cuda|p888_native_train_world2)$' --output-on-failure --no-tests=error --output-junit /tmp/task15-gate.xml
python3 -c 'import xml.etree.ElementTree as E; r=E.parse("/tmp/task15-gate.xml").getroot(); c=r.findall(".//testcase"); assert len(c)==8, len(c); assert not any(x.find("skipped") is not None or x.find("failure") is not None or x.find("error") is not None for x in c)'
"@
Invoke-NativeChecked 'queued Task-15 production trainer gates' { docker exec mgt-gpu-queue python3 scripts/gpu_queue_submit.py --label task15-production-trainer-gates --wait -- bash -lc $gpuGate }
Invoke-NativeChecked 'check runtime source manifest' { py scripts/runtime_source_manifest.py --repo-root . --check }
Invoke-NativeChecked 'check Task-15 diff' { git diff --check }
Invoke-NativeChecked 'inspect Task-15 status' { git status --short }
  ```

- [ ] Commit only the implementation payload:

  ```powershell
  git add native/tools/mgt_native_train_smoke.cu native/include/mgt/train_plan.hpp native/src/train_plan.cpp native/include/mgt/batch_norm_training.hpp native/src/batch_norm_training.cpp native/tests/test_batch_norm_training_plan.cpp native/include/mgt/epoch_schedule.hpp native/src/epoch_schedule.cpp native/tests/test_epoch_schedule.cpp native/include/mgt/p888_sample_generator.hpp native/src/p888_sample_generator.cpp native/tests/test_p888_sample_generator.cpp native/tests/fixtures/p888_generator_v1.json scripts/generate_p888_generator_fixture.py scripts/generate_p888_checkpoint_v3_fixture.py native/tests/fixtures/p888_checkpoint_v3_golden.json scripts/run_cuda_2rank_test.py native/cuda/mgt_cuda/p888_sample_generator.cuh native/cuda/p888_sample_generator.cu native/tests/cuda/test_p888_sample_generator_cuda.cu native/include/mgt/training_artifacts.hpp native/src/training_artifacts.cpp native/tests/test_training_artifacts.cpp native/include/mgt/checkpoint_consensus.hpp native/src/checkpoint_consensus.cpp native/tests/test_checkpoint_consensus.cpp native/tests/cuda/test_checkpoint_consensus_2rank.cu native/cuda/mgt_cuda/checkpoint_snapshot.cuh native/cuda/checkpoint_snapshot.cu native/tests/cuda/test_checkpoint_snapshot.cu native/tests/cuda/test_weight_export_cuda.cu native/tests/cuda/test_p888_native_train_world2.cu native/include/mgt/training_telemetry.hpp native/src/training_telemetry.cpp native/tests/test_training_telemetry.cpp native/tests/cuda/test_training_telemetry.cu native/tests/cuda/test_training_health_2rank.cu native/cuda/mgt_cuda/training_telemetry.cuh native/cuda/training_telemetry.cu native/cuda/mgt_cuda/mlp_batch_norm_forward.cuh native/cuda/mlp_batch_norm_forward.cu native/cuda/sync_batch_norm.cuh native/cuda/sync_batch_norm.cu native/cuda/mgt_cuda/linear_train_ops.cuh native/cuda/mlp_backward.cu native/cuda/mgt_cuda/adamw.cuh native/cuda/adamw.cu native/cuda/mgt_cuda/allreduce_nccl.cuh native/cuda/allreduce_nccl.cu native/include/mgt/weight_export.hpp native/src/weight_export.cpp native/tests/test_weight_export.cpp native/CMakeLists.txt
  git commit -m "feat: train p888 with resumable production sync batchnorm"
  ```
---

## Task 16: Freeze the Runtime, Validate A100 Diagnostics, and Select the Authoritative T4 Policy

**Setup-commit files:**

- Modify: `scripts/cluster/run_p888_bn_a100_release.sbatch`
- Modify: `scripts/package_kaggle_gitclone_kernel.py`
- Create: `scripts/tests/test_package_kaggle_gitclone_kernel.py`
- Create: `scripts/p888_external_controller.py`
- Create: `scripts/tests/test_p888_external_controller.py`
- Create: `scripts/p888_resume_bundle.py`
- Create: `scripts/prepare_p888_resume_dataset.py`
- Create: `scripts/verify_p888_resume_bundle.py`
- Create: `scripts/tests/test_p888_resume_bundle.py`
- Create: `kaggle/kernel/run_p888_bn_validation_2xt4.sh`
- Create: `kaggle/kernel/run_p888_full_32692_2xt4.sh`
- Create: `kaggle/kernel/run_puzzle0_model_compare_2xt4.sh`
- Create: `kaggle/resume_dataset/dataset-metadata.json`

**Evidence files created only after external runs:**

- Create: `test_results/p888_bn_a100_validation.md`
- Create: `test_results/p888_bn_2xt4_validation.md`
- Create: `test_results/p888_original_puzzle0_preflight.md`
- Create once: `test_results/p888_runtime_setup_locks/<runtime_source_sha>.json`
- Create/update active pointer: `test_results/p888_runtime_setup_lock.json`
- Create once: `test_results/p888_lineages/<lineage_id>.json`
- Create/update active pointer: `test_results/p888_active_lineage.json`
- Create immutable snapshot: `test_results/p888_controller_operations_task16.jsonl`

- [ ] Put every external mutation behind one exact-commit controller. `p888_external_controller.py` must use `subprocess.run(argv, shell=False, check=True, text=True)` for every `git`, `kaggle`, Python, and verifier child; it may retry read-only network calls with bounded exponential backoff, but never converts a failed push, verifier, create, or version command into success. Expose these subcommands:

  ```text
  assert-runtime
  run-kernel
  resume-kernel
  open-or-bootstrap-lineage
  promote-active-selection
  train-lineage
  run-final-comparison
  verify-a100-summary
  ingest-a100-summary
  render-validation-reports
  render-training-reports
  ```

  Every subcommand first proves the checkout from which the controller script itself executes has the locked source SHA, Git-tree SHA, and runtime-tree SHA256, is detached, has empty tracked/index/untracked runtime state, and contains the locked controller file SHA256. Before each `kernels push`, `datasets create`, or `datasets version`, rerun the same proof. The sole target-checkout exception is `assert-runtime --repo-root <main> --allow-evidence-only-commits`: the controller still executes from the verified detached checkout, but may inspect an attached main checkout only when its HEAD descends from the locked setup SHA, every intervening commit changes only the exact reviewed evidence allowlists, its index is empty, its runtime pathspecs have no tracked or untracked changes, and its recomputed runtime-tree SHA256 equals the lock. Without that flag, an attached target is rejected. Tests cover non-descendant HEAD, a mixed runtime/evidence commit, dirty index, dirty/untracked runtime path, and unchanged evidence-only descendants. The controller writes canonical, fsync-and-rename operation ledgers after every state transition. No later command is copied from mutable `main`; all controller invocations execute the script from the detached setup checkout created below.

- [ ] Make native-command failure explicit in every displayed PowerShell block. Define these helpers once at Task-16 execution start and use them for every native process, including tests and Git:

  ```powershell
  Set-StrictMode -Version Latest
  $ErrorActionPreference = 'Stop'
  function Invoke-NativeChecked([string]$Label, [scriptblock]$Command) {
      & $Command
      $code = $LASTEXITCODE
      if ($code -ne 0) { throw "$Label failed with exit code $code" }
  }
  function Get-NativeChecked([string]$Label, [scriptblock]$Command) {
      $lines = @(& $Command)
      $code = $LASTEXITCODE
      if ($code -ne 0) { throw "$Label failed with exit code $code" }
      return $lines
  }
  function Assert-StagedExactly([string[]]$Expected) {
      $actual = @(Get-NativeChecked 'read staged paths' { git diff --cached --name-only })
      $delta = @(Compare-Object ($Expected | Sort-Object) ($actual | Sort-Object))
      if ($delta.Count -ne 0) { throw "unexpected staged payload: $($delta | Out-String)" }
      Invoke-NativeChecked 'cached diff check' { git diff --cached --check }
  }
  $script:ProtectedBackupPath = 'kaggle/kernel/run_ranks_2xt4.sh~31640'
  function Assert-ProtectedBackupUnchanged {
      if (-not (Test-Path -LiteralPath $script:ProtectedBackupPath -PathType Leaf)) { throw 'protected backup is missing or is not a leaf' }
      $info = Get-Item -LiteralPath $script:ProtectedBackupPath
      if ($info.Length -ne 5258 -or $info.LastWriteTimeUtc.Ticks -ne 639188887422510152) { throw 'protected backup metadata differs from plan baseline; do not inspect contents' }
  }
  Assert-ProtectedBackupUnchanged
  ```

  PowerShell 5.1 does not stop on a native nonzero exit merely because `$ErrorActionPreference='Stop'`; the wrappers are mandatory. Unit tests execute the documented command paths with fake children and prove a failed verifier cannot reach dataset mutation, a failed push cannot enter polling, and a failed Git push cannot produce a setup lock.

- [ ] Extend `package_kaggle_gitclone_kernel.py` with mandatory immutable submission identity:

  ```text
  --submission-nonce <32 lowercase hex>
  --mode validation|original_preflight|training|final_compare
  --expected-input-bundle-sha256 none|<64 lowercase hex>
  --expected-parent-session-index <integer|-1>
  --runtime-source-sha <40 lowercase hex>
  --runtime-git-tree-sha <40 lowercase hex>
  --runtime-tree-sha256 <64 lowercase hex>
  ```

  The controller generates the nonce with `secrets.token_hex(16)` and creates its submission ledger with `O_CREAT|O_EXCL` before push. The generated runner bakes every identity field as a Python literal, rejects conflicting environment values, and writes `/kaggle/working/submission_identity.json` as its first action, before Git/network/build work. That canonical file contains schema, nonce, mode, kernel ID, output subdirectory, expected input/session, all runtime hashes, package-manifest SHA256, and state `started`. The final runner manifest repeats every field and state `complete` or `error`. The package fetches the exact commit object, checks out detached `FETCH_HEAD`, and requires:

  ```text
  rev-parse HEAD              == runtime_source_sha
  rev-parse HEAD^{tree}       == runtime_git_tree_sha
  runtime_source_manifest.py  == runtime_tree_sha256
  status --porcelain=v1 --untracked-files=all is empty
  git diff --quiet and git diff --cached --quiet both succeed
  ```

  Build/dependency trees live under `/tmp`; only compact artifacts go under `/kaggle/working`. Repeat the clean/runtime proof after the entry script. Tests inspect generated AST/text and run a fake repository for success plus wrong nonce, commit, tree, runtime hash, dirty tree, and conflicting-environment failures.

- [ ] Make shared-kernel polling identity-safe and restart-idempotent. Under an operation-specific exclusive lock, `run-kernel` is create-or-resume: when no ledger exists it rejects a queued/running prior version, creates the nonce ledger, packages, validates, and checks the exit code of exactly one `kaggle kernels push`; when the ledger already exists it requires every supplied immutable argument to equal the write-once ledger byte-for-byte and resumes from the recorded state without packaging or pushing again. A mismatched retry fails closed. It polls every 30 seconds with a configured hard timeout. A terminal status by itself is never accepted because it may belong to the previous shared-ID version. On each terminal observation, download into a fresh attempt directory and read `submission_identity.json`:

  - missing or different nonce means stale output; preserve its identity in the polling ledger, wait, and continue;
  - matching nonce plus `ERROR` preserves all available output/status and fails;
  - matching nonce plus `COMPLETE` still requires the mode-specific final manifest and verifier before atomically promoting that fresh attempt directory to the operation output directory;
  - a target output directory that already exists is never merged or overwritten;
  - timeout without a matching identity is `unresolved_submission`, never success.

  `resume-kernel` accepts only the common locked-runtime arguments plus `--operation`; it loads every kernel/mode/entry/output/parent/environment value from the write-once submission ledger, revalidates the package and ledger digests, and continues from the recorded state without another push. These are the literal interruption-recovery commands after recreating the candidate context; rerunning the original `run-kernel` command with byte-identical arguments is equivalent:

  ```powershell
  Invoke-NativeChecked 'resume Task-16 validation polling' { py $controller resume-kernel @common --operation task16-validation }
  Invoke-NativeChecked 'resume Task-16 original preflight polling' { py $controller resume-kernel @common --operation task16-original-preflight }
  ```

  Use only the command for the interrupted operation. If its ledger is absent, `resume-kernel` fails and never creates or pushes anything. Tests simulate old `COMPLETE`, old `ERROR`, status lag after push, matching success, matching failure, corrupt identity, download failure, timeout, process restart, mismatched create-or-resume arguments, missing-ledger resume, and prove both recovery forms never create a second version.

- [ ] Make the three checked-in shell runners fail closed. All use `set -euo pipefail`, require exactly two visible Tesla T4 devices on Kaggle, require SM75 Release binaries, set `MGT_REQUIRE_TWO_GPUS=1`, accept only baked submission identity, use only `--policy-json`, preserve both NCCL rendezvous identities for two-range mode, and emit one final machine-readable manifest. Validation alone may set `MGT_RETUNE=allow` and must also set `MGT_TASK16_ACCEPTANCE=1`; full/resume and puzzle comparison hardcode `MGT_RETUNE=never`. No runner sources `selected.env` or reconstructs policy from legacy `MGT_*` tuning flags. Rank 0 prints the required per-step line without buffering; every script traps failure and updates the matching-nonce final manifest before exiting nonzero.

- [ ] Define the private resume dataset and deterministic bundle exactly:

  ```json
  {
    "title": "__MGT_DATASET_TITLE__",
    "id": "__MGT_DATASET_ID__",
    "licenses": [{"name": "other"}]
  }
  ```

  The checked-in file is a render template, not a publishable metadata file. The controller requires both sentinels exactly once, substitutes canonical values in memory, and rejects any remaining `__MGT_` token. Dataset base ID is `trydotatwo/p888-syncbn-training-resume`; the actual immutable lineage ID is `trydotatwo/p888-syncbn-training-resume-<first16(lineage_id)>`. The filename is singular `dataset-metadata.json`; `kaggle datasets create` is called without `-u`, so it stays private. A staging directory is flat and contains only `dataset-metadata.json`, `resume_bundle.zip`, `resume_bundle.zip.sha256`, and `publish_manifest.json`; use `-r skip`, and forbid `datasets version -d`. Build `resume_bundle.zip` with standard-library `zipfile`, deterministic `ZIP_STORED` entries, sorted canonical JSON, UTF-8, compact separators, one terminal LF, fixed timestamps/permissions, and no host paths.

  ZIP contents are `bundle_manifest.json` plus sidecar, exact accepted policy plus sidecar, training contract, original preflight, integrity-chain metadata, and either no checkpoint for `genesis`, one complete schema-v3 checkpoint for `paused`, or checkpoint plus final folded/export/training artifacts for `complete`. Reject duplicate/absolute/backslash/`..`/symlink-like entries, unknown or missing files, size/hash mismatch, malformed JSON, wrong policy/source/runtime/contract/preflight, cursor rollback, lineage branching, parent mismatch, paused without checkpoint, and complete without exactly 32,692 epochs, 326,920 steps, and final export.

  `publish_manifest.json` records schema, actual dataset ID, lineage ID/generation nonce, session/parent indices, input/output dataset versions, parent/output bundle SHA256, raw ZIP SHA256/bytes, state, all runtime hashes, contract/policy/preflight hashes, checkpoint SHA256, global step, semantic epoch, next batch/sample, integrity chain, `writer_nonce`, and originating `submission_nonce`. Before genesis the controller creates a 32-hex `lineage_generation_nonce` with `O_EXCL`; `lineage_id` is SHA256 of domain `MGTP8LIN1` plus that nonce, runtime hashes, contract, accepted policy, input manifest, original preflight, and seed. The generation nonce changes identity only, never training RNG. Each dataset version contains only the newest checkpoint; older versions retain recovery history.

- [ ] Implement a real single-writer and compare-and-swap protocol in the controller. Acquire `test_results/p888_controller_state/active-writer.lock` with `os.open(lock_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)` and store operation ID, writer nonce, PID, host, lineage, phase, and ledger path. Never auto-delete or auto-steal a surviving lock; after interruption, resume the recorded operation and reconcile remote state. On clean completion atomically move the lock into immutable history.

  Immediately before `datasets create/version`, download and fully verify the latest remote parent into a new directory. For an append, require its version number, raw ZIP SHA256, `publish_manifest.json` bytes, lineage, and bundle SHA256 to equal the recorded parent. After upload, require remote version exactly `input+1`, download it fresh, and require exact published ZIP/bundle/lineage/writer/submission/runtime fields. Kaggle CLI has no server-side conditional update: the local lock prevents authorized in-workspace races, while post-upload checks detect an external race. If a stale child was published as another version, mark that dataset reference tainted in an immutable conflict record and never consume its latest state. Recovery creates a new generation nonce, lineage record, and suffixed private dataset from genesis; the tainted dataset and every old setup/lineage record remain untouched. Tests cover lock/restart, parent change, concurrent increment, taint/new-slug recovery, upload-timeout exact-child reconciliation, and roundtrip mismatch.

- [ ] Fully define existing/new lineage selection. `open-or-bootstrap-lineage --dataset-base-id` creates or resumes one immutable `test_results/p888_lineages/<lineage_id>.json` containing setup-lock digest, actual suffixed dataset ID, generation nonce, validation/preflight operations, and genesis bundle identity; mutable current session/state lives only in verified remote versions and ignored controller ledgers. It distinguishes absence only from explicit HTTP 404/not-found plus a successful authenticated exact-ID listing; every other error blocks. If absent, create privately and exact-roundtrip version 1. If present, require ready and fully verify the exact expected lineage/runtime/policy/contract/preflight/cursor/ZIP/bundle. A matching genesis/paused/complete lineage resumes; an unrelated or tainted suffixed ID causes a new generation nonce/slug, never overwrite/delete/blind append. At this stage write only an ignored candidate-lineage selector under `p888_controller_state`; do not modify either tracked active pointer. There is no hard-coded `v001`.

- [ ] Run all local gates, then create and push one exact setup commit before any A100 job, Kaggle kernel run, or dataset create/version action. Task 1's checked read-only original-model acquisition is the sole earlier exception:

  ```powershell
  $setupFiles = @(
    'scripts/cluster/run_p888_bn_a100_release.sbatch',
    'scripts/package_kaggle_gitclone_kernel.py',
    'scripts/tests/test_package_kaggle_gitclone_kernel.py',
    'scripts/p888_external_controller.py',
    'scripts/tests/test_p888_external_controller.py',
    'scripts/p888_resume_bundle.py',
    'scripts/prepare_p888_resume_dataset.py',
    'scripts/verify_p888_resume_bundle.py',
    'scripts/tests/test_p888_resume_bundle.py',
    'kaggle/kernel/run_p888_bn_validation_2xt4.sh',
    'kaggle/kernel/run_p888_full_32692_2xt4.sh',
    'kaggle/kernel/run_puzzle0_model_compare_2xt4.sh',
    'kaggle/resume_dataset/dataset-metadata.json'
  )
  $preStaged = @(Get-NativeChecked 'read existing index before tests' { git diff --cached --name-only })
  if ($preStaged.Count -ne 0) { throw "index must be empty before Task 16: $preStaged" }
  $dirtyTracked = @(Get-NativeChecked 'read tracked working changes' { git diff --name-only })
  $dirtyUntracked = @(Get-NativeChecked 'read untracked working files' { git ls-files --others --exclude-standard })
  $protectedBackup = $script:ProtectedBackupPath
  Assert-ProtectedBackupUnchanged
  $actualSetupDirty = @($dirtyTracked + $dirtyUntracked | Where-Object { $_ -ne $protectedBackup } | Sort-Object -Unique)
  $dirtyDelta = @(Compare-Object ($setupFiles | Sort-Object) $actualSetupDirty)
  if ($dirtyDelta.Count -ne 0) { throw "Task-16 working payload differs from setup allowlist: $($dirtyDelta | Out-String)" }
  Invoke-NativeChecked 'queued full local gate' { docker exec mgt-gpu-queue python3 scripts/gpu_queue_submit.py --label task16-full-local-gate --wait -- bash -lc "cmake --build native/build --config Release --parallel && ctest --test-dir native/build -C Release --output-on-failure --no-tests=error" }
  Invoke-NativeChecked 'controller unit tests' { py -m unittest scripts.tests.test_package_kaggle_gitclone_kernel scripts.tests.test_p888_external_controller scripts.tests.test_p888_resume_bundle scripts.tests.test_autotune_2xt4 -v }
  Invoke-NativeChecked 'A100 script syntax' { bash -n scripts/cluster/run_p888_bn_a100_release.sbatch }
  Invoke-NativeChecked 'validation script syntax' { bash -n kaggle/kernel/run_p888_bn_validation_2xt4.sh }
  Invoke-NativeChecked 'training script syntax' { bash -n kaggle/kernel/run_p888_full_32692_2xt4.sh }
  Invoke-NativeChecked 'comparison script syntax' { bash -n kaggle/kernel/run_puzzle0_model_compare_2xt4.sh }
  Invoke-NativeChecked 'working diff check' { git diff --check }
  $dirtyTracked = @(Get-NativeChecked 'recheck tracked working changes' { git diff --name-only })
  $dirtyUntracked = @(Get-NativeChecked 'recheck untracked working files' { git ls-files --others --exclude-standard })
  Assert-ProtectedBackupUnchanged

  $actualSetupDirty = @($dirtyTracked + $dirtyUntracked | Where-Object { $_ -ne $protectedBackup } | Sort-Object -Unique)
  $dirtyDelta = @(Compare-Object ($setupFiles | Sort-Object) $actualSetupDirty)
  if ($dirtyDelta.Count -ne 0) { throw "tests changed Task-16 working payload: $($dirtyDelta | Out-String)" }
  Invoke-NativeChecked 'stage exact setup files' { git add -- $setupFiles }
  Assert-StagedExactly $setupFiles
  Invoke-NativeChecked 'commit setup' { git commit -m 'build: prepare reproducible p888 validation and resume' }
  Invoke-NativeChecked 'push setup' { git push origin HEAD:codex-native-trainer-implementation }
  Assert-ProtectedBackupUnchanged
  ```

  The protected `kaggle/kernel/run_ranks_2xt4.sh~31640` backup is never staged, removed, or inspected as runtime source. Any unexpected staged path aborts before commit.

- [ ] Derive hashes from a clean detached controller clone and create the setup lock once. Do not compute the runtime hash from the mutable main checkout:

  ```powershell
  $mainRoot = (Get-Location).Path
  $runtimeSourceSha = ((Get-NativeChecked 'read setup source SHA' { git rev-parse HEAD })[0]).Trim()
  $runtimeGitTreeSha = ((Get-NativeChecked 'read setup tree SHA' { git rev-parse "${runtimeSourceSha}^{tree}" })[0]).Trim()
  if ($runtimeSourceSha -notmatch '^[0-9a-f]{40}$' -or $runtimeGitTreeSha -notmatch '^[0-9a-f]{40}$') { throw 'invalid setup Git hashes' }
  $controllerRoot = Join-Path $mainRoot "test_results/p888_controller_checkouts/$runtimeSourceSha"
  New-Item -ItemType Directory -Force -Path (Split-Path -Parent $controllerRoot) | Out-Null
  if (-not (Test-Path -LiteralPath $controllerRoot)) {
      Invoke-NativeChecked 'clone detached controller' { git clone --no-local https://github.com/TryDotAtwo/MultiGPUTrainMLP.git $controllerRoot }
  }
  Invoke-NativeChecked 'fetch exact setup object' { git -C $controllerRoot fetch origin codex-native-trainer-implementation }
  Invoke-NativeChecked 'checkout exact setup object' { git -C $controllerRoot checkout --detach $runtimeSourceSha }
  $runtimeTreeSha256 = ((Get-NativeChecked 'hash detached runtime tree' { py "$controllerRoot/scripts/runtime_source_manifest.py" --repo-root $controllerRoot --print-sha })[0]).Trim()
  $mainRuntimeTreeSha256 = ((Get-NativeChecked 'hash committed main runtime tree' { py "$controllerRoot/scripts/runtime_source_manifest.py" --repo-root $mainRoot --print-sha })[0]).Trim()
  if ($mainRuntimeTreeSha256 -ne $runtimeTreeSha256) { throw 'tested main runtime differs from detached setup commit' }
  $controllerSha256 = (Get-FileHash -LiteralPath "$controllerRoot/scripts/p888_external_controller.py" -Algorithm SHA256).Hash.ToLowerInvariant()
  if ($runtimeTreeSha256 -notmatch '^[0-9a-f]{64}$' -or $controllerSha256 -notmatch '^[0-9a-f]{64}$') { throw 'invalid detached hashes' }
  Invoke-NativeChecked 'prove detached controller runtime' { py "$controllerRoot/scripts/p888_external_controller.py" assert-runtime --repo-root $controllerRoot --runtime-source-sha $runtimeSourceSha --runtime-git-tree-sha $runtimeGitTreeSha --runtime-tree-sha256 $runtimeTreeSha256 --controller-sha256 $controllerSha256 }
  $remoteHead = ((Get-NativeChecked 'read remote setup head' { git ls-remote origin refs/heads/codex-native-trainer-implementation })[0] -split '\s+')[0]
  if ($remoteHead -ne $runtimeSourceSha) { throw 'setup commit was not the pushed branch head at freeze time' }
  $lock = [ordered]@{schema=1; branch='codex-native-trainer-implementation'; repository='https://github.com/TryDotAtwo/MultiGPUTrainMLP.git'; runtime_source_sha=$runtimeSourceSha; runtime_git_tree_sha=$runtimeGitTreeSha; runtime_tree_sha256=$runtimeTreeSha256; controller_sha256=$controllerSha256}
  $lockJson = ($lock | ConvertTo-Json -Compress) + "`n"
  $versionedLockRel = "test_results/p888_runtime_setup_locks/$runtimeSourceSha.json"
  $versionedLockPath = Join-Path $mainRoot $versionedLockRel
  New-Item -ItemType Directory -Force -Path (Split-Path -Parent $versionedLockPath) | Out-Null
  if (Test-Path -LiteralPath $versionedLockPath) {
      if ([IO.File]::ReadAllText($versionedLockPath) -cne $lockJson) { throw 'versioned setup lock is immutable and mismatched' }
  } else {
      $stream = [IO.File]::Open($versionedLockPath,[IO.FileMode]::CreateNew,[IO.FileAccess]::Write,[IO.FileShare]::None)
      try { $bytes=[Text.UTF8Encoding]::new($false).GetBytes($lockJson); $stream.Write($bytes,0,$bytes.Length); $stream.Flush($true) } finally { $stream.Dispose() }
  }
  $versionedLockSha = (Get-FileHash -LiteralPath $versionedLockPath -Algorithm SHA256).Hash.ToLowerInvariant()
  $candidateSetup = [ordered]@{schema=1; setup_lock_path=$versionedLockRel; setup_lock_sha256=$versionedLockSha}
  $candidateSetupJson = ($candidateSetup | ConvertTo-Json -Compress) + "`n"
  $candidateSetupPath = Join-Path $mainRoot 'test_results/p888_controller_state/setup-candidate.json'
  New-Item -ItemType Directory -Force -Path (Split-Path -Parent $candidateSetupPath) | Out-Null
  $candidateSetupTmp = "$candidateSetupPath.tmp-$PID"
  $candidateStream = [IO.File]::Open($candidateSetupTmp,[IO.FileMode]::CreateNew,[IO.FileAccess]::Write,[IO.FileShare]::None)
  try { $candidateBytes=[Text.UTF8Encoding]::new($false).GetBytes($candidateSetupJson); $candidateStream.Write($candidateBytes,0,$candidateBytes.Length); $candidateStream.Flush($true) } finally { $candidateStream.Dispose() }
  Move-Item -LiteralPath $candidateSetupTmp -Destination $candidateSetupPath -Force
  ```

  On reuse, validate candidate-selector schema/digest, versioned-lock schema and hex lengths, `rev-parse <source>^{tree}`, controller SHA, detached cleanliness, and remote reachability/ancestry. Never rewrite a versioned lock. A new setup creates a new SHA-named lock and changes only the ignored candidate selector; the previously promoted active setup/lineage remains authoritative until every new gate and genesis roundtrip succeeds.

  The A100 wait may outlive the current local shell. After any shell reset, first paste the Task-16 checked-native/backup helper block, then run this complete candidate-only initialization before ingest, T4 validation, preflight, bootstrap, promotion, or report rendering. It deliberately reads neither active pointer nor lineage:

  ```powershell
  Assert-ProtectedBackupUnchanged
  $mainRoot = (Get-Location).Path
  $candidateSetupPath = Join-Path $mainRoot 'test_results/p888_controller_state/setup-candidate.json'
  if (-not (Test-Path -LiteralPath $candidateSetupPath -PathType Leaf)) { throw 'candidate setup pointer is missing' }
  $candidateSetup = Get-Content $candidateSetupPath -Raw | ConvertFrom-Json
  if ([int]$candidateSetup.schema -ne 1 -or [string]$candidateSetup.setup_lock_path -notmatch '^test_results/p888_runtime_setup_locks/[0-9a-f]{40}\.json$' -or [string]$candidateSetup.setup_lock_sha256 -notmatch '^[0-9a-f]{64}$') { throw 'malformed candidate setup pointer' }
  $versionedLockRel = [string]$candidateSetup.setup_lock_path
  $versionedLockPath = Join-Path $mainRoot $versionedLockRel
  if (-not (Test-Path -LiteralPath $versionedLockPath -PathType Leaf)) { throw 'candidate versioned lock is missing' }
  $versionedLockSha = (Get-FileHash -LiteralPath $versionedLockPath -Algorithm SHA256).Hash.ToLowerInvariant()
  if ($versionedLockSha -ne [string]$candidateSetup.setup_lock_sha256) { throw 'candidate versioned-lock digest mismatch' }
  $runtimeLock = Get-Content $versionedLockPath -Raw | ConvertFrom-Json
  $runtimeSourceSha = [string]$runtimeLock.runtime_source_sha
  $runtimeGitTreeSha = [string]$runtimeLock.runtime_git_tree_sha
  $runtimeTreeSha256 = [string]$runtimeLock.runtime_tree_sha256
  $controllerSha256 = [string]$runtimeLock.controller_sha256
  if ([int]$runtimeLock.schema -ne 1 -or [string]$runtimeLock.branch -ne 'codex-native-trainer-implementation' -or [string]$runtimeLock.repository -ne 'https://github.com/TryDotAtwo/MultiGPUTrainMLP.git' -or $runtimeSourceSha -notmatch '^[0-9a-f]{40}$' -or $runtimeGitTreeSha -notmatch '^[0-9a-f]{40}$' -or $runtimeTreeSha256 -notmatch '^[0-9a-f]{64}$' -or $controllerSha256 -notmatch '^[0-9a-f]{64}$') { throw 'malformed candidate versioned lock' }
  if ([IO.Path]::GetFileNameWithoutExtension($versionedLockPath) -cne $runtimeSourceSha) { throw 'candidate lock filename/source mismatch' }
  $derivedTree = ((Get-NativeChecked 'derive candidate Git tree' { git rev-parse "${runtimeSourceSha}^{tree}" })[0]).Trim()
  if ($derivedTree -ne $runtimeGitTreeSha) { throw 'candidate source/tree mismatch' }
  $controllerRoot = Join-Path $mainRoot "test_results/p888_controller_checkouts/$runtimeSourceSha"
  New-Item -ItemType Directory -Force -Path (Split-Path -Parent $controllerRoot) | Out-Null
  if (-not (Test-Path -LiteralPath $controllerRoot)) { Invoke-NativeChecked 'restore candidate detached clone' { git clone --no-local ([string]$runtimeLock.repository) $controllerRoot } }
  Invoke-NativeChecked 'fetch candidate setup commit' { git -C $controllerRoot fetch origin codex-native-trainer-implementation }
  Invoke-NativeChecked 'checkout candidate setup commit' { git -C $controllerRoot checkout --detach $runtimeSourceSha }
  $actualControllerSha = (Get-FileHash -LiteralPath "$controllerRoot/scripts/p888_external_controller.py" -Algorithm SHA256).Hash.ToLowerInvariant()
  if ($actualControllerSha -ne $controllerSha256) { throw 'candidate controller bytes mismatch' }
  $detachedRuntimeTreeSha = ((Get-NativeChecked 'hash candidate detached runtime' { py "$controllerRoot/scripts/runtime_source_manifest.py" --repo-root $controllerRoot --print-sha })[0]).Trim()
  $mainRuntimeTreeSha = ((Get-NativeChecked 'hash candidate main runtime' { py "$controllerRoot/scripts/runtime_source_manifest.py" --repo-root $mainRoot --print-sha })[0]).Trim()
  if ($detachedRuntimeTreeSha -ne $runtimeTreeSha256 -or $mainRuntimeTreeSha -ne $runtimeTreeSha256) { throw 'candidate runtime-tree mismatch' }
  $remoteHead = ((Get-NativeChecked 'read candidate remote head' { git ls-remote origin refs/heads/codex-native-trainer-implementation })[0] -split '\s+')[0]
  if ($remoteHead -ne $runtimeSourceSha) { throw 'candidate setup is no longer the remote branch head' }
  $controller = "$controllerRoot/scripts/p888_external_controller.py"
  $common = @('--repo-root',$controllerRoot,'--artifacts-root',(Join-Path $mainRoot 'test_results'),'--runtime-source-sha',$runtimeSourceSha,'--runtime-git-tree-sha',$runtimeGitTreeSha,'--runtime-tree-sha256',$runtimeTreeSha256,'--controller-sha256',$controllerSha256,'--setup-lock-path',$versionedLockPath,'--setup-lock-sha256',$versionedLockSha)
  Invoke-NativeChecked 'prove candidate context' { py $controller assert-runtime @common --candidate-setup-pointer $candidateSetupPath }
  Assert-ProtectedBackupUnchanged
  ```

  `assert-runtime --candidate-setup-pointer` strictly parses that selector, rejects duplicate/unknown fields, and proves it names the same immutable lock already present in `$common`. Never adapt the Task-17 active-lineage block for this pre-promotion phase and never rely on remembered PowerShell variables.

- [ ] Run A100 diagnostics only after exact hashes are checked and before `sbatch`. The implementing agent substitutes the four literal values from the lock into the pasteable commands; the user enters them, while the agent only reads terminal output and never executes SSH:

  ```bash
  set -euo pipefail
  export MGT_RUNTIME_SOURCE_SHA='<40-hex from setup lock>'
  export MGT_RUNTIME_GIT_TREE_SHA='<40-hex from setup lock>'
  export MGT_RUNTIME_TREE_SHA256='<64-hex from setup lock>'
  export MGT_CONTROLLER_SHA256='<64-hex from setup lock>'
  cd /mnt/pool/6/vokirova/p888-a100-smoke
  git -C source fetch origin codex-native-trainer-implementation
  git -C source checkout --detach "$MGT_RUNTIME_SOURCE_SHA"
  test "$(git -C source rev-parse HEAD)" = "$MGT_RUNTIME_SOURCE_SHA"
  test "$(git -C source rev-parse 'HEAD^{tree}')" = "$MGT_RUNTIME_GIT_TREE_SHA"
  test "$(python3 source/scripts/runtime_source_manifest.py --repo-root source --print-sha)" = "$MGT_RUNTIME_TREE_SHA256"
  test -z "$(git -C source status --porcelain=v1 --untracked-files=all)"
  job_line=$(sbatch -p kaf12 --export=ALL,MGT_EXPECTED_RUNTIME_SOURCE_SHA="$MGT_RUNTIME_SOURCE_SHA",MGT_EXPECTED_RUNTIME_GIT_TREE_SHA="$MGT_RUNTIME_GIT_TREE_SHA",MGT_EXPECTED_RUNTIME_TREE_SHA256="$MGT_RUNTIME_TREE_SHA256",MGT_EXPECTED_CONTROLLER_SHA256="$MGT_CONTROLLER_SHA256" source/scripts/cluster/run_p888_bn_a100_release.sbatch)
  jid=$(printf '%s\n' "$job_line" | awk '/Submitted batch job/{print $4}')
  case "$jid" in (''|*[!0-9]*) echo "invalid sbatch result: $job_line" >&2; exit 1;; esac
  printf 'JOB_ID=%s\n' "$jid"
  scontrol show job "$jid"
  while squeue -h -j "$jid" | grep -q .; do
    date -u '+WAIT_UTC=%Y-%m-%dT%H:%M:%SZ'
    squeue -h -j "$jid" -o 'STATE=%T ELAPSED=%M NODE=%N'
    sleep 30
  done
  record=''
  for attempt in $(seq 1 12); do
    record=$(sacct -n -X -j "$jid" --format=JobIDRaw,State,ExitCode -P | awk -F'|' -v id="$jid" '$1==id{print;exit}')
    test -n "$record" && break
    sleep 10
  done
  test -n "$record"
  state=$(printf '%s\n' "$record" | cut -d'|' -f2)
  exit_code=$(printf '%s\n' "$record" | cut -d'|' -f3)
  case "$state" in (COMPLETED*) ;; (*) echo "SLURM_STATE=$state" >&2; exit 1;; esac
  test "$exit_code" = '0:0'
  summary="results/${jid}/a100_summary.json"
  envelope="results/${jid}/a100_import_envelope.json"
  test -s "$summary"
  python3 source/scripts/p888_external_controller.py verify-a100-summary --repo-root source --summary "$summary" --job-id "$jid" --runtime-source-sha "$MGT_RUNTIME_SOURCE_SHA" --runtime-git-tree-sha "$MGT_RUNTIME_GIT_TREE_SHA" --runtime-tree-sha256 "$MGT_RUNTIME_TREE_SHA256" --controller-sha256 "$MGT_CONTROLLER_SHA256" --expected-world 8 --slurm-state "$state" --slurm-exit-code "$exit_code" --verified-envelope-out "$envelope"
  test -s "$envelope"
  test "$(wc -c < "$envelope")" -le 16384
  envelope_sha=$(sha256sum "$envelope" | awk '{print $1}')
  printf 'A100_IMPORT_ENVELOPE_SHA256=%s\nA100_IMPORT_ENVELOPE_BASE64_BEGIN\n' "$envelope_sha"
  python3 -c 'import base64,sys; print(base64.b64encode(open(sys.argv[1],"rb").read()).decode("ascii"))' "$envelope"
  printf 'A100_IMPORT_ENVELOPE_BASE64_END\n'
  ```

  The SLURM script independently rechecks all four literals, including controller bytes, before build, uses `#SBATCH --nodes=1`, `--ntasks-per-node=8`, `--cpus-per-task=4`, `--gres=gpu:8`, and `-p kaf12`, and never uses `--gpus-per-task` or `--gpu-bind`. It builds Release SM80, proves all eight ranks and full/final rows, and atomically writes `a100_summary.json` only after all required tests/profiles finish. That in-job summary contains job/source/tree/runtime hashes, build/hardware identity, diagnostics, and remote log/Nsight artifact paths plus SHA256; it cannot claim a terminal SLURM state before the job exits. After `sacct` proves `COMPLETED/0:0`, `verify-a100-summary` validates the summary and writes one canonical, at-most-16-KiB import envelope that binds the exact summary SHA256 to the observed SLURM state/exit code and locked hashes. A100 timing never selects a T4 default; Systems precedes Compute, and Compute profiles only the one Systems-proven dominant kernel.

  The implementing agent copies the printed job ID, envelope digest, and one-line Base64 from the visible terminal into this local block; neither the agent nor controller opens SSH. Keep the Base64 out of the child-process argument vector: write exactly one ASCII, LF-terminated line into the ignored staging file, then pass only its path:

  ```powershell
  $a100JobId = '<numeric job ID printed above>'
  $a100EnvelopeSha256 = '<64 lowercase hex printed above>'
  $a100EnvelopeBase64 = '<single Base64 line printed above>'
  if ($a100JobId -notmatch '^[0-9]+$' -or $a100EnvelopeSha256 -notmatch '^[0-9a-f]{64}$' -or $a100EnvelopeBase64 -notmatch '^[A-Za-z0-9+/]+={0,2}$') { throw 'malformed copied A100 envelope fields' }
  $a100ImportStage = Join-Path $mainRoot "test_results/a100_import_staging/$a100JobId"
  New-Item -ItemType Directory -Force -Path $a100ImportStage | Out-Null
  $a100Base64Path = Join-Path $a100ImportStage 'envelope.b64'
  [IO.File]::WriteAllText($a100Base64Path,$a100EnvelopeBase64 + "`n",[Text.ASCIIEncoding]::new())
  Invoke-NativeChecked 'ingest verified A100 summary' { py $controller ingest-a100-summary @common --job-id $a100JobId --expected-envelope-sha256 $a100EnvelopeSha256 --base64-file $a100Base64Path }
  ```

  Ingest requires exactly one LF-terminated Base64 line and rejects noncanonical Base64, a decoded envelope over 16 KiB, wrong envelope/summary digest, job/runtime/world/terminal state, or malformed JSON. It writes a fresh directory under `test_results/cluster_outputs/<job-id>` atomically and re-reads it. If that destination already exists, re-read every canonical byte: exact envelope/digest/job/runtime equality returns success with `reused=true`, while any mismatch fails without overwrite. Unit tests cover every rejection plus crash-after-rename/exact retry and a forged `COMPLETED` field without the remote envelope binding. Do not begin T4 validation until the local verified import exists.

- [ ] Run the authoritative T4 validation through the detached controller; this is the sole `retune allow` operation:

  ```powershell
  $controller = "$controllerRoot/scripts/p888_external_controller.py"
  $common = @('--repo-root',$controllerRoot,'--artifacts-root',(Join-Path $mainRoot 'test_results'),'--runtime-source-sha',$runtimeSourceSha,'--runtime-git-tree-sha',$runtimeGitTreeSha,'--runtime-tree-sha256',$runtimeTreeSha256,'--controller-sha256',$controllerSha256,'--setup-lock-path',$versionedLockPath,'--setup-lock-sha256',$versionedLockSha)
  Invoke-NativeChecked 'authoritative T4 validation' { py $controller run-kernel @common --operation task16-validation --kernel-id trydotatwo/native-multigpu-mlp-convergence-2xt4 --mode validation --entry kaggle/kernel/run_p888_bn_validation_2xt4.sh --output-subdir p888_bn_validation --expected-input-bundle-sha256 none --expected-parent-session-index -1 --verify-kind p888-validation --env MGT_RETUNE=allow --env MGT_TASK16_ACCEPTANCE=1 }
  ```

  The run executes strict and the finite Task-14 matrix on two T4s, full/final 300-sample timing with pair indices 0/1/2, 1,000-step health/NCCL stress, ten uninterrupted epochs, 3+resume+7 epochs, and folded-export parity. It emits policy bytes/SHA, all rejected rows, per-rank JSONL, max-rank summary, memory series/ledger, snapshot/build/source manifests, and checkpoint/export artifacts. The controller re-reads the accepted policy through Python and native validate-only loaders.

- [ ] Promotion uses the Task-14 gates exactly. If no optimized candidate clears 3% but measured strict FP32 passes every nonperformance gate—correctness, full/final/unequal-final rows, finite/NCCL stress, 14.5-GiB limit, checkpoint/resume, and folded export—publish strict with `acceptance_status=accepted` and `selection_reason=strict_fallback_no_candidate_cleared_margin`; mark only the 3% gate `not_applicable` and preserve the fastest rejected candidate/reasons. Task 17 may proceed. If strict fails a nonperformance gate, no accepted policy exists and all later work stops.

- [ ] Run the original-only puzzle-0 preflight as a later sequential version of the same kernel ID, again through nonce-verified control:

  ```powershell
  Invoke-NativeChecked 'original puzzle-zero preflight' { py $controller run-kernel @common --operation task16-original-preflight --kernel-id trydotatwo/native-multigpu-mlp-convergence-2xt4 --mode original_preflight --entry kaggle/kernel/run_puzzle0_model_compare_2xt4.sh --output-subdir p888_original_preflight --expected-input-bundle-sha256 none --expected-parent-session-index -1 --verify-kind original-preflight --dataset-source arabidopsisthalian/ihes-model-1778521793 --dataset-source arabidopsisthalian/model-ihes-1780290207-e40960 --env MGT_COMPARE_MODE=original_preflight }
  ```

  Verify each available Task-1 model by source path/SHA/tensor keys. Run puzzle 0, depth 100, beam 10,000,000, existing two-GPU beam search, identical scoring, and one console line per depth. At least one original must solve; otherwise quality baseline is undefined and full training stops. Record shortest solved-original length and preflight SHA256.

- [ ] Open or bootstrap the private lineage only after validation and preflight both succeed:

  ```powershell
  Invoke-NativeChecked 'open or bootstrap resume lineage' { py $controller open-or-bootstrap-lineage @common --dataset-base-id trydotatwo/p888-syncbn-training-resume --metadata-template "$controllerRoot/kaggle/resume_dataset/dataset-metadata.json" --validation-operation task16-validation --preflight-operation task16-original-preflight }
  Invoke-NativeChecked 'promote verified setup and lineage' { py $controller promote-active-selection @common --candidate-setup-pointer $candidateSetupPath --candidate-lineage-pointer (Join-Path $mainRoot 'test_results/p888_controller_state/lineage-candidate.json') --active-setup-out (Join-Path $mainRoot 'test_results/p888_runtime_setup_lock.json') --active-lineage-out (Join-Path $mainRoot 'test_results/p888_active_lineage.json') --validation-operation task16-validation --preflight-operation task16-original-preflight }
  ```

  The controller follows the explicit absence/existing branch, enforces writer lock/roundtrip, and creates the immutable lineage record plus ignored candidate selector. `promote-active-selection` then re-verifies A100 import, T4 acceptance, original preflight, setup lock, exact genesis/resume roundtrip, and both candidate selectors under one exclusive activation journal. It fsyncs and atomically replaces the two active pointers as one recoverable transaction: interruption either leaves the old pair authoritative or is deterministically completed on exact retry; a mixed pair is never accepted by any subcommand. Mutable transition ledgers stay ignored under `p888_controller_state/<operation>/`; report rendering creates the immutable Task-16 snapshot. Tests cover failure before/between/after pointer renames, exact retry, and old-pair preservation. No caller types or assumes a dataset version.

- [ ] Render compact reports from verified controller ledgers, assert main runtime equivalence through the exact detached controller, and commit only the evidence allowlist:

  ```powershell
  Invoke-NativeChecked 'render validation reports' { py $controller render-validation-reports @common --a100-import-root (Join-Path $mainRoot 'test_results/cluster_outputs') --validation-operation task16-validation --preflight-operation task16-original-preflight --operations-snapshot (Join-Path $mainRoot 'test_results/p888_controller_operations_task16.jsonl') }
  Invoke-NativeChecked 'prove main runtime still matches setup' { py $controller assert-runtime --repo-root $mainRoot --runtime-source-sha $runtimeSourceSha --runtime-git-tree-sha $runtimeGitTreeSha --runtime-tree-sha256 $runtimeTreeSha256 --controller-sha256 $controllerSha256 --allow-evidence-only-commits }
  $activeSetupPath = Join-Path $mainRoot 'test_results/p888_runtime_setup_lock.json'
  $activeSetup = Get-Content $activeSetupPath -Raw | ConvertFrom-Json
  $versionedLockRel = [string]$activeSetup.setup_lock_path
  $activeLineagePath = Join-Path $mainRoot 'test_results/p888_active_lineage.json'
  $activeLineage = Get-Content $activeLineagePath -Raw | ConvertFrom-Json
  $lineageRecordRel = [string]$activeLineage.lineage_record_path
  if ($versionedLockRel -notmatch '^test_results/p888_runtime_setup_locks/[0-9a-f]{40}\.json$' -or $lineageRecordRel -notmatch '^test_results/p888_lineages/[0-9a-f]{64}\.json$') { throw 'unsafe dynamic evidence path' }
  if ((Get-FileHash -LiteralPath (Join-Path $mainRoot $versionedLockRel) -Algorithm SHA256).Hash.ToLowerInvariant() -ne [string]$activeSetup.setup_lock_sha256) { throw 'active setup pointer digest mismatch' }
  if ((Get-FileHash -LiteralPath (Join-Path $mainRoot $lineageRecordRel) -Algorithm SHA256).Hash.ToLowerInvariant() -ne [string]$activeLineage.lineage_record_sha256) { throw 'active lineage pointer digest mismatch' }
  $evidenceFiles = @('test_results/p888_bn_a100_validation.md','test_results/p888_bn_2xt4_validation.md','test_results/p888_original_puzzle0_preflight.md','test_results/p888_runtime_setup_lock.json',$versionedLockRel,'test_results/p888_active_lineage.json',$lineageRecordRel,'test_results/p888_controller_operations_task16.jsonl')
  $preStaged = @(Get-NativeChecked 'read evidence index' { git diff --cached --name-only })
  if ($preStaged.Count -ne 0) { throw "index must be empty before evidence staging: $preStaged" }
  Invoke-NativeChecked 'stage exact validation evidence' { git add -f -- $evidenceFiles }
  Assert-StagedExactly $evidenceFiles
  Invoke-NativeChecked 'commit validation evidence' { git commit -m 'test: record p888 production validation evidence' }
  Invoke-NativeChecked 'push validation evidence' { git push origin HEAD:codex-native-trainer-implementation }
  Assert-ProtectedBackupUnchanged
  ```

  Reports contain setup/runtime/controller/submission hashes, clean flags, hardware/software/build fingerprints, exact policy/snapshot/workspace, row vectors, correctness, visible/hidden generator timing, 300-sample max-rank timing, peak memory, rejection matrix, stress/resume/export results, original preflight, exact artifact paths, and A100 job IDs. No runtime file belongs in the evidence commit.
---

## Task 17: Complete Full Training Across Verified Sessions and Accept Puzzle 0

**Files created only after external work:**

- Create: `test_results/p888_full_training_2xt4.md`
- Create: `test_results/p888_puzzle0_depth100_beam10m.md`
- Create: `test_results/p888_resume_dataset_versions.jsonl`
- Create immutable snapshot: `test_results/p888_controller_operations_task17.jsonl`

- [ ] Load and prove the immutable setup before doing anything external. Reuse the Task-16 checked-native helpers; do not run controller code from the current main checkout:

  ```powershell
  Set-StrictMode -Version Latest
  $ErrorActionPreference = 'Stop'
  function Invoke-NativeChecked([string]$Label, [scriptblock]$Command) { & $Command; $code=$LASTEXITCODE; if($code -ne 0){ throw "$Label failed with exit code $code" } }
  function Get-NativeChecked([string]$Label, [scriptblock]$Command) { $lines=@(& $Command); $code=$LASTEXITCODE; if($code -ne 0){ throw "$Label failed with exit code $code" }; return $lines }
  function Assert-StagedExactly([string[]]$Expected) { $actual=@(Get-NativeChecked 'read staged paths' { git diff --cached --name-only }); $delta=@(Compare-Object ($Expected|Sort-Object) ($actual|Sort-Object)); if($delta.Count-ne0){throw "unexpected staged payload: $($delta|Out-String)"}; Invoke-NativeChecked 'cached diff check' { git diff --cached --check } }
  $script:ProtectedBackupPath = 'kaggle/kernel/run_ranks_2xt4.sh~31640'
  function Assert-ProtectedBackupUnchanged { if(-not (Test-Path -LiteralPath $script:ProtectedBackupPath -PathType Leaf)){throw 'protected backup is missing or is not a leaf'}; $info=Get-Item -LiteralPath $script:ProtectedBackupPath; if($info.Length-ne5258 -or $info.LastWriteTimeUtc.Ticks-ne639188887422510152){throw 'protected backup metadata differs from plan baseline; do not inspect contents'} }
  Assert-ProtectedBackupUnchanged
  $mainRoot = (Get-Location).Path
  $activeSetup = Get-Content (Join-Path $mainRoot 'test_results/p888_runtime_setup_lock.json') -Raw | ConvertFrom-Json
  if ([int]$activeSetup.schema -ne 1 -or [string]$activeSetup.setup_lock_path -notmatch '^test_results/p888_runtime_setup_locks/[0-9a-f]{40}\.json$' -or [string]$activeSetup.setup_lock_sha256 -notmatch '^[0-9a-f]{64}$') { throw 'malformed active setup pointer' }
  $runtimeLockPath = Join-Path $mainRoot ([string]$activeSetup.setup_lock_path)
  $actualLockSha = (Get-FileHash -LiteralPath $runtimeLockPath -Algorithm SHA256).Hash.ToLowerInvariant()
  if ($actualLockSha -ne [string]$activeSetup.setup_lock_sha256) { throw 'versioned setup lock digest mismatch' }
  $runtimeLock = Get-Content $runtimeLockPath -Raw | ConvertFrom-Json
  if ([int]$runtimeLock.schema -ne 1) { throw 'unsupported versioned setup lock schema' }
  $runtimeSourceSha = [string]$runtimeLock.runtime_source_sha
  $runtimeGitTreeSha = [string]$runtimeLock.runtime_git_tree_sha
  $runtimeTreeSha256 = [string]$runtimeLock.runtime_tree_sha256
  $controllerSha256 = [string]$runtimeLock.controller_sha256
  if ($runtimeSourceSha -notmatch '^[0-9a-f]{40}$' -or $runtimeGitTreeSha -notmatch '^[0-9a-f]{40}$' -or $runtimeTreeSha256 -notmatch '^[0-9a-f]{64}$' -or $controllerSha256 -notmatch '^[0-9a-f]{64}$') { throw 'malformed setup lock' }
  $derivedTree = ((Get-NativeChecked 'derive locked Git tree' { git rev-parse "${runtimeSourceSha}^{tree}" })[0]).Trim()
  if ($derivedTree -ne $runtimeGitTreeSha) { throw 'setup source/tree mismatch' }
  $controllerRoot = Join-Path $mainRoot "test_results/p888_controller_checkouts/$runtimeSourceSha"
  New-Item -ItemType Directory -Force -Path (Split-Path -Parent $controllerRoot) | Out-Null
  if (-not (Test-Path -LiteralPath $controllerRoot)) {
      Invoke-NativeChecked 'restore detached controller clone' { git clone --no-local ([string]$runtimeLock.repository) $controllerRoot }
  }
  Invoke-NativeChecked 'fetch locked controller commit' { git -C $controllerRoot fetch origin codex-native-trainer-implementation }
  Invoke-NativeChecked 'checkout locked controller commit' { git -C $controllerRoot checkout --detach $runtimeSourceSha }
  $actualControllerSha = (Get-FileHash -LiteralPath "$controllerRoot/scripts/p888_external_controller.py" -Algorithm SHA256).Hash.ToLowerInvariant()
  if ($actualControllerSha -ne $controllerSha256) { throw 'controller bytes differ from setup lock' }
  $controller = "$controllerRoot/scripts/p888_external_controller.py"
  $common = @('--repo-root',$controllerRoot,'--artifacts-root',(Join-Path $mainRoot 'test_results'),'--runtime-source-sha',$runtimeSourceSha,'--runtime-git-tree-sha',$runtimeGitTreeSha,'--runtime-tree-sha256',$runtimeTreeSha256,'--controller-sha256',$controllerSha256,'--setup-lock-path',$runtimeLockPath,'--setup-lock-sha256',$actualLockSha)
  $activeLineagePath = Join-Path $mainRoot 'test_results/p888_active_lineage.json'
  $initialActiveLineageSha = (Get-FileHash -LiteralPath $activeLineagePath -Algorithm SHA256).Hash.ToLowerInvariant()
  $activeLineage = Get-Content $activeLineagePath -Raw | ConvertFrom-Json
  if ([int]$activeLineage.schema -ne 1 -or [string]$activeLineage.lineage_record_path -notmatch '^test_results/p888_lineages/[0-9a-f]{64}\.json$' -or [string]$activeLineage.lineage_record_sha256 -notmatch '^[0-9a-f]{64}$') { throw 'malformed active lineage pointer' }
  $initialLineageRecordRel = [string]$activeLineage.lineage_record_path
  $lineageRecordPath = Join-Path $mainRoot ([string]$activeLineage.lineage_record_path)
  if ((Get-FileHash -LiteralPath $lineageRecordPath -Algorithm SHA256).Hash.ToLowerInvariant() -ne [string]$activeLineage.lineage_record_sha256) { throw 'lineage record digest mismatch' }
  $lineageRecord = Get-Content $lineageRecordPath -Raw | ConvertFrom-Json
  $datasetId = [string]$lineageRecord.dataset_id
  if ($datasetId -notmatch '^trydotatwo/p888-syncbn-training-resume-[0-9a-f]{16}$' -or [string]$lineageRecord.setup_lock_sha256 -ne $actualLockSha) { throw 'lineage/setup binding mismatch' }
  Invoke-NativeChecked 'prove locked controller and lineage' { py $controller assert-runtime @common --require-lineage-record $lineageRecordPath }
  ```

  The assertion verifies detached cleanliness, source ancestry/reachability, runtime hash, controller bytes, and equality between the versioned setup lock and selected lineage record. A mismatch cannot be fixed in place: create a new setup commit, SHA-named lock, generation nonce, suffixed dataset, and genesis lineage while retaining the old records.

- [ ] Make the full runner enforce the complete contract before allocation: accepted-policy bytes and SHA, training contract, input/preflight hashes, exact runtime/hardware compatibility, exactly two T4s, schema-v3 checkpoint, generator version/seed/integrity cursor, 32,692 semantic epochs, 326,920 optimizer steps, and `--retune never`. Every completed epoch proves 999,978 committed samples and depth count 34,482 for every depth 1..29. Any prefetched but untrained delta is absent from the checkpoint and dataset bundle.

- [ ] Keep each session time-bounded and checkpoint-atomic. Use 1,800 seconds for the first real session and 36,000 afterward. Fresh ranks initialize identical `local_last_effective_step_nanoseconds=max(1000000,ceil(max(accepted_full_q95_ms,accepted_final_q95_ms)*1000000))` from the locked Task-16 decimal-millisecond strings using checked integer conversion. After every healthy committed telemetry record, compute `measured_ns=ceil(effective_step_seconds*1e9)` with overflow/nonpositive rejection and set `local_ns=max(measured_ns,ceil(9*previous_local_ns/10))`, where integer decay is `(9*previous_local_ns+9)/10`. At every epoch/checkpoint/pause/final fence, the 86-word integrity allgather chooses `canonical_ns=max(local_ns over ranks)`, writes it back to every rank, and stores that one little-endian `u64` in the checkpoint; resume restores the same canonical value everywhere.

  Before an attempted batch, each rank compares its monotonic session timer with the common wall limit and sets only its local preallocated stop-request word when `remaining_ns < max(2*local_ns,300000000000)`. It must still enqueue the identical admission prefix defined in Task 15. The first input-BN SUM globalizes stop; one rank's request makes every rank pause before current-step BN state mutation. The mandatory order is: let the first input-BN health/stop consensus accept or reject the previous `PendingStepCommit` exactly once; discard the newly admitted current/prefetched deltas; wait `terminal_health_ready`; run the fixed one-word terminal-health AllGather on `runtime.bn_context` as confirmation; require `Finish == kOk`, `validation_code == 0`, and `global_health_mask == 0`; any failure takes the common no-integrity/no-checkpoint exit, while exact success enqueues and drains the newly accepted prior telemetry, update local fixed-point timing, run the 86-word integrity/timing/chain AllGather on the same context, serialize/digest-consensus/publish the checkpoint, execute the shared publication-status broadcast, build/verify the bundle, print `RUN_STATE=paused`, and exit 0. No local stop decision changes control flow before the shared SUM, and stop never masquerades as health. Test one-rank-only stop requests with unequal clocks/timings at full/final shapes, resume of canonical timing, last-step timing inclusion, and byte-identical checkpoints. A complete run prints `RUN_STATE=complete`. Any nonfinite/global-health failure, digest disagreement, sample/policy/lineage drift, invalid cursor, unexpected nominal exit, or failed bundle verification exits nonzero and publishes no dataset version.

- [ ] Rank 0 prints one concise, unbuffered line for every optimizer step after its asynchronous slot completes: semantic epoch, batch index, global step, global/local rows, loss, generator duration, visible generator wait, hidden overlap, compute time, effective step time, samples/s, elapsed wall time, ETA, global health, and checkpoint state. Every rank writes the same sequence-keyed JSONL row with local fields. Delayed ring completion is allowed, but records remain ordered, exactly once, and fully drained before pause/checkpoint/final output.

- [ ] Drive the entire lineage with one restart-safe controller command rather than hand-written per-session commands:

  ```powershell
  Invoke-NativeChecked 'complete P888 training lineage' { py $controller train-lineage @common --dataset-id $datasetId --kernel-id trydotatwo/native-multigpu-mlp-convergence-2xt4 --entry kaggle/kernel/run_p888_full_32692_2xt4.sh --output-subdir p888_full_training --first-wall-seconds 1800 --later-wall-seconds 36000 --checkpoint-epoch-interval 100 --max-new-sessions 512 --max-taint-restarts 1 }
  $activeLineage = Get-Content (Join-Path $mainRoot 'test_results/p888_active_lineage.json') -Raw | ConvertFrom-Json
  if ([string]$activeLineage.lineage_record_path -notmatch '^test_results/p888_lineages/[0-9a-f]{64}\.json$') { throw 'unsafe post-training lineage path' }
  $lineageRecordPath = Join-Path $mainRoot ([string]$activeLineage.lineage_record_path)
  if ((Get-FileHash -LiteralPath $lineageRecordPath -Algorithm SHA256).Hash.ToLowerInvariant() -ne [string]$activeLineage.lineage_record_sha256) { throw 'post-training active lineage digest mismatch' }
  $lineageRecord = Get-Content $lineageRecordPath -Raw | ConvertFrom-Json
  $datasetId = [string]$lineageRecord.dataset_id
  if ($datasetId -notmatch '^trydotatwo/p888-syncbn-training-resume-[0-9a-f]{16}$' -or [string]$lineageRecord.setup_lock_sha256 -ne $actualLockSha) { throw 'post-training lineage/setup binding mismatch' }
  ```

  Here `--dataset-id` is a guard for the lineage selected at command start, not permission to keep writing a tainted slug. If the one allowed taint recovery fires, `train-lineage` derives a fresh generation nonce and suffixed dataset from the immutable setup/base identity, atomically advances the active-lineage pointer only after verified genesis, and continues internally; the post-command reload above is therefore mandatory.

  `train-lineage` is a deterministic state machine persisted after each transition:

  ```text
  OPEN_PARENT
    -> PACKAGE_CREATED
    -> KERNEL_PUSHED
    -> MATCHING_KERNEL_TERMINAL
    -> OUTPUT_VERIFIED
    -> DATASET_STAGE_VERIFIED
    -> PARENT_CAS_REVERIFIED
    -> DATASET_VERSION_REQUESTED
    -> ROUNDTRIP_VERIFIED
    -> OPEN_PARENT or LINEAGE_COMPLETE
  ```

  At `OPEN_PARENT`, query ready status, download the actual latest remote dataset version into a fresh directory, and verify the exact lineage, runtime hashes, raw ZIP SHA, bundle SHA, accepted policy, cursor, and integrity chain. Never use a hard-coded `v001`. For every session generate one submission nonce and bake `MGT_RETUNE=never`, expected parent bundle SHA, parent session index, input dataset version, wall limit, and checkpoint interval into the runner. The same shared kernel ID is strictly serial; a queued/running prior version blocks a new push. Terminal status is accepted only with the matching submission identity and mode-specific output verifier from Task 16.

- [ ] Make restart reconciliation idempotent and nonbranching:

  - if the operation ledger says `KERNEL_PUSHED`, resume nonce polling without pushing again;
  - if matching output is verified but unpublished, redownload/reverify the recorded parent and publish only if its version/bundle/manifest still match;
  - if upload may have succeeded before a local timeout, inspect the latest dataset: an exact child with the recorded output bundle, writer nonce, submission nonce, parent, and version `input+1` is roundtripped and adopted; the controller never uploads it twice;
  - if latest is still the exact parent, resume publish; if it is neither exact parent nor exact child, write an immutable taint/conflict record, archive the current writer state, never consume that dataset again, and let the top-level controller use its one allowed taint restart to create a fresh generation nonce/suffixed dataset from genesis; a second taint stops;
  - retain an interrupted active-writer lock and require `train-lineage` to resume its matching ledger; never steal/delete it automatically;
  - after `ROUNDTRIP_VERIFIED`, atomically archive the lock and append one canonical ledger row before launching another kernel.

  Fake-CLI tests interrupt at every transition and prove one kernel push and at most one dataset version per session.

- [ ] Before every dataset version, apply the Task-16 CAS sequence: acquire exclusive writer lock; read/download/verify latest parent; verify kernel output against that exact parent, submission nonce, session index, runtime, policy, and cursor; build deterministic stage; download/reverify latest parent again immediately before upload; require unchanged version/raw ZIP/manifest/bundle; invoke checked `datasets version`; require ready and exactly `input+1`; download a fresh roundtrip and require exact published raw ZIP SHA, output bundle SHA, lineage, writer nonce, submission nonce, and all hashes. The next session reads only this roundtrip. Never use `-d`, never publish an `ERROR` output, and never launch the next kernel before roundtrip success.

- [ ] Require monotonic progress and the forced resume proof. The first real session must use 1,800 seconds, advance global step, end `paused`, and publish a valid child. Session 2 must restore byte-identical checkpoint tensors, FP16 mirror, optimizer, BN state, accepted policy, generator state, committed integrity accumulator, rolling integrity chain, and schedule cursor; its first global step equals session 1's `next_global_step`. Every later paused session must strictly advance step or epoch. Three consecutive no-progress sessions, any cursor rollback, or more than 512 new sessions stops with the last verified dataset intact.

- [ ] Append one canonical row to `test_results/p888_resume_dataset_versions.jsonl` only after each exact roundtrip. Record controller operation/submission/writer nonces, kernel ID and reported version/status, input/output dataset versions, session/parent indices, input/output ZIP and bundle SHA, lineage, start/end step/epoch/batch/sample, integrity-chain SHA, run state, package/output/stage/roundtrip paths, and all runtime/policy hashes. Re-read the appended row and fsync it. This local ledger is evidence, not authority; the remote roundtripped bundle remains the resume parent.

- [ ] At completion, require exactly 32,692 epochs, 326,920 optimizer steps, every epoch checksum/depth histogram in the rolling integrity evidence, final schema-v3 checkpoint, byte-identical accepted policy, unbroken parent chain from genesis, and all folded-export gates. Publish and exact-roundtrip the final `complete` bundle before any beam search.

- [ ] Run final comparison through a new nonce-identified sequential kernel version, loading the exact complete dataset and the two original model datasets:

  ```powershell
  Invoke-NativeChecked 'final puzzle-zero comparison' { py $controller run-final-comparison @common --dataset-id $datasetId --kernel-id trydotatwo/native-multigpu-mlp-convergence-2xt4 --entry kaggle/kernel/run_puzzle0_model_compare_2xt4.sh --output-subdir p888_puzzle0_compare --original-dataset arabidopsisthalian/ihes-model-1778521793 --original-dataset arabidopsisthalian/model-ihes-1780290207-e40960 }
  ```

  The controller opens and verifies the latest `complete` bundle, bakes its exact dataset version/bundle/policy/runtime hashes and a fresh nonce, and uses mode `final_compare`. In one run revalidate Task-1 original manifests, then run every available original and the final folded model on puzzle 0, depth 100, beam 10,000,000, identical scoring and the existing two-GPU beam search, logging every depth. Require at least one original to solve in this same run. Baseline is the shortest solution among those solved originals. Require the final model to solve and `final_length <= best_original_length + 2`; a previous native checkpoint may be reported but never substitutes for the original baseline.

- [ ] Render the full-training and puzzle reports only from verified controller/dataset/kernel ledgers. Include model/checkpoint/policy/runtime/controller/submission hashes, complete lineage, loss trajectory, visible/hidden generator timing, training throughput/wall time, solution lengths, expanded states, beam time, per-depth logs, exact commands, job/kernel/dataset versions where available, and artifact paths. Then run final repository gates while the detached controller proves the main runtime still equals setup:

  ```powershell
  Invoke-NativeChecked 'render final reports' { py $controller render-training-reports @common --dataset-id $datasetId --operations-snapshot (Join-Path $mainRoot 'test_results/p888_controller_operations_task17.jsonl') }
  Invoke-NativeChecked 'prove main runtime before evidence' { py $controller assert-runtime --repo-root $mainRoot --runtime-source-sha $runtimeSourceSha --runtime-git-tree-sha $runtimeGitTreeSha --runtime-tree-sha256 $runtimeTreeSha256 --controller-sha256 $controllerSha256 --allow-evidence-only-commits }
  Invoke-NativeChecked 'queued final CUDA gate' { docker exec mgt-gpu-queue python3 scripts/gpu_queue_submit.py --label task17-final-local-gate --wait -- bash -lc "cmake --build native/build --config Release --parallel && ctest --test-dir native/build -C Release --output-on-failure --no-tests=error" }
  Invoke-NativeChecked 'final controller tests' { py -m unittest scripts.tests.test_autotune_2xt4 scripts.tests.test_p888_external_controller scripts.tests.test_p888_resume_bundle -v }
  Invoke-NativeChecked 'final diff check' { git diff --check }
  $changed = @(Get-NativeChecked 'list changed source files' { git diff --name-only c770e20..HEAD -- '*.md' '*.cpp' '*.cu' '*.cuh' '*.hpp' '*.py' '*.sh' '*.json' })
  $renderedEvidence = @('test_results/p888_full_training_2xt4.md','test_results/p888_puzzle0_depth100_beam10m.md','test_results/p888_resume_dataset_versions.jsonl','test_results/p888_controller_operations_task17.jsonl')
  $scanFiles = @($changed + $renderedEvidence + @('test_results/p888_active_lineage.json',[string]$activeLineage.lineage_record_path) | Sort-Object -Unique | Where-Object { Test-Path -LiteralPath $_ })
  if ($scanFiles.Count -gt 0) {
      & rg -n 'T[B]D|T[O]DO|F[I]XME|implement lat[e]r|appropriate error handlin[g]|similar t[o]|<fill[-_ a-z]*>' -- $scanFiles
      $scanCode = $LASTEXITCODE
      if ($scanCode -eq 0) { throw 'placeholder prose found in source or rendered evidence' }
      if ($scanCode -ne 1) { throw "placeholder scan failed with exit $scanCode" }
  }
  ```

  The exact detached controller, not a mutable helper, performs the runtime comparison. Untracked files outside the reviewed runtime pathspecs—including the protected backup—do not enter the runtime hash and remain untouched; any tracked or untracked file that matches a runtime pathspec invalidates evidence.

- [ ] Commit the four compact final-evidence files and, only if taint recovery switched lineage, the changed active-lineage pointer plus its new immutable record; packages, raw outputs, mutable controller state, detached checkout, and dataset staging stay ignored:

  ```powershell
  $finalEvidenceBase = @('test_results/p888_full_training_2xt4.md','test_results/p888_puzzle0_depth100_beam10m.md','test_results/p888_resume_dataset_versions.jsonl','test_results/p888_controller_operations_task17.jsonl')
  $currentActiveLineageSha = (Get-FileHash -LiteralPath $activeLineagePath -Algorithm SHA256).Hash.ToLowerInvariant()
  $currentLineageRecordRel = [string]$activeLineage.lineage_record_path
  $dynamicLineageEvidence = @()
  if ($currentActiveLineageSha -ne $initialActiveLineageSha) {
      if ($currentLineageRecordRel -eq $initialLineageRecordRel -or $currentLineageRecordRel -notmatch '^test_results/p888_lineages/[0-9a-f]{64}\.json$') { throw 'lineage switch did not select one new safe record' }
      $currentLineageRecordPath = Join-Path $mainRoot $currentLineageRecordRel
      if (-not (Test-Path -LiteralPath $currentLineageRecordPath -PathType Leaf)) { throw 'new lineage record is missing' }
      $actualRecordSha = (Get-FileHash -LiteralPath $currentLineageRecordPath -Algorithm SHA256).Hash.ToLowerInvariant()
      if ($actualRecordSha -ne [string]$activeLineage.lineage_record_sha256) { throw 'new lineage record digest mismatch' }
      $trackedRecord = @(Get-NativeChecked 'inspect new lineage tracking state' { git ls-files -- $currentLineageRecordRel })
      if ($trackedRecord.Count -gt 1 -or ($trackedRecord.Count -eq 1 -and $trackedRecord[0] -ne $currentLineageRecordRel)) { throw 'ambiguous lineage tracking state' }
      $dynamicLineageEvidence = @('test_results/p888_active_lineage.json',$currentLineageRecordRel)
  }
  $finalEvidence = @($finalEvidenceBase + $dynamicLineageEvidence)
  $preStaged = @(Get-NativeChecked 'read final evidence index' { git diff --cached --name-only })
  if ($preStaged.Count -ne 0) { throw "index must be empty before final evidence staging: $preStaged" }
  Invoke-NativeChecked 'stage exact final evidence' { git add -f -- $finalEvidence }
  Assert-StagedExactly $finalEvidence
  Invoke-NativeChecked 'commit final evidence' { git commit -m 'test: record p888 full training and puzzle zero evidence' }
  Invoke-NativeChecked 'push final evidence' { git push origin HEAD:codex-native-trainer-implementation }
  Assert-ProtectedBackupUnchanged
  ```

  If any runtime source, runner, packager, verifier, or controller requires a fix after setup, do not label it evidence-only: stop, make a new setup commit, recompute hashes, create a new immutable SHA-named setup lock and generation-nonce lineage on a new suffixed dataset, advance the active pointers only after verification, and rerun every affected external gate. Never overwrite old lock/lineage/dataset evidence.
---

## Optional Experiments After the Production Gate

These are not prerequisites for full training and must remain disabled unless they independently pass the same correctness/performance protocol:

- compact FP16 saved `xhat` to reduce roughly 250 MiB/rank;
- recompute/checkpoint activations instead of storing every `xhat`;
- CUDA Graph capture after the full/final-batch graphs are stable;
- concurrent dW/dX GEMMs on separate streams;
- forced CUTLASS input-gradient kernels;
- measured tile-bucket overlap on a separate explicit policy, never the reserved placeholder;
- reduce-scatter/allgather with sharded optimizer ownership.

Each experiment needs a new explicit enum value, capacity query, dedicated tests, separate artifact row, and fail-closed dispatch.

## Final Acceptance Checklist

- [ ] The original source-backed input and residual contracts are executable and green.
- [ ] Original logical parameter count is exactly 15,357,749.
- [ ] Physical padding is zero, invisible to logical export, and never trained as logical state.
- [ ] All 34 true global SyncBN sites match the CPU/PyTorch oracle.
- [ ] Strict FP32 and selected production policies remain separately callable.
- [ ] Every hot-path allocation, handle, stream, event, and workspace is precreated.
- [ ] Input-gradient sparse owner-write is not the production default.
- [ ] The accepted T4 policy is either a measured >=3% winner or an explicitly validated strict fallback, never an A100 transfer.
- [ ] Full and final batches use exact local-row partitions.
- [ ] The epoch contains 999,978 fresh examples with equal depth counts.
- [ ] Full training completes 32,692 semantic epochs and 326,920 optimizer steps.
- [ ] Checkpoint/resume preserves linear, BN, optimizer, schedule, generator, embedded policy, and source/runtime identity.
- [ ] Every Kaggle session is a verified single-parent dataset lineage with roundtrip-valid bundle bytes.
- [ ] Folded export matches unfused evaluation and preserves beam scoring.
- [ ] Puzzle 0 solves at depth at most 100 with beam 10M.
- [ ] Final solution is within two moves of the best original model in the same run.
- [ ] All accepted/rejected measurements, hashes, commands, profiles, and logs are reproducible.

## Handoff to the Implementing Agent

Start with Task 1 and do not begin CUDA optimization until Tasks 1-3 are committed and green. Use `superpowers:subagent-driven-development` when independent test/implementation reviews can run in parallel; otherwise use `superpowers:executing-plans`. At each task boundary, report only:

```text
completed task and commit
focused tests
broader tests
measured result, if applicable
remaining blocker
next numbered task
```
