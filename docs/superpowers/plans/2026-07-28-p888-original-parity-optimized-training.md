# P888 Original-Parity Optimized Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add optimized synchronized BatchNorm and reproduce the original P888 training workload exactly at the statistical-contract level on 2x T4.

**Architecture:** Extend the native model layout with FP32 BatchNorm state, implement a CPU oracle and fused CUDA/NCCL forward-backward path, preserve the exact epoch/data/batch contract, then export a BN-folded inference model for the existing multi-GPU beam search.

**Tech Stack:** C++17, CUDA, CUTLASS/cuBLASLt, NCCL, CMake/CTest, Python/PyTorch reference fixtures, Kaggle 2x T4.

## Global Constraints

- Work only on 2x T4.
- Do not interrupt existing Kaggle sessions.
- Do not start implementation or GPU jobs until the user confirms a free Kaggle slot.
- Preserve global batch 100,000 and final batch 99,978.
- Preserve 34,482 examples at each depth 1 through 29 per semantic epoch.
- Use true synchronized global BatchNorm; rank-local BatchNorm is invalid.
- Accept different initialization and RNG order, but not a changed architecture or training distribution.
- Reject unverified autotune candidates and unknown policies.
- Do not alter the inference beam-search scoring contract.

---

## Task 1: Freeze the Original Contract in Executable Tests

**Files:**

- Create: `native/tests/reference/generate_p888_bn_fixture.py`
- Create: `native/tests/fixtures/p888_bn_contract.json`
- Create: `native/tests/test_p888_training_contract.cpp`
- Modify: `native/CMakeLists.txt`

- [ ] Add a failing contract test that asserts the layer sizes, 34 BatchNorm sites, epsilon, momentum, depth range, walkers per depth, samples per epoch, batch sizes, optimizer, and epoch count.

- [ ] Run:

  `cmake --build native/build --config Release --target test_p888_training_contract`

  Expected: failure because the contract API and fixture do not exist.

- [ ] Add a small PyTorch fixture generator containing `Linear -> BatchNorm1d -> ReLU` and one residual block. Emit inputs, parameters, running state, forward outputs, loss, and backward gradients as JSON-compatible binary arrays plus metadata.

- [ ] Add a typed `P888TrainingContract` in `native/include/mgt/config.hpp` with:

  ```cpp
  struct P888TrainingContract {
    static constexpr int kMinDepth = 1;
    static constexpr int kMaxDepth = 29;
    static constexpr int kWalkersPerDepth = 34482;
    static constexpr int kSamplesPerEpoch = 999978;
    static constexpr int kGlobalBatch = 100000;
    static constexpr int kEpochs = 32692;
    static constexpr float kBatchNormEpsilon = 1.0e-5f;
    static constexpr float kBatchNormMomentum = 0.1f;
  };
  ```

- [ ] Generate the fixture and run the test again.

  Expected: pass with all original-contract assertions.

- [ ] Commit:

  `git add native/include/mgt/config.hpp native/tests/reference/generate_p888_bn_fixture.py native/tests/fixtures/p888_bn_contract.json native/tests/test_p888_training_contract.cpp native/CMakeLists.txt`

  `git commit -m "test: freeze original p888 training contract"`

## Task 2: Extend Model Layout with BatchNorm State

**Files:**

- Modify: `native/include/mgt/model_layout.hpp`
- Modify: `native/src/model_layout.cpp`
- Modify: `native/tests/test_model_layout.cpp`
- Modify: `native/tests/test_parameter_mapping.cpp`

- [ ] Add failing tests for 34 ordered BatchNorm sites and contiguous FP32 affine/running-state offsets.

- [ ] Define:

  ```cpp
  struct BatchNormSlice {
    std::string name;
    int features;
    size_t gamma_offset;
    size_t beta_offset;
    size_t running_mean_offset;
    size_t running_var_offset;
  };
  ```

  Add `std::vector<BatchNormSlice> batch_norms` and explicit counts for trainable and running-state scalars to `ModelLayout`.

- [ ] Generate sites in strict forward order: input, hidden, then block `0..15` `fc1`, `fc2`.

- [ ] Initialize `gamma=1`, `beta=0`, `running_mean=0`, and `running_var=1` in the trainer state constructor.

- [ ] Run:

  `ctest --test-dir native/build -C Release -R "model_layout|parameter_mapping" --output-on-failure`

  Expected: pass.

- [ ] Commit:

  `git add native/include/mgt/model_layout.hpp native/src/model_layout.cpp native/tests/test_model_layout.cpp native/tests/test_parameter_mapping.cpp`

  `git commit -m "feat: add batchnorm model state layout"`

## Task 3: Implement the CPU BatchNorm Oracle

**Files:**

- Create: `native/include/mgt/batch_norm.hpp`
- Create: `native/src/batch_norm_cpu.cpp`
- Create: `native/tests/test_batch_norm_cpu.cpp`
- Modify: `native/src/mlp_cpu_ref.cpp`
- Modify: `native/include/mgt/mlp_cpu_ref.hpp`
- Modify: `native/CMakeLists.txt`

- [ ] Write failing fixture-based tests for training forward, evaluation forward, running-stat updates, `dx`, `dgamma`, and `dbeta`.

- [ ] Define APIs:

  ```cpp
  void batch_norm_forward_cpu(
      const float* x, int rows, int cols,
      const float* gamma, const float* beta,
      float* running_mean, float* running_var,
      float momentum, float epsilon, bool training,
      float* y, BatchNormCache* cache);

  void batch_norm_backward_cpu(
      const float* dy, const BatchNormCache& cache,
      const float* gamma, float* dx,
      float* dgamma, float* dbeta);
  ```

- [ ] Match PyTorch semantics: biased variance for normalization and unbiased correction for `running_var`.

- [ ] Insert BatchNorm at all 34 sites in the CPU MLP reference.

- [ ] Run:

  `ctest --test-dir native/build -C Release -R "batch_norm_cpu|mlp_cpu" --output-on-failure`

  Expected: pass within fixture tolerances.

- [ ] Commit:

  `git add native/include/mgt/batch_norm.hpp native/src/batch_norm_cpu.cpp native/tests/test_batch_norm_cpu.cpp native/src/mlp_cpu_ref.cpp native/include/mgt/mlp_cpu_ref.hpp native/CMakeLists.txt`

  `git commit -m "feat: implement p888 batchnorm cpu reference"`

## Task 4: Implement Single-GPU CUDA BatchNorm

**Files:**

- Create: `native/cuda/batch_norm.cuh`
- Create: `native/cuda/batch_norm.cu`
- Create: `native/tests/cuda/test_cuda_batch_norm_reference.cu`
- Modify: `native/CMakeLists.txt`

- [ ] Add failing CUDA tests over feature widths 2,556 and 218, including full and final per-rank batch sizes.

- [ ] Implement reduction kernels that accumulate sum and sum of squares in FP32.

- [ ] Implement fused training forward:

  `bias + normalize + gamma/beta + ReLU + optional FP16 store`

- [ ] Implement backward sufficient-statistic reduction and fused `dx`, `dgamma`, and `dbeta`.

- [ ] Reuse caller-provided workspace; assert that the hot path performs no allocation.

- [ ] Run:

  `ctest --test-dir native/build -C Release -R cuda_batch_norm_reference --output-on-failure`

  Expected: pass against the CPU oracle, including non-divisible row counts.

- [ ] Commit:

  `git add native/cuda/batch_norm.cuh native/cuda/batch_norm.cu native/tests/cuda/test_cuda_batch_norm_reference.cu native/CMakeLists.txt`

  `git commit -m "feat: add fused cuda batchnorm kernels"`

## Task 5: Add True Two-Rank Synchronized BatchNorm

**Files:**

- Create: `native/include/mgt/sync_batch_norm.hpp`
- Create: `native/cuda/sync_batch_norm.cu`
- Create: `native/tests/cuda/test_sync_batch_norm_2rank.cu`
- Create: `native/tests/run_sync_batch_norm_2rank.ps1`
- Modify: `native/CMakeLists.txt`

- [ ] Add a two-rank failing test where rank distributions differ deliberately, so rank-local BatchNorm cannot accidentally pass.

- [ ] Implement an API accepting the existing NCCL communicator and CUDA stream:

  ```cpp
  void sync_batch_norm_forward_cuda(
      const BatchNormForwardArgs& args,
      ncclComm_t comm, cudaStream_t stream);

  void sync_batch_norm_backward_cuda(
      const BatchNormBackwardArgs& args,
      ncclComm_t comm, cudaStream_t stream);
  ```

- [ ] All-reduce sum, sum of squares, and backward sufficient statistics. Use the true global row count supplied by the distributed batch scheduler.

- [ ] Pack compatible reductions to reduce launch latency without crossing a forward/backward dependency.

- [ ] Run on 2x T4:

  `powershell -ExecutionPolicy Bypass -File native/tests/run_sync_batch_norm_2rank.ps1`

  Expected: both ranks report pass, and concatenated outputs/gradients agree with the single logical reference batch.

- [ ] Commit:

  `git add native/include/mgt/sync_batch_norm.hpp native/cuda/sync_batch_norm.cu native/tests/cuda/test_sync_batch_norm_2rank.cu native/tests/run_sync_batch_norm_2rank.ps1 native/CMakeLists.txt`

  `git commit -m "feat: implement two-rank synchronized batchnorm"`

## Task 6: Integrate BatchNorm into Native MLP Training

**Files:**

- Modify: `native/cuda/mlp_forward.cu`
- Modify: `native/cuda/mlp_backward.cu`
- Modify: `native/cuda/mlp_backward.cuh`
- Modify: `native/tools/mgt_native_train_smoke.cu`
- Modify: `native/tests/cuda/test_cuda_mlp_cpu_reference.cu`
- Create: `native/tests/cuda/test_cuda_mlp_pytorch_fixture.cu`

- [ ] Extend the PyTorch fixture test to cover the complete 16-block network and fail against the current no-BatchNorm path.

- [ ] Replace every direct bias/ReLU stage with the synchronized BatchNorm stage in training mode.

- [ ] Retain FP32 BatchNorm state and master parameters while allowing FP16 GEMM tensors in production mode.

- [ ] Include BatchNorm affine parameters in Adam with zero weight decay.

- [ ] Add finite-value guards after statistics, activations, gradients, and optimizer updates.

- [ ] Run:

  `ctest --test-dir native/build -C Release -R "cuda_mlp_cpu_reference|cuda_mlp_pytorch_fixture" --output-on-failure`

  Expected: FP32 parity passes tightly; mixed precision passes the declared activation, loss, and gradient tolerances.

- [ ] Commit:

  `git add native/cuda/mlp_forward.cu native/cuda/mlp_backward.cu native/cuda/mlp_backward.cuh native/tools/mgt_native_train_smoke.cu native/tests/cuda/test_cuda_mlp_cpu_reference.cu native/tests/cuda/test_cuda_mlp_pytorch_fixture.cu`

  `git commit -m "feat: train native p888 with synchronized batchnorm"`

## Task 7: Enforce Exact Epoch and Distributed Batch Semantics

**Files:**

- Modify: `native/include/mgt/training_data_pipeline.hpp`
- Modify: `native/cuda/training_data_pipeline.cu`
- Modify: `native/tools/mgt_native_train_smoke.cu`
- Modify: `native/tests/test_training_data_pipeline.cpp`
- Create: `native/tests/test_epoch_schedule.cpp`

- [ ] Add failing tests for equal per-depth counts, total 999,978, nine 100,000 batches, one 99,978 batch, and exact two-rank splits.

- [ ] Add:

  ```cpp
  struct EpochBatch {
    int global_offset;
    int global_rows;
    int rank_offset;
    int rank_rows;
  };

  std::vector<EpochBatch> make_epoch_schedule(int rank, int world_size);
  ```

- [ ] Generate exactly 34,482 non-backtracking walks at every depth 1 through 29 before assigning disjoint rank shards.

- [ ] Use actual `global_rows` for MSE and synchronized BatchNorm denominators.

- [ ] Log semantic epoch, optimizer step, depth histogram, global batch rows, samples per second, and loss.

- [ ] Run:

  `ctest --test-dir native/build -C Release -R "training_data_pipeline|epoch_schedule" --output-on-failure`

  Expected: pass with no padding or duplicated indices.

- [ ] Commit:

  `git add native/include/mgt/training_data_pipeline.hpp native/cuda/training_data_pipeline.cu native/tools/mgt_native_train_smoke.cu native/tests/test_training_data_pipeline.cpp native/tests/test_epoch_schedule.cpp`

  `git commit -m "feat: reproduce original p888 epoch schedule"`

## Task 8: Extend Checkpoint/Resume and BN-Folded Export

**Files:**

- Modify: `native/include/mgt/training_artifacts.hpp`
- Modify: `native/src/training_artifacts.cpp`
- Modify: `native/tests/test_training_artifacts.cpp`
- Modify: `native/include/mgt/weight_export.hpp`
- Modify: `native/src/weight_export.cpp`
- Modify: `native/tests/test_weight_export.cpp`

- [ ] Add failing round-trip tests for BatchNorm state, affine Adam moments, optimizer step, semantic epoch, and contract fingerprint.

- [ ] Introduce checkpoint schema version 3 and reject missing BatchNorm training state on parity resume.

- [ ] Store only completed semantic epochs as resumable boundaries.

- [ ] Fold BatchNorm into the preceding linear transform during inference export:

  ```text
  scale = gamma / sqrt(running_var + epsilon)
  folded_weight[row, :] = weight[row, :] * scale[row]
  folded_bias[row] = beta[row] + (bias[row] - running_mean[row]) * scale[row]
  ```

- [ ] Compare unfused evaluation output with folded inference output on a fixed batch.

- [ ] Run:

  `ctest --test-dir native/build -C Release -R "training_artifacts|weight_export" --output-on-failure`

  Expected: checkpoint resume and folded-export parity pass.

- [ ] Commit:

  `git add native/include/mgt/training_artifacts.hpp native/src/training_artifacts.cpp native/tests/test_training_artifacts.cpp native/include/mgt/weight_export.hpp native/src/weight_export.cpp native/tests/test_weight_export.cpp`

  `git commit -m "feat: checkpoint and export batchnorm training state"`

## Task 9: Autotune and Validate on 2x T4

**Files:**

- Modify: `native/include/mgt/autotune.hpp`
- Modify: `native/src/autotune.cpp`
- Modify: `native/tools/mgt_native_train_smoke.cu`
- Create: `kaggle/kernel/run_p888_parity_2xt4.sh`
- Create: `test_results/p888_bn_2xt4_validation.md`

- [ ] Add candidate policies for GEMM selection, BatchNorm block geometry, reduction packing, workspace size, and safe communication overlap.

- [ ] Require every candidate to pass a fixed-output checksum/tolerance gate before timing.

- [ ] Benchmark warmup plus repeated steady-state semantic batches. Record median and tail step time, samples per second, loss, maximum memory, and phase timings.

- [ ] Compare the fastest verified native policy with a PyTorch DistributedDataParallel plus synchronized-BatchNorm reference under the same global batch and model.

- [ ] Run short checkpoint/resume training and verify decreasing loss, finite running state, and resumed/uninterrupted agreement.

- [ ] Save the winning policy keyed by the exact two-T4 hardware/software fingerprint.

- [ ] Commit:

  `git add native/include/mgt/autotune.hpp native/src/autotune.cpp native/tools/mgt_native_train_smoke.cu kaggle/kernel/run_p888_parity_2xt4.sh test_results/p888_bn_2xt4_validation.md`

  `git commit -m "perf: autotune original-parity training on two t4s"`

## Task 10: Run Full Training and Puzzle-0 Acceptance

**Files:**

- Create: `kaggle/kernel/run_p888_full_32692_2xt4.sh`
- Create: `test_results/p888_full_training_2xt4.md`
- Create: `test_results/p888_puzzle0_depth100_beam10m.md`
- Modify: `kaggle/kernel/run_puzzle0_model_compare_2xt4.sh`

- [ ] Launch full training for 32,692 semantic epochs with periodic version-3 checkpoints and explicit resume verification.

- [ ] Confirm artifacts report approximately 326,920 optimizer steps and exactly 999,978 generated examples per completed epoch.

- [ ] Export the final BN-folded model.

- [ ] Run the original comparison checkpoints and the native model on puzzle 0 with maximum depth 100 and beam 10,000,000 using the existing multi-GPU beam search and per-step console logs.

- [ ] Require the native model to solve and to produce a solution no more than two moves longer than the best original result in the same run.

- [ ] Record checkpoint hashes, autotune policy, environment fingerprint, training throughput, final loss, solution lengths, expanded states, and wall time.

- [ ] Run final verification:

  `ctest --test-dir native/build -C Release --output-on-failure`

  `git diff --check`

  `rg -n "T[B]D|T[O]DO|implement lat[e]r|appropriate error handlin[g]|similar t[o]" docs/superpowers native kaggle/kernel test_results`

  Expected: tests pass, diff check is clean, and the placeholder scan has no hits introduced by this work.

- [ ] Commit:

  `git add kaggle/kernel/run_p888_full_32692_2xt4.sh kaggle/kernel/run_puzzle0_model_compare_2xt4.sh test_results/p888_full_training_2xt4.md test_results/p888_puzzle0_depth100_beam10m.md`

  `git commit -m "test: validate full p888 parity training and puzzle quality"`

## Final Acceptance Checklist

- [ ] Architecture contains all 34 original BatchNorm sites.
- [ ] Global synchronized BatchNorm matches the PyTorch reference.
- [ ] Epoch means 999,978 fresh examples, not one optimizer step.
- [ ] Full run completes 32,692 semantic epochs.
- [ ] Native 2x T4 training is faster than the measured PyTorch reference.
- [ ] Checkpoint/resume preserves model, BatchNorm, optimizer, step, and epoch state.
- [ ] Exported folded model matches evaluation output.
- [ ] Puzzle 0 is solved at depth 100 and beam 10M.
- [ ] Solution length is within two moves of the best original checkpoint.
- [ ] Reproducible logs, hashes, profiles, and commands are committed.
