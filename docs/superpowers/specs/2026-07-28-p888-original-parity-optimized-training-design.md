# P888 Original-Parity Optimized Training Design

## Purpose

Reproduce the original P888 model and training process at the mathematical and statistical level while retaining the optimized native multi-GPU implementation for 2x T4.

The result must train the same network family on the same data distribution for the same amount of work as the original code. Exact PyTorch RNG order, byte-identical initialization, and byte-identical optimizer trajectories are not required.

## Original Training Contract

- Architecture: `6336 -> 2556 -> 218`, followed by 16 residual blocks of `218 -> 218 -> 218`, then a scalar output.
- BatchNorm placement:
  - after the input linear layer;
  - after the hidden linear layer;
  - after `fc1` and `fc2` in every residual block;
  - no BatchNorm after the output layer.
- Activation: ReLU after every BatchNorm.
- Objective: mean squared error.
- Optimizer: Adam, learning rate `1e-4`, no weight decay.
- Depth distribution: equal sample count for every depth from 1 through 29.
- Walk generation: fresh non-backtracking random walks.
- Walkers per depth: 34,482.
- Samples per epoch: `34,482 * 29 = 999,978`.
- Global batch size: 100,000, producing nine full batches and one 99,978-sample batch per epoch.
- Original training length: 32,692 epochs, or approximately 326,920 optimizer updates and 32.69 billion generated examples.

For two ranks, every full global batch is split into 50,000 samples per rank. The final batch is split into 49,989 samples per rank.

## Accepted Differences

- Parameter initialization may differ from the original process.
- Random-walk RNG algorithm and sample ordering may differ.
- Floating-point reduction order may differ.
- Mixed-precision production mode may use FP16 tensors with FP32 accumulation and master state.

These differences are accepted only if the fixed-batch parity tests, training-behavior tests, full-workload contract, and puzzle-quality gate pass.

## BatchNorm Semantics

Training BatchNorm follows `torch.nn.BatchNorm1d` defaults:

- epsilon: `1e-5`;
- momentum: `0.1`;
- trainable `gamma` and `beta`;
- batch mean and biased batch variance for normalization;
- unbiased variance correction when updating `running_var`;
- FP32 running mean, running variance, affine parameters, gradients, and optimizer state.

The production two-rank implementation computes local sum and sum of squares, all-reduces them with NCCL, and derives one global mean and variance. Consequently, it is true synchronized global BatchNorm over the 100,000-sample global batch, not independent per-rank BatchNorm.

Backward propagation all-reduces the global sufficient statistics required for `dx`, `dgamma`, and `dbeta`. Rank-local approximations are not permitted in parity or production modes.

## Native Execution Architecture

Two execution modes share one model layout and checkpoint format:

1. `parity_fp32`: FP32 linear operations, FP32 BatchNorm, deterministic fixed fixtures where practical. This mode is the correctness oracle.
2. `production_mixed`: FP16 linear inputs and weights, FP32 GEMM accumulation and master weights, FP32 BatchNorm statistics and optimizer state.

The production path uses:

- CUTLASS or cuBLASLt GEMMs selected by the existing autotuner;
- fused bias, normalization, affine transform, ReLU, and FP16 output conversion;
- reusable workspaces with no per-layer hot-path allocation;
- packed BatchNorm state and gradients;
- batched or grouped NCCL reductions where doing so preserves the dependency order;
- CUDA events and explicit correctness gates for every autotuned candidate.

Autotuning must fail closed: an unknown policy or an unverified candidate cannot become the production choice.

## Data and Epoch Scheduling

An epoch is a semantic unit of exactly 999,978 newly generated examples. The native generator must produce exactly 34,482 examples for each target depth 1 through 29 before shuffling or distributed scheduling.

The two ranks receive disjoint shards of every global batch. Neither padding nor silent sample duplication is allowed. The last batch uses its true size in MSE and BatchNorm denominators.

The checkpoint records the completed semantic epoch and optimizer step. Resume begins at the next complete epoch; mid-epoch recovery is outside this design because epoch generation is deterministic only at the distributional, not byte-exact, level.

## Checkpoint and Export

The checkpoint schema is extended to contain:

- linear parameters;
- BatchNorm `gamma`, `beta`, `running_mean`, `running_var`, and batch counter;
- Adam first and second moments for all trainable parameters;
- optimizer step;
- completed semantic epoch;
- model and training-contract fingerprints.

Loading an older checkpoint without BatchNorm training state is rejected for parity training. It remains available only through the existing legacy inference import path.

Inference export folds each BatchNorm into the preceding linear weights and bias using the stored running statistics. The exported inference layout therefore remains compatible with the optimized beam-search evaluator.

## Validation Gates

1. **CPU reference:** native FP32 BatchNorm forward and backward agree with a PyTorch fixture for full and final batch sizes within declared tolerances.
2. **Single-GPU CUDA parity:** every BatchNorm placement, loss, representative activations, and gradients agree with the CPU/PyTorch reference.
3. **Two-GPU synchronization:** 2x T4 synchronized BatchNorm agrees with the same global batch evaluated on one logical reference rank.
4. **Short training:** loss decreases, all BatchNorm parameters update, running statistics remain finite, and checkpoint/resume reproduces the uninterrupted run within mixed-precision tolerance.
5. **Throughput:** the fastest correctness-approved 2x T4 policy is selected automatically and is faster in samples per second than the equivalent PyTorch 2x T4 reference run. A profile must attribute BatchNorm, GEMM, communication, generation, and optimizer time separately.
6. **Full workload:** training completes the equivalent of 32,692 original epochs, not merely 32,692 optimizer steps.
7. **Puzzle quality:** on puzzle 0 with depth 100 and beam 10,000,000, the trained model must solve the puzzle. Its solution may be at most two moves longer than the best original checkpoint tested with the identical search configuration.

## Operational Boundary

All implementation, tuning, and validation in this phase target exactly two NVIDIA T4 GPUs. No configuration or optimization for other GPU counts is part of the acceptance path.

Existing Kaggle sessions must not be interrupted. Implementation and GPU execution begin only after the user confirms that a Kaggle GPU slot is free.

## Completion Criteria

The work is complete only when:

- all seven validation gates pass;
- the chosen autotune policy and its measurements are saved as artifacts;
- a resumable full-training checkpoint exists;
- the puzzle-0 comparison report includes the original and native results under the identical search contract;
- the branch contains the code, tests, launch configuration, and reproducible commands.
