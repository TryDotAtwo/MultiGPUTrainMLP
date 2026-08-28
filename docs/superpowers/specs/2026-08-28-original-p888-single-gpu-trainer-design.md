# Original p888 Single-GPU Trainer Design

## Goal

Build a fast, correctness-gated single-GPU trainer for the source-backed original p888 network. Develop and profile on the local RTX 3070 Laptop GPU in the shared Docker GPU queue. Preserve Tesla T4 SM75 as the production acceptance target.

This subsystem is the compute baseline for later data-parallel training. It contains no NCCL and no multi-rank control flow.

## Fixed model contract

- Input: 72 positions with 72 categorical values, logically `72 * 72 = 5184` features.
- Input projection: logical width 2556, physical aligned width 2560.
- Hidden projection: logical width 218, physical aligned width 224.
- Residual stack: 16 blocks, each with two logical `218 -> 218` linear layers.
- Output: one scalar.
- BatchNorm: after the input projection and after both linear layers in every residual block.
- Activation: ReLU after every BatchNorm, preserving the source-backed residual ordering.
- Padding lanes are zero/inert and never affect loss, gradients, running statistics, checkpoints, or exported weights.

## Precision contract

Production mixed precision uses FP16 operands for Tensor Core linear work with FP32 accumulation. Master parameters, gradients at optimizer boundaries, BatchNorm statistics and state, loss, and AdamW state remain FP32.

The existing strict FP32 CPU/CUDA path remains the correctness oracle. Performance changes are ineligible unless forward values, representative activations, all parameter gradients, one-step AdamW state, and padding satisfy declared mixed-precision tolerances.

## Ownership boundary

Rust owns configuration, validation, CLI, run directories, checkpoint/artifact policy, telemetry serialization, and process-level error reporting.

C++20 owns the trainer object and CUDA resource lifetime. CUDA C++ owns the steady-state data plane: random-walk generation, forward, loss, backward, and AdamW. cuBLASLt/CUTLASS provide GEMM implementations; custom CUDA kernels are used where layout-aware fusion removes launches or global-memory traffic. CUB/CCCL is preferred for generic reductions.

Rust calls a narrow versioned C ABI. The initial interface supports create, prepare, train step, metrics query, checkpoint/export, and destroy. Handles are opaque. Every call returns a stable status code and writes bounded diagnostic text through an explicit error-query function.

No Rust callback, allocation, file I/O, CPU readback, dynamic library-plan construction, or device-wide synchronization is allowed inside the steady-state step.

## Runtime shape

`prepare` validates SM capability and capacity, selects algorithms for the exact shapes, allocates one persistent device arena plus library workspaces, creates streams/events, uploads immutable puzzle data, initializes parameters, and warms all selected operations.

`train_step` consumes only prepared state and a step index. The GPU generates the rank-zero random-walk batch, executes the complete original model, computes MSE and gradients, applies AdamW, and records device-side timing events. Metrics are copied only at configured observation boundaries.

CUDA Graph capture is optional and may be retained only after an uninstrumented paired benchmark demonstrates a stable improvement. It is not the first implementation milestone.

## Development and acceptance targets

Local development uses the existing `mgt-gpu-queue` container on RTX 3070 Laptop GPU, CUDA architecture 86, and available Nsight Systems/Compute tools. Every GPU command goes through the queue.

Local profiling identifies launch gaps, synchronization, GEMM selection, memory traffic, and dominant kernels. RTX 3070 results are diagnostic only.

The production result is selected independently on one Tesla T4, architecture 75. SM86 algorithm choices, graph decisions, throughput, and capacity do not authorize SM75 defaults.

## Verification sequence

1. CPU tests validate configuration, aligned layout, arena capacity, ABI versioning, and failure behavior.
2. A deterministic small-shape CUDA test compares complete forward/backward/AdamW behavior with the FP32 oracle.
3. A production-shape one-step test validates all 16 blocks, BatchNorm state, finite values, and inert padding.
4. A short training test demonstrates decreasing loss and checkpoint/resume equivalence.
5. An unprofiled warm benchmark records step Q50/Q95, samples/s, memory, and FLOP estimate.
6. Nsight Systems attributes the full critical path; Nsight Compute is used only for the dominant one or two kernels.

Correctness runs and performance runs are separate artifacts. Profiled timing is never used as headline throughput.

## First implementation slice

The first vertical slice adds the versioned C ABI and Rust client around an allocation-free prepared trainer lifecycle. It runs the original p888 contract for one deterministic small batch using the existing validated BatchNorm and linear primitives. It does not yet add fusion, graphs, autotuning, NCCL, or full-workload training.

Success means Rust can prepare and execute one correct CUDA step through the ABI, retrieve structured metrics, and destroy all resources; the same native step remains directly testable from C++.

## Explicit non-goals

- Multi-GPU/NCCL behavior.
- Changing the original architecture, BatchNorm semantics, label distribution, optimizer, or training length.
- Reusing A100 BF16 policy as a T4/RTX policy.
- Python or Rust in the per-step data plane.
- Headline performance claims from RTX 3070 or profiled runs.
