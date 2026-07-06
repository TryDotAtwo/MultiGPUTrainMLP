# Native trainer optimization plan

Date: 2026-07-06
Baseline: Kaggle 2xT4 v8 best `b53248_t48_ws16m_bucket4m`, 511027.88 states/s, step 208.402 ms, backward 189.710 ms, allreduce tail 14.622 ms.

## Rules

- Keep the trainer generic for different puzzles and MLP shapes.
- Do not hardcode p888-specific dimensions in kernels or scheduling code.
- Preserve online random-walk training; no shared dataset on disk.
- Keep correctness checks next to every speed change.
- Prefer fixed buffers and precomputed plans in the steady-state path.
- Measure on local Docker first when meaningful, then on real Kaggle 2xT4 for T4 claims.

## Plan

1. Profile the current best `tile=48` configuration.
   - Add a profile row for `b53248_t48_ws16m_profile`.
   - Run the same correctness gates before profiling.
   - Record stage timings for input forward, residual forward/backward, hidden backward, input-gradient GEMM, allreduce and Adam.

2. Add cuBLASLt autotune/cache for all MLP GEMM shapes.
   - Build the autotune around manifest-derived shapes, not one puzzle.
   - Cache selected algorithms by shape, transpose flags, precision and workspace budget.
   - Keep a safe default path when no candidate passes `cublasLtMatmulAlgoCheck`.
   - Verify loss and gradients against the existing mixed-precision correctness tests.

3. Improve gradient allreduce overlap by bucket readiness.
   - Emit buckets as soon as their owner range is complete.
   - Keep deterministic bucket order across ranks.
   - Avoid forcing compute streams to idle-wait on communication streams.
   - Measure visible allreduce tail, not only total NCCL time.

4. Specialize residual-block backward without losing generic model support.
   - Use the layer manifest to select kernels per supported block type.
   - Reduce unnecessary activation/materialization traffic.
   - Keep fp32 accumulation where required for training stability.

5. Add CUDA Graph capture for stable train-step templates.
   - Capture after `TrainPlan` allocation and cuBLASLt plan selection.
   - Keep dynamic config changes outside graph replay.
   - Fall back to normal launch only for unsupported debugging/profile modes.

6. Evaluate bf16/fp16 precision modes separately for A100/H100.
   - Keep fp16 as the T4 default.
   - Add bf16 only where hardware and correctness checks support it.
   - Compare against the float32 oracle on small models and against loss stability on full-model smoke runs.

## Current next action

Start with item 1, then use the profile to choose the exact first implementation slice for item 2.