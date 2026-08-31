# Fixed-shape training graph feasibility

## Scope

Determine whether a CUDA Graph can replay the original SM86 p888 training step
with fresh input and optimizer parameters, without recapturing the graph each
step. This is a diagnostic gate, not a new public trainer mode.

## Contract

- Docker `mgt-gpu-queue`, CUDA 12.8, RTX3070 Laptop / SM86 only.
- Original p888, batch4096 for performance; FP32 master/Adam/BN and FP16 dense
  operands remain unchanged. No frozen-batch benchmark.
- Preserve the C ABI, Rust API, arena query, default eager path, and nonlocal
  cuBLAS stream-reset behavior. No T4 or convergence claim.
- Same-stream binding must preserve a caller's cuBLAS workspace. A real stream
  change still calls `cublasSetStream` and retains its documented reset behavior.
- Allocate an explicit 4 MiB workspace outside capture in the diagnostic only.
- Identify dynamic kernels by actual function address, not node order/name.
  Require exactly one RandomWalk, one AdamWWithHalfMirror, and one AdamW node.
- Copy typed captured configs and update epoch, sample offset, optimizer step.
  Fixed active rows only; reject shape changes. Record the completion event
  after graph launch. Do not mutate trainer bookkeeping during replay setup.
- No graph allocation/free nodes. No recapture in the timed loop. Check exact
  data/meta each step, complete model-state bytes for single-CTA small cases,
  finite production metrics, and sanitizer results.
- Paired unprofiled comparisons must separate the workspace effect from graph
  replay. Preserve all attempts and report run means, not per-step medians.

## Evidence and sources

The diagnostic failed with three allocation/free pairs despite a supplied
workspace; preserving same-stream binding removed all six nodes and allowed
whole-graph update. Queue jobs: RED `8f5a1162e26a`, GREEN `a0aae41f4f75`.

NVIDIA documents that `cublasSetStream` resets workspace even when the stream is
unchanged, and that an explicit workspace avoids cuBLAS allocation/free graph
nodes. CUDA permits updating individual nodes between launches; these updates
do not modify already enqueued executions.

- [cuBLAS 12.8](https://docs.nvidia.com/cuda/archive/12.8.0/cublas/index.html#cuda-graphs-support)
- [CUDA 12.8 graphs](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-c-programming-guide/index.html#cuda-graphs)

## Decision gate

Only the workspace-preserving helper and its regression test are eligible for
production in this phase. Replay remains test/tool-only until repeatable
correctness and throughput evidence justify a separate lifecycle/API design.
