# Explicit single-GPU training graph lifecycle

## Basis

[Feasibility audit](../../audits/2026-08-31-single-gpu-graph-feasibility.md):
capture-once, three typed node updates, original p888 on SM86, +4.53% confirmed
throughput with unchanged kernels and memory events. Promote this scheduling
mechanism into the native trainer, C ABI and Rust ownership wrapper.

## Required behavior

- Explicit opt-in `SingleGpuExecutionMode::kFixedBatchGraph`; eager remains the
  default. Do not alter the original model, data generation, precision or Adam.
- Graph support in this phase requires the validated CUDA>=12.8 build. Unsupported
  builds reject explicit graph mode before GPU allocation, never silently switch.
- Graph mode adds one256-byte-aligned4MiB cuBLAS workspace slice to the existing
  arena. `QuerySingleGpuTrainerBytes` includes it. No extra hot-path allocation.
- Create owns stream/event/BLAS/arena as before. Prepare initializes weights,
  binds workspace and captures the full-capacity shape exactly once. Capturing
  must not advance sequence, write a ticket or execute a training update.
- Separate enqueue-only training work from request validation and host completion
  bookkeeping. Both eager and capture call the same device-work body.
- Full batches update RandomWalk and both Adam nodes by typed config copies.
  Match function handles, not names or graph order. Reject duplicate/missing
  nodes and allocation/free nodes. Keep source graph parameter storage alive.
- Smaller valid batches take the existing eager path on the same stream and
  workspace. This is a declared tail path, not recovery from a graph error.
  Test full→tail→full and epoch rollover with fresh data and optimizer steps.
- Validate row bounds, nonzero/next optimizer step and epoch slice before any
  enqueue. Invalid requests leave sequence and ticket untouched.
- Capture/update/launch or completion-record failures mark the trainer failed;
  later prepare/step/metrics return failure, and destroy remains available. Do
  not replay a step, fall back or reinitialize partially modified training state.
- Completion event is recorded after actual eager/graph work. Queued steps may
  proceed without per-step synchronization; updates affect future launches only.
- Destroy drains the stream, then releases executable/source graph before arena
  and BLAS resources. All partial-construction paths retain safe ownership.
- Preserve every V1 config/step/metrics layout and reserved-zero rule. Add a
  separate16-byte `MgtSingleGpuExecutionOptionsV1` and
  `mgt_single_gpu_v1_create_with_options(config, options, out)`. Existing create
  remains eager. The new call requires valid version/size/mode/reserved fields.
- Rust keeps `SingleGpuConfig` source/layout behavior and `create(config)`.
  Add `SingleGpuExecutionMode` and `create_with_mode(config, mode)`; do not leak
  handles on errors. Existing train-step/ticket/Drop behavior remains compatible.
- Expose `mgt_single_gpu_v1_read_metrics` plus Rust `enqueue_step`/`read_metrics`.
  Enqueue passes a null metrics pointer through the existing C ABI to avoid a
  per-step GPU synchronization/DtoH copy; explicit read drains the latest step.
  Keep the existing synchronous `train_step` convenience method compatible.
- Benchmark can select mode explicitly; report it and queried arena size.
  Compare native graph against eager with unchanged workload and source hashes.

## Verification boundary

Native exact small-batch state and data comparisons, queued launches, tails,
epoch rollover, arena accounting, fail-stop state, C ABI layout/error checks,
Rust ownership and sanitizer gates precede performance claims. Then paired
unprofiled ABBAAB and fresh node-level Nsight confirm the integrated route.
Graph metadata allocation is not part of arena accounting or a total-VRAM claim.
No complete convergence, plugin publication, T4 or multi-GPU readiness claim.
