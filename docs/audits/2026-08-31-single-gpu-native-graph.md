# Native single-GPU graph lifecycle and asynchronous Rust API

Date: 2026-08-31. Original p888, RTX3070 Laptop/SM86, Docker queue only.
Follows the [graph feasibility audit](2026-08-31-single-gpu-graph-feasibility.md).

## Accepted scope

Fixed-capacity CUDA Graph replay is now an explicit native/C ABI/Rust option.
The legacy path remains eager. Smaller valid batches use eager on the same
stream; graph failures never trigger a retry or silent eager fallback. Model,
random-walk schedule, FP32 master/Adam/BN and FP16 dense operands are unchanged.

The arena is739806720 bytes eager or744001024 bytes graph at batch4096.
The difference is an aligned4MiB cuBLAS workspace, not graph metadata or total
VRAM. CUDA>=12.8 is the validated build floor, not a claim that every used API
was first introduced in12.8. Other GPU/toolkit combinations remain unmeasured.

## Ownership and data flow

1. Create validates mode and allocates the existing stream/event/BLAS/arena.
2. Prepare initializes model/state, binds the workspace, captures the unchanged
   enqueue-only training body and instantiates one executable graph. Capture
   leaves sequence0 and model bytes equal to eager initialization.
3. Each full-capacity request changes exactly three kernel nodes: RandomWalk
   receives epoch, sample offset and optimizer step; the weight/mirror Adam and
   affine Adam nodes receive the new step. All data pointers remain resident.
4. A smaller request executes the same eager body. Subsequent full requests
   replay the original executable without recapture, including epoch rollover.
5. Record the completion event and advance the ticket only after successful
   enqueue. Metrics explicitly drain the latest submitted step and copy one loss.
6. Destroy drains the stream, releases executable/source graph, then BLAS,
   arena, event and stream. Workspace outlives BLAS cleanup.

Kernel identity uses function handles, not node order or textual names. Driver
enumeration handles private cuBLAS nodes; Runtime parameters are read only for
the three owned kernels. Duplicate/missing dynamic nodes or allocation/free
nodes reject preparation. The source graph owns argument storage throughout
replay. Per-launch typed copies update executable parameters without mutating
source-node storage or allocating host buffers.

Invalid row/epoch/step requests are rejected before work and leave sequence and
ticket unchanged. Capture/enqueue/update/event/metrics failures poison the
trainer until destruction. There is no replay-after-partial-update recovery.
The fail-stop test injects a logical graph-state failure before CUDA execution;
it is not a hardware-context failure experiment.

The established capture and workspace rules follow NVIDIA's
[CUDA12.8 graph guide](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-c-programming-guide/index.html#cuda-graphs),
[cuBLAS graph/workspace documentation](https://docs.nvidia.com/cuda/archive/12.8.0/cublas/index.html#cuda-graphs-support)
and [Driver graph parameter API](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-driver-api/group__CUDA__GRAPH.html).

## C ABI and Rust

The80-byte V1 config and existing step/metrics structures are unchanged;
reserved fields must still be zero. The additive16-byte
`MgtSingleGpuExecutionOptionsV1` selects eager0 or fixed-batch graph1 through
`mgt_single_gpu_v1_create_with_options`. Size/version/mode/reserved values are
validated. Legacy `create` remains eager.

Rust adds `SingleGpuExecutionMode`, `create_with_mode`, `enqueue_step` and
`read_metrics`. `enqueue_step` passes a null metrics pointer to the existing C
ABI, eliminating the wrapper's mandatory per-step synchronization/readback.
The synchronous `train_step` convenience API is retained.

```rust
let mut trainer = SingleGpuTrainer::create_with_mode(
    config, SingleGpuExecutionMode::FixedBatchGraph)?;
trainer.prepare()?;
trainer.enqueue_step(4096, 1, 0, 0)?;
trainer.enqueue_step(554, 2, 0, 999424)?;
let metrics = trainer.read_metrics()?;
```

The example illustrates dispatch, not a complete epoch traversal. Calls on one
handle must be serialized. Rust's mutable methods enforce that locally; Drop
waits for outstanding work but cannot report a cleanup error. Call
`read_metrics` to observe asynchronous execution errors before Drop.

Self-review also found an inherited teardown leak: native destruction consumed
its owner even when CUDA cleanup returned failure, while C ABI retained an
unusable wrapper. The C ABI now consumes/nulls the handle in both outcomes and
reports errors through `last_error(NULL,...)`. A test-only native backend
injects this failure without damaging a CUDA context.

## Correctness and failure evidence

All queue logs below are retained locally under `.gpu_queue/logs/`.

| Job | Outcome and scope |
|---|---|
| 89e90e881159 | Expected native RED: graph mode did not add the required workspace arena budget. |
| b2b669ef675f | Native lifecycle/capture/replay GREEN; exact model bytes at rows4/256, production4096 with4095-row tail; memcheck/leak-check clean. |
| e4e805dea9a4 | Expected C ABI RED: stub implementation accepted invalid execution options. |
| 79baff054d0a | Native graph and both C ABI tests, Rust test, C ABI memcheck/leak-check pass. |
| 12caa64f7393 | Full pre-publication gate:24 CUDA regressions,4 CPU/C ABI tests,6 graph/ABI/linear tests (sets overlap), Rust test and targeted sanitizers. |
| 264fb0448bc1 | Measurement-validator tests, rebuilt benchmark identity, explicit-eager C ABI option, C ABI memcheck, paired runs and native/Rust Nsight gates pass. |
| 1a8de368924d | Expected destroy-ownership RED: failure left a non-null consumed wrapper. Test cleans its own RED allocation. |
| 5cc306ef019b | Final native/C ABI4 CTests and Rust pass; benchmark cmp passes. Later optional host-ASan enters DEADLYSIGNAL loop; exact test process terminated, overall job143. Not a successful ASan run. |

Native tests cover full→tail→full, epoch transition, queued requests, repeated
reads, zero/skipped optimizer steps, row/epoch bounds, uint64 offset extremes,
workspace alignment/accounting, preparation without a train update and
fail-stop behavior. Complete model/gradient/moment bytes are compared at4/256
rows. At4096, exact generated state/label/meta plus finite metrics are checked;
the test does not claim full production-model byte equality.

The full gate includes the actual4096-capacity/554-row epoch tail, native graph
memcheck/initcheck/racecheck/synccheck, full4096 graph benchmark memcheck with
leak checking, activation-tape/overwrite/BN regressions and the Rust ownership
test. CUDA sanitizer reports are zero errors/hazards/leaks in these runs.
Sanitizer timings, including the multi-second instrumented steps, are not
performance measurements. Existing world1 NCCL regressions do not validate2 GPUs.

## Binary provenance

| Identity | Path | SHA256 |
|---|---|---|
| Before native integration | /tmp/mgt-blas-stream-preserved | 235d03f34f8dd6da52c12aebd1c125a4114393d96cde785d0120c424fe78ebda |
| Initial native graph | /tmp/mgt-native-graph-v1 | 408bdf0d10fed209140856683a362444a393b0c9c0cfccd912ccf50125b1ec4d |
| Final native graph | /tmp/mgt-native-graph-final | 82f7d55c236b90c224d1c06f834460ec1bf39b5e5f97a6c06f00699dd7eaaed7 |

Job52669395fe8e stopped before tests/performance because final rebuilding
changed the initial SHA. Inspection found exactly12 differing bytes, all in
three NVCC `tmpxft_*` source filenames in ELF `.strtab`; preceding sections,
including CPU machine code and `.nv_fatbin`, compare exactly. The old file was
not overwritten; final unprofiled runs use the newly frozen82f7d55c identity.

Workload hashes remain group JSON
`f2d7cae9a387d8acbe7e4082711179dc5a309232e4278733c90853534c649e02`,
target binary
`107de2bc788e11029f7851f8e1b0b5afb4e34379c709fc840689ebd3d1f51b5b`.
Driver572.70, CUDA12.8.93, Nsight Systems2025.6.3; no clock/power/driver changes.

## Unprofiled paired measurements

ABBAAB, batch4096,140 warmup/100 timed steps per process;240 full batches fit
inside one999978-row semantic epoch. Values are medians of three run means,
not medians of individual steps. All six records in every series are retained.
Binary hashes are checked before/after, along with mode, arena, finite metrics,
workload and telemetry. The validator has two tests including12 corruptions.

| Series | A run means, ms | B run means, ms | Median A→B, ms | Interpretation |
|---|---|---|---|---|
| Eager old→integrated | 22.5549,22.5538,22.5777 | 22.5576,22.5676,22.5569 | 22.5549→22.5576 | -0.012% throughput; unchanged, all sampled active clocks780MHz. |
| Initial native eager→graph | 22.5521,22.5543,22.5535 | 21.4627,21.4670,21.4604 | 22.5535→21.4627 | +5.0823% throughput; graph includes an810MHz sample, descriptive. |
| Final native, four preconditioners | 18.3982,22.5494,22.5676 | 18.3013,21.2835,21.4610 | 22.5494→21.2835 | +5.9478% descriptive only; substantial frequency drift in both variants. |
| Final native,16 preconditioners | 22.5662,22.5628,22.5563 | 21.4622,21.4666,21.4606 | 22.5628→21.4622 | **+5.1281% throughput**; graph samples780MHz, eager samples780/810MHz. |

The four preconditioning runs were insufficient to prevent the earlier final
series' frequency transition. Its low18ms values are not attributed to the
code. The separate confirmation used a predefined16 preconditioning processes
before ABBAAB, with no filtering or pooling of earlier runs. Temperatures overlap
at77–79C versus78–79C. Clocks were not locked; the isolated810MHz eager telemetry
sample makes the5.1281% figure observational rather than a fixed-clock claim.

All artifacts are under `test_results/native_graph_20260831/`:

- `paired-eager-regression.jsonl`, SHA256
  `7c75bfcddf96453eddbc0732a1d576bdbd7cb37b1999ce7290cba9c8130b5b7e`.
- `paired-native-graph.jsonl`, SHA256
  `ba7c5f7a4bb9eea4c6b0692393506409dd012515dd54d19f8822fdd5273ce2f8`.
- `paired-native-confirm.jsonl`, job06a9472a25f5, SHA256
  `7ee9840b481b3525f103cd5ae76e2eea982a4cc46713115500e4ae70c052c8b3`.
- `paired-native-soaked.jsonl`, job4f581aee95e9, SHA256
  `676977d2ff89537e49e8f910414162631234e7aaa0bb5dad04d80e6c5bf13240`.

Graph prepare includes capture/instantiate once, outside step timing. Initial
native runs report174.594/177.413/189.385ms versus eager39.136/40.383/39.760ms;
preparation cost depends on initialization and the environment, not just capture.

## Nsight proof of identical device work

Native traces:100 warmup/four timed steps. Strict comparison passes all four
steps: identical kernel names/geometry/registers/shared-memory metadata and
copy/memset kinds, byte counts and order. Exactly1 capture,1 instantiate,
104 graph launches and312 dynamic-node updates. Per step:363 kernels,
71 memsets/156136 bytes and68 DtoD copies/78000 bytes.

| Mean over measured steps2–4 | Eager, ms | Native graph, ms |
|---|---:|---:|
| Step span | 23.063710 | 21.895490 |
| Sum of kernel durations | 21.168442 | 21.047303 |
| Kernel interval union | 21.168442 | 21.036775 |
| All device-activity interval union | 21.490632 | 21.364432 |
| Uncovered span | 1.573078 | 0.531058 |
| Sparse dW consumer | 3.734461 | 3.734006 |

Interval unions handle small reported overlaps. Uncovered time is not assumed
to be entirely CPU idle. This is scheduling improvement, not fewer FLOPs or
a faster sparse kernel. The remaining graph trace costs include input gather
1.823955ms, weight Adam1.403473ms and BN-backward partials1.489203ms.

- `eager-warmed.sqlite`, SHA256
  `7e4a8dfb8313cca195b9cb2e433cfa91804eb58b15c9a1d7c7a3fef890353524`.
- `graph-warmed.sqlite`, SHA256
  `ac189f858bd629e9c599743c1105f7b0103dccdf9618dcf30a05abe6ec85cd9a`.
- `native-profile.json`: strict gate and per-step/aggregate analysis.

The actual Rust test binary was also traced:1 capture/instantiate,3 graph
replays,9 node updates and one eager tail. Only three4-byte DtoH copies occur:
legacy eager control metrics, explicit read after queued work and repeat read
after an invalid request. Final queued replay is drained by Drop without a
loss read. `rust-async.sqlite` SHA256
`b18030d28dc47ff20787ea833e9a2fde6e168cb3acf25ff91c2fde1282054f4a`;
`rust-async-summary.json` passes. This proves the Rust dispatch path, not a
separate Rust4096 end-to-end throughput benchmark.

The destroy-failure test also passes host ASan+LeakSanitizer when its recursive
SIGSEGV handler is disabled, plus standalone UBSan. The default ASan signal
handler loop in job5cc306ef019b is retained as a tooling failure, not counted as
a code failure or silently omitted. Jobc8a7e8562de3 then passes bounded
ASan/LSan, UBSan and graph C ABI Compute Sanitizer memcheck/leak-check.

## Remaining boundaries

No convergence/quality or checkpoint/resume claim: checkpoint is still explicitly
unimplemented in this API. No automatic graph selection, concurrent-host capture
policy, general dynamic-shape graph support, T4/2xT4 result or training-plugin
release. Sparse dW remains about3.73ms and is the next device-side target.
Queue logs and large traces are ignored local evidence, not bundled artifacts.
