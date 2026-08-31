# Fixed-shape CUDA Graph feasibility

Historical prototype phase. Production integration and final validation are in
the [native graph lifecycle audit](2026-08-31-single-gpu-native-graph.md).

Date: 2026-08-31. Original p888, RTX3070 Laptop / SM86, CUDA12.8.93,
driver572.70, Docker `mgt-gpu-queue`. No subagents used.

## Decision and scope

The production change preserves a caller-provided cuBLAS workspace when a handle
is already bound to the requested stream. It does not allocate a new workspace
or enable graphs in the default trainer. C ABI, Rust API, arena query and model
precision are unchanged. Nonlocal MLP stream binding still calls the original
`cublasSetStream`; a real stream change in the local helper also retains cuBLAS's
workspace-reset semantics.

The replay implementation is deliberately test/tool-only. It proves that one
fixed-row training graph can process changing data/epoch/Adam parameters, with
no repeated capture. It is not a production graph lifecycle or a T4 port.
The graph diagnostics are built only with CUDA>=12.8; the helper does not raise
the production toolkit floor.

## Why the workspace fix is required

An initial capture had three allocation/free node pairs. Whole-graph update was
rejected. Binding a4MiB workspace alone did not remove them: the trainer called
`cublasSetStream` repeatedly, discarding that workspace even for the same stream.
`detail::BindBlasStream` first obtains the current stream and returns immediately
when it already matches. Explicit workspace then removes all six nodes.

This behavior is documented by NVIDIA: setting a stream resets user workspace;
explicit workspace avoids cuBLAS graph allocation/free nodes. Four MiB is the
recommended workspace size for the architecture class containing SM86.
[cuBLAS12.8 documentation](https://docs.nvidia.com/cuda/archive/12.8.0/cublas/index.html#cuda-graphs-support)

The caller still owns workspace lifetime and must rebind after a real stream
change. This patch does not pretend to preserve workspace across that change.

## Replay design

The test support includes the private trainer, RandomWalk and Adam translation
units; their objects are not pulled a second time from the static archive. This
keeps private stream/arena access out of the exported API. It is not the layout
to use for eventual production integration.

One capture contains363kernels,71memsets and68DtoD copies at rows4096. The small
rows4 path contains332kernels. Capture enqueues no device work: the diagnostic
restores the launch function's host sequence/in-flight fields after capture,
then instantiates the graph. Every actual replay records completion afterward.

Exactly three kernel nodes are selected by function identity:

| Node | Changed arguments | Preserved arguments |
|---|---|---|
| RandomWalk | config sample offset, epoch, optimizer step | row count, seed, rank, move/target/output pointers |
| AdamWWithHalfMirror | config optimizer step | parameter count, optimizer constants, all buffers |
| AdamW | config optimizer step | affine count, optimizer constants, all buffers |

Captured configurations are copied into typed local values; local argument
arrays replace only these entries. The source graph and its parameter storage
remain alive. `cudaGraphExecKernelNodeSetParams` updates the executable, not the
source node's owned storage. Requests with a different row count, invalid
sequence or out-of-epoch slice are rejected before launch.

The initial Runtime API query failed on cuBLAS's driver-loaded kernel nodes.
The final diagnostic uses Driver API enumeration and `cudaGetFuncBySymbol` to
identify our three nodes, then obtains Runtime parameters only for those nodes.
It does not swallow CUDA errors or identify kernels by name/order. All closed
cuBLAS nodes are left unchanged.
[Driver graph parameter ownership](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-driver-api/group__CUDA__GRAPH.html),
[Runtime/Driver interoperability](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-runtime-api/group__CUDART__DRIVER.html)

Individual node updates are appropriate when only a few parameters change;
they affect future launches, not executions already queued. The test explicitly
checks this by enqueueing four changing requests before synchronization.
[NVIDIA dynamic-graph design](https://developer.nvidia.com/blog/constructing-cuda-graphs-with-dynamic-parameters/)

## Correctness and negative controls

- Cold and warm capture/whole-update at rows4: model-state bytes, data and loss
  match eager; rows4096 captures have exact data/meta and finite metrics.
- Capture-once replay at rows4 and256: eight real optimizer steps, changing
  offsets/epochs, final legal epoch slice, exact weights/gradients/moments,
  FP16 weight mirror, BN affine/gradients/moments/running state, outputs and loss.
- Rows4096: same request sequence and queued-launch test, exact states/labels/
  walk metadata, finite metrics. Cross-CTA BN atomics do not promise full-model
  bitwise identity; no long-run convergence claim is made.
- Three independent negative controls omit the walk, weight-Adam or affine-Adam
  update. Each agrees at step1 and is detected at step2 in its expected field.
- Invalid requests leave the trainer sequence and caller ticket unchanged.
- Stream helper covers null handle, repeated same stream, real stream changes
  and rebinding to the default stream. Explicit-workspace capture is the
  regression against accidental same-stream workspace reset.

Queue evidence (logs under `.gpu_queue/logs/<id>.log`):

| Job | Result |
|---|---|
|75a14e3527aa|Initial diagnostic:3alloc+3free; second whole-update rejected. Not a production failure.|
|8f5a1162e26a|Meaningful workspace RED: explicit workspace still lost by old stream binding.|
|a0aae41f4f75|Workspace GREEN: no alloc/free; cold/warm small and production update pass; memcheck0.|
|2fa83fb23490|Test scaffolding compile error, corrected; not semantic RED.|
|6ba1d0055dc2|Diagnostic Runtime query rejected a foreign cuBLAS function; not semantic RED.|
|99acda70e433|Meaningful replay RED: stale captured request produces `byte mismatch: states`.|
|ac36ae939e38|Capture/replay CTests, rows256/4096, all three negative controls, queued launches and memcheck pass.|
|9bfc13f96eeb|24CUDA+4CPU/C ABI+1Rust tests; old-path sanitizer gates; graph init/race/sync checks; capture mem/leak check all pass.|
|974a34857276|Malformed expected SHA rejected before any paired measurement. Four preconditioning runs retained.|
|b1a7167be20c|Three complete ABBAAB comparisons, all18runs retained; measurement helper tests pass.|
|2ccab7bd6bf4|Final CMake rebuild matches both frozen binaries; graph/FP16 CTests and three measurement-helper tests pass; confirmation A/B and strict Nsight audit pass.|

The full gate includes activation-tape mem/init, local and nonlocal gradient
overwrite memcheck, a production4096 eager memcheck, BN bias mem/init/race/sync,
graph replay mem/init/race/sync and graph capture mem/leak checks. No reported
sanitizer errors, leaks or race hazards. These are targeted gates, not every
historical repository test or a multi-GPU regression.

## Unprofiled measurements

Each run: batch4096,140warmup,100timed optimizer steps, original production
inputs, same seed and monotonically changing sample offset. All240steps fit in
one999978-sample epoch. Timing includes launch/update overhead and final GPU
completion, excludes initialization/capture. These are medians of three run
means, not medians of individual steps. ABBAAB retains all attempts.

| Comparison | A run means, ms | B run means, ms | Median A → B, ms | Throughput change |
|---|---|---|---|---|
|Default production → stream helper|22.5562,22.4270,22.4359|22.4546,22.4379,22.4389|22.4359→22.4389|−0.0134%|
|Probe eager default → explicit workspace|22.4351,22.4305,22.4361|22.4371,22.4295,22.4340|22.4351→22.4340|+0.0049%|
|Probe eager workspace → graph|22.4321,22.4584,22.4432|21.3606,21.3629,21.3588|22.4432→21.3606|+5.0682%|

Graph saves1.0826ms/step in this series (4.8237% latency), approximately191.8k
samples/s. Active samples(util>=90%, memory>0) are780MHz on both graph/eager
sides; temperatures overlap84–86C vs83–86C. This is sampled state, not locked
clocks. The workspace-control series includes one885MHz active sample; its
near-zero change is not used as a controlled performance claim.

The benchmark's `memory_bytes` is owned arena plus optional4MiB workspace:
739806720→744001024bytes. It is not total process VRAM or CUDA graph metadata.
Capture/instantiate time is separately reported; it is not free and must be
amortized across training steps.

An independent six-run confirmation retained:

- Eager workspace:22.5560,22.4361,22.4366ms; median22.4366ms.
- Graph:21.4649,21.4653,21.3566ms; median21.4649ms.
- **0.9717ms lower latency /4.3309% latency reduction /4.5269% throughput gain**.
- All active samples780MHz; eager76–82C, graph79–82C. No sample filtering beyond
  the declared activity filter and no outlier removal. The first series is not
  pooled with this confirmation or replaced by it.

## Nsight Systems: unchanged work, shorter gaps

Fresh traces use Nsight Systems2025.6.3 with
`--trace=cuda,nvtx,osrt,cublas --cuda-graph-trace=node --sample=none --cpuctxsw=none`,
100warmup+4measured steps. Both variants use the same frozen probe and explicit
workspace. Means below use measured steps2–4, not unprofiled throughput.

| Metric | Eager workspace | Graph |
|---|---:|---:|
|Kernels/step|363|363|
|Sum kernel time, ms|21.069226|21.047577|
|First RandomWalk→last Adam span, ms|22.951857|21.894113|
|GPU activity union, ms|21.390861|21.363623|
|Uncovered within span, ms|1.560996|0.530490|
|Sparse dW, ms|3.682253|3.733568|
|Memsets / bytes|71 /156136|71 /156136|
|DtoD copies / bytes|68 /78000|68 /78000|

Strict comparison passes for every measured step: exact kernel names, grids,
blocks, dynamic/static shared memory, register/local-memory metadata, transfer
kinds/bytes/values and event order. The complete graph trace records one capture,
one instantiation,104launches and312individual node updates. Exactly three
Runtime kernel-parameter queries are made after Driver identity matching.

The observed1.0577ms span reduction mostly follows a1.0305ms reduction in gaps;
summed kernel duration changes by only0.0216ms. This supports a scheduling
benefit without attributing the win to faster GEMMs. Uncovered time is an
interval-union measurement, not a claim that every gap is CPU idle or removable.
Small reported activity overlaps are handled by union, not double-counted.

The remaining largest single kernels are sparse input dW(3.734ms), input gather
(1.825ms) and weight Adam(1.404ms). Graph replay does not solve their device-side
cost; it must not be described as ideal GPU saturation.

## Frozen artifacts

Inputs: group SHA256
`f2d7cae9a387d8acbe7e4082711179dc5a309232e4278733c90853534c649e02`;
target `107de2bc788e11029f7851f8e1b0b5afb4e34379c709fc840689ebd3d1f51b5b`.

| Binary | SHA256 |
|---|---|
|`/tmp/mgt-bn-input-bias`|`b16ac6c32d93fabbf6d1f028e059e14d1acc6fc9d9cce901c50116f119b946bb`|
|`/tmp/mgt-blas-stream-preserved`|`235d03f34f8dd6da52c12aebd1c125a4114393d96cde785d0120c424fe78ebda`|
|`/tmp/mgt-graph-three-node-probe`|`8c2a3d63e2c29da8d14d26e8d58d761d03ee8838e731d7cc09234810064411a0`|
|`/tmp/mgt-graph-workspace-green`|`7e808dc8388c923ae4064c3ca2b5c4de3523ae84ffca1932ccaa92854597a01c`|
|`/tmp/mgt-graph-replay-stale-red`|`4320a5faf96e30f0edbcbdaf7cdf77ec18b11d817d9e564491d2fb710c001fbe`|

Local retained measurement files in `test_results/graph_probe_20260831/`:

| JSONL | SHA256 |
|---|---|
|`paired-helper.jsonl`|`453c0f5bceee9838e219f856f73d4345200308ef87d401c010cf4212fec35a95`|
|`paired-workspace.jsonl`|`fc2b52ddf142665ad09428e8c775356eea6a8d9e2896732f76fd531d33dc9dab`|
|`paired-replay.jsonl`|`77305e2d5a2dfbfc943a00d2086b2307f2733942b5524c3bfbb26bad4210e053`|
|`paired-replay-confirm.jsonl`|`e8870ed572e2388132d2ce4724f323d9bd15f2367b693f7b5b2e3465dcea6958`|

Fresh SQLite trace hashes:

- `eager-workspace-warmed.sqlite`:
  `538b2458b52dcbac62596584f23cba64a2dd12fa60cab403feb1f89fcb559516`.
- `graph-warmed.sqlite`:
  `cdd4ecc69e1873261e35b194b04a56d11115b5235251d0b71503b09bdb9ca41b`.

`replay-profile.json` retains per-step sequences, metadata, runtime counts and
timing summaries. The measurement validator rejects dropped/reordered runs,
wrong hashes/commands/modes/bytes, failed or nonfinite metrics, changed hardware
and missing activity samples. Trace comparison negative controls reject altered
kernel/event sequences and metadata.

## Production integration boundary

A measured prototype does not yet make graphs the default. The next integration
must separate enqueue-only work from host ticket bookkeeping, own workspace in
the queried arena/lifecycle, expose an explicit supported mode without reusing
reserved C ABI fields, handle full batches and epoch tails, and test Rust/C ABI
failure/lifetime behavior. Captured pointers must outlive the executable graph.
No partial-state fallback after a failed capture/update is permitted.

No plugin release, ideal GPU saturation, MXFP8, multi-GPU or T4 result is claimed.
