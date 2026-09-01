# Compact structural input-gradient audit

Date: 2026-09-01. Target: original p888, batch 4096, RTX 3070 Laptop /
SM86, CUDA Graph mode. Baseline source: `73f6b4fcc2a2d43c49c3ac44d262a00145be91b3`.

## Decision

Accepted for the trainer-owned local-FP16 path. Original p888 has only 24
structurally reachable values at each of 72 positions, so only 1728 of the 5184
input-table bins can receive a gradient. A host-side move-orbit analysis now
builds a sorted position-major `uint16_t` map of those bins. The graph uploads it
once and builds row lists in compact active-index space.

The consumer still accumulates each reachable bin in the original row order and
writes it to its full-table address. Therefore active outputs are bitwise equal
to the previous grouped-row path. Inactive outputs are never touched after the
trainer's initial full-arena zero. This is safe because active bins are fully
owner-written on every step, Adam reads but does not write the gradient table,
all other gradient owners are disjoint, fallback performs a full memset, and a
failed trainer cannot be reused.

The compact path is gated on packed-u16 adjacent-pair dispatch, a non-empty
proper structural subset, a valid device map, and the trainer's persistent-zero
promise. Public low-level callers and full-orbit puzzles retain the old full
overwrite path.

## Memory and launch shape

| Metric | Baseline | Candidate | Delta |
|---|---:|---:|---:|
| Structural bins | 5184 | 1728 | -3456 |
| Grouped-row scratch, bytes | 42,488,064 | 14,162,688 | -28,325,376 |
| Trainer arena, bytes | 744,001,024 | 744,011,520 | +10,496 |
| Builder grid / block | 648 / 256 | 432 / 128 | compact map |
| Consumer grid-x / block | 1296 / 128 | 864 / 64 | two active bins/CTA |

The scratch aliases the existing BN workspace, so the 28.3 MB logical reduction
does not shrink the arena. The only arena cost is the aligned 10,368-byte map
slice, producing a 10,496-byte total-layout increase.

## Correctness and platform gates

All GPU work ran through `mgt-gpu-queue`; job IDs map to retained
`.gpu_queue/logs/<id>.log` files.

| Job | Outcome |
|---|---|
| `149449eb0add` | CPU TDD red: production helper was undefined. |
| `8f65ed6a9e6f` | CPU green: exact production oracle, sorted 1728-bin map, full-orbit fallback, and invalid-config/puzzle cases passed. |
| `dcbc9e554fcf` | CUDA TDD red: compact builder and consumer symbols were absent. |
| `31e450f61b74` | CUDA green: bitwise active results, ordered row IDs, tails 1/31/32/33/257, owner writes, persistent-zero reuse, immutable inputs/map, and canaries passed. |
| `2c3a4079d7b6` | Native graph full/tail tests verified the uploaded map and bitwise `+0` for all 3456 impossible bins after graph/eager steps and queued reuse. |
| `0571b9518606` | Isolated compact-kernel memcheck, initcheck, racecheck, and synccheck passed with zero findings. |
| `9ef713693ada` | Production p888 graph memcheck and initcheck passed with zero leaks and zero errors. |
| `c377d7a21c47` | Final SM86 targeted CPU/CUDA/NCCL/graph regression passed 37/37. |
| `89554e3886a7` | Clean Rust `single_gpu_ffi` test passed 1/1. |
| `880c92d0d285` | Exact `sm_75` build selected the SM75 tensor-op policy; helper, compact kernel, lifecycle, graph, and FP16 full-step tests passed 5/5, followed by a production runtime smoke on the available SM86 device. |

SM75 job `78d2a93a7e1e` was discarded after Docker returned a bind-mount `EIO` and
the queue container exited 255. Docker reported `OOMKilled=false`; the restarted
queue recovered the stale job, and the clean retry above passed. This was an
infrastructure interruption, not a code failure. The SM75 gate proves compile /
runtime compatibility, not T4 speed.

## Frozen binaries and paired timing

| Variant | Frozen path | SHA256 |
|---|---|---|
| Baseline | `/tmp/mgt-residual-beta1-bin` | `893e5be07f0884968a09011da761a7d96f4f2ba37bc352d64cc228dd4a993763` |
| Candidate | `/tmp/mgt-compact-active-bin` | `61d6b5055952164f1bcd04f81380b001d81307e9f1705fa62b01230e0b57e067` |

Job `8aedfa8003f5` used four baseline preheat passes, then strict ABBAAB order.
Every process checked its frozen-binary hash and ran original p888, batch 4096,
140 warmup steps, and 100 timed graph steps. All retained telemetry samples were
at 780 MHz.

| Variant | Run means, ms | Median, ms | Samples/s at median |
|---|---|---:|---:|
| Baseline | 20.3371, 20.3346, 20.3335 | 20.3346 | about 201,430 |
| Candidate | 20.2014, 20.1972, 20.1971 | 20.1972 | about 202,800 |

Latency falls by **0.675696%** and throughput rises by **0.680292%**. The
absolute median delta is **-0.1374 ms/step**.

- `paired-compact-active-stable.jsonl` SHA256:
  `cc12912353376aa7fc04df9da7b7bc5fae680d01ea05950e21ec4b1aa188cee`.
- `paired-compact-active-stable-summary.json` SHA256:
  `5f81f00d7d5bdf2b603bfe46df7c1b4282a4f67e581c1cbbaf65a028c1c2888f`.

Job `d3310ddb8f66` retained the first series but it was rejected: one candidate
process was externally stalled at 68.9036 ms and baseline telemetry crossed
780/825/975/1095 MHz. It does not contribute to the speed claim.

## Nsight mechanism proof

Nsight Systems job `dac935731c5c` used warmed frozen binaries. The strict analyzer
found identical graph-node order and unchanged metadata for every unrelated
kernel: 347 kernels, 71 memsets / 156,136 bytes, and zero copies per graph step.

| Stage | Baseline, ms | Candidate, ms | Delta, ms |
|---|---:|---:|---:|
| Row-list builder | 0.226176 | 0.094939 | -0.131237 |
| Sparse consumer | 3.605793 | 3.587378 | -0.018415 |
| Combined sparse stage | 3.831969 | 3.682316 | -0.149653 |
| Mean graph-step span | 20.834474 | 20.717447 | -0.117027 |

Baseline SQLite SHA256 is
`86ad6b5d301dd0defec84e2aff3d45f504716fe2abf051ac129a22f30195bdfe`;
candidate SHA256 is
`ccf885563eb9e51b65639ceb914ce3706948014edd22fce36b6f1899638890ee`.
Nsight perturbs absolute timing; unprofiled ABBAAB owns the published percentage.

Nsight Compute job `dd247fcbd262` isolated the builder:

| Metric | Baseline | Candidate |
|---|---:|---:|
| Duration, us | 224.576 | 95.968 |
| Executed instructions | 14,263,335 | 5,173,632 |
| Registers/thread | 30 | 21 |
| Allocated registers/thread | 32 | 24 |
| Achieved occupancy | 90.96% | 89.32% |

The win is reduced structural work, not higher occupancy. Baseline CSV SHA256 is
`4cc423199a5a0d761fd886e2f88b3f42db9ee9fa411c0762a3f9a268cb9f2b5f`;
candidate SHA256 is
`28f682cba18cf808a0bae2bb84a9700fca7c96f6db6d3f025e080108f0f6c4d6`.

## Boundary and next target

This checkpoint proves exact local single-GPU dataflow, persistent-zero alias
semantics, SM75 code compatibility, CUDA Graph behavior, and SM86 speed. It does
not claim T4 timing, multi-GPU scaling, or convergence.

After compaction the sparse consumer remains about 3.587 ms, input gather about
1.783 ms, and FP16-mirror Adam about 1.403 ms. Earlier exact register row scans
and dense-half-GEMM input gradients were materially slower and remain rejected.
The next audit target is the input-gather dataflow; any new sparse-consumer design
must reduce the active-row work itself rather than only removing impossible bins.
