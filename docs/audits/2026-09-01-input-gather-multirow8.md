# Eight-row input-gather audit

Date: 2026-09-01. Target: original p888, batch 4096, RTX 3070 Laptop /
SM86, CUDA Graph mode. Baseline source:
`c7a759cb7421c6193c0e2179b24a2a05d4f4ea63`.

## Decision

Accepted for the SM86/u32 input-half dispatch. One 128-thread CTA now owns eight
independent rows for one 256-feature tile. It stages all eight state-offset rows
in shared memory and keeps eight `float2` accumulators per lane. This exposes
load/add instruction-level parallelism across rows without changing a row's
bias-first, position-major `__fadd_rn` sequence or its coalesced half2 feature
loads.

Rows below eight retain the previous one-row CTA. The u64 fallback and the
non-SM86 production paths are unchanged. Row tails are predicated, output
padding remains positive zero, and inputs are read-only.

## Candidate selection

The exact CUDA test covers rows/CTA 1, 2, 4, and 8 over full and partial feature
tiles, row tails, logical padding, half extremes, accumulation order, immutable
inputs, and canaries. Job `94e44394e49f` was the TDD green after the expected
undefined-symbol red in `2b8bfa16f605`.

Job `8eb53bcd5e4a` used a deterministic p888-like fixture with 24 reachable
values per position. Its isolated event sweep measured 1839.57, 1806.12,
1743.15, and 1671.77 us for rows/CTA 1, 2, 4, and 8 respectively. A same-CTA
deduplication experiment took 4290.33 us and was deleted. The benchmark fixture
models p888's per-position cardinality; it is not a claim that its generated
states are exact random-walk samples.

Production build and graph gates passed in job `896e696f5464`. Frozen binaries:

| Variant | Path | SHA256 |
|---|---|---|
| Baseline | `/tmp/mgt-compact-active-bin` | `61d6b5055952164f1bcd04f81380b001d81307e9f1705fa62b01230e0b57e067` |
| Candidate | `/tmp/mgt-input-multirow8-bin` | `84f58bc080634ca4d3073b97bdaae321293c723bff26e10a7a69e3dfd8d9516c` |

## Paired timing

Job `a1d69bd77ce0` thermally gated every process to an 80--81 C start, then ran
strict ABBAAB. Each process verified its frozen-binary hash and ran original
p888, batch 4096, 140 warmup steps, and 100 timed graph steps. All retained
telemetry samples were exactly 780 MHz.

| Variant | Run means, ms | Median, ms | Samples/s at median |
|---|---|---:|---:|
| Baseline | 20.2077, 20.0992, 20.1030 | 20.1030 | about 203,751 |
| Candidate | 20.0383, 19.9477, 19.9476 | 19.9477 | about 205,338 |

The absolute median delta is **-0.1553 ms/step**. Latency falls by
**0.772522%** and throughput rises by **0.778536%**.

- Paired JSONL SHA256:
  `be20efcdb48b9f72944561be62f52b6db858bb92c7b785e5159cf3640e444c6c`.
- Summary SHA256:
  `11227a8a3d52c83dc6bdd142f25b28e0f40d6fb99e233457fa9cca2b7d093119`.

Jobs `9de89ba96c64`, `9126b76be4e7`, and `daae7897517e` were rejected because
the laptop crossed clock states or thermally throttled. Job `76c65c010847` was
rejected by the benchmark's epoch-bound configuration gate. A temporary clock
lock attempt in `971a8e75642d` returned unsupported and reset safely. None of
those series contributes to the speed claim.

## Nsight mechanism proof

Nsight Systems job `8de9b8993489` used warmed frozen binaries. The strict
analyzer passed exact graph-event order and identical metadata for every
unrelated kernel. Each retained step has 347 kernels, 71 memsets / 156,136
bytes, and zero copies.

| Mean over graph steps 2--4 | Baseline, ms | Candidate, ms | Delta, ms |
|---|---:|---:|---:|
| Input gather | 1.802263 | 1.645905 | -0.156358 |
| Kernel sum | 20.088207 | 19.925686 | -0.162521 |
| Step span | 20.635177 | 20.473522 | -0.161654 |

The gather grid changes from `(4096,10,1)` to `(512,10,1)`. Both use 128-thread
blocks. Registers/thread change 39 to 40 and static shared memory changes 320 to
2560 bytes. No local-memory spill exists. Baseline SQLite SHA256 is
`f222d3335f5e7162ab509866fcfcd5c7a3f3f694e19b3075779295bcd14de0a8`;
candidate SHA256 is
`19ec43339e37c5c91aab22e43ef4459bde8605951a9e36ebf0e6298539e6333f`.
Nsight perturbs absolute timing; unprofiled ABBAAB owns the published percentage.

Nsight Compute job `cc54bd18d10b` confirms the intended latency-hiding
mechanism on the same active24 shape:

| Metric | One row/CTA | Eight rows/CTA |
|---|---:|---:|
| Executed instructions | 123,703,296 | 137,405,440 |
| Long-scoreboard stalls / issued instruction | 14.7226 | 9.6087 |
| Eligible warps / scheduler cycle | 1.1832 | 1.5282 |
| Issue-active instructions / cycle | 0.47 | 0.66 |
| Achieved occupancy | 94.06% | 96.11% |
| L2 sector hit rate | 99.18% | 98.86% |

The candidate performs more bookkeeping instructions but exposes enough
independent row work to reduce memory-dependency stalls. NCU replay-order
durations are cache-distorted and are intentionally excluded from timing.
Baseline CSV SHA256 is
`cc8e4810c60854d8357167e6ea8f502845131a3e14ea93a9863b59949f4e83fd`;
candidate SHA256 is
`9f90b8d0ad29f7c791b365cf6ca65d70ac6662d9d05a3769576fe3b43211b203`.

## Correctness and platform gates

All GPU work ran through `mgt-gpu-queue`; job IDs map to retained
`.gpu_queue/logs/<id>.log` files.

| Job | Outcome |
|---|---|
| `7109a40c769c` | Isolated gather memcheck, initcheck, racecheck, and synccheck passed: 5 cases / 30 exact variants, zero leaks, errors, or hazards. |
| `11dd43b8346d` | Production dispatch and full p888 graph memcheck/initcheck passed with zero leaks and zero errors. |
| `51215e3103b0` | Final SM86 CPU/CUDA/NCCL/graph targeted regression passed 37/37. |
| `7c97d963c3cc` | Exact `sm_75` build plus gather, local-dispatch, lifecycle, full-step, and graph tests passed 5/5; production runtime smoke passed on the available SM86 device. |

The SM75 gate proves compile and compatibility behavior, not T4 performance.

## Boundary and next target

This checkpoint proves exact per-row arithmetic, tail and padding semantics,
graph preservation, SM86 end-to-end speed, and SM75 code compatibility. It does
not claim T4 timing, multi-GPU scaling, or convergence.

After this change the sparse input-gradient consumer remains about 3.600 ms,
the FP16-mirror Adam kernel about 1.403 ms, and the largest GEMM families about
1.68 and 1.65 ms. The next experiment should reduce the sparse consumer's
active-row work; if that cannot beat its exact ordered implementation, the Adam
mirror is the next independent target.
