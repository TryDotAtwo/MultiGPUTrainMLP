# AdamW shared bias-correction audit

Date: 2026-09-01. Target: original p888, batch 4096, FP16 input-gradient
policy, RTX 3070 Laptop / SM86, CUDA Graph mode. Starting checkpoint:
`6ea3e02e3a0a3cd44c15f3c4410550ded8cd2d57`.

## Decision

Accepted. `AdamWWithHalfMirrorKernel` now computes the two step-wide bias
corrections once per block, publishes them through 8 bytes of shared memory,
and updates 128 parameters per block. The old kernel repeated both inlined
`powf` implementations in every thread and used 256-thread blocks.

The kernel name, argument list, graph node, FP32 update order, RN-half mirror,
optimizer state, and arena size are unchanged. Bias correction stays on the
device, so the optimization does not replace CUDA `powf` with host-libm
semantics. There is no extra kernel, copy, allocation, or graph update field.

## Selection evidence

Nsight Compute job `1130b795b5ef` measured the frozen baseline at 93.709% of
DRAM peak with 247.403264 MB read, 216.233856 MB written, and 464.122432 MB of
L2 traffic. This is already approximately the physical four-FP32-read plus
three-FP32/one-FP16-write minimum. SASS job `0772840be770` nevertheless showed
both full `powf` implementations in the per-thread hot path.

Isolated sweep job `cb8533ac1ff6` covered 128/256/512-thread launches,
per-block shared bias, host-precomputed bias, and one/two/four items per thread.
All one- and two-item variants were byte-exact over steps 1, 2, 997, and 65,535.
Four items per thread lost about 76% through strided/coalescing pressure. The
earlier persistent-grid candidate (`5df629425abe`) was also rejected after a
0.8% loss. Shared bias with one item and 128 threads was selected because it
preserves device arithmetic and reached the memory roof without host-libm or
graph-parameter changes.

## Exactness contract

`cuda_adamw_half_mirror_exact` uses 259 parameters to exercise a non-block tail
and starts weights and both moments from nonzero values. Across steps 1, 2, 997,
and 65,535 it compares the fused half-mirror path byte-for-byte against
`LaunchAdamWKernel` followed by `LaunchFloatToHalf`. FP32 weights, first and
second moments, and every half mirror value must all match.

The characterization test passed before the refactor (`bce2e101fa54`) and after
it (`85addffc9920`). Graph capture/native-graph and the exact test passed 3/3 in
candidate-freeze job `e5434edb4863`.

Frozen benchmark ELFs:

| Variant | Path | SHA256 |
|---|---|---|
| Baseline | `/tmp/mgt-sparse-half-policy-bin` | `23ff68bda5ebb71b73f68c05d62e420080d493367fdcd1cd2aff8201a7a72239` |
| Shared bias | `/tmp/mgt-adam-shared-bias-bin` | `d1eb23fc929962dd7dd68d90f07996f077b8fc121bbdc78d1cb82699a0590b9f` |

## Paired timing

Job `2c69dfe470c6` ran strict ABBAAB. Each process was thermally gated to an
80--81 C start and ran 140 warmup plus 100 timed graph steps. Hashes were checked
before and after each run. Every retained active sample was exactly 780 MHz for
both binaries.

| Variant | Run means, ms | Median, ms |
|---|---|---:|
| Baseline | 18.3736, 18.3729, 18.3730 | 18.3730 |
| Shared bias | 18.2905, 18.2895, 18.2922 | 18.2905 |

The absolute median delta is **-0.0825 ms/step**. Latency falls by
**0.449028%** and throughput rises by **0.451054%**.

- Retained JSONL SHA256:
  `2da5c99f701b1e8dfbb8d930c5eb4fea8cda639ff1b57fef1e271c83d03bf0a3`.
- Fail-closed revalidated summary SHA256:
  `9d62302c43a788e67105fa278484f7185784667326f16b54001c12ceda94edf8`.
- Summary revalidation job: `c2c16bfe8493`.

The clock was not administratively locked; the accepted protocol requires one
unique active clock per variant and equality between variants.

## Nsight mechanism proof

Nsight Systems job `c9eb89cfd4b3`, analyzed by `b951a844bc95`, preserves exact
graph event order. Every measured step retains 347 kernels, 71 memsets / 156,136
bytes, and zero copies. The only changed kernel metadata belongs to the weight
Adam node: `(60392 x 256 threads, 25 registers, 0 shared bytes)` becomes
`(120784 x 128 threads, 24 registers, 8 shared bytes)`.

| Mean over graph steps 2--4 | Baseline, ms | Shared bias, ms | Delta, ms |
|---|---:|---:|---:|
| Weight Adam | 1.402761 | 1.280184 | -0.122577 |
| Kernel sum | 18.302010 | 18.180409 | -0.121601 |
| Step span | 18.866473 | 18.735822 | -0.130650 |

Adam falls by **8.7383%** in the trace; unrelated kernel deltas stay below
0.002 ms. Baseline SQLite SHA256 is
`e3d47859d21360135713ff7b5248b5fef9d3ecbf63c835ba33b0c56d095287dc`;
candidate SHA256 is
`0de9f509187e99852e3d426ea707ef4e7a9a3968572bcdc85a6a3f31dd56e612`.
Strict analysis JSON SHA256 is
`45b7d65f814b9583c14374844cd35b2ed1be60df75d2807bcac37d2d674f07b5`.
Nsight perturbs absolute timing; ABBAAB owns the end-to-end claim.

Nsight Compute candidate job `97edb07b5517` confirms the mechanism:

| Metric | Baseline | Shared bias |
|---|---:|---:|
| Block / grid | 256 / 60,392 | 128 / 120,784 |
| Registers/thread | 25 | 24 |
| Static shared/block | 0 B | 8 B |
| Executed instructions | 147,559,791 | 94,268,578 |
| DRAM throughput | 93.709% | 93.905% |
| DRAM read / write | 247.403 / 216.234 MB | 247.374 / 216.219 MB |
| Replay duration | 1.289600 ms | 1.286752 ms |

Instruction count falls by **36.1150%** while physical traffic is unchanged and
DRAM remains the roof. NCU's uncontrolled-clock replay shows only a 0.22% time
delta and is used as mechanism evidence, not as the performance claim.

## Sanitizer, regression, and platform gates

All GPU work ran through `mgt-gpu-queue`; job IDs map to retained
`.gpu_queue/logs/<id>.log` files.

| Job | Outcome |
|---|---|
| `dc48ac0a6b24` | Exact Adam suite passed memcheck, initcheck, racecheck, and synccheck; full p888 graph FP16 candidate passed memcheck/initcheck. Zero leaks, errors, warnings, or hazards. |
| `da7e7e66b77c` | Final SM86 CPU/CUDA/NCCL/graph targeted regression passed 38/38. |
| `a8c02fd2affa` | Exact `CMAKE_CUDA_ARCHITECTURES=75` build passed exact-Adam and native-graph tests 2/2 on the available SM86 device. |

The SM75 gate proves code generation and runtime compatibility, not T4 timing.
Optimizer convergence is unchanged by construction because the exact test
proves byte-identical FP32 state and half mirrors over disparate steps.

## Boundary and next target

Adam now consumes about 1.280 ms and is at 93.9% of measured DRAM peak. Further
exact single-kernel work is bounded to roughly a few percent unless gradient or
moment storage changes precision or the optimizer is fused into producers.

The next largest measured custom kernel is
`SparseInputGradCompactActiveAdjacent2PackedHalfU16` at 1.943824 ms. It becomes
the next audit target; changes must retain the explicit RN-half-source / ordered
FP32-accumulation contract.
