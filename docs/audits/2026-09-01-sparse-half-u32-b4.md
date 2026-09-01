# Sparse-half u32/B4 consumer audit

Date: 2026-09-01. Target: original p888, batch 4096, RTX 3070 Laptop /
SM86, CUDA Graph mode. Starting checkpoint:
`fd8b5bd8f3842fc1a3dfef3a587294b4d1ab9ff9`.

## Decision

Accepted for the existing explicit FP16 input-gradient policy on SM86. The
compact sparse consumer now indexes its source and destination as half2/float2
with proven-u32 element extents and packs four independent bin warps into one
128-thread CTA. Each lane still owns the same two adjacent features and walks
the same ascending u16 row list with explicit RN FP32 additions. The result is
byte-identical to the prior RN-half/serial-FP32 contract.

Selection is fail-closed. It requires the already-selected compact-active u16
half path, an SM86 device policy, even/aligned half2/float2 storage, and u32-fit
active-list, source, and output element extents. Other architectures, including
SM75/T4, and oversized shapes retain the previous 64-bit two-warp kernel until
they are measured natively. There is no new allocation, graph node, copy,
synchronization, reduction, precision change, or optimizer change.

## Search and rejected dataflow

The current half consumer was measured at about 1.94 ms with roughly 1.37 GB of
L2 traffic. A position-pair candidate loaded each dZ row once for two positions
and stored 48 ordered accumulators per lane in 12--12.7 KiB shared memory. Job
`38b4fa0dcb4c` proved byte-exact output, but it ran at 6.47--9.65 ms versus a
2.03 ms builder-plus-current-consumer pipeline. One warp consumed about 12 KiB
shared memory and generated about 6 GB shared traffic, so the candidate was
rejected without entering production.

Jobs `3ae4c6095a29` and `b220566a78b8` then swept exact u32 half2 indexing,
read-only spelling, manual unrolling, and one/two/four/eight/sixteen bin warps
per CTA. The hot repeat measured the existing consumer at 1.938 ms and the
u32/four-warp candidate at 1.797 ms. One warp was badly underpacked, eight and
sixteen warps regressed, fixed-p888 indexing did not help, and explicit `__ldg`
was slower. These isolated repeated-buffer timings selected a candidate; they
are not the published end-to-end claim.

## Arithmetic and bounds contract

`SparseInputGradCompactActiveAdjacent2PackedHalfU16U32` computes list, dZ-half2,
and output-float2 element indices in `unsigned`. Host preflight proves:

- `active_count * rows <= UINT_MAX`;
- `rows * (hd1 / 2) <= UINT_MAX`;
- `state_len * state_value_pad * (hd1 / 2) <= UINT_MAX`.

The exact compact suite now exercises both the prior and u32 kernels against a
serial RN-half oracle. It covers a non-multiple-of-four active-map tail, rows
1/31/32/33/257, changed states, inactive-output ownership, persistent zeros,
immutable inputs/maps, and allocation canaries. Build job `d0fa28842630` passed
this suite plus native graph and graph capture; the production smoke reported
`source=half ... u32=1`.

Frozen binaries:

| Variant | Path | SHA256 |
|---|---|---|
| Adam shared-bias baseline | `/tmp/mgt-adam-shared-bias-bin` | `d1eb23fc929962dd7dd68d90f07996f077b8fc121bbdc78d1cb82699a0590b9f` |
| Sparse u32/B4 candidate | `/tmp/mgt-sparse-half-u32-b4-bin` | `d71de98df84c93c1e292adb02665827332f31644fb0c31abe045bd9cd73d2359` |

## Thermal-gated full-trainer A/B

Job `93b55d46339d` retained strict ABBAAB. Every process used original p888,
batch 4096, graph mode, FP16 input-gradient policy, 140 warmup steps, and 100
timed steps. All active telemetry samples for both variants were exactly
780 MHz.

| Variant | Run means, ms | Median, ms |
|---|---|---:|
| Baseline | 18.3386, 18.2881, 18.2899 | 18.2899 |
| u32/B4 | 18.1873, 18.1862, 18.1846 | 18.1862 |

The median delta is **-0.1037 ms/step**: latency falls **0.566980%** and
throughput rises **0.570213%**.

- Paired JSONL SHA256:
  `444cd82dabd0be4d60a5435ee78d9c17afd015dd6ec16f1c005590e8d43557dc`.
- Summary SHA256:
  `4292db8d465a4d325a044018d14bfc93f047d172ccaf57745b314942dad99a14`.

## Nsight mechanism proof

Nsight Systems job `c5a86856d2af` passed exact canonical graph event order.
Only the sparse-half kernel identity and launch metadata changed; every step
retained 347 kernels, 71 memsets / 156,136 bytes, and zero copies.

| Mean over graph steps 2--4 | Baseline, ms | Candidate, ms | Delta, ms |
|---|---:|---:|---:|
| Sparse consumer | 1.943824 | 1.827256 | -0.116568 |
| Kernel sum | 18.180409 | 18.054211 | -0.126198 |
| Step span | 18.735822 | 18.600823 | -0.134999 |

The sparse launch changes from `(864,40,1)x64` with 30 registers/thread to
`(432,40,1)x128` with 28 registers/thread. Baseline SQLite SHA256 is
`0de9f509187e99852e3d426ea707ef4e7a9a3968572bcdc85a6a3f31dd56e612`;
candidate SQLite SHA256 is
`c1635b56850ecb4f4e758aa578caafe4c69d792e7b87fad8603c0c69b07d6e9a`.
The strict analysis JSON SHA256 is
`718987b0c19d9c593f808d1dfa227d5c5d9cdbc5b75e9b7d99b264a9c00a8a86`.

Nsight Compute job `289ffe4789c8` confirms the mechanism:

| Metric | Baseline | Candidate |
|---|---:|---:|
| Executed instructions | 151,566,000 | 133,129,840 |
| Registers/thread | 30 | 28 |
| Active warps, % peak | 57.783 | 84.424 |
| DRAM read / write, MB | 21.645 / 18.178 | 21.670 / 18.155 |
| L2 traffic, GB | 1.370 | 1.400 |
| Replay duration, ms | 1.9376 | 1.0312 |

Instructions fall 12.16% while physical traffic remains effectively unchanged.
The large replay-duration delta is cache/replay evidence only; the thermal
ABBAAB and Nsight Systems trace own timing claims. NCU report and CSV SHA256 are
`17a18e16d27970f78f494928756a15459c4b5232c1862a6dc943828da6f81191`
and `81ec96e60e265059bf8107ccee54406c7e06c4adb3a60aceeb16104db8f98834`.

## Safety, regression, and platform gates

| Job | Outcome |
|---|---|
| `24d7c6b5183f` | Exact suite passed memcheck/initcheck/racecheck/synccheck; full p888 graph passed memcheck/initcheck with zero errors or leaks. |
| `a9c79afb7f43` | Full targeted CPU/CUDA/NCCL/graph regression passed 38/38. |
| `19fc5db52dd2` | Exact `CMAKE_CUDA_ARCHITECTURES=75` build passed compact sparse, input-half policy, native graph, and Adam exact tests 4/4 on the available SM86 device. |

The SM75 build proves code generation and forward compatibility, not T4 timing.
The SM86-only selector keeps the u32/B4 optimization disabled on a native T4
until that device is benchmarked.

## Boundary and next target

The optimized sparse consumer remains the largest custom kernel at about
1.827 ms, followed by two cuBLAS GEMMs around 1.65--1.68 ms and the input gather
around 1.646 ms. Further exact sparse gains likely need a representation or
dataflow change; blind CTA packing, fixed-shape indexing, shared position-pair
accumulation, and dense one-hot GEMM have been exhausted or rejected.
