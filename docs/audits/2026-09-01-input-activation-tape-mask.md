# SM86 input activation-tape mask audit

Date: 2026-09-01. Target: original p888, batch 4096, RTX 3070 Laptop /
SM86, CUDA Graph mode. Baseline: the accepted SM86 u32-index input-gather
checkpoint `f4b23c1f8a620a611ee69ba8bba85130e89b73b6`.

## Decision

Accepted for automatic selection on the existing local-FP16 SM86/u32-fit path.
At input BN site 0, the already-authoritative FP16 activation tape receives the
rounded ReLU result while the 40 MiB FP32 gather matrix remains untouched. The
input BN backward kernels recover the exact ReLU predicate from
`fma(normalized, gamma, beta) > 0` instead of reading that FP32 activation
matrix twice.

The preserved-input forward specialization uses `__fmaf_rn`; the recomputed
backward predicate uses the same operation. Other forward specializations keep
their previous expression and code generation. BN normalized values, running
statistics, affine parameters and gradients, dense/sparse gradients, Adam
state, and optimizer precision are unchanged. This checkpoint removes one
FP32 activation store and two later FP32 activation reads; it does not remove
the input gather itself.

Selection is intentionally narrow. It requires the local FP16 context, exact
SM86 architecture policy, even input widths, and the same proved u32 element
bounds as the input gather. SM75/T4, nonlocal/synchronized BN, non-FP16,
oversized, and odd-width paths retain the previous FP32 activation
materialization and mask reads.

## Correctness, alias, and platform gates

All job IDs refer to retained `.gpu_queue/logs/<id>.log` files.

| Job | Outcome |
|---|---|
| `d13c1766387d` | RED: tests referenced the missing preserve/recompute API. |
| `a4d1c6288438` | Primitive forward/backward tests passed, 2/2. |
| `f8a6bd92e58f` | Final candidate build plus independent rounded CPU full-step oracle and integration set passed, 9/9. |
| `13fc87e44595` | Backward exact-mask and fail-stop alias suite passed: 13 apply, 7 full, 70 invalid cases. |
| `9858f8091f83` | Compute Sanitizer memcheck passed for forward epilogue, backward recomputed mask, and full FP16 activation-tape trainer; three zero-error summaries. |
| `8baf6aff0000` | Compile-only SM75/T4 build passed for both primitives, the full activation-tape test, and production benchmark. Runtime T4 speed is not claimed. |
| `83d8fb2984db` | Final regression set passed 33/33 plus native full/tail/fail-stop graph and graph capture/update. The job then stopped only at a whole-ELF metadata comparison described below. |
| `47b7771c6d88` | Measurement validators and Rust FFI passed. |

The full-step oracle includes an even-width input shape that selects the
production SM86 path. It independently reconstructs the forward activation,
loss, every weight and affine gradient, running statistics, tape contents,
padding, overwrite semantics, changing live-row counts, and cache lifetime.
The primitive suite covers NaN, signed zero, partial tiles, in-place `dy==dx`,
guard canaries, exact FP32 ordering, and the recomputed affine predicate.

The public alias contract rejects partial writable overlap before any launch.
`activated` and `relu_beta` are mutually exclusive; residual gradient requires
the materialized activation route. `relu_beta` may alias read-only apply inputs,
but full backward rejects overlap with writable `dgamma`, `dbeta`, or the
statistics workspace.

## Frozen code and paired end-to-end result

| Variant | Frozen path | SHA256 |
|---|---|---|
| Baseline | `/tmp/mgt-input-half-u32-sm86` | `e0fe40aa9cfbdfba55607a43288a32446d114354c18d2ec1633a03a4eb1c1c5a` |
| Clean measured candidate | `/tmp/mgt-input-tape-mask-final` | `55c21ac62c3075c0eedc4d0567f44098705f0af1a902c1cd9e9eea7dc6018fc0` |
| Final source rebuild | `/tmp/mgt-single-sm86/mgt_single_gpu_benchmark` | `9982e052ef671d04edb8997e548e351dd8a4d889ede665352f776e0096a130c8` |

Job `dfb339e087a6` proves that the final source rebuild and clean measured
candidate have byte-identical `cuobjdump` SASS, PTX, host `.text`, and host
`.rodata`. CUDA relinking changed non-code ELF metadata, so a whole-file hash
is retained as provenance but is not treated as a code-generation difference.
Host `.text` SHA256 is
`24f11a7e18228adeb36251e35e272b5785f1f40c6bf3c5948649c4e9620b08a0`;
host `.rodata` SHA256 is
`8b8c4fa572dda5bfae418ba454d9308e899a4426051c6a0a40d81c715151da89`.

Clean job `5a37585e93a1` retained all six ABBAAB run means. Each process checked
binary hashes before and after execution and used original p888, batch 4096,
140 warmup steps, 100 timed graph steps, and the unchanged 744001024-byte
arena. All active clock samples were 780 MHz.

| Variant | Run means, ms | Median, ms | Samples/s at median |
|---|---|---:|---:|
| Baseline | 21.0512, 21.0517, 21.0478 | 21.0512 | about 194,573 |
| Candidate | 20.7845, 20.7847, 20.7819 | 20.7845 | about 197,070 |

Latency falls by **1.266911%** and throughput rises by **1.283168%**.
`paired-input-tape-mask-final.jsonl` SHA256 is
`af42f1add2e3845f29d4647d542b0561a23dda802d2ef6034780b68132f5ae74`.

An exact-whole-file repeat, job `1a77e3277250`, was rejected: an external GPU
load moved active clocks through 210-780 MHz and produced 39-365 ms run means.
Neither that run nor clock-smoke `620bb14d74c6` contributes to the published
speed claim. The local ABBA recorder was tightened to reject varying active
clocks and more than 1% within-variant run-mean spread.

## Nsight Systems structural and timing proof

Jobs `a54927cf7937` and `aed94825c7ad` captured both binary orderings after 100
warmup steps. The strict analyzer compared four complete graph steps in each
trace. Both pairs preserve 363 kernels, 71 memsets, zero copies, every launch
geometry, and event order. Only the input site-0 forward/partial/apply kernel
identities change; backward apply moves from 20 to 22 registers/thread.

| Site-0 operation | Baseline -> candidate, ms | Reverse pair, ms |
|---|---:|---:|
| Forward apply | 0.411115 -> 0.338515 | 0.411116 -> 0.338291 |
| Backward partial | 0.340400 -> 0.231907 | 0.340424 -> 0.232344 |
| Backward apply | 0.477552 -> 0.426947 | 0.477658 -> 0.427160 |
| Sum delta | -0.231698 | -0.231404 |

The two target deltas average **-0.231551 ms/step**. Nsight perturbs absolute
timing; the unprofiled ABBAAB result owns the end-to-end percentage.

- Forward baseline/candidate SQLite SHA256:
  `392d35a0d25b65434d581df0775af3a653738bb7ef25cf5ee629a8e2dd1af7a1`,
  `c638a0ead5853ac2762fe5050cdf2774b46e74e27c1761847d47848a18c0d7ba`.
- Reverse baseline/candidate SQLite SHA256:
  `5d6097aafadbb8bb6a3ef258aea6a2bbf523aaffdb4deae6ae81b6dbc03bd6dd`,
  `3217d192e35da9e1e7d33eb9a22a92f0c9e35f24e65741810b18638f2462ce66`.
- `input-tape-mask-final-paired-profile-v2.json` SHA256:
  `40fd0c416c0a4a6e1432305960304405fd2f62eddd2ef003aafe4c0220af8146`.

## Nsight Compute mechanism proof

Jobs `147add2dc7bc` and corrected site-0 baseline capture `32ef1d5f3dbd` used
the exact production geometries. The strict analyzer rejects the earlier wrong
hidden-site baseline selection.

| Operation | Duration, us | DRAM read delta, MB | DRAM write delta, MB | Registers | Instruction delta |
|---|---:|---:|---:|---:|---:|
| Forward apply | 408.960 -> 314.304 | +0.066560 | **-41.883776** | 20 -> 20 | -983,040 |
| Backward partial | 345.984 -> 237.120 | **-41.889152** | +0.000896 | 40 -> 40 | -266,240 |
| Backward apply | 472.352 -> 401.056 | **-42.082688** | -0.550272 | 20 -> 22 | +2,621,440 |

The mechanism is measured, not inferred: one approximately 40 MiB FP32 store
and two approximately 40 MiB FP32 reads disappear. The extra two FMA operands
increase apply instructions/registers, but avoiding the activation read is a
net win. NCU replay used eager launch and uncontrolled clocks/cache state, so
its durations are mechanism evidence only.

`input-tape-mask-ncu-profile.json` SHA256 is
`f5dbb8887e254a4aa5e82b4f9ed2210aada1ed4585703a606b80cb07b97259b3`.

## Boundary and next target

This checkpoint proves single-GPU SM86 correctness, alias semantics, memory
safety, graph structure, mechanism, and speed. It does not prove convergence,
runtime T4 speed, or multi-GPU scaling. The next measured material costs remain
the approximately 1.8 ms input gather and 3.55-3.61 ms packed sparse-gradient
consumer. Removing or fusing the still-materialized 40 MiB gather is the next
forward dataflow target; sparse-gradient work remains the larger individual
kernel family.
