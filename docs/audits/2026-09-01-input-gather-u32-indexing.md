# SM86 input-gather u32-index audit

Date: 2026-09-01. Target: original p888, batch 4096, RTX 3070 Laptop /
SM86, CUDA Graph mode. Baseline: packed-u16 sparse-gradient checkpoint
`406e8755b5b32ee4701f224d7e7275efd433f795`.

## Decision

Accepted for automatic SM86 selection. `InputHalf2Row32<128>` retains the
existing one-CTA-per-row/feature-tile ownership, bias-first position order,
half2 loads, explicit FP32 round-to-nearest additions, output padding, launch
geometry, and storage types. It changes only address/index arithmetic from u64
to u32 when both complete element ranges fit `UINT_MAX`.

The dispatcher proves the weight and output bounds before launch. A cached
device policy enables the path only for compute capability 8.6. T4/SM75, A100,
other architectures, oversized shapes, odd physical widths, and odd logical
widths retain their previous paths. There is no precision conversion,
allocation, graph node, synchronization, reduction, optimizer, or model
change. For p888 the proved extents are 13,273,600 half-weight elements and
10,485,760 FP32 output elements.

## Correctness, policy, and sanitizer gates

All job IDs refer to retained `.gpu_queue/logs/<id>.log` files.

| Job | Outcome |
|---|---|
| `574cc63fba6c` | RED: exact test referenced the missing u32 kernel. |
| `a0263362e778` | GREEN: u32 kernel was bit-exact against the independent ordered-FP32 oracle and existing u64 variants. |
| `74b225c2a621` | RED: SM mapping assertions referenced the missing architecture policy. |
| `1e00d48f858a` | GREEN: six production/input/graph/capture tests passed with the SM86-only policy. |
| `e6c4bd808644` | Compute Sanitizer memcheck: 5 cases, 15 variants, zero errors. |
| `8b64d8ed58db` | Compile-only SM75/T4 build passed for the test and production benchmark. Runtime T4 performance is not claimed. |
| `3d7f9c444a02` | Final gate: 33/33 CUDA/C ABI tests, native full/tail/fail-stop graph, capture/update, measurement validators, exact frozen-ELF identity, and Rust FFI passed. |

The exact suite compares FP32 bits, not a tolerance. It covers partial feature
tiles, changing live row counts, padding, half extrema, cancellation, NaN,
signed zero, immutable inputs, and allocation canaries. Compile-time policy
assertions select only SM86 and reject SM75, SM80, SM89, and SM90.

## Frozen binaries and unprofiled end-to-end result

| Variant | Frozen path | SHA256 |
|---|---|---|
| Packed-u16 baseline | `/tmp/mgt-sparse-u16-packed` | `b643e49b5e4ec1075e8c515e9ea5085b527b0668e36ba27388aee411059ac14d` |
| Initial u32 measurement | `/tmp/mgt-input-half-u32` | `57e73e7480266b6cb14ff448cf68067ecb89e955508053a35ccaab09c818a1aa` |
| Final SM86-policy ELF | `/tmp/mgt-input-half-u32-sm86` | `e0fe40aa9cfbdfba55607a43288a32446d114354c18d2ec1633a03a4eb1c1c5a` |

Every ABBAAB series retains all six run means. Each process verifies binary
hashes before and after execution and uses original p888, batch 4096, 140
warmup steps, 100 timed graph steps, and the unchanged 744001024-byte arena.
Clocks were observed, not locked.

| Series/job | Median baseline, ms | Median candidate, ms | Throughput |
|---|---:|---:|---:|
| `af3d0277c65f` | 20.9759 | 20.9554 | +0.09783% |
| `3710af931c2e` | 20.9775 | 20.9611 | +0.07824% |
| `2b674e9850d6`, exact final ELF | 20.9758 | 20.9549 | +0.09974% |

The exact final series moves the median from about 195.27k to 195.47k
samples/s. This is a small but repeated end-to-end win; it is not rounded up to
the larger profiler deltas.

- `paired-input-half-u32.jsonl`, SHA256
  `247ad9ed33e1cc47c01221dc7c3d7ff5a3d291a2f0a858a6d9a043fd2e0cf740`.
- `paired-input-half-u32-confirm.jsonl`, SHA256
  `a7f17e6b8110b46bf9be9f1352fe43c3b5f2ec5c2091509ccdab41e74387d37d`.
- `paired-input-half-u32-final.jsonl`, SHA256
  `234cd9387539fc2e2d08ecd545408fec29b5a24c31f6a1e636e99bba3ba140e1`.

The policy-only rebuild changes host code and whole-ELF metadata. Job
`66f17f542200` proves the initial measured and final policy ELFs have
byte-identical emitted SASS and PTX; the final frozen ELF was then measured
directly by the third series.

## Nsight Systems structural and timing proof

Jobs `141d508aea2b` and `5e243e8fc9cc` captured both binary orderings after 100
warmup steps. The strict analyzer compared four complete graph steps in each
trace. Both pairs preserve the exact event order, all 363 kernels, 71 memsets,
zero copies, all launch geometries, and every unrelated kernel resource tuple.
Only the input-gather signature/resource tuple changes: 40 to 39
registers/thread and 640 to 320 bytes static shared memory per CTA.

| Nsight order | Baseline gather, ms | Candidate gather, ms | Delta, ms | Unchanged sparse delta, ms |
|---|---:|---:|---:|---:|
| Baseline then candidate | 1.823994 | 1.771166 | -0.052828 | -0.052758 |
| Candidate then baseline | 1.825023 | 1.800101 | -0.024922 | +0.005627 |

The first ordering visibly contains whole-run drift because the unchanged
sparse kernel moves by the same amount as the gather. The reverse ordering
still keeps the gather win while the unchanged sparse kernel moves against it.
The two profiled gather deltas average -0.038875 ms, but Nsight perturbs this
kernel; only unprofiled ABBAAB is used for the published full-step percentage.

- Forward baseline/candidate SQLite SHA256:
  `15e1e3780da67361687563ad6c07059af1f7dccfd39902242357efb69cbd6279`,
  `cc351ffc4155ac1633515e2a5e263f05422c36bed7c60dce4c23d68fa7d3c4b6`.
- Reverse baseline/candidate SQLite SHA256:
  `ecf22b982f9c1a74c99f2ecfa28d305611ae274a90d32bf3719ea7ddec72238b`,
  `eb3f1cd3ef1bc03a3e13beb44b7949bd5f11416f5cd132afe35bbc2f6b0ed872`.
- `input-half-u32-paired-profile.json`, SHA256
  `4082cc1516adf35fa14b67f511653ad65ff08e6e0499681d905386e4f2b0e0c0`.

## Nsight Compute mechanism proof

Jobs `884865bb8433`, `f2a043fc831f`, and `53bd6612f6a7` used NCU 2025.1.1,
the exact production shape, 100 warmup launches, one measured gather launch,
and no profiler clock control.

| NCU metric | u64 address path | u32 address path |
|---|---:|---:|
| Duration, ms | 1.816256 | 1.782560 |
| Executed instructions | 128,946,176 | 123,703,296 |
| Registers/thread | 40 | 39 |
| DRAM throughput, GB/s | 6.931 | 7.032 |
| L2 sector hit rate | 96.155% | 96.188% |
| Achieved occupancy | 94.529% | 94.309% |

The mechanism is reduced integer address work: about 4.1% fewer executed
instructions and one fewer register. NCU replay changes scheduler/stall
behavior, so its -0.033696 ms duration is mechanism evidence, not the
unprofiled latency claim.

- u64 raw CSV SHA256
  `4a1ca11ec62760eadd38a49272c17aa624bd079af3222094d3e9366acd7340c2`.
- u32 raw CSV SHA256
  `1eaa81726ff6b8b9613a2b6132f5c2148f0a4cc44e9165b626ed72985163c2dd`.

## Rejected local variants

The exact harness rejected wider per-thread coarsening, a fixed-p888 unroll,
software prefetch, and a 64-thread CTA. Representative isolated results were
1.813 ms for 64x4, 1.869 ms for 32x8, 1.805 ms for fixed-p888, 1.813 ms for
prefetch, and 1.804 ms for 64 threads versus about 1.798 ms for the accepted
128-thread u32 candidate. No rejected path remains in production.

## Boundary and next target

This checkpoint proves exact SM86 execution and compile-time SM75 fallback. It
does not prove T4 runtime speed, convergence, or multi-GPU scaling. The input
gather is now about 1.80 ms/step. The packed sparse-gradient consumer remains
larger at roughly 3.55-3.61 ms/step; another material forward win likely needs
a dataflow change such as avoiding or fusing the 40 MiB gather materialization,
not another launch-geometry tweak.
