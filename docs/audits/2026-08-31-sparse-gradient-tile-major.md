# Accepted SM86 sparse-gradient tile traversal

Date: 2026-08-31. Follow-up to the
[baseline NCU investigation](2026-08-31-sm86-sparse-gradient-ncu.md).

Accepted: original p888, batch4096, RTX3070 Laptop/SM86. Median run-mean
full-step latency **34.2866 ->28.9943ms** (-15.44%); throughput **+18.25%**,
approximately119.5k ->141.3k samples/s. One CTA coordinate swap, unchanged
arithmetic, no new allocation, unchanged739806720-byte arena.

## What changed

The existing grouped sparse-gradient kernel was moved to private
`native/cuda/sparse_input_grad_grouped_rows.cuh` so an isolated regression can
exercise the real implementation. Anonymous namespace linkage keeps the local,
distributed and test translation units independent.

Before: `grid=(ceil(hd1/256), bins)`, `h=blockIdx.x*256+threadIdx.x`,
`bin=blockIdx.y`. After: `grid=(bins, ceil(hd1/256))`,
`h=blockIdx.y*256+threadIdx.x`, `bin=blockIdx.x`.

Every thread still owns one output and sums its ascending row-ID list in the
same order. Builder, row-ID layout, FP32 additions, padding, scratch capacity
fallback, BN lifetime and dispatch policy are unchanged. Auto remains SM86-only;
explicit grouped-row performance on other GPUs was not measured.

The mapping favors reuse of a fixed feature tile across bins/positions. It is
not a correctness dependency on CUDA block scheduling order. Thread count256
was held constant;128/64 tiles and a new builder were not mixed into this test.

## Identity and verification

Existing Docker container `mgt-gpu-queue`, image `mgt-single-gpu-dev:2026-08-28`,
CUDA12.8.93, SM86 Release/Ninja, Nsight Compute2025.1.1,
Nsight Systems2025.6.3. All GPU commands used `gpu_queue_submit.py`.

| Binary | SHA256 |
|---|---|
| Pre-extraction accepted BN-fused baseline, frozen `/tmp/mgt-sparse-tile-baseline` | `3422df2cfed1cb9cab1b5922b3162a95d615e03a8df2179fb640ca8eb9e63747` |
| Post-extraction old mapping, `/tmp/mgt-sparse-tile-extracted-baseline` | `6865f5e24b490ef5408cb711b9d92dabdc10d59bf1fef6d40c82ef32cd9645a5` |
| Accepted new mapping, `/tmp/mgt-single-sm86/mgt_single_gpu_benchmark` | `46a97247b835e9c9d0ac93a9e77e7a52111a2994785e261c61ecfa75a421b35e` |

The A/B uses the post-extraction baseline to isolate mapping from refactoring.
Production inputs and model/optimizer contract are the same as in the
[BN epilogue audit](2026-08-31-bn-epilogue-fusion.md).

Runtime RED: job`6b484c6d7b54` built successfully, then the new-grid test failed
against old kernel coordinates: bin2,h0 retained sentinel`0xcdcdcdcd` instead
of zero. GREEN job`2ce20d59b3ba`:10/10 CUDA CTests passed, including21 isolated
gradient cases. Full isolated memcheck:0 errors,0 leaked bytes; quick
racecheck:0 errors/warnings/hazards; quick synccheck:0 errors.
Job`b2f83218e0e8`: production4096-row complete-step memcheck:0 errors.

The regression combines independent old-grid GPU serial additions, bounded
CPU serial additions, hand-derived rounding/cancellation witnesses and guards.
It covers rows1/31/32/33/4095/4096/4097, reused allocations with changing row-ID
strides, empty bins, collisions/skew, odd widths/tails, zero padding and the
full72x72x4096x2560 shape. It is not a full independent training trajectory.
Independent read-only review found no actionable issue in the scoped change.

Pre-publication smoke:4/4 CPU/C ABI CTests passed in job`55a0a7229ee9`.
Its Rust binary built but initially could not locate the shared library;
job`3a5257289583` passed the Rust FFI/RAII GPU test with
`LD_LIBRARY_PATH=/tmp/mgt-single-sm86` prepended for that process. No Rust/API
change was needed. These are targeted checks, not the full multi-GPU suite.

## Unprofiled full-step A/B

Raw files/scripts under `test_results/sparse_gradient_20260831/`:
`run_paired.py`, `summarize_paired.py`, `paired-tile256*.jsonl` and matching
`*-summary.json`. Order A-B-B-A-A-B; three runs/variant. No run was removed.
Each reported value is the benchmark's mean over100 timed steps, not an
individual-step median. Fixed seed, original inputs, batch4096 and arena size
were unchanged.

| Dataset / queue job | Warmup | A run-means, ms | B run-means, ms | Median A ->B |
|---|---:|---|---|---|
| `paired-tile256`, `26ae631363d9` | 100 | 34.2767,34.2390,34.2390 | 28.9649,29.0229,28.9690 | 34.2390 ->28.9690 |
| `paired-tile256-confirm`, `71c66c86170e` | 140 | 34.2866,34.2743,34.2971 | 28.9426,29.0414,28.9943 | 34.2866 ->28.9943 |

The first dataset sampled780/795MHz and is retained as preliminary evidence.
The confirmation sampled780MHz for all active samples and all six final2s
tails. Active temperatures were80-86C for A and83-87C for B. Telemetry is sparse
and asynchronous, not exact timed-window clock locking or a statistical CI.
No clocks/power settings were changed. The separate confirmation, not a pooled
selection of best runs, supplies the headline result.

Training losses at240 steps overlap (A18.2661-19.0245, B17.7924-19.6372).
Loss variation across separate runs is not evidence of a mapping-induced
arithmetic difference, nor is this short test proof of convergence. Exact
isolated gradients and unchanged algorithm are the correctness gate here;
future precision changes still need convergence/time-to-quality evaluation.

## Nsight attribution

Systems job`f7125d316a2d`:104 steps/profile, analyze only the four after100
warmups; means below use steps102-104. Raw `map-baseline-warmed` and
`tile256-warmed` reports/SQLite, `tile256-warmed-profile-summary.json`.

| Metric | Old mapping | New mapping |
|---|---:|---:|
| Full GPU step span, ms | 35.008859 | 29.702147 |
| Kernel time sum, ms | 32.586577 | 27.291954 |
| Sparse gradient, ms | 8.950073 | 3.689962 |
| Row builder, ms | 1.842940 | 1.831929 |
| Forward span, ms | 8.676007 | 8.663133 |
| Kernel count | 463 | 463 |
| FloatToHalf / GEMM mains / split-K reductions | 33 /99 /33 | 33 /99 /33 |

All four ordered kernel/geometry sequences match exactly after transposing only
the sparse-gradient grid. Full-step gain is almost entirely explained by that
kernel. Profiled span is not the unprofiled throughput measurement.

The earlier zero-warmup Systems job`0aaa4880a249` retained correct sequence
evidence but had an anomalous candidate span78.88ms with broad slowdown of
unchanged kernels. Its times are excluded from performance attribution, not
silently pooled/deleted. The precise transient cause was not established;
the warmed rerun and independent A/B resolve acceptance without guessing it.

Compute:15-pass kernel-replay/default-cache-flush candidate in
`tile256.ncu-rep` (job`0aaa4880a249`) versus previous baseline investigation:
duration8.9456 ->3.728064ms; L2 hit3.7055 ->79.2643%; registers40 ->40;
executed warp instructions221,393,760 ->221,393,760. SM throughput34.73 ->83.32%
(new limiting contributor is LSU instruction throughput); occupancy98.34
->96.18%. Higher occupancy was not the reason for improvement.

Direct memory-counter confirmation, job`b2f83218e0e8`: application replay,
`--cache-control none --clock-control none`, three application passes each,
step101 after100 warmups. Reports/CSV: `map-baseline-bytes`, `tile256-bytes`.

| Counter | Old mapping | New mapping |
|---|---:|---:|
| DRAM bytes read, decimal GB | 2.970514 | 0.60419648 |
| DRAM bytes written, decimal MB | 53.355648 | 53.125248 |
| L2 sector hit rate, % | 3.666765 | 80.195864 |
| Kernel duration, ms | 8.949952 | 3.674080 |
| GPC frequency, MHz | 779.994891 | 779.997714 |

Reads fell approximately79.7%, with nearly unchanged writes. This supports the
cache-reuse explanation using measured traffic, not just logical FLOP/byte
estimates. Multi-pass diagnostics are still not unprofiled timing.

## Training-plugin evidence and next work

The `write-cuda-hot-paths`, `use-nsight-compute`, `use-nsight-systems`,
`use-compute-sanitizer` and TDD workflows shaped this experiment: one variable,
exact oracle, isolated failure, memory counters, then full-step acceptance.

Transferable result: a memory-saturated kernel can still waste most traffic;
change reuse distance before assuming bandwidth saturation is optimal.
Do not turn this specific SM86 mapping into a universal hardware recipe.

The 3070 trainer is not declared fully optimized. Remaining work includes the
row builder, remaining linear/BN backward traffic and launch scheduling, plus
stronger training-quality/runner validation. The dedicated training plugin and
T4 port remain later stages; no T4 default or Kaggle runner changed here.
