# SM86 sparse gradient: two independent columns per thread

Date: 2026-08-31. Original p888, batch4096, RTX3070 Laptop/SM86.
The experiment follows [BN mask fusion](2026-08-31-bn-backward-mask-fusion.md).
Correctness gates passed. The first measured benefit is only about0.11% of the
full step; X2 is not promoted as a meaningful optimization by this audit.

## Arithmetic and launch contract

The [test-only candidate](../../native/tests/cuda/sparse_input_grad_layout_candidates.cuh)
keeps the 256-feature tile and bin-major grid. A 128-thread CTA assigns each
thread columns `tile*256+thread` and `tile*256+thread+128`. Each column retains
its original ascending-row FP32 recurrence; independent accumulators are not
partial sums and are never combined. This changes ownership, not precision or
summation order. There are no new atomics, shuffles, barriers or shared memory.

The row-ID load and row-base calculation are shared across the two columns.
The total number of mathematical dZ loads, additions and gradient stores is
unchanged. Empty bins still overwrite both valid outputs with positive zero.
The second load and store are guarded independently at feature tails. Physical
padding is accumulated like every other column, not skipped by logical width.

Production launch: grid5184x10x1, block128x1x1, versus the old block256x1x1.
CTA count remains51,840. The grid divisor and unrelated `T=256` launches do not
change. No new allocation or model/optimizer buffer is introduced.

The recorded experimental host routing selected X2 only in
the local implementation on SM86, for both auto and explicit `grouped_rows`.
Other architectures and the nonlocal implementation retain the old consumer.
Selection is cached by device and relevant shape. The existing insufficient-
workspace fallback is unchanged. The `columns_per_thread` diagnostic reports
the selected grouped policy, not proof that this fallback was not taken;
production acceptance requires the actual Nsight launch geometry. This routing
was not retained: final production dispatch remains the baseline. The test-only
candidate and its expanded regression coverage remain for reproducibility.

## Numerical and memory gates

[The regression](../../native/tests/cuda/test_sparse_input_grad_grouped_rows.cu)
has independent old-grid GPU and serial CPU oracles. The candidate uses a
fixed256-feature grid independent of its128-thread block. GPU comparisons are
bitwise, including NaNs; CPU/literal comparisons permit only NaN-payload
differences. The test does not obtain expected results from the new kernel.

Coverage:37 full cases,6 quick cases. This includes counts0/1/31/32/33/63/64/65,
feature widths around128/256, allocation reuse with changed row strides,
poisoned unused row IDs and capacity rows, finite/NaN output poison, complete
input immutability, padding and allocation guards. Cancellation literals cover
both accumulators. Signed zero, smallest subnormals, normal/subnormal boundaries,
infinities, NaN and FP32 overflow retain the baseline arithmetic mode: the
effective NVCC command has no `--ftz` or `--use_fast_math` option. Tight allocations
at width129/count1 and width257/count3 allow memcheck to detect discarded tail
overreads beyond the final active row.

Large cases include72x72 categorical bins and4096x2560 gradients, including
logical2556/physical2560. A synthetic three-orbit fixture has3456 empty bins and
170/171 rows per occupied bin. It is correctness coverage, not a measured
production-walk histogram or performance workload.

- [Runtime RED `2c6ce9f6f010`](../../.gpu_queue/logs/2c6ce9f6f010.log): the old
  one-column body deliberately launched with128 threads and256-feature tiles
  leaves output poison at bin0,column256. Compilation succeeded. Before RED,
  the old real kernel passed all37 cases and quick memcheck/leak-check.
- [Unit GREEN `510a205daa24`](../../.gpu_queue/logs/510a205daa24.log): all37 X2
  cases passed; full memcheck/leak-check and initcheck report0 errors/0 leaks;
  quick racecheck reports0 hazards and synccheck0 errors.
- [Integration `bf0890043458`](../../.gpu_queue/logs/bf0890043458.log): eight
  CTests pass (X2 unit plus seven local/nonlocal full-step regressions),
  activation-tape memcheck/leak-check, gradient-overwrite initcheck and complete
  production batch4096 memcheck report0 errors. Local execution reports the
  selected two-column policy. Sanitizer durations are not benchmark results.

The test source SHA256 at the recorded unit/integration gates is
`1606413a17ab9da7a154b6ac9d9199d5281ee20d017988a02888cb6d836ab534`.
The isolated GREEN executable SHA256 is
`e01e4025fdb020777b586dbcf669babb6c737fee5df6701b05afcd873eb67070`.
Small/full-step gates do not establish arbitrary-run bit identity or long-run
convergence; existing cross-CTA BN/optimizer reduction behavior is unchanged.

## Frozen benchmark identities

| Variant | Executable | SHA256 |
|---|---|---|
| BN-mask baseline | `/tmp/mgt-bn-backward-mask` | `ca5b2a2300e7c39e0466d52d57627ecc251a715a6f840edf3437ffa6d1cd05eb` |
| X2 candidate | `/tmp/mgt-sparse-cols2` | `81ca865474eea2225a11f71f891b1fd893d513bcf5f13f685e28a5bad48c9cae` |

Both use original p888 inputs, unchanged arena739806720 bytes, FP32 master/Adam
and sparse gradients, FP16 dense operands and FP32 dense accumulation. The
baseline trace is `test_results/bn_backward_residual_20260831/mask-warmed.sqlite`,
SHA256 `6cabe52687cb00656f95068e17a2b54dd4163da0b21b99c7656bfd5a9519d319`.

## Unprofiled A/B and strict timeline

[Job `fabf059b5219`](../../.gpu_queue/logs/fabf059b5219.log) passed22 CPU analyzer
checks, then ran four separate baseline preconditioning processes followed by
six ABBAAB attempts (140 warmup/100 measured steps each). All attempts succeeded
and remain in [paired-cols2.jsonl](../../test_results/sparse_cols2_20260831/paired-cols2.jsonl),
SHA256 `4d52fde17f46f924dc9e69078beced2cdd474cd3039f095b675277e949622a22`.

| Variant | Three run means, ms | Median of run means, ms |
|---|---|---:|
| BN-mask baseline | 23.1482,23.1501,23.1728 | 23.1501 |
| X2 | 23.1270,23.1244,23.1220 | 23.1244 |

The median difference is0.0257ms, about+0.1111% reciprocal throughput. All active
samples were780MHz; clocks were not locked. Baseline temperatures81-84C and
X2 temperatures82-84C overlap. This tiny difference is not treated as evidence
of a materially faster trainer. The Docker daemon had restarted between earlier
BN measurements and this series; the unchanged baseline itself now runs faster
than its earlier23.8682ms snapshot. That cross-session difference is not credited
to X2, and its precise cause has not been established.

[The strict Nsight result](../../test_results/sparse_cols2_20260831/cols2-warmed-profile-summary.json)
passes all four measured steps after100 warmup steps: one exact sparse replacement,
block256->128, grid5184x10 unchanged;396 kernels,71 memsets/156136 bytes and68
device-to-device copies/78000 bytes. Every other kernel name/geometry/order and
memory event is unchanged. Mean measured steps2-4 consumer duration is
3.743909->3.650903ms, and whole-step span24.464062->23.747041ms. The baseline
trace is from the earlier session: most of that span difference is outside the
changed consumer and is not an X2 speedup claim. The simultaneous unprofiled
A/B above is the relevant whole-step comparison.

## NCU diagnosis

[Job `26c76981e6fe`](../../.gpu_queue/logs/26c76981e6fe.log) profiles one production
consumer after100 warmup invocations, not the oracle-warmed unit. It uses
`--cache-control all --clock-control none`,14 replay passes per kernel. The
profiler warns about unmodified clocks. The reported GPC clocks are779.986736
and779.988276MHz; these diagnostic replay durations are not unprofiled step times.

| Metric | Baseline | X2 |
|---|---:|---:|
| Consumer duration, ms | 3.722368 | 3.697984 |
| Executed warp instructions | 221393760 | 149106640 |
| Registers/thread (allocated) | 40 (40) | 26 (32) |
| Achieved warp occupancy, % | 95.419381 | 98.013565 |
| L2 throughput, % of peak | 95.212144 | 95.290788 |
| L2 sector hit rate, % | 80.054064 | 80.016530 |
| DRAM reads, MB | 609.422208 | 607.017856 |
| DRAM writes, MB | 54.068096 | 54.012800 |
| Eligible warps/cycle | 1.140707 | 0.509154 |

Sources: [baseline CSV](../../test_results/sparse_cols2_20260831/baseline-consumer-ncu.csv),
[X2 CSV](../../test_results/sparse_cols2_20260831/cols2-consumer-ncu.csv).
Instruction count falls by about33%, while memory traffic and near-saturated L2
throughput barely change. This supports an L2-traffic bottleneck, not an
instruction-count bottleneck. Warp-stall ratios have different instruction
denominators and must not be read as absolute stall-time changes.

Next experiment: narrower per-bin feature stripes and multiple independent bins
within a CTA, preserving each scalar FP32 recurrence. This aims to improve cache
reuse rather than merely reduce address instructions. Ampere's cache and
occupancy tradeoffs are described in the
[NVIDIA tuning guide](https://docs.nvidia.com/cuda/ampere-tuning-guide/index.html).
Results for that different layout are recorded in the
[WarpBins rejection audit](2026-08-31-sparse-gradient-warp-bins.md).

Raw queue logs and `test_results/` are ignored local artifacts and may not be
present in a clean checkout. No T4, multi-GPU, ideal saturation or training-plugin
release claim follows from this experiment.
