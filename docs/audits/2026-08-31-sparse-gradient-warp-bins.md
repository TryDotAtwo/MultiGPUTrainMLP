# SM86 sparse-gradient layout experiments: no production promotion

Date: 2026-08-31. RTX3070 Laptop, original p888, batch4096. This follows
[two-column ownership](2026-08-31-sparse-gradient-two-columns.md).

Decision: retain the original production consumer. WarpBins32 and WarpBins64
pass numerical/memory gates but regress whole-step performance. X2's earlier
0.11% difference is too small to justify promotion. All three candidates now
live in a [test-only header](../../native/tests/cuda/sparse_input_grad_layout_candidates.cuh);
there is no new architecture policy, precision change, allocation or production
kernel in this checkpoint.

## Hypothesis and arithmetic contract

Each thread still computes one complete ascending-row FP32 sum. Instead of a
256-feature stripe for one bin, a CTA handles several independent bins with
narrower feature stripes. There is no partial-sum reassociation, shared memory,
atomic accumulation or cross-warp communication. Invalid bin/feature lanes may
return independently; every valid output, including empty bins, is overwritten.

| Mapping | Bins/CTA | Features/bin | Production grid | Threads/CTA |
|---|---:|---:|---|---:|
| Baseline | 1 | 256 | 5184x10 | 256 |
| WarpBins32 | 8 | 32 | 648x80 | 256 |
| WarpBins64 | 4 | 64 | 1296x40 | 256 |

All use51,840 CTAs. One4096-row feature stripe spans4MiB,512KiB or1MiB
respectively. This is a locality hypothesis, not a reduction in the number of
mathematical dZ loads. Cache reuse and concurrent CTA placement still determine
actual traffic. For device-cache context, see the
[NVIDIA Ampere tuning guide](https://docs.nvidia.com/cuda/ampere-tuning-guide/index.html).

During the recorded experiments only the local SM86 grouped-row consumer was
changed. The builder, buffer ownership, device/shape cache, insufficient-scratch
fallback, nonlocal path and other architectures were unchanged. Diagnostics
reported selected_features_per_bin; actual geometry was independently checked
in Nsight. This temporary host routing was removed after the comparisons.

## Correctness gates

The [shared regression](../../native/tests/cuda/test_sparse_input_grad_grouped_rows.cu)
uses an independent fixed256-thread old-grid GPU oracle, serial CPU arithmetic,
literal cancellation/count witnesses, poisoned outputs/unused row IDs, immutable
inputs and allocation canaries. Candidate geometry never controls oracle geometry.
Coverage is37 full cases and6 quick cases per candidate; it includes partial
bin groups, widths1/31/32/33/127/128/129/255/256/257/511/513/2556/2560, row-count
tails, allocation reuse through4097/33/4096/31/4095/32/1/4097 rows, all-empty dZ,
skewed bins, tight final-row allocations, signed zero/subnormals/NaNs/infinities
and production-sized synthetic matrices. The three-orbit fixture is synthetic
coverage, not a measured random-walk histogram.

- [RED b62fbf3f8c92](../../.gpu_queue/logs/b62fbf3f8c92.log): compilation succeeded;
  the old one-bin body under the new launch leaves bin1,column0 poisoned.
- [Unit GREEN c0a8473b5fdd](../../.gpu_queue/logs/c0a8473b5fdd.log): both32 and64
  pass37 cases, full memcheck/leak-check and initcheck, plus quick racecheck and
  synccheck; all report zero errors/hazards/leaks.
- [Warp32 integration 477c6e862cb9](../../.gpu_queue/logs/477c6e862cb9.log) and
  [Warp64 integration bfa4fd3d1be4](../../.gpu_queue/logs/bfa4fd3d1be4.log): each
  passes10 CTests, activation-tape memcheck/leak-check, gradient-overwrite
  initcheck and full production batch4096 memcheck. Sanitizer walltime is not
  a performance measurement.

These gates establish the tested scalar-recurrence/ownership contract, not
arbitrary-length bit-identical training. Cross-CTA BN reductions are unchanged.

After moving the rejected kernels into test-only storage,
[final jobff6af9a50cf7](../../.gpu_queue/logs/ff6af9a50cf7.log) passes24 CUDA
regressions, four CPU/C-ABI tests and one Rust FFI test. Activation-tape
memcheck/leak-check and initcheck, and local/nonlocal gradient-overwrite
memcheck/leak-check report zero errors. The rebuilt production benchmark is
byte-identical (`cmp`) to the frozen ca5b2a23 baseline; production code is unchanged.

## Frozen identities

| Variant | Executable | SHA256 |
|---|---|---|
| Baseline | /tmp/mgt-bn-backward-mask | ca5b2a2300e7c39e0466d52d57627ecc251a715a6f840edf3437ffa6d1cd05eb |
| Warp32 | /tmp/mgt-sparse-warp32 | d37ff6b7f52d0cf6ff4301b75a9f3b2cae488716d995a557c4737c903dd66ee4 |
| Warp64 | /tmp/mgt-sparse-warp64 | 2ed78f742588970a122376e8494e204944c42141d8cdb26c52c6ff65723647f9 |

Same input hashes as the preceding audit,739806720-byte arena, FP32 master/Adam/
sparse gradients, FP16 dense operands and FP32 dense accumulation. Every paired
run checks executable SHA256 before and after execution.

## Unprofiled ABBAAB, separate datasets

Each series begins with four separate baseline preconditioning processes, then
all six140-warmup/100-measured-step runs.240 total steps fit within the244-full-
batch epoch. Values below are medians of three run means, not per-step medians.
No attempt is discarded. Each series passed22 CPU analyzer/summarizer tests.

| Series | Baseline run means, ms | Candidate run means, ms | Median baseline -> candidate, ms | Latency change |
|---|---|---|---|---:|
| Warp32 | 23.2635,23.2721,23.1499 | 23.4973,23.5027,23.4052 | 23.2635 ->23.4973 | +1.0050% |
| Warp64 | 23.1495,23.1723,23.1464 | 23.2365,23.2047,23.1927 | 23.1495 ->23.2047 | +0.23845% |

All active samples (utilization>=90%, allocated memory>0MiB) are780MHz in both
series; clocks are not locked and sampling includes initialization/warmup.
Temperatures overlap: Warp32 baseline77-82C/candidate79-82C; Warp64 both81-84C.
The earliest Warp32 preconditioning runs were18.8-19.0ms and are not comparison
samples. The unchanged baseline moved between sessions; no cross-session
baseline change is credited to a kernel optimization.

Raw evidence:

- [Warp32 job3c640dfb07ba](../../.gpu_queue/logs/3c640dfb07ba.log),
  [paired JSONL](../../test_results/sparse_warp_bins_20260831/paired-warp32.jsonl),
  SHA2567f179447784f386b9f9262b44020272306e44b11bdebe2c7b9307fadcc3c310c.
- [Warp64 job066c79d9c922](../../.gpu_queue/logs/066c79d9c922.log),
  [paired JSONL](../../test_results/sparse_warp64_20260831/paired-warp64.jsonl),
  SHA256d0a6bc3a9077ff3ccd57151ae77fcfeca52059081de41080f963439bc610739d.

## Strict Nsight Systems comparison

Each trace has100 warmup/four measured steps. Both analyzers require exactly one
sparse replacement per measured step, the table's exact geometry,396 kernels,
71 memsets/156136 bytes and68 DtoD copies/78000 bytes. All other kernel names,
geometry, order and memory-event sequences must match. CPU tests exercise the
real baseline SQLite reader/comparator and inject failures in names, geometry,
counts, bytes, copy kind, stream and order; no synthetic fixture is a GPU result.

| Trace | Sparse consumer, ms | All kernels, ms | Step span, ms |
|---|---:|---:|---:|
| Historical baseline | 3.743909 | 22.450786 | 24.464062 |
| Warp32 | 3.957207 | 22.051188 | 24.046308 |
| Warp64 | 3.743603 | 21.851250 | 23.847663 |

These are means of measured steps2-4. The baseline trace is from an earlier
session: lower overall spans in the candidates do not demonstrate a speedup.
The simultaneous unprofiled comparisons above are the relevant evidence.

- [Warp32 SQLite](../../test_results/sparse_warp_bins_20260831/warp32-warmed.sqlite),
  SHA256942cdeb3f8183de0aba4b18535e9203668fc698e5650e463d61d73b00bff8395.
- [Warp64 SQLite](../../test_results/sparse_warp64_20260831/warp64-warmed.sqlite),
  SHA256ac4aa42628909fb48c16948f2c30004da348ae7bf6b130c61be669e4082cd0b7.

## NCU limitation and next target

Kernel-replay job3f999e5b7904 failed with exit9 after reporting incompatible
driver/internal module loading and LaunchFailed. Its partial report is retained
but is not accepted as counter evidence; Warp64 was not reached in that job.
Application-replay [job20a3143b0db9](../../.gpu_queue/logs/20a3143b0db9.log)
completed both candidates and CSV exports with exit0,15 application passes each,
without a driver or environment change. The incompatible-driver/internal-module
warnings remain; these are diagnostic counters, not independent acceptance of
the whole profiler installation. NVIDIA documents that this mode reruns the
application instead of saving/restoring kernel memory in the
[Nsight profiling guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#application-replay).

| Counter | Prior baseline kernel replay | Warp32 application replay | Warp64 application replay |
|---|---:|---:|---:|
| Consumer duration, ms | 3.722368 | 3.952256 | 3.713472 |
| GPC clock, MHz | 779.986736 | 779.987177 | 779.987211 |
| Warp instructions | 221393760 | 226244640 | 226287814 |
| Registers/thread | 40 | 40 | 40 |
| Achieved warp occupancy, % | 95.419381 | 43.213880 | 64.253545 |
| L2 throughput, % of peak | 95.212144 | 82.967070 | 90.807488 |
| L2 sector hit rate, % | 80.054064 | 98.165345 | 97.388188 |
| DRAM read, MB | 609.422208 | 51.113216 | 76.220288 |
| DRAM write, MB | 54.068096 | 52.081152 | 52.154624 |
| Eligible warps/cycle | 1.140707 | 0.832249 | 1.224559 |

[Warp32 CSV](../../test_results/sparse_warp_bins_20260831/warp32-consumer-app-ncu.csv)
and [Warp64 CSV](../../test_results/sparse_warp_bins_20260831/warp64-consumer-app-ncu.csv).
All use100 warmup launches, cache-control=all and clock-control=none; replay mode
differs from the earlier baseline. The locality hypothesis is supported by the
traffic counters, but occupancy falls enough that unprofiled training does not
improve. Mixing empty/occupied bin warps within a CTA is a plausible explanation,
not a separately measured cause. A future compaction experiment would need to
preserve complete writes of empty gradient bins and ascending per-column sums.

Next bounded target:33 Bias kernels consume about0.88ms in the new traces.
Fuse bias reads into BN statistics/apply, retaining the old rounded FP32 input
sum, instead of materializing a separate biased matrix. That change needs its
own tests, full-step benchmark and trace; no gain is claimed here.

Queue logs and test_results are ignored local evidence, not bundled artifacts.
No T4, multi-GPU, long-run convergence or training-plugin release claim follows.
