# SM86 BN backward mask fusion: staged evidence

Date: 2026-08-31. Original p888, batch 4096, RTX 3070 Laptop / SM86;
logical/physical widths 2556/2560 and 218/224, 16 residual blocks, scalar output.
The original BN/ReLU/residual, MSE and Adam semantics and existing FP16 GEMM
operand / FP32 accumulation contract are unchanged.

Both stages have passed the recorded correctness gates and strict trace checks.
The separate phase-2 confirmation, against phase 1, records a median of three
run means per variant **24.1523 -> 23.8682 ms** (-1.1763% latency, +1.1903%
reciprocal throughput). All active clock samples in that confirmation were
780 MHz, but clocks were not locked; this is **not a controlled-clock speedup
claim**. The validated phase-2 trace records exactly 16 residual-mask folds,
412 -> 396 kernels per step, with other event sequences unchanged.

Earlier phase-1 evidence records **24.5378 -> 24.1517 ms** (-1.5735% latency,
+1.5986% reciprocal throughput), descriptively. The two separately collected
A/B series are not combined into a new end-to-end speedup estimate.

## Binary identities and scope

| Stage | Benchmark artifact at the recorded job | SHA256 |
|---|---|---|
| Existing BN-backward half-mirror baseline | `/tmp/mgt-bn-backward-half` | `48f54cbb9fac4303d838125b55aa289ea49aabda13c4ed64e174e033016aa2da` |
| Phase 1: ReLU masks folded into BN | `/tmp/mgt-bn-backward-relu` | `2844b2c4403dea3c186b1c8aa6d62cd05a62270e3543773b903461e3fb9a766f` |
| Phase 2: residual masks also folded into BN | `/tmp/mgt-bn-backward-mask` | `ca5b2a2300e7c39e0466d52d57627ecc251a715a6f840edf3437ffa6d1cd05eb` |

The first two binaries were frozen and identified in
[phase-1 A/B job `5e47912b8939`](../../.gpu_queue/logs/5e47912b8939.log).
The phase-2 SHA is recorded by
[gate job `03c48a4708b0`](../../.gpu_queue/logs/03c48a4708b0.log) and frozen/rechecked
by [final gate job `a25f3db19f98`](../../.gpu_queue/logs/a25f3db19f98.log).
All quoted GPU executions came from
the shared queue. Reported arena size remains 739,806,720 bytes in these benchmark
runs. This audit makes no convergence, optimality, T4, or multi-rank scaling claim.

## Preserved arithmetic and data lifetimes

[The backward descriptor](../../native/cuda/mgt_cuda/local_batch_norm.cuh) retains
`half_output` first and adds optional `activated` and `residual_grad`. A null
`activated` keeps plain BN. Otherwise the original FP32 post-ReLU activation
selects `activated[i] > 0 ? incoming_dy[i] : +0`. The predicate is not evaluated
from a rounded half activation, and the mask is not multiplication by a boolean.

[The CUDA implementation](../../native/cuda/local_batch_norm.cu) applies this
select in both partial statistics and the final BN derivative. The existing
32-column x 8-row CTA, 256-row partial tile, per-lane row-add order, lane-fold
order, atomic accumulation, statistics clears, and two statistics copies are
retained. The final derivative remains the original FP32 expression:

```text
scale = gamma[c] * inv_std[c] / rows
dx[i] = scale * (rows * masked_dy[i] - dbeta[c] - normalized[i] * dgamma[c])
```

A masked-off element can have nonzero `dx` because BN couples rows. In particular,
the literal supplied-statistics fixture has masked `dy=0`, `dbeta=2`, and
`gamma=inv_std=rows=1`, so `dx=-2`, not zero. A masked-off NaN incoming gradient is
selected away, but `0 * NaN` from a NaN normalized value must still propagate
through the BN arithmetic. NaN and nonpositive activations select positive zero;
positive activations preserve the incoming value, including negative zero.

When requested, `residual_grad` receives incoming **masked dY, not BN dx** on every
physical lane. Positive padded activations therefore retain their incoming
residual gradients. BN `dx` and its RN half mirror remain positive zero in padding.
The input gradient is captured before an exact in-place `dy == dx` store.

[Trainer integration](../../native/cuda/mlp_batch_norm_forward.cu) is staged:

| Local production site | Count | Phase 1 | Phase 2 |
|---|---:|---|---|
| Residual FC1 and hidden BN | 17 | Fused ReLU + BN + half mirror | Same |
| Input BN | 1 | Fused ReLU + BN, half off | Same |
| Residual FC2 BN | 16 | Separate `ResB`, then plain BN + half mirror | Fused residual mask + BN + half mirror |

Strict-legacy and grouped/exact input-gradient routes select alternative input
implementations; they do not execute two input BN calls. Input BN remains
explicitly half-off even when `hd1 == hd2`. At each FC2, `d_residual` survives
the FC2 GEMMs and FC1 BN/GEMMs until the unchanged `AddInPlace`; the next block
reuses it only afterward on the same stream.

The existing 33 dense BN producers publish the FP16 mirror of their authoritative
FP32 `dx`. The adjacent dW/dX GEMMs consume this mirror; they never substitute
`d_residual` for it. The public FP16-step wrapper still clears cache tags on entry
and every exit, and its active B range is disjoint from residual and other live
storage. See [wrapper/preflight](../../native/cuda/local_mlp_batch_norm.cu).
No new buffer or precision change is introduced by masking.

The nonlocal build retains separate `ReluB`/`ResB` followed by the existing
[synchronous-BN selector](../../native/cuda/mgt_cuda/sync_batch_norm_selector.cuh).
Global-row normalization, NCCL calls and residual addition order are unchanged.

New epilogue aliases are checked before full-BN statistics are cleared.
`activated` may share read-only dy/normalized storage but not dx or full-backward
writable dgamma/dbeta/statistics. Residual output requires activated and is
disjoint from the live float views and half mirror. Exact dy/dx in-place is
supported; partial overlap is rejected. These BN checks do not promise rollback
of forward work already performed by an invalid whole-step call.

## Numerical and safety gates

The independent test is
[test_local_batch_norm_backward_mask.cu](../../native/tests/cuda/test_local_batch_norm_backward_mask.cu).
Its oracle materializes the old mask, then uses independent old partial/apply
kernels; it does not call the fused API to construct its expected result.
The fields-only implementation failed at runtime in
[RED job `d8ffdf3843a1`](../../.gpu_queue/logs/d8ffdf3843a1.log):
`dx[11] actual=5/0x40a00000 expected=-2/0xc0000000`. Compilation succeeded;
the ignored mask, rather than a build error, caused the failure.

[Unit GREEN job `82b35a6ee6eb`](../../.gpu_queue/logs/82b35a6ee6eb.log) records:

- Four targeted local BN CTests passed, including existing forward/backward
  epilogue and plain-BN regressions.
- New full suite: 20 supplied-stat Apply fixtures, 7 full-backward fixtures,
  and 63 invalid/pre-write probes. Quick suite: 13 Apply, 4 full-backward,
  and the same 63 rejection probes.
- Full memcheck: 0 errors and 0 leaked bytes; full initcheck: 0 errors.
  Quick racecheck: 0 hazards/errors/warnings; quick synccheck: 0 errors.

Coverage includes RN half ties, underflow/overflow, finite FP32 bit patterns,
signed zero, NaN classification, masked-off nonzero BN dx, normalized-NaN
propagation, and literal padded residual values `{-0, -7, +0}`. Shapes include
odd stride, rows 1/3/17/255/256/257/513/4096, production 218/224 and generic
2556/2560, reused capacities, in-place/out-of-place, optional outputs, read-only
activation aliases, allocation canaries and byte-preserved inputs/tails.

Supplied-stat Apply requires exact old-GPU FP32 results, except that NaN payloads
are compared by classification. Full-backward comparisons are exact for a single
row-partial CTA and for the multi-CTA dyadic fixtures, whose finite sums are
exactly representable independently of atomic arrival order. General decimal
multi-CTA data uses tolerances against the old GPU composition and independent
double-precision CPU sums/formula. This does not assert arbitrary training runs
are bit-identical or prove long-run convergence.

Both [phase-1 integration job `87e6b52436d3`](../../.gpu_queue/logs/87e6b52436d3.log)
and [phase-2 integration job `03c48a4708b0`](../../.gpu_queue/logs/03c48a4708b0.log)
passed the same seven targeted CTests: local FP32 full step, local FP16 full step,
activation tape, gradient overwrite, grouped-rows full step, NCCL gradient
overwrite, and shared-source full backward. Each job also records activation-tape
memcheck/leak-check with 0 errors/0 leaked bytes, gradient-overwrite initcheck with
0 errors, and an original-p888 batch-4096 complete step under memcheck with
`status=ok` and 0 errors. Sanitizer step times are not performance measurements.
The NCCL tests here exercise single-rank compatibility, not multi-rank execution.

The activation-tape regression retains CPU mixed/FP32 gradient oracles, poisoned
gradient buffers, stale-cache rejection and lifecycle checks, and the equal-width
input/hidden fixture. The overwrite regression covers padded physical gradients,
Adam state and changed-row reuse; no gradient poison gate was removed.

[Final phase-2 gate job `a25f3db19f98`](../../.gpu_queue/logs/a25f3db19f98.log)
subsequently passed 21 targeted CUDA CTests, four CPU/C-ABI CTests, and the actual
Rust `single_gpu_ffi` test (1 passed). Full activation-tape memcheck/leak-check and
initcheck reported 0 errors, with 0 leaked bytes; local and NCCL-world1 overwrite
memcheck/leak-check also reported 0 errors and 0 leaked bytes. The rebuilt
benchmark compared byte-identically with the frozen phase-2 binary and retained
the SHA above. This expands targeted coverage without implying a multi-rank run.

## Phase 1 unprofiled A/B: all six runs retained

Original p888 arguments were batch 4096, warmup 140, measured steps 100.
Order was baseline, ReLU, ReLU, baseline, baseline, ReLU. Each number below is
a benchmark's **run mean**, not an individual step latency. The separate initial
baseline preconditioning run (19.6187 ms in the queue log) is not one of these
six paired attempts. All six attempts succeeded and are retained in
[paired-relu.jsonl](../../test_results/bn_backward_mask_20260831/paired-relu.jsonl).

| Variant | Three run means, ms | Median of run means, ms |
|---|---|---:|
| Half-mirror baseline | 24.5372, 24.5425, 24.5378 | 24.5378 |
| Phase 1 ReLU fusion | 24.1549, 24.1453, 24.1517 | 24.1517 |

The median difference is 0.3861 ms: -1.5735% latency, or +1.5986% reciprocal
throughput. [The summary](../../test_results/bn_backward_mask_20260831/paired-relu-summary.json)
explicitly classifies this as descriptive. Active telemetry means utilization
at least 90% with nonzero memory usage. The first baseline run included
780-1215 MHz active samples; the other five runs' active samples were 780 MHz.
All six final-two-second active sample sets were
780 MHz, but that diagnostic tail is not the entire timed interval. Sampling
also spans initialization/warmup and is not aligned to timed steps. These sparse
snapshots do not establish controlled clocks, and no run was discarded to make
that claim. Benchmark loss values are not used as a convergence comparison.

## Phase 1 Nsight: measured structure and timing boundaries

[A/B and profiling job `5e47912b8939`](../../.gpu_queue/logs/5e47912b8939.log)
recorded the candidate after 100 warmup steps with four measured steps. The
frozen baseline trace is
[mirror-warmed.sqlite](../../test_results/bn_backward_half_20260831/mirror-warmed.sqlite),
SHA256 `9a008c63775ee0eda44028ca2a94e46996bb786b953ddf8a9d5a93a4e889c673`.
The candidate is [relu-warmed.sqlite](../../test_results/bn_backward_mask_20260831/relu-warmed.sqlite),
SHA256 `c3d44e271fc7c2949274b6ba9e2f618543e12c7271f226fbd0fa9168f9536fff`.

[The analyzer](../../test_results/bn_backward_mask_20260831/analyze_nsys.py) skips
100 warmup boundaries, starts each step at `RandomWalkKernel`, and ends it at
the second terminal Adam kernel. It validates all four measured steps; the table
uses the mean of measured steps 2-4, excluding measured step 1. Each step contains
one ordered training stream, with no memcpy/memset crossing its boundaries.
[Raw summary](../../test_results/bn_backward_mask_20260831/relu-warmed-profile-summary.json):

| Metric per step | Half-mirror baseline | Phase 1 |
|---|---:|---:|
| Kernel count | 430 | 412 |
| Standalone ReLU / residual-mask counts | 18 / 16 | 0 / 16 |
| Backward BN partial / apply counts | 34 / 34 | 34 / 34 |
| Kernel-duration sum, ms | 23.094147 | 22.603508 |
| Standalone ReLU duration, ms | 0.972909 | 0 |
| Standalone residual-mask duration, ms | 0.753172 | 0.750546 |
| Backward BN partial duration, ms | 1.169221 | 1.459993 |
| Backward BN apply duration, ms | 1.686333 | 1.971744 |
| Memset count / bytes | 71 / 156,136 | 71 / 156,136 |
| Memset duration, ms | 0.137023 | 0.127496 |
| Memcpy count / bytes | 68 / 78,000 | 68 / 78,000 |
| Memcpy duration, ms | 0.191277 | 0.190717 |
| Kernel+memcpy+memset interval union, ms | 23.422447 | 22.921721 |
| Whole-step GPU span, ms | 25.263077 | 24.623815 |
| Span not covered by those activities, ms | 1.840630 | 1.702094 |

The strict trace comparison associates each of the 18 removed ReLU masks with
its BN site. It checks 17 masked half-output applies, one masked float-only
input apply, and 16 unchanged plain half-output FC2 applies. After normalizing
those intended replacements, all other kernel names/geometry/order and the
memcpy/memset event sequences match across all four steps. The 68 observed
copies have `copyKind=8, srcKind=2, dstKind=2` (device-to-device), two immediately
after each BN apply: 66 copies of 872 bytes and two copies of 10,224 bytes.

The fused partial/apply kernels now consume the activation predicate, and their
recorded aggregate durations rise. ReLU + BN partial + BN apply sums decrease
from 3.828463 to 3.431737 ms, approximately 0.396727 ms. The total kernel sum
decreases by 0.490639 ms and whole-step span by 0.639262 ms; these are different
quantities. The latter also contains changed transfer durations and uncovered
intervals. It is not attributable in full to the removed masks. The uncovered
interval is only the complement of recorded GPU activity, not a demonstrated
CPU bottleneck or a second measure of kernel time. These instrumented spans are
also separate from the unprofiled run means above.

## Phase 2 NCU diagnostic: not a whole-step comparison

[Job `94258815e8ec`](../../.gpu_queue/logs/94258815e8ec.log) captured the first FC2
partial/apply pair after 100 warmup steps: the filter matches backward partial
and apply kernels, skips 6800 matching launches (100 x 34 x 2), and captures two.
Each captured kernel used eight profiling passes. Clock and cache controls were
both `none`; both profiler logs explicitly warn about uncontrolled caches and
unmodified clocks. The clock metric is `gpc__cycles_elapsed.avg.per_second`,
whose CSV unit is **GHz**, not MHz.

| Captured kernel | Phase-1 duration, us | Phase-2 duration, us | Phase-1 GPC clock, GHz | Phase-2 GPC clock, GHz |
|---|---:|---:|---:|---:|
| First FC2 backward partial | 25.792 | 37.664 | 1.702589 | 1.674287 |
| First FC2 backward apply | 37.088 | 56.576 | 1.686179 | 1.673893 |

Sources: [phase-1 CSV](../../test_results/bn_backward_mask_20260831/relu-fc2-ncu.csv),
[phase-2 CSV](../../test_results/bn_backward_mask_20260831/mask-fc2-ncu.csv), and
[phase-1](../../test_results/bn_backward_mask_20260831/relu-fc2-ncu.log) /
[phase-2](../../test_results/bn_backward_mask_20260831/mask-fc2-ncu.log) replay logs.
These are neither steady-780-MHz nor cold-cache-controlled measurements.
The selected pair excludes the removed standalone `ResB`, so its two durations
alone cannot establish the net cost or benefit of phase 2. Profiler-instrumented
whole-process benchmark times from this job are not unprofiled latency evidence.

## Phase 2 unprofiled A/B: separate confirmation

Here the baseline is the frozen **phase-1 ReLU binary**, not the earlier
half-mirror baseline. Both phase-2 series use original p888 batch 4096,
140 warmup steps, 100 measured steps, and ABBAAB order. Each reported latency
is a run mean; every paired attempt is retained in its own dataset.

The first [A/B and profiling job `1c028a1abb23`](../../.gpu_queue/logs/1c028a1abb23.log)
completed all six benchmark attempts successfully. The separate initial
baseline preconditioning run was 19.0048 ms. Its
[paired-mask.jsonl](../../test_results/bn_backward_residual_20260831/paired-mask.jsonl),
SHA256 `3ea607790ce5966b700ea2dbebc772ee4bc8349250e156233e4c456657c54065`,
contains:

| First series, not accepted as a speedup | Three run means, ms | Median of run means, ms |
|---|---|---:|
| Phase 1 baseline | 19.1340, 22.8508, 24.1449 | 22.8508 |
| Phase 2 residual fusion | 19.0050, 19.0887, 23.8721 | 19.0887 |

[Its summary](../../test_results/bn_backward_residual_20260831/paired-mask-summary.json)
shows active sampled clocks spanning 780-1710 MHz; even the final-two-second
sample sets are not matched. The apparent +19.7085% reciprocal-throughput change
is **not accepted as the effect of fusion**. This series is preserved, not
silently discarded or pooled with the confirmation.

[Confirmation job `e187231751e0`](../../.gpu_queue/logs/e187231751e0.log) exited 0.
It first ran four independent phase-1 preconditioning processes with means
19.1586, 19.2362, 19.6314 and 24.0201 ms. Those processes preceded the paired
series and are not paired samples or post-hoc outlier removals. The subsequent
six ABBAAB attempts all succeeded:

| Confirmation variant | Three run means, ms | Median of run means, ms |
|---|---|---:|
| Phase 1 baseline | 24.1408, 24.1600, 24.1523 | 24.1523 |
| Phase 2 residual fusion | 23.8699, 23.8662, 23.8682 | 23.8682 |

Source: [paired-mask-confirm.jsonl](../../test_results/bn_backward_residual_20260831/paired-mask-confirm.jsonl),
SHA256 `a25d3cdec62a3295010262f86af6f1258276a0f7f465d29e7e561f4df1ea7024`,
and [its summary](../../test_results/bn_backward_residual_20260831/paired-mask-confirm-summary.json).
All three attempts per variant retain the respective `2844...` / `ca5b...`
binary identities listed above. The median difference is 0.2841 ms:
-1.1762855% latency, or +1.1902867% reciprocal throughput.

Every active and final-two-second active clock sample in this confirmation
was 780 MHz. Baseline active temperatures were 75-79 C; phase 2 was 76-79 C.
The same utilization-at-least-90%/nonzero-memory filter is used as in phase 1.
These are asynchronous samples spanning initialization/warmup as well as work,
not clock controls or measurements aligned to each timed step. The confirmation
therefore supports a small measured improvement under matched sampled clocks,
not a locked-clock causal estimate. No convergence inference is made from the
benchmark's loss values.

## Phase 2 Nsight: validated 16 residual-mask folds

Job `1c028a1abb23` generated and exported the phase-2 trace after 100 warmup
steps with four measured steps. Its GPU work succeeded, but the subsequent
CPU analyzer raised `NameError: name 'relu' is not defined`, causing queue
exit code 2. The [original failed summary](../../test_results/bn_backward_residual_20260831/mask-warmed-profile-summary.json)
is preserved; it is not the validation result used here.

The corrected [analyzer](../../test_results/bn_backward_residual_20260831/analyze_nsys.py)
produced the separate
[successful recheck summary](../../test_results/bn_backward_residual_20260831/mask-warmed-profile-summary-recheck.json),
SHA256 `718c8d9ecffb0271d76100d193a606d83d43e1a2cc171858400068b08090f255`.
Its [regression fixtures](../../test_results/bn_backward_residual_20260831/test_helpers.py)
include the actual SQLite-to-comparison path.
It compares the phase-1 `relu-warmed.sqlite` identified above with
[mask-warmed.sqlite](../../test_results/bn_backward_residual_20260831/mask-warmed.sqlite),
SHA256 `6cabe52687cb00656f95068e17a2b54dd4163da0b21b99c7656bfd5a9519d319`.
The status is `ok`: all four measured steps pass strict site-associated event
checks. As in phase 1, the table averages only measured steps 2-4, from
`RandomWalkKernel` through the second terminal Adam kernel. There is one
ordered training stream and no boundary-crossing memcpy/memset activity.

| Metric per step | Phase 1 | Phase 2 |
|---|---:|---:|
| Kernel count | 412 | 396 |
| Standalone ReLU / residual-mask counts | 0 / 16 | 0 / 0 |
| Backward BN partial / apply counts | 34 / 34 | 34 / 34 |
| Kernel-duration sum, ms | 22.603508 | 22.450786 |
| Standalone residual-mask duration, ms | 0.750546 | 0 |
| Backward BN partial duration, ms | 1.459993 | 1.635156 |
| Backward BN apply duration, ms | 1.971744 | 2.309694 |
| Memset count / bytes | 71 / 156,136 | 71 / 156,136 |
| Memset duration, ms | 0.127496 | 0.137295 |
| Memcpy count / bytes | 68 / 78,000 | 68 / 78,000 |
| Memcpy duration, ms | 0.190717 | 0.190921 |
| Kernel+memcpy+memset interval union, ms | 22.921721 | 22.779001 |
| Whole-step GPU span, ms | 24.623815 | 24.464062 |
| Span not covered by those activities, ms | 1.702094 | 1.685061 |

Each measured step has exactly 16 additional `ResB` folds at residual BN sites:
34 masked BN partials, 17 mask+half applies, 16 mask+residual+half applies, and
one input mask-only float apply. After canonicalizing only those intended
replacements, the existing 18 ReLU sites and all other kernel names, launch
geometry and order (including GEMMs), plus memcpy/memset event sequences,
match. All 68 copies remain device-to-device (`copyKind=8, srcKind=2,
dstKind=2`): 66 x 872 bytes and 2 x 10,224 bytes, associated with the same BN
statistics-copy sites. No transfer or clear was removed by this mask fusion.

Residual mask + BN partial + BN apply sums decrease from 4.182283 to 3.944850 ms,
a 0.237433 ms difference. This is distinct from the 0.152723 ms decrease in
all kernel durations, the 0.142720 ms decrease in activity union, and the
0.159753 ms decrease in whole-step span. The latter also includes transfer
duration changes and a 0.017033 ms decrease in uncovered intervals. Neither
the full-span change nor the uncovered interval is attributed wholly to the
removed masks. These instrumented measurements are separate from the
23.8682 ms unprofiled confirmation median and from the selected NCU kernels.

Queue logs and `test_results/` links are local raw/ignored evidence paths; they
may not be present in a clean source checkout. The audit records their identities
and does not treat their absence from version control as a new execution result.
