# Accepted SM86 input-gather feature tiling

Date: 2026-08-31. Original p888, batch4096, RTX3070 Laptop/SM86.
Follow-up to [exact-tree column sums](2026-08-31-column-sum-tiling.md).

Full-step median of three run means **25.8899 ->25.2253ms** (-2.5670%
latency, +2.6347% throughput), with all active sampled clocks780MHz in the
confirmation series. One warmed Nsight Compute gather **2.445888 ->1.798816ms**.
Measured DRAM reads **746.282752 ->12.045440MB** per gather; the output and
mathematical weight demand are unchanged. Arena remains739806720 bytes.
These are local optimization results, not optimality, convergence or T4 claims.

## Ownership and numerical contract

Previously one256-thread CTA owned all2560 physical features of a row, looping
over five512-feature chunks. Now a128-thread CTA owns one row and one256-feature
tile. Production grid changes4096x1 ->4096x10, and each lane owns one half2 pair.
The grid's row index varies fastest within a feature tile. This favors cache
reuse; CUDA scheduling order is not a correctness assumption.

A5184-row FP16 weight slice of256 features is2.53125MiB, versus5.0625MiB for
512 features and25.3125MiB for the entire table. The measured device has4MiB L2.
This capacity calculation motivates T128, but does not guarantee cache residency:
output traffic, concurrent CTAs and cache policy still matter. The counters below
validate the traffic reduction on the actual production input.

The private header `native/cuda/input_half_tiled.cuh` shares the real kernel with
its isolated test. Each CTA initializes the72 live shared offsets, then all lanes
cross the barrier before feature-tail lanes return. The640-byte shared offset
array is unchanged. Bias and position0..71 are added in the original order with
explicit FP32 round-to-nearest adds, then written to the same FP32 activation.
The [CUDA12.8.1 intrinsic contract](https://docs.nvidia.com/cuda/archive/12.8.1/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__SINGLE.html)
defines this addition behavior. There is no half-precision accumulation, new
approximation, extra buffer or changed Adam/BN/model setting.

The dispatcher retains both even-physical and even-logical guards. Odd widths
continue through the unchanged scalar fallback. Index/address arithmetic in the
tiled kernel uses64-bit products; padded feature pairs are written as positive
zero without reading poisoned weight padding. No new allocation or launch is added.

## Identity and correctness evidence

Docker `mgt-gpu-queue`, image `mgt-single-gpu-dev:2026-08-28`, CUDA12.8.93,
Release/Ninja SM86, driver572.70, Systems2025.6.3, Compute2025.1.1. All GPU work
went through the shared queue; no clock/power policy was changed.

| Binary | SHA256 |
|---|---|
| Accepted C4/C8 column-sum baseline, `/tmp/mgt-column-sum-c4c8` | `2e762d39635128cfbc9e690fe338b33fc74ffa714a4e3fc88eb59a1adcfbc741` |
| Tiled gather, `/tmp/mgt-input-half-t128` | `1b661b812751618932e63cac09c4e4cadcfeebed0b596a3d07af8785e8973212` |

Unlike the column-sum experiment, the full-step baseline predates private-header
extraction. The kernel extraction and ownership change are one production delta;
the original row-loop implementation is also preserved as an independent test
oracle. Nsight verifies that only the gather launch changed in the full step.

Runtime RED job`b4a95d1a2f42`: the extracted old row-loop kernel under the new
single-feature-tile launch wrote feature256, which belongs to another tile.
Expected untouched sentinel0xcdcdcdcd. This tests the new ownership contract;
the old implementation was correct under its original row-only launch.

Isolated GREEN job`c91124271a07`:19 fixtures x2 specializations =38 comparisons.
The CPU oracle independently decodes half bits and forces FP32 addition; a literal
cancellation/extremes witness distinguishes ordering and half accumulation.
Large cases additionally compare to the old row-loop GPU oracle. Coverage includes
partial/full feature grids, even widths254/256/258/510/512/514/1022/1024/1026/2560,
rows1/17/4096 plus shrink-to3 in reused capacity, production72x72 inputs, NaN
weight padding, invalid unused state padding, positive-zero outputs, canaries and
input immutability. Ownership is checked through partial-grid sentinels and the
mapping, not an instrumented atomic write-count counter.

Production gate job`1ecf39e623bc`:13/13 targeted CUDA CTests; all38 variants under
memcheck (0 errors,0 leaked bytes) and initcheck (0 errors);10 quick variants
under racecheck (0 hazards/errors/warnings) and synccheck (0 errors). The complete
production4096 training step passed memcheck. Existing direct-input tests still
cover odd physical/logical combinations. Job`d5bffc383623`:4/4 CPU/C-ABI tests and
the actual Rust FFI/RAII training-step test passed. This is not a full multi-GPU
regression. Independent code review found no actionable defects.

Publication recheck job`0920f9cd45b1`:6/6 targeted tests passed (both new exact
kernel tests and four existing public NCCL-single-rank backward/full-step tests).
The benchmark rebuilt to the same frozen candidate SHA256. Those one-rank
checks exercise shared-source compatibility, not multi-rank collectives.

## Unprofiled A/B: every run retained

Ignored local artifact directory: `test_results/input_half_tiled_20260831/`.
Runner records six A-B-B-A-A-B runs, frozen binary SHA256, args, metrics, stderr,
selected environment and asynchronous100ms nvidia-smi telemetry. Each run has140
warmup +100 timed steps, within the244-full-batch original epoch. Reported step_ms
is a run mean; the summary is not individual-step p50.

| Dataset / variant | Three run means, ms | Median, ms |
|---|---|---:|
| Initial row loop |25.9533,25.9784,25.9927|25.9784|
| Initial T128 |25.3377,25.3343,25.3337|25.3343|
| Confirmation row loop |25.9793,25.8500,25.8899|25.8899|
| Confirmation T128 |25.2280,25.2126,25.2253|25.2253|

Initial job`af7944582128`, `paired-t128.jsonl`: startup frequency drift in the
baseline, while all runs' last2s active samples were780MHz. Retained as a
descriptive result (+2.5424%), not a controlled comparison.

Confirmation job`d5bffc383623`, `paired-t128-confirm.jsonl`: every active sample
in all six runs780MHz; temperatures baseline76-83C, candidate78-83C. Active means
utilization>=90% and memory.used>0. Samples are sparse and not synchronized with
timed step boundaries: they support, but do not prove, constant clocks throughout
the entire measurement. No clocks were locked; no runs or losses were filtered.
Short diagnostic losses are not convergence validation.

The synthetic isolated benchmark used100 warmup +300 CUDA-event iterations in
one fixed order: old-row4438.118490us, T1281652.186483us, T2562479.267782us.
These are different inputs/cache conditions without matched clock telemetry.
They select a candidate; they are not the production speedup or a T4 tile rule.

## Profiling attribution and limitations

Systems job`c889aa7fdfbc`:100 warmup +4 final steps; means use steps102-104.
Reports `row-warmed.nsys-rep`, `t128-warmed.nsys-rep`, SQLite exports and
`t128-profile-summary.json`. All four final sequences contain463 kernels and
match after canonicalizing exactly one gather launch; all other geometries match.
Registers/thread40, shared640B, local memory/thread0 in both gathers.

Gather time2.425842 ->1.841469ms. The first baseline trace has much faster
unrelated compute: total kernel sum19.728905ms vs23.696826ms and step span
21.650982ms vs26.116739ms. Those unmatched profiler runs cannot measure end-to-end
speedup or forward-span change. Use the unprofiled paired confirmation above;
do not claim that the candidate regressed based on these profile spans.

Compute: original row loop collected in job`d5bffc383623`, T128 in
job`c889aa7fdfbc`; one gather after100 warmup steps,15 replay passes, default
cache flush, `--clock-control none`. Raw `row.csv` and `t128.csv` unit rows are
respected below; MB and GB are decimal. Replay traffic is not whole-step traffic.

| NCU metric | Row loop | T128 |
|---|---:|---:|
| Duration, ms |2.445888|1.798816|
| GPC clock, MHz |779.983929|779.983426|
| DRAM read, MB |746.282752|12.045440|
| DRAM write, MB |42.830720|41.373184|
| DRAM throughput, GB/s |322.628621|29.696547|
| L2 sector hit rate, % |52.138810|99.167031|
| Global-load requests |11972608|12083200|
| Global-load sectors |47847565|48007489|
| Executed warp instructions |123645952|128946176|
| SM throughput, % peak sustained |47.320804|66.436475|
| Achieved occupancy, % |94.345359|94.291422|
| Eligible warps/scheduler/cycle |1.152703|1.449946|

DRAM bandwidth utilization falls because much less DRAM traffic is needed,
not because the kernel became slower. Global demand and instruction count rise
slightly as each row's offsets are prepared ten times, but L2 reuse wins. Output
stores total40MiB mathematically; hardware DRAM counters include cache/writeback
effects and need not equal that number exactly. High occupancy alone did not
distinguish the two versions. SM throughput is not Tensor-Core TFLOPS.

## Remaining work

The next bounded cleanup is33 redundant dense weight/bias-gradient clears,
guarded by full physical-output overwrite proofs and poisoned-buffer tests.
Keep loss/output-bias atomic initialization and sparse fallback guards. Longer
convergence, complete trainer saturation, plugin packaging and T4 validation
remain separate work; no result here establishes any of them.
