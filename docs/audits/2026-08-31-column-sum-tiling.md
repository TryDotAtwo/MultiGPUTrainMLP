# Exact-tree column-sum tiling: original p888, SM86

Date: 2026-08-31. Follow-up to the local single-GPU training hot-path work.
Scope: improve bias-gradient column reductions without changing their FP32
accumulation order, model, optimizer, tensor layout or collective placement.

Observed result: the latest complete six-run A/B series has median run-mean
step time27.4168 ->26.0016ms (-5.16% latency, +5.44% throughput), with unchanged
arena size. **All three series have clock drift; this is an observational
result, not a controlled-clock speedup claim.** Separate Nsight attribution
shows column-sum kernel time2.419273 ->0.972139ms per measured training step.

## Contract and implementation

The measured workload is original p888 on one RTX3070 Laptop GPU, SM86,
batch4096, logical hidden widths2556/218, physical strides2560/224,
16 residual blocks and scalar output. The new private implementation is
`native/cuda/column_sum.cuh`; production call sites are in
`native/cuda/mlp_batch_norm_forward.cu`.

- A fixed256-thread CTA owns several adjacent columns. Width4 is used for
  the33 narrow reductions per step; width8 for the single wide reduction.
  Narrow grids become56 rather than224 CTAs; the wide grid320 rather
  than2560. The number of reduction launches remains34 per step.
- Each column retains256 **virtual** row lanes. Virtual lane `t` accumulates
  rows`t,t+256,...` in order, then participates in the same binary tree with
  offsets128,64,...,1. This is not a64- or32-lane replacement reduction.
- For tile width`C`, each physical thread owns`C` virtual sums. The upper
  tree levels are folded in registers using the original pairs; the remaining
  levels use the same1KiB shared-memory allocation. Every addition uses
  `__fadd_rn`; no sum is reassociated across the original tree edges.
- Padded columns produce positive zero without reading their NaN-poisoned
  inputs. Inactive lanes still initialize shared memory and reach all barriers;
  partial final column tiles cannot write beyond the physical stride.
- No new allocation, scratch arena, host synchronization or atomic operation
  is introduced. The tested trainer arena remains739,806,720 bytes.

The reduction source is shared by local/common training code. The measured
and sanitizer-checked trainer here is the local no-collective path (the build
also includes NCCL); no
multi-GPU correctness or performance claim follows from this audit.

## Correctness and review gates

| Gate | Recorded evidence |
|---|---|
| Runtime RED | Job`a28c4d8b3d84`: extracted old kernel builds, then fails column3, got`0xcdcdcdcd`, expected132 |
| Initial exact-tree GREEN | Job`83a8624eee34`:35 fixtures /175 tile-width variants passed |
| Targeted CUDA suite | Job`fb86115e7923`:12/12 CTest tests passed |
| Full column-sum memcheck | Same job:175 variants,0 errors; leak check enabled |
| Quick racecheck | Same job:35 variants,0 hazards,0 errors,0 warnings |
| Quick synccheck | Same job:35 variants,0 errors |
| Full initcheck | Same job:175 variants,0 errors |
| Production4096 memcheck | Same job:one complete trainer step, status`ok`,0 errors |
| Contract/CPU/C ABI | Job`cc2fcf5cf239`:4/4 CTest tests passed |
| Rust owner integration | Same job:1 passed,0 failed |
| Independent code review | No remaining findings reported |

`native/tests/cuda/test_column_sum_tiled.cu` launches the actual private kernel,
not a duplicate candidate. Its old-grid GPU oracle explicitly uses RN FP32
addition. Its independent CPU oracle assigns rows to256 residue classes,
forces an FP32 store after every addition, and applies the specified tree.
A hand-derived257-row cancellation witness distinguishes the required result
from serial/reduced-lane accumulation. The first RED deliberately launches
width4 with stride9/grid3: the old one-column-per-block implementation leaves
column3 unwritten, causing a value mismatch without an illegal memory access.

Widths1/2/4/8/16 cover row counts0,1,31,32,33,255,256,257,4095,4096,4097;
strides1,7,9,31,32,33,224,257,2560; and production logical tails218/2556.
Checks include positive-zero padding, NaN-poisoned inactive capacity,
shrinking/growing row counts with reused allocations, unchanged input bytes,
and guards on both sides of every allocation. The largest input is40MiB.
Existing FP32/FP16 full-step, grouped-row, activation-tape, input-half,
BN/epilogue, activation and trainer-lifecycle tests remain in the targeted
suite. These gates do not establish a long-run Adam trajectory or convergence.

## Frozen binaries and measurement record

Queue:`mgt-gpu-queue`; measured GPU:`NVIDIA GeForce RTX 3070 Laptop GPU`,
SM86. Runs use the existing single-GPU development build and GPU queue.
No clock/power settings were changed; NCU explicitly used`--clock-control none`.
The source data is`native/production_inputs/p888.json` with
`native/tests/fixtures/p888-target.bin`.

| Variant | Frozen executable | SHA256 |
|---|---|---|
| Extracted baseline | `/tmp/mgt-column-sum-extracted-baseline` | `a879769514fb382c823d47a8093c5613f131195752fa9ab36801de6780573b16` |
| Width4/8 candidate | `/tmp/mgt-column-sum-c4c8` | `2e762d39635128cfbc9e690fe338b33fc74ffa714a4e3fc88eb59a1adcfbc741` |

The paired runner invoked the candidate at
`/tmp/mgt-single-sm86/mgt_single_gpu_benchmark`; every recorded candidate hash
matches the frozen width4/8 executable. Do not reinterpret later contents of
that mutable build path as the measured candidate.

The initial GREEN/microbenchmark job also records the earlier width4 trainer
SHA256`1b666d6387ea1fcdfe084905cf56d28dbb83ebb5bab54a8a06dbeca6b509bfb1`,
frozen separately at`/tmp/mgt-column-sum-c4` in job`b16b3cbe5e78`.
That preliminary trainer is not the paired candidate. The final12-test suite,
sanitizers, production memcheck and ABI/Rust gates record the width4/8 hash
`2e762d...` above; the isolated test itself exercises all five widths.

Every paired run uses140 warmup +100 timed steps, batch4096, and reports the
mean step time over that run. Order is A-B-B-A-A-B, three runs per variant;
GPU telemetry is sampled approximately every100ms. All18 runs are retained,
with each six-run dataset summarized independently:

| Dataset / queue job | Baseline run means, ms | Candidate run means, ms | Medians, baseline -> candidate |
|---|---|---|---|
| `paired-c4-c8.jsonl` /`dfdb24812989` |26.2244,27.4166,27.3245|25.9761,25.9803,25.8976|27.3245 ->25.9761|
| `paired-c4-c8-confirm.jsonl` /`98f6b7f5bf08` |27.4284,27.4116,27.4298|25.9770,25.9919,25.9836|27.4284 ->25.9836|
| `paired-c4-c8-final.jsonl` /`cd0d84254495` |27.3975,27.4168,27.4198|26.0016,26.0166,25.8885|27.4168 ->26.0016|

Their descriptive throughput increases are5.1909%,5.5604%,5.4427%, respectively.
These are medians of run means, not per-step medians, confidence intervals or
independent controlled replications. No slow/cold run is dropped or pooled
away. In particular, the first baseline run26.2244ms remains in the first set.

All three summary files explicitly report
`all_runs_same_constant_sampled_clock=false`. In the last series, active
baseline samples span780-1260MHz and74-82C; candidate samples show780MHz and
77-82C. Last-two-second active samples are780MHz for all six final-series
runs, but this diagnostic tail is **not aligned to or sufficient to cover
the full timing window**. Initialization/warmup and transient behavior remain
in the complete telemetry. The final1.4152ms median difference is useful
observational evidence when combined with kernel attribution, not proof of
a thermally controlled5.44% gain.

## Nsight Systems attribution

Baseline job`a93f5c6dc126`; candidate job`2df6da9073fd`. Both profiles run
100 warmups followed by four measured steps. The summary's legacy key
`mean_steps_2_to_4` means the final three post-warmup windows, corresponding to
training steps102-104. The first post-warmup window101 is excluded from
averages; the underlying trace also retains the100 warmup steps.
Window:RandomWalk start through final AdamW end.

| Mean per measured step | Extracted baseline | Width4/8 |
|---|---:|---:|
| Whole-step GPU span, ms |31.314674|26.711676|
| Whole-step kernel sum, ms |25.808725|24.313569|
| Total kernel count |463|463|
| All column sums, count |34|34|
| All column sums, ms |2.419273|0.972139|
| Narrow width4 column sums, count / ms |Included above|33 /0.829934|
| Wide width8 column sum, count / ms |Included above|1 /0.142205|
| Dense GEMM mains, count / ms |99 /5.889068|99 /5.884888|
| FP32-to-FP16 casts, count / ms |33 /0.757056|33 /0.757162|
| GPU-span interval without recorded GPU activity, ms |4.868538|1.978919|

Column sums save1.447133ms; the total kernel-sum difference is1.495156ms.
This directly supports the reduction-kernel attribution. The baseline trace
also contains substantially more gaps without GPU activity. Therefore the
larger31.314674 ->26.711676ms span reduction must **not** be attributed wholly
to this optimization or advertised as its training speedup. Uncovered time
is not automatically removable CPU launch overhead.

The structural comparison passes for captured steps101-104: after
canonicalizing the34 column-sum replacements, remaining kernel sequence and
geometry match. This is separate from the numerical-equivalence tests.
Both traces retain68 device copies/78,000 bytes and104 memsets/8,902,888 bytes
per step, with no boundary-crossing memory events. Their memory-operation
durations need not match; no copy/memset removal is claimed.

## NCU and tile-width evidence

NCU2025.1.1 profiles one narrow and one wide reduction after100 warmups.
Baseline job`4c6baa0caf47` uses the **pre-extraction** executable
`/tmp/mgt-column-sum-baseline`; candidate job`2df6da9073fd` uses the measured
width4/8 implementation. These CSVs are a separate per-kernel record, not
the paired extracted-baseline executable record above. Their recorded GPC
frequencies are approximately778-780MHz; clock control remained disabled.

| CSV / kernel | Duration, us | DRAM bandwidth, GB/s | Registers/thread | Executed warp instructions |
|---|---:|---:|---:|---:|
| `serial-small.csv`,224-column grid |53.792|86.396193|26|412,826|
| `tiled-small.csv`,width4 /56-column-tile grid |24.672|188.850843|48|339,232|
| `serial-large.csv`,2560-column grid |676.512|64.010785|26|4,833,788|
| `tiled-large.csv`,width8 /320-column-tile grid |146.400|295.492022|45|4,934,728|

All four use256-thread CTAs and1024 bytes static shared memory. The tiled
wide case has slightly **more**, not fewer, executed instructions; the
bandwidth/time improvement is consistent with the changed memory access
pattern. Nsight reports localMemoryPerThread=0 for baseline and both tiled
variants. Its nonzero aggregate localMemoryTotal export field is not treated
as measured spill traffic.

The quick CUDA-event sweep in job`83a8624eee34` uses100 warmup +300 timed
launches per width. Narrow224-column timings for widths1/2/4/8/16 are
26.217813/17.582080/9.932800/12.721493/16.042666us. Wide2560-column timings are
403.278503/232.366079/170.502828/131.324590/128.812370us. Width4 wins the narrow
sample; width16 is slightly faster than width8 in this isolated wide sample.
Thus this audit does **not** claim width8 is the absolute microbenchmark
winner or globally optimal. Width4/8 is the configuration actually subjected
to the recorded full-step A/B and resource/attribution gates. Microbench,
NCU replay, Nsight kernel sums and unprofiled training timings are distinct
measurements and must not be interchanged.

## Artifacts and limits

Raw evidence remains local and ignored under`test_results/column_sum_20260831/`:
all three paired JSONL files and corresponding`*-summary.json` files;
`{serial,tiled}-warmed.nsys-rep` and SQLite;
`tiled-profile-summary.json`;`{serial,tiled}-{small,large}.csv/.ncu-rep`;
and the analysis/runner scripts. Queue logs remain under`.gpu_queue/logs/`
using the job IDs above. This audit is the compact tracked evidence record;
raw reports were neither deleted nor published.

Remaining limits: no fixed-clock study, full convergence, ideal-trainer
claim, T4 result or multi-GPU validation. Exact isolated column sums do not
make the entire trainer deterministic: other BN/loss atomic reductions can
still vary. [InputHalf feature tiling](2026-08-31-input-gather-feature-tiling.md)
is an accepted separate follow-up and is excluded from every result claimed here.
