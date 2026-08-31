# Accepted SM86 stable warp-ballot row builder

Date: 2026-08-31. Follow-up to the accepted
[sparse-gradient tile traversal](2026-08-31-sparse-gradient-tile-major.md).

Original p888, batch4096, RTX3070 Laptop/SM86: median run-mean full-step
latency **28.9696 -> 27.3462ms** (-5.60%); throughput **+5.94%**, approximately
141.4k ->149.8k samples/s. Warmed Nsight builder time **1.841559 ->0.228068ms**
(8.07x). Arena remains739806720 bytes. This is another accepted local
optimization, not a claim that the trainer is optimal or that T4 is verified.

## Change and exactness contract

`BuildGroupedInputRows` moved into private `native/cuda/grouped_input_rows.cuh`,
shared by the local/distributed translation units and an isolated regression
with internal linkage. The unchanged serial implementation was first rebuilt
and frozen, separating extraction from the algorithm experiment.

Before:72 CTAs x72 threads, one thread per position/value bin, serial scan of
4096 rows. After:648 CTAs x256 threads, eight warps per CTA, one warp per bin.
Each warp visits consecutive32-row chunks and uses ballot/popcount to compact
matching rows by increasing lane. Chunks are processed in increasing order,
so output row IDs and subsequent FP32 additions are unchanged.

Only whole unused warps return. Tail lanes participate in every full-mask
ballot but predicate off input reads. `bin`, row/chunk indices and address
products use64-bit arithmetic; counts and stored IDs remain unsigned32-bit.
The ballot is not a memory barrier: writes are disjoint, and the subsequent
consumer runs after kernel completion in the same stream. These requirements
follow the [CUDA12.8.1 warp-vote contract](https://docs.nvidia.com/cuda/archive/12.8.1/cuda-c-programming-guide/index.html#warp-vote-functions).

Scratch layout `counts[bin] + row_ids[bin*rows+i]`, BN scratch lifetime,
capacity fallback, tile-major gradient consumer, auto SM86 dispatch and all
precision/model/optimizer settings are unchanged. No allocations, atomics,
extra kernel launches or new environment switches were added. No convergence
claim is inferred from short losses; this change has an exact integer oracle.

## Identity and gates

Existing Docker `mgt-gpu-queue`, image `mgt-single-gpu-dev:2026-08-28`,
CUDA12.8.93, SM86 Release/Ninja, driver572.70, Nsight Systems2025.6.3 and
Compute2025.1.1. All GPU work used `scripts/gpu_queue_submit.py`. Base commit
`1bc696b952453f581c93e2587617d1eaf9f79875`; no clock/power setting changes.

| Binary | SHA256 |
|---|---|
| Pre-extraction accepted baseline, `/tmp/mgt-row-builder-baseline` | `46a97247b835e9c9d0ac93a9e77e7a52111a2994785e261c61ecfa75a421b35e` |
| Extracted serial baseline, `/tmp/mgt-row-builder-extracted-baseline` | `76800828acba2fadff530d3b88bd2db32ef466d18be48b52c9911e75019f2786` |
| Ballot candidate, `/tmp/mgt-single-sm86/mgt_single_gpu_benchmark` | `d272c5dce706f806dbc336e933ce559caacb86350d71b24aebb13dd6bbd81409` |

TDD runtime RED job`cab91977d809`: the test compiled, then old kernel under
the proposed flat-grid launch left bin6 unfilled (`0xcdcdcdcd`, expected11).
This checks the new launch contract; the old algorithm was correct with its
original launch geometry.

GREEN job`2e3cc1d7ec55`:11/11 targeted CUDA CTests passed, including the new
22-case builder test and existing21-case sparse-gradient test. New coverage:
rows0/1/31/32/33/255/256/257/4095/4096/4097; bins1/9/35/5184; balanced,
all-in-one, skewed, mixed-invalid/all-invalid; partial warp/CTA; changing live
stride in reused allocations. Independent row-major CPU append oracle,
hand-derived literal check, exact counts/IDs, ascending order, every unused
tail, allocation guards and immutable input bytes are verified.

Same job: full builder memcheck0 errors/0 leaked bytes, full initcheck0 errors,
quick racecheck0 hazards/errors/warnings and quick synccheck0 errors. A full
production4096 training step passed memcheck0 errors. Job`25f958e24057`:
4/4 CPU/C-ABI CTests and the actual Rust FFI/RAII training-step test passed.
These are targeted gates, not the entire multi-GPU test suite.

Independent read-only code review found no critical, important or actionable
minor defects in kernel, launch, lifetimes, tests or CMake wiring.

## Unprofiled full-step A/B

Artifacts live in ignored `test_results/grouped_builder_20260831/`; raw GPU
reports are retained locally, not included in the source commit. Runner
`run_paired.py` records all six A-B-B-A-A-B runs, binaries/SHA256, args, selected
environment, stderr, metrics and asynchronous100ms nvidia-smi telemetry.
Both datasets use140 warmup+100 timed steps (240 total, within the original
single-epoch244 full-batch limit), with the same original p888 input files.
Summary aggregation is the median of three run means, not individual-step p50.

| Dataset / variant | Three run means, ms | Median, ms |
|---|---|---:|
| Initial serial |29.0441,28.9439,28.9451|28.9451|
| Initial ballot |27.4236,27.4176,27.4164|27.4176|
| Confirmation serial |29.1556,28.9696,28.9385|28.9696|
| Confirmation ballot |27.4311,27.3462,27.3319|27.3462|

Initial job`6fffe509d77f`, `paired-ballot.jsonl`: startup frequency drift in
serial samples780/975/1020/1050/1215MHz; every run's last2s active samples780MHz.
Retained but not used for the headline matched-sampled-clock comparison.

Confirmation job`7cc6806b446a`, `paired-ballot-confirm.jsonl`: all active and
last2s active samples780MHz. Active temperatures serial77-85C, ballot81-86C.
Active means utilization>=90% and memory.used>0. Sparse asynchronous telemetry
does not prove locked clocks throughout the timed window; these are short
matched-sampled-clock measurements, not confidence intervals or thermal control.
No runs were discarded. Exact computed delta:1.6234ms, latency-5.6038%,
throughput+5.9365%. Final losses serial17.6206/18.1656/17.9981 versus
ballot18.4616/17.9456/17.8900 are diagnostic only, not a convergence test.

## Nsight attribution

Systems captured100 warmup+4 final steps and analyzed steps102-104, using
RandomWalk ->last AdamW GPU boundaries. Reports: `serial-warmed.nsys-rep` and
`ballot-warmed.nsys-rep`, matching SQLite exports and
`ballot-profile-summary.json`. Do not substitute profiler wall time for A/B.

| Metric, ms unless noted | Serial | Ballot |
|---|---:|---:|
| GPU step span |29.846305|28.065928|
| Sum of kernel durations |27.376592|25.677997|
| Builder |1.841559|0.228068|
| Sparse-gradient consumer |3.743196|3.689414|
| Forward span |8.686859|8.650021|
| Kernel count |463|463|
| Cast / main GEMM / split-K reduce counts |33 /99 /33|33 /99 /33|

All four final kernel sequences and launch geometries match after changing
only builder72x72 ->648x256. The saved builder time explains most of the
full-step gain; small differences in unchanged kernels/gaps are not attributed
to new arithmetic. Builder now accounts for0.81% of the profiled GPU step span.

Baseline profile job`44cb87112308`. Candidate Systems capture/export in
job`d408ba305839` succeeded, but that job ended nonzero because an accidental
trailing post-processing command referenced a nonexistent path. No report was
overwritten: successful job`d1e5a4f570c0` independently read/validated the SQLite
exports, compared all final steps, and collected candidate Compute data.

Compute: one builder after100 warmup steps, kernel replay/default cache flush,
15 passes each, `--clock-control none`; reports `serial.ncu-rep`,
`ballot.ncu-rep` and raw CSV. Units below are converted from each CSV unit row.

| NCU metric | Serial | Ballot |
|---|---:|---:|
| Kernel duration, ms |1.840928|0.230784|
| GPC clock, MHz |779.968309|779.739208|
| SM throughput, % peak sustained |6.168682|49.954911|
| Achieved occupancy, % |11.249364|90.414901|
| Eligible warps/scheduler/cycle |0.047848|1.070822|
| Long-scoreboard stall ratio, cycles/instruction |25.835271|14.424506|
| Executed warp instructions |10,401,912|14,415,966|
| Registers/thread |24|29|
| L2 sector hit rate, % |98.661958|98.956880|
| DRAM throughput, GB/s |0.619583|4.805324|

Instruction count rises38.6%, yet latency falls: the improvement comes from
parallelizing the serial scan and exposing enough warps to hide latency, not
from reducing arithmetic count or introducing approximate math. SM throughput
here is not Tensor-Core utilization, TFLOPS or full-trainer efficiency.

## Next measured target / training-plugin lessons

Stop tuning this builder for now. Current profile has input gather2.4631ms,
34 bias-gradient `ColumnSum` launches totaling2.4169ms and sparse gradient
3.6894ms. A useful next bounded investigation is `ColumnSum`: each warp reads
one column across strided rows. First profile it, then evaluate coalesced
access while explicitly preserving or validating its reduction order. Do not
remove pre-BN bias updates merely because an idealized derivative sums to zero.

For the future GPU-training plugin: preserve discrete grouping order before
changing gradient reduction; full-mask tail participation is mandatory;
high occupancy is evidence only when full-step timing improves; instruction
count alone can mislead; freeze binaries and compare one change with warmed
profiles plus exact correctness gates. Plugin packaging and T4 remain later
stages, not delivered by this checkpoint.
