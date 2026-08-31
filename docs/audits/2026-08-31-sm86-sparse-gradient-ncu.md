# SM86 sparse-gradient bottleneck: measured, not yet optimized

Date: 2026-08-31. Scope: read-only profiling and next-hypothesis selection.
No CUDA implementation or default-dispatch change was made in this audit.

Subsequent accepted experiment:
[tile-major sparse gradient](2026-08-31-sparse-gradient-tile-major.md),
34.2866 ->28.9943ms unprofiled full step. The sections below retain the original
baseline investigation and distinguish then-unvalidated hypotheses.

## Current project sequence

The user's current sequence supersedes earlier T4-first readiness priorities:

1. Optimize the original single-GPU trainer on RTX 3070 Laptop using Nsight.
2. Distill validated, transferable GPU-training practices into a dedicated plugin.
3. Validate and tune on one T4, then scale to two T4s on Kaggle.

The 3070 stage needs measured full-step gains, explanations for the dominant
resource limits, and training-quality evidence. Nsight alone does not establish
convergence. Hardware-specific dispatch policies must remain separate from
general training principles. No T4/Kaggle work was performed here.

## Reproduction

- GPU: NVIDIA GeForce RTX 3070 Laptop, SM86, 40 SMs, 4 MiB L2.
- Existing container: `mgt-gpu-queue`; all GPU work used `gpu_queue_submit.py`.
- CUDA: 12.8.93; Nsight Compute: `/opt/nvidia/nsight-compute/2025.1.1/ncu`.
- Binary: `/tmp/mgt-single-sm86/mgt_single_gpu_benchmark`.
- SHA256: `3422df2cfed1cb9cab1b5922b3162a95d615e03a8df2179fb640ca8eb9e63747`.
- Existing dirty source includes the accepted activation-tape and BN-epilogue
  work; HEAD alone is not a reproduction identifier.
- Original p888: batch4096, 72 categorical positions/72 values, hidden physical
  width2560, compact state stride80 bytes. FP32 sparse-gradient accumulation.
- Benchmark args: `4096 100 1 /work/native/production_inputs/p888.json
  /work/native/tests/fixtures/p888-target.bin`.
- Both runs filtered `BuildGroupedInputRows` and `SparseInputGradGroupedRows`,
  skipped200 matching launches and collected2: step101 after100 warmups.
- Both used `--clock-control none`; no power/clock settings were modified.
  Measured GPC clock was approximately780 MHz and DRAM clock5.464 GHz.

Queue jobs, both exit0:

| Job | Replay/cache | Sections |
|---|---|---|
| `b87dfacd2aba` | Kernel replay, default cache flush, 15 passes/kernel | SpeedOfLight, LaunchStats, Occupancy, MemoryWorkloadAnalysis, SchedulerStats, WarpStateStats, InstructionStats |
| `7b5114ac71bf` | Application replay, `--cache-control none`, 7 application passes | SpeedOfLight, MemoryWorkloadAnalysis, LaunchStats |

Raw artifacts: `test_results/sparse_gradient_20260831/baseline.ncu-rep`,
`baseline-raw.csv`, `natural-cache.ncu-rep`, `natural-cache-raw.csv`.
Exact commands and replay logs are in `.gpu_queue/logs/<job>.log`.
CSV raw pages are wide: header, unit row, then kernel rows.

## Results

| Metric | Builder, flushed | Builder, natural | Gradient, flushed | Gradient, natural |
|---|---:|---:|---:|---:|
| Kernel duration, ms | 1.839008 | 1.842976 | 8.945600 | 8.944096 |
| DRAM throughput, GB/s | 0.614940 | 0.247530 | 337.979825 | 338.120535 |
| DRAM throughput, NCU % | 0.18 approx. | 0.070780 | 96.641336 | 96.681011 |
| SM throughput, NCU % | 6.167823 | 5.719325 | 34.732682 | 34.033461 |
| L2 sector hit rate, % | 97.845571 | 98.252556 | 3.705467 | 3.738971 |
| Registers/thread | 24 | 24 | 40 | 40 |

Additional flushed-run counters:

- Builder: occupancy11.249201%, eligible warps/scheduler0.047846,
  issue-active4.782096%, long-scoreboard25.839065 cycles/issued instruction,
  10,401,912 executed warp-level instructions. Launch72 blocks x72 threads.
- Gradient: occupancy98.340113%, eligible warps/scheduler0.291020,
  issue-active19.918331%, long-scoreboard54.597760 cycles/issued instruction,
  221,393,760 executed warp-level instructions. Launch51,840 blocks x256 threads.

Interpretation: the builder underfills the GPU and has a long dependent scan.
The gradient saturates DRAM in both collection modes; more occupancy alone is
not the remedy. These are per-kernel counters, not whole-step GPU utilization.
The natural-cache check corroborates that cache flushing is not the principal
cause of the gradient's low L2 hit rate in this workload.

These are diagnostic profiles, not a new unprofiled performance benchmark.
NCU's application `step_ms` includes collection overhead and must not be used as
trainer throughput. Clocks were not locked, counters can come from different
passes, and application-replay losses vary; this is not a numerical-replay or
convergence test. The latest accepted unprofiled baseline remains34.2634ms
median run-mean in [the BN epilogue audit](2026-08-31-bn-epilogue-fusion.md).

## Next hypothesis, not a validated optimization

Source: `native/cuda/mlp_batch_norm_forward.cu`, kernels at lines131/140 and
grouped-row launch at535. Independent architecture review agrees with the
measured distinction between the two bottlenecks.

First experiment: change only the gradient CTA coordinate mapping so bins are
adjacent for a fixed hidden-feature tile. Keep the ascending-row summation loop
unchanged. First compare mapping alone at256 threads, then independently test
128/64-feature tiles if useful. The logical dZ footprint per tile is4/2/1 MiB
respectively, versus40 MiB for the entire dZ matrix. Current logical dZ reads are
`72 * 4096 * 2560 * 4 = 3,019,898,880` bytes, before output/index traffic; this
is a source-level count, not an NCU DRAM-byte counter.

This is an empirical cache-locality hypothesis, not a scheduling guarantee:
CUDA permits blocks to execute in any order. Row-index reads and gradient
stores also compete for cache. Accept only after reduced measured DRAM traffic
and paired unprofiled full-step improvement; otherwise reject the hypothesis.

Separate later experiment: one warp/bin stable ballot compaction for the row
builder, preserving ascending row IDs and scratch ownership. Do not combine it
with the CTA-mapping change in the first A/B. Direct kernel oracles must cover
warp tails, empty/colliding bins, hidden-width tails and exact gradient bits;
then sanitizer and existing full-step tests. Precision-changing algorithms
require additional convergence/time-to-quality gates instead of bitwise parity.

## Transferable lessons for the future training plugin

- Systems first identifies critical-path shares; Compute then distinguishes
  underfilled execution from saturated memory or arithmetic resources.
- High occupancy or busy DRAM does not prove the algorithm is efficient:
  unnecessary rereads may saturate memory while wasting full-step time.
- Check the profiler's cache/replay policy before drawing cache-locality
  conclusions. Keep diagnostic timing separate from unprofiled throughput.
- Change one mapping/layout/precision variable at a time; preserve an oracle,
  workload identity, binary identity and rejected-experiment evidence.
- The proposed CTA mapping and builder are not yet validated plugin recipes.

Primary references: NVIDIA's [NCU2025.1 profiling guide](https://docs.nvidia.com/nsight-compute/2025.1/ProfilingGuide/index.html#cache-control)
describes default cache flushing and application replay; the
[CUDA12.8.1 programming model](https://docs.nvidia.com/cuda/archive/12.8.1/cuda-c-programming-guide/index.html#thread-hierarchy)
requires block independence and permits arbitrary scheduling order.
