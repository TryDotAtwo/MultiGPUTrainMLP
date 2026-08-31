# Local BN input-bias fusion

Date: 2026-08-31. Original p888, batch4096, RTX3070 Laptop/SM86.
This follows [backward mask fusion](2026-08-31-bn-backward-mask-fusion.md) and
the [rejected sparse-layout experiments](2026-08-31-sparse-gradient-warp-bins.md).

## Change and invariants

Thirty-three local dense Bias passes are replaced by bias reads in the existing
BN partial-statistics and apply kernels. Each consumer computes the same rounded
FP32 x+bias value using `__fadd_rn`, matching the old Bias global-store boundary.
BN then computes statistics/normalization in its original order. This is not
folding a bias into the post-normalization affine beta, dropping bias parameters,
or changing training BN semantics.

The optional `LocalBatchNormForwardEpilogue::input_bias` contains logical-feature
FP32 values. Null retains the existing path. The input-gather BN site remains
unbiased here: its table bias is already included by the unchanged input backend.
The hidden site and32 residual FC sites receive their original bias slices.
The nonlocal/synchronized path retains the original materialized Bias with
the same `s.hd2`/`lh2` launch arguments.

Partial/apply specialize on whether input_bias is present. Each bias load is
inside the logical-column guard; padding does not read bias. Normalized padding
stays+0, followed by the original optional residual/ReLU/half epilogue. No extra
matrix is written. FP32 normalized values, authoritative activations, BN state,
master weights, sparse gradients and Adam remain FP32; half GEMM mirrors and
GEMM algorithms are unchanged. No new arena buffer, collective or hot-path
allocation is introduced.

Both Apply and Forward reject bias overlap with writable outputs before work is
enqueued. Forward additionally rejects overlap with means/inverse standard
deviations, running state and statistics workspace. Disjoint read-only aliases
remain allowed. In-place x==y remains supported with a separate bias slice.

The removed matrix traffic is33*(4096*218 +4096*224)*4 =238,977,024 bytes of
logical activation loads/stores per step, excluding feature-vector loads,
caches and transaction granularity. This is an analytical pass count, not
measured DRAM traffic or a prediction that all old Bias time is removable.

Source: [local BN](../../native/cuda/local_batch_norm.cu),
[contract](../../native/cuda/mgt_cuda/local_batch_norm.cuh),
[trainer routing](../../native/cuda/mlp_batch_norm_forward.cu).

## Numerical and memory verification

The expanded [epilogue test](../../native/tests/cuda/test_local_batch_norm_epilogue.cu)
has an independent Bias kernel that first materializes FP32 biased inputs. The
reference then executes the previous affine/activation composition; the fused
consumer never supplies its own expected values.

- [Runtime RED1c6a834854db](../../.gpu_queue/logs/1c6a834854db.log): the new field
  existed but the old implementation ignored it. Compilation succeeded; the
  first rounding witness failed at output2. Frozen RED SHA256:
  0f7ed7d99fc6226ee2e84d44a86935ebdaae19ca9cf8d5ab32b5fc23d7a7839e.
- [Unit GREEN66d090c299a5](../../.gpu_queue/logs/66d090c299a5.log):148 apply cases,
  16 complete-forward cases and19 new bias-alias rejection cases pass. Full
  memcheck/leak-check and initcheck report zero errors; quick racecheck has
  zero hazards and synccheck zero errors. The previous no-bias invalid cases
  also remain in the full suite.
- [Integration977f24f48b1a](../../.gpu_queue/logs/977f24f48b1a.log):24 CUDA
  regressions, four CPU/C-ABI tests and one Rust FFI test pass. Full production
  batch4096 memcheck, activation-tape memcheck/leak-check/initcheck, local and
  nonlocal gradient-overwrite memcheck/leak-check all report zero errors.
  Full epilogue memcheck/initcheck now uses tightly allocated logical bias
  vectors, detecting even discarded padding overreads.
- [Final5f27cc04df9e](../../.gpu_queue/logs/5f27cc04df9e.log) repeats those gates
  after restoring the exact original nonlocal Bias launch arguments and adds
  quick racecheck/synccheck on the final tightly allocated test. All pass.
  The rebuilt local benchmark remains byte-identical to the frozen candidate.

Coverage includes all supported plain/ReLU/residual/half combinations, in-place
and out-of-place execution, widths around256, logical2556/physical2560,
logical218/physical224, row-tile boundaries255/256/257 and rows4095/4096. Exact
dyadic full-forward fixtures compare output, normalized tape, half mirrors,
means, inverse standard deviations, running state and statistics workspace.
Literal apply cases protect rounding of1+2^-24, -1-2^-24, 2^24+1, subnormals and
signed zero; GPU-composition cases also cover NaN/infinity behavior. Immutable
inputs, bias slices and guards are checked. These gates do not establish
arbitrary-run bit-identical training or long-run convergence; cross-CTA atomic
BN reduction ordering is unchanged.

Final epilogue test source SHA256 at the final gates:
b1e8e54e19cd0b91df4de9dc28a9c27000658c9db68ed1bb6c491963501301ab.
The earlier unit-only frozen executable6fd4a9df... precedes the tighter bias
allocations; it is not the final test-binary identity.

## Frozen production identities and first A/B

| Variant | Executable | SHA256 |
|---|---|---|
| Baseline | /tmp/mgt-bn-backward-mask | ca5b2a2300e7c39e0466d52d57627ecc251a715a6f840edf3437ffa6d1cd05eb |
| Input-bias fusion | /tmp/mgt-bn-input-bias | b16ac6c32d93fabbf6d1f028e059e14d1acc6fc9d9cce901c50116f119b946bb |

Same production group/target hashes as preceding audits,739806720-byte arena,
original aligned model and precision. Each run verifies executable hashes before
and after execution. GPU work is serialized through the existing Docker queue;
no clock, power, driver or environment change is part of this patch.

[First measured job6d5688717ef2](../../.gpu_queue/logs/6d5688717ef2.log): four
separate baseline preconditioning processes, then six ABBAAB runs, each140
warmup/100 measured steps. All240 steps fit inside the244-full-batch epoch.

| Variant | Three run means, ms | Median of run means, ms |
|---|---|---:|
| Baseline | 23.2734,23.2630,23.2697 | 23.2697 |
| Bias fusion | 22.5673,22.5600,22.5576 | 22.5600 |

Observed difference0.7097ms, -3.0499% latency /+3.1458% throughput. Baseline
active samples include780-810MHz, candidate780MHz; clocks are not locked.
Temperatures overlap76-81C/77-81C. This first series is descriptive, not a
perfectly frequency-controlled comparison.

[Raw first JSONL](../../test_results/bn_input_bias_20260831/paired-bias.jsonl)
SHA256116214a2fe0ecdebdafa79d4f5691d80ec9cfc588da479231eca0e666f99d7f0.
The first attempt0ca1101805b0 stopped in a CPU analyzer regression before GPU
benchmarking: a mutation intended for an unrelated kernel also changed checked
BN geometry. That test mutation was narrowed to RandomWalk. The benchmark path
capitalization was also corrected before execution. The recheck passes all26
CPU analyzer/summarizer tests; no GPU timing attempt was discarded.

## Strict timeline and confirmation

The initial100-warmup/four-step trace passes exact structural comparison:
396 ->363 kernels, exactly33 Bias launches removed. There are still34 forward
partials and34 forward applies, with one no-bias input site and33 biased dense
sites. The precise ReLU/residual/half specialization counts and input/narrow
site geometries are checked. Every other kernel name, geometry and order is
unchanged, as are71 memsets/156136 bytes and68 DtoD copies/78000 bytes per step.

CPU tests exercise the actual historical baseline SQLite through the complete
comparator, plus in-memory mutations of names, geometry, bias-site selection,
residual/half policy, counts, bytes, copy kind, stream and event order. Synthetic
SQLite transformations are tests only, never measured GPU results.

[Initial candidate SQLite](../../test_results/bn_input_bias_20260831/bias-warmed.sqlite)
SHA256fe983afd61a6fdc39439e51ecb2e76c897ad81f58334a231801b9e1146f19d0a.
Mean measured steps2-4:21.180824ms kernel sum,23.077221ms span,3.733919ms sparse,
0.564886ms forward partials and1.804674ms forward applies. That comparator uses
the historical baseline trace, so its cross-session whole-span difference is
not the optimization's speedup.

[Confirmation job53fb6c0e0e28](../../.gpu_queue/logs/53fb6c0e0e28.log) repeats
four baseline preconditions and the same six-run ABBAAB protocol:

| Variant | Three run means, ms | Median of run means, ms |
|---|---|---:|
| Baseline | 23.1647,23.1802,23.1719 | 23.1719 |
| Bias fusion | 22.4354,22.4296,22.4334 | 22.4334 |

Confirmed difference0.7385ms, -3.18705% latency /+3.29197% throughput,
approximately182.6k samples/s. All active samples in both variants are780MHz;
clocks are not locked. Temperatures81-84C and82-85C overlap. The first and second
series remain separate; no cross-series baseline drift is credited to fusion.
[Confirmation JSONL](../../test_results/bn_input_bias_20260831/paired-bias-confirm.jsonl)
SHA25694822e373af71d288298a4a260a51d7a83dfa789fcb0b7262481638f928da9e5.

The same job captures a fresh baseline and candidate after100 warmup steps;
the strict four-step comparator passes again. Mean measured steps2-4:

| Metric | Fresh baseline | Bias fusion |
|---|---:|---:|
| All kernels, ms | 21.777778 | 21.088736 |
| Step span, ms | 23.754994 | 22.987331 |
| Sparse consumer, ms | 3.682601 | 3.682100 |
| Bias kernels, ms | 0.877594 | 0 |
| Forward BN partials, ms | 0.381088 | 0.563320 |
| Forward BN applies, ms | 1.802425 | 1.804030 |
| Uncovered interval, ms | 1.652548 | 1.577619 |

Bias removal saves0.877594ms but partials gain0.182232ms and applies0.001604ms:
the changed families save about0.693758ms, consistent with the measured full
kernel-sum reduction0.689043ms. The uncovered interval is not automatically
removable CPU overhead. Profiled spans are not the unprofiled step times above.

- [Fresh baseline SQLite](../../test_results/bn_input_bias_20260831/baseline-fresh.sqlite),
  SHA256931500c3fed65d578f7e4be1dc56295995d5abed2b03592b4bdac0e764e21c58.
- [Fresh candidate SQLite](../../test_results/bn_input_bias_20260831/bias-fresh.sqlite),
  SHA256564859e4443e0a61de5952e8a96666074d74cbad77b702d791d6ea5affdf4649.

Decision: promote local Bias-to-BN fusion. No subagents or external reviewers
were used for this checkpoint; it was checked by code review, runtime oracles,
sanitizers, paired full-step runs and exact timeline comparison.

Queue logs and test_results are local ignored evidence. This is a local BN
optimization, not T4 validation, multi-GPU scaling, ideal saturation, CUDA Graph
acceptance, long-run convergence or a training-plugin release.
