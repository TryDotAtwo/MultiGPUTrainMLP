# BN-backward half mirror: original p888, SM86

## Result and scope

Fuse the existing FP32-to-FP16 gradient conversion into local BN backward apply.
For the measured original p888 step, exactly **33 conversion kernels disappear:
463 -> 430 total kernels**, with no allocation growth. The six retained paired
runs give 25.0926 -> 24.4922 ms for the median of run-mean step times: an observed
2.393% latency reduction / 2.451% throughput increase in a matched-sampled-clock
comparison. Clocks were not locked; this is not a controlled or portable speedup.

Scope: RTX 3070 Laptop GPU, SM86; original p888 (72 positions, 72 values), batch
4096, logical widths 2556/218, physical strides 2560/224, 16 residual blocks,
scalar output. FP16 GEMM operands, FP32 BN/master weights/gradients/Adam state,
the model, optimizer, dispatch and fallback policy remain unchanged. Every paired
run reports the same 739,806,720-byte arena. The comparison starts from the
[dense-gradient clear checkpoint](2026-08-31-dense-gradient-overwrite.md), not
from a pre-clear binary. The final targeted correctness/ABI gates passed below.

## Dataflow, precision and lifetime contract

In [local_batch_norm.cu](../../native/cuda/local_batch_norm.cu),
`BackwardApplyCoalesced<HalfMirror>` computes the original FP32 expression once,
stores authoritative FP32 `dx`, and optionally stores `__float2half_rn(value)`.
The arithmetic expression and reduction order are unchanged. Physical padding
is written as positive zero in both outputs; the mirror does not read FP32 `dx`
back from memory. Half rounding, including finite FP32 values overflowing half,
matches the existing conversion contract, not a new finite-half guarantee.

The [local BN API](../../native/cuda/mgt_cuda/local_batch_norm.cuh) adds an optional
backward epilogue and a supplied-statistics apply entry point. Exact `dy == dx`
is supported; partial overlap is rejected. `dx` must be disjoint from normalized
values and feature inputs. The half output must not overlap float inputs,
outputs or statistics. Validation precedes reduction/output writes. Existing
partial reductions, statistics clears and two gradient copies per site remain.

In [mlp_batch_norm_forward.cu](../../native/cuda/mlp_batch_norm_forward.cu), only
the explicit hidden/residual dense-BN sites publish a gradient mirror in the
existing `operand_b` buffer. For p888 these are 32 residual sites plus one hidden
site. Input BN remains FP32-only even when `hd1 == hd2`; eligibility is not
inferred from equal dimensions. Local FP32 and the public nonlocal/NCCL
compilation retain their prior backward route without this mirror.

The cache tag is the exact FP32 source pointer plus active element count.
It is published only after successful enqueue, reused by adjacent dW and dX
GEMMs, and cleared after dX, on failures and on unrelated GEMM paths. The public
[local FP16 step](../../native/cuda/local_mlp_batch_norm.cu) clears stale tags on
entry and every return, including validation failures. This is a host-side
lifetime boundary, not GPU completion: storage reuse still requires same-stream
ordering or an explicit completion dependency, as documented in the
[step API](../../native/cuda/mgt_cuda/local_mlp_batch_norm.cuh).

Preflight checks the mirror's active `rows * max(hd2, output_dim)` half extent,
not logical width or the caller's full advertised capacity. It rejects overlap
with live model/optimizer data, gradients, inputs, outputs, activation tape,
workspace and loss, using overflow-checked byte-range arithmetic. Physical
padding is included. Plan slices and active row products are validated; valid
shrinking-row views/custom offsets are not replaced by a packed-layout guess.
Adjacent ranges and unused capacity suffixes remain legal; unused FC1/residual
scratch for zero residual blocks is not treated as live. Existing unrelated
float/float lifetime reuse is outside these new mirror restrictions.

## Correctness evidence and attempt chronology

All job IDs below refer to retained `.gpu_queue/logs/<id>.log`. A failed build
is not a runtime RED, irrespective of its queue label.

| Job | Verified outcome |
| --- | --- |
| `a169926b0b16` | Test compile failure: range-for initializer-list deduction; no runtime RED. |
| `c38e2dae2124` | Same test compile failure; no GREEN tests or sanitizers executed. |
| `aa81686d1ff7` | After the test compiled, runtime RED: literal RN half mirror mismatch at index 0. |
| `4979511a7599` | Unit GREEN: 3/3 CTests; full memcheck/leak check and initcheck clean; quick racecheck/synccheck clean. |
| `d0a370b6e2f0` | Lifecycle runtime RED: cache escaped the training-step lifetime in all six shapes. |
| `e3464bad3d49` | Unknown Ninja target `bench_single_gpu_train_step`; stopped before compilation/tests. |
| `161c495a673e` | Correct targets: 4/4 CTests; activation-tape full memcheck/leak check and initcheck clean. |
| `73a94990074a` | Production batch-4096 full-step memcheck: zero errors. Subsequent `ctest -N` only listed tests. |
| `33dfd997f355` | Six paired runs, profile capture/export and exact four-step structural comparison completed, exit 0. |
| `658642fd7e20` | Two filtered NCU apply captures completed, eight replay passes each, exit 0. |
| `4c120dbfb0cc` | Unknown target `test_mlp_batch_norm_full_step`; stopped before compilation/tests. |
| `876122ae992a` | Correct `test_mlp_batch_norm_full_backward` target; final regression, sanitizers, identity check, ABI tests and CSV exports passed, exit 0. |

The new [backward-epilogue test](../../native/tests/cuda/test_local_batch_norm_backward_epilogue.cu)
contains 21 full apply fixtures, two integration fixtures and 51 invalid calls;
quick mode uses six apply fixtures, retaining both integration and all invalid
cases. A literal copy of the old GPU apply gives a bit-exact FP32 oracle with
fixed supplied statistics, independent of cross-run atomic reduction order.
A host double formula supplies a numerical check; independent host RN conversion
and literal half bit patterns check ties, signed zero, underflow and overflow.
Coverage includes mirror off/on, exact in-place operation, row/block tails,
odd widths, 218/224 and 2556/2560 strides, rows through 4097, shrinking reuse,
NaN padding, positive-zero output padding, canaries and input-byte immutability.
The two full-wrapper fixtures use a single partial-reduction CTA. Invalid
pointers/shapes/overlaps must reject without writes; work uses a nondefault stream.
The target is registered in [native/CMakeLists.txt](../../native/CMakeLists.txt).

The [activation-tape test](../../native/tests/cuda/test_local_mlp_fp16_activation_tape.cu)
retains its independent CPU oracle and all-gradient poison checks. Six shapes
exercise scalar/vector outputs, zero/two residual blocks, output wider than
hidden width and equal hidden widths, each with rows 4,3,4. It poisons operand B,
seeds stale tags, checks tag retirement on success/error, verifies the last dense
half mirror and preserves inactive suffix guards. The four-test integrated gate
also includes local FP32/FP16 and NCCL-world-one FP32 complete gradient-overwrite
fixtures described in the prior audit; it does not establish multi-rank behavior.

The final `876122ae992a` gate passed 20/20 targeted CUDA CTests, including public
NCCL backward/full-backward and both overwrite fixtures. Updated activation-tape
tests passed 183 rejected alias probes and 18 legal reuse probes across six
shapes, including equal/partial/reverse overlaps, physical padding, adjacent
ranges and unused suffixes. Rejections check no writes; legal probes retain
finite checks. Full activation-tape memcheck/leak check and initcheck were clean;
local and NCCL-world-one overwrite memcheck/leak checks were also clean.
Four CPU/C-ABI CTests and the one Rust FFI test passed. Byte comparison confirmed
the rebuilt benchmark equals frozen `48f54c...`; this is the profiled binary.
These final alias tests are evidenced by this job, not earlier `161c495a673e`.

## Paired timings: all runs retained

Job `33dfd997f355` used frozen binaries, batch 4096, 140 warmup steps and 100
timed steps per run, in baseline/mirror/mirror/baseline/baseline/mirror order.
A separate baseline preconditioning run reported 25.1939 ms and is not included
in the six-run statistic. No run was removed as an outlier. Each value below is
the benchmark's run-mean step time, not a median of individual timed steps.

| Index | Variant | Step ms | Samples/s | Reported loss |
| --- | --- | ---: | ---: | ---: |
| 0 | baseline | 25.2036 | 162517 | 18.1213 |
| 1 | mirror | 24.6031 | 166483 | 17.9515 |
| 2 | mirror | 24.4922 | 167237 | 18.5030 |
| 3 | baseline | 25.0926 | 163235 | 17.9523 |
| 4 | baseline | 25.0720 | 163370 | 19.2626 |
| 5 | mirror | 24.4696 | 167391 | 18.1620 |

Median run means: **25.0926 -> 24.4922 ms**. The throughput comparison is
`25.0926 / 24.4922 - 1 = 2.4514%`; latency reduction is 2.3927%.
All active telemetry samples in all six runs were 780 MHz; clocks were not
locked. Active samples use utilization >=90% and nonzero allocated memory.
Requested 100-ms asynchronous sampling also covers initialization/warmup and
does not align exactly to timed steps; the last-two-second subset is diagnostic
only. Baseline active temperatures span 77-82 C, mirror 79-82 C; median sampled
powers are 66.51 W and 67.325 W. These snapshots are not a controlled thermal or
power experiment. Reported losses are retained, not convergence/equivalence
evidence; correctness rests on the bounded oracle tests above.

## Nsight: exact structure and bounded attribution

Compare the retained dense-clear `elided-final-warmed.sqlite` with new
`mirror-warmed.sqlite`: 100 warmup steps, four captured measured steps, timing
means below from steps 2-4. The analyzer validates every one of the four steps.
The step window runs from `RandomWalkKernel` through terminal AdamW, not CPU wall
time. It canonicalizes only the exact 33 old apply-plus-cast pairs into mirrored
apply and the remaining input apply into non-mirrored apply. All other kernel
names, launch geometry/order and memory-event sequences must match exactly.

| Per-step structural quantity | Baseline | Mirror |
| --- | ---: | ---: |
| Kernel launches | 463 | 430 |
| `FloatToHalf` launches | 33 | 0 |
| `BackwardPartialCoalesced` launches | 34 | 34 |
| Old non-template backward apply | 34 | 0 |
| Backward apply `<true>` / `<false>` | 0 / 0 | 33 / 1 |
| Memsets / bytes | 71 / 156136 | 71 / 156136 |
| Device-to-device copies / bytes | 68 / 78000 | 68 / 78000 |

The 34 atomic partial-reduction launches and 68 statistics copies are unchanged;
this optimization does not remove BN reductions. At the measured shape, the
removed conversion input reads total `33 * 4096 * 224 * 4 = 121110528` bytes,
or **115.5 MiB nominal FP32 reread per step**. This is not measured DRAM traffic:
cache service is unknown and the required half writes still occur in apply.

| Mean over profile steps 2-4, ms | Baseline | Mirror |
| --- | ---: | ---: |
| Step span | 25.904979 | 25.263077 |
| Sum of kernel durations | 23.696345 | 23.094147 |
| Standalone casts | 0.752721 | 0 |
| All backward applies | 1.530400 | 1.686333 |
| Backward partial reductions | 1.170450 | 1.169221 |
| Memsets | 0.131174 | 0.137023 |
| Copies | 0.190984 | 0.191277 |
| Uncovered interval in step span | 1.886476 | 1.840630 |

Apply gets 0.155933 ms slower while separate casts disappear: the combined
apply-plus-cast time falls 2.283121 -> 1.686333 ms, about 0.596788 ms. The whole
kernel sum falls about 0.602198 ms. The 0.641902-ms span difference also includes
changed gaps/memory-event durations; it is not all conversion or kernel savings.
These two profiles provide mechanism evidence, not independent controlled A/B
timing or proof of an ideal bandwidth limit.

NCU job `658642fd7e20` captured the first dense backward apply after 100 warmup
steps (`--launch-skip 3400 --launch-count 1`, filtered to this kernel family).
Both captures use eight replay passes, `--clock-control none --cache-control
none`; NCU warns that uncontrolled clocks/caches may make results inconsistent.
Raw CSV units are microseconds, MHz, registers/thread, percent and decimal MB:

| One replay-profiled apply | Baseline | Mirror |
| --- | ---: | ---: |
| Duration, us | 37.632 | 42.464 |
| Measured clock, MHz | 779.562606 | 779.738916 |
| Registers/thread | 20 | 18 |
| Achieved occupancy, % | 80.532470 | 81.286066 |
| DRAM read / write, MB | 7.348864 / 3.567360 | 7.349376 / 5.273600 |

Apply alone costs 4.832 us more and performs the additional half writes. This
capture excludes the removed cast; it is not a production-step measurement or
a measurement of 115.5 MiB less DRAM traffic. The combined net evidence is the
separate Nsight Systems apply-plus-cast accounting above.

## Reproduction identities and retained artifacts

- Baseline `/tmp/mgt-dense-clear-final`: SHA256
  `149ae33480bb06397ad13d1b1694a49579d61ab3d7e9121efcde5b1e661c775a`.
- Candidate `/tmp/mgt-bn-backward-half`: SHA256
  `48f54cbb9fac4303d838125b55aa289ea49aabda13c4ed64e174e033016aa2da`.
- Both identities remain stable before/after all six paired runs. Raw data and
  helpers are local ignored artifacts under `test_results/bn_backward_half_20260831/`:
  `paired-mirror.jsonl`, `paired-mirror-summary.json`, `mirror-warmed.nsys-rep`,
  `mirror-warmed.sqlite`, `mirror-warmed-profile-summary.json`, `run_paired.py`,
  `summarize_paired.py`, `analyze_nsys.py`, `analyze_profiles.py`.
  NCU retains `baseline-apply-ncu` / `mirror-apply-ncu` `.ncu-rep`, `.log`, `.csv`.
- Paired JSONL SHA256: `0cb3cb436fcba081d2126318469bbe087257f22806db631fd388e8f6b71510b6`.
- Baseline SQLite SHA256: `9eb41f68cdb06d117f249dc09ea4d7ed637fcde0484583040e141b1f52acf3f7`.
- Mirror SQLite SHA256: `9a008c63775ee0eda44028ca2a94e46996bb786b953ddf8a9d5a93a4e889c673`.

This audit makes no new convergence, T4, multi-GPU scaling, plugin or ideal
throughput claim. Evidence supports this bounded local-SM86 optimization and
the listed targeted gates, not an assertion that every repository test ran.
