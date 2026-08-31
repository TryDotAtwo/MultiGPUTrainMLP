# Dense-gradient overwrite: original p888, SM86

## Decision and scope

Accept the dense-gradient clear cleanup against checkpoint `0125509` as a
structural reduction in redundant work. For the measured original p888 step,
33 memset calls disappear and the output-head clear is narrowed to its atomic
bias accumulator. The exact observed budget changes from **104 calls /
8,902,888 bytes** to **71 calls / 156,136 bytes** per step. Kernel launches and
copy counts/bytes remain unchanged in all four captured steps.

This is **not a controlled full-step speedup claim**. The final six-run dataset
has a descriptive median of run means of 25.3633 -> 25.1633 ms, but sparse clock
telemetry and slow runs in both variants do not support attributing that entire
difference to this change.

Scope: one RTX 3070 Laptop GPU, SM86, original p888 (72 positions, 72 values),
batch 4096, logical hidden widths 2556/218, physical strides 2560/224, 16 residual
blocks, scalar output, and the existing local FP16 training path with FP32
master weights/gradients/Adam state. Arena usage remains 739,806,720 bytes.
The shared CUDA source is also compiled for the public nonlocal NCCL path;
world-size-one FP32 tests cover that compilation. This audit does not establish
multi-rank behavior, T4 performance, convergence, or any plugin result.
The separate BN-backward half-mirror work under development is excluded.

## What changed, and why the old contents are irrelevant

In [mlp_batch_norm_forward.cu](../../native/cuda/mlp_batch_norm_forward.cu):

- Remove the complete dW+bias clear preceding each residual FC1/FC2 gradient
  and the hidden-layer gradient: 32 residual clears plus one hidden clear.
- Narrow the output-head weight-gradient clear to `output_dim` bias elements.
  Retain the loss clear and the output-bias clear because those accumulate
  atomically. The scalar output dW is a beta-zero GEMV; vector output dW is a
  beta-zero GEMM.
- Dense GEMM dW writes the entire physical matrix with `beta=0`.
  The FP16 path uses the existing `LaunchFp16LinearGradWeight` helper, also with
  beta zero and FP32 output. `ColumnSum` writes the complete physical bias
  stride, including zero padding.
- Retain all input-gradient clears required by sparse/fallback paths, all
  model/optimizer semantics, precision, allocation sizes, and dispatch choices.

At the measured physical shape, the removed traffic is
`32*(224*224+224)*4 + (2560*224+224)*4 + 224*4 = 8,746,752` bytes/step.
The last term is the output weight portion removed from the narrowed clear;
the bias clear itself still executes. This matches the measured byte delta.

Nonzero padded weights and Adam moments are legal: the proof does not assume
that they are zero. Padding claims concern only gradients whose support is
zero because the corresponding post-BN activation or dY is zero. No dense
reduction arithmetic or accumulation order is changed by deleting a clear.

## Correctness evidence and its limits

[test_local_mlp_fp16_activation_tape.cu](../../native/tests/cuda/test_local_mlp_fp16_activation_tape.cu)
at this checkpoint poisons every physical weight-gradient element, every affine-gradient
element, and loss before each real step, alternating NaNs and a finite nonzero
sentinel. Its existing independent CPU references, input states, Adam settings,
five shape variants, and row sequence `4,3,4` are preserved. This remains the
unpadded numerical oracle; it is not relabeled as a padded-model oracle.

[test_local_mlp_gradient_overwrite.cu](../../native/tests/cuda/test_local_mlp_gradient_overwrite.cu)
adds two complementary checks:

1. Complete-step zero-versus-poison twins start with the same model and finite
   nonzero weights/moments, including physical padding. They retain their own
   updated states through rows `4,2,1,4`. Physical widths 5/3 and 8/4, logical
   widths 3/2, one residual block, and output widths 1/2 cover odd/scalar and
   aligned paths. The local executable has eight configurations / 32 paired
   steps (64 train-step calls), across FP32 and FP16+tape. The NCCL executable
   has four configurations / 16 paired steps (32 public FP32 train-step calls),
   with `local_rows == global_rows` and a real world-size-one context.
2. Fixed post-BN operands directly exercise FP32 `cublasSgemm(beta=0)`, the
   production FP16 gradient helper, and the actual private `ColumnSum<4>` and
   `<8>` kernels. The three fixtures are `(rows,in,out,logical_in,logical_out)`
   = `(17,5,3,3,2)`, `(257,8,4,3,2)`, `(257,224,224,219,221)`. In each executable,
   six precision variants make 12 zero-versus-poison comparisons, using both
   `0xffffffff` quiet NaNs and `0x5a5a5a5a` finite sentinels. All dW/bias float
   bit patterns must match and be finite. Input addresses, data and BLAS handle
   stay fixed, so this check is independent of BN atomic-reduction variation.
   Small fixtures additionally check host double-accumulated dot products;
   all fixtures check host bias sums. Inputs are half-exact dyadic values.

The full-step twins compare all physical weight gradients, affine gradients,
loss, weights, affine parameters, both sets of Adam m/v, running statistics,
and outputs. Every compared value must be finite. Their tolerance is
`3e-6 + 3e-5*max(abs(a),abs(b))`; they do not require bitwise equality across
independent full BN/atomic executions. The fixed-operand checks do require
bitwise equality. Both check exact numerical zero on mathematically padded
gradient support, allocation guards and immutable input bytes; mixed full
steps also check the updated half-weight mirror. CPU MSE is checked from the
outputs and labels. These bounded checks are not a long-run convergence test.

### Recorded gate chronology

| Queue job | Actual result |
|---|---|
| [`3759bbfaaf62`](../../.gpu_queue/logs/3759bbfaaf62.log) | Before production edits, all three semantic characterization CTests passed. The separate trace budget assertion failed as intended: `(104,8902888,463)` versus expected `(71,156136,463)`. The wrapper recorded `EXPECTED RED` and exited 0. Only the structural budget was RED; the semantic poison tests were already GREEN. |
| [`24b782fc302e`](../../.gpu_queue/logs/24b782fc302e.log) | Initial clear-elided build: nine targeted CTests passed; local overwrite memcheck/leak-check and initcheck reported zero errors, NCCL world-one overwrite memcheck/leak-check reported zero errors/leaks, and one production batch4096 step passed memcheck. Exit 0. This version still emitted an unused-`pc` declaration warning. |
| [`75562722316d`](../../.gpu_queue/logs/75562722316d.log) | Final build including the fixed-operand checks: all three targeted CTests passed, then complete local memcheck/leak-check, local initcheck and NCCL memcheck/leak-check passed with zero errors/leaks. The job subsequently **failed, exit 1**, because `cmp` compared the newly built benchmark to the old frozen warning-version binary. The logged SHA256 values differ. Its later planned A/B and ABI commands did **not** execute. |
| [`f443741f3ae3`](../../.gpu_queue/logs/f443741f3ae3.log) | Recovery froze the final binary and logged its SHA256; production batch4096 memcheck passed; one preconditioning run was logged; all six final paired runs completed; final Nsight capture/export and exact-budget/sequence analysis succeeded; four CPU/C ABI CTests and Rust FFI passed. Exit 0. |

The nine initial CTests were `local_mlp_batch_norm_full_step`,
`local_mlp_batch_norm_fp16_full_step`, `local_mlp_fp16_activation_tape`,
`local_mlp_gradient_overwrite`, `local_mlp_batch_norm_grouped_rows_full_step`,
`column_sum_tiled`, `input_half_tiled`, `single_gpu_trainer_lifecycle`, and
`nccl_mlp_gradient_overwrite`. The final three were the two overwrite targets
and activation-tape target. The recovery CPU/C ABI targets were
`single_gpu_contract`, `puzzle_io`, `random_walk_cpu`, and
`single_gpu_trainer_ffi`; Rust ran `ffi_layout_and_raii_owner` (one passed).
These are targeted sets, not a claim that every repository test was rerun.

## Binary identity and separate timing datasets

| Role | Frozen benchmark | SHA256 |
|---|---|---|
| Checkpoint baseline | `/tmp/mgt-input-half-t128` | `1b661b812751618932e63cac09c4e4cadcfeebed0b596a3d07af8785e8973212` |
| Initial clear-elided, unused declaration retained | `/tmp/mgt-dense-clear-elided` | `2635893d6eaf30fcdf6659761acfd1bbed7d435236e13bd8699a306e9ddbd2cf` |
| Final clear-elided, unused declaration removed | `/tmp/mgt-dense-clear-final` | `149ae33480bb06397ad13d1b1694a49579d61ab3d7e9121efcde5b1e661c775a` |

The initial and final candidate binaries are not byte-identical and their
datasets are not pooled. Both datasets use 140 warmup steps and 100 timed steps
per run, order baseline/elided/elided/baseline/baseline/elided, batch4096,
`native/production_inputs/p888.json` and `native/tests/fixtures/p888-target.bin`.
Every recorded run is retained. Each number below is the benchmark's reported
mean step time for that run, not an individual-step sample.

| Run index | Variant | Initial dataset, ms | Final dataset, ms |
|---:|---|---:|---:|
| 0 | baseline | 21.7472 | 25.3376 |
| 1 | elided | 25.1622 | 25.1471 |
| 2 | elided | 25.1546 | 25.1633 |
| 3 | baseline | 25.3432 | 25.3633 |
| 4 | baseline | 25.3431 | 27.0986 |
| 5 | elided | 25.1629 | 26.2387 |

Initial medians: 25.3431 -> 25.1622 ms, descriptive throughput difference
`(25.3431/25.1622-1)*100 = 0.7189%`. Final medians: 25.3633 -> 25.1633 ms,
descriptive latency reduction 0.7885% and throughput difference 0.7948%.
The separately logged final preconditioning baseline run was 25.3607 ms;
it is not one of the six paired records and is not included in either median.

Initial active SM-clock samples span 780--1305 MHz; the first baseline run is
visibly faster during higher clocks. Final baseline samples contain 780 and
1185 MHz, while final candidate active samples are 780 MHz. Final temperatures
span 77--80 C for baseline and 78--81 C for candidate. Both final variants have
a slow run (27.0986 and 26.2387 ms). The last candidate idle check also reports
137 MiB with 0% utilization; it is not evidence of a fully isolated thermal
experiment. No run is discarded as an outlier.

Telemetry is asynchronous `nvidia-smi` sampling, requested at approximately
0.1-second intervals, filtered to utilization >=90% and memory used >0 MiB.
It covers process activity including initialization/warmup and is not aligned
to the timed steps. The last-two-second summary is diagnostic only. Clocks
were not locked; neither dataset has a common constant sampled clock across
all runs. **The all-run medians are descriptive, not a controlled speedup.**
Per-run losses are finite but vary; the benchmark values are not used as a
convergence or bitwise full-training equivalence claim.

## Nsight Systems: exact work budget versus elapsed span

The final capture used 100 warmup steps followed by four measured steps.
The read-only analyzer splits steps at `RandomWalkKernel` and ends each at the
terminal AdamW kernel. All four steps are checked structurally; timings below
are means of captured steps 2--4. Durations are derived from nanoseconds and
reported in milliseconds; byte counts are bytes, not KiB.

| Per-step quantity | Baseline | Final |
|---|---:|---:|
| Kernel launches | 463 | 463 |
| Memcpy calls | 68 | 68 |
| Memcpy bytes | 78,000 | 78,000 |
| Memset calls | 104 | 71 |
| Memset bytes | 8,902,888 | 156,136 |
| Summed kernel time, ms | 23.6968263 | 23.6963447 |
| Summed memset time, ms | 0.2305510 | 0.1311737 |
| Summed memcpy time, ms | 0.1916573 | 0.1909843 |
| GPU activity union, ms | 24.1190347 | 24.0185027 |
| Uncovered time inside step span, ms | 1.9977043 | 1.8864763 |
| Step span, ms | 26.1167390 | 25.9049790 |

All four normalized kernel launch sequences and geometries compare equal.
This does not assert identical machine code. Kernel time is effectively
unchanged (difference about 0.000482 ms), whereas memset time falls by about
0.099377 ms. The 0.211760 ms span difference also includes about 0.111228 ms
less uncovered time and a small copy-time difference. It must not be described
as an all-GPU-kernel acceleration or attributed wholly to removed memset GPU
execution. The initial candidate trace separately reported kernel time
23.6931207 ms, memset time 0.1301307 ms and span 25.9005347 ms; those values are
retained under its own summary and are not substituted for the final trace.

## Retained artifacts and replay boundary

Under [test_results/dense_gradient_clear_20260831](../../test_results/dense_gradient_clear_20260831):

- `paired-elided.jsonl` and `paired-elided-summary.json`: initial six runs;
  JSONL SHA256 `e37861a78cc2b01c216c5459f0a9abb6c318e6224494e6e163c1f87c9f7517f7`.
- `paired-elided-final.jsonl` and `paired-elided-final-summary.json`: final six
  runs; JSONL SHA256 `e9eb6f5f77cf3ff518c9750f783d758493b689ae31d6051fbb9ddfd4a0c34b89`.
- `elided-warmed.nsys-rep`, `elided-warmed.sqlite`, `elided-profile-summary.json`:
  initial candidate trace; SQLite SHA256
  `3edde7aecadc9210d2620bf5073b2f1601cdcab80743991d4ca977c01d4712e1`.
- `elided-final-warmed.nsys-rep`, `elided-final-warmed.sqlite`,
  `elided-final-warmed-profile-summary.json`: final trace; SQLite SHA256
  `9eb41f68cdb06d117f249dc09ea4d7ed637fcde0484583040e141b1f52acf3f7`.
- Baseline trace: `test_results/input_half_tiled_20260831/t128-warmed.sqlite`,
  SHA256 `e79d38d47ac40771a427e325691bff9819a2d98d44a527a95ecd8253e64c7a38`.

`run_paired.py`, `summarize_paired.py`, `analyze_nsys.py`, `analyze_profiles.py`,
and `check_memset_contract.py` retain commands and attribution logic. The paired
runner currently selects the final binary; recreating the initial experiment
requires explicitly selecting its recorded old binary, not overwriting either
dataset. Output files are created exclusively; use a fresh output name for any
new run. Queue logs linked above retain the exact build, CTest, sanitizer,
profiling and ABI commands. GPU work must remain serialized through the shared
queue; this document's evidence audit itself performed no new GPU run.

The fixed-operand test source SHA256 recorded at these gates is
`d51f357c49df3c743f1f0b5eff514dc62be1081bc0f8b4bcce4ed26a0bc3d9a9`;
the activation-tape test SHA256 is
`15972ab0b86c1663f62c5f7408999052c7b371903ed8d1c49200bb763d2320d4`.
The later BN-backward mirror checkpoint extends the activation-tape test with
cache/alias cases and a sixth shape; the hash and five-shape description here
refer to the earlier dense-clear gate, not to that extended test source.
An independent read-only review reported no actionable code issues; acceptance
rests on the overwrite contract, targeted correctness gates, and exact trace
budget above, not on a claimed controlled throughput gain.
