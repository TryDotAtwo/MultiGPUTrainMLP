# Local BN epilogue fusion: original p888, SM86

Date: 2026-08-31. Follow-up to
[the activation-tape/IO audit](2026-08-31-io-aware-mlp-training.md).
Scope: fuse local BN apply -> optional residual -> ReLU -> optional FP16 mirror.
No model, optimizer, distributed SyncBN, or GEMM-policy redesign.

Accepted result: median run-mean step time35.4149 ->34.2634ms
(-3.25% latency, +3.36% throughput), no arena growth. Nsight forward span
9.877088 ->8.676607ms (-12.15%);34 standalone activation launches eliminated.
Profiled spans and unprofiled throughput are separate measurements.

## Contract and implementation

The workload remains batch4096, logical hidden widths2556/218, physical
strides2560/224, 16 residual blocks and34 BN sites, scalar MSE output.
FP32 normalized values, activations, residuals, running statistics and master
weights remain authoritative. FP16 is only a consumer-side GEMM mirror.

- `LaunchLocalStridedBatchNormApply` applies already-computed full-batch
  statistics. Five compile-time variants cover plain BN and ReLU with/without
  residual and half output. No allocations, synchronization or new scratch.
- Local forward routes all34 sites through the fused consumer, including the
  FP32 local backend. Distributed SyncBN keeps separate activation calls.
- Existing partial/finalize reduction algorithms and their launch boundaries
  remain. Integer launch-bound arithmetic is made overflow-safe; rows*stride
  exceeding INT_MAX is rejected. GEMM and backward policy are unchanged.
- Scalar heads emit no unused final half slot, including the zero-block case.
  Vector heads retain their final half slot.
- x==y is legal; partial overlap and destructive normalized/residual/half
  aliases are rejected. Full forward checks half overlap against running
  mean/variance and reduction workspace before any asynchronous writes.
- Padding first gets normalized=0 and affine=0, then executes residual/ReLU.
  A positive residual in padded columns survives, matching the old composition.

This is the IO-aware *local consumer fusion* principle, not FlashAttention
itself. BN still needs its full-batch reduction barrier. Nominal intermediate
FP32 traffic removed is `4096*(2560+33*224)*8 = 326107136 bytes = 311 MiB/step`:
one affine store plus its subsequent activation read. This is a source-level
global-memory traffic count, **not measured DRAM bytes**; cache residency matters.
Arena size remains739,806,720 bytes, unchanged from activation tape.

## Precision boundary

Baseline SM86 SASS is normalize `FADD -> FMUL`, affine `FFMA -> STG`.
The fused kernel retains the ordinary affine expression and uses explicit
`__fadd_rn(residual, affine)` before ReLU. This preserves the separately rounded
addition across the former FP32 store. NVIDIA documents that this intrinsic
cannot merge into a multiply-add instruction:
[CUDA12.8 single-precision intrinsics](https://docs.nvidia.com/cuda/archive/12.8.1/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__SINGLE.html).

The test includes `(1+2^-23)*(1-2^-23)-1 = -2^-46`, then residual2^-45,
giving positive FP32 activation2^-46 but FP16 mirror+0. It also tests values
that would change if affine and residual were reassociated, signed zero, NaN
ReLU policy, positive padding, and exact in-place operation.

## Correctness and review gates

| Gate | Evidence |
|---|---|
| Initial runtime RED | `a3217039dd8e`: old apply/stub fails `FP32 output bit mismatch at 0` |
| Initial9 CUDA tests | `2b9bd07e1e1a`:9/9 passed |
| Alias-regression RED | `9856cd9b77e2`: `full-forward half_output alias accepted: running_mean` |
| Final9 CUDA tests | `aa34cdf0740f`:9/9 passed |
| Final epilogue memcheck | Same job:0 errors |
| Final epilogue racecheck | Same job:0 hazards,0 errors,0 warnings |
| Final production4096 memcheck | Same job:1 complete step,0 errors |
| Independent code review | Running-state alias gap and missing invalid-call payload checks fixed; re-review has no remaining findings |

The new epilogue test compares exact FP32/FP16 bits with an independent
old-affine kernel followed by a separate activation kernel. Shapes cover
255/256/257-element boundaries, odd strides, scalar width,2556:2560, and
4096x218:224; all valid descriptor variants run in/out-of-place. Guard regions,
unchanged input/parameter buffers and invalid-call payloads are checked.
The full-forward dyadic fixture allows bitwise comparison of statistics,
running state, normalized values and outputs despite atomic arrival order.

Existing full-step FP32/FP16, grouped-row, activation-tape edge, input-half and
trainer-lifecycle tests are included. These are targeted gates, not an
independent long-run Adam/convergence proof. Atomic BN/loss accumulation remains
non-deterministic; final training-loss differences alone do not prove a cause.

## Measurement record

All GPU runs use the existing `mgt-gpu-queue` Docker queue and image
`mgt-single-gpu-dev:2026-08-28`. RTX3070 Laptop, SM86, CUDA12.8.93, Release/Ninja,
CUTLASS commit`ffa119a`; no power/clock settings changed.
The accepted tape binary was copied before editing and SHA256 recorded:
`285cd22c13cdfa5000b763d2d6904f645e978ac584eb2210d0545bb2979860e6`.

Paired wall-clock benchmark:4096 rows,5 warmups,20 timed training steps,
original non-identity `native/production_inputs/p888.json` and matching target.
Order A-B-B-A-A-B,3 independent runs per variant; each reported step_ms is a
run mean, not a per-step percentile. GPU telemetry is sampled every100ms.
Files under `test_results/bn_epilogue_20260831/` are retained separately:

- `paired.jsonl`: preliminary fusion, before host-only alias validation fix.
- `paired-final.jsonl`: final binary, but initial clocks drifted; not used for
  the headline speed ratio. Keep this entire dataset as evidence.
- `paired-confirm.jsonl`: another short final-binary set, also clock-drifty.
- `{tape,fused}.nsys-rep` and SQLite:4 steps each, first cold step excluded.
- `codegen.json`: frozen baseline/final hashes and BN apply SASS/resources.

The short `paired-confirm.jsonl` also drifted and is **not** a headline dataset.
Its complete measurements remain in `benchmark-summary.json` alongside the
other short sets. Do not pool the preliminary and final binary hashes or select
only convenient pairs from these sets.

### Warmed final wall-clock comparison

Final confirmation uses **100 warmup +100 timed steps**, unchanged A-B-B-A-A-B
order,3 runs per variant, job`38f7e3648ac3`. Raw data:
`paired-warmed100.jsonl`, analysis:`benchmark-warmed100-summary.json`.
Final binary SHA256:
`3422df2cfed1cb9cab1b5922b3162a95d615e03a8df2179fb640ca8eb9e63747`.

| Variant | Three run means, ms/step | Median run mean | Approx. samples/s |
|---|---|---:|---:|
| Accepted activation tape |35.4457,35.4149,35.3847|35.4149|115,658|
| Local BN epilogue fusion |34.2634,34.2771,34.2453|34.2634|119,544|

Delta:-1.1515ms/step, -3.2515% latency, +3.3607% throughput. All six runs and
their telemetry are retained. All last-two-second active samples were780MHz;
five runs were780MHz throughout active samples, while the first tape run
included early780-1350MHz warmup samples. Tail temperatures rose76-84C over
the full A/B sequence. Telemetry is sparse and not precisely aligned to the
timed window: these are warmed observed results, not a locked-clock experiment
or a confidence interval. The tight timing ranges support the observed gain.
All variants retain739,806,720 arena bytes.

The attempted200+100 run (`a3dfdcf6b641`) failed in the **baseline** with exit8:
the benchmark always passes epoch0 and does not wrap sample offsets. The
999,978-sample original epoch permits244 complete4096-row batches, not300.
The trainer correctly rejects an epoch-crossing request. No trainer contract
was changed;100+100 remains inside one epoch. The failed empty output and queue
log are retained. Benchmark multi-epoch rollover remains outside this patch.

### Nsight attribution

Job`28acecfed5de`; means over steps2-4, first cold step excluded. Whole-step
window:RandomWalk start to final AdamW end; forward:InputHalf2Row start to
OutputBias end. `analyze_profiles.py` reads SQLite in read-only mode and
exclusive-creates`profile-summary.json`; all count and sequence checks passed.

| Mean / count per step | Tape | Fused |
|---|---:|---:|
| Whole-step GPU span, ms |36.203016|35.013122|
| Whole-step kernel sum, ms |33.690115|32.610157|
| Forward GPU span, ms |9.877088|8.676607|
| Forward kernel sum, ms |9.058773|7.973883|
| Total kernels |497|463|
| Forward kernels |205|171|
| BN apply kernels |34|34|
| Separate forward activation kernels |34|0|
| BN apply kernel time, ms |1.529120|1.938753|
| Separate activation kernel time, ms |1.511317|0|
| BN partial / finalize counts |34 /34|34 /34|
| All FloatToHalf count |33|33|
| FloatToHalf time, ms |0.757220|0.757304|
| Dense GEMM mains / split-K reductions |99 /33|99 /33|
| Dense GEMM main time, ms |5.881411|5.885760|
| Interval not covered by recorded GPU activity, ms |2.085601|1.983850|

The former BN-apply + activation sum3.040436ms becomes1.938753ms:
**1.101684ms saved** in those kernels. Remaining kernel names/order and
grid/block/dynamic-shared geometry match for all four steps after removing
standalone activations and canonicalizing BN apply variants. This structural
comparison is not a numerical-equivalence proof; numerical tests are separate.

Both traces have68 D2D copies/78,000 bytes and104 device memsets/8,902,888
bytes per step; no H2D/DtoH in the measured window, no boundary-crossing memory
events. Uncovered time is not automatically CPU idle or removable launch cost.

SASS/resource evidence: baseline BN apply22 registers, fused variants20-22,
all STACK=0 and LOCAL=0, no local-load/store instructions in the inspected
BN apply kernels. Residual variants retain FFMA followed by a separate FADD.
Nsight also reports localMemoryPerThread=0; its identical nonzero
localMemoryTotal export field is not interpreted as spill traffic.

## Remaining work

- Largest remaining measured group: sparse input-weight gradient8.958232ms
  plus grouped-row construction1.841283ms =10.799515ms/step. This is the next
  high-impact audit target; inspect traffic/ownership before changing kernels.
- A separate, smaller IO-fusion candidate remains BN backward apply -> FP16 dY
  mirror:33 casts currently cost0.757304ms. Keep FP32 derivatives for BN/affine
  consumers and preserve dW/dX half reuse; do not combine this with a sparse
  gradient rewrite in one unmeasured patch.
- Full independent Adam trajectories, full convergence, T4 and multi-GPU
  validation are not established by these SM86 targeted gates.

Workflow: `write-cuda-hot-paths` constrained precision, alias/lifetime and
resource checks; `test-driven-development` required runtime RED before each
implementation/fix; `use-compute-sanitizer` and `use-nsight-systems` supplied
the memory/race and attribution gates. Independent numerical, test and code
reviews complemented rather than replaced execution evidence.
