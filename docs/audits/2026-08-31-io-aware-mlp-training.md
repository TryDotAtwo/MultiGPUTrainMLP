# IO-aware training audit: original p888 on SM86

Date: 2026-08-31. Scope: original Rust/C++/CUDA single-GPU trainer, not a
Transformer redesign. Current change: persistent per-step FP16 activation tape.
Research and independent reviews precede further kernel fusion. This document
records the tape stage. The subsequently implemented local BN epilogue fusion
and its separate measurements are in the
[follow-up audit](2026-08-31-bn-epilogue-fusion.md).

## Contract and actual execution

- Input: 72 categorical positions with 72 values, 5184 logical one-hot features;
  compact states stay on GPU instead of materializing the one-hot matrix.
- Hidden widths: logical 2556/218, physical 2560/224; 16 residual blocks, 34 BN
  sites; batch4096; scalar MSE head.
- FP32 master weights, Adam state, BN statistics/normalized values, residuals
  and saved activations. Dense GEMMs consume FP16 operands with FP32 accumulation.
- Tape-stage local forward: input gather or GEMM -> bias -> BN partial moments -> finalize
  -> BN apply -> ReLU or residual+ReLU -> next linear. The local implementation
  overrides the distributed BN selector; `MGT_BN_FUSED_EPILOGUE` does not select
  this local path.
- References in code: `native/cuda/local_batch_norm.cu`,
  `native/cuda/mlp_batch_norm_forward.cu`,
  `native/cuda/mgt_cuda/fp16_linear_train_ops.cuh`.

## FlashAttention: what transfers

FA1 tiles attention and keeps compact softmax statistics so the large attention
matrix need not be materialized. Backward recomputes selected tiles on chip.
This illustrates that more arithmetic can be faster when it removes expensive
memory traffic. Its attention-specific IO bound is not an MLP complexity result.
Our activation tape makes the opposite capacity tradeoff: retain an extra small
representation to avoid reconstructing it twice. Both decisions must be judged
by traffic and lifetime, not by a universal preference for recomputation.
[FA1, sections 3.1-3.2](https://arxiv.org/html/2205.14135v2).

FA2 reduces non-matmul work and communication between warps and creates more
independent output tiles. Its attention forward splits Q ownership rather than
requiring inter-warp partial-output reductions. Larger tiles can still lose to
register pressure or spills. This motivates shape-specific ownership and tile
tuning, not a blanket ban on split-K: our residual dW has a small 224x224 output
and a 4096 reduction dimension, where extra reduction partitions may usefully
increase parallelism.
[FA2, sections 3.1-3.3](https://arxiv.org/html/2307.08691v1).

FA3 overlaps movement, matrix work and softmax using Hopper-specific TMA,
asynchronous WGMMA and producer/consumer warpgroups. Its FP8 path is a separate
precision decision. The scheduling principle is useful; those hardware
instructions are not an SM86 implementation recipe.
[FA3, sections 2.2-3.2](https://arxiv.org/html/2407.08608v1).

FA4 extends the resource accounting beyond Tensor FLOPs and DRAM: shared-memory
traffic and scalar/exponential throughput can become the limiting resources.
Its Blackwell pipeline uses TMEM and 2-CTA MMA; its approximate exponential and
conditional softmax rescaling are attention-specific numerical choices, not
permission to approximate BN. The portable lesson is to identify the limiting
resource separately for forward/backward and remove duplicated staging.
[FA4, sections 3.1-3.2](https://arxiv.org/html/2603.05451v1).

Current-source caveat: the FA4 README emphasizes Hopper/Blackwell, but its public
interface contains an SM80 forward dispatch for architecture8.x. The inspected
backward entry has an early architecture assertion allowing 9/10/11/12; do not
infer full Ampere training support from the imported SM80 backward class. This
is source inspection, not a runtime compatibility test. Similarly, the current
README links a separate contributor-maintained Turing implementation; the old
blanket advice to use only FA1 on T4 is outdated.
[FA interface, main inspected 2026-08-31](https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/cute/interface.py),
[official support overview](https://github.com/Dao-AILab/flash-attention),
[Turing implementation](https://github.com/ssiu/flash-attention-turing).

Ampere supports asynchronous global-to-shared copies. SM86 has 100 KiB shared
memory per SM and at most 99 KiB per block. A whole 224x224 FP16 weight matrix
already takes 98 KiB, before activations/pipeline buffers. Whole-layer shared
residency is consequently not an attractive default. Compile/tune for SM86;
revalidate separately on SM75/T4.
[CUDA 12.8 Ampere guide](https://docs.nvidia.com/cuda/archive/12.8.0/ampere-tuning-guide/index.html).

CUTLASS explains hierarchical tiling, coalesced epilogues and double-buffered
operand movement. A custom elementwise epilogue may avoid an intermediate
round-trip, but changing GEMM implementation can cost more than that saving.
Its split-K scheme also requires a completion reduction; statistics must be
computed from completed preactivations, never squares of split-K partials.
[CUTLASS efficient GEMM](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cpp/efficient_gemm.md).

## Why BatchNorm is a real boundary

For each channel, training BN needs all batch rows before normalization. A row
tile's partial moments do not suffice. Backward similarly needs the complete
reductions of dY and dY*normalized. Online combination of partial moments does
not remove these dependencies. Microbatch-local BN, LayerNorm or frozen running
statistics would change the requested model.
[Original BN algorithm](https://arxiv.org/html/1502.03167v3).

```text
GEMM/bias -> partial moments -> complete moments/running-state update
                                      |
                                      v
                  BN affine + residual/ReLU + FP32/FP16 outputs
                                      |
                                      v
                                 next GEMM
```

The proposed next fusion crosses only the final elementwise edge, not this
batch-wide boundary. Keep normalized FP32 values for backward: ReLU is not
invertible, and division by gamma is not a safe reconstruction method.

cuDNN documents BN+activation and BN+add+activation, supporting the fusion
pattern. Its fast semi-persistent path requires specific packed NHWC/half
layouts and workspace/reserve space; it is not a drop-in replacement for our
FP32, logical218/stride224 BN tensors. The legacy Ex APIs are deprecated in
cuDNN9, so any library substitution needs a separate layout/numerical audit.
[cuDNN BN operations](https://docs.nvidia.com/deeplearning/cudnn/backend/latest/api/cudnn-ops-library.html#cudnnbatchnormalizationforwardtrainingex).

## Current tape and byte ledger

Store the FP16 representation in its activation producer once, consume it in
both the next forward GEMM and that linear's dW. Preserve the unrounded FP32
activation for residuals, ReLU masks and scalar output. dY conversion remains
once per dense site and is already reused between dW and dX.

Production tape: `4096*(2560+32*224)` half elements = 76 MiB. It replaces a
20 MiB conversion scratch, so net arena growth is 56 MiB (58,720,256 bytes):
681,086,464 -> 739,806,720 bytes. This is a speed/capacity tradeoff, not a claim
of lower total VRAM use. Expected standalone casts: 99 -> 33 per step.

For a narrow site: physical P=4096*224=917,504; logical L=4096*218=892,928.
The table counts logical bulk global loads/stores, not measured DRAM traffic.
Cache hits, transaction granularity, atomics and kernel scheduling are excluded.

| Candidate after tape | Analytical pass saving | Launch saving | Risk |
|---|---:|---:|---|
| BN apply + activation/residual + optional half output | 311 MiB/step | 34 | Low: pointwise rounding/padding |
| BN backward apply + half operand output | 115.5 MiB/step | 33 | Medium: dY cache ownership |
| Bias + existing forward statistics producer | 112.40625 MiB/step | 33 | Medium: atomic scheduling |
| BN backward apply + bias-gradient partials | Up to 112.40625 MiB, minus partials | Not yet fixed | Higher: reduction order |
| GEMM bias/statistics epilogue | Backend-dependent | Backend-dependent | Higher: tile occupancy/split-K |

Forward fusion removes one FP32 intermediate store and reload: 8 bytes per
physical element, over `4096*(2560+33*224)` elements = 311 MiB. These opportunities
are not additive speedup forecasts. The old trace's ReLU/residual total around
1.16 ms also is not guaranteed removable time.

## Independent review and acceptance gates

Three independent experts reviewed FlashAttention transferability, tape
lifetime/boundaries, and BN fusion. They converged on pointwise BN epilogue
fusion after measuring the tape. Static review found two generic API defects:

1. nrd=0/scalar: hidden activation tried writing a non-existent half slot.
2. vector output: final activation was neither stored nor mapped for forward
   and output dW.

Neither shape is the fixed production p888 contract; both are legal generic
shapes and must remain supported. Add a final slot only for vector heads;
scalar heads need no final half activation.

Acceptance sequence:

1. RED reproduction, then GREEN: zero/multiple blocks, vector heads, canaries,
   capacity-minus-one, active rows4->3->4, every saved half slot and gradients.
2. Existing FP32/FP16/grouped-input/lifecycle tests; memcheck/racecheck.
3. Same-session alternating HEAD/tape unprofiled runs, same workload/seed/
   warmup/build flags; report thermal/clock context, not old-vs-new clock drift.
4. Separate Nsight trace: verify cast count, unchanged GEMM count, forward and
   full-step breakdown. Profiling time is not unprofiled throughput.

Preserve epsilon1e-5, momentum0.1, biased training variance, unbiased running
variance, FP32 optimizer authority, padding zeros and weight-update ordering.
Fusion must retain rounding between BN affine and residual addition. Compare
all gradients and state, not only finite loss. Existing atomic reductions mean
bitwise end-to-end reproducibility is not assumed.

## Runtime evidence

Existing queue container restarted without building a new image. Verified CUDA
12.8.93, Nsight Systems2025.6.3, CUTLASS commit `ffa119a` and CUDA architecture86.
Frozen pre-tape HEAD `ebf785d` built in an isolated Docker temporary directory;
working-tree files and unrelated build directories were not replaced.

### Correctness

- RED queue job `e251cef45442`: zero-block scalar canary overwritten at half
  offset12; vector heads failed at launch. These failures occurred before the
  minimal capacity/final-slot/producer fixes.
- `eef1f962f97d`: **8/8 targeted CUDA tests pass**: local BN, FP32 full step,
  FP16 full step, activation-tape edges, grouped-row full step, input half,
  trainer lifecycle, activation kernels.
- The new edge test covers five model shapes, three consecutive active-row
  sizes4/3/4, every weight/BN affine gradient, running-state updates, exact
  activation/weight half mirrors, guards and capacity-minus-one rejection.
  Against its independent mixed-precision CPU oracle, maximum absolute errors
  were 3.815e-6 for weight/affine gradients, 4.769e-7 for outputs and 1.193e-7
  for running state. It re-bases each step on the current master weights; it is
  not an independently simulated multi-step optimizer trajectory.
- The first test fixture had near-constant BN channels and was numerically
  ill-conditioned under FP16 perturbations. It was replaced with fixed dyadic
  weights/non-correlated input pairs; tolerances were **not widened**.
- Same gate job: edge test memcheck **0 errors, 0 leaked bytes**; racecheck
  **0 errors/warnings**. `5d9d7cb753cc`: production4096, one complete step,
  memcheck **0 errors**, loss295.083. These sanitizer runs are not timings.
- This is targeted acceptance, not a complete repository regression, long-run
  convergence study or a claim of all possible concurrency hazards being absent.
- Final independent code review found no new Critical/Important defects.
  Non-blocking follow-up: a dedicated logical<physical padded tape oracle case;
  existing local-BN padding tests and production memcheck are not that oracle.
  Operand-A sizing/lifetime is documented at its public context field. Existing
  operand-B capacity checks still occur later in backward; this patch does not
  claim transactional preflight of every supplied buffer.

### Unprofiled paired performance

Both variants use original p888, batch4096, five warmup steps and twenty timed
steps, identical seed/data/flags and the same queue container. Each reported
run is average wall-clock time per step; the summary is median of three runs.
Order: HEAD, tape, tape, HEAD, HEAD, tape. GPU telemetry is sampled concurrently
and retained. No GPU clock/power settings were changed.

| Final build, job71542e3577f7 | HEAD ebf785d | Tape |
|---|---:|---:|
| Run averages, ms | 36.8929 / 36.9224 / 36.9070 | 35.3791 / 35.3648 / 35.3528 |
| Median, ms/step | **36.9070** | **35.3648** |
| Throughput from median, samples/s | 110,982 | 115,821 |
| Arena bytes | 681,086,464 | 739,806,720 |

Result: **1.5422 ms less per step, -4.18% time / +4.36% throughput**, at the cost
of **56 MiB** additional arena storage. Active samples in both variants report
780 MHz; temperature80-82 C. Results characterize this laptop's observed state,
not maximum possible hardware throughput. The earlier independent paired set
`6c4b5bc04db1` also showed 36.9631 -> 35.4218 ms median.

Final losses after25 steps vary: HEAD171.940-172.684, tape172.233-172.677.
Ranges overlap, but this does not prove equivalent long-run convergence.
BN/sparse atomic scheduling and mixed-precision threshold sensitivity are
plausible causes; no causal attribution is claimed from these loss scalars.

Raw artifacts: `test_results/io_audit_20260831/paired.jsonl`,
`paired-final.jsonl`, `run_paired.py`; logs in `.gpu_queue/logs/` by job ID.

### Nsight attribution

Job `ded4cbeaf997` profiled four steps of each variant. Exclude cold step1.
Step span is RandomWalk start to final Adam end; forward span is InputHalf2Row
start to OutputBias end. These GPU spans are **not** unprofiled wall-clock
measurements. The trace precedes the generic zero-block/vector-head fixes;
those fixes do not change the production shape's selected operations.

| Mean over steps2-4 | HEAD | Tape |
|---|---:|---:|
| Whole-step span, ms | 37.920576 | 36.153184 |
| Whole-step kernel sum, ms | 35.259658 | 33.661111 |
| Forward span, ms | 10.587286 | 9.860891 |
| Forward kernel sum, ms | 9.695005 | 9.051291 |
| All FloatToHalf kernels / step | **99** | **33** |
| Forward FloatToHalf kernels / step | **33** | **0** |
| All kernels / step | 563 | 497 |
| GEMM main kernels / step | 99 | 99 |
| split-K reduction kernels / step | 33 | 33 |
| Cast kernel time, ms | 2.618796 | 0.755254 |
| Forward activation kernel time, ms | 1.240371 | 1.509451 |
| Memset time, ms | 0.221778 | 0.232190 |
| Memcpy time, ms | 0.190567 | 0.192019 |
| Interval uncovered by recorded GPU kernels/memory ops, ms | 2.248572 | 2.067863 |

Half dual-write adds 0.269080 ms to the activation kernels while eliminating
1.863542 ms of standalone casts. This explains why net gain is smaller than
removed conversion time. The 99 GEMM mains include98 Ampere library kernels
and1 CUTLASS main; scalar-output GEMV mains and their reductions are separate.

Each step has68 D2D copies totaling78,000 bytes and104 memsets totaling8,902,888
bytes. There are **no H2D/DtoH copies inside this step window**. D2D copy sizes
match two affine-gradient copyouts at every BN site; caller attribution is an
inference from code/sizes, not a per-event API correlation. The uncovered
interval is not automatically CPU idleness or removable launch overhead.

Independent review found identical remaining kernel order and launch geometry
after removing FloatToHalf and equating activation/mirror variants. This is a
structural check, not numerical equivalence proof.

Raw artifacts: `test_results/io_audit_20260831/{head,tape}.nsys-rep`, matching
SQLite files, `profile-summary.json`, `analyze_profiles.py`.

## Next bounded implementation

Implemented and verified in the [follow-up audit](2026-08-31-bn-epilogue-fusion.md).
The original acceptance scope follows:

Fuse only local BN apply + optional residual + ReLU + optional half output.
Do not touch BN partial/finalize, GEMM choice, normalized FP32 tape, or the
optimizer. First test fixed supplied statistics and padding/rounding at element
counts255/256/257; then full-step gradients/running state and production4096.
Accept only after another paired unprofiled measurement and Nsight check that
34 activation launches disappear without increasing register spills or changing
the reduction boundary. Backward half-output fusion follows as a separate patch.
