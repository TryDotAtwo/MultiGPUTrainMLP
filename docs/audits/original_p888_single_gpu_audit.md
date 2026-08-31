# Original p888 single-GPU audit map

Audit order is dataflow order. Performance promotion is forbidden until the
semantic and numerical gates before it are green.

## Latest accepted measurements (2026-08-31)

The A0-A12 entries below retain earlier-stage measurements; their old full-step
timings are not the current performance snapshot. The latest validated original
p888/SM86 batch4096 path is **22.4334ms/step** (median of three100-step run means,
140 warmup steps/run), approximately182.6k samples/s, arena739806720 bytes.
All active telemetry samples in the latest A/B were780MHz; clocks were not
locked and sparse samples do not prove a fully controlled timing window.

- [Column-sum tiling](2026-08-31-column-sum-tiling.md): same256-leaf FP32 tree,
  adjacent-column loads;34 reductions total2.4193 ->0.9721ms. Full-step gain is
  observational because all three A/B series include frequency drift.
- [Input-gather feature tiling](2026-08-31-input-gather-feature-tiling.md): same
  ordered half2-to-FP32 sums; NCU DRAM reads746.3 ->12.0MB. Matched-sampled-clock
  full-step confirmation25.8899 ->25.2253ms (+2.63% throughput).
- [Dense-gradient overwrite](2026-08-31-dense-gradient-overwrite.md): remove
  redundant clears after proving complete physical dW/bias writes, including
  padding.104 ->71 memsets,8,902,888 ->156,136 bytes/step; no kernel changes.
- [BN-backward half mirror](2026-08-31-bn-backward-half-mirror.md): emit RN-half
  dense operands with FP32 BN dX;33 casts disappear,463 ->430 kernels. A/B
  25.0926 ->24.4922ms (+2.45% throughput), all sampled active clocks780MHz.
  Nsight apply+cast time2.283121 ->1.686333ms; reductions/copies remain unchanged.
- [BN-backward mask fusion](2026-08-31-bn-backward-mask-fusion.md): original FP32
  activation predicates move into BN partial/apply; residual output retains
  incoming masked dY, not BN dX.18 ReLU and16 residual-mask launches disappear,
  430 ->396 kernels. Separate A/B stages measured24.5378 ->24.1517ms (descriptive,
  some warmup clock drift) and24.1523 ->23.8682ms (+1.19% throughput; all active
  confirmation samples780MHz). The initial clock-drifty residual series is
  retained but not used to claim its apparent19.7% gain. No new buffer or change
  to BN reduction order, FP32 accumulation, or the nonlocal BN selector.
- [BN input-bias fusion](2026-08-31-bn-input-bias-fusion.md): preserve rounded
  FP32 x+bias inside partial/apply, remove33 separate Bias passes;396 ->363
  kernels. Confirmation23.1719 ->22.4334ms (+3.29% throughput), all active
  samples780MHz. Fresh Nsight changed-family time falls by0.694ms; sparse
  remains3.683ms. Earlier baseline drift is not attributed to this patch.
- [CUDA Graph feasibility](2026-08-31-single-gpu-graph-feasibility.md): same-stream
  binding now preserves user cuBLAS workspace; default eager throughput is
  unchanged. A test/tool-only fixed-shape replay updates RandomWalk and both
  Adam nodes, not frozen data. Confirmation22.4366→21.4649ms(+4.53% throughput),
  all active samples780MHz; Nsight verifies identical363kernels/memory events
  and reduces uncovered time1.561→0.530ms. Public trainer/C ABI/Rust graph mode
  and tail-batch lifecycle are **not integrated**; the default snapshot above
  must not be replaced with this experimental timing.

No new precision/model approximation. CUDA exact oracles, targeted full-step
regressions, sanitizer checks and Rust FFI passed. Earlier checkpoints also had
independent reviews; the input-bias and graph-feasibility checkpoints were
completed without subagents. Neither convergence, complete single-GPU saturation,
production CUDA Graph integration, a training-plugin
release nor T4/multi-GPU readiness follows from these local checkpoints.

## A0. Source provenance and contract

- Exact hashes for group JSON, target tensor/binary, original trainer source,
  model metadata, and comparison checkpoint.
- Structured JSON parsing; field order and unrelated numeric strings must not
  change the move table.
- Validate 18 named permutations, 72 positions, target range, inverse map, and
  model parameter count.
- Current status: **green**. Production moves and archived target
  exist; the binary target was checked against the archived PyTorch tensor. The
  benchmark rejects an identity-only move set and loading rejects noncanonical
  inverse pairs. The parser reads named schema fields independently of field
  order, ignores unrelated numeric text, accepts exactly one `moves`/`actions`
  table, and validates exact dimensions/permutations. Automatic hashes in the
  benchmark JSON remain a reporting improvement, not a correctness gate.
- Frozen SHA256: group JSON
  `f2d7cae9a387d8acbe7e4082711179dc5a309232e4278733c90853534c649e02`;
  target binary
  `107de2bc788e11029f7851f8e1b0b5afb4e34379c709fc840689ebd3d1f51b5b`.

## A1. Batch generation semantics

- Exactly 34,482 samples at every depth 1..29 per semantic epoch.
- Exclude the inverse of the previous move, including the original move-zero
  `inverse_moves[-1]` quirk; do not merely exclude repetition of the same move.
- Preserve walk direction, label definition, epoch shuffle/sharding, tail batch,
  and resume position.
- CPU golden versus CUDA exact state/label/meta bytes; depth histogram,
  duplicates, cancellation rate, and cross-rank overlap.
- Current status: **green for single-GPU generation**. CPU and CUDA use a
  deterministic bijection over all 999,978 epoch positions, producing exactly
  34,482 rows per depth. Both implement canonical inverse exclusion including
  the move-zero quirk. A fixed production-input slice is byte-identical for
  CPU/CUDA `state`, `label`, and `meta`. C++, C ABI, and Rust carry semantic
  epoch plus epoch-sample offset. Multi-rank shard overlap remains an A1 gate
  for the later 2xT4 phase.

## A2. Generated storage and handoff

- `state[rows][80]` (`uint8`, 72 live + 8 zero padding), `label[rows]` FP32,
  `meta[rows]` 16 bytes.
- Verify alignment, coalescing, bytes written, padding, lifetime, and that meta is
  required downstream. Check whether generation can overlap the previous step.
- Current status: **green correctness / acceptable performance**. Static
  contracts prove 80-byte/16-byte layouts; CPU/CUDA exact-byte parity and
  memcheck pass. At batch 4096 NCU measured 0.276 ms. The 32-block grid reaches
  only 7.9% occupancy and uses 127 registers/thread, but the stage is below 1%
  of the end-to-end step, so it is not an optimization target.

## A3. Sparse input projection

- Reference equation for the 72 table lookups plus bias.
- Audit FP16 table/master ownership, load pattern, cache behavior, accumulation
  error, padding lanes, and produced activation layout.
- Measure useful bytes/s and NCU memory-stall counters.
- Current status: **green**. The promoted FP16 path assigns one block to each
  row, loads its 72 table offsets once into shared memory, and processes two
  features per thread with `half2`; odd physical/logical widths retain the
  scalar fallback. Direct scalar/CUDA comparison is bitwise exact for production
  rows 1/17/4096 and four parity combinations; memcheck reports 0 errors and
  racecheck reports 0 hazards. NCU 2025.1.1 changed the kernel from 4.01 to
  2.25 ms, 212.22 to 349.35 GB/s, and 362.4M to 123.6M executed instructions.
  Three unprofiled 20-step runs measured 43.7567--44.0184 ms/step versus the
  frozen 44.9383 ms baseline.

## A4. BatchNorm and activation

- PyTorch parity for all 34 sites, tail rows, running statistics, affine grads,
  aliasing, and padding.
- Attribute reduction/apply kernels, memory traffic, and launch count.
- Current status: **green correctness / acceptable performance**. The shared
  strided implementation is selected by all 34 plan sites. CPU/CUDA forward and
  backward parity now covers the production `2556/2560` input site, the
  `218/224` residual sites, a 4095-row tail, running statistics, affine grads,
  and exact zero padding; memcheck is clean. A four-step Nsight Systems window
  recorded exactly 136 instances of each of the five BN kernels (34 per step).
  Their summed GPU time is about 4.52 ms/step: apply 2.92 ms, reductions 1.46
  ms, finalize 0.14 ms. This is material but no longer the first red gate.

## A5. Hidden and residual forward

- Per-layer shape/layout/dtype/leading dimensions and FP32 reference parity.
- Record selected cuBLAS/CUTLASS algorithms, Tensor Core eligibility, issued
  versus useful FLOPs, and epilogue/materialization traffic.
- Current status: **green correctness / profiled**. The local FP16 and FP32 full
  steps match the frozen CPU-derived fixture, including all residual parameter
  gradients. SM86 selects `ampere_s168*` Tensor Core kernels (plus CUTLASS for
  the hidden weight-gradient shape). Physical dense work is about 53.56 GFLOP
  per complete train step; the GEMM kernels consume about 5.87 ms, or 9.12
  issued TFLOP/s inside GEMMs. Against the full 44.01 ms median step this is
  only 1.22 issued TFLOP/s, confirming that non-GEMM stages dominate.

## A6. Output and loss

- Scalar head, bias, MSE normalization, loss reduction, and label alignment.
- Compare output/loss/dY against the reference on fixed real states.
- Current status: **green for scalar-head semantics**. The fixed CPU-derived
  fixture checks scalar output, MSE loss `3.6869926453`, output weight/bias
  gradients, and both FP32 and FP16 complete local steps. Nsight measures the
  output bias, scalar loss reduction, output-input gradient, and two GEMVs as
  negligible relative to the step. A larger real-state fixture remains useful
  evidence hardening, not the first correctness blocker.

## A7. Dense and BatchNorm backward

- Gradient parity per site and layer, accumulation ownership, zeroing, aliases,
  FP16 conversion traffic, and launch topology.
- Current status: **green correctness / profiled**. Frozen CPU-derived fixtures
  cover output, residual-stack, hidden, BN affine, and optimizer-visible weight
  gradients in FP32 and FP16 local steps. The 32 residual layers issue the
  expected Tensor Core dW/dX pairs. Reusing each freshly converted `dY` for its
  adjacent dW/dX pair removes exactly 33 `FloatToHalf` launches per step (529 to
  397 instances over four profiled steps) without changing the fixture. Three
  20-step runs measured 35.7139--35.8679 ms/step after this promotion.

## A8. Sparse table gradient

- Exact CPU/CUDA parity including collisions and tail rows.
- Audit algorithmic complexity, memory layout, atomics/reductions, occupancy,
  and NCU bandwidth/stall counters.
- Current status: **green correctness / promoted on SM86**. The former
  owner-write kernel took 17.36 ms, achieved 18.19% occupancy, and executed
  457.3M instructions. The replacement deterministically builds ascending row
  lists for every `(position,value)` in dead BN workspace, then launches
  shared-free gathers which preserve the original FP32 accumulation order.
  NCU measures 8.19 ms, 98.31% occupancy, 370.31 GB/s, and 221.4M instructions.
  Production memcheck reports 0 errors and racecheck reports 0 hazards;
  FP32/FP16 fixed-step regressions pass.
  Three 20-step runs measured 36.4226--37.2828 ms/step, versus 44.0124 ms after
  A3. Auto selection is intentionally limited to SM86; T4/A100 require their
  own measurements.

The [SM86 tile-major traversal audit](2026-08-31-sparse-gradient-tile-major.md)
preserves exact sparse-gradient accumulation, lowers measured DRAM reads from
2.97GB to0.60GB and reduces full-step latency34.2866 ->28.9943ms.

Latest A8 follow-up: the [stable warp-ballot builder audit](2026-08-31-stable-grouped-row-builder.md)
preserves exact ascending row IDs, reduces grouping1.8416 ->0.2281ms and
full-step latency28.9696 ->27.3462ms (+5.94% throughput). Arena and precision
are unchanged; all confirmation active telemetry samples were780MHz (clocks
were not locked). Original p888/SM86 only.

The [ownership/layout follow-up](2026-08-31-sparse-gradient-warp-bins.md)
rejects WarpBins32/64 after exact correctness gates but +1.005%/+0.238% paired
step latency. X2's +0.11% throughput is not promoted either. Production remains
byte-identical to the accepted BN-mask baseline; the candidates are test-only.

## A9. Optimizer and mirrors

- Adam semantics, bias correction, step numbering, FP32 master/m/v, FP16 mirror,
  finite checks, and checkpoint contents. Verify one update against PyTorch.
- Current status: **green for the live step**. CUDA AdamW matches the CPU
  reference including weight decay and bias correction. The local complete-step
  fixture checks the first master-weight and affine updates; the fused weight
  update writes the FP16 mirror in the same kernel and its lifecycle test passes.
  Checkpoint/resume byte identity belongs to the later runner audit, outside this
  in-memory step gate.

## A10. Runtime scheduling

- No steady-state allocation or host readback; explicit stream/event ownership.
- NVTX ranges per stage, CUDA Graph opportunity, CPU launch gaps, overlap, and
  stable clocks/power during measurements.
- Current status: **first unresolved performance gate**. The prepared lifecycle
  performs allocation before timed steps and the fresh timeline has no semantic
  failure, but the step still issues hundreds of conversion/BN kernels and has
  no accepted CUDA Graph or stage-overlap policy. During a 100-step run the
  laptop GPU held 96--99% utilization at only 780 MHz and 80--81 C, producing
  37.1181 ms/step versus the shorter-run 35.7755 ms median. Scheduling and the
  thermal/power envelope must be frozen before attributing smaller deltas.

## A11. Training behavior

- Short real-data run: loss trend, depth-conditioned error, BN health, finite
  parameters, checkpoint/resume, and deterministic replay where promised.
- Final gate: identical beam-search contract against an original checkpoint.

## A12. Throughput accounting

- Report separately: useful FLOPs, issued/padded FLOPs, kernel throughput,
  stage throughput, and end-to-end samples/s.
- Freeze hardware, power/clocks, binary hash, inputs, batch/tail shape, warmup,
  measurement window, Nsight report, and NCU counters.
- Current SM86 snapshot: 53.558 GFLOP of physically issued dense work and
  51.074 GFLOP of logical/useful dense work per train step. At 35.7755 ms this
  is 1.497 issued and 1.428 useful end-to-end TFLOP/s, plus non-dense work not
  represented by those FLOP counts. GEMM-only throughput is about 9.12 TFLOP/s.
  At the observed 780 MHz, the 40-SM FP16 Tensor Core envelope is about 15.97
  TFLOP/s: GEMMs reach roughly 57%, while the full step reaches roughly 9.4%
  because sparse gather, BN, conversions, and optimizer are not dense GEMMs.
