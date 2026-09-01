# Sparse input-gradient FP16 mirror audit

Date: 2026-09-01. Target: original p888, batch 4096, RTX 3070 Laptop /
SM86, CUDA Graph mode. Starting checkpoint:
`ffb3f3177d08bb3f46dae3f2164ecbbc48d8a7d2`.

## Decision

Accepted as an explicit trainer precision policy. The default remains exact
ordered FP32 input-gradient consumption. With `kFp16Mirror`, the input BN apply
kernel publishes one RN-half copy of its authoritative FP32 `dX`, and the sparse
embedding-gradient kernel consumes half2 values while retaining ordered FP32
accumulation and FP32 outputs. Master weights, accumulated gradients, Adam
moments, and optimizer arithmetic remain FP32.

The mirror aliases the prefix of the now-dead input-activation tape in
`operand_a`. It adds no arena bytes and no conversion kernel. Host preflight
requires the exact tape prefix, sufficient active-row capacity, half2 alignment,
and disjointness from every live FP32 tensor, the weight mirror, `operand_b`, the
active-bin map, and training state.

Selection is fail-closed: it requires the existing compact-active, adjacent-two,
packed-u16 sparse path plus sufficient workspace and mirror capacity. Otherwise
the authoritative FP32 consumer remains selected. Unknown public policy values
are rejected. The benchmark default and existing callers therefore retain FP32
semantics unless they opt in.

## Arithmetic contract and tests

`SparseInputGradCompactActiveAdjacent2PackedHalfU16` loads half2, converts each
pair to `float2`, and applies the same row-ID order with explicit `__fadd_rn`
FP32 sums. The exact oracle is a serial sum of values after RN-half rounding,
not a bitwise match to the former FP32-source result.

The CUDA unit covers active-bin ownership, row-count tails (1, 31, 32, 33, and
257), persistent-zero inactive outputs, immutable states/source/map, canaries,
and exact serial RN-half order. The lifecycle and native graph tests cover
unknown-policy rejection, default FP32 selection, explicit FP16 selection, and
graph/eager parity for both policies. Jobs `f1af5e42cf32`, `737500922208`, and
`784f712526c7` passed the final policy build and targeted tests.

Alias preflight is exercised across the activation-tape matrix. It rejects a
shifted mirror, short capacity, null pointer with nonzero capacity, FP32 overlap,
and active-map overlap; the exact tape prefix is accepted. Final sanitizer jobs
`ee27f6f29f12` and `b2a5405aeb4c` passed this suite under memcheck and initcheck.

Frozen benchmark ELF:

| Path | SHA256 |
|---|---|
| `/tmp/mgt-sparse-half-policy-bin` | `23ff68bda5ebb71b73f68c05d62e420080d493367fdcd1cd2aff8201a7a72239` |

The same ELF implements both policies; only the final CLI policy argument differs.

## Convergence gate

`mgt_single_gpu_convergence_probe` runs all 999,978 samples per epoch: 244 full
graph batches plus one 554-row eager tail. It reports the mean loss over the last
16 full batches so the smaller tail does not dominate comparison. Both runs used
the same configured seed and epoch/sample schedule; CUDA reduction scheduling is
not claimed bitwise deterministic.

Jobs `8ee6525f40a1` (FP32) and `aa37c7f599a5` (FP16 mirror) completed ten epochs /
2,450 optimizer steps with finite loss throughout:

| Mean of last 16 full batches | Epoch 1 | Epoch 5 | Epoch 10 |
|---|---:|---:|---:|
| FP32 source | 18.671722 | 15.296594 | 14.800935 |
| FP16 mirror | 18.196780 | 15.249225 | 14.714521 |

At epoch 10 the FP16 value is 0.584% lower, not degraded. This is a short-run
training-parity gate for the original workload, not a claim about final model
quality or a replacement for longer T4 training validation.

## Paired timing

Job `aa5510b38cc2` thermally gated every process to an 80--81 C start, verified
the frozen ELF hash before and after every run, and executed strict ABBAAB. Each
process ran 140 warmup plus 100 timed graph steps. All retained active telemetry
samples were exactly 780 MHz for both policies.

| Policy | Run means, ms | Median, ms | Samples/s at median |
|---|---|---:|---:|
| FP32 source | 20.0402, 19.9504, 19.9464 | 19.9504 | about 205,309 |
| FP16 mirror | 18.3715, 18.3744, 18.3726 | 18.3726 | about 222,941 |

The absolute median delta is **-1.5778 ms/step**. Latency falls by
**7.908613%** and throughput rises by **8.587788%**.

- Paired JSONL SHA256:
  `8e21a5d6ac0738317da344b723d235c49710f525659ce83d234268618224b772`.
- Summary SHA256:
  `bc6d7b29443ce84fda7755500331e2e7a49bc28108ef58027a1c9e7e0d527279`.

An explicit `nvidia-smi -lgc 780,780` attempt was rejected by the current WSL
driver before any run and produced no measurement artifact. The accepted series
uses the established fail-closed thermal protocol: one unique active clock per
variant is required, and the two variants must match.

## Nsight mechanism proof

Nsight Systems graph-node profiles use the same frozen ELF with 100 warmup and
four measured steps. Job `4ac1d94b85e0` passed exact canonical event order,
identical unrelated kernel geometry/resources, and exactly two identity changes
at the same event indices in every step:

1. input BN apply FP32-only -> FP32 plus RN-half publication;
2. sparse FP32 source -> sparse half source.

Every measured step retains 347 kernels, 71 memsets / 156,136 bytes, and zero
copies.

| Mean over graph steps 2--4 | FP32 source, ms | FP16 mirror, ms | Delta, ms |
|---|---:|---:|---:|
| Sparse input gradient | 3.602514 | 1.944063 | -1.658451 |
| Input BN apply | 0.427042 | 0.452817 | +0.025774 |
| Kernel sum | 19.929720 | 18.302010 | -1.627711 |
| Step span | 20.477526 | 18.866473 | -1.611053 |

The FP16 publication cost is paid inside the existing BN node; no node or copy
is added. Baseline SQLite SHA256 is
`092bc82be684157f61b076bc8626e8e9e2945287071ad271a8202d032c13a5ac`;
candidate SHA256 is
`e3d47859d21360135713ff7b5248b5fef9d3ecbf63c835ba33b0c56d095287dc`.
The strict profile JSON SHA256 is
`7cdb5ea5817de536a9ce63183b669ccbf6e12c98ce7625f6919e8943489beffd`.
Nsight perturbs absolute timing; unprofiled ABBAAB owns the published percentage.

Nsight Compute job `7a0b12b1cc3a` isolates one sparse launch per policy at the
same `(864,40,1)` grid and 64-thread block. Analyzer job `f7d62e8a03bb` passed:

| Metric | FP32 source | FP16 mirror |
|---|---:|---:|
| DRAM reads | 46.081408 MB | 21.644672 MB |
| DRAM writes | 18.730880 MB | 18.178304 MB |
| L2 traffic | 2.854393 GB | 1.369729 GB |
| Registers/thread | 40 | 30 |
| Executed instructions | 118,480,520 | 151,566,000 |
| Issue active | 26.369% | 62.660% |
| Replay duration | 3.595840 ms | 1.937600 ms |

The half path executes conversion instructions, but cuts source traffic roughly
in half, lowers register pressure, and more than doubles scheduler issue activity.
NCU replay uses eager launch and uncontrolled clocks/caches; it is mechanism
evidence, not the timing claim. The strict NCU profile JSON SHA256 is
`d1da593c9ab0c6dfc1a0acf262877bc75b4fac908ab125dbfbbc635236333ba9`.

## Sanitizer, regression, and platform gates

All GPU work ran through `mgt-gpu-queue`; job IDs map to retained
`.gpu_queue/logs/<id>.log` files.

| Jobs | Outcome |
|---|---|
| `33cc537b3f39`, `1c8cc759ac20`, `5f2b7ce9f882`, `c61aafa2e6cd` | Isolated sparse FP32/RN-half suite passed memcheck, initcheck, racecheck, and synccheck with zero errors/hazards. |
| `ee27f6f29f12`, `b2a5405aeb4c` | Activation-tape alias suite passed memcheck/initcheck with zero errors. |
| `dd877423fb75`, `f18a9d53d2dc` | Full p888 graph FP16 policy selected the half path and passed memcheck/initcheck with zero errors. |
| `fdd8a4413b75` | Final SM86 CPU/CUDA/NCCL/graph targeted regression passed 37/37. |
| `6e62d4fea008` | Legacy benchmark CLI with no mode/precision arguments selected eager FP32 and passed. |
| `d09dd878582b`, `075e43cacafc`, `fa070bee2b3b`, `ae25cd9a3b67` | Exact `CMAKE_CUDA_ARCHITECTURES=75` build passed 7/7 targeted tests; the SM75-built FP16-policy ELF passed compatibility runtime smoke on the available SM86 device. |

The SM75 gate proves code generation and forward compatibility, not T4 timing.

## Boundary and next target

This checkpoint proves explicit precision semantics, alias lifetime, short-run
training parity, SM86 speed, graph preservation, sanitizer cleanliness, and SM75
code compatibility. It does not claim native T4 performance, two-GPU scaling,
or final convergence.

After this change the largest measured non-sparse standalone kernel is the FP16
mirror Adam update at about 1.403 ms. The next single-GPU audit target is Adam
traffic/register pressure while retaining FP32 master weights and moments.
