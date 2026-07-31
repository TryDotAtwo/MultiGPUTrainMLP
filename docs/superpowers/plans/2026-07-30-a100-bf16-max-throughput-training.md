# 8xA100 BF16 Maximum-Throughput SyncBN Training Implementation Plan

> **For the primary implementation agent:** REQUIRED SUB-SKILL: Use superpowers:executing-plans and implement this plan task-by-task without subagent delegation. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the current 8xA100 diagnostic FP32/TF32 SyncBatchNorm step into the fastest correctness-approved production P888 trainer, with every Tensor-Core-eligible GEMM using explicit BF16 operands and FP32 accumulation.

**Architecture:** Preserve strict FP32 as a separate numerical-oracle mode. Build one beam-search-style prepared BF16 production runner: a sealed, versioned execution profile fixes every backend, algorithm, buffer offset, allocation domain, stream, event edge, collective order, active-row graph, and capacity before launch. Candidate discovery happens only in an explicit tuning process; production loads one exact profile, validates it collectively, allocates exactly its declared static domains, and either runs that exact DAG or exits before step 0. It never changes precision, backend, graph mode, workspace, tile, or collective implementation.

**Tech Stack:** C++20, CUDA C++17, CUDA SM80, `__nv_bfloat16`, cuBLASLt, optional CUTLASS SM80 BF16 TensorOp kernels, CUB/CCCL, NCCL, CUDA Graphs, NVTX, Nsight Systems, Nsight Compute, CMake/CTest, Python 3, SLURM on MEPhI `basis:kaf12`.

**Authoritative hardware/API contracts:**

- [MultiGPUBeamSearch](https://github.com/TryDotAtwo/MultiGPUBeamSearch): structural reference for explicit invariants, budgeted configuration, static scratch, fixed slot/event ownership, repeated graph templates, and fatal capacity violations; training uses its own math and an even stricter sealed-profile boundary.
- [NVIDIA A100 datasheet](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet.pdf): use the dense BF16 Tensor Core peak for denominator; do not report sparse peak unless 2:4 sparsity is actually executed.
- [cuBLAS documentation](https://docs.nvidia.com/cuda/cublas/index.html): explicit `CUDA_R_16BF` inputs with FP32 compute/accumulation; selected algorithms and workspace are checked before warmup.
- [NCCL CUDA Graph guidance](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/cudagraph.html): graph capture is version/topology-gated and rank-uniform.
- [NCCL Device API](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/deviceapi.html): LSA is a late, fail-closed single-node experiment; A100 does not use multimem.
- [Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/2024.3/ProfilingGuide/index.html): kernel counters support utilization claims only after the cluster grants counter access.

## Global Constraints

- Base the implementation on branch `codex/sparse-input-grad-v3` at or after commit `97f510d`.
- This is an A100-only delta plan. It supersedes the FP16/T4 performance choices in `docs/superpowers/plans/2026-07-28-p888-syncbn-hotpath-production.md` only for the 8xA100 line; it does not transfer an A100 winner to 2xT4.
- Target hardware is exactly one node with 8x NVIDIA A100 40 GB, SM80, one process and one CUDA device per rank.
- Use full global batch `100000`, local full batch `12500`, final global batch `99978`, and final rank rows `12498,12498,12497,12497,12497,12497,12497,12497`.
- The optimized production policy is BF16. FP16 is not a candidate in this A100 plan. Strict FP32 remains callable only through a separate explicit oracle executable/mode; it is never a production fallback.
- Every Tensor-Core-eligible forward, dW, dX, and input-table GEMM uses `CUDA_R_16BF` A/B operands and `CUBLAS_COMPUTE_32F` accumulation. Do not rely on a per-call FP32-to-BF16 fast mode.
- FP32 remains authoritative for master parameters, parameter gradients, Adam first/second moments, loss, BatchNorm sums/sumsq, mean, variance, inverse standard deviation, gamma, beta, affine gradients, and running state.
- The dataflow invariant is: FP32 `dX`/upstream/residual accumulation -> BN backward -> BF16 `dZ` -> BF16 GEMM operands -> FP32 dX/dW outputs. The optimized path stores post-BN activations and the reusable `dZ` ring in BF16; it does not claim that residual/upstream FP32 accumulators are BF16. FP32 and compact-BF16 `xhat` are distinct explicit candidate policies, never runtime fallbacks.
- The hot loop performs no allocation, descriptor/algorithm discovery, stream/event/handle creation, synchronous host readback, `cudaDeviceSynchronize`, or `cudaStreamSynchronize`.
- No accepted healthy-path application/custom kernel may contain a global atomic instruction. Closed-library GEMM candidates require zero measured global ATOM/RED instructions for the selected key; NCCL internals are the only explicit exclusion. Error detection uses preallocated per-CTA words plus fixed-order two-pass reduction. A device-detected data/health branch may not index invalid memory or exit before the collective drain; an NCCL transport failure follows coordinated communicator abort instead.
- Do not remove or approximate the 34 forward and 34 backward global SyncBN dependencies. Local/ghost/group BatchNorm, gradient accumulation, fewer BN sites, and changed optimizer cadence are different training algorithms and are outside this plan.
- Eliminate synchronization around the 68 mandatory dependencies: host waits, redundant events, serialized independent dW work, exposed weight all-reduce, repeated conversion, and launch gaps.
- Allocate every workspace, NCCL buffer, graph object, event, stream, cuBLAS/cuBLASLt handle, CUB temporary buffer, and telemetry slot before warmup.
- Unknown policies, unsupported active-row shapes, stale hardware fingerprints, insufficient capacity, nonzero values in padded rows/lanes, invalid categorical values, and collective-order mismatches fail closed.
- Primary performance KPI is max-rank steady-state step wall time and time-to-quality. Secondary KPIs are BF16 Tensor Core utilization, GPU busy time, exposed NCCL time, HBM efficiency, p95 latency, and peak memory.
- Never increase executed FLOPs merely to inflate utilization unless the resulting end-to-end max-rank step is faster.
- A candidate is timed only after it passes correctness on the same source revision and deterministic nonzero snapshot.
- Run cluster compute only through SLURM under `/mnt/pool/6/vokirova/<jobname>` with `sbatch -p kaf12`; never compute on a login node.
- The user enters SSH/SLURM commands in the built-in Codex terminal. The implementing agent prepares exact commands and reads the resulting job IDs/logs; it does not operate an external terminal.
- Do not read, hash, modify, stage, rename, or delete `kaggle/kernel/run_ranks_2xt4.sh~31640`.
- Before each commit, require an empty pre-existing index, stage only the task's listed paths, run `git diff --cached --check`, and inspect `git status --short`.
- Push only green commits to the current scoped branch; never force-push or rewrite history.
## Beam-Style Exact Execution Contract

This plan adopts the useful control-plane properties of MultiGPUBeamSearch: exhaustive invariants, a budgeted configuration build, one statically planned device arena, fixed slot/event ownership, CUDA Graph templates only for repeated jobs, and fatal capacity/contract violations. It does not copy unrelated beam-search algorithms or its host-history storage behavior.

### Separate tuner, oracle, and production entry points

There are three explicit modes; none may call another as recovery:

~~~text
mgt_a100_bf16_tuner_runner
    enumerates named candidates, runs correctness/performance gates, and writes
    candidate ledger + one accepted execution profile

mgt_a100_bf16_production_runner --execution-profile accepted_profile.json
    accepts no auto/backend/tile/workspace defaults, performs no discovery or timing,
    and runs exactly the sealed profile

mgt_strict_fp32_oracle_runner
    produces reference fixtures/trajectories for gates and is never entered by
    the BF16 production process
~~~

A rejected tuner candidate is evidence, not a runtime fallback. The accepted choice can explicitly name host NCCL, Device LSA, eager, graph, gather, tiled GEMM, FP32 xhat, compact-BF16 xhat, cuBLASLt, or CUTLASS. Once sealed, that exact choice is mandatory. A production launcher has no auto, try, or prefer enum value and no retune flag.

### Sealed profile and canonical identity

Create a versioned P888A100ExecutionProfileV1 containing:

~~~text
schema/version/endian and canonical serialization version
source SHA, clean-tree/build recipe, binary SHA256, compiler and flags
CUDA driver/runtime, cuBLASLt, NCCL, CUB/CCCL, CUTLASS commit
all eight GPU UUIDs, SM, memory, clocks policy, topology/P2P matrix
model/layout/checkpoint version and full/tail row catalogs
complete A100Bf16Policy bytes and algorithm-table SHA256
static-arena layout SHA256 and required bytes/alignment
stream IDs, priority, handle ownership, event IDs, slot counts
complete writer/reader/event DAG and workspace lifetime classes
ordered communicator IDs, collective sequence, count/type/op for every site
full/tail graph-capture recipe and canonical node/edge manifest hashes plus capability identity
snapshot/gate hashes and acceptance report hash
~~~

Canonical serialization is deterministic and endian-explicit; unknown fields, missing fields, duplicate keys, unknown enum values, and schema drift are fatal. Rank 0 does not dictate a profile opportunistically: every rank loads the same bytes, computes the same SHA256, and all-gathers the hash before any runtime resource is created. A mismatch exits collectively.

### One static memory plan with fixed allocation domains

`BuildA100StaticArenaPlan` must return a typed `A100StaticArenaPlanV1`, not merely a byte total. It declares exactly one ordinary CUDA device-allocation domain, exactly one pinned-host telemetry domain, and, only when the sealed backend is Device LSA, exactly one separately sized NCCL symmetric device domain required by `ncclMemAlloc`. Every overflow-checked, 256-byte-aligned device slice and cache-line-aligned host slice records domain, name, dtype, offset, bytes, capacity, owner stream, first writer event, last reader event, and lifetime class. Slices cover persistent FP32 model/BN/optimizer state, gradients, the BF16 weight mirror, fixed batch slots, startup self-test shadow state, activations, BF16 dZ ring, FP32 dX/residual scratch, FP32 or BF16 xhat, ReLU masks, GEMM workspaces, split-K partials, BN partials/results, optional symmetric LSA stats/health, weight-reduction tiles, device/pinned-host telemetry rings, fatal-health words, and graph control storage.

Use beam-style lifetime overlays only within the same allocation domain and only when the event DAG proves non-overlap. Each domain size is the maximum of mutually exclusive phase layouts plus persistent ranges; the startup shadow may overlay steady-state-only scratch after the startup transaction completes. Allocate the declared domains once during preparation and derive every pointer from checked typed offsets; no side allocation is permitted. The steady-state step contains no CUDA/NCCL allocation/free, host allocation, container growth, lazy descriptor construction, or workspace resizing. Capacity overflow sets a preallocated fatal record and drains the already-scheduled collective sequence; it never truncates, resizes, or switches implementation.

### Startup transaction: exact backend or no run

Production performs this transaction before it can emit a training checkpoint or enqueue step 0:

1. Parse and canonicalize the profile; validate every local arithmetic, layout, alignment, capacity, dtype, and active-row invariant.
2. All-gather profile, source, binary, model, snapshot, and library hashes; require rank-uniform equality.
3. Verify exactly 8xA100 SM80 40 GB, GPU UUID ordering, topology, peer access, driver/runtime/library versions, and communicator rendezvous identities.
4. Reconstruct each cuBLASLt algorithm from public algorithm ID plus recorded attributes, then run cublasLtMatmulAlgoCheck; validate each CUTLASS/custom kernel version and split-K layout. Missing or incompatible entries are fatal.
5. Validate the selected collective backend. If the profile says Device LSA, require its exact NCCL ABI, symmetric-memory, P2P, barrier, and graph capabilities. If it says host NCCL, do not probe LSA as an alternative.
6. Allocate exactly the sealed ordinary domain and, if selected, the sealed symmetric domain; create fixed streams, handles, communicators, events, plans, and per-rank full/tail graph execs.
7. Run a deterministic nonzero prepared-step self-test into scratch state, verify guard regions, collectives, finite values, checksums, and error words on all ranks, then restore the sealed initial state.
8. Execute a final all-rank readiness barrier. Any rank failure causes a coordinated nonzero exit with the same stage/error code; no training step is launched.

The error artifact records profile SHA256, rank, startup stage, stable error code, failing key/slice/site, CUDA/cuBLASLt/NCCL status, and peer rank when applicable. It contains no substitute policy.

### No-fallback matrix

| Profile contract failure | Production response |
| --- | --- |
| BF16 algorithm key missing or AlgoCheck fails | Exit before warmup; never invoke a heuristic/default GEMM |
| CUTLASS commit/kernel version differs | Exit; never use cuBLASLt in its place |
| Selected graph capture/instantiate/replay unsupported | Exit; eager is valid only in a separately sealed eager profile |
| Selected Device LSA unsupported or peer/window/barrier check fails | Exit; host NCCL is valid only in a separately sealed host-NCCL profile |
| Active rows absent from the sealed catalog | Exit; never pad to another accepted logical shape |
| Arena/workspace/symmetric buffer too small or misaligned | Exit; never allocate a side buffer or shrink a tile |
| Collective sequence/count/type/hash differs across ranks | Exit collectively before enqueue; if healthy communicators detect a device data fault mid-step, drain neutral payloads; an NCCL transport error triggers coordinated abort |
| Nonfinite/invalid category/guard corruption | Record fatal state, drain the fixed collective suffix with neutral payloads, skip updates, then exit |
| Profiler counters unavailable | Mark utilization evidence blocked; wall-time runs may continue but no percentage is invented |

### Dispatcher, slots, and graph sessions

One host dispatcher owns the fixed training state machine. Each slice has one writer, enumerated readers, one ready event, and one reuse event. The BF16 dZ ring, split-K partial slots, batch buffers, telemetry slots, and graph control blocks use monotonically increasing sequence numbers so ABA reuse is detectable. The dispatcher may query completed telemetry outside the critical path, but it never makes a data-dependent backend decision.

Each rank owns one persistent, fixed-address `P888StepControlV1` in the ordinary arena. A single-writer, non-atomic graph-entry kernel validates the committed cursor, materializes the in-flight sequence/optimizer step, derives the double-buffer batch slot, checks the expected full/tail graph kind, and publishes immutable per-replay scalars to later kernels. A graph-exit commit kernel advances the committed cursor only after successful update/health joins; a drained faulty step leaves it unchanged. Generation writes fixed arena batch slots and records `batch_ready[slot,sequence]`; custom input/loss kernels resolve the selected slot through the control block, then every cuBLASLt/CUTLASS call uses fixed internal addresses. Production never patches graph node parameters, changes cuBLAS pointers, or calls `cudaGraphExec*SetParams` in the hot loop.

The graph always writes a sequence-tagged record into a fixed device telemetry ring. Outside the critical DAG, the telemetry stream copies completed contiguous record batches asynchronously into the fixed pinned-host ring after nonblocking event queries. Slow logging may report a sequence gap/drop counter but may never delay slot reuse, graph replay, or optimizer progress. `publish_metrics_record` is a host-consumer choice only and is not copied into the device control block.

CUDA Graphs capture repeated prepared-step templates, while checkpointing, JSON emission, terminal health reporting, and epoch/batch dispatch remain outside. NCCL graph capture is collective: create exactly two world-8 capture sessions, one with all ranks at 12500 rows and one tail session where ranks 0-1 capture 12498 while ranks 2-7 simultaneously capture 12497. Each rank stores only its own full and tail graph exec. Independent per-shape collective captures are forbidden.


## Execution Strategy at a Glance

| Phase | What changes | Hard gate before continuing |
| --- | --- | --- |
| 1. Freeze truth | One strict prepared step, rank-independent nonzero snapshot, max-rank full/tail baseline | Same mathematics and trustworthy paired timings |
| 2. Build the beam-style runtime | Sealed profile, one typed static arena, fixed streams/events/communicators, startup transaction | Zero hot allocation/sync/discovery; contract mismatch is fatal |
| 3. Move arithmetic to BF16 Tensor Cores | Persistent BF16 weight mirror, BF16 activations/dZ, exact cuBLASLt/CUTLASS table, BF16 input GEMMs | Rounded-BF16 operand oracle passes for every role/key |
| 4. Remove bandwidth and launch waste | Fused casts/ReLU mask, row-tiled SyncBN, no healthy-path atomics/clears/full passes | All 68 global SyncBN dependencies and numerics unchanged |
| 5. Hide unavoidable communication | Concurrent dX/dW, protected dZ/split-K slots, tail and tiled reductions after BN critical path | Complete range coverage, fixed NCCL order, 10k-step no-deadlock soak |
| 6. Reduce host overhead | Exactly two collective graph sessions and fixed dispatcher replay | Graph candidate beats named eager candidate and passes checksum/stress gates |
| 7. Seal and train | Automatic tuner chooses one complete profile; production loads only that profile | Full/tail acceptance, NCU evidence, convergence, resume, full train, puzzle 0 |

Performance priorities are ordered by wall-time leverage, not novelty:

1. Put every eligible forward/dW/dX/input-table GEMM on explicit BF16 Tensor Core operands with FP32 accumulation.
2. Make the long input gradient and repeated residual GEMMs large/parallel enough to occupy SM80 Tensor Cores; use deterministic split-K only when end-to-end timing wins.
3. Keep the 68 exact SyncBN barriers but make local work row-coalesced and overlap all independent dW/reduction work without delaying the next BN collective.
4. Eliminate CPU launch bubbles, ordinary-step synchronizations, contended atomics, repeated conversion/materialization, and dynamic memory.
5. Judge utilization per exact kernel key and over the union of active intervals. Eight A100s provide a theoretical dense BF16 aggregate peak of 2.496 PFLOP/s, but the acceptance target is minimum max-rank step time; non-GEMM SyncBN, NCCL, elementwise work, and Adam are not falsely counted as Tensor Core work.

The last observed 42.6166 ms step remains provisional until Phase 1. Milestones are <=28 ms for the first complete BF16 path, 20-24 ms for production, and 15-20 ms only if measured fusion/graphs/LSA pass every gate.

## Measured Starting Point

| Item | Current result | Planning consequence |
| --- | ---: | --- |
| 8xA100 full step | last observed `42.6166 ms` | Provisional reference; Task 1 must remeasure/freeze the shared-step baseline |
| Throughput | `2,346,505 samples/s` | Use max-rank wall time as source of truth |
| Forward | `8.234 ms` | Includes input gather, 34 GEMMs/BN sites, and BN collectives |
| Residual backward | `13.048 ms` | Contains narrow GEMMs, BN, and residual work |
| Input backward | `19.942 ms` | Contains input BN, sparse table gradient, bias sum, and 50.63 MiB SUM |
| NCCL GPU time | about `26.3%` | Mostly tiny-message latency and serialized dependencies |
| BN collective count | `68` | Mathematically mandatory for exact global SyncBN |
| Total collective count | `73` | Five non-BN reductions can be rescheduled |
| Fused-feature candidate | `58.6362 ms`, `-27.3%` | Never reuse one-block-per-feature row loops |
| NCU status | `ERR_NVGPUCTRPERM` | No Tensor Core utilization claim until counters are enabled |

The current `scripts/estimate_train_flops.py` credits the sparse input gradient with a dense one-hot GEMM. That is a useful-equivalent count, not issued hardware work. Every report created by this plan must carry three separate fields:

```text
useful_model_flops
backend_issued_flops
tensor_core_eligible_flops
```


## Cross-Cutting Promotion Gates

Every optimization follows the same order: operator correctness, full prepared-step correctness, distributed/tail correctness, paired wall-time measurement, then profiler confirmation. A faster candidate that skips a gate is rejected.

Before any performance promotion, require:

```text
nonzero deterministic operator fixtures
all 34 forward BN sites and all 34 backward BN sites checked
forward outputs, dW, dX, dgamma, dbeta, bias grads checked by tensor class
running mean/variance checked per BN site
FP32 master, Adam m/v, and raw BF16 mirror checked after update
world-8 full 100000 and uneven final 99978 checked
world-1 logical oracle versus concatenated world-8 result where semantics permit
1-step, 100-step, and 1000-step trajectories
same-policy repeatability and zero NaN/Inf
```

Report max-absolute, relative L2, cosine, norm ratio, and nonfinite count separately for each tensor class; never hide a bad layer in one aggregate vector. A BF16 candidate that fails is rejected. Only an independently named policy already enumerated by this plan (for example FP32-xhat storage) may be measured, and it must still keep every Tensor-Core-eligible GEMM operand BF16. Do not weaken the strict FP32/global-SyncBN oracle.

Performance reports use max-rank complete-DAG wall time. Nsight stage ranges may overlap and therefore are descriptive, not additive. A100 peak percentages use backend-issued, Tensor-Core-eligible FLOPs only; useful sparse-equivalent FLOPs and the custom scalar head are reported separately.
## Target Dataflow

```text
FP32 master weights
    |
    +-- Adam update in FP32 --> persistent BF16 weight mirror
                                   |
states --> BF16 input formulation -+--> BF16 Tensor Core GEMM, FP32 C
                                                    |
                    FP32 local moments, FP32 NCCL SUM, FP32 finalize
                                                    |
                     fused affine + ReLU/residual + BF16 activation
                                                    |
                                    next BF16 Tensor Core GEMM

backward:
FP32 upstream/residual accumulator
    -> fused activation gate + FP32 BN partial statistics
    -> FP32 NCCL SUM
    -> FP32 BN dZ formula + one BF16 dZ cast
    -> concurrent BF16 Tensor Core dX and dW, FP32 outputs
    -> overlapped FP32 weight-gradient reductions
    -> FP32 Adam + BF16 mirror update
```

The BF16 path deliberately keeps GEMM C/D in FP32 where BatchNorm or parameter accumulation consumes it. It casts once in a fused BN epilogue and then reuses that BF16 value; it never stages the same tensor through repeated FP32-to-BF16 conversion kernels.

## File Structure

### New core policy and runtime files

- `native/include/mgt/a100_bf16_policy.hpp`: portable policy enums, validation, canonical fingerprint fields.
- `native/src/a100_bf16_policy.cpp`: fail-closed policy validation and serialization helpers.
- `native/include/mgt/a100_execution_profile.hpp`: candidate versus accepted profile types, canonical serialization, and sealed identity.
- `native/src/a100_execution_profile.cpp`: exact profile validation, cross-rank hash inputs, and stable startup error codes.
- `native/cuda/mgt_cuda/a100_static_arena.cuh` and `native/cuda/a100_static_arena.cu`: beam-style typed device/pinned-host/symmetric domains, checked offsets, lifetime overlays, and guard regions.
- `native/cuda/mgt_cuda/p888_step_control.cuh`: fixed-address committed/in-flight cursor, batch-slot, graph-kind, and fatal-health contract.
- `native/cuda/mgt_cuda/a100_bf16_runtime.cuh`: opaque prepared runtime and step interfaces.
- `native/cuda/a100_bf16_runtime.cu`: ownership of streams, handles, communicators, workspaces, events, and graph executables.
- `native/include/mgt/distributed_batch.hpp` and `native/src/distributed_batch.cpp`: the only balanced global-batch partition implementation.
- `native/include/mgt/benchmark_snapshot.hpp` and `native/src/benchmark_snapshot.cpp`: rank-partition-independent nonzero parameter/data snapshots and canonical hashes.
- `native/cuda/mgt_cuda/prepared_p888_train_step.cuh` and `native/cuda/prepared_p888_train_step.cu`: the single step API shared by benchmark, production trainer, and graph wrapper.
- `native/include/mgt/a100_bf16_algorithm_table.hpp` and `native/src/a100_bf16_algorithm_table.cpp`: canonical, hashable cuBLASLt/CUTLASS algorithm selections; accepted runs never benchmark algorithms at startup.

The benchmark is not a second trainer. It may prepare inputs and timing records, but it must call the same `LaunchPreparedP888TrainStep` entry point used by production. Directly assembling forward/backward/NCCL/Adam in a benchmark is a stop-the-line violation.

Every sample is generated from `(seed, global_sample_id, position)` and every parameter/state element from `(seed, logical_parameter_index)`. No snapshot value may depend on rank or world size.

### New BF16 primitives

- `native/cuda/mgt_cuda/bf16_linear_train_ops.cuh`
- `native/cuda/bf16_linear_train_ops.cu`
- `native/cuda/mgt_cuda/input_embedding_bf16.cuh`
- `native/cuda/input_embedding_bf16.cu`
- `native/cuda/mgt_cuda/sync_batch_norm_tiled.cuh`
- `native/cuda/sync_batch_norm_tiled.cu`
- `native/cuda/mgt_cuda/bf16_gemm_bn.cuh`
- `native/cuda/bf16_gemm_bn.cu`
- `scripts/cutlass_sm80_bf16.lock`: the exact tested CUTLASS commit; no floating `main` checkout is accepted.

### New communication, benchmarking, and autotuning files

- `native/include/mgt/bn_communication_plan.hpp`
- `native/src/bn_communication_plan.cpp`
- `scripts/run_cuda_multirank_test.py`
- `scripts/summarize_a100_bf16_benchmark.py`
- `scripts/autotune_8xa100_bf16.py`
- `scripts/cluster/run_a100_bf16_gates.sbatch`
- `scripts/cluster/run_a100_bf16_autotune.sbatch`
- `scripts/cluster/run_a100_bf16_nsys.sbatch`
- `scripts/cluster/run_a100_bf16_ncu.sbatch`

The existing `native/cuda/mlp_backward.cu` remains the source for proven one-hot/cuBLASLt logic during extraction. The new production runtime must not retain its lazy descriptor cache, event creation, or in-launch autotuning.

---

### Task 1: Establish One Shared Strict Step and a Trustworthy 8xA100 Benchmark

**Files:**

- Modify: `native/tools/mgt_bn_step_benchmark.cu`
- Create: `native/tools/mgt_strict_fp32_oracle_runner.cu`
- Create: `native/include/mgt/distributed_batch.hpp`
- Create: `native/src/distributed_batch.cpp`
- Create: `native/tests/test_distributed_batch.cpp`
- Create: `native/include/mgt/benchmark_snapshot.hpp`
- Create: `native/src/benchmark_snapshot.cpp`
- Create: `native/tests/test_benchmark_snapshot.cpp`
- Create: `native/cuda/mgt_cuda/prepared_p888_train_step.cuh`
- Create: `native/cuda/prepared_p888_train_step.cu`
- Create: `native/tests/cuda/test_prepared_p888_train_step.cu`
- Create: `scripts/run_cuda_multirank_test.py`
- Create: `scripts/summarize_a100_bf16_benchmark.py`
- Create: `scripts/tests/test_summarize_a100_bf16_benchmark.py`
- Create: `scripts/cluster/run_a100_bf16_gates.sbatch`
- Modify: `native/CMakeLists.txt`

**Interfaces:**

```cpp
struct DistributedBatchSlice {
    std::uint32_t active_rows;
    std::uint64_t global_offset;
};

mgt::Status PartitionGlobalBatch(
    std::uint32_t global_rows,
    std::uint32_t world,
    std::uint32_t rank,
    DistributedBatchSlice* out);

struct PreparedTrainStepRequest {
    std::uint32_t batch_slot;
    std::uint32_t active_rows;
    std::uint32_t global_rows;
    std::uint64_t global_offset;
    std::uint64_t optimizer_step;
    bool publish_metrics_record;  // never changes the collective DAG
};

struct PreparedTrainStepTicket {
    cudaEvent_t completion_event;
    std::uint64_t sequence;
};

struct PreparedP888StrictRuntimeCreateInfo {
    CudaMlpShape shape;
    std::uint32_t capacity_rows;
    const std::uint32_t* supported_active_rows;
    std::uint32_t supported_active_row_count;
    std::uint32_t device_id;
    std::uint32_t rank;
    std::uint32_t world;
    const char* strict_nccl_id_file;
    MlpBatchNormStepBuffers buffers;
    mgt::BatchNormTrainingPlan batch_norm_plan;
    AdamWKernelConfig adam;
    std::array<const mgt::TrainStateStorage*, 2> state_slots;
    std::array<const float*, 2> label_slots;
};

struct PreparedP888TrainRuntime;

mgt::Status QueryPreparedP888StrictRuntimeBytes(
    const PreparedP888StrictRuntimeCreateInfo& info,
    std::uint64_t* bytes);
mgt::Status CreatePreparedP888StrictRuntime(
    const PreparedP888StrictRuntimeCreateInfo& info,
    PreparedP888TrainRuntime** out);
mgt::Status DestroyPreparedP888TrainRuntime(
    PreparedP888TrainRuntime* runtime);


mgt::Status LaunchPreparedP888TrainStep(
    PreparedP888TrainRuntime* runtime,
    const PreparedTrainStepRequest& request,
    PreparedTrainStepTicket* ticket);
```

- Produces benchmark JSON schema `mgt_a100_train_step_v1`.
- Produces deterministic nonzero full/tail snapshots shared by every later A/B test.
- Produces max-rank repeat-average timing for acceptance and a separate max-rank per-step diagnostic stream for p95; rank-0-only timing is never accepted.

- [ ] Extract the current strict global-SyncBN forward/backward/Adam sequence into `PreparedP888TrainRuntime` and `LaunchPreparedP888TrainStep` without changing its mathematics. Implement and test `QueryPreparedP888StrictRuntimeBytes`, `CreatePreparedP888StrictRuntime`, and `DestroyPreparedP888TrainRuntime`; bind both state/label slots, all model/optimizer buffers, communicator, supported-row catalog, descriptors, workspaces, streams, and events during creation. The step request carries only slot/cursor metadata and never supplies a launch-time data pointer. Keep the old launch order covered by `mlp_batch_norm_full_backward`; make `mgt_bn_step_benchmark.cu` a caller of the shared entry point, not an owner of the sequence.

- [ ] Build `mgt_strict_fp32_oracle_runner` as an explicit executable over the same prepared-step API. It accepts only the strict runtime contract, emits reference fixtures/trajectories, and is not linked or callable from `mgt_a100_bf16_production_runner`.

- [ ] Replace the all-zero benchmark state with a deterministic host snapshot generated from SplitMix64 seed `888`:

  ```cpp
  std::uint64_t SplitMix64(std::uint64_t x);
  float SignedUnit(std::uint64_t x);
  void FillBenchmarkSnapshot(
      std::uint64_t seed,
      std::uint64_t global_sample_offset,
      std::uint32_t active_rows,
      const CudaMlpShape& shape,
      BenchmarkMutableState* mutable_state,
      std::vector<mgt::TrainStateStorage>* states,
      std::vector<float>* labels);
  ```

  Each sample value is a pure function of `(seed, global_sample_offset + row, position)` and each parameter, BN state, Adam state, and BF16 mirror element is a pure function of `(seed, logical_parameter_index)`. Parameters are identical on all ranks; repartitioning the same global batch across 1/2/4/8 ranks yields byte-identical concatenated states/labels and one canonical `snapshot_sha256`.

- [ ] Add explicit options:

  ```text
  --device
  --rank
  --world
  --nccl-id-file
  --expected-local-rows
  --global-rows
  --seed
  --warmup
  --steps
  --repeat
  --mode strict-syncbn
  --output-jsonl
  --profile-stages
  ```

  Reject unknown/duplicate options, zero steps, and `rank>=world`. Derive `active_rows` and `global_offset` only through `PartitionGlobalBatch`; `--expected-local-rows` is optional and may only assert the derived value.

- [ ] Unit-test balanced partitions for strong-scaling diagnostics and use the world-8 rows for production acceptance:

  ```text
  world1 full/final: [100000] / [99978]
  world2 full/final: [50000,50000] / [49989,49989]
  world4 full:       [25000,25000,25000,25000]
  world4 final:      [24995,24995,24994,24994]
  world8 full:       [12500,12500,12500,12500,12500,12500,12500,12500]
  world8 final:      [12498,12498,12497,12497,12497,12497,12497,12497]
  ```

- [ ] Primary throughput mode records one event pair around each uninstrumented repeated region and reports `region_ms / measured_steps`; it never calls this a per-step distribution. The region-stop event waits on the final `PreparedTrainStepTicket`, which represents the complete multistream/NCCL/Adam DAG. A separate `--profile-step-samples` invocation uses a precreated start/completion event pair per ticket, performs one host wait only after enqueueing the full region, and reports max-rank step samples for diagnostic p95. Stage profiling is a third separate invocation; overlapping stage durations are never added into a fake total.

- [ ] All ranks emit:

  ```json
  {
    "schema": "mgt_a100_train_step_v1",
    "rank": 0,
    "world": 8,
    "active_rows": 12500,
    "global_rows": 100000,
    "case": "full",
    "row_vector": [12500,12500,12500,12500,12500,12500,12500,12500],
    "pair_index": 0,
    "pair_attempt_nonce": "fresh-process-pair-id",
    "pair_role": "baseline",
    "pair_order": 0,
    "snapshot_sha256": "64-lowercase-hex",
    "runtime_tree_sha256": "64-lowercase-hex",
    "acceptance_eligible": true,
    "seed": 888,
    "source_sha": "40-or-64-lowercase-hex",
    "policy_sha256": "sha256-of-strict-fp32-syncbn",
    "timing_mode": "region_average",
    "repeat": 0,
    "measured_steps": 100,
    "region_ms": 4200.0,
    "avg_step_ms": 42.0,
    "stage_ms": {},
    "collective_counts": {"bn": 68, "weight": 4, "loss": 1},
    "workspace_bytes": 0,
    "status": "ok"
  }
  ```

- [ ] `summarize_a100_bf16_benchmark.py` joins region rows only when source, policy, snapshot, runtime tree, case, row vector, pair nonce, pair index, and repeat agree. It divides each rank region by the same measured-step count, takes the maximum rank average for that repeat, and reports Q50 across repeats. It separately joins instrumented rows by step index, takes max-rank duration per step, and reports diagnostic Q95. It labels Q95 unavailable rather than deriving it from region averages.

- [ ] Add Python tests for missing rank, duplicate rank, unequal sample count, source/policy mismatch, max-rank selection, Q50/Q95, full/tail mixing, malformed JSON, and nonfinite duration.

- [ ] Each A/B side starts in a fresh 8-rank process group, reloads the identical canonical snapshot, and runs adjacent to its mate with alternating order. A candidate can be promoted only against a source-identical baseline; the frozen Task-1 result is historical context, not a permanent comparison control.

- [ ] Add NVTX ranges:

  ```text
  step
  forward
  output_loss
  residual_backward
  hidden_backward
  input_backward
  weight_reduction
  adam
  ```

- [ ] `run_a100_bf16_gates.sbatch` must use:

  ```text
  --partition=kaf12
  --nodes=1
  --ntasks-per-node=8
  --cpus-per-task=4
  --gres=gpu:8
  --mem=128G
  --time=00:30:00
  ```

  It verifies `MGT_EXPECTED_SHA`, configures `Release` with `CMAKE_CUDA_ARCHITECTURES=80`, builds named targets, runs CTest, then uses `srun --ntasks=8 --kill-on-bad-exit=1`.

- [ ] Run CPU tests:

  ```powershell
  py -m unittest scripts.tests.test_summarize_a100_bf16_benchmark -v
  git diff --check
  ```

- [ ] After the user submits the cluster gate, require seven interleaved strict region-average repeats plus one instrumented p95 run for full and tail. Freeze both baseline artifacts before Task 2.

- [ ] Commit:

  ```powershell
  git add native/include/mgt/distributed_batch.hpp native/src/distributed_batch.cpp native/tests/test_distributed_batch.cpp native/include/mgt/benchmark_snapshot.hpp native/src/benchmark_snapshot.cpp native/tests/test_benchmark_snapshot.cpp native/cuda/mgt_cuda/prepared_p888_train_step.cuh native/cuda/prepared_p888_train_step.cu native/tests/cuda/test_prepared_p888_train_step.cu native/tools/mgt_bn_step_benchmark.cu native/tools/mgt_strict_fp32_oracle_runner.cu scripts/run_cuda_multirank_test.py scripts/summarize_a100_bf16_benchmark.py scripts/tests/test_summarize_a100_bf16_benchmark.py scripts/cluster/run_a100_bf16_gates.sbatch native/CMakeLists.txt
  git commit -m "test: harden 8xa100 syncbn benchmark"
  ```

---

### Task 2: Define the Explicit BF16 Policy and Resource Owner

**Files:**

- Create: `native/include/mgt/a100_bf16_policy.hpp`
- Create: `native/src/a100_bf16_policy.cpp`
- Create: `native/tests/test_a100_bf16_policy.cpp`
- Create: `native/include/mgt/a100_execution_profile.hpp`
- Create: `native/src/a100_execution_profile.cpp`
- Create: `native/tests/test_a100_execution_profile.cpp`
- Create: `native/cuda/mgt_cuda/a100_static_arena.cuh`
- Create: `native/cuda/a100_static_arena.cu`
- Create: `native/tests/cuda/test_a100_static_arena.cu`
- Create: `native/cuda/mgt_cuda/a100_bf16_runtime.cuh`
- Create: `native/cuda/p888_step_control.cu`
- Create: `native/tests/cuda/test_p888_step_control.cu`
- Create: `native/cuda/mgt_cuda/p888_step_control.cuh`
- Create: `native/cuda/a100_bf16_runtime.cu`
- Create: `native/tests/cuda/test_a100_bf16_runtime.cu`
- Modify: `native/cuda/mgt_cuda/prepared_p888_train_step.cuh`
- Modify: `native/cuda/prepared_p888_train_step.cu`
- Modify: `native/tools/mgt_bn_step_benchmark.cu`
- Modify: `native/CMakeLists.txt`

**Interfaces:**

```cpp
enum class A100LinearBackend : std::uint32_t {
    kCublasLtBf16 = 1,
    kCutlassBf16 = 2,
};

enum class A100InputForwardBackend : std::uint32_t {
    kGatherFromBf16Table = 1,
    kPositionTiledBf16Gemm = 2,
};

enum class A100InputGradBackend : std::uint32_t {
    kPositionTiledBf16Gemm = 1,
};

enum class A100BatchNormBackend : std::uint32_t {
    kStridedFp32Reference = 1,
    kRowTiledFp32StatsBf16Output = 2,
    kCutlassPartialMomentsBf16 = 3,
};

enum class A100WeightReduceBackend : std::uint32_t {
    kSectionedReference = 1,
    kTailPlusInputTilesOverlap = 2,
};

enum class A100GraphBackend : std::uint32_t {
    kDisabled = 1,
    kComputeAndBn = 2,
    kFullMultistream = 3,
};

enum class A100XhatStorage : std::uint32_t {
    kFp32 = 1,
    kBf16 = 2,
};

enum class A100DwDxSchedule : std::uint32_t {
    kSerial = 1,
    kConcurrentProtected = 2,
};

enum class A100InputGradReduction : std::uint32_t {
    kOneContiguousSum = 1,
    kPositionTiledSums = 2,
};

enum class A100InputTileMaterialization : std::uint32_t {
    kExplicitBf16OneHot = 1,
    kImplicitCutlassIterator = 2,
};

enum class A100BnCollectiveBackend : std::uint32_t {
    kNcclHostAllReduce = 1,
    kNcclDeviceLsa = 2,
};

struct A100Bf16Policy {
    std::uint32_t schema_version;
    A100LinearBackend linear;
    A100InputForwardBackend input_forward;
    A100InputGradBackend input_grad;
    A100BatchNormBackend batch_norm;
    A100WeightReduceBackend weight_reduce;
    A100GraphBackend graph;
    A100XhatStorage xhat_storage;
    A100DwDxSchedule dw_dx_schedule;
    A100InputGradReduction input_grad_reduction;
    A100InputTileMaterialization input_tile_materialization;
    A100BnCollectiveBackend bn_collective;
    std::uint32_t input_positions_per_tile;
    std::uint32_t bn_row_chunk;
    std::uint32_t bn_feature_tile;
    std::uint32_t dz_ring_slots;
    std::uint32_t padded_rows_multiple;
    std::uint64_t lt_workspace_bytes;
    std::array<std::uint8_t, 32> algorithm_table_sha256;
};
```

```cpp
enum class A100ExecutionProfileUse : std::uint32_t {
    kCandidateForTuner = 1,
    kAcceptedForProduction = 2,
};

struct P888A100ExecutionProfileV1;
enum class A100LocalGraphSlot : std::uint32_t {
    kFull = 1,
    kTail = 2,
};

struct P888StepControlV1 {
    std::uint32_t schema_version;
    std::uint32_t graph_slot;
    std::uint32_t active_rows;
    std::uint32_t global_rows;
    std::uint64_t global_offset;
    std::uint64_t committed_sequence;
    std::uint64_t committed_optimizer_step;
    std::uint64_t inflight_sequence;
    std::uint64_t inflight_optimizer_step;
    std::uint64_t semantic_epoch;
    std::uint32_t batch_in_epoch;
    std::uint32_t batch_slot;
    std::uint64_t generation_seed;
    std::uint32_t fatal_health;
};

mgt::Status InitializeP888StepControl(
    P888StepControlV1* control,
    std::uint64_t initial_sequence,
    std::uint64_t initial_optimizer_step,
    std::uint64_t generation_seed);

mgt::Status LaunchBeginP888StepControl(
    P888StepControlV1* control,
    A100LocalGraphSlot slot,
    cudaStream_t stream);
mgt::Status LaunchCommitP888StepControl(
    P888StepControlV1* control,
    cudaStream_t stream);

struct A100Bf16RuntimeCreateInfo {
    CudaMlpShape shape;
    std::uint32_t logical_hd1;
    std::uint32_t logical_hd2;
    std::uint32_t capacity_rows;
    const std::uint32_t* supported_active_rows;
    std::uint32_t supported_active_row_count;
    std::uint32_t device_id;
    std::uint32_t rank;
    std::uint32_t world;
    const char* bn_nccl_id_file;
    const char* weight_nccl_id_file;
    const char* metrics_nccl_id_file;
    const P888A100ExecutionProfileV1* execution_profile;
    A100ExecutionProfileUse profile_use;
};

struct A100StaticArenaPlanV1;
struct A100StaticArenaView {
    void* ordinary_base;
    std::uint64_t ordinary_bytes;
    void* symmetric_base;
    std::uint64_t symmetric_bytes;
    void* pinned_host_base;
    std::uint64_t pinned_host_bytes;
    const A100StaticArenaPlanV1* plan;
};
struct A100Bf16Runtime;

mgt::Status BuildA100StaticArenaPlan(
    const A100Bf16RuntimeCreateInfo& info,
    A100StaticArenaPlanV1* out);
mgt::Status CreateA100Bf16Runtime(
    const A100Bf16RuntimeCreateInfo& info,
    const A100StaticArenaPlanV1& arena,
    A100Bf16Runtime** out);
mgt::Status DestroyA100Bf16Runtime(A100Bf16Runtime* runtime);
```
- [ ] Define the lifecycle explicitly. A candidate profile is complete and executable but carries `profile_state=candidate` and is accepted only by the tuner/benchmark entry point. An accepted profile is a byte-for-byte copy of one passing candidate plus immutable gate/artifact hashes and `profile_state=accepted`; only Task 13 may seal it. The production entry point rejects candidate profiles, while Task 12 uses candidate profiles solely through the tuner harness to exercise the real trainer before sealing.

- [ ] Implement `Initialize/LaunchBegin/LaunchCommitP888StepControl` in this task so every eager task from Task 2 onward already uses the same fixed-address cursor contract that graphs later capture unchanged. Before enqueue, the host validates `PreparedTrainStepRequest` slot/full-tail/cursor metadata against its monotonic shadow. Begin materializes in-flight fields and never advances committed state; commit runs only after update/metrics/global-health joins. Full/full/.../tail order, fault-without-commit, resume, overflow, wrong slot, duplicate launch, and 1000-step wrap/reuse tests must pass without atomics or a host wait.

- [ ] `BuildA100StaticArenaPlan` emits all typed slices and lifetime overlays, then hashes the canonical layout. `CreateA100Bf16Runtime` requires the plan hash to equal the execution-profile hash and allocates exactly the declared ordinary device domain, pinned-host domain, and optional symmetric domain. Production may not edit or regenerate a profile to fit currently free memory.

- [ ] Write failing policy/profile tests for every unknown enum, unsupported schema, missing or zero SHA, zero/oversized tile, unsupported row multiple, non-256-byte Lt workspace, duplicate active row, world outside `{1,2,4,8}`, rank overflow, arithmetic overflow, and non-canonical serialization. Verify that xhat storage, dW/dX schedule, input-gradient reduction, input-tile materialization, BN collective backend, dZ-ring slots, and algorithm-table SHA each change the canonical profile hash. The production launcher separately requires world 8; world 1/2/4 exists only for correctness and scaling diagnostics.

- [ ] Validate optimized A100 policy constraints:

  ```text
  linear = cublasLt BF16 or CUTLASS BF16
  input grad BF16 tile in {8,12,16,24,32,48,56,64,72}
  row chunk in {256,512,1024}
  feature tile in {32,64}
  dZ ring slots in {2,3,4}
  padded rows multiple in {1,16,32}
  Lt workspace in {16,32,64,128,256} MiB
  ```

- [ ] The runtime owns exactly:

  ```text
  compute_stream
  auxiliary_gemm_stream
  weight_stream
  generation_stream
  telemetry_stream
  bn_context
  weight_context
  metrics_context
  compute_blas_lt
  auxiliary_blas_lt
  prepared GEMM/input/BN plans
  all workspaces and CUB temporary storage
  all dependency events
  graph and graph-exec objects
  ```

- [ ] Resource construction occurs in `CreateA100Bf16Runtime`; destruction occurs only after the caller has stopped launching steps. The launch path never calls a create/destroy API.

- [ ] Bind each prepared runtime to its declared active-row catalog and reject every other row count before enqueue. Production world 8 declares `{12500,12498,12497}`. Scaling diagnostics prepare the balanced full/final shapes `{100000,99978,50000,49989,25000,24995,24994,12500,12498,12497}` as capacity permits; an explicit OOM is a diagnostic result, not a silent shape substitution.

- [ ] Require separate NCCL rendezvous files and communicators for critical BN traffic, weight-gradient traffic, and metrics. Enqueue the same scalar metrics SUM every step so graph topology never depends on logging cadence; `publish_metrics_record` controls only whether a completed ring slot is exposed to the CPU. Define one rank-uniform host issue order: all `F0..F33,B33..B0` BN collectives, tail, ascending input tiles, metrics. Fingerprint and validate `NCCL_LAUNCH_ORDER_IMPLICIT`; if concurrent multi-communicator graph execution cannot honor that order, reject the candidate. Log communicator/global issue counters per rank.

- [ ] BF16 benchmark mode replaces the strict `--nccl-id-file` with required `--bn-nccl-id-file`, `--weight-nccl-id-file`, and `--metrics-nccl-id-file`. Reject duplicate paths and never initialize two logical contexts from the same rendezvous bytes.

- [ ] Make `PreparedP888TrainRuntime` own or bind one `A100Bf16Runtime` while preserving the Task-1 launch signature. Add required `--candidate-execution-profile PATH` only to the tuner/benchmark entry point in this task; do not add it to production. Strict-oracle and BF16-candidate runtimes must still reach the same `LaunchPreparedP888TrainStep`; mode-specific low-level launch assembly in the benchmark is forbidden.

- [ ] Validate `PreparedTrainStepRequest.active_rows/global_offset` against `PartitionGlobalBatch` before enqueueing anything. Bind immutable model/optimizer buffers once during runtime preparation; the request contains batch identity and cursor only.

- [ ] Test partial-construction cleanup by injecting failure after every owned resource, repeated create/destroy, wrong current device, duplicate ID paths, insufficient capacity, and successful zero-allocation launch setup.

- [ ] Run:

  ```powershell
  cmake --build native/build-cpu-codex --config Release --target test_a100_bf16_policy test_a100_execution_profile test_p888_step_control
  ctest --test-dir native/build-cpu-codex -C Release -R "^(a100_bf16_policy|a100_execution_profile|p888_step_control)$" --output-on-failure --no-tests=error
  git diff --check
  ```

- [ ] Commit:

  ```powershell
  git add native/include/mgt/a100_bf16_policy.hpp native/src/a100_bf16_policy.cpp native/tests/test_a100_bf16_policy.cpp native/include/mgt/a100_execution_profile.hpp native/src/a100_execution_profile.cpp native/tests/test_a100_execution_profile.cpp native/cuda/mgt_cuda/p888_step_control.cuh native/cuda/p888_step_control.cu native/tests/cuda/test_p888_step_control.cu native/cuda/mgt_cuda/a100_static_arena.cuh native/cuda/a100_static_arena.cu native/tests/cuda/test_a100_static_arena.cu native/cuda/mgt_cuda/a100_bf16_runtime.cuh native/cuda/a100_bf16_runtime.cu native/tests/cuda/test_a100_bf16_runtime.cu native/cuda/mgt_cuda/prepared_p888_train_step.cuh native/cuda/prepared_p888_train_step.cu native/tools/mgt_bn_step_benchmark.cu native/CMakeLists.txt
  git commit -m "feat: add prepared a100 bf16 runtime"
  ```

---

### Task 3: Implement Prepared BF16 Tensor Core Linear Operations

**Files:**

- Create: `native/cuda/mgt_cuda/bf16_linear_train_ops.cuh`
- Create: `native/cuda/bf16_linear_train_ops.cu`
- Create: `native/tests/cuda/test_cuda_bf16_linear_train_ops.cu`
- Create: `native/include/mgt/a100_bf16_algorithm_table.hpp`
- Create: `native/src/a100_bf16_algorithm_table.cpp`
- Create: `native/tests/test_a100_bf16_algorithm_table.cpp`
- Modify: `native/cuda/mgt_cuda/adamw.cuh`
- Modify: `native/cuda/adamw.cu`
- Modify: `native/tests/cuda/test_cuda_adamw_smoke.cu`
- Modify: `scripts/ensure_cutlass.sh`
- Create: `scripts/cutlass_sm80_bf16.lock`
- Modify: `native/CMakeLists.txt`

**Interfaces:**

```cpp
struct Bf16LinearProblem {
    std::uint32_t active_rows;
    std::uint32_t site_id;
    std::uint32_t compute_rows;
    std::uint32_t input_features;
    std::uint32_t output_features;
};
enum class Bf16GemmRole : std::uint32_t {
    kInputForward = 1,
    kHiddenForward = 2,
    kResidualForward = 3,
    kGradWeight = 4,
    kGradInput = 5,
    kInputTableGrad = 6,
};

struct Bf16GemmKeyV1 {
    std::uint32_t schema_version;
    Bf16GemmRole role;
    std::uint32_t site_id;
    std::uint32_t order_a;
    std::uint32_t order_b;
    std::uint32_t order_c;
    std::uint32_t order_d;
    std::uint32_t alignment_a_bytes;
    std::uint32_t alignment_b_bytes;
    std::uint32_t alignment_c_bytes;
    std::uint32_t alignment_d_bytes;
    std::uint32_t batch_count;
    std::uint64_t stride_a_bytes;
    std::uint64_t stride_b_bytes;
    std::uint64_t stride_c_bytes;
    std::uint64_t stride_d_bytes;
    std::uint32_t active_rows;
    std::uint32_t compute_rows;
    std::uint32_t m;
    std::uint32_t n;
    std::uint32_t k;
    std::uint32_t op_a;
    std::uint32_t op_b;
    std::uint64_t lda;
    std::uint64_t ldb;
    std::uint64_t ldc;
    std::uint64_t ldd;
    std::uint32_t a_type;
    std::uint32_t b_type;
    std::uint32_t c_type;
    std::uint32_t d_type;
    std::uint32_t compute_type;
    std::uint32_t epilogue;
    std::uint32_t beta_bits;
};

struct Bf16SplitKContractV1 {
    std::uint32_t split_count;
    std::uint32_t partition_kind;
    std::uint32_t k_granularity;
    std::uint32_t reduction_scheme;
    std::uint32_t finalize_kernel_version;
    std::uint32_t scratch_layout_version;
    std::uint64_t scratch_offset;
    std::uint64_t scratch_bytes;
    std::uint64_t scratch_alignment;
    std::uint32_t slot_count;
};

struct Bf16GemmChoiceV1 {
    std::uint32_t backend;
    std::int32_t cublaslt_algo_id;
    std::uint32_t tile_id;
    std::uint32_t stages_id;
    std::uint32_t split_k;
    std::uint32_t reduction_scheme;
    std::uint32_t cta_swizzle;
    std::uint32_t custom_option;
    std::uint32_t custom_kernel_version;
    std::uint64_t workspace_offset;
    std::uint64_t workspace_bytes;
    std::uint64_t workspace_alignment;
    Bf16SplitKContractV1 split_k_contract;
};

struct Bf16AlgorithmRecordV1 {
    Bf16GemmKeyV1 key;
    Bf16GemmChoiceV1 choice;
};

struct Bf16AlgorithmTable;
mgt::Status LookupBf16GemmChoice(
    const Bf16AlgorithmTable& table,
    const Bf16GemmKeyV1& key,
    const Bf16GemmChoiceV1** out);

struct Bf16LinearTrainOpsPlan;

mgt::Status CreateBf16LinearTrainOpsPlan(
    const Bf16LinearProblem* problems,
    std::uint32_t problem_count,
    std::uint32_t device_id,
    cublasLtHandle_t handle,
    const Bf16AlgorithmTable& algorithms,
    const A100StaticArenaView& arena,
    Bf16LinearTrainOpsPlan** out);

mgt::Status LaunchBf16LinearForwardToFloat(
    const Bf16LinearTrainOpsPlan* plan,
    Bf16LinearProblem problem,
    const __nv_bfloat16* input,
    const __nv_bfloat16* weight,
    float* output,
    cudaStream_t stream);

mgt::Status LaunchBf16LinearGradWeightToFloat(
    const Bf16LinearTrainOpsPlan* plan,
    Bf16LinearProblem problem,
    const __nv_bfloat16* input,
    const __nv_bfloat16* grad_output,
    float* grad_weight,
    cudaStream_t stream);

mgt::Status LaunchBf16LinearGradInputToFloat(
    const Bf16LinearTrainOpsPlan* plan,
    Bf16LinearProblem problem,
    const __nv_bfloat16* grad_output,
    const __nv_bfloat16* weight,
    float* grad_input,
    float beta,
    cudaStream_t stream);
```

- [ ] Create descriptors for explicit `CUDA_R_16BF` A/B, `CUDA_R_32F` C/D, `CUBLAS_COMPUTE_32F`, and FP32 alpha/beta. Require 16-byte pointer alignment and 256-byte workspace alignment.

- [ ] Pin CUTLASS to the exact commit stored in `scripts/cutlass_sm80_bf16.lock`. Refuse an existing checkout at a different commit and include the resolved commit in every profile and benchmark artifact.

- [ ] Implement `Bf16GemmKeyV1` and `Bf16GemmChoiceV1` as the only lookup contract. Serialize canonical little-endian binary and a sorted-key JSON rendering; hash the binary. The key includes role/site, active/compute rows, exact m/n/k, transposes, matrix orders, leading dimensions, guaranteed pointer alignments, batch count/byte strides, all dtypes, compute type, epilogue, and the exact FP32 beta bit pattern. Reject duplicate keys and non-unique lookups. Non-batched production GEMMs explicitly record `batch_count=1` and zero batch strides rather than relying on defaults.

- [ ] Never serialize opaque `cublasLtMatmulAlgo_t` bytes. Store the public algorithm ID and every configured public attribute: tile ID, stages ID, split-K count, reduction scheme, CTA swizzle, and custom option. Production reconstructs with `cublasLtMatmulAlgoInit` plus attribute setters and requires `cublasLtMatmulAlgoCheck` to approve the recorded layout/workspace. No heuristic API is linked into the production target.

- [ ] Reject in-place/atomic split-K reduction schemes. For application and CUTLASS cubins, scan SASS for global `ATOM`/`RED`. For each closed cuBLASLt candidate, collect the exact-key global atomic instruction/counter metrics in a bounded NCU run; a nonzero value makes it ineligible. If required counters are unavailable, the no-atomic proof for that candidate is `blocked`, so it cannot be finally sealed even though wall-time exploration may continue.

- [ ] Provide a separate candidate-only discovery command that queries heuristics and times algorithms before sealing. Build explicit candidate/diagnostic row catalogs for:

  ```text
  production world-8 rows: 12500,12498,12497
  BF16 scaling rows: 100000,99978,50000,49989,25000,24995,24994
  compute rows: exact round_up(active_rows, candidate padded_rows_multiple)
  2560 -> 224 and 224 -> 224
  forward, dW, dX transposed variants
  dX beta bit-patterns +0.0 and +1.0
  output head excluded (custom row dot/reduction; not Tensor-Core eligible)
  ```

  Production seals only the world-8 catalog. Each world-1/2/4 scaling run uses a separate complete diagnostic candidate profile/table; if a row cannot fit or a key cannot be prepared, that diagnostic is `not_runnable`/OOM and exits rather than borrowing another shape or algorithm.

- [ ] Treat residual dW as a separate occupancy problem. Its FP32 output is only `224x224`; a `128x128` kernel exposes four output CTAs and can leave most of an A100 idle even though the arithmetic uses Tensor Cores.

- [ ] For every residual dW key, sweep cuBLASLt workspaces `16/32/64/128/256 MiB` and all returned split-K-capable algorithms. If the best cuBLASLt plan remains occupancy-limited, add an explicit deterministic CUTLASS BF16 split-K candidate:

  ```text
  split count {2,4,8,16}
  exact contiguous or balanced K partition recorded in the choice
  each CTA writes a disjoint FP32 partial [slot][split][224][224]
  one versioned coalesced fixed-order finalize writes final dW
  no atomicAdd and no launch-time allocation
  ```

  The choice records split count, K partition/granularity, finalize kernel version, reduction order, scratch layout/version/offset/bytes/alignment, beta mode, and slot count. Reject any mismatch between the top-level choice fields and `split_k_contract`. Runtime assigns each concurrent dW a unique scratch slot; `splitk_finalize_done[slot]` is the last-reader/reuse event. The critical dX and auxiliary dW handles have independent aligned workspace slices. Select split count and concurrent-SM schedule through paired full-step time.

- [ ] Include the algorithm-table SHA256 in policy, static-arena plan, execution profile, checkpoint, and benchmark fingerprint. A missing shape, duplicate entry, stale device/library identity, unsupported workspace, scratch overlap, or key mismatch is fatal.

- [ ] Zero-fill only compute padding rows through one prepared vector kernel. BN, loss, samples/s, and denominators always use `active_rows`; padding is never counted as data.

- [ ] Add:

  ```cpp
  mgt::Status LaunchAdamWKernelWithBfloat16Mirror(
      const AdamWKernelConfig& config,
      float* master,
      __nv_bfloat16* mirror,
      const float* grad,
      float* moment1,
      float* moment2,
      cudaStream_t stream);
  ```

  It computes Adam in FP32, writes FP32 master/moments, and casts the final parameter once with round-to-nearest BF16. The BF16 mirror is initialized once after fresh initialization and once after checkpoint restore.

- [ ] Test nonsymmetric forward/dW/dX matrices, beta +0/+1, every role/site/row key in each prepared catalog, active versus padded rows, invalid alignment/capacity/device/shape, duplicate/missing keys, public-algorithm reconstruction, split-K scratch slot reuse, stale mirror rejection, and exact BF16 mirror bits after Adam.

- [ ] Use two separate numerical gates. First prove the primitive against an FP32 accumulator fed the already BF16-rounded A and B operands:

  ```text
  every output finite
  relative L2 <= 1e-4
  scaled max error <= 2e-3
  padded output exactly +0
  split-K and non-split implementations meet the same oracle
  ```

  Then apply the looser BF16-policy-versus-strict-FP32 trajectory gate (`relative L2 <= 0.02`, cosine `>=0.999`, norm ratio `[0.97,1.03]`). A primitive that fails the rounded-input oracle is an implementation bug, not acceptable BF16 drift.

- [ ] Cluster NCU gate, after performance counters are enabled, is keyed by the exact `Bf16GemmKeyV1` and recorded choice rather than by dimensions alone. For every material production key record descriptors, BF16 HMMA/SASS evidence, Nsight Systems/Compute duration source, calls per step, and concurrent interval. Define:

  ```text
  plain GEMM issued FLOPs = 2*m*n*k using recorded compute dimensions
  split-K partial FLOPs = sum_s(2*m*n*k_s)
  split-K finalize FP32 ops = m*n*(split_count-1), reported outside BF16 denominator
  per-key TFLOP/s = issued FLOPs per invocation / median kernel duration
  aggregate active-window TFLOP/s = sum(production calls * issued FLOPs) /
                                    union of their GPU-active intervals
  end-to-end eligible rate = eligible issued FLOPs per step / max-rank step time
  ```

  Require explicit BF16 A/B descriptors, BF16 Tensor Core instructions, and no TF32 instructions. Target large-role keys at `>=40%` and narrow-role keys at `>=25%` of one-A100 312-TFLOP/s dense BF16 peak, but never average away a bad role. If counters return `ERR_NVGPUCTRPERM` or any required metric is unavailable, emit `blocked` and no utilization percentage.


- [ ] Commit:

  ```powershell
  git add native/include/mgt/a100_bf16_algorithm_table.hpp native/src/a100_bf16_algorithm_table.cpp native/tests/test_a100_bf16_algorithm_table.cpp native/cuda/mgt_cuda/bf16_linear_train_ops.cuh native/cuda/bf16_linear_train_ops.cu native/tests/cuda/test_cuda_bf16_linear_train_ops.cu native/cuda/mgt_cuda/adamw.cuh native/cuda/adamw.cu native/tests/cuda/test_cuda_adamw_smoke.cu scripts/ensure_cutlass.sh scripts/cutlass_sm80_bf16.lock native/CMakeLists.txt
  git commit -m "perf: add prepared bf16 tensor core linear ops"
  ```

---

### Task 4: Wire a Single-Cast BF16 Activation and Gradient Dataflow

**Files:**

- Modify: `native/cuda/mgt_cuda/mlp_batch_norm_forward.cuh`
- Modify: `native/cuda/mlp_batch_norm_forward.cu`
- Create: `native/cuda/mgt_cuda/bf16_activation.cuh`
- Create: `native/cuda/bf16_activation.cu`
- Create: `native/tests/cuda/test_mlp_batch_norm_bf16_dataflow.cu`
- Modify: `native/CMakeLists.txt`

**Interfaces:**

```cpp
struct MlpBatchNormBf16WorkspacePlan {
    std::uint32_t capacity_rows;
    std::uint64_t saved_activation_bf16_offset;
    std::uint64_t saved_activation_bf16_count;
    std::uint64_t saved_xhat_offset;
    std::uint64_t saved_xhat_count;
    std::uint64_t relu_mask_offset_bytes;
    std::uint64_t relu_mask_bytes;
    std::uint64_t dz_ring_bf16_offset;
    std::uint64_t dz_ring_bf16_count;
    std::uint64_t preactivation_f32_offset;
    std::uint64_t grad_input_f32_offset;
    std::uint64_t total_bytes;
};

mgt::Status BuildMlpBatchNormBf16WorkspacePlan(
    const CudaMlpShape& shape,
    std::uint32_t capacity_rows,
    std::uint32_t dz_ring_slots,
    A100XhatStorage xhat_storage,
    MlpBatchNormBf16WorkspacePlan* out);
```

- [ ] Use BF16 saved activation order:

  ```text
  input BN/ReLU output
  hidden BN/ReLU output
  residual block 0 fc1 output
  residual block 0 fc2/residual/ReLU output
  ...
  residual block 15 fc1 output
  residual block 15 fc2/residual/ReLU output
  ```

  The count is:

  ```text
  capacity_rows * (physical_hd1 + 33 * physical_hd2)
  ```

- [ ] Save a compact ReLU mask from the FP32 post-affine/residual value before BF16 rounding. Map one bit to each logical activation, use warp ballot, and let exactly one lane write each aligned `uint32_t`; no atomic or clear is allowed. Padded feature bits and padded rows are written as zero. Backward gates from this mask, not by re-testing rounded BF16 activations.

- [ ] Account for all 34 activation masks with overflow-checked byte arithmetic and 256-byte slice alignment. Mask storage is part of the stable graph workspace and is restored by recomputation after resume, not serialized in checkpoints.

- [ ] Keep only reusable FP32 preactivation and grad-input scratch. Do not preserve all FP32 activations when the BF16 policy is active.

- [ ] Implement FP32 `xhat` and compact BF16 `xhat` as two complete, named candidate policies. Both write BF16 post-BN activations and BF16 `dZ`; both keep linear dX and residual/upstream accumulation FP32. The tuner may accept one, but the sealed runtime may not switch between them.

- [ ] Every conversion is fused into a producer:

  ```text
  BN forward epilogue -> BF16 activation
  BN backward epilogue -> BF16 dZ operand for the current linear dW/dX GEMMs
  Adam -> BF16 weight mirror
  ```

  A standalone conversion kernel is permitted only for initialization, checkpoint restore, and a test oracle.

- [ ] Zero padded activation/gradient lanes and compute padding rows in the same producer kernel.

- [ ] Test every saved slice offset, overlap rejection, exact capacity, one-byte-short capacity, all full/tail rows, residual aliasing, explicit FP32-xhat and BF16-xhat profile modes, FP32 dX/residual accumulation, BF16 dZ bits, and checkpoint mirror restoration.

- [ ] Full-step numerical gate:

  ```text
  all tensors finite
  per-class relative L2 <= 0.025
  per-class cosine >= 0.999
  norm ratio in [0.97,1.03]
  running mean relative L2 <= 0.01
  running variance relative L2 <= 0.01
  one-step parameter relative L2 <= 0.01
  ```

- [ ] Commit:

  ```powershell
  git add native/cuda/mgt_cuda/mlp_batch_norm_forward.cuh native/cuda/mlp_batch_norm_forward.cu native/cuda/mgt_cuda/bf16_activation.cuh native/cuda/bf16_activation.cu native/tests/cuda/test_mlp_batch_norm_bf16_dataflow.cu native/CMakeLists.txt
  git commit -m "perf: wire bf16 syncbn activation dataflow"
  ```

---

### Task 5: Put Input Forward and Input Gradient on BF16 Tensor Cores

**Files:**

- Create: `native/cuda/mgt_cuda/input_embedding_bf16.cuh`
- Create: `native/cuda/input_embedding_bf16.cu`
- Create: `native/tests/cuda/test_cuda_input_embedding_bf16.cu`
- Modify: `native/cuda/mlp_backward.cu`
- Modify: `native/cuda/mlp_batch_norm_forward.cu`
- Modify: `native/tools/mgt_bn_input_grad_benchmark.cu`
- Modify: `native/CMakeLists.txt`

**Interfaces:**

```cpp
struct InputEmbeddingBf16Config {
    A100InputForwardBackend forward;
    A100InputGradBackend backward;
    A100InputTileMaterialization materialization;
    std::uint32_t positions_per_tile;
};

struct InputEmbeddingBf16Plan;

mgt::Status CreateInputEmbeddingBf16Plan(
    const CudaMlpShape& shape,
    const std::uint32_t* active_rows,
    std::uint32_t active_row_count,
    InputEmbeddingBf16Config config,
    const P888StepControlV1* step_control,
    const std::array<const mgt::TrainStateStorage*, 2>& state_slots,
    cublasLtHandle_t handle,
    const Bf16AlgorithmTable& algorithms,
    const A100StaticArenaView& arena,
    InputEmbeddingBf16Plan** out);

mgt::Status LaunchInputEmbeddingForwardBf16(
    const InputEmbeddingBf16Plan* plan,
    const __nv_bfloat16* table,
    std::uint32_t active_rows,
    float* preactivation_f32,
    cudaStream_t stream);

mgt::Status LaunchInputEmbeddingGradBf16(
    const InputEmbeddingBf16Plan* plan,
    const __nv_bfloat16* dz,
    std::uint32_t active_rows,
    float* table_grad_f32,
    cudaStream_t stream);

mgt::Status GetInputEmbeddingTileReadyEvent(
    const InputEmbeddingBf16Plan* plan,
    std::uint32_t tile,
    cudaEvent_t* out);
```

- [ ] Factor the existing position-tiled one-hot builder and cuBLASLt dispatch from `native/cuda/mlp_backward.cu`; do not copy its global lazy cache, FP16 type, launch-time autotuner, or any default algorithm path.

- [ ] Implement one-hot tiles as fully overwritten BF16 matrices. No memset and no atomic is permitted.

- [ ] Forward tile `0` uses GEMM beta `0`; later tiles use beta `1` into the same FP32 preactivation. Backward tiles write disjoint FP32 parameter ranges with beta `0`.

- [ ] Prepare exact algorithm-table entries for every forward/backward, full/last tile, beta +0/+1, active/compute-row combination in the production and diagnostic row catalogs:

  ```text
  positions per tile {8,12,16,24,32,48,56,64,72}
  rows {12500,12498,12497} for production plus the Task-3 scaling catalog in diagnostic profiles
  input columns = tile_positions * 72
  output columns = 2560
  ```

- [ ] `CreateInputEmbeddingBf16Plan` binds both fixed state slots and the fixed control block, resolves every required key exactly once, and stores direct prepared handles. Forward/backward derive the slot only from `step_control->batch_slot`; no launch-time state pointer exists. Missing full tile, final tile, beta mode, row shape, or custom-kernel version is fatal during preparation; launch performs no lookup that can choose a substitute.

- [ ] Retain BF16 gather forward as an autotune candidate. Dense BF16 forward is promoted only if max-rank full-step time improves; Tensor Core percentage alone is insufficient.

- [ ] The gather candidate reads persistent BF16 table rows with aligned `__nv_bfloat162` vector loads, accumulates the 72 positions in FP32, validates categories before address formation, and writes FP32 preactivation. It is a legitimate winner even with low Tensor Core utilization if wall time is lower.

- [ ] Profile explicit one-hot materialization separately. Only if builder/one-hot HBM traffic remains `>10%` of full-step wall time, add a bounded CUTLASS implicit-one-hot iterator candidate that synthesizes BF16 fragments without a materialized matrix; subject it to the same no-atomic, correctness, and paired wall-time gates.

- [ ] Each one-hot builder writes one `uint32_t` per-CTA health word after validating categories. Deterministically OR/max-reduce locally. Host-NCCL profiles convert that result to an exact FP32 `0.0/1.0` slot in the first input-BN SUM and test the global sum for nonzero; Device-LSA profiles use the separately typed symmetric `uint32_t` health slot. Invalid values produce zero rows and no invalid address. No rank exits early: the fixed collective suffix is drained with neutral payloads, parameter/running-state updates are skipped, then all ranks exit with the same fatal code.

- [ ] Record one precreated event after every input-gradient tile. Task 8 consumes those events to overlap each contiguous tile reduction.

- [ ] Test CPU table-gradient and forward oracles, partial final tile, all category values, invalid category on one rank, disjoint tile writes, exact beta bit patterns, every production/diagnostic row, exact table-key lookup, missing/duplicate entry rejection, exact arena workspace slice, alignment, padded lanes, collective drain, no allocations, and no `atomic` instruction in the optimized kernels.

- [ ] Promotion gate:

  ```text
  input backward stage improves by at least 20%
  full-step median improves by at least 10%
  full-step p95 does not regress
  tail throughput remains at least 98% of its paired baseline
  numerical gates from Task 4 pass
  ```

- [ ] Commit:

  ```powershell
  git add native/cuda/mgt_cuda/input_embedding_bf16.cuh native/cuda/input_embedding_bf16.cu native/tests/cuda/test_cuda_input_embedding_bf16.cu native/cuda/mlp_backward.cu native/cuda/mlp_batch_norm_forward.cu native/tools/mgt_bn_input_grad_benchmark.cu native/CMakeLists.txt
  git commit -m "perf: move input embedding gradients to bf16 tensor cores"
  ```

---

### Task 6: Remove Healthy-Path Atomics, Redundant Clears, and Full Tensor Passes

**Files:**

- Create: `native/cuda/mgt_cuda/deterministic_reductions.cuh`
- Create: `native/cuda/deterministic_reductions.cu`
- Create: `native/tests/cuda/test_deterministic_reductions.cu`
- Modify: `native/cuda/mlp_batch_norm_forward.cu`
- Modify: `native/cuda/finite_check.cu`
- Modify: `native/CMakeLists.txt`

**Interfaces:**

```cpp
struct DeterministicReductionPlan;

mgt::Status CreateDeterministicReductionPlan(
    std::uint32_t capacity_rows,
    std::uint32_t max_ctas,
    const P888StepControlV1* step_control,
    const std::array<const float*, 2>& label_slots,
    const A100StaticArenaView& arena,
    DeterministicReductionPlan** out);

mgt::Status LaunchScalarMseAndGrad(
    const DeterministicReductionPlan* plan,
    const float* output,
    const float* label,
    std::uint32_t active_rows,
    float global_scale,
    float* loss,
    __nv_bfloat16* dy_bf16,
    float* output_bias_grad,
    cudaStream_t stream);

mgt::Status LaunchBf16ScalarHeadForward(
    const DeterministicReductionPlan* plan,
    const __nv_bfloat16* activation,
    const __nv_bfloat16* weight,
    const float* bias,
    std::uint32_t active_rows,
    float* output,
    cudaStream_t stream);

mgt::Status LaunchBf16ScalarHeadBackward(
    const DeterministicReductionPlan* plan,
    const __nv_bfloat16* activation,
    const __nv_bfloat16* weight,
    const __nv_bfloat16* dy,
    std::uint32_t active_rows,
    float* grad_activation_f32,
    float* grad_weight,
    float* grad_bias,
    cudaStream_t stream);
```

- [ ] `label` is a fixed graph argument equal to the base of the two contiguous, capacity-strided label slots bound by the plan; kernels select the effective slot from `step_control->batch_slot`. Plan creation rejects noncontiguous/misaligned slots, and no per-step label pointer is patched.

- [ ] Replace both `atomicAdd` loss paths with:

  ```text
  CTA partial loss/bias sums
  -> deterministic compact finalize
  -> non-input-table tail bucket for dW/dbias; fixed per-step metrics SUM for loss
  ```

- [ ] Keep the `224 -> 1` output head on one fixed custom path because its N=1 geometry is not a useful Tensor Core GEMM. Forward uses a vectorized BF16 row dot with warp/CTA FP32 reduction; backward writes FP32 `dX = dy * W` and reduces 224 FP32 dW columns through disjoint CTA partials plus fixed-order finalize. It contains no global atomic and is excluded from the Tensor-Core-eligible denominator.

- [ ] Output-head dW and bias join the contiguous non-input-table tail bucket. Enqueue the scalar FP32 loss SUM through `metrics_context` every step, after tail and every input-tile collective in the sealed order. Logging cadence only publishes or ignores the completed telemetry slot; it never changes kernels/collectives and never stalls critical BN backward.

- [ ] Profile the fixed scalar head as memory/reduction work and optimize only its vector width, CTA shape, and deterministic finalize inside the versioned custom kernel. Do not add a second runtime backend and never count its N=1 work as a Tensor Core utilization failure.

- [ ] Replace `finite_check.cu` atomics with one health word per CTA followed by a prepared fixed-order two-pass max/OR reduction with disjoint writes; do not use an implementation whose look-back/state machine emits global ATOM/RED instructions. Every CTA overwrites its word each step, so no clear is needed.

- [ ] Remove a `cudaMemsetAsync` before any matrix/bias gradient that is completely overwritten by beta-zero GEMM or deterministic reduction.

- [ ] Keep `CopyGrads` and the BN-related `ColumnSum` in the reference path until Task 7 has a row-tiled oracle; this task must not depend on files that Task 7 has not created yet.

- [ ] Keep existing BN activation-gate and residual semantics untouched in this task. Their fusion belongs to Task 7 after the row-tiled BN kernel and its oracle exist.

- [ ] Remove only clears and passes whose producer already completely overwrites the destination in the current implementation. Defer `CopyGrads`, BN-affine, and linear-bias fusion to Task 7.

- [ ] Add a source gate:

  ```powershell
  rg -n "atomic(Add|Exch|Or|CAS)" native/cuda/deterministic_reductions.cu native/cuda/input_embedding_bf16.cu
  ```

  Expected: exit code `1`, meaning no match in optimized kernels.

- [ ] Require bitwise repeatability for the same rank count, source, policy, snapshot, and launch order. Cross-policy bitwise equality is not required.

- [ ] Commit:

  ```powershell
  git add native/cuda/mgt_cuda/deterministic_reductions.cuh native/cuda/deterministic_reductions.cu native/tests/cuda/test_deterministic_reductions.cu native/cuda/mlp_batch_norm_forward.cu native/cuda/finite_check.cu native/CMakeLists.txt
  git commit -m "perf: remove hot path atomics and redundant clears"
  ```

---

### Task 7: Replace Strided SyncBN with Row-Tiled, Fused BF16 SyncBN

**Files:**

- Create: `native/cuda/mgt_cuda/sync_batch_norm_tiled.cuh`
- Create: `native/cuda/sync_batch_norm_tiled.cu`
- Create: `native/tests/cuda/test_sync_batch_norm_tiled.cu`
- Create: `native/tests/cuda/test_sync_batch_norm_tiled_8rank.cu`
- Modify: `native/cuda/mgt_cuda/sync_batch_norm_selector.cuh`
- Modify: `native/cuda/sync_batch_norm.cu`
- Modify: `native/cuda/mlp_batch_norm_forward.cu`
- Modify: `native/CMakeLists.txt`

**Interfaces:**

```cpp
struct TiledSyncBatchNormConfig {
    std::uint32_t row_chunk;
    std::uint32_t feature_tile;
    A100XhatStorage xhat_storage;
};

struct TiledSyncBatchNormWorkspace {
    float* partials;
    std::uint64_t partial_count;
    std::uint32_t* health_words;
    std::uint64_t health_word_count;
};
```

```cpp
mgt::Status LaunchTiledSyncBatchNormForward(
    const TiledSyncBatchNormConfig& config,
    const float* preactivation,
    const float* bias,
    const __nv_bfloat16* residual,
    std::uint32_t active_rows,
    std::uint32_t global_rows,
    std::uint32_t logical_cols,
    std::uint32_t stride,
    const float* gamma,
    const float* beta,
    float* running_mean,
    float* running_variance,
    float* saved_mean,
    float* saved_inv_std,
    void* saved_xhat,
    __nv_bfloat16* activation,
    TiledSyncBatchNormWorkspace workspace,
    NcclRankContext* context,
    cudaStream_t stream);
```

- [ ] Forward local moments use a 2D logical tile:

  ```text
  warp lane -> adjacent feature
  warp index -> rows inside one row chunk
  partial layout -> [row_chunk_index][sum features][sumsq features]
  ```

  Warp lanes must read adjacent FP32 columns. No CTA may loop over all rows for one feature.

- [ ] A compact deterministic finalize reduces row chunks to packed FP32 `[sum,sumsq]`. Health remains typed `uint32_t` locally. A host-NCCL profile appends one separately defined numeric FP32 health slot (`0.0f` or `1.0f`) to the same SUM and converts `global_health_sum != 0` back to `uint32_t`; an LSA profile uses its typed symmetric health word. Raw integer bits are never interpreted as FP32 and statistics never share an untyped layout.

- [ ] The post-NCCL row-major epilogue fuses:

  ```text
  global mean/variance
  unbiased running-variance update
  xhat
  gamma/beta
  optional residual add
  ReLU
  BF16 activation cast
  padded-lane zero
  ```

- [ ] Backward local statistics fuse activation gating and residual split, accumulate FP32 `sum(dy)` and `sum(dy*xhat)`, then use the selected exact collective backend. The row-major epilogue computes the current linear layer's dZ in FP32 and casts that dZ once to BF16 for dW/dX GEMM operands. The linear dX output and every residual/upstream accumulation remain FP32.

- [ ] Gate backward from Task 4's saved pre-rounding ReLU mask and write the residual skip gradient exactly once. Do not infer the mask from BF16 values and do not materialize a second gated tensor.

- [ ] The backward epilogue writes `dgamma`, `dbeta`, and linear-bias CTA partials directly to their final/preallocated reduction ranges. Delete optimized-path `CopyGrads` and strided `ColumnSum` only after per-site oracle tests pass.

- [ ] Source/SASS gates require no user `atomic(Add|Exch|Or|CAS)` in `sync_batch_norm_tiled.cu`; NCCL internals are explicitly outside this rule.

- [ ] Preserve exactly one forward and one backward collective per BN site: `68` BN collectives per step. Reject any trace with a different count/order.

- [ ] Keep `sync_batch_norm_fused_feature.cu` available only as a named rejected benchmark candidate. It cannot become automatic.

- [ ] Sweep only:

  ```text
  row_chunk {256,512,1024}
  feature_tile {32,64}
  xhat_storage {kFp32,kBf16}
  ```

- [ ] Tests compare concatenated eight-rank results with one logical CPU/PyTorch BF16-input FP32-BN reference for deliberately different per-rank distributions and unequal tail rows.

- [ ] Correctness covers output, xhat, mean, biased variance, running mean, unbiased running variance, BF16 dZ, FP32 linear dX/upstream, dgamma, dbeta, residual gradient, bias gradient, padding, typed health, one-rank fault drain, and 100 repeated steps.

- [ ] Promotion gate:

  ```text
  forward+BN backward aggregate improves by at least 10%
  full-step improves by at least 5%
  no full/tail p95 regression
  same-policy repeated checksum matches
  all Task 4 numerical gates pass
  ```

- [ ] Commit:

  ```powershell
  git add native/cuda/mgt_cuda/sync_batch_norm_tiled.cuh native/cuda/sync_batch_norm_tiled.cu native/tests/cuda/test_sync_batch_norm_tiled.cu native/tests/cuda/test_sync_batch_norm_tiled_8rank.cu native/cuda/mgt_cuda/sync_batch_norm_selector.cuh native/cuda/sync_batch_norm.cu native/cuda/mlp_batch_norm_forward.cu native/CMakeLists.txt
  git commit -m "perf: add row tiled bf16 sync batch norm"
  ```

---

### Task 8: Overlap dW/dX and Hide Weight-Gradient AllReduce

**Files:**

- Create: `native/include/mgt/bn_communication_plan.hpp`
- Create: `native/src/bn_communication_plan.cpp`
- Create: `native/tests/test_bn_communication_plan.cpp`
- Create: `native/tests/cuda/test_mlp_batch_norm_overlap_8rank.cu`
- Modify: `native/cuda/mgt_cuda/a100_bf16_runtime.cuh`
- Modify: `native/cuda/a100_bf16_runtime.cu`
- Modify: `native/cuda/mlp_batch_norm_forward.cu`
- Modify: `native/cuda/allreduce_nccl.cu`
- Modify: `native/CMakeLists.txt`

**Interfaces:**

```cpp
struct GradientRange {
    std::uint64_t offset_floats;
    std::uint64_t count_floats;
};

struct BnCommunicationPlan {
    GradientRange non_input_table_tail;  // [InputBias(shape), total_parameters)
    std::array<GradientRange, 9> input_tiles;
    std::uint32_t input_tile_count;
    std::uint64_t total_parameter_floats;
};

mgt::Status BuildA100BnCommunicationPlan(
    const CudaMlpShape& shape,
    std::uint32_t input_positions_per_tile,
    BnCommunicationPlan* out);
```

- [ ] Derive ranges from the authoritative layout helpers: input tiles partition `[InputTable(shape),InputBias(shape))`; `non_input_table_tail=[InputBias(shape),TotalParameters(shape))` includes input bias, hidden/residual/output weights, and every linear bias. Define one writer per range and prove sorted union coverage with no gap/overlap over all `15,460,289` physical linear parameter floats. Unit-test the exact boundaries so `input_bias` can never disappear between the two classes.

- [ ] After each BN backward:

  ```text
  critical compute stream -> dX GEMM
  auxiliary GEMM stream   -> dW GEMM
  ```

  Both wait on the same precreated `dz_ready` event. The next layer waits only for dX; the dW result publishes a range-ready event.

- [ ] Use a BF16 dZ ring with policy-controlled slots `{2,3,4}`. A slot cannot be reused until both its dW and dX readers have completed and the recorded `dz_slot_reuse` event fires. FP32 linear dX/upstream scratch is a separate arena slice. No busy polling or host wait is allowed.

- [ ] For `kConcurrentProtected`, enqueue critical dX first on the highest supported priority and auxiliary dW second on a lower-priority stream, both after `dz_ready`. Seal the exact priorities and algorithms. Reject the concurrent candidate if isolated dX duration regresses by more than `2%`, if dW causes a longer critical BN-to-BN interval, or if paired full-step time does not improve; stream priority is treated only as a scheduling hint, never as proof of protection.

- [ ] Materialize and hash this minimum event/ownership DAG; every event is precreated and every edge is device-side:

  | Producer/event | Consumers | Reuse condition |
  | --- | --- | --- |
  | input batch batch_ready[slot,seq] | first input formulation | prepared step records batch_consumed[slot,seq] |
  | linear GEMM preactivation_ready[site] | local BN moments | BN epilogue completes |
  | BN local finalize bn_local_ready[site] | critical BN collective | global stats result consumed |
  | BN epilogue activation_ready[site] | next forward GEMM | last forward reader completes |
  | BN backward dz_ready[site,slot,seq] | one dX reader and one dW reader | both record dz_slot_reuse |
  | linear dX dx_ready[site] | residual add / preceding BN backward | last FP32 upstream reader completes |
  | each dW/bias writer | its range join | tail or tile NCCL completion |
  | final tail writer join tail_ready plus all_bn_done | tail NCCL | tail_reduce_done |
  | input dW tile_ready[i] | input-tile NCCL | tile_reduce_done[i] |
  | final weight collective weights_done | Adam | update_done |
  | per-step metrics SUM metrics_done | telemetry slot / final join | telemetry slot sequence is consumed or skipped |
  | update_done plus metrics_done -> step_done[seq] | checkpoint fence / next step | all next-step readers bind seq+1 |


- [ ] Publish `tail_ready` only after every asynchronous dW/bias writer touching `[InputBias(shape),TotalParameters(shape))` has joined and the final input-bias writer completes. After the final input-BN backward SUM, the critical stream separately records `all_bn_done`. The weight stream waits on both `all_bn_done` and `tail_ready` before launching the tail SUM; neither event implies the other. This launch may overlap the long BF16 input-table dW.

- [ ] For input-table weights, each position tile is contiguous. The dispatcher issues the tail collective first, then the weight stream consumes ascending `tile_ready` events and enqueues each FP32 SUM while compute builds/GEMMs later tiles. A tile scratch/range is not reusable until its NCCL completion event fires.

- [ ] BN collectives use `bn_context` on the critical stream; weight collectives use `weight_context` on the weight stream; the per-step scalar loss uses `metrics_context` after the last input-tile collective is issued. Each communicator has one owning stream. The rank-uniform global host issue order is `F0..F33,B33..B0,tail,input_tile[0..N-1],metrics`; capture/replay preserve it for every step. Fingerprint `NCCL_LAUNCH_ORDER_IMPLICIT`, communicator identities, and sequence hash. Reject any concurrency/graph candidate that cannot honor it.

- [ ] Record `weights_done` after the final input tile. The compute stream waits on it once before Adam. No other step-wide synchronization is permitted.

- [ ] Do not convert BN statistics or affine gradients to BF16. Do not use BF16 NCCL as the initial implementation.

- [ ] Test delayed tail dW writer, delayed input tile, delayed NCCL, two/three/four dZ-ring slots, full/tail rows, one-rank device-health fault with neutral collective drain, and a separate injected NCCL transport failure with coordinated communicator abort. Also require exact per-context/global issue counters, exact input-bias coverage, no gap/overlap/double reduction, scratch reuse events, Adam wait, 1000 healthy steps, and rank-specific delay without deadlock.

- [ ] Compare overlap policies:

  ```text
  sectioned reference
  tail + one input SUM
  tail + input tile SUMs
  concurrent dW/dX off/on
  ring slots 2/3/4
  ```

- [ ] Promotion requires paired median step-time improvement `>=3%`, corresponding examples/s improvement `>=3%`, no p95 regression, and zero correctness/deadlock failures.

- [ ] Commit:

  ```powershell
  git add native/include/mgt/bn_communication_plan.hpp native/src/bn_communication_plan.cpp native/tests/test_bn_communication_plan.cpp native/tests/cuda/test_mlp_batch_norm_overlap_8rank.cu native/cuda/mgt_cuda/a100_bf16_runtime.cuh native/cuda/a100_bf16_runtime.cu native/cuda/mlp_batch_norm_forward.cu native/cuda/allreduce_nccl.cu native/CMakeLists.txt
  git commit -m "perf: overlap bf16 backward and weight reductions"
  ```

---

### Task 9: Profile the Stable Eager DAG and Prepare Graph Infrastructure

**Files:**

- Create: `native/cuda/mgt_cuda/a100_training_graph.cuh`
- Modify: `native/cuda/mgt_cuda/p888_step_control.cuh`
- Create: `native/cuda/a100_training_graph.cu`
- Create: `native/tests/cuda/test_a100_training_graph_8rank.cu`
- Modify: `native/cuda/mgt_cuda/a100_bf16_runtime.cuh`
- Modify: `native/cuda/a100_bf16_runtime.cu`
- Modify: `native/tools/mgt_bn_step_benchmark.cu`
- Create: `scripts/cluster/run_a100_bf16_nsys.sbatch`
- Modify: `native/CMakeLists.txt`

**Interfaces:**

```cpp
// Imported from p888_step_control.cuh for reference; do not redefine here.
enum class A100LocalGraphSlot : std::uint32_t {
    kFull = 1,
    kTail = 2,
};

struct P888StepControlV1 {
    std::uint32_t schema_version;
    std::uint32_t graph_slot;
    std::uint32_t active_rows;
    std::uint32_t global_rows;
    std::uint64_t global_offset;
    std::uint64_t committed_sequence;
    std::uint64_t committed_optimizer_step;
    std::uint64_t inflight_sequence;
    std::uint64_t inflight_optimizer_step;
    std::uint64_t semantic_epoch;
    std::uint32_t batch_in_epoch;
    std::uint32_t batch_slot;
    std::uint64_t generation_seed;
    std::uint32_t fatal_health;
};

struct A100CollectiveGraphCaptureInfo {
    std::uint32_t world;
    std::uint32_t rank;
    std::uint32_t local_full_rows;  // 12500 on every rank
    std::uint32_t local_tail_rows;  // 12498 on ranks 0-1, else 12497
};

mgt::Status CaptureA100TrainingGraphSessions(
    A100Bf16Runtime* runtime,
    const A100CollectiveGraphCaptureInfo& info);
mgt::Status LaunchA100TrainingGraph(
    A100Bf16Runtime* runtime,
    A100LocalGraphSlot slot);
```

- [ ] First run Nsight Systems on the strict shared production step and the complete eager BF16 step from Task 8. Trace only a bounded post-warmup window and preserve max-rank `.nsys-rep` plus exported CUDA/NCCL/NVTX summaries.

- [ ] Structural eager gate: ordinary steps contain zero `cudaDeviceSynchronize`, `cudaStreamSynchronize`, synchronous D2H, `cudaMalloc/cudaFree`, descriptor/heuristic creation, and blocking host polling. Show the 68 unavoidable BN edges separately from removable host/stream gaps.

- [ ] Capture Task 2's already-tested fixed-address `P888StepControlV1` begin/commit kernels unchanged and add only the public graph capture/launch interfaces. The graph launch API always uses the runtime-owned sealed stream and never accepts an external stream or per-node parameters. Verify graph begin derives `batch_slot=inflight_sequence&1`, graph commit remains ordered after `update_done`, `metrics_done`, and zero global health, and a drained mismatch leaves the committed cursor unchanged. Keep `A100GraphBackend::kDisabled` as the only acceptance-eligible choice in this task.

- [ ] Probe and record NCCL graph support, CUDA `>=11.3`, `NCCL_LAUNCH_ORDER_IMPLICIT`, and the exact communicator/topology identity; one process owns one GPU. A graph candidate whose required capability is absent is rejected by the tuner. A sealed graph profile treats any later capability mismatch as fatal and never launches eager. Do not promote a final graph until Tasks 10-11 freeze kernels and collectives.

- [ ] Add a correctness-only smoke capture behind a test flag using exactly two world-8 collective capture sessions. Full capture has 12500 rows on all ranks. Tail capture is one simultaneous session with 12498 rows on ranks 0-1 and 12497 on ranks 2-7; every rank enqueues the identical collective sequence/count/type even though local kernels differ. Each rank retains only `full_exec` and its own `tail_exec`. Never capture 12498 and 12497 as independent collective graphs. This smoke graph is invalidated whenever Task 10 or 11 changes a kernel or collective.

- [ ] Checkpoint, terminal health reporting, and host JSON serialization remain outside the repeated graph.

- [ ] Run 100 correctness replays against eager BF16. Defer the 10,000-replay stress, final capture, and graph performance promotion to Task 13 after the production pipeline is stable.

- [ ] Eager readiness gate:

  ```text
  full multistream completion event joins compute + dW + BN/weight/metrics NCCL + Adam
  no hidden host synchronization or allocation
  no long gradient collective overlaps a critical BN collective
  event waits match the intended DAG on all ranks
  strict and BF16 traces use the same shared step entry point
  the profiler identifies the next top three exposed costs
  ```

- [ ] Commit:

  ```powershell
  git add native/cuda/mgt_cuda/a100_training_graph.cuh native/cuda/a100_training_graph.cu native/tests/cuda/test_a100_training_graph_8rank.cu native/cuda/mgt_cuda/p888_step_control.cuh native/cuda/mgt_cuda/a100_bf16_runtime.cuh native/cuda/a100_bf16_runtime.cu native/tools/mgt_bn_step_benchmark.cu scripts/cluster/run_a100_bf16_nsys.sbatch native/CMakeLists.txt
  git commit -m "perf: profile eager bf16 dag and prepare graphs"
  ```

---

### Task 10: Evaluate CUTLASS BF16 GEMM Epilogues that Emit BN Partials

**Files:**

- Create: `native/cuda/mgt_cuda/bf16_gemm_bn.cuh`
- Create: `native/cuda/bf16_gemm_bn.cu`
- Create: `native/tests/cuda/test_bf16_gemm_bn.cu`
- Modify: `native/cuda/mgt_cuda/bf16_linear_train_ops.cuh`
- Modify: `native/cuda/bf16_linear_train_ops.cu`
- Modify: `native/cuda/mgt_cuda/sync_batch_norm_tiled.cuh`
- Modify: `native/cuda/sync_batch_norm_tiled.cu`
- Modify: `native/CMakeLists.txt`

**Interfaces:**

```cpp
struct Bf16GemmBnProblem {
    Bf16LinearProblem linear;
    std::uint32_t logical_output_features;
    std::uint32_t row_chunk;
};

struct Bf16GemmBnPlan;
mgt::Status CreateBf16GemmBnPlan(
    const Bf16GemmBnProblem* problems,
    std::uint32_t problem_count,
    const Bf16AlgorithmTable& algorithms,
    const A100StaticArenaView& arena,
    Bf16GemmBnPlan** out);
mgt::Status DestroyBf16GemmBnPlan(Bf16GemmBnPlan* plan);

mgt::Status LaunchBf16GemmWithBnPartials(
    const Bf16GemmBnPlan* plan,
    const Bf16GemmBnProblem& problem,
    const __nv_bfloat16* input,
    const __nv_bfloat16* weight,
    const float* bias,
    float* preactivation_f32,
    float* partial_sum,
    float* partial_sumsq,
    cudaStream_t stream);
```

- [ ] Implement fixed SM80 CUTLASS candidates with BF16 TensorOp operands and FP32 accumulators for:

  ```text
  12512x2560x224
  12512x224x224
  tail active rows 12498/12497 with compute_rows rounded by the sealed policy and padded rows exactly +0
  ```

- [ ] The epilogue adds FP32 bias, writes the authoritative FP32 preactivation required across the global BN barrier, and writes FP32 sum/sumsq partials from the same accumulator values. It uses no global atomics. A BF16-preactivation variant is a different numerical policy and is not part of this first production experiment.
- [ ] Represent every fused candidate as an exact `Bf16GemmKeyV1/Bf16GemmChoiceV1` record with a versioned custom kernel, tile, stage count, BN-partial layout, and arena slices. `CreateBf16GemmBnPlan` resolves the complete catalog before warmup; launch performs no selection, descriptor creation, allocation, or workspace lookup.


- [ ] Reuse Task 7's compact deterministic finalize and NCCL path. Do not duplicate BatchNorm formulas.

- [ ] Compare CUTLASS tiles:

  ```text
  128x128x32
  128x64x32
  64x128x32
  stages 3 and 4 where supported
  ```

- [ ] Run this experiment only when Task 9's Nsight trace shows the FP32 preactivation write plus separate local-moment read on a material critical path. Acceptance requires a source-identical paired full-step median improvement `>=3%`, no full/final p95 regression, and all per-site numerical gates.

- [ ] If no fused candidate clears the gate, record each as rejected; the independently measured cuBLASLt-plus-row-tiled-BN candidate remains eligible for later sealing. Do not broaden the search and do not place both implementations behind a runtime recovery branch.

- [ ] Whichever backend wins invalidates Task 9's smoke graph. Freeze the eager kernel table here; final graph capture occurs only in Task 13.

- [ ] Commit:

  ```powershell
  git add native/cuda/mgt_cuda/bf16_gemm_bn.cuh native/cuda/bf16_gemm_bn.cu native/tests/cuda/test_bf16_gemm_bn.cu native/cuda/mgt_cuda/bf16_linear_train_ops.cuh native/cuda/bf16_linear_train_ops.cu native/cuda/mgt_cuda/sync_batch_norm_tiled.cuh native/cuda/sync_batch_norm_tiled.cu native/CMakeLists.txt
  git commit -m "perf: evaluate fused bf16 gemm bn partials"
  ```

---

### Task 11: Evaluate Device-Initiated LSA for Tiny BN Collectives

**Files:**

- Create: `native/cuda/mgt_cuda/device_lsa_bn.cuh`
- Create: `native/cuda/device_lsa_bn.cu`
- Create: `native/tests/cuda/test_device_lsa_bn_8rank.cu`
- Modify: `native/include/mgt/a100_bf16_policy.hpp`
- Modify: `native/src/a100_bf16_policy.cpp`
- Modify: `native/cuda/mgt_cuda/a100_bf16_runtime.cuh`
- Modify: `native/cuda/a100_bf16_runtime.cu`
- Modify: `native/CMakeLists.txt`

**Interfaces:**

```cpp
// A100BnCollectiveBackend is defined once in Task 2's sealed policy.
struct DeviceLsaBnContractV1 {
    std::uint32_t expected_world;
    std::uint32_t block_threads;
    std::uint32_t grid_blocks;
    std::uint32_t lsa_barrier_count;
    std::uint32_t max_stats_floats;
    std::array<std::uint8_t, 32> nccl_device_abi_sha256;
};
struct DeviceLsaBnPlan;

mgt::Status QueryDeviceLsaBnSupportForTuner(
    const DeviceLsaBnContractV1& contract,
    bool* supported,
    std::string* reason);
mgt::Status CreateDeviceLsaBnPlan(
    const DeviceLsaBnContractV1& contract,
    const P888A100ExecutionProfileV1& profile,
    const A100StaticArenaView& arena,
    ncclComm_t host_comm,
    DeviceLsaBnPlan** out);
mgt::Status DestroyDeviceLsaBnPlan(DeviceLsaBnPlan* plan);

mgt::Status LaunchDeviceLsaBnAllReduce(
    const DeviceLsaBnPlan* plan,
    const float* symmetric_local_stats,
    float* reduced_local_stats,
    const std::uint32_t* symmetric_local_health,
    std::uint32_t* reduced_local_health,
    std::uint32_t float_count,
    cudaStream_t stream);
```

This is a conditional tuner experiment. Host NCCL and Device LSA are independent exact candidate profiles, not primary/backup paths. The official NCCL Device API is available only in sufficiently new NCCL releases and LSA requires peer-accessible symmetric memory. A100 may use LSA with full eight-GPU peer access; it cannot use Hopper-only multimem/NVLink SHARP. If the sealed winner names LSA, failure of any LSA contract is fatal. If it names host NCCL, production never initializes or probes LSA.

- [ ] Attempt this task only if Task 9 shows exposed latency from the 68 tiny BN collectives remains a top-three critical-path cost after row-tiled kernels and no long gradient collective is interfering. Otherwise record `not_triggered` and continue.

- [ ] The tuner admits an LSA candidate only when compile and runtime NCCL ABI/API identity match the recorded supported NCCL `>=2.28` build, world is exactly 8 on one node, full pairwise P2P/LSA connectivity exists, `ncclMemAlloc` symmetric allocation and window registration succeed, and topology matches. A failed tuner check rejects only the LSA candidate. The accepted-profile startup repeats every check and exits collectively on failure; it never mutates the policy to host NCCL.

- [ ] Allocate fixed symmetric FP32 source/result buffers during runtime preparation. For each BN site, the local finalize writes packed `[sum,sumsq]` into the symmetric source, then one device kernel performs:

  ```text
  LSA barrier: all eight local buffers ready
  each output element loads ranks 0..7 in fixed order
  FP32 deterministic local accumulation, no atomic
  write only this rank's reduced result
  LSA barrier: all peer reads complete before buffer reuse
  ```

  Health uses a separately aligned symmetric `uint32_t` source/result word with deterministic rank-order OR/max. Every CTA obeys the barrier contract even after health becomes nonzero; health bits are never reinterpreted as floats.

- [ ] Preserve exactly the same 68-site order, FP32 statistics, global row denominators, running-state update, and following BN epilogue. The optimization removes host-enqueued collective latency; it does not remove a global dependency or change to local/ghost BN.

- [ ] Seal one block size and one `gridDim` sized for the largest `2 * 2560` FP32 statistics buffer, identical on all ranks and all 68 sites. Require `reqs.lsaBarrierCount == gridDim.x`. Every CTA participates in both LSA barriers, including CTAs with no valid element and every fatal-health path; early return before the second barrier is forbidden. Adjacent lanes handle adjacent values and no global atomic is used. Do not add NVSHMEM or a second communication runtime.

- [ ] Validate every production BN count, deliberately different per-rank distributions, full/final rows, peer disable, unsupported/mismatched NCCL ABI, one-rank health fault, zero-work CTAs, barrier-count mismatch, buffer reuse, both collective graph sessions, and numerical equality within the same FP32 per-site tolerance. In tuner tests peer/API failures must reject LSA; in sealed-production tests they must produce the expected fatal startup code.

- [ ] Run a 10,000-step eight-rank deadlock/reuse soak with injected rank delays and NCCL async-error polling. Require identical collective counters and final model checksums across repeated same-policy runs. Test teardown in the fixed order: destroy graph execs, destroy device communicators, deregister windows, `ncclMemFree` symmetric allocations, destroy host communicators; no object may outlive memory it can access.

- [ ] Seal an LSA profile only if source-identical paired tests show `>=5%` full-step Q50 improvement, no full/final diagnostic Q95 regression, no loss of convergence gates, and Task 9's trace confirms exposed tiny-collective latency disappeared. Otherwise mark LSA rejected and separately seal the measured host-NCCL candidate if it passes all gates. Production never transitions between the two.

- [ ] Any accepted/rejected LSA result invalidates the smoke graph. Final graph capture remains in Task 13.

- [ ] Commit:

  ```powershell
  git add native/cuda/mgt_cuda/device_lsa_bn.cuh native/cuda/device_lsa_bn.cu native/tests/cuda/test_device_lsa_bn_8rank.cu native/include/mgt/a100_bf16_policy.hpp native/src/a100_bf16_policy.cpp native/cuda/mgt_cuda/a100_bf16_runtime.cuh native/cuda/a100_bf16_runtime.cu native/CMakeLists.txt
  git commit -m "perf: evaluate device lsa syncbn collectives"
  ```

---

### Task 12: Integrate the Prepared BF16 Step into the Real Trainer

**Files:**

- Modify: `native/tools/mgt_native_train_smoke.cu`
- Create: `native/tools/mgt_a100_bf16_tuner_runner.cu`
- Create: `native/tools/mgt_a100_bf16_production_runner.cu`
- Modify: `native/cuda/mgt_cuda/prepared_p888_train_step.cuh`
- Modify: `native/cuda/prepared_p888_train_step.cu`
- Modify: `native/include/mgt/train_plan.hpp`
- Modify: `native/src/train_plan.cpp`
- Modify: `native/include/mgt/training_artifacts.hpp`
- Modify: `native/src/training_artifacts.cpp`
- Modify: `native/include/mgt/weight_export.hpp`
- Modify: `native/src/weight_export.cpp`
- Create: `native/tests/cuda/test_p888_bf16_train_8rank.cu`
- Modify: `native/tests/test_training_artifacts.cpp`
- Modify: `native/tests/test_weight_export.cpp`
- Modify: `native/CMakeLists.txt`

**Interfaces:**

- The tuner runner consumes only a complete `profile_state=candidate` execution profile; the production runner consumes only `profile_state=accepted` `P888A100ExecutionProfileV1` bytes.
- Produces resumable checkpoints containing FP32 master state and exact BF16 mirror bits.
- Produces inference export matching the effective BF16-trained model.

- [ ] Add two explicit executables over the same trainer library:

  ```text
  mgt_a100_bf16_tuner_runner --candidate-execution-profile PATH
  mgt_a100_bf16_production_runner --execution-profile PATH
  ```

  The tuner executable rejects accepted profiles and is the only path Task 13 may use for candidate timing. The production executable rejects candidate profiles and, until Task 13 seals one, its expected test result is `profile_not_accepted`. Neither executable accepts legacy `--linear-fp16`, `--input-grad-fp16`, TF32, backend/tile environment variables, `auto`, or retuning.

- [ ] Split link targets: `mgt_a100_bf16_executor` contains only profile parsing/validation, prepared kernels, fixed algorithm reconstruction, and dispatcher logic; `mgt_a100_bf16_discovery` contains heuristics and timers and is linked only into the tuner runner. Add a production symbol/import scan that rejects heuristic discovery, strict-oracle entry points, legacy FP32/TF32 eligible-GEMM launchers, and environment-policy parsers.

- [ ] The trainer owns generation, logging, checkpoint, and epoch control, but every optimizer step calls `LaunchPreparedP888TrainStep`. Add a regression/source gate that the trainer does not directly invoke the low-level BN/GEMM/backward/Adam launch sequence.

- [ ] Build the exact epoch:

  ```text
  999978 newly generated examples
  depths 1..29
  34482 examples per depth
  nine global batches of 100000
  one final global batch of 99978
  ten optimizer steps
  ```

- [ ] Double-buffer states/labels in fixed ordinary-arena slots. `generation_stream` fills batch `n+1` while the prepared step consumes batch `n` through the backend named in the profile. The persistent device control block derives the slot from the monotonic sequence; only custom ingress/loss kernels dereference that slot, while every prepared GEMM uses fixed internal addresses. Events enforce one writer, one reader, and one reuse sequence per buffer. A graph-profile launch failure is fatal; it does not execute eager.

- [ ] Replace per-step `cudaEventSynchronize` and per-step device-to-host loss copies with a 256-slot device telemetry ring plus a 256-slot pinned-host mirror from the static arena plan. The graph writes the device record every step. The CPU uses `cudaEventQuery`; the telemetry stream batches contiguous asynchronous D2H copies, and the CPU prints completed records in sequence order. Ring overrun increments a visible drop counter instead of stalling the compute graph.

- [ ] A checkpoint fence may synchronize only after all ranks agree to stop at the same completed sequence. No synchronization or file I/O is allowed inside an ordinary optimizer step. A device data/health fault drains the sealed collective suffix with neutral payloads, skips updates, reaches the coordinated fence, writes the fatal artifact, and exits. An NCCL async transport error instead causes the watchdog to request the same ordered communicator abort on every rank; no checkpoint is claimed recoverable from a partially completed step.

- [ ] Checkpoint payload includes:

  ```text
  FP32 master linear weights
  raw BF16 mirror bits
  FP32 linear Adam m/v
  FP32 BN gamma/beta
  FP32 BN Adam m/v
  FP32 running mean/variance
  optimizer step
  semantic epoch and batch cursor
  exact execution-profile bytes and SHA256
  exact policy bytes and algorithm-table SHA256
  static-arena layout SHA256
  selected BN collective backend and topology fingerprint
  graph/eager backend identity
  source/binary/library/hardware fingerprints
  ```

- [ ] Resume restores the BF16 mirror byte-for-byte; it does not regenerate it from FP32 master.

- [ ] Training forward uses the BF16 mirror as the effective model. BN-folded inference export therefore widens BF16 mirror values to FP32, folds FP32 running statistics/affine parameters, then casts once to the evaluator's existing FP16 layout.

- [ ] Validate unfused BF16 evaluation against the reloaded folded export:

  ```text
  max absolute <= 0.01
  relative L2 <= 0.02
  same score ordering on the fixed evaluation fixture
  repeated export byte-identical
  pre/post-resume export byte-identical
  ```

- [ ] Test 20 uninterrupted steps versus 7-step checkpoint plus 13 resumed steps through the candidate tuner runner. Require identical cursor, optimizer step, execution-profile bytes, policy bytes, BF16 mirror bits, and final checksum. Separately prove that production rejects candidate/mutated profiles, accepts a test-sealed profile only with all gate hashes, and never invokes tuner/heuristic/oracle symbols.

- [ ] Commit:

  ```powershell
  git add native/tools/mgt_native_train_smoke.cu native/tools/mgt_a100_bf16_tuner_runner.cu native/tools/mgt_a100_bf16_production_runner.cu native/cuda/mgt_cuda/prepared_p888_train_step.cuh native/cuda/prepared_p888_train_step.cu native/include/mgt/train_plan.hpp native/src/train_plan.cpp native/include/mgt/training_artifacts.hpp native/src/training_artifacts.cpp native/include/mgt/weight_export.hpp native/src/weight_export.cpp native/tests/cuda/test_p888_bf16_train_8rank.cu native/tests/test_training_artifacts.cpp native/tests/test_weight_export.cpp native/CMakeLists.txt
  git commit -m "feat: integrate production a100 bf16 syncbn trainer"
  ```

---

### Task 13: Final Graph Capture, Automatic 8xA100 BF16 Autotuning, and Profiling

**Files:**

- Create: `scripts/autotune_8xa100_bf16.py`
- Create: `scripts/tests/test_autotune_8xa100_bf16.py`
- Create: `scripts/cluster/run_a100_bf16_autotune.sbatch`
- Modify: `scripts/cluster/run_a100_bf16_nsys.sbatch`
- Create: `scripts/cluster/run_a100_bf16_ncu.sbatch`
- Modify: `scripts/summarize_a100_bf16_benchmark.py`
- Create: `scripts/poll_a100_memory.py`
- Create: `scripts/tests/test_poll_a100_memory.py`
- Modify: `native/cuda/mgt_cuda/a100_training_graph.cuh`
- Modify: `native/cuda/a100_training_graph.cu`
- Modify: `native/tests/cuda/test_a100_training_graph_8rank.cu`
- Modify: `native/tools/mgt_bn_step_benchmark.cu`

**Interfaces:**

- Produces canonical `accepted_execution_profile.json`, `policy.json`, `algorithm_table.json`, `static_arena.json`, `matrix.jsonl`, `summary.json`, Nsight artifacts, and a rejected-candidate ledger.
- The production runner consumes only the accepted execution-profile bytes with exact source/binary/hardware/software/workload/gate fingerprints.

- [ ] Fingerprint:

  ```text
  ordered GPU UUIDs, names, SM versions, and 40-GB memory identity
  GPU count, total-memory identity, application clocks, power limit, and persistence policy
  observed free memory plus required domain bytes as launch evidence (gate `free >= required`, but do not hash transient free bytes)
  CUDA driver/runtime
  cuBLASLt version
  NCCL version
  CUTLASS source identity
  algorithm-table SHA256 and every checked algorithm record
  BN collective backend plus NCCL Device API capability
  graph-support/capture fingerprint
  prepared-step binary SHA256
  compiler and Release flags
  CMAKE_CUDA_ARCHITECTURES=80
  topology matrix
  source Git SHA
  full/tail row vectors
  model dimensions
  ```

- [ ] Separate behavior by executable. The tuner may reuse a byte-identical accepted candidate as prior evidence or run the complete matrix when explicitly invoked. The production runner never tunes: exact accepted-profile match starts the startup transaction; any mismatch exits with a stable code.

- [ ] Use this bounded coarse-to-fine search:

  ```text
  Stage A: exact cuBLASLt/CUTLASS choice, workspace, split-K contract, and row padding per GEMM key
  Stage B: input gather vs BF16 GEMM; explicit vs triggered implicit one-hot; tiles 8,12,16,24,32,48,56,64,72; one-SUM vs tiled-SUM
  Stage C: BN row chunks 256,512,1024; feature tiles 32,64; explicit FP32-xhat vs BF16-xhat
  Stage D: serial vs protected concurrent dW/dX; dZ ring 2,3,4; tail/input reductions
  Stage E: exact host-NCCL vs Device-LSA candidate when triggered
  Stage F: exact eager, compute+BN graph, or full-multistream graph candidate
  Stage G: top eight deduplicated full-profile interaction candidates
  ```

- [ ] After Tasks 10-12 freeze eager kernels, collective backend, arena, and real-trainer buffers, invalidate every earlier graph. For each graph candidate perform exactly two collective capture sessions: full with all ranks at 12500, and tail with ranks 0-1 at 12498 while ranks 2-7 are simultaneously at 12497. Compare named eager, compute+BN-graph, and full-multistream-graph candidate profiles through the shared prepared-step API.

- [ ] Build each graph through a logical node registry with stable role/site/stream/sequence labels. Hash the canonical node types and labeled dependency edges, never CUDA pointer values or transient node handles. Production recaptures the two sessions, reconstructs the same manifests, and requires both hashes to match before graph instantiate/replay.

- [ ] A graph candidate requires checksum equality with the named eager candidate, 10,000 full/full/tail replays without deadlock, identical collective sequence, host launch gaps `<2%`, paired Q50 improvement `>=3%`, and no diagnostic Q95/tail regression. Unsupported capture rejects the graph candidate. A separately measured eager candidate remains eligible for sealing; this decision occurs only in the tuner, never during production startup/replay.

- [ ] Use four bounded cost tiers; do not spend five production A/B pairs on every raw cuBLASLt candidate:

  ```text
  micro screening:       5 warmups + 20-30 measurements, one A100, retain top-K compatible algorithms
  component screening:   5 warmups + 30 steps, one fresh 8-rank A/B pair, retain 2-3 per component
  interaction screening: 20 warmups + 50 steps, three fresh A/B pairs, retain top eight policies
  final acceptance:       20 warmups + 100 steps, five fresh A/B pairs, full and final cases
  every tier: canonical snapshot reload, correctness first, failures/rejections recorded
  every pair: fresh processes, adjacent sides, alternating order, identical snapshot/source/runtime tree
  primary Q50: max-rank region-average timing
  diagnostic Q95: separate max-rank per-step instrumentation run
  ```

- [ ] Candidate promotion:

  ```text
  all correctness/stress gates pass
  full Q50 <= 0.97 * paired baseline
  full diagnostic Q95 <= paired baseline diagnostic Q95
  final throughput >= 0.98 * paired baseline
  peak memory <= 30 GiB per rank
  no rank timeout, NCCL error, nonfinite value, or checksum instability
  ```

- [ ] Select by epoch score:

  ```text
  epoch_score_ms = 9 * full_Q50_ms + final_Q50_ms
  ```

  Within `0.5%` of the best score, choose lower combined p95, then lower memory, then lexicographically lower policy ID.

- [ ] Run BF16 strong-scaling diagnostics at fixed global batch `100000` for `1,2,4,8` GPUs using Task 1's rank-independent snapshot, exact balanced partitions, and the complete world-specific diagnostic profile/algorithm table prepared in Tasks 3 and 5; plus weak scaling at `12500` rows/rank. Missing key/profile or OOM is an explicit `not_runnable` result and exits that diagnostic; never borrow the world-8 accepted profile, reduce batch, or change snapshot. Only world-8 full/tail evidence can seal production.

- [ ] Run Nsight Systems in this order: Task-1 strict shared step, first complete eager BF16 step, final eager winner, then final graph candidate/winner. The final report must show:

  ```text
  max-rank critical path
  GPU busy/idle
  exposed versus overlapped NCCL
  host launch gaps
  graph replay
  compute/auxiliary/weight/generation stream overlap
  exact collective sequence
  zero ordinary-step cudaDeviceSynchronize/cudaStreamSynchronize
  zero synchronous D2H and runtime allocation/free
  zero descriptor/heuristic creation inside warm steady state
  no blocking host polling
  68 mandatory BN edges identified explicitly
  top exposed costs ranked by max-rank wall time
  ```

- [ ] Nsight Compute profiles only representative kernels that Nsight Systems proves material, one invocation per kernel class:

  ```text
  2560x224 BF16 GEMM
  224x224 BF16 GEMM
  BF16 input-gradient GEMM
  tiled BN forward moments
  tiled BN backward moments
  BN forward/backward epilogues
  ```

  Collect hierarchical Tensor Roofline, memory workload, occupancy, warp state, BF16 Tensor Core instructions, global ATOM/RED instructions, DRAM throughput, and duration. Join each capture to the exact algorithm-table key/choice, production call count, and Task-3 FLOP formula. Do not run whole-pipeline NCU. `ERR_NVGPUCTRPERM` or a missing required counter yields `blocked`; wall-time/correctness exploration may continue, but utilization percentages and final no-atomic acceptance remain unavailable.

- [ ] Overall acceptance targets:

  ```text
  minimum useful milestone: <= 28 ms full step
  production target: 20-24 ms
  stretch target: 15-20 ms
  GPU busy >= 90%
  host launch gaps < 2%
  all eligible GEMMs use BF16 Tensor Cores
  BF16 input-gradient GEMM >= 60% of one-A100 312-TFLOP/s dense BF16 peak
  aggregate eligible BF16 GEMMs >= 50% peak while those GEMMs are active
  scalar N=1 head excluded from Tensor-Core-eligible denominator
  exposed NCCL <= 20% of step
  ```

- [ ] Seal exactly one winning candidate only after every required full/tail gate passes. Copy candidate bytes unchanged, add immutable evidence hashes and `profile_state=accepted`, recanonicalize once, and cross-check policy/table/arena/graph hashes. Never synthesize a winner from fields of multiple candidates. Unit tests cover matrix completeness, screening tiers, deduplication, caps, paired ordering, max-rank quantiles, Q95 separation, every fingerprint drift, stale profile, exact tie-breaking, failed rows, OOM, timeout, malformed logs, graph invalidation, collective-capture sessions, production rejection of candidate/mutated profiles, no fallback symbol path, and absence of FP16 candidates.

- [ ] Commit:

  ```powershell
  git add scripts/autotune_8xa100_bf16.py scripts/tests/test_autotune_8xa100_bf16.py scripts/cluster/run_a100_bf16_autotune.sbatch scripts/cluster/run_a100_bf16_nsys.sbatch scripts/cluster/run_a100_bf16_ncu.sbatch scripts/poll_a100_memory.py scripts/tests/test_poll_a100_memory.py scripts/summarize_a100_bf16_benchmark.py native/cuda/mgt_cuda/a100_training_graph.cuh native/cuda/a100_training_graph.cu native/tests/cuda/test_a100_training_graph_8rank.cu native/tools/mgt_bn_step_benchmark.cu
  git commit -m "perf: autotune bf16 training on 8xa100"
  ```

---

### Task 14: Accept Time-to-Quality, Not Just a Fast Microbenchmark

**Files:**

- Create: `scripts/cluster/run_a100_bf16_stability.sbatch`
- Create: `scripts/cluster/run_a100_bf16_full_training.sbatch`
- Create: `test_results/a100_bf16_acceptance.md`
- Create: `test_results/a100_bf16_execution_profile.json`
- Create: `test_results/a100_bf16_policy.json`
- Create: `test_results/a100_bf16_static_arena.json`
- Create: `test_results/a100_bf16_algorithm_table.json`
- Create: `test_results/a100_bf16_autotune_summary.json`
- Create: `test_results/a100_bf16_training_summary.json`
- Create: `test_results/a100_bf16_puzzle0_compare.json`

**Interfaces:**

- Consumes the exact accepted execution profile through `mgt_a100_bf16_production_runner`; the tuner and strict oracle are separate comparison processes.
- Produces reproducible performance, convergence, checkpoint/resume, and puzzle-quality evidence.

- [ ] Run a 1000-step stress with full/full/tail cycling, per-rank delay injection, periodic NCCL async-error polling, finite checks, exact collective counters, and same-policy repeated checksums.

- [ ] Run a 10,000-step training-behavior comparison against strict FP32 from matched initialization/data keys:

  ```text
  every loss finite
  no BN running-state divergence
  BF16 final loss <= 1.05 * strict final loss
  BF16 median loss over final 1000 steps <= 1.05 * strict
  parameter norm ratio in [0.9,1.1]
  checkpoint/resume matches uninterrupted BF16
  ```

- [ ] Freeze accepted-profile bytes/SHA, source and binary SHA, policy SHA, algorithm-table SHA, static-arena SHA, collective schedule/backend/topology fingerprint, graph capture recipe and canonical node/edge manifests, environment manifest, and checkpoint format before the full run. Re-run the production startup transaction before stability and full training. Any change invalidates the evidence; production never retunes or substitutes.

- [ ] Complete the original training workload:

  ```text
  32692 semantic epochs
  326920 optimizer steps
  32.69 billion generated examples
  ```

- [ ] Evaluate puzzle 0 with depth `100` and beam `10,000,000` using identical search settings for:

  ```text
  best available original model
  final BF16-trained folded export
  ```

  Require both to run in the same comparison job. The final model must solve; its solution length must be no more than two moves longer than the best solved original.

- [ ] `a100_bf16_acceptance.md` reports:

  ```text
  exact source/policy/environment hashes
  full/tail Q50/Q95 and samples/s
  useful/issued/Tensor-Core-eligible FLOPs separately
  per-kernel BF16 Tensor Core metrics, or explicit counters-unavailable status with no utilization claim
  exposed NCCL and overlap
  peak memory
  strong/weak scaling
  convergence and resume
  full training wall time
  puzzle comparison
  accepted and rejected candidates
  exact SLURM job IDs and artifact paths
  ```

- [ ] Run final gates:

  ```powershell
  py -m unittest scripts.tests.test_summarize_a100_bf16_benchmark scripts.tests.test_autotune_8xa100_bf16 -v
  cmake --build native/build-cpu-codex --config Release --parallel
  ctest --test-dir native/build-cpu-codex -C Release --output-on-failure --no-tests=error
  git diff --check
  ```

- [ ] Commit:

  Stage only the compact evidence below.

  ```powershell
  git add -f test_results/a100_bf16_acceptance.md test_results/a100_bf16_execution_profile.json test_results/a100_bf16_policy.json test_results/a100_bf16_algorithm_table.json test_results/a100_bf16_static_arena.json test_results/a100_bf16_autotune_summary.json test_results/a100_bf16_training_summary.json test_results/a100_bf16_puzzle0_compare.json
  git add scripts/cluster/run_a100_bf16_stability.sbatch scripts/cluster/run_a100_bf16_full_training.sbatch
  git commit -m "test: record 8xa100 bf16 training acceptance"
  ```

## Stop-the-Line Conditions

Stop the current numbered task, preserve the last green commit, and report the exact failing gate when:

- any production code changes backend, precision, algorithm, tile, workspace, graph mode, xhat mode, input reduction, or collective implementation from the sealed profile;
- a candidate is faster only on rank 0 but slower by max-rank wall time;
- collective order differs across ranks or any rank hangs;
- final-batch rows are padded, duplicated, dropped, or counted in BN/loss incorrectly;
- a hot-loop allocation, handle creation, host synchronization, or synchronous readback appears;
- an accepted application/custom kernel contains any global atomic, or a selected closed-library GEMM reports a nonzero global ATOM/RED counter;
- running statistics, gradients, moments, padding, or BF16 mirror become nonfinite;
- numerical acceptance requires weakening the strict FP32 oracle;
- execution-profile, policy, algorithm-table, arena, graph, source, binary, library, hardware, or topology identity does not match the exact runtime;
- a performance claim lacks paired source-identical full and tail runs;
- NCU counters are unavailable but a hardware-utilization percentage is being claimed;
- a device data/health fault lets one rank exit before all healthy communicators drain the same suffix, or an NCCL transport fault fails to trigger coordinated communicator abort;
- source changes after full-training evidence is frozen.

## Expected Outcome

The implementation should first reach a reliable BF16 production path around `20-28 ms/step`, then use row-tiled BN, protected communication overlap, final graphs, measured CUTLASS fusion, and topology-gated device LSA to pursue `15-20 ms/step`. The plan does not promise 70-100% end-to-end BF16 peak because exact global SyncBN, communication, elementwise work, and Adam cannot execute on Tensor Cores. It instead requires:

```text
high BF16 Tensor Core utilization inside every eligible GEMM
minimum max-rank wall time
minimum exposed communication
no healthy-path CPU stalls or contended atomics
unchanged global SyncBN contract
successful full training and puzzle-quality gate
```

## Self-Review Checklist

- [x] The optimized path is BF16-only; FP16 is excluded from A100 policy search.
- [x] Strict FP32 remains an immutable oracle.
- [x] Mandatory global BN dependencies are distinguished from removable synchronization.
- [x] Input forward and backward both have Tensor Core candidates.
- [x] Hot-path atomics, memsets, conversion passes, and host synchronizations have explicit removal tasks.
- [x] Resource ownership and collective order are explicit.
- [x] Full and unequal final batches are covered.
- [x] Autotuning is automatic, bounded, paired, and fail-closed.
- [x] Nsight Systems precedes per-kernel Nsight Compute.
- [x] Performance is subordinate to correctness and time-to-quality.
- [x] Production trainer, checkpoint/resume, export, and puzzle quality are included.
- [x] A100 results remain isolated from the 2xT4 production line.
- [x] Benchmark, eager trainer, and graph wrapper share one prepared-step entry point.
- [x] Snapshot identity is independent of world/rank partitioning.
- [x] Accepted runtime loads a hashed algorithm table and performs no startup timing.
- [x] Long gradient collectives cannot delay the 68 critical BN collectives.
- [x] Final graph capture occurs only after eager kernels, communication backend, and production buffers are frozen.
- [x] Candidate tuning and production execution are separate entry points; production has no runtime fallback or retuning path.
- [x] The sealed profile fixes every backend, algorithm key, arena slice, stream/event edge, collective order, and graph session.
- [x] BF16 dZ is distinguished from FP32 linear dX/upstream/residual accumulation.
- [x] Input bias is covered by the non-input-table tail reduction.
- [x] Tail NCCL waits on both all-BN-done and tail-writers-done.
- [x] Full and unequal-tail NCCL graphs use exactly two collective capture sessions.
- [x] Health remains typed; data faults drain a rank-uniform suffix, while transport faults use coordinated communicator abort.
- [x] Every material NCU claim is tied to an exact algorithm-table key and explicit issued-FLOP formula.

## Execution Handoff

Implement Tasks 1-13 sequentially in the primary agent without subagents, as requested. At each task boundary perform an explicit spec/correctness self-review, preserve the last green commit, and do not start the next task until listed gates pass. Run Task 14 only after source, binary, accepted execution profile, algorithm table, arena, collective backend, and graph capture manifests are frozen. At every task boundary report:

```text
task and commit
focused correctness gates
broader gates
full/tail paired performance
accepted/rejected decision
remaining blocker
next task
```
