# Residual-gradient GEMM beta fusion audit

Date: 2026-09-01. Target: original p888, batch 4096, RTX 3070 Laptop /
SM86, CUDA Graph mode. Baseline source: `775ee0a0f11c067a8b3db92f763933da9aec2611`.

## Decision

Accepted for the generic residual-stack backward path. Each FC1 backward computes

`input_gradient = W^T * activation_gradient + residual_gradient`.

For an even residual depth, the implementation now writes the dX GEMM directly
into the residual-gradient buffer with `beta=1`, then swaps the two existing FP32
gradient-buffer roles. This removes the separate elementwise residual-add launch
without adding storage, copies, or graph nodes. Original p888 has 16 residual
blocks, so all 16 adds disappear.

The dW GEMM remains `beta=0`; only dX receives `beta=1`. The two dX operands stay
distinct. For an odd residual depth, the final block retains one legacy add so the
public API still returns the final gradient through its fixed `block_grad` pointer.
Zero-depth behavior is unchanged.

## Correctness and platform gates

All GPU work ran through `mgt-gpu-queue`; job IDs map to retained
`.gpu_queue/logs/<id>.log` files.

| Job | Outcome |
|---|---|
| `b6f7a73fdf37` | SM86 candidate build passed the one-block odd-tail test, all local-FP16 activation-tape/oracle cases including two-block ping-pong, and native full/tail/fail-stop graph tests. |
| `c77b72efa981` | Compute Sanitizer memcheck and initcheck passed the full local-FP16 activation-tape trainer; zero leaks and two zero-error summaries. |
| `5ae05416e982` | Exact Kaggle/T4 compile policy (`sm_75`, NCCL, CUTLASS AUTO `input_embedding_grad`) passed the odd/even oracle, native graph, and production p888 runtime on the available SM86 device. This is an SM75 code/runtime compatibility gate, not a T4 speed claim. |
| `747ecd5b3a5a` | Final SM86 regression passed 33/33 CUDA/NCCL/graph tests, explicit capture/update and full/tail/fail-stop graph checks, plus Rust FFI 1/1. The rebuilt benchmark stayed byte-identical to the frozen candidate. |

The independent rounded mixed-precision CPU oracle checks loss, outputs, all
weight and affine gradients, running statistics, live-row changes, overwrite
semantics, tape guards, scratch capacity, and cache lifetime. It covers no
residual blocks and two residual blocks; the standalone residual-stack test
covers one block and therefore the odd-depth compatibility tail.

Folding the addition into the GEMM epilogue preserves the mathematical FP32
accumulation contract but is not specified as bitwise-identical to a separate
kernel. The mixed-precision oracle and downstream optimizer gates, rather than
bitwise baseline identity, own correctness.

## Frozen binaries and paired timing

| Variant | Frozen path | SHA256 |
|---|---|---|
| Baseline | `/tmp/mgt-input-tape-mask-final` | `55c21ac62c3075c0eedc4d0567f44098705f0af1a902c1cd9e9eea7dc6018fc0` |
| Candidate | `/tmp/mgt-residual-beta1-bin` | `893e5be07f0884968a09011da761a7d96f4f2ba37bc352d64cc228dd4a993763` |

Job `f8aab4d3b74e` retained strict ABBAAB order. Every process checked the
corresponding hash before and after execution and used original p888, batch
4096, 140 warmup steps, 100 timed graph steps, and the unchanged 744001024-byte
arena. All active telemetry samples were 780 MHz.

| Variant | Run means, ms | Median, ms | Samples/s at median |
|---|---|---:|---:|
| Baseline | 20.8809, 20.7785, 20.7775 | 20.7785 | about 197,127 |
| Candidate | 20.4407, 20.3370, 20.3374 | 20.3374 | about 201,402 |

Latency falls by **2.122867%** and throughput rises by **2.168910%**.
The absolute median delta is **-0.4411 ms/step**.

- `paired-residual-beta1-final.jsonl` SHA256:
  `5f5ce306c039e2e50d98e1b4bed4b89c0f9ee015e191f7ef059c13ddcf5626c9`.
- `paired-residual-beta1-final-summary.json` SHA256:
  `d7f1fef0fb41d5dea57fe76f14aa981ca3ec9bbb44f90b4d8aee67c5dea2caee`.

Jobs `edaf8bc1f30d` and `7616c9db92b3` retained earlier attempts but were
rejected because active SM clocks crossed multiple DVFS states. They do not
contribute to the speed claim.

## Nsight Systems mechanism proof

Job `de736879520d` profiled baseline then candidate after 100 warmup steps. The
strict read-only analyzer selected four complete graph steps from each trace.

| Metric | Baseline | Candidate | Delta |
|---|---:|---:|---:|
| Mean summed GPU-kernel time, ms | 20.671056 | 20.219019 | -0.452036 |
| `AddInPlace` kernels/step | 16 | 0 | -16 |
| `AddInPlace` time/step, ms | 0.448910 | 0 | -0.448910 |
| GEMM kernels/step | 99 | 99 | 0 |
| GEMM time/step, ms | 5.819571 | 5.857474 | +0.037903 |

All six GEMM kernel identities and their counts remain unchanged. The fused
dX `beta=1` epilogue makes the 32 residual TN GEMMs about 0.039 ms/step more
expensive in aggregate, while removing the 0.449 ms standalone add family.
Nsight perturbs absolute timing; the unprofiled ABBAAB result owns the published
end-to-end percentage.

- Baseline SQLite SHA256:
  `84f63525a005f940d2b485a532f62e55eb7854abcb3cdd149618ab68b482e5bc`.
- Candidate SQLite SHA256:
  `c5b0a62e648394980ef6190bc3f161cc520a43f65149805345c47e418275c705`.
- Analyzer SHA256:
  `42ca7d0cdc48d5fec8a3ad0458b77ade3cc264acc825f4ca0dd7a5d1fae6eef2`.

## Boundary and next target

This checkpoint proves the single-GPU dataflow change, generic odd/even pointer
semantics, memory initialization, SM75 code compatibility, graph capture, and
SM86 speed. It does not claim T4 timing, multi-GPU scaling, or convergence.

The post-change low-clock Nsight ranking is led by the packed sparse input
gradient at about 3.606 ms, input gather at about 1.802 ms, FP16-mirror Adam at
about 1.403 ms, and the residual GEMM families. The rejected u32 sparse-address
experiment already showed no end-to-end win. The next material dataflow target
is therefore the still-materialized input gather/BN boundary or a larger sparse
algorithm change, not another legacy cuBLAS algorithm enum.
