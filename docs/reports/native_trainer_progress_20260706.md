# Native trainer progress, 2026-07-06

This report records the state after the first step of the six-point optimization plan in
`docs/plans/native_trainer_optimization_plan_20260706.md`.

## Scope

- Target model: archive-shaped p888 MLP, output_dim=1.
- Data path: online random walks, no shared dataset on disk.
- Fast path: half activations/weights for linear layers, cuBLASLt/CUTLASS-backed GEMM wrappers, tiled one-hot input-gradient GEMM.
- Multi-GPU path: NCCL allreduce by deterministic buckets with overlap enabled.

## Code changes verified

- Added the tracked six-point optimization plan.
- Added the `b53248_t48_ws16m_profile` Kaggle sweep row for stage-level timing.
- Increased the cuBLASLt matmul plan cache capacity from 64 to 256 entries.
- Made cuBLASLt descriptor cache fallback safe: if no workspace heuristic is available, descriptors stay cached and the wrapper uses the default no-workspace cuBLASLt algorithm instead of failing.
- Rejected the first 32-candidate heuristic picker attempt because it regressed local throughput.

## Verification

Local Docker CUDA checks:

- Build target: `test_cuda_train_step_smoke`, `test_cuda_mlp_backward_mixed_precision_error`, `mgt_native_train`.
- CTest filter: `cuda_train_step_smoke|cuda_mlp_backward_mixed_precision_error`.
- Result: 2/2 tests passed.

Kaggle v10 checks:

- Kernel status: complete.
- Git revision: `9aff99eeff0659451dea19e3545e1f92d4dc6922`.
- GPU preflight: 2 x Tesla T4.
- CTest: 17/17 tests passed.
- Sweep rows: 27 ok, 0 failed.
- Loss log scan: 432 finite loss records, no non-finite values, observed range 292.668 to 297.923.

Kaggle v11 autotune checks:

- Kernel status: complete.
- Git revision: `9c1f6d6649aa2eb005925476d85f0e8737a11205`.
- GPU preflight: 2 x Tesla T4.
- CTest: 17/17 tests passed.
- Sweep rows: 4 ok, 0 failed.
- Loss log scan: 64 finite loss records, no non-finite values, observed range 293.116 to 297.344.

Kaggle v12 residual-subprofile checks:

- Kernel status: complete.
- Git revision: `0cde775084697fa6b701e4777596aeb0bbefb23c`.
- GPU preflight: 2 x Tesla T4.
- CTest: 17/17 tests passed.
- Sweep rows: 3 ok, 0 failed.
- Loss log scan: 48 finite loss records, no non-finite values, observed range 293.116 to 297.344.

## Performance

Local one-GPU run after the safe cuBLASLt fallback:

| Run | Batch | Tile | Throughput, states/s | Step ms | Backward ms | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `local_auto_b24576_tile48_ltcache_safe_ws16m_20260706` | 24576 | 48 | 280131.49 | 88.17 | 86.07 | No proven speedup; noisy and below the earlier local tile48 default run. |

Kaggle two-T4 runs:

| Run | Commit | Best config | Throughput, states/s | Step ms | Backward ms | Allreduce ms | Notes |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| v9 tile48 profile | `55a60e6` | `b53248_t48_ws16m_bucket4m` | 457828.31 | 232.65 | 213.89 | 28.16 | Noisy allreduce tail; useful profile row. |
| v10 safe cuBLASLt cache | `9aff99e` | `b53248_t56_ws16m_bucket4m` | 513110.86 | 207.55 | 188.65 | 15.17 | Best observed 2xT4 result so far, but treat as no-regression plus slight observed improvement because Kaggle run-to-run noise is material. |
| v11 cuBLASLt autotune | `9c1f6d6` | `b53248_t56_auto` | 518031.40 | 205.58 | 187.01 | 15.97 | New best observed 2xT4 result; +0.96% over v10 best, still small enough to treat as measured improvement rather than a guaranteed universal gain. |
| v12 residual subprofile | `0cde775` | `b53248_t56_auto` | 521582.82 | 204.19 | 185.28 | 16.06 | New best observed 2xT4 result; +0.69% over v11 best, with the same caveat about Kaggle run-to-run noise. |

Kaggle v10 top configs:

| Config | Throughput, states/s | Step ms | Backward ms | Allreduce ms |
| --- | ---: | ---: | ---: | ---: |
| `b53248_t56_ws16m_bucket4m` | 513110.86 | 207.55 | 188.65 | 15.17 |
| `b53248_t48_ws16m_bucket4m` | 510093.78 | 208.78 | 189.97 | 16.23 |
| `b53248_t64_ws16m_bucket4m` | 508577.42 | 209.41 | 190.58 | 16.94 |
| `b53248_t48_ws16m_profile` | 508355.44 | 209.50 | 191.09 | 16.47 |
| `b53248_t72_ws16m_bucket4m` | 506326.15 | 210.33 | n/a | n/a |

Do not switch the default tile only from this evidence. Across v8 and v10, tile 56 is slightly ahead on average, but the margin is small enough that a repeated focused sweep is needed.

Kaggle v11 autotune configs:

| Config | Throughput, states/s | Step ms | Backward ms | Allreduce ms |
| --- | ---: | ---: | ---: | ---: |
| `b53248_t56_auto` | 518031.40 | 205.58 | 187.01 | 15.97 |
| `b53248_t48_auto` | 516508.34 | 206.19 | 187.13 | 15.25 |
| `b53248_t48_auto_profile` | 513811.17 | 207.27 | 188.57 | 14.71 |
| `b53248_t64_auto` | 510361.21 | 208.68 | 189.53 | 16.91 |

Kaggle v12 residual-subprofile configs:

| Config | Throughput, states/s | Step ms | Backward ms | Allreduce ms |
| --- | ---: | ---: | ---: | ---: |
| `b53248_t56_auto` | 521582.82 | 204.19 | 185.28 | 16.06 |
| `b53248_t56_auto_profile_substage` | 518069.25 | 205.57 | 186.95 | 15.11 |
| `b53248_t48_auto_profile_substage` | 515005.61 | 206.79 | 188.27 | 14.72 |

## Stage profile

Kaggle v10 profile row `b53248_t48_ws16m_profile`:

| Stage | Avg ms |
| --- | ---: |
| Input forward | 27.02 |
| Hidden forward | 2.99 |
| Residual forward | 36.51 |
| Output | 4.67 |
| Residual backward | 60.78 |
| Hidden backward | 14.44 |
| Input gradient | 45.36 |
| Allreduce | 16.47 |
| Adam | 1.88 |

Main compute bottlenecks are now residual backward, input-gradient GEMM, residual forward, and input forward. The communication tail is visible but smaller in the v10 best run than in v9.

Kaggle v11 profile row `b53248_t48_auto_profile`:

| Stage | Avg ms |
| --- | ---: |
| Input forward | 27.30 |
| Hidden forward | 2.96 |
| Residual forward | 36.89 |
| Output | 4.38 |
| Residual backward | 61.03 |
| Hidden backward | 14.23 |
| Input gradient | 42.99 |
| Allreduce | 14.71 |
| Adam | 1.89 |

Compared with the v10 tile48 profile, autotune mainly improved input gradient from 45.36 ms to 42.99 ms and output from 4.67 ms to 4.38 ms. Residual backward stayed the dominant compute stage at about 61 ms.

Kaggle v12 residual-subprofile rows:

| Config | Residual backward ms | FC2 dZ ms | FC1 dZ ms | Input gradient ms |
| --- | ---: | ---: | ---: | ---: |
| `b53248_t48_auto_profile_substage` | 63.57 | 13.63 | 10.89 | 40.81 |
| `b53248_t56_auto_profile_substage` | 63.31 | 13.59 | 10.88 | 40.02 |

The two-T4 profile confirms that residual backward is still the first target. The activation-gradient kernels alone cost about 24.5 ms per profiled step on T4, but they are memory-heavy and launch-count-sensitive; naive two-dimensional tiling did not help locally.

## Next work

1. Treat tile 56 as the current measured 2xT4 default candidate, but do not hard-code it into the engine; it remains a shape/runtime-tuned value.
2. Attack residual backward first, because v12 shows about 63 ms there on T4 and the two activation-gradient kernels alone cost about 24.5 ms.
3. Keep input-gradient GEMM as the second target; autotune helped, but this stage still costs about 40 ms in the v12 profile rows.
4. Move repeated residual-backward launch structure toward CUDA Graph capture or real kernel fusion; naive residual dZ tiling was measured and rejected.
5. Continue the CUTLASS path only where it gives more control than cuBLASLt autotune for a shape family, not as a blind replacement.
6. Keep correctness checks tied to every speed step: CTest, finite-loss scan, and at least one profile row.

## cuBLASLt autotune slice

Implemented after the v10 report:

- Added `LtMatmulAutotuneConfig` and public `ConfigureLtMatmulAutotune` / `CurrentLtMatmulAutotuneConfig` API.
- Each cuBLASLt plan now stores checked heuristic candidates for the exact GEMM shape key: A/B/C layout sizes, transpose flags, precision path, and workspace budget.
- Optional warmup/timing autotune runs only when `beta == 0`, so repeated trial GEMMs overwrite C instead of accumulating into it.
- The first valid heuristic remains the default when autotune is disabled.
- Runner flags and Kaggle env knobs were added: `--lt-autotune`, `--lt-autotune-candidates`, `--lt-autotune-warmups`, `--lt-autotune-iters` and matching `MGT_LT_AUTOTUNE*` variables.
- Logs, `profile.jsonl`, `metadata.env`, and Kaggle sweep summaries now record the autotune settings.

Fresh local verification:

- Docker build target: `test_cuda_train_step_smoke`, `test_cuda_mlp_backward_mixed_precision_error`, `mgt_native_train` passed.
- CTest filter `cuda_train_step_smoke|cuda_mlp_backward_mixed_precision_error`: 2/2 passed.
- `bash -n` for `kaggle/kernel/run_ranks_2xt4.sh` and `kaggle/kernel/run_sweep_2xt4.sh`: passed.

Local one-GPU comparison, same build and workload: world_size=1, batch=24576, tile=48, workspace=16 MiB, steps=8.

| Run | Autotune | Throughput, states/s | Step ms | Backward ms | Notes |
| --- | ---: | ---: | ---: | ---: | --- |
| `local_lt_autotune_off_b24576_t48_20260706` | 0 | 188776.76 | 133.61 | 130.34 | Noisy, first off run. |
| `local_lt_autotune_off_b24576_t48_repeat_20260706` | 0 | 174689.28 | 142.75 | 139.56 | Repeated off run confirms bad first heuristic locally. |
| `local_lt_autotune_on_b24576_t48_20260706` | 1 | 278586.46 | 88.82 | 86.65 | Same workload with per-shape timing selection. |

Interpretation: local autotune is materially better than the first cuBLASLt heuristic for this machine and this shape family. Kaggle v11 confirms the same direction on 2xT4, but with a smaller observed gain: 518031 states/s for the v11 best row versus 513111 states/s for the v10 best row.

## Residual backward subprofile slice, 2026-07-07

Implemented after the v11 autotune slice:

- Added residual-backward substage timers to `MlpBackwardProfile` without changing the non-profiled training path.
- Exported these fields to `profile.jsonl` and to the rank-wrapper `throughput_summary.json` aggregation.
- The profiled residual-backward total is now the sum of its measured substages, which makes the next optimization target visible.

Fresh local verification:

- Docker build target: `test_cuda_train_step_smoke`, `test_cuda_mlp_backward_cpu_compare`, `test_cuda_mlp_backward_mixed_precision_error`, `mgt_native_train` passed.
- CTest filter `cuda_mlp_backward_cpu_compare|cuda_mlp_backward_mixed_precision_error|cuda_train_step_smoke`: 3/3 passed.
- `bash -n` for `kaggle/kernel/run_ranks_2xt4.sh` and `kaggle/kernel/run_sweep_2xt4.sh`: passed.
- `git diff --check`: passed.

Local one-GPU no-profile check, same p888 workload as before: world_size=1, batch=24576, tile=48, workspace=16 MiB, cuBLASLt autotune enabled.

| Run | Steps | Throughput, states/s | Step ms | Backward ms | Notes |
| --- | ---: | ---: | ---: | ---: | --- |
| `local_residual_subprofile_wrapper_b24576_t48_20steps_20260707` | 20 | 309296.17 | 79.60 | 77.37 | Average over steps 1..19. |
| same run, last 10 steady steps | 10 | 317724.91 | 77.43 | 75.13 | Better warm-state estimate on the laptop GPU. |

The earlier 8-step checks in this session were much noisier because the local laptop GPU started from low clocks; the 20-step wrapper run is the more useful local no-regression check. Loss stayed finite: 19 steady records, observed range 291.078827 to 296.950378, non-finite count 0.

Local one-GPU profile wrapper row: `local_residual_subprofile_wrapper_profile_b24576_t48_20260707`, batch=24576, tile=48, profile enabled. Profile mode adds synchronization and is for attribution, not headline throughput.

| Residual backward substage | Avg ms |
| --- | ---: |
| FC2 dZ | 5.23 |
| FC2 grad weight | 3.95 |
| FC2 bias | 2.10 |
| FC2 backprop | 3.38 |
| FC1 dZ | 4.14 |
| FC1 grad weight | 3.51 |
| FC1 bias | 1.91 |
| FC1 backprop | 3.13 |
| Skip add | 3.63 |
| Residual backward total | 30.97 |

This points to the next real optimization slice: reduce residual-backward launch count and memory traffic first. The largest individual buckets are the two activation-gradient kernels, then weight-gradient GEMMs, then skip add. The previous bias-gradient and beta=1 skip-add attempts were measured and rejected because they slowed the non-profiled path.

## Rejected residual dZ tiling attempts, 2026-07-07

After the v12 Kaggle run, three local residual activation-gradient variants were tested against the stable one-GPU baseline. All three kept correctness but lost throughput, so the code was restored to the `0cde775` residual-backward path.

Stable local baseline:

| Run | Throughput, states/s | Step ms | Backward ms | Notes |
| --- | ---: | ---: | ---: | --- |
| `local_residual_subprofile_wrapper_b24576_t48_20steps_20260707` | 309296.17 | 79.60 | 77.37 | Average over steps 1..19. |
| same run, last 10 steady steps | 317724.91 | 77.43 | 75.13 | Warm-state estimate. |

Rejected local variants:

| Variant | Throughput, states/s | Step ms | Backward ms | Result |
| --- | ---: | ---: | ---: | --- |
| Fixed `128x2` tiled residual dZ | 288679.62 | 85.16 | 83.08 | Rejected; slower than baseline. |
| Adaptive `224x2` residual dZ | 302565.30 | 81.35 | 79.10 | Rejected; still slower than baseline. |
| Adaptive `224x1` residual dZ | 283213.27 | 87.03 | 84.66 | Rejected; worst of the tested variants. |

Correctness verification for the adaptive row-1 variant: CTest filter `cuda_mlp_backward_cpu_compare|cuda_mlp_backward_mixed_precision_error|cuda_train_step_smoke` passed 3/3. The performance result is the blocker, not numerical correctness.

Conclusion: do not continue with naive residual dZ launch tiling. The next higher-probability optimization is either CUDA Graph capture for the repeated residual-backward launch chain, or a real fused residual-backward kernel that removes multiple global-memory round trips instead of only changing the launch geometry.
## Rejected backward CUDA Graph attempt, 2026-07-07

A local optional `--backward-graph` path was prototyped for the single-GPU, no-profile, no-callback case. The first step used the normal launcher to warm cuBLASLt plans, then later steps captured and replayed the backward graph. Correctness was fine, but throughput did not improve, so the source change was reverted.

Verification before rejection:

- Docker build target: `mgt_native_train_smoke`, `test_cuda_train_step_smoke`, `test_cuda_mlp_backward_mixed_precision_error` passed.
- CTest filter `cuda_mlp_backward_mixed_precision_error|cuda_train_step_smoke|native_train_backward_graph_smoke`: 3/3 passed.
- Both A/B runs ended with the same final loss: `295.444`.

Local A/B workload: world_size=1, batch=24576, tile=48, workspace=16 MiB, cuBLASLt autotune disabled, steps=12.

| Variant | Window | Throughput, states/s | Step ms | Backward ms | Graph captured | Loss range |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| Baseline | steps 1..11 | 289217.57 | 84.97 | 82.85 | 0 | 291.078827..296.814453 |
| Baseline | last 6 | 289524.53 | 84.88 | 82.75 | 0 | 291.078827..295.443909 |
| Backward graph | steps 1..11 | 282123.40 | 87.11 | 85.00 | 1 | 291.078827..296.814453 |
| Backward graph | last 6 | 288477.23 | 85.19 | 83.06 | 1 | 291.078827..295.443909 |

Conclusion: plain CUDA Graph capture around the current backward launcher is not worth keeping. It does not remove enough work because the heavy stages are still cuBLASLt GEMMs and memory-heavy activation-gradient kernels. The next useful graph/fusion work should be lower-level: either capture a larger steady-state step with communication constraints solved, or fuse residual-backward kernels so the graph reduces real memory traffic and launch count instead of only wrapping the existing sequence.
## Rejected half-only residual skip attempt, 2026-07-07

A half-only residual `fc2 dz` path was prototyped to avoid writing the `dzfc2` float matrix in the half-linear path. The prototype added a separate half skip buffer, computed `fc2` bias with a one-owner-per-column half reduction, and performed skip-add from half. It used no atomics and kept owner writes deterministic.

Correctness verification before rejection:

- Docker build target: `test_cuda_mlp_backward_mixed_precision_error`, `test_cuda_train_step_smoke`, `mgt_native_train_smoke` passed.
- CTest filter `cuda_mlp_backward_mixed_precision_error|cuda_train_step_smoke`: 2/2 passed.

Local A/B workload: world_size=1, batch=24576, tile=48, workspace=16 MiB, cuBLASLt autotune disabled, steps=12.

| Variant | Window | Throughput, states/s | Step ms | Backward ms | Loss range |
| --- | --- | ---: | ---: | ---: | --- |
| Baseline | steps 1..11 | 289217.57 | 84.97 | 82.85 | 291.078827..296.814453 |
| Baseline | last 6 | 289524.53 | 84.88 | 82.75 | 291.078827..295.443909 |
| Half-only skip prototype | steps 1..11 | 259396.47 | 94.74 | 92.30 | 291.098694..296.816467 |
| Half-only skip prototype | last 6 | 268169.59 | 91.64 | 89.33 | 291.098694..295.474365 |

Conclusion: replacing the float `dzfc2` path with half-only bias and skip kernels made the residual backward path slower. The likely issue is that the custom bias reduction and extra half buffer cost more than the saved float write/read on this shape. The source change was reverted. The next fusion attempt should avoid adding another reduction kernel; it needs to combine existing elementwise work with a downstream operation or target a larger GEMM-side layout issue.

## Fused residual fc1 epilogue, 2026-07-07

Implemented a first real GEMM-side fusion for the half-linear training path:

- residual `fc1` forward now uses a cached cuBLASLt half-output GEMM with fused `bias + ReLU` epilogue when `hd2` is aligned;
- the output buffer is written directly as half activation for the following `fc2` GEMM;
- backward `fc1 dZ` reads the ReLU mask from that half activation, so the fast path no longer needs the residual `fc1` float activation/preactivation write;
- the cuBLASLt bias epilogue is run through the transposed column-major view of the same row-major buffer, because cuBLASLt broadcasts bias over D rows;
- plan descriptors are cached by `(m, n, k, workspace_bytes)` and only the per-block bias pointer is updated in the hot path.

Correctness verification:

- Docker CUDA build `build-fused-epilogue-docker`: passed.
- Focused CTest filter `cuda_mlp_backward_smoke|cuda_mlp_backward_mixed_precision_error|cuda_train_step_smoke|native_train_profile_smoke`: 4/4 passed.
- Full CTest in the GPU container: 34/34 passed, including `native_train_output_dim3_smoke` and `native_train_output_dim128_smoke`.

Local one-GPU no-profile workload: world_size=1, batch=24576, tile=48, workspace=16 MiB, cuBLASLt autotune enabled, steps=20, steady window steps 1..19.

| Variant | Throughput, states/s | Step ms | Backward ms | Loss range |
| --- | ---: | ---: | ---: | --- |
| Previous stable baseline | 309296.17 | 79.60 | 77.37 | 291.078827..296.950378 |
| Fused fc1 epilogue, uncached descriptors | 310363.87 | 79.18 | 77.15 | 291.078827..296.950378 |
| Fused fc1 epilogue, cached descriptors | 309358.18 | 79.44 | 77.40 | 291.078827..296.950378 |

Profile-mode attribution for the cached path, batch=24576, tile=48, profile enabled, steps 1..7:

| Stage | Avg ms |
| --- | ---: |
| Residual forward | 9.99 |
| Residual backward total | 28.77 |
| FC1 dZ | 3.68 |
| FC1 grad weight | 3.06 |
| Input gradient | 26.23 |

Interpretation: the fusion is numerically safe and removes a real residual `fc1` global-memory pass, but the headline local throughput is neutral within noise. It is worth keeping as a structural step toward a fused residual block, not as a claimed throughput win. The next higher-impact fusion target is `fc2 dZ + residual skip/relu state` and/or moving more of the residual block to a controlled CUTLASS epilogue, because input gradient and residual backward still dominate.

## Rejected fused fc2 dZ plus bias owner-column attempt, 2026-07-07

A custom half-linear residual `fc2 dZ + bias` kernel was tested after the fused `fc1` epilogue. The kernel used one owner block per output column, wrote `dzfc2` and `dzfc2_half`, and reduced the column bias gradient in the same launch. It used no atomics and no extra reduction kernel.

Correctness verification before rejection:

- Docker CUDA build passed.
- CTest filter `cuda_mlp_backward_mixed_precision_error|cuda_train_step_smoke|native_train_profile_smoke`: 3/3 passed.

Local no-profile workload: world_size=1, batch=24576, tile=48, workspace=16 MiB, cuBLASLt autotune enabled, steps=20, steady window steps 1..19.

| Variant | Throughput, states/s | Step ms | Backward ms | Loss range |
| --- | ---: | ---: | ---: | --- |
| Fused fc1 epilogue cached baseline | 309358.18 | 79.44 | 77.40 | 291.078827..296.950378 |
| Owner-column fused fc2 dZ+bias | 195414.20 | 125.76 | 123.64 | 291.078827..296.950378 |

Conclusion: this fusion is numerically correct but structurally wrong for the hot path. The owner-column layout makes writes to `dzfc2`/half `dzfc2` strided by `hd2`, so the saved bias GEMV is overwhelmed by poor memory coalescing. Do not repeat column-owned dZ+bias fusion for this matrix layout. A future `fc2` fusion needs row/tile-coalesced writes and either a separate planned reduction or a GEMM/CUTLASS epilogue that preserves coalescing.

## Local input-gradient tile check after fc1 fusion, 2026-07-07

After the fused `fc1` epilogue, the input-gradient tile size was rechecked locally because input gradient remains the largest single stage. All runs used world_size=1, batch=24576, workspace=16 MiB, cuBLASLt autotune enabled, half input gradients, half linear weights, and finite identical loss range.

Profile-mode sequential checks, steps 1..11:

| Tile positions | Throughput, states/s | Step ms | Backward ms | Profiled input-grad ms |
| ---: | ---: | ---: | ---: | ---: |
| 36 | 283056.86 | 86.82 | 84.80 | 27.54 |
| 48 | 281440.81 | 87.32 | 85.30 | 26.50 |
| 80 | 267799.14 | 91.77 | 89.66 | 26.52 |

No-profile checks, steps 1..19:

| Tile positions | Throughput, states/s | Step ms | Backward ms |
| ---: | ---: | ---: | ---: |
| 36 | 298640.80 | 82.29 | 80.21 |
| 48 | 309358.18 | 79.44 | 77.40 |
| 56 | 308733.92 | 79.60 | 77.49 |

Conclusion: local best remains tile 48; tile 56 is effectively tied but slightly lower; tile 80 is not a good local setting even though the padded state length is 80. Keep tile size as a runtime/config parameter and tune per GPU class. Do not hardwire tile 80 into the input-gradient path.
