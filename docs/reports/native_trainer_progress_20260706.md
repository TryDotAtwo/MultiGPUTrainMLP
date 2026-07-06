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

## Next work

1. Repeat the v11 autotune 2xT4 sweep once more if choosing a permanent default tile; tile 56 is ahead in v10 and v11, but by a small margin.
2. Attack residual backward first, because it remains the largest profiled compute stage.
3. Keep input-gradient GEMM as the second target; autotune helped, but this stage still costs about 43 ms in the profile row.
4. Move more repeated launch structure toward CUDA Graph capture after the residual backward path is stable.
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