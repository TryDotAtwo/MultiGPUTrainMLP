# Production-shape convergence probe on 2xT4, 2026-07-22

- Kernel: `trydotatwo/native-multigpu-mlp-convergence-2xt4`, version 1.
- Status: `COMPLETE`; runner return code: `0`.
- Source revision: `d89cde1`.
- Hardware: exactly 2 x Tesla T4 with NCCL.
- Workload: real p888, full `2556 -> 218` model, 16 residual blocks, batch 57,344 per rank, 1,000 steps.
- Held-out set: 4,096 deterministic samples from a domain-separated seed, evaluated every 100 steps on both ranks.
- Rank evaluation series matched exactly and all values were finite.

## Result

| Completed steps | Held-out MSE |
| ---: | ---: |
| 0 | 298.067627 |
| 100 | 36.696354 |
| 200 | 35.686222 |
| 300 | 35.493256 |
| 400 | 35.419548 |
| 500 | 35.409969 |
| 600 | 35.402161 |
| 700 | 35.409233 |
| 800 | 35.424782 |
| 900 | 35.403160 |
| 1,000 | 35.433983 |

Relative baseline-to-final improvement: 88.11%. The best observed point was near step 600; constant `lr=1e-4` then plateaued and slightly regressed. A longer constant-LR run is therefore not accepted as useful full training.

Measured training throughput: 522,068 states/s average, 220.044 ms/step. Evaluation runs are outside the per-step timing.

Next gate: learning-rate schedule and LR/schedule sweep, followed by checkpointed production training.

Artifacts: `test_results/kaggle_convergence_2xt4_v1_20260722/`.
