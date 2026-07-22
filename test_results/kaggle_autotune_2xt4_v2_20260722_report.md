# Automatic 2xT4 production-shape autotune, 2026-07-22

- Kernel: `trydotatwo/native-multigpu-mlp-autotune-2xt4`, version 2.
- Status: `COMPLETE`; runner return code: `0`.
- Hardware contract: exactly 2 x Tesla T4, compute capability 7.5; driver 580.159.04.
- Workload: real p888 inputs, full model `2556 -> 218`, 16 residual blocks, K=1..29, NCCL world size 2.
- Method: staged sweep by one family at a time: input-gradient tile, then batch size, then LT workspace/allreduce bucket.
- Selection metric: highest successful minimum steady-step throughput; failed/non-finite rows are ineligible.
- Cache: schema v1, exact SHA-256 fingerprint, atomic publication, fail-closed mismatch, quick reuse drift threshold 85%.

## Selected configuration

- `MGT_BATCH_SIZE=57344`
- `MGT_INPUT_GRAD_POSITION_TILE=48`
- `MGT_LT_WORKSPACE_BYTES=16777216`
- `MGT_ALLREDUCE_BUCKET_BYTES=2097152`
- `MGT_INPUT_GRAD_SPARSE=0`
- `MGT_CUTLASS_HALF_GEMM_KINDS=input_embedding_grad,forward`
- `MGT_LT_AUTOTUNE=1`

Final combined stage winner: `ws16m_b2m`.

- Average throughput: 530,343.21 states/s.
- Minimum steady-step throughput: 523,732.54 states/s.
- Tile stage conservative winner: tile 48, 546,595.84 states/s minimum.
- Batch stage conservative winner: 57,344/rank, 540,927.55 states/s minimum.
- All 13 measured rows succeeded.

Artifacts: `test_results/kaggle_autotune_2xt4_v2_20260722/`.
