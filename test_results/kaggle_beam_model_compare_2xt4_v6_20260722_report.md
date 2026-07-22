# p888 model beam comparison on 2xT4

- Kaggle kernel: `trydotatwo/p888-model-beam-compare-2xt4`, version 6, `COMPLETE`.
- Hardware: 2x NVIDIA T4; CUDA architecture 75.
- Beam runner revision: `33acff6` from `TryDotAtwo/MultiGPUBeamSearch`.
- Fixed search contract: puzzle 7, depth limit 12, beam width 262144, two ranks.
- All model inputs were exported as FP16 because the runner's BF16 CUTLASS path requires SM80+.

| model | solved | solution length | seconds |
|---|---:|---:|---:|
| native_step600 | 1 | 5 | 1.043450 |
| ihes_e32692 | 1 | 5 | 0.876571 |
| ihes_e40960 | 1 | 5 | 0.888957 |

The native 600-step model matches both IHES checkpoints on solve success and
solution length in this stable smoke case. Its end-to-end runtime is 19.0%
slower than e32692 and 17.4% slower than e40960, so there is no demonstrated
beam-search speed win yet.

An attempted harder fixed case, puzzle 11 at depth 12 and the same beam width,
did not produce a model comparison: rank 1 aborted with `double free or
corruption (!prev)` while testing the native model. This is a multi-rank beam
runner defect, not a model-quality result, and must be fixed before a broader
solve-rate comparison is trustworthy.

Artifacts:

- `test_results/beam_compare_v6/beam-model-comparison/comparison.csv`
- `test_results/beam_compare_v6/beam-model-comparison/summary.json`
- `test_results/beam_compare_v6/beam-model-comparison/*_rank*.log`
- `test_results/beam_compare_v4/beam-model-comparison/native_step600_p11_rank1.log`
