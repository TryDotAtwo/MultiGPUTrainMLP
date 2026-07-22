# Native training contract on Kaggle 2xT4, 2026-07-22

- Kernel: `trydotatwo/native-multigpu-mlp-training-contract-2xt4`, version 2.
- Status: `COMPLETE`; runner return code: `0`.
- Source HEAD: `fdbff67`, containing correctness commit `e049b46`.
- Input: real archived p888 move table, fingerprint prefix `real:`; 18 moves, state length 72.
- Topology: world size 2, one rank per device, NCCL enabled. The gate itself rejects hardware unless exactly two reported devices are Tesla T4.
- Continuous run: 4 steps.
- Resume run: 2 steps plus resume for 2 further steps.
- All four final manifests report `completed_steps=4`.
- SHA-256 for continuous rank0/rank1 and resumed rank0/rank1 checkpoint state: `109F2AA7E5C487A18AC07A6409FC54C6197A05EDE7BC93DF98B013643B23172B`.
- Periodic step-2 checkpoint and weight exports exist for both ranks.
- Losses remained finite; the CUDA gate checked loss, gradients, and updated weights every step.

This validates the distributed training contract and deterministic resume on real puzzle inputs. It is not evidence of completed production training: the run used the small correctness model and only four optimizer steps. A production-shape convergence run and held-out/solver metric remain required.

Artifacts: `test_results/kaggle_training_contract_2xt4_v2_20260722/`.
