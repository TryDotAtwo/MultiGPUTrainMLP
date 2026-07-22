# Production Training Contract Design

## Goal

Turn `mgt_native_train` from a synthetic performance smoke into a fail-closed trainer that consumes a real puzzle definition, resumes deterministically, and preserves progress through periodic atomic artifacts. GPU execution and performance validation remain restricted to Kaggle 2xT4.

## Scope

This phase covers training correctness only:

- real `group.json` and target binary inputs;
- an explicit synthetic benchmark mode for existing smoke/performance jobs;
- checkpoint metadata validation and cumulative global step restoration;
- periodic checkpoint and weight export;
- optimizer parameter plumbing and decoupled AdamW semantics;
- finite-value failure gates;
- CPU/unit coverage locally and end-to-end validation on 2xT4.

The staged performance autotuner and new CUDA/NCCL kernels are separate follow-up phases. They must benchmark the real workload produced by this contract.

## Input Contract

Production mode requires both `--group-json PATH` and `--target-bin PATH`. Missing either input is an error. Synthetic puzzle generation remains available only through `--synthetic-benchmark 1`; production inputs and synthetic mode are mutually exclusive.

The loaded puzzle must agree with the configured group, state length, move count, and state-value limits. The trainer records stable hashes of both input files in every checkpoint and rejects a resume when the hashes or model shape differ.

## Checkpoint Contract

Checkpoint version 2 stores:

- cumulative completed step;
- parameter count and model/puzzle fingerprint;
- seed and optimizer hyperparameters;
- weights, first moment, and second moment;
- byte count and checksum for the state payload.

Resume uses `global_step = completed_step + local_step`. The global step drives random-walk generation and Adam bias correction, so an uninterrupted run and a split/resumed run consume the same examples and updates.

Checkpoint writes use a sibling temporary directory followed by an atomic rename. Rank 0 writes artifacts only after all ranks have completed the step and synchronized. A failed write leaves the previous complete checkpoint intact.

## Periodic Artifacts

`--checkpoint-period-steps N` and `--weight-export-period-steps N` use cumulative steps. Zero disables the corresponding periodic write. The final step is always written when artifacts are enabled, even when it is not a period boundary.

Each periodic export is written to a step-qualified directory. `checkpoint/latest` and `weights/latest` are replaced atomically only after the new artifact is complete.

## Optimizer and Numerical Contract

The native CLI accepts `--adam-beta1`, `--adam-beta2`, and `--adam-eps`. AdamW applies moments to the raw gradient and applies weight decay as a separate parameter update. Existing zero-decay behavior remains numerically unchanged.

Loss, gradients, and updated parameters must be finite. A non-finite value terminates every rank with a non-zero exit and records the failing cumulative step.

## Validation Gates

Local gates are CPU-only parsing, manifest, checksum, periodic-schedule, and optimizer reference tests. GPU gates run only on Kaggle 2xT4 and include:

1. real-input launch on both ranks;
2. uninterrupted versus resumed state equality;
3. rank-0/rank-1 parameter checksum equality;
4. periodic artifact survival and final export loading;
5. a longer held-out convergence run before calling full training complete.

## Follow-up Performance Phase

After these gates pass, the selected autotuner performs a full first-run sweep per exact 2xT4/workload fingerprint, caches the winner, and performs a quick drift check on reuse. Explicit CLI overrides take precedence over the cache; unknown or mismatched fingerprints fail closed to validated defaults.
