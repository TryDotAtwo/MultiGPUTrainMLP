# BN residual-gradient publication audit

Date: 2026-09-01. Target: original p888, batch 4096, RTX 3070 Laptop /
SM86, CUDA Graph mode. Base candidate: BN affine-gradient publication
`fb75ae7240d5f3c2fe59085663a6e637c89d7526`.

## Decision

Accepted for the SM86 single-GPU trainer. In a full residual BN backward,
`BackwardPartialCoalesced` already computes the exact masked incoming `dY` used
by the affine reductions. It now publishes that same FP32 value to
`residual_grad` for every physical row-stride lane. `BackwardApplyCoalesced`
consumes the published value instead of rereading both `dy` and `activated`, and
does not write it a second time.

The public apply-only API keeps the old read-and-publish path. Full backward
extends the partial grid to `row_stride` only when residual publication is
enabled. A `cols=1, stride=65` exact case proves that padding beyond the first
32-column tile is covered. The original select semantics for NaN, signed zero,
masked padding and in-place `dy == dx` are unchanged.

There is no new allocation, graph node, kernel, synchronization, arithmetic,
precision mode or reduction-order change. At the sixteen `218/224` residual
sites, the source-level traffic delta is nominally 55,574,528 fewer bytes per
step: one FP32 read saved per logical lane, offset by one added read per padding
lane. This is not a measured DRAM-byte claim because caches remain in the path.

## Correctness and sanitizer evidence

All job IDs refer to retained `.gpu_queue/logs/<id>.log` files.

| Job | Outcome |
|---|---|
| `978ec7aa30f0` | Build plus eight BN/full-step/graph tests passed. |
| `7e2f0c6088ce` | Exact quick suite passed 19 cases, including multi-tile padding; rows4 capture remained 403 nodes = 332 kernels + 71 memsets. Frozen candidate created. |
| `07bdc015e904` | Memcheck and leak-check completed with zero errors/leaks; the combined command then stopped on an invalid initcheck CLI spelling. |
| `a16f577a9459` | Harness-only retry established that this Compute Sanitizer version takes `--track-unused-memory` as a flag. No target program ran. |
| `b61afcfa9f02` | Corrected initcheck, racecheck and synccheck reported zero errors/hazards. Native graph lifecycle and one production graph step passed memcheck with zero leaks/errors. |
| `4dc87b9c7ac0` | Final gate: 30/30 CUDA/C ABI tests, rows4096 full/tail/fail-stop native graph, rows4096 capture/update, frozen-binary identity, two measurement-validator tests and Rust FFI passed. |

The production graph remains 434 nodes = 363 kernels + 71 zero-memsets, with
no memcpy nodes. The exact oracle covers separate and aliased activation,
in-place output, literal RN-half values, inactive capacity canaries, affine
gradient publication/fallback, and full residual padding.

## Binary and workload identity

| Variant | Frozen path | SHA256 |
|---|---|---|
| Previous BN publication | `/tmp/mgt-bn-publish-grad` | `916e64a365843601e2ea16364ecda46d79913ce37e2200139163be91c6596d08` |
| Residual publication | `/tmp/mgt-bn-residual-publish` | `7ebd7bcc92f9646375cccf8142569e947d2dce1d2c2f787f1771b0c2b94f5cc8` |

Every unprofiled run uses the same original p888 JSON/target, batch 4096,
140 warmup steps, 100 timed graph steps and 744001024-byte arena. Binary hashes
are checked before and after every run. Active telemetry reports 780MHz; clocks
were observed, not locked.

## Unprofiled measurements

ABBAAB retains all six runs and compares the median of three run means per
variant. The effect is below run/process variance, so no stable fixed percentage
is claimed.

| Series/job | Baseline run means, ms | Candidate run means, ms | Median A -> B | Throughput |
|---|---|---|---|---:|
| `8f6e8c93fa78` | 21.2667, 21.1604, 21.1569 | 21.1070, 21.1051, 21.1069 | 21.1604 -> 21.1069 | +0.2535% |
| `f9e3630ea209` | 21.3682, 21.1549, 21.1551 | 73.6364, 21.2108, 21.1009 | 21.1551 -> 21.2108 | -0.2626% |
| `3da538954914` | 21.2673, 21.1546, 21.1550 | 21.2140, 21.1052, 21.1052 | 21.1550 -> 21.1052 | +0.2360% |

The second series is retained in full; its 73.6364ms candidate run is an obvious
latency outlier but is not deleted or silently filtered. As a descriptive
cross-check, the median of all nine run means is 21.1569 -> 21.1069ms,
about 194.1k samples/s and +0.2369% throughput.

- `paired-bn-residual.jsonl`, SHA256
  `e5b33d8695eb769de4f8dc2bb9411c827796065d5cb5f79c616ff59056c5111c`.
- `paired-bn-residual-confirm.jsonl`, SHA256
  `358971f493c76b223aaf5b1541fa72f7d9de7991ca57ee7375239aedd89d03f9`.
- `paired-bn-residual-adjudicate.jsonl`, SHA256
  `af6e733fa069ac930ee1d8c22f956ab2bf6a9897c91cb69387819f1b81fddc76`.

## Nsight structural proof

Job `2144bf2f9a49` captured fresh baseline/candidate traces after 100 warmup
steps; job `0361cbcdfd09` strictly analyzed four complete graph steps. Kernel
order, grid/block/dynamic-shared geometry, all 71 memsets, zero-copy budget and
complete event order match after canonicalizing only the changed BN backward
template signatures. No unrelated kernel resource metadata changed.

| Mean, measured steps 2-4 | Baseline, ms | Candidate, ms | Delta, ms |
|---|---:|---:|---:|
| Step span | 21.657760 | 21.615781 | -0.041979 |
| Sum of 363 kernels | 21.092095 | 21.056235 | -0.035860 |
| 34 BN backward partials | 1.489674 | 1.751926 | +0.262252 |
| 34 BN backward applies | 2.186613 | 1.891168 | -0.295445 |
| Partial + apply | 3.676287 | 3.643094 | -0.033193 |

The residual specialization occurs 16 times per step; the remaining 18 sites
retain the non-residual specialization. Apply resource metadata changes only
within the edited family: observed register variants move from 20/21 to 20/22,
with static shared memory 0, local memory/thread 0 and unchanged geometry.

- Baseline SQLite SHA256
  `c31e70f5e932282d3cb5c9aa8d21806ee16d6e0034e7b6036e7002a40fba9f80`.
- Candidate SQLite SHA256
  `74dc105c3518e495bbb50dd1965c13f44b8be87540eb947522b3d068ec38040d`.
- `bn-residual-profile.json`, SHA256
  `9140c60b88f66c90e1d37bdd3c6ee8f69fa358e17ac38f4334199427136bc518`.

## Boundaries

This is an exact dataflow/traffic cleanup, not a precision or convergence
experiment. The full-step delta is small and one retained series reverses sign;
the accepted evidence is structural plus a positive isolated BN-kernel delta,
not a promised +0.24% on every process run. It establishes nothing for T4 or
two-GPU scaling. Large ignored Nsight/JSONL artifacts stay local.

The next optimization must start from the updated trace. The largest individual
kernel family remains sparse input-gradient consumption (~3.7ms); prior compact
active-bin and tile-width probes were slower, so the next attempt needs a new
ownership/dataflow change rather than another local launch-geometry sweep.
