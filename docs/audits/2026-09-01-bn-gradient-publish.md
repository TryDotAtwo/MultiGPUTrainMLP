# SM86 BN backward gradient publication

Date: 2026-09-01. Original p888, RTX 3070 Laptop/SM86, Docker queue only.
This follows the [native graph lifecycle](2026-08-31-single-gpu-native-graph.md).

## Accepted change

Each of the 34 local BN backward sites already reduces `dgamma` and `dbeta`
into the two observable FP32 `stats_workspace` ranges. The old path then ran
the existing backward-apply kernel and enqueued two device-to-device copies to
the public gradient ranges. The graph therefore contained 68 copy nodes per
full step.

The apply kernel now uses its first `cols` lanes to publish those completed
workspace reductions while it computes `dX`. Reduction ownership, atomic and
lane-fold order, FP32 values, workspace contents, `dX`, half mirror and Adam
inputs are unchanged. Apply-only calls pass null publication pointers and still
treat supplied affine gradients as immutable.

Publishing is used only when both output feature ranges are disjoint from all
apply inputs/outputs and the reduction workspace. The legacy post-apply copies
remain for accepted aliases such as `dgamma == gamma` and `dbeta == inv_std`;
writing those ranges at the start of the apply kernel would otherwise race its
own reads. A dedicated exact test covers this fallback. There is no new buffer,
kernel, FLOP, precision mode or arena allocation.

## Test-driven and sanitizer evidence

All job IDs refer to retained `.gpu_queue/logs/<id>.log` files.

| Job | Outcome |
|---|---|
| `ff1d6ed92f1e` | Expected RED: rows4 capture had 471 nodes = 332 kernels + 68 copies + 71 memsets; the new zero-copy assertion failed. |
| `7416c87bd6ce` | Intermediate implementation kept `stats_workspace` poisoned and correctly failed the old-GPU-statistics oracle. This rejected reducing directly into public outputs. |
| `ab75b701cdeb` | GREEN: eight BN/full-step/graph tests passed after restoring workspace reductions and publishing in apply. |
| `015cec1aba64` | Exact alias fallback passed. Rows4 capture/update had 403 nodes = 332 kernels + 71 memsets and zero copies. |
| `aa14facf632f` | Memcheck/leak-check, initcheck, racecheck and synccheck passed; zero errors/hazards/leaks. Native graph lifecycle and a production graph step also passed memcheck. |
| `399c634dfee9` | Final gate: 30/30 CUDA/C ABI tests, rows4096 full/tail/fail-stop native graph, rows4096 capture/update, benchmark binary identity, two measurement-validator tests and Rust FFI all passed. |

The rows4096 captured graph now has 434 nodes = 363 kernels + 71 memsets.
The final gate includes the real 4095-row tail used by the native graph test;
the production runtime still executes non-capacity tails eagerly.

## Binary and workload identity

| Variant | Frozen path | SHA256 |
|---|---|---|
| Previous native graph | `/tmp/mgt-native-graph-final` | `82f7d55c236b90c224d1c06f834460ec1bf39b5e5f97a6c06f00699dd7eaaed7` |
| BN publication | `/tmp/mgt-bn-publish-grad` | `916e64a365843601e2ea16364ecda46d79913ce37e2200139163be91c6596d08` |

Both use batch4096, 140 warmup and 100 timed graph steps, the same original
p888 JSON/target, seed and744001024-byte graph arena. Hashes are checked before
and after every run. Active telemetry in both retained series reports780MHz;
clocks were observed, not locked.

## Unprofiled measurements

ABBAAB retains every run mean and uses the median of the three means per
variant. The small effect is not reported as a stable fixed percentage.

| Series/job | Baseline run means, ms | Candidate run means, ms | Median A -> B | Throughput |
|---|---|---|---|---:|
| `46a561044f12` | 21.3650, 21.2557, 21.2558 | 21.2534, 21.1511, 21.1512 | 21.2558 -> 21.1512 | +0.4945% |
| `7787a307413d` | 21.3637, 21.2557, 21.2599 | 21.2580, 21.2549, 21.1463 | 21.2599 -> 21.2549 | +0.0235% |

Pooling the six retained run means only as a descriptive cross-check gives
21.25785 -> 21.20230ms, +0.2620% throughput. The intended delta is about the
same size as process-level variance, so neither the first +0.49% nor the pooled
value is a fixed-clock performance claim. Both predefined series are nonnegative.

- `paired-bn-publish.jsonl`, SHA256
  `296954976568fb539643acc4a8aeadce692cbf22cd190cb400005c87908db411`.
- `paired-bn-publish-confirm.jsonl`, SHA256
  `4abcd98c62c49155035215643b5c7e2692197b8d0f42afbbc162f7a2ebbf619f`.

## Nsight structural proof

Fresh baseline and candidate traces each skip 100 warmup steps and analyze four
full graph steps. The fail-closed analyzer verifies the identical ordered 363
kernel tokens after canonicalizing only the two added apply pointer arguments,
identical geometry and resource metadata, identical 71 zero-memsets, and the
same capture/replay/update API budget. Removing the baseline copy tokens makes
the complete event sequence exactly equal to the candidate sequence.

| Mean, measured steps 2-4 | Baseline, ms | Candidate, ms |
|---|---:|---:|
| Step span | 21.787693 | 21.644238 |
| Sum of kernel durations | 21.056121 | 21.094013 |
| 34 backward-apply kernels | 2.147321 | 2.189212 |
| 68 D2D copies / 78,000 bytes | 0.170357 | 0 |
| 71 memsets / 156,136 bytes | 0.109286 | 0.108243 |

The apply timing difference is descriptive profiler noise; registers, static
shared memory and local-memory metadata are identical. The measured structural
result is zero D2D copies and a0.143455ms shorter traced span, with no other
event-order change.

- Baseline SQLite SHA256
  `a2b41dab5ca171e01acfd8b657dfc4849210e296afa1918d3bfbe49ec6a2965b`.
- Candidate SQLite SHA256
  `74a6686224fa516997cf5adb60a8a45e946bf2386845e6bef961e3481d0fd138`.
- `bn-publish-profile.json`, SHA256
  `3c9824d6a517eb9852caa82322d56337964437a8ec4f1ea3534b23722d34d46f`;
  trace jobs `369f97613f07`, `a7a46df594b1`, analyzer job `0aacdeb33401`.

## Boundaries

This is a scheduling/traffic cleanup, not a convergence or numerical-quality
experiment. It says nothing new about T4 or two-GPU scaling. Large ignored
Nsight/JSONL artifacts stay local; source, tests and this audit are published.
The next optimization must be selected from the updated candidate trace rather
than extrapolated from this sub-noise full-step delta.
