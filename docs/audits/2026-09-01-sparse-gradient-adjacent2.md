# SM86 sparse-gradient adjacent-feature ownership audit

Date: 2026-09-01. Target: original p888, batch 4096, RTX 3070 Laptop /
SM86, CUDA Graph mode. Baseline: residual-gradient publication
`5989e39fcb46d328f9b77c7640fc022262d297de`.

## Decision

Accepted for automatic SM86 selection. One sparse input-gradient thread now
owns two adjacent features instead of one. It loads one row ID and one aligned
`float2` per list entry, then maintains two independent serial FP32 sums. Each
feature therefore preserves the baseline row order and FP32 addition order.

The production fast path requires an even physical width and 8-byte-aligned
input/output pointers. Otherwise it launches the prior scalar owner. The kernel
also has a complete odd-width scalar tail, exercised by the isolated regression.
Explicit `grouped_rows` remains scalar, and automatic T4/A100 policy is
unchanged. There is no new allocation, graph node, synchronization, reduction,
precision mode, or optimizer change.

## Search and mechanism evidence

The initial production-shaped probe selected 64 threads after rejecting wider
two-feature blocks and all four-feature owners. Two steady, alternately warmed
sweeps measured the scalar consumer at 3.8121/3.8211 ms and the 64-thread
two-feature owner at 3.6026/3.6016 ms. Earlier compact-bin, warp-bin, direct
group-eight, and four-feature candidates were slower.

A pre-integration NCU 2025.1.1 mechanism pair used the same 4096x2560 tensor,
ordered row lists, 5184 bins, and selected 64-thread mapping. NCU replays alter
cache state, so these counters explain the mechanism rather than predict the
unprofiled end-to-end delta.

| NCU metric | Scalar owner | Adjacent-2 owner |
|---|---:|---:|
| Duration, ms | 3.725024 | 3.652544 |
| Blocks / threads | 51,840 / 256 | 103,680 / 64 |
| Registers/thread | 40 | 40 |
| Executed instructions | 221,456,880 | 122,342,760 |
| Achieved occupancy | 95.73% | 65.27% |
| DRAM throughput, GB/s | 180.43 | 38.30 |
| L2 sector hit rate | 79.68% | 97.04% |
| Eligible warps/scheduler | 1.141 | 0.372 |
| Long-scoreboard stalls/issued instruction | 16.62 | 25.57 |

The win comes from roughly halving row-ID/address/instruction work and greatly
increasing L2 reuse. It does not come from higher occupancy or DRAM bandwidth;
both fall, and the remaining kernel is latency/scheduling limited.

## Correctness and sanitizer gates

All job IDs refer to retained `.gpu_queue/logs/<id>.log` files.

| Job | Outcome |
|---|---|
| `89637728d21c` | RED: the new exact target failed because the adjacent-2 kernel did not exist. |
| `628ab8264adc` | First integration build exposed a missing translation-unit import; no test ran. |
| `c05bd6c991c8` | GREEN: adjacent-2 exact suite, scalar suite, grouped/full FP16 steps, capture and native graph passed (6/6). |
| `999b6515d8a7` | Full 37-case exact suite plus memcheck/initcheck/racecheck/synccheck: zero errors, leaks, or hazards. Production graph memcheck passed. |
| `52cd5f7fc174` | Candidate frozen and a production graph smoke selected `features=2`. |
| `6b66c99e49aa` | Final gate: 31/31 CUDA/C ABI tests, full/tail/fail-stop native graph, capture/update, frozen identity, measurement validators, and Rust FFI passed. |

The exact GPU oracle owns one output per thread and accumulates the same
ascending row-ID list. It covers cancellation witnesses, RN-half boundaries,
NaN, infinity, signed zero, subnormals, empty bins, tight and guarded
allocations, odd/even widths through 2560, rows 1/31/32/33/4095/4096/4097,
skewed production bins, padding, canaries, and immutable inputs. Finite small
cases also compare with an independent CPU serial oracle.

The production graph remains 434 nodes = 363 kernels + 71 zero-memsets and no
copies. Capture/update and native graph lifecycle both log the effective SM86
dispatch as `kernel=grouped_rows ... features=2`.

## Binary and unprofiled A/B

| Variant | Frozen path | SHA256 |
|---|---|---|
| Residual publication | `/tmp/mgt-bn-residual-publish` | `7ebd7bcc92f9646375cccf8142569e947d2dce1d2c2f787f1771b0c2b94f5cc8` |
| Adjacent-2 | `/tmp/mgt-sparse-adjacent2` | `498d5f5a54479c907fd2761bd34ebc5a74db82e204a02621820160866d34640d` |

Each ABBAAB series retains all six run means. Every process uses the same
original p888 inputs, batch 4096, 140 warmup steps, 100 timed graph steps, and
744001024-byte arena; hashes are checked before and after every run. Clocks were
observed, not locked.

| Series/job | Baseline run means, ms | Candidate run means, ms | Median A -> B | Throughput |
|---|---|---|---|---:|
| `caf0ca0eba6a` | 21.2210, 21.1081, 21.1024 | 21.1523, 21.0497, 21.0515 | 21.1081 -> 21.0515 | +0.2689% |
| `e3012b715dee` | 21.2072, 21.1062, 21.1053 | 21.1536, 21.0493, 21.0526 | 21.1062 -> 21.0526 | +0.2546% |

The descriptive median of all six run means is 21.10715 -> 21.05205 ms,
194.6k samples/s and +0.2617% throughput. The effect is consistent across these
two series but small enough that it remains a measured snapshot, not a hardware
independent promise. Queue job `c21e70b39b0e` failed before opening an output
file because the submitted baseline SHA was truncated; no benchmark ran, and
the corrected confirm is retained above.

- `paired-sparse-adjacent2.jsonl`, SHA256
  `fe47f6b6c0013f5d29851ba6b3dba44acc832246d7c9f3e71e2f8606c7f716c4`.
- `paired-sparse-adjacent2-confirm.jsonl`, SHA256
  `f60df7ad754d3d2f9aafed8d2992e25e23c4ad55967007d0e52d4fa1014d7054`.

## Fresh Nsight Systems proof

Job `dd241acd4f0a` captured baseline and candidate after 100 warmup steps. A
strict read-only analyzer compared four complete graph steps. Kernel and memory
event order are identical after permitting only the sparse kernel name and
launch geometry to change. All unrelated kernel geometry/resources match.

| Mean, measured steps 2-4 | Baseline, ms | Candidate, ms | Delta, ms |
|---|---:|---:|---:|
| Step span | 21.614777 | 21.487637 | -0.127140 |
| Sum of 363 kernels | 21.060622 | 20.926753 | -0.133870 |
| Sparse consumer | 3.739168 | 3.662699 | -0.076470 |

The sparse grid changes from `(5184,10,1)x(256,1,1)` to
`(5184,20,1)x(64,1,1)`. Both use 40 registers/thread, no static/dynamic shared
memory, and no per-thread local memory. The remaining trace delta includes
normal pairwise cache/frequency effects; only the sparse-family reduction is
directly attributable to the source change.

- Baseline SQLite SHA256
  `f8bb888cd917afcdeb4d4326fbdf606d5a6774ec337ef3adb659df48eb93273f`.
- Candidate SQLite SHA256
  `a009e71f65b44636c93a8670c885b1474f40aed4d331227c7741a06135eb289d`.
- `sparse-adjacent2-profile.json`, SHA256
  `0df99df3e427f1a42a5a5fcecf72869a630d68c625d340cd5393e986ec448c02`.

## Boundary and next target

This checkpoint proves exact SM86 execution, not convergence and not T4 or
multi-GPU performance. Sparse input-gradient consumption is still the largest
individual family at 3.663 ms/step. Its adjacent-feature reuse has now consumed
the safe local instruction reduction; further work needs a different dataflow
or representation, not another blind block-size sweep.
