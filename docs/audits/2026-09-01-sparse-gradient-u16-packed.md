# SM86 sparse-gradient packed-u16 row-list audit

Date: 2026-09-01. Target: original p888, batch 4096, RTX 3070 Laptop /
SM86, CUDA Graph mode. Baseline: adjacent-feature ownership
`a1d5d7d548589393a9e3e6a050b22d8e27e3ebdb`.

## Decision

Accepted for automatic SM86 selection. The grouped-row builder emits exact
ascending `uint16_t` row IDs when the live row count is at most 65535. The
consumer packs four independent bins into one 128-thread CTA: one warp owns a
bin, and every lane owns two adjacent features with two independent serial FP32
sums. Row order and each feature's FP32 addition order are unchanged.

The production fast path additionally requires an even physical width and
8-byte-aligned input/output pointers. Any failed condition uses the previous
u32 builder and adjacent-2/scalar consumer. Explicit `grouped_rows`, T4 and A100
policy are unchanged. The isolated consumer still implements an exact odd-width
tail.

For p888, row-list storage falls from 84,934,656 to 42,467,328 bytes. This
scratch aliases the existing BN workspace, so the graph arena remains
744001024 bytes. There is no new allocation, graph node, synchronization,
reduction, precision mode, optimizer change, or model approximation.

## Correctness and sanitizer gates

All job IDs refer to retained `.gpu_queue/logs/<id>.log` files.

| Job | Outcome |
|---|---|
| `45f403c36c80` | RED: both new exact targets failed to compile because the u16 builder and packed consumer did not exist. |
| `d79b183d1585` | GREEN: u16 builder/consumer, u32/adjacent baselines, full FP32/FP16 steps, capture and native graph passed (10/10). |
| `ef6aa574d20f` | Full 22-case builder and 37-case consumer memcheck; quick initcheck/racecheck/synccheck for both; production graph memcheck: zero errors, leaks, or hazards. |
| `1111c98a13d1` | Candidate frozen and production smoke selected `features=2 row_index=2`. |
| `6d50739020dd` | Final gate: 33/33 CUDA/C ABI tests, full/tail/fail-stop native graph, capture/update, measurement validators, code-section provenance, and Rust FFI passed. |

The u16 builder suite checks exact counts, ascending IDs, reused smaller row
strides, invalid values, empty bins, production 72x72 layouts, untouched tails,
canaries, and immutable inputs. The packed consumer suite retains the exact GPU
oracle and finite CPU oracle used by adjacent-2. It covers cancellation, RN-half
boundaries, NaN, infinity, signed zero, subnormals, empty bins, tight and guarded
allocations, odd/even widths through 2560, rows 1/31/32/33/4095/4096/4097,
skewed production bins, padding, and canaries.

The production graph remains 434 nodes = 363 kernels + 71 zero-memsets and no
copies. Capture/update and native graph lifecycle both report the effective
SM86 dispatch as `kernel=grouped_rows ... features=2 row_index=2`.

## Binary provenance and unprofiled A/B

| Variant | Frozen path | SHA256 |
|---|---|---|
| Adjacent-2 u32 | `/tmp/mgt-sparse-adjacent2` | `498d5f5a54479c907fd2761bd34ebc5a74db82e204a02621820160866d34640d` |
| Packed u16 | `/tmp/mgt-sparse-u16-packed` | `b643e49b5e4ec1075e8c515e9ea5085b527b0668e36ba27388aee411059ac14d` |

Each ABBAAB series retains all six run means. Every process uses the same
original p888 inputs, batch 4096, 140 warmup steps, 100 timed graph steps, and
744001024-byte arena; hashes are checked before and after every run. Clocks were
observed, not locked.

| Series/job | Baseline run means, ms | Candidate run means, ms | Median A -> B | Throughput |
|---|---|---|---|---:|
| `4280edc982f3` | 21.1563, 21.0617, 21.0454 | 21.0840, 20.9811, 20.9734 | 21.0617 -> 20.9811 | +0.3842% |
| `69a459caffa8` | 21.1555, 21.0511, 21.0501 | 21.0822, 20.9760, 20.9670 | 21.0511 -> 20.9760 | +0.3580% |

The descriptive median of all six run means is 21.0564 -> 20.97855 ms,
195.25k samples/s and +0.3711% throughput. The second baseline series observed
both 780 and 885 MHz while every candidate sample observed 780 MHz; therefore
the exact end-to-end percentage is a retained local snapshot. Both independent
series and the isolated profiler agree on direction.

- `paired-sparse-u16-packed.jsonl`, SHA256
  `aca72435732e4943e7d99d9ac8e874904ac973eea31bc532bbf53fd12b0fc7db`.
- `paired-sparse-u16-packed-confirm.jsonl`, SHA256
  `f5b932f627e61e4d623142bc8b807fd3ff4f4ca3ee25763434645a60e09a20c2`.

Whole-ELF identity is not used across the post-review rebuild: final job
`6d50739020dd` produced SHA256
`f52b1a2996cab080eca706ee005fd820bc030363a4ee41031f0390a6d34d1b6c`,
while the frozen measured ELF has the hash above. Jobs `50f0cdf05102` and
`969cad85d15c` prove equal file sizes and byte-identical host `.text`, `.rodata`,
emitted SASS, and PTX. Their respective code hashes are:

- host `.text`: `d2d535a5d04b7b42de7c045e668b9d79bc62d9b2342b462b3c6a7d7d6971f085`;
- host `.rodata`: `3149a0392f5241a9b6fd0048a01b10c67d0a59f7b1c1c9785d255d4fa1e9a24a`;
- SASS dump: `967ba711b7387cc19c18a2d580365c0bb8325eae21fbdf7a32a5428c19d4f507`;
- PTX dump: `09e0b18a8c335be33eae5829be445524d75279efee957e15bde9aae830ccbd47`.

## Fresh Nsight Systems proof

Job `17536d770855` captured both frozen binaries after 100 warmup steps. A
strict analyzer compared four complete graph steps. Kernel/memory event order,
all unrelated kernel geometries/resources, 363 kernels, 71 memsets, and zero
copies are preserved.

| Mean, measured steps 2-4 | Baseline, ms | Candidate, ms | Delta, ms |
|---|---:|---:|---:|
| Step span | 21.547609 | 21.490824 | -0.056785 |
| Sum of 363 kernels | 20.993517 | 20.929257 | -0.064260 |
| Row-list builder | 0.224204 | 0.226553 | +0.002349 |
| Sparse consumer | 3.681077 | 3.607299 | -0.073779 |
| Builder + consumer | 3.905282 | 3.833852 | -0.071430 |

The consumer grid changes from `(5184,20,1)x(64,1,1)` to
`(1296,40,1)x(128,1,1)`. Both use 40 registers/thread and no shared or local
memory. The builder keeps `(648,1,1)x(256,1,1)` and moves from 29 to 30
registers/thread.

- Baseline SQLite SHA256
  `6d4db01914ec3e401bc670c402100ce3e925132661fc41eb3730d760126b0eab`.
- Candidate SQLite SHA256
  `bbe6470534b271913170820387f2fe871538b3cd02a6840b43a603a43bb6f351`.
- `sparse-u16-packed-profile.json`, SHA256
  `80676fbd73895fd37ad424ba123b580225b1080be8701fb2e17cea02541ef644`.

## Nsight Compute mechanism proof

Jobs `2bacc2dcfef6` and `6603818365f5` used NCU 2025.1.1, original p888,
100 warmup launches, one production sparse-consumer launch, and no profiler
clock control. Replay counters explain the mechanism; they do not replace the
unprofiled paired result.

| NCU metric | Adjacent-2 u32 | Packed u16 |
|---|---:|---:|
| Duration, ms | 3.658272 | 3.602688 |
| Blocks / threads | 103,680 / 64 | 51,840 / 128 |
| Registers/thread | 40 | 40 |
| Executed instructions | 120,483,120 | 124,563,080 |
| DRAM throughput, GB/s | 38.283 | 30.060 |
| L2 sector hit rate | 97.162% | 98.164% |
| Achieved occupancy | 65.330% | 67.350% |
| Eligible warps/scheduler | 0.36654 | 0.40768 |
| Long-scoreboard stalls/issued instruction | 25.9993 | 24.2842 |

The packed kernel executes slightly more instructions because of u16 widening
and four-bin mapping, but cuts list traffic, halves block count, improves L2 hit
rate and scheduler eligibility, and reduces long-scoreboard pressure. That is
consistent with the measured 0.0556 ms NCU and 0.0738 ms Nsight consumer wins.

- Baseline NCU raw CSV SHA256
  `367018e53cac6329b9986616e1746f626721cd7956f5c612d3ac489694f7d490`.
- Candidate NCU raw CSV SHA256
  `95b09614202eb84e8260a62748b0090e49c91dc3fc021526abbb5e588cb6e7ac`.

## Boundary and next target

This checkpoint proves exact SM86 execution, not convergence and not T4 or
multi-GPU performance. The packed-u16 path is intentionally disabled above
65535 live rows. The sparse consumer remains the largest individual family at
3.607 ms/step; `InputHalf2Row` is next at 1.823 ms/step. Further sparse work
needs a different dataflow or representation rather than another blind launch
geometry sweep.
