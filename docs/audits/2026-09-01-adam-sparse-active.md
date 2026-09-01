# Structurally sparse input-table Adam on RTX 3070

## Result

The original p888 single-GPU graph no longer launches Adam over input-table
rows that the puzzle can never reach.  At the production physical shape:

- full input table: `5,184 * 2,560 = 13,271,040` parameters;
- structurally active input rows: `1,728 * 2,560 = 4,423,680` parameters;
- skipped input parameters: `8,847,360`;
- full weight vector: `15,460,289` parameters;
- launched live weight-Adam vector: `6,612,929` parameters (`42.7736%`).

The thermal-gated fixed-workload ABBAAB result is `18.242 -> 17.509 ms/step`,
or `-4.0182%` latency and `+4.1864%` throughput.  This checkpoint does not
change the model, input-gradient precision, graph node count, arena size, data
schedule, optimizer hyperparameters, or parameter layout.

## Exactness argument and fail-closed boundary

`BuildInputActiveBins` owns a sorted, position-major superset of every input
bin reachable from the target through the puzzle moves.  Preparation zeros the
whole arena, initializes the FP32 master weights, and creates the coherent FP16
weight mirror before step zero.  The compact input-gradient path writes only
the structural active rows; every inactive gradient and both inactive Adam
moments therefore remain positive zero for the trainer lifetime.

For `weight_decay == 0`, dense Adam is an identity on those skipped elements:
zero gradient and zero moments produce zero update, unchanged FP32 weight,
unchanged FP16 rounding, and zero moments.  The optimization is selected only
when the trainer owns all of these provenance facts.  It is disabled when the
map is absent/full, persistent-zero ownership is absent, optimizer provenance
is absent, or weight decay is nonzero.  The low-level sparse Adam validator
also rejects partial map descriptions, overflow, and nonzero weight decay.

The compact kernel keeps the existing six-argument Adam kernel identity used by
the CUDA graph.  Logical indices first map through the device-owned active-bin
array, then continue densely after the full input-table prefix.  Graph
instantiation validates the reconstructed physical count (`15,460,289`) rather
than incorrectly requiring the launched logical count to equal the allocation.
Affine Adam remains dense.

Exact gates:

- dense versus sparse master weights, FP16 mirror, moment1 and moment2 compare
  byte-for-byte over active, inactive and dense-tail regions after steps
  `1, 2, 997, 65535`;
- invalid partial-map, weight-decay and out-of-range-count configurations are
  rejected;
- native graph tests inspect the captured pointer, row width, active count,
  live count and reconstructed physical count, and retain exact eager/graph
  state comparisons at small shapes;
- production structural-map ownership and persistent-zero inactive gradients
  remain covered.

## Paired measurement

Frozen binaries:

| Variant | Path | SHA256 |
|---|---|---|
| baseline | `/tmp/mgt-sparse-half-u32-b4-bin` | `d71de98df84c93c1e292adb02665827332f31644fb0c31abe045bd9cd73d2359` |
| sparse Adam | `/tmp/mgt-adam-sparse-active-bin` | `704364224d9b0ea1ca8014baa4b4cda0b619921692c8b546b01a82e1513ce5bf` |

A final clean relink produced full-file SHA256
`209b620e3b5a3fed54f724194b44fc0810ef4ebf03a22a16e557e2ef42a0db71`.
The two ELF files have identical size, GNU build ID, loaded sections and CUDA
fatbin.  Their only 20 differing bytes are five nvcc temporary PID strings in
the nonloaded `.strtab`; after `objcopy --strip-debug --strip-unneeded`, both
hash to `6d1d36356d046da957305ccb966944954054b858369d1e6278d4b604b0400a15`.
Thus the measured and final production code/data are byte-identical after
removing nondeterministic symbol names.

Both variants used RTX 3070 Laptop GPU SM86, batch `4096`, graph mode, FP16
input-gradient mirror, `140` warmup steps and `100` timed steps in order
ABBAAB.  All active telemetry samples were `780 MHz`; both variants retained
the same `744,011,520`-byte arena budget.

| Variant | Retained run means (ms) | Median (ms) |
|---|---:|---:|
| baseline | `18.2420, 18.2420, 18.1838` | `18.2420` |
| sparse Adam | `17.5128, 17.5090, 17.4565` | `17.5090` |

Artifacts:

- `paired-adam-sparse-active.jsonl` SHA256
  `2caadcee047d8361bcb5528bf042a6be70d9f079ed7b926f6d1f09f34d4c2f38`;
- `paired-adam-sparse-active-summary.json` SHA256
  `8c2669a8b03a8ce135e27ce65f7c044092727ade992c0f1682cd9c63b214a547`.

## Nsight evidence

Nsight Systems preserves the exact graph event order after normalizing only the
weight-Adam launch geometry: `347` kernels, `71` memsets (`156,136` bytes), and
zero copies per measured step.  Mean steps 2--4:

| Metric | Baseline | Sparse Adam | Delta |
|---|---:|---:|---:|
| graph span (ms) | `18.600823` | `17.878548` | `-0.722275` |
| kernel sum (ms) | `18.054211` | `17.334238` | `-0.719973` |
| weight Adam (ms) | `1.280695` | `0.553709` | `-0.726985` |
| weight Adam grid | `120784x1x1` | `51664x1x1` | live extent only |

The only resource-metadata changes are the recompiled weight and affine Adam
kernels.  Weight Adam drops from `24` to `23` registers per thread; its identity
and `128`-thread block stay unchanged.  Candidate Nsight Systems report SHA256
is `0cfb9e12c023595e82e48041d4354d190ed6c3edb3fc50e644c4b6600da73d6e`,
SQLite SHA256 is
`8ceec3cea4f6819d90d324d27da9b28d7db5d70109f0d817f45319ec0a1b83ee`,
and strict analysis SHA256 is
`c833f86fc7768314f9b35f7424f21cdd04539ac854eb91c1ced9dd7874573aff`.

Nsight Compute explains the reduction:

| Metric | Baseline | Sparse Adam | Candidate/baseline |
|---|---:|---:|---:|
| replay duration (ms) | `1.285312` | `0.558400` | `0.43445` |
| DRAM read (MB) | `247.392768` | `105.827712` | `0.42777` |
| DRAM write (MB) | `216.272640` | `92.384384` | `0.42717` |
| L2 bytes (MB) | `464.110944` | `198.672448` | `0.42807` |
| instructions | `94,268,578` | `35,559,781` | `0.37722` |
| issue active | `28.04%` | `51.21%` | `1.8264x` |

Physical traffic tracks the independently derived `0.427736` live-parameter
ratio.  NCU replay time is mechanism evidence, not the published timing claim;
the thermal-gated unprofiled ABBAAB owns that claim.  Strict NCU analysis SHA256
is `7be25670498841000dc26211a224b9259f78842bfd0813bca2905cf7afab71bc`.

## Verification

All CUDA build/test/profile work ran through the shared Docker GPU queue.

| Queue job | Evidence |
|---|---|
| `6082850b2405` | SM86 build; exact Adam and graph gates `3/3` pass |
| `fba4d287bf5d` | thermal-gated ABBAAB; all six retained |
| `7438dbbcf707` | candidate Nsight Systems trace/export |
| `8e6e63b30193` | paired baseline/candidate Nsight Compute |
| `915c999d6e09` | exact mem/init/race/sync clean; full graph mem/init clean |
| `40e9be3f1aac` | full relevant regression `38/38` pass |
| `f93a019d5e14` | `CMAKE_CUDA_ARCHITECTURES=75`; exact/graph gates `4/4` pass |
| `a94bb2f9ba19` | final SM86 rebuild and exact/graph gates `2/2` pass |
| `b3cd2b1d00fb` | final ELF freeze; identical build ID and loaded size |

The SM75 result proves compilation and exact execution compatibility on the
available SM86 GPU.  It is not a Tesla T4 performance selection; native T4
profiling remains required before publishing an SM75 throughput claim.

## Next measured bottlenecks

After removing dense inactive Adam work, the largest candidate families are the
sparse input-gradient consumer (`1.832 ms`), the two main FP16 GEMM families
(`1.680` and `1.655 ms`), and input embedding forward (`1.642 ms`).  Any next
checkpoint must isolate one of these families and retain this sparse Adam
baseline.
