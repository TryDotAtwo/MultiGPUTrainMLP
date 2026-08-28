# Original p888 single-GPU baseline — RTX 3070 Laptop

Date: 2026-08-28

## Configuration

- GPU: NVIDIA GeForce RTX 3070 Laptop GPU (SM86)
- Container image: `mgt-single-gpu-dev:2026-08-28`
- Build: Release, CUDA arch 86, `MGT_ENABLE_NCCL=OFF`
- Model: logical `5184 -> 2556 -> 16 x (218 -> 218) -> 1`
- Physical alignment: `2560 / 224`
- Batch: 4096
- Warmup: 2 steps
- Inputs: `native/production_inputs/p888.json` plus the archived-equivalent
  identity target `native/tests/fixtures/p888-target.bin`
- Measurement: 10 serially enqueued steps, one terminal synchronization
- Arena: 681,086,464 bytes, allocated before steady state

## Unprofiled result

```json
{"gpu":"NVIDIA GeForce RTX 3070 Laptop GPU","arch":86,"batch":4096,"warmup":2,"steps":10,"step_ms":44.9383,"samples_s":91147.1,"memory_bytes":681086464,"loss":239.155,"status":"ok"}
```

An earlier 45.1725 ms result used an identity-only test move set and is invalid
as a production-data baseline. The benchmark now rejects that fixture. The
number above uses real p888 moves and the audited equal-depth, inverse-excluding
single-GPU schedule.

## Sparse input projection promotion

The row-owned/shared-offset `half2` gather is bitwise equal to a scalar CUDA
reference for production shape at 1, 17, and 4096 rows. Three unprofiled runs
with 5 warmup and 20 measured steps produced 44.0124, 44.0184, and 43.7567
ms/step (median 44.0124 ms, 93,064.6 samples/s), a 2.06% median step-time
reduction from the frozen baseline.

NCU 2025.1.1 measured the isolated gather at 2.25 ms, 349.35 GB/s, 91.06% DRAM
throughput, and 123,645,952 executed instructions. The prior kernel measured
4.01 ms, 212.22 GB/s, and 362,414,080 instructions. Memcheck reported 0 errors;
racecheck reported 0 hazards, 0 errors, and 0 warnings.

## Nsight Systems attribution

The profile used 2 warmup plus 5 measured steps. Dominant GPU kernels:

| Kernel group | GPU time |
|---|---:|
| Sparse input gradient | 39.9% |
| Sparse FP16 input gather | 9.2% |
| FP32-to-FP16 operand conversion | 7.7% |
| Column reductions | 5.6% |
| Tensor Core GEMMs (visible top variants) | 11.0% |
| Fused AdamW plus FP16 mirror | 3.7% |
| Local BN coalesced kernels | 11.5% |

The prior feature-major local BN kernels accounted for 43.3% of GPU time. The coalesced tiled implementation reduced the 4096-batch step from 70.6136 ms to 45.9194 ms. Fusing AdamW with FP16 mirror refresh reduced it further to 44.6885 ms in the development build.

## Gates

- Local BN reference test: pass
- FP32 local full train step: pass
- FP16 local full train step: pass
- Prepared single-GPU lifecycle: pass
- Compute Sanitizer memcheck: 0 errors
- `MGT_ENABLE_NCCL=OFF` lifecycle and benchmark: pass
- `ldd` on benchmark and C ABI shared library: no NCCL dependency
- C ABI lifecycle: pass
- Rust RAII owner integration: 1 passed, 0 failed

The sequential audit continues at A4 BatchNorm/activation. Sparse input gradient
remains the largest already-known performance target, but is gated behind the
intermediate correctness audit.
