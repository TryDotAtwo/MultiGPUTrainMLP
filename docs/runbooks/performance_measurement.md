# Performance measurement rules

## GPU idle gate

Benchmark scripts wait for an idle GPU by default before running CTest or training sweeps.

Environment knobs:

- `MGT_WAIT_GPU_IDLE=1` enables the gate, `0` disables it.
- `MGT_GPU_IDLE_MAX_UTIL=15` sets the maximum accepted GPU utilization percent.
- `MGT_GPU_IDLE_MAX_MEMORY_MB=1536` sets the maximum accepted used memory per visible GPU.
- `MGT_GPU_IDLE_TIMEOUT_SEC=900` sets how long the script waits before failing.
- `MGT_GPU_IDLE_POLL_SEC=5` sets the polling interval.

The gate prints `gpu_idle_check=ok`, `busy`, `timeout`, or `skipped`. A benchmark with `busy` waits in queue; a benchmark that times out should not be used as a performance result.

## FLOP estimate

`scripts/estimate_train_flops.py` estimates the train-step work from model shape:

```bash
python3 scripts/estimate_train_flops.py \
  --hd1 2560 \
  --hd2 224 \
  --nrd 16 \
  --output-dim 1 \
  --batch-size 24576 \
  --throughput-states-s 300000 \
  --peak-tflops 80
```

The estimate is intentionally explicit and conservative. It counts sparse input forward, one-hot input-gradient GEMM, hidden/residual/output forward plus backward linear work, and small activation/loss work. Sweep summaries include `flop_estimate`, `achieved_tflops`, and `peak_utilization_percent` when `MGT_GPU_PEAK_TFLOPS` is set.

Use `MGT_GPU_PEAK_TFLOPS` for the relevant precision path of one measured GPU. Single-GPU summaries use it directly; multi-rank summaries multiply it by `world_size` for the global utilization and also store `peak_tflops_per_gpu`. Do not compare a profiled run with a non-profiled run without noting `backward_profile`, because profile events change timing.