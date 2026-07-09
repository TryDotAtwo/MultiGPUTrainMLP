#!/usr/bin/env python3
"""Estimate native MLP trainer FLOPs and achieved TFLOP/s from run parameters."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass


@dataclass
class TrainFlopEstimate:
    state_len: int
    state_value_pad: int
    hd1: int
    hd2: int
    residual_blocks: int
    output_dim: int
    batch_size: int
    sparse_input_forward_flops_per_sample: int
    input_embedding_grad_flops_per_sample: int
    hidden_linear_train_flops_per_sample: int
    residual_linear_train_flops_per_sample: int
    output_linear_train_flops_per_sample: int
    activation_and_loss_flops_per_sample: int
    total_flops_per_sample: int
    total_flops_per_step: int
    throughput_states_s: float | None = None
    achieved_tflops: float | None = None
    peak_tflops: float | None = None
    peak_utilization_percent: float | None = None


def estimate_train_flops(
    *,
    state_len: int,
    state_value_pad: int,
    hd1: int,
    hd2: int,
    residual_blocks: int,
    output_dim: int,
    batch_size: int,
    throughput_states_s: float | None = None,
    peak_tflops: float | None = None,
) -> TrainFlopEstimate:
    # Linear train cost uses forward + dW + dX GEMM/MATMUL work: 3 * (2*m*n) per sample.
    sparse_input_forward = state_len * hd1
    input_embedding_grad = 2 * state_len * state_value_pad * hd1
    hidden_linear_train = 6 * hd1 * hd2
    residual_linear_train = residual_blocks * 2 * 6 * hd2 * hd2
    output_linear_train = 6 * hd2 * output_dim
    activation_and_loss = hd1 + hd2 * (2 * residual_blocks + 2) + output_dim * 8
    total_per_sample = (
        sparse_input_forward
        + input_embedding_grad
        + hidden_linear_train
        + residual_linear_train
        + output_linear_train
        + activation_and_loss
    )
    total_per_step = total_per_sample * batch_size
    achieved = None
    utilization = None
    if throughput_states_s is not None:
        achieved = throughput_states_s * total_per_sample / 1.0e12
        if peak_tflops and peak_tflops > 0:
            utilization = achieved / peak_tflops * 100.0
    return TrainFlopEstimate(
        state_len=state_len,
        state_value_pad=state_value_pad,
        hd1=hd1,
        hd2=hd2,
        residual_blocks=residual_blocks,
        output_dim=output_dim,
        batch_size=batch_size,
        sparse_input_forward_flops_per_sample=sparse_input_forward,
        input_embedding_grad_flops_per_sample=input_embedding_grad,
        hidden_linear_train_flops_per_sample=hidden_linear_train,
        residual_linear_train_flops_per_sample=residual_linear_train,
        output_linear_train_flops_per_sample=output_linear_train,
        activation_and_loss_flops_per_sample=activation_and_loss,
        total_flops_per_sample=total_per_sample,
        total_flops_per_step=total_per_step,
        throughput_states_s=throughput_states_s,
        achieved_tflops=achieved,
        peak_tflops=peak_tflops,
        peak_utilization_percent=utilization,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-len", type=int, default=80)
    parser.add_argument("--state-value-pad", type=int, default=2)
    parser.add_argument("--hd1", type=int, required=True)
    parser.add_argument("--hd2", type=int, required=True)
    parser.add_argument("--nrd", type=int, required=True)
    parser.add_argument("--output-dim", type=int, default=1)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--throughput-states-s", type=float)
    parser.add_argument("--peak-tflops", type=float)
    args = parser.parse_args()
    estimate = estimate_train_flops(
        state_len=args.state_len,
        state_value_pad=args.state_value_pad,
        hd1=args.hd1,
        hd2=args.hd2,
        residual_blocks=args.nrd,
        output_dim=args.output_dim,
        batch_size=args.batch_size,
        throughput_states_s=args.throughput_states_s,
        peak_tflops=args.peak_tflops,
    )
    print(json.dumps(asdict(estimate), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())