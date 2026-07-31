#!/usr/bin/env python3
"""Strictly join per-rank A100 benchmark JSONL records."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import defaultdict
from typing import Iterable, Mapping, Sequence


class BenchmarkFormatError(ValueError):
    """Raised when benchmark evidence cannot be joined without ambiguity."""


_IDENTITY_FIELDS = (
    "schema",
    "world",
    "case",
    "row_vector",
    "pair_index",
    "pair_attempt_nonce",
    "source_sha",
    "policy_sha256",
    "snapshot_sha256",
    "runtime_tree_sha256",
)


def _freeze(value):
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, dict):
        return tuple(sorted((key, _freeze(item)) for key, item in value.items()))
    return value


def _require(row: Mapping, field: str):
    if field not in row:
        raise BenchmarkFormatError(f"missing field: {field}")
    return row[field]


def _identity(row: Mapping):
    return tuple(_freeze(_require(row, field)) for field in _IDENTITY_FIELDS)


def _finite_positive(row: Mapping, field: str) -> float:
    value = _require(row, field)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BenchmarkFormatError(f"{field} must be numeric")
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise BenchmarkFormatError(f"{field} must be finite and positive")
    return value


def _nonnegative_integer(row: Mapping, field: str) -> int:
    value = _require(row, field)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise BenchmarkFormatError(f"{field} must be a nonnegative integer")
    return value


def _validate_common(row: Mapping) -> tuple[int, int, tuple]:
    if _require(row, "schema") != "mgt_a100_train_step_v1":
        raise BenchmarkFormatError("unsupported schema")
    if _require(row, "status") != "ok":
        raise BenchmarkFormatError("non-ok benchmark row")
    world = _nonnegative_integer(row, "world")
    rank = _nonnegative_integer(row, "rank")
    if world == 0 or rank >= world:
        raise BenchmarkFormatError("invalid rank/world")
    row_vector = _require(row, "row_vector")
    if not isinstance(row_vector, list) or len(row_vector) != world:
        raise BenchmarkFormatError("row_vector length must equal world")
    return world, rank, _identity(row)


def _join_complete(groups, world: int, label: str):
    joined = {}
    for key, rank_rows in groups.items():
        if set(rank_rows) != set(range(world)):
            raise BenchmarkFormatError(f"{label} is missing one or more ranks")
        joined[key] = rank_rows
    return joined


def _nearest_rank_q95(values: Sequence[float]) -> float:
    ordered = sorted(values)
    index = max(0, math.ceil(0.95 * len(ordered)) - 1)
    return ordered[index]


def summarize_records(records: Iterable[Mapping]) -> dict:
    region_groups = defaultdict(dict)
    sample_groups = defaultdict(dict)
    common_identity = None
    common_world = None

    for row in records:
        if not isinstance(row, Mapping):
            raise BenchmarkFormatError("each JSONL row must be an object")
        world, rank, identity = _validate_common(row)
        if common_identity is None:
            common_identity, common_world = identity, world
        elif identity != common_identity or world != common_world:
            raise BenchmarkFormatError("benchmark identity mismatch")

        mode = _require(row, "timing_mode")
        if mode == "region_average":
            repeat = _nonnegative_integer(row, "repeat")
            steps = _nonnegative_integer(row, "measured_steps")
            if steps == 0:
                raise BenchmarkFormatError("measured_steps must be positive")
            region_ms = _finite_positive(row, "region_ms")
            key = (identity, repeat)
            if rank in region_groups[key]:
                raise BenchmarkFormatError("duplicate rank in region repeat")
            region_groups[key][rank] = (steps, region_ms)
        elif mode == "step_samples":
            step_index = _nonnegative_integer(row, "step_index")
            step_ms = _finite_positive(row, "step_ms")
            key = (identity, step_index)
            if rank in sample_groups[key]:
                raise BenchmarkFormatError("duplicate rank in step sample")
            sample_groups[key][rank] = step_ms
        else:
            raise BenchmarkFormatError("unsupported timing_mode")

    if not region_groups:
        raise BenchmarkFormatError("no region_average rows")

    joined_regions = _join_complete(region_groups, common_world, "region repeat")
    repeat_values = []
    for (_, repeat), rank_rows in sorted(joined_regions.items(), key=lambda item: item[0][1]):
        step_counts = {value[0] for value in rank_rows.values()}
        if len(step_counts) != 1:
            raise BenchmarkFormatError("measured_steps differ across ranks")
        steps = next(iter(step_counts))
        repeat_values.append((repeat, max(region_ms / steps for steps, region_ms in rank_rows.values())))

    sample_values = []
    if sample_groups:
        joined_samples = _join_complete(sample_groups, common_world, "step sample")
        for (_, step_index), rank_rows in sorted(joined_samples.items(), key=lambda item: item[0][1]):
            sample_values.append((step_index, max(rank_rows.values())))

    repeat_max = [value for _, value in repeat_values]
    step_max = [value for _, value in sample_values]
    return {
        "schema": "mgt_a100_train_step_summary_v1",
        "world": common_world,
        "case": common_identity[_IDENTITY_FIELDS.index("case")],
        "repeat_max_avg_step_ms": repeat_max,
        "q50_step_ms": statistics.median(repeat_max),
        "diagnostic_step_max_ms": step_max,
        "diagnostic_q95_step_ms": _nearest_rank_q95(step_max) if step_max else None,
    }


def load_jsonl(lines: Iterable[str]) -> list[dict]:
    rows = []
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except (json.JSONDecodeError, ValueError) as exc:
            raise BenchmarkFormatError(f"malformed JSON at line {line_number}") from exc
        if not isinstance(row, dict):
            raise BenchmarkFormatError(f"line {line_number} is not a JSON object")
        rows.append(row)
    return rows


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("jsonl", help="per-rank benchmark JSONL")
    args = parser.parse_args(argv)
    try:
        with open(args.jsonl, "r", encoding="utf-8") as handle:
            summary = summarize_records(load_jsonl(handle))
    except (OSError, BenchmarkFormatError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(summary, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
