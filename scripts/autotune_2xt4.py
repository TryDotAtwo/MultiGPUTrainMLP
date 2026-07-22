"""Pure selection and cache rules for the 2xT4 autotuner."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Iterable, Mapping

CACHE_SCHEMA = 1


def build_fingerprint(context: Mapping[str, Any]) -> str:
    payload = json.dumps(context, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def cache_matches(cache: Mapping[str, Any], fingerprint: str) -> bool:
    return cache.get("schema") == CACHE_SCHEMA and cache.get("fingerprint") == fingerprint


def select_stable_candidate(rows: Iterable[Mapping[str, Any]]) -> Mapping[str, Any]:
    valid = []
    for row in rows:
        if row.get("status") != "ok":
            continue
        conservative = float(row.get("min_throughput_states_s", float("nan")))
        average = float(row.get("avg_throughput_states_s", float("nan")))
        if math.isfinite(conservative) and conservative > 0 and math.isfinite(average) and average > 0:
            valid.append((conservative, average, str(row.get("config_id", "")), row))
    if not valid:
        raise ValueError("no stable successful autotune candidates")
    return max(valid, key=lambda item: (item[0], item[1], item[2]))[3]


def select_final_stage_candidate(stages: Iterable[Iterable[Mapping[str, Any]]]) -> Mapping[str, Any]:
    final_rows = None
    for rows in stages:
        materialized = list(rows)
        if materialized:
            final_rows = materialized
    if final_rows is None:
        raise ValueError("no autotune stages")
    return select_stable_candidate(final_rows)


def drift_is_acceptable(measured: float, cached: float, minimum_ratio: float = 0.85) -> bool:
    return (
        math.isfinite(measured)
        and math.isfinite(cached)
        and math.isfinite(minimum_ratio)
        and measured > 0
        and cached > 0
        and 0 < minimum_ratio <= 1
        and measured >= cached * minimum_ratio
    )
_ENV_FIELDS = {
    "batch_size": "MGT_BATCH_SIZE",
    "input_grad_position_tile": "MGT_INPUT_GRAD_POSITION_TILE",
    "lt_workspace_bytes": "MGT_LT_WORKSPACE_BYTES",
    "allreduce_bucket_bytes": "MGT_ALLREDUCE_BUCKET_BYTES",
    "input_grad_sparse": "MGT_INPUT_GRAD_SPARSE",
    "cutlass_half_gemm_kinds": "MGT_CUTLASS_HALF_GEMM_KINDS",
    "lt_autotune": "MGT_LT_AUTOTUNE",
}


def render_selected_env(row: Mapping[str, Any]) -> str:
    config = row.get("config")
    if not isinstance(config, Mapping):
        raise ValueError("selected row is missing config")
    lines = []
    for field, variable in _ENV_FIELDS.items():
        if field not in config:
            raise ValueError(f"selected row is missing {field}")
        value = str(config[field])
        rendered = value if value.isdigit() else "'" + value.replace("'", "'\\''") + "'"
        lines.append(f"{variable}={rendered}")
    return "\n".join(lines) + "\n"


def reuse_decision(cache: Mapping[str, Any], fingerprint: str,
                   measured_throughput: float | None,
                   minimum_ratio: float = 0.85) -> str:
    if not cache_matches(cache, fingerprint):
        return "full_tune"
    if measured_throughput is None:
        return "quick_check"
    try:
        cached = float(cache["baseline_throughput_states_s"])
    except (KeyError, TypeError, ValueError):
        return "full_tune"
    return "reuse" if drift_is_acceptable(measured_throughput, cached, minimum_ratio) else "full_tune"