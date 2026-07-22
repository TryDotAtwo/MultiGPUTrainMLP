import json
import tempfile
import unittest
from pathlib import Path

from autotune_2xt4 import (
    build_fingerprint,
    cache_matches,
    drift_is_acceptable,
    select_stable_candidate,
    render_selected_env,
    reuse_decision,
)


class Autotune2xT4Tests(unittest.TestCase):
    def test_fingerprint_is_order_independent_and_exact(self):
        left = build_fingerprint({"gpu": ["T4", "T4"], "batch": 53248, "tile": 56})
        right = build_fingerprint({"tile": 56, "batch": 53248, "gpu": ["T4", "T4"]})
        changed = build_fingerprint({"gpu": ["T4", "T4"], "batch": 49152, "tile": 56})
        self.assertEqual(left, right)
        self.assertNotEqual(left, changed)

    def test_cache_fails_closed_on_schema_or_fingerprint_mismatch(self):
        expected = "abc"
        self.assertTrue(cache_matches({"schema": 1, "fingerprint": expected}, expected))
        self.assertFalse(cache_matches({"schema": 2, "fingerprint": expected}, expected))
        self.assertFalse(cache_matches({"schema": 1, "fingerprint": "other"}, expected))
        self.assertFalse(cache_matches({}, expected))

    def test_selects_best_conservative_throughput_not_fastest_average(self):
        rows = [
            {"config_id": "fragile", "status": "ok", "avg_throughput_states_s": 600.0,
             "min_throughput_states_s": 410.0},
            {"config_id": "stable", "status": "ok", "avg_throughput_states_s": 570.0,
             "min_throughput_states_s": 555.0},
            {"config_id": "failed", "status": "failed", "avg_throughput_states_s": 999.0,
             "min_throughput_states_s": 999.0},
        ]
        self.assertEqual(select_stable_candidate(rows)["config_id"], "stable")

    def test_render_selected_env_is_shell_safe_and_complete(self):
        row = {
            "config": {
                "batch_size": 53248,
                "input_grad_position_tile": 56,
                "lt_workspace_bytes": 16777216,
                "allreduce_bucket_bytes": 4194304,
                "input_grad_sparse": 0,
                "cutlass_half_gemm_kinds": "input_embedding_grad,forward",
                "lt_autotune": 1,
            }
        }
        text = render_selected_env(row)
        self.assertIn("MGT_BATCH_SIZE=53248", text)
        self.assertIn("MGT_INPUT_GRAD_POSITION_TILE=56", text)
        self.assertIn("MGT_CUTLASS_HALF_GEMM_KINDS='input_embedding_grad,forward'", text)
        self.assertTrue(text.endswith("\n"))

    def test_reuse_decision_fails_closed_and_rechecks_drift(self):
        cache = {"schema": 1, "fingerprint": "fp", "baseline_throughput_states_s": 600.0}
        self.assertEqual(reuse_decision(cache, "other", None), "full_tune")
        self.assertEqual(reuse_decision(cache, "fp", None), "quick_check")
        self.assertEqual(reuse_decision(cache, "fp", 520.0), "reuse")
        self.assertEqual(reuse_decision(cache, "fp", 500.0), "full_tune")

    def test_drift_check_requires_finite_threshold(self):
        self.assertTrue(drift_is_acceptable(510.0, 600.0, 0.85))
        self.assertFalse(drift_is_acceptable(509.9, 600.0, 0.85))
        self.assertFalse(drift_is_acceptable(float("nan"), 600.0, 0.85))
        self.assertFalse(drift_is_acceptable(600.0, 0.0, 0.85))


if __name__ == "__main__":
    unittest.main()
