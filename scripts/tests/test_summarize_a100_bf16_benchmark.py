import math
import unittest

from scripts.summarize_a100_bf16_benchmark import BenchmarkFormatError, summarize_records


IDENTITY = {
    "schema": "mgt_a100_train_step_v1",
    "world": 2,
    "case": "full",
    "row_vector": [5, 5],
    "pair_index": 3,
    "pair_attempt_nonce": "nonce-a",
    "source_sha": "a" * 40,
    "policy_sha256": "b" * 64,
    "snapshot_sha256": "c" * 64,
    "runtime_tree_sha256": "d" * 64,
    "status": "ok",
}


def region(rank, repeat, region_ms, measured_steps=10, **changes):
    row = dict(IDENTITY)
    row.update({
        "rank": rank,
        "timing_mode": "region_average",
        "repeat": repeat,
        "measured_steps": measured_steps,
        "region_ms": region_ms,
    })
    row.update(changes)
    return row


def sample(rank, step_index, duration_ms, **changes):
    row = dict(IDENTITY)
    row.update({
        "rank": rank,
        "timing_mode": "step_samples",
        "step_index": step_index,
        "step_ms": duration_ms,
    })
    row.update(changes)
    return row


class SummarizeBenchmarkTest(unittest.TestCase):
    def test_max_rank_per_repeat_then_q50(self):
        rows = [
            region(0, 0, 100.0), region(1, 0, 120.0),
            region(0, 1, 90.0), region(1, 1, 110.0),
            region(0, 2, 130.0), region(1, 2, 125.0),
        ]
        summary = summarize_records(rows)
        self.assertEqual(summary["repeat_max_avg_step_ms"], [12.0, 11.0, 13.0])
        self.assertEqual(summary["q50_step_ms"], 12.0)
        self.assertIsNone(summary["diagnostic_q95_step_ms"])

    def test_step_samples_take_max_rank_then_nearest_rank_q95(self):
        rows = [
            region(0, 0, 100.0), region(1, 0, 110.0),
            sample(0, 0, 4.0), sample(1, 0, 5.0),
            sample(0, 1, 8.0), sample(1, 1, 7.0),
            sample(0, 2, 6.0), sample(1, 2, 9.0),
        ]
        summary = summarize_records(rows)
        self.assertEqual(summary["diagnostic_step_max_ms"], [5.0, 8.0, 9.0])
        self.assertEqual(summary["diagnostic_q95_step_ms"], 9.0)

    def assertRejected(self, rows):
        with self.assertRaises(BenchmarkFormatError):
            summarize_records(rows)

    def test_rejects_missing_rank(self):
        self.assertRejected([region(0, 0, 100.0)])

    def test_rejects_duplicate_rank(self):
        self.assertRejected([region(0, 0, 100.0), region(0, 0, 101.0), region(1, 0, 102.0)])

    def test_rejects_unequal_measured_step_count(self):
        self.assertRejected([region(0, 0, 100.0, 10), region(1, 0, 100.0, 11)])

    def test_rejects_source_or_policy_mismatch(self):
        self.assertRejected([region(0, 0, 100.0), region(1, 0, 100.0, source_sha="e" * 40)])
        self.assertRejected([region(0, 0, 100.0), region(1, 0, 100.0, policy_sha256="e" * 64)])

    def test_rejects_full_tail_mixing(self):
        self.assertRejected([region(0, 0, 100.0), region(1, 0, 100.0, case="tail", row_vector=[4, 3])])

    def test_rejects_nonfinite_duration(self):
        self.assertRejected([region(0, 0, math.inf), region(1, 0, 100.0)])
        self.assertRejected([sample(0, 0, math.nan), sample(1, 0, 1.0), region(0, 0, 1.0), region(1, 0, 1.0)])

    def test_rejects_malformed_json_line(self):
        from scripts.summarize_a100_bf16_benchmark import load_jsonl
        with self.assertRaises(BenchmarkFormatError):
            load_jsonl(["{broken\n"])


if __name__ == "__main__":
    unittest.main()
