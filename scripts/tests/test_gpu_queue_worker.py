import json
import pathlib
import sys
import tempfile
import unittest
from unittest import mock

from scripts import gpu_queue_worker


class GpuQueueWorkerTest(unittest.TestCase):
    def test_job_json_is_loaded_before_pending_to_running_rename(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            paths = gpu_queue_worker.ensure_dirs(root)
            job_path = paths["pending"] / "job.json"
            gpu_queue_worker.write_json(job_path, {
                "id": "load-before-rename",
                "label": "unit",
                "cwd": directory,
                "env": [],
                "command": [sys.executable, "-c", "print('ok')"],
            })
            original_load = gpu_queue_worker.load_json

            def reject_post_rename_load(path):
                self.assertEqual(path, job_path)
                return original_load(path)

            with mock.patch.object(
                    gpu_queue_worker, "load_json", side_effect=reject_post_rename_load):
                gpu_queue_worker.run_job(job_path, paths, root, cooldown_sec=0)

            done_path = paths["done"] / job_path.name
            self.assertTrue(done_path.is_file())
            result = json.loads(done_path.read_text(encoding="utf-8"))
            self.assertEqual(result["status"], "done")
            self.assertEqual(result["return_code"], 0)
            self.assertIn("ok", pathlib.Path(result["log_path"]).read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
