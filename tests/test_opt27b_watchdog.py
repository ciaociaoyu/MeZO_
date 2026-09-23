import contextlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import opt27b_radius_watchdog as watchdog


class WatchdogTests(unittest.TestCase):
    def test_tail_skips_incomplete_last_line(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "records.jsonl"
            path.write_text('{"step": 10}\n{"step":')
            self.assertEqual(watchdog.tail_record(path), {"step": 10})

    def test_claim_excludes_second_audit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "logs/watchdog").mkdir(parents=True)
            with watchdog.claim(root) as first:
                self.assertTrue(first)
                with watchdog.claim(root) as second:
                    self.assertFalse(second)
            with watchdog.claim(root) as released:
                self.assertTrue(released)

    def test_only_declared_array_counts_as_training_worker(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            watchdog.save(root / "jobs.json", [{"id": "one", "kind": "train"}])
            watchdog.save(root / "l4_submission.json", {"job_id": "123"})
            unrelated = subprocess.CompletedProcess([], 0, "999_0|RUNNING|unrelated|node\n", "")
            with patch.object(watchdog.subprocess, "run", return_value=unrelated):
                state = watchdog.snapshot(root)
            self.assertEqual(state["workers"], [])
            self.assertTrue(state["issues"])
            own = subprocess.CompletedProcess([], 0, "123_0|RUNNING|our_worker|node\n", "")
            with patch.object(watchdog.subprocess, "run", return_value=own):
                state = watchdog.snapshot(root)
            self.assertEqual(state["issues"], [])

    def test_extra_array_counts_as_training_worker(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            watchdog.save(root / "jobs.json", [{"id": "one", "kind": "train"}])
            watchdog.save(root / "l4_submission.json", {"job_id": "123"})
            watchdog.save(root / "l4_submission_extra.json", {"job_id": "456"})
            queue = subprocess.CompletedProcess([], 0, "456_2|RUNNING|our_worker|node\n", "")
            with patch.object(watchdog.subprocess, "run", return_value=queue):
                state = watchdog.snapshot(root)
            self.assertEqual(state["array_ids"], ["123", "456"])
            self.assertEqual(state["workers"], ["456_2|RUNNING|our_worker|node"])
            self.assertEqual(state["issues"], [])

    def test_healthy_audit_does_not_invoke_agent(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "logs/watchdog").mkdir(parents=True)
            state = {"time": "now", "runs": [], "issues": [], "all_terminal": True}
            with patch.object(watchdog, "snapshot", return_value=state), patch.object(watchdog, "invoke_agent") as agent:
                watchdog.audit(root, agent=True)
            agent.assert_not_called()

    def test_active_cpu_handoff_not_duplicated(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            watchdog.save(root / "logs/watchdog/next_cpu_job.json", {"job_id": "123"})
            active = subprocess.CompletedProcess([], 0, "PENDING\n", "")
            with patch.object(watchdog.subprocess, "run", return_value=active), patch.object(watchdog.subprocess, "check_output") as submit:
                self.assertEqual(watchdog.schedule_next(root, 0, True), "123")
            submit.assert_not_called()


if __name__ == "__main__":
    unittest.main()
