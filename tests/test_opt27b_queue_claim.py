"""Queue ownership must survive independent workers on a shared filesystem."""
import multiprocessing as mp
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from opt27b_radius_extension import distributed_claim, recover_training_logs, safe_aggregate, append, records


def hold(root, ready, release):
    with distributed_claim(Path(root), "one_run") as acquired:
        ready.put(acquired)
        release.wait(30)


class ClaimTest(unittest.TestCase):
    def test_reporting_failure_does_not_raise(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch("opt27b_radius_extension.aggregate", side_effect=ImportError("simulated plotting failure")):
                self.assertFalse(safe_aggregate(root))
            self.assertEqual(len(records(root / "logs/aggregation_failures.jsonl")), 1)

    def test_restart_archives_uncheckpointed_records(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for step in range(4):
                append(root / "train.jsonl", {"step": step})
                append(root / "eval.jsonl", {"step": step})
            recover_training_logs(root, 3)
            self.assertEqual([r["step"] for r in records(root / "train.jsonl")], [0, 1, 2])
            archives = list((root / "interrupted_records").glob("*/train.jsonl"))
            self.assertEqual(len(records(archives[0])), 4)
            recover_training_logs(root, 1)
            self.assertEqual(records(root / "train.jsonl"), [])
            self.assertEqual(records(root / "eval.jsonl"), [])

    def test_exclusive_and_released(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "locks").mkdir()
            (root / "logs").mkdir()
            ctx = mp.get_context("fork")
            ready, release = ctx.Queue(), ctx.Event()
            process = ctx.Process(target=hold, args=(directory, ready, release))
            process.start()
            try:
                self.assertTrue(ready.get(timeout=20))
                with distributed_claim(root, "one_run") as duplicate:
                    self.assertFalse(duplicate)
                with distributed_claim(root, "different_run") as other:
                    self.assertTrue(other)
            finally:
                release.set()
                process.join(timeout=20)
            self.assertEqual(process.exitcode, 0)
            with distributed_claim(root, "one_run") as reclaimed:
                self.assertTrue(reclaimed)


if __name__ == "__main__":
    unittest.main()
