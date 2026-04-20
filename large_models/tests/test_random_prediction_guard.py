import sys
import unittest
from pathlib import Path
from types import SimpleNamespace


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "large_models"))

from trainer import OurTrainer  # noqa: E402


class LargeRandomPredictionGuardTest(unittest.TestCase):
    def _build_trainer(self, *, task_name: str, global_step: int = 2000, num_labels: int = 0):
        trainer = OurTrainer.__new__(OurTrainer)
        trainer.args = SimpleNamespace(
            task_name=task_name,
            output_dir="/tmp/mezo_large_random_guard_test",
            random_prediction_guard_enabled=True,
            random_prediction_guard_step=2000,
            random_prediction_guard_acc_tolerance=0.05,
            random_prediction_guard_loss_tolerance=0.03,
            random_prediction_guard_bad_loss_excess=0.5,
            random_prediction_guard_recent_evals=2,
            random_prediction_guard_min_loss_drop=0.05,
            random_prediction_guard_min_acc_gain=0.02,
        )
        trainer.state = SimpleNamespace(
            global_step=global_step,
            log_history=[{"loss": 1.60}, {"eval_loss": 1.61}, {"loss": 1.58}],
        )
        trainer.model = SimpleNamespace(config=SimpleNamespace(num_labels=num_labels))
        trainer._eval_history = []
        trainer._eval_loss_history = []
        trainer._random_prediction_guard_initial_train_loss = 1.60
        return trainer

    def test_num_label_resolution_for_supported_pilot_tasks(self):
        self.assertEqual(self._build_trainer(task_name="SST-5")._random_prediction_guard_num_labels(), 5)
        self.assertEqual(self._build_trainer(task_name="MNLI")._random_prediction_guard_num_labels(), 3)
        self.assertEqual(self._build_trainer(task_name="BoolQ")._random_prediction_guard_num_labels(), 2)

    def test_extract_eval_acc_accepts_mnli_and_boolq_metric_keys(self):
        trainer = self._build_trainer(task_name="MNLI")
        self.assertEqual(trainer._extract_eval_acc({"eval_mnli/acc": 0.47}), 0.47)
        self.assertEqual(trainer._extract_eval_acc({"eval_accuracy": 0.73}), 0.73)

    def test_random_prediction_guard_triggers_on_random_plateau_for_boolq(self):
        trainer = self._build_trainer(task_name="BoolQ")
        trainer._eval_history = [
            {"global_step": 1000, "eval_loss": 0.70, "eval_acc": 0.52},
            {"global_step": 2000, "eval_loss": 0.70, "eval_acc": 0.51},
        ]
        payload = trainer._random_prediction_guard_payload(
            train_loss=1.58,
            eval_loss=0.70,
            eval_acc=0.51,
            eval_loss_avg5=0.70,
        )
        self.assertIsNotNone(payload)
        self.assertEqual(payload["reason"], "random_plateau")
        self.assertEqual(payload["num_labels"], 2)

    def test_random_prediction_guard_allows_clear_progress_for_sst5(self):
        trainer = self._build_trainer(task_name="SST-5")
        trainer._eval_history = [
            {"global_step": 1000, "eval_loss": 1.58, "eval_acc": 0.21},
            {"global_step": 2000, "eval_loss": 1.44, "eval_acc": 0.33},
        ]
        payload = trainer._random_prediction_guard_payload(
            train_loss=1.40,
            eval_loss=1.44,
            eval_acc=0.33,
            eval_loss_avg5=1.50,
        )
        self.assertIsNone(payload)

    def test_random_prediction_guard_can_trigger_on_severe_eval_loss_for_mnli(self):
        trainer = self._build_trainer(task_name="MNLI")
        trainer._eval_history = [
            {"global_step": 1000, "eval_loss": 1.65, "eval_acc": 0.41},
            {"global_step": 2000, "eval_loss": 1.72, "eval_acc": 0.40},
        ]
        payload = trainer._random_prediction_guard_payload(
            train_loss=1.65,
            eval_loss=1.72,
            eval_acc=0.40,
            eval_loss_avg5=1.69,
        )
        self.assertIsNotNone(payload)
        self.assertEqual(payload["reason"], "severe_eval_loss")
        self.assertEqual(payload["num_labels"], 3)


if __name__ == "__main__":
    unittest.main()
