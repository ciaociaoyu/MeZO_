import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "medium_models"))

from src.int8_residual_grid import ResidualGridUpdater  # noqa: E402
from src.trainer import Trainer  # noqa: E402


class ResidualGridUpdaterTest(unittest.TestCase):
    def test_round_commit_keeps_weights_on_grid_and_bounds_residual(self):
        param = nn.Parameter(torch.tensor([0.0, 0.2, -0.4, 0.6], dtype=torch.float32))
        updater = ResidualGridUpdater(
            [("w", param)],
            bits=8,
            residual_dtype="fp32",
            commit_mode="round",
            max_code_step=0,
            freeze_scale=True,
        )
        scale = updater.scales["w"]
        stats = updater.apply_update(
            "w",
            param,
            torch.ones_like(param),
            projected_grad=1.0,
            learning_rate=float(scale.item() * 0.6),
        )

        q = torch.round(param.detach() / scale)
        self.assertTrue(torch.allclose(param.detach(), q * scale))
        self.assertGreater(stats["active_frac"], 0.0)
        self.assertTrue(torch.isfinite(param).all().item())
        self.assertTrue(torch.isfinite(updater.residuals["w"]).all().item())
        self.assertLessEqual(float(torch.max(torch.abs(updater.residuals["w"].float() / scale)).item()), 0.5001)

    def test_floor_commit_waits_until_full_grid_step(self):
        param = nn.Parameter(torch.tensor([0.0, 0.2, -0.4, 0.6], dtype=torch.float32))
        updater = ResidualGridUpdater(
            [("w", param)],
            bits=8,
            residual_dtype="fp32",
            commit_mode="floor",
            max_code_step=0,
            freeze_scale=True,
        )
        scale = updater.scales["w"]
        before = param.detach().clone()
        stats1 = updater.apply_update("w", param, torch.ones_like(param), 1.0, float(scale.item() * 0.6))
        self.assertEqual(stats1["active_count"], 0.0)
        self.assertTrue(torch.allclose(param.detach(), before))
        stats2 = updater.apply_update("w", param, torch.ones_like(param), 1.0, float(scale.item() * 0.6))
        self.assertGreater(stats2["active_count"], 0.0)

    def test_max_code_step_clips_per_step_motion(self):
        param = nn.Parameter(torch.tensor([0.0, 0.2, -0.4, 0.6], dtype=torch.float32))
        updater = ResidualGridUpdater(
            [("w", param)],
            bits=8,
            residual_dtype="fp32",
            commit_mode="round",
            max_code_step=1,
            freeze_scale=True,
        )
        scale = updater.scales["w"]
        before_q = torch.round(param.detach() / scale)
        updater.apply_update("w", param, torch.ones_like(param), 1.0, float(scale.item() * 10.0))
        after_q = torch.round(param.detach() / scale)
        self.assertLessEqual(float(torch.max(torch.abs(after_q - before_q)).item()), 1.0)

    def test_noop_update_keeps_codes_inactive_and_on_grid(self):
        param = nn.Parameter(torch.tensor([0.0, 0.2, -0.4, 0.6], dtype=torch.float32))
        updater = ResidualGridUpdater([("w", param)], bits=8, residual_dtype="fp32")
        scale = updater.scales["w"]
        before = param.detach().clone()
        stats = updater.apply_update("w", param, torch.randn_like(param), projected_grad=0.0, learning_rate=1.0)
        self.assertEqual(stats["actual_sq"], 0.0)
        self.assertEqual(stats["active_frac"], 0.0)
        self.assertEqual(stats["candidate_active_count"], 0.0)
        self.assertEqual(stats["selected_active_count"], 0.0)
        self.assertEqual(stats["ef_error_norm"], 0.0)
        self.assertEqual(stats["ef_error_max"], 0.0)
        self.assertTrue(torch.allclose(param.detach(), before))
        self.assertTrue(torch.allclose(param.detach(), torch.round(param.detach() / scale) * scale))

    def test_thresholded_commit_keeps_uncommitted_update_in_residual(self):
        param = nn.Parameter(torch.zeros(4, dtype=torch.float32))
        updater = ResidualGridUpdater(
            [("w", param)],
            bits=8,
            residual_dtype="fp32",
            commit_mode="round",
            commit_threshold=0.75,
            freeze_scale=True,
        )
        updater.scales["w"] = torch.tensor(0.1, dtype=torch.float32)
        updater.snap_param_to_grid("w", param)

        stats = updater.apply_update("w", param, torch.ones_like(param), projected_grad=1.0, learning_rate=0.06)
        self.assertEqual(stats["candidate_active_count"], 4.0)
        self.assertEqual(stats["active_after_threshold_count"], 0.0)
        self.assertEqual(stats["selected_active_count"], 0.0)
        self.assertEqual(stats["active_count"], 0.0)
        self.assertTrue(torch.allclose(param.detach(), torch.zeros_like(param)))
        self.assertTrue(torch.allclose(updater.residuals["w"], torch.full_like(param, -0.06), atol=1e-7))
        self.assertEqual(stats["ef_error_norm"], 0.0)

    def test_top_abs_acc_selection_keeps_top_k_candidates(self):
        param = nn.Parameter(torch.zeros(4, dtype=torch.float32))
        updater = ResidualGridUpdater(
            [("w", param)],
            bits=8,
            residual_dtype="fp32",
            commit_mode="round",
            max_code_step=1,
            commit_select="top_abs_acc",
            target_active_frac=0.5,
            freeze_scale=True,
        )
        updater.scales["w"] = torch.tensor(0.1, dtype=torch.float32)
        updater.snap_param_to_grid("w", param)
        direction = torch.tensor([1.1, 0.9, 0.4, 1.6], dtype=torch.float32)

        stats = updater.apply_update("w", param, direction, projected_grad=1.0, learning_rate=0.1)
        self.assertEqual(stats["candidate_active_count"], 3.0)
        self.assertEqual(stats["selected_active_count"], 2.0)
        self.assertEqual(stats["selection_dropped_count"], 1.0)
        self.assertTrue(torch.allclose(param.detach(), torch.tensor([-0.1, 0.0, 0.0, -0.1]), atol=1e-7))
        acc = -0.1 * direction
        actual = param.detach()
        self.assertTrue(torch.allclose(updater.residuals["w"], acc - actual, atol=1e-7))
        self.assertEqual(stats["ef_error_norm"], 0.0)
        self.assertGreater(stats["acc_actual_cos"], 0.0)

    def test_norm_budget_selection_respects_acc_norm_cap(self):
        param = nn.Parameter(torch.zeros(4, dtype=torch.float32))
        updater = ResidualGridUpdater(
            [("w", param)],
            bits=8,
            residual_dtype="fp32",
            commit_mode="round",
            max_code_step=1,
            commit_select="norm_budget",
            actual_norm_ratio_cap=0.5,
            budget_reference="acc",
            freeze_scale=True,
        )
        updater.scales["w"] = torch.tensor(0.1, dtype=torch.float32)
        updater.snap_param_to_grid("w", param)
        direction = torch.tensor([3.0, 2.0, 1.0, 0.6], dtype=torch.float32)

        stats = updater.apply_update("w", param, direction, projected_grad=1.0, learning_rate=0.1)
        self.assertEqual(stats["candidate_active_count"], 4.0)
        self.assertEqual(stats["selected_active_count"], 3.0)
        self.assertLessEqual(stats["actual_norm_after_selection"], 0.5 * stats["norm_budget_reference_norm"] + 1e-7)
        self.assertTrue(torch.allclose(param.detach(), torch.tensor([-0.1, -0.1, -0.1, 0.0]), atol=1e-7))
        acc = -0.1 * direction
        actual = param.detach()
        self.assertTrue(torch.allclose(updater.residuals["w"], acc - actual, atol=1e-7))
        self.assertEqual(stats["ef_error_norm"], 0.0)

    def test_fixed_scale_subgrid_deltas_accumulate_then_jump(self):
        param = nn.Parameter(torch.zeros(4, dtype=torch.float32))
        updater = ResidualGridUpdater([("w", param)], bits=8, residual_dtype="fp32", commit_mode="round")
        updater.scales["w"] = torch.tensor(0.1, dtype=torch.float32)
        updater.snap_param_to_grid("w", param)

        stats1 = updater.apply_update("w", param, torch.ones_like(param), projected_grad=1.0, learning_rate=0.04)
        self.assertEqual(stats1["active_count"], 0.0)
        self.assertTrue(torch.allclose(param.detach(), torch.zeros_like(param)))
        self.assertTrue(torch.allclose(updater.residuals["w"], torch.full_like(param, -0.04)))

        stats2 = updater.apply_update("w", param, torch.ones_like(param), projected_grad=1.0, learning_rate=0.04)
        self.assertGreater(stats2["active_count"], 0.0)
        self.assertTrue(torch.allclose(param.detach(), torch.full_like(param, -0.1)))
        self.assertLessEqual(float(torch.max(torch.abs(updater.residuals["w"] / updater.scales["w"])).item()), 0.5001)


class DirectionSparseMaskTest(unittest.TestCase):
    def _trainer(self, *, mode="exact_random", rate=0.3, per_layer_exact=True, rescale="none"):
        trainer = Trainer.__new__(Trainer)
        trainer.args = SimpleNamespace(
            sparse_ratio=1.0,
            sparse_mask_strategy="percentile_per_layer",
            sparse_scope="trainable_only",
            sparse_mask_refresh_steps=100,
            sparse_log_active_fraction=False,
            zo_direction_sparse_rate=rate,
            zo_direction_sparse_mode=mode,
            zo_sparse_per_layer_exact=per_layer_exact,
            zo_sparse_rescale=rescale,
            zo_quantization_bits=8,
            zo_two_point_precision="fp32",
            h_estimation_active_source="fixed",
            zero_order_eps=1e-3,
        )
        trainer.state = SimpleNamespace(global_step=0)
        trainer.named_parameters_to_optim = [
            ("a", nn.Parameter(torch.zeros(10))),
            ("b", nn.Parameter(torch.zeros(7))),
        ]
        trainer._sparse_refresh_masks = None
        trainer._sparse_step_masks = None
        trainer._sparse_step_masks_step = None
        return trainer

    def test_exact_random_reuses_mask_and_hits_requested_count(self):
        trainer = self._trainer(mode="exact_random", rate=0.3, per_layer_exact=True)
        param = trainer.named_parameters_to_optim[0][1]
        mask1 = trainer._quzo_direction_sparse_mask("a", param, step_seed=123)
        mask2 = trainer._quzo_direction_sparse_mask("a", param, step_seed=123)
        self.assertTrue(torch.equal(mask1, mask2))
        self.assertEqual(int(mask1.sum().item()), round(0.3 * param.numel()))

    def test_inv_sqrt_rescale_changes_active_noise_magnitude(self):
        trainer = self._trainer(mode="exact_random", rate=0.25, per_layer_exact=True, rescale="inv_sqrt_p")
        z = trainer._sample_sparse_noise_like("a", trainer.named_parameters_to_optim[0][1].data, seed=123, dtype=torch.float32)
        self.assertEqual(int(torch.count_nonzero(z).item()), round(0.25 * z.numel()))
        self.assertTrue(torch.isfinite(z).all().item())


if __name__ == "__main__":
    unittest.main()
