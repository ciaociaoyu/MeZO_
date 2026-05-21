import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "medium_models"))

from src.h_schedules import parse_h_grid, resolve_h_schedule  # noqa: E402


def _args(**overrides):
    base = dict(
        zero_order_eps=1e-3,
        max_steps=100,
        precision_mode="fp32",
        zo_two_point_precision="fp32",
        zo_quantization_bits=32,
        h_schedule="fixed",
        h_schedule_grid="",
        h_schedule_grid_policy="continuous",
        h_schedule_window_min=1e-5,
        h_schedule_window_max=1e-2,
        h_schedule_h0=0.0,
        h_schedule_gamma=0.101,
        h_schedule_total_steps=0,
        h_schedule_d_eff=1.0,
        h_schedule_n_eff=1.0,
        h_schedule_lipschitz_l=0.0,
        h_schedule_c_delta=1.0,
        h_schedule_fd_clip_min=1e-5,
        h_schedule_fd_clip_policy="none",
        h_schedule_fd_floor_min=1e-5,
        h_schedule_fd_clip_max=0.0,
        h_schedule_fd_int8_policy="fp16_proxy_raw",
        h_schedule_allow_out_of_window=True,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


class HScheduleTest(unittest.TestCase):
    def test_mezo_default_returns_zero_order_eps(self):
        h, meta = resolve_h_schedule(_args(h_schedule="mezo_default", zero_order_eps=1e-3), step=7)
        self.assertAlmostEqual(h, 1e-3)
        self.assertAlmostEqual(meta["raw_h"], 1e-3)
        self.assertEqual(meta["canonical_schedule"], "mezo_default")

    def test_fixed_small_returns_onee_minus_five(self):
        args = _args(h_schedule="fixed_small")
        h0, meta0 = resolve_h_schedule(args, step=0)
        h7, meta7 = resolve_h_schedule(args, step=7)
        self.assertEqual(h0, 1e-5)
        self.assertEqual(h7, 1e-5)
        self.assertEqual(meta0["canonical_schedule"], "fixed_small_1e-5")
        self.assertEqual(meta7["baseline_role"], "fixed_small_h_1e-5")

    def test_fixed_small_rejects_wrong_h0(self):
        args = _args(h_schedule="fixed_small", h_schedule_h0=3e-5)
        with self.assertRaisesRegex(ValueError, "fixed_small"):
            resolve_h_schedule(args, step=0)

    def test_fd_eps13_raw_fp32_returns_machine_epsilon_third(self):
        expected = float(np.finfo(np.float32).eps ** (1.0 / 3.0))
        h, meta = resolve_h_schedule(_args(h_schedule="fd_eps13_raw", precision_mode="fp32"), step=0)
        self.assertAlmostEqual(expected, 0.004921565763652325)
        self.assertAlmostEqual(meta["raw_h"], expected)
        self.assertAlmostEqual(h, expected)
        self.assertTrue(meta["fd_principled"])
        self.assertFalse(meta["out_of_window_raw"])

    def test_fd_eps13_raw_fp16_is_not_capped(self):
        expected = float(np.finfo(np.float16).eps ** (1.0 / 3.0))
        h, meta = resolve_h_schedule(_args(h_schedule="fd_eps13_raw", precision_mode="fp16"), step=0)
        self.assertAlmostEqual(expected, 0.0992431640625)
        self.assertAlmostEqual(meta["raw_h"], expected)
        self.assertAlmostEqual(h, expected)
        self.assertTrue(meta["fd_principled"])
        self.assertTrue(meta["out_of_window_raw"])
        self.assertEqual(meta["cap_reason"], "")

    def test_fd_eps13_raw_int8_uses_fp16_proxy_raw(self):
        expected = float(np.finfo(np.float16).eps ** (1.0 / 3.0))
        h, meta = resolve_h_schedule(_args(h_schedule="fd_eps13_raw", precision_mode="int8"), step=0)
        self.assertAlmostEqual(h, expected)
        self.assertAlmostEqual(meta["raw_h"], expected)
        self.assertFalse(meta["fd_principled"])
        self.assertIn("no machine-epsilon analogue", meta["fd_exception_reason"])
        self.assertTrue(meta["out_of_window_raw"])

    def test_fd_eps13_raw_int8_skip_raises(self):
        args = _args(h_schedule="fd_eps13_raw", precision_mode="int8", h_schedule_fd_int8_policy="skip")
        with self.assertRaisesRegex(ValueError, "machine-epsilon analogue"):
            resolve_h_schedule(args, step=0)

    def test_fd_eps13_raw_capped_stress_int8_is_marked_unprincipled(self):
        args = _args(
            h_schedule="fd_eps13_raw",
            precision_mode="int8",
            h_schedule_fd_int8_policy="capped_stress",
            h_schedule_fd_clip_max=1e-2,
        )
        h, meta = resolve_h_schedule(args, step=0)
        self.assertAlmostEqual(h, 1e-2)
        self.assertFalse(meta["fd_principled"])
        self.assertIn("capped stress", meta["fd_exception_reason"])

    def test_fd_eps13_lower_floor_policy_can_floor_future_small_values(self):
        args = _args(
            h_schedule="fd_eps13_raw",
            precision_mode="fp32",
            h_schedule_fd_clip_policy="lower_floor_only",
            h_schedule_fd_floor_min=1e-2,
        )
        h, meta = resolve_h_schedule(args, step=0)
        self.assertAlmostEqual(h, 1e-2)
        self.assertEqual(meta["cap_reason"], "raw h below safety min")

    def test_continuous_grid_policy_does_not_snap(self):
        args = _args(
            zero_order_eps=2.4e-3,
            h_schedule="mezo_default",
            h_schedule_grid="1e-3, 3e-3 1e-2",
            h_schedule_grid_policy="continuous",
        )
        parsed = parse_h_grid(args.h_schedule_grid)
        for actual, expected in zip(parsed, [1e-3, 3e-3, 1e-2]):
            self.assertAlmostEqual(actual, expected)
        h, meta = resolve_h_schedule(args, step=0)
        self.assertAlmostEqual(h, 2.4e-3)
        self.assertFalse(meta["grid_used"])

    def test_nearest_grid_policy_snaps_when_requested(self):
        args = _args(
            zero_order_eps=2.4e-3,
            h_schedule="mezo_default",
            h_schedule_grid="1e-3, 3e-3 1e-2",
            h_schedule_grid_policy="nearest",
        )
        h, meta = resolve_h_schedule(args, step=0)
        self.assertAlmostEqual(h, 3e-3)
        self.assertTrue(meta["grid_used"])

    def test_legacy_ji_theory_clip_raises_if_lipschitz_l_is_nonpositive(self):
        args = _args(h_schedule="ji_theory_clip", h_schedule_lipschitz_l=0.0)
        with self.assertRaisesRegex(ValueError, "h_schedule_lipschitz_l"):
            resolve_h_schedule(args, step=0)

    def test_legacy_pf_vrzo_clip_decreases_as_inverse_step_plus_one(self):
        args = _args(h_schedule="pf_vrzo_clip", h_schedule_h0=1e-2)
        h0, _ = resolve_h_schedule(args, step=0)
        h4, _ = resolve_h_schedule(args, step=4)
        self.assertAlmostEqual(h0, 1e-2)
        self.assertAlmostEqual(h4, 2e-3)


if __name__ == "__main__":
    unittest.main()
