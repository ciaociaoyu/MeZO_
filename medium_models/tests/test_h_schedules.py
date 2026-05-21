import math
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
        h_schedule_fd_clip_max=1e-2,
        h_schedule_fd_int8_policy="capped_stress",
    )
    base.update(overrides)
    return SimpleNamespace(**base)


class HScheduleTest(unittest.TestCase):
    def test_mezo_default_returns_zero_order_eps(self):
        h, meta = resolve_h_schedule(_args(h_schedule="mezo_default", zero_order_eps=3e-4), step=7)
        self.assertAlmostEqual(h, 3e-4)
        self.assertAlmostEqual(meta["raw_h"], 3e-4)
        self.assertEqual(meta["canonical_schedule"], "mezo_default")

    def test_fd_eps13_fp32_returns_machine_epsilon_third(self):
        expected = float(np.finfo(np.float32).eps ** (1.0 / 3.0))
        h, meta = resolve_h_schedule(_args(h_schedule="fd_eps13", precision_mode="fp32"), step=0)
        self.assertAlmostEqual(meta["raw_h"], expected)
        self.assertAlmostEqual(h, expected)
        self.assertTrue(meta["fd_principled"])

    def test_fd_eps13_fp16_raw_and_capped_final(self):
        expected_raw = float(np.finfo(np.float16).eps ** (1.0 / 3.0))
        h, meta = resolve_h_schedule(_args(h_schedule="fd_eps13", precision_mode="fp16"), step=0)
        self.assertAlmostEqual(meta["raw_h"], expected_raw)
        self.assertAlmostEqual(h, 1e-2)
        self.assertTrue(meta["fd_principled"])
        self.assertIn("fp16", meta["cap_reason"])
        self.assertTrue(meta["window_clipped"])

    def test_fd_eps13_int8_capped_stress(self):
        h, meta = resolve_h_schedule(_args(h_schedule="fd_eps13", precision_mode="int8"), step=0)
        self.assertAlmostEqual(h, 1e-2)
        self.assertFalse(meta["fd_principled"])
        self.assertIn("no machine-epsilon analogue", meta["fd_exception_reason"])

    def test_fd_eps13_int8_skip_raises(self):
        args = _args(h_schedule="fd_eps13", precision_mode="int8", h_schedule_fd_int8_policy="skip")
        with self.assertRaisesRegex(ValueError, "machine-epsilon analogue"):
            resolve_h_schedule(args, step=0)

    def test_fd_eps13_respects_lower_safety_floor(self):
        raw = float(np.finfo(np.float32).eps ** (1.0 / 3.0))
        args = _args(
            h_schedule="fd_eps13",
            precision_mode="fp32",
            h_schedule_fd_clip_min=1e-2,
            h_schedule_fd_clip_max=1e-1,
        )
        h, meta = resolve_h_schedule(args, step=0)
        self.assertLess(raw, 1e-2)
        self.assertAlmostEqual(h, 1e-2)
        self.assertTrue(meta["window_clipped"])
        self.assertIn("safety min", meta["cap_reason"])

    def test_spall_ck_decreases_continuously(self):
        args = _args(h_schedule="spall_ck", h_schedule_h0=1e-3, h_schedule_gamma=0.101)
        h0, meta0 = resolve_h_schedule(args, step=0)
        h9, meta9 = resolve_h_schedule(args, step=9)
        self.assertAlmostEqual(h0, 1e-3)
        self.assertLess(h9, h0)
        self.assertFalse(meta0["grid_used"])
        self.assertFalse(meta9["grid_used"])

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

    def test_window_clipping_metadata(self):
        args = _args(
            h_schedule="spall_ck",
            h_schedule_h0=1e-1,
            h_schedule_window_min=1e-5,
            h_schedule_window_max=1e-2,
        )
        h, meta = resolve_h_schedule(args, step=0)
        self.assertAlmostEqual(h, 1e-2)
        self.assertTrue(meta["window_clipped"])
        self.assertIn("safety max", meta["cap_reason"])

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
