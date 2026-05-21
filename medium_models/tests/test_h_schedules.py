import math
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "medium_models"))

from src.h_schedules import parse_h_grid, resolve_h_schedule  # noqa: E402


def _args(**overrides):
    base = dict(
        zero_order_eps=1e-3,
        max_steps=100,
        h_schedule="fixed",
        h_schedule_grid="",
        h_schedule_window_min=0.0,
        h_schedule_window_max=0.0,
        h_schedule_h0=0.0,
        h_schedule_gamma=0.101,
        h_schedule_total_steps=0,
        h_schedule_d_eff=1.0,
        h_schedule_n_eff=1.0,
        h_schedule_lipschitz_l=0.0,
        h_schedule_c_delta=1.0,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


class HScheduleTest(unittest.TestCase):
    def test_fixed_returns_zero_order_eps(self):
        h, meta = resolve_h_schedule(_args(zero_order_eps=3e-4), step=7)
        self.assertAlmostEqual(h, 3e-4)
        self.assertAlmostEqual(meta["raw_h"], 3e-4)
        self.assertEqual(meta["schedule"], "fixed")

    def test_spall_clip_decreases_slowly_and_respects_window(self):
        args = _args(
            h_schedule="spall_clip",
            h_schedule_h0=1e-2,
            h_schedule_window_min=1e-5,
            h_schedule_window_max=1e-2,
        )
        h0, _ = resolve_h_schedule(args, step=0)
        h100, _ = resolve_h_schedule(args, step=100)
        self.assertAlmostEqual(h0, 1e-2)
        self.assertGreaterEqual(h100, 1e-5)
        self.assertLessEqual(h100, 1e-2)
        self.assertLess(h100, h0)
        self.assertGreater(h100, 1e-3)

    def test_shamir_clip_is_step_independent_and_respects_total_steps(self):
        args = _args(
            h_schedule="shamir_clip",
            h_schedule_h0=1e-2,
            h_schedule_total_steps=100,
            h_schedule_d_eff=4.0,
        )
        h0, _ = resolve_h_schedule(args, step=0)
        h999, _ = resolve_h_schedule(args, step=999)
        self.assertAlmostEqual(h0, 2e-3)
        self.assertAlmostEqual(h999, h0)

    def test_ji_sqrtk_clip_decreases_as_inverse_sqrt_step_plus_one(self):
        args = _args(h_schedule="ji_sqrtk_clip", h_schedule_h0=1e-2)
        h0, _ = resolve_h_schedule(args, step=0)
        h3, _ = resolve_h_schedule(args, step=3)
        self.assertAlmostEqual(h0, 1e-2)
        self.assertAlmostEqual(h3, 5e-3)
        self.assertAlmostEqual(h3, h0 / math.sqrt(4.0))

    def test_pf_vrzo_clip_decreases_as_inverse_step_plus_one(self):
        args = _args(h_schedule="pf_vrzo_clip", h_schedule_h0=1e-2)
        h0, _ = resolve_h_schedule(args, step=0)
        h4, _ = resolve_h_schedule(args, step=4)
        self.assertAlmostEqual(h0, 1e-2)
        self.assertAlmostEqual(h4, 2e-3)

    def test_grid_snapping_maps_to_nearest_grid(self):
        args = _args(
            zero_order_eps=2.4e-3,
            h_schedule="fixed",
            h_schedule_grid="1e-3, 3e-3 1e-2",
        )
        parsed = parse_h_grid(args.h_schedule_grid)
        for actual, expected in zip(parsed, [1e-3, 3e-3, 1e-2]):
            self.assertAlmostEqual(actual, expected)
        h, meta = resolve_h_schedule(args, step=0)
        self.assertAlmostEqual(h, 3e-3)
        self.assertTrue(meta["grid_used"])

    def test_ji_theory_clip_raises_if_lipschitz_l_is_nonpositive(self):
        args = _args(h_schedule="ji_theory_clip", h_schedule_lipschitz_l=0.0)
        with self.assertRaisesRegex(ValueError, "h_schedule_lipschitz_l"):
            resolve_h_schedule(args, step=0)


if __name__ == "__main__":
    unittest.main()
