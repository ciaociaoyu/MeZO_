"""Finite-difference perturbation-radius schedules for schedule-only MeZO baselines."""

from __future__ import annotations

import math
import re
from typing import Any


H_SCHEDULE_CHOICES = {
    "fixed",
    "spall_clip",
    "shamir_clip",
    "ji_sqrtk_clip",
    "ji_theory_clip",
    "pf_vrzo_clip",
}


def parse_h_grid(grid_str: str) -> list[float]:
    """Parse comma/whitespace-separated h values."""
    raw = str(grid_str or "").strip()
    if not raw:
        return []
    values = []
    for token in re.split(r"[\s,]+", raw):
        if not token:
            continue
        try:
            value = float(token)
        except ValueError as exc:
            raise ValueError(f"Invalid h grid value {token!r}") from exc
        if (not math.isfinite(value)) or value <= 0.0:
            raise ValueError(f"h grid values must be finite positive floats; got {token!r}")
        values.append(value)
    return values


def clip_to_window(h: float, h_min: float, h_max: float) -> float:
    """Clip h to a positive-sided optional window."""
    h_val = float(h)
    min_val = float(h_min or 0.0)
    max_val = float(h_max or 0.0)
    if min_val > 0.0 and max_val > 0.0 and min_val > max_val:
        raise ValueError(f"h schedule window_min ({min_val}) must be <= window_max ({max_val})")
    if min_val > 0.0:
        h_val = max(h_val, min_val)
    if max_val > 0.0:
        h_val = min(h_val, max_val)
    return float(h_val)


def nearest_grid(h: float, grid: list[float]) -> float:
    """Map h to the closest grid point, preserving grid order for ties."""
    if not grid:
        return float(h)
    return float(min(grid, key=lambda value: abs(float(value) - float(h))))


def _float_attr(args: Any, name: str, default: float) -> float:
    try:
        return float(getattr(args, name, default))
    except Exception:
        return float(default)


def _int_attr(args: Any, name: str, default: int) -> int:
    try:
        return int(getattr(args, name, default))
    except Exception:
        return int(default)


def _base_h(args: Any) -> float:
    h0 = _float_attr(args, "h_schedule_h0", 0.0)
    if h0 > 0.0:
        return h0
    window_max = _float_attr(args, "h_schedule_window_max", 0.0)
    if window_max > 0.0:
        return window_max
    return _float_attr(args, "zero_order_eps", 1e-3)


def resolve_h_schedule(args: Any, step: int) -> tuple[float, dict]:
    """Resolve the active h value and metadata for a zero-based optimizer step."""
    try:
        step_i = int(step)
    except Exception as exc:
        raise ValueError(f"h schedule step must be an integer, got {step!r}") from exc
    if step_i < 0:
        raise ValueError(f"h schedule step must be >= 0, got {step_i}")

    schedule = str(getattr(args, "h_schedule", "fixed") or "fixed").strip().lower()
    if schedule not in H_SCHEDULE_CHOICES:
        raise ValueError(f"Invalid h_schedule={schedule!r}. Allowed: {sorted(H_SCHEDULE_CHOICES)}")

    denom = float(step_i + 1)
    zero_order_eps = _float_attr(args, "zero_order_eps", 1e-3)

    if schedule == "fixed":
        raw_h = zero_order_eps
    elif schedule == "spall_clip":
        raw_h = _base_h(args) / (denom ** _float_attr(args, "h_schedule_gamma", 0.101))
    elif schedule == "shamir_clip":
        total_steps = _int_attr(args, "h_schedule_total_steps", 0)
        if total_steps <= 0:
            total_steps = _int_attr(args, "max_steps", 1)
        total_steps = max(int(total_steps), 1)
        raw_h = (
            _float_attr(args, "h_schedule_c_delta", 1.0)
            * _base_h(args)
            * math.sqrt(_float_attr(args, "h_schedule_d_eff", 1.0) / float(total_steps))
        )
    elif schedule == "ji_sqrtk_clip":
        raw_h = _base_h(args) / math.sqrt(denom)
    elif schedule == "ji_theory_clip":
        lipschitz_l = _float_attr(args, "h_schedule_lipschitz_l", 0.0)
        if lipschitz_l <= 0.0:
            raise ValueError("--h_schedule_lipschitz_l must be > 0 for h_schedule=ji_theory_clip")
        d_eff = _float_attr(args, "h_schedule_d_eff", 1.0)
        if d_eff <= 0.0:
            raise ValueError("--h_schedule_d_eff must be > 0 for h_schedule=ji_theory_clip")
        raw_h = 1.0 / (lipschitz_l * math.sqrt(d_eff * denom))
    elif schedule == "pf_vrzo_clip":
        raw_h = _base_h(args) / denom
    else:
        raise AssertionError(f"Unhandled h schedule {schedule!r}")

    if (not math.isfinite(raw_h)) or raw_h <= 0.0:
        raise ValueError(f"h_schedule={schedule} produced invalid raw_h={raw_h}")

    window_min = _float_attr(args, "h_schedule_window_min", 0.0)
    window_max = _float_attr(args, "h_schedule_window_max", 0.0)
    clipped_h = raw_h
    if window_min > 0.0 or window_max > 0.0:
        clipped_h = clip_to_window(raw_h, window_min, window_max)

    grid_str = str(getattr(args, "h_schedule_grid", "") or "")
    grid = parse_h_grid(grid_str)
    final_h = nearest_grid(clipped_h, grid) if grid else clipped_h

    meta = {
        "raw_h": float(raw_h),
        "final_h": float(final_h),
        "schedule": schedule,
        "step": int(step_i),
        "window_min": float(window_min),
        "window_max": float(window_max),
        "grid_used": bool(grid),
        "grid": list(grid),
        "grid_str": grid_str,
    }
    return float(final_h), meta
