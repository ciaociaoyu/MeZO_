"""Finite-difference perturbation-radius schedules for schedule-only MeZO baselines."""

from __future__ import annotations

import bisect
import math
import re
from typing import Any

import numpy as np


H_SCHEDULE_CHOICES = {
    "fixed",
    "mezo_default",
    "fd_eps13",
    "spall_ck",
    # Legacy schedule-only baselines kept for old scripts.
    "spall_clip",
    "shamir_clip",
    "ji_sqrtk_clip",
    "ji_theory_clip",
    "pf_vrzo_clip",
}

H_GRID_POLICIES = {"continuous", "nearest", "floor", "ceil"}
H_FD_INT8_POLICIES = {"capped_stress", "skip"}


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


def _grid_snap(h: float, grid: list[float], policy: str) -> float:
    policy = str(policy or "continuous").strip().lower()
    if policy == "continuous":
        return float(h)
    if policy not in H_GRID_POLICIES:
        raise ValueError(f"--h_schedule_grid_policy must be one of {sorted(H_GRID_POLICIES)}")
    if not grid:
        raise ValueError(f"--h_schedule_grid_policy={policy} requires nonempty --h_schedule_grid")
    sorted_grid = sorted(float(v) for v in grid)
    h_val = float(h)
    if policy == "nearest":
        return nearest_grid(h_val, sorted_grid)
    idx = bisect.bisect_right(sorted_grid, h_val)
    if policy == "floor":
        return float(sorted_grid[max(0, idx - 1)])
    if policy == "ceil":
        return float(sorted_grid[min(len(sorted_grid) - 1, idx)])
    raise AssertionError(f"Unhandled h grid policy {policy!r}")


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


def _str_attr(args: Any, name: str, default: str) -> str:
    try:
        return str(getattr(args, name, default) or default)
    except Exception:
        return str(default)


def _base_h(args: Any, *, legacy_window_fallback: bool = True) -> float:
    h0 = _float_attr(args, "h_schedule_h0", 0.0)
    if h0 > 0.0:
        return h0
    if legacy_window_fallback:
        window_max = _float_attr(args, "h_schedule_window_max", 0.0)
        if window_max > 0.0:
            return window_max
    return _float_attr(args, "zero_order_eps", 1e-3)


def _infer_precision_mode(args: Any) -> str:
    precision = _str_attr(args, "precision_mode", "").strip().lower()
    if precision:
        return precision
    two_point = _str_attr(args, "zo_two_point_precision", "").strip().lower()
    if two_point in {"fp32", "fp16", "bf16"}:
        return two_point
    bits = _int_attr(args, "zo_quantization_bits", 0)
    if bits == 8:
        return "int8"
    if bits == 16:
        return "fp16"
    if bits == 32:
        return "fp32"
    return "fp32"


def _fd_eps13_raw(args: Any) -> tuple[float, bool, str, str]:
    precision = _infer_precision_mode(args)
    if precision == "fp32":
        return float(np.finfo(np.float32).eps ** (1.0 / 3.0)), True, "", ""
    if precision in {"fp16", "float16"}:
        return (
            float(np.finfo(np.float16).eps ** (1.0 / 3.0)),
            True,
            "",
            "",
        )
    if precision == "int8":
        policy = _str_attr(args, "h_schedule_fd_int8_policy", "capped_stress").strip().lower()
        if policy not in H_FD_INT8_POLICIES:
            raise ValueError(f"--h_schedule_fd_int8_policy must be one of {sorted(H_FD_INT8_POLICIES)}")
        if policy == "skip":
            raise ValueError(
                "--h_schedule fd_eps13 with precision_mode=int8 is undefined because INT8 has no "
                "machine-epsilon analogue; use --h_schedule_fd_int8_policy capped_stress to run the "
                "capped stress baseline."
            )
        h_max = _float_attr(args, "h_schedule_fd_clip_max", 1e-2)
        return (
            float(h_max),
            False,
            "INT8 has no machine-epsilon analogue; using capped stress baseline",
            "",
        )
    return float(np.finfo(np.float32).eps ** (1.0 / 3.0)), True, "", ""


def resolve_h_schedule(args: Any, step: int) -> tuple[float, dict]:
    """Resolve the active h value and metadata for a zero-based optimizer step."""
    try:
        step_i = int(step)
    except Exception as exc:
        raise ValueError(f"h schedule step must be an integer, got {step!r}") from exc
    if step_i < 0:
        raise ValueError(f"h schedule step must be >= 0, got {step_i}")

    schedule = _str_attr(args, "h_schedule", "fixed").strip().lower()
    if schedule not in H_SCHEDULE_CHOICES:
        raise ValueError(f"Invalid h_schedule={schedule!r}. Allowed: {sorted(H_SCHEDULE_CHOICES)}")

    denom = float(step_i + 1)
    zero_order_eps = _float_attr(args, "zero_order_eps", 1e-3)
    precision_mode = _infer_precision_mode(args)
    fd_principled = True
    fd_exception_reason = ""
    cap_reason = ""
    h0 = 0.0
    gamma = _float_attr(args, "h_schedule_gamma", 0.101)

    canonical_schedule = schedule
    if schedule in {"fixed", "mezo_default"}:
        canonical_schedule = "mezo_default"
        raw_h = zero_order_eps
        clip_min = 0.0
        clip_max = 0.0
    elif schedule == "fd_eps13":
        canonical_schedule = "fd_eps13"
        raw_h, fd_principled, fd_exception_reason, cap_reason = _fd_eps13_raw(args)
        clip_min = _float_attr(args, "h_schedule_fd_clip_min", 1e-5)
        clip_max = _float_attr(args, "h_schedule_fd_clip_max", 1e-2)
    elif schedule == "spall_ck":
        canonical_schedule = "spall_ck"
        h0 = _base_h(args, legacy_window_fallback=False)
        raw_h = h0 / (denom ** gamma)
        clip_min = _float_attr(args, "h_schedule_window_min", 1e-5)
        clip_max = _float_attr(args, "h_schedule_window_max", 1e-2)
    elif schedule == "spall_clip":
        raw_h = _base_h(args) / (denom ** gamma)
        h0 = _base_h(args)
        clip_min = _float_attr(args, "h_schedule_window_min", 0.0)
        clip_max = _float_attr(args, "h_schedule_window_max", 0.0)
    elif schedule == "shamir_clip":
        total_steps = _int_attr(args, "h_schedule_total_steps", 0)
        if total_steps <= 0:
            total_steps = _int_attr(args, "max_steps", 1)
        total_steps = max(int(total_steps), 1)
        h0 = _base_h(args)
        raw_h = (
            _float_attr(args, "h_schedule_c_delta", 1.0)
            * h0
            * math.sqrt(_float_attr(args, "h_schedule_d_eff", 1.0) / float(total_steps))
        )
        clip_min = _float_attr(args, "h_schedule_window_min", 0.0)
        clip_max = _float_attr(args, "h_schedule_window_max", 0.0)
    elif schedule == "ji_sqrtk_clip":
        h0 = _base_h(args)
        raw_h = h0 / math.sqrt(denom)
        clip_min = _float_attr(args, "h_schedule_window_min", 0.0)
        clip_max = _float_attr(args, "h_schedule_window_max", 0.0)
    elif schedule == "ji_theory_clip":
        lipschitz_l = _float_attr(args, "h_schedule_lipschitz_l", 0.0)
        if lipschitz_l <= 0.0:
            raise ValueError("--h_schedule_lipschitz_l must be > 0 for h_schedule=ji_theory_clip")
        d_eff = _float_attr(args, "h_schedule_d_eff", 1.0)
        if d_eff <= 0.0:
            raise ValueError("--h_schedule_d_eff must be > 0 for h_schedule=ji_theory_clip")
        raw_h = 1.0 / (lipschitz_l * math.sqrt(d_eff * denom))
        clip_min = _float_attr(args, "h_schedule_window_min", 0.0)
        clip_max = _float_attr(args, "h_schedule_window_max", 0.0)
    elif schedule == "pf_vrzo_clip":
        h0 = _base_h(args)
        raw_h = h0 / denom
        clip_min = _float_attr(args, "h_schedule_window_min", 0.0)
        clip_max = _float_attr(args, "h_schedule_window_max", 0.0)
    else:
        raise AssertionError(f"Unhandled h schedule {schedule!r}")

    if (not math.isfinite(raw_h)) or raw_h <= 0.0:
        raise ValueError(f"h_schedule={schedule} produced invalid raw_h={raw_h}")

    clipped_h = clip_to_window(raw_h, clip_min, clip_max)
    window_clipped = not math.isclose(float(clipped_h), float(raw_h), rel_tol=0.0, abs_tol=0.0)
    if window_clipped and not cap_reason:
        if schedule == "fd_eps13" and precision_mode in {"fp16", "float16"} and clip_max > 0.0 and raw_h > clip_max:
            cap_reason = "fp16 eps^(1/3) exceeds safety max"
        elif clip_max > 0.0 and raw_h > clip_max:
            cap_reason = "raw h exceeds safety max"
        elif clip_min > 0.0 and raw_h < clip_min:
            cap_reason = "raw h below safety min"
        else:
            cap_reason = "raw h clipped to safety window"

    grid_str = _str_attr(args, "h_schedule_grid", "")
    grid_policy = _str_attr(args, "h_schedule_grid_policy", "continuous").strip().lower()
    if grid_policy not in H_GRID_POLICIES:
        raise ValueError(f"--h_schedule_grid_policy must be one of {sorted(H_GRID_POLICIES)}")
    grid = parse_h_grid(grid_str)
    final_h = _grid_snap(clipped_h, grid, grid_policy)
    grid_used = bool(grid) and grid_policy != "continuous"

    meta = {
        "schedule": schedule,
        "canonical_schedule": canonical_schedule,
        "step": int(step_i),
        "raw_h": float(raw_h),
        "clipped_h": float(clipped_h),
        "final_h": float(final_h),
        "precision_mode": precision_mode,
        "fd_principled": bool(fd_principled),
        "fd_exception_reason": fd_exception_reason,
        "cap_reason": cap_reason,
        "h0": float(h0),
        "gamma": float(gamma),
        "window_min": float(clip_min),
        "window_max": float(clip_max),
        "window_clipped": bool(window_clipped),
        "grid_policy": grid_policy,
        "grid_used": bool(grid_used),
        "grid": list(grid),
        "grid_str": grid_str,
        "zero_order_eps": float(zero_order_eps),
    }
    return float(final_h), meta
