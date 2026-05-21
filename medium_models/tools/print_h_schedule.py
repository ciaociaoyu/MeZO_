#!/usr/bin/env python
"""Print resolved h schedule values for quick diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from types import SimpleNamespace


MEDIUM_ROOT = Path(__file__).resolve().parents[1]
if str(MEDIUM_ROOT) not in sys.path:
    sys.path.insert(0, str(MEDIUM_ROOT))

from src.h_schedules import H_SCHEDULE_CHOICES, resolve_h_schedule  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Print first N finite-difference h schedule values.")
    parser.add_argument("--steps", "-n", type=int, default=10, help="Number of zero-based steps to print.")
    parser.add_argument("--format", choices=["jsonl", "csv"], default="jsonl", help="Output format.")
    parser.add_argument("--zero_order_eps", type=float, default=1e-3)
    parser.add_argument("--max_steps", type=int, default=0)
    parser.add_argument("--precision_mode", default="")
    parser.add_argument("--zo_two_point_precision", default="fp32")
    parser.add_argument("--zo_quantization_bits", type=int, default=32)
    parser.add_argument("--h_schedule", choices=sorted(H_SCHEDULE_CHOICES), default="fixed")
    parser.add_argument("--h_schedule_grid", default="")
    parser.add_argument("--h_schedule_grid_policy", choices=["continuous", "nearest", "floor", "ceil"], default="continuous")
    parser.add_argument("--h_schedule_window_min", type=float, default=0.0)
    parser.add_argument("--h_schedule_window_max", type=float, default=0.0)
    parser.add_argument("--h_schedule_h0", type=float, default=0.0)
    parser.add_argument("--h_schedule_gamma", type=float, default=0.101)
    parser.add_argument("--h_schedule_total_steps", type=int, default=0)
    parser.add_argument("--h_schedule_d_eff", type=float, default=1.0)
    parser.add_argument("--h_schedule_n_eff", type=float, default=1.0)
    parser.add_argument("--h_schedule_lipschitz_l", type=float, default=0.0)
    parser.add_argument("--h_schedule_c_delta", type=float, default=1.0)
    parser.add_argument("--h_schedule_fd_clip_min", type=float, default=1e-5)
    parser.add_argument("--h_schedule_fd_clip_policy", choices=["none", "lower_floor_only", "cap", "skip"], default="none")
    parser.add_argument("--h_schedule_fd_floor_min", type=float, default=1e-5)
    parser.add_argument("--h_schedule_fd_clip_max", type=float, default=0.0)
    parser.add_argument("--h_schedule_fd_int8_policy", choices=["fp16_proxy_raw", "capped_stress", "skip"], default="fp16_proxy_raw")
    parser.add_argument("--h_schedule_allow_out_of_window", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--h_schedule_log_csv", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--include_steps",
        default="",
        help="Optional comma/space-separated extra zero-based steps to print, e.g. 20000.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    ns = parser.parse_args()
    if ns.steps <= 0:
        raise ValueError("--steps must be > 0")
    args = SimpleNamespace(**vars(ns))

    steps = list(range(int(ns.steps)))
    if ns.include_steps:
        for token in ns.include_steps.replace(",", " ").split():
            step = int(token)
            if step < 0:
                raise ValueError("--include_steps must be >= 0")
            if step not in steps:
                steps.append(step)
    rows = []
    for step in steps:
        h_value, meta = resolve_h_schedule(args, step)
        rows.append({
            "step": step,
            "h_schedule": meta["schedule"],
            "canonical_schedule": meta["canonical_schedule"],
            "raw_h": meta["raw_h"],
            "final_h": h_value,
            "precision_mode": meta["precision_mode"],
            "fd_principled": meta["fd_principled"],
            "fd_exception_reason": meta["fd_exception_reason"],
            "cap_reason": meta["cap_reason"],
            "out_of_window_raw": meta["out_of_window_raw"],
            "out_of_window_reason": meta["out_of_window_reason"],
            "baseline_role": meta["baseline_role"],
            "window_min": meta["window_min"],
            "window_max": meta["window_max"],
            "grid_policy": meta["grid_policy"],
            "grid_used": meta["grid_used"],
            "grid": ns.h_schedule_grid,
            "zero_order_eps": ns.zero_order_eps,
        })

    if ns.format == "jsonl":
        for row in rows:
            print(json.dumps(row, sort_keys=True))
        return 0

    writer = csv.DictWriter(sys.stdout, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
