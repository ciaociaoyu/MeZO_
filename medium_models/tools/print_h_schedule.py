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
    parser.add_argument("--h_schedule", choices=sorted(H_SCHEDULE_CHOICES), default="fixed")
    parser.add_argument("--h_schedule_grid", default="")
    parser.add_argument("--h_schedule_window_min", type=float, default=0.0)
    parser.add_argument("--h_schedule_window_max", type=float, default=0.0)
    parser.add_argument("--h_schedule_h0", type=float, default=0.0)
    parser.add_argument("--h_schedule_gamma", type=float, default=0.101)
    parser.add_argument("--h_schedule_total_steps", type=int, default=0)
    parser.add_argument("--h_schedule_d_eff", type=float, default=1.0)
    parser.add_argument("--h_schedule_n_eff", type=float, default=1.0)
    parser.add_argument("--h_schedule_lipschitz_l", type=float, default=0.0)
    parser.add_argument("--h_schedule_c_delta", type=float, default=1.0)
    parser.add_argument("--h_schedule_log_csv", action=argparse.BooleanOptionalAction, default=True)
    return parser


def main() -> int:
    parser = build_parser()
    ns = parser.parse_args()
    if ns.steps <= 0:
        raise ValueError("--steps must be > 0")
    args = SimpleNamespace(**vars(ns))

    rows = []
    for step in range(int(ns.steps)):
        h_value, meta = resolve_h_schedule(args, step)
        rows.append({
            "step": step,
            "h_schedule": meta["schedule"],
            "raw_h": meta["raw_h"],
            "final_h": h_value,
            "window_min": meta["window_min"],
            "window_max": meta["window_max"],
            "grid_used": meta["grid_used"],
            "grid": ns.h_schedule_grid,
            "precision_mode": ns.precision_mode,
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
