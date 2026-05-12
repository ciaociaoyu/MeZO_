#!/usr/bin/env python3
"""Summarize short INT8 residual_grid diagnostic runs."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


FIELDS = [
    "run_name",
    "backend",
    "h",
    "lr",
    "residual_dtype",
    "commit_mode",
    "max_code_step",
    "update_norm_clip",
    "steps_completed",
    "final_train_loss",
    "final_eval_loss",
    "best_acc",
    "final_acc",
    "global_active_frac_last",
    "global_cos_intended_actual_last",
    "global_actual_over_intended_norm_ratio_last",
    "saturation_frac_last",
    "residual_over_scale_p99_last",
    "residual_over_scale_max_last",
    "unsaturated_residual_bound_violation_frac_last",
    "grid_error_norm_last",
    "scale_drift_max",
    "nan_occurred",
]


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def is_bad_number(value: Any) -> bool:
    if isinstance(value, float):
        return not math.isfinite(value)
    return False


def load_last_metrics_csv(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return rows[-1] if rows else {}


def first_accuracy(metrics: Dict[str, Any]) -> Optional[Any]:
    for key, value in metrics.items():
        if "acc" in str(key).lower():
            return value
    return None


def max_scale_drift(run_dir: Path) -> Optional[float]:
    path = run_dir / "scale_drift.csv"
    if not path.exists():
        return None
    max_val = 0.0
    found = False
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            raw = row.get("scale_delta_max")
            if raw in (None, ""):
                continue
            try:
                value = float(raw)
            except ValueError:
                continue
            if math.isfinite(value):
                max_val = max(max_val, value)
                found = True
    return max_val if found else None


def iter_run_summaries(run_root: Path) -> Iterable[Path]:
    yield from sorted(run_root.glob("*/seed*/run_summary.json"))


def summarize_run(summary_path: Path) -> Dict[str, Any]:
    run_dir = summary_path.parent
    run_name = run_dir.parent.name
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    cfg = (summary.get("config", {}) or {}).get("training_args", {}) or {}
    updates = read_jsonl(run_dir / "update_stats.jsonl")
    last = updates[-1] if updates else ((summary.get("artifacts", {}) or {}).get("update_stats_last_row", {}) or {})
    metrics_last = load_last_metrics_csv(run_dir / "metrics_logs" / "metrics_adaptiveH-0_cscale-0.csv")
    eval_loss = None
    final_acc = None
    for metrics in (summary.get("eval", {}) or {}).values():
        if isinstance(metrics, dict):
            eval_loss = metrics.get("eval_loss", eval_loss)
            acc = first_accuracy(metrics)
            final_acc = acc if acc is not None else final_acc
    train_loss = metrics_last.get("train_loss") or last.get("train_loss")
    nan_occurred = any(any(is_bad_number(v) for v in row.values()) for row in updates)
    return {
        "run_name": run_name,
        "backend": cfg.get("zo_update_backend", last.get("update_backend")),
        "h": cfg.get("zero_order_eps", last.get("h")),
        "lr": cfg.get("learning_rate", last.get("lr")),
        "residual_dtype": cfg.get("residual_dtype", ""),
        "commit_mode": cfg.get("residual_commit_mode", ""),
        "max_code_step": cfg.get("residual_max_code_step", ""),
        "update_norm_clip": cfg.get("zo_update_norm_clip", last.get("zo_update_norm_clip")),
        "steps_completed": len(updates) or ((summary.get("train", {}) or {}).get("global_step")),
        "final_train_loss": train_loss,
        "final_eval_loss": eval_loss,
        "best_acc": final_acc,
        "final_acc": final_acc,
        "global_active_frac_last": last.get("global_active_frac", last.get("active_frac")),
        "global_cos_intended_actual_last": last.get("global_cos_intended_actual", last.get("cos_intended_actual")),
        "global_actual_over_intended_norm_ratio_last": last.get(
            "global_actual_over_intended_norm_ratio", last.get("actual_over_intended_norm_ratio")
        ),
        "saturation_frac_last": last.get("global_saturation_frac", last.get("saturation_frac")),
        "residual_over_scale_p99_last": last.get("residual_over_scale_p99"),
        "residual_over_scale_max_last": last.get("residual_over_scale_max"),
        "unsaturated_residual_bound_violation_frac_last": last.get("unsaturated_residual_bound_violation_frac"),
        "grid_error_norm_last": last.get("grid_error_norm"),
        "scale_drift_max": max_scale_drift(run_dir),
        "nan_occurred": nan_occurred,
    }


def format_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root")
    args = parser.parse_args()
    run_root = Path(args.run_root).resolve()
    rows = [summarize_run(path) for path in iter_run_summaries(run_root)]
    csv_path = run_root / "summary.csv"
    md_path = run_root / "summary.md"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    with md_path.open("w", encoding="utf-8") as f:
        f.write("| " + " | ".join(FIELDS) + " |\n")
        f.write("| " + " | ".join(["---"] * len(FIELDS)) + " |\n")
        for row in rows:
            f.write("| " + " | ".join(format_cell(row.get(field)) for field in FIELDS) + " |\n")
    print(f"summary_csv={csv_path}")
    print(f"summary_md={md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
