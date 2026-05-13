#!/usr/bin/env python3
"""Summarize small probe-window training validation runs."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


FIELDS = [
    "run_name",
    "precision_mode",
    "zo_quantization",
    "update_backend",
    "direction_type",
    "h",
    "sparse_rate",
    "h_active",
    "sparse_mode",
    "sparse_rescale",
    "lr",
    "steps_completed",
    "final_train_loss",
    "final_train_acc",
    "final_eval_loss",
    "final_acc",
    "best_acc",
    "nan_occurred",
]


def iter_summaries(run_root: Path) -> Iterable[Path]:
    yield from sorted(run_root.glob("*/seed*/run_summary.json"))


def read_csv(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def safe_float(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def first_accuracy(metrics: Dict[str, Any]) -> Optional[Any]:
    for key, value in metrics.items():
        if "acc" in str(key).lower():
            return value
    return None


def final_eval(summary: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {"final_eval_loss": None, "final_acc": None}
    for metrics in (summary.get("eval", {}) or {}).values():
        if not isinstance(metrics, dict):
            continue
        out["final_eval_loss"] = metrics.get("eval_loss", out["final_eval_loss"])
        acc = first_accuracy(metrics)
        if acc is not None:
            out["final_acc"] = acc
    return out


def summarize_one(path: Path) -> Dict[str, Any]:
    run_dir = path.parent
    run_name = run_dir.parent.name
    summary = json.loads(path.read_text(encoding="utf-8"))
    cfg = (summary.get("config", {}) or {}).get("training_args", {}) or {}
    metrics_rows = read_csv(run_dir / "metrics_logs" / "metrics_adaptiveH-0_cscale-0.csv")
    last_metrics = metrics_rows[-1] if metrics_rows else {}
    acc_values = [safe_float(row.get("eval_acc")) for row in metrics_rows]
    acc_values = [x for x in acc_values if x is not None]
    sparse_rate = safe_float(cfg.get("zo_direction_sparse_rate"))
    h_value = safe_float(cfg.get("zero_order_eps"))
    h_active = None
    if h_value is not None and sparse_rate is not None and sparse_rate > 0:
        if str(cfg.get("zo_sparse_rescale", "none")) == "inv_sqrt_p":
            h_active = h_value / math.sqrt(sparse_rate)
        else:
            h_active = h_value
    eval_info = final_eval(summary)
    train = summary.get("train", {}) or {}
    nan_occurred = any(
        any(safe_float(value) is None and str(value).lower() in {"nan", "inf", "-inf"} for value in row.values())
        for row in metrics_rows
    )
    direction_type = "sparse" if sparse_rate is not None and sparse_rate < 1.0 else "dense"
    return {
        "run_name": run_name,
        "precision_mode": cfg.get("precision_mode") or cfg.get("zo_two_point_precision"),
        "zo_quantization": cfg.get("zo_quantization"),
        "update_backend": cfg.get("zo_update_backend"),
        "direction_type": direction_type,
        "h": h_value,
        "sparse_rate": sparse_rate,
        "h_active": h_active,
        "sparse_mode": cfg.get("zo_direction_sparse_mode"),
        "sparse_rescale": cfg.get("zo_sparse_rescale"),
        "lr": cfg.get("learning_rate"),
        "steps_completed": train.get("global_step"),
        "final_train_loss": last_metrics.get("train_loss"),
        "final_train_acc": last_metrics.get("train_acc"),
        "final_eval_loss": eval_info["final_eval_loss"],
        "final_acc": eval_info["final_acc"],
        "best_acc": max(acc_values) if acc_values else eval_info["final_acc"],
        "nan_occurred": nan_occurred,
    }


def format_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def maybe_write_plots(rows: List[Dict[str, Any]], run_root: Path) -> None:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        return

    plots_dir = run_root / "plots"
    plots_dir.mkdir(exist_ok=True)

    dense = [row for row in rows if row.get("direction_type") == "dense"]
    sparse = [row for row in rows if row.get("direction_type") == "sparse"]

    def plot_grouped(data: List[Dict[str, Any]], x_key: str, y_key: str, group_key: str, filename: str, ylabel: str) -> None:
        grouped: Dict[str, List[Dict[str, Any]]] = {}
        for row in data:
            x = safe_float(row.get(x_key))
            y = safe_float(row.get(y_key))
            if x is None or y is None:
                continue
            grouped.setdefault(str(row.get(group_key, "")), []).append(row)
        if not grouped:
            return
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for group, group_rows in sorted(grouped.items()):
            points = sorted(
                ((safe_float(row.get(x_key)), safe_float(row.get(y_key))) for row in group_rows),
                key=lambda item: item[0] if item[0] is not None else 0.0,
            )
            xs = [point[0] for point in points if point[0] is not None and point[1] is not None]
            ys = [point[1] for point in points if point[0] is not None and point[1] is not None]
            if xs and ys:
                ax.plot(xs, ys, marker="o", label=group)
        ax.set_xscale("log")
        ax.set_xlabel(x_key)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(plots_dir / filename, dpi=160)
        plt.close(fig)

    plot_grouped(dense, "h", "final_acc", "precision_mode", "dense_training_final_acc_vs_h.png", "final_acc")
    plot_grouped(dense, "h", "final_eval_loss", "precision_mode", "dense_training_eval_loss_vs_h.png", "final_eval_loss")
    plot_grouped(sparse, "h_active", "final_acc", "sparse_rate", "sparse_training_final_acc_vs_h_active.png", "final_acc")
    plot_grouped(sparse, "h_active", "final_eval_loss", "sparse_rate", "sparse_training_eval_loss_vs_h_active.png", "final_eval_loss")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root")
    args = parser.parse_args()
    run_root = Path(args.run_root).resolve()
    rows = [summarize_one(path) for path in iter_summaries(run_root)]
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
    maybe_write_plots(rows, run_root)
    print(f"rows={len(rows)}")
    print(f"summary_csv={csv_path}")
    print(f"summary_md={md_path}")
    plots_dir = run_root / "plots"
    if plots_dir.exists():
        print(f"plots_dir={plots_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
