#!/usr/bin/env python3
"""Generate plots for future probe-window training experiments."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional


def safe_float(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out


def read_csv(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def metrics_for_run(run_dir: Path) -> List[Dict[str, Any]]:
    return read_csv(run_dir / "metrics_logs" / "metrics_adaptiveH-0_cscale-0.csv")


def plot_eval_acc_by_group(root: Path, rows: List[Dict[str, Any]], family: str, group_keys: List[str], filename: str) -> None:
    import matplotlib.pyplot as plt  # type: ignore

    fig, ax = plt.subplots(figsize=(8, 5))
    plotted = False
    for row in rows:
        if row.get("family") != family:
            continue
        run_name = str(row.get("run_name"))
        seed = str(row.get("seed"))
        run_dir = root / run_name / f"seed{seed}"
        metrics = metrics_for_run(run_dir)
        points = []
        for m in metrics:
            step = safe_float(m.get("global_step"))
            acc = safe_float(m.get("eval_acc"))
            if step is not None and acc is not None:
                points.append((step, acc))
        if not points:
            continue
        points.sort()
        label = ", ".join(f"{k}={row.get(k)}" for k in group_keys)
        ax.plot([p[0] for p in points], [p[1] for p in points], marker="o", linewidth=1.2, label=label)
        plotted = True
    if not plotted:
        plt.close(fig)
        return
    ax.set_xlabel("step")
    ax.set_ylabel("eval_acc")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(root / "plots" / filename, dpi=160)
    plt.close(fig)


def scatter(rows: List[Dict[str, Any]], x_key: str, y_key: str, group_key: str, out: Path, xlabel: str, ylabel: str, logx: bool = False) -> None:
    import matplotlib.pyplot as plt  # type: ignore

    groups: Dict[str, List[tuple[float, float]]] = defaultdict(list)
    for row in rows:
        x = safe_float(row.get(x_key))
        y = safe_float(row.get(y_key))
        if x is None or y is None:
            continue
        groups[str(row.get(group_key, ""))].append((x, y))
    if not groups:
        return
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for label, points in sorted(groups.items()):
        points.sort()
        ax.plot([p[0] for p in points], [p[1] for p in points], marker="o", linestyle="-", label=label)
    if logx:
        ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root")
    args = parser.parse_args()
    root = Path(args.run_root).resolve()
    (root / "plots").mkdir(exist_ok=True)

    try:
        import matplotlib  # noqa: F401
    except Exception:
        print("matplotlib unavailable; skipping plots")
        return 0

    rows = read_csv(root / "summary_all.csv")
    if not rows:
        print("summary_all.csv missing or empty; run summarize_future_probe_window_training.py first")
        return 1
    dense = [r for r in rows if r.get("family") == "dense"]
    sparse = [r for r in rows if r.get("family") == "sparse"]
    checkpoint = read_csv(root / "summary_checkpoint_probe.csv")

    plot_eval_acc_by_group(root, rows, "dense", ["h_raw", "seed"], "dense_eval_acc_vs_step_by_h.png")
    plot_eval_acc_by_group(root, rows, "sparse", ["p", "h_active", "lr"], "sparse_eval_acc_vs_step_by_p_hactive_lr.png")
    scatter(sparse, "lr", "best_acc", "h_active", root / "plots" / "sparse_best_acc_vs_lr_by_hactive.png", "lr", "best_acc", logx=True)
    scatter(sparse, "h_active", "best_acc", "p", root / "plots" / "sparse_best_acc_vs_hactive_by_p.png", "h_active", "best_acc", logx=True)
    scatter(checkpoint, "step", "corr_fd_true", "run_name", root / "plots" / "checkpoint_corr_fd_true_vs_step.png", "checkpoint step", "corr_fd_true")

    # If initial probe summaries are available under the previous packaged run, relate initial corr to training.
    probe_summary = Path("runs/probe_window_dense_20260512_193200/summary.csv")
    probe_rows = read_csv(probe_summary)
    if probe_rows and dense:
        by_h = {}
        for row in probe_rows:
            if row.get("precision_mode") == "int8" and row.get("direction_type") == "dense":
                h = safe_float(row.get("h_raw"))
                corr = safe_float(row.get("corr_fd_true"))
                if h is not None and corr is not None:
                    by_h[round(h, 12)] = corr
        relation = []
        for row in dense:
            h = safe_float(row.get("h_raw"))
            acc = safe_float(row.get("best_acc"))
            if h is not None and acc is not None and round(h, 12) in by_h:
                relation.append({"initial_corr_fd_true": by_h[round(h, 12)], "best_acc": acc, "h_raw": h})
        scatter(relation, "initial_corr_fd_true", "best_acc", "h_raw", root / "plots" / "probe_corr_vs_training_best_acc.png", "initial corr_fd_true", "best_acc")

    print(f"plots_dir={root / 'plots'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
