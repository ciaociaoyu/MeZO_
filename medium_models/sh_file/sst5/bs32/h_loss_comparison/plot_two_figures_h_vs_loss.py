#!/usr/bin/env python3

import csv
import math
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path("/Users/jichaoyu/Documents/GitHub/MeZO")
OUT_DIR = ROOT / "medium_models/sh_file/sst5/bs32/h_loss_comparison"

LEFT_SOURCE = (
    ROOT
    / "medium_models/sh_file/sst5/bs32/h-loss/analysis/converged_eval_loss_vs_h_points.csv"
)
OLD_FULL_SOURCE = (
    ROOT
    / "medium_models/sh_file/sst5/bs32/h_precision_sweep_16/analysis/mezo_fp16_roberta_sst5_oldfull_loss_mse.csv"
)

POINTS_CSV = OUT_DIR / "two_figures_h_vs_loss_points.csv"
WIDE_CSV = OUT_DIR / "two_figures_h_vs_loss_wide.csv"
PLOT_PNG = OUT_DIR / "two_figures_h_vs_loss_overlay.png"
PLOT_SVG = OUT_DIR / "two_figures_h_vs_loss_overlay.svg"


def safe_float(value: object) -> float:
    try:
        result = float(value)
    except Exception:
        return float("nan")
    return result if math.isfinite(result) else float("nan")


def load_hloss_sweep() -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    with LEFT_SOURCE.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            h = safe_float(row.get("eps"))
            loss = safe_float(row.get("converged_eval_loss"))
            if not (math.isfinite(h) and math.isfinite(loss)):
                continue
            # Match the filtered plot in h-loss/anl_hloss.py.
            if math.isclose(h, 1e-9, rel_tol=1e-6, abs_tol=1e-12):
                continue
            if math.isclose(h, 3e-9, rel_tol=1e-6, abs_tol=1e-12):
                continue
            rows.append(
                {
                    "source": "fp32",
                    "h": h,
                    "loss": loss,
                    "loss_column": "converged_eval_loss",
                    "source_file": str(LEFT_SOURCE),
                }
            )
    return sorted(rows, key=lambda item: float(item["h"]))


def load_old_full_sweep() -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    with OLD_FULL_SOURCE.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            h = safe_float(row.get("h"))
            loss = safe_float(row.get("eval_loss"))
            status = (row.get("status") or "").strip()
            if status != "completed":
                continue
            if not (math.isfinite(h) and math.isfinite(loss)):
                continue
            rows.append(
                {
                    "source": "fp16",
                    "h": h,
                    "loss": loss,
                    "loss_column": "eval_loss",
                    "source_file": str(OLD_FULL_SOURCE),
                }
            )
    return sorted(rows, key=lambda item: float(item["h"]))


def write_points(rows: List[Dict[str, object]]) -> None:
    with POINTS_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["source", "h", "loss", "loss_column", "source_file"]
        )
        writer.writeheader()
        writer.writerows(rows)


def write_wide(left_rows: List[Dict[str, object]], old_rows: List[Dict[str, object]]) -> None:
    by_h: Dict[float, Dict[str, object]] = {}
    for row in left_rows:
        h = float(row["h"])
        by_h.setdefault(h, {"h": h})["h_loss_converged_eval_loss"] = row["loss"]
    for row in old_rows:
        h = float(row["h"])
        by_h.setdefault(h, {"h": h})["old_full_dataset_eval_loss"] = row["loss"]

    with WIDE_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "h",
                "h_loss_converged_eval_loss",
                "old_full_dataset_eval_loss",
            ],
        )
        writer.writeheader()
        for h in sorted(by_h):
            writer.writerow(by_h[h])


def plot(left_rows: List[Dict[str, object]], old_rows: List[Dict[str, object]]) -> None:
    fig, ax = plt.subplots(figsize=(8.8, 5.2))

    for rows, label, color, marker in [
        (left_rows, "fp32", "#1f77b4", "o"),
        (old_rows, "fp16", "#d62728", "s"),
    ]:
        xs = [float(row["h"]) for row in rows]
        ys = [float(row["loss"]) for row in rows]
        ax.plot(xs, ys, marker=marker, linewidth=2.0, markersize=5.0, label=label, color=color)

    ax.set_xscale("log")
    ax.set_xlabel("h")
    ax.set_ylabel("loss")
    ax.set_title("SST-5 RoBERTa-large: h vs loss")
    ax.set_ylim(1.0, 1.72)
    ax.grid(True, which="both", alpha=0.28)
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(PLOT_PNG, dpi=240)
    fig.savefig(PLOT_SVG)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    left_rows = load_hloss_sweep()
    old_rows = load_old_full_sweep()
    write_points([*left_rows, *old_rows])
    write_wide(left_rows, old_rows)
    plot(left_rows, old_rows)
    print(f"wrote {POINTS_CSV}")
    print(f"wrote {WIDE_CSV}")
    print(f"wrote {PLOT_PNG}")
    print(f"wrote {PLOT_SVG}")


if __name__ == "__main__":
    main()
