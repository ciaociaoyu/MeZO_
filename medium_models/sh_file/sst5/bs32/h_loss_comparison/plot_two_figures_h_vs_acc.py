#!/usr/bin/env python3

import csv
import json
import math
import re
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path("/Users/jichaoyu/Documents/GitHub/MeZO")
OUT_DIR = ROOT / "medium_models/sh_file/sst5/bs32/h_loss_comparison"

FP32_ROOT = ROOT / "medium_models/sh_file/sst5/bs32/h-loss"
FP16_SUMMARY = (
    ROOT
    / "medium_models/sh_file/sst5/bs32/h_precision_sweep_16/workspace/result/sst-5-bs32-full-fp16-h-sweep-seed16/summary.jsonl"
)

POINTS_CSV = OUT_DIR / "two_figures_h_vs_acc_points.csv"
WIDE_CSV = OUT_DIR / "two_figures_h_vs_acc_wide.csv"
PLOT_PNG = OUT_DIR / "two_figures_h_vs_acc_overlay.png"
PLOT_SVG = OUT_DIR / "two_figures_h_vs_acc_overlay.svg"


def safe_float(value: object) -> float:
    try:
        result = float(value)
    except Exception:
        return float("nan")
    return result if math.isfinite(result) else float("nan")


def parse_h_from_eps_dir(path: Path) -> Optional[float]:
    match = re.search(r"eps([0-9]+(?:\.[0-9]+)?(?:e[+-]?[0-9]+)?)", path.name)
    if not match:
        return None
    h = safe_float(match.group(1))
    return h if math.isfinite(h) else None


def excluded_h(h: float) -> bool:
    return math.isclose(h, 1e-9, rel_tol=1e-6, abs_tol=1e-12) or math.isclose(
        h, 3e-9, rel_tol=1e-6, abs_tol=1e-12
    )


def read_eval_accs(metrics_csv: Path) -> List[float]:
    accs: List[float] = []
    with metrics_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            acc = safe_float(row.get("eval_acc"))
            if math.isfinite(acc):
                accs.append(acc)
    return accs


def load_fp32_rows() -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for eps_dir in sorted(FP32_ROOT.glob("sst5_bs32_hloss_eps*")):
        h = parse_h_from_eps_dir(eps_dir)
        if h is None or excluded_h(h):
            continue

        metrics_paths = sorted(eps_dir.glob("seed*/metrics_logs/*.csv"))
        if not metrics_paths:
            continue

        # Pick the longest metrics file if multiple are present.
        metrics_csv = max(metrics_paths, key=lambda p: sum(1 for _ in p.open("r", encoding="utf-8", errors="ignore")))
        accs = read_eval_accs(metrics_csv)
        if not accs:
            continue

        last5 = accs[-5:]
        acc = sum(last5) / len(last5)
        rows.append(
            {
                "source": "fp32",
                "h": h,
                "acc": acc,
                "acc_column": "eval_acc_last5_mean",
                "source_file": str(metrics_csv),
            }
        )
    return sorted(rows, key=lambda item: float(item["h"]))


def first_metrics(block: object) -> Dict[str, object]:
    if not isinstance(block, dict) or not block:
        return {}
    first = next(iter(block.values()))
    return first if isinstance(first, dict) else {}


def load_fp16_rows() -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    with FP16_SUMMARY.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            h = safe_float(record.get("h"))
            metrics = first_metrics(record.get("eval"))
            acc = safe_float(metrics.get("eval_acc"))
            loss = safe_float(metrics.get("eval_loss"))
            if not (math.isfinite(h) and math.isfinite(acc) and math.isfinite(loss)):
                continue
            rows.append(
                {
                    "source": "fp16",
                    "h": h,
                    "acc": acc,
                    "acc_column": "eval_acc",
                    "source_file": str(FP16_SUMMARY),
                }
            )
    return sorted(rows, key=lambda item: float(item["h"]))


def write_points(rows: List[Dict[str, object]]) -> None:
    with POINTS_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["source", "h", "acc", "acc_column", "source_file"]
        )
        writer.writeheader()
        writer.writerows(rows)


def write_wide(fp32_rows: List[Dict[str, object]], fp16_rows: List[Dict[str, object]]) -> None:
    by_h: Dict[float, Dict[str, object]] = {}
    for row in fp32_rows:
        h = float(row["h"])
        by_h.setdefault(h, {"h": h})["fp32_eval_acc_last5_mean"] = row["acc"]
    for row in fp16_rows:
        h = float(row["h"])
        by_h.setdefault(h, {"h": h})["fp16_eval_acc"] = row["acc"]

    with WIDE_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["h", "fp32_eval_acc_last5_mean", "fp16_eval_acc"]
        )
        writer.writeheader()
        for h in sorted(by_h):
            writer.writerow(by_h[h])


def plot(fp32_rows: List[Dict[str, object]], fp16_rows: List[Dict[str, object]]) -> None:
    fig, ax = plt.subplots(figsize=(8.8, 5.2))

    for rows, label, color, marker in [
        (fp32_rows, "fp32", "#1f77b4", "o"),
        (fp16_rows, "fp16", "#d62728", "s"),
    ]:
        xs = [float(row["h"]) for row in rows]
        ys = [float(row["acc"]) for row in rows]
        ax.plot(xs, ys, marker=marker, linewidth=2.0, markersize=5.0, label=label, color=color)

    ax.set_xscale("log")
    ax.set_xlabel("h")
    ax.set_ylabel("accuracy")
    ax.set_title("SST-5 RoBERTa-large: h vs accuracy")
    ax.set_ylim(0.1, 0.56)
    ax.grid(True, which="both", alpha=0.28)
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(PLOT_PNG, dpi=240)
    fig.savefig(PLOT_SVG)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fp32_rows = load_fp32_rows()
    fp16_rows = load_fp16_rows()
    write_points([*fp32_rows, *fp16_rows])
    write_wide(fp32_rows, fp16_rows)
    plot(fp32_rows, fp16_rows)
    print(f"wrote {POINTS_CSV}")
    print(f"wrote {WIDE_CSV}")
    print(f"wrote {PLOT_PNG}")
    print(f"wrote {PLOT_SVG}")


if __name__ == "__main__":
    main()
