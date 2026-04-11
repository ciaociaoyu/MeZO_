#!/usr/bin/env python3
import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
SUMMARY_PATH = SCRIPT_DIR / "workspace" / "result" / "sst-5-bs32-full-adam-fp16-h-sweep-seed16" / "summary.jsonl"
OUT_DIR = SCRIPT_DIR / "figures"
SUMMARY_CSV = OUT_DIR / "adam_fp16_h_acc_summary.csv"
PNG_PATH = OUT_DIR / "adam_fp16_h_vs_acc.png"
SVG_PATH = OUT_DIR / "adam_fp16_h_vs_acc.svg"


def _safe_float(value):
    try:
        out = float(value)
    except Exception:
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def _load_records():
    records_by_h = {}
    if not SUMMARY_PATH.exists():
        raise FileNotFoundError(f"Missing summary file: {SUMMARY_PATH}")

    for line in SUMMARY_PATH.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        h_value = _safe_float(record.get("h"))
        if not math.isfinite(h_value):
            continue

        metrics_last = ((record.get("artifacts") or {}).get("metrics_csv_last_row") or {})
        eval_metrics = ((record.get("eval") or {}).get("sst-5") or {})
        test_metrics = ((record.get("test") or {}).get("sst-5") or {})

        records_by_h[h_value] = {
            "h": h_value,
            "train_acc_last": _safe_float(metrics_last.get("train_acc")),
            "dev_acc": _safe_float(eval_metrics.get("eval_acc")),
            "test_acc": _safe_float(test_metrics.get("eval_acc")),
            "output_dir": record.get("output_dir", ""),
        }

    return [records_by_h[h] for h in sorted(records_by_h)]


def _write_summary_csv(records):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with SUMMARY_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["h", "train_acc_last", "dev_acc", "test_acc", "output_dir"],
        )
        writer.writeheader()
        writer.writerows(records)


def _plot(records):
    xs = np.array([row["h"] for row in records], dtype=float)
    train_acc = np.array([row["train_acc_last"] for row in records], dtype=float)
    dev_acc = np.array([row["dev_acc"] for row in records], dtype=float)
    test_acc = np.array([row["test_acc"] for row in records], dtype=float)

    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    for ys, label, color, marker in (
        (train_acc, "train_acc_last", "#1f77b4", "o"),
        (dev_acc, "dev_acc", "#d62728", "s"),
        (test_acc, "test_acc", "#2ca02c", "^"),
    ):
        mask = np.isfinite(ys)
        ax.plot(xs[mask], ys[mask], marker=marker, linewidth=2, markersize=5, label=label, color=color)

    ax.set_xscale("log")
    ax.set_xlabel("h (zero_order_eps)")
    ax.set_ylabel("accuracy")
    ax.set_title("SST-5 Adam + fp16: acc vs h")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(PNG_PATH, dpi=200)
    fig.savefig(SVG_PATH)
    plt.close(fig)


def main():
    records = _load_records()
    _write_summary_csv(records)
    _plot(records)
    print(f"wrote {SUMMARY_CSV}")
    print(f"wrote {PNG_PATH}")
    print(f"wrote {SVG_PATH}")


if __name__ == "__main__":
    main()
