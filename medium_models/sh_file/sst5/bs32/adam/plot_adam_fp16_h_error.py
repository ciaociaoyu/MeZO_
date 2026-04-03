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
SUMMARY_CSV = OUT_DIR / "adam_fp16_h_error_summary.csv"
PNG_PATH = OUT_DIR / "adam_fp16_h_vs_probe_error.png"
SVG_PATH = OUT_DIR / "adam_fp16_h_vs_probe_error.svg"


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

        probe_last = ((record.get("artifacts") or {}).get("zo_directional_probe_last_row") or {})
        records_by_h[h_value] = {
            "h": h_value,
            "probe_mae": _safe_float(probe_last.get("mae")),
            "probe_rmse": _safe_float(probe_last.get("rmse")),
            "probe_corr": _safe_float(probe_last.get("corr")),
            "probe_sign_acc": _safe_float(probe_last.get("sign_acc")),
            "output_dir": record.get("output_dir", ""),
        }

    return [records_by_h[h] for h in sorted(records_by_h)]


def _write_summary_csv(records):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with SUMMARY_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["h", "probe_mae", "probe_rmse", "probe_corr", "probe_sign_acc", "output_dir"],
        )
        writer.writeheader()
        writer.writerows(records)


def _plot(records):
    xs = np.array([row["h"] for row in records], dtype=float)
    mae = np.array([row["probe_mae"] for row in records], dtype=float)
    rmse = np.array([row["probe_rmse"] for row in records], dtype=float)
    corr = np.array([row["probe_corr"] for row in records], dtype=float)
    sign_acc = np.array([row["probe_sign_acc"] for row in records], dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8))

    mask_mae = np.isfinite(mae) & (mae > 0)
    mask_rmse = np.isfinite(rmse) & (rmse > 0)
    axes[0].plot(xs[mask_mae], mae[mask_mae], marker="o", linewidth=2, label="MAE", color="#d62728")
    axes[0].plot(xs[mask_rmse], rmse[mask_rmse], marker="s", linewidth=2, label="RMSE", color="#1f77b4")
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("h (zero_order_eps)")
    axes[0].set_ylabel("error")
    axes[0].set_title("SST-5 Adam + fp16: probe error vs h")
    axes[0].grid(True, which="both", alpha=0.3)
    axes[0].legend()

    mask_corr = np.isfinite(corr)
    mask_sign = np.isfinite(sign_acc)
    axes[1].plot(xs[mask_corr], corr[mask_corr], marker="o", linewidth=2, label="corr", color="#9467bd")
    axes[1].plot(xs[mask_sign], sign_acc[mask_sign], marker="^", linewidth=2, label="sign_acc", color="#2ca02c")
    axes[1].set_xscale("log")
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].set_xlabel("h (zero_order_eps)")
    axes[1].set_ylabel("quality")
    axes[1].set_title("SST-5 Adam + fp16: probe quality vs h")
    axes[1].grid(True, which="both", alpha=0.3)
    axes[1].legend()

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
