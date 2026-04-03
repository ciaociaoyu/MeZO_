#!/usr/bin/env python3

import csv
import json
import math
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt


ROOT = Path("/Users/jichaoyu/Documents/GitHub/MeZO/medium_models")
TASKS = [
    {
        "task": "SST-2",
        "summary": ROOT / "sh_file/sst-2/full_fp16_h_sweep_16_workspace/result/SST-2-bs32-full-fp16-h-sweep-seed16/summary.jsonl",
    },
    {
        "task": "sst-5",
        "summary": ROOT / "sh_file/sst5/bs32/h_precision_sweep_16/workspace/result/sst-5-bs32-full-fp16-h-sweep-seed16/summary.jsonl",
    },
    {
        "task": "MNLI",
        "summary": ROOT / "sh_file/MNLI_bs8/full_fp16_h_sweep_16_workspace/result/MNLI-bs32-full-fp16-h-sweep-seed16/summary.jsonl",
    },
    {
        "task": "RTE",
        "summary": ROOT / "sh_file/RTE_bs8/full_fp16_h_sweep_16_workspace/result/RTE-bs32-full-fp16-h-sweep-seed16/summary.jsonl",
    },
]
OUT_DIR = ROOT / "sh_file/h_probe_error_figures"
H_MARKS = {
    "SST-2": {"h_two_point": 1.52e-4, "h_additive": 4.76e-4},
    "sst-5": {"h_two_point": 2.322e-4, "h_additive": 5.859e-4},
    "MNLI": {"h_two_point": 2.614e-4, "h_additive": 5.94e-4},
    "RTE": {"h_two_point": 2.104e-4, "h_additive": 5.304e-4},
}


def _safe_float(value) -> float:
    try:
        result = float(value)
    except Exception:
        return float("nan")
    return result if math.isfinite(result) else float("nan")


def load_rows(summary_path: Path, task_name: str) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    with summary_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            probe = (record.get("artifacts") or {}).get("zo_directional_probe_last_row") or {}
            rows.append(
                {
                    "task": task_name,
                    "h": _safe_float(record.get("h")),
                    "probe_mae": _safe_float(probe.get("mae")),
                    "probe_rmse": _safe_float(probe.get("rmse")),
                    "probe_sign_acc": _safe_float(probe.get("sign_acc")),
                    "probe_corr": _safe_float(probe.get("corr")),
                    "fd_mean": _safe_float(probe.get("fd_mean")),
                    "td_mean": _safe_float(probe.get("td_mean")),
                }
            )
    rows.sort(key=lambda row: row["h"])
    return rows


def write_csv(all_rows: List[Dict[str, float]], out_path: Path) -> None:
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "task",
                "h",
                "probe_mae",
                "probe_rmse",
                "probe_sign_acc",
                "probe_corr",
                "fd_mean",
                "td_mean",
            ],
        )
        writer.writeheader()
        for row in all_rows:
            writer.writerow(row)


def _plot_metric(ax, rows: List[Dict[str, float]], key: str, label: str, color: str, linestyle: str = "-") -> None:
    xs = []
    ys = []
    for row in rows:
        x = row["h"]
        y = row[key]
        if math.isfinite(x) and math.isfinite(y) and y > 0:
            xs.append(x)
            ys.append(y)
    if xs:
        ax.plot(xs, ys, marker="o", linewidth=2, markersize=5, label=label, color=color, linestyle=linestyle)


def _add_h_markers(ax, task: str) -> None:
    marks = H_MARKS.get(task)
    if not marks:
        return

    two_point = marks["h_two_point"]
    additive = marks["h_additive"]

    ax.axvline(two_point, color="#2ca02c", linestyle="--", linewidth=1.8, alpha=0.9, label="h_two_point")
    ax.axvline(additive, color="#9467bd", linestyle="-.", linewidth=1.8, alpha=0.9, label="h_additive")
    ax.text(
        two_point,
        0.98,
        f"two-point\n{two_point:.2e}",
        color="#2ca02c",
        rotation=90,
        transform=ax.get_xaxis_transform(),
        ha="right",
        va="top",
        fontsize=8,
        backgroundcolor="white",
    )
    ax.text(
        additive,
        0.82,
        f"additive\n{additive:.2e}",
        color="#9467bd",
        rotation=90,
        transform=ax.get_xaxis_transform(),
        ha="left",
        va="top",
        fontsize=8,
        backgroundcolor="white",
    )


def plot(all_task_rows: Dict[str, List[Dict[str, float]]], out_prefix: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    axes = axes.flatten()

    for ax, (task, rows) in zip(axes, all_task_rows.items()):
        _plot_metric(ax, rows, "probe_mae", "MAE", "#d62728")
        _plot_metric(ax, rows, "probe_rmse", "RMSE", "#1f77b4", linestyle="--")
        _add_h_markers(ax, task)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("h")
        ax.set_ylabel("error")
        ax.set_title(task)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()

    fig.suptitle("FP16 Full-Dataset h vs Directional-Derivative Error", fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_prefix.with_suffix(".png"), dpi=220)
    fig.savefig(out_prefix.with_suffix(".svg"))
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_rows: List[Dict[str, float]] = []
    by_task: Dict[str, List[Dict[str, float]]] = {}

    for item in TASKS:
        rows = load_rows(item["summary"], item["task"])
        by_task[item["task"]] = rows
        all_rows.extend(rows)

    write_csv(all_rows, OUT_DIR / "fp16_full_h_probe_error_summary.csv")
    plot(by_task, OUT_DIR / "fp16_full_h_vs_probe_error")
    print(f"[done] wrote figure and csv to {OUT_DIR}")


if __name__ == "__main__":
    main()
