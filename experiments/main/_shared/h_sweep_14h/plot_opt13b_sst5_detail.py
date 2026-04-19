#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt


H_VALUES = [
    "1e-8",
    "3e-8",
    "1e-7",
    "3e-7",
    "1e-6",
    "3e-6",
    "1e-5",
    "3e-5",
    "1e-4",
    "3e-4",
    "1e-3",
    "3e-3",
    "1e-2",
    "3e-2",
]


def to_float(value):
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    number = float(text)
    if math.isnan(number):
        return None
    return number


def last_probe_row(path: Path):
    if not path.exists():
        return None
    with path.open() as handle:
        rows = list(csv.DictReader(handle))
    return rows[-1] if rows else None


def load_rows(base_dir: Path):
    rows = []
    for h_str in H_VALUES:
        run_dir = base_dir / f"h_{h_str}" / "seed_42"
        summary_path = run_dir / "run_summary.json"
        probe_path = run_dir / "zo_directional_probe.csv"

        summary = json.loads(summary_path.read_text())
        metrics_csv = Path(summary["paths"]["metrics_csv"])
        if not metrics_csv.exists():
            metrics_csv = run_dir / Path(summary["paths"]["metrics_csv"]).name

        eval_loss = None
        eval_acc = None
        with metrics_csv.open() as handle:
            for row in csv.DictReader(handle):
                if row.get("phase") != "eval" or row.get("split") != "eval":
                    continue
                metric = row.get("metric")
                value = to_float(row.get("value"))
                if metric == "eval_loss":
                    eval_loss = value
                elif metric in {"accuracy", "eval_acc"}:
                    eval_acc = value

        probe_row = last_probe_row(probe_path)
        mse = to_float(probe_row["mse"]) if probe_row else None
        rows.append(
            {
                "h_str": h_str,
                "h": float(h_str),
                "eval_loss": eval_loss,
                "eval_acc": eval_acc,
                "probe_mse": mse,
            }
        )
    return rows


def best_point(rows, key, maximize=False):
    valid = [row for row in rows if row[key] is not None]
    if not valid:
        return None
    return max(valid, key=lambda row: row[key]) if maximize else min(valid, key=lambda row: row[key])


def annotate_best(ax, point, label, ylog=False):
    if point is None:
        return
    x = point["h"]
    y = point[label]
    ax.scatter([x], [y], color="black", marker="*", s=100, zorder=5)
    ax.annotate(
        f"best {point['h_str']}",
        xy=(x, y),
        xytext=(8, 8),
        textcoords="offset points",
        fontsize=9,
        color="black",
    )


def write_csv(rows, output_csv: Path):
    with output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["h_str", "h", "eval_loss", "eval_acc", "probe_mse"])
        writer.writeheader()
        writer.writerows(rows)


def plot(rows, output_png: Path):
    fig, axes = plt.subplots(3, 1, figsize=(11, 12), sharex=True)
    x = [row["h"] for row in rows]

    series = [
        ("eval_loss", "Final eval_loss", False, False),
        ("eval_acc", "Final eval_acc", True, False),
        ("probe_mse", "Directional probe mse", False, True),
    ]

    for ax, (key, ylabel, maximize, ylog) in zip(axes, series):
        y = [row[key] for row in rows]
        ax.plot(x, y, marker="o", linewidth=2.5, color="#0072b2")
        ax.set_xscale("log")
        if ylog:
            ax.set_yscale("log")
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle="--", alpha=0.35)
        point = best_point(rows, key, maximize=maximize)
        annotate_best(ax, point, key, ylog=ylog)

    axes[0].set_title("opt-1.3b / SST5: h vs eval_loss, eval_acc, probe mse", fontsize=15, pad=12)
    axes[-1].set_xlabel("h")
    axes[-1].set_xticks([float(h) for h in H_VALUES])
    axes[-1].set_xticklabels(H_VALUES, rotation=45, ha="right")

    fig.tight_layout()
    fig.savefig(output_png, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_report(rows, output_txt: Path):
    best_loss = best_point(rows, "eval_loss", maximize=False)
    best_acc = best_point(rows, "eval_acc", maximize=True)
    best_mse = best_point(rows, "probe_mse", maximize=False)

    lines = [
        "opt-1.3b / SST5",
        f"best eval_loss: {best_loss['h_str']} ({best_loss['eval_loss']:.6f})" if best_loss else "best eval_loss: n/a",
        f"best eval_acc: {best_acc['h_str']} ({best_acc['eval_acc']:.6f})" if best_acc else "best eval_acc: n/a",
        f"best probe_mse: {best_mse['h_str']} ({best_mse['probe_mse']:.6f})" if best_mse else "best probe_mse: n/a",
        "note: probe_mse comes from zo_directional_probe.csv, not the evaluation loss.",
    ]
    output_txt.write_text("\n".join(lines) + "\n")


def main():
    root = Path(__file__).resolve().parent
    base_dir = root / "results" / "quzo16" / "opt-1.3b" / "sst5"
    out_dir = root / "analysis"
    out_dir.mkdir(exist_ok=True)

    rows = load_rows(base_dir)
    output_png = out_dir / "opt13b_sst5_loss_acc_mse.png"
    output_csv = out_dir / "opt13b_sst5_loss_acc_mse.csv"
    output_txt = out_dir / "opt13b_sst5_loss_acc_mse.txt"

    write_csv(rows, output_csv)
    plot(rows, output_png)
    write_report(rows, output_txt)

    print(f"plot_png={output_png}")
    print(f"summary_csv={output_csv}")
    print(f"report_txt={output_txt}")


if __name__ == "__main__":
    main()
