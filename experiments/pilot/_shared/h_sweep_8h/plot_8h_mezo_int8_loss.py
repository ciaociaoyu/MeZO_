#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[4]
RESULT_ROOT = REPO_ROOT / "experiments" / "pilot" / "mezo" / "roberta-large"
ANALYSIS_ROOT = REPO_ROOT / "experiments" / "pilot" / "_shared" / "h_sweep_8h" / "analysis"
TASKS = [("sst5", "SST-5"), ("mnli", "MNLI")]


def task_result_root(task: str) -> Path:
    return RESULT_ROOT / task / "int8" / "h_sweep_8h" / "results"


def safe_task_block(summary: dict, field: str) -> dict:
    block = summary.get(field, {})
    return next(iter(block.values())) if block else {}


def count_probe_rows(path: Path) -> int:
    if not path.exists():
        return -1
    with path.open() as handle:
        return sum(1 for _ in csv.DictReader(handle))


def load_task_rows(task: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    task_root = task_result_root(task)
    for h_dir in sorted(task_root.glob("h_*"), key=lambda p: float(p.name[2:])):
        summary_path = h_dir / "run_mezo_int8" / "seed16" / "run_summary.json"
        probe_path = h_dir / "run_mezo_int8" / "seed16" / "zo_directional_probe.csv"
        if not summary_path.exists():
            continue

        summary = json.loads(summary_path.read_text())
        eval_block = safe_task_block(summary, "eval")
        test_block = safe_task_block(summary, "test")
        tail = summary.get("artifacts", {}).get("tail_perf_metrics", {}) or {}
        last5 = summary.get("artifacts", {}).get("eval_loss_last5", {}) or {}

        rows.append(
            {
                "task": task,
                "h": float(h_dir.name[2:]),
                "h_str": h_dir.name[2:],
                "dev_loss": eval_block.get("eval_loss"),
                "test_loss": test_block.get("eval_loss"),
                "sec_per_step": tail.get("tail_perf_wallclock_per_step"),
                "samples_per_sec": tail.get("tail_perf_samples_per_second"),
                "last5_mean": last5.get("eval_loss_last5_mean"),
                "probe_rows": count_probe_rows(probe_path),
                "summary_json": summary_path.as_posix(),
                "probe_csv": probe_path.as_posix(),
            }
        )
    return rows


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    fieldnames = [
        "task",
        "h_str",
        "h",
        "dev_loss",
        "test_loss",
        "sec_per_step",
        "samples_per_sec",
        "last5_mean",
        "probe_rows",
        "summary_json",
        "probe_csv",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def best_row(rows: list[dict[str, object]], key: str) -> dict[str, object] | None:
    numeric_rows = [row for row in rows if isinstance(row.get(key), (int, float))]
    if not numeric_rows:
        return None
    return min(numeric_rows, key=lambda row: float(row[key]))


def write_report(rows_by_task: dict[str, list[dict[str, object]]], path: Path) -> None:
    lines = [
        "8h mezo_int8 roberta-large loss report",
        "",
        f"result_root: {RESULT_ROOT} / <task> / int8 / h_sweep_8h / results",
        "",
        "Notes:",
        "- This report only covers local 8h results under experiments/pilot/mezo/roberta-large/<task>/int8/h_sweep_8h/results.",
        "- All current zo_directional_probe.csv files are header-only, so MSE is unavailable in this workspace.",
        "",
    ]
    for task, title in TASKS:
        rows = rows_by_task.get(task, [])
        lines.append(f"{title}: {len(rows)} points")
        best_dev = best_row(rows, "dev_loss")
        best_test = best_row(rows, "test_loss")
        if best_dev:
            lines.append(
                f"- best dev loss: h={best_dev['h_str']} loss={float(best_dev['dev_loss']):.6f}"
            )
        if best_test:
            lines.append(
                f"- best test loss: h={best_test['h_str']} loss={float(best_test['test_loss']):.6f}"
            )
        lines.append("")
    path.write_text("\n".join(lines))


def plot(rows_by_task: dict[str, list[dict[str, object]]], output_path: Path) -> None:
    available_styles = set(plt.style.available)
    if "seaborn-v0_8-whitegrid" in available_styles:
        plt.style.use("seaborn-v0_8-whitegrid")
    elif "seaborn-whitegrid" in available_styles:
        plt.style.use("seaborn-whitegrid")
    else:
        plt.style.use("default")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), constrained_layout=True)

    colors = {
        "dev_loss": "#d55e00",
        "test_loss": "#0072b2",
    }

    for ax, (task, title) in zip(axes, TASKS):
        rows = rows_by_task.get(task, [])
        xs = [float(row["h"]) for row in rows]
        dev_losses = [float(row["dev_loss"]) for row in rows]
        test_losses = [float(row["test_loss"]) for row in rows]

        ax.plot(xs, dev_losses, marker="o", linewidth=2, color=colors["dev_loss"], label="dev loss")
        ax.plot(
            xs,
            test_losses,
            marker="s",
            linewidth=2,
            linestyle="--",
            color=colors["test_loss"],
            label="test loss",
        )
        ax.set_xscale("log")
        ax.set_title(title)
        ax.set_xlabel("h")
        ax.set_ylabel("loss")
        ax.legend(frameon=False)

        best_dev = best_row(rows, "dev_loss")
        if best_dev:
            ax.scatter(
                [float(best_dev["h"])],
                [float(best_dev["dev_loss"])],
                s=80,
                color=colors["dev_loss"],
                edgecolor="black",
                linewidth=0.8,
                zorder=5,
            )
            ax.annotate(
                f"best dev\nh={best_dev['h_str']}",
                (float(best_dev["h"]), float(best_dev["dev_loss"])),
                xytext=(8, 8),
                textcoords="offset points",
                fontsize=9,
            )

        ax.text(
            0.03,
            0.97,
            "MSE unavailable:\nprobe CSVs are header-only",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "#cccccc"},
        )

    fig.suptitle("8h mezo_int8 sweep: roberta-large loss curves", fontsize=15)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ANALYSIS_ROOT.mkdir(parents=True, exist_ok=True)

    rows_by_task = {task: load_task_rows(task) for task, _ in TASKS}
    all_rows = [row for rows in rows_by_task.values() for row in rows]

    output_png = ANALYSIS_ROOT / "mezo_int8_8h_roberta_loss.png"
    output_csv = ANALYSIS_ROOT / "mezo_int8_8h_roberta_loss.csv"
    output_report = ANALYSIS_ROOT / "mezo_int8_8h_roberta_loss_report.txt"

    plot(rows_by_task, output_png)
    write_csv(all_rows, output_csv)
    write_report(rows_by_task, output_report)

    print(f"plot_png={output_png}")
    print(f"summary_csv={output_csv}")
    print(f"report_txt={output_report}")


if __name__ == "__main__":
    main()
