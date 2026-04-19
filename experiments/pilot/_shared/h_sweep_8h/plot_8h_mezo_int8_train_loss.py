#!/usr/bin/env python3
from __future__ import annotations

import csv
import statistics
from pathlib import Path

import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[4]
RESULT_ROOT = REPO_ROOT / "experiments" / "pilot" / "mezo" / "roberta-large"
ANALYSIS_ROOT = REPO_ROOT / "experiments" / "pilot" / "_shared" / "h_sweep_8h" / "analysis"
H_ORDER = ["1e-6", "3e-6", "1e-5", "3e-5", "1e-4", "3e-4", "1e-3", "3e-3"]
TASKS = [("sst5", "SST-5"), ("mnli", "MNLI")]


def task_result_root(task: str) -> Path:
    return RESULT_ROOT / task / "int8" / "h_sweep_8h" / "results"


def rolling_mean(values: list[float], window: int = 25) -> list[float]:
    if not values:
        return []
    if window <= 1:
        return values[:]
    out: list[float] = []
    running_sum = 0.0
    for idx, value in enumerate(values):
        running_sum += value
        if idx >= window:
            running_sum -= values[idx - window]
        count = min(idx + 1, window)
        out.append(running_sum / count)
    return out


def load_series(task: str, h_text: str) -> dict[str, object] | None:
    metrics_path = (
        task_result_root(task)
        / f"h_{h_text}"
        / "run_mezo_int8"
        / "seed16"
        / "metrics_logs"
        / "metrics_adaptiveH-0_cscale-0.csv"
    )
    if not metrics_path.exists():
        return None

    dedup_rows: list[dict[str, str]] = []
    seen_steps: set[int] = set()
    with metrics_path.open() as handle:
        for row in csv.DictReader(handle):
            step = int(float(row["global_step"]))
            if step in seen_steps:
                continue
            seen_steps.add(step)
            dedup_rows.append(row)

    steps = [int(float(row["global_step"])) for row in dedup_rows]
    losses = [float(row["train_loss"]) for row in dedup_rows]
    smoothed = rolling_mean(losses, window=25)
    tail = losses[-50:] if len(losses) >= 50 else losses

    return {
        "task": task,
        "h": h_text,
        "steps": steps,
        "train_loss": losses,
        "train_loss_smooth": smoothed,
        "metrics_csv": metrics_path.as_posix(),
        "n_points": len(losses),
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "min_loss": min(losses),
        "max_loss": max(losses),
        "tail_mean": statistics.mean(tail),
        "tail_std": statistics.pstdev(tail) if len(tail) > 1 else 0.0,
    }


def write_summary_csv(series_by_task: dict[str, list[dict[str, object]]], output_csv: Path) -> None:
    fieldnames = [
        "task",
        "h",
        "n_points",
        "initial_loss",
        "final_loss",
        "min_loss",
        "max_loss",
        "tail_mean",
        "tail_std",
        "metrics_csv",
    ]
    with output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for task in [task for task, _ in TASKS]:
            for series in series_by_task.get(task, []):
                writer.writerow({key: series[key] for key in fieldnames})


def style_plot() -> None:
    available_styles = set(plt.style.available)
    if "seaborn-v0_8-whitegrid" in available_styles:
        plt.style.use("seaborn-v0_8-whitegrid")
    elif "seaborn-whitegrid" in available_styles:
        plt.style.use("seaborn-whitegrid")
    else:
        plt.style.use("default")


def plot_task(task: str, title: str, series_list: list[dict[str, object]], output_path: Path) -> None:
    style_plot()
    fig, axes = plt.subplots(4, 2, figsize=(13, 14), constrained_layout=True, sharex=True)
    axes_flat = list(axes.flat)

    for ax, h_text in zip(axes_flat, H_ORDER):
        series = next((item for item in series_list if item["h"] == h_text), None)
        if series is None:
            ax.set_axis_off()
            continue

        steps = series["steps"]
        losses = series["train_loss"]
        smoothed = series["train_loss_smooth"]

        ax.plot(steps, losses, color="#9ecae1", linewidth=1.0, alpha=0.5, label="raw")
        ax.plot(steps, smoothed, color="#08519c", linewidth=2.0, label="rolling mean (25 pts)")
        ax.set_title(f"h={h_text} | final={float(series['final_loss']):.3f}", fontsize=11)
        ax.set_xlabel("global_step")
        ax.set_ylabel("train_loss")
        ax.text(
            0.03,
            0.95,
            f"tail mean={float(series['tail_mean']):.3f}\ntail std={float(series['tail_std']):.3f}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8.5,
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#cccccc"},
        )

    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.suptitle(f"8h mezo_int8 train loss by h: roberta-large / {title}", fontsize=16)
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.985))
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_report(series_by_task: dict[str, list[dict[str, object]]], output_path: Path) -> None:
    lines = [
        "8h mezo_int8 train-loss stability report",
        "",
        f"result_root: {RESULT_ROOT} / <task> / int8 / h_sweep_8h / results",
        "",
        "Notes:",
        "- Each metrics CSV repeats every global_step 10 times. The plots deduplicate to one point per step before drawing.",
        "- The rolling mean uses a 25-point window, which corresponds to roughly 250 training steps.",
        "",
    ]
    for task, title in TASKS:
        lines.append(f"{title}")
        for series in series_by_task.get(task, []):
            lines.append(
                f"- h={series['h']}: final={float(series['final_loss']):.4f}, "
                f"min={float(series['min_loss']):.4f}, max={float(series['max_loss']):.4f}, "
                f"tail_mean={float(series['tail_mean']):.4f}, tail_std={float(series['tail_std']):.4f}"
            )
        lines.append("")
    output_path.write_text("\n".join(lines))


def main() -> None:
    ANALYSIS_ROOT.mkdir(parents=True, exist_ok=True)

    series_by_task = {
        task: [load_series(task, h) for h in H_ORDER]
        for task, _ in TASKS
    }
    series_by_task = {
        task: [series for series in series_list if series is not None]
        for task, series_list in series_by_task.items()
    }

    plot_task(
        "sst5",
        "SST-5",
        series_by_task["sst5"],
        ANALYSIS_ROOT / "mezo_int8_8h_roberta_sst5_train_loss_by_h.png",
    )
    plot_task(
        "mnli",
        "MNLI",
        series_by_task["mnli"],
        ANALYSIS_ROOT / "mezo_int8_8h_roberta_mnli_train_loss_by_h.png",
    )
    write_summary_csv(
        series_by_task,
        ANALYSIS_ROOT / "mezo_int8_8h_roberta_train_loss_by_h_summary.csv",
    )
    write_report(
        series_by_task,
        ANALYSIS_ROOT / "mezo_int8_8h_roberta_train_loss_by_h_report.txt",
    )

    print(
        "sst5_plot="
        + str(ANALYSIS_ROOT / "mezo_int8_8h_roberta_sst5_train_loss_by_h.png")
    )
    print(
        "mnli_plot="
        + str(ANALYSIS_ROOT / "mezo_int8_8h_roberta_mnli_train_loss_by_h.png")
    )
    print(
        "summary_csv="
        + str(ANALYSIS_ROOT / "mezo_int8_8h_roberta_train_loss_by_h_summary.csv")
    )
    print(
        "report_txt="
        + str(ANALYSIS_ROOT / "mezo_int8_8h_roberta_train_loss_by_h_report.txt")
    )


if __name__ == "__main__":
    main()
