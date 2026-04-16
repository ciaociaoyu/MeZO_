#!/usr/bin/env python3
from __future__ import annotations

import csv
import math
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


EXPECTED_H = [
    1e-8,
    3e-8,
    1e-7,
    3e-7,
    1e-6,
    3e-6,
    1e-5,
    3e-5,
    1e-4,
    3e-4,
    1e-3,
    3e-3,
    1e-2,
    3e-2,
]

COMBO_ORDER = [
    ("roberta-large", "sst5"),
    ("roberta-large", "mnli"),
    ("opt-1.3b", "sst5"),
    ("opt-1.3b", "mnli"),
]

VARIANT_ORDER = ["quzo8", "quzo16"]
VARIANT_COLORS = {
    "quzo8": "#d55e00",
    "quzo16": "#0072b2",
}

PATH_RE = re.compile(
    r"results/(?P<variant>quzo\d+)/(?P<model>[^/]+)/(?P<task>[^/]+)/h_(?P<h>[^/]+)/"
)


def safe_float(value: str | None) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    if math.isnan(number):
        return math.nan
    return number


def format_h(value: float) -> str:
    return f"{value:.0e}".replace("e-0", "e-").replace("e+0", "e+")


def title_case_task(task: str) -> str:
    return {
        "sst5": "SST-5",
        "mnli": "MNLI",
    }.get(task, task)


def parse_context(path: Path) -> dict[str, str] | None:
    match = PATH_RE.search(path.as_posix())
    if not match:
        return None
    return match.groupdict()


def parse_opt_metrics(path: Path) -> tuple[float | None, float | None, int | None]:
    eval_loss = None
    eval_acc = None
    last_eval_step = None

    with path.open() as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            phase = (row.get("phase") or "").strip()
            split = (row.get("split") or "").strip()
            metric = (row.get("metric") or "").strip()
            value = safe_float(row.get("value"))
            step = row.get("step")

            if phase != "eval" and split != "eval":
                continue

            if metric == "eval_loss":
                eval_loss = value
                last_eval_step = int(step) if step else last_eval_step
            elif metric in {"accuracy", "eval_acc"}:
                eval_acc = value
                last_eval_step = int(step) if step else last_eval_step

    return eval_loss, eval_acc, last_eval_step


def parse_roberta_metrics(path: Path) -> tuple[float | None, float | None, int | None]:
    eval_loss = None
    eval_acc = None
    last_eval_step = None

    with path.open() as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if (row.get("eval_ran") or "").strip() != "YES":
                continue

            loss = safe_float(row.get("eval_loss"))
            acc = safe_float(row.get("eval_acc"))
            step_text = (row.get("global_step") or "").strip()

            if loss is not None:
                eval_loss = loss
            if acc is not None:
                eval_acc = acc
            if step_text:
                last_eval_step = int(float(step_text))

    return eval_loss, eval_acc, last_eval_step


def extract_record(path: Path) -> dict[str, object] | None:
    ctx = parse_context(path)
    if ctx is None:
        return None

    if "metrics_logs" in path.parts:
        eval_loss, eval_acc, step = parse_roberta_metrics(path)
    else:
        eval_loss, eval_acc, step = parse_opt_metrics(path)

    return {
        "variant": ctx["variant"],
        "model": ctx["model"],
        "task": ctx["task"],
        "h_str": ctx["h"],
        "h": float(ctx["h"]),
        "eval_loss": eval_loss,
        "eval_acc": eval_acc,
        "last_eval_step": step,
        "source_csv": path.as_posix(),
    }


def load_records(results_dir: Path) -> list[dict[str, object]]:
    records = []
    for variant_dir in sorted(results_dir.iterdir()):
        if not variant_dir.is_dir() or not variant_dir.name.startswith("quzo"):
            continue
        for csv_path in sorted(variant_dir.rglob("metrics*.csv")):
            record = extract_record(csv_path)
            if record is not None:
                records.append(record)
    return records


def sort_key(record: dict[str, object]) -> tuple:
    variant = str(record["variant"])
    model = str(record["model"])
    task = str(record["task"])
    h = float(record["h"])
    variant_idx = VARIANT_ORDER.index(variant) if variant in VARIANT_ORDER else 99
    combo_idx = COMBO_ORDER.index((model, task)) if (model, task) in COMBO_ORDER else 99
    return combo_idx, variant_idx, h


def write_summary(records: list[dict[str, object]], output_csv: Path) -> None:
    fieldnames = [
        "variant",
        "model",
        "task",
        "h_str",
        "h",
        "eval_loss",
        "eval_acc",
        "last_eval_step",
        "source_csv",
    ]
    with output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in sorted(records, key=sort_key):
            writer.writerow(record)


def build_groups(records: list[dict[str, object]]) -> dict[tuple[str, str], dict[str, list[dict[str, object]]]]:
    grouped: dict[tuple[str, str], dict[str, list[dict[str, object]]]] = defaultdict(lambda: defaultdict(list))
    for record in records:
        grouped[(str(record["model"]), str(record["task"]))][str(record["variant"])].append(record)
    for variant_map in grouped.values():
        for variant_records in variant_map.values():
            variant_records.sort(key=lambda item: float(item["h"]))
    return grouped


def annotate_best(ax, records: list[dict[str, object]], metric_key: str, maximize: bool) -> None:
    valid = []
    for record in records:
        value = record.get(metric_key)
        if value is None or (isinstance(value, float) and math.isnan(value)):
            continue
        valid.append(record)

    if not valid:
        return

    best = max(valid, key=lambda item: float(item[metric_key])) if maximize else min(
        valid, key=lambda item: float(item[metric_key])
    )
    x = float(best["h"])
    y = float(best[metric_key])
    ax.scatter([x], [y], color="black", marker="*", s=90, zorder=5)
    ax.annotate(
        f"best {format_h(x)}",
        xy=(x, y),
        xytext=(6, 6),
        textcoords="offset points",
        fontsize=8,
        color="black",
    )


def plot_grid(records: list[dict[str, object]], output_png: Path) -> list[str]:
    grouped = build_groups(records)

    fig, axes = plt.subplots(2, 4, figsize=(22, 9), sharex=False)
    best_lines: list[str] = []
    legend_handles = []
    legend_labels = []

    for col, combo in enumerate(COMBO_ORDER):
        model, task = combo
        variant_map = grouped.get(combo, {})
        loss_ax = axes[0, col]
        acc_ax = axes[1, col]

        loss_ax.set_title(f"{model} / {title_case_task(task)}", fontsize=12, pad=10)
        loss_ax.set_xscale("log")
        acc_ax.set_xscale("log")

        all_records_for_combo: list[dict[str, object]] = []
        for variant in VARIANT_ORDER + sorted(set(variant_map) - set(VARIANT_ORDER)):
            variant_records = variant_map.get(variant)
            if not variant_records:
                continue
            all_records_for_combo.extend(variant_records)

            xs = [float(item["h"]) for item in variant_records]
            loss_ys = [item["eval_loss"] for item in variant_records]
            acc_ys = [item["eval_acc"] for item in variant_records]
            color = VARIANT_COLORS.get(variant)

            loss_line = loss_ax.plot(xs, loss_ys, marker="o", linewidth=2, label=variant, color=color)[0]
            acc_ax.plot(xs, acc_ys, marker="o", linewidth=2, label=variant, color=color)

            if variant not in legend_labels:
                legend_handles.append(loss_line)
                legend_labels.append(variant)

        annotate_best(loss_ax, all_records_for_combo, "eval_loss", maximize=False)
        annotate_best(acc_ax, all_records_for_combo, "eval_acc", maximize=True)

        loss_ax.grid(True, alpha=0.3, linestyle="--")
        acc_ax.grid(True, alpha=0.3, linestyle="--")
        loss_ax.set_xticks(EXPECTED_H)
        acc_ax.set_xticks(EXPECTED_H)
        loss_ax.set_xticklabels([format_h(h) for h in EXPECTED_H], rotation=45, ha="right", fontsize=8)
        acc_ax.set_xticklabels([format_h(h) for h in EXPECTED_H], rotation=45, ha="right", fontsize=8)

        if col == 0:
            loss_ax.set_ylabel("eval_loss")
            acc_ax.set_ylabel("eval_acc")
        acc_ax.set_xlabel("h")

        valid_loss_h = {
            float(r["h"])
            for r in all_records_for_combo
            if r.get("eval_loss") is not None
            and not (isinstance(r.get("eval_loss"), float) and math.isnan(r["eval_loss"]))
        }
        valid_acc_h = {
            float(r["h"])
            for r in all_records_for_combo
            if r.get("eval_acc") is not None
            and not (isinstance(r.get("eval_acc"), float) and math.isnan(r["eval_acc"]))
        }
        best_loss = _best_summary(all_records_for_combo, "eval_loss", maximize=False)
        best_acc = _best_summary(all_records_for_combo, "eval_acc", maximize=True)
        best_lines.append(
            f"{model}/{task}: valid_loss={len(valid_loss_h)}/14, valid_acc={len(valid_acc_h)}/14, "
            f"best_loss={best_loss}, best_acc={best_acc}"
        )

    fig.suptitle("14h sweep: final eval_loss / eval_acc vs h", fontsize=16, y=0.98)
    if legend_handles:
        fig.legend(legend_handles, legend_labels, loc="upper center", ncol=len(legend_labels), frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(output_png, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return best_lines


def _best_summary(records: list[dict[str, object]], metric_key: str, maximize: bool) -> str:
    valid = []
    for record in records:
        value = record.get(metric_key)
        if value is None or (isinstance(value, float) and math.isnan(value)):
            continue
        valid.append(record)
    if not valid:
        return "n/a"
    best = max(valid, key=lambda item: float(item[metric_key])) if maximize else min(
        valid, key=lambda item: float(item[metric_key])
    )
    return f"{format_h(float(best['h']))} ({float(best[metric_key]):.4f})"


def write_text_report(lines: list[str], output_txt: Path) -> None:
    output_txt.write_text("\n".join(lines) + "\n")


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    results_dir = base_dir / "results"
    output_dir = base_dir / "analysis"
    output_dir.mkdir(exist_ok=True)

    records = load_records(results_dir)
    summary_csv = output_dir / "hsweep14h_summary.csv"
    plot_png = output_dir / "hsweep14h_loss_acc.png"
    report_txt = output_dir / "hsweep14h_report.txt"

    write_summary(records, summary_csv)
    report_lines = plot_grid(records, plot_png)
    write_text_report(report_lines, report_txt)

    print(f"records={len(records)}")
    print(f"summary_csv={summary_csv}")
    print(f"plot_png={plot_png}")
    print(f"report_txt={report_txt}")
    print("")
    for line in report_lines:
        print(line)


if __name__ == "__main__":
    main()
