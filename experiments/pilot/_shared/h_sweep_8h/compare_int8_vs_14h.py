#!/usr/bin/env python3
from __future__ import annotations

import csv
import math
import os
from pathlib import Path

import matplotlib.pyplot as plt


EXPECTED_H_14 = [
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

EXPECTED_H_8 = [
    1e-6,
    3e-6,
    1e-5,
    3e-5,
    1e-4,
    3e-4,
    1e-3,
    3e-3,
]

EXPECTED_H_ALL = EXPECTED_H_14

COMBO_ORDER = [
    ("roberta-large", "sst5"),
    ("roberta-large", "mnli"),
    ("opt-1.3b", "sst5"),
    ("opt-1.3b", "mnli"),
]

SWEEP_CONFIGS = {
    "8h_mezo_int8": {
        "label": "8h mezo_int8",
        "color": "#d55e00",
        "marker": "s",
        "expected_h": EXPECTED_H_8,
    },
    "14h_quzo8": {
        "label": "14h quzo8",
        "color": "#0072b2",
        "marker": "o",
        "expected_h": EXPECTED_H_14,
    },
}

REPO_ROOT = Path(__file__).resolve().parents[4]
ANALYSIS_ROOT = REPO_ROOT / "experiments" / "pilot" / "_shared" / "h_sweep_8h" / "analysis"


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


def existing_root(candidates: list[Path]) -> Path | None:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def resolve_roots(repo_root: Path) -> dict[str, Path | None]:
    env_8h = os.environ.get("HSWEEP8H_INT8_ROOT")
    env_14h = os.environ.get("HSWEEP14H_INT8_ROOT")

    roots_8h = []
    roots_14h = []
    if env_8h:
        roots_8h.append(Path(env_8h))
    if env_14h:
        roots_14h.append(Path(env_14h))

    roots_8h.extend(
        [
            repo_root / "experiments" / "pilot" / "mezo",
            repo_root / "experiments" / "h_sweep_8h" / "results" / "mezo_int8",
            Path("/scratch/jy03364/MeZO_/experiments/h_sweep_8h/results/mezo_int8"),
        ]
    )
    roots_14h.extend(
        [
            repo_root / "experiments" / "smoke" / "mezo",
            repo_root / "experiments" / "h_sweep_14h" / "results" / "quzo8",
            Path("/scratch/jy03364/MeZO_/experiments/h_sweep_14h/results/quzo8"),
        ]
    )

    return {
        "8h_mezo_int8": existing_root(roots_8h),
        "14h_quzo8": existing_root(roots_14h),
    }


def parse_medium_metrics(path: Path) -> float | None:
    eval_loss = None
    with path.open() as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if (row.get("eval_ran") or "").strip() != "YES":
                continue
            loss = safe_float(row.get("eval_loss"))
            if loss is not None:
                eval_loss = loss
    return eval_loss


def parse_large_metrics(path: Path) -> float | None:
    eval_loss = None
    with path.open() as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            phase = (row.get("phase") or "").strip()
            split = (row.get("split") or "").strip()
            metric = (row.get("metric") or "").strip()
            if phase != "eval" and split != "eval":
                continue
            if metric == "eval_loss":
                loss = safe_float(row.get("value"))
                if loss is not None:
                    eval_loss = loss
    return eval_loss


def parse_metrics(path: Path) -> float | None:
    with path.open() as handle:
        reader = csv.DictReader(handle)
        try:
            first_row = next(reader)
        except StopIteration:
            return None
    header_names = set(first_row.keys())
    if "eval_ran" in header_names:
        return parse_medium_metrics(path)
    return parse_large_metrics(path)


def parse_probe_mse(path: Path) -> float | None:
    with path.open() as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return None
    return safe_float(rows[-1].get("mse"))


def find_first(path: Path, pattern: str) -> Path | None:
    matches = sorted(path.rglob(pattern))
    return matches[0] if matches else None


def resolve_combo_root(root: Path | None, sweep_key: str, model: str, task: str) -> Path | None:
    if root is None:
        return None

    if sweep_key == "8h_mezo_int8":
        candidates = [
            root / model / task / "int8" / "h_sweep_8h" / "results",
            root / model / task,
        ]
    elif sweep_key == "14h_quzo8":
        candidates = [
            root / model / task,
            root / model / task / "int8" / "h_sweep_14h" / "results",
            root / model / task / "int8" / "h_sweep_14h" / "smoke" / task / "results",
        ]
    else:
        candidates = [root / model / task]

    return existing_root(candidates)


def load_combo_records(root: Path | None, sweep_key: str, model: str, task: str) -> list[dict[str, object]]:
    combo_root = resolve_combo_root(root, sweep_key, model, task)
    if combo_root is None or not combo_root.exists():
        return []

    records: list[dict[str, object]] = []
    for h_dir in sorted(combo_root.glob("h_*")):
        h_text = h_dir.name[len("h_") :]
        try:
            h_value = float(h_text)
        except ValueError:
            continue

        metrics_csv = find_first(h_dir, "metrics*.csv")
        probe_csv = find_first(h_dir, "zo_directional_probe.csv")
        run_summary = find_first(h_dir, "run_summary.json")

        eval_loss = parse_metrics(metrics_csv) if metrics_csv is not None else None
        mse = parse_probe_mse(probe_csv) if probe_csv is not None else None

        if eval_loss is None and mse is None and run_summary is None:
            continue

        records.append(
            {
                "sweep": sweep_key,
                "model": model,
                "task": task,
                "h": h_value,
                "h_str": h_text,
                "eval_loss": eval_loss,
                "mse": mse,
                "metrics_csv": metrics_csv.as_posix() if metrics_csv else "",
                "probe_csv": probe_csv.as_posix() if probe_csv else "",
                "run_summary": run_summary.as_posix() if run_summary else "",
            }
        )

    records.sort(key=lambda item: float(item["h"]))
    return records


def write_summary_csv(records: list[dict[str, object]], output_csv: Path) -> None:
    fieldnames = [
        "sweep",
        "model",
        "task",
        "h_str",
        "h",
        "eval_loss",
        "mse",
        "metrics_csv",
        "probe_csv",
        "run_summary",
    ]
    with output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(record)


def summarize_records(records: list[dict[str, object]]) -> str:
    total = len(records)
    valid_loss = sum(
        1
        for record in records
        if record["eval_loss"] is not None and not math.isnan(float(record["eval_loss"]))
    )
    valid_mse = sum(
        1 for record in records if record["mse"] is not None and not math.isnan(float(record["mse"]))
    )
    return f"points={total}, valid_loss={valid_loss}, valid_mse={valid_mse}"


def add_series(ax, records: list[dict[str, object]], metric_key: str, sweep_key: str) -> bool:
    xs = []
    ys = []
    for record in records:
        value = record[metric_key]
        if value is None:
            continue
        numeric = float(value)
        if math.isnan(numeric):
            continue
        xs.append(float(record["h"]))
        ys.append(numeric)

    if not xs:
        return False

    cfg = SWEEP_CONFIGS[sweep_key]
    ax.plot(
        xs,
        ys,
        marker=cfg["marker"],
        linewidth=2,
        markersize=6,
        color=cfg["color"],
        label=cfg["label"],
    )
    return True


def annotate_missing(ax, message: str) -> None:
    ax.text(
        0.5,
        0.5,
        message,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=10,
        color="#666666",
    )


def plot_grid(records_by_sweep: dict[str, dict[tuple[str, str], list[dict[str, object]]]], output_png: Path) -> list[str]:
    fig, axes = plt.subplots(2, 4, figsize=(22, 10), sharex=False)
    summary_lines: list[str] = []
    legend_handles = []
    legend_labels = []

    for col, combo in enumerate(COMBO_ORDER):
        model, task = combo
        loss_ax = axes[0, col]
        mse_ax = axes[1, col]
        loss_ax.set_title(f"{model} / {title_case_task(task)}", fontsize=12, pad=10)

        any_loss = False
        any_mse = False
        positive_mse = []

        for sweep_key in ["8h_mezo_int8", "14h_quzo8"]:
            records = records_by_sweep.get(sweep_key, {}).get(combo, [])
            any_loss |= add_series(loss_ax, records, "eval_loss", sweep_key)
            any_mse |= add_series(mse_ax, records, "mse", sweep_key)
            positive_mse.extend(
                float(record["mse"])
                for record in records
                if record["mse"] is not None and not math.isnan(float(record["mse"])) and float(record["mse"]) > 0
            )

            summary_lines.append(
                f"{SWEEP_CONFIGS[sweep_key]['label']} {model}/{task}: {summarize_records(records)}"
            )

        for ax in (loss_ax, mse_ax):
            ax.set_xscale("log")
            ax.set_xticks(EXPECTED_H_ALL)
            ax.set_xticklabels([format_h(h) for h in EXPECTED_H_ALL], rotation=45, ha="right", fontsize=8)
            ax.grid(True, alpha=0.3, linestyle="--")

        if positive_mse:
            mse_ax.set_yscale("log")

        if col == 0:
            loss_ax.set_ylabel("eval_loss")
            mse_ax.set_ylabel("probe mse")

        loss_ax.set_xlabel("h")
        mse_ax.set_xlabel("h")

        if not any_loss:
            annotate_missing(loss_ax, "No loss data found")
        if not any_mse:
            annotate_missing(mse_ax, "No mse data found")

        handles, labels = loss_ax.get_legend_handles_labels()
        for handle, label in zip(handles, labels):
            if label not in legend_labels:
                legend_handles.append(handle)
                legend_labels.append(label)

    fig.suptitle("INT8 sweep comparison: 8h pilot vs 14h sweep", fontsize=16, y=0.98)
    if legend_handles:
        fig.legend(legend_handles, legend_labels, loc="upper center", ncol=len(legend_labels), frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(output_png, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return summary_lines


def write_report(
    output_txt: Path,
    roots: dict[str, Path | None],
    records_by_sweep: dict[str, dict[tuple[str, str], list[dict[str, object]]]],
    plot_png: Path,
    summary_csv: Path,
    summary_lines: list[str],
) -> None:
    lines = []
    lines.append("INT8 sweep comparison report")
    lines.append("")
    for sweep_key in ["8h_mezo_int8", "14h_quzo8"]:
        root = roots.get(sweep_key)
        label = SWEEP_CONFIGS[sweep_key]["label"]
        lines.append(f"{label} root: {root if root is not None else 'NOT FOUND'}")
    lines.append("")
    lines.append(f"plot_png: {plot_png}")
    lines.append(f"summary_csv: {summary_csv}")
    lines.append("")
    lines.extend(summary_lines)
    lines.append("")
    lines.append("Notes:")
    lines.append("- Current script prefers the reorganized pilot/smoke layout and falls back to the legacy h_sweep directories when needed.")
    lines.append("- The comparison is still labeled as 8h `mezo_int8` vs 14h `quzo8` to preserve the original analysis naming.")
    lines.append("- Loss is read from the final available eval point in `metrics*.csv`.")
    lines.append("- MSE is read from the last row of `zo_directional_probe.csv`.")
    lines.append("- If an 8h result root is missing in the current workspace, the plot will only show the 14h side.")
    output_txt.write_text("\n".join(lines) + "\n")


def main() -> None:
    repo_root = REPO_ROOT
    roots = resolve_roots(repo_root)

    ANALYSIS_ROOT.mkdir(exist_ok=True)
    output_png = ANALYSIS_ROOT / "int8_8h_vs_14h_loss_mse.png"
    summary_csv = ANALYSIS_ROOT / "int8_8h_vs_14h_summary.csv"
    report_txt = ANALYSIS_ROOT / "int8_8h_vs_14h_report.txt"

    all_records: list[dict[str, object]] = []
    records_by_sweep: dict[str, dict[tuple[str, str], list[dict[str, object]]]] = {}

    for sweep_key in ["8h_mezo_int8", "14h_quzo8"]:
        sweep_records: dict[tuple[str, str], list[dict[str, object]]] = {}
        root = roots.get(sweep_key)
        for combo in COMBO_ORDER:
            model, task = combo
            records = load_combo_records(root, sweep_key, model, task)
            sweep_records[combo] = records
            all_records.extend(records)
        records_by_sweep[sweep_key] = sweep_records

    write_summary_csv(all_records, summary_csv)
    summary_lines = plot_grid(records_by_sweep, output_png)
    write_report(report_txt, roots, records_by_sweep, output_png, summary_csv, summary_lines)

    print(f"plot_png={output_png}")
    print(f"summary_csv={summary_csv}")
    print(f"report_txt={report_txt}")
    for line in summary_lines:
        print(line)


if __name__ == "__main__":
    main()
