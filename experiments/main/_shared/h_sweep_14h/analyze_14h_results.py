#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[4]
ANALYSIS_ROOT = REPO_ROOT / "experiments" / "main" / "_shared" / "h_sweep_14h" / "analysis"

SWEEPS = {
    "quzo8": {
        "color": "#d55e00",
        "marker": "s",
        "label": "quzo8",
    },
    "quzo16": {
        "color": "#0072b2",
        "marker": "o",
        "label": "quzo16",
    },
}

COMBOS = [
    ("roberta-large", "sst5", "RoBERTa-large / SST-5"),
    ("roberta-large", "mnli", "RoBERTa-large / MNLI"),
    ("opt-1.3b", "sst5", "OPT-1.3B / SST-5"),
    ("opt-1.3b", "mnli", "OPT-1.3B / MNLI"),
]


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


def existing_files(path: Path, pattern: str) -> list[Path]:
    return sorted(path.rglob(pattern))


def find_first(path: Path, pattern: str) -> Path | None:
    matches = existing_files(path, pattern)
    return matches[0] if matches else None


def existing_path(candidates: list[Path]) -> Path | None:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def combo_root_candidates(sweep_key: str, model: str, task: str) -> list[Path]:
    if sweep_key == "quzo16":
        return [
            REPO_ROOT / "experiments" / "main" / "mezo" / model / task / "fp16" / "h_sweep_14h" / "results",
            REPO_ROOT / "experiments" / "h_sweep_14h" / "results" / "quzo16" / model / task,
        ]

    return [
        REPO_ROOT / "experiments" / "h_sweep_14h" / "results" / "quzo8" / model / task,
        REPO_ROOT / "experiments" / "smoke" / "mezo" / model / task / "int8" / "h_sweep_14h" / "results",
        REPO_ROOT / "experiments" / "smoke" / "mezo" / model / task / "int8" / "h_sweep_14h" / "smoke" / task / "results",
    ]


def describe_root_preferences(sweep_key: str) -> str:
    return ", ".join(path.as_posix() for path in combo_root_candidates(sweep_key, "<model>", "<task>"))


def parse_medium_metrics(path: Path) -> float | None:
    eval_loss = None
    with path.open() as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if (row.get("eval_ran") or "").strip() != "YES":
                continue
            value = safe_float(row.get("eval_loss"))
            if value is not None:
                eval_loss = value
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
                value = safe_float(row.get("value"))
                if value is not None:
                    eval_loss = value
    return eval_loss


def parse_metrics(path: Path | None) -> float | None:
    if path is None or not path.exists():
        return None
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


def parse_probe_mse(path: Path | None) -> float | None:
    if path is None or not path.exists():
        return None
    with path.open() as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return None
    return safe_float(rows[-1].get("mse"))


def parse_summary_statuses(summary_path: Path | None) -> dict[str, str]:
    if summary_path is None or not summary_path.exists():
        return {}
    statuses: dict[str, str] = {}
    for line in summary_path.read_text().splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        h_value = str(item.get("h", "")).strip()
        if h_value:
            statuses[h_value] = item.get("status", "<missing>")
    return statuses


def load_records(sweep_key: str, model: str, task: str) -> list[dict[str, object]]:
    combo_root = existing_path(combo_root_candidates(sweep_key, model, task))
    if combo_root is None or not combo_root.exists():
        return []

    statuses = parse_summary_statuses(find_first(combo_root, "summary.jsonl"))
    records: list[dict[str, object]] = []
    h_texts = set(statuses.keys())
    h_texts.update(h_dir.name[2:] for h_dir in combo_root.glob("h_*"))

    for h_text in sorted(h_texts, key=float):
        h_dir = combo_root / f"h_{h_text}"
        metrics_csv = find_first(h_dir, "metrics*.csv")
        probe_csv = find_first(h_dir, "zo_directional_probe.csv")
        run_summary = find_first(h_dir, "run_summary.json")

        eval_loss = parse_metrics(metrics_csv)
        mse = parse_probe_mse(probe_csv)

        if eval_loss is None and mse is None and run_summary is None and h_text not in statuses:
            continue

        records.append(
            {
                "sweep": sweep_key,
                "model": model,
                "task": task,
                "h": float(h_text),
                "h_str": h_text,
                "status": statuses.get(h_text, "unknown"),
                "eval_loss": eval_loss,
                "mse": mse,
                "metrics_csv": metrics_csv.as_posix() if metrics_csv else "",
                "probe_csv": probe_csv.as_posix() if probe_csv else "",
                "run_summary": run_summary.as_posix() if run_summary else "",
            }
        )

    return records


def summarize_records(records: list[dict[str, object]]) -> str:
    status_counts = Counter(str(record["status"]) for record in records)
    valid_loss = sum(
        1
        for record in records
        if record["eval_loss"] is not None and not math.isnan(float(record["eval_loss"]))
    )
    valid_mse = sum(
        1 for record in records if record["mse"] is not None and not math.isnan(float(record["mse"]))
    )
    status_text = ", ".join(f"{k}={v}" for k, v in sorted(status_counts.items()))
    return f"points={len(records)}, valid_loss={valid_loss}, valid_mse={valid_mse}, {status_text}"


def style_plot() -> None:
    available_styles = set(plt.style.available)
    if "seaborn-v0_8-whitegrid" in available_styles:
        plt.style.use("seaborn-v0_8-whitegrid")
    elif "seaborn-whitegrid" in available_styles:
        plt.style.use("seaborn-whitegrid")
    else:
        plt.style.use("default")


def write_summary_csv(records: list[dict[str, object]], output_csv: Path) -> None:
    fieldnames = [
        "sweep",
        "model",
        "task",
        "h_str",
        "h",
        "status",
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


def plot(records_by_combo: dict[tuple[str, str], dict[str, list[dict[str, object]]]], output_png: Path) -> None:
    style_plot()
    fig, axes = plt.subplots(len(COMBOS), 2, figsize=(14, 16), constrained_layout=True)

    legend_handles = []
    legend_labels = []

    for row_idx, (model, task, title) in enumerate(COMBOS):
        loss_ax = axes[row_idx, 0]
        mse_ax = axes[row_idx, 1]
        loss_ax.set_title(title, fontsize=12)

        combo_data = records_by_combo.get((model, task), {})
        any_loss = False
        any_mse = False

        for sweep_key in ["quzo8", "quzo16"]:
            records = combo_data.get(sweep_key, [])
            if not records:
                continue

            cfg = SWEEPS[sweep_key]
            xs = [float(record["h"]) for record in records]
            ys_loss = [
                float(record["eval_loss"])
                for record in records
                if record["eval_loss"] is not None and not math.isnan(float(record["eval_loss"]))
            ]
            xs_loss = [
                float(record["h"])
                for record in records
                if record["eval_loss"] is not None and not math.isnan(float(record["eval_loss"]))
            ]
            ys_mse = [
                float(record["mse"])
                for record in records
                if record["mse"] is not None and not math.isnan(float(record["mse"]))
            ]
            xs_mse = [
                float(record["h"])
                for record in records
                if record["mse"] is not None and not math.isnan(float(record["mse"]))
            ]

            if xs_loss:
                any_loss = True
                loss_ax.plot(
                    xs_loss,
                    ys_loss,
                    marker=cfg["marker"],
                    linewidth=2,
                    color=cfg["color"],
                    label=cfg["label"],
                )
            if xs_mse:
                any_mse = True
                mse_ax.plot(
                    xs_mse,
                    ys_mse,
                    marker=cfg["marker"],
                    linewidth=2,
                    color=cfg["color"],
                    label=cfg["label"],
                )

            handles, labels = loss_ax.get_legend_handles_labels()
            for handle, label in zip(handles, labels):
                if label not in legend_labels:
                    legend_handles.append(handle)
                    legend_labels.append(label)

        for ax in [loss_ax, mse_ax]:
            ax.set_xscale("log")
            ax.set_xlabel("h")

        loss_ax.set_ylabel("eval_loss")
        mse_ax.set_ylabel("probe mse")
        mse_ax.set_yscale("log")

        if not any_loss:
            loss_ax.text(0.5, 0.5, "No valid loss points", ha="center", va="center", transform=loss_ax.transAxes)
        if not any_mse:
            mse_ax.text(0.5, 0.5, "No valid MSE points", ha="center", va="center", transform=mse_ax.transAxes)

    fig.suptitle("14h sweep overview: current quzo8/quzo16 results", fontsize=16)
    if legend_handles:
        fig.legend(legend_handles, legend_labels, loc="upper center", ncol=len(legend_labels), frameon=False)
    fig.savefig(output_png, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_report(records_by_combo: dict[tuple[str, str], dict[str, list[dict[str, object]]]], output_report: Path) -> None:
    lines = [
        "14h sweep analysis report",
        "",
        f"quzo8 roots: {describe_root_preferences('quzo8')}",
        f"quzo16 roots: {describe_root_preferences('quzo16')}",
        "",
    ]

    for model, task, title in COMBOS:
        lines.append(title)
        combo_data = records_by_combo.get((model, task), {})
        for sweep_key in ["quzo8", "quzo16"]:
            records = combo_data.get(sweep_key, [])
            if not records:
                lines.append(f"- {sweep_key}: no local records")
                continue
            lines.append(f"- {sweep_key}: {summarize_records(records)}")

            valid_loss_records = [
                record
                for record in records
                if record["eval_loss"] is not None and not math.isnan(float(record["eval_loss"]))
            ]
            if valid_loss_records:
                best = min(valid_loss_records, key=lambda item: float(item["eval_loss"]))
                lines.append(
                    f"  best loss at h={best['h_str']}: eval_loss={float(best['eval_loss']):.6f}, status={best['status']}"
                )

            valid_mse_records = [
                record
                for record in records
                if record["mse"] is not None and not math.isnan(float(record["mse"]))
            ]
            if valid_mse_records:
                best = min(valid_mse_records, key=lambda item: float(item["mse"]))
                lines.append(
                    f"  best mse at h={best['h_str']}: mse={float(best['mse']):.6f}, status={best['status']}"
                )
        lines.append("")

    output_report.write_text("\n".join(lines))


def main() -> None:
    ANALYSIS_ROOT.mkdir(parents=True, exist_ok=True)

    records_by_combo: dict[tuple[str, str], dict[str, list[dict[str, object]]]] = {}
    all_records: list[dict[str, object]] = []

    for model, task, _ in COMBOS:
        combo_key = (model, task)
        records_by_combo[combo_key] = {}
        for sweep_key in ["quzo8", "quzo16"]:
            records = load_records(sweep_key, model, task)
            records_by_combo[combo_key][sweep_key] = records
            all_records.extend(records)

    output_png = ANALYSIS_ROOT / "h_sweep_14h_overview_loss_mse.png"
    output_csv = ANALYSIS_ROOT / "h_sweep_14h_overview_summary.csv"
    output_report = ANALYSIS_ROOT / "h_sweep_14h_overview_report.txt"

    plot(records_by_combo, output_png)
    write_summary_csv(all_records, output_csv)
    write_report(records_by_combo, output_report)

    print(f"plot_png={output_png}")
    print(f"summary_csv={output_csv}")
    print(f"report_txt={output_report}")


if __name__ == "__main__":
    main()
