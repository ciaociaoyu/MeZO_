#!/usr/bin/env python3

import csv
import math
from pathlib import Path
from typing import Dict, List


ROOT = Path("/Users/jichaoyu/Documents/GitHub/MeZO/medium_models")
PROBE_CSV = ROOT / "sh_file/h_probe_error_figures/fp16_full_h_probe_error_summary.csv"
LOSS_CSV = ROOT / "sh_file/h_loss_figures/fp16_full_h_loss_summary.csv"
OUT_DIR = ROOT / "sh_file/h_tables"
SELECTED_H = [1e-3, 1e-5, 3e-5]
TASK_ORDER = ["SST-2", "sst-5", "MNLI", "RTE"]


def _safe_float(value: str) -> float:
    try:
        result = float(value)
    except Exception:
        return float("nan")
    return result if math.isfinite(result) else float("nan")


def _load_csv(path: Path) -> Dict[str, Dict[float, Dict[str, float]]]:
    by_task: Dict[str, Dict[float, Dict[str, float]]] = {}
    with path.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            task = row["task"]
            clean = {"task": task}
            for key, value in row.items():
                if key == "task":
                    continue
                clean[key] = _safe_float(value)
            by_task.setdefault(task, {})[clean["h"]] = clean
    return by_task


def _format_float(value: float, digits: int = 6) -> str:
    if not math.isfinite(value):
        return "nan"
    return f"{value:.{digits}f}"


def _format_h(value: float) -> str:
    return f"{value:.0e}"


def build_rows() -> List[Dict[str, float]]:
    probe_by_task = _load_csv(PROBE_CSV)
    loss_by_task = _load_csv(LOSS_CSV)
    rows: List[Dict[str, float]] = []

    for task in TASK_ORDER:
        probe_rows = [row for row in probe_by_task[task].values() if math.isfinite(row["probe_mae"])]
        best_error_row = min(probe_rows, key=lambda row: row["probe_mae"])
        target_hs = [best_error_row["h"], *SELECTED_H]

        for idx, h in enumerate(target_hs):
            probe_row = probe_by_task[task].get(h)
            loss_row = loss_by_task[task].get(h)
            if probe_row is None or loss_row is None:
                continue

            rows.append(
                {
                    "task": task,
                    "setting": "best_error" if idx == 0 else f"h={_format_h(h)}",
                    "h": h,
                    "probe_mae": probe_row["probe_mae"],
                    "probe_rmse": probe_row["probe_rmse"],
                    "dev_loss": loss_row["dev_loss"],
                    "dev_acc": loss_row["dev_acc"],
                    "test_loss": loss_row["test_loss"],
                    "test_acc": loss_row["test_acc"],
                }
            )
    return rows


def write_csv(rows: List[Dict[str, float]], out_path: Path) -> None:
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "task",
                "setting",
                "h",
                "probe_mae",
                "probe_rmse",
                "dev_loss",
                "dev_acc",
                "test_loss",
                "test_acc",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_markdown(rows: List[Dict[str, float]], out_path: Path) -> None:
    lines = [
        "# FP16 Selected-h Table",
        "",
        "Best error is defined by minimum `probe_mae`. In the current results, the minimum-`probe_rmse` point is the same `h=3e-4` for all four tasks.",
        "",
        "| Task | Setting | h | MAE | RMSE | Dev Loss | Dev Acc | Test Loss | Test Acc |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]

    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["task"],
                    row["setting"],
                    _format_h(row["h"]),
                    _format_float(row["probe_mae"]),
                    _format_float(row["probe_rmse"]),
                    _format_float(row["dev_loss"]),
                    _format_float(row["dev_acc"]),
                    _format_float(row["test_loss"]),
                    _format_float(row["test_acc"]),
                ]
            )
            + " |"
        )

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = build_rows()
    write_csv(rows, OUT_DIR / "fp16_selected_h_table.csv")
    write_markdown(rows, OUT_DIR / "fp16_selected_h_table.md")
    print(f"[done] wrote table files to {OUT_DIR}")


if __name__ == "__main__":
    main()
