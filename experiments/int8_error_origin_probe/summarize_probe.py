#!/usr/bin/env python3
"""Summarize INT8 finite-difference probe rows into the diagnostic table."""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional


SUMMARY_FIELDS = [
    "h",
    "rows",
    "step_min",
    "step_max",
    "mse",
    "g2",
    "nmse",
    "corr",
    "sign_acc",
    "fd_zero_ratio",
    "changed_ratio",
    "changed_numel",
    "fd_abs_median",
    "td_abs_median",
    "fd_mean",
    "td_mean",
    "mse_u2",
    "nmse_u2",
    "corr_u2",
    "sign_acc_u2",
    "diagnosis",
]


def to_float(value: object) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip()
    if text == "" or text.lower() in {"none", "null"}:
        return None
    try:
        out = float(text)
    except ValueError:
        return None
    if not math.isfinite(out):
        return None
    return out


def mean(values: Iterable[Optional[float]]) -> Optional[float]:
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not vals:
        return None
    return sum(vals) / len(vals)


def fmt(value: Optional[float]) -> str:
    if value is None:
        return ""
    if value == 0:
        return "0"
    abs_v = abs(value)
    if abs_v < 1e-3 or abs_v >= 1e4:
        return f"{value:.6e}"
    return f"{value:.6g}"


def diagnose(row: Dict[str, Optional[float]]) -> str:
    nmse = row.get("nmse")
    corr = row.get("corr")
    sign = row.get("sign_acc")
    fd_zero = row.get("fd_zero_ratio")
    changed = row.get("changed_ratio")

    if nmse is None:
        return "missing_nmse"
    abs_corr = abs(corr) if corr is not None else None
    sign_random = sign is not None and 0.4 <= sign <= 0.6

    if nmse < 0.1 and corr is not None and corr > 0.7:
        return "usable_signal"
    if nmse < 0.5 and corr is not None and corr > 0.3:
        return "partly_usable_signal"
    if 0.5 <= nmse <= 2.0:
        if (abs_corr is not None and abs_corr < 0.2) or sign_random:
            return "dead_zone_or_signal_collapse"
        if fd_zero is not None and fd_zero > 0.5:
            return "dead_zone_or_signal_collapse"
        if changed is not None and changed < 0.01:
            return "dead_zone_or_signal_collapse"
    if nmse > 5.0:
        if fd_zero is not None and fd_zero > 0.5:
            return "dead_zone_with_large_gradient_scale"
        return "extra_noise_or_impl_mismatch"
    return "mixed"


def summarize(rows: List[Dict[str, str]]) -> List[Dict[str, object]]:
    groups: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        h = to_float(row.get("eps"))
        if h is None:
            continue
        groups[f"{h:.17g}"].append(row)

    summary: List[Dict[str, object]] = []
    for key, items in groups.items():
        steps = [to_float(item.get("global_step")) for item in items]
        numeric = {
            "h": float(key),
            "mse": mean(to_float(item.get("mse")) for item in items),
            "g2": mean(to_float(item.get("g2")) for item in items),
            "nmse": mean(to_float(item.get("nmse")) for item in items),
            "corr": mean(to_float(item.get("corr")) for item in items),
            "sign_acc": mean(to_float(item.get("sign_acc")) for item in items),
            "fd_zero_ratio": mean(to_float(item.get("fd_zero_ratio")) for item in items),
            "changed_ratio": mean(to_float(item.get("param_changed_ratio_mean")) for item in items),
            "changed_numel": mean(to_float(item.get("perturb_changed_numel_mean")) for item in items),
            "fd_abs_median": mean(to_float(item.get("fd_abs_median")) for item in items),
            "td_abs_median": mean(to_float(item.get("td_abs_median")) for item in items),
            "fd_mean": mean(to_float(item.get("fd_mean")) for item in items),
            "td_mean": mean(to_float(item.get("td_mean")) for item in items),
            "mse_u2": mean(to_float(item.get("mse_u2_debug")) for item in items),
            "nmse_u2": mean(to_float(item.get("nmse_u2_debug")) for item in items),
            "corr_u2": mean(to_float(item.get("corr_u2_debug")) for item in items),
            "sign_acc_u2": mean(to_float(item.get("sign_acc_u2_debug")) for item in items),
        }
        out = {
            "h": numeric["h"],
            "rows": len(items),
            "step_min": min(v for v in steps if v is not None) if any(v is not None for v in steps) else "",
            "step_max": max(v for v in steps if v is not None) if any(v is not None for v in steps) else "",
            **numeric,
        }
        out["diagnosis"] = diagnose(numeric)
        summary.append(out)

    summary.sort(key=lambda row: float(row["h"]))
    return summary


def write_csv(path: Path, summary: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for row in summary:
            writer.writerow({field: row.get(field, "") for field in SUMMARY_FIELDS})


def write_markdown(path: Path, summary: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = ["h", "mse", "g2", "nmse", "corr", "sign_acc", "fd_zero_ratio", "changed_ratio", "diagnosis"]
    with path.open("w", encoding="utf-8") as handle:
        handle.write("# INT8 Error Origin Probe Summary\n\n")
        handle.write("| " + " | ".join(columns) + " |\n")
        handle.write("| " + " | ".join(["---"] * len(columns)) + " |\n")
        for row in summary:
            cells = []
            for col in columns:
                value = row.get(col)
                cells.append(str(value) if isinstance(value, str) else fmt(value if isinstance(value, float) else to_float(value)))
            handle.write("| " + " | ".join(cells) + " |\n")


def print_table(summary: List[Dict[str, object]]) -> None:
    columns = ["h", "mse", "g2", "nmse", "corr", "sign_acc", "fd_zero_ratio", "changed_ratio", "diagnosis"]
    print("\t".join(columns))
    for row in summary:
        cells = []
        for col in columns:
            value = row.get(col)
            cells.append(str(value) if isinstance(value, str) else fmt(value if isinstance(value, float) else to_float(value)))
        print("\t".join(cells))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(args.input)
    with args.input.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    summary = summarize(rows)
    output_dir = args.output_dir or (args.input.parent / "analysis")
    write_csv(output_dir / "summary.csv", summary)
    write_markdown(output_dir / "summary.md", summary)
    print_table(summary)


if __name__ == "__main__":
    main()
