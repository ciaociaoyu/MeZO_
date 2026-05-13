#!/usr/bin/env python3
"""Summarize probe-window JSONL diagnostics and draw matplotlib plots."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Dict, Iterable, List, Optional, Tuple


SUMMARY_FIELDS = [
    "precision_mode",
    "direction_type",
    "sparse_rate",
    "sparse_mode",
    "sparse_rescale",
    "h_raw",
    "h_active",
    "num_probe_rows",
    "num_probe_directions",
    "num_probe_batches",
    "probe_active_frac_mean",
    "probe_alignment_mean",
    "probe_norm_ratio_mean",
    "delta_q_norm_mean",
    "nominal_delta_norm_mean",
    "fd_zero_ratio",
    "fd_mean",
    "fd_std",
    "d_true_mean",
    "d_true_std",
    "corr_fd_true",
    "nMSE_fd_true",
    "sign_agreement",
    "window_candidate",
]


def finite_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, bool):
        return float(value)
    text = str(value).strip()
    if text == "" or text.lower() in {"none", "null", "nan"}:
        return None
    try:
        out = float(text)
    except Exception:
        return None
    return out if math.isfinite(out) else None


def collect(rows: Iterable[Dict[str, Any]], key: str) -> List[float]:
    vals = []
    for row in rows:
        val = finite_float(row.get(key))
        if val is not None:
            vals.append(val)
    return vals


def avg(rows: Iterable[Dict[str, Any]], key: str) -> Optional[float]:
    vals = collect(rows, key)
    return mean(vals) if vals else None


def std(vals: List[float]) -> Optional[float]:
    if not vals:
        return None
    return pstdev(vals) if len(vals) > 1 else 0.0


def corr(xs: List[float], ys: List[float]) -> Optional[float]:
    pairs = [(x, y) for x, y in zip(xs, ys) if math.isfinite(x) and math.isfinite(y)]
    if len(pairs) < 2:
        return None
    x_mean = mean(x for x, _ in pairs)
    y_mean = mean(y for _, y in pairs)
    x_var = sum((x - x_mean) ** 2 for x, _ in pairs)
    y_var = sum((y - y_mean) ** 2 for _, y in pairs)
    if x_var <= 0.0 or y_var <= 0.0:
        return None
    cov = sum((x - x_mean) * (y - y_mean) for x, y in pairs)
    return cov / math.sqrt(x_var * y_var)


def read_rows(root: Path) -> List[Dict[str, Any]]:
    paths = [root] if root.is_file() else sorted(root.glob("**/probe_stats.jsonl"))
    rows: List[Dict[str, Any]] = []
    for path in paths:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                row["_source"] = str(path)
                rows.append(row)
    return rows


def group_key(row: Dict[str, Any]) -> Tuple[Any, ...]:
    return (
        row.get("precision_mode", row.get("zo_two_point_precision", "")),
        row.get("direction_type", "sparse" if finite_float(row.get("p", row.get("sparse_rate", 1.0))) not in (None, 1.0) else "dense"),
        finite_float(row.get("sparse_rate", row.get("p", 1.0))) or 1.0,
        row.get("sparse_mode", "none"),
        row.get("sparse_rescale", "none"),
        finite_float(row.get("h_raw", row.get("eps"))),
    )


def summarize_group(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    first = rows[0]
    fd = collect(rows, "d_fd") or collect(rows, "fd_mean")
    true = collect(rows, "d_true") or collect(rows, "td_mean")
    fd_zero_vals = []
    for row in rows:
        if "fd_is_zero" in row:
            fd_zero_vals.append(1.0 if bool(row.get("fd_is_zero")) else 0.0)
        else:
            val = finite_float(row.get("fd_zero_ratio"))
            if val is not None:
                fd_zero_vals.append(val)
    corr_fd_true = corr(fd, true) if fd and true and len(fd) == len(true) else avg(rows, "corr")
    nmse = None
    if fd and true and len(fd) == len(true):
        mse = mean((x - y) ** 2 for x, y in zip(fd, true))
        denom = mean(y * y for y in true)
        nmse = mse / denom if denom > 0.0 else None
    else:
        nmse = avg(rows, "nmse")
    sign_vals = []
    for row in rows:
        if "sign_match" in row and row.get("sign_match") is not None:
            sign_vals.append(1.0 if bool(row.get("sign_match")) else 0.0)
        else:
            val = finite_float(row.get("sign_acc"))
            if val is not None:
                sign_vals.append(val)

    active = avg(rows, "probe_active_frac")
    align = avg(rows, "probe_alignment")
    ratio = avg(rows, "probe_norm_ratio")
    fd_zero = mean(fd_zero_vals) if fd_zero_vals else None
    candidate = (
        active is not None and active > 0.01
        and align is not None and align > 0.5
        and ratio is not None and 0.3 <= ratio <= 3.0
        and fd_zero is not None and fd_zero < 0.5
    )
    p = finite_float(first.get("sparse_rate", first.get("p", 1.0))) or 1.0
    h_raw = finite_float(first.get("h_raw", first.get("eps"))) or 0.0
    h_active = finite_float(first.get("h_active"))
    if h_active is None:
        h_active = h_raw / math.sqrt(p) if str(first.get("sparse_rescale", "none")) == "inv_sqrt_p" and p > 0.0 else h_raw
    return {
        "precision_mode": first.get("precision_mode", first.get("zo_two_point_precision", "")),
        "direction_type": first.get("direction_type", "sparse" if p < 1.0 else "dense"),
        "sparse_rate": p,
        "sparse_mode": first.get("sparse_mode", "none"),
        "sparse_rescale": first.get("sparse_rescale", "none"),
        "h_raw": h_raw,
        "h_active": h_active,
        "num_probe_rows": len(rows),
        "num_probe_directions": len({row.get("seed", row.get("direction_index", i)) for i, row in enumerate(rows)}),
        "num_probe_batches": len({row.get("batch_index", 0) for row in rows}),
        "probe_active_frac_mean": active,
        "probe_alignment_mean": align,
        "probe_norm_ratio_mean": ratio,
        "delta_q_norm_mean": avg(rows, "delta_q_norm"),
        "nominal_delta_norm_mean": avg(rows, "nominal_delta_norm"),
        "fd_zero_ratio": fd_zero,
        "fd_mean": mean(fd) if fd else None,
        "fd_std": std(fd),
        "d_true_mean": mean(true) if true else None,
        "d_true_std": std(true),
        "corr_fd_true": corr_fd_true,
        "nMSE_fd_true": nmse,
        "sign_agreement": mean(sign_vals) if sign_vals else None,
        "window_candidate": candidate,
    }


def write_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in SUMMARY_FIELDS})


def fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def estimate_windows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    grouped: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["precision_mode"], row["direction_type"], row["sparse_rate"], row["sparse_rescale"])].append(row)
    for key, group in grouped.items():
        group = sorted(group, key=lambda r: float(r["h_active"] if r["direction_type"] == "sparse" else r["h_raw"]))
        candidates = [r for r in group if bool(r.get("window_candidate"))]
        corr_rows = [r for r in group if finite_float(r.get("corr_fd_true")) is not None]
        align_rows = [r for r in group if finite_float(r.get("probe_alignment_mean")) is not None]
        x_name = "h_active" if group[0]["direction_type"] == "sparse" else "h_raw"
        lower = min((float(r[x_name]) for r in candidates), default=None)
        upper = max((float(r[x_name]) for r in candidates), default=None)
        out.append({
            "precision_mode": key[0],
            "direction_type": key[1],
            "sparse_rate": key[2],
            "sparse_rescale": key[3],
            "best_h_by_corr": max(corr_rows, key=lambda r: float(r["corr_fd_true"]))[x_name] if corr_rows else None,
            "best_h_by_alignment": max(align_rows, key=lambda r: float(r["probe_alignment_mean"]))[x_name] if align_rows else None,
            "lower_visible_h": lower,
            "upper_stable_h": upper,
            "window_width_log10": math.log10(upper / lower) if lower and upper and upper > lower else None,
        })
    return out


def write_markdown(rows: List[Dict[str, Any]], windows: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = [
        "precision_mode", "direction_type", "sparse_rate", "h_raw", "h_active",
        "probe_active_frac_mean", "probe_alignment_mean", "probe_norm_ratio_mean",
        "fd_zero_ratio", "corr_fd_true", "nMSE_fd_true", "sign_agreement", "window_candidate",
    ]
    with path.open("w", encoding="utf-8") as f:
        f.write("# Probe Window Summary\n\n")
        f.write("Window candidate heuristic: active_frac > 0.01, alignment > 0.5, norm_ratio in [0.3, 3.0], fd_zero_ratio < 0.5. If true-gradient rows exist, use corr/nMSE as diagnostics rather than hard filters.\n\n")
        f.write("| " + " | ".join(cols) + " |\n")
        f.write("| " + " | ".join(["---"] * len(cols)) + " |\n")
        for row in rows:
            f.write("| " + " | ".join(fmt(row.get(c)) for c in cols) + " |\n")
        f.write("\n## Estimated Windows\n\n")
        wcols = ["precision_mode", "direction_type", "sparse_rate", "sparse_rescale", "best_h_by_corr", "best_h_by_alignment", "lower_visible_h", "upper_stable_h", "window_width_log10"]
        f.write("| " + " | ".join(wcols) + " |\n")
        f.write("| " + " | ".join(["---"] * len(wcols)) + " |\n")
        for row in windows:
            f.write("| " + " | ".join(fmt(row.get(c)) for c in wcols) + " |\n")


def maybe_plot(rows: List[Dict[str, Any]], out_dir: Path) -> List[Path]:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return []
    out_dir.mkdir(parents=True, exist_ok=True)
    plots: List[Path] = []

    def plot_grouped(x_key: str, y_key: str, group_key_name: str, filename: str, title: str) -> None:
        fig, ax = plt.subplots(figsize=(6.5, 4.2))
        drew = False
        groups: Dict[Any, List[Dict[str, Any]]] = defaultdict(list)
        for row in rows:
            x = finite_float(row.get(x_key))
            y = finite_float(row.get(y_key))
            if x is not None and y is not None:
                groups[row.get(group_key_name)].append(row)
        for label, group in sorted(groups.items(), key=lambda kv: str(kv[0])):
            pts = sorted((float(r[x_key]), float(r[y_key])) for r in group if finite_float(r.get(x_key)) is not None and finite_float(r.get(y_key)) is not None)
            if not pts:
                continue
            ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", label=str(label))
            drew = True
        if drew:
            ax.set_xscale("log")
            ax.set_xlabel(x_key)
            ax.set_ylabel(y_key)
            ax.set_title(title)
            ax.grid(True, which="both", alpha=0.3)
            ax.legend()
            fig.tight_layout()
            path = out_dir / filename
            fig.savefig(path, dpi=160)
            plots.append(path)
        plt.close(fig)

    dense = [r for r in rows if r.get("direction_type") == "dense"]
    sparse = [r for r in rows if r.get("direction_type") == "sparse"]
    old_rows = rows[:]
    if dense:
        rows[:] = dense
        for y in ["probe_active_frac_mean", "probe_alignment_mean", "probe_norm_ratio_mean", "corr_fd_true", "nMSE_fd_true", "fd_zero_ratio"]:
            plot_grouped("h_raw", y, "precision_mode", f"dense_{y}_vs_h.png", f"{y} vs h")
    if sparse:
        rows[:] = sparse
        for y in ["probe_active_frac_mean", "probe_alignment_mean", "probe_norm_ratio_mean", "corr_fd_true", "nMSE_fd_true", "fd_zero_ratio"]:
            plot_grouped("h_raw", y, "sparse_rate", f"sparse_{y}_vs_h_raw.png", f"{y} vs raw h")
            plot_grouped("h_active", y, "sparse_rate", f"sparse_{y}_vs_h_active.png", f"{y} vs h_active")
    rows[:] = old_rows
    return plots


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    raw_rows = read_rows(args.run_root)
    groups: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = defaultdict(list)
    for row in raw_rows:
        groups[group_key(row)].append(row)
    summary = [summarize_group(group) for _, group in sorted(groups.items(), key=lambda kv: tuple(str(x) for x in kv[0]))]
    windows = estimate_windows(summary)
    out_dir = args.output_dir or args.run_root
    write_csv(summary, out_dir / "summary.csv")
    write_markdown(summary, windows, out_dir / "summary.md")
    plots = maybe_plot(summary, out_dir / "plots")
    print(f"rows={len(summary)}")
    print(f"summary_csv={out_dir / 'summary.csv'}")
    print(f"summary_md={out_dir / 'summary.md'}")
    print(f"plots_dir={out_dir / 'plots'}")
    for plot in plots:
        print(f"plot={plot}")


if __name__ == "__main__":
    main()
