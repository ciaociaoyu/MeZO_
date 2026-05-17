#!/usr/bin/env python3
"""Summarize groupwise INT8 block-256 continuation artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.summarize_gptq256_rerun import read_jsonl, safe_float, summarize_run


PROBE_FIELDS = [
    "precision_mode",
    "quantization_algorithm",
    "group_size",
    "block_size",
    "direction_type",
    "sparse_rate",
    "sparse_mode",
    "sparse_rescale",
    "h_raw",
    "h_active",
    "num_probe_rows",
    "probe_active_frac_mean",
    "probe_alignment_mean",
    "probe_norm_ratio_mean",
    "fd_zero_ratio",
    "corr_fd_true",
    "nMSE_fd_true",
    "sign_agreement",
]

ALL_FIELDS = [
    "run_name",
    "model",
    "dataset",
    "quantization_algorithm",
    "group_size",
    "update_backend",
    "direction_type",
    "h",
    "h_active",
    "sparse_rate",
    "lr",
    "steps_completed",
    "seed",
    "best_eval_acc",
    "last_eval_acc",
    "best_eval_loss",
    "last_eval_loss",
    "corr_fd_true",
    "nMSE_fd_true",
    "probe_alignment",
    "probe_norm_ratio",
    "active_frac",
    "cos_intended_actual",
    "actual_over_intended_norm_ratio",
    "acc_actual_cos",
    "actual_over_acc_norm_ratio",
    "residual_bound_violation_frac",
    "grid_error_norm",
    "scale_drift_max",
    "nan_occurred",
    "status",
    "notes",
]


def collect(rows: Iterable[Dict[str, Any]], key: str) -> List[float]:
    vals = []
    for row in rows:
        val = safe_float(row.get(key))
        if val is not None:
            vals.append(val)
    return vals


def corr(xs: List[float], ys: List[float]) -> Optional[float]:
    pairs = [(x, y) for x, y in zip(xs, ys) if math.isfinite(x) and math.isfinite(y)]
    if len(pairs) < 2:
        return None
    xm = mean(x for x, _ in pairs)
    ym = mean(y for _, y in pairs)
    xv = sum((x - xm) ** 2 for x, _ in pairs)
    yv = sum((y - ym) ** 2 for _, y in pairs)
    if xv <= 0.0 or yv <= 0.0:
        return None
    return sum((x - xm) * (y - ym) for x, y in pairs) / math.sqrt(xv * yv)


def summarize_probe_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[
            (
                row.get("precision_mode"),
                row.get("quantization_algorithm", ""),
                row.get("group_size", row.get("block_size", "")),
                row.get("block_size", ""),
                row.get("direction_type"),
                safe_float(row.get("sparse_rate", row.get("p", 1.0))),
                row.get("sparse_mode", ""),
                row.get("sparse_rescale", ""),
                safe_float(row.get("h_raw")),
            )
        ].append(row)

    out = []
    for _, group in sorted(grouped.items(), key=lambda kv: (str(kv[0][4]), float(kv[0][5] or 0.0), float(kv[0][8] or 0.0))):
        first = group[0]
        fd = collect(group, "d_fd")
        true = collect(group, "d_true")
        fd_zero = [1.0 if bool(row.get("fd_is_zero")) else 0.0 for row in group if "fd_is_zero" in row]
        sign = [1.0 if bool(row.get("sign_match")) else 0.0 for row in group if row.get("sign_match") is not None]
        nmse = None
        if fd and true and len(fd) == len(true):
            denom = mean(y * y for y in true)
            nmse = mean((x - y) ** 2 for x, y in zip(fd, true)) / denom if denom > 0.0 else None
        out.append({
            "precision_mode": first.get("precision_mode"),
            "quantization_algorithm": first.get("quantization_algorithm"),
            "group_size": first.get("group_size", first.get("block_size")),
            "block_size": first.get("block_size"),
            "direction_type": first.get("direction_type"),
            "sparse_rate": safe_float(first.get("sparse_rate", first.get("p", 1.0))),
            "sparse_mode": first.get("sparse_mode"),
            "sparse_rescale": first.get("sparse_rescale"),
            "h_raw": safe_float(first.get("h_raw")),
            "h_active": safe_float(first.get("h_active")),
            "num_probe_rows": len(group),
            "probe_active_frac_mean": mean(collect(group, "probe_active_frac")) if collect(group, "probe_active_frac") else None,
            "probe_alignment_mean": mean(collect(group, "probe_alignment")) if collect(group, "probe_alignment") else None,
            "probe_norm_ratio_mean": mean(collect(group, "probe_norm_ratio")) if collect(group, "probe_norm_ratio") else None,
            "fd_zero_ratio": mean(fd_zero) if fd_zero else None,
            "corr_fd_true": corr(fd, true) if fd and true and len(fd) == len(true) else None,
            "nMSE_fd_true": nmse,
            "sign_agreement": mean(sign) if sign else None,
        })
    return out


def write_csv(path: Path, rows: List[Dict[str, Any]], fields: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def write_table_md(path: Path, title: str, rows: List[Dict[str, Any]], fields: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write(f"# {title}\n\n")
        f.write("| " + " | ".join(fields) + " |\n")
        f.write("| " + " | ".join(["---"] * len(fields)) + " |\n")
        for row in rows:
            f.write("| " + " | ".join(fmt(row.get(field)) for field in fields) + " |\n")


def probe_as_all_rows(probe_rows: List[Dict[str, Any]], notes: str) -> List[Dict[str, Any]]:
    out = []
    for row in probe_rows:
        out.append({
            "run_name": f"{row.get('direction_type')}_probe_p{fmt(row.get('sparse_rate'))}_h{fmt(row.get('h_raw'))}",
            "model": "roberta-large",
            "dataset": "SST-5",
            "quantization_algorithm": row.get("quantization_algorithm"),
            "group_size": row.get("group_size"),
            "update_backend": "probe_only",
            "direction_type": row.get("direction_type"),
            "h": row.get("h_raw"),
            "h_active": row.get("h_active"),
            "sparse_rate": row.get("sparse_rate"),
            "lr": 0,
            "steps_completed": 0,
            "seed": 16,
            "corr_fd_true": row.get("corr_fd_true"),
            "nMSE_fd_true": row.get("nMSE_fd_true"),
            "probe_alignment": row.get("probe_alignment_mean"),
            "probe_norm_ratio": row.get("probe_norm_ratio_mean"),
            "status": "probe_only",
            "notes": notes,
        })
    return out


def maybe_plots(root: Path, dense_probe: List[Dict[str, Any]], sparse_probe: List[Dict[str, Any]], all_rows: List[Dict[str, Any]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    plot_dir = root / "07_plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    def plot_xy(rows: List[Dict[str, Any]], x: str, y: str, name: str, *, logx: bool = False) -> None:
        pts = [(safe_float(row.get(x)), safe_float(row.get(y))) for row in rows]
        pts = [(a, b) for a, b in pts if a is not None and b is not None]
        if not pts:
            return
        pts.sort(key=lambda item: item[0])
        fig, ax = plt.subplots(figsize=(6.5, 4.0))
        ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o")
        if logx:
            ax.set_xscale("log")
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(plot_dir / name, dpi=160)
        plt.close(fig)

    plot_xy(dense_probe, "h_raw", "corr_fd_true", "dense_probe_corr_vs_h.png", logx=True)
    plot_xy(dense_probe, "h_raw", "nMSE_fd_true", "dense_probe_nmse_vs_h.png", logx=True)
    train_rows = [row for row in all_rows if row.get("update_backend") != "probe_only"]
    plot_xy(train_rows, "h", "best_eval_acc", "training_best_acc_vs_h.png", logx=True)
    if sparse_probe:
        fig, ax = plt.subplots(figsize=(7.0, 4.5))
        for p in sorted({safe_float(row.get("sparse_rate")) for row in sparse_probe if safe_float(row.get("sparse_rate")) is not None}):
            rows = [row for row in sparse_probe if safe_float(row.get("sparse_rate")) == p]
            rows.sort(key=lambda row: safe_float(row.get("h_active")) or 0.0)
            ax.plot([safe_float(row.get("h_active")) for row in rows], [safe_float(row.get("corr_fd_true")) for row in rows], marker="o", label=f"p={p:g}")
        ax.set_xscale("log")
        ax.set_xlabel("h_active")
        ax.set_ylabel("corr_fd_true")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(plot_dir / "sparse_probe_corr_vs_h_active.png", dpi=160)
        plt.close(fig)


def best_by(rows: List[Dict[str, Any]], key: str) -> Optional[Dict[str, Any]]:
    valid = [row for row in rows if safe_float(row.get(key)) is not None]
    if not valid:
        return None
    return max(valid, key=lambda row: safe_float(row.get(key)) or -1e9)


def find_h(rows: List[Dict[str, Any]], h: float) -> Optional[Dict[str, Any]]:
    for row in rows:
        val = safe_float(row.get("h_raw", row.get("h")))
        if val is not None and abs(val - h) <= max(1e-12, h * 1e-6):
            return row
    return None


def write_final_report(root: Path, dense_probe: List[Dict[str, Any]], sparse_probe: List[Dict[str, Any]], dense_train: List[Dict[str, Any]], residual_rows: List[Dict[str, Any]], all_rows: List[Dict[str, Any]]) -> None:
    best_dense_probe = best_by(dense_probe, "corr_fd_true")
    best_dense_train = best_by(dense_train, "best_eval_acc")
    best_sparse_probe = best_by(sparse_probe, "corr_fd_true")
    h1e3 = find_h(dense_probe, 1e-3)
    h15e3 = find_h(dense_probe, 1.5e-3)
    h2e3 = find_h(dense_probe, 2e-3)
    h3e3 = find_h(dense_probe, 3e-3)
    h1e2 = find_h(dense_probe, 1e-2)
    residual = best_by(residual_rows, "best_eval_acc")

    lines = [
        "# groupwise_int8_block256 Final Report",
        "",
        "## 1. Actual Quantizer",
        "",
        "This run uses `groupwise_int8_block256`: symmetric group-wise INT8 quantization with group/block size 256 and `calibration_samples=0`.",
        "",
        "It is not exact GPTQ. No Hessian-based GPTQ calibration is used in this code path.",
        "",
        "## 2. Dense Window",
        "",
        f"Best dense probe h by corr_fd_true: `{fmt(best_dense_probe.get('h_raw')) if best_dense_probe else ''}` with corr `{fmt(best_dense_probe.get('corr_fd_true')) if best_dense_probe else ''}` and nMSE `{fmt(best_dense_probe.get('nMSE_fd_true')) if best_dense_probe else ''}`.",
        "",
        f"h=1e-3: corr `{fmt(h1e3.get('corr_fd_true')) if h1e3 else ''}`, nMSE `{fmt(h1e3.get('nMSE_fd_true')) if h1e3 else ''}`.",
        f"h=1.5e-3: corr `{fmt(h15e3.get('corr_fd_true')) if h15e3 else ''}`, nMSE `{fmt(h15e3.get('nMSE_fd_true')) if h15e3 else ''}`.",
        f"h=2e-3: corr `{fmt(h2e3.get('corr_fd_true')) if h2e3 else ''}`, nMSE `{fmt(h2e3.get('nMSE_fd_true')) if h2e3 else ''}`.",
        f"h=3e-3: corr `{fmt(h3e3.get('corr_fd_true')) if h3e3 else ''}`, nMSE `{fmt(h3e3.get('nMSE_fd_true')) if h3e3 else ''}`.",
        f"h=1e-2: corr `{fmt(h1e2.get('corr_fd_true')) if h1e2 else ''}`, nMSE `{fmt(h1e2.get('nMSE_fd_true')) if h1e2 else ''}`.",
        "",
        "Interpretation should treat h=1e-2 as unreliable if derivative correlation/locality is poor, even when probe geometry looks active.",
        "",
        "## 3. Dense Training",
        "",
        f"Best dense FP16-master training row: `{best_dense_train.get('run_name') if best_dense_train else ''}` with best_eval_acc `{fmt(best_dense_train.get('best_eval_acc')) if best_dense_train else ''}` and last_eval_acc `{fmt(best_dense_train.get('last_eval_acc')) if best_dense_train else ''}`.",
        "",
        "Late collapse should be judged from the gap between best_eval_acc and last_eval_acc plus last_eval_loss.",
        "",
        "## 4. Residual Grid",
        "",
        f"Best residual-grid row: `{residual.get('run_name') if residual else ''}` with best_eval_acc `{fmt(residual.get('best_eval_acc')) if residual else ''}`, last_eval_acc `{fmt(residual.get('last_eval_acc')) if residual else ''}`, grid_error_norm `{fmt(residual.get('grid_error_norm')) if residual else ''}`, and scale_drift_max `{fmt(residual.get('scale_drift_max')) if residual else ''}`.",
        "",
        "Residual-grid remains mechanically clean only if grid_error_norm, scale_drift_max, and residual_bound_violation_frac stay zero or numerically negligible.",
        "",
        "## 5. Sparse Rate",
        "",
        f"Best sparse probe row: p=`{fmt(best_sparse_probe.get('sparse_rate')) if best_sparse_probe else ''}`, h_active=`{fmt(best_sparse_probe.get('h_active')) if best_sparse_probe else ''}`, corr `{fmt(best_sparse_probe.get('corr_fd_true')) if best_sparse_probe else ''}`, nMSE `{fmt(best_sparse_probe.get('nMSE_fd_true')) if best_sparse_probe else ''}`.",
        "",
        "Sparse training should only be started after this probe table points to a stable h_active/p pair.",
        "",
        "## 6. Recommendation",
        "",
        "Keep this as a robustness / quantizer-ablation setting unless repeated training confirms that the wider groupwise block-256 window improves stability, not just probe geometry.",
        "",
        "If groupwise block-256 is promoted later, use the best dense-training h from this continuation rather than h=1e-2, because h=1e-2 remains a locality risk.",
        "",
    ]
    (root / "06_summaries" / "groupwise256_final_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    args = parser.parse_args()
    root = Path(args.root).expanduser().resolve()

    dense_probe = summarize_probe_rows(read_jsonl(root / "02_dense_probe_window" / "probe_stats.jsonl"))
    write_csv(root / "02_dense_probe_window" / "groupwise256_dense_probe_summary.csv", dense_probe, PROBE_FIELDS)
    write_table_md(root / "02_dense_probe_window" / "groupwise256_dense_probe_summary.md", "groupwise_int8_block256 Dense Probe Summary", dense_probe, PROBE_FIELDS)

    sparse_raw = []
    for path in sorted((root / "04_sparse_probe_by_rate").glob("**/probe_stats.jsonl")):
        sparse_raw.extend(read_jsonl(path))
    sparse_probe = summarize_probe_rows(sparse_raw)
    write_csv(root / "04_sparse_probe_by_rate" / "groupwise256_sparse_probe_by_rate.csv", sparse_probe, PROBE_FIELDS)
    write_table_md(root / "04_sparse_probe_by_rate" / "groupwise256_sparse_probe_by_rate.md", "groupwise_int8_block256 Sparse Probe By Rate", sparse_probe, PROBE_FIELDS)

    dense_train = [summarize_run(path.parent, root) for path in sorted((root / "03_dense_fp16master_training").glob("**/run_summary.json"))]
    write_csv(root / "03_dense_fp16master_training" / "dense_training_summary.csv", dense_train, ALL_FIELDS)
    write_table_md(root / "03_dense_fp16master_training" / "dense_training_summary.md", "groupwise_int8_block256 Dense FP16-Master Training", dense_train, ALL_FIELDS)

    residual_rows = [summarize_run(path.parent, root) for path in sorted((root / "06_summaries").glob("residual_grid*/**/run_summary.json"))]
    write_csv(root / "06_summaries" / "residual_grid_groupwise256_summary.csv", residual_rows, ALL_FIELDS)
    write_table_md(root / "06_summaries" / "residual_grid_groupwise256_summary.md", "groupwise_int8_block256 Residual Grid Summary", residual_rows, ALL_FIELDS)

    all_rows = probe_as_all_rows(dense_probe, "02_dense_probe_window") + probe_as_all_rows(sparse_probe, "04_sparse_probe_by_rate") + dense_train + residual_rows
    write_csv(root / "06_summaries" / "groupwise256_all_summary.csv", all_rows, ALL_FIELDS)
    write_table_md(root / "06_summaries" / "groupwise256_all_summary.md", "groupwise_int8_block256 All Summary", all_rows, ALL_FIELDS)
    maybe_plots(root, dense_probe, sparse_probe, all_rows)
    write_final_report(root, dense_probe, sparse_probe, dense_train, residual_rows, all_rows)
    print(json.dumps({"dense_probe_rows": len(dense_probe), "sparse_probe_rows": len(sparse_probe), "dense_train_rows": len(dense_train), "residual_rows": len(residual_rows), "root": str(root)}, sort_keys=True))


if __name__ == "__main__":
    main()
