#!/usr/bin/env python3
"""Summarize future probe-window training experiments."""

from __future__ import annotations

import ast
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


ALL_FIELDS = [
    "run_name", "family", "seed", "backend", "direction_type", "h_raw", "h_active", "p",
    "sparse_mode", "sparse_rescale", "lr", "steps_completed", "final_train_loss",
    "final_eval_loss", "final_acc", "best_acc", "nan_occurred", "projected_grad_mean",
    "projected_grad_std", "d_fd_mean", "d_fd_std", "actual_active_frac",
    "update_active_frac_last", "cos_intended_actual_last", "actual_over_intended_norm_ratio_last",
    "saturation_frac_last", "residual_over_scale_p99_last",
    "residual_bound_violation_frac_last", "grid_error_norm_last", "scale_drift_max",
]


def safe_float(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def read_csv(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def first_accuracy(metrics: Dict[str, Any]) -> Optional[Any]:
    for key, value in metrics.items():
        if "acc" in str(key).lower():
            return value
    return None


def final_eval(summary: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {"final_eval_loss": None, "final_acc": None}
    for metrics in (summary.get("eval", {}) or {}).values():
        if not isinstance(metrics, dict):
            continue
        out["final_eval_loss"] = metrics.get("eval_loss", out["final_eval_loss"])
        acc = first_accuracy(metrics)
        if acc is not None:
            out["final_acc"] = acc
    return out


def metric_mean_std(values: Iterable[Any]) -> tuple[Optional[float], Optional[float]]:
    vals = [safe_float(v) for v in values]
    vals = [v for v in vals if v is not None]
    if not vals:
        return None, None
    if len(vals) == 1:
        return vals[0], 0.0
    return statistics.mean(vals), statistics.pstdev(vals)


def parse_log_dicts(path: Path) -> List[Dict[str, Any]]:
    rows = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        idx = line.find("{'loss'")
        if idx < 0:
            continue
        try:
            rows.append(ast.literal_eval(line[idx:]))
        except Exception:
            continue
    return rows


def max_scale_drift(run_dir: Path) -> Optional[float]:
    path = run_dir / "scale_drift.csv"
    if not path.exists():
        return None
    vals = []
    for row in read_csv(path):
        val = safe_float(row.get("scale_delta_max"))
        if val is not None:
            vals.append(val)
    return max(vals) if vals else None


def infer_family(run_name: str, cfg: Dict[str, Any]) -> str:
    backend = str(cfg.get("zo_update_backend") or "")
    p = safe_float(cfg.get("zo_direction_sparse_rate"))
    if backend == "residual_grid":
        return "residual"
    if p is not None and p < 1.0:
        return "sparse"
    return "dense"


def summarize_run(path: Path, root: Path) -> Dict[str, Any]:
    run_dir = path.parent
    run_name = run_dir.parent.name
    summary = json.loads(path.read_text(encoding="utf-8"))
    cfg = (summary.get("config", {}) or {}).get("training_args", {}) or {}
    metrics_rows = read_csv(run_dir / "metrics_logs" / "metrics_adaptiveH-0_cscale-0.csv")
    last_metrics = metrics_rows[-1] if metrics_rows else {}
    eval_accs = [safe_float(row.get("eval_acc")) for row in metrics_rows]
    eval_accs = [x for x in eval_accs if x is not None]
    update_rows = read_jsonl(run_dir / "update_stats.jsonl")
    last_update = update_rows[-1] if update_rows else ((summary.get("artifacts", {}) or {}).get("update_stats_last_row", {}) or {})
    log_rows = parse_log_dicts(root / "logs" / f"{run_name}.log")
    pg_mean, pg_std = metric_mean_std(row.get("projected_grad") for row in update_rows)
    fd_mean, fd_std = metric_mean_std(row.get("d_fd") for row in update_rows)
    active_mean, _ = metric_mean_std(row.get("direction_sparse_active_fraction") for row in log_rows)
    h_raw = safe_float(cfg.get("zero_order_eps"))
    p = safe_float(cfg.get("zo_direction_sparse_rate"))
    sparse_rescale = cfg.get("zo_sparse_rescale")
    h_active = h_raw
    if h_raw is not None and p is not None and p > 0 and sparse_rescale == "inv_sqrt_p":
        h_active = h_raw / math.sqrt(p)
    eval_info = final_eval(summary)
    return {
        "run_name": run_name,
        "family": infer_family(run_name, cfg),
        "seed": cfg.get("seed"),
        "backend": cfg.get("zo_update_backend"),
        "direction_type": "sparse" if p is not None and p < 1.0 else "dense",
        "h_raw": h_raw,
        "h_active": h_active,
        "p": p,
        "sparse_mode": cfg.get("zo_direction_sparse_mode"),
        "sparse_rescale": sparse_rescale,
        "lr": cfg.get("learning_rate"),
        "steps_completed": (summary.get("train", {}) or {}).get("global_step"),
        "final_train_loss": last_metrics.get("train_loss") or last_update.get("train_loss"),
        "final_eval_loss": eval_info["final_eval_loss"],
        "final_acc": eval_info["final_acc"],
        "best_acc": max(eval_accs) if eval_accs else eval_info["final_acc"],
        "nan_occurred": any(str(v).lower() in {"nan", "inf", "-inf"} for row in metrics_rows for v in row.values()),
        "projected_grad_mean": pg_mean,
        "projected_grad_std": pg_std,
        "d_fd_mean": fd_mean,
        "d_fd_std": fd_std,
        "actual_active_frac": active_mean,
        "update_active_frac_last": last_update.get("global_active_frac", last_update.get("active_frac")),
        "cos_intended_actual_last": last_update.get("global_cos_intended_actual", last_update.get("cos_intended_actual")),
        "actual_over_intended_norm_ratio_last": last_update.get("global_actual_over_intended_norm_ratio", last_update.get("actual_over_intended_norm_ratio")),
        "saturation_frac_last": last_update.get("global_saturation_frac", last_update.get("saturation_frac")),
        "residual_over_scale_p99_last": last_update.get("residual_over_scale_p99"),
        "residual_bound_violation_frac_last": last_update.get("unsaturated_residual_bound_violation_frac"),
        "grid_error_norm_last": last_update.get("grid_error_norm"),
        "scale_drift_max": max_scale_drift(run_dir),
    }


def summarize_checkpoint_probe(root: Path) -> List[Dict[str, Any]]:
    rows = []
    for path in sorted(root.glob("*/seed*/checkpoint_probe_stats.jsonl")):
        run_name = path.parent.parent.name
        grouped: Dict[tuple, List[Dict[str, Any]]] = defaultdict(list)
        for row in read_jsonl(path):
            key = (
                row.get("checkpoint_step"),
                row.get("h_raw"),
                row.get("h_active"),
                row.get("p", row.get("sparse_rate")),
            )
            grouped[key].append(row)
        for (step, h_raw, h_active, p), group in grouped.items():
            fd_vals = [safe_float(r.get("d_fd")) for r in group]
            true_vals = [safe_float(r.get("d_true")) for r in group]
            pairs = [(x, y) for x, y in zip(fd_vals, true_vals) if x is not None and y is not None]
            corr = None
            nmse = None
            if len(pairs) >= 2:
                xs, ys = zip(*pairs)
                mx, my = statistics.mean(xs), statistics.mean(ys)
                vx = sum((x - mx) ** 2 for x in xs)
                vy = sum((y - my) ** 2 for y in ys)
                denom = math.sqrt(vx * vy)
                corr = sum((x - mx) * (y - my) for x, y in pairs) / denom if denom > 0 else None
                denom_mse = sum(y * y for y in ys) / len(ys)
                mse = sum((x - y) ** 2 for x, y in pairs) / len(pairs)
                nmse = mse / (denom_mse + 1e-12)
            align_vals = [safe_float(r.get("probe_alignment")) for r in group]
            align_vals = [v for v in align_vals if v is not None]
            ratio_vals = [safe_float(r.get("probe_norm_ratio")) for r in group]
            ratio_vals = [v for v in ratio_vals if v is not None]
            rows.append({
                "run_name": run_name,
                "step": step,
                "h_raw": h_raw,
                "h_active": h_active,
                "p": p,
                "probe_alignment": statistics.mean(align_vals) if align_vals else None,
                "probe_norm_ratio": statistics.mean(ratio_vals) if ratio_vals else None,
                "corr_fd_true": corr,
                "nMSE_fd_true": nmse,
            })
    return rows


def write_csv(path: Path, rows: List[Dict[str, Any]], fields: List[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_md(path: Path, rows: List[Dict[str, Any]], fields: List[str], title: str) -> None:
    def fmt(v: Any) -> str:
        if v is None:
            return ""
        if isinstance(v, float):
            return f"{v:.6g}"
        return str(v)
    with path.open("w", encoding="utf-8") as f:
        f.write(f"# {title}\n\n")
        f.write("| " + " | ".join(fields) + " |\n")
        f.write("| " + " | ".join(["---"] * len(fields)) + " |\n")
        for row in rows:
            f.write("| " + " | ".join(fmt(row.get(field)) for field in fields) + " |\n")


def aggregate(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[tuple, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (row.get("family"), row.get("h_active"), row.get("p"), row.get("lr"), row.get("backend"))
        groups[key].append(row)
    out = []
    for (family, h_active, p, lr, backend), group in sorted(groups.items(), key=lambda kv: str(kv[0])):
        best_vals = [safe_float(r.get("best_acc")) for r in group]
        final_vals = [safe_float(r.get("final_acc")) for r in group]
        best_vals = [x for x in best_vals if x is not None]
        final_vals = [x for x in final_vals if x is not None]
        out.append({
            "method": family,
            "backend": backend,
            "p": p,
            "h_active": h_active,
            "lr": lr,
            "n": len(group),
            "mean_best_acc": statistics.mean(best_vals) if best_vals else None,
            "std_best_acc": statistics.pstdev(best_vals) if len(best_vals) > 1 else 0.0 if best_vals else None,
            "mean_final_acc": statistics.mean(final_vals) if final_vals else None,
            "std_final_acc": statistics.pstdev(final_vals) if len(final_vals) > 1 else 0.0 if final_vals else None,
        })
    return out


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root")
    args = parser.parse_args()
    root = Path(args.run_root).resolve()
    rows = [summarize_run(path, root) for path in sorted(root.glob("*/seed*/run_summary.json"))]
    dense = [r for r in rows if r.get("family") == "dense"]
    sparse = [r for r in rows if r.get("family") == "sparse"]
    residual = [r for r in rows if r.get("family") == "residual"]
    checkpoint = summarize_checkpoint_probe(root)
    agg = aggregate([r for r in rows if r.get("family") in {"dense", "sparse"}])

    write_csv(root / "summary_all.csv", rows, ALL_FIELDS)
    write_csv(root / "summary_dense.csv", dense, ALL_FIELDS)
    write_csv(root / "summary_sparse.csv", sparse, ALL_FIELDS)
    write_csv(root / "summary_residual.csv", residual, ALL_FIELDS)
    write_csv(root / "summary_promoted_compare.csv", agg, ["method", "backend", "p", "h_active", "lr", "n", "mean_best_acc", "std_best_acc", "mean_final_acc", "std_final_acc"])
    write_csv(root / "summary_checkpoint_probe.csv", checkpoint, ["run_name", "step", "h_raw", "h_active", "p", "probe_alignment", "probe_norm_ratio", "corr_fd_true", "nMSE_fd_true"])

    write_md(root / "summary.md", rows, ["run_name", "family", "seed", "h_raw", "h_active", "p", "lr", "best_acc", "final_acc", "final_eval_loss", "nan_occurred"], "Future Probe Window Training Summary")
    print(f"rows={len(rows)}")
    print(f"summary_all={root / 'summary_all.csv'}")
    print(f"summary_dense={root / 'summary_dense.csv'}")
    print(f"summary_sparse={root / 'summary_sparse.csv'}")
    print(f"summary_checkpoint_probe={root / 'summary_checkpoint_probe.csv'}")
    print(f"summary_residual={root / 'summary_residual.csv'}")
    print(f"summary_md={root / 'summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
