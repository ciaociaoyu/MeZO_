#!/usr/bin/env python3
"""Summarize local residual_grid H100 runs."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


FIELDS = [
    "run_name",
    "h",
    "lr",
    "residual_dtype",
    "commit_mode",
    "max_code_step",
    "update_norm_clip",
    "steps_completed",
    "best_eval_acc",
    "best_eval_step",
    "last_eval_acc",
    "last_eval_step",
    "best_eval_loss",
    "best_eval_loss_step",
    "last_eval_loss",
    "last_eval_loss_step",
    "final_train_loss",
    "nan_occurred",
    "early_stopped",
    "active_frac_last",
    "cos_intended_actual_last",
    "actual_over_intended_norm_ratio_last",
    "saturation_frac_last",
    "residual_over_scale_p99_last",
    "residual_over_scale_max_last",
    "residual_bound_violation_frac_last",
    "grid_error_norm_last",
    "scale_drift_max",
]


def safe_float(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def safe_int(value: Any) -> Optional[int]:
    out = safe_float(value)
    if out is None:
        return None
    return int(out)


def read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
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


def manifest_cases(root: Path) -> Dict[str, Dict[str, Any]]:
    manifest = read_json(root / "config_manifest.json")
    cases = manifest.get("cases", []) if isinstance(manifest, dict) else []
    return {
        str(case["run_name"]): case
        for case in cases
        if isinstance(case, dict) and case.get("run_name")
    }


def cfg_get(cfg: Dict[str, Any], manifest: Dict[str, Any], *names: str, default: Any = None) -> Any:
    for name in names:
        if cfg.get(name) not in (None, ""):
            return cfg[name]
        if manifest.get(name) not in (None, ""):
            return manifest[name]
    return default


def metrics_path(run_dir: Path) -> Path:
    return run_dir / "metrics_logs" / "metrics_adaptiveH-0_cscale-0.csv"


def eval_extrema(metrics: List[Dict[str, Any]], summary: Dict[str, Any]) -> Dict[str, Any]:
    best_acc = best_acc_step = None
    last_acc = last_acc_step = None
    best_loss = best_loss_step = None
    last_loss = last_loss_step = None
    for row in metrics:
        step = safe_int(row.get("global_step"))
        acc = safe_float(row.get("eval_acc"))
        loss = safe_float(row.get("eval_loss"))
        if acc is not None:
            last_acc, last_acc_step = acc, step
            if best_acc is None or acc > best_acc:
                best_acc, best_acc_step = acc, step
        if loss is not None:
            last_loss, last_loss_step = loss, step
            if best_loss is None or loss < best_loss:
                best_loss, best_loss_step = loss, step

    if last_acc is None or last_loss is None:
        step = safe_int((summary.get("train", {}) or {}).get("global_step"))
        for metrics_dict in (summary.get("eval", {}) or {}).values():
            if not isinstance(metrics_dict, dict):
                continue
            loss = safe_float(metrics_dict.get("eval_loss"))
            acc = None
            for key, value in metrics_dict.items():
                if "acc" in str(key).lower():
                    acc = safe_float(value)
                    break
            if acc is not None and last_acc is None:
                last_acc, last_acc_step = acc, step
                if best_acc is None or acc > best_acc:
                    best_acc, best_acc_step = acc, step
            if loss is not None and last_loss is None:
                last_loss, last_loss_step = loss, step
                if best_loss is None or loss < best_loss:
                    best_loss, best_loss_step = loss, step

    return {
        "best_eval_acc": best_acc,
        "best_eval_step": best_acc_step,
        "last_eval_acc": last_acc,
        "last_eval_step": last_acc_step,
        "best_eval_loss": best_loss,
        "best_eval_loss_step": best_loss_step,
        "last_eval_loss": last_loss,
        "last_eval_loss_step": last_loss_step,
    }


def final_train_loss(metrics: List[Dict[str, Any]], updates: List[Dict[str, Any]]) -> Optional[float]:
    if updates:
        loss = safe_float(updates[-1].get("train_loss"))
        if loss is not None:
            return loss
    for row in reversed(metrics):
        loss = safe_float(row.get("train_loss"))
        if loss is not None:
            return loss
    return None


def contains_bad_number(rows: Iterable[Dict[str, Any]]) -> bool:
    for row in rows:
        for value in row.values():
            if isinstance(value, str) and value.strip().lower() in {"nan", "inf", "-inf"}:
                return True
            if isinstance(value, float) and not math.isfinite(value):
                return True
    return False


def max_scale_drift(run_dir: Path) -> Optional[float]:
    vals = []
    for row in read_csv(run_dir / "scale_drift.csv"):
        val = safe_float(row.get("scale_delta_max"))
        if val is not None:
            vals.append(val)
    return max(vals) if vals else None


def find_run_dirs(root: Path) -> List[Path]:
    dirs = set()
    for pattern in ("*/seed*/run_summary.json", "*/seed*/run_metadata.json", "*/seed*/update_stats.jsonl"):
        for path in root.glob(pattern):
            dirs.add(path.parent)
    return sorted(dirs, key=lambda p: (p.parent.name, p.name))


def summarize_run(run_dir: Path, cases: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    run_name = run_dir.parent.name
    manifest = cases.get(run_name, {})
    summary = read_json(run_dir / "run_summary.json")
    cfg = (summary.get("config", {}) or {}).get("training_args", {}) or {}
    metrics = read_csv(metrics_path(run_dir))
    updates = read_jsonl(run_dir / "update_stats.jsonl")
    last = updates[-1] if updates else ((summary.get("artifacts", {}) or {}).get("update_stats_last_row", {}) or {})
    ev = eval_extrema(metrics, summary)
    max_steps = safe_int(cfg_get(cfg, manifest, "max_steps", "steps"))
    steps_completed = safe_int((summary.get("train", {}) or {}).get("global_step"))
    if steps_completed is None:
        steps_completed = len(updates) if updates else None
    early_stopped = True
    if summary and max_steps is not None and steps_completed is not None:
        early_stopped = steps_completed < max_steps
    elif summary:
        early_stopped = False
    return {
        "run_name": run_name,
        "h": cfg_get(cfg, manifest, "zero_order_eps", "h", "h_raw", default=last.get("h")),
        "lr": cfg_get(cfg, manifest, "learning_rate", "lr", default=last.get("lr")),
        "residual_dtype": cfg_get(cfg, manifest, "residual_dtype"),
        "commit_mode": cfg_get(cfg, manifest, "residual_commit_mode", "commit_mode", default=last.get("residual_commit_mode")),
        "max_code_step": cfg_get(cfg, manifest, "residual_max_code_step", "max_code_step", default=last.get("residual_max_code_step")),
        "update_norm_clip": cfg_get(cfg, manifest, "zo_update_norm_clip", "update_norm_clip", default=last.get("zo_update_norm_clip")),
        "steps_completed": steps_completed,
        **ev,
        "final_train_loss": final_train_loss(metrics, updates),
        "nan_occurred": contains_bad_number(metrics) or contains_bad_number(updates),
        "early_stopped": early_stopped,
        "active_frac_last": last.get("global_active_frac", last.get("active_frac")),
        "cos_intended_actual_last": last.get("global_cos_intended_actual", last.get("cos_intended_actual")),
        "actual_over_intended_norm_ratio_last": last.get(
            "global_actual_over_intended_norm_ratio", last.get("actual_over_intended_norm_ratio")
        ),
        "saturation_frac_last": last.get("global_saturation_frac", last.get("saturation_frac")),
        "residual_over_scale_p99_last": last.get("residual_over_scale_p99"),
        "residual_over_scale_max_last": last.get("residual_over_scale_max"),
        "residual_bound_violation_frac_last": last.get("unsaturated_residual_bound_violation_frac"),
        "grid_error_norm_last": last.get("grid_error_norm"),
        "scale_drift_max": max_scale_drift(run_dir),
    }


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in FIELDS})


def fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def write_md(path: Path, rows: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write("# Residual Local H100 Summary\n\n")
        f.write("| " + " | ".join(FIELDS) + " |\n")
        f.write("| " + " | ".join(["---"] * len(FIELDS)) + " |\n")
        for row in rows:
            f.write("| " + " | ".join(fmt(row.get(field)) for field in FIELDS) + " |\n")


def update_series(run_dir: Path, field: str) -> Tuple[List[float], List[float]]:
    xs, ys = [], []
    for row in read_jsonl(run_dir / "update_stats.jsonl"):
        x = safe_float(row.get("global_step"))
        y = safe_float(row.get(field))
        if y is None and field.startswith("global_"):
            y = safe_float(row.get(field[len("global_") :]))
        if x is not None and y is not None:
            xs.append(x)
            ys.append(y)
    return xs, ys


def metric_series(run_dir: Path, field: str) -> Tuple[List[float], List[float]]:
    xs, ys = [], []
    for row in read_csv(metrics_path(run_dir)):
        x = safe_float(row.get("global_step"))
        y = safe_float(row.get(field))
        if x is not None and y is not None:
            xs.append(x)
            ys.append(y)
    return xs, ys


def plot_if_available(root: Path, rows: List[Dict[str, Any]]) -> None:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        return

    plots = root / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    dirs = {run_dir.parent.name: run_dir for run_dir in find_run_dirs(root)}

    metric_specs = [
        ("eval_acc", "eval_acc_vs_step.png", "eval_acc", metric_series),
        ("eval_loss", "eval_loss_vs_step.png", "eval_loss", metric_series),
        ("global_cos_intended_actual", "cos_intended_actual_vs_step.png", "cos_intended_actual", update_series),
        (
            "global_actual_over_intended_norm_ratio",
            "actual_over_intended_norm_ratio_vs_step.png",
            "actual_over_intended_norm_ratio",
            update_series,
        ),
        ("global_active_frac", "active_frac_vs_step.png", "active_frac", update_series),
        ("residual_over_scale_p99", "residual_over_scale_p99_vs_step.png", "residual_over_scale_p99", update_series),
    ]
    for field, filename, ylabel, loader in metric_specs:
        plt.figure(figsize=(9, 5))
        plotted = False
        for row in rows:
            run_name = str(row["run_name"])
            run_dir = dirs.get(run_name)
            if run_dir is None:
                continue
            xs, ys = loader(run_dir, field)
            if not xs:
                continue
            plt.plot(xs, ys, label=run_name, linewidth=1.2)
            plotted = True
        if plotted:
            plt.xlabel("step")
            plt.ylabel(ylabel)
            plt.title(f"{ylabel} vs step")
            if len(rows) <= 12:
                plt.legend(fontsize=7)
            plt.tight_layout()
            plt.savefig(plots / filename, dpi=160)
        plt.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    root = Path(args.run_root).resolve()
    cases = manifest_cases(root)
    rows = [summarize_run(run_dir, cases) for run_dir in find_run_dirs(root)]
    rows.sort(key=lambda r: str(r.get("run_name")))
    write_csv(root / "summary_residual.csv", rows)
    write_md(root / "summary_residual.md", rows)
    if not args.no_plots:
        plot_if_available(root, rows)
    print(f"rows={len(rows)}")
    print(f"summary_csv={root / 'summary_residual.csv'}")
    print(f"summary_md={root / 'summary_residual.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
