#!/usr/bin/env python3
"""Summarize next window/sparse/residual experiments with explicit best/last eval metrics."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


FINAL_ACC_ALIAS_WARNING = "final_acc may refer to best or last in old summaries"

SUMMARY_FIELDS = [
    "run_name",
    "seed",
    "precision_mode",
    "zo_quantization",
    "update_backend",
    "direction_type",
    "h_raw",
    "h_active",
    "sparse_rate",
    "sparse_mode",
    "sparse_rescale",
    "lr",
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
    "final_train_acc",
    "nan_occurred",
    "early_stopped",
    "final_acc",
    "final_acc_alias_warning",
    "family",
    "max_steps",
    "eval_every",
    "actual_active_frac_last",
    "actual_active_frac_mean",
    "projected_grad_mean",
    "projected_grad_std",
    "d_fd_mean",
    "d_fd_std",
    "cos_intended_actual_last",
    "actual_over_intended_norm_ratio_last",
    "saturation_frac_last",
    "residual_over_scale_p50_last",
    "residual_over_scale_p90_last",
    "residual_over_scale_p99_last",
    "residual_bound_violation_frac_last",
    "grid_error_norm_last",
    "scale_drift_max",
    "completed",
    "notes",
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
    val = safe_float(value)
    if val is None:
        return None
    return int(val)


def mean_std(values: Iterable[Any]) -> tuple[Optional[float], Optional[float]]:
    vals = [safe_float(v) for v in values]
    vals = [v for v in vals if v is not None]
    if not vals:
        return None, None
    if len(vals) == 1:
        return vals[0], 0.0
    return statistics.mean(vals), statistics.pstdev(vals)


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


def load_manifest(root: Path) -> Dict[str, Dict[str, Any]]:
    manifest = read_json(root / "config_manifest.json")
    cases = manifest.get("cases", []) if isinstance(manifest, dict) else []
    out = {}
    for case in cases:
        if isinstance(case, dict) and case.get("run_name"):
            out[str(case["run_name"])] = case
    return out


def first_acc(metrics: Dict[str, Any]) -> Optional[Any]:
    for key, value in metrics.items():
        if "acc" in str(key).lower():
            return value
    return None


def summary_eval(summary: Dict[str, Any]) -> tuple[Optional[float], Optional[float]]:
    last_loss = None
    last_acc = None
    for metrics in (summary.get("eval", {}) or {}).values():
        if isinstance(metrics, dict):
            last_loss = safe_float(metrics.get("eval_loss", last_loss))
            acc = first_acc(metrics)
            if acc is not None:
                last_acc = safe_float(acc)
    return last_loss, last_acc


def infer_run_name(run_dir: Path) -> str:
    if run_dir.name.startswith("seed"):
        return run_dir.parent.name
    return run_dir.name


def metrics_path(run_dir: Path) -> Path:
    return run_dir / "metrics_logs" / "metrics_adaptiveH-0_cscale-0.csv"


def eval_rows(metrics: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    for row in metrics:
        if safe_float(row.get("eval_acc")) is not None or safe_float(row.get("eval_loss")) is not None:
            rows.append(row)
    return rows


def best_last_eval(metrics: List[Dict[str, Any]], summary: Dict[str, Any]) -> Dict[str, Any]:
    rows = eval_rows(metrics)
    best_acc = best_acc_step = None
    best_loss = best_loss_step = None
    last_acc = last_acc_step = None
    last_loss = last_loss_step = None
    for row in rows:
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
        summary_loss, summary_acc = summary_eval(summary)
        step = safe_int((summary.get("train", {}) or {}).get("global_step"))
        if last_acc is None and summary_acc is not None:
            last_acc, last_acc_step = summary_acc, step
            best_acc, best_acc_step = (summary_acc, step) if best_acc is None else (best_acc, best_acc_step)
        if last_loss is None and summary_loss is not None:
            last_loss, last_loss_step = summary_loss, step
            best_loss, best_loss_step = (summary_loss, step) if best_loss is None else (best_loss, best_loss_step)
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


def last_train(metrics: List[Dict[str, Any]], updates: List[Dict[str, Any]]) -> tuple[Optional[float], Optional[float]]:
    for row in reversed(metrics):
        loss = safe_float(row.get("train_loss"))
        acc = safe_float(row.get("train_acc"))
        if loss is not None or acc is not None:
            return loss, acc
    if updates:
        return safe_float(updates[-1].get("train_loss")), None
    return None, None


def contains_nan(rows: Iterable[Dict[str, Any]]) -> bool:
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


def cfg_get(cfg: Dict[str, Any], manifest: Dict[str, Any], *names: str, default: Any = None) -> Any:
    for name in names:
        if name in cfg and cfg[name] not in (None, ""):
            return cfg[name]
        if name in manifest and manifest[name] not in (None, ""):
            return manifest[name]
    return default


def summarize_run(run_dir: Path, root: Path, manifest_cases: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    run_name = infer_run_name(run_dir)
    manifest = manifest_cases.get(run_name, {})
    summary = read_json(run_dir / "run_summary.json")
    cfg = (summary.get("config", {}) or {}).get("training_args", {}) or {}
    metadata = read_json(run_dir / "run_metadata.json")
    metrics = read_csv(metrics_path(run_dir))
    updates = read_jsonl(run_dir / "update_stats.jsonl")
    last_update = updates[-1] if updates else {}
    ev = best_last_eval(metrics, summary)
    train_loss, train_acc = last_train(metrics, updates)
    pg_mean, pg_std = mean_std(row.get("projected_grad") for row in updates)
    fd_mean, fd_std = mean_std(row.get("d_fd") for row in updates)
    active_mean, _ = mean_std(row.get("direction_sparse_active_fraction") for row in updates)
    h_raw = safe_float(cfg_get(cfg, manifest, "zero_order_eps", "h_raw", "zo_h"))
    sparse_rate = safe_float(cfg_get(cfg, manifest, "zo_direction_sparse_rate", "sparse_rate", "p", default=1.0))
    sparse_rescale = str(cfg_get(cfg, manifest, "zo_sparse_rescale", "sparse_rescale", default="none"))
    h_active = safe_float(cfg_get(cfg, manifest, "h_active"))
    if h_active is None:
        h_active = h_raw / math.sqrt(sparse_rate) if h_raw is not None and sparse_rate and sparse_rescale == "inv_sqrt_p" else h_raw
    max_steps = safe_int(cfg_get(cfg, manifest, "max_steps", "steps"))
    steps_completed = safe_int((summary.get("train", {}) or {}).get("global_step"))
    if steps_completed is None and updates:
        steps_completed = safe_int(updates[-1].get("global_step"))
    completed = bool(summary)
    early_stopped = False
    if max_steps is not None and steps_completed is not None:
        early_stopped = steps_completed < max_steps
    if not completed:
        early_stopped = True
    family = str(cfg_get(cfg, manifest, "family", default=""))
    if not family:
        backend = str(cfg_get(cfg, manifest, "zo_update_backend", "update_backend", default=""))
        direction = str(cfg_get(cfg, manifest, "direction_type", default="dense"))
        family = "residual" if backend == "residual_grid" else "sparse_screen" if direction == "sparse" else "dense"
    last_acc = ev["last_eval_acc"]
    return {
        "run_name": run_name,
        "seed": cfg_get(cfg, manifest, "seed"),
        "precision_mode": cfg_get(cfg, manifest, "precision_mode", default="int8"),
        "zo_quantization": cfg_get(cfg, manifest, "zo_quantization", default=metadata.get("zo_quantization")),
        "update_backend": cfg_get(cfg, manifest, "zo_update_backend", "update_backend"),
        "direction_type": cfg_get(cfg, manifest, "direction_type", default="sparse" if sparse_rate and sparse_rate < 1 else "dense"),
        "h_raw": h_raw,
        "h_active": h_active,
        "sparse_rate": sparse_rate,
        "sparse_mode": cfg_get(cfg, manifest, "zo_direction_sparse_mode", "sparse_mode"),
        "sparse_rescale": sparse_rescale,
        "lr": cfg_get(cfg, manifest, "learning_rate", "lr"),
        "steps_completed": steps_completed,
        **ev,
        "final_train_loss": train_loss,
        "final_train_acc": train_acc,
        "nan_occurred": contains_nan(metrics) or contains_nan(updates),
        "early_stopped": early_stopped,
        "final_acc": last_acc,
        "final_acc_alias_warning": FINAL_ACC_ALIAS_WARNING,
        "family": family,
        "max_steps": max_steps,
        "eval_every": cfg_get(cfg, manifest, "eval_steps", "eval_every"),
        "actual_active_frac_last": last_update.get("direction_sparse_active_fraction"),
        "actual_active_frac_mean": active_mean,
        "projected_grad_mean": pg_mean,
        "projected_grad_std": pg_std,
        "d_fd_mean": fd_mean,
        "d_fd_std": fd_std,
        "cos_intended_actual_last": last_update.get("global_cos_intended_actual", last_update.get("cos_intended_actual")),
        "actual_over_intended_norm_ratio_last": last_update.get("global_actual_over_intended_norm_ratio", last_update.get("actual_over_intended_norm_ratio")),
        "saturation_frac_last": last_update.get("global_saturation_frac", last_update.get("saturation_frac")),
        "residual_over_scale_p50_last": last_update.get("residual_over_scale_p50"),
        "residual_over_scale_p90_last": last_update.get("residual_over_scale_p90"),
        "residual_over_scale_p99_last": last_update.get("residual_over_scale_p99"),
        "residual_bound_violation_frac_last": last_update.get("unsaturated_residual_bound_violation_frac"),
        "grid_error_norm_last": last_update.get("grid_error_norm"),
        "scale_drift_max": max_scale_drift(run_dir),
        "completed": completed,
        "notes": "" if completed else "partial run; no run_summary.json",
    }


def find_run_dirs(root: Path) -> List[Path]:
    dirs = set()
    for path in root.glob("*/seed*/run_metadata.json"):
        dirs.add(path.parent)
    for path in root.glob("*/seed*/run_summary.json"):
        dirs.add(path.parent)
    for path in root.glob("*/seed*/update_stats.jsonl"):
        dirs.add(path.parent)
    return sorted(dirs)


def write_csv(path: Path, rows: List[Dict[str, Any]], fields: List[str] = SUMMARY_FIELDS) -> None:
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


def write_md(path: Path, rows: List[Dict[str, Any]], title: str) -> None:
    fields = [
        "run_name",
        "family",
        "seed",
        "h_raw",
        "h_active",
        "sparse_rate",
        "lr",
        "steps_completed",
        "best_eval_acc",
        "best_eval_step",
        "last_eval_acc",
        "last_eval_step",
        "best_eval_loss",
        "last_eval_loss",
        "early_stopped",
        "completed",
    ]
    with path.open("w", encoding="utf-8") as f:
        f.write(f"# {title}\n\n")
        f.write(f"`final_acc` compatibility note: {FINAL_ACC_ALIAS_WARNING}.\n\n")
        f.write("| " + " | ".join(fields) + " |\n")
        f.write("| " + " | ".join(["---"] * len(fields)) + " |\n")
        for row in rows:
            f.write("| " + " | ".join(fmt(row.get(field)) for field in fields) + " |\n")


def aggregate(rows: List[Dict[str, Any]], group_fields: List[str]) -> List[Dict[str, Any]]:
    groups: Dict[tuple, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row.get(field) for field in group_fields)].append(row)
    out = []
    for key, group in sorted(groups.items(), key=lambda item: str(item[0])):
        best_vals = [safe_float(r.get("best_eval_acc")) for r in group]
        last_vals = [safe_float(r.get("last_eval_acc")) for r in group]
        loss_vals = [safe_float(r.get("last_eval_loss")) for r in group]
        best_vals = [v for v in best_vals if v is not None]
        last_vals = [v for v in last_vals if v is not None]
        loss_vals = [v for v in loss_vals if v is not None]
        row = {field: value for field, value in zip(group_fields, key)}
        row.update({
            "n": len(group),
            "mean_best_eval_acc": statistics.mean(best_vals) if best_vals else None,
            "std_best_eval_acc": statistics.pstdev(best_vals) if len(best_vals) > 1 else 0.0 if best_vals else None,
            "mean_last_eval_acc": statistics.mean(last_vals) if last_vals else None,
            "std_last_eval_acc": statistics.pstdev(last_vals) if len(last_vals) > 1 else 0.0 if last_vals else None,
            "mean_last_eval_loss": statistics.mean(loss_vals) if loss_vals else None,
        })
        out.append(row)
    return out


def plot_if_available(root: Path, rows: List[Dict[str, Any]]) -> None:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        return
    plots = root / "plots"
    plots.mkdir(exist_ok=True)

    def run_metric_series(run_name: str, field: str) -> tuple[List[float], List[float]]:
        candidates = list(root.glob(f"{run_name}/seed*/metrics_logs/metrics_adaptiveH-0_cscale-0.csv"))
        if not candidates:
            return [], []
        xs, ys = [], []
        for row in read_csv(candidates[0]):
            y = safe_float(row.get(field))
            x = safe_float(row.get("global_step"))
            if x is not None and y is not None:
                xs.append(x)
                ys.append(y)
        return xs, ys

    for family, name in [("dense", "dense"), ("sparse_screen", "sparse"), ("residual", "residual")]:
        fam_rows = [r for r in rows if r.get("family") == family]
        if not fam_rows:
            continue
        for metric in ["eval_acc", "eval_loss"]:
            plt.figure(figsize=(9, 5))
            for row in fam_rows:
                xs, ys = run_metric_series(str(row["run_name"]), metric)
                if xs:
                    label = f"{row.get('run_name')}"
                    plt.plot(xs, ys, label=label, linewidth=1.2)
            plt.xlabel("step")
            plt.ylabel(metric)
            plt.title(f"{name} {metric} vs step")
            if len(fam_rows) <= 12:
                plt.legend(fontsize=7)
            plt.tight_layout()
            plt.savefig(plots / f"{name}_{metric}_vs_step.png", dpi=160)
            plt.close()

    dense = [r for r in rows if r.get("family") == "dense"]
    if dense:
        agg = aggregate(dense, ["h_active"])
        xs = [safe_float(r.get("h_active")) for r in agg]
        ys = [safe_float(r.get("mean_best_eval_acc")) for r in agg]
        y_last = [safe_float(r.get("mean_last_eval_acc")) for r in agg]
        pairs = [(x, y, z) for x, y, z in zip(xs, ys, y_last) if x is not None and y is not None and z is not None]
        if pairs:
            pairs.sort()
            plt.figure(figsize=(7, 4))
            plt.plot([p[0] for p in pairs], [p[1] for p in pairs], marker="o", label="best")
            plt.plot([p[0] for p in pairs], [p[2] for p in pairs], marker="o", label="last")
            plt.xscale("log")
            plt.xlabel("h")
            plt.ylabel("eval_acc")
            plt.title("Dense INT8 eval_acc by h")
            plt.legend()
            plt.tight_layout()
            plt.savefig(plots / "dense_best_last_eval_acc_by_h.png", dpi=160)
            plt.close()

    sparse = [r for r in rows if r.get("family") in {"sparse_screen", "sparse_promote"}]
    if sparse:
        plt.figure(figsize=(8, 5))
        for key, group in defaultdict(list, {k: [r for r in sparse if (r.get("sparse_rate"), r.get("lr")) == k] for k in {(r.get("sparse_rate"), r.get("lr")) for r in sparse}}).items():
            pts = []
            for row in group:
                x = safe_float(row.get("h_active"))
                y = safe_float(row.get("best_eval_acc"))
                if x is not None and y is not None:
                    pts.append((x, y))
            if pts:
                pts.sort()
                plt.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", label=f"p={key[0]},lr={key[1]}")
        plt.xlabel("h_active")
        plt.ylabel("best_eval_acc")
        plt.title("Sparse best eval_acc by h_active")
        plt.legend(fontsize=7)
        plt.tight_layout()
        plt.savefig(plots / "sparse_best_eval_acc_by_h_active.png", dpi=160)
        plt.close()

    residual = [r for r in rows if r.get("family") == "residual"]
    if residual:
        for field, out_name in [
            ("global_cos_intended_actual", "residual_cos_intended_actual_vs_step.png"),
            ("global_actual_over_intended_norm_ratio", "residual_norm_ratio_vs_step.png"),
            ("global_active_frac", "residual_active_frac_vs_step.png"),
            ("unsaturated_residual_bound_violation_frac", "residual_bound_violation_vs_step.png"),
        ]:
            plt.figure(figsize=(9, 5))
            for row in residual:
                path = next(iter(root.glob(f"{row['run_name']}/seed*/update_stats.jsonl")), None)
                if path is None:
                    continue
                xs, ys = [], []
                for update in read_jsonl(path):
                    x = safe_float(update.get("global_step"))
                    y = safe_float(update.get(field))
                    if x is not None and y is not None:
                        xs.append(x)
                        ys.append(y)
                if xs:
                    plt.plot(xs, ys, label=str(row["run_name"]), linewidth=1.1)
            plt.xlabel("step")
            plt.ylabel(field)
            plt.title(field)
            plt.legend(fontsize=7)
            plt.tight_layout()
            plt.savefig(plots / out_name, dpi=160)
            plt.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    root = Path(args.run_root).resolve()
    manifest = load_manifest(root)
    rows = [summarize_run(run_dir, root, manifest) for run_dir in find_run_dirs(root)]
    rows.sort(key=lambda r: (str(r.get("family")), str(r.get("run_name"))))

    write_csv(root / "summary.csv", rows)
    write_csv(root / "summary_dense.csv", [r for r in rows if r.get("family") == "dense"])
    write_csv(root / "summary_sparse_screen.csv", [r for r in rows if r.get("family") == "sparse_screen"])
    write_csv(root / "summary_promoted.csv", [r for r in rows if r.get("family") == "sparse_promote"])
    write_csv(root / "summary_residual.csv", [r for r in rows if r.get("family") == "residual"])

    dense_agg = aggregate([r for r in rows if r.get("family") == "dense"], ["h_active"])
    sparse_agg = aggregate([r for r in rows if r.get("family") in {"sparse_screen", "sparse_promote"}], ["sparse_rate", "h_active", "lr"])
    write_csv(root / "summary_dense_by_h.csv", dense_agg, ["h_active", "n", "mean_best_eval_acc", "std_best_eval_acc", "mean_last_eval_acc", "std_last_eval_acc", "mean_last_eval_loss"])
    write_csv(root / "summary_sparse_by_setting.csv", sparse_agg, ["sparse_rate", "h_active", "lr", "n", "mean_best_eval_acc", "std_best_eval_acc", "mean_last_eval_acc", "std_last_eval_acc", "mean_last_eval_loss"])
    write_md(root / "summary.md", rows, "Next Experiment Summary")

    if not args.no_plots:
        plot_if_available(root, rows)

    print(f"rows={len(rows)}")
    print(f"summary={root / 'summary.csv'}")
    print(f"summary_md={root / 'summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
