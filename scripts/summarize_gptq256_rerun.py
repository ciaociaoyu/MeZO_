#!/usr/bin/env python3
"""Summarize GPTQ-256 INT8 rerun artifacts.

The run may use the honest groupwise_int8_block256 fallback when exact GPTQ is
unavailable in medium_models.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional, Tuple


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

PROBE_FIELDS = [
    "precision_mode",
    "quantization_algorithm",
    "group_size",
    "direction_type",
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


def safe_float(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        out = float(value)
    except Exception:
        return None
    return out if math.isfinite(out) else None


def safe_int(value: Any) -> Optional[int]:
    val = safe_float(value)
    return int(val) if val is not None else None


def read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
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
        except Exception:
            continue
    return rows


def read_csv(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def collect(rows: Iterable[Dict[str, Any]], key: str) -> List[float]:
    out = []
    for row in rows:
        val = safe_float(row.get(key))
        if val is not None:
            out.append(val)
    return out


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
                row.get("direction_type"),
                safe_float(row.get("h_raw")),
            )
        ].append(row)
    out = []
    for _, group in sorted(grouped.items(), key=lambda kv: (str(kv[0][3]), safe_float(kv[0][4]) or 0.0)):
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
            "direction_type": first.get("direction_type"),
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


def metrics_path(run_dir: Path) -> Path:
    return run_dir / "metrics_logs" / "metrics_adaptiveH-0_cscale-0.csv"


def eval_extrema(metrics: List[Dict[str, Any]], summary: Dict[str, Any]) -> Dict[str, Any]:
    best_acc = last_acc = best_loss = last_loss = None
    for row in metrics:
        acc = safe_float(row.get("eval_acc"))
        loss = safe_float(row.get("eval_loss"))
        if acc is not None:
            last_acc = acc
            best_acc = acc if best_acc is None or acc > best_acc else best_acc
        if loss is not None:
            last_loss = loss
            best_loss = loss if best_loss is None or loss < best_loss else best_loss
    if last_acc is None or last_loss is None:
        for metrics_dict in (summary.get("eval", {}) or {}).values():
            if not isinstance(metrics_dict, dict):
                continue
            if last_loss is None:
                last_loss = safe_float(metrics_dict.get("eval_loss"))
                best_loss = last_loss if best_loss is None else best_loss
            if last_acc is None:
                for key, value in metrics_dict.items():
                    if "acc" in str(key).lower():
                        last_acc = safe_float(value)
                        best_acc = last_acc if best_acc is None else best_acc
                        break
    return {
        "best_eval_acc": best_acc,
        "last_eval_acc": last_acc,
        "best_eval_loss": best_loss,
        "last_eval_loss": last_loss,
    }


def max_scale_drift(run_dir: Path) -> Optional[float]:
    vals = collect(read_csv(run_dir / "scale_drift.csv"), "scale_delta_max")
    return max(vals) if vals else None


def has_bad_number(rows: Iterable[Dict[str, Any]]) -> bool:
    for row in rows:
        for value in row.values():
            if isinstance(value, str) and value.strip().lower() in {"nan", "inf", "-inf"}:
                return True
            if isinstance(value, float) and not math.isfinite(value):
                return True
    return False


def infer_section(run_dir: Path, root: Path) -> str:
    try:
        rel = run_dir.relative_to(root)
    except Exception:
        return ""
    return rel.parts[0] if rel.parts else ""


def summarize_run(run_dir: Path, root: Path) -> Dict[str, Any]:
    summary = read_json(run_dir / "run_summary.json")
    cfg = (summary.get("config", {}) or {}).get("training_args", {}) or {}
    data_cfg = (summary.get("config", {}) or {}).get("data_args", {}) or {}
    model_cfg = (summary.get("config", {}) or {}).get("model_args", {}) or {}
    metrics = read_csv(metrics_path(run_dir))
    update_rows = read_jsonl(run_dir / "update_stats.jsonl")
    last_update = update_rows[-1] if update_rows else ((summary.get("artifacts", {}) or {}).get("update_stats_last_row", {}) or {})
    run_name = run_dir.parent.name
    section = infer_section(run_dir, root)
    status = "completed" if (run_dir / "run_summary.json").exists() else "partial"
    if has_bad_number(metrics) or has_bad_number(update_rows):
        status = "nan_or_inf"
    h_raw = safe_float(cfg.get("zero_order_eps") or last_update.get("h"))
    sparse_rate = safe_float(cfg.get("zo_direction_sparse_rate") or cfg.get("sparse_rate"))
    h_active = h_raw
    if cfg.get("direction_type") == "sparse" and cfg.get("zo_sparse_rescale") == "inv_sqrt_p" and h_raw is not None and sparse_rate and sparse_rate > 0:
        h_active = h_raw / math.sqrt(sparse_rate)
    raw_steps_completed = safe_int((summary.get("train", {}) or {}).get("global_step")) or safe_int(last_update.get("global_step"))
    max_steps = safe_int(cfg.get("max_steps"))
    steps_completed = raw_steps_completed
    if raw_steps_completed is not None and max_steps is not None and max_steps > 0:
        steps_completed = min(raw_steps_completed, max_steps)
    row = {
        "run_name": run_name,
        "model": model_cfg.get("model_name_or_path", "roberta-large"),
        "dataset": data_cfg.get("task_name", "SST-5"),
        "quantization_algorithm": cfg.get("quantization_algorithm") or last_update.get("quantization_algorithm"),
        "group_size": cfg.get("quantization_group_size") or last_update.get("group_size"),
        "update_backend": cfg.get("zo_update_backend") or last_update.get("update_backend"),
        "direction_type": cfg.get("direction_type"),
        "h": h_raw,
        "h_active": h_active,
        "sparse_rate": sparse_rate,
        "lr": cfg.get("learning_rate") or last_update.get("lr"),
        "steps_completed": steps_completed,
        "seed": cfg.get("seed"),
        **eval_extrema(metrics, summary),
        "corr_fd_true": None,
        "nMSE_fd_true": None,
        "probe_alignment": None,
        "probe_norm_ratio": None,
        "active_frac": last_update.get("global_active_frac", last_update.get("active_frac")),
        "cos_intended_actual": last_update.get("global_cos_intended_actual", last_update.get("cos_intended_actual")),
        "actual_over_intended_norm_ratio": last_update.get("global_actual_over_intended_norm_ratio", last_update.get("actual_over_intended_norm_ratio")),
        "acc_actual_cos": last_update.get("global_acc_actual_cos", last_update.get("acc_actual_cos")),
        "actual_over_acc_norm_ratio": last_update.get("global_actual_over_acc_norm_ratio", last_update.get("actual_over_acc_norm_ratio")),
        "residual_bound_violation_frac": last_update.get("unsaturated_residual_bound_violation_frac"),
        "grid_error_norm": last_update.get("grid_error_norm"),
        "scale_drift_max": max_scale_drift(run_dir),
        "nan_occurred": status == "nan_or_inf",
        "status": status,
        "notes": section,
    }
    return row


def probe_as_all_rows(probe_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for row in probe_rows:
        out.append({
            "run_name": f"probe_h_{row.get('h_raw')}",
            "model": "roberta-large",
            "dataset": "SST-5",
            "quantization_algorithm": row.get("quantization_algorithm"),
            "group_size": row.get("group_size"),
            "update_backend": "probe_only",
            "direction_type": row.get("direction_type"),
            "h": row.get("h_raw"),
            "h_active": row.get("h_active"),
            "sparse_rate": 1.0,
            "lr": 0,
            "steps_completed": 0,
            "seed": 16,
            "corr_fd_true": row.get("corr_fd_true"),
            "nMSE_fd_true": row.get("nMSE_fd_true"),
            "probe_alignment": row.get("probe_alignment_mean"),
            "probe_norm_ratio": row.get("probe_norm_ratio_mean"),
            "status": "probe_only",
            "notes": "02_probe_window",
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


def maybe_plots(root: Path, all_rows: List[Dict[str, Any]], probe_rows: List[Dict[str, Any]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    plot_dir = root / "08_plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    def plot_xy(rows: List[Dict[str, Any]], x: str, y: str, name: str, *, logx: bool = False) -> None:
        pts = [(safe_float(row.get(x)), safe_float(row.get(y)), str(row.get("run_name", ""))) for row in rows]
        pts = [(a, b, label) for a, b, label in pts if a is not None and b is not None]
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

    plot_xy(probe_rows, "h_raw", "corr_fd_true", "probe_corr_vs_h.png", logx=True)
    plot_xy(probe_rows, "h_raw", "nMSE_fd_true", "probe_nmse_vs_h.png", logx=True)
    plot_xy(all_rows, "h", "best_eval_acc", "train_best_acc_vs_h.png", logx=True)
    plot_xy(all_rows, "h", "last_eval_loss", "train_last_loss_vs_h.png", logx=True)


def write_comparison_docs(root: Path, all_rows: List[Dict[str, Any]], probe_rows: List[Dict[str, Any]]) -> None:
    summaries = root / "07_summaries"
    summaries.mkdir(parents=True, exist_ok=True)
    exact = any(str(row.get("quantization_algorithm")) == "gptq" for row in all_rows)
    actual_algos = sorted({str(row.get("quantization_algorithm")) for row in all_rows if row.get("quantization_algorithm")})
    best_probe = max(
        [row for row in probe_rows if safe_float(row.get("corr_fd_true")) is not None],
        key=lambda r: safe_float(r.get("corr_fd_true")) or -1e9,
        default=None,
    )
    train_rows = [row for row in all_rows if row.get("update_backend") != "probe_only"]
    best_train = max(
        [row for row in train_rows if safe_float(row.get("best_eval_acc")) is not None],
        key=lambda r: safe_float(r.get("best_eval_acc")) or -1e9,
        default=None,
    )
    text = [
        "# GPTQ-256 vs Previous INT8 Summary",
        "",
        f"Exact GPTQ used: `{exact}`.",
        f"Actual quantizer labels observed: `{', '.join(actual_algos) if actual_algos else 'none yet'}`.",
        "",
        "This rerun should be interpreted as exact GPTQ only if the quantizer report says so. In the current code path, the expected fallback is `groupwise_int8_block256`.",
        "",
        "## Probe Window",
        "",
        f"Best observed probe h by corr_fd_true: `{fmt(best_probe.get('h_raw')) if best_probe else ''}` with corr `{fmt(best_probe.get('corr_fd_true')) if best_probe else ''}`.",
        "",
        "Previous INT8 expectation: useful signal around h=2e-3 to 3e-3; too-small h distorted; h=1e-2 can look geometrically active but fail locality.",
        "",
        "## Training",
        "",
        f"Best observed training run: `{best_train.get('run_name') if best_train else ''}` best_eval_acc=`{fmt(best_train.get('best_eval_acc')) if best_train else ''}`.",
        "",
        "Dense, sparse, direct-update, and residual-grid conclusions should be filled from completed rows in `gptq256_all_runs_summary.csv`.",
        "",
    ]
    (summaries / "gptq256_vs_previous_int8_summary.md").write_text("\n".join(text), encoding="utf-8")
    (root / "02_probe_window" / "gptq256_vs_previous_int8_probe_comparison.md").write_text("\n".join(text[:14]), encoding="utf-8")

    final = [
        "# Final GPTQ-256 Report",
        "",
        "1. Was GPTQ-256 actually used?",
        f"   - Exact GPTQ observed: `{exact}`. Actual labels: `{', '.join(actual_algos) if actual_algos else 'none yet'}`.",
        "2. Was exact GPTQ used or groupwise fallback?",
        "   - See `01_quantizer_checks/gptq256_quantizer_report.md`; current medium_models path is expected to use `groupwise_int8_block256` fallback.",
        "3. Does it reproduce the previous INT8 probe window?",
        f"   - Current best probe h: `{fmt(best_probe.get('h_raw')) if best_probe else ''}`.",
        "4. Does selected h remain around 2e-3 to 3e-3?",
        "   - Answer from probe rows above; do not overclaim before full grid completes.",
        "5. Does dense INT8 + FP16 master still train best inside the window?",
        f"   - Current best training row: `{best_train.get('run_name') if best_train else ''}`.",
        "6. Does direct INT8 update still fail?",
        "   - Check rows from `05_int8_direct_update` for active_frac and norm-ratio distortion.",
        "7. Does sparse behavior match prior sparse behavior?",
        "   - Check rows from `04_int8_fp16master_sparse`.",
        "8. Does residual-grid remain clean and useful?",
        "   - Check rows from `06_int8_residual_grid` for grid_error_norm, scale_drift_max, and EF metrics.",
        "9. Did block/group size 256 materially improve or shift the window?",
        "   - Compare probe and dense training rows against previous INT8 summaries.",
        "10. Should GPTQ-256 be included in the paper?",
        "   - Treat as a robustness/appendix ablation unless exact GPTQ is later implemented and results are clearly material.",
        "",
    ]
    (summaries / "final_gptq256_report.md").write_text("\n".join(final), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    args = parser.parse_args()
    root = Path(args.root).expanduser().resolve()

    probe_jsonl = root / "02_probe_window" / "probe_stats.jsonl"
    probe_rows = summarize_probe_rows(read_jsonl(probe_jsonl))
    write_csv(root / "02_probe_window" / "gptq256_dense_probe_summary.csv", probe_rows, PROBE_FIELDS)
    write_table_md(root / "02_probe_window" / "gptq256_dense_probe_summary.md", "GPTQ-256 Dense Probe Summary", probe_rows, PROBE_FIELDS)

    train_rows = []
    for path in sorted(root.glob("0[3-6]_*/**/run_summary.json")):
        train_rows.append(summarize_run(path.parent, root))
    all_rows = probe_as_all_rows(probe_rows) + train_rows
    write_csv(root / "07_summaries" / "gptq256_all_runs_summary.csv", all_rows, ALL_FIELDS)
    write_table_md(root / "07_summaries" / "gptq256_all_runs_summary.md", "GPTQ-256 All Runs Summary", all_rows, ALL_FIELDS)
    maybe_plots(root, all_rows, probe_rows)
    write_comparison_docs(root, all_rows, probe_rows)
    print(json.dumps({"probe_rows": len(probe_rows), "train_rows": len(train_rows), "root": str(root)}, sort_keys=True))


if __name__ == "__main__":
    main()
