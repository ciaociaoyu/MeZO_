#!/usr/bin/env python3
"""Build V10 supplement audit/results bundle from existing MeZO artifacts.

This script intentionally reuses existing logs and summaries. It does not
launch training. Missing priority experiments are recorded in README and notes.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATE = os.environ.get("V10_DATE") or "20260626"
OUT = ROOT / f"v10_supplement_results_{DATE}"
FIG = OUT / "figures"
RAW = OUT / "raw_run_summaries"


def read_csv(path: str | Path) -> pd.DataFrame:
    p = ROOT / path
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p)


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def write_csv(df: pd.DataFrame, name: str) -> Path:
    path = OUT / name
    df.to_csv(path, index=False)
    return path


def savefig(fig, name: str) -> None:
    import matplotlib.pyplot as plt

    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"{name}.{ext}", bbox_inches="tight", dpi=180)
    plt.close(fig)


def safe_float(x):
    try:
        if pd.isna(x):
            return np.nan
        return float(x)
    except Exception:
        return np.nan


def normalize_task(x: str) -> str:
    x = str(x).lower().strip()
    return {"sst5": "sst-5", "sst-5": "sst-5", "sst2": "sst-2", "sst-2": "sst-2"}.get(x, x)


def h_policy_from_run_name(name: str, h: float | None = None) -> str:
    n = str(name).lower()
    if "fixed_small" in n or "h1e-5" in n:
        return "fixed-small"
    if "mezo_default" in n or "standard_1e-3" in n or (h is not None and abs(h - 1e-3) < 1e-12):
        return "default"
    if "clean" in n or "hstar" in n or "formula" in n or "reference" in n:
        return "reference"
    if "low_1e-6" in n:
        return "low-control"
    return "other"


def collect_high_precision_plateau(run_index: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Collect existing FP32/FP16 plateau rows; no new training is run."""
    dfs = []
    for path in [
        "experiments/main_latest/mezo/roberta-large/sst5/fp32_fp16_h_sweep_11h_seed16_bs64_ckpt1k_20260517/summaries/summary_all.csv",
        "experiments/main_latest/mezo/roberta-large/sst5/fp32_fp16_h_sweep_left_tail_seed16_bs64_ckpt1k_20260519/summaries/merged_fp32_fp16_h_sweep_1e-9_to_1e-2.csv",
    ]:
        df = read_csv(path)
        if not df.empty:
            df["source_path"] = path
            dfs.append(df)
    if not dfs:
        existing = pd.DataFrame()
    else:
        existing = pd.concat(dfs, ignore_index=True, sort=False)
    if not existing.empty:
        for col in ["h", "best_eval_acc", "last_eval_acc", "seed", "data_seed", "steps_completed"]:
            if col in existing:
                existing[col] = pd.to_numeric(existing[col], errors="coerce")
        existing["model"] = "roberta-large"
        existing["task"] = "sst-5"
        existing["run_type"] = np.where(existing.get("steps_completed", 0).fillna(0) >= 20000, "full", "partial")
        existing["h_policy"] = existing["h"].map(lambda h: f"h={h:g}" if pd.notna(h) else "")
        existing["git_commit"] = git_commit()
    # Variance components only valid with multi-seed per h.
    if existing.empty or existing.groupby("h")["seed"].nunique().max() < 2:
        var = pd.DataFrame(
            [
                {
                    "model": "roberta-large",
                    "task": "sst-5",
                    "precision": "fp32/fp16",
                    "status": "insufficient_existing_multiseed_data",
                    "reason": "Existing high-precision h sweeps found are seed16 only; paired seed variance decomposition requires seeds {16,32,64,128,256}.",
                    "var_h_policy": np.nan,
                    "var_seed_direction": np.nan,
                    "var_residual": np.nan,
                    "num_runs": int(len(existing)),
                }
            ]
        )
    else:
        rows = []
        for prec, g in existing.groupby("precision_mode" if "precision_mode" in existing else "precision"):
            tab = g.dropna(subset=["h", "seed", "best_eval_acc"])
            grand = tab["best_eval_acc"].mean()
            h_means = tab.groupby("h")["best_eval_acc"].mean()
            s_means = tab.groupby("seed")["best_eval_acc"].mean()
            rows.append(
                {
                    "precision": prec,
                    "status": "computed_from_existing",
                    "var_h_policy": float(np.var(h_means - grand)),
                    "var_seed_direction": float(np.var(s_means - grand)),
                    "var_residual": float(np.var(tab["best_eval_acc"] - tab["h"].map(h_means) - tab["seed"].map(s_means) + grand)),
                    "num_runs": int(len(tab)),
                }
            )
        var = pd.DataFrame(rows)
    return existing, var


def collect_sparse_prefix_visibility() -> pd.DataFrame:
    rows = []
    sparse_sources = []
    # Multi-task sparse p=0.1 true MSE probes.
    for sub in ["probes_sparse_p0p1_minmse", "probes_sparse_p0p1_refine"]:
        base = ROOT / "outputs/int4_sparsep0p1_probe_minmse_vs_default_2k_20260522_181148" / sub
        for task_dir in sorted(base.glob("*")):
            f = task_dir / "summary.csv"
            if f.exists():
                sparse_sources.append((normalize_task(task_dir.name), f))
    # General sparse probe focused on SST-5.
    for f in [
        ROOT / "outputs/rtnclip_int4_sparse_mezo_nmse_probe_20260522_dirs32/summary.csv",
        ROOT / "outputs/rtnclip_int4_sparse_mezo_nmse_probe_20260522_fixeddirs32/summary.csv",
    ]:
        if f.exists():
            sparse_sources.append(("sst-5", f))
    for task, f in sparse_sources:
        df = pd.read_csv(f)
        for _, r in df.iterrows():
            h = safe_float(r.get("h"))
            rows.append(
                {
                    "model": "roberta-large",
                    "task": task,
                    "mode": "sparse_p0p1",
                    "precision": "int4",
                    "quantizer": "G128_RTNClip",
                    "h": h,
                    "h_policy": policy_for_h(h, None),
                    "true_directional_nmse": first_present(r, ["default_fd_true_nmse", "fd_true_nmse"]),
                    "directional_corr": first_present(r, ["default_corr_fd_true", "corr_fd_true"]),
                    "sign_agreement": np.nan,
                    "active_frac": first_present(r, ["active_frac_mean", "code_change_frac_mean"]),
                    "norm_ratio": first_present(r, ["norm_ratio_mean"]),
                    "visible_direction_cos": first_present(r, ["alignment_mean"]),
                    "corr_g_vh_g_u": np.nan,
                    "n_directions": first_present(r, ["directions", "n_directions"]),
                    "source_path": rel(f),
                    "notes": "Sparse p=0.1 probe; default metric is d_h_minus_gTu where available.",
                }
            )
    # Prefix probes. Existing true-MSE summaries mostly do not include visibility geometry.
    prefix_sources = [
        ("sst-5", ROOT / "outputs/rtnclip_int4_prefix_fd_nmse_bound_20260522_185011/summary.csv", "prefix_int4_old"),
        ("sst-5", ROOT / "outputs/rtnclip_int4_prefix_mezo32_probe64_20260523_144315/summary.csv", "prefix_mezo32_probe64"),
        ("sst-2", ROOT / "outputs/rtnclip_int4base_prefix_fp32_probe64_sst2_h1_to_1em8_20260523_151122/summary.csv", "prefix_fp32_probe64"),
        ("sst-5", ROOT / "outputs/prefix_mezo_fp32_probe64_sst5_h1_to_1em8_20260523_150345/summary.csv", "prefix_fp32_probe64"),
    ]
    for task, f, tag in prefix_sources:
        if not f.exists():
            continue
        df = pd.read_csv(f)
        for _, r in df.iterrows():
            h = safe_float(r.get("h"))
            rows.append(
                {
                    "model": "roberta-large",
                    "task": task,
                    "mode": "prefix_int4" if "int4" in tag else "prefix_fp32_or_mixed",
                    "precision": "int4" if "int4" in tag else "mixed",
                    "quantizer": "G128_RTNClip_or_prefix_path",
                    "h": h,
                    "h_policy": policy_for_h(h, None),
                    "true_directional_nmse": first_present(r, ["fd_true_nmse"]),
                    "directional_corr": first_present(r, ["corr_fd_true"]),
                    "sign_agreement": np.nan,
                    "active_frac": np.nan,
                    "norm_ratio": np.nan,
                    "visible_direction_cos": np.nan,
                    "corr_g_vh_g_u": np.nan,
                    "n_directions": first_present(r, ["n_directions"]),
                    "source_path": rel(f),
                    "notes": f"{tag}; true-MSE/corr available, visibility geometry mostly unavailable in this summary.",
                }
            )
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.drop_duplicates(subset=["task", "mode", "h", "source_path"]).sort_values(["mode", "task", "h", "source_path"])
    return out


def first_present(row, keys):
    for k in keys:
        if k in row and pd.notna(row[k]):
            return row[k]
    return np.nan


def policy_for_h(h: float, ref: float | None) -> str:
    if h is None or pd.isna(h):
        return "unknown"
    if abs(h - 1e-5) <= 1e-12:
        return "fixed-small"
    if abs(h - 1e-3) <= 1e-12:
        return "default"
    if ref is not None and pd.notna(ref) and abs(math.log10(h / ref)) < 0.05:
        return "reference"
    return "grid"


def rel(p: Path) -> str:
    try:
        return str(p.resolve().relative_to(ROOT))
    except Exception:
        return str(p)


def collect_training_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw = read_csv("paper_artifacts_final/data/raw/roberta_int4_all_runs.csv")
    main = read_csv("paper_artifacts_final/data/processed/roberta_int4_multitask_main.csv")
    if raw.empty:
        raw = read_csv("hwindow_final_experiments_bundle_v2/roberta_multitask_appendix_full.csv")
    if main.empty:
        main = read_csv("hwindow_final_experiments_bundle_v2/roberta_multitask_main.csv")
    if not raw.empty:
        raw["task"] = raw["task"].map(normalize_task)
    prefix = raw[raw.get("mode", pd.Series(dtype=str)).astype(str).str.contains("prefix", na=False)].copy() if not raw.empty else pd.DataFrame()
    sparse = raw[raw.get("mode", pd.Series(dtype=str)).astype(str).str.contains("sparse_p0p1", na=False)].copy() if not raw.empty else pd.DataFrame()
    return raw, prefix, sparse


def aggregate_multiseed(df: pd.DataFrame, name: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    cols = ["task", "mode", "raw_h_policy", "h_value", "seed", "run_type", "steps", "best_dev_acc", "final_dev_acc", "source_path", "run_name"]
    present = [c for c in cols if c in df.columns]
    per = df[present].copy()
    if "raw_h_policy" in per:
        per["h_policy"] = per["raw_h_policy"].map(lambda x: h_policy_from_run_name(x))
    elif "run_name" in per:
        per["h_policy"] = per["run_name"].map(lambda x: h_policy_from_run_name(x))
    else:
        per["h_policy"] = "unknown"
    per.to_csv(OUT / f"{name}_per_run.csv", index=False)
    grouped = per.groupby(["task", "mode", "h_policy", "h_value"], dropna=False).agg(
        seeds=("seed", lambda x: ",".join(map(lambda y: str(int(y)) if pd.notna(y) else "", sorted(set(x))))),
        n_seeds=("seed", "nunique"),
        run_types=("run_type", lambda x: ",".join(sorted(set(map(str, x))))),
        best_eval_acc_mean=("best_dev_acc", "mean"),
        best_eval_acc_std=("best_dev_acc", "std"),
        final_eval_acc_mean=("final_dev_acc", "mean"),
        final_eval_acc_std=("final_dev_acc", "std"),
        n_runs=("best_dev_acc", "size"),
    ).reset_index()
    return grouped


def probe_update_mismatch() -> pd.DataFrame:
    rows = []
    # Canonical SST-5 true-MSE contains INT8/INT4 dense visibility diagnostics.
    canon = read_csv("paper_artifacts_final/data/processed/sst5_true_directional_mse.csv")
    if not canon.empty:
        for _, r in canon.iterrows():
            rows.append(
                {
                    "model": r.get("model"),
                    "task": r.get("task"),
                    "precision": r.get("precision"),
                    "mode": r.get("mode"),
                    "quantizer": r.get("quantizer"),
                    "h": r.get("h"),
                    "active_frac": r.get("crossing_active_fraction"),
                    "norm_ratio": r.get("displacement_norm_ratio"),
                    "cos_vh_u": r.get("displacement_alignment"),
                    "jump_distribution": "",
                    "d_Q_scale_or_variance": np.nan,
                    "directional_nmse": r.get("true_directional_nmse"),
                    "directional_corr": r.get("directional_correlation"),
                    "source_path": r.get("source_path"),
                }
            )
    sp = collect_sparse_prefix_visibility()
    for _, r in sp.iterrows():
        if r.get("task") == "sst-5" and str(r.get("mode")).startswith("sparse"):
            rows.append(
                {
                    "model": r.get("model"),
                    "task": r.get("task"),
                    "precision": r.get("precision"),
                    "mode": r.get("mode"),
                    "quantizer": r.get("quantizer"),
                    "h": r.get("h"),
                    "active_frac": r.get("active_frac"),
                    "norm_ratio": r.get("norm_ratio"),
                    "cos_vh_u": r.get("visible_direction_cos"),
                    "jump_distribution": "",
                    "d_Q_scale_or_variance": np.nan,
                    "directional_nmse": r.get("true_directional_nmse"),
                    "directional_corr": r.get("directional_corr"),
                    "source_path": r.get("source_path"),
                }
            )
    return pd.DataFrame(rows).drop_duplicates()


def audit_tasks(raw: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if raw.empty:
        return pd.DataFrame()
    for mode, task in [("prefix", "rte"), ("sparse_p0p1", "trec")]:
        sub = raw[(raw["mode"].astype(str).str.contains(mode, na=False)) & (raw["task"].map(normalize_task) == task)]
        policies = sorted(set(sub.get("raw_h_policy", pd.Series(dtype=str)).dropna().astype(str)))
        full = sub[sub.get("run_type", "") == "full"] if "run_type" in sub else pd.DataFrame()
        rows.append(
            {
                "audit_item": f"{mode}_{task}_completeness",
                "num_rows": int(len(sub)),
                "num_full_rows": int(len(full)),
                "policies_found": ";".join(policies),
                "comparable_for_main_table": bool(len(full) >= 3 and any("mezo_default" in p for p in policies) and any("fixed_small" in p for p in policies)),
                "recommendation": "include only if all fixed/default/reference rows are same family and full" if len(full) >= 3 else "omit from main stress-test table or keep appendix/audit only",
            }
        )
    # Family mixing check.
    for (task, mode), sub in raw.groupby(["task", "mode"], dropna=False):
        sources = sorted(set(map(str, sub.get("source_path", pd.Series(dtype=str)).dropna())))
        rows.append(
            {
                "audit_item": f"family_sources_{mode}_{task}",
                "num_rows": int(len(sub)),
                "num_full_rows": int((sub.get("run_type", pd.Series(dtype=str)) == "full").sum()),
                "policies_found": ";".join(sorted(set(map(str, sub.get("raw_h_policy", pd.Series(dtype=str)).dropna())))),
                "comparable_for_main_table": len(sources) == 1,
                "recommendation": f"sources={';'.join(sources[:4])}" + (";..." if len(sources) > 4 else ""),
            }
        )
    return pd.DataFrame(rows)


def copy_raw_summaries(raw: pd.DataFrame) -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    copied = []
    for sp in sorted(set(map(str, raw.get("source_path", pd.Series(dtype=str)).dropna()))):
        p = ROOT / sp
        if p.exists():
            target = RAW / sp.replace("/", "__")
            shutil.copy2(p, target)
            copied.append({"source_path": sp, "copied_to": str(target.relative_to(OUT))})
    # Also copy per-run JSON summaries/configs when the run index can map
    # paper-table rows back to run directories.
    run_index = read_csv("all_experiment_results_package_20260626/indices/run_index.csv")
    if not run_index.empty and "run_name" in raw.columns:
        wanted = set(map(str, raw["run_name"].dropna()))
        idx = run_index[run_index["run_name"].astype(str).isin(wanted)].copy()
        for _, row in idx.iterrows():
            run_dir = str(row.get("run_dir", ""))
            for filename in ["run_summary.json", "run_config.json", "run_manifest_row.json", "metrics.csv", "eval_metrics.jsonl"]:
                candidates = [
                    ROOT / run_dir / filename,
                    ROOT / "all_experiment_results_package_20260626" / "artifacts" / run_dir / filename,
                ]
                for p in candidates:
                    if p.exists():
                        target = RAW / f"{run_dir}__{filename}".replace("/", "__")
                        shutil.copy2(p, target)
                        copied.append({"source_path": rel(p), "copied_to": str(target.relative_to(OUT))})
                        break
    pd.DataFrame(copied).to_csv(OUT / "raw_summary_copies.csv", index=False)


def make_figures(high: pd.DataFrame, var: pd.DataFrame, vis: pd.DataFrame, prefix_agg: pd.DataFrame, sparse_agg: pd.DataFrame, mismatch: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt

    FIG.mkdir(parents=True, exist_ok=True)
    if not high.empty and {"h", "best_eval_acc"}.issubset(high.columns):
        fig, ax = plt.subplots(figsize=(6, 4))
        for key, g in high.dropna(subset=["h", "best_eval_acc"]).groupby(high.get("precision_mode", high.get("precision", "precision")).astype(str)):
            ax.plot(g["h"], g["best_eval_acc"], marker="o", label=key)
        ax.set_xscale("log")
        ax.set_xlabel("h")
        ax.set_ylabel("best eval accuracy")
        ax.set_title("Existing high-precision RoBERTa/SST-5 h sweep (mostly seed16)")
        ax.legend()
        savefig(fig, "priority1_existing_accuracy_vs_h")

    if not var.empty:
        fig, ax = plt.subplots(figsize=(6, 3.5))
        vals = [
            safe_float(var.iloc[0].get("var_h_policy")),
            safe_float(var.iloc[0].get("var_seed_direction")),
            safe_float(var.iloc[0].get("var_residual")),
        ]
        ax.bar(["h policy", "seed/direction", "residual"], [0 if np.isnan(v) else v for v in vals])
        ax.set_ylabel("variance component")
        ax.set_title(str(var.iloc[0].get("status", "")))
        savefig(fig, "priority1_variance_components")

    if not vis.empty:
        plot = vis[vis["h_policy"].isin(["fixed-small", "default", "reference", "grid"])].copy()
        for metric in ["true_directional_nmse", "directional_corr", "active_frac", "norm_ratio", "visible_direction_cos"]:
            if metric not in plot or plot[metric].notna().sum() == 0:
                continue
            fig, ax = plt.subplots(figsize=(7, 4))
            for (mode, task), g in plot.dropna(subset=["h", metric]).groupby(["mode", "task"]):
                ax.plot(g["h"], g[metric], marker="o", label=f"{mode}:{task}")
            ax.set_xscale("log")
            ax.set_xlabel("h")
            ax.set_ylabel(metric)
            ax.set_title(f"Sparse/prefix probe diagnostic: {metric}")
            ax.legend(fontsize=7, ncol=2)
            savefig(fig, f"priority2_sparse_prefix_{metric}")

    for name, agg in [("prefix", prefix_agg), ("sparse", sparse_agg)]:
        if agg.empty:
            continue
        fig, ax = plt.subplots(figsize=(8, 4))
        for pol, g in agg.groupby("h_policy"):
            ax.scatter(g["task"].astype(str), g["best_eval_acc_mean"], label=pol)
            for _, r in g.iterrows():
                if pd.notna(r.get("best_eval_acc_std")):
                    ax.errorbar(str(r["task"]), r["best_eval_acc_mean"], yerr=r["best_eval_acc_std"], fmt="none", alpha=0.4)
        ax.set_ylabel("best eval accuracy mean")
        ax.set_title(f"Existing INT4 {name} policy results")
        ax.legend(fontsize=8)
        savefig(fig, f"priority_training_{name}_policy_accuracy")

    if not mismatch.empty:
        for metric in ["active_frac", "cos_vh_u", "norm_ratio", "directional_nmse", "directional_corr"]:
            if metric not in mismatch or mismatch[metric].notna().sum() == 0:
                continue
            fig, ax = plt.subplots(figsize=(7, 4))
            for (precision, mode), g in mismatch.dropna(subset=["h", metric]).groupby(["precision", "mode"]):
                ax.plot(g["h"], g[metric], marker="o", label=f"{precision}:{mode}")
            ax.set_xscale("log")
            ax.set_xlabel("h")
            ax.set_ylabel(metric)
            ax.set_title(f"Probe/update mismatch diagnostic: {metric}")
            ax.legend(fontsize=8)
            savefig(fig, f"priority5_mismatch_{metric}")


def write_notes(high, var, vis, raw, prefix_agg, sparse_agg, audit, mismatch):
    readme = f"""# V10 Supplement Results ({DATE})

Generated from existing MeZO / ZO perturbation-radius artifacts.

Git commit at build time: `{git_commit()}`

This folder is an audit-and-aggregation supplement. It does not launch new
training. Missing multi-seed experiments are explicitly marked as missing or
insufficient rather than fabricated.

## Contents

- `priority1_high_precision_existing_runs.csv`
- `priority1_variance_components.csv`
- `sparse_prefix_true_mse_visibility.csv`
- `prefix_int4_multiseed_per_run.csv`
- `prefix_int4_multiseed_aggregate.csv`
- `sparse_int4_multiseed_per_run.csv`
- `sparse_int4_multiseed_aggregate.csv`
- `probe_update_mismatch_diagnostics.csv`
- `audit_prefix_rte_sparse_trec.csv`
- `v10_table_values_audit.csv`
- `figures/*.png` and `figures/*.pdf`
- `raw_run_summaries/` copied source summary CSVs
- `paper_update_notes.md`

## Priority status

1. High-precision plateau multi-seed variance decomposition:
   existing h-sweeps are mostly seed16 only. The required paired seeds
   {{16,32,64,128,256}} were not found, so variance decomposition is marked
   insufficient.

2. Sparse/prefix INT4 probe diagnostics:
   sparse p=0.1 true-MSE/visibility data were found for several tasks from
   existing probe summaries. Prefix true-MSE/correlation summaries were found,
   but many prefix summaries do not include active fraction/norm-ratio geometry.

3. Prefix INT4 multi-seed confirmation:
   existing paper-facing table data are primarily seed16. Aggregates are
   emitted, but n_seeds indicates whether a row is genuinely multi-seed.

4. Sparse INT4 multi-seed confirmation:
   existing paper-facing table data are primarily seed16. Aggregates are
   emitted, but n_seeds indicates whether a row is genuinely multi-seed.

5. Probe/update mismatch diagnostic:
   canonical SST-5 dense INT8/INT4 diagnostics and available sparse p=0.1
   diagnostics are aggregated. Prefix mismatch geometry is incomplete.

## Important caveats

- Do not interpret missing nMSE/visibility fields as zero.
- Do not mix old highest-abs sparse mask runs with seed-fixed/task-gradient
  sparse runs in the main table.
- Do not use residual-grid or QZO/QES-like runs as mainline optimizer claims.
- Prefix RTE and sparse TREC are audited separately for comparability.
"""
    (OUT / "README.md").write_text(readme)

    notes = f"""# Paper Update Notes for V10

## Central interpretation

The aggregated results support the V10 framing that h is a finite-difference
probe guardrail. Inside a safe region, h-policy differences are not reliably an
accuracy knob. Low precision, sparse, and prefix settings are stress tests.

## Priority 1

The existing high-precision RoBERTa/SST-5 h sweeps are useful for showing a
plateau, but not enough for the requested paired multi-seed variance
decomposition. `priority1_variance_components.csv` records this as
`insufficient_existing_multiseed_data`.

Paper use: state as a planned/needed supplement unless new paired seeds are
run.

## Priority 2

`sparse_prefix_true_mse_visibility.csv` aggregates probe-level evidence.
Sparse p=0.1 includes true directional nMSE/correlation and visibility
diagnostics. Prefix summaries include true directional nMSE/correlation where
available but often lack active fraction and norm-ratio.

Paper use: sparse p=0.1 can support default-safe visibility claims; prefix
default-failure needs either training evidence or additional geometry probes if
active/norm diagnostics are required.

## Priorities 3 and 4

Existing seed-fixed INT4 sparse/prefix tables are mostly seed16. The aggregate
CSVs preserve `n_seeds`, so single-seed rows are clearly visible.

Paper use: do not call these multi-seed confirmations unless `n_seeds >= 3`.

## Priority 5

`probe_update_mismatch_diagnostics.csv` combines canonical dense INT8/INT4
SST-5 mismatch diagnostics with available sparse p=0.1 diagnostics.

Paper use: supports the low-precision boundary-case section. Prefix mismatch
needs an additional probe if the paper needs active fraction and per-coordinate
jump distributions for prefix.

## Audit tasks

- Prefix RTE: see `audit_prefix_rte_sparse_trec.csv`. Include only if same
  family/full fixed/default/reference rows exist.
- Sparse TREC: see `audit_prefix_rte_sparse_trec.csv`. In prior final artifacts
  sparse TREC often appears as medium/incomplete; keep appendix unless complete.
- V10 table value consistency: see `v10_table_values_audit.csv`.
- Residual-grid/QES-like runs: keep as update-side diagnostics only.
"""
    (OUT / "paper_update_notes.md").write_text(notes)

    missing = []
    missing.append({
        "priority": 1,
        "missing_item": "paired high-precision multiseed plateau full runs",
        "details": "Need RoBERTa-large/SST-5 FP32 or FP16/BF16 h={1e-5,1e-4,1e-3,3e-3} with seeds {16,32,64,128,256}; existing sweep is mostly seed16.",
    })
    missing.append({
        "priority": 2,
        "missing_item": "prefix INT4 visibility geometry",
        "details": "Existing prefix true-MSE summaries generally lack active fraction, norm ratio, visible-direction cosine, and per-coordinate jump distribution.",
    })
    missing.append({
        "priority": 3,
        "missing_item": "prefix INT4 multiseed training confirmation",
        "details": "Existing paper-facing prefix rows are mostly single-seed seed16.",
    })
    missing.append({
        "priority": 4,
        "missing_item": "sparse INT4 multiseed training confirmation",
        "details": "Existing paper-facing sparse rows are mostly single-seed seed16.",
    })
    pd.DataFrame(missing).to_csv(OUT / "missing_for_full_v10_supplement.csv", index=False)


def main():
    OUT.mkdir(exist_ok=True)
    FIG.mkdir(exist_ok=True)
    run_index = read_csv("all_experiment_results_package_20260626/indices/run_index.csv")
    if run_index.empty:
        run_index = pd.DataFrame()
    high, var = collect_high_precision_plateau(run_index)
    vis = collect_sparse_prefix_visibility()
    raw, prefix, sparse = collect_training_tables()
    prefix_agg = aggregate_multiseed(prefix, "prefix_int4_multiseed")
    sparse_agg = aggregate_multiseed(sparse, "sparse_int4_multiseed")
    audit = audit_tasks(raw)
    mismatch = probe_update_mismatch()

    write_csv(high, "priority1_high_precision_existing_runs.csv")
    write_csv(var, "priority1_variance_components.csv")
    write_csv(vis, "sparse_prefix_true_mse_visibility.csv")
    write_csv(prefix_agg, "prefix_int4_multiseed_aggregate.csv")
    write_csv(sparse_agg, "sparse_int4_multiseed_aggregate.csv")
    write_csv(mismatch, "probe_update_mismatch_diagnostics.csv")
    write_csv(audit, "audit_prefix_rte_sparse_trec.csv")
    write_csv(raw, "v10_table_values_audit.csv")

    copy_raw_summaries(raw)
    make_figures(high, var, vis, prefix_agg, sparse_agg, mismatch)
    write_notes(high, var, vis, raw, prefix_agg, sparse_agg, audit, mismatch)

    meta = {
        "date": DATE,
        "repo_root": str(ROOT),
        "git_commit": git_commit(),
        "generated_by": "tools/build_v10_supplement_results.py",
        "no_new_training_launched": True,
        "outputs": sorted(p.name for p in OUT.iterdir()),
    }
    (OUT / "metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
