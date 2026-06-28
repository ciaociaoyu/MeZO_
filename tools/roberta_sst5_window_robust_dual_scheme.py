#!/usr/bin/env python3
"""Build robust dual-scheme RoBERTa/SST-5 H-window artifacts.

This is a probe-only post-processor. It reuses the saved probe metrics from
`roberta_sst5_theoretical_windows_20260627/` and existing SST-5 h-sweep
accuracy CSVs. It does not train and it does not reload the RoBERTa model.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DEFAULT = ROOT / "roberta_sst5_theoretical_windows_20260627"
DATE_DEFAULT = os.environ.get("HWINDOW_DATE", "20260628")
TAUS_PLUGIN = [1.0, 5.0, 10.0]
TAUS_FIT = [1.0, 5.0, 10.0]
DEFAULT_H = 1e-3
TINY_H = 1e-5
ZERO_TOL = 1e-12


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def safe_float(x, default=np.nan) -> float:
    try:
        y = float(x)
    except Exception:
        return default
    return y if math.isfinite(y) else default


def fmt_interval(low, high) -> str:
    if pd.isna(low) or pd.isna(high):
        return ""
    return f"[{float(low):.6g}, {float(high):.6g}]"


def interval_contains(low, high, h: float):
    if pd.isna(low) or pd.isna(high):
        return None
    return bool(float(low) <= h <= float(high))


def truthy(value) -> bool:
    if value is None:
        return False
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y"}
    if pd.isna(value):
        return False
    return bool(value)


def solve_window(A: float, B: float, tau: float):
    if not (math.isfinite(A) and math.isfinite(B) and A > 0 and B > 0):
        return np.nan, np.nan, np.nan, "invalid_coefficients"
    rho_min = 2.0 * math.sqrt(A * B)
    disc = tau * tau - 4.0 * A * B
    if disc < 0 or rho_min > tau:
        return rho_min, np.nan, np.nan, "no_window"
    x_low = (tau - math.sqrt(max(disc, 0.0))) / (2.0 * B)
    x_high = (tau + math.sqrt(max(disc, 0.0))) / (2.0 * B)
    return rho_min, math.sqrt(max(x_low, 0.0)), math.sqrt(max(x_high, 0.0)), "window"


def read_required(source: Path, name: str) -> pd.DataFrame:
    path = source / name
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def copy_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def empirical_windows(source: Path, out: Path) -> pd.DataFrame:
    src = read_required(source, "empirical_accuracy_good_sets.csv").copy()
    rows = []
    for _, row in src.iterrows():
        low = safe_float(row.get("h_good_low"))
        high = safe_float(row.get("h_good_high"))
        rows.append(
            {
                **row.to_dict(),
                "default_h": DEFAULT_H,
                "tiny_h": TINY_H,
                "default_in_empirical_window": interval_contains(low, high, DEFAULT_H),
                "tiny_in_empirical_window": interval_contains(low, high, TINY_H),
                "status": "ok" if not pd.isna(low) and not pd.isna(high) else "missing_accuracy",
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(out / "empirical_accuracy_windows.csv", index=False)
    return df


def add_zero_fraction(summary: pd.DataFrame, raw: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (precision, h), group in raw.groupby(["precision", "h"], dropna=False):
        dh = group["d_h"].astype(float).to_numpy()
        rows.append(
            {
                "precision": precision,
                "h": float(h),
                "dh_zero_fraction": float(np.mean(np.abs(dh) <= ZERO_TOL)) if len(dh) else np.nan,
            }
        )
    z = pd.DataFrame(rows)
    merged = summary.merge(z, on=["precision", "h"], how="left")
    return merged


def write_raw_and_summary(source: Path, out: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = read_required(source, "raw_probe_metrics.csv")
    summary = read_required(source, "probe_summary_by_h.csv")
    summary = add_zero_fraction(summary, raw)
    raw.to_csv(out / "raw_probe_metrics.csv", index=False)
    summary.to_csv(out / "probe_summary_by_h.csv", index=False)
    return raw, summary


def practical_status_for_row(row, primary: bool) -> bool:
    corr_thr = 0.90 if primary else 0.80
    nmse_thr = 0.10 if primary else 0.20
    sign_thr = 0.80 if primary else 0.70
    zero_thr = 0.95 if primary else 0.99
    return (
        safe_float(row.get("directional_corr")) >= corr_thr
        and safe_float(row.get("scalar_nmse")) <= nmse_thr
        and safe_float(row.get("sign_agreement")) >= sign_thr
        and safe_float(row.get("dh_zero_fraction"), 1.0) < zero_thr
        and safe_float(row.get("d_h_std"), 0.0) > 0.0
    )


def practical_windows(summary: pd.DataFrame, out: Path) -> pd.DataFrame:
    rows = []
    for precision, group in summary.groupby("precision", sort=False):
        group = group.sort_values("h")
        primary = group[group.apply(lambda r: practical_status_for_row(r, True), axis=1)]
        relaxed = group[group.apply(lambda r: practical_status_for_row(r, False), axis=1)]
        first_nonzero = group[group["dh_zero_fraction"].fillna(1.0) < 1.0]

        def bounds(g):
            if g.empty:
                return np.nan, np.nan, ""
            vals = sorted(float(x) for x in g["h"].tolist())
            return vals[0], vals[-1], " ".join(f"{x:.8g}" for x in vals)

        p_low, p_high, p_vals = bounds(primary)
        r_low, r_high, r_vals = bounds(relaxed)
        nz_low, _, _ = bounds(first_nonzero)
        status = "primary_available" if not primary.empty else ("relaxed_only" if not relaxed.empty else "no_practical_probe_visible_window")
        rows.append(
            {
                "precision": precision,
                "checkpoint_id": group["checkpoint_id"].iloc[0],
                "primary_status": "available" if not primary.empty else "unavailable",
                "h_left_primary": p_low,
                "h_right_primary": p_high,
                "primary_h_values": p_vals,
                "relaxed_status": "available" if not relaxed.empty else "unavailable",
                "h_left_relaxed": r_low,
                "h_right_relaxed": r_high,
                "relaxed_h_values": r_vals,
                "first_nonzero_dh_h": nz_low,
                "default_h": DEFAULT_H,
                "tiny_h": TINY_H,
                "default_in_practical_primary": interval_contains(p_low, p_high, DEFAULT_H),
                "default_in_practical_relaxed": interval_contains(r_low, r_high, DEFAULT_H),
                "tiny_in_practical_primary": interval_contains(p_low, p_high, TINY_H),
                "status": status,
                "thresholds_primary": "corr>=0.90, scalar_nmse<=0.10, sign>=0.80, dh_zero_fraction<0.95, dh_std>0",
                "thresholds_relaxed": "corr>=0.80, scalar_nmse<=0.20, sign>=0.70, dh_zero_fraction<0.99, dh_std>0",
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(out / "practical_probe_windows.csv", index=False)
    return df


def curvature_table(raw: pd.DataFrame) -> pd.DataFrame:
    rows = []
    temp = raw.copy()
    temp["kappa"] = (temp["loss_plus"] - 2.0 * temp["loss_base"] + temp["loss_minus"]).abs() / (
        temp["h"].astype(float) ** 2 * temp["norm_u2"].astype(float)
    )
    for precision, group in temp.groupby("precision"):
        # Use middle h values to avoid floating-point dead zones and large-h locality blowups.
        safe = group[(group["h"].astype(float) >= 1e-4) & (group["h"].astype(float) <= 3e-3)]
        if safe.empty:
            safe = group[np.isfinite(group["kappa"])]
        k = safe["kappa"].replace([np.inf, -np.inf], np.nan).dropna()
        rows.append(
            {
                "precision": precision,
                "curvature_h_low": float(safe["h"].min()) if not safe.empty else np.nan,
                "curvature_h_high": float(safe["h"].max()) if not safe.empty else np.nan,
                "n_curvature_samples": int(len(k)),
                "L_loc_median": float(k.median()) if len(k) else np.nan,
                "L_loc_p90": float(k.quantile(0.90)) if len(k) else np.nan,
                "L_loc_p95": float(k.quantile(0.95)) if len(k) else np.nan,
                "L_loc_status": "ok" if len(k) else "unavailable",
            }
        )
    return pd.DataFrame(rows)


def delta_estimates(source: Path, summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for precision, group in summary.groupby("precision"):
        source_kind = "unavailable"
        delta = np.nan
        notes = ""
        if precision in {"int8", "int4"}:
            qpath = source / f"quantizer_summary_{precision}.json"
            if qpath.exists():
                q = json.loads(qpath.read_text())
                delta = safe_float(q.get("scale_median_weighted"))
                source_kind = "quantizer_scale_median_weighted"
                notes = f"scale_min={q.get('scale_min_global')}; scale_max={q.get('scale_max_global')}; saturation={q.get('saturation_frac')}"
        else:
            # Numeric visibility estimate. Use the first h where finite differences
            # are not all zero and scalar nMSE/correlation indicate a usable signal.
            usable = group[
                (group["dh_zero_fraction"].fillna(1.0) < 1.0)
                & (group["d_h_std"].fillna(0.0) > 0.0)
                & (group["scalar_nmse"].fillna(np.inf) < 1.0)
            ].sort_values("h")
            if not usable.empty:
                h_vis = float(usable["h"].iloc[0])
                delta = 2.0 * h_vis
                source_kind = "numeric_visibility_transition_delta_approx_2h"
                notes = f"h_vis={h_vis:.8g}; FP precision estimate, not hardware ulp"
        rows.append({"precision": precision, "Delta_eff": delta, "Delta_eff_source": source_kind, "Delta_notes": notes})
    return pd.DataFrame(rows)


def pure_theory_plugin(raw: pd.DataFrame, summary: pd.DataFrame, source: Path, out: Path) -> pd.DataFrame:
    curv = curvature_table(raw)
    delta = delta_estimates(source, summary)
    params = (
        summary.groupby("precision")
        .agg(checkpoint_id=("checkpoint_id", "first"), G=("G", "first"), G2=("G2", "first"), d=("d", "first"))
        .reset_index()
    )
    base = params.merge(delta, on="precision", how="left").merge(curv, on="precision", how="left")
    rows = []
    for _, r in base.iterrows():
        precision = r["precision"]
        d = safe_float(r["d"])
        G2 = safe_float(r["G2"])
        Delta = safe_float(r["Delta_eff"])
        L = safe_float(r["L_loc_p90"])
        status = "ok"
        notes = []
        if pd.isna(Delta) or Delta <= 0:
            status = "no_delta_eff"
            notes.append("Delta_eff unavailable")
        if pd.isna(L) or L <= 0:
            status = "no_L_loc" if status == "ok" else status + "+no_L_loc"
            notes.append("L_loc unavailable")
        if pd.isna(d) or pd.isna(G2) or G2 <= 0:
            status = "bad_G_or_d"
            notes.append("G2/d unavailable")

        if status == "ok":
            c_d = (d + 4.0) / (d + 1.0)
            A = c_d * Delta * Delta / 4.0
            B = c_d * 4.0 * L * L * d * (d + 2.0) / G2
            h_ref = (A / B) ** 0.25 if A > 0 and B > 0 else np.nan
        else:
            A = B = h_ref = np.nan

        for tau in TAUS_PLUGIN:
            rho_min, h_low, h_high, wstatus = solve_window(A, B, tau) if status == "ok" else (np.nan, np.nan, np.nan, status)
            rows.append(
                {
                    **r.to_dict(),
                    "L_loc": L,
                    "L_loc_source": "directional_curvature_p90_middle_h",
                    "A_plugin": A,
                    "B_plugin": B,
                    "h_ref_plugin": h_ref,
                    "tau": tau,
                    "rho_min_plugin": rho_min,
                    "h_low": h_low,
                    "h_high": h_high,
                    "default_h": DEFAULT_H,
                    "tiny_h": TINY_H,
                    "default_in_window": interval_contains(h_low, h_high, DEFAULT_H),
                    "tiny_in_window": interval_contains(h_low, h_high, TINY_H),
                    "status": wstatus if status == "ok" else status,
                    "notes": "; ".join(notes) if notes else ("conservative plug-in from direct Delta/L estimates" if wstatus == "window" else "rho_min exceeds tau"),
                }
            )
    df = pd.DataFrame(rows)
    df.to_csv(out / "pure_theory_plugin_windows.csv", index=False)
    curv.to_csv(out / "configs" / "directional_curvature_estimates.csv", index=False)
    delta.to_csv(out / "configs" / "delta_eff_estimates.csv", index=False)
    return df


def rho_fit_windows(source: Path, out: Path) -> pd.DataFrame:
    old = read_required(source, "fitted_windows.csv")
    # Take one row per precision for the primary fit coefficients, then recompute
    # the requested tau values.
    prim = (
        old[old["fit_method"].astype(str).eq("dep_log_soft_l1")]
        .sort_values(["precision", "tau"])
        .groupby("precision", as_index=False)
        .first()
    )
    rows = []
    for _, r in prim.iterrows():
        A = safe_float(r.get("A"))
        B = safe_float(r.get("B"))
        stable = str(r.get("fit_stability_flag")) == "stable"
        h_ref = (A / B) ** 0.25 if stable and A > 0 and B > 0 else np.nan
        for tau in TAUS_FIT:
            rho_min, h_low, h_high, wstatus = solve_window(A, B, tau) if stable else (np.nan, np.nan, np.nan, str(r.get("fit_stability_flag")))
            rows.append(
                {
                    "precision": r["precision"],
                    "checkpoint_id": r.get("checkpoint_id"),
                    "fit_method": "dep_log_soft_l1_reused",
                    "A": A,
                    "B": B,
                    "C": safe_float(r.get("C")),
                    "h_ref_fit": h_ref,
                    "tau": tau,
                    "rho_min_dep": rho_min,
                    "h_low": h_low,
                    "h_high": h_high,
                    "default_h": DEFAULT_H,
                    "tiny_h": TINY_H,
                    "default_in_window": interval_contains(h_low, h_high, DEFAULT_H),
                    "tiny_in_window": interval_contains(h_low, h_high, TINY_H),
                    "fit_quality_r2_or_log_error": r.get("fit_quality_r2_or_log_error"),
                    "fit_stability_flag": r.get("fit_stability_flag"),
                    "status": wstatus if stable else "no_stable_smooth_fit",
                    "notes": "" if stable else f"fit not used: {r.get('notes')}",
                }
            )
    df = pd.DataFrame(rows)
    df.to_csv(out / "rho_fit_windows.csv", index=False)
    return df


def row_for_threshold(emp: pd.DataFrame, precision: str, threshold: str) -> dict:
    m = emp[(emp["precision"] == precision) & (emp["threshold_type"] == threshold)]
    return m.iloc[0].to_dict() if not m.empty else {}


def window_row(df: pd.DataFrame, precision: str, tau: float, status_col="status") -> dict:
    if df.empty:
        return {}
    m = df[(df["precision"] == precision) & (df["tau"].astype(float) == float(tau))]
    return m.iloc[0].to_dict() if not m.empty else {}


def practical_row(df: pd.DataFrame, precision: str) -> dict:
    m = df[df["precision"] == precision]
    return m.iloc[0].to_dict() if not m.empty else {}


def final_message(emp_row, pure_w1, prac, fit_w1) -> str:
    default_emp = emp_row.get("default_in_empirical_window")
    default_prac = prac.get("default_in_practical_primary")
    default_prac_rel = prac.get("default_in_practical_relaxed")
    pure_status = str(pure_w1.get("status", ""))
    fit_status = str(fit_w1.get("status", ""))
    if truthy(pure_w1.get("default_in_window")) and pure_status == "window":
        return "certified default-safe"
    if truthy(default_prac) or truthy(default_prac_rel):
        return "broad default-safe" if truthy(default_emp) else "practical window but no smooth fit"
    if truthy(default_emp):
        return "empirical plateau, no theory certificate"
    if pure_status in {"no_delta_eff", "no_L_loc", "bad_G_or_d"} and fit_status == "no_stable_smooth_fit":
        return "insufficient data"
    if str(prac.get("status", "")).startswith("no_practical"):
        return "boundary/no certificate"
    return "default-risk"


def comparison(emp: pd.DataFrame, pure: pd.DataFrame, practical: pd.DataFrame, fit: pd.DataFrame, out: Path) -> pd.DataFrame:
    precisions = ["fp32", "fp16", "int8", "int4"]
    rows = []
    for p in precisions:
        e = row_for_threshold(emp, p, "best_acc_minus_0.01")
        pw1 = window_row(pure, p, 1.0)
        pw5 = window_row(pure, p, 5.0)
        pw10 = window_row(pure, p, 10.0)
        pr = practical_row(practical, p)
        fw1 = window_row(fit, p, 1.0)
        fw5 = window_row(fit, p, 5.0)
        fw10 = window_row(fit, p, 10.0)
        rows.append(
            {
                "precision": p,
                "checkpoint_id": pr.get("checkpoint_id") or fw1.get("checkpoint_id") or pw1.get("checkpoint_id"),
                "empirical_good_primary": fmt_interval(e.get("h_good_low", np.nan), e.get("h_good_high", np.nan)),
                "default_in_empirical_primary": e.get("default_in_empirical_window"),
                "tiny_1e5_in_empirical_primary": e.get("tiny_in_empirical_window"),
                "pure_theory_status": pw1.get("status", ""),
                "pure_h_ref": pw1.get("h_ref_plugin", np.nan),
                "pure_W1": fmt_interval(pw1.get("h_low", np.nan), pw1.get("h_high", np.nan)),
                "pure_W5": fmt_interval(pw5.get("h_low", np.nan), pw5.get("h_high", np.nan)),
                "pure_W10": fmt_interval(pw10.get("h_low", np.nan), pw10.get("h_high", np.nan)),
                "default_in_pure_W1": pw1.get("default_in_window"),
                "default_in_pure_W5": pw5.get("default_in_window"),
                "tiny_1e5_in_pure_W1": pw1.get("tiny_in_window"),
                "practical_status": pr.get("status", ""),
                "practical_W_primary": fmt_interval(pr.get("h_left_primary", np.nan), pr.get("h_right_primary", np.nan)),
                "practical_W_relaxed": fmt_interval(pr.get("h_left_relaxed", np.nan), pr.get("h_right_relaxed", np.nan)),
                "default_in_practical_primary": pr.get("default_in_practical_primary"),
                "default_in_practical_relaxed": pr.get("default_in_practical_relaxed"),
                "tiny_1e5_in_practical_primary": pr.get("tiny_in_practical_primary"),
                "rho_fit_status": fw1.get("status", ""),
                "rho_fit_h_ref": fw1.get("h_ref_fit", np.nan),
                "rho_fit_W1": fmt_interval(fw1.get("h_low", np.nan), fw1.get("h_high", np.nan)),
                "rho_fit_W5": fmt_interval(fw5.get("h_low", np.nan), fw5.get("h_high", np.nan)),
                "rho_fit_W10": fmt_interval(fw10.get("h_low", np.nan), fw10.get("h_high", np.nan)),
                "final_message": final_message(e, pw1, pr, fw1),
                "notes": "",
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(out / "window_comparison_for_paper.csv", index=False)
    return df


def discovery_report(source: Path, out: Path) -> None:
    ckpt = pd.read_csv(source / "checkpoint_manifest.csv") if (source / "checkpoint_manifest.csv").exists() else pd.DataFrame()
    acc_src = pd.read_csv(source / "accuracy_sweep_points.csv") if (source / "accuracy_sweep_points.csv").exists() else pd.DataFrame()
    lines = [
        "# Discovery Report",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        f"Git commit: `{git_commit()}`",
        "",
        "## Source",
        "",
        f"- Reused probe-only source directory: `{source}`",
        "- No new training was run.",
        "- No model reload was required for this robust dual-scheme pass; it reuses saved raw probe metrics.",
        "",
        "## Checkpoints Found/Selected",
        "",
    ]
    if ckpt.empty:
        lines.append("- No checkpoint manifest found.")
    else:
        for _, r in ckpt.iterrows():
            lines.append(f"- {r.get('precision')}: `{r.get('checkpoint_path')}` ({r.get('source_run_id')})")
    lines += [
        "",
        "## Accuracy Sweep Data",
        "",
    ]
    if acc_src.empty:
        lines.append("- No accuracy sweep points found.")
    else:
        for p, g in acc_src.groupby("precision"):
            srcs = sorted(set(str(x) for x in g["source_path"]))
            lines.append(f"- {p}: {len(g)} h points; sources: " + "; ".join(f"`{s}`" for s in srcs))
    lines += [
        "",
        "## Precision Modes",
        "",
        "- FP32 and FP16: high-precision forward modes from existing sweep/probe.",
        "- INT8 and INT4: G128 RTNClip shared-grid fake quantization on Linear.weight from the existing probe.",
        "",
        "## Fallback Choices",
        "",
        "- Practical windows are computed even when smooth rho fit is unstable.",
        "- Pure plug-in windows use direct Delta/L estimates when available and otherwise emit unavailable status rows.",
        "- FP32/FP16 Delta estimates are numeric visibility approximations, not hardware ulp claims.",
    ]
    (out / "discovery_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def readme(out: Path, source: Path) -> None:
    text = f"""# RoBERTa/SST-5 Robust Dual-Scheme Window Results

Generated: {datetime.now().isoformat(timespec='seconds')}

This folder is probe-only. No new training was run.

Source probe directory: `{source}`

Main outputs:
- `empirical_accuracy_windows.csv`: existing h-sweep accuracy good sets.
- `probe_summary_by_h.csv`: scalar nMSE, vector-level rho, corr/sign, d_h statistics.
- `pure_theory_plugin_windows.csv`: direct plug-in windows from Delta_eff and L_loc estimates, with status fields.
- `practical_probe_windows.csv`: threshold-based practical probe-visible windows, including FP16 dead-zone handling.
- `rho_fit_windows.csv`: smooth A/h^2+B h^2 rho-fit windows only where stable.
- `window_comparison_for_paper.csv`: single-row-per-precision comparison for paper use.

Important definitions:
- `scalar_nmse` is not rho.
- `rho_raw = mean((d_h-d_star)^2 ||u||^2) / V_dir_sample`.
- Practical windows use fixed probe thresholds and are not tuned by accuracy.
- Empirical accuracy windows use existing h-sweep results with `best_acc - 0.01` as the primary threshold.
"""
    (out / "README.md").write_text(text, encoding="utf-8")


def plot_probe_metrics(summary: pd.DataFrame, practical: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
    metrics = [
        ("scalar_nmse", "Scalar directional nMSE"),
        ("directional_corr", "corr(d_h, d*)"),
        ("sign_agreement", "Sign agreement"),
        ("dh_zero_fraction", "d_h zero fraction"),
    ]
    for ax, (col, title) in zip(axes.ravel(), metrics):
        for precision, g in summary.groupby("precision"):
            g = g.sort_values("h")
            ax.plot(g["h"], g[col], marker="o", linewidth=1.4, label=precision)
        ax.axvline(TINY_H, color="0.5", linestyle=":", linewidth=1)
        ax.axvline(DEFAULT_H, color="k", linestyle="--", linewidth=1)
        for _, r in practical.iterrows():
            low = safe_float(r.get("h_left_primary"))
            high = safe_float(r.get("h_right_primary"))
            if not pd.isna(low) and not pd.isna(high):
                ax.axvspan(low, high, color="tab:green", alpha=0.04)
        ax.set_xscale("log")
        ax.set_title(title)
        ax.grid(True, which="both", alpha=0.25)
    axes[0, 0].set_yscale("log")
    axes[0, 0].legend(fontsize=8)
    fig.supxlabel("h")
    fig.tight_layout()
    for ext in ["pdf", "png"]:
        fig.savefig(out / f"fig_probe_metrics_by_precision.{ext}", bbox_inches="tight", dpi=180)
    plt.close(fig)


def plot_left_boundary(summary: pd.DataFrame, out: Path) -> None:
    sub = summary[summary["precision"].isin(["fp16", "int4"])].copy()
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharex=True)
    for ax, col, title in zip(axes, ["d_h_std", "dh_zero_fraction", "directional_corr"], ["std(d_h)", "d_h zero fraction", "corr(d_h,d*)"]):
        for precision, g in sub.groupby("precision"):
            g = g.sort_values("h")
            ax.plot(g["h"], g[col], marker="o", label=precision)
        ax.axvline(TINY_H, color="0.5", linestyle=":", linewidth=1)
        ax.axvline(DEFAULT_H, color="k", linestyle="--", linewidth=1)
        ax.set_xscale("log")
        ax.set_title(title)
        ax.grid(True, which="both", alpha=0.25)
    axes[0].set_yscale("log")
    axes[0].legend(fontsize=8)
    fig.supxlabel("h")
    fig.tight_layout()
    for ext in ["pdf", "png"]:
        fig.savefig(out / f"fig_left_boundary_zoom_fp16_int4.{ext}", bbox_inches="tight", dpi=180)
    plt.close(fig)


def plot_window_comparison(comp: pd.DataFrame, emp: pd.DataFrame, pure: pd.DataFrame, practical: pd.DataFrame, fit: pd.DataFrame, out: Path) -> None:
    precisions = comp["precision"].tolist()
    fig, ax = plt.subplots(figsize=(10, 5.2))
    y = np.arange(len(precisions))
    ax.set_xscale("log")
    ax.set_xlim(1e-9, 3e-2)
    ax.set_ylim(-0.55, len(precisions) - 0.05)
    ax.set_yticks(y)
    ax.set_yticklabels(precisions)
    ax.axvline(TINY_H, color="0.5", linestyle=":", linewidth=1, label="tiny 1e-5")
    ax.axvline(DEFAULT_H, color="k", linestyle="--", linewidth=1, label="default 1e-3")
    for i, p in enumerate(precisions):
        er = row_for_threshold(emp, p, "best_acc_minus_0.01")
        if er:
            ax.hlines(i - 0.22, safe_float(er.get("h_good_low")), safe_float(er.get("h_good_high")), color="tab:blue", linewidth=7, alpha=0.35)
        pr = practical_row(practical, p)
        if pr:
            low = safe_float(pr.get("h_left_primary"))
            high = safe_float(pr.get("h_right_primary"))
            if not pd.isna(low):
                ax.hlines(i, low, high, color="tab:green", linewidth=5, alpha=0.75)
            low = safe_float(pr.get("h_left_relaxed"))
            high = safe_float(pr.get("h_right_relaxed"))
            if not pd.isna(low):
                ax.hlines(i + 0.08, low, high, color="tab:green", linewidth=2, alpha=0.45)
        pw = window_row(pure, p, 1.0)
        if pw and pw.get("status") == "window":
            ax.hlines(i + 0.22, safe_float(pw.get("h_low")), safe_float(pw.get("h_high")), color="tab:purple", linewidth=5, alpha=0.75)
        else:
            ax.text(1.4e-8, i + 0.22, "no plug-in cert", fontsize=7, va="center")
        fw = window_row(fit, p, 1.0)
        href = safe_float(fw.get("h_ref_fit"))
        if not pd.isna(href):
            ax.scatter([href], [i + 0.32], marker="D", color="tab:red", s=32)
        elif fw:
            ax.text(1.4e-8, i + 0.32, "no stable fit", fontsize=7, va="center")
    ax.set_xlabel("h")
    ax.grid(True, which="both", axis="x", alpha=0.25)
    ax.set_title("RoBERTa/SST-5 window comparison")
    handles = [
        plt.Line2D([0], [0], color="tab:blue", lw=7, alpha=0.35, label="empirical acc good set"),
        plt.Line2D([0], [0], color="tab:green", lw=5, alpha=0.75, label="practical probe primary"),
        plt.Line2D([0], [0], color="tab:purple", lw=5, alpha=0.75, label="pure plug-in W1"),
        plt.Line2D([0], [0], color="tab:red", marker="D", lw=0, label="rho-fit h_ref"),
    ]
    ax.legend(handles=handles, fontsize=8, loc="lower right")
    fig.tight_layout()
    for ext in ["pdf", "png"]:
        fig.savefig(out / f"fig_window_comparison_combined.{ext}", bbox_inches="tight", dpi=180)
    plt.close(fig)


def plot_rho_fit(summary: pd.DataFrame, fit: pd.DataFrame, out: Path) -> None:
    stable = fit[(fit["tau"] == 1.0) & (fit["status"] == "window")]
    if stable.empty:
        (out / "fig_rho_fit_if_stable.skipped.txt").write_text("No stable rho fits available.\n", encoding="utf-8")
        return
    n = len(stable)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 3.8), squeeze=False)
    for ax, (_, r) in zip(axes.ravel(), stable.iterrows()):
        p = r["precision"]
        g = summary[summary["precision"] == p].sort_values("h")
        hs = g["h"].astype(float).to_numpy()
        A = safe_float(r.get("A"))
        B = safe_float(r.get("B"))
        pred = A / (hs * hs) + B * hs * hs
        ax.plot(hs, g["rho_raw"], "o", label="rho_raw")
        ax.plot(hs, pred, "-", label="A/h^2+B h^2")
        ax.axhline(1.0, color="0.5", linestyle=":")
        ax.axvline(DEFAULT_H, color="k", linestyle="--", linewidth=1)
        ax.axvline(safe_float(r.get("h_ref_fit")), color="tab:red", linestyle="-.", linewidth=1)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(p)
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(fontsize=8)
    fig.tight_layout()
    for ext in ["pdf", "png"]:
        fig.savefig(out / f"fig_rho_fit_if_stable.{ext}", bbox_inches="tight", dpi=180)
    plt.close(fig)


def write_report(comp: pd.DataFrame, emp: pd.DataFrame, pure: pd.DataFrame, practical: pd.DataFrame, fit: pd.DataFrame, out: Path) -> None:
    lines = ["# Paper Window Result Summary", ""]
    for _, row in comp.iterrows():
        p = row["precision"]
        lines += [
            f"## {p}",
            "",
            f"1. Empirical accuracy window: `{row['empirical_good_primary'] or 'missing'}`.",
            f"2. Default h=1e-3 inside empirical window: `{row['default_in_empirical_primary']}`.",
            f"3. Tiny h=1e-5 inside empirical window: `{row['tiny_1e5_in_empirical_primary']}`.",
            f"4. Pure-theory plug-in status: `{row['pure_theory_status']}`; W1: `{row['pure_W1'] or 'none'}`.",
            f"5. Practical probe status: `{row['practical_status']}`; primary: `{row['practical_W_primary'] or 'none'}`; relaxed: `{row['practical_W_relaxed'] or 'none'}`.",
            f"6. Smooth rho fit status: `{row['rho_fit_status']}`; W1: `{row['rho_fit_W1'] or 'none'}`.",
            f"7. Paper wording: `{row['final_message']}`.",
            "",
        ]
        if p == "fp16":
            lines.append("FP16 note: small h has a d_h=0 dead zone; this is not the same as no practical window.")
            lines.append("")
        if p == "int4":
            lines.append("INT4 note: current true directional probe remains a boundary case; accuracy can be empirically default-safe without a stable smooth rho certificate.")
            lines.append("")
    (out / "paper_window_result_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=SOURCE_DEFAULT)
    parser.add_argument("--out", type=Path, default=ROOT / f"roberta_sst5_window_robust_dual_scheme_{DATE_DEFAULT}")
    args = parser.parse_args()

    source = args.source.resolve()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / "logs").mkdir(exist_ok=True)
    (out / "configs").mkdir(exist_ok=True)

    readme(out, source)
    discovery_report(source, out)
    raw, summary = write_raw_and_summary(source, out)
    emp = empirical_windows(source, out)
    practical = practical_windows(summary, out)
    pure = pure_theory_plugin(raw, summary, source, out)
    fit = rho_fit_windows(source, out)
    comp = comparison(emp, pure, practical, fit, out)
    write_report(comp, emp, pure, practical, fit, out)

    # Lightweight provenance/config copies.
    for name in [
        "checkpoint_manifest.csv",
        "accuracy_sweep_points.csv",
        "quantizer_summary_int8.json",
        "quantizer_summary_int4.json",
        "env.json",
    ]:
        copy_if_exists(source / name, out / "configs" / name)
    copy_if_exists(source / "logs" / "run_final.log", out / "logs" / "source_run_final.log")

    plot_probe_metrics(summary, practical, out)
    plot_window_comparison(comp, emp, pure, practical, fit, out)
    plot_left_boundary(summary, out)
    plot_rho_fit(summary, fit, out)

    metadata = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "source": str(source),
        "no_training": True,
        "default_h": DEFAULT_H,
        "tiny_h": TINY_H,
        "zero_tol": ZERO_TOL,
        "notes": "Robust dual-scheme post-processing of existing probe-only metrics.",
    }
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"Wrote {out}")
    print(comp.to_string(index=False))


if __name__ == "__main__":
    main()
