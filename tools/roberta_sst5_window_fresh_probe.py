#!/usr/bin/env python3
"""Fresh FP16/INT4 RoBERTa/SST-5 probe and robust window summaries.

This script intentionally recomputes raw probe metrics. It does not read the
historical `roberta_sst5_theoretical_windows_20260627/raw_probe_metrics.csv`.
It reuses the true-probe implementation in `tools/roberta_sst5_theoretical_windows.py`
for model loading, gradients, two-sided finite differences, and RTNClip oracle
logic, then adds dead-zone-aware practical windows and filtered rho fits.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import roberta_sst5_theoretical_windows as tw


ROOT = Path(__file__).resolve().parents[1]
DATE_DEFAULT = os.environ.get("HWINDOW_DATE", datetime.now().strftime("%Y%m%d"))
DEFAULT_OUT = ROOT / f"roberta_sst5_window_fresh_probe_{DATE_DEFAULT}"
DEFAULT_H = 1e-3
TINY_H = 1e-5
ZERO_TOL = 1e-12
TAUS = [0.1, 0.5, 1.0, 2.0]

FP16_GRID = [3e-6, 5e-6, 7e-6, 1e-5, 2e-5, 3e-5, 5e-5, 7e-5, 1e-4, 3e-4, 1e-3, 1.5e-3, 2e-3, 3e-3, 5e-3]
INT4_GRID = [1e-5, 3e-5, 1e-4, 3e-4, 5e-4, 7e-4, 1e-3, 1.2e-3, 1.5e-3, 2e-3, 3e-3, 5e-3]


def finite(value) -> float | None:
    try:
        x = float(value)
    except Exception:
        return None
    return x if math.isfinite(x) else None


def interval_contains(low, high, h: float):
    lo = finite(low)
    hi = finite(high)
    if lo is None or hi is None:
        return None
    return bool(lo <= h <= hi)


def fmt_interval(low, high) -> str:
    lo = finite(low)
    hi = finite(high)
    if lo is None or hi is None:
        return ""
    return f"[{lo:.6g}, {hi:.6g}]"


def truthy(value) -> bool:
    if value is None:
        return False
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y"}
    try:
        if pd.isna(value):
            return False
    except Exception:
        pass
    return bool(value)


def write_csv(path: Path, rows: Sequence[Dict[str, object]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames), extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def add_output_gitignore(out: Path) -> None:
    (out / ".gitignore").write_text("checkpoints/*.pt\nraw_probe_metrics.jsonl\n", encoding="utf-8")


def compute_zero_fraction(raw: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (precision, h), g in raw.groupby(["precision", "h"], dropna=False):
        dh = g["d_h"].astype(float).to_numpy()
        rows.append(
            {
                "precision": precision,
                "h": float(h),
                "dh_zero_fraction": float(np.mean(np.abs(dh) <= ZERO_TOL)) if len(dh) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def add_zero_to_summary(out: Path) -> pd.DataFrame:
    raw = pd.read_csv(out / "raw_probe_metrics.csv")
    summary = pd.read_csv(out / "probe_summary_by_h.csv")
    z = compute_zero_fraction(raw)
    summary = summary.drop(columns=["dh_zero_fraction"], errors="ignore").merge(z, on=["precision", "h"], how="left")
    summary.to_csv(out / "probe_summary_by_h.csv", index=False)
    return summary


def pass_practical(row: pd.Series, primary: bool) -> bool:
    corr_thr = 0.90 if primary else 0.80
    nmse_thr = 0.10 if primary else 0.20
    sign_thr = 0.80 if primary else 0.70
    zero_thr = 0.95 if primary else 0.99
    corr = finite(row.get("directional_corr"))
    nmse = finite(row.get("scalar_nmse"))
    sign = finite(row.get("sign_agreement"))
    zero = finite(row.get("dh_zero_fraction"))
    dh_std = finite(row.get("d_h_std"))
    return (
        corr is not None
        and corr >= corr_thr
        and nmse is not None
        and nmse <= nmse_thr
        and sign is not None
        and sign >= sign_thr
        and zero is not None
        and zero < zero_thr
        and dh_std is not None
        and dh_std > 0
    )


def bounds(group: pd.DataFrame) -> tuple[float, float, str]:
    if group.empty:
        return np.nan, np.nan, ""
    vals = sorted(float(x) for x in group["h"].tolist())
    return vals[0], vals[-1], " ".join(f"{x:.9g}" for x in vals)


def practical_windows(summary: pd.DataFrame, out: Path) -> pd.DataFrame:
    rows = []
    for precision, g in summary.groupby("precision", sort=False):
        g = g.sort_values("h")
        primary = g[g.apply(lambda r: pass_practical(r, True), axis=1)]
        relaxed = g[g.apply(lambda r: pass_practical(r, False), axis=1)]
        nonzero = g[g["dh_zero_fraction"].fillna(1.0) < 1.0]
        p_low, p_high, p_vals = bounds(primary)
        r_low, r_high, r_vals = bounds(relaxed)
        nz_low, _, _ = bounds(nonzero)
        status = "primary_available" if not primary.empty else ("relaxed_only" if not relaxed.empty else "no_practical_probe_visible_window")
        rows.append(
            {
                "precision": precision,
                "checkpoint_id": g["checkpoint_id"].iloc[0],
                "first_nonzero_dh_h": nz_low,
                "primary_status": "available" if not primary.empty else "unavailable",
                "h_left_primary": p_low,
                "h_right_primary": p_high,
                "primary_h_values": p_vals,
                "relaxed_status": "available" if not relaxed.empty else "unavailable",
                "h_left_relaxed": r_low,
                "h_right_relaxed": r_high,
                "relaxed_h_values": r_vals,
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


def filtered_fit_points(group: pd.DataFrame) -> pd.DataFrame:
    g = group.copy()
    return g[
        (g["dh_zero_fraction"].fillna(1.0) < 0.99)
        & (g["d_h_std"].fillna(0.0) > 0.0)
        & (g["scalar_nmse"].replace([np.inf, -np.inf], np.nan).fillna(np.inf) <= 5.0)
        & (g["directional_corr"].replace([np.inf, -np.inf], np.nan).fillna(-np.inf) >= 0.20)
        & (g["rho_raw"].replace([np.inf, -np.inf], np.nan).fillna(np.nan) > 0)
    ].sort_values("h")


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


def smooth_rho_windows(summary: pd.DataFrame, out: Path) -> pd.DataFrame:
    rows = []
    for precision, g in summary.groupby("precision", sort=False):
        fit_g = filtered_fit_points(g)
        if len(fit_g) >= 4:
            fit = tw.fit_rho(
                fit_g["h"].astype(float).to_numpy(),
                fit_g["rho_raw"].astype(float).to_numpy(),
                tw.PRIMARY_FIT_METHOD,
            )
            A = finite(fit.get("A"))
            B = finite(fit.get("B"))
            C = finite(fit.get("C")) or 0.0
            stable = fit.get("fit_stability_flag") == "stable"
            h_ref = (A / B) ** 0.25 if stable and A and B and A > 0 and B > 0 else np.nan
            fit_quality = fit.get("fit_quality_r2_log")
            notes_base = fit.get("fit_notes") or ""
            status_base = "stable" if stable else "no_stable_smooth_fit"
        else:
            A = B = C = h_ref = fit_quality = np.nan
            stable = False
            notes_base = f"insufficient filtered points after excluding dead-zone/noisy points: {len(fit_g)}"
            status_base = "no_stable_smooth_fit"
        for tau in TAUS:
            if stable and A and B:
                rho_min, h_low, h_high, wstatus = solve_window(A, B, tau)
                status = wstatus
            else:
                rho_min, h_low, h_high = np.nan, np.nan, np.nan
                status = status_base
            rows.append(
                {
                    "precision": precision,
                    "checkpoint_id": g["checkpoint_id"].iloc[0],
                    "fit_method": "dep_log_soft_l1_filtered_deadzone",
                    "filter": "dh_zero_fraction<0.99, d_h_std>0, scalar_nmse<=5, corr>=0.20",
                    "n_total_h": int(len(g)),
                    "n_fit_h": int(len(fit_g)),
                    "fit_h_values": " ".join(f"{float(x):.9g}" for x in fit_g["h"].tolist()),
                    "A": A,
                    "B": B,
                    "C": C,
                    "h_ref": h_ref,
                    "rho_min_dep": rho_min,
                    "tau": tau,
                    "h_low": h_low,
                    "h_high": h_high,
                    "default_h": DEFAULT_H,
                    "default_in_window": interval_contains(h_low, h_high, DEFAULT_H),
                    "rho_dep_at_default": A / (DEFAULT_H * DEFAULT_H) + B * DEFAULT_H * DEFAULT_H if stable and A and B else np.nan,
                    "fit_quality_r2_or_log_error": fit_quality,
                    "fit_stability_flag": "stable" if stable else "unavailable",
                    "status": status,
                    "notes": notes_base if notes_base else "fit excludes finite-difference dead-zone points",
                }
            )
    df = pd.DataFrame(rows)
    df.to_csv(out / "rho_fit_windows.csv", index=False)
    return df


def read_empirical(out: Path) -> pd.DataFrame:
    src = pd.read_csv(out / "empirical_accuracy_good_sets.csv")
    rows = []
    for _, row in src.iterrows():
        low = finite(row.get("h_good_low"))
        high = finite(row.get("h_good_high"))
        rows.append(
            {
                **row.to_dict(),
                "default_h": DEFAULT_H,
                "tiny_h": TINY_H,
                "default_in_empirical_window": interval_contains(low, high, DEFAULT_H),
                "tiny_in_empirical_window": interval_contains(low, high, TINY_H),
                "status": "ok" if low is not None and high is not None else "missing_accuracy",
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(out / "empirical_accuracy_windows.csv", index=False)
    return df


def empirical_primary(emp: pd.DataFrame, precision: str) -> Dict[str, object]:
    m = emp[(emp["precision"] == precision) & (emp["threshold_type"] == "best_acc_minus_0.01")]
    return m.iloc[0].to_dict() if not m.empty else {}


def practical_one(practical: pd.DataFrame, precision: str) -> Dict[str, object]:
    m = practical[practical["precision"] == precision]
    return m.iloc[0].to_dict() if not m.empty else {}


def fit_one(fit: pd.DataFrame, precision: str, tau: float = 1.0) -> Dict[str, object]:
    m = fit[(fit["precision"] == precision) & (fit["tau"].astype(float) == tau)]
    return m.iloc[0].to_dict() if not m.empty else {}


def final_message(emp: Dict[str, object], prac: Dict[str, object], fit: Dict[str, object]) -> str:
    if truthy(prac.get("default_in_practical_primary")) or truthy(prac.get("default_in_practical_relaxed")):
        return "practical default-safe; smooth fit optional"
    if truthy(emp.get("default_in_empirical_window")):
        return "empirical default-safe, no practical probe certificate"
    if str(prac.get("status", "")).startswith("no_practical"):
        return "boundary/no practical probe window"
    if str(fit.get("status", "")) == "no_stable_smooth_fit":
        return "no smooth rho fit"
    return "default-risk or insufficient data"


def comparison(emp: pd.DataFrame, practical: pd.DataFrame, fit: pd.DataFrame, out: Path) -> pd.DataFrame:
    rows = []
    for precision in ["fp16", "int4"]:
        e = empirical_primary(emp, precision)
        p = practical_one(practical, precision)
        f1 = fit_one(fit, precision, 1.0)
        rows.append(
            {
                "precision": precision,
                "checkpoint_id": p.get("checkpoint_id") or f1.get("checkpoint_id"),
                "empirical_good_primary": fmt_interval(e.get("h_good_low"), e.get("h_good_high")),
                "default_in_empirical_primary": e.get("default_in_empirical_window"),
                "tiny_1e5_in_empirical_primary": e.get("tiny_in_empirical_window"),
                "first_nonzero_dh_h": p.get("first_nonzero_dh_h"),
                "practical_status": p.get("status"),
                "practical_W_primary": fmt_interval(p.get("h_left_primary"), p.get("h_right_primary")),
                "practical_W_relaxed": fmt_interval(p.get("h_left_relaxed"), p.get("h_right_relaxed")),
                "default_in_practical_primary": p.get("default_in_practical_primary"),
                "default_in_practical_relaxed": p.get("default_in_practical_relaxed"),
                "tiny_1e5_in_practical_primary": p.get("tiny_in_practical_primary"),
                "rho_fit_status": f1.get("status"),
                "rho_fit_h_ref": f1.get("h_ref"),
                "rho_fit_W1": fmt_interval(f1.get("h_low"), f1.get("h_high")),
                "rho_fit_n_fit_h": f1.get("n_fit_h"),
                "rho_fit_h_values": f1.get("fit_h_values"),
                "final_message": final_message(e, p, f1),
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(out / "window_comparison_for_paper.csv", index=False)
    return df


def plot_left_boundary(summary: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.0), sharex=True)
    panels = [
        ("d_h_std", "std(d_h)", True),
        ("dh_zero_fraction", "d_h zero fraction", False),
        ("directional_corr", "corr(d_h, d*)", False),
        ("scalar_nmse", "scalar true directional nMSE", True),
    ]
    for ax, (col, title, logy) in zip(axes.ravel(), panels):
        for precision, g in summary.groupby("precision", sort=False):
            g = g.sort_values("h")
            ax.plot(g["h"], g[col], marker="o", linewidth=1.4, label=precision)
        ax.axvline(TINY_H, color="0.5", linestyle=":", linewidth=1, label="1e-5" if col == "d_h_std" else None)
        ax.axvline(DEFAULT_H, color="k", linestyle="--", linewidth=1, label="1e-3" if col == "d_h_std" else None)
        ax.set_xscale("log")
        if logy:
            ax.set_yscale("log")
        ax.set_title(title)
        ax.grid(True, which="both", alpha=0.25)
    axes[0, 0].legend(fontsize=8)
    fig.supxlabel("h")
    fig.tight_layout()
    fig.savefig(out / "fig_left_boundary_zoom_fp16_int4.pdf", bbox_inches="tight")
    fig.savefig(out / "fig_left_boundary_zoom_fp16_int4.png", bbox_inches="tight", dpi=180)
    plt.close(fig)


def plot_probe_metrics(summary: pd.DataFrame, practical: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.8), sharex=True)
    for ax, col, title in zip(
        axes,
        ["scalar_nmse", "directional_corr", "sign_agreement"],
        ["scalar true directional nMSE", "corr(d_h,d*)", "sign agreement"],
    ):
        for precision, g in summary.groupby("precision", sort=False):
            g = g.sort_values("h")
            ax.plot(g["h"], g[col], marker="o", label=precision)
        ax.axvline(DEFAULT_H, color="k", linestyle="--", linewidth=1)
        ax.axvline(TINY_H, color="0.5", linestyle=":", linewidth=1)
        ax.set_xscale("log")
        if col == "scalar_nmse":
            ax.set_yscale("log")
        ax.set_title(title)
        ax.grid(True, which="both", alpha=0.25)
    axes[0].legend(fontsize=8)
    fig.supxlabel("h")
    fig.tight_layout()
    fig.savefig(out / "fig_probe_metrics_fp16_int4.pdf", bbox_inches="tight")
    fig.savefig(out / "fig_probe_metrics_fp16_int4.png", bbox_inches="tight", dpi=180)
    plt.close(fig)


def write_summary(out: Path, comp: pd.DataFrame, summary: pd.DataFrame) -> None:
    lines = [
        "# Fresh Probe Window Result Summary",
        "",
        "This folder was generated from a fresh probe run. It does not reuse the historical `roberta_sst5_theoretical_windows_20260627/raw_probe_metrics.csv`.",
        "",
        "- model/task: RoBERTa-large / SST-5 full data",
        "- precisions: FP16 and INT4",
        "- probe target: `d_star=<grad,u>`, `d_h=[F(w+hu)-F(w-hu)]/(2h)`, `e_h=d_h-d_star`",
        "- vector rho: `mean(e_h^2 ||u||^2) / V_dir_sample`; scalar nMSE is reported separately.",
        "",
    ]
    for _, row in comp.iterrows():
        p = row["precision"]
        lines += [
            f"## {p}",
            "",
            f"- d_h=0 -> nonzero transition: `{row['first_nonzero_dh_h']}`.",
            f"- Empirical accuracy good set: `{row['empirical_good_primary'] or 'missing'}`.",
            f"- Practical primary window: `{row['practical_W_primary'] or 'none'}`.",
            f"- Practical relaxed window: `{row['practical_W_relaxed'] or 'none'}`.",
            f"- Smooth rho fit status: `{row['rho_fit_status']}`; W1: `{row['rho_fit_W1'] or 'none'}`.",
            f"- Interpretation: `{row['final_message']}`.",
            "",
        ]
        sg = summary[summary["precision"] == p].sort_values("h")
        if not sg.empty:
            best_nmse = sg.loc[sg["scalar_nmse"].astype(float).idxmin()]
            lines.append(f"- Best scalar nMSE point in this fresh grid: h=`{float(best_nmse['h']):.9g}`, nMSE=`{float(best_nmse['scalar_nmse']):.6g}`, corr=`{best_nmse['directional_corr']}`.")
            lines.append("")
    (out / "paper_window_result_summary.md").write_text("\n".join(lines), encoding="utf-8")


def update_readme(out: Path, num_directions: int, comp: pd.DataFrame) -> None:
    ckpt = pd.read_csv(out / "checkpoint_manifest.csv") if (out / "checkpoint_manifest.csv").exists() else pd.DataFrame()
    rows = []
    for _, r in comp.iterrows():
        rows.append(
            f"| {r['precision']} | {r['first_nonzero_dh_h']} | {r['practical_W_primary'] or 'none'} | "
            f"{r['practical_W_relaxed'] or 'none'} | {r['rho_fit_status']} | {r['rho_fit_W1'] or 'none'} | {r['final_message']} |"
        )
    ckpt_lines = []
    if not ckpt.empty:
        for _, r in ckpt.iterrows():
            ckpt_lines.append(f"- {r.get('precision')}: `{r.get('checkpoint_path')}`")
    else:
        ckpt_lines.append("- checkpoint manifest unavailable")
    text = f"""# RoBERTa/SST-5 Fresh FP16/INT4 Probe

This is a fresh probe-only run. It recomputes `d_star`, `d_h`, `e_h`,
`scalar_nmse`, `rho_raw`, correlation, sign agreement, `dh_std`, and
`dh_zero_fraction`.

It does not reuse `/scratch/jy03364/MeZO_/roberta_sst5_theoretical_windows_20260627/raw_probe_metrics.csv`.

Requested directions: {num_directions}

FP16 h grid: {', '.join(f'{h:g}' for h in FP16_GRID)}

INT4 h grid: {', '.join(f'{h:g}' for h in INT4_GRID)}

## Probe Setup

- model/task: `roberta-large` / SST-5 full data
- seed/data_seed: 16/16
- batch size: 64; num_batches: 1
- direction seed base: 730000
- trainable subspace: all floating model parameters, matching the RTNClip dense runner
- low-bit forward oracle: G128 RTNClip shared-grid fake quantization on Linear.weight; non-Linear parameters remain unquantized in the forward state
- rho denominator: sampled vector random-direction floor `V_dir_sample`; scalar nMSE is reported separately and is not rho.

## Checkpoints

The underlying probe loader regenerates deterministic task-start states. Checkpoint files are local reproducibility artifacts and are ignored by `.gitignore`.

{chr(10).join(ckpt_lines)}

## Result Summary

| precision | first nonzero d_h h | practical primary | practical relaxed | smooth fit status | smooth W1 | interpretation |
|---|---:|---|---|---|---|---|
{chr(10).join(rows)}

## File Notes

- `raw_probe_metrics.csv` is freshly computed in this folder.
- `probe_summary_by_h.csv` adds `dh_zero_fraction`.
- `practical_probe_windows.csv` is threshold based and does not use accuracy.
- `rho_fit_windows.csv` is the dead-zone-filtered smooth rho fit.
- `rho_fit_windows_unfiltered_base_script.csv`, if present, is only the base script's unfiltered fit provenance and should not be used as the final fit table.
"""
    (out / "README.md").write_text(text, encoding="utf-8")


def run_fresh_probe(out: Path, num_directions: int, batch_size: int) -> None:
    os.environ.setdefault("DATALOADER_SHUFFLE", "True")
    tw.DEFAULT_H_GRIDS["fp16"] = list(FP16_GRID)
    tw.DEFAULT_H_GRIDS["int4"] = list(INT4_GRID)
    args = SimpleNamespace(
        output_dir=str(out),
        precisions=["fp16", "int4"],
        model_id="roberta-large",
        seed=16,
        data_seed=16,
        batch_size=batch_size,
        num_batches=1,
        num_directions=num_directions,
        direction_seed_base=730000,
        group_size=128,
        h_grid="",
        progress_every=8,
        reuse_raw_metrics=False,
    )
    tw.run_probe(args)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--num_directions", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--skip_probe", action="store_true", help="Only recompute summaries from this output folder's own fresh raw metrics.")
    args = parser.parse_args()

    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    add_output_gitignore(out)

    if not args.skip_probe:
        if (out / "raw_probe_metrics.csv").exists():
            raise FileExistsError(f"{out / 'raw_probe_metrics.csv'} already exists; remove it or use --skip_probe for own-folder postprocessing")
        run_fresh_probe(out, args.num_directions, args.batch_size)
        add_output_gitignore(out)

    summary = add_zero_to_summary(out)
    practical = practical_windows(summary, out)
    rho_fit = smooth_rho_windows(summary, out)
    emp = read_empirical(out)
    comp = comparison(emp, practical, rho_fit, out)
    comp.to_csv(out / "comparison_table_for_paper.csv", index=False)

    base_fit = out / "fitted_windows.csv"
    if base_fit.exists():
        dst = out / "rho_fit_windows_unfiltered_base_script.csv"
        if dst.exists():
            dst.unlink()
        base_fit.rename(dst)

    plot_left_boundary(summary, out)
    plot_probe_metrics(summary, practical, out)
    write_summary(out, comp, summary)
    update_readme(out, args.num_directions, comp)

    # Keep local reproducibility checkpoint files out of accidental git adds.
    add_output_gitignore(out)
    print(f"Wrote fresh probe outputs to {out}")
    print(comp.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
