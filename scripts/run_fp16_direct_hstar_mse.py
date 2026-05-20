#!/usr/bin/env python3
"""Direct FP16 h-star MSE evaluation.

This script evaluates the formula-derived continuous h-star directly. It does
not train and does not select h by minimizing probe MSE. The grid is used for
G visibility/stability, L plateau diagnostics, visibility clamping, and an
oracle reference only after h-star has been computed.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import os
import shutil
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

from run_fp16_hstar_generalization import (  # noqa: E402
    EPS,
    H_GRID,
    Setting,
    direction_seeds,
    env_report,
    estimate_g,
    estimate_ulp,
    finite_corr,
    hstar,
    l_candidates,
    load_context,
    old_snr_l,
    resolve_settings,
    restore_external_backups,
    safe_float,
    select_l,
    set_mode_fp16,
    set_mode_fp32,
    two_point_fd,
    pair_effective_stats,
    compute_true_grads,
    true_directional_from_grads,
    write_json,
)


COMPONENT_FIELDS = [
    "group",
    "model",
    "dataset",
    "seed",
    "checkpoint",
    "d_trainable",
    "Delta_mode",
    "Delta_value",
    "G_method",
    "G_hat",
    "h_G",
    "L_mode",
    "L_q",
    "L_hat",
    "h2_L",
    "hstar_cont",
    "hstar_cont_q50",
    "hstar_cont_q95",
    "notes",
]

DIRECT_FIELDS = [
    "group",
    "model",
    "dataset",
    "seed",
    "checkpoint",
    "selector_name",
    "hstar_cont",
    "h_selected",
    "clamp_changed",
    "clamp_reason",
    "mse_at_hstar_cont",
    "nmse_at_hstar_cont",
    "corr_at_hstar_cont",
    "bias_at_hstar_cont",
    "mae_at_hstar_cont",
    "median_abs_error_at_hstar_cont",
    "alignment_cont",
    "norm_ratio_cont",
    "zero_coord_frac_cont",
    "rms_snap_error_cont",
    "mse_at_h_selected",
    "nmse_at_h_selected",
    "corr_at_h_selected",
    "bias_at_h_selected",
    "mae_at_h_selected",
    "median_abs_error_at_h_selected",
    "alignment_selected",
    "norm_ratio_selected",
    "zero_coord_frac_selected",
    "rms_snap_error_selected",
    "empirical_min_nmse_h",
    "empirical_min_nmse",
    "empirical_max_corr_h",
    "empirical_max_corr",
    "nmse_ratio_cont",
    "nmse_ratio_selected",
    "corr_gap_cont",
    "corr_gap_selected",
    "raw_cont_pass",
    "selected_pass",
    "strict_selected_pass",
]

VIS_FIELDS = [
    "group",
    "model",
    "dataset",
    "seed",
    "h",
    "alignment_eff",
    "norm_ratio_eff",
    "zero_coord_frac_eff",
    "rms_snap_error",
    "visibility_pass",
]

GRID_FIELDS = [
    "group",
    "model",
    "dataset",
    "seed",
    "h",
    "nmse",
    "corr",
    "mse",
    "bias",
    "mae",
    "median_abs_error",
    "alignment_eff",
    "norm_ratio_eff",
    "zero_coord_frac_eff",
    "rms_snap_error",
    "visibility_pass",
    "empirical_oracle_flags",
]


def write_csv(path: Path, rows: Iterable[Dict[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fields), extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})


def metric_row(fds: Sequence[float], d_true: Sequence[float]) -> Dict[str, float]:
    a = np.asarray(fds, dtype=np.float64)
    b = np.asarray(d_true, dtype=np.float64)
    mask = np.isfinite(a) & np.isfinite(b)
    if int(mask.sum()) == 0:
        return {
            "mse": float("nan"),
            "nmse": float("nan"),
            "corr": float("nan"),
            "bias": float("nan"),
            "mae": float("nan"),
            "median_abs_error": float("nan"),
        }
    err = a[mask] - b[mask]
    mse = float(np.mean(err * err))
    return {
        "mse": mse,
        "nmse": float(mse / (float(np.mean(b[mask] * b[mask])) + EPS)),
        "corr": finite_corr(a, b),
        "bias": float(np.mean(err)),
        "mae": float(np.mean(np.abs(err))),
        "median_abs_error": float(np.median(np.abs(err))),
    }


def visibility_pass(row: Dict[str, Any]) -> bool:
    align = safe_float(row.get("alignment_eff"))
    norm = safe_float(row.get("norm_ratio_eff"))
    zero = safe_float(row.get("zero_coord_frac_eff"))
    return bool(
        align is not None
        and align >= 0.99
        and norm is not None
        and 0.9 <= norm <= 1.1
        and zero is not None
        and zero <= 0.10
    )


def average_visibility(ctx: Any, seeds: Sequence[int], h: float, visibility_dirs: int) -> Dict[str, float]:
    vals: List[Dict[str, float]] = []
    for seed in list(seeds)[: max(1, min(int(visibility_dirs), len(seeds)))]:
        vals.append(pair_effective_stats(ctx, int(seed), float(h)))
    out: Dict[str, float] = {}
    for key in ["alignment_eff", "norm_ratio_eff", "zero_coord_frac_eff", "rms_snap_error"]:
        arr = [float(v[key]) for v in vals if key in v and math.isfinite(float(v[key]))]
        out[key] = float(np.mean(arr)) if arr else float("nan")
    out["visibility_pass"] = visibility_pass(out)
    return out


def compute_fd_values(ctx: Any, seeds: Sequence[int], h: float) -> List[float]:
    fds: List[float] = []
    for seed in seeds:
        fd, _, _ = two_point_fd(ctx, int(seed), float(h))
        fds.append(float(fd))
    return fds


def build_grid_probe(
    ctx: Any,
    seeds: Sequence[int],
    d_true: Sequence[float],
    visibility_dirs: int,
) -> Tuple[Dict[float, List[float]], Dict[float, Dict[str, float]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    fd_by_h: Dict[float, List[float]] = {}
    vis_by_h: Dict[float, Dict[str, float]] = {}
    vis_rows: List[Dict[str, Any]] = []
    grid_rows: List[Dict[str, Any]] = []
    for h in H_GRID:
        t0 = time.time()
        fds = compute_fd_values(ctx, seeds, float(h))
        vis = average_visibility(ctx, seeds, float(h), visibility_dirs)
        met = metric_row(fds, d_true)
        fd_by_h[float(h)] = fds
        vis_by_h[float(h)] = vis
        base = {
            "group": ctx.setting.group,
            "model": ctx.setting.model_label,
            "dataset": ctx.setting.dataset,
            "seed": ctx.setting.seed,
            "h": float(h),
            **vis,
        }
        vis_rows.append(base)
        grid_rows.append({**base, **met, "empirical_oracle_flags": ""})
        print(
            f"[grid-ref] {ctx.setting.group}/{ctx.setting.model_label}/{ctx.setting.dataset}/seed{ctx.setting.seed} "
            f"h={h:g} nmse={met['nmse']:.4g} corr={met['corr']:.4g} vis={vis['visibility_pass']} elapsed={time.time()-t0:.1f}s",
            flush=True,
        )
    return fd_by_h, vis_by_h, vis_rows, grid_rows


def make_probe_rows_for_g(ctx: Any, grid_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for r in grid_rows:
        rows.append(
            {
                "group": ctx.setting.group,
                "model": ctx.setting.model_label,
                "dataset": ctx.setting.dataset,
                "seed": ctx.setting.seed,
                "h": float(r["h"]),
                "alignment_eff": r.get("alignment_eff"),
                "norm_ratio_eff": r.get("norm_ratio_eff"),
                "zero_coord_frac_eff": r.get("zero_coord_frac_eff"),
            }
        )
    return rows


def visibility_clamp(
    ctx: Any,
    seeds: Sequence[int],
    hstar_cont: float,
    vis_by_h: Dict[float, Dict[str, float]],
    visibility_dirs: int,
    binary_steps: int,
) -> Tuple[float, Dict[str, float], bool, str]:
    if not math.isfinite(float(hstar_cont)) or float(hstar_cont) <= 0.0:
        return float("nan"), {}, False, "invalid_hstar"
    vis_cont = average_visibility(ctx, seeds, float(hstar_cont), visibility_dirs)
    if visibility_pass(vis_cont):
        return float(hstar_cont), vis_cont, False, "raw_hstar_visible"

    candidates = [h for h in H_GRID if h >= float(hstar_cont) and visibility_pass(vis_by_h.get(float(h), {}))]
    if not candidates:
        h_sel = max(H_GRID)
        vis = vis_by_h.get(float(h_sel)) or average_visibility(ctx, seeds, h_sel, visibility_dirs)
        return float(h_sel), vis, True, "no_visible_h_ge_hstar_on_scan;selected_max_grid"
    high = float(min(candidates))
    lower_grid = [h for h in H_GRID if h < high]
    low = float(hstar_cont)
    if lower_grid:
        low = max(float(hstar_cont), max([h for h in lower_grid if h < high] or [float(hstar_cont)]))
    vis_low = average_visibility(ctx, seeds, low, visibility_dirs)
    if visibility_pass(vis_low):
        return low, vis_low, True, "selected_previous_visible_scan_point"
    best_h = high
    best_vis = vis_by_h.get(high) or average_visibility(ctx, seeds, high, visibility_dirs)
    for _ in range(max(0, int(binary_steps))):
        mid = math.sqrt(max(low, 1e-12) * max(high, 1e-12))
        vis_mid = average_visibility(ctx, seeds, mid, visibility_dirs)
        if visibility_pass(vis_mid):
            best_h, best_vis, high = mid, vis_mid, mid
        else:
            low = mid
    reason = "binary_visibility_clamp" if binary_steps > 0 else "selected_from_grid_visibility_scan"
    return float(best_h), best_vis, True, reason


def selector_components(
    ctx: Any,
    ulp: Dict[str, Any],
    g_rows: Sequence[Dict[str, Any]],
    l_selected: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    primary_g = next((r for r in g_rows if r.get("G_method") == "absG"), None)
    rich_g = next((r for r in g_rows if r.get("G_method") == "richardsonG_candidate"), None)
    clean = next((r for r in l_selected if r.get("L_mode") == "L_clean32" and r.get("selector") == "plateau_q90_primary"), None)
    oracle = next((r for r in l_selected if r.get("L_mode") == "L_oracle_precision" and r.get("selector") == "plateau_q90_primary"), None)
    old = next((r for r in l_selected if r.get("selector") == "old_snr_max_fallback_ablation"), None)
    out: List[Dict[str, Any]] = []

    def add(name: str, g: Optional[Dict[str, Any]], lrow: Optional[Dict[str, Any]], q: str, notes: str = "") -> None:
        if not g or not lrow:
            return
        delta = float(ulp.get("delta_ulp_rms", float("nan")))
        ghat = float(g.get("G_hat", float("nan")))
        lval = float(lrow.get(f"selected_L_{q}", float("nan")))
        cont = hstar(delta, ghat, lval, ctx.d_trainable)
        q50 = hstar(delta, ghat, float(lrow.get("selected_L_q50", float("nan"))), ctx.d_trainable)
        q95 = hstar(delta, ghat, float(lrow.get("selected_L_q95", float("nan"))), ctx.d_trainable)
        out.append(
            {
                "group": ctx.setting.group,
                "model": ctx.setting.model_label,
                "dataset": ctx.setting.dataset,
                "seed": ctx.setting.seed,
                "checkpoint": "initial_pretrained",
                "d_trainable": ctx.d_trainable,
                "Delta_mode": "delta_ulp_rms",
                "Delta_value": delta,
                "G_method": g.get("G_method"),
                "G_hat": ghat,
                "h_G": g.get("h_G"),
                "L_mode": lrow.get("L_mode"),
                "L_q": q,
                "L_hat": lval,
                "h2_L": lrow.get("selected_h2"),
                "hstar_cont": cont,
                "hstar_cont_q50": q50,
                "hstar_cont_q95": q95,
                "selector_name": name,
                "notes": notes,
            }
        )

    add("calibrated_hstar_absG_Lclean32_q90", primary_g, clean, "q90")
    add("calibrated_hstar_absG_Lclean32_q50", primary_g, clean, "q50")
    add("calibrated_hstar_absG_Lclean32_q95", primary_g, clean, "q95")
    add("calibrated_hstar_absG_Loracle_q90", primary_g, oracle, "q90", "diagnostic oracle-precision L")
    add("oldSNR_L_ablation", primary_g, old, "q90", "old SNR L ablation")
    add("calibrated_hstar_richardsonG_Lclean32_q90", rich_g, clean, "q90", "diagnostic richardsonG")
    return out


def direct_eval_selector(
    ctx: Any,
    comp: Dict[str, Any],
    seeds: Sequence[int],
    d_true: Sequence[float],
    vis_by_h: Dict[float, Dict[str, float]],
    empirical_min: Dict[str, Any],
    empirical_corr: Dict[str, Any],
    visibility_dirs: int,
    binary_steps: int,
) -> Dict[str, Any]:
    h_cont = float(comp.get("hstar_cont", float("nan")))
    fds_cont = compute_fd_values(ctx, seeds, h_cont) if math.isfinite(h_cont) and h_cont > 0.0 else []
    met_cont = metric_row(fds_cont, d_true)
    vis_cont = average_visibility(ctx, seeds, h_cont, visibility_dirs) if fds_cont else {}

    h_sel, vis_sel, clamp_changed, reason = visibility_clamp(ctx, seeds, h_cont, vis_by_h, visibility_dirs, binary_steps)
    if math.isfinite(h_sel) and h_sel > 0.0:
        if abs(h_sel - h_cont) <= max(1e-12, 1e-9 * max(abs(h_sel), abs(h_cont))):
            fds_sel = fds_cont
            met_sel = met_cont
        else:
            fds_sel = compute_fd_values(ctx, seeds, h_sel)
            met_sel = metric_row(fds_sel, d_true)
    else:
        met_sel = metric_row([], d_true)

    emp_min = float(empirical_min["nmse"])
    emp_corr = float(empirical_corr["corr"])
    nmse_ratio_cont = float(met_cont["nmse"] / (emp_min + EPS)) if math.isfinite(met_cont["nmse"]) else float("nan")
    nmse_ratio_sel = float(met_sel["nmse"] / (emp_min + EPS)) if math.isfinite(met_sel["nmse"]) else float("nan")
    corr_gap_cont = float(emp_corr - met_cont["corr"]) if math.isfinite(met_cont["corr"]) and math.isfinite(emp_corr) else float("nan")
    corr_gap_sel = float(emp_corr - met_sel["corr"]) if math.isfinite(met_sel["corr"]) and math.isfinite(emp_corr) else float("nan")
    return {
        "group": ctx.setting.group,
        "model": ctx.setting.model_label,
        "dataset": ctx.setting.dataset,
        "seed": ctx.setting.seed,
        "checkpoint": "initial_pretrained",
        "selector_name": comp.get("selector_name"),
        "hstar_cont": h_cont,
        "h_selected": h_sel,
        "clamp_changed": bool(clamp_changed),
        "clamp_reason": reason,
        "mse_at_hstar_cont": met_cont["mse"],
        "nmse_at_hstar_cont": met_cont["nmse"],
        "corr_at_hstar_cont": met_cont["corr"],
        "bias_at_hstar_cont": met_cont["bias"],
        "mae_at_hstar_cont": met_cont["mae"],
        "median_abs_error_at_hstar_cont": met_cont["median_abs_error"],
        "alignment_cont": vis_cont.get("alignment_eff"),
        "norm_ratio_cont": vis_cont.get("norm_ratio_eff"),
        "zero_coord_frac_cont": vis_cont.get("zero_coord_frac_eff"),
        "rms_snap_error_cont": vis_cont.get("rms_snap_error"),
        "mse_at_h_selected": met_sel["mse"],
        "nmse_at_h_selected": met_sel["nmse"],
        "corr_at_h_selected": met_sel["corr"],
        "bias_at_h_selected": met_sel["bias"],
        "mae_at_h_selected": met_sel["mae"],
        "median_abs_error_at_h_selected": met_sel["median_abs_error"],
        "alignment_selected": vis_sel.get("alignment_eff"),
        "norm_ratio_selected": vis_sel.get("norm_ratio_eff"),
        "zero_coord_frac_selected": vis_sel.get("zero_coord_frac_eff"),
        "rms_snap_error_selected": vis_sel.get("rms_snap_error"),
        "empirical_min_nmse_h": empirical_min["h"],
        "empirical_min_nmse": emp_min,
        "empirical_max_corr_h": empirical_corr["h"],
        "empirical_max_corr": emp_corr,
        "nmse_ratio_cont": nmse_ratio_cont,
        "nmse_ratio_selected": nmse_ratio_sel,
        "corr_gap_cont": corr_gap_cont,
        "corr_gap_selected": corr_gap_sel,
        "raw_cont_pass": bool((math.isfinite(nmse_ratio_cont) and nmse_ratio_cont <= 1.25) or (math.isfinite(corr_gap_cont) and corr_gap_cont <= 0.01)),
        "selected_pass": bool((math.isfinite(nmse_ratio_sel) and nmse_ratio_sel <= 1.25) or (math.isfinite(corr_gap_sel) and corr_gap_sel <= 0.01)),
        "strict_selected_pass": bool(math.isfinite(nmse_ratio_sel) and nmse_ratio_sel <= 1.10),
    }


def plot_setting(
    out_dir: Path,
    setting: Setting,
    grid_rows: Sequence[Dict[str, Any]],
    direct_rows: Sequence[Dict[str, Any]],
    l_rows: Sequence[Dict[str, Any]],
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    pdir = out_dir / "plots" / f"{setting.group}_{setting.model_label}_{setting.dataset}_seed{setting.seed}".replace("/", "_")
    pdir.mkdir(parents=True, exist_ok=True)
    hs = np.asarray([float(r["h"]) for r in grid_rows], dtype=np.float64)
    marks = [r for r in direct_rows if r.get("selector_name") == "calibrated_hstar_absG_Lclean32_q90"]

    def mark(ax):
        for r in marks:
            hc = safe_float(r.get("hstar_cont"))
            hs_ = safe_float(r.get("h_selected"))
            if hc is not None:
                ax.axvline(hc, color="tab:red", linestyle="--", label="hstar_cont")
            if hs_ is not None:
                ax.axvline(hs_, color="tab:green", linestyle=":", label="h_selected")

    for key, fname, ylabel in [("nmse", "nmse_vs_h.png", "nMSE"), ("corr", "corr_vs_h.png", "corr")]:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(hs, [float(r[key]) for r in grid_rows], marker="o")
        mark(ax)
        ax.set_xscale("log")
        ax.set_xlabel("h")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(pdir / fname, dpi=160)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    for key in ["alignment_eff", "norm_ratio_eff", "zero_coord_frac_eff"]:
        ax.plot(hs, [float(r[key]) for r in grid_rows], marker="o", label=key)
    mark(ax)
    ax.set_xscale("log")
    ax.set_xlabel("h")
    ax.set_ylabel("visibility diagnostic")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(pdir / "visibility_vs_h.png", dpi=160)
    plt.close(fig)

    rows = [r for r in l_rows if r["L_mode"] == "L_clean32"]
    if rows:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot([float(r["h2"]) for r in rows], [float(r["lambda_q90"]) for r in rows], marker="o")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("h2")
        ax.set_ylabel("L q90")
        fig.tight_layout()
        fig.savefig(pdir / "L_q90_vs_h2.png", dpi=160)
        plt.close(fig)


def summarize_markdown(direct_rows: Sequence[Dict[str, Any]], out_dir: Path) -> str:
    primary = [r for r in direct_rows if r.get("selector_name") == "calibrated_hstar_absG_Lclean32_q90"]
    lines = [
        "# Direct FP16 h-star MSE evaluation",
        "",
        f"Output directory: `{out_dir}`",
        "",
        "The selector first computes continuous `hstar_cont` from Delta/G/L, then applies a visibility-only clamp to get `h_selected`. The empirical grid optimum is an oracle reference only.",
        "",
        "## Table 1: Direct hstar evaluation",
        "",
        "| model | dataset | seed | hstar_cont | nMSE(hstar_cont) | corr(hstar_cont) | h_selected | nMSE(h_selected) | corr(h_selected) | empirical min h | selected nMSE ratio | pass |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in primary:
        lines.append(
            f"| {r['model']} | {r['dataset']} | {r['seed']} | {float(r['hstar_cont']):.6g} | "
            f"{float(r['nmse_at_hstar_cont']):.6g} | {float(r['corr_at_hstar_cont']):.6g} | "
            f"{float(r['h_selected']):.6g} | {float(r['nmse_at_h_selected']):.6g} | "
            f"{float(r['corr_at_h_selected']):.6g} | {float(r['empirical_min_nmse_h']):.6g} | "
            f"{float(r['nmse_ratio_selected']):.6g} | {r['selected_pass']} |"
        )
    lines += [
        "",
        "## Table 2: Clamp behavior",
        "",
        "| model | dataset | seed | hstar_cont | h_selected | clamp changed? | reason |",
        "|---|---|---:|---:|---:|---|---|",
    ]
    for r in primary:
        lines.append(
            f"| {r['model']} | {r['dataset']} | {r['seed']} | {float(r['hstar_cont']):.6g} | "
            f"{float(r['h_selected']):.6g} | {r['clamp_changed']} | {r['clamp_reason']} |"
        )
    lines += [
        "",
        "## Table 3: Group pass rates",
        "",
        "| group | raw hstar pass rate | selected pass rate | strict selected pass rate | median selected nMSE ratio | max selected nMSE ratio |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for group in ["A_seed", "B_task", "C_model"]:
        xs = [r for r in primary if r.get("group") == group]
        if not xs:
            lines.append(f"| {group} | n/a | n/a | n/a | n/a | n/a |")
            continue
        raw = float(np.mean([bool(r["raw_cont_pass"]) for r in xs]))
        sel = float(np.mean([bool(r["selected_pass"]) for r in xs]))
        strict = float(np.mean([bool(r["strict_selected_pass"]) for r in xs]))
        ratios = [float(r["nmse_ratio_selected"]) for r in xs]
        lines.append(f"| {group} | {raw:.3g} | {sel:.3g} | {strict:.3g} | {float(np.median(ratios)):.6g} | {float(np.max(ratios)):.6g} |")
    lines += [
        "",
        "## Interpretation Notes",
        "",
        "- `hstar_cont` is formula-derived from Delta/G/L only.",
        "- `h_selected` is chosen only by visibility diagnostics, not by MSE/correlation.",
        "- `empirical_min_nmse_h` is an oracle grid reference and was not used for selection.",
    ]
    return "\n".join(lines) + "\n"


def analyze_setting(setting: Setting, args: argparse.Namespace, out_dir: Path, diagnostics: Dict[str, Any]):
    import torch

    print(f"[setting] start {setting.group}/{setting.model_label}/{setting.dataset}/seed{setting.seed}", flush=True)
    device = torch.device("cuda:0")
    ctx = load_context(setting, device, args.batch_size)
    diagnostics.setdefault("settings", []).append(
        {
            "group": setting.group,
            "model": setting.model_label,
            "dataset": setting.dataset,
            "seed": setting.seed,
            "data_info": ctx.data_info,
            "d_trainable": ctx.d_trainable,
        }
    )
    ulp = estimate_ulp(ctx)
    _, grads, truth_kind = compute_true_grads(ctx)
    fp32_backups = [b.detach().clone() for b in ctx.backups]
    probe_seeds = direction_seeds(setting.seed, args.num_probe_dirs, 0)
    l_seeds = direction_seeds(setting.seed, args.num_L_dirs, 1)
    vis_dirs = max(1, min(int(args.visibility_dirs), len(probe_seeds)))
    set_mode_fp16(ctx)
    d_true = [true_directional_from_grads(ctx, grads, int(seed)) for seed in probe_seeds]
    fd_by_h, vis_by_h, vis_rows, grid_rows = build_grid_probe(ctx, probe_seeds, d_true, vis_dirs)
    probe_rows_for_g = make_probe_rows_for_g(ctx, grid_rows)
    g_rows = estimate_g(ctx, probe_rows_for_g, fd_by_h)

    ctx.model.float()
    ctx.forward_precision = "fp32"
    ctx.direction_dtype_name = "float32"
    restore_external_backups(ctx, fp32_backups)
    clean_rows = l_candidates(ctx, "L_clean32", l_seeds)
    ctx.model.float()
    ctx.forward_precision = "fp32"
    ctx.direction_dtype_name = "float32"
    restore_external_backups(ctx, fp32_backups)
    oracle_rows = l_candidates(ctx, "L_oracle_precision", l_seeds)
    l_rows = clean_rows + oracle_rows
    l_selected = [select_l(clean_rows, "plateau_q90_primary"), select_l(oracle_rows, "plateau_q90_primary"), old_snr_l(oracle_rows)]

    components = selector_components(ctx, ulp, g_rows, l_selected)
    empirical_min = min(grid_rows, key=lambda r: float(r["nmse"]) if safe_float(r.get("nmse")) is not None else float("inf"))
    empirical_corr = max(grid_rows, key=lambda r: float(r["corr"]) if safe_float(r.get("corr")) is not None else -float("inf"))
    for r in grid_rows:
        flags = []
        if abs(float(r["h"]) - float(empirical_min["h"])) <= 1e-12:
            flags.append("empirical_min_nmse")
        if abs(float(r["h"]) - float(empirical_corr["h"])) <= 1e-12:
            flags.append("empirical_max_corr")
        r["empirical_oracle_flags"] = ";".join(flags)

    ctx.model.float()
    ctx.forward_precision = "fp32"
    ctx.direction_dtype_name = "float32"
    restore_external_backups(ctx, fp32_backups)
    set_mode_fp16(ctx)
    direct_rows = [
        direct_eval_selector(ctx, comp, probe_seeds, d_true, vis_by_h, empirical_min, empirical_corr, vis_dirs, args.binary_visibility_steps)
        for comp in components
    ]
    plot_setting(out_dir, setting, grid_rows, direct_rows, l_rows)
    print(f"[setting] done {setting.group}/{setting.model_label}/{setting.dataset}/seed{setting.seed}", flush=True)
    return components, direct_rows, vis_rows, grid_rows, l_rows, l_selected, g_rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="")
    parser.add_argument("--num_probe_dirs", type=int, default=32)
    parser.add_argument("--num_L_dirs", type=int, default=16)
    parser.add_argument("--visibility_dirs", type=int, default=8)
    parser.add_argument("--binary_visibility_steps", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--skip_opt", action="store_true")
    parser.add_argument("--only_group", default="")
    parser.add_argument("--max_settings", type=int, default=0)
    args = parser.parse_args()

    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir) if args.output_dir else REPO_ROOT / "analysis" / f"fp16_direct_hstar_mse_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=False)
    (out_dir / "plots").mkdir(parents=True, exist_ok=True)
    report = env_report()
    (out_dir / "env_report.txt").write_text(report, encoding="utf-8")
    diagnostics: Dict[str, Any] = {
        "start_time": dt.datetime.now().isoformat(),
        "h_grid": H_GRID,
        "num_probe_dirs": args.num_probe_dirs,
        "num_L_dirs": args.num_L_dirs,
        "visibility_dirs": args.visibility_dirs,
        "warnings": [],
        "skipped_settings": [],
    }
    if args.num_probe_dirs < 64 or args.num_L_dirs < 32:
        diagnostics["warnings"].append(
            f"direction counts reduced for runtime: m_probe={args.num_probe_dirs}, m_L={args.num_L_dirs}"
        )
    try:
        import torch

        if not torch.cuda.is_available():
            (out_dir / "failure_report.txt").write_text("CUDA unavailable; refusing to write empty scientific CSVs.\n", encoding="utf-8")
            write_json(out_dir / "diagnostics.json", diagnostics)
            return 2
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    except Exception as exc:
        (out_dir / "failure_report.txt").write_text(f"torch/CUDA startup failed: {exc}\n", encoding="utf-8")
        write_json(out_dir / "diagnostics.json", diagnostics)
        return 2

    settings, skipped = resolve_settings(include_opt=not args.skip_opt)
    if args.only_group:
        settings = [s for s in settings if s.group == args.only_group]
    if args.max_settings > 0:
        settings = settings[: args.max_settings]
    diagnostics["skipped_settings"].extend(skipped)

    all_components: List[Dict[str, Any]] = []
    all_direct: List[Dict[str, Any]] = []
    all_vis: List[Dict[str, Any]] = []
    all_grid: List[Dict[str, Any]] = []
    all_l: List[Dict[str, Any]] = []
    all_lsel: List[Dict[str, Any]] = []
    all_g: List[Dict[str, Any]] = []

    for setting in settings:
        try:
            components, direct_rows, vis_rows, grid_rows, l_rows, lsel, grows = analyze_setting(setting, args, out_dir, diagnostics)
            all_components.extend(components)
            all_direct.extend(direct_rows)
            all_vis.extend(vis_rows)
            all_grid.extend(grid_rows)
            all_l.extend(l_rows)
            all_lsel.extend(lsel)
            all_g.extend([{k: v for k, v in r.items() if not str(k).startswith("_")} for r in grows])
            write_csv(out_dir / "hstar_components.csv", all_components, COMPONENT_FIELDS)
            write_csv(out_dir / "hstar_direct_eval.csv", all_direct, DIRECT_FIELDS)
            write_csv(out_dir / "visibility_scan.csv", all_vis, VIS_FIELDS)
            write_csv(out_dir / "optional_grid_reference.csv", all_grid, GRID_FIELDS)
            write_csv(out_dir / "L_candidates.csv", all_l, list(all_l[0].keys()) if all_l else [])
            write_csv(out_dir / "L_selected.csv", all_lsel, list(all_lsel[0].keys()) if all_lsel else [])
            write_csv(out_dir / "G_estimates.csv", all_g, list(all_g[0].keys()) if all_g else [])
            write_json(out_dir / "diagnostics.json", diagnostics)
        except Exception as exc:
            diagnostics.setdefault("skipped_settings", []).append(
                {
                    "group": setting.group,
                    "model": setting.model_label,
                    "dataset": setting.dataset,
                    "seed": setting.seed,
                    "reason": repr(exc),
                    "traceback": traceback.format_exc(),
                }
            )
            print(f"[setting] skipped {setting.group}/{setting.model_label}/{setting.dataset}/seed{setting.seed}: {exc}", flush=True)
            write_json(out_dir / "diagnostics.json", diagnostics)

    write_csv(out_dir / "hstar_components.csv", all_components, COMPONENT_FIELDS)
    write_csv(out_dir / "hstar_direct_eval.csv", all_direct, DIRECT_FIELDS)
    write_csv(out_dir / "visibility_scan.csv", all_vis, VIS_FIELDS)
    write_csv(out_dir / "optional_grid_reference.csv", all_grid, GRID_FIELDS)
    if all_l:
        write_csv(out_dir / "L_candidates.csv", all_l, list(all_l[0].keys()))
    if all_lsel:
        write_csv(out_dir / "L_selected.csv", all_lsel, list(all_lsel[0].keys()))
    if all_g:
        write_csv(out_dir / "G_estimates.csv", all_g, list(all_g[0].keys()))
    (out_dir / "hstar_direct_summary.md").write_text(summarize_markdown(all_direct, out_dir), encoding="utf-8")
    diagnostics["end_time"] = dt.datetime.now().isoformat()
    primary = [r for r in all_direct if r.get("selector_name") == "calibrated_hstar_absG_Lclean32_q90"]
    diagnostics["settings_completed"] = len(primary)
    write_json(out_dir / "diagnostics.json", diagnostics)

    raw_rate = float(np.mean([bool(r["raw_cont_pass"]) for r in primary])) if primary else float("nan")
    sel_rate = float(np.mean([bool(r["selected_pass"]) for r in primary])) if primary else float("nan")
    strict_rate = float(np.mean([bool(r["strict_selected_pass"]) for r in primary])) if primary else float("nan")
    print(f"output directory: {out_dir}")
    print(f"total settings completed: {len(primary)}")
    print(f"raw hstar pass rate: {raw_rate:.4g}")
    print(f"visibility-clamped selected pass rate: {sel_rate:.4g}")
    print(f"strict selected pass rate: {strict_rate:.4g}")
    for group in ["A_seed", "B_task", "C_model"]:
        xs = [r for r in primary if r.get("group") == group]
        rate = float(np.mean([bool(r["selected_pass"]) for r in xs])) if xs else float("nan")
        print(f"{group}: n={len(xs)} selected_pass_rate={rate:.4g}")
    if diagnostics.get("skipped_settings"):
        print("skipped settings:")
        for item in diagnostics["skipped_settings"]:
            print(f"  {item}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
