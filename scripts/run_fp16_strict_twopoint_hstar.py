#!/usr/bin/env python3
"""Strict symmetric two-point FP16 h-star diagnostic.

This is an offline probe-only analyzer. It estimates G and third-directional
derivative moments directly, computes continuous h-star variants, and evaluates
FP16 two-point finite differences at those h values. It does not train.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

from run_fp16_direct_hstar_mse import (  # noqa: E402
    average_visibility,
    build_grid_probe,
    compute_fd_values,
    metric_row,
    visibility_pass,
    write_csv,
)
from run_fp16_hstar_generalization import (  # noqa: E402
    EPS,
    H_GRID,
    Setting,
    apply_signed,
    compute_loss,
    compute_true_grads,
    direction_norm_sq,
    direction_seeds,
    env_report,
    estimate_ulp,
    finite_corr,
    load_context,
    reset_backups,
    resolve_settings,
    restore,
    restore_external_backups,
    safe_float,
    set_mode_fp16,
    sign_flip_rate,
    true_directional_from_grads,
    write_json,
)


G_FIELDS = [
    "group",
    "model",
    "dataset",
    "seed",
    "G_true",
    "G_true_available",
    "grad_norm_dtype",
    "d_trainable",
    "all_trainable_params_included",
    "G_fd_multi",
    "G_fd_at_3e-4",
    "G_fd_at_1e-3",
    "G_fd_smallest_stable",
    "G_fd_over_true",
    "relative_error_G_fd",
    "selected_hG_values",
    "fallback_flags",
]

T3_CAND_FIELDS = [
    "group",
    "model",
    "dataset",
    "seed",
    "h3",
    "m_T3",
    "S3_sq",
    "S3_rms",
    "T3_abs_mean",
    "T3_abs_median",
    "rho3_q50",
    "rho3_q90",
    "rho3_q95",
    "finite_rate",
    "zero_T3_frac",
    "sign_negative_frac",
    "S3_sq_stability_2x",
    "S3_sq_stability_next",
    "rho3_q90_stability_2x",
    "rho3_q90_stability_next",
    "log_slope_S3_next",
    "log_slope_rho_next",
    "low_h3_noise_suspected",
    "large_h3_nonlocal_suspected",
]

T3_SEL_FIELDS = [
    "group",
    "model",
    "dataset",
    "seed",
    "selected_h3_values",
    "S3_sq_multi",
    "S3_rms_multi",
    "rho3_q90_selected",
    "selection_status",
    "fallback_flags",
]

EVAL_FIELDS = [
    "group",
    "model",
    "dataset",
    "seed",
    "formula_name",
    "Delta_mode",
    "Delta_value",
    "G_mode",
    "G_value",
    "S3_sq",
    "rho3_value",
    "L_value_if_used",
    "hstar_cont",
    "empirical_min_nmse_h",
    "h_over_oracle",
    "oracle_over_h",
    "empirical_min_nmse",
    "empirical_max_corr_h",
    "empirical_max_corr",
    "mse",
    "nmse",
    "corr",
    "bias",
    "mae",
    "median_abs_error",
    "nmse_ratio",
    "corr_gap",
    "alignment_eff",
    "norm_ratio_eff",
    "zero_coord_frac_eff",
    "rms_snap_error",
    "pass",
    "strict_pass",
    "notes",
]

COMPARE_FIELDS = [
    "group",
    "model",
    "dataset",
    "seed",
    "h4_fdG",
    "h6_fdG_S3",
    "h6_trueG_S3",
    "h_oracle",
    "oracle_over_h4",
    "oracle_over_h6_fdG",
    "oracle_over_h6_trueG",
    "nmse_ratio_h4",
    "nmse_ratio_h6_fdG",
    "nmse_ratio_h6_trueG",
]


def formula_audit() -> Dict[str, Any]:
    def phi(t: float, a0: float, a1: float, a2: float, a3: float) -> float:
        return a0 + a1 * t + a2 * t * t / 2.0 + a3 * t * t * t / 6.0

    a0, a1, a2, a3 = 0.7, -1.3, 2.4, -5.5
    h = 0.037
    stencil = (phi(2 * h, a0, a1, a2, a3) - 2 * phi(h, a0, a1, a2, a3) + 2 * phi(-h, a0, a1, a2, a3) - phi(-2 * h, a0, a1, a2, a3)) / (2 * h**3)
    A, B = 2.3, 7.1
    h_min = (A / (2.0 * B)) ** (1.0 / 6.0)
    delta, G, S3 = 0.03, 4.2, 8.7
    A2 = delta * delta * G * G / 4.0
    B2 = S3 / 36.0
    h6 = ((9.0 / 2.0) * delta * delta * G * G / S3) ** (1.0 / 6.0)
    h6_from_envelope = (A2 / (2.0 * B2)) ** (1.0 / 6.0)
    return {
        "third_stencil_expected": a3,
        "third_stencil_observed": stencil,
        "third_stencil_abs_error": abs(stencil - a3),
        "third_stencil_pass": bool(abs(stencil - a3) < 1e-10),
        "generic_envelope_A": A,
        "generic_envelope_B": B,
        "generic_minimizer": h_min,
        "h6_formula": h6,
        "h6_from_envelope": h6_from_envelope,
        "h6_abs_error": abs(h6 - h6_from_envelope),
        "h6_minimizer_pass": bool(abs(h6 - h6_from_envelope) < 1e-12),
        "h6_constant": "((9/2) * Delta^2 * G^2 / S3_sq)^(1/6)",
        "rho3_constant": "(9 * Delta^2 * G^2 / (2 * rho3_q90^2 * d*(d+2)*(d+4)))^(1/6)",
        "rho3_gate_constant": "(9 * Delta^2 * G^2 / (128 * rho3_q90^2 * d*(d+2)*(d+4)))^(1/6)",
    }


def load_previous_l(input_package: Path) -> Dict[Tuple[str, str, int], Dict[str, float]]:
    path = input_package / "hstar_components.csv"
    out: Dict[Tuple[str, str, int], Dict[str, float]] = {}
    if not path.exists():
        return out
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if not (
                row.get("G_method") == "absG"
                and row.get("L_mode") == "L_clean32"
                and row.get("L_q") == "q90"
            ):
                continue
            key = (row.get("model", ""), row.get("dataset", ""), int(float(row.get("seed", "0"))))
            out[key] = {
                "L_hat": float(row.get("L_hat", "nan")),
                "h2_L": float(row.get("h2_L", "nan")),
            }
    return out


def grad_norm(grads: Sequence[Any]) -> float:
    total = 0.0
    for g in grads:
        if g is not None:
            total += float((g.detach().float() * g.detach().float()).sum().item())
    return math.sqrt(max(total, 0.0))


def h4_star(delta: float, g: float, lval: float, d: int) -> float:
    if min(delta, g, lval, float(d)) <= 0 or not all(map(math.isfinite, [delta, g, lval])):
        return float("nan")
    return float((delta * delta * g * g / (16.0 * lval * lval * float(d) * float(d + 2))) ** 0.25)


def h6_s3_star(delta: float, g: float, s3_sq: float) -> float:
    if min(delta, g, s3_sq) <= 0 or not all(map(math.isfinite, [delta, g, s3_sq])):
        return float("nan")
    return float(((9.0 / 2.0) * delta * delta * g * g / s3_sq) ** (1.0 / 6.0))


def h6_rho_star(delta: float, g: float, rho: float, d: int, gate: bool = False) -> float:
    if min(delta, g, rho, float(d)) <= 0 or not all(map(math.isfinite, [delta, g, rho])):
        return float("nan")
    const = 128.0 if gate else 2.0
    denom = const * rho * rho * float(d) * float(d + 2) * float(d + 4)
    return float((9.0 * delta * delta * g * g / denom) ** (1.0 / 6.0))


def compute_g_fd(
    ctx: Any,
    fd_by_h: Dict[float, List[float]],
    vis_by_h: Dict[float, Dict[str, float]],
    g_true: float,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    cand: List[Dict[str, Any]] = []
    for h in H_GRID:
        fd = np.asarray(fd_by_h[float(h)], dtype=np.float64)
        fin = fd[np.isfinite(fd)]
        g_abs = float(math.sqrt(math.pi / 2.0) * np.mean(np.abs(fin))) if fin.size else float("nan")
        g_rms = float(math.sqrt(np.mean(fin * fin))) if fin.size else float("nan")
        h2 = next((x for x in H_GRID if abs(float(x) - 2.0 * float(h)) <= max(1e-12, 1e-9 * abs(2.0 * float(h)))), None)
        corr2 = finite_corr(fd_by_h[float(h)], fd_by_h[float(h2)]) if h2 is not None and float(h2) in fd_by_h else float("nan")
        flip = sign_flip_rate(fd_by_h[float(h)], fd_by_h[float(h2)]) if h2 is not None and float(h2) in fd_by_h else float("nan")
        rel = float("nan")
        if h2 is not None and float(h2) in fd_by_h:
            fd2 = np.asarray(fd_by_h[float(h2)], dtype=np.float64)
            fin2 = fd2[np.isfinite(fd2)]
            g2 = float(math.sqrt(math.pi / 2.0) * np.mean(np.abs(fin2))) if fin2.size else float("nan")
            rel = abs(g_abs - g2) / (abs(g_abs) + EPS) if math.isfinite(g_abs) and math.isfinite(g2) else float("nan")
        vis = vis_by_h.get(float(h), {})
        stable = (
            visibility_pass(vis)
            and math.isfinite(g_abs)
            and (h2 is None or not math.isfinite(corr2) or (corr2 >= 0.90 and flip <= 0.10))
        )
        score = 0.0
        a = safe_float(vis.get("alignment_eff"))
        n = safe_float(vis.get("norm_ratio_eff"))
        z = safe_float(vis.get("zero_coord_frac_eff"))
        if a is not None:
            score += max(0.0, 0.99 - a)
        else:
            score += 10.0
        if n is not None:
            score += abs(n - 1.0)
        else:
            score += 10.0
        if z is not None:
            score += z
        else:
            score += 10.0
        if math.isfinite(corr2):
            score += max(0.0, 0.90 - corr2)
        if math.isfinite(flip):
            score += max(0.0, flip - 0.10)
        cand.append(
            {
                "h": float(h),
                "G_abs": g_abs,
                "G_rms": g_rms,
                "corr_d2_h_2h": corr2,
                "sign_flip_rate_h_2h": flip,
                "relative_G_change": rel,
                "stable": stable,
                "score": score,
                **vis,
            }
        )
    stable_rows = [r for r in cand if r["stable"]]
    fallback = ""
    if stable_rows:
        g_multi = float(np.median([r["G_abs"] for r in stable_rows]))
        h_values = [r["h"] for r in stable_rows]
        g_small = min(stable_rows, key=lambda r: r["h"])["G_abs"]
    else:
        best = min(cand, key=lambda r: float(r["score"]))
        g_multi = float(best["G_abs"])
        h_values = [float(best["h"])]
        g_small = float(best["G_abs"])
        fallback = "fallback_G_fd"
    by_h = {float(r["h"]): r for r in cand}
    g_3e4 = by_h.get(3e-4, {}).get("G_abs", float("nan"))
    g_1e3 = by_h.get(1e-3, {}).get("G_abs", float("nan"))
    row = {
        "group": ctx.setting.group,
        "model": ctx.setting.model_label,
        "dataset": ctx.setting.dataset,
        "seed": ctx.setting.seed,
        "G_true": g_true,
        "G_true_available": math.isfinite(g_true),
        "grad_norm_dtype": "fp32",
        "d_trainable": ctx.d_trainable,
        "all_trainable_params_included": True,
        "G_fd_multi": g_multi,
        "G_fd_at_3e-4": g_3e4,
        "G_fd_at_1e-3": g_1e3,
        "G_fd_smallest_stable": g_small,
        "G_fd_over_true": g_multi / (g_true + EPS) if math.isfinite(g_true) else float("nan"),
        "relative_error_G_fd": abs(g_multi - g_true) / (g_true + EPS) if math.isfinite(g_true) else float("nan"),
        "selected_hG_values": ";".join(f"{x:.12g}" for x in h_values),
        "fallback_flags": fallback,
    }
    return row, cand


def third_order_one(ctx: Any, seed: int, h3: float) -> Tuple[float, float]:
    restore(ctx)
    norm_sq = direction_norm_sq(ctx, int(seed))
    restore(ctx)
    apply_signed(ctx, int(seed), 2.0 * float(h3), +1.0)
    p2 = compute_loss(ctx)
    restore(ctx)
    apply_signed(ctx, int(seed), float(h3), +1.0)
    p1 = compute_loss(ctx)
    restore(ctx)
    apply_signed(ctx, int(seed), float(h3), -1.0)
    m1 = compute_loss(ctx)
    restore(ctx)
    apply_signed(ctx, int(seed), 2.0 * float(h3), -1.0)
    m2 = compute_loss(ctx)
    restore(ctx)
    t3 = (p2 - 2.0 * p1 + 2.0 * m1 - m2) / (2.0 * float(h3) ** 3)
    rho = abs(float(t3)) / (max(norm_sq, 0.0) ** 1.5 + EPS)
    return float(t3), float(rho)


def summarize_t3_rows(rows: List[Dict[str, Any]]) -> None:
    rows.sort(key=lambda r: float(r["h3"]))
    by_h = {float(r["h3"]): r for r in rows}
    hs = [float(r["h3"]) for r in rows]
    for i, row in enumerate(rows):
        h = float(row["h3"])
        h2 = next((x for x in hs if abs(x - 2.0 * h) <= max(1e-12, 1e-9 * abs(2.0 * h))), None)
        if h2 is not None:
            other = by_h[h2]
            row["S3_sq_stability_2x"] = abs(float(row["S3_sq"]) - float(other["S3_sq"])) / (abs(float(row["S3_sq"])) + EPS)
            row["rho3_q90_stability_2x"] = abs(float(row["rho3_q90"]) - float(other["rho3_q90"])) / (abs(float(row["rho3_q90"])) + EPS)
        if i + 1 < len(rows):
            nxt = rows[i + 1]
            row["S3_sq_stability_next"] = abs(float(row["S3_sq"]) - float(nxt["S3_sq"])) / (abs(float(row["S3_sq"])) + EPS)
            row["rho3_q90_stability_next"] = abs(float(row["rho3_q90"]) - float(nxt["rho3_q90"])) / (abs(float(row["rho3_q90"])) + EPS)
            row["log_slope_S3_next"] = abs(math.log(float(nxt["S3_sq"]) + EPS) - math.log(float(row["S3_sq"]) + EPS)) / abs(math.log(float(nxt["h3"])) - math.log(h))
            row["log_slope_rho_next"] = abs(math.log(float(nxt["rho3_q90"]) + EPS) - math.log(float(row["rho3_q90"]) + EPS)) / abs(math.log(float(nxt["h3"])) - math.log(h))
    for i, row in enumerate(rows):
        s3 = safe_float(row.get("S3_sq"))
        rho = safe_float(row.get("rho3_q90"))
        low = False
        if safe_float(row.get("finite_rate")) is not None and float(row["finite_rate"]) < 0.95:
            low = True
        if s3 is not None:
            larger = [float(r["S3_sq"]) for r in rows if float(r["h3"]) > float(row["h3"]) and safe_float(r.get("S3_sq")) is not None]
            if larger and s3 / (float(np.median(larger)) + EPS) >= 5.0:
                low = True
            if i + 1 < len(rows) and s3 / (float(rows[i + 1]["S3_sq"]) + EPS) >= 5.0:
                low = True
        if rho is not None and i + 1 < len(rows) and rho / (float(rows[i + 1]["rho3_q90"]) + EPS) >= 5.0:
            low = True
        row["low_h3_noise_suspected"] = bool(low)
        row["large_h3_nonlocal_suspected"] = bool(i >= len(rows) - 2 and safe_float(row.get("log_slope_S3_next")) is not None and float(row["log_slope_S3_next"]) > 2.0)


def compute_t3(ctx: Any, seeds: Sequence[int]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    import torch

    ctx.model.float()
    ctx.forward_precision = "fp32"
    ctx.direction_dtype_name = "float16"
    reset_backups(ctx)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    rows: List[Dict[str, Any]] = []
    for h3 in H_GRID:
        vals: List[float] = []
        rhos: List[float] = []
        t0 = time.time()
        for seed in seeds:
            t3, rho = third_order_one(ctx, int(seed), float(h3))
            vals.append(t3)
            rhos.append(rho)
        arr = np.asarray(vals, dtype=np.float64)
        rho_arr = np.asarray(rhos, dtype=np.float64)
        fin = arr[np.isfinite(arr)]
        fin_rho = rho_arr[np.isfinite(rho_arr)]
        rows.append(
            {
                "group": ctx.setting.group,
                "model": ctx.setting.model_label,
                "dataset": ctx.setting.dataset,
                "seed": ctx.setting.seed,
                "h3": float(h3),
                "m_T3": len(seeds),
                "S3_sq": float(np.mean(fin * fin)) if fin.size else float("nan"),
                "S3_rms": float(math.sqrt(np.mean(fin * fin))) if fin.size else float("nan"),
                "T3_abs_mean": float(np.mean(np.abs(fin))) if fin.size else float("nan"),
                "T3_abs_median": float(np.median(np.abs(fin))) if fin.size else float("nan"),
                "rho3_q50": float(np.quantile(fin_rho, 0.50)) if fin_rho.size else float("nan"),
                "rho3_q90": float(np.quantile(fin_rho, 0.90)) if fin_rho.size else float("nan"),
                "rho3_q95": float(np.quantile(fin_rho, 0.95)) if fin_rho.size else float("nan"),
                "finite_rate": float(np.mean(np.isfinite(arr))) if arr.size else 0.0,
                "zero_T3_frac": float(np.mean(np.abs(fin) < EPS)) if fin.size else float("nan"),
                "sign_negative_frac": float(np.mean(fin < 0.0)) if fin.size else float("nan"),
                "S3_sq_stability_2x": float("nan"),
                "S3_sq_stability_next": float("nan"),
                "rho3_q90_stability_2x": float("nan"),
                "rho3_q90_stability_next": float("nan"),
                "log_slope_S3_next": float("nan"),
                "log_slope_rho_next": float("nan"),
                "low_h3_noise_suspected": False,
                "large_h3_nonlocal_suspected": False,
            }
        )
        print(
            f"[T3] {ctx.setting.group}/{ctx.setting.model_label}/{ctx.setting.dataset}/seed{ctx.setting.seed} "
            f"h3={h3:g} elapsed={time.time()-t0:.1f}s",
            flush=True,
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    summarize_t3_rows(rows)
    stable = [
        r
        for r in rows
        if float(r["finite_rate"]) >= 0.95
        and not bool(r["low_h3_noise_suspected"])
        and not bool(r["large_h3_nonlocal_suspected"])
        and (
            (math.isfinite(float(r["S3_sq_stability_next"])) and float(r["S3_sq_stability_next"]) <= 0.5)
            or (math.isfinite(float(r["S3_sq_stability_2x"])) and float(r["S3_sq_stability_2x"]) <= 0.5)
        )
    ]
    flags = ""
    if not stable:
        stable = [
            min(
                rows,
                key=lambda r: (
                    bool(r["low_h3_noise_suspected"]),
                    bool(r["large_h3_nonlocal_suspected"]),
                    safe_float(r.get("S3_sq_stability_next")) or float("inf"),
                    float(r["h3"]),
                ),
            )
        ]
        flags = "fallback_T3"
    s3_vals = [float(r["S3_sq"]) for r in stable if math.isfinite(float(r["S3_sq"]))]
    rho_vals = [float(r["rho3_q90"]) for r in stable if math.isfinite(float(r["rho3_q90"]))]
    selected = {
        "group": ctx.setting.group,
        "model": ctx.setting.model_label,
        "dataset": ctx.setting.dataset,
        "seed": ctx.setting.seed,
        "selected_h3_values": ";".join(f"{float(r['h3']):.12g}" for r in stable),
        "S3_sq_multi": float(np.median(s3_vals)) if s3_vals else float("nan"),
        "S3_rms_multi": float(math.sqrt(np.median(s3_vals))) if s3_vals else float("nan"),
        "rho3_q90_selected": float(np.median(rho_vals)) if rho_vals else float("nan"),
        "selection_status": "selected" if not flags else "fallback_unreliable",
        "fallback_flags": flags,
    }
    return rows, selected


def evaluate_variant(
    ctx: Any,
    seeds: Sequence[int],
    d_true: Sequence[float],
    formula: Dict[str, Any],
    empirical_min: Dict[str, Any],
    empirical_corr: Dict[str, Any],
    visibility_dirs: int,
) -> Dict[str, Any]:
    h = float(formula.get("hstar_cont", float("nan")))
    if math.isfinite(h) and h > 0.0:
        fds = compute_fd_values(ctx, seeds, h)
        met = metric_row(fds, d_true)
        vis = average_visibility(ctx, seeds, h, visibility_dirs)
    else:
        met = {
            "mse": float("nan"),
            "nmse": float("nan"),
            "corr": float("nan"),
            "bias": float("nan"),
            "mae": float("nan"),
            "median_abs_error": float("nan"),
        }
        vis = {}
    emp_h = float(empirical_min["h"])
    emp_nmse = float(empirical_min["nmse"])
    emp_corr = float(empirical_corr["corr"])
    nmse_ratio = float(met["nmse"] / (emp_nmse + EPS)) if math.isfinite(met["nmse"]) else float("nan")
    corr_gap = float(emp_corr - met["corr"]) if math.isfinite(met["corr"]) and math.isfinite(emp_corr) else float("nan")
    return {
        "group": ctx.setting.group,
        "model": ctx.setting.model_label,
        "dataset": ctx.setting.dataset,
        "seed": ctx.setting.seed,
        **formula,
        "empirical_min_nmse_h": emp_h,
        "h_over_oracle": float(h / emp_h) if math.isfinite(h) and emp_h > 0 else float("nan"),
        "oracle_over_h": float(emp_h / h) if math.isfinite(h) and h > 0 else float("nan"),
        "empirical_min_nmse": emp_nmse,
        "empirical_max_corr_h": empirical_corr["h"],
        "empirical_max_corr": emp_corr,
        **met,
        "nmse_ratio": nmse_ratio,
        "corr_gap": corr_gap,
        "alignment_eff": vis.get("alignment_eff"),
        "norm_ratio_eff": vis.get("norm_ratio_eff"),
        "zero_coord_frac_eff": vis.get("zero_coord_frac_eff"),
        "rms_snap_error": vis.get("rms_snap_error"),
        "pass": bool((math.isfinite(nmse_ratio) and nmse_ratio <= 1.25) or (math.isfinite(corr_gap) and corr_gap <= 0.01)),
        "strict_pass": bool(math.isfinite(nmse_ratio) and nmse_ratio <= 1.10),
    }


def plot_setting(out_dir: Path, setting: Setting, grid_rows: Sequence[Dict[str, Any]], eval_rows: Sequence[Dict[str, Any]], t3_rows: Sequence[Dict[str, Any]], g_candidates: Sequence[Dict[str, Any]]) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    pdir = out_dir / "plots" / f"{setting.group}_{setting.model_label}_{setting.dataset}_seed{setting.seed}".replace("/", "_")
    pdir.mkdir(parents=True, exist_ok=True)
    hs = np.asarray([float(r["h"]) for r in grid_rows], dtype=np.float64)
    marks = [r for r in eval_rows if r.get("formula_name") in {"h4_fdG", "h6_fdG_S3", "h6_trueG_S3"}]
    oracle_h = float(min(grid_rows, key=lambda r: float(r["nmse"]))["h"]) if grid_rows else float("nan")

    def mark(ax):
        if math.isfinite(oracle_h):
            ax.axvline(oracle_h, color="black", linestyle="-", label="oracle grid")
        for r in marks:
            h = safe_float(r.get("hstar_cont"))
            if h is not None:
                ax.axvline(h, linestyle="--", label=r.get("formula_name"))

    for key, fname, ylabel in [("nmse", "nmse_vs_h_strict2pt.png", "nMSE"), ("corr", "corr_vs_h_strict2pt.png", "corr")]:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(hs, [float(r[key]) for r in grid_rows], marker="o")
        mark(ax)
        ax.set_xscale("log")
        ax.set_xlabel("h")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(pdir / fname, dpi=160)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot([float(r["h3"]) for r in t3_rows], [float(r["S3_sq"]) for r in t3_rows], marker="o")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("h3")
    ax.set_ylabel("S3_sq")
    fig.tight_layout()
    fig.savefig(pdir / "S3_sq_vs_h3.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot([float(r["h3"]) for r in t3_rows], [float(r["rho3_q90"]) for r in t3_rows], marker="o")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("h3")
    ax.set_ylabel("rho3_q90")
    fig.tight_layout()
    fig.savefig(pdir / "rho3_q90_vs_h3.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot([float(r["h"]) for r in g_candidates], [float(r["G_abs"]) for r in g_candidates], marker="o")
    ax.set_xscale("log")
    ax.set_xlabel("h_G")
    ax.set_ylabel("G_abs")
    fig.tight_layout()
    fig.savefig(pdir / "G_fd_vs_hG.png", dpi=160)
    plt.close(fig)


def summarize_markdown(eval_rows: Sequence[Dict[str, Any]], g_rows: Sequence[Dict[str, Any]], t3_sel_rows: Sequence[Dict[str, Any]], compare_rows: Sequence[Dict[str, Any]], out_dir: Path) -> str:
    by_key = {(r["model"], r["dataset"], int(r["seed"]), r["formula_name"]): r for r in eval_rows}

    def truth(v: Any) -> bool:
        return str(v).lower() in {"true", "1", "yes"}

    lines = [
        "# Strict Symmetric Two-Point H-Star Summary",
        "",
        f"Output directory: `{out_dir}`",
        "",
        "This is probe-only. The empirical oracle h is used only as a reference, not for selecting any h-star.",
        "",
        "## Table 1: h4 vs strict h6",
        "",
        "| model | dataset | seed | h4_fdG | h6_fdG_S3 | h6_trueG_S3 | h_oracle | oracle/h4 | oracle/h6_fdG | nMSE ratio h4 | nMSE ratio h6_fdG | pass h4 | pass h6 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for r in compare_rows:
        key = (r["model"], r["dataset"], int(r["seed"]))
        h4 = by_key.get((*key, "h4_fdG"), {})
        h6 = by_key.get((*key, "h6_fdG_S3"), {})
        lines.append(
            f"| {r['model']} | {r['dataset']} | {r['seed']} | {float(r['h4_fdG']):.6g} | "
            f"{float(r['h6_fdG_S3']):.6g} | {float(r['h6_trueG_S3']):.6g} | {float(r['h_oracle']):.6g} | "
            f"{float(r['oracle_over_h4']):.6g} | {float(r['oracle_over_h6_fdG']):.6g} | "
            f"{float(r['nmse_ratio_h4']):.6g} | {float(r['nmse_ratio_h6_fdG']):.6g} | "
            f"{h4.get('pass', '')} | {h6.get('pass', '')} |"
        )
    lines += [
        "",
        "## Table 2: Group comparison",
        "",
        "| group | median oracle/h4 | median oracle/h6_fdG | h4 pass rate | h6 pass rate | h4 strict pass | h6 strict pass |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for group in ["A_seed", "B_task", "C_model", "overall"]:
        xs = compare_rows if group == "overall" else [r for r in compare_rows if r["group"] == group]
        if not xs:
            lines.append(f"| {group} | n/a | n/a | n/a | n/a | n/a | n/a |")
            continue
        h4_rows = [r for r in eval_rows if r["formula_name"] == "h4_fdG" and (group == "overall" or r["group"] == group)]
        h6_rows = [r for r in eval_rows if r["formula_name"] == "h6_fdG_S3" and (group == "overall" or r["group"] == group)]
        lines.append(
            f"| {group} | {float(np.median([float(r['oracle_over_h4']) for r in xs])):.6g} | "
            f"{float(np.median([float(r['oracle_over_h6_fdG']) for r in xs])):.6g} | "
            f"{float(np.mean([truth(r['pass']) for r in h4_rows])):.3g} | "
            f"{float(np.mean([truth(r['pass']) for r in h6_rows])):.3g} | "
            f"{float(np.mean([truth(r['strict_pass']) for r in h4_rows])):.3g} | "
            f"{float(np.mean([truth(r['strict_pass']) for r in h6_rows])):.3g} |"
        )
    lines += [
        "",
        "## Table 3: G estimation",
        "",
        "| model | dataset | seed | G_true | G_fd_multi | G_fd/G_true |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for r in g_rows:
        lines.append(f"| {r['model']} | {r['dataset']} | {r['seed']} | {float(r['G_true']):.6g} | {float(r['G_fd_multi']):.6g} | {float(r['G_fd_over_true']):.6g} |")
    lines += [
        "",
        "## Table 4: T3 stability",
        "",
        "| model | dataset | seed | selected h3 values | S3_sq_multi | stability flags |",
        "|---|---|---:|---|---:|---|",
    ]
    for r in t3_sel_rows:
        lines.append(f"| {r['model']} | {r['dataset']} | {r['seed']} | {r['selected_h3_values']} | {float(r['S3_sq_multi']):.6g} | {r['fallback_flags']} |")
    lines += [
        "",
        "## Interpretation Notes",
        "",
        "- `h6_fdG_S3` is the deployable-like strict two-point formula using FP16 finite-difference G and clean-FP32 third-moment estimates.",
        "- `h6_trueG_S3` is diagnostic: it tests the formula with exact FP32 gradient norm.",
        "- `h_oracle` is only the grid min-nMSE reference and was not used to choose h.",
    ]
    return "\n".join(lines) + "\n"


def analyze_setting(setting: Setting, args: argparse.Namespace, out_dir: Path, previous_l: Dict[Tuple[str, str, int], Dict[str, float]], diagnostics: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
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
    fp32_backups = [b.detach().clone() for b in ctx.backups]
    ulp = estimate_ulp(ctx)
    _, grads, truth_kind = compute_true_grads(ctx)
    g_true = grad_norm(grads)
    max_dirs = max(args.num_probe_dirs, args.num_G_dirs, args.num_T3_dirs)
    base_seeds = direction_seeds(setting.seed, max_dirs, 0)
    probe_seeds = base_seeds[: args.num_probe_dirs]
    g_seeds = base_seeds[: args.num_G_dirs]
    t3_seeds = base_seeds[: args.num_T3_dirs]
    d_true = [true_directional_from_grads(ctx, grads, int(s)) for s in probe_seeds]

    set_mode_fp16(ctx)
    fd_by_h, vis_by_h, vis_rows, grid_rows = build_grid_probe(ctx, probe_seeds, d_true, args.visibility_dirs)
    empirical_min = min(grid_rows, key=lambda r: float(r["nmse"]))
    empirical_corr = max(grid_rows, key=lambda r: float(r["corr"]) if math.isfinite(float(r["corr"])) else -float("inf"))

    g_row, g_candidates = compute_g_fd(ctx, fd_by_h, vis_by_h, g_true)
    ctx.model.float()
    ctx.forward_precision = "fp32"
    ctx.direction_dtype_name = "float16"
    restore_external_backups(ctx, fp32_backups)
    t3_rows, t3_sel = compute_t3(ctx, t3_seeds)

    key = (setting.model_label, setting.dataset, setting.seed)
    l_info = previous_l.get(key, {})
    l_val = float(l_info.get("L_hat", float("nan")))
    delta = float(ulp.get("delta_ulp_rms", float("nan")))
    g_fd = float(g_row["G_fd_multi"])
    s3 = float(t3_sel["S3_sq_multi"])
    rho = float(t3_sel["rho3_q90_selected"])
    d = int(ctx.d_trainable)
    formulas = [
        {
            "formula_name": "h4_trueG",
            "Delta_mode": "delta_ulp_rms",
            "Delta_value": delta,
            "G_mode": "G_true",
            "G_value": g_true,
            "S3_sq": "",
            "rho3_value": "",
            "L_value_if_used": l_val,
            "hstar_cont": h4_star(delta, g_true, l_val, d),
            "notes": "fourth-root L-smooth envelope with true G",
        },
        {
            "formula_name": "h4_fdG",
            "Delta_mode": "delta_ulp_rms",
            "Delta_value": delta,
            "G_mode": "G_fd_multi",
            "G_value": g_fd,
            "S3_sq": "",
            "rho3_value": "",
            "L_value_if_used": l_val,
            "hstar_cont": h4_star(delta, g_fd, l_val, d),
            "notes": "fourth-root L-smooth envelope with finite-difference G",
        },
        {
            "formula_name": "h6_trueG_S3",
            "Delta_mode": "delta_ulp_rms",
            "Delta_value": delta,
            "G_mode": "G_true",
            "G_value": g_true,
            "S3_sq": s3,
            "rho3_value": "",
            "L_value_if_used": "",
            "hstar_cont": h6_s3_star(delta, g_true, s3),
            "notes": "strict two-point third-moment formula with true G; constant 9/2",
        },
        {
            "formula_name": "h6_fdG_S3",
            "Delta_mode": "delta_ulp_rms",
            "Delta_value": delta,
            "G_mode": "G_fd_multi",
            "G_value": g_fd,
            "S3_sq": s3,
            "rho3_value": "",
            "L_value_if_used": "",
            "hstar_cont": h6_s3_star(delta, g_fd, s3),
            "notes": "strict two-point deployable-like formula with finite-difference G; constant 9/2",
        },
        {
            "formula_name": "h6_fdG_rho3",
            "Delta_mode": "delta_ulp_rms",
            "Delta_value": delta,
            "G_mode": "G_fd_multi",
            "G_value": g_fd,
            "S3_sq": "",
            "rho3_value": rho,
            "L_value_if_used": "",
            "hstar_cont": h6_rho_star(delta, g_fd, rho, d, gate=False),
            "notes": "bound-style rho3 formula; denominator constant 2",
        },
        {
            "formula_name": "h6_fdG_rho3_gate",
            "Delta_mode": "delta_ulp_rms",
            "Delta_value": delta,
            "G_mode": "G_fd_multi",
            "G_value": g_fd,
            "S3_sq": "",
            "rho3_value": rho,
            "L_value_if_used": "",
            "hstar_cont": h6_rho_star(delta, g_fd, rho, d, gate=True),
            "notes": "conservative gate-bound rho3 formula; denominator constant 128",
        },
    ]

    set_mode_fp16(ctx)
    eval_rows = [evaluate_variant(ctx, probe_seeds, d_true, f, empirical_min, empirical_corr, args.visibility_dirs) for f in formulas]
    by_formula = {r["formula_name"]: r for r in eval_rows}
    compare = {
        "group": setting.group,
        "model": setting.model_label,
        "dataset": setting.dataset,
        "seed": setting.seed,
        "h4_fdG": by_formula.get("h4_fdG", {}).get("hstar_cont", float("nan")),
        "h6_fdG_S3": by_formula.get("h6_fdG_S3", {}).get("hstar_cont", float("nan")),
        "h6_trueG_S3": by_formula.get("h6_trueG_S3", {}).get("hstar_cont", float("nan")),
        "h_oracle": empirical_min["h"],
        "oracle_over_h4": by_formula.get("h4_fdG", {}).get("oracle_over_h", float("nan")),
        "oracle_over_h6_fdG": by_formula.get("h6_fdG_S3", {}).get("oracle_over_h", float("nan")),
        "oracle_over_h6_trueG": by_formula.get("h6_trueG_S3", {}).get("oracle_over_h", float("nan")),
        "nmse_ratio_h4": by_formula.get("h4_fdG", {}).get("nmse_ratio", float("nan")),
        "nmse_ratio_h6_fdG": by_formula.get("h6_fdG_S3", {}).get("nmse_ratio", float("nan")),
        "nmse_ratio_h6_trueG": by_formula.get("h6_trueG_S3", {}).get("nmse_ratio", float("nan")),
    }
    plot_setting(out_dir, setting, grid_rows, eval_rows, t3_rows, g_candidates)
    print(f"[setting] done {setting.group}/{setting.model_label}/{setting.dataset}/seed{setting.seed}", flush=True)
    return [
        ("G_direct_estimates", g_row),
        ("T3_selected", t3_sel),
        ("compare", compare),
        ("eval_rows", eval_rows),
        ("T3_candidates", t3_rows),
        ("grid_rows", grid_rows),
    ], {"truth_kind": truth_kind, "ulp": ulp, "G_candidates": g_candidates, "vis_rows": vis_rows}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_package", default="analysis/fp16_direct_hstar_mse_20260520_183229_combined")
    parser.add_argument("--output_dir", default="")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_probe_dirs", type=int, default=32)
    parser.add_argument("--num_G_dirs", type=int, default=32)
    parser.add_argument("--num_T3_dirs", type=int, default=16)
    parser.add_argument("--visibility_dirs", type=int, default=8)
    parser.add_argument("--skip_opt", action="store_true")
    parser.add_argument("--only_group", default="")
    parser.add_argument("--max_settings", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = Path(args.output_dir) if args.output_dir else Path("analysis") / f"fp16_strict_twopoint_directG_T3_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=False)
    (out_dir / "plots").mkdir(exist_ok=True)
    (out_dir / "env_report.txt").write_text(env_report(), encoding="utf-8")
    audit = formula_audit()
    write_json(out_dir / "formula_audit.json", audit)
    (out_dir / "formula_audit.txt").write_text("\n".join(f"{k}: {v}" for k, v in audit.items()) + "\n", encoding="utf-8")
    diagnostics: Dict[str, Any] = {
        "start_time": dt.datetime.now().isoformat(),
        "input_package": str(args.input_package),
        "num_probe_dirs": args.num_probe_dirs,
        "num_G_dirs": args.num_G_dirs,
        "num_T3_dirs": args.num_T3_dirs,
        "visibility_dirs": args.visibility_dirs,
        "warnings": ["direction counts reduced for runtime: m_probe=32, m_G=32, m_T3=16"],
        "skipped_settings": [],
    }
    if not audit["third_stencil_pass"] or not audit["h6_minimizer_pass"]:
        diagnostics["formula_audit_failed"] = True
        write_json(out_dir / "diagnostics.json", diagnostics)
        raise SystemExit("formula audit failed")

    try:
        import torch

        if not torch.cuda.is_available():
            (out_dir / "failure_report.txt").write_text("CUDA unavailable\n", encoding="utf-8")
            return 2
    except Exception as exc:
        (out_dir / "failure_report.txt").write_text(f"torch/CUDA check failed: {exc}\n", encoding="utf-8")
        return 2

    previous_l = load_previous_l(Path(args.input_package))
    settings, skipped = resolve_settings(include_opt=not args.skip_opt)
    diagnostics["skipped_settings"].extend(skipped)
    if args.only_group:
        settings = [s for s in settings if s.group == args.only_group]
    if args.max_settings > 0:
        settings = settings[: args.max_settings]

    g_rows: List[Dict[str, Any]] = []
    t3_rows_all: List[Dict[str, Any]] = []
    t3_sel_rows: List[Dict[str, Any]] = []
    eval_rows: List[Dict[str, Any]] = []
    compare_rows: List[Dict[str, Any]] = []

    for setting in settings:
        try:
            chunks, extra = analyze_setting(setting, args, out_dir, previous_l, diagnostics)
            diagnostics.setdefault("setting_diagnostics", []).append({"setting": setting.__dict__, **extra})
            for name, payload in chunks:
                if name == "G_direct_estimates":
                    g_rows.append(payload)
                elif name == "T3_selected":
                    t3_sel_rows.append(payload)
                elif name == "compare":
                    compare_rows.append(payload)
                elif name == "eval_rows":
                    eval_rows.extend(payload)
                elif name == "T3_candidates":
                    t3_rows_all.extend(payload)
        except Exception as exc:
            traceback.print_exc()
            diagnostics.setdefault("skipped_settings", []).append(
                {
                    "group": setting.group,
                    "model": setting.model_label,
                    "dataset": setting.dataset,
                    "seed": setting.seed,
                    "reason": repr(exc),
                }
            )
        write_json(out_dir / "diagnostics.json", diagnostics)
        write_csv(out_dir / "G_direct_estimates.csv", g_rows, G_FIELDS)
        write_csv(out_dir / "T3_candidates.csv", t3_rows_all, T3_CAND_FIELDS)
        write_csv(out_dir / "T3_selected.csv", t3_sel_rows, T3_SEL_FIELDS)
        write_csv(out_dir / "strict_twopoint_hstar_eval.csv", eval_rows, EVAL_FIELDS)
        write_csv(out_dir / "hcont_vs_horacle_strict2pt.csv", compare_rows, COMPARE_FIELDS)
        (out_dir / "strict_twopoint_summary.md").write_text(
            summarize_markdown(eval_rows, g_rows, t3_sel_rows, compare_rows, out_dir),
            encoding="utf-8",
        )

    diagnostics["end_time"] = dt.datetime.now().isoformat()
    write_json(out_dir / "diagnostics.json", diagnostics)
    primary_h4 = [r for r in eval_rows if r.get("formula_name") == "h4_fdG"]
    primary_h6 = [r for r in eval_rows if r.get("formula_name") == "h6_fdG_S3"]
    true_h6 = [r for r in eval_rows if r.get("formula_name") == "h6_trueG_S3"]

    def rate(rows: Sequence[Dict[str, Any]]) -> float:
        return float(np.mean([bool(r["pass"]) for r in rows])) if rows else float("nan")

    def median_oracle(rows: Sequence[Dict[str, Any]]) -> float:
        vals = [float(r["oracle_over_h"]) for r in rows if math.isfinite(float(r["oracle_over_h"]))]
        return float(np.median(vals)) if vals else float("nan")

    print(f"output directory: {out_dir}")
    print(f"total settings completed: {len(compare_rows)}")
    print(f"median oracle/h4_fdG: {median_oracle(primary_h4):.6g}")
    print(f"median oracle/h6_fdG_S3: {median_oracle(primary_h6):.6g}")
    print(f"median oracle/h6_trueG_S3: {median_oracle(true_h6):.6g}")
    print(f"h4 pass rate: {rate(primary_h4):.4g}")
    print(f"h6_fdG pass rate: {rate(primary_h6):.4g}")
    print(f"h6_trueG pass rate: {rate(true_h6):.4g}")
    print(f"skipped settings: {len(diagnostics.get('skipped_settings', []))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
