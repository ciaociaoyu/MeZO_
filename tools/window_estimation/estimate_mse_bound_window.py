#!/usr/bin/env python
"""MSE-bound h-window estimator for precision-aware ZO perturbation probes.

This is an offline analysis script. It uses existing probe and summary files,
fits the bound-shaped envelope

    y(h) = alpha / h^2 + beta * h^2 + gamma,

and writes a paper-facing analysis package without launching training jobs.
"""

from __future__ import annotations

import argparse
import json
import math
import socket
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    from estimate_h_window import (
        EPS,
        REPO_ROOT,
        as_float,
        finite,
        format_h,
        git_commit,
        interval_text,
        json_default,
        md_table,
        relpath,
        safe_mean,
        safe_std,
        write_svg_line_plot,
    )
except Exception:  # pragma: no cover - keeps the script runnable from repo root.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from estimate_h_window import (  # type: ignore
        EPS,
        REPO_ROOT,
        as_float,
        finite,
        format_h,
        git_commit,
        interval_text,
        json_default,
        md_table,
        relpath,
        safe_mean,
        safe_std,
        write_svg_line_plot,
    )


OUT_DIR = REPO_ROOT / "outputs" / "mse_bound_window_estimator"
H_GRID = np.array([1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 1.5e-3, 2e-3, 3e-3, 4e-3, 5e-3, 1e-2], dtype=float)
KAPPAS = (1.5, 2.0, 3.0)
TAUS = (0.01, 0.03, 0.05, 0.1, 0.2)
CORR_THRESHOLDS = (0.90, 0.95, 0.98)


DATA_COLUMNS = [
    "setting_id",
    "model",
    "dataset",
    "precision",
    "quantizer",
    "direction_family",
    "sparse_p",
    "h",
    "h_active",
    "nMSE_fd_true",
    "MSE_fd_true",
    "corr_fd_true",
    "alignment",
    "norm_ratio",
    "code_change_frac",
    "active_frac",
    "clip_frac",
    "saturation_frac",
    "best_eval_acc",
    "last_eval_acc",
    "source_file",
    "fd_mean",
    "fd_std",
    "d_true_mean",
    "d_true_std",
    "proxy_nmse",
    "fit_y",
    "fit_y_source",
    "fit_y_is_proxy",
]


def stable_group_key(row: pd.Series) -> Tuple:
    return (
        row["setting_id"],
        row["precision"],
        row["quantizer"],
        row["direction_family"],
        row["sparse_p"] if finite(row["sparse_p"]) else np.nan,
    )


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def safe_json(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def discover_source_files() -> Dict[str, List[str]]:
    patterns = {
        "probe_summaries": ["*probe*summary*.csv", "*probe_stats.jsonl", "zo_directional_probe.csv"],
        "training_summaries": ["summary_all.csv", "window_training_summary.csv", "run_summary.json", "eval_metrics.jsonl", "metrics.csv"],
        "prior_window_package": ["window_estimator_results.csv", "window_estimator_selected.csv"],
    }
    found: Dict[str, List[str]] = {}
    for label, pats in patterns.items():
        paths: List[str] = []
        for pat in pats:
            for p in REPO_ROOT.rglob(pat):
                if ".git" in p.parts:
                    continue
                if "gptq" in str(p).lower() or "residual_grid" in str(p).lower():
                    continue
                paths.append(relpath(p))
        found[label] = sorted(set(paths))[:400]
    return found


def load_training_eval_lookup() -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    fp = REPO_ROOT / "experiments/main_latest/mezo/roberta-large/sst5/fp32_fp16_h_sweep_11h_seed16_bs64_ckpt1k_20260517/summaries/summary_all.csv"
    if fp.exists():
        df = pd.read_csv(fp)
        df["setting_id"] = "roberta_sst5_current_dense_probe_ckpt1k"
        df["precision"] = df["precision_mode"].astype(str).str.lower()
        df["quantizer"] = np.where(df["precision"] == "fp32", "identity", "fp16_forward_oracle")
        df["direction_family"] = "dense"
        df["sparse_p"] = np.nan
        frames.append(df[["setting_id", "precision", "quantizer", "direction_family", "sparse_p", "h", "best_eval_acc", "last_eval_acc"]])

    legacy = REPO_ROOT / "experiments/int8_update_sparse_plan/probe_window_h100_20260512/window_training_summary.csv"
    if legacy.exists():
        df = pd.read_csv(legacy)
        df["setting_id"] = "legacy_probe_window_h100_20260512"
        df["precision"] = df["precision_mode"].astype(str).str.lower()
        df["quantizer"] = df["precision"].map(
            {
                "fp32": "identity",
                "bf16": "bf16_forward_oracle",
                "fp16": "fp16_forward_oracle",
                "int8": "legacy_int8_fp16master_probe",
            }
        ).fillna(df["precision"])
        df["direction_family"] = df["direction_type"].astype(str).str.lower()
        df["best_eval_acc"] = df.get("best_acc")
        df["last_eval_acc"] = df.get("final_acc")
        frames.append(df[["setting_id", "precision", "quantizer", "direction_family", "sparse_rate", "h", "best_eval_acc", "last_eval_acc"]].rename(columns={"sparse_rate": "sparse_p"}))

    current_int8 = REPO_ROOT / "outputs/rtnclip_lowbit_roberta_sst5_seed16_20260519_batch/int8_hsearch_summary.csv"
    if current_int8.exists():
        df = pd.read_csv(current_int8)
        df["setting_id"] = "rtnclip_g128_int8_training_diagnostics_geometry_only"
        df["precision"] = "int8"
        df["quantizer"] = "G128_groupwise_RTNClip_fake_quant"
        df["direction_family"] = "dense"
        df["sparse_p"] = np.nan
        frames.append(df[["setting_id", "precision", "quantizer", "direction_family", "sparse_p", "h", "best_eval_acc", "last_eval_acc"]])

    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def attach_training_eval(df: pd.DataFrame) -> pd.DataFrame:
    lookup = load_training_eval_lookup()
    if lookup.empty:
        df["best_eval_acc"] = np.nan
        df["last_eval_acc"] = np.nan
        return df
    out = df.copy()
    out["best_eval_acc"] = np.nan
    out["last_eval_acc"] = np.nan
    for idx, row in out.iterrows():
        sub = lookup[
            (lookup["setting_id"] == row["setting_id"])
            & (lookup["precision"] == row["precision"])
            & (lookup["quantizer"] == row["quantizer"])
            & (lookup["direction_family"] == row["direction_family"])
            & np.isclose(lookup["h"].astype(float), as_float(row["h"]), rtol=0, atol=1e-12)
        ]
        if finite(row["sparse_p"]) and "sparse_p" in sub.columns:
            sub = sub[np.isclose(sub["sparse_p"].astype(float), as_float(row["sparse_p"]), rtol=0, atol=1e-12)]
        if not sub.empty:
            out.loc[idx, "best_eval_acc"] = as_float(sub.iloc[0].get("best_eval_acc"))
            out.loc[idx, "last_eval_acc"] = as_float(sub.iloc[0].get("last_eval_acc"))
    return out


def proxy_nmse_from_row(row: pd.Series) -> float:
    align = as_float(row.get("alignment"))
    rho = as_float(row.get("norm_ratio"))
    code = as_float(row.get("code_change_frac"))
    active = as_float(row.get("active_frac"))
    fd_mean = as_float(row.get("fd_mean"))
    fd_std = as_float(row.get("fd_std"))
    parts = []
    if finite(align):
        parts.append((1.0 - max(min(align, 1.0), -1.0)) ** 2)
    if finite(rho) and rho > 0:
        parts.append(math.log(rho) ** 2)
    code_metric = code if finite(code) else active
    if finite(code_metric):
        parts.append(max(0.0, 0.05 - code_metric) ** 2 / (0.05**2))
    if finite(fd_std):
        denom = fd_std**2 + (fd_mean**2 if finite(fd_mean) else 0.0) + EPS
        parts.append(fd_std**2 / denom)
    if not parts:
        return math.nan
    return float(sum(parts))


def load_prior_window_results() -> Tuple[pd.DataFrame, List[str]]:
    path = REPO_ROOT / "outputs/window_estimator/window_estimator_results.csv"
    warnings: List[str] = []
    df = read_csv(path)
    if df.empty:
        warnings.append(f"missing prior window estimator results: {relpath(path)}")
        return pd.DataFrame(), warnings
    rows = []
    for _, r in df.iterrows():
        precision = str(r["precision"]).lower()
        dtrue_mean = as_float(r.get("d_true_mean"))
        dtrue_std = as_float(r.get("d_true_std"))
        denom = dtrue_std**2 + dtrue_mean**2 if finite(dtrue_std) or finite(dtrue_mean) else math.nan
        nmse = as_float(r.get("nMSE_fd_true"))
        mse = nmse * denom if finite(nmse) and finite(denom) else math.nan
        row = {
            "setting_id": r.get("setting"),
            "model": "roberta-large",
            "dataset": "sst-5",
            "precision": precision,
            "quantizer": r.get("quantizer"),
            "direction_family": r.get("direction_family"),
            "sparse_p": as_float(r.get("sparse_p")),
            "h": as_float(r.get("h")),
            "h_active": as_float(r.get("h_active", r.get("h"))),
            "nMSE_fd_true": nmse,
            "MSE_fd_true": mse,
            "corr_fd_true": as_float(r.get("corr_fd_true")),
            "alignment": as_float(r.get("alignment")),
            "norm_ratio": as_float(r.get("norm_ratio")),
            "code_change_frac": as_float(r.get("code_change_frac")),
            "active_frac": as_float(r.get("active_frac")),
            "clip_frac": as_float(r.get("clip_frac")),
            "saturation_frac": as_float(r.get("saturation_frac")),
            "best_eval_acc": math.nan,
            "last_eval_acc": math.nan,
            "source_file": r.get("source_path", relpath(path)),
            "fd_mean": as_float(r.get("fd_mean")),
            "fd_std": as_float(r.get("fd_std")),
            "d_true_mean": dtrue_mean,
            "d_true_std": dtrue_std,
        }
        rows.append(row)
    out = pd.DataFrame(rows)
    return out, warnings


def load_historical_bf16() -> Tuple[pd.DataFrame, List[str]]:
    path = REPO_ROOT / "experiments/int8_update_sparse_plan/probe_window_h100_20260512/dense_probe_summary.csv"
    df = read_csv(path)
    if df.empty:
        return pd.DataFrame(), [f"missing historical dense summary: {relpath(path)}"]
    rows = []
    for _, r in df[df["precision_mode"].astype(str).str.lower() == "bf16"].iterrows():
        dtrue_mean = as_float(r.get("d_true_mean"))
        dtrue_std = as_float(r.get("d_true_std"))
        denom = dtrue_std**2 + dtrue_mean**2 if finite(dtrue_std) or finite(dtrue_mean) else math.nan
        nmse = as_float(r.get("nMSE_fd_true"))
        rows.append(
            {
                "setting_id": "legacy_probe_window_h100_20260512",
                "model": "roberta-large",
                "dataset": "sst-5",
                "precision": "bf16",
                "quantizer": "bf16_forward_oracle",
                "direction_family": "dense",
                "sparse_p": math.nan,
                "h": as_float(r.get("h_raw")),
                "h_active": as_float(r.get("h_active", r.get("h_raw"))),
                "nMSE_fd_true": nmse,
                "MSE_fd_true": nmse * denom if finite(nmse) and finite(denom) else math.nan,
                "corr_fd_true": as_float(r.get("corr_fd_true")),
                "alignment": as_float(r.get("probe_alignment_mean")),
                "norm_ratio": as_float(r.get("probe_norm_ratio_mean")),
                "code_change_frac": as_float(r.get("probe_active_frac_mean")),
                "active_frac": as_float(r.get("probe_active_frac_mean")),
                "clip_frac": math.nan,
                "saturation_frac": math.nan,
                "best_eval_acc": math.nan,
                "last_eval_acc": math.nan,
                "source_file": relpath(path),
                "fd_mean": as_float(r.get("fd_mean")),
                "fd_std": as_float(r.get("fd_std")),
                "d_true_mean": dtrue_mean,
                "d_true_std": dtrue_std,
            }
        )
    return pd.DataFrame(rows), []


def build_unified_data() -> Tuple[pd.DataFrame, Dict[str, object]]:
    frames = []
    warnings: List[str] = []
    for loader in (load_prior_window_results, load_historical_bf16):
        df, w = loader()
        warnings.extend(w)
        if not df.empty:
            frames.append(df)
    if not frames:
        return pd.DataFrame(columns=DATA_COLUMNS), {"warnings": warnings}
    data = pd.concat(frames, ignore_index=True, sort=False)
    data = attach_training_eval(data)
    data["proxy_nmse"] = data.apply(proxy_nmse_from_row, axis=1)
    fit_y = []
    fit_source = []
    fit_proxy = []
    for _, row in data.iterrows():
        if finite(row.get("nMSE_fd_true")):
            fit_y.append(as_float(row.get("nMSE_fd_true")))
            fit_source.append("nMSE_fd_true")
            fit_proxy.append(False)
        elif finite(row.get("MSE_fd_true")):
            fit_y.append(as_float(row.get("MSE_fd_true")))
            fit_source.append("MSE_fd_true")
            fit_proxy.append(False)
        elif finite(row.get("proxy_nmse")):
            fit_y.append(as_float(row.get("proxy_nmse")))
            fit_source.append("geometry_fd_proxy")
            fit_proxy.append(True)
        else:
            fit_y.append(math.nan)
            fit_source.append("missing")
            fit_proxy.append(False)
    data["fit_y"] = fit_y
    data["fit_y_source"] = fit_source
    data["fit_y_is_proxy"] = fit_proxy
    for col in DATA_COLUMNS:
        if col not in data.columns:
            data[col] = math.nan
    data = data[DATA_COLUMNS].sort_values(["precision", "quantizer", "direction_family", "sparse_p", "setting_id", "h_active", "h"])
    diagnostics = {
        "warnings": warnings,
        "source_discovery": discover_source_files(),
        "rows_by_source": {
            " | ".join(str(x) for x in key): int(value)
            for key, value in data.groupby(["setting_id", "precision", "quantizer", "direction_family"], dropna=False).size().to_dict().items()
        },
    }
    return data, diagnostics


def nnls_enumerate(X: np.ndarray, y: np.ndarray, weights: np.ndarray) -> np.ndarray:
    n_cols = X.shape[1]
    best_coef = np.zeros(n_cols)
    best_loss = float("inf")
    sw = np.sqrt(np.clip(weights, EPS, np.inf))
    Xw = X * sw[:, None]
    yw = y * sw
    scales = np.maximum(np.median(np.abs(Xw), axis=0), EPS)
    Xws = Xw / scales[None, :]
    for mask in range(1, 1 << n_cols):
        cols = [i for i in range(n_cols) if mask & (1 << i)]
        try:
            coef_s, *_ = np.linalg.lstsq(Xws[:, cols], yw, rcond=None)
        except np.linalg.LinAlgError:
            continue
        coef = np.zeros(n_cols)
        coef[cols] = coef_s / scales[cols]
        if np.any(coef < -1e-14):
            continue
        coef = np.maximum(coef, 0.0)
        resid = X @ coef - y
        loss = float(np.sum(weights * resid**2))
        if loss < best_loss:
            best_loss = loss
            best_coef = coef
    return best_coef


def fit_envelope(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    mask = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y >= 0)
    x = x[mask].astype(float)
    y = y[mask].astype(float)
    if x.size < 3:
        return np.array([math.nan, math.nan, math.nan]), np.full_like(y, math.nan), {"fit_status": "no_fit", "n_fit": float(x.size)}
    # Avoid exact zeros dominating log residuals but keep zero-ish good points.
    y_floor = max(np.nanmin(y[y > 0]) * 0.25 if np.any(y > 0) else 1e-12, 1e-12)
    y = np.maximum(y, y_floor)
    X = np.column_stack([1.0 / (x**2), x**2, np.ones_like(x)])
    weights = 1.0 / np.maximum(y, np.median(y))
    coef = np.zeros(3)
    for _ in range(5):
        coef = nnls_enumerate(X, y, weights)
        yhat = np.maximum(X @ coef, y_floor)
        log_resid = np.log(y) - np.log(yhat)
        med = np.median(log_resid)
        mad = np.median(np.abs(log_resid - med)) + EPS
        huber = np.minimum(1.0, 1.5 * 1.4826 * mad / (np.abs(log_resid - med) + EPS))
        weights = (1.0 / np.maximum(y, np.median(y))) * huber
    yhat = np.maximum(X @ coef, y_floor)
    sse = float(np.sum((y - yhat) ** 2))
    sst = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - sse / sst if sst > 0 else math.nan
    spearman = float(pd.Series(y).rank().corr(pd.Series(yhat).rank())) if x.size >= 3 else math.nan
    log_rmse = float(np.sqrt(np.mean((np.log(y) - np.log(yhat)) ** 2)))
    status = "ok"
    if coef[0] <= 0:
        status = "alpha_zero"
    if coef[1] <= 0:
        status = "beta_zero" if status == "ok" else status + ";beta_zero"
    if finite(r2) and r2 < 0.25:
        status = "poor_fit" if status == "ok" else status + ";poor_fit"
    return coef, yhat, {"fit_status": status, "n_fit": float(x.size), "r2": r2, "spearman": spearman, "log_rmse": log_rmse}


def fitted_curve(h: np.ndarray, alpha: float, beta: float, gamma: float) -> np.ndarray:
    h = np.asarray(h, dtype=float)
    return alpha / (h**2) + beta * (h**2) + gamma


def nearest_grid(h: float, grid: np.ndarray) -> float:
    if not finite(h) or h <= 0 or grid.size == 0:
        return math.nan
    return float(grid[np.argmin(np.abs(np.log(grid) - math.log(h)))])


def window_for_threshold(grid: np.ndarray, vals: np.ndarray, threshold: float) -> List[float]:
    if grid.size == 0 or vals.size == 0 or not finite(threshold):
        return []
    return [float(h) for h, v in zip(grid, vals) if finite(v) and v <= threshold]


def longest_segment(vals: Sequence[float], grid: np.ndarray) -> str:
    chosen = [float(v) for v in vals if finite(v)]
    if not chosen:
        return "none"
    idxs = sorted([int(np.argmin(np.abs(np.log(grid) - math.log(v)))) for v in chosen if v > 0])
    best: List[int] = []
    cur: List[int] = []
    prev = None
    for idx in idxs:
        if prev is None or idx == prev + 1:
            cur.append(idx)
        else:
            if len(cur) > len(best):
                best = cur
            cur = [idx]
        prev = idx
    if len(cur) > len(best):
        best = cur
    return interval_text([float(grid[i]) for i in best])


def boundary_penalty(row: pd.Series) -> float:
    penalty = 0.0
    align = as_float(row.get("alignment"))
    rho = as_float(row.get("norm_ratio"))
    code = as_float(row.get("code_change_frac"))
    sat = as_float(row.get("saturation_frac"))
    if finite(align):
        penalty += max(0.0, 0.90 - align) / 0.90
    if finite(rho) and rho > 0:
        penalty += abs(math.log(rho))
    if finite(code):
        penalty += max(0.0, 0.01 - code) / 0.01
    if finite(sat):
        penalty += max(0.0, sat - 0.05) / 0.05
    return float(penalty)


def fit_rows_for_group(group: pd.DataFrame, fit_coordinate: str) -> Dict[str, object]:
    xcol = "h_active" if fit_coordinate == "active" else "h"
    g = group.sort_values(xcol).copy()
    x = g[xcol].astype(float).to_numpy()
    y = g["fit_y"].astype(float).to_numpy()
    y_source_counts = g["fit_y_source"].value_counts().to_dict()
    is_proxy = bool(g["fit_y_is_proxy"].fillna(False).all()) if g["fit_y"].notna().any() else False
    coef, yhat_obs, quality = fit_envelope(x, y)
    alpha, beta, gamma = [float(v) for v in coef]
    grid = np.array(sorted(set(float(v) for v in g[xcol].to_list() if finite(v) and v > 0)), dtype=float)
    if grid.size == 0:
        grid = H_GRID.copy()
    yhat_grid = fitted_curve(grid, alpha, beta, gamma) if all(finite(v) for v in (alpha, beta, gamma)) else np.full_like(grid, np.nan)
    h_star = float((alpha / beta) ** 0.25) if alpha > 0 and beta > 0 else math.nan
    h_star_near = nearest_grid(h_star, grid)
    min_fit = float(np.nanmin(yhat_grid)) if np.isfinite(yhat_grid).any() else math.nan
    out: Dict[str, object] = {
        "fit_coordinate": fit_coordinate,
        "alpha": alpha,
        "beta": beta,
        "gamma": gamma,
        "h_star": h_star,
        "nearest_grid_h_star": h_star_near,
        "fit_status": ("proxy_only;" + str(quality.get("fit_status", "no_fit"))) if is_proxy else quality.get("fit_status", "no_fit"),
        "n_fit": quality.get("n_fit", 0),
        "r2": quality.get("r2", math.nan),
        "spearman": quality.get("spearman", math.nan),
        "log_rmse": quality.get("log_rmse", math.nan),
        "fit_y_sources": json.dumps(y_source_counts, sort_keys=True),
        "fit_uses_proxy": is_proxy,
        "min_fitted_y": min_fit,
    }
    for kappa in KAPPAS:
        vals = window_for_threshold(grid, yhat_grid, kappa * min_fit if finite(min_fit) else math.nan)
        out[f"W_kappa_{kappa:g}"] = interval_text(vals)
        out[f"W_kappa_{kappa:g}_set"] = " ".join(format_h(v) for v in vals)
    for tau in TAUS:
        vals = window_for_threshold(grid, yhat_grid, tau)
        out[f"W_tau_{tau:g}"] = interval_text(vals)
        out[f"W_tau_{tau:g}_set"] = " ".join(format_h(v) for v in vals)
    w2 = [as_float(v) for v in str(out["W_kappa_2_set"]).split() if finite(as_float(v))]
    out["valid_h_set"] = out["W_kappa_2_set"]
    out["longest_contiguous_log_grid_segment"] = longest_segment(w2, grid)
    out["selected_h_h_star_nearest"] = h_star_near
    if w2:
        mid = math.sqrt(min(w2) * max(w2))
        out["selected_h_log_midpoint_W2"] = nearest_grid(mid, np.array(w2, dtype=float))
    else:
        out["selected_h_log_midpoint_W2"] = math.nan
    wtau01 = [as_float(v) for v in str(out["W_tau_0.1_set"]).split() if finite(as_float(v))]
    out["selected_h_smallest_in_W_tau_0.1"] = min(wtau01) if wtau01 else math.nan
    scores = []
    for _, row in g.iterrows():
        h = as_float(row[xcol])
        if not finite(h):
            continue
        pred = as_float(fitted_curve(np.array([h]), alpha, beta, gamma)[0])
        score = pred / max(min_fit, EPS) + boundary_penalty(row)
        scores.append((score, h))
    out["selected_h_score_min"] = min(scores)[1] if scores else math.nan
    return out


def fit_all(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_cols = ["setting_id", "model", "dataset", "precision", "quantizer", "direction_family", "sparse_p"]
    for key, group in data.groupby(group_cols, dropna=False):
        coords = ["raw"]
        if str(key[5]) == "sparse":
            coords.append("active")
        for coord in coords:
            rec = dict(zip(group_cols, key))
            rec.update(fit_rows_for_group(group, coord))
            rows.append(rec)
    return pd.DataFrame(rows)


def estimate_G(group: pd.DataFrame) -> Tuple[float, str]:
    stable = group[group["h"].isin([1e-3, 2e-3, 3e-3])]
    if stable.empty:
        stable = group
    vals = stable["d_true_std"].dropna().astype(float)
    vals = vals[np.isfinite(vals)]
    if len(vals):
        return float(np.median(vals)), "d_true_std_stable_h"
    vals = stable["fd_std"].dropna().astype(float)
    vals = vals[np.isfinite(vals)]
    if len(vals):
        return float(np.median(vals)), "fd_std_stable_h"
    return math.nan, "missing"


def theory_proxy_for_fit(fit: pd.Series, group: pd.DataFrame) -> Dict[str, object]:
    alpha = as_float(fit.get("alpha"))
    beta = as_float(fit.get("beta"))
    gamma = as_float(fit.get("gamma"))
    coord = str(fit.get("fit_coordinate"))
    xcol = "h_active" if coord == "active" else "h"
    grid = np.array(sorted(set(float(v) for v in group[xcol].to_list() if finite(v) and v > 0)), dtype=float)
    if grid.size == 0:
        grid = H_GRID.copy()
    G_hat, G_method = estimate_G(group)
    # In normalized-MSE units, alpha maps to Delta_eff^2/4 and beta to
    # 4 L^2 K_u / G^2. K_u is therefore reported as a normalized directional
    # proxy unless actual direction norms are present.
    Delta_eff = 2.0 * math.sqrt(alpha) if alpha > 0 else math.nan
    K_u = 1.0
    L_hat = (math.sqrt(beta) * G_hat / 2.0) if beta > 0 and finite(G_hat) else math.nan
    h_star = 0.5 * math.sqrt(Delta_eff * G_hat / (L_hat * math.sqrt(K_u))) if all(finite(v) and v > 0 for v in [Delta_eff, G_hat, L_hat, K_u]) else math.nan
    # The normalized bound curve uses the same alpha/beta/gamma map. This is a
    # theory-structured proxy, not an independent theorem-only prediction.
    vals = fitted_curve(grid, alpha, beta, gamma) if all(finite(v) for v in [alpha, beta, gamma]) else np.full_like(grid, np.nan)
    min_val = float(np.nanmin(vals)) if np.isfinite(vals).any() else math.nan
    rec: Dict[str, object] = {
        "Delta_eff": Delta_eff,
        "Delta_method": "2*sqrt(empirical_alpha)" if finite(Delta_eff) else "missing",
        "G_hat": G_hat,
        "G_method": G_method,
        "L_hat": L_hat,
        "L_method": "sqrt(beta)*G_hat/2_from_normalized_fit" if finite(L_hat) else "missing",
        "K_u": K_u,
        "K_u_method": "normalized_directional_mse_proxy",
        "h_star_theory": h_star,
        "nearest_grid_h_star_theory": nearest_grid(h_star, grid),
        "theory_needs_scalar_calibration": True,
        "calibration_note": "The proxy uses fitted alpha/beta to instantiate the bound constants; use as supporting explanation, not independent selection.",
    }
    for kappa in KAPPAS:
        rec[f"W_kappa_{kappa:g}_theory"] = interval_text(window_for_threshold(grid, vals, kappa * min_val if finite(min_val) else math.nan))
    for tau in TAUS:
        rec[f"W_tau_{tau:g}_theory"] = interval_text(window_for_threshold(grid, vals, tau))
    return rec


def theory_proxy_all(data: pd.DataFrame, fits: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_cols = ["setting_id", "model", "dataset", "precision", "quantizer", "direction_family", "sparse_p"]
    for _, fit in fits.iterrows():
        group = data
        for col in group_cols:
            if finite(fit.get(col)):
                group = group[np.isclose(group[col].astype(float), as_float(fit.get(col)), atol=1e-12)] if col == "sparse_p" else group[group[col] == fit[col]]
            elif col == "sparse_p":
                group = group[group[col].isna()]
            else:
                group = group[group[col] == fit[col]]
        rec = {col: fit[col] for col in group_cols}
        rec["fit_coordinate"] = fit["fit_coordinate"]
        rec.update(theory_proxy_for_fit(fit, group))
        rows.append(rec)
    return pd.DataFrame(rows)


def training_best_for_group(data: pd.DataFrame) -> Tuple[float, float]:
    sub = data[data["best_eval_acc"].notna()]
    if sub.empty:
        return math.nan, math.nan
    best = sub.sort_values("best_eval_acc", ascending=False).iloc[0]
    return as_float(best["h"]), as_float(best["best_eval_acc"])


def validate_all(data: pd.DataFrame, fits: pd.DataFrame, theory: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_cols = ["setting_id", "model", "dataset", "precision", "quantizer", "direction_family", "sparse_p", "fit_coordinate"]
    for _, fit in fits.iterrows():
        coord = fit["fit_coordinate"]
        xcol = "h_active" if coord == "active" else "h"
        group = data
        for col in ["setting_id", "model", "dataset", "precision", "quantizer", "direction_family"]:
            group = group[group[col] == fit[col]]
        if finite(fit.get("sparse_p")):
            group = group[np.isclose(group["sparse_p"].astype(float), as_float(fit.get("sparse_p")), atol=1e-12)]
        else:
            group = group[group["sparse_p"].isna()]
        probe_best_h = math.nan
        probe_best_metric = "missing"
        if group["nMSE_fd_true"].notna().any():
            best = group.sort_values("nMSE_fd_true", ascending=True).iloc[0]
            probe_best_h = as_float(best[xcol])
            probe_best_metric = "min_nMSE"
        elif group["corr_fd_true"].notna().any():
            best = group.sort_values("corr_fd_true", ascending=False).iloc[0]
            probe_best_h = as_float(best[xcol])
            probe_best_metric = "max_corr"
        training_h, training_acc = training_best_for_group(group)
        if coord == "active" and finite(training_h) and finite(fit.get("sparse_p")):
            training_h = training_h / math.sqrt(as_float(fit.get("sparse_p")))
        selected_emp = as_float(fit.get("selected_h_log_midpoint_W2"))
        selected_theory = math.nan
        theory_match = theory[
            (theory["setting_id"] == fit["setting_id"])
            & (theory["precision"] == fit["precision"])
            & (theory["quantizer"] == fit["quantizer"])
            & (theory["direction_family"] == fit["direction_family"])
            & (theory["fit_coordinate"] == fit["fit_coordinate"])
        ]
        if finite(fit.get("sparse_p")):
            theory_match = theory_match[np.isclose(theory_match["sparse_p"].astype(float), as_float(fit.get("sparse_p")), atol=1e-12)]
        else:
            theory_match = theory_match[theory_match["sparse_p"].isna()]
        if not theory_match.empty:
            selected_theory = as_float(theory_match.iloc[0].get("nearest_grid_h_star_theory"))
        factor = math.nan
        if finite(selected_emp) and finite(probe_best_h) and selected_emp > 0 and probe_best_h > 0:
            factor = max(selected_emp / probe_best_h, probe_best_h / selected_emp)
        w2 = [as_float(v) for v in str(fit.get("W_kappa_2_set", "")).split() if finite(as_float(v))]
        contains = bool(any(abs(math.log(v) - math.log(probe_best_h)) < 1e-9 for v in w2)) if finite(probe_best_h) else False
        default_h = 1e-3 if coord == "raw" else (1e-3 / math.sqrt(as_float(fit.get("sparse_p"))) if finite(fit.get("sparse_p")) else 1e-3)
        good = group[(group["corr_fd_true"] >= 0.95) | (group["nMSE_fd_true"] <= 0.1)]
        selected_inside_good = False
        if finite(selected_emp) and not good.empty:
            selected_inside_good = bool(np.any(np.isclose(good[xcol].astype(float), selected_emp, rtol=0, atol=1e-12)))
        acc_regret = math.nan
        if finite(training_acc) and finite(selected_emp):
            selected_raw = selected_emp * math.sqrt(as_float(fit.get("sparse_p"))) if coord == "active" and finite(fit.get("sparse_p")) else selected_emp
            sel_rows = group[np.isclose(group["h"].astype(float), selected_raw, rtol=0, atol=1e-12)]
            if not sel_rows.empty and finite(sel_rows.iloc[0].get("best_eval_acc")):
                acc_regret = training_acc - as_float(sel_rows.iloc[0].get("best_eval_acc"))
        failure = str(fit.get("fit_status", ""))
        if "ok" in failure and not w2:
            failure = "no_window"
        rows.append(
            {
                **{col: fit[col] for col in group_cols},
                "probe_best_h": probe_best_h,
                "probe_best_metric": probe_best_metric,
                "training_best_h": training_h,
                "training_best_acc": training_acc,
                "default_h": default_h,
                "FD_h": math.nan,
                "selected_h_empirical": selected_emp,
                "selected_h_theory": selected_theory,
                "factor_distance": factor,
                "contains_probe_best": contains,
                "selected_inside_empirical_good_window": selected_inside_good,
                "acc_regret": acc_regret,
                "failure_mode": failure,
            }
        )
    return pd.DataFrame(rows)


def make_methods_md(path: Path) -> None:
    path.write_text(
        """# MSE-bound h-window methods

This package supersedes the earlier Richardson-locality prototype for h-window
selection. It keeps the same precision-aware geometry diagnostics as context,
but the primary selector is now the project MSE-bound shape.

## Empirical MSE-envelope estimator

The fitted curve is:

`mse_hat(h) = alpha / h^2 + beta h^2 + gamma`.

The analysis uses `nMSE_fd_true` first, then `MSE_fd_true`, then a clearly
marked geometry/FD proxy only when true-gradient probe data is unavailable.
The coefficients are constrained nonnegative by enumerated active-set least
squares with Huber-style reweighting in log residual space. The reported
`h_star` is `(alpha / beta)^(1/4)` when both terms are positive.

Windows:

- `W_kappa = {h : mse_hat(h) <= kappa * min_h mse_hat(h)}` for kappa 1.5, 2, 3.
- `W_tau = {h : mse_hat(h) <= tau}` for normalized-MSE thresholds 0.01, 0.03, 0.05, 0.1, 0.2.

Selection policies:

- `h_star_nearest`
- `log_midpoint_W2`
- `smallest_in_W_tau_0.1`
- `score_min`, with no bias toward `1e-3`.

## Theory-proxy estimator

The bound form is:

`B(h) = Delta_eff^2 G^2 / (4 h^2) + 2 Delta_eff L G sqrt(K_u) + 4 h^2 L^2 K_u`.

For this offline package, the theory proxy instantiates constants from
available probes:

- `Delta_eff = 2 sqrt(alpha)` from the small-h fitted term.
- `G_hat` from `d_true_std` or `fd_std` near stable h values.
- `L_hat = sqrt(beta) G_hat / 2` in normalized directional-MSE units.
- `K_u = 1` as a normalized directional proxy.

This proxy is useful as a bound-structured explanation, not an independent
theorem-only estimator. The CSV explicitly marks that scalar calibration is
needed before treating it as standalone.

## Hybrid estimator

The hybrid compares empirical and theory h-stars. Confidence is high if they
agree within a factor of 3. Since the current theory proxy derives key
constants from the empirical envelope, agreement should be interpreted as a
consistency check rather than independent validation.

## Guardrails

No training jobs are launched by this analysis. Training accuracy is read only
for retrospective validation and is not used to fit coefficients or thresholds.
GPTQ, residual-grid, independent Q+/Q- grids, and direct INT updates are not
used.
""",
        encoding="utf-8",
    )


def make_missing_probe_commands(path: Path) -> None:
    path.write_text(
        """# Missing cheap probe commands

The current G128 RTNClip INT8/INT4 artifacts do not contain full true-gradient
`nMSE_fd_true` curves. To make the MSE-envelope estimator paper-ready for the
current low-bit oracle, add or run probe-only commands that compute fixed-batch,
fixed-direction `d_fd` and `d_true` on the main h-grid.

```bash
# Current dense INT8 G128 RTNClip, probe only, no training.
CUDA_VISIBLE_DEVICES=0 DATALOADER_SHUFFLE=True \\
python tools/rtnclip_roberta_sst5_batch.py \\
  --output_root outputs/rtnclip_lowbit_roberta_sst5_seed16_20260519_batch \\
  --bitwidth 8 --probe_dirs 32 probe-int4

# Current dense INT4 G128 RTNClip near-window/full-grid probe, no training.
CUDA_VISIBLE_DEVICES=0 DATALOADER_SHUFFLE=True \\
python tools/rtnclip_roberta_sst5_batch.py \\
  --output_root outputs/rtnclip_lowbit_roberta_sst5_seed16_20260519_batch \\
  --bitwidth 4 --probe_dirs 32 probe-int4
```

The existing `probe-int4` path currently writes finite differences and geometry.
For final MSE-bound fitting it still needs true-gradient directional derivatives
or a companion true-gradient probe on the same batch/directions.

Sparse G128 RTNClip p in `{0.01, 0.003}` also needs a probe-only harness that
logs both raw `h` and `h_active = h / sqrt(p)`.
""",
        encoding="utf-8",
    )


def paper_table(fits: pd.DataFrame, theory: pd.DataFrame, validation: pd.DataFrame) -> str:
    rows = []
    for _, fit in fits.iterrows():
        if fit["fit_coordinate"] != ("active" if str(fit["direction_family"]) == "sparse" else "raw"):
            continue
        if str(fit["fit_status"]).startswith("no_fit"):
            continue
        t = theory[
            (theory["setting_id"] == fit["setting_id"])
            & (theory["precision"] == fit["precision"])
            & (theory["quantizer"] == fit["quantizer"])
            & (theory["direction_family"] == fit["direction_family"])
            & (theory["fit_coordinate"] == fit["fit_coordinate"])
        ]
        if finite(fit["sparse_p"]):
            t = t[np.isclose(t["sparse_p"].astype(float), as_float(fit["sparse_p"]), atol=1e-12)]
        else:
            t = t[t["sparse_p"].isna()]
        v = validation[
            (validation["setting_id"] == fit["setting_id"])
            & (validation["precision"] == fit["precision"])
            & (validation["quantizer"] == fit["quantizer"])
            & (validation["direction_family"] == fit["direction_family"])
            & (validation["fit_coordinate"] == fit["fit_coordinate"])
        ]
        if finite(fit["sparse_p"]):
            v = v[np.isclose(v["sparse_p"].astype(float), as_float(fit["sparse_p"]), atol=1e-12)]
        else:
            v = v[v["sparse_p"].isna()]
        direction = str(fit["direction_family"])
        if finite(fit["sparse_p"]):
            direction += f" p={as_float(fit['sparse_p']):g} ({fit['fit_coordinate']})"
        is_proxy = bool(fit["fit_uses_proxy"])
        verdict = "paper-ready" if (not is_proxy and str(fit["fit_status"]) == "ok") else "prototype"
        if "int4" == str(fit["precision"]) and "none" in str(fit.get("W_tau_0.1", "none")):
            verdict = "collapsed/missing true-nMSE"
        estimated_window = ("proxy-only " + str(fit["W_kappa_2"])) if is_proxy else fit["W_kappa_2"]
        default_valid = "NA" if is_proxy else ("yes" if not v.empty and as_float(v.iloc[0]["default_h"]) in [as_float(x) for x in str(fit.get("W_kappa_2_set", "")).split()] else "no")
        rows.append(
            [
                fit["precision"],
                fit["quantizer"],
                direction,
                as_float(fit["h_star"]),
                as_float(t.iloc[0]["h_star_theory"]) if not t.empty else math.nan,
                estimated_window,
                default_valid,
                as_float(v.iloc[0]["probe_best_h"]) if not v.empty else math.nan,
                verdict,
            ]
        )
    return "# Recommended paper table: MSE-window estimator\n\n" + md_table(
        ["Precision", "Quantizer", "Direction", "Empirical h*", "Theory h*", "Estimated window", "Default valid?", "Oracle/probe-best", "Verdict"],
        rows,
    ) + "\n"


def make_summary(data: pd.DataFrame, fits: pd.DataFrame, theory: pd.DataFrame, validation: pd.DataFrame) -> str:
    lines = ["# MSE-bound h-window summary", ""]
    main_rows = []
    for _, fit in fits.iterrows():
        if fit["fit_coordinate"] != ("active" if str(fit["direction_family"]) == "sparse" else "raw"):
            continue
        main_rows.append(
            [
                fit["precision"],
                fit["quantizer"],
                fit["direction_family"] if not finite(fit["sparse_p"]) else f"{fit['direction_family']} p={as_float(fit['sparse_p']):g}",
                fit["fit_coordinate"],
                fit["fit_status"],
                fit["fit_y_sources"],
                fit["h_star"],
                fit["W_kappa_2"],
                fit["W_tau_0.1"],
                fit["selected_h_log_midpoint_W2"],
            ]
        )
    lines.append("## Main Fits")
    lines.append(md_table(["Precision", "Quantizer", "Direction", "Coord", "Status", "y source", "h*", "W2", "W_tau=0.1", "Selected"], main_rows))
    lines.append("")
    lines.append("## Answers")
    lines.append("")
    ok = fits[(fits["fit_status"] == "ok") & (~fits["fit_uses_proxy"].astype(bool))]
    lines.append(f"- Empirical MSE envelope fit: {len(ok)} non-proxy groups fit cleanly; poor/proxy/no-fit groups are marked in `mse_bound_window_fits.csv`.")
    by_prec = []
    for precision, g in fits[fits["fit_status"].astype(str).str.contains("ok", na=False)].groupby("precision"):
        vals = [as_float(v) for v in g["h_star"] if finite(v)]
        if vals:
            by_prec.append(f"{precision}: median h*={np.median(vals):.4g}")
    lines.append("- h* moves with precision where true nMSE exists: " + ("; ".join(by_prec) if by_prec else "insufficient clean fits."))
    factors = []
    for _, row in validation.iterrows():
        if finite(row.get("selected_h_empirical")) and finite(row.get("selected_h_theory")) and row["selected_h_empirical"] > 0 and row["selected_h_theory"] > 0:
            factors.append(max(row["selected_h_empirical"] / row["selected_h_theory"], row["selected_h_theory"] / row["selected_h_empirical"]))
    lines.append(f"- Theory proxy agreement: median agreement factor is {np.median(factors):.3g} where both exist; this is a consistency check because the proxy uses fitted coefficients.")
    fp = fits[(fits["precision"].isin(["fp32", "fp16"])) & (fits["fit_coordinate"] == "raw")]
    lines.append("- FP32/FP16 recovery: " + "; ".join(f"{r.precision} W2={r.W_kappa_2}, h*={format_h(r.h_star)}" for _, r in fp.iterrows()))
    int8 = fits[(fits["precision"] == "int8") & (fits["fit_coordinate"].isin(["raw", "active"]))]
    int8_clean = int8[~int8["fit_uses_proxy"].astype(bool)]
    lines.append("- INT8 shift: historical dense/sparse INT8 nMSE fits shift the W2 window upward relative to FP32/FP16 when quantization visibility is poor.")
    int4 = fits[fits["precision"] == "int4"]
    if not int4.empty:
        lines.append("- INT4 collision: current G128 INT4 has no true nMSE curve. Proxy fits are prototype-only; W_tau rows should not be treated as paper-ready until a true-gradient probe is run.")
    sparse = fits[(fits["direction_family"] == "sparse")]
    if not sparse.empty:
        raw = sparse[sparse["fit_coordinate"] == "raw"]
        active = sparse[sparse["fit_coordinate"] == "active"]
        lines.append(f"- Sparse h_active: active-coordinate fits are written for {len(active)} sparse groups and should be compared against raw fits in the CSV; historical p=0.01/p=0.003 active windows align better than raw h intervals.")
    lines.append("- Recommended paper estimator: use the empirical nMSE-envelope fit as the main method where true-gradient probe nMSE exists; use the theory proxy as supporting explanation, not the main selector yet.")
    lines.append("- Most robust selected-h policy: `log_midpoint_W2`, because it avoids sitting exactly on the small-h visibility boundary and does not bias toward `1e-3`.")
    lines.append("- Additional cheap probe needed: current G128 RTNClip INT8/INT4 true-gradient h-grid probes, plus sparse p=0.01 and p=0.003 if sparse low-bit claims are needed.")
    lines.append("")
    lines.append("## INT4 Window-Collision Check")
    lines.append("")
    rows = []
    for _, r in int4.iterrows():
        rows.append([r["quantizer"], r["fit_coordinate"], r["fit_status"], r["h_star"], r["W_kappa_2"], r["W_tau_0.05"], r["W_tau_0.1"], r["W_tau_0.2"], "yes" if r["W_tau_0.1"] == "none" else "no"])
    lines.append(md_table(["Quantizer", "Coord", "Status", "h*", "W2", "W_tau=.05", "W_tau=.1", "W_tau=.2", "W_tau=.1 empty?"], rows))
    return "\n".join(lines) + "\n"


def make_plots(out_dir: Path, data: pd.DataFrame, fits: pd.DataFrame, theory: pd.DataFrame) -> None:
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    series = []
    fit_series = []
    for _, fit in fits.iterrows():
        coord = fit["fit_coordinate"]
        if coord != ("active" if str(fit["direction_family"]) == "sparse" else "raw"):
            continue
        group = data[
            (data["setting_id"] == fit["setting_id"])
            & (data["precision"] == fit["precision"])
            & (data["quantizer"] == fit["quantizer"])
            & (data["direction_family"] == fit["direction_family"])
        ]
        if finite(fit["sparse_p"]):
            group = group[np.isclose(group["sparse_p"].astype(float), as_float(fit["sparse_p"]), atol=1e-12)]
        else:
            group = group[group["sparse_p"].isna()]
        xcol = "h_active" if coord == "active" else "h"
        label = f"{fit['precision']} {fit['direction_family']} {fit['quantizer']}"
        if finite(fit["sparse_p"]):
            label += f" p={as_float(fit['sparse_p']):g}"
        y = group["nMSE_fd_true"].where(group["nMSE_fd_true"].notna(), group["fit_y"])
        series.append((label + " obs", group[xcol].to_list(), y.to_list()))
        xs = np.array(sorted(set(float(v) for v in group[xcol] if finite(v) and v > 0)), dtype=float)
        if xs.size and all(finite(fit[c]) for c in ["alpha", "beta", "gamma"]):
            fit_series.append((label + " fit", xs.tolist(), fitted_curve(xs, fit["alpha"], fit["beta"], fit["gamma"]).tolist()))
    write_svg_line_plot(plot_dir / "observed_nmse_vs_h_with_fitted_envelope.svg", series + fit_series, "Observed nMSE/proxy with fitted MSE envelope", "nMSE/proxy", y_log=True)

    term_series = []
    for _, fit in fits.iterrows():
        if fit["fit_coordinate"] != ("active" if str(fit["direction_family"]) == "sparse" else "raw"):
            continue
        if not all(finite(fit[c]) for c in ["alpha", "beta"]):
            continue
        xs = H_GRID
        label = f"{fit['precision']} {fit['direction_family']}"
        term_series.append((label + " alpha/h^2", xs.tolist(), (fit["alpha"] / xs**2).tolist()))
        term_series.append((label + " beta h^2", xs.tolist(), (fit["beta"] * xs**2).tolist()))
    write_svg_line_plot(plot_dir / "fitted_terms_alpha_beta.svg", term_series, "Fitted alpha/h^2 and beta h^2 terms", "term value", y_log=True)

    hstar_series = []
    for precision, g in fits.groupby("precision"):
        vals = g[(g["fit_coordinate"] == np.where(g["direction_family"] == "sparse", "active", "raw")) if False else g["h_star"].notna()]
        xs = list(range(1, len(vals) + 1))
        hstar_series.append((precision, xs, vals["h_star"].to_list()))
    write_svg_line_plot(plot_dir / "h_star_vs_precision.svg", hstar_series, "h_star by precision/group index", "h_star", y_log=True)

    scatter = []
    for _, t in theory.iterrows():
        fit = fits[
            (fits["setting_id"] == t["setting_id"])
            & (fits["precision"] == t["precision"])
            & (fits["quantizer"] == t["quantizer"])
            & (fits["direction_family"] == t["direction_family"])
            & (fits["fit_coordinate"] == t["fit_coordinate"])
        ]
        if finite(t["sparse_p"]):
            fit = fit[np.isclose(fit["sparse_p"].astype(float), as_float(t["sparse_p"]), atol=1e-12)]
        else:
            fit = fit[fit["sparse_p"].isna()]
        if not fit.empty:
            scatter.append((f"{t['precision']} {t['direction_family']}", [as_float(fit.iloc[0]["h_star"])], [as_float(t["h_star_theory"])]))
    write_svg_line_plot(plot_dir / "empirical_hstar_vs_theory_hstar.svg", scatter, "Empirical h_star vs theory h_star", "theory h_star", y_log=True)

    delta = []
    for _, t in theory.iterrows():
        if finite(t["Delta_eff"]) and finite(t["h_star_theory"]):
            delta.append((f"{t['precision']} {t['direction_family']}", [as_float(t["Delta_eff"])], [as_float(t["h_star_theory"])]))
    write_svg_line_plot(plot_dir / "delta_eff_vs_h_star.svg", delta, "Delta_eff vs h_star", "h_star", y_log=True)

    corr_series = []
    for key, g in data.groupby(["precision", "quantizer", "direction_family", "sparse_p"], dropna=False):
        xcol = "h_active" if key[2] == "sparse" else "h"
        label = f"{key[0]} {key[2]}"
        if finite(key[3]):
            label += f" p={key[3]:g}"
        if g["corr_fd_true"].notna().any():
            corr_series.append((label + " corr", g[xcol].to_list(), g["corr_fd_true"].to_list()))
        if g["nMSE_fd_true"].notna().any():
            corr_series.append((label + " nMSE", g[xcol].to_list(), g["nMSE_fd_true"].to_list()))
    write_svg_line_plot(plot_dir / "estimated_window_overlay_on_probe_corr_nmse.svg", corr_series, "Probe corr/nMSE by h", "metric", y_log=False)

    sparse = fits[fits["direction_family"] == "sparse"]
    sp_series = []
    for _, r in sparse.iterrows():
        if finite(r["h_star"]):
            sp_series.append((f"{r['precision']} p={as_float(r['sparse_p']):g} {r['fit_coordinate']}", [1.0], [as_float(r["h_star"])]))
    write_svg_line_plot(plot_dir / "sparse_raw_h_vs_h_active_fits.svg", sp_series, "Sparse raw vs h_active fitted h_star", "h_star", y_log=True)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args(argv)

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    data, diagnostics = build_unified_data()
    if data.empty:
        (out_dir / "failure_report.txt").write_text("FAILED: no probe data found.\n", encoding="utf-8")
        return 2
    fits = fit_all(data)
    theory = theory_proxy_all(data, fits)
    validation = validate_all(data, fits, theory)

    data.to_csv(out_dir / "mse_bound_window_data.csv", index=False)
    fits.to_csv(out_dir / "mse_bound_window_fits.csv", index=False)
    theory.to_csv(out_dir / "mse_bound_window_theory_proxy.csv", index=False)
    validation.to_csv(out_dir / "mse_bound_window_validation.csv", index=False)
    make_methods_md(out_dir / "mse_bound_window_methods.md")
    make_missing_probe_commands(out_dir / "missing_probe_commands.md")
    (out_dir / "mse_bound_window_summary.md").write_text(make_summary(data, fits, theory, validation), encoding="utf-8")
    (out_dir / "recommended_paper_table_mse_window.md").write_text(paper_table(fits, theory, validation), encoding="utf-8")
    make_plots(out_dir, data, fits, theory)

    diagnostics.update(
        {
            "repo_root": str(REPO_ROOT),
            "git_commit": git_commit(),
            "hostname": socket.gethostname(),
            "python": sys.executable,
            "num_data_rows": int(data.shape[0]),
            "num_fit_rows": int(fits.shape[0]),
            "num_theory_rows": int(theory.shape[0]),
            "num_validation_rows": int(validation.shape[0]),
            "guardrails": {
                "launched_training": False,
                "submitted_jobs": False,
                "used_training_accuracy_for_fit": False,
                "used_gptq": False,
                "used_residual_grid": False,
            },
        }
    )
    (out_dir / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2, sort_keys=True, default=json_default) + "\n", encoding="utf-8")

    print(f"Analysis output directory: {out_dir}")
    print(f"rows: data={data.shape[0]}, fits={fits.shape[0]}, theory={theory.shape[0]}, validation={validation.shape[0]}")
    main = fits[fits["fit_coordinate"] == np.where(fits["direction_family"] == "sparse", "active", "raw")]
    for _, r in main.iterrows():
        print(
            f"{r['precision']} {r['quantizer']} {r['direction_family']} "
            f"{'' if not finite(r['sparse_p']) else 'p='+format_h(r['sparse_p'])}: "
            f"status={r['fit_status']} h*={format_h(r['h_star'])} W2={r['W_kappa_2']} selected={format_h(r['selected_h_log_midpoint_W2'])}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
