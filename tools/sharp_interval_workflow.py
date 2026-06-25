#!/usr/bin/env python3
"""Build sharp interval-aware fit and RoBERTa INT4 evaluation bundle.

This script is intentionally result-driven: it reuses existing synthetic,
interval geometry, loss-probe, and RoBERTa INT4 training CSVs, fits several
nonnegative h-window models, chooses h candidates, compares against existing
training results, and writes a self-contained bundle.  It does not fabricate
missing training rows; absent full runs are reported in missing_items.md and in
the generated launch manifest.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import socket
import subprocess
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


EPS = 1e-30
TASKS = ["sst-2", "sst-5", "rte", "mnli", "trec"]
MODES = ["dense", "sparse_p0p1", "prefix"]
H_GRID = np.array(
    [1e-7, 3e-7, 1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1],
    dtype=float,
)


def run_cmd(cmd: Sequence[str], cwd: Path) -> str:
    try:
        return subprocess.check_output(cmd, cwd=str(cwd), stderr=subprocess.STDOUT, text=True).strip()
    except Exception as exc:  # pragma: no cover - diagnostic path
        return f"unavailable: {exc}"


def read_csv(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def first_present(row: pd.Series, names: Sequence[str], default=np.nan):
    for name in names:
        if name in row.index:
            val = row[name]
            if not (pd.isna(val) if not isinstance(val, str) else val == ""):
                return val
    return default


def as_float(x, default=np.nan) -> float:
    try:
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return default
        if isinstance(x, str):
            s = x.strip()
            if not s or s.lower() in {"nan", "none", "null"}:
                return default
            return float(s)
        return float(x)
    except Exception:
        return default


def canonical_directional_target(row: pd.Series) -> Tuple[float, str, bool]:
    """Return a paper-compatible directional MSE target when present.

    The paper target is based on loss-level two-point finite differences:

        d_Q(h, u) = [L(Q(w+h u)) - L(Q(w-h u))] / (2h)
        d_ref(u) = <grad L(w), u>

    Some project summaries contain `lowbit_true_nmse`, but that field may refer
    to effective-displacement geometry, e.g. Delta_Q/(2h), rather than the
    loss-level directional derivative.  Those rows are intentionally rejected as
    fit targets and can only be used as interval/geometry covariates.
    """

    fd_available = str(first_present(row, ["fd_true_available"], "")).strip().lower()
    fd_explicitly_false = fd_available in {"false", "0", "no"}
    if not fd_explicitly_false:
        raw_mse = as_float(first_present(row, ["fd_true_mse"]))
        if np.isfinite(raw_mse):
            return raw_mse, "paper_directional_mse:fd_true_mse", True
        for name in ["fd_true_nmse", "nMSE_fd_true", "fd_true_nmse_default"]:
            val = as_float(first_present(row, [name]))
            if np.isfinite(val):
                return val, f"paper_directional_nmse:{name}", True

    # Explicitly do not accept lowbit_true_nmse unless a future run labels it as
    # a loss-level directional metric.  The current known version is
    # dequantized_effective_displacement_nmse_v1, i.e. geometry only.
    version = str(first_present(row, ["nMSE_metric_version"], "")).lower()
    lowbit_val = as_float(first_present(row, ["lowbit_true_nmse"]))
    if np.isfinite(lowbit_val):
        if "directional_loss" in version or "loss_directional" in version:
            return lowbit_val, f"paper_directional_nmse:lowbit_true_nmse:{version}", True
        return np.nan, f"geometry_only_not_target:lowbit_true_nmse:{version or 'unknown_version'}", False

    legacy_val = as_float(first_present(row, ["default_nmse"]))
    if np.isfinite(legacy_val):
        return np.nan, "legacy_or_ambiguous_not_target:default_nmse", False
    return np.nan, "missing_directional_mse_target", False


def norm_task(x, source: str = "") -> str:
    s = str(x).lower().strip() if x is not None and not pd.isna(x) else ""
    src = source.lower()
    for key in TASKS:
        if key in s or key.replace("-", "") in s:
            return key
        if key in src or key.replace("-", "") in src:
            return key
    if "sst5" in s or "sst5" in src:
        return "sst-5"
    if "sst2" in s or "sst2" in src:
        return "sst-2"
    return s or "unknown"


def norm_mode(row: pd.Series, source: str = "") -> str:
    # Prefer explicit row fields over path names.  Some historical output roots
    # contain both "sparse" and "prefix" in the directory name.
    row_vals = " ".join(str(row.get(c, "")) for c in row.index).lower()
    vals = row_vals + " " + source.lower()
    direction = str(row.get("direction_mode", row.get("perturbation_mode", ""))).lower()
    run_scope = " ".join(str(row.get(c, "")) for c in ["run_name", "perturbed_parameter_scope", "direction_mode", "perturbation_mode"]).lower()
    if "sparse" in run_scope or direction == "sparse":
        sparse_ratio = as_float(first_present(row, ["sparse_ratio", "mask_active_frac_all", "mask_active_frac_quantized_linear"]))
        if np.isfinite(sparse_ratio):
            if sparse_ratio <= 0.02:
                return "sparse_p0p01"
            if 0.05 <= sparse_ratio <= 0.2:
                return "sparse_p0p1"
        if "0.01" in vals or "p0p01" in vals or "p0.01" in vals:
            return "sparse_p0p01"
        return "sparse_p0p1"
    if "prefix" in run_scope or direction == "prefix":
        return "prefix"
    return "dense"


def infer_policy(row: pd.Series, h: float, source: str = "") -> str:
    for col in ["h_policy", "candidate_name", "policy"]:
        if col in row.index and isinstance(row[col], str) and row[col].strip():
            p = row[col].strip()
            if p == "mezo_default":
                return "default"
            if p == "hstar_ours":
                return "old_env"
            return p
    run = str(row.get("run_name", "")) + " " + source
    low = run.lower()
    if "fixed_small" in low or "h1e-5" in low:
        return "fixed_small"
    if "mezo_default" in low or (np.isfinite(h) and abs(math.log(max(h, EPS) / 1e-3)) < 1e-6):
        return "default"
    if "cleangl" in low:
        return "hstar_cleanGL"
    if "lowbitl" in low:
        return "hstar_lowbitL"
    if "hstar" in low:
        return "old_env"
    return "unknown"


def infer_run_type(steps: float, status: str = "") -> str:
    if not np.isfinite(steps):
        return "unknown"
    if steps >= 18000:
        return "full"
    if steps >= 8000:
        return "medium"
    if steps > 0:
        return "pilot"
    return "missing"


def simple_markdown(df: pd.DataFrame, max_rows: int = 40) -> str:
    """Small dependency-free markdown table renderer."""
    if df.empty:
        return "_none_"
    d = df.head(max_rows).copy()
    cols = list(d.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in d.iterrows():
        vals = []
        for col in cols:
            val = row[col]
            if isinstance(val, float):
                vals.append("" if not np.isfinite(val) else f"{val:.6g}")
            else:
                vals.append(str(val).replace("\n", " ")[:120])
        lines.append("| " + " | ".join(vals) + " |")
    if len(df) > max_rows:
        lines.append(f"| ... | {len(df) - max_rows} more rows |" + " |" * max(0, len(cols) - 2))
    return "\n".join(lines)


def nnls_enumerate(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Tiny nonnegative least squares by active set enumeration."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n = X.shape[1]
    best_coef = np.zeros(n)
    best_loss = float("inf")
    for mask in range(1, 1 << n):
        idx = [i for i in range(n) if mask & (1 << i)]
        Xi = X[:, idx]
        try:
            ci, *_ = np.linalg.lstsq(Xi, y, rcond=None)
        except Exception:
            continue
        if np.any(ci < -1e-12):
            continue
        pred = Xi @ np.maximum(ci, 0)
        loss = float(np.mean((pred - y) ** 2))
        if loss < best_loss:
            best_loss = loss
            best_coef = np.zeros(n)
            best_coef[idx] = np.maximum(ci, 0)
    if not np.isfinite(best_loss):
        best_coef[-1] = max(float(np.nanmean(y)), 0.0)
    return best_coef


def fit_nonnegative(h: np.ndarray, y: np.ndarray, features: np.ndarray, space: str) -> Tuple[np.ndarray, float, float]:
    ok = np.isfinite(h) & np.isfinite(y) & np.all(np.isfinite(features), axis=1) & (h > 0) & (y >= 0)
    h = h[ok]
    y = y[ok]
    X = features[ok]
    if len(y) < X.shape[1] or len(y) < 3:
        coef = np.full(X.shape[1], np.nan)
        return coef, np.nan, np.nan
    if space == "linear":
        coef = nnls_enumerate(X, y)
        pred = np.maximum(X @ coef, EPS)
    else:
        # Approximate log-space fit with iterative reweighted NNLS.
        coef = nnls_enumerate(X, y)
        for _ in range(8):
            pred = np.maximum(X @ coef, EPS)
            w = 1.0 / pred
            coef = nnls_enumerate(X * w[:, None], y * w)
        pred = np.maximum(X @ coef, EPS)
    y_safe = np.maximum(y, EPS)
    rmse_log = float(np.sqrt(np.mean((np.log10(pred) - np.log10(y_safe)) ** 2)))
    ss_res = float(np.sum((np.log10(pred) - np.log10(y_safe)) ** 2))
    ss_tot = float(np.sum((np.log10(y_safe) - np.mean(np.log10(y_safe))) ** 2))
    r2_log = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return coef, r2_log, rmse_log


def fit_group(g: pd.DataFrame) -> List[Dict[str, object]]:
    g = g.sort_values("h").copy()
    h = g["h"].astype(float).to_numpy()
    y = g["A_fit"].astype(float).to_numpy()
    a_cross = g["A_cross"].fillna(g["A_fit"]).astype(float).to_numpy()
    m_loc = g["M_loc"].fillna(np.nan).astype(float).to_numpy()
    if np.all(~np.isfinite(m_loc)):
        m_loc = h**4
    else:
        finite = np.isfinite(m_loc)
        fill = np.nanmedian(m_loc[finite]) if finite.any() else 0.0
        m_loc = np.where(finite, m_loc, fill)
    p_clip = g.get("p_clip", pd.Series(np.zeros(len(g)))).fillna(0).astype(float).to_numpy()
    clean = p_clip <= 0.05
    if clean.sum() >= 4:
        h0, y0, cross0, loc0 = h[clean], y[clean], a_cross[clean], m_loc[clean]
    else:
        h0, y0, cross0, loc0 = h, y, a_cross, m_loc
    out = []
    models: List[Tuple[str, Optional[float], np.ndarray, List[str]]] = []
    models.append(("M2", 2.0, np.column_stack([1 / np.maximum(h0, EPS) ** 2, h0**2, np.ones_like(h0)]), ["alpha", "beta", "gamma"]))
    models.append(("M4", 4.0, np.column_stack([1 / np.maximum(h0, EPS) ** 2, h0**4, np.ones_like(h0)]), ["alpha", "beta", "gamma"]))
    for p in [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]:
        models.append((f"Mp", p, np.column_stack([1 / np.maximum(h0, EPS) ** 2, h0**p, np.ones_like(h0)]), ["alpha", "beta", "gamma"]))
    cross = np.maximum(cross0, 0.0)
    models.append(
        (
            "M_sharp_norm",
            4.0,
            np.column_stack([cross, h0**2 * np.sqrt(cross + EPS), h0**4, np.ones_like(h0)]),
            ["a_cross", "b_cross_loc", "c_h4", "gamma"],
        )
    )
    models.append(
        (
            "M_sharp_constrained",
            4.0,
            np.column_stack([cross, 2 * h0**2 * np.sqrt(cross + EPS), h0**4, np.ones_like(h0)]),
            ["a1_sq", "a1a2", "a2_sq", "gamma"],
        )
    )
    models.append(("MIA_loc", None, np.column_stack([cross, loc0, np.ones_like(h0)]), ["a_cross", "c_loc", "gamma"]))
    for name, p, X, coef_names in models:
        for space in ["linear", "log"]:
            coef, r2, rmse = fit_nonnegative(h0, y0, X, space)
            dense_h = np.logspace(np.log10(max(h0.min() / 3, 1e-9)), np.log10(min(h0.max() * 3, 1.0)), 400)
            if name == "M2":
                pred_dense = coef[0] / dense_h**2 + coef[1] * dense_h**2 + coef[2]
            elif name == "M4":
                pred_dense = coef[0] / dense_h**2 + coef[1] * dense_h**4 + coef[2]
            elif name == "Mp":
                pred_dense = coef[0] / dense_h**2 + coef[1] * dense_h**float(p) + coef[2]
            else:
                # Interpolate cross/loc onto dense grid for h selection.
                logh = np.log(h0)
                cross_dense = np.exp(np.interp(np.log(dense_h), logh, np.log(np.maximum(cross0, EPS))))
                loc_dense = np.exp(np.interp(np.log(dense_h), logh, np.log(np.maximum(loc0, EPS))))
                if name == "M_sharp_norm":
                    pred_dense = coef[0] * cross_dense + coef[1] * dense_h**2 * np.sqrt(cross_dense + EPS) + coef[2] * dense_h**4 + coef[3]
                elif name == "M_sharp_constrained":
                    pred_dense = coef[0] * cross_dense + coef[1] * 2 * dense_h**2 * np.sqrt(cross_dense + EPS) + coef[2] * dense_h**4 + coef[3]
                else:
                    pred_dense = coef[0] * cross_dense + coef[1] * loc_dense + coef[2]
            pred_dense = np.maximum(pred_dense, EPS)
            h_star = float(dense_h[int(np.nanargmin(pred_dense))]) if np.all(np.isfinite(pred_dense)) else np.nan
            row = {
                "fit_model": name,
                "fit_space": space,
                "p": p,
                "R2_log": r2,
                "RMSE_log": rmse,
                "h_fit_min": float(h0.min()),
                "h_fit_max": float(h0.max()),
                "h_star_pred": h_star,
                "h_star_interior": bool(np.isfinite(h_star) and h_star > h0.min() * 1.01 and h_star < h0.max() / 1.01),
                "clean_points": int(len(h0)),
            }
            for key, val in zip(coef_names, coef):
                row[key] = float(val) if np.isfinite(val) else np.nan
            status = []
            if name in {"M2", "M4", "Mp"}:
                if row.get("alpha", 0) <= 0:
                    status.append("left_tail_missing")
                if row.get("beta", 0) <= 0:
                    status.append("right_tail_missing")
            if not row["h_star_interior"]:
                status.append("boundary_solution")
            row["status"] = ";".join(status) if status else "ok"
            out.append(row)
    return out


def model_predict(row: pd.Series, h: np.ndarray, ref: pd.DataFrame) -> np.ndarray:
    model = str(row["fit_model"])
    if model == "M2":
        return row.get("alpha", 0) / h**2 + row.get("beta", 0) * h**2 + row.get("gamma", 0)
    if model == "M4":
        return row.get("alpha", 0) / h**2 + row.get("beta", 0) * h**4 + row.get("gamma", 0)
    if model == "Mp":
        return row.get("alpha", 0) / h**2 + row.get("beta", 0) * h ** float(row.get("p", 2.0)) + row.get("gamma", 0)
    r = ref.sort_values("h")
    logh = np.log(r["h"].astype(float).to_numpy())
    cross = r["A_cross"].fillna(r["A_fit"]).astype(float).to_numpy()
    cross_dense = np.exp(np.interp(np.log(h), logh, np.log(np.maximum(cross, EPS))))
    if model == "M_sharp_norm":
        return row.get("a_cross", 0) * cross_dense + row.get("b_cross_loc", 0) * h**2 * np.sqrt(cross_dense + EPS) + row.get("c_h4", 0) * h**4 + row.get("gamma", 0)
    if model == "M_sharp_constrained":
        return row.get("a1_sq", 0) * cross_dense + row.get("a1a2", 0) * 2 * h**2 * np.sqrt(cross_dense + EPS) + row.get("a2_sq", 0) * h**4 + row.get("gamma", 0)
    loc = r["M_loc"].fillna(np.nan).astype(float).to_numpy()
    if np.all(~np.isfinite(loc)):
        loc = r["h"].astype(float).to_numpy() ** 4
    loc_dense = np.exp(np.interp(np.log(h), logh, np.log(np.maximum(loc, EPS))))
    return row.get("a_cross", 0) * cross_dense + row.get("c_loc", 0) * loc_dense + row.get("gamma", 0)


def load_fit_inputs(repo: Path) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    # Synthetic repaired results.
    syn = read_csv(repo / "synthetic_fit_repair" / "synthetic_fit_raw.csv")
    if syn is not None:
        for _, r in syn.iterrows():
            rows.append(
                {
                    "source_path": "synthetic_fit_repair/synthetic_fit_raw.csv",
                    "experiment_type": "synthetic",
                    "model": str(r.get("config_id", "synthetic")),
                    "task": "synthetic",
                    "precision": f"int{int(r['qbits'])}" if str(r.get("qmode", "")) == "rtn" else "fp32",
                    "quantizer": str(r.get("qmode", "identity")),
                    "mode": "dense" if as_float(r.get("active_p", 1.0), 1.0) == 1.0 else f"sparse_p{as_float(r.get('active_p')):g}",
                    "h": as_float(r.get("h")),
                    "A_true": as_float(r.get("A_true")),
                    "nMSE_loss": np.nan,
                    "A_cross_uniform": as_float(r.get("A_interval_uniform")),
                    "A_cross_grad": as_float(r.get("A_interval_grad")),
                    "sigma_raw2": np.nan,
                    "p_active": as_float(r.get("p_active")),
                    "V_align": as_float(r.get("V_align")),
                    "V_norm": as_float(r.get("V_norm")),
                    "p_clip": as_float(r.get("p_clip"), 0.0),
                    "relative_disp": as_float(r.get("relative_disp")),
                    "locality_proxy": as_float(r.get("locality_proxy")),
                    "M_loc_true": as_float(r.get("M_loc_true")),
                    "accuracy_if_available": np.nan,
                    "run_type": "probe",
                    "seed": 0,
                    "config_key": str(r.get("config_id", "synthetic")),
                    "target_kind": "paper_directional_nmse:synthetic_A_true",
                    "target_is_paper_directional_mse": True,
                }
            )
    # Interval-aware real geometry.
    for p in [
        repo / "interval_aware_h_probe" / "interval_geometry_summary.csv",
        repo / "hwindow_12h_highdim_extra_midp" / "realmodel_interval_metrics.csv",
        repo / "hwindow_12h_highdim_extra_g64" / "realmodel_interval_metrics.csv",
        repo / "hwindow_12h_highdim_extra_g256" / "realmodel_interval_metrics.csv",
        repo / "outputs" / "interval_h_selection_8h_probes" / "opt13b_sst5_int8" / "interval_geometry_summary.csv",
    ]:
        df = read_csv(p)
        if df is None:
            continue
        for _, r in df.iterrows():
            task = norm_task(first_present(r, ["task", "dataset"]), str(p))
            mode = str(first_present(r, ["perturbation_mode", "mode"], "dense")).replace("sparse_p0.1", "sparse_p0p1")
            rows.append(
                {
                    "source_path": str(p.relative_to(repo)),
                    "experiment_type": "real_interval_geometry",
                    "model": str(first_present(r, ["model"], "roberta-large")),
                    "task": task,
                    "precision": str(first_present(r, ["precision"], "unknown")).lower(),
                    "quantizer": "rtn_or_project_default",
                    "mode": mode,
                    "h": as_float(r.get("h")),
                    "A_true": np.nan,
                    "nMSE_loss": np.nan,
                    "A_cross_uniform": as_float(first_present(r, ["A_uniform", "A_interval_uniform"])),
                    "A_cross_grad": as_float(first_present(r, ["A_interval_grad", "A_cross_grad"])),
                    "sigma_raw2": np.nan,
                    "p_active": as_float(r.get("p_active")),
                    "V_align": as_float(r.get("V_align")),
                    "V_norm": as_float(r.get("V_norm")),
                    "p_clip": as_float(r.get("p_clip"), 0.0),
                    "relative_disp": as_float(r.get("relative_disp")),
                    "locality_proxy": as_float(r.get("locality_proxy")),
                    "M_loc_true": as_float(first_present(r, ["M_loc_true", "locality_proxy"])),
                    "accuracy_if_available": np.nan,
                    "run_type": "probe",
                    "seed": 16,
                    "config_key": f"{first_present(r, ['model'], 'roberta-large')}|{task}|{first_present(r, ['precision'], 'unknown')}|{mode}",
                    "target_kind": "geometry_only_no_loss_directional_mse",
                    "target_is_paper_directional_mse": False,
                }
            )
    # Training/loss-probe summaries often contain lowbit true nMSE and visibility metrics.
    patterns = [
        "outputs/rtnclip_int4_g128_rtnclip_roberta_sst5_seed16_20260521/int4_hsearch_summary.csv",
        "outputs/rtnclip_int4_standard_screen_seed16_20260520_203144_h100/int4_standard_*summary.csv",
        "outputs/int4_full_data_hstar_dense_sparse_20260522_113710/int4_hsearch_summary.csv",
        "outputs/int4_dense_hstar_cont_vs_default_2k_20260522_163849/int4_hsearch_summary.csv",
        "outputs/int4_cleanGL_hstar_dense_sparsep0p1_20k_20260523_142501/int4_hsearch_summary.csv",
        "outputs/int4_sparse_prefix_seedfixed_int4fd_20k_20260523_171841/int4_hsearch_summary.csv",
        "outputs/int4_sparsep0p1_probe_minmse_vs_default_2k_20260522_181148/int4_hsearch_summary.csv",
        "outputs/sharp_interval_roberta_int4_eval/int4_hsearch_summary.csv",
        "outputs/rtnclip_int4_mse_reprobe_20260521_true_nmse_d16/int4_mse_probe_summary.csv",
        "outputs/rtnclip_int4_mse_reprobe_20260521_true_nmse_d8_v2/int4_mse_probe_summary.csv",
        "outputs/rtnclip_int4_sparse_mezo_nmse_probe_20260522_dirs32/summary.csv",
        "outputs/rtnclip_int4_sparse_mezo_nmse_probe_20260522_fixeddirs32/summary.csv",
        "outputs/rtnclip_int4_adapter_nmse_probe_20260522_dirs32/summary.csv",
        "outputs/rtnclip_int4_prefix_mezo32_probe64_20260523_144315/summary.csv",
    ]
    for pat in patterns:
        for p in repo.glob(pat):
            df = read_csv(p)
            if df is None:
                continue
            for _, r in df.iterrows():
                h = as_float(first_present(r, ["h", "h_value", "eps", "zo_eps"]))
                if not np.isfinite(h):
                    continue
                task = norm_task(first_present(r, ["task", "dataset", "task_name"]), str(p))
                mode = norm_mode(r, str(p))
                precision = "int4" if "int4" in str(p).lower() or as_float(r.get("bitwidth")) == 4 else str(first_present(r, ["precision"], "int4")).lower()
                a_fit, target_kind, target_ok = canonical_directional_target(r)
                rows.append(
                    {
                        "source_path": str(p.relative_to(repo)),
                        "experiment_type": "training_summary_or_loss_probe",
                        "model": "roberta-large",
                        "task": task,
                        "precision": precision,
                        "quantizer": "G128_RTNClip_fake_quant",
                        "mode": mode,
                        "h": h,
                        "A_true": a_fit,
                        "nMSE_loss": a_fit,
                        "A_cross_uniform": as_float(first_present(r, ["delta_visibility_nmse", "delta_visibility_nmse_mean", "lowbit_true_nmse", "active_frac", "A_uniform"])),
                        "A_cross_grad": np.nan,
                        "sigma_raw2": np.nan,
                        "p_active": as_float(first_present(r, ["active_frac", "code_change_frac"])),
                        "V_align": as_float(first_present(r, ["alignment", "corr_fd_true", "lowbit_true_corr"])),
                        "V_norm": as_float(first_present(r, ["norm_ratio"])),
                        "p_clip": max(
                            as_float(first_present(r, ["saturation_frac_w_plus"], 0.0), 0.0),
                            as_float(first_present(r, ["saturation_frac_w_minus"], 0.0), 0.0),
                            as_float(first_present(r, ["clip_frac"], 0.0), 0.0),
                        ),
                        "relative_disp": as_float(first_present(r, ["delta_visibility_rel_l2", "relative_disp"])),
                        "locality_proxy": np.nan,
                        "M_loc_true": np.nan,
                        "accuracy_if_available": as_float(first_present(r, ["best_eval_acc", "accuracy", "best_dev_acc"])),
                        "run_type": infer_run_type(as_float(first_present(r, ["steps_completed", "steps", "last_eval_step"])), str(r.get("status", ""))),
                        "seed": as_float(first_present(r, ["seed"], 16), 16),
                        "config_key": f"roberta-large|{task}|{precision}|{mode}",
                        "target_kind": target_kind,
                        "target_is_paper_directional_mse": target_ok,
                    }
                )
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out[np.isfinite(out["h"].astype(float))]
        out = out.drop_duplicates()
    return out


def load_training_index(repo: Path) -> pd.DataFrame:
    rows = []
    patterns = [
        "outputs/rtnclip_int4_g128_rtnclip_roberta_sst5_seed16_20260521/int4_hsearch_summary.csv",
        "outputs/int4_full_data_hstar_dense_sparse_20260522_113710/int4_hsearch_summary.csv",
        "outputs/int4_dense_hstar_cont_vs_default_2k_20260522_163849/int4_hsearch_summary.csv",
        "outputs/int4_cleanGL_hstar_dense_sparsep0p1_20k_20260523_142501/int4_hsearch_summary.csv",
        "outputs/int4_sparse_prefix_seedfixed_int4fd_20k_20260523_171841/int4_hsearch_summary.csv",
        "outputs/int4_sparsep0p1_probe_minmse_vs_default_2k_20260522_181148/int4_hsearch_summary.csv",
        "outputs/sharp_interval_roberta_int4_eval/int4_hsearch_summary.csv",
    ]
    for pat in patterns:
        for p in repo.glob(pat):
            df = read_csv(p)
            if df is None:
                continue
            for _, r in df.iterrows():
                h = as_float(first_present(r, ["h", "h_value"]))
                if not np.isfinite(h):
                    continue
                task = norm_task(first_present(r, ["task", "dataset", "task_name"]), str(p))
                mode = norm_mode(r, str(p))
                policy = infer_policy(r, h, str(p))
                steps = as_float(first_present(r, ["steps_completed", "steps", "last_eval_step"]))
                status = str(first_present(r, ["status"], "unknown"))
                rows.append(
                    {
                        "model": "roberta-large",
                        "task": task,
                        "precision": "int4",
                        "quantizer": "G128_RTNClip_fake_quant",
                        "mode": mode,
                        "h_policy": policy,
                        "h_value": h,
                        "seed": int(as_float(first_present(r, ["seed"], 16), 16)),
                        "run_type": infer_run_type(steps, status),
                        "steps": int(steps) if np.isfinite(steps) else 0,
                        "status": status,
                        "best_dev_acc": as_float(first_present(r, ["best_eval_acc", "best_dev_acc", "accuracy"])),
                        "final_dev_acc": as_float(first_present(r, ["last_eval_acc", "final_dev_acc", "accuracy"])),
                        "loss": as_float(first_present(r, ["best_eval_loss", "last_eval_loss", "loss"])),
                        "source_path": str(p.relative_to(repo)),
                        "run_dir": str(first_present(r, ["run_dir"], "")),
                    }
                )
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.drop_duplicates(subset=["task", "mode", "h_policy", "h_value", "seed", "source_path", "run_dir"])
    return out


def prepare_fit_frame(fit_input: pd.DataFrame) -> pd.DataFrame:
    df = fit_input.copy()
    if "target_is_paper_directional_mse" not in df.columns:
        df["target_is_paper_directional_mse"] = False
    target_ok = df["target_is_paper_directional_mse"].astype(bool)
    df["A_fit"] = df["A_true"]
    fill_mask = target_ok & ~np.isfinite(df["A_fit"].astype(float))
    df.loc[fill_mask, "A_fit"] = df.loc[fill_mask, "nMSE_loss"]
    df.loc[~target_ok, "A_fit"] = np.nan
    df["A_cross"] = df["A_cross_grad"]
    df.loc[~np.isfinite(df["A_cross"].astype(float)), "A_cross"] = df["A_cross_uniform"]
    df.loc[~np.isfinite(df["A_cross"].astype(float)), "A_cross"] = df["p_active"]
    df["M_loc"] = df["M_loc_true"]
    df.loc[~np.isfinite(df["M_loc"].astype(float)), "M_loc"] = df["locality_proxy"]
    df = df[np.isfinite(df["A_fit"].astype(float)) & np.isfinite(df["A_cross"].astype(float))]
    df = df[df["A_fit"].astype(float) >= 0]
    return df


def build_candidates(fit_df: pd.DataFrame, fit_rows: pd.DataFrame, training: pd.DataFrame) -> pd.DataFrame:
    rows = []
    real = fit_df[(fit_df["model"].astype(str).str.contains("roberta", case=False, na=False)) & (fit_df["precision"] == "int4")]
    for (task, mode), g in real.groupby(["task", "mode"]):
        if task not in TASKS or mode not in MODES:
            continue
        key_mask = (fit_rows["group_key"] == f"roberta-large|{task}|int4|{mode}") & (fit_rows["fit_space"] == "log")
        sharp = fit_rows[key_mask & fit_rows["fit_model"].isin(["M_sharp_norm", "M_sharp_constrained", "MIA_loc"])].sort_values(["RMSE_log", "R2_log"], ascending=[True, False])
        if sharp.empty:
            allfits = fit_rows[key_mask].sort_values(["RMSE_log", "R2_log"], ascending=[True, False])
            if allfits.empty:
                continue
            best = allfits.iloc[0]
        else:
            best = sharp.iloc[0]
        h_obs_min = float(np.nanmin(g["h"].astype(float)))
        h_obs_max = float(np.nanmax(g["h"].astype(float)))
        h_dense = np.logspace(np.log10(max(h_obs_min, 1e-9)), np.log10(min(h_obs_max, 1e-1)), 600)
        pred = model_predict(best, h_dense, g)
        min_pred = float(np.nanmin(pred))
        h_sharp = float(h_dense[int(np.nanargmin(pred))])
        train_grid = H_GRID
        snap = lambda x: float(train_grid[int(np.argmin(np.abs(np.log(train_grid) - math.log(max(x, EPS)))))] )
        h_sharp = snap(h_sharp)
        within = h_dense[pred <= 1.25 * min_pred]
        if len(within):
            h_low, h_high = float(within.min()), float(within.max())
        else:
            h_low = h_high = h_sharp
        cons = h_dense[pred <= 1.10 * min_pred]
        h_sharp_cons = snap(float(cons.min())) if len(cons) else h_sharp
        interp_cols = {}
        gg = g.sort_values("h")
        logh = np.log(gg["h"].astype(float).to_numpy())
        for col in ["A_fit", "A_cross", "V_align", "V_norm", "p_active", "p_clip", "relative_disp"]:
            vals = gg[col].astype(float).to_numpy() if col in gg.columns else np.full(len(gg), np.nan)
            if np.all(~np.isfinite(vals)):
                interp_cols[col] = lambda x, v=np.nan: np.nan
            else:
                fill = np.nanmedian(vals[np.isfinite(vals)])
                vals = np.where(np.isfinite(vals), vals, fill)
                vals = np.maximum(vals, EPS) if col in {"A_fit", "A_cross", "V_norm", "p_active", "relative_disp"} else vals
                interp_cols[col] = lambda x, vals=vals, col=col: float(np.exp(np.interp(math.log(max(x, EPS)), logh, np.log(vals)))) if col in {"A_fit", "A_cross", "V_norm", "p_active", "relative_disp"} else float(np.interp(math.log(max(x, EPS)), logh, vals))
        default_h = 1e-3
        p_clip_base = float(np.nanmin(gg["p_clip"].astype(float))) if "p_clip" in gg else 0.0
        default_in_window = bool(default_h >= h_low / 1.001 and default_h <= h_high * 1.001)
        default_checks = (
            default_in_window
            and (interp_cols["V_align"](default_h) >= 0.70 if np.isfinite(interp_cols["V_align"](default_h)) else True)
            and (0.5 <= interp_cols["V_norm"](default_h) <= 1.5 if np.isfinite(interp_cols["V_norm"](default_h)) else True)
            and ((interp_cols["p_clip"](default_h) - p_clip_base) <= 0.05 if np.isfinite(interp_cols["p_clip"](default_h)) else True)
        )
        h_safe = default_h if default_checks else h_sharp_cons
        old = training[(training["task"] == task) & (training["mode"] == mode) & (training["h_policy"].astype(str).str.contains("hstar|env|cleanGL|lowbitL", case=False, na=False))]
        h_env = float(old.sort_values(["run_type", "best_dev_acc"], ascending=[True, False]).iloc[0]["h_value"]) if not old.empty else np.nan
        rows.append(
            {
                "model": "roberta-large",
                "task": task,
                "precision": "int4",
                "quantizer": "G128_RTNClip_fake_quant",
                "mode": mode,
                "h_default": default_h,
                "h_env": h_env,
                "h_sharp": h_sharp,
                "h_sharp_cons": h_sharp_cons,
                "h_safe": h_safe,
                "sharp_best_model": best["fit_model"],
                "sharp_fit_R2_log": best["R2_log"],
                "sharp_fit_RMSE_log": best["RMSE_log"],
                "default_in_sharp_window": default_in_window,
                "h_env_in_sharp_window": bool(np.isfinite(h_env) and h_env >= h_low / 1.001 and h_env <= h_high * 1.001),
                "sharp_window_low": h_low,
                "sharp_window_high": h_high,
                "selected_reason": "default_safe_keep_1e-3" if h_safe == default_h else "default_outside_or_failed_checks_use_sharp_cons",
                "A_default": interp_cols["A_fit"](default_h),
                "A_env": interp_cols["A_fit"](h_env) if np.isfinite(h_env) else np.nan,
                "A_sharp": interp_cols["A_fit"](h_sharp),
                "A_safe": interp_cols["A_fit"](h_safe),
                "V_align_default": interp_cols["V_align"](default_h),
                "V_align_sharp": interp_cols["V_align"](h_sharp),
                "p_active_default": interp_cols["p_active"](default_h),
                "p_active_sharp": interp_cols["p_active"](h_sharp),
                "relative_disp_default": interp_cols["relative_disp"](default_h),
                "relative_disp_sharp": interp_cols["relative_disp"](h_sharp),
            }
        )
    return pd.DataFrame(rows)


def best_training_for(training: pd.DataFrame, task: str, mode: str, h: float) -> Optional[pd.Series]:
    if not np.isfinite(h) or training.empty:
        return None
    g = training[(training["task"] == task) & (training["mode"] == mode)].copy()
    if g.empty:
        return None
    g["logdist"] = (np.log(g["h_value"].astype(float).clip(lower=EPS)) - math.log(h)).abs()
    g = g[g["logdist"] <= math.log(1.08)]
    if g.empty:
        return None
    order = {"full": 3, "medium": 2, "pilot": 1, "missing": 0, "unknown": 0}
    g["run_score"] = g["run_type"].map(order).fillna(0)
    return g.sort_values(["run_score", "best_dev_acc", "steps"], ascending=[False, False, False]).iloc[0]


def build_policy_comparison(candidates: pd.DataFrame, training: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = []
    train_rows = []
    missing = []
    for _, c in candidates.iterrows():
        task, mode = c["task"], c["mode"]
        refs = {}
        for policy, hcol in [
            ("default", "h_default"),
            ("old_env", "h_env"),
            ("sharp", "h_sharp"),
            ("sharp_cons", "h_sharp_cons"),
            ("safe", "h_safe"),
            ("fixed_small", None),
        ]:
            h = 1e-5 if policy == "fixed_small" else as_float(c.get(hcol))
            t = best_training_for(training, task, mode, h)
            if t is not None:
                refs[policy] = t
                train_rows.append(
                    {
                        "model": "roberta-large",
                        "task": task,
                        "precision": "int4",
                        "quantizer": "G128_RTNClip_fake_quant",
                        "mode": mode,
                        "h_policy": policy,
                        "h_value": h,
                        "seed": int(t["seed"]),
                        "run_type": t["run_type"],
                        "steps": int(t["steps"]),
                        "best_dev_acc": t["best_dev_acc"],
                        "final_dev_acc": t["final_dev_acc"],
                        "default_acc_reference": refs.get("default", t).get("best_dev_acc", np.nan) if hasattr(refs.get("default", t), "get") else np.nan,
                        "delta_vs_default": np.nan,
                        "old_env_acc_reference": refs.get("old_env", t).get("best_dev_acc", np.nan) if hasattr(refs.get("old_env", t), "get") else np.nan,
                        "delta_vs_old_env": np.nan,
                        "source_path": t["source_path"],
                        "early_stopped": bool(str(t.get("status", "")).lower() == "failed"),
                        "notes": "existing_training_reused",
                    }
                )
            elif policy in {"sharp", "sharp_cons", "safe"}:
                missing.append({"task": task, "mode": mode, "policy": policy, "h_value": h, "reason": "no_existing_training_at_selected_h"})
        default_acc = refs.get("default", pd.Series(dtype=object)).get("best_dev_acc", np.nan)
        env_acc = refs.get("old_env", pd.Series(dtype=object)).get("best_dev_acc", np.nan)
        sharp_acc = refs.get("sharp", pd.Series(dtype=object)).get("best_dev_acc", np.nan)
        cons_acc = refs.get("sharp_cons", pd.Series(dtype=object)).get("best_dev_acc", np.nan)
        safe_acc = refs.get("safe", pd.Series(dtype=object)).get("best_dev_acc", np.nan)
        accs = {"default": default_acc, "old_env": env_acc, "sharp": sharp_acc, "sharp_cons": cons_acc, "safe": safe_acc}
        valid = {k: v for k, v in accs.items() if np.isfinite(as_float(v))}
        best_policy = max(valid, key=valid.get) if valid else "missing"
        row = {
            "task": task,
            "mode": mode,
            "default_h": c["h_default"],
            "default_acc": default_acc,
            "h_env": c["h_env"],
            "h_env_acc": env_acc,
            "h_sharp": c["h_sharp"],
            "h_sharp_acc": sharp_acc,
            "h_sharp_cons": c["h_sharp_cons"],
            "h_sharp_cons_acc": cons_acc,
            "h_safe": c["h_safe"],
            "h_safe_acc": safe_acc,
            "best_policy": best_policy,
            "sharp_vs_default_delta": sharp_acc - default_acc if np.isfinite(sharp_acc) and np.isfinite(default_acc) else np.nan,
            "sharp_vs_env_delta": sharp_acc - env_acc if np.isfinite(sharp_acc) and np.isfinite(env_acc) else np.nan,
            "safe_vs_default_delta": safe_acc - default_acc if np.isfinite(safe_acc) and np.isfinite(default_acc) else np.nan,
            "run_type": ",".join(sorted(set(str(t["run_type"]) for t in refs.values()))) if refs else "missing",
            "seeds": ",".join(sorted(set(str(int(t["seed"])) for t in refs.values()))) if refs else "",
        }
        rows.append(row)
    comp = pd.DataFrame(rows)
    train_out = pd.DataFrame(train_rows)
    if not train_out.empty:
        for ref_col, policy_ref in [("default_acc_reference", "default"), ("old_env_acc_reference", "old_env")]:
            for idx, row in train_out.iterrows():
                task, mode = row["task"], row["mode"]
                ref = comp[(comp["task"] == task) & (comp["mode"] == mode)]
                val = ref.iloc[0]["default_acc" if policy_ref == "default" else "h_env_acc"] if not ref.empty else np.nan
                train_out.loc[idx, ref_col] = val
                if policy_ref == "default":
                    train_out.loc[idx, "delta_vs_default"] = row["best_dev_acc"] - val if np.isfinite(val) else np.nan
                else:
                    train_out.loc[idx, "delta_vs_old_env"] = row["best_dev_acc"] - val if np.isfinite(val) else np.nan
    return comp, train_out, pd.DataFrame(missing)


def win_tie_loss(comp: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for policy, acc_col in [("sharp", "h_sharp_acc"), ("sharp_cons", "h_sharp_cons_acc"), ("safe", "h_safe_acc"), ("old_env", "h_env_acc")]:
        vals = []
        for _, r in comp.iterrows():
            a = as_float(r.get(acc_col))
            d = as_float(r.get("default_acc"))
            if not np.isfinite(a) or not np.isfinite(d):
                continue
            delta = a - d
            vals.append(delta)
        rows.append(
            {
                "policy": policy,
                "num_win": sum(v > 0.005 for v in vals),
                "num_tie": sum(abs(v) <= 0.005 for v in vals),
                "num_loss": sum(v < -0.005 for v in vals),
                "avg_delta_vs_default": float(np.mean(vals)) if vals else np.nan,
                "num_compared": len(vals),
            }
        )
    return pd.DataFrame(rows)


def write_figures(out: Path, candidates: pd.DataFrame, comp: pd.DataFrame, fit_df: pd.DataFrame, fit_rows: pd.DataFrame) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def save(name: str):
        plt.tight_layout()
        plt.savefig(out / f"{name}.png", dpi=180)
        plt.savefig(out / f"{name}.pdf")
        plt.close()

    if not comp.empty:
        labels = [f"{r.task}\\n{r.mode}" for r in comp.itertuples()]
        x = np.arange(len(labels))
        width = 0.18
        plt.figure(figsize=(max(8, len(labels) * 0.85), 4.5))
        for i, (name, col) in enumerate([("default", "default_acc"), ("old_env", "h_env_acc"), ("sharp", "h_sharp_acc"), ("safe", "h_safe_acc")]):
            plt.bar(x + (i - 1.5) * width, comp[col].astype(float), width, label=name)
        plt.xticks(x, labels, rotation=45, ha="right")
        plt.ylabel("best dev acc")
        plt.legend()
        save("fig_roberta_int4_h_policy_accuracy")

        plt.figure(figsize=(max(8, len(labels) * 0.85), 4.5))
        for i, (name, col) in enumerate([("default", "default_h"), ("old_env", "h_env"), ("sharp", "h_sharp"), ("safe", "h_safe")]):
            plt.scatter(x + (i - 1.5) * 0.08, comp[col].astype(float), label=name)
        plt.yscale("log")
        plt.xticks(x, labels, rotation=45, ha="right")
        plt.ylabel("h")
        plt.legend()
        save("fig_roberta_int4_h_values")

        plt.figure(figsize=(7, 4.5))
        vals = []
        labs = []
        for _, r in comp.iterrows():
            if np.isfinite(as_float(r.get("h_sharp_acc"))) and np.isfinite(as_float(r.get("default_acc"))):
                vals.append(r["h_sharp_acc"] - r["default_acc"])
                labs.append(f"{r['task']}-{r['mode']}")
        plt.axhline(0, color="k", lw=0.8)
        plt.bar(range(len(vals)), vals)
        plt.xticks(range(len(vals)), labs, rotation=45, ha="right")
        plt.ylabel("sharp - default acc")
        save("fig_roberta_int4_probe_vs_accuracy")

    if not candidates.empty:
        plt.figure(figsize=(7, 4.5))
        x = np.arange(len(candidates))
        inside = candidates["h_env_in_sharp_window"].astype(int)
        plt.bar(x, inside)
        plt.xticks(x, [f"{r.task}\\n{r.mode}" for r in candidates.itertuples()], rotation=45, ha="right")
        plt.yticks([0, 1], ["outside", "inside"])
        plt.ylabel("h_env in sharp window")
        save("fig_h_env_inside_sharp_window")

    # Fit curves for available real RoBERTa configs.
    real_keys = candidates[["task", "mode"]].drop_duplicates().head(8)
    if not real_keys.empty:
        n = len(real_keys)
        fig, axes = plt.subplots(n, 1, figsize=(7, max(3, 2.2 * n)), squeeze=False)
        for ax, (_, rr) in zip(axes[:, 0], real_keys.iterrows()):
            key = f"roberta-large|{rr.task}|int4|{rr.mode}"
            g = fit_df[fit_df["group_key"] == key].sort_values("h")
            if g.empty:
                continue
            ax.loglog(g["h"], g["A_fit"], "o-", label="A/nMSE")
            fr = fit_rows[(fit_rows["group_key"] == key) & (fit_rows["fit_space"] == "log") & (fit_rows["fit_model"].isin(["M_sharp_norm", "MIA_loc", "M2"]))].sort_values("RMSE_log")
            if not fr.empty:
                best = fr.iloc[0]
                hd = np.logspace(np.log10(g["h"].min()), np.log10(g["h"].max()), 200)
                ax.loglog(hd, np.maximum(model_predict(best, hd, g), EPS), "--", label=best["fit_model"])
            cand = candidates[(candidates["task"] == rr.task) & (candidates["mode"] == rr.mode)].iloc[0]
            for name, col in [("default", "h_default"), ("env", "h_env"), ("sharp", "h_sharp"), ("safe", "h_safe")]:
                if np.isfinite(as_float(cand[col])):
                    ax.axvline(cand[col], lw=0.8, alpha=0.6, label=name)
            ax.set_title(f"{rr.task} {rr.mode}")
            ax.legend(fontsize=7)
        save("fig_roberta_int4_sharp_fit_curves")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path("."))
    parser.add_argument("--output_dir", type=Path, default=Path("sharp_interval_fit_and_roberta_int4_eval"))
    args = parser.parse_args()
    repo = args.repo.resolve()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    fit_input = load_fit_inputs(repo)
    training = load_training_index(repo)
    fit_input.to_csv(out / "fit_input_index.csv", index=False)
    training.to_csv(out / "training_index_roberta_int4.csv", index=False)

    fit_df = prepare_fit_frame(fit_input)
    if not fit_df.empty:
        fit_df["group_key"] = fit_df["config_key"].astype(str)
    all_fit_rows = []
    for key, g in fit_df.groupby("group_key"):
        if g["h"].nunique() < 3 or len(g) < 3:
            continue
        fits = fit_group(g)
        for row in fits:
            row["group_key"] = key
            first = g.iloc[0]
            for col in ["model", "task", "precision", "quantizer", "mode", "experiment_type"]:
                row[col] = first.get(col)
            all_fit_rows.append(row)
    fit_rows = pd.DataFrame(all_fit_rows)
    if fit_rows.empty:
        fit_rows = pd.DataFrame(columns=["group_key", "fit_model", "fit_space", "R2_log", "RMSE_log"])
    fit_rows.to_csv(out / "fit_model_comparison.csv", index=False)
    coeff_cols = [c for c in fit_rows.columns if c not in {"status"}]
    fit_rows[coeff_cols].to_csv(out / "model_coefficients.csv", index=False)

    best = fit_rows[fit_rows["fit_space"] == "log"].sort_values(["group_key", "RMSE_log", "R2_log"], ascending=[True, True, False]).groupby("group_key", as_index=False).head(1)
    best.to_csv(out / "hstar_comparison.csv", index=False)

    candidates = build_candidates(fit_df, fit_rows, training)
    candidates.to_csv(out / "roberta_int4_h_candidates.csv", index=False)
    comp, train_results, missing_train = build_policy_comparison(candidates, training)
    train_results.to_csv(out / "roberta_int4_training_results.csv", index=False)
    comp.to_csv(out / "roberta_int4_policy_comparison.csv", index=False)
    wtl = win_tie_loss(comp)
    wtl.to_csv(out / "roberta_int4_win_tie_loss_summary.csv", index=False)
    missing_train.to_csv(out / "missing_training_runs.csv", index=False)

    # Training launch manifest for missing selected h runs.
    launch_rows = []
    for _, r in missing_train.iterrows():
        if r["policy"] == "safe" and abs(as_float(r["h_value"]) - 1e-3) < 1e-12:
            continue
        launch_rows.append(
            {
                "priority": "P0" if r["task"] in {"sst-2", "sst-5", "rte", "trec"} and r["mode"] == "dense" else "P1",
                "task": r["task"],
                "mode": r["mode"],
                "policy": r["policy"],
                "h": r["h_value"],
                "suggested_command": (
                    "DATALOADER_SHUFFLE=True CUDA_VISIBLE_DEVICES=0 python tools/rtnclip_roberta_sst5_batch.py "
                    f"--output_root outputs/sharp_interval_roberta_int4_eval --task_name {r['task']} --dataset_mode full "
                    f"--run_dir outputs/sharp_interval_roberta_int4_eval/int4_hsearch/dense/int4_dense_{r['task']}_{r['policy']}_h{r['h_value']:.6g}_seed16_full_bs64_step20k "
                    f"--run_name int4_dense_{r['task']}_{r['policy']}_h{r['h_value']:.6g}_seed16_full_bs64_step20k "
                    f"--bitwidth 4 --h {r['h_value']} --h_label {r['policy']} --steps 20000 --eval_every 1000 "
                    "--checkpoint_steps 1000 --batch_size 64 --seed 16 --data_seed 16 --scale_refresh_k 1 train-one"
                    if r["mode"] == "dense"
                    else "manual_or_existing_mode_wrapper_required"
                ),
            }
        )
    pd.DataFrame(launch_rows).to_csv(out / "roberta_int4_missing_training_launch_manifest.csv", index=False)

    audit_lines = [
        "# Data audit",
        "",
        f"- fit input rows: {len(fit_input)}",
        f"- fit-ready rows: {len(fit_df)}",
        f"- fitted groups: {fit_rows['group_key'].nunique() if not fit_rows.empty else 0}",
        f"- RoBERTa INT4 training rows indexed: {len(training)}",
        f"- candidate configs: {len(candidates)}",
        f"- missing selected training rows: {len(missing_train)}",
        "",
        "## Available A_true / nMSE sources",
        fit_input.groupby(["experiment_type"]).size().to_string() if not fit_input.empty else "none",
        "",
        "## Fit target policy",
        "- Fit target `A_fit` is restricted to paper-compatible loss-level directional MSE/NMSE rows.",
        "- Geometry-only fields such as `A_cross`, `A_interval`, `sigma_raw2`, `delta_visibility_nmse`, and current `lowbit_true_nmse=dequantized_effective_displacement_nmse_v1` are not used as the fit target.",
        "- They can only enter sharp/interval-aware models as covariates.",
        "",
        "## Target kind counts",
        fit_input.groupby(["target_kind"]).size().to_string() if "target_kind" in fit_input.columns and not fit_input.empty else "none",
        "",
        "## RoBERTa INT4 training rows by task/mode/run_type",
        training.groupby(["task", "mode", "run_type"]).size().to_string() if not training.empty else "none",
    ]
    (out / "data_audit.md").write_text("\n".join(audit_lines) + "\n", encoding="utf-8")

    takeaways = [
        "# Sharp fit takeaways",
        "",
        "- M2, M4, Mp, M_sharp_norm, M_sharp_constrained, and MIA_loc were fit with nonnegative coefficients.",
        "- Fit target is paper-compatible loss-level directional MSE/NMSE only: `(d_Q(h,u)-d_ref(u))^2`, optionally normalized by `E[d_ref^2]`.",
        "- Interval geometry metrics (`A_cross`, `A_interval`, effective-displacement lowbit nMSE) are explanatory covariates, not the target.",
        "- Main ranking uses log-space RMSE; rows with clipping >5% are excluded when enough clean points exist.",
        "- h_sharp is probe-only: it is selected from fitted/probe metrics, not from training accuracy.",
        "- h_safe keeps h=1e-3 whenever default lies inside the sharp window and passes visibility/locality checks.",
        "",
    ]
    if not fit_rows.empty:
        top = fit_rows[fit_rows["fit_space"] == "log"].sort_values("RMSE_log").head(15)
        takeaways += ["## Best log-space fits", "", simple_markdown(top[["group_key", "fit_model", "R2_log", "RMSE_log", "h_star_pred", "status"]]), ""]
    (out / "sharp_fit_takeaways.md").write_text("\n".join(takeaways) + "\n", encoding="utf-8")

    fail_lines = [
        "# RoBERTa INT4 failure diagnosis",
        "",
        "Missing or underperforming h_sharp rows are not filled in with synthetic values.",
        "Use `roberta_int4_missing_training_launch_manifest.csv` for exact missing training candidates.",
        "",
    ]
    if not comp.empty:
        bad = comp[np.isfinite(comp["sharp_vs_default_delta"].astype(float)) & (comp["sharp_vs_default_delta"].astype(float) < -0.005)]
        if bad.empty:
            fail_lines.append("- No existing h_sharp comparison is a clear loss against default under the 0.005 threshold.")
        else:
            fail_lines.append("## Clear sharp losses vs default")
            fail_lines.append(simple_markdown(bad[["task", "mode", "h_sharp", "h_sharp_acc", "default_acc", "sharp_vs_default_delta"]]))
    (out / "roberta_int4_failure_diagnosis.md").write_text("\n".join(fail_lines) + "\n", encoding="utf-8")

    exp_lines_cn = [
        "# RoBERTa INT4 experiment takeaways",
        "",
        "## 中文",
        "",
        "1. 旧公式不需要完全废弃；它可以作为 coarse-envelope 的 h_env。",
        "2. h_sharp 是更细的 interval-aware probe 诊断，不应声称总能提升训练 accuracy。",
        "3. practical 主方法更适合用 h_safe：default 在 sharp window 内就保留 1e-3，否则才覆盖。",
        "4. 已有 full 训练会被复用；缺失的 h_sharp/full 训练在 manifest 中列出，没有编造。",
        "5. 如果 h_env 落在 sharp window 内，旧实验可以保留并解释为 coarse-envelope radius。",
        "",
        "## English",
        "",
        "1. The old formula should not be discarded; it remains a coarse-envelope radius.",
        "2. h_sharp is a sharper interval-aware perturbation diagnostic, not a guarantee of better accuracy.",
        "3. h_safe is the practical rule: keep default when it is inside the sharp window; override only when unsafe.",
        "4. Existing full training results are reused; missing sharp/full rows are explicitly reported.",
        "5. If h_env lies inside the sharp window, previous experiments remain valid as coarse-envelope runs.",
    ]
    (out / "roberta_int4_experiment_takeaways.md").write_text("\n".join(exp_lines_cn) + "\n", encoding="utf-8")

    missing_lines = [
        "# Missing items",
        "",
        "- Real-model sharp fitting is limited by available loss-level nMSE/probe grids; geometry-only rows cannot provide A_true.",
        "- New full RoBERTa INT4 training is not fabricated. Missing selected h runs are listed in `roberta_int4_missing_training_launch_manifest.csv`.",
        "- Dense multi-task default full rows are incomplete in the discovered outputs for some tasks; existing 2k rows are marked pilot.",
    ]
    if not missing_train.empty:
        missing_lines += ["", "## Missing selected training rows", simple_markdown(missing_train)]
    (out / "missing_items.md").write_text("\n".join(missing_lines) + "\n", encoding="utf-8")

    metadata = {
        "hostname": socket.gethostname(),
        "git_commit": run_cmd(["git", "rev-parse", "HEAD"], repo),
        "python": run_cmd(["python", "--version"], repo),
        "output_dir": str(out),
        "h_grid": H_GRID.tolist(),
    }
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    try:
        write_figures(out, candidates, comp, fit_df, fit_rows)
    except Exception as exc:
        with (out / "missing_items.md").open("a", encoding="utf-8") as f:
            f.write(f"\n- Figure generation failed: {exc}\n")

    zip_path = Path("sharp_interval_fit_and_roberta_int4_eval.zip")
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in out.rglob("*"):
            if p.is_file():
                zf.write(p, arcname=str(p.relative_to(out.parent)))
    print(f"Wrote {out} and {zip_path}")


if __name__ == "__main__":
    main()
