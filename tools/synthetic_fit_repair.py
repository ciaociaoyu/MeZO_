#!/usr/bin/env python
"""Repair synthetic h-window fits for perturbation-window theory.

The goal is a clean synthetic suite, not a broad parameter sweep.  It separates
visibility-only, locality-only, and combined quantized nonlinear regimes, then
fits several nonnegative models:

  M2:    alpha / h^2 + beta h^2 + gamma
  M4:    alpha / h^2 + beta h^4 + gamma
  Mp:    alpha / h^2 + beta h^p + gamma
  MIAp:  c_vis A_interval_grad(h) + beta h^p + gamma
  MIA_loc: c_vis A_interval_grad(h) + c_loc M_loc_true(h) + gamma

All outputs are written under synthetic_fit_repair/.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import shutil
import socket
import subprocess
import sys
import time
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
H_GRID_BASE = np.array(
    [
        1e-9,
        3e-9,
        1e-8,
        3e-8,
        1e-7,
        3e-7,
        1e-6,
        3e-6,
        1e-5,
        3e-5,
        1e-4,
        3e-4,
        1e-3,
        3e-3,
        1e-2,
        3e-2,
        1e-1,
        3e-1,
        1.0,
    ],
    dtype=np.float64,
)
P_GRID = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]


@dataclass(frozen=True)
class Config:
    config_id: str
    family: str
    d: int
    n_dirs: int
    direction: str
    qmode: str
    qbits: int
    delta: float
    clip: bool
    qrange_scale: float
    groupwise: bool
    group_size: int
    scale_sigma: float
    nonlinear_a: float
    c_hetero: float
    active_p: float = 1.0


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    except Exception:
        return ""


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def stable_log_cosh(x: torch.Tensor) -> torch.Tensor:
    ax = torch.abs(x)
    return ax + torch.nn.functional.softplus(-2.0 * ax) - math.log(2.0)


def make_delta_vec(cfg: Config, device: torch.device, gen: torch.Generator) -> torch.Tensor:
    if not cfg.groupwise:
        return torch.full((cfg.d,), cfg.delta, device=device)
    n_groups = math.ceil(cfg.d / cfg.group_size)
    scales = torch.exp(torch.randn(n_groups, device=device, generator=gen) * cfg.scale_sigma)
    return (cfg.delta * scales).repeat_interleave(cfg.group_size)[: cfg.d]


def quantize(x: torch.Tensor, delta: torch.Tensor, cfg: Config) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if cfg.qmode == "identity":
        code = x / delta
        clip_mask = torch.zeros_like(x, dtype=torch.bool)
        return x, code, clip_mask
    code = torch.round(x / delta)
    if cfg.clip:
        qmin = -(2 ** (cfg.qbits - 1)) * cfg.qrange_scale
        qmax = (2 ** (cfg.qbits - 1) - 1) * cfg.qrange_scale
        clipped = (code <= qmin) | (code >= qmax)
        code = code.clamp(qmin, qmax)
    else:
        clipped = torch.zeros_like(code, dtype=torch.bool)
    return code * delta, code, clipped


def make_problem(cfg: Config, device: torch.device, gen: torch.Generator) -> Dict[str, torch.Tensor]:
    delta = make_delta_vec(cfg, device, gen)
    # Keep weights in a numerically safe range. Quantized configs use grid-scale
    # initialization so visibility is controlled by Delta rather than base clipping.
    if cfg.qmode == "identity":
        w = torch.empty(cfg.d, device=device).uniform_(-0.25, 0.25, generator=gen)
    else:
        base_code = torch.empty(cfg.d, device=device).uniform_(-4.0, 4.0, generator=gen)
        w = base_code * delta
    if cfg.family == "linear":
        g = torch.randn(cfg.d, device=device, generator=gen)
        a = torch.ones(cfg.d, device=device)
        c = torch.ones(cfg.d, device=device)
    else:
        # Log-cosh: F_i=(c_i/a_i) log cosh(a_i w_i), grad_i=c_i tanh(a_i w_i).
        a = cfg.nonlinear_a * torch.exp(0.25 * torch.randn(cfg.d, device=device, generator=gen))
        c = torch.exp(cfg.c_hetero * torch.randn(cfg.d, device=device, generator=gen))
        g = c * torch.tanh(a * w)
    if cfg.active_p < 1.0:
        mask = torch.zeros(cfg.d, dtype=torch.bool, device=device)
        k = max(1, int(round(cfg.d * cfg.active_p)))
        idx = torch.randperm(cfg.d, device=device, generator=gen)[:k]
        mask[idx] = True
    else:
        mask = torch.ones(cfg.d, dtype=torch.bool, device=device)
    return {"w": w, "delta": delta, "g": g, "a": a, "c": c, "mask": mask}


def eval_F(x: torch.Tensor, cfg: Config, problem: Dict[str, torch.Tensor]) -> torch.Tensor:
    if cfg.family == "linear":
        return torch.sum(problem["g"].unsqueeze(0) * x, dim=1)
    a = problem["a"].unsqueeze(0)
    c = problem["c"].unsqueeze(0)
    return torch.sum((c / a) * stable_log_cosh(a * x), dim=1)


def run_config(cfg: Config, h_grid: np.ndarray, device: torch.device, seed: int) -> pd.DataFrame:
    gen = torch.Generator(device=device)
    gen.manual_seed(seed + abs(hash(cfg.config_id)) % 1_000_000)
    problem = make_problem(cfg, device, gen)
    w = problem["w"]
    delta = problem["delta"]
    g = problem["g"]
    mask = problem["mask"]
    g_norm_sq = float(torch.sum(g * g).item())
    delta_rms = float(torch.sqrt(torch.mean(delta * delta)).item())
    delta_grad = math.sqrt(float(torch.sum(g * g * delta * delta).item()) / max(g_norm_sq, 1e-30))
    rows: List[Dict[str, Any]] = []
    chunk = min(16, cfg.n_dirs)

    for h in h_grid:
        accum = {
            "mse": 0.0,
            "dstar2": 0.0,
            "interval_uniform": 0.0,
            "interval_grad": 0.0,
            "active": 0.0,
            "jump": 0.0,
            "jump0": 0.0,
            "jump1": 0.0,
            "jumpge2": 0.0,
            "norm_ratio": 0.0,
            "align": 0.0,
            "clip": 0.0,
            "mloc": 0.0,
            "disp": 0.0,
            "relative_disp": 0.0,
            "locality_proxy": 0.0,
        }
        seen = 0
        while seen < cfg.n_dirs:
            cur = min(chunk, cfg.n_dirs - seen)
            u = torch.randn(cur, cfg.d, device=device, generator=gen)
            u[:, ~mask] = 0.0
            z = u / math.sqrt(cfg.d) if cfg.direction == "normalized" else u
            plus = w.unsqueeze(0) + float(h) * z
            minus = w.unsqueeze(0) - float(h) * z
            q_plus, code_plus, clip_plus = quantize(plus, delta.unsqueeze(0), cfg)
            q_minus, code_minus, clip_minus = quantize(minus, delta.unsqueeze(0), cfg)
            delta_q = q_plus - q_minus
            b = delta_q / (2.0 * float(h))
            d_star = torch.sum(g.unsqueeze(0) * z, dim=1)
            d_q = (eval_F(q_plus, cfg, problem) - eval_F(q_minus, cfg, problem)) / (2.0 * float(h))
            d_lin = torch.sum(g.unsqueeze(0) * b, dim=1)
            err = d_q - d_star
            interval_err = b - z
            jump = torch.abs(code_plus - code_minus)
            intended = 2.0 * float(h) * z
            norm_delta = torch.linalg.vector_norm(delta_q, dim=1)
            norm_intended = torch.linalg.vector_norm(intended, dim=1).clamp_min(1e-30)
            dot = torch.sum(delta_q * intended, dim=1)
            align = dot / (norm_delta.clamp_min(1e-30) * norm_intended)
            e_plus = q_plus - w.unsqueeze(0)
            e_minus = q_minus - w.unsqueeze(0)
            disp2 = 0.5 * (torch.sum(e_plus * e_plus, dim=1) + torch.sum(e_minus * e_minus, dim=1))
            clip = clip_plus | clip_minus
            accum["mse"] += float(torch.sum(err * err).item())
            accum["dstar2"] += float(torch.sum(d_star * d_star).item())
            accum["interval_uniform"] += float(torch.sum(interval_err * interval_err).item())
            accum["interval_grad"] += float(torch.sum((g.unsqueeze(0) ** 2) * (interval_err**2)).item())
            accum["active"] += float(torch.sum((jump > 0).float()).item())
            accum["jump"] += float(torch.sum(jump.float()).item())
            accum["jump0"] += float(torch.sum((jump == 0).float()).item())
            accum["jump1"] += float(torch.sum((jump == 1).float()).item())
            accum["jumpge2"] += float(torch.sum((jump >= 2).float()).item())
            accum["norm_ratio"] += float(torch.sum(norm_delta / norm_intended).item())
            accum["align"] += float(torch.sum(torch.nan_to_num(align, nan=0.0)).item())
            accum["clip"] += float(torch.sum(clip.float()).item())
            accum["mloc"] += float(torch.sum((d_q - d_lin) ** 2).item())
            accum["disp"] += float(torch.sum(disp2).item())
            accum["relative_disp"] += float(torch.sum(torch.sqrt(disp2)).item())
            accum["locality_proxy"] += float(torch.sum((2.0 * disp2) ** 2 / (16.0 * float(h) * float(h) + 1e-30)).item())
            seen += cur

        denom = max(cfg.n_dirs, 1)
        coord_denom = max(cfg.n_dirs * cfg.d, 1)
        dstar2 = max(accum["dstar2"], 1e-30)
        rows.append(
            {
                "config_id": cfg.config_id,
                "family": cfg.family,
                "d": cfg.d,
                "n_dirs": cfg.n_dirs,
                "direction": cfg.direction,
                "qmode": cfg.qmode,
                "qbits": cfg.qbits,
                "Delta": cfg.delta,
                "Delta_rms": delta_rms,
                "Delta_grad": delta_grad,
                "clip_enabled": cfg.clip,
                "qrange_scale": cfg.qrange_scale,
                "groupwise": cfg.groupwise,
                "group_size": cfg.group_size,
                "scale_sigma": cfg.scale_sigma,
                "nonlinear_a": cfg.nonlinear_a,
                "c_hetero": cfg.c_hetero,
                "active_p": cfg.active_p,
                "h": float(h),
                "A_true": accum["mse"] / dstar2,
                "A_interval_uniform": accum["interval_uniform"] / coord_denom,
                "A_interval_grad": accum["interval_grad"] / (denom * max(g_norm_sq, 1e-30)),
                "A_coarse_delta_rms": (delta_rms**2) / (4.0 * float(h) ** 2),
                "A_coarse_delta_grad": (delta_grad**2) / (4.0 * float(h) ** 2),
                "p_active": accum["active"] / coord_denom,
                "jump_mean": accum["jump"] / coord_denom,
                "jump_zero_frac": accum["jump0"] / coord_denom,
                "jump_one_frac": accum["jump1"] / coord_denom,
                "jump_ge2_frac": accum["jumpge2"] / coord_denom,
                "V_norm": accum["norm_ratio"] / denom,
                "V_align": accum["align"] / denom,
                "p_clip": accum["clip"] / coord_denom,
                "M_loc_true": accum["mloc"] / dstar2,
                "disp_rms": math.sqrt(accum["disp"] / coord_denom),
                "relative_disp": accum["relative_disp"] / (denom * max(float(torch.linalg.vector_norm(w).item()), 1e-30)),
                "locality_proxy": accum["locality_proxy"] / denom,
                "dstar2_mean": accum["dstar2"] / denom,
                "g_norm_sq": g_norm_sq,
                "V_dir_gaussian": (cfg.d + 1.0) * g_norm_sq,
                "V_dir_eff": (int(mask.sum().item()) + 1.0) * g_norm_sq,
            }
        )
    return pd.DataFrame(rows)


def nnls_enumerate(X: np.ndarray, y: np.ndarray, weights: Optional[np.ndarray] = None) -> Tuple[np.ndarray, float]:
    if weights is not None:
        sw = np.sqrt(weights).reshape(-1, 1)
        Xw = X * sw
        yw = y * sw.reshape(-1)
    else:
        Xw, yw = X, y
    n = X.shape[1]
    best_coef = np.zeros(n)
    best_sse = float("inf")
    for r in range(1, n + 1):
        for subset in itertools.combinations(range(n), r):
            sub = list(subset)
            try:
                coef_sub, *_ = np.linalg.lstsq(Xw[:, sub], yw, rcond=None)
            except Exception:
                continue
            if np.any(coef_sub < -1e-14):
                continue
            coef = np.zeros(n)
            coef[sub] = np.maximum(coef_sub, 0.0)
            pred = Xw @ coef
            sse = float(np.sum((pred - yw) ** 2))
            if sse < best_sse:
                best_sse = sse
                best_coef = coef
    return best_coef, best_sse


def fit_one(group: pd.DataFrame, model: str, p_value: Optional[float], fit_space: str) -> Dict[str, Any]:
    clean = group[np.isfinite(group["A_true"]) & (group["A_true"] >= 0) & (group["p_clip"] <= 0.05)].copy()
    if len(clean) < 5:
        clean = group[np.isfinite(group["A_true"]) & (group["A_true"] >= 0)].copy()
    h = clean["h"].to_numpy(dtype=float)
    y = clean["A_true"].to_numpy(dtype=float)
    eps = max(float(np.nanmin(y[y > 0])) * 1e-3 if np.any(y > 0) else 1e-12, 1e-12)
    if model == "M2":
        names = ["alpha", "beta", "gamma"]
        X = np.stack([1.0 / (h**2), h**2, np.ones_like(h)], axis=1)
        p = 2.0
    elif model == "M4":
        names = ["alpha", "beta", "gamma"]
        X = np.stack([1.0 / (h**2), h**4, np.ones_like(h)], axis=1)
        p = 4.0
    elif model == "Mp":
        p = float(p_value)
        names = ["alpha", "beta", "gamma"]
        X = np.stack([1.0 / (h**2), h**p, np.ones_like(h)], axis=1)
    elif model == "MIA2":
        p = 2.0
        names = ["c_vis", "beta", "gamma"]
        X = np.stack([clean["A_interval_grad"].to_numpy(dtype=float), h**2, np.ones_like(h)], axis=1)
    elif model == "MIAp":
        p = float(p_value)
        names = ["c_vis", "beta", "gamma"]
        X = np.stack([clean["A_interval_grad"].to_numpy(dtype=float), h**p, np.ones_like(h)], axis=1)
    elif model == "MIA_loc":
        p = np.nan
        names = ["c_vis", "c_loc", "gamma"]
        X = np.stack(
            [
                clean["A_interval_grad"].to_numpy(dtype=float),
                clean["M_loc_true"].to_numpy(dtype=float),
                np.ones_like(h),
            ],
            axis=1,
        )
    else:
        raise ValueError(model)
    valid = np.all(np.isfinite(X), axis=1) & np.isfinite(y)
    X, y, h = X[valid], y[valid], h[valid]
    if len(y) < 4:
        return {"status": "fit_failed_too_few_points", "fit_model": model, "p": p, "fit_space": fit_space}
    weights = None
    if fit_space == "log":
        weights = 1.0 / np.maximum(y, eps) ** 2
        weights = np.minimum(weights, np.nanpercentile(weights, 90))
    coef, sse = nnls_enumerate(X, y, weights=weights)
    pred = np.maximum(X @ coef, eps)
    log_y = np.log(np.maximum(y, eps))
    log_pred = np.log(pred)
    rmse_log = float(np.sqrt(np.mean((log_y - log_pred) ** 2)))
    ss_log = float(np.sum((log_y - log_y.mean()) ** 2))
    r2_log = 1.0 - float(np.sum((log_y - log_pred) ** 2)) / ss_log if ss_log > 0 else np.nan
    ss = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - float(np.sum((y - pred) ** 2)) / ss if ss > 0 else np.nan
    out = {
        "fit_model": model,
        "fit_space": fit_space,
        "p": p,
        "R2_linear": r2,
        "R2_log": r2_log,
        "RMSE_log": rmse_log,
        "clean_points": int(len(y)),
        "h_fit_min": float(np.min(h)),
        "h_fit_max": float(np.max(h)),
        "status": "ok",
    }
    for name, value in zip(names, coef):
        out[name] = float(value)
    alpha = float(out.get("alpha", np.nan))
    beta = float(out.get("beta", np.nan))
    if math.isfinite(alpha) and math.isfinite(beta) and alpha > 0 and beta > 0 and math.isfinite(p):
        out["h_star"] = float((2.0 * alpha / (p * beta)) ** (1.0 / (p + 2.0)))
    else:
        # For interval-aware models, use fitted grid minimum.
        idx = int(np.argmin(pred))
        out["h_star"] = float(h[idx])
    h_sorted = np.sort(h)
    hs = out["h_star"]
    out["h_star_interior"] = bool(math.isfinite(hs) and hs > h_sorted[min(1, len(h_sorted) - 1)] and hs < h_sorted[max(len(h_sorted) - 2, 0)])
    out["left_tail_visible"] = bool(y[0] >= 2.0 * np.min(y))
    out["right_tail_visible"] = bool(y[-1] >= 2.0 * np.min(y))
    if out.get("alpha", 1.0) <= 0:
        out["status"] = "left_tail_missing"
    if model in {"M2", "M4", "Mp", "MIA2", "MIAp"} and out.get("beta", 1.0) <= 0:
        out["status"] = "right_tail_missing"
    if not out["h_star_interior"]:
        out["status"] = f"{out['status']};boundary_solution"
    return out


def fit_all(raw: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    fit_rows: List[Dict[str, Any]] = []
    best_rows: List[Dict[str, Any]] = []
    compare_rows: List[Dict[str, Any]] = []
    repair_rows: List[Dict[str, Any]] = []
    for cid, group in raw.groupby("config_id"):
        meta_cols = [
            "config_id",
            "family",
            "d",
            "direction",
            "qmode",
            "qbits",
            "Delta",
            "clip_enabled",
            "groupwise",
            "scale_sigma",
            "nonlinear_a",
            "active_p",
        ]
        meta = {c: group.iloc[0][c] for c in meta_cols if c in group.columns}
        model_specs: List[Tuple[str, Optional[float]]] = [("M2", None), ("M4", None), ("MIA2", None), ("MIA_loc", None)]
        model_specs.extend(("Mp", p) for p in P_GRID)
        model_specs.extend(("MIAp", p) for p in P_GRID)
        for model, p in model_specs:
            for space in ["linear", "log"]:
                row = {**meta, **fit_one(group, model, p, space)}
                fit_rows.append(row)
                compare_rows.append(row.copy())
        fdf = pd.DataFrame([r for r in fit_rows if r.get("config_id") == cid and r.get("fit_space") == "log"])
        ok = fdf[fdf["R2_log"].notna()].sort_values(["RMSE_log", "R2_log"], ascending=[True, False])
        best = ok.iloc[0].to_dict() if not ok.empty else {**meta, "fit_model": "none", "status": "fit_failed"}
        reason = "best_log_rmse"
        if meta.get("family") == "linear":
            reason = "visibility_only_oracle_not_for_full_u_shape"
        elif best.get("fit_model") == "M4":
            reason = "central_difference_locality_tail_prefers_h4"
        elif str(best.get("fit_model", "")).startswith("MIA"):
            reason = "interval_aware_model_best"
        best["selection_reason"] = reason
        best_rows.append(best)

        # Lightweight repair log.
        m2 = fdf[fdf["fit_model"].eq("M2")]
        if not m2.empty:
            r = m2.iloc[0]
            failure = []
            if not bool(r.get("left_tail_visible", False)):
                failure.append("left_tail_missing")
            if not bool(r.get("right_tail_visible", False)):
                failure.append("right_tail_missing")
            if not bool(r.get("h_star_interior", False)):
                failure.append("boundary_solution")
            if float(r.get("R2_log", np.nan)) < 0.90:
                failure.append("low_R2_log")
            repair_rows.append(
                {
                    "attempt": 0,
                    "family": meta.get("family"),
                    "config_id": cid,
                    "failure_reason": ";".join(failure) if failure else "none",
                    "action_taken": "compare_M4_Mp_MIA_and_clean_clip_range",
                    "result_status": "ok" if not failure else "diagnosed",
                    "best_model": best.get("fit_model"),
                    "R2_log": best.get("R2_log"),
                    "h_star": best.get("h_star"),
                }
            )
    return pd.DataFrame(fit_rows), pd.DataFrame(best_rows), pd.DataFrame(compare_rows), pd.DataFrame(repair_rows)


def window_summary(raw: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cid, g in raw.groupby("config_id"):
        clean = g[g["p_clip"] <= 0.05].copy()
        if clean.empty:
            clean = g.copy()
        y = clean["A_true"].to_numpy(dtype=float)
        h = clean["h"].to_numpy(dtype=float)
        min_y = np.nanmin(y)
        for kappa in [1.05, 1.10, 1.25, 1.50]:
            ok = y <= kappa * min_y
            rows.append(
                {
                    "config_id": cid,
                    "kappa": kappa,
                    "h_min": float(np.nanmin(h[ok])) if ok.any() else np.nan,
                    "h_max": float(np.nanmax(h[ok])) if ok.any() else np.nan,
                    "window_width_log10": float(np.log10(np.nanmax(h[ok]) / np.nanmin(h[ok]))) if ok.sum() > 1 else 0.0 if ok.any() else np.nan,
                    "default_h_in_window": bool(ok[np.argmin(np.abs(h - 1e-3))]) if len(h) else False,
                }
            )
    return pd.DataFrame(rows)


def old_diagnosis(out_dir: Path) -> None:
    zip_path = REPO_ROOT / "hwindow_12h_highdim_bundle.zip"
    lines = ["# Synthetic Old Fit Diagnosis", ""]
    if not zip_path.exists():
        lines.append("No previous hwindow_12h_highdim_bundle.zip found.")
        (out_dir / "synthetic_old_fit_diagnosis.md").write_text("\n".join(lines) + "\n")
        return
    import zipfile

    with zipfile.ZipFile(zip_path) as zf:
        names = set(zf.namelist())
        fit_name = "hwindow_12h_highdim_bundle/synthetic_highdim_fit.csv"
        raw_name = "hwindow_12h_highdim_bundle/synthetic_highdim_raw.csv"
        if fit_name not in names:
            lines.append("Previous zip did not contain synthetic_highdim_fit.csv.")
        else:
            fit = pd.read_csv(zf.open(fit_name))
            bad = fit[(fit.get("alpha", 0).fillna(0) <= 0) | (fit.get("beta", 0).fillna(0) <= 0) | (~np.isfinite(fit.get("h_star", np.nan)))]
            lines.append(f"- Previous fit rows: {len(fit)}")
            lines.append(f"- Rows with alpha<=0, beta<=0, or invalid h_star: {len(bad)}")
            if "metric" in fit.columns:
                lines.append("")
                lines.append("Bad rows by metric:")
                lines.append(bad.groupby("metric").size().reset_index(name="rows").to_csv(index=False).strip())
            lines.append("")
            lines.append("Likely causes:")
            lines.append("- Linear oracle rows naturally lack a locality tail, so beta and h_star are not meaningful.")
            lines.append("- Ideal symmetric central differences on smooth oracles often produce a squared h^4 tail, not h^2.")
            lines.append("- Some previous configs mixed clipping/saturation regimes into a single OLS fit.")
            lines.append("- Linear-space OLS can be dominated by extreme small-h points; log/weighted fits are needed.")
        if raw_name in names:
            raw = pd.read_csv(zf.open(raw_name), usecols=lambda c: c in {"config_id", "A_true", "A_interval_grad", "p_clip"})
            if {"A_true", "A_interval_grad"}.issubset(raw.columns):
                corr = np.corrcoef(np.log(np.maximum(raw["A_true"], 1e-12)), np.log(np.maximum(raw["A_interval_grad"], 1e-12)))[0, 1]
                lines.append("")
                lines.append(f"Global log-correlation between A_interval_grad and A_true in previous raw table: {corr:.4f}")
                lines.append("This suggests interval-aware terms are still useful even when alpha/beta envelope fits fail.")
    (out_dir / "synthetic_old_fit_diagnosis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_results(out_dir: Path, raw: pd.DataFrame, fit: pd.DataFrame, best: pd.DataFrame, windows: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt

    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    def fitted_curve(g: pd.DataFrame, row: pd.Series) -> np.ndarray:
        h = g["h"].to_numpy(dtype=float)
        model = row["fit_model"]
        if model in {"M2", "M4", "Mp"}:
            p = float(row["p"])
            return row.get("alpha", 0.0) / (h**2) + row.get("beta", 0.0) * (h**p) + row.get("gamma", 0.0)
        if model in {"MIA2", "MIAp"}:
            p = float(row["p"])
            return row.get("c_vis", 0.0) * g["A_interval_grad"].to_numpy(dtype=float) + row.get("beta", 0.0) * (h**p) + row.get("gamma", 0.0)
        if model == "MIA_loc":
            return row.get("c_vis", 0.0) * g["A_interval_grad"].to_numpy(dtype=float) + row.get("c_loc", 0.0) * g["M_loc_true"].to_numpy(dtype=float) + row.get("gamma", 0.0)
        return np.full_like(h, np.nan)

    combined_candidates = best[best["family"].eq("combined")]
    clean = combined_candidates.sort_values("RMSE_log").head(1)
    clean_id = clean.iloc[0]["config_id"] if not clean.empty else raw["config_id"].iloc[0]
    g = raw[raw["config_id"].eq(clean_id)].sort_values("h")
    best_row = best[best["config_id"].eq(clean_id)].iloc[0]

    plt.figure(figsize=(7, 5))
    plt.loglog(g["h"], g["A_true"], "o-", label="A_true")
    plt.loglog(g["h"], fitted_curve(g, best_row), "--", label=f"best {best_row['fit_model']}")
    if math.isfinite(float(best_row.get("h_star", np.nan))):
        plt.axvline(float(best_row["h_star"]), color="k", ls=":", label="h_star")
    plt.xlabel("h")
    plt.ylabel("normalized MSE")
    plt.title("Clean combined quantized nonlinear window")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "fig_clean_u_shape_m2.pdf")
    plt.savefig(out_dir / "fig_clean_u_shape_m2.png")
    plt.close()

    plt.figure(figsize=(7, 5))
    plt.loglog(g["h"], g["A_true"], "o-", label="A_true")
    for model in ["M2", "M4", "Mp", "MIAp", "MIA_loc"]:
        rows = fit[(fit["config_id"].eq(clean_id)) & (fit["fit_model"].eq(model)) & (fit["fit_space"].eq("log"))].sort_values("RMSE_log")
        if rows.empty:
            continue
        r = rows.iloc[0]
        plt.loglog(g["h"], fitted_curve(g, r), "--", label=f"{model}, p={r.get('p', np.nan)}")
    plt.xlabel("h")
    plt.ylabel("normalized MSE")
    plt.title("Model comparison on same combined config")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_dir / "fig_clean_u_shape_model_comparison.pdf")
    plt.savefig(out_dir / "fig_clean_u_shape_model_comparison.png")
    plt.close()

    for family, fname, title in [
        ("linear", "fig_visibility_only_linear", "Visibility-only linear oracle"),
        ("locality", "fig_locality_only_fullprecision", "Full-precision nonlinear locality oracle"),
        ("combined", "fig_combined_quantized_nonlinear", "Combined quantized nonlinear oracle"),
    ]:
        cid = best[best["family"].eq(family)].sort_values("RMSE_log").iloc[0]["config_id"]
        gg = raw[raw["config_id"].eq(cid)].sort_values("h")
        plt.figure(figsize=(7, 5))
        plt.loglog(gg["h"], gg["A_true"], "o-", label="A_true")
        if family != "locality":
            plt.loglog(gg["h"], gg["A_interval_grad"], "s--", label="A_interval_grad")
        if family != "linear":
            plt.loglog(gg["h"], gg["M_loc_true"], "^--", label="M_loc_true")
        plt.xlabel("h")
        plt.ylabel("normalized MSE / component")
        plt.title(title)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"{fname}.pdf")
        plt.savefig(out_dir / f"{fname}.png")
        plt.close()

    plt.figure(figsize=(7, 5))
    plt.loglog(g["h"], g["A_true"], "o-", label="A_true")
    plt.loglog(g["h"], g["A_interval_grad"], "s--", label="A_interval_grad")
    plt.loglog(g["h"], g["A_coarse_delta_grad"], ":", label="coarse Delta_grad^2/(4h^2)")
    plt.xlabel("h")
    plt.ylabel("metric")
    plt.title("Interval-aware vs coarse visibility")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "fig_interval_aware_vs_coarse.pdf")
    plt.savefig(out_dir / "fig_interval_aware_vs_coarse.png")
    plt.close()

    # Repair example: old coarse model vs best interval-aware.
    plt.figure(figsize=(7, 5))
    plt.loglog(g["h"], g["A_true"], "o-", label="A_true")
    for model in ["M2", str(best_row["fit_model"])]:
        rows = fit[(fit["config_id"].eq(clean_id)) & (fit["fit_model"].eq(model)) & (fit["fit_space"].eq("log"))].sort_values("RMSE_log")
        if not rows.empty:
            plt.loglog(g["h"], fitted_curve(g, rows.iloc[0]), "--", label=model)
    plt.xlabel("h")
    plt.ylabel("normalized MSE")
    plt.title("Repair example: model choice / h range")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "fig_h_range_repair_example.pdf")
    plt.savefig(out_dir / "fig_h_range_repair_example.png")
    plt.close()

    scaling = best[best["family"].eq("scaling")].copy()
    if not scaling.empty:
        plt.figure(figsize=(7, 5))
        for delta, gg in scaling.groupby("Delta"):
            by = gg.groupby("d")["h_star"].median().reset_index()
            plt.loglog(by["d"], by["h_star"], "o-", label=f"Delta={delta:g}")
        plt.xlabel("d")
        plt.ylabel("h_star")
        plt.title("High-dimensional h_star scaling")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "fig_highdim_window_scaling.pdf")
        plt.savefig(out_dir / "fig_highdim_window_scaling.png")
        plt.close()

    plt.figure(figsize=(7, 5))
    for d, gg in raw[(raw["family"].eq("scaling")) & (raw["Delta"].eq(1e-3)) & (raw["active_p"].eq(1.0))].groupby("d"):
        rho = gg["A_true"].to_numpy() * gg["dstar2_mean"].to_numpy() * gg["d"].to_numpy() / np.maximum(gg["V_dir_eff"].to_numpy(), 1e-30)
        plt.loglog(gg["h"], rho, "o-", label=f"d={int(d):g}")
    plt.xlabel("h")
    plt.ylabel("rho proxy")
    plt.title("rho(h) by dimension")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "fig_rho_highdim.pdf")
    plt.savefig(out_dir / "fig_rho_highdim.png")
    plt.close()


def run_suite(args: argparse.Namespace) -> None:
    out_dir = REPO_ROOT / "synthetic_fit_repair"
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    write_json(
        out_dir / "metadata.json",
        {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "hostname": socket.gethostname(),
            "git_commit": git_commit(),
            "python": sys.executable,
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "device": str(device),
            "args": vars(args),
        },
    )
    old_diagnosis(out_dir)

    configs: List[Config] = [
        Config("linear_visibility_d1e4_int4", "linear", 10_000, 256, "coordinate", "rtn", 4, 1e-3, False, 1.0, False, 128, 0.0, 1.0, 0.0),
        Config("locality_fp_d1e4_a4", "locality", 10_000, 256, "coordinate", "identity", 32, 1.0, False, 1.0, False, 128, 0.0, 4.0, 0.25),
        Config("locality_fp_norm_d1e4_a12", "locality", 10_000, 256, "normalized", "identity", 32, 1.0, False, 1.0, False, 128, 0.0, 12.0, 0.25),
        Config("combined_clean_d1e4_int4_D1e-3_a8", "combined", 10_000, 256, "coordinate", "rtn", 4, 1e-3, False, 1.0, False, 128, 0.0, 8.0, 0.25),
        Config("combined_clean_d1e4_int8_D1e-4_a8", "combined", 10_000, 256, "coordinate", "rtn", 8, 1e-4, False, 1.0, False, 128, 0.0, 8.0, 0.25),
        Config("combined_clip_appendix_d1e4_int4", "combined", 10_000, 256, "coordinate", "rtn", 4, 1e-3, True, 2.0, False, 128, 0.0, 8.0, 0.25),
    ]
    # High-dimensional scaling after clean config.
    for d in [1_000, 10_000, 100_000, 1_000_000]:
        n_dirs = 256 if d <= 10_000 else 128 if d <= 100_000 else 32
        for delta in [1e-5, 1e-4, 1e-3]:
            for qbits in [8, 4]:
                for p in [1.0, 0.1, 0.01]:
                    configs.append(
                        Config(
                            f"scaling_d{d}_p{p:g}_int{qbits}_D{delta:g}",
                            "scaling",
                            d,
                            n_dirs,
                            "coordinate",
                            "rtn",
                            qbits,
                            delta,
                            False,
                            1.0,
                            False,
                            128,
                            0.0,
                            8.0,
                            0.25,
                            active_p=p,
                        )
                    )
    frames = []
    start = time.time()
    for i, cfg in enumerate(configs, 1):
        print(f"[run] {i}/{len(configs)} {cfg.config_id}", flush=True)
        frames.append(run_config(cfg, H_GRID_BASE, device, args.seed))
        if i % 12 == 0:
            pd.concat(frames, ignore_index=True).to_csv(out_dir / "synthetic_fit_raw.csv", index=False)
            print(f"[checkpoint] {i}/{len(configs)} elapsed_min={(time.time()-start)/60:.1f}", flush=True)
    raw = pd.concat(frames, ignore_index=True)
    raw.to_csv(out_dir / "synthetic_fit_raw.csv", index=False)
    fit, best, comp, repairs = fit_all(raw)
    windows = window_summary(raw)
    fit.to_csv(out_dir / "synthetic_fit_summary.csv", index=False)
    best.to_csv(out_dir / "synthetic_fit_best_model.csv", index=False)
    comp.to_csv(out_dir / "synthetic_model_comparison.csv", index=False)
    repairs.to_csv(out_dir / "synthetic_repair_log.csv", index=False)
    windows.to_csv(out_dir / "synthetic_window_summary.csv", index=False)
    scaling = best[best["family"].eq("scaling")].copy()
    scaling["active_dimension"] = scaling["d"].astype(float) * scaling["active_p"].astype(float)
    scaling.to_csv(out_dir / "synthetic_highdim_scaling.csv", index=False)
    try:
        plot_results(out_dir, raw, fit, best, windows)
    except ModuleNotFoundError as exc:
        (out_dir / "missing_items.md").write_text(f"- plotting skipped because dependency is missing: {exc}\n", encoding="utf-8")
    except Exception as exc:
        (out_dir / "missing_items.md").write_text(f"- plotting failed after metrics were computed: {exc}\n", encoding="utf-8")

    # README / summary.
    lines = [
        "# Synthetic Fit Repair Summary",
        "",
        "## Why the previous fit was not clean",
        "The previous broad sweep mixed linear, nonlinear, clipping, and high-saturation regimes and used coarse linear least-squares fits. Linear oracles do not contain a locality tail, and ideal symmetric central differences on smooth functions often produce an h^4 squared-error tail rather than h^2.",
        "",
        "## What succeeded",
    ]
    show = best[best["family"].isin(["linear", "locality", "combined"])][
        ["config_id", "family", "fit_model", "p", "R2_log", "RMSE_log", "h_star", "h_star_interior", "selection_reason", "status"]
    ]
    lines.append(show.to_csv(index=False).strip())
    lines += [
        "",
        "## Interpretation",
        "- Linear visibility-only rows should be used only to show quantization/interval crossing; beta and h_star are not meaningful there.",
        "- Full-precision nonlinear central differences usually prefer M4 or learned-p tails, which is expected for squared central-difference bias.",
        "- Combined quantized nonlinear rows should be interpreted through interval-aware models when M2 is not the best log-space fit.",
        "- The h^2 term is best described as a practical envelope / convergence-level proxy, not a strict Taylor law for every ideal central-difference synthetic oracle.",
        "",
        "## Main-paper recommendation",
        "Use `fig_clean_u_shape_model_comparison.pdf`, `fig_combined_quantized_nonlinear.pdf`, and `fig_interval_aware_vs_coarse.pdf` in the main paper. Put the linear visibility-only and clipping appendix rows in the appendix.",
        "",
        "## Claims not to make",
        "- Do not claim M2 is universally best.",
        "- Do not claim h_star is meaningful for linear visibility-only oracles.",
        "- Do not use clipping-dominated rows as clean window evidence.",
    ]
    (out_dir / "synthetic_fit_repair_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    zip_path = REPO_ROOT / "synthetic_fit_repair.zip"
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in out_dir.rglob("*"):
            if p.is_file():
                zf.write(p, arcname=str(p.relative_to(out_dir.parent)))
    print(f"[done] {out_dir} {zip_path} elapsed_min={(time.time()-start)/60:.1f}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=16)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run_suite(parse_args())
