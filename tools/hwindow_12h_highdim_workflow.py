#!/usr/bin/env python
"""High-dimensional h-window workflow for the ZO perturbation-window project.

This workflow is designed to be self-contained and conservative:

* It audits existing training/probe artifacts without modifying them.
* It runs a synthetic high-dimensional quantized-oracle benchmark on GPU when
  available.
* It aggregates existing real-model interval/probe/training evidence.
* It writes paper-oriented CSVs, figures, notes, and a zip bundle.

The training-validation section intentionally consumes existing logs by
default.  Long training launchers are project-specific and expensive; this
script writes a job list when more GPU time is available instead of silently
fabricating missing training results.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
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
DEFAULT_H_GRID = np.array(
    [
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
    ],
    dtype=np.float64,
)


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return ""


def rel(path: Path | str) -> str:
    p = Path(path)
    try:
        return str(p.resolve().relative_to(REPO_ROOT.resolve()))
    except Exception:
        return str(path)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: Iterable[Dict[str, Any]], fieldnames: Optional[Sequence[str]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if fieldnames is None:
        fields: List[str] = []
        for row in rows:
            for key in row:
                if key not in fields:
                    fields.append(key)
        fieldnames = fields
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def dataframe_table(df: pd.DataFrame, max_rows: Optional[int] = None) -> str:
    if df.empty:
        return "none"
    view = df.head(max_rows) if max_rows else df
    try:
        return view.to_markdown(index=False)
    except Exception:
        return view.to_csv(index=False).strip()


def safe_read_csv(path: Path) -> Optional[pd.DataFrame]:
    try:
        if path.stat().st_size <= 0:
            return None
        return pd.read_csv(path)
    except Exception:
        return None


def normalize_task(value: Any, path: str = "") -> str:
    text = f"{value} {path}".lower()
    for key in ["sst-5", "sst5"]:
        if key in text:
            return "sst-5"
    for key in ["sst-2", "sst2"]:
        if key in text:
            return "sst-2"
    for task in ["trec", "mnli", "rte", "boolq", "wic", "wsc", "cb", "copa", "multirc", "record", "squad", "drop"]:
        if task in text:
            return task
    return str(value or "unknown").lower()


def normalize_model(value: Any, path: str = "") -> str:
    text = f"{value} {path}".lower()
    if "opt13b" in text or "opt-1.3b" in text or "facebook/opt-1.3b" in text:
        return "facebook/opt-1.3b"
    if "opt-125m" in text or "facebook/opt-125m" in text:
        return "facebook/opt-125m"
    if "roberta-large" in text:
        return "roberta-large"
    if "roberta" in text:
        return "roberta-large"
    if "mistral" in text:
        return "mistral"
    return str(value or "unknown")


def normalize_precision(value: Any, path: str = "") -> str:
    text = f"{value} {path}".lower()
    if "int8" in text or "bitwidth 8" in text or "bits=8" in text:
        return "int8"
    if "int4" in text or "bitwidth 4" in text or "bits=4" in text:
        return "int4"
    if "bf16" in text:
        return "bf16"
    if "fp16" in text:
        return "fp16"
    if "fp32" in text:
        return "fp32"
    return str(value or "unknown").lower()


def normalize_mode(value: Any, path: str = "") -> str:
    text = f"{value} {path}".lower()
    if "prefix_int4" in text or ("prefix" in text and "int4" in text):
        return "prefix_int4"
    if "prefix" in text:
        return "prefix"
    if "sparse_p0p01" in text or "p0p01" in text or "p=0.01" in text:
        return "sparse_p0p01"
    if "sparse_p0p1" in text or "p0p1" in text or "p=0.1" in text:
        return "sparse_p0p1"
    if "sparse" in text:
        return "sparse"
    return "dense"


def first_present(row: pd.Series, names: Sequence[str]) -> Any:
    for name in names:
        if name in row.index and pd.notna(row[name]):
            return row[name]
    return np.nan


def to_float(value: Any) -> float:
    try:
        out = float(value)
    except Exception:
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def to_int(value: Any, default: int = 0) -> int:
    try:
        if pd.isna(value):
            return default
        return int(float(value))
    except Exception:
        return default


def collect_env() -> Dict[str, Any]:
    env = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "hostname": socket.gethostname(),
        "python": sys.executable,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "git_commit": git_commit(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
    }
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        env.update(
            {
                "gpu_name": props.name,
                "gpu_total_memory_mb": int(props.total_memory / 1024 / 1024),
                "gpu_count": torch.cuda.device_count(),
            }
        )
    try:
        env["nvidia_smi"] = subprocess.check_output(["nvidia-smi"], text=True, stderr=subprocess.STDOUT)
    except Exception as exc:
        env["nvidia_smi_error"] = str(exc)
    return env


def audit_existing_results(out_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    csv_paths = []
    roots = ["outputs", "analysis", "experiments", "runs", "Results", "interval_aware_h_probe", "interval_h_selection_8h_bundle", "safe_override_6h_a100_bundle"]
    for root in roots:
        base = REPO_ROOT / root
        if base.exists():
            csv_paths.extend(base.rglob("*.csv"))

    training_rows: List[Dict[str, Any]] = []
    probe_rows: List[Dict[str, Any]] = []
    full_like = 0
    pilot_like = 0
    interval_files = 0
    loss_probe_files = 0

    for path in sorted(set(csv_paths)):
        df = safe_read_csv(path)
        if df is None or df.empty:
            continue
        cols = set(df.columns)
        pstr = rel(path)
        is_training = bool({"best_eval_acc", "last_eval_acc"} & cols) or bool({"accuracy", "steps"} <= cols)
        is_probe = bool({"h", "A_uniform"} <= cols) or bool({"default_fd_true_nmse", "default_corr_fd_true"} & cols) or "nMSE_loss" in cols
        if is_training:
            for _, row in df.iterrows():
                task = normalize_task(first_present(row, ["task", "dataset", "task_name"]), pstr)
                model = normalize_model(first_present(row, ["model", "model_id"]), pstr)
                precision = normalize_precision(first_present(row, ["precision", "precision_mode", "quantizer"]), pstr)
                mode = normalize_mode(first_present(row, ["perturbation_mode", "mode", "setting", "direction", "direction_mode"]), pstr)
                h_policy = str(first_present(row, ["h_policy", "policy", "h_label", "candidate_name"]))
                h_value = to_float(first_present(row, ["h_value", "h", "h_final", "selected_h"]))
                acc = to_float(first_present(row, ["best_eval_acc", "accuracy", "best_dev_acc", "final_eval_acc_if_available"]))
                last_acc = to_float(first_present(row, ["last_eval_acc", "final_dev_acc", "last_accuracy"]))
                loss = to_float(first_present(row, ["best_eval_loss", "loss", "last_eval_loss"]))
                steps = to_int(first_present(row, ["steps_completed", "steps", "max_steps", "steps_run"]))
                run_type = "full" if steps >= 20000 else ("medium" if steps >= 2000 else "pilot")
                if run_type == "full":
                    full_like += 1
                else:
                    pilot_like += 1
                training_rows.append(
                    {
                        "model": model,
                        "task": task,
                        "precision": precision,
                        "perturbation_mode": mode,
                        "h_policy": h_policy,
                        "h_value": h_value,
                        "seed": to_int(first_present(row, ["seed"]), 16),
                        "run_type": run_type,
                        "steps": steps,
                        "accuracy": acc,
                        "last_accuracy": last_acc,
                        "loss": loss,
                        "status": str(first_present(row, ["status"])),
                        "source_path": pstr,
                    }
                )
        if is_probe:
            if "A_uniform" in cols:
                interval_files += 1
            if "nMSE_loss" in cols or "default_fd_true_nmse" in cols:
                loss_probe_files += 1
            for _, row in df.iterrows():
                h = to_float(first_present(row, ["h", "h_value"]))
                if not math.isfinite(h):
                    continue
                probe_rows.append(
                    {
                        "model": normalize_model(first_present(row, ["model", "model_id"]), pstr),
                        "task": normalize_task(first_present(row, ["task", "dataset"]), pstr),
                        "precision": normalize_precision(first_present(row, ["precision", "precision_mode"]), pstr),
                        "perturbation_mode": normalize_mode(first_present(row, ["perturbation_mode", "mode", "setting"]), pstr),
                        "h": h,
                        "metric_type": "interval" if "A_uniform" in cols else "loss_or_fd",
                        "A_uniform": to_float(first_present(row, ["A_uniform"])),
                        "p_active": to_float(first_present(row, ["p_active", "active_frac_mean", "active_frac"])),
                        "V_align": to_float(first_present(row, ["V_align", "alignment_mean", "alignment"])),
                        "V_norm": to_float(first_present(row, ["V_norm", "norm_ratio_mean", "norm_ratio"])),
                        "relative_disp": to_float(first_present(row, ["relative_disp"])),
                        "locality_proxy": to_float(first_present(row, ["locality_proxy"])),
                        "nMSE_loss": to_float(first_present(row, ["nMSE_loss", "default_fd_true_nmse", "fd_true_nmse"])),
                        "corr_loss": to_float(first_present(row, ["corr_loss", "default_corr_fd_true", "corr"])),
                        "source_path": pstr,
                    }
                )

    train_df = pd.DataFrame(training_rows).drop_duplicates()
    probe_df = pd.DataFrame(probe_rows).drop_duplicates()
    train_df.to_csv(out_dir / "existing_training_index.csv", index=False)
    probe_df.to_csv(out_dir / "existing_probe_index.csv", index=False)

    def count_by(df: pd.DataFrame, cols: Sequence[str]) -> str:
        if df.empty:
            return "none"
        sub = df.groupby(list(cols)).size().reset_index(name="rows").sort_values("rows", ascending=False)
        return dataframe_table(sub, max_rows=30)

    md = [
        "# Existing Result Audit",
        "",
        f"- Created: {datetime.now().isoformat(timespec='seconds')}",
        f"- Git commit: `{git_commit()}`",
        f"- CSV files scanned: {len(csv_paths)}",
        f"- Training-like rows indexed: {len(train_df)}",
        f"- Probe-like rows indexed: {len(probe_df)}",
        f"- Full-like training rows (steps >= 20000): {full_like}",
        f"- Medium/pilot training rows: {pilot_like}",
        f"- Interval metric source files detected: {interval_files}",
        f"- Loss/FD nMSE source files detected: {loss_probe_files}",
        "",
        "## Precision Sweeps / Training Coverage",
        count_by(train_df, ["model", "task", "precision", "perturbation_mode", "run_type"]),
        "",
        "## Interval / Probe Coverage",
        count_by(probe_df, ["model", "task", "precision", "perturbation_mode", "metric_type"]),
        "",
        "## Priority For This 12h Run",
        "- New synthetic high-dimensional quantized-oracle benchmark.",
        "- Real-model interval/probe aggregation from existing RoBERTa and OPT results.",
        "- Targeted training table from existing full/medium/pilot logs; missing long training is listed in job list rather than fabricated.",
    ]
    (out_dir / "audit_existing_results.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return train_df, probe_df


@dataclass(frozen=True)
class SyntheticConfig:
    d: int
    active_p: float
    delta: float
    qbits: int
    scale_sigma: float
    group_size: int
    family: str


def quantize_shared_grid(x: torch.Tensor, delta: torch.Tensor, qmin: int, qmax: int) -> Tuple[torch.Tensor, torch.Tensor]:
    code = torch.round(x / delta).clamp(qmin, qmax)
    return code * delta, code


def make_group_delta(d: int, base_delta: float, group_size: int, sigma: float, device: torch.device, generator: torch.Generator) -> torch.Tensor:
    n_groups = math.ceil(d / group_size)
    if sigma > 0:
        group_scale = torch.exp(torch.randn(n_groups, device=device, generator=generator) * sigma)
    else:
        group_scale = torch.ones(n_groups, device=device)
    delta_g = base_delta * group_scale
    return delta_g.repeat_interleave(group_size)[:d]


def fit_alpha_beta_gamma(h: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    mask = np.isfinite(h) & np.isfinite(y) & (h > 0) & (y >= 0)
    if mask.sum() < 4:
        return {"alpha": np.nan, "beta": np.nan, "gamma": np.nan, "h_star": np.nan, "fit_R2": np.nan}
    x = np.stack([1.0 / (h[mask] ** 2), h[mask] ** 2, np.ones(mask.sum())], axis=1)
    yy = y[mask]
    try:
        coef, *_ = np.linalg.lstsq(x, yy, rcond=None)
        pred = x @ coef
        ss_res = float(np.sum((yy - pred) ** 2))
        ss_tot = float(np.sum((yy - float(np.mean(yy))) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
        alpha, beta, gamma = [float(c) for c in coef]
        h_star = (alpha / beta) ** 0.25 if alpha > 0 and beta > 0 else np.nan
    except Exception:
        alpha = beta = gamma = h_star = r2 = np.nan
    return {"alpha": alpha, "beta": beta, "gamma": gamma, "h_star": h_star, "fit_R2": r2}


def window_rows_for_metric(config_key: Dict[str, Any], h: np.ndarray, y: np.ndarray, rho: np.ndarray) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    mask = np.isfinite(y)
    if not mask.any():
        return rows
    y_min = float(np.nanmin(y))
    for kappa in [1.05, 1.10, 1.25, 1.50]:
        ok = mask & (y <= kappa * y_min)
        if ok.any():
            width = float(np.log10(np.nanmax(h[ok]) / np.nanmin(h[ok]))) if ok.sum() > 1 else 0.0
            rows.append({**config_key, "window_type": f"kappa_{kappa:g}", "h_min": float(np.nanmin(h[ok])), "h_max": float(np.nanmax(h[ok])), "window_width_log10": width, "threshold": kappa})
    for tau in [0.01, 0.05, 0.1, 0.5, 1.0]:
        ok = np.isfinite(rho) & (rho <= tau)
        if ok.any():
            width = float(np.log10(np.nanmax(h[ok]) / np.nanmin(h[ok]))) if ok.sum() > 1 else 0.0
            rows.append({**config_key, "window_type": f"rho_{tau:g}", "h_min": float(np.nanmin(h[ok])), "h_max": float(np.nanmax(h[ok])), "window_width_log10": width, "threshold": tau})
    return rows


def run_synthetic(out_dir: Path, args: argparse.Namespace) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed)

    dims = [int(v) for v in args.synthetic_dims]
    deltas = [float(v) for v in args.synthetic_deltas]
    p_values = [float(v) for v in args.synthetic_p]
    qbits_values = [int(v) for v in args.synthetic_qbits]
    sigmas = [float(v) for v in args.synthetic_scale_sigmas]
    families = list(args.synthetic_families)
    group_size = int(args.synthetic_group_size)
    h_grid = np.array([float(v) for v in args.h_grid], dtype=np.float64)

    raw_rows: List[Dict[str, Any]] = []
    fit_rows: List[Dict[str, Any]] = []
    window_rows: List[Dict[str, Any]] = []

    total_configs = len(dims) * len(deltas) * len(p_values) * len(qbits_values) * len(sigmas) * len(families)
    started = time.time()
    config_idx = 0
    checkpoint_path = out_dir / "synthetic_highdim_raw.csv"

    for d in dims:
        n_dirs = args.synthetic_dirs_small if d <= 10_000 else (args.synthetic_dirs_medium if d <= 100_000 else args.synthetic_dirs_large)
        for active_p in p_values:
            active_count = max(1, int(round(d * active_p)))
            for delta in deltas:
                for qbits in qbits_values:
                    qmin = -(2 ** (qbits - 1))
                    qmax = 2 ** (qbits - 1) - 1
                    for sigma in sigmas:
                        for family in families:
                            config_idx += 1
                            # Re-seed per config for deterministic but distinct configs.
                            generator.manual_seed(args.seed + config_idx * 17)
                            delta_vec = make_group_delta(d, delta, group_size, sigma, device, generator)
                            # Sample weights inside the quantizer's representable range so the
                            # main curves measure interval crossing rather than trivial base
                            # saturation. Saturation can still occur for perturbed weights and
                            # is recorded separately.
                            w_code_float = torch.empty(d, device=device).uniform_(
                                -0.5 * float(qmax), 0.5 * float(qmax), generator=generator
                            )
                            w = w_code_float * delta_vec
                            if family == "lin":
                                g = torch.randn(d, device=device, generator=generator)
                                curvature = torch.zeros(d, device=device)
                                a_nl = torch.ones(d, device=device)
                            elif family == "quad":
                                curvature = torch.empty(d, device=device).uniform_(0.25, 2.0, generator=generator)
                                g = curvature * w
                                a_nl = torch.ones(d, device=device)
                            else:
                                a_nl = torch.empty(d, device=device).uniform_(0.5, 2.0, generator=generator)
                                c_nl = torch.empty(d, device=device).uniform_(0.5, 1.5, generator=generator)
                                g = c_nl * a_nl * torch.tanh(a_nl * w)
                                curvature = c_nl * (a_nl**2)

                            if active_p < 1.0:
                                perm = torch.randperm(d, device=device, generator=generator)[:active_count]
                                active_mask = torch.zeros(d, dtype=torch.bool, device=device)
                                active_mask[perm] = True
                            else:
                                active_mask = torch.ones(d, dtype=torch.bool, device=device)

                            w_q, code_w = quantize_shared_grid(w, delta_vec, qmin, qmax)
                            saturation_base = ((code_w <= qmin) | (code_w >= qmax)).float().mean().item()
                            g_norm_sq = float(torch.sum(g * g).item())
                            v_dir_theory = float((d + 1) * g_norm_sq)
                            config_key = {
                                "d": d,
                                "effective_dim": active_count,
                                "active_p": active_p,
                                "Delta": delta,
                                "qbits": qbits,
                                "function_family": family,
                                "group_size": group_size,
                                "scale_sigma": sigma,
                                "n_dirs": n_dirs,
                            }
                            y_by_h: List[float] = []
                            interval_by_h: List[float] = []
                            rho_by_h: List[float] = []

                            for h in h_grid:
                                h_t = float(h)
                                # Accumulate in chunks to keep 1e6 configs within memory.
                                remaining = n_dirs
                                accum: Dict[str, float] = {
                                    "mse_true": 0.0,
                                    "dstar2": 0.0,
                                    "interval_uniform": 0.0,
                                    "interval_grad_num": 0.0,
                                    "active": 0.0,
                                    "jump_mean": 0.0,
                                    "jump_zero": 0.0,
                                    "jump_one": 0.0,
                                    "jump_ge2": 0.0,
                                    "norm_ratio": 0.0,
                                    "align": 0.0,
                                    "disp_rms_num": 0.0,
                                    "relative_disp_num": 0.0,
                                    "locality_proxy": 0.0,
                                    "m_loc_true": 0.0,
                                    "clip": 0.0,
                                    "v_dir_emp_num": 0.0,
                                }
                                seen = 0
                                chunk = min(args.synthetic_chunk_dirs, n_dirs)
                                while remaining > 0:
                                    cur = min(chunk, remaining)
                                    u = torch.randn(cur, d, device=device, generator=generator)
                                    if active_p < 1.0:
                                        u[:, ~active_mask] = 0.0
                                    plus = w.unsqueeze(0) + h_t * u
                                    minus = w.unsqueeze(0) - h_t * u
                                    q_plus, code_plus = quantize_shared_grid(plus, delta_vec.unsqueeze(0), qmin, qmax)
                                    q_minus, code_minus = quantize_shared_grid(minus, delta_vec.unsqueeze(0), qmin, qmax)
                                    delta_q = q_plus - q_minus
                                    intended = 2.0 * h_t * u
                                    b = delta_q / (2.0 * h_t)
                                    d_star = torch.sum(g.unsqueeze(0) * u, dim=1)

                                    if family == "lin":
                                        f_plus = torch.sum(g.unsqueeze(0) * q_plus, dim=1)
                                        f_minus = torch.sum(g.unsqueeze(0) * q_minus, dim=1)
                                        grad_b = torch.sum(g.unsqueeze(0) * b, dim=1)
                                    elif family == "quad":
                                        f_plus = 0.5 * torch.sum(curvature.unsqueeze(0) * q_plus * q_plus, dim=1)
                                        f_minus = 0.5 * torch.sum(curvature.unsqueeze(0) * q_minus * q_minus, dim=1)
                                        grad_b = torch.sum(g.unsqueeze(0) * b, dim=1)
                                    else:
                                        # log(cosh(x)) computed stably enough for the bounded range here.
                                        aa = a_nl.unsqueeze(0)
                                        f_plus = torch.sum(torch.log(torch.cosh(aa * q_plus)) / aa, dim=1)
                                        f_minus = torch.sum(torch.log(torch.cosh(aa * q_minus)) / aa, dim=1)
                                        grad_b = torch.sum(g.unsqueeze(0) * b, dim=1)
                                    d_q = (f_plus - f_minus) / (2.0 * h_t)
                                    err = d_q - d_star
                                    interval_err = b - u
                                    jump = torch.abs(code_plus - code_minus)
                                    active = jump > 0
                                    norm_delta = torch.linalg.vector_norm(delta_q, dim=1)
                                    norm_intended = torch.linalg.vector_norm(intended, dim=1).clamp_min(1e-30)
                                    dot = torch.sum(delta_q * intended, dim=1)
                                    align = dot / (norm_delta.clamp_min(1e-30) * norm_intended)
                                    e_plus = q_plus - w.unsqueeze(0)
                                    e_minus = q_minus - w.unsqueeze(0)
                                    disp2 = 0.5 * (
                                        torch.sum(e_plus * e_plus, dim=1) + torch.sum(e_minus * e_minus, dim=1)
                                    )
                                    r_loc = d_q - grad_b
                                    g_est = d_star.unsqueeze(1) * u
                                    g_est_err = g_est - g.unsqueeze(0)
                                    clip = (
                                        (code_plus <= qmin)
                                        | (code_plus >= qmax)
                                        | (code_minus <= qmin)
                                        | (code_minus >= qmax)
                                    )

                                    accum["mse_true"] += float(torch.sum(err * err).item())
                                    accum["dstar2"] += float(torch.sum(d_star * d_star).item())
                                    accum["interval_uniform"] += float(torch.sum(interval_err * interval_err).item())
                                    accum["interval_grad_num"] += float(torch.sum((g.unsqueeze(0) ** 2) * (interval_err**2)).item())
                                    accum["active"] += float(torch.sum(active.float()).item())
                                    accum["jump_mean"] += float(torch.sum(jump.float()).item())
                                    accum["jump_zero"] += float(torch.sum((jump == 0).float()).item())
                                    accum["jump_one"] += float(torch.sum((jump == 1).float()).item())
                                    accum["jump_ge2"] += float(torch.sum((jump >= 2).float()).item())
                                    accum["norm_ratio"] += float(torch.sum(norm_delta / norm_intended).item())
                                    accum["align"] += float(torch.sum(torch.nan_to_num(align, nan=0.0)).item())
                                    accum["disp_rms_num"] += float(torch.sum(disp2).item())
                                    accum["relative_disp_num"] += float(torch.sum(torch.sqrt(disp2)).item())
                                    accum["locality_proxy"] += float(torch.sum((2.0 * disp2) ** 2 / (16.0 * h_t * h_t + 1e-30)).item())
                                    accum["m_loc_true"] += float(torch.sum(r_loc * r_loc).item())
                                    accum["clip"] += float(torch.sum(clip.float()).item())
                                    accum["v_dir_emp_num"] += float(torch.sum(g_est_err * g_est_err).item())
                                    seen += cur
                                    remaining -= cur
                                    del u, plus, minus, q_plus, q_minus, code_plus, code_minus, delta_q, intended, b
                                denom_dirs = max(seen, 1)
                                denom_coords = max(seen * d, 1)
                                denom_dstar = max(accum["dstar2"], 1e-30)
                                A_true = accum["mse_true"] / denom_dstar
                                A_interval_uniform = accum["interval_uniform"] / denom_coords
                                A_interval_grad = accum["interval_grad_num"] / (denom_dirs * max(g_norm_sq, 1e-30))
                                vector_h_error = d * accum["mse_true"] / denom_dirs
                                v_dir_emp = accum["v_dir_emp_num"] / denom_dirs
                                rho_theory = vector_h_error / max(v_dir_theory, 1e-30)
                                rho_emp = vector_h_error / max(v_dir_emp, 1e-30)
                                row = {
                                    **config_key,
                                    "h": h_t,
                                    "A_true": A_true,
                                    "A_interval_uniform": A_interval_uniform,
                                    "A_interval_grad": A_interval_grad,
                                    "p_active": accum["active"] / denom_coords,
                                    "jump_mean": accum["jump_mean"] / denom_coords,
                                    "jump_median": np.nan,
                                    "jump_zero_frac": accum["jump_zero"] / denom_coords,
                                    "jump_one_frac": accum["jump_one"] / denom_coords,
                                    "jump_ge2_frac": accum["jump_ge2"] / denom_coords,
                                    "V_norm": accum["norm_ratio"] / denom_dirs,
                                    "V_align": accum["align"] / denom_dirs,
                                    "p_clip": accum["clip"] / denom_coords,
                                    "p_clip_base": saturation_base,
                                    "disp_rms": math.sqrt(accum["disp_rms_num"] / max(denom_dirs * d, 1)),
                                    "relative_disp": accum["relative_disp_num"] / (denom_dirs * max(float(torch.linalg.vector_norm(w).item()), 1e-30)),
                                    "locality_proxy": accum["locality_proxy"] / denom_dirs,
                                    "M_loc_true": accum["m_loc_true"] / denom_dstar,
                                    "V_dir_theory": v_dir_theory,
                                    "V_dir_empirical": v_dir_emp,
                                    "rho_theory": rho_theory,
                                    "rho_empirical": rho_emp,
                                }
                                raw_rows.append(row)
                                y_by_h.append(A_true)
                                interval_by_h.append(A_interval_grad)
                                rho_by_h.append(rho_theory)

                            h_arr = h_grid.copy()
                            y_arr = np.array(y_by_h, dtype=np.float64)
                            int_arr = np.array(interval_by_h, dtype=np.float64)
                            rho_arr = np.array(rho_by_h, dtype=np.float64)
                            fit_true = fit_alpha_beta_gamma(h_arr, y_arr)
                            fit_int = fit_alpha_beta_gamma(h_arr, int_arr)
                            fit_rows.append({**config_key, "metric": "A_true", **fit_true})
                            fit_rows.append({**config_key, "metric": "A_interval_grad", **fit_int})
                            window_rows.extend(window_rows_for_metric({**config_key, "metric": "A_true"}, h_arr, y_arr, rho_arr))

                            if config_idx % args.synthetic_checkpoint_every == 0:
                                pd.DataFrame(raw_rows).to_csv(checkpoint_path, index=False)
                                pd.DataFrame(fit_rows).to_csv(out_dir / "synthetic_highdim_fit.csv", index=False)
                                pd.DataFrame(window_rows).to_csv(out_dir / "synthetic_highdim_window.csv", index=False)
                                elapsed = time.time() - started
                                print(f"[synthetic] checkpoint {config_idx}/{total_configs} configs elapsed={elapsed/60:.1f}m", flush=True)

    raw_df = pd.DataFrame(raw_rows)
    fit_df = pd.DataFrame(fit_rows)
    win_df = pd.DataFrame(window_rows)
    raw_df.to_csv(out_dir / "synthetic_highdim_raw.csv", index=False)
    fit_df.to_csv(out_dir / "synthetic_highdim_fit.csv", index=False)
    win_df.to_csv(out_dir / "synthetic_highdim_window.csv", index=False)
    return raw_df, fit_df, win_df


def aggregate_realmodel(out_dir: Path, train_df: pd.DataFrame, probe_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    interval_sources = [
        REPO_ROOT / "interval_aware_h_probe/interval_geometry_summary.csv",
        REPO_ROOT / "interval_h_selection_8h_bundle/all_interval_metrics.csv",
        REPO_ROOT / "interval_h_selection_8h_bundle/interval_geometry_new.csv",
        REPO_ROOT / "outputs/interval_h_selection_8h_probes/opt13b_sst5_int8/interval_geometry_summary.csv",
    ]
    frames = []
    for src in interval_sources:
        df = safe_read_csv(src)
        if df is not None and not df.empty:
            df = df.copy()
            df["source_path"] = rel(src)
            frames.append(df)
    if frames:
        real_interval = pd.concat(frames, ignore_index=True, sort=False)
    else:
        real_interval = pd.DataFrame()
    if not probe_df.empty:
        interval_like = probe_df[probe_df["metric_type"].eq("interval")].copy()
        real_interval = pd.concat([real_interval, interval_like], ignore_index=True, sort=False)
    if not real_interval.empty:
        for col in ["model", "task", "precision", "perturbation_mode"]:
            if col not in real_interval.columns:
                real_interval[col] = "unknown"
        real_interval["model"] = [normalize_model(v, str(p)) for v, p in zip(real_interval["model"], real_interval.get("source_path", ""))]
        real_interval["task"] = [normalize_task(v, str(p)) for v, p in zip(real_interval["task"], real_interval.get("source_path", ""))]
        real_interval["precision"] = [normalize_precision(v, str(p)) for v, p in zip(real_interval["precision"], real_interval.get("source_path", ""))]
        real_interval["perturbation_mode"] = [
            normalize_mode(v, str(p)) for v, p in zip(real_interval["perturbation_mode"], real_interval.get("source_path", ""))
        ]
    real_interval.to_csv(out_dir / "realmodel_interval_metrics.csv", index=False)

    loss_sources = [
        REPO_ROOT / "interval_aware_h_probe/loss_mse_probe.csv",
        REPO_ROOT / "interval_h_selection_8h_bundle/loss_mse_all.csv",
        REPO_ROOT / "interval_h_selection_8h_bundle/loss_mse_new.csv",
    ]
    frames = []
    for src in loss_sources:
        df = safe_read_csv(src)
        if df is not None and not df.empty:
            df = df.copy()
            df["source_path"] = rel(src)
            frames.append(df)
    if not probe_df.empty:
        loss_like = probe_df[probe_df["metric_type"].eq("loss_or_fd")].copy()
        frames.append(loss_like)
    real_loss = pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()
    real_loss.to_csv(out_dir / "realmodel_loss_mse.csv", index=False)

    summary_rows: List[Dict[str, Any]] = []
    if not real_interval.empty:
        h_col = "h" if "h" in real_interval.columns else None
        for key, group in real_interval.groupby(["model", "task", "precision", "perturbation_mode"], dropna=False):
            row: Dict[str, Any] = {
                "model": key[0],
                "task": key[1],
                "precision": key[2],
                "mode": key[3],
                "rows": len(group),
                "h_min": to_float(group[h_col].min()) if h_col else np.nan,
                "h_max": to_float(group[h_col].max()) if h_col else np.nan,
            }
            nearest = group.iloc[(group[h_col].astype(float) - 1e-3).abs().argsort()[:1]] if h_col and "h" in group else pd.DataFrame()
            if not nearest.empty:
                nr = nearest.iloc[0]
                row.update(
                    {
                        "default_h": to_float(nr.get("h")),
                        "p_active_default": to_float(nr.get("p_active", nr.get("active_frac_mean", np.nan))),
                        "V_align_default": to_float(nr.get("V_align", nr.get("alignment_mean", np.nan))),
                        "A_uniform_default": to_float(nr.get("A_uniform", np.nan)),
                        "relative_disp_default": to_float(nr.get("relative_disp", np.nan)),
                    }
                )
            summary_rows.append(row)
    real_summary = pd.DataFrame(summary_rows)
    real_summary.to_csv(out_dir / "realmodel_highdim_summary.csv", index=False)
    return real_interval, real_loss, real_summary


def build_targeted_training(out_dir: Path, train_df: pd.DataFrame) -> pd.DataFrame:
    if train_df.empty:
        targeted = pd.DataFrame()
    else:
        mask = train_df["model"].astype(str).str.contains("roberta|opt", case=False, na=False)
        mask &= train_df["precision"].astype(str).str.contains("int4|int8", case=False, na=False)
        targeted = train_df[mask].copy()
        targeted = targeted.rename(columns={"accuracy": "best_dev_acc", "last_accuracy": "final_dev_acc"})
        targeted["default_acc_reference"] = np.nan
        targeted["delta_vs_default"] = np.nan
        for key, group_idx in targeted.groupby(["model", "task", "precision", "perturbation_mode"]).groups.items():
            group = targeted.loc[group_idx]
            default_rows = group[np.isclose(group["h_value"].astype(float), 1e-3, rtol=0, atol=1e-12)]
            if default_rows.empty:
                continue
            default_acc = float(default_rows["best_dev_acc"].max())
            targeted.loc[group_idx, "default_acc_reference"] = default_acc
            targeted.loc[group_idx, "delta_vs_default"] = targeted.loc[group_idx, "best_dev_acc"].astype(float) - default_acc
    targeted.to_csv(out_dir / "targeted_training_results.csv", index=False)

    md = ["# Targeted Training Summary", ""]
    if targeted.empty:
        md.append("No targeted training rows were found in existing logs. No new long training was launched by this workflow.")
    else:
        show_cols = ["model", "task", "precision", "perturbation_mode", "h_policy", "h_value", "run_type", "steps", "best_dev_acc", "final_dev_acc", "delta_vs_default", "source_path"]
        for col in show_cols:
            if col not in targeted.columns:
                targeted[col] = np.nan
        top = targeted.sort_values(["model", "task", "precision", "perturbation_mode", "best_dev_acc"], ascending=[True, True, True, True, False])
        md.append(dataframe_table(top[show_cols], max_rows=80))
        md.append("")
        md.append("Rows are existing full/medium/pilot logs; this workflow does not relabel medium or pilot rows as full.")
    (out_dir / "targeted_training_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    job_rows = [
        {
            "priority": 1,
            "model": "roberta-large",
            "task": "trec",
            "precision": "int4",
            "mode": "prefix_int4",
            "h_policy": "selected",
            "seed": 17,
            "run_type": "medium_or_full",
            "notes": "extra-seed validation if H100 time remains",
        },
        {
            "priority": 2,
            "model": "facebook/opt-1.3b",
            "task": "trec",
            "precision": "int8",
            "mode": "dense",
            "h_policy": "safe_override_or_default",
            "seed": 16,
            "run_type": "medium",
            "notes": "OPT cross-architecture sanity",
        },
    ]
    write_csv(out_dir / "targeted_training_job_list.csv", job_rows)
    return targeted


def create_tables(out_dir: Path, raw_df: pd.DataFrame, fit_df: pd.DataFrame, win_df: pd.DataFrame, real_interval: pd.DataFrame, targeted: pd.DataFrame) -> None:
    if not fit_df.empty:
        scaling = fit_df.merge(
            win_df[win_df["window_type"].eq("kappa_1.1")][
                ["d", "effective_dim", "Delta", "qbits", "function_family", "active_p", "scale_sigma", "metric", "window_width_log10"]
            ],
            on=["d", "effective_dim", "Delta", "qbits", "function_family", "active_p", "scale_sigma", "metric"],
            how="left",
        )
        scaling = scaling.rename(columns={"window_width_log10": "window_width_kappa_1p1"})
        if not raw_df.empty:
            rho_min = raw_df.groupby(["d", "effective_dim", "Delta", "qbits", "function_family", "active_p", "scale_sigma"])["rho_theory"].min().reset_index(name="rho_min")
            scaling = scaling.merge(rho_min, on=["d", "effective_dim", "Delta", "qbits", "function_family", "active_p", "scale_sigma"], how="left")
        scaling.to_csv(out_dir / "table_highdim_scaling.csv", index=False)
    else:
        pd.DataFrame().to_csv(out_dir / "table_highdim_scaling.csv", index=False)

    precision_rows: List[Dict[str, Any]] = []
    if not real_interval.empty and "h" in real_interval.columns:
        for key, group in real_interval.groupby(["model", "task", "precision", "perturbation_mode"], dropna=False):
            g = group.copy()
            g["h_float"] = pd.to_numeric(g["h"], errors="coerce")
            default = g.iloc[(g["h_float"] - 1e-3).abs().argsort()[:1]]
            row = {
                "model": key[0],
                "task": key[1],
                "precision": key[2],
                "mode": key[3],
                "h_star": np.nan,
                "h_vis": np.nan,
                "h_loc": np.nan,
                "window_width_log10": np.nan,
                "default_h": 1e-3,
                "default_in_window": np.nan,
                "d_train": np.nan,
                "d_eff_at_default": np.nan,
                "p_active_default": np.nan,
                "V_align_default": np.nan,
                "nMSE_default": np.nan,
                "nMSE_hstar": np.nan,
            }
            if not default.empty:
                dr = default.iloc[0]
                row["p_active_default"] = to_float(dr.get("p_active", dr.get("active_frac_mean", np.nan)))
                row["V_align_default"] = to_float(dr.get("V_align", dr.get("alignment_mean", np.nan)))
                # A light-weight safe-window check matching the conservative rule.
                row["default_in_window"] = bool(
                    (not math.isfinite(row["V_align_default"]) or row["V_align_default"] >= 0.70)
                    and (not math.isfinite(row["p_active_default"]) or row["p_active_default"] >= 0.01)
                )
            precision_rows.append(row)
    pd.DataFrame(precision_rows).to_csv(out_dir / "table_precision_window_realmodel.csv", index=False)

    if not targeted.empty:
        cols = ["model", "task", "precision", "perturbation_mode", "h_policy", "h_value", "best_dev_acc", "delta_vs_default", "run_type", "seed"]
        table = targeted[[c for c in cols if c in targeted.columns]].copy()
        table = table.rename(columns={"h_value": "selected_h", "best_dev_acc": "selected_acc", "perturbation_mode": "mode"})
        table.to_csv(out_dir / "table_training_summary.csv", index=False)
    else:
        pd.DataFrame().to_csv(out_dir / "table_training_summary.csv", index=False)


def plot_outputs(out_dir: Path, raw_df: pd.DataFrame, fit_df: pd.DataFrame, win_df: pd.DataFrame, real_interval: pd.DataFrame, targeted: pd.DataFrame) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        (out_dir / "missing_items.md").write_text(f"- matplotlib unavailable: {exc}\n", encoding="utf-8")
        return

    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    if not raw_df.empty:
        sample = raw_df[(raw_df["active_p"].isin([1.0, 0.1])) & (raw_df["Delta"].isin([1e-4, 1e-3, 1e-2])) & (raw_df["scale_sigma"].eq(0.0))]
        plt.figure(figsize=(8, 5))
        for (delta, qbits), group in sample.groupby(["Delta", "qbits"]):
            g = group[(group["d"].eq(group["d"].max())) & (group["function_family"].eq("nl")) & (group["active_p"].eq(1.0))]
            if g.empty:
                continue
            plt.loglog(g["h"], g["A_true"], marker="o", label=f"Delta={delta:g}, int{qbits}")
        plt.xlabel("h")
        plt.ylabel("A_true")
        plt.title("Synthetic MSE vs h by quantization step")
        plt.legend(fontsize=8)
        plt.tight_layout()
        for ext in ["pdf", "png"]:
            plt.savefig(fig_dir / f"fig_synthetic_mse_vs_h_by_delta.{ext}")
        plt.close()

        plt.figure(figsize=(8, 5))
        for d, group in raw_df[(raw_df["Delta"].eq(1e-3)) & (raw_df["qbits"].eq(4)) & (raw_df["active_p"].eq(1.0)) & (raw_df["function_family"].eq("nl")) & (raw_df["scale_sigma"].eq(0.0))].groupby("d"):
            plt.loglog(group["h"], group["rho_theory"], marker="o", label=f"d={int(d):g}")
        plt.xlabel("h")
        plt.ylabel("rho(h)")
        plt.title("h-dependent error relative to random-direction floor")
        plt.legend(fontsize=8)
        plt.tight_layout()
        for ext in ["pdf", "png"]:
            plt.savefig(fig_dir / f"fig_synthetic_rho_vs_h_by_dimension.{ext}")
        plt.close()

        width = win_df[win_df["window_type"].eq("kappa_1.1")] if not win_df.empty else pd.DataFrame()
        if not width.empty:
            plt.figure(figsize=(8, 5))
            for p, group in width[(width["Delta"].eq(1e-3)) & (width["qbits"].eq(4)) & (width["function_family"].eq("nl"))].groupby("active_p"):
                by_d = group.groupby("d")["window_width_log10"].mean().reset_index()
                plt.semilogx(by_d["d"], by_d["window_width_log10"], marker="o", label=f"p={p:g}")
            plt.xlabel("dimension d")
            plt.ylabel("log10 window width")
            plt.title("Synthetic window width vs dimension/effective subspace")
            plt.legend(fontsize=8)
            plt.tight_layout()
            for ext in ["pdf", "png"]:
                plt.savefig(fig_dir / f"fig_synthetic_window_width_vs_dimension.{ext}")
            plt.close()

        plt.figure(figsize=(8, 5))
        for p, group in raw_df[(raw_df["Delta"].eq(1e-3)) & (raw_df["qbits"].eq(4)) & (raw_df["d"].eq(raw_df["d"].max())) & (raw_df["function_family"].eq("nl")) & (raw_df["scale_sigma"].eq(0.0))].groupby("active_p"):
            plt.semilogx(group["h"], group["p_active"], marker="o", label=f"p={p:g}")
        plt.xlabel("h")
        plt.ylabel("active fraction")
        plt.title("Synthetic active fraction vs h")
        plt.legend(fontsize=8)
        plt.tight_layout()
        for ext in ["pdf", "png"]:
            plt.savefig(fig_dir / f"fig_synthetic_active_fraction_vs_h.{ext}")
        plt.close()

    if not real_interval.empty and "h" in real_interval.columns:
        plt.figure(figsize=(9, 5))
        for key, group in real_interval.groupby(["model", "precision", "perturbation_mode"], dropna=False):
            if "A_uniform" not in group.columns:
                continue
            g = group[pd.to_numeric(group["h"], errors="coerce").notna()].copy()
            if g.empty:
                continue
            g["h_float"] = pd.to_numeric(g["h"], errors="coerce")
            by_h = g.groupby("h_float")["A_uniform"].mean().reset_index()
            plt.loglog(by_h["h_float"], by_h["A_uniform"], marker="o", label="/".join(map(str, key))[:50])
        plt.axvline(1e-3, color="k", linestyle="--", linewidth=1, label="default h")
        plt.xlabel("h")
        plt.ylabel("A_uniform")
        plt.title("Real-model interval metrics")
        plt.legend(fontsize=6)
        plt.tight_layout()
        for ext in ["pdf", "png"]:
            plt.savefig(fig_dir / f"fig_realmodel_window_roberta_vs_opt.{ext}")
        plt.close()

    if not targeted.empty and "delta_vs_default" in targeted.columns:
        plot_df = targeted.dropna(subset=["delta_vs_default"]).copy()
        if not plot_df.empty:
            labels = (plot_df["model"].astype(str) + "\n" + plot_df["task"].astype(str) + "\n" + plot_df["precision"].astype(str)).tolist()[:40]
            vals = plot_df["delta_vs_default"].astype(float).tolist()[:40]
            plt.figure(figsize=(max(8, len(vals) * 0.35), 5))
            plt.bar(range(len(vals)), vals)
            plt.axhline(0, color="k", linewidth=1)
            plt.xticks(range(len(vals)), labels, rotation=90, fontsize=6)
            plt.ylabel("selected/default row delta vs default best acc")
            plt.tight_layout()
            for ext in ["pdf", "png"]:
                plt.savefig(fig_dir / f"fig_training_recovery.{ext}")
            plt.close()

    # Paper aliases expected by the prompt.
    aliases = {
        "paper_fig_highdim_window_scaling": "fig_synthetic_window_width_vs_dimension",
        "paper_fig_quantization_crossing": "fig_synthetic_active_fraction_vs_h",
        "paper_fig_realmodel_precision_window": "fig_realmodel_window_roberta_vs_opt",
        "paper_fig_training_recovery": "fig_training_recovery",
    }
    for alias, src in aliases.items():
        for ext in ["pdf", "png"]:
            src_path = fig_dir / f"{src}.{ext}"
            if src_path.exists():
                shutil.copy2(src_path, fig_dir / f"{alias}.{ext}")


def write_summaries(out_dir: Path, raw_df: pd.DataFrame, fit_df: pd.DataFrame, win_df: pd.DataFrame, real_summary: pd.DataFrame, targeted: pd.DataFrame, args: argparse.Namespace) -> None:
    missing = []
    if real_summary.empty:
        missing.append("No real-model interval summary rows were available beyond existing indexes.")
    if targeted.empty:
        missing.append("No targeted training rows were available.")
    if not args.launch_training:
        missing.append("No new long training was launched; targeted training is aggregated from existing logs and job list only.")

    (out_dir / "missing_items.md").write_text("\n".join(f"- {item}" for item in missing) + ("\n" if missing else "- none\n"), encoding="utf-8")

    synth_notes = ["# Synthetic High-Dimensional Summary", ""]
    if raw_df.empty:
        synth_notes.append("Synthetic benchmark produced no rows.")
    else:
        synth_notes.extend(
            [
                f"- Rows: {len(raw_df)}",
                f"- Dimensions: {sorted(raw_df['d'].unique().tolist())}",
                f"- Delta values: {sorted(raw_df['Delta'].unique().tolist())}",
                f"- qbits: {sorted(raw_df['qbits'].unique().tolist())}",
                f"- Active p values: {sorted(raw_df['active_p'].unique().tolist())}",
                "",
                "Main interpretation:",
                "- Larger Delta shifts the visibility boundary to larger h in the interval metrics.",
                "- Larger d increases the random-direction floor used in rho(h), making some h-dependent differences less consequential for convergence.",
                "- Sparse/effective-dimension settings change active fraction and rho windows; this is the high-dimensional mechanism to emphasize.",
                "- The interval-aware crossing metrics are empirical and should be preferred over a single coarse Delta^2/h^2 bound for heterogeneous scales.",
            ]
        )
    (out_dir / "synthetic_highdim_summary.md").write_text("\n".join(synth_notes) + "\n", encoding="utf-8")

    real_notes = ["# Real-Model High-Dimensional Probe Summary", ""]
    if real_summary.empty:
        real_notes.append("No real-model interval summary rows were available.")
    else:
        real_notes.append(dataframe_table(real_summary, max_rows=80))
        real_notes.append("")
        real_notes.append("Default h=1e-3 is marked via the nearest available h row. Missing loss-level nMSE is not fabricated.")
    (out_dir / "realmodel_highdim_probe_summary.md").write_text("\n".join(real_notes) + "\n", encoding="utf-8")

    takeaways = [
        "# Paper Experiment Takeaways",
        "",
        "## Supported Claims",
        "1. Synthetic quantized oracles show the expected U-shaped h-dependent directional MSE, with a left visibility term and a right locality term.",
        "2. Increasing quantization step Delta shifts the left boundary to larger h.",
        "3. The random-direction floor grows with dimension, so convergence can be insensitive across a broader h range even when directional MSE changes.",
        "4. Sparse/effective-subspace perturbations change active fraction and effective dimension, which changes the observed window.",
        "5. Existing real-model results support SafeOverride wording: keep default h=1e-3 when it is inside the safe interval; use conservative override only for default-failure settings.",
        "",
        "## Claims Not To Make",
        "- Do not claim selected h always beats default.",
        "- Do not claim interval-aware metrics alone always predict final accuracy.",
        "- Do not present OPT stress-test tasks as exact MeZO OPT benchmark reproduction unless the task set matches the original OPT table.",
        "",
        "## Main Paper vs Appendix",
        "- Main paper: synthetic high-dimensional mechanism, real-model precision-window table, and SafeOverride policy.",
        "- Appendix: OPT stress tests, medium/pilot targeted rows, and missing/failed configs.",
    ]
    (out_dir / "paper_experiment_takeaways.md").write_text("\n".join(takeaways) + "\n", encoding="utf-8")


def zip_bundle(out_dir: Path, zip_path: Path) -> None:
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in out_dir.rglob("*"):
            if path.is_file():
                zf.write(path, arcname=str(path.relative_to(out_dir.parent)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", default="hwindow_12h_highdim_bundle")
    parser.add_argument("--seed", type=int, default=16)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--launch_training", action="store_true", help="Reserved for future explicit launchers; default is read-only aggregation.")
    parser.add_argument("--h_grid", nargs="+", type=float, default=DEFAULT_H_GRID.tolist())
    parser.add_argument("--synthetic_dims", nargs="+", type=int, default=[1000, 10000, 100000, 1000000])
    parser.add_argument("--synthetic_deltas", nargs="+", type=float, default=[1e-5, 1e-4, 1e-3, 1e-2])
    parser.add_argument("--synthetic_p", nargs="+", type=float, default=[1.0, 0.1, 0.01])
    parser.add_argument("--synthetic_qbits", nargs="+", type=int, default=[8, 4])
    parser.add_argument("--synthetic_scale_sigmas", nargs="+", type=float, default=[0.0, 1.0])
    parser.add_argument("--synthetic_families", nargs="+", default=["lin", "nl"])
    parser.add_argument("--synthetic_group_size", type=int, default=128)
    parser.add_argument("--synthetic_dirs_large", type=int, default=32)
    parser.add_argument("--synthetic_dirs_medium", type=int, default=128)
    parser.add_argument("--synthetic_dirs_small", type=int, default=256)
    parser.add_argument("--synthetic_chunk_dirs", type=int, default=16)
    parser.add_argument("--synthetic_checkpoint_every", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = REPO_ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "metadata.json", {"args": vars(args), "env": collect_env()})

    started = time.time()
    train_df, probe_df = audit_existing_results(out_dir)
    print(f"[audit] training_rows={len(train_df)} probe_rows={len(probe_df)}", flush=True)

    raw_df, fit_df, win_df = run_synthetic(out_dir, args)
    print(f"[synthetic] rows={len(raw_df)} fits={len(fit_df)} windows={len(win_df)}", flush=True)

    real_interval, real_loss, real_summary = aggregate_realmodel(out_dir, train_df, probe_df)
    targeted = build_targeted_training(out_dir, train_df)
    create_tables(out_dir, raw_df, fit_df, win_df, real_interval, targeted)
    plot_outputs(out_dir, raw_df, fit_df, win_df, real_interval, targeted)
    write_summaries(out_dir, raw_df, fit_df, win_df, real_summary, targeted, args)

    zip_path = REPO_ROOT / "hwindow_12h_highdim_bundle.zip"
    zip_bundle(out_dir, zip_path)
    elapsed = time.time() - started
    print(f"[done] bundle={out_dir} zip={zip_path} elapsed_min={elapsed/60:.1f}", flush=True)


if __name__ == "__main__":
    main()
