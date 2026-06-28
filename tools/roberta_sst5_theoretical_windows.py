#!/usr/bin/env python
"""Probe-only vector-level guardrail windows for RoBERTa-large / SST-5.

This script computes the V13/V4 guardrail quantity

    rho(h) = E[(d_h - d*)^2 ||u||^2] / V_dir

from saved or deterministic task-start RoBERTa checkpoints.  It deliberately
keeps scalar directional nMSE separate from rho.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import re
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = REPO_ROOT / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import smoke_rtnclip_roberta_sst5 as smoke  # noqa: E402

EPS = 1e-30
PRIMARY_FIT_METHOD = "dep_log_soft_l1"
DEFAULT_H_GRIDS = {
    "fp32": [1e-9, 1e-8, 1e-7, 3e-7, 1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 1.5e-3, 2e-3, 3e-3, 4e-3, 5e-3, 1e-2],
    "fp16": [3e-7, 1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 1.5e-3, 2e-3, 3e-3, 5e-3, 1e-2],
    "int8": [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 1.5e-3, 2e-3, 3e-3, 5e-3, 1e-2],
    "int4": [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 1.5e-3, 2e-3, 3e-3, 5e-3, 1e-2],
}
TAUS = [0.1, 0.5, 1.0, 2.0]


@dataclass
class PrecisionSpec:
    name: str
    model_dtype: torch.dtype
    master_dtype: torch.dtype
    quant_bits: Optional[int]
    h_grid: List[float]


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: Sequence[Dict[str, object]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames), extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def append_jsonl(path: Path, row: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, sort_keys=True, default=str) + "\n")


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def finite_float(value) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        x = float(value)
    except Exception:
        return None
    return x if math.isfinite(x) else None


def mean(xs: Iterable[float]) -> Optional[float]:
    vals = [float(x) for x in xs if math.isfinite(float(x))]
    return sum(vals) / len(vals) if vals else None


def std(xs: Iterable[float]) -> Optional[float]:
    vals = [float(x) for x in xs if math.isfinite(float(x))]
    if len(vals) < 2:
        return None
    m = sum(vals) / len(vals)
    return math.sqrt(sum((x - m) ** 2 for x in vals) / (len(vals) - 1))


def corr(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    pairs = [(float(x), float(y)) for x, y in zip(xs, ys) if math.isfinite(float(x)) and math.isfinite(float(y))]
    if len(pairs) < 2:
        return None
    xv = np.array([p[0] for p in pairs], dtype=np.float64)
    yv = np.array([p[1] for p in pairs], dtype=np.float64)
    if float(np.var(xv)) <= EPS or float(np.var(yv)) <= EPS:
        return None
    return float(np.corrcoef(xv, yv)[0, 1])


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    except Exception:
        return ""


def collect_env() -> Dict[str, object]:
    env = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "hostname": platform.node(),
        "python": sys.executable,
        "python_version": sys.version.replace("\n", " "),
        "git_commit": git_commit(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "DATALOADER_SHUFFLE": os.environ.get("DATALOADER_SHUFFLE", ""),
    }
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(torch.cuda.current_device())
        env.update({"gpu_name": props.name, "gpu_total_memory_mb": int(props.total_memory / 1024 / 1024)})
    return env


def make_specs(precisions: Sequence[str], h_grid_override: Optional[List[float]]) -> List[PrecisionSpec]:
    specs: List[PrecisionSpec] = []
    for p in precisions:
        p = p.lower()
        if p == "fp32":
            specs.append(PrecisionSpec(p, torch.float32, torch.float32, None, h_grid_override or DEFAULT_H_GRIDS[p]))
        elif p == "fp16":
            specs.append(PrecisionSpec(p, torch.float16, torch.float16, None, h_grid_override or DEFAULT_H_GRIDS[p]))
        elif p == "int8":
            specs.append(PrecisionSpec(p, torch.float16, torch.float16, 8, h_grid_override or DEFAULT_H_GRIDS[p]))
        elif p == "int4":
            specs.append(PrecisionSpec(p, torch.float16, torch.float16, 4, h_grid_override or DEFAULT_H_GRIDS[p]))
        else:
            raise ValueError(f"unsupported precision {p}")
    return specs


def move_batch(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}


def forward_loss(model, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
    payload = dict(batch)
    if "token_type_ids" not in payload or payload["token_type_ids"] is None:
        payload["token_type_ids"] = torch.zeros_like(payload["input_ids"])
    return model(**payload)[0].float()


def set_model_from_master(params: Dict[str, torch.nn.Parameter], master: Dict[str, torch.Tensor], dtype: torch.dtype) -> None:
    with torch.no_grad():
        for name, param in params.items():
            if name in master:
                param.copy_(master[name].to(dtype=param.dtype if param.is_floating_point() else dtype))


def compute_grads(model, params: Dict[str, torch.nn.Parameter], master: Dict[str, torch.Tensor], batch, dtype: torch.dtype):
    set_model_from_master(params, master, dtype)
    model.zero_grad(set_to_none=True)
    loss = forward_loss(model, batch)
    loss.backward()
    grads = {}
    g2 = torch.zeros((), device=loss.device, dtype=torch.float64)
    d = 0
    for name, param in params.items():
        if name not in master or param.grad is None:
            continue
        grad = param.grad.detach().float()
        grads[name] = grad.clone()
        g2 += grad.double().square().sum()
        d += int(grad.numel())
    return float(loss.detach().cpu()), grads, float(g2.detach().cpu()), int(d)


def sample_directions(master: Dict[str, torch.Tensor], seed: int, dtype: torch.dtype) -> Dict[str, torch.Tensor]:
    first = next(iter(master.values()))
    gen = torch.Generator(device=first.device).manual_seed(int(seed))
    directions: Dict[str, torch.Tensor] = {}
    for name, tensor in master.items():
        if tensor.is_floating_point():
            directions[name] = torch.randn(tensor.shape, device=tensor.device, generator=gen, dtype=dtype)
    return directions


def direction_stats(grads: Dict[str, torch.Tensor], directions: Dict[str, torch.Tensor]) -> Tuple[float, float]:
    d_star = torch.zeros((), device=next(iter(directions.values())).device, dtype=torch.float64)
    norm_u2 = torch.zeros_like(d_star)
    for name, direction in directions.items():
        u = direction.float()
        norm_u2 += u.double().square().sum()
        if name in grads:
            d_star += (grads[name].double() * u.double()).sum()
    return float(d_star.detach().cpu()), float(norm_u2.detach().cpu())


def build_quantizer_states(master: Dict[str, torch.Tensor], params: Dict[str, torch.nn.Parameter], bits: int, group_size: int):
    q_names = [name for name in smoke.linear_weight_names_from_params(params) if name in master] if hasattr(smoke, "linear_weight_names_from_params") else []
    if not q_names:
        q_names = [name for name in smoke.linear_weight_names(_MODEL_REF) if name in master]  # type: ignore[name-defined]
    states, rows = smoke.refresh_quantizer_states(master, q_names, bits, group_size)
    numel_by_name = {name: params[name].numel() for name in q_names if name in params}
    return states, smoke.aggregate_quantizer_stats(rows, numel_by_name)


def finite_difference(
    model,
    params: Dict[str, torch.nn.Parameter],
    master: Dict[str, torch.Tensor],
    directions: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    h: float,
    model_dtype: torch.dtype,
    states: Optional[Dict[str, smoke.RTNClipState]],
) -> Tuple[float, float, float]:
    with torch.no_grad():
        if states:
            smoke.copy_master_to_model(params, master, directions, h, +1.0, states)
        else:
            for name, param in params.items():
                if name in master:
                    value = master[name].float().add(directions[name].float(), alpha=float(h))
                    param.copy_(value.to(dtype=param.dtype))
        loss_plus = forward_loss(model, batch)
        if states:
            smoke.copy_master_to_model(params, master, directions, h, -1.0, states)
        else:
            for name, param in params.items():
                if name in master:
                    value = master[name].float().add(directions[name].float(), alpha=-float(h))
                    param.copy_(value.to(dtype=param.dtype))
        loss_minus = forward_loss(model, batch)
        set_model_from_master(params, master, model_dtype)
    lp = float(loss_plus.detach().cpu())
    lm = float(loss_minus.detach().cpu())
    return lp, lm, (lp - lm) / (2.0 * float(h))


def high_precision_pair_stats(master: Dict[str, torch.Tensor], directions: Dict[str, torch.Tensor], h: float) -> Dict[str, object]:
    norm_u2 = 0.0
    for direction in directions.values():
        norm_u2 += float(direction.float().double().square().sum().detach().cpu())
    intended_norm = 2.0 * float(h) * math.sqrt(max(norm_u2, 0.0))
    return {
        "probe_active_frac": 1.0,
        "probe_alignment": 1.0,
        "probe_norm_ratio": 1.0,
        "delta_q_norm": intended_norm,
        "nominal_delta_norm": intended_norm,
        "multi_code_jump_frac": None,
        "saturation_frac": 0.0,
    }


def lowbit_pair_stats(master: Dict[str, torch.Tensor], directions: Dict[str, torch.Tensor], states: Dict[str, smoke.RTNClipState], h: float) -> Dict[str, object]:
    stats = smoke.perturbation_metrics(master, directions, states, float(h))
    return {
        "probe_active_frac": stats.get("active_frac"),
        "probe_alignment": stats.get("alignment"),
        "probe_norm_ratio": stats.get("norm_ratio"),
        "delta_q_norm": stats.get("delta_q_norm"),
        "nominal_delta_norm": stats.get("ideal_displacement_norm"),
        "multi_code_jump_frac": None,
        "saturation_frac": stats.get("saturation_frac"),
        "delta_visibility_nmse": stats.get("delta_visibility_nmse"),
    }


def fit_rho(hs: np.ndarray, rho: np.ndarray, method: str) -> Dict[str, object]:
    from scipy.optimize import least_squares, nnls

    mask = np.isfinite(hs) & np.isfinite(rho) & (hs > 0) & (rho > 0)
    hs = hs[mask].astype(np.float64)
    rho = rho[mask].astype(np.float64)
    if len(hs) < 4:
        return {"fit_method": method, "fit_stability_flag": "insufficient_points", "A": np.nan, "B": np.nan, "C": np.nan}
    dep_only = method == PRIMARY_FIT_METHOD
    if dep_only:
        X = np.stack([1.0 / (hs * hs), hs * hs], axis=1)
    else:
        X = np.stack([1.0 / (hs * hs), hs * hs, np.ones_like(hs)], axis=1)
    if method == "linear_nnls":
        coef, _ = nnls(X, rho)
    else:
        init, _ = nnls(X, rho)
        init = np.maximum(init, 1e-30)

        def resid(log_coef):
            coef = np.exp(log_coef)
            pred = X @ coef
            return np.log(pred + EPS) - np.log(rho + EPS)

        res = least_squares(resid, np.log(init), loss="soft_l1", max_nfev=20000)
        coef = np.exp(res.x)
    if dep_only:
        coef = np.array([float(coef[0]), float(coef[1]), 0.0], dtype=np.float64)
        X = np.stack([1.0 / (hs * hs), hs * hs, np.ones_like(hs)], axis=1)
    pred_total = X @ coef
    y = np.log(rho + EPS)
    yp = np.log(pred_total + EPS)
    ss_res = float(np.sum((y - yp) ** 2))
    ss_tot = float(np.sum((y - float(np.mean(y))) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    rmse_log = math.sqrt(ss_res / max(len(y), 1))
    A, B, C = [float(x) for x in coef]
    flag = "stable"
    notes: List[str] = []
    if A <= 0 or B <= 0:
        flag = "missing_tail"
        notes.append("A or B nonpositive")
    elif not np.isfinite(r2) or r2 < 0.8 or rmse_log > 1.0:
        flag = "poor_fit"
        notes.append(f"insufficient fit quality: log_R2={r2:.4g}, log_RMSE={rmse_log:.4g}")
    else:
        h_ref = (A / B) ** 0.25
        h_min = float(np.min(hs))
        h_max = float(np.max(hs))
        dep_min = 2.0 * math.sqrt(A * B)
        # A fitted h_ref far outside the probed h range is a boundary/identifiability
        # solution, not a useful guardrail certificate. This catches cases where C
        # absorbs the curve and the right-tail coefficient effectively vanishes.
        if not np.isfinite(h_ref) or h_ref < h_min / 10.0 or h_ref > h_max * 10.0:
            flag = "boundary_solution"
            notes.append(f"h_ref={h_ref:.6g} outside [{h_min:.6g}, {h_max:.6g}] by >10x")
        if not np.isfinite(dep_min) or dep_min <= 0:
            flag = "missing_tail"
            notes.append("rho_min_dep nonpositive/invalid")
        if B < 1e-30:
            flag = "missing_tail"
            notes.append("B effectively zero; no identifiable locality/right tail")
    return {
        "fit_method": method,
        "A": A,
        "B": B,
        "C": C,
        "fit_quality_r2_log": r2,
        "fit_rmse_log": rmse_log,
        "fit_stability_flag": flag,
        "fit_notes": "; ".join(notes),
    }


def window_rows_for_fit(precision: str, checkpoint_id: str, fit: Dict[str, object]) -> List[Dict[str, object]]:
    A = finite_float(fit.get("A"))
    B = finite_float(fit.get("B"))
    C = finite_float(fit.get("C"))
    rows: List[Dict[str, object]] = []
    stable = fit.get("fit_stability_flag") == "stable"
    if A is None or B is None or A <= 0 or B <= 0 or not stable:
        for tau in TAUS:
            note = fit.get("fit_notes") or "A or B unavailable/nonpositive"
            if A is not None and B is not None and A > 0 and B > 0 and not stable:
                note = f"fit unstable: {fit.get('fit_stability_flag')}; {note}".strip()
            rows.append({
                "precision": precision, "checkpoint_id": checkpoint_id, "fit_method": fit.get("fit_method"),
                "A": A, "B": B, "C": C, "tau": tau, "h_ref": None, "rho_min_dep": None,
                "h_low": None, "h_high": None, "default_h": 1e-3, "default_in_window": None,
                "rho_dep_at_default": None, "fit_quality_r2_or_log_error": fit.get("fit_quality_r2_log"),
                "fit_stability_flag": "unavailable" if stable else fit.get("fit_stability_flag"),
                "notes": note,
            })
        return rows
    h_ref = (A / B) ** 0.25
    rho_min = 2.0 * math.sqrt(A * B)
    rho_default = A / (1e-3 ** 2) + B * (1e-3 ** 2)
    for tau in TAUS:
        disc = tau * tau - 4.0 * A * B
        h_low = h_high = None
        default_in = None
        notes = ""
        if disc < 0 or rho_min > tau:
            notes = "no certified window"
        else:
            x_low = (tau - math.sqrt(max(disc, 0.0))) / (2.0 * B)
            x_high = (tau + math.sqrt(max(disc, 0.0))) / (2.0 * B)
            h_low = math.sqrt(max(x_low, 0.0))
            h_high = math.sqrt(max(x_high, 0.0))
            default_in = h_low <= 1e-3 <= h_high
        rows.append({
            "precision": precision,
            "checkpoint_id": checkpoint_id,
            "fit_method": fit.get("fit_method"),
            "A": A,
            "B": B,
            "C": C,
            "h_ref": h_ref,
            "rho_min_dep": rho_min,
            "tau": tau,
            "h_low": h_low,
            "h_high": h_high,
            "default_h": 1e-3,
            "default_in_window": default_in,
            "rho_dep_at_default": rho_default,
            "fit_quality_r2_or_log_error": fit.get("fit_quality_r2_log"),
            "fit_stability_flag": fit.get("fit_stability_flag"),
            "notes": notes,
        })
    return rows


def summarize_by_h(raw_rows: Sequence[Dict[str, object]], checkpoint_id: str, precision: str) -> Tuple[List[Dict[str, object]], Dict[str, float]]:
    groups: Dict[float, List[Dict[str, object]]] = defaultdict(list)
    for row in raw_rows:
        if row["precision"] == precision and row["checkpoint_id"] == checkpoint_id:
            groups[float(row["h"])].append(row)
    out = []
    vdir_values = []
    for h, rows in sorted(groups.items()):
        dstar = [float(r["d_star"]) for r in rows]
        dh = [float(r["d_h"]) for r in rows]
        e = [float(r["e_h"]) for r in rows]
        vec = [float(r["vector_error_h"]) for r in rows]
        dstar2 = [x * x for x in dstar]
        vdir = [float(r["V_dir_sample_direction"]) for r in rows]
        vdir_sample = mean(vdir)
        vdir_values.extend(vdir)
        vh_raw = mean(vec)
        scalar_den = max(mean(dstar2) or 0.0, EPS)
        scalar_nmse = (mean([x * x for x in e]) or 0.0) / scalar_den
        sign = mean([1.0 if np.sign(a) == np.sign(b) else 0.0 for a, b in zip(dh, dstar)])
        out.append({
            "precision": precision,
            "checkpoint_id": checkpoint_id,
            "h": h,
            "n_directions": len(rows),
            "G": rows[0]["G"],
            "G2": rows[0]["G2"],
            "d": rows[0]["d"],
            "V_dir_formula": rows[0]["V_dir_formula"],
            "V_dir_sample": vdir_sample,
            "scalar_nmse": scalar_nmse,
            "directional_corr": corr(dh, dstar),
            "sign_agreement": sign,
            "V_h_raw": vh_raw,
            "rho_raw": (vh_raw or 0.0) / max(vdir_sample or 0.0, EPS),
            "rho_raw_std_or_bootstrap_ci": std([x / max(vs, EPS) for x, vs in zip(vec, vdir)]),
            "d_h_mean": mean(dh),
            "d_h_std": std(dh),
            "d_star_mean": mean(dstar),
            "d_star_std": std(dstar),
            "probe_active_frac": mean([float(r["probe_active_frac"]) for r in rows if finite_float(r.get("probe_active_frac")) is not None]),
            "probe_alignment": mean([float(r["probe_alignment"]) for r in rows if finite_float(r.get("probe_alignment")) is not None]),
            "probe_norm_ratio": mean([float(r["probe_norm_ratio"]) for r in rows if finite_float(r.get("probe_norm_ratio")) is not None]),
            "saturation_frac": mean([float(r["saturation_frac"]) for r in rows if finite_float(r.get("saturation_frac")) is not None]),
        })
    return out, {"V_dir_sample_global": mean(vdir_values) or float("nan")}


def accuracy_sources() -> Dict[str, Path]:
    return {
        "fp32_fp16": REPO_ROOT / "experiments/main_latest/mezo/roberta-large/sst5/fp32_fp16_h_sweep_left_tail_seed16_bs64_ckpt1k_20260519/summaries/merged_fp32_fp16_h_sweep_1e-9_to_1e-2.csv",
        "int8": REPO_ROOT / "outputs/rtnclip_lowbit_roberta_sst5_seed16_20260519_batch/int8_hsearch_summary.csv",
        "int4": REPO_ROOT / "outputs/rtnclip_int4_g128_rtnclip_roberta_sst5_seed16_20260521/int4_hsearch_summary.csv",
    }


def accuracy_sweep_points() -> List[Dict[str, object]]:
    sources = accuracy_sources()
    points: List[Dict[str, object]] = []
    for row in read_csv_rows(sources["fp32_fp16"]):
        prec = str(row.get("precision_mode", "")).lower()
        h = finite_float(row.get("h"))
        acc = finite_float(row.get("best_eval_acc"))
        if prec in {"fp32", "fp16"} and h is not None and acc is not None:
            points.append({"precision": prec, "h": h, "best_eval_acc": acc, "source_path": str(sources["fp32_fp16"])})
    for prec, key in [("int8", "int8"), ("int4", "int4")]:
        for row in read_csv_rows(sources[key]):
            h = finite_float(row.get("h"))
            acc = finite_float(row.get("best_eval_acc"))
            if h is not None and acc is not None:
                points.append({"precision": prec, "h": h, "best_eval_acc": acc, "source_path": str(sources[key])})
    return sorted(points, key=lambda r: (str(r["precision"]), float(r["h"])))


def accuracy_good_sets(points: Optional[Sequence[Dict[str, object]]] = None) -> List[Dict[str, object]]:
    points = list(points) if points is not None else accuracy_sweep_points()
    by_prec: Dict[str, List[Tuple[float, float, str]]] = defaultdict(list)
    for row in points:
        h = finite_float(row.get("h"))
        acc = finite_float(row.get("best_eval_acc"))
        prec = str(row.get("precision", "")).lower()
        if h is not None and acc is not None:
            by_prec[prec].append((h, acc, str(row.get("source_path", ""))))
    out: List[Dict[str, object]] = []
    for prec, vals in by_prec.items():
        if not vals:
            continue
        max_acc = max(acc for _, acc, _ in vals)
        h_best = min([h for h, acc, _ in vals if abs(acc - max_acc) <= 1e-12])
        src = vals[0][2]
        for delta in [0.005, 0.01, 0.02]:
            good = sorted(h for h, acc, _ in vals if acc >= max_acc - delta)
            out.append({
                "precision": prec,
                "threshold_type": f"best_acc_minus_{delta:g}",
                "max_acc": max_acc,
                "h_best_acc": h_best,
                "h_good_low": min(good) if good else None,
                "h_good_high": max(good) if good else None,
                "all_good_h_values": " ".join(f"{h:.9g}" for h in good),
                "source_path": src,
            })
    return out


def make_comparison_rows(fit_rows: Sequence[Dict[str, object]], acc_rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    acc_primary = {r["precision"]: r for r in acc_rows if r["threshold_type"] == "best_acc_minus_0.01"}
    rows = []
    for prec in ["fp32", "fp16", "int8", "int4"]:
        w1 = [
            r for r in fit_rows
            if r["precision"] == prec
            and r["fit_method"] == PRIMARY_FIT_METHOD
            and float(r["tau"]) == 1.0
            and r.get("fit_stability_flag") == "stable"
        ]
        if not w1:
            w1 = [
                r for r in fit_rows
                if r["precision"] == prec
                and float(r["tau"]) == 1.0
                and r.get("fit_stability_flag") == "stable"
            ]
        fit = w1[0] if w1 else {}
        acc = acc_primary.get(prec, {})
        h_low = finite_float(fit.get("h_low"))
        h_high = finite_float(fit.get("h_high"))
        default_in = fit.get("default_in_window")
        if not fit:
            interp = "no stable fit"
        elif h_low is None or h_high is None:
            interp = "empirically default-safe but uncertified"
        elif bool(default_in):
            interp = "broad default-safe" if h_low <= 1e-3 <= h_high else "conservative certified"
        else:
            interp = "default-risk"
        rows.append({
            "precision": prec,
            "checkpoint_id": fit.get("checkpoint_id"),
            "theoretical_h_ref": fit.get("h_ref"),
            "theoretical_W1_low": h_low,
            "theoretical_W1_high": h_high,
            "default_in_W1": default_in,
            "empirical_good_low": acc.get("h_good_low"),
            "empirical_good_high": acc.get("h_good_high"),
            "h_best_acc": acc.get("h_best_acc"),
            "interpretation": interp,
        })
    return rows


def plot_outputs(
    out_dir: Path,
    summary_rows: Sequence[Dict[str, object]],
    fit_rows: Sequence[Dict[str, object]],
    acc_rows: Sequence[Dict[str, object]],
    acc_points: Sequence[Dict[str, object]],
) -> None:
    import matplotlib.pyplot as plt

    colors = {"fp32": "#1f77b4", "fp16": "#2ca02c", "int8": "#ff7f0e", "int4": "#d62728"}
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 1, figsize=(7.2, 6.2), sharex=True)
    for prec in ["fp32", "fp16", "int8", "int4"]:
        rows = sorted([r for r in summary_rows if r["precision"] == prec], key=lambda r: float(r["h"]))
        if rows:
            axes[0].plot([r["h"] for r in rows], [r["scalar_nmse"] for r in rows], marker="o", label=prec, color=colors[prec])
        acc_src = acc_rows
        # Accuracy curves from raw sources are intentionally not joined here; good-set figure carries intervals.
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_ylabel("scalar true directional nMSE")
    axes[0].legend()
    for prec in ["fp32", "fp16", "int8", "int4"]:
        pts = sorted([r for r in acc_points if r["precision"] == prec], key=lambda r: float(r["h"]))
        if pts:
            axes[1].plot([float(r["h"]) for r in pts], [float(r["best_eval_acc"]) for r in pts], marker="o", label=prec, color=colors[prec])
    axes[1].axvline(1e-3, color="black", linestyle="--", linewidth=1, label="default 1e-3")
    axes[1].set_xscale("log")
    axes[1].set_ylabel("best acc good-set markers")
    axes[1].set_xlabel("h")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_roberta_sst5_nmse_and_accuracy_vs_h.pdf")
    fig.savefig(out_dir / "fig_roberta_sst5_nmse_and_accuracy_vs_h.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for prec in ["fp32", "fp16", "int8", "int4"]:
        rows = sorted([r for r in summary_rows if r["precision"] == prec], key=lambda r: float(r["h"]))
        if not rows:
            continue
        hs = np.array([float(r["h"]) for r in rows])
        ax.scatter(hs, [r["rho_raw"] for r in rows], color=colors[prec], label=f"{prec} raw")
        fits = [
            r for r in fit_rows
            if r["precision"] == prec
            and r["fit_method"] == PRIMARY_FIT_METHOD
            and float(r["tau"]) == 1.0
            and r.get("fit_stability_flag") == "stable"
        ]
        if fits and finite_float(fits[0].get("A")) and finite_float(fits[0].get("B")):
            A = float(fits[0]["A"]); B = float(fits[0]["B"])
            grid = np.logspace(np.log10(min(hs)), np.log10(max(hs)), 200)
            ax.plot(grid, A / (grid * grid) + B * grid * grid, color=colors[prec], linewidth=1.5)
            href = finite_float(fits[0].get("h_ref"))
            if href:
                ax.scatter([href], [A / (href * href) + B * href * href], marker="D", color=colors[prec], edgecolor="black", zorder=5)
    ax.axhline(1.0, color="gray", linestyle=":", label="tau=1")
    ax.axvline(1e-3, color="black", linestyle="--", linewidth=1, label="default 1e-3")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("h"); ax.set_ylabel("vector-level rho")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_roberta_sst5_rho_fit_vs_h.pdf")
    fig.savefig(out_dir / "fig_roberta_sst5_rho_fit_vs_h.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    ylabels = ["fp32", "fp16", "int8", "int4"]
    acc_primary = {r["precision"]: r for r in acc_rows if r["threshold_type"] == "best_acc_minus_0.01"}
    for yi, prec in enumerate(ylabels):
        acc = acc_primary.get(prec)
        if acc and finite_float(acc.get("h_good_low")) and finite_float(acc.get("h_good_high")):
            ax.plot([float(acc["h_good_low"]), float(acc["h_good_high"])], [yi, yi], color=colors[prec], linewidth=8, alpha=0.25)
        fit = [
            r for r in fit_rows
            if r["precision"] == prec
            and r["fit_method"] == PRIMARY_FIT_METHOD
            and float(r["tau"]) == 1.0
            and r.get("fit_stability_flag") == "stable"
        ]
        if fit:
            href = finite_float(fit[0].get("h_ref"))
            lo = finite_float(fit[0].get("h_low"))
            hi = finite_float(fit[0].get("h_high"))
            if lo and hi:
                ax.plot([lo, hi], [yi, yi], color=colors[prec], linewidth=4)
            else:
                ax.text(1.2e-5, yi + 0.08, "no tau=1 certificate", fontsize=8, color=colors[prec])
            if href:
                ax.scatter([href], [yi], marker="D", color=colors[prec], edgecolor="black", zorder=5)
        else:
            ax.text(1.2e-5, yi + 0.08, "no stable rho fit", fontsize=8, color=colors[prec])
    ax.axvline(1e-3, color="black", linestyle="--", linewidth=1)
    ax.set_xscale("log")
    ax.set_yticks(range(len(ylabels))); ax.set_yticklabels(ylabels)
    ax.set_ylim(-0.45, len(ylabels) - 0.25)
    ax.set_xlabel("h")
    ax.set_title("Empirical accuracy good set (light) vs theoretical W1 (dark)")
    fig.tight_layout()
    fig.savefig(out_dir / "fig_roberta_sst5_windows_comparison.pdf")
    fig.savefig(out_dir / "fig_roberta_sst5_windows_comparison.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for prec in ["fp32", "fp16", "int8", "int4"]:
        rows = sorted([r for r in summary_rows if r["precision"] == prec], key=lambda r: float(r["h"]))
        fit = [
            r for r in fit_rows
            if r["precision"] == prec
            and r["fit_method"] == PRIMARY_FIT_METHOD
            and float(r["tau"]) == 1.0
            and r.get("fit_stability_flag") == "stable"
        ]
        if not rows or not fit or not finite_float(fit[0].get("A")) or not finite_float(fit[0].get("B")):
            continue
        A = float(fit[0]["A"]); B = float(fit[0]["B"])
        hs = np.array([float(r["h"]) for r in rows])
        residual = np.log(np.array([float(r["rho_raw"]) for r in rows]) + EPS) - np.log(A / (hs * hs) + B * hs * hs + EPS)
        ax.plot(hs, residual, marker="o", label=prec, color=colors[prec])
    ax.axhline(0.0, color="gray", linestyle=":")
    ax.set_xscale("log")
    ax.set_xlabel("h"); ax.set_ylabel("log residual: raw rho - dep fit")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_roberta_sst5_fit_diagnostics.pdf")
    fig.savefig(out_dir / "fig_roberta_sst5_fit_diagnostics.png", dpi=200)
    plt.close(fig)


def run_probe(args: argparse.Namespace) -> None:
    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "env.json", collect_env())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA is required for RoBERTa-large probe")
    h_override = [float(x) for x in args.h_grid.split(",")] if args.h_grid else None
    specs = make_specs(args.precisions, h_override)
    raw_path = out / "raw_probe_metrics.csv"
    raw_jsonl = out / "raw_probe_metrics.jsonl"
    if raw_jsonl.exists():
        raw_jsonl.unlink()

    raw_rows: List[Dict[str, object]] = []
    checkpoint_manifest: List[Dict[str, object]] = []
    global _MODEL_REF

    for spec in specs:
        spec_start = time.time()
        model, train_loader, _dev_loader, data_args, train_sampler = smoke.load_prompt_model_and_data(
            SimpleNamespace(
                repo_root=REPO_ROOT,
                model_id=args.model_id,
                task_name="sst-5",
                seed=args.seed,
                data_seed=args.data_seed,
                batch_size=args.batch_size,
                eval_batch_size=args.batch_size,
                dataset_mode="full",
                data_dir=None,
                num_k=16,
            ),
            device,
        )
        _MODEL_REF = model
        model.to(dtype=spec.model_dtype)
        model.eval()
        params = smoke.named_parameter_map(model)
        master = {
            name: p.detach().clone().to(device=device, dtype=spec.master_dtype)
            for name, p in params.items()
            if p.detach().is_floating_point()
        }
        checkpoint_id = "task_start_seed16_deterministic"
        ckpt_path = out / "checkpoints" / f"{checkpoint_id}_{spec.name}_master.pt"
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"checkpoint_id": checkpoint_id, "precision": spec.name, "master": {k: v.cpu() for k, v in master.items()}}, ckpt_path)
        checkpoint_manifest.append({
            "precision": spec.name,
            "checkpoint_id": checkpoint_id,
            "checkpoint_step": 0,
            "checkpoint_path": str(ckpt_path),
            "source_run_id": "deterministic task-start model load",
            "source_precision_mode": "common task-start",
            "common_across_precision_modes": True,
        })
        states = None
        quant_summary: Dict[str, object] = {}
        if spec.quant_bits is not None:
            q_names = [name for name in smoke.linear_weight_names(model) if name in master]
            states, q_rows = smoke.refresh_quantizer_states(master, q_names, int(spec.quant_bits), args.group_size)
            quant_summary = smoke.aggregate_quantizer_stats(q_rows, {name: params[name].numel() for name in q_names})
            write_json(out / f"quantizer_summary_{spec.name}.json", quant_summary)
        batch_iter = iter(train_loader)
        batches = [move_batch(next(batch_iter), device) for _ in range(args.num_batches)]
        for batch_index, batch in enumerate(batches):
            base_loss, grads, G2, d = compute_grads(model, params, master, batch, spec.model_dtype)
            V_dir_formula = (float(d) + 1.0) * G2
            for direction_index in range(args.num_directions):
                direction_seed = args.direction_seed_base + direction_index
                directions = sample_directions(master, direction_seed, spec.master_dtype)
                d_star, norm_u2 = direction_stats(grads, directions)
                V_dir_sample_direction = d_star * d_star * norm_u2 - 2.0 * d_star * d_star + G2
                for h in spec.h_grid:
                    lp, lm, d_h = finite_difference(model, params, master, directions, batch, float(h), spec.model_dtype, states)
                    e_h = d_h - d_star
                    if states:
                        pstats = lowbit_pair_stats(master, directions, states, float(h))
                    else:
                        pstats = high_precision_pair_stats(master, directions, float(h))
                    row = {
                        "precision": spec.name,
                        "checkpoint_id": checkpoint_id,
                        "checkpoint_step": 0,
                        "h": float(h),
                        "batch_index": batch_index,
                        "direction_index": direction_index,
                        "direction_seed": direction_seed,
                        "d_star": d_star,
                        "d_h": d_h,
                        "e_h": e_h,
                        "norm_u2": norm_u2,
                        "vector_error_h": e_h * e_h * norm_u2,
                        "loss_plus": lp,
                        "loss_minus": lm,
                        "loss_base": base_loss,
                        "G": math.sqrt(max(G2, 0.0)),
                        "G2": G2,
                        "d": d,
                        "V_dir_formula": V_dir_formula,
                        "V_dir_sample_direction": V_dir_sample_direction,
                        "direction_distribution": "torch.randn_like trainable parameters",
                        "direction_dtype": str(spec.master_dtype),
                        "quantizer": "none" if spec.quant_bits is None else "G128_RTNClip_shared_grid_fake_quant",
                        "quant_bits": spec.quant_bits or 32,
                        "group_size": args.group_size if spec.quant_bits else 0,
                        "probe_active_frac": pstats.get("probe_active_frac"),
                        "probe_alignment": pstats.get("probe_alignment"),
                        "probe_norm_ratio": pstats.get("probe_norm_ratio"),
                        "delta_q_norm": pstats.get("delta_q_norm"),
                        "nominal_delta_norm": pstats.get("nominal_delta_norm"),
                        "saturation_frac": pstats.get("saturation_frac"),
                    }
                    raw_rows.append(row)
                    append_jsonl(raw_jsonl, row)
                if (direction_index + 1) % max(1, args.progress_every) == 0:
                    print(f"[{spec.name}] batch {batch_index+1}/{len(batches)} direction {direction_index+1}/{args.num_directions}", flush=True)
                del directions
        print(f"[{spec.name}] done in {time.time() - spec_start:.1f}s", flush=True)
        del model
        torch.cuda.empty_cache()

    raw_fields = [
        "precision", "checkpoint_id", "checkpoint_step", "h", "batch_index", "direction_index", "direction_seed",
        "d_star", "d_h", "e_h", "norm_u2", "vector_error_h", "loss_plus", "loss_minus", "loss_base",
        "G", "G2", "d", "V_dir_formula", "V_dir_sample_direction", "direction_distribution", "direction_dtype",
        "quantizer", "quant_bits", "group_size", "probe_active_frac", "probe_alignment", "probe_norm_ratio",
        "delta_q_norm", "nominal_delta_norm", "saturation_frac",
    ]
    write_csv(raw_path, raw_rows, raw_fields)

    summary_rows: List[Dict[str, object]] = []
    fit_rows: List[Dict[str, object]] = []
    for spec in specs:
        rows, _ = summarize_by_h(raw_rows, "task_start_seed16_deterministic", spec.name)
        summary_rows.extend(rows)
        hs = np.array([float(r["h"]) for r in rows], dtype=np.float64)
        rho = np.array([float(r["rho_raw"]) for r in rows], dtype=np.float64)
        for method in [PRIMARY_FIT_METHOD, "log_soft_l1", "linear_nnls"]:
            fit = fit_rho(hs, rho, method)
            fit_rows.extend(window_rows_for_fit(spec.name, "task_start_seed16_deterministic", fit))

    summary_fields = [
        "precision", "checkpoint_id", "h", "n_directions", "G", "G2", "d", "V_dir_formula", "V_dir_sample",
        "scalar_nmse", "directional_corr", "sign_agreement", "V_h_raw", "rho_raw", "rho_raw_std_or_bootstrap_ci",
        "d_h_mean", "d_h_std", "d_star_mean", "d_star_std", "probe_active_frac", "probe_alignment",
        "probe_norm_ratio", "saturation_frac",
    ]
    fit_fields = [
        "precision", "checkpoint_id", "fit_method", "A", "B", "C", "h_ref", "rho_min_dep", "tau",
        "h_low", "h_high", "default_h", "default_in_window", "rho_dep_at_default",
        "fit_quality_r2_or_log_error", "fit_stability_flag", "notes",
    ]
    acc_points = accuracy_sweep_points()
    acc_rows = accuracy_good_sets(acc_points)
    comparison_rows = make_comparison_rows(fit_rows, acc_rows)

    write_csv(out / "probe_summary_by_h.csv", summary_rows, summary_fields)
    write_csv(out / "fitted_windows.csv", fit_rows, fit_fields)
    write_csv(out / "empirical_accuracy_good_sets.csv", acc_rows, [
        "precision", "threshold_type", "max_acc", "h_best_acc", "h_good_low", "h_good_high",
        "all_good_h_values", "source_path",
    ])
    write_csv(out / "accuracy_sweep_points.csv", acc_points, [
        "precision", "h", "best_eval_acc", "source_path",
    ])
    write_csv(out / "comparison_table_for_paper.csv", comparison_rows, [
        "precision", "checkpoint_id", "theoretical_h_ref", "theoretical_W1_low", "theoretical_W1_high",
        "default_in_W1", "empirical_good_low", "empirical_good_high", "h_best_acc", "interpretation",
    ])
    write_csv(out / "checkpoint_manifest.csv", checkpoint_manifest, [
        "precision", "checkpoint_id", "checkpoint_step", "checkpoint_path", "source_run_id",
        "source_precision_mode", "common_across_precision_modes",
    ])
    plot_outputs(out, summary_rows, fit_rows, acc_rows, acc_points)
    write_readme(out, args, specs, checkpoint_manifest, comparison_rows)
    write_paper_note(out, comparison_rows, fit_rows, acc_rows)
    print(f"Wrote {out}")


def write_readme(out: Path, args: argparse.Namespace, specs: Sequence[PrecisionSpec], ckpts: Sequence[Dict[str, object]], comparison: Sequence[Dict[str, object]]) -> None:
    lines = [
        "# RoBERTa/SST-5 Theoretical Guardrail Windows",
        "",
        "Probe-only run. No training was launched.",
        "",
        "## Checkpoints",
        "",
        "No common saved step-0 task-start checkpoint was found in the historical h-sweeps. The script therefore loads the deterministic RoBERTa-large/SST-5 task-start state with seed/data_seed 16, saves that master state under this output directory, and uses it as a common checkpoint across precision modes. This avoids mixing checkpoints trained at different h values.",
        "",
    ]
    for c in ckpts:
        lines.append(f"- {c['precision']}: `{c['checkpoint_path']}`")
    lines.extend([
        "",
        "Note: the `.pt` master checkpoint files are local reproducibility artifacts and are intentionally excluded from Git because they are several GB. They are deterministic task-start states and can be regenerated by rerunning this script without `--reuse_raw_metrics`.",
        "",
        "## Probe Setup",
        "",
        f"- model: `{args.model_id}`",
        "- task: SST-5 full data",
        f"- seed/data_seed: {args.seed}/{args.data_seed}",
        f"- batch size: {args.batch_size}; num_batches: {args.num_batches}",
        f"- directions per precision: {args.num_directions}",
        f"- direction seed base: {args.direction_seed_base}",
        "- trainable subspace: all floating model parameters, matching the RTNClip dense runner",
        "- low-bit forward oracle: G128 RTNClip shared-grid fake quantization on Linear.weight; non-Linear parameters remain unquantized in the forward state",
        "- rho denominator: sampled vector random-direction floor `V_dir_sample`; `V_dir_formula` is also reported.",
        "- scalar nMSE is reported separately and is not used as rho.",
        "",
        "## Precision H Grids",
        "",
    ])
    for s in specs:
        lines.append(f"- {s.name}: " + ", ".join(f"{h:g}" for h in s.h_grid))
    lines.extend(["", "## Comparison Summary", "", "| precision | h_ref | W1 | default in W1 | empirical good set | interpretation |", "|---|---:|---|---|---|---|"])
    for row in comparison:
        w1 = "n/a"
        if finite_float(row.get("theoretical_W1_low")) is not None and finite_float(row.get("theoretical_W1_high")) is not None:
            w1 = f"[{float(row['theoretical_W1_low']):.3g}, {float(row['theoretical_W1_high']):.3g}]"
        emp = "n/a"
        if row.get("empirical_good_low") not in {None, ""}:
            emp = f"[{float(row['empirical_good_low']):.3g}, {float(row['empirical_good_high']):.3g}]"
        href = "n/a" if row.get("theoretical_h_ref") in {None, ""} else f"{float(row['theoretical_h_ref']):.3g}"
        lines.append(f"| {row['precision']} | {href} | {w1} | {row.get('default_in_W1')} | {emp} | {row.get('interpretation')} |")
    (out / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_paper_note(out: Path, comparison: Sequence[Dict[str, object]], fit_rows: Sequence[Dict[str, object]], acc_rows: Sequence[Dict[str, object]]) -> None:
    lines = ["# Paper Window Result Summary", ""]
    for row in comparison:
        prec = row["precision"]
        default_in = row.get("default_in_W1")
        w1 = "no tau=1 certificate"
        if finite_float(row.get("theoretical_W1_low")) is not None:
            w1 = f"[{float(row['theoretical_W1_low']):.3g}, {float(row['theoretical_W1_high']):.3g}]"
        emp = "unavailable"
        if row.get("empirical_good_low") not in {None, ""}:
            emp = f"[{float(row['empirical_good_low']):.3g}, {float(row['empirical_good_high']):.3g}]"
        lines.extend([
            f"## {prec}",
            "",
            f"- Default h=1e-3 inside W1: {default_in}.",
            f"- Theoretical W1: {w1}.",
            f"- Empirical accuracy good set: {emp}.",
            f"- Interpretation: {row.get('interpretation')}.",
            "- Wording: treat this as a guardrail certificate/probe diagnostic, not an accuracy optimum.",
            "",
        ])
    (out / "paper_window_result_summary.md").write_text("\n".join(lines), encoding="utf-8")


def regenerate_from_raw(args: argparse.Namespace) -> None:
    out = Path(args.output_dir).resolve()
    raw_path = out / "raw_probe_metrics.csv"
    if not raw_path.exists():
        raise FileNotFoundError(f"--reuse_raw_metrics requested but {raw_path} does not exist")
    h_override = [float(x) for x in args.h_grid.split(",")] if args.h_grid else None
    specs = make_specs(args.precisions, h_override)
    raw_rows = read_csv_rows(raw_path)
    checkpoint_manifest = read_csv_rows(out / "checkpoint_manifest.csv")

    summary_rows: List[Dict[str, object]] = []
    fit_rows: List[Dict[str, object]] = []
    for spec in specs:
        rows, _ = summarize_by_h(raw_rows, "task_start_seed16_deterministic", spec.name)
        summary_rows.extend(rows)
        hs = np.array([float(r["h"]) for r in rows], dtype=np.float64)
        rho = np.array([float(r["rho_raw"]) for r in rows], dtype=np.float64)
        for method in [PRIMARY_FIT_METHOD, "log_soft_l1", "linear_nnls"]:
            fit = fit_rho(hs, rho, method)
            fit_rows.extend(window_rows_for_fit(spec.name, "task_start_seed16_deterministic", fit))

    summary_fields = [
        "precision", "checkpoint_id", "h", "n_directions", "G", "G2", "d", "V_dir_formula", "V_dir_sample",
        "scalar_nmse", "directional_corr", "sign_agreement", "V_h_raw", "rho_raw", "rho_raw_std_or_bootstrap_ci",
        "d_h_mean", "d_h_std", "d_star_mean", "d_star_std", "probe_active_frac", "probe_alignment",
        "probe_norm_ratio", "saturation_frac",
    ]
    fit_fields = [
        "precision", "checkpoint_id", "fit_method", "A", "B", "C", "h_ref", "rho_min_dep", "tau",
        "h_low", "h_high", "default_h", "default_in_window", "rho_dep_at_default",
        "fit_quality_r2_or_log_error", "fit_stability_flag", "notes",
    ]
    acc_points = accuracy_sweep_points()
    acc_rows = accuracy_good_sets(acc_points)
    comparison_rows = make_comparison_rows(fit_rows, acc_rows)

    write_csv(out / "probe_summary_by_h.csv", summary_rows, summary_fields)
    write_csv(out / "fitted_windows.csv", fit_rows, fit_fields)
    write_csv(out / "empirical_accuracy_good_sets.csv", acc_rows, [
        "precision", "threshold_type", "max_acc", "h_best_acc", "h_good_low", "h_good_high",
        "all_good_h_values", "source_path",
    ])
    write_csv(out / "accuracy_sweep_points.csv", acc_points, [
        "precision", "h", "best_eval_acc", "source_path",
    ])
    write_csv(out / "comparison_table_for_paper.csv", comparison_rows, [
        "precision", "checkpoint_id", "theoretical_h_ref", "theoretical_W1_low", "theoretical_W1_high",
        "default_in_W1", "empirical_good_low", "empirical_good_high", "h_best_acc", "interpretation",
    ])
    plot_outputs(out, summary_rows, fit_rows, acc_rows, acc_points)
    write_readme(out, args, specs, checkpoint_manifest, comparison_rows)
    write_paper_note(out, comparison_rows, fit_rows, acc_rows)
    print(f"Regenerated summaries and figures from {raw_path}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="")
    parser.add_argument("--precisions", nargs="+", default=["fp32", "fp16", "int8", "int4"])
    parser.add_argument("--model_id", default="roberta-large")
    parser.add_argument("--seed", type=int, default=16)
    parser.add_argument("--data_seed", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_batches", type=int, default=1)
    parser.add_argument("--num_directions", type=int, default=64)
    parser.add_argument("--direction_seed_base", type=int, default=730000)
    parser.add_argument("--group_size", type=int, default=128)
    parser.add_argument("--h_grid", default="", help="Optional comma-separated h grid shared by all precision modes.")
    parser.add_argument("--progress_every", type=int, default=4)
    parser.add_argument("--reuse_raw_metrics", action="store_true", help="Regenerate summaries/figures from existing raw_probe_metrics.csv without GPU probing.")
    args = parser.parse_args()
    if not args.output_dir:
        args.output_dir = str(REPO_ROOT / f"roberta_sst5_theoretical_windows_{datetime.now().strftime('%Y%m%d')}")
    if args.reuse_raw_metrics:
        regenerate_from_raw(args)
    else:
        run_probe(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
