#!/usr/bin/env python
"""Offline h-window estimator for precision-aware ZO perturbation probes.

This script is intentionally analysis-only. It reads existing probe summaries
and per-direction JSONL files, then evaluates geometry, self-consistency, and
hybrid h-window estimators without launching training or submitting jobs.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import socket
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - report generation handles this.
    plt = None


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "window_estimator"
EPS = 1e-12

REQUIRED_RESULT_COLUMNS = [
    "setting",
    "precision",
    "quantizer",
    "direction_family",
    "sparse_p",
    "h",
    "h_active",
    "alignment",
    "norm_ratio",
    "code_change_frac",
    "clip_frac",
    "saturation_frac",
    "richardson_relerr",
    "corr_fd_true",
    "nMSE_fd_true",
    "geometry_visible",
    "fd_local",
    "valid",
]

EXTRA_RESULT_COLUMNS = [
    "source_kind",
    "source_path",
    "probe_rows",
    "finite_rate",
    "active_frac",
    "zero_effective_displacement_frac",
    "fd_zero_ratio",
    "sign_agreement",
    "fd_mean",
    "fd_std",
    "d_true_mean",
    "d_true_std",
    "loss_plus_mean",
    "loss_minus_mean",
    "loss_diff_abs_median",
    "richardson_pair_h",
    "richardson_pair_type",
    "richardson_n",
    "richardson_iqr",
    "fd_local_available",
    "loss_snr_visible",
    "visibility_defect",
    "boundary_penalty",
    "score",
    "failure_mode",
    "notes",
]

RESULT_COLUMNS = REQUIRED_RESULT_COLUMNS + EXTRA_RESULT_COLUMNS


@dataclass(frozen=True)
class Thresholds:
    tau_align: float = 0.70
    tau_rho_low: float = 0.70
    tau_rho_high: float = 1.50
    tau_code: float = 1e-2
    tau_active: float = 1e-2
    tau_richardson: float = 0.30
    loss_snr_k: float = 5.0
    lambda_v: float = 1.0
    lambda_l: float = 1.0
    lambda_s: float = 0.25


def relpath(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve()))
    except Exception:
        return str(path)


def as_float(value) -> float:
    if value is None:
        return math.nan
    try:
        if pd.isna(value):
            return math.nan
    except Exception:
        pass
    if isinstance(value, str):
        value = value.strip()
        if not value or value.lower() in {"nan", "none", "null"}:
            return math.nan
    try:
        return float(value)
    except Exception:
        return math.nan


def finite(value) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except Exception:
        return False


def finite_array(values: Iterable) -> np.ndarray:
    arr = np.array([as_float(v) for v in values], dtype=np.float64)
    return arr[np.isfinite(arr)]


def safe_mean(values: Iterable) -> float:
    arr = finite_array(values)
    return float(arr.mean()) if arr.size else math.nan


def safe_median(values: Iterable) -> float:
    arr = finite_array(values)
    return float(np.median(arr)) if arr.size else math.nan


def safe_std(values: Iterable) -> float:
    arr = finite_array(values)
    return float(arr.std(ddof=1)) if arr.size > 1 else (0.0 if arr.size == 1 else math.nan)


def safe_iqr(values: Iterable) -> float:
    arr = finite_array(values)
    if not arr.size:
        return math.nan
    q75, q25 = np.percentile(arr, [75, 25])
    return float(q75 - q25)


def safe_corr(x_values: Iterable, y_values: Iterable) -> float:
    x = np.array([as_float(v) for v in x_values], dtype=np.float64)
    y = np.array([as_float(v) for v in y_values], dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 2 or np.std(x) <= 0 or np.std(y) <= 0:
        return math.nan
    return float(np.corrcoef(x, y)[0, 1])


def safe_nmse(fd_values: Iterable, true_values: Iterable) -> float:
    fd = np.array([as_float(v) for v in fd_values], dtype=np.float64)
    true = np.array([as_float(v) for v in true_values], dtype=np.float64)
    mask = np.isfinite(fd) & np.isfinite(true)
    fd = fd[mask]
    true = true[mask]
    if not fd.size:
        return math.nan
    return float(np.mean((fd - true) ** 2) / (np.mean(true**2) + EPS))


def json_default(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer, np.int64)):
        return int(value)
    if isinstance(value, (np.floating, np.float64)):
        value = float(value)
        return None if math.isnan(value) else value
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, float) and math.isnan(value):
        return None
    return str(value)


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(REPO_ROOT),
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return ""


def read_jsonl(path: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def first_present(row: pd.Series, names: Sequence[str]) -> float:
    for name in names:
        if name in row.index and finite(row[name]):
            return as_float(row[name])
    return math.nan


def first_present_obj(row: pd.Series, names: Sequence[str]):
    for name in names:
        if name in row.index and not pd.isna(row[name]):
            return row[name]
    return None


def h_key(h: float) -> int:
    return int(round(float(h) * 1_000_000_000_000))


def same_h(a: float, b: float, rtol: float = 1e-6) -> bool:
    if not (finite(a) and finite(b)):
        return False
    return abs(a - b) <= rtol * max(abs(a), abs(b), 1.0)


def find_reference_h(h: float, available: Sequence[float]) -> Tuple[float, str]:
    candidates = [x for x in available if finite(x) and x < h * (1 - 1e-9)]
    for divisor, label in ((2.0, "h_over_2"), (3.0, "h_over_3")):
        target = h / divisor
        for x in candidates:
            if same_h(x, target):
                return x, label
    if not candidates:
        return math.nan, ""
    # Smoothness fallback: keep it lower-scale and close enough to be useful.
    valid_candidates = []
    for x in candidates:
        ratio = x / h
        if 0.25 <= ratio <= 0.75:
            valid_candidates.append((abs(math.log(h) - math.log(x)), x, ratio))
    if valid_candidates:
        _, x, ratio = min(valid_candidates)
        return x, f"nearest_lower_ratio_{ratio:.3g}"
    return math.nan, ""


def match_key_from_row(row: Dict[str, object], key_fields: Sequence[str]) -> str:
    if key_fields:
        return "|".join(str(row.get(k, "")) for k in key_fields)
    for names in (
        ("batch_index", "direction_index", "seed"),
        ("batch_index", "direction_index"),
        ("direction_index", "seed"),
        ("k_dir",),
    ):
        if all(k in row for k in names):
            return "|".join(str(row.get(k, "")) for k in names)
    return str(row.get("_row_index", ""))


def aggregate_directional_records(
    records: List[Dict[str, object]],
    meta: Dict[str, object],
    h_field: str,
    fd_fields: Sequence[str],
    true_fields: Sequence[str],
    key_fields: Sequence[str],
) -> pd.DataFrame:
    if not records:
        return pd.DataFrame()

    enriched: List[Dict[str, object]] = []
    for i, row in enumerate(records):
        r = dict(row)
        r["_row_index"] = i
        h = as_float(r.get(h_field))
        if not finite(h):
            continue
        r["_h"] = h
        r["_h_active"] = as_float(r.get("h_active", h))
        fd = math.nan
        for field in fd_fields:
            if finite(r.get(field)):
                fd = as_float(r.get(field))
                break
        r["_fd"] = fd
        true = math.nan
        for field in true_fields:
            if finite(r.get(field)):
                true = as_float(r.get(field))
                break
        r["_d_true"] = true
        r["_match_key"] = match_key_from_row(r, key_fields)
        enriched.append(r)

    if not enriched:
        return pd.DataFrame()

    df = pd.DataFrame(enriched)
    hs = sorted(df["_h"].dropna().unique())
    richardson: Dict[int, Dict[str, object]] = {}
    by_h_key: Dict[int, Dict[str, float]] = {}
    for h in hs:
        sub = df[np.isclose(df["_h"], h)]
        by_h_key[h_key(h)] = {
            str(row["_match_key"]): as_float(row["_fd"])
            for _, row in sub.iterrows()
            if finite(row.get("_fd"))
        }

    for h in hs:
        ref_h, pair_type = find_reference_h(float(h), hs)
        if not finite(ref_h):
            continue
        current = by_h_key.get(h_key(float(h)), {})
        ref = by_h_key.get(h_key(float(ref_h)), {})
        common = sorted(set(current).intersection(ref))
        relerrs = []
        for key in common:
            d_h = current[key]
            d_ref = ref[key]
            if finite(d_h) and finite(d_ref):
                relerrs.append(abs(d_h - d_ref) / max(abs(d_ref), EPS))
        if relerrs:
            richardson[h_key(float(h))] = {
                "richardson_relerr": safe_median(relerrs),
                "richardson_iqr": safe_iqr(relerrs),
                "richardson_pair_h": float(ref_h),
                "richardson_pair_type": pair_type,
                "richardson_n": len(relerrs),
            }

    rows: List[Dict[str, object]] = []
    for h in hs:
        sub = df[np.isclose(df["_h"], h)]
        first = sub.iloc[0]
        fd = sub["_fd"].to_numpy(dtype=np.float64)
        d_true = sub["_d_true"].to_numpy(dtype=np.float64)
        fd_finite = np.isfinite(fd)
        dtrue_finite = np.isfinite(d_true)
        finite_rate = float(fd_finite.mean()) if fd.size and np.isfinite(fd).any() else math.nan

        active_frac = safe_mean(first_present(row, ["probe_active_frac", "active_frac"]) for _, row in sub.iterrows())
        code_change = safe_mean(first_present(row, ["code_change_frac", "probe_active_frac", "active_frac"]) for _, row in sub.iterrows())
        alignment = safe_mean(first_present(row, ["probe_alignment", "alignment"]) for _, row in sub.iterrows())
        norm_ratio = safe_mean(first_present(row, ["probe_norm_ratio", "norm_ratio"]) for _, row in sub.iterrows())
        clip_frac = safe_mean(
            max(
                first_present(row, ["clip_frac", "clip_frac_w"]),
                first_present(row, ["clip_frac_w_plus"]),
                first_present(row, ["clip_frac_w_minus"]),
            )
            for _, row in sub.iterrows()
        )
        saturation_frac = safe_mean(
            max(
                first_present(row, ["saturation_frac", "saturation_frac_w"]),
                first_present(row, ["saturation_frac_w_plus"]),
                first_present(row, ["saturation_frac_w_minus"]),
            )
            for _, row in sub.iterrows()
        )
        zero_eff = safe_mean(
            first_present(row, ["zero_effective_displacement_frac", "fd_is_zero"])
            for _, row in sub.iterrows()
        )
        if "fd_is_zero" in sub.columns:
            fd_zero = float(np.mean([bool(x) for x in sub["fd_is_zero"].fillna(False).to_list()]))
        else:
            fd_zero = math.nan
        if "sign_match" in sub.columns:
            sign_agreement = safe_mean([1.0 if bool(x) else 0.0 for x in sub["sign_match"].dropna().to_list()])
        else:
            sign_agreement = math.nan

        row = {
            **meta,
            "h": float(h),
            "h_active": safe_median(sub["_h_active"]),
            "alignment": alignment,
            "norm_ratio": norm_ratio,
            "code_change_frac": code_change,
            "clip_frac": clip_frac,
            "saturation_frac": saturation_frac,
            "corr_fd_true": safe_corr(fd, d_true) if dtrue_finite.any() else math.nan,
            "nMSE_fd_true": safe_nmse(fd, d_true) if dtrue_finite.any() else math.nan,
            "probe_rows": int(len(sub)),
            "finite_rate": finite_rate,
            "active_frac": active_frac,
            "zero_effective_displacement_frac": zero_eff,
            "fd_zero_ratio": fd_zero,
            "sign_agreement": sign_agreement,
            "fd_mean": safe_mean(fd),
            "fd_std": safe_std(fd),
            "d_true_mean": safe_mean(d_true),
            "d_true_std": safe_std(d_true),
            "loss_plus_mean": safe_mean(sub["loss_plus"]) if "loss_plus" in sub.columns else math.nan,
            "loss_minus_mean": safe_mean(sub["loss_minus"]) if "loss_minus" in sub.columns else math.nan,
            "loss_diff_abs_median": safe_median(
                abs(as_float(row.get("loss_plus")) - as_float(row.get("loss_minus")))
                for _, row in sub.iterrows()
            )
            if {"loss_plus", "loss_minus"}.issubset(sub.columns)
            else math.nan,
        }
        source_paths = sorted(set(str(x) for x in sub.get("_source_path", pd.Series(dtype=str)).dropna().to_list()))
        if source_paths:
            row["source_path"] = ";".join(source_paths[:4])
            if len(source_paths) > 4:
                row["source_path"] += f";...(+{len(source_paths)-4})"
        row.update(richardson.get(h_key(float(h)), {}))
        rows.append(row)

    return pd.DataFrame(rows)


def load_current_fp32_fp16() -> Tuple[pd.DataFrame, List[str]]:
    base = REPO_ROOT / "experiments" / "main_latest" / "mezo" / "roberta-large" / "sst5" / "fp32_fp16_h_sweep_11h_seed16_bs64_ckpt1k_20260517"
    warnings: List[str] = []
    frames = []
    for precision in ("fp32", "fp16"):
        paths = sorted(
            (base / precision).glob("h_sweep_11h_checkpointed/results/*/seed16/checkpoint_probe_stats.jsonl")
        )
        records: List[Dict[str, object]] = []
        for path in paths:
            for row in read_jsonl(path):
                row["_source_path"] = relpath(path)
                records.append(row)
        if not records:
            warnings.append(f"missing current {precision} checkpoint_probe_stats.jsonl under {relpath(base)}")
            continue
        meta = {
            "setting": "roberta_sst5_current_dense_probe_ckpt1k",
            "precision": precision,
            "quantizer": "identity" if precision == "fp32" else "fp16_forward_oracle",
            "direction_family": "dense",
            "sparse_p": math.nan,
            "source_kind": "current_fp32_fp16_checkpoint_probe_jsonl",
            "source_path": relpath(base / precision),
            "notes": "RoBERTa-large/SST-5 seed16 bs64 checkpoint probe; no training launched by this analysis.",
        }
        frames.append(
            aggregate_directional_records(
                records=records,
                meta=meta,
                h_field="h_raw",
                fd_fields=("d_fd", "fd"),
                true_fields=("d_true",),
                key_fields=("batch_index", "direction_index", "seed"),
            )
        )
    return (pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(), warnings)


def load_legacy_dense_sparse_summaries() -> Tuple[pd.DataFrame, List[str]]:
    root = REPO_ROOT / "experiments" / "int8_update_sparse_plan" / "probe_window_h100_20260512"
    paths = [root / "dense_probe_summary.csv", root / "sparse_probe_summary.csv"]
    rows: List[Dict[str, object]] = []
    warnings: List[str] = []
    for path in paths:
        if not path.exists():
            warnings.append(f"missing legacy probe summary {relpath(path)}")
            continue
        df = pd.read_csv(path)
        for _, r in df.iterrows():
            precision = str(r.get("precision_mode", "")).lower()
            if precision not in {"fp32", "fp16", "int8", "int4"}:
                continue
            direction = str(r.get("direction_type", "dense")).lower()
            sparse_p = as_float(r.get("sparse_rate"))
            if direction == "dense":
                sparse_p = math.nan
            quantizer = {
                "fp32": "identity",
                "fp16": "fp16_forward_oracle",
                "int8": "legacy_int8_fp16master_probe",
                "int4": "legacy_int4_probe",
            }.get(precision, precision)
            rows.append(
                {
                    "setting": "legacy_probe_window_h100_20260512",
                    "precision": precision,
                    "quantizer": quantizer,
                    "direction_family": direction,
                    "sparse_p": sparse_p,
                    "h": as_float(r.get("h_raw", r.get("h"))),
                    "h_active": as_float(r.get("h_active", r.get("h_raw", r.get("h")))),
                    "alignment": as_float(r.get("probe_alignment_mean")),
                    "norm_ratio": as_float(r.get("probe_norm_ratio_mean")),
                    "code_change_frac": as_float(r.get("probe_active_frac_mean")),
                    "clip_frac": math.nan,
                    "saturation_frac": math.nan,
                    "richardson_relerr": math.nan,
                    "corr_fd_true": as_float(r.get("corr_fd_true")),
                    "nMSE_fd_true": as_float(r.get("nMSE_fd_true")),
                    "source_kind": "legacy_dense_sparse_probe_summary_csv",
                    "source_path": relpath(path),
                    "probe_rows": as_float(r.get("num_probe_rows")),
                    "finite_rate": math.nan,
                    "active_frac": as_float(r.get("probe_active_frac_mean")),
                    "fd_zero_ratio": as_float(r.get("fd_zero_ratio")),
                    "sign_agreement": as_float(r.get("sign_agreement")),
                    "fd_mean": as_float(r.get("fd_mean")),
                    "fd_std": as_float(r.get("fd_std")),
                    "d_true_mean": as_float(r.get("d_true_mean")),
                    "d_true_std": as_float(r.get("d_true_std")),
                    "notes": "Historical summary used for calibration/evaluation; no Richardson pairs in this summary file.",
                }
            )
    return pd.DataFrame(rows), warnings


def load_groupwise256_int8_jsonl() -> Tuple[pd.DataFrame, List[str]]:
    root = (
        REPO_ROOT
        / "experiments"
        / "main_latest"
        / "roberta-large"
        / "sst5"
        / "groupwise_int8_block256_window_continuation_seed16_20260517"
    )
    specs = [
        (root / "02_dense_probe_window" / "probe_stats.jsonl", "dense", math.nan),
        (root / "04_sparse_probe_by_rate" / "bernoulli_p0p01" / "probe_stats.jsonl", "sparse", 0.01),
        (root / "04_sparse_probe_by_rate" / "bernoulli_p0p003" / "probe_stats.jsonl", "sparse", 0.003),
    ]
    frames = []
    warnings: List[str] = []
    for path, direction, p in specs:
        records = read_jsonl(path)
        if not records:
            warnings.append(f"missing historical groupwise INT8 probe JSONL {relpath(path)}")
            continue
        for row in records:
            row["_source_path"] = relpath(path)
        meta = {
            "setting": "historical_groupwise256_int8_probe",
            "precision": "int8",
            "quantizer": "groupwise_int8_block256_historical",
            "direction_family": direction,
            "sparse_p": p,
            "source_kind": "historical_groupwise256_probe_stats_jsonl",
            "source_path": relpath(path),
            "notes": "Historical groupwise256 INT8 reference, not current G128 RTNClip.",
        }
        frames.append(
            aggregate_directional_records(
                records=records,
                meta=meta,
                h_field="h_raw",
                fd_fields=("d_fd", "fd"),
                true_fields=("d_true",),
                key_fields=("batch_index", "direction_index", "seed"),
            )
        )
    return (pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(), warnings)


def h_from_run_name(name: str) -> float:
    match = re.search(r"_h([^_]+)_seed", name)
    if not match:
        return math.nan
    label = match.group(1).replace("p", ".")
    try:
        return float(label)
    except Exception:
        return math.nan


def load_rtnclip_int8_geometry() -> Tuple[pd.DataFrame, List[str]]:
    root = REPO_ROOT / "outputs" / "rtnclip_lowbit_roberta_sst5_seed16_20260519_batch"
    hsearch = root / "int8_hsearch_summary.csv"
    rows: List[Dict[str, object]] = []
    warnings: List[str] = []
    if hsearch.exists():
        summary = pd.read_csv(hsearch)
        for _, sr in summary.iterrows():
            run_dir = Path(str(sr.get("run_dir", "")))
            diag = run_dir / "perturbation_diagnostics.jsonl"
            if not diag.exists():
                continue
            h = as_float(sr.get("h"))
            drows = read_jsonl(diag)
            if not drows:
                continue
            rows.append(
                {
                    "setting": "rtnclip_g128_int8_training_diagnostics_geometry_only",
                    "precision": "int8",
                    "quantizer": "G128_groupwise_RTNClip_fake_quant",
                    "direction_family": "dense",
                    "sparse_p": math.nan,
                    "h": h,
                    "h_active": h,
                    "alignment": safe_median(r.get("alignment") for r in drows),
                    "norm_ratio": safe_median(r.get("norm_ratio") for r in drows),
                    "code_change_frac": safe_median(r.get("code_change_frac", r.get("active_frac")) for r in drows),
                    "clip_frac": safe_median(
                        max(as_float(r.get("clip_frac_w_plus")), as_float(r.get("clip_frac_w_minus")))
                        for r in drows
                    ),
                    "saturation_frac": safe_median(
                        max(as_float(r.get("saturation_frac_w_plus")), as_float(r.get("saturation_frac_w_minus")))
                        for r in drows
                    ),
                    "richardson_relerr": math.nan,
                    "corr_fd_true": as_float(sr.get("corr_fd_true")),
                    "nMSE_fd_true": as_float(sr.get("nMSE_fd_true")),
                    "source_kind": "current_rtnclip_int8_perturbation_diagnostics_jsonl",
                    "source_path": relpath(diag),
                    "probe_rows": len(drows),
                    "active_frac": safe_median(r.get("active_frac") for r in drows),
                    "zero_effective_displacement_frac": safe_median(
                        r.get("zero_effective_displacement_frac") for r in drows
                    ),
                    "finite_rate": math.nan,
                    "notes": "Existing training-path perturbation diagnostics; geometry only, no fixed-batch Richardson pairs.",
                }
            )
    else:
        warnings.append(f"missing RTNClip INT8 hsearch summary {relpath(hsearch)}")

    smoke = root / "smoke_summary.csv"
    if smoke.exists():
        sdf = pd.read_csv(smoke)
        sdf = sdf[(sdf.get("bitwidth") == 8) & (sdf.get("scale_refresh_k") == 1)]
        for _, sr in sdf.iterrows():
            h = as_float(sr.get("h"))
            rows.append(
                {
                    "setting": "rtnclip_g128_int8_smoke_geometry_only",
                    "precision": "int8",
                    "quantizer": "G128_groupwise_RTNClip_fake_quant",
                    "direction_family": "dense",
                    "sparse_p": math.nan,
                    "h": h,
                    "h_active": h,
                    "alignment": as_float(sr.get("alignment")),
                    "norm_ratio": as_float(sr.get("norm_ratio")),
                    "code_change_frac": as_float(sr.get("active_frac")),
                    "clip_frac": math.nan,
                    "saturation_frac": as_float(sr.get("saturation_frac_w")),
                    "richardson_relerr": math.nan,
                    "corr_fd_true": math.nan,
                    "nMSE_fd_true": math.nan,
                    "source_kind": "current_rtnclip_int8_smoke_summary_csv",
                    "source_path": relpath(smoke),
                    "probe_rows": as_float(sr.get("steps_completed")),
                    "active_frac": as_float(sr.get("active_frac")),
                    "finite_rate": as_float(sr.get("d_h_finite_rate")),
                    "zero_effective_displacement_frac": as_float(sr.get("zero_effective_displacement_frac")),
                    "notes": "50-step smoke summary at h=1e-3; geometry only.",
                }
            )
    else:
        warnings.append(f"missing RTNClip smoke summary {relpath(smoke)}")

    if not rows:
        warnings.append("no current RTNClip INT8 geometry rows found")
    return pd.DataFrame(rows), warnings


def load_rtnclip_int4_probe() -> Tuple[pd.DataFrame, List[str]]:
    path = (
        REPO_ROOT
        / "outputs"
        / "rtnclip_lowbit_roberta_sst5_seed16_20260519_batch"
        / "int4_probe"
        / "probe_stats.jsonl"
    )
    warnings: List[str] = []
    records = read_jsonl(path)
    if not records:
        warnings.append(f"missing current RTNClip INT4 probe JSONL {relpath(path)}")
        return pd.DataFrame(), warnings
    for row in records:
        row["_source_path"] = relpath(path)
        row["h_active"] = row.get("h")
    meta = {
        "setting": "rtnclip_g128_int4_current_probe",
        "precision": "int4",
        "quantizer": "G128_groupwise_RTNClip_fake_quant",
        "direction_family": "dense",
        "sparse_p": math.nan,
        "source_kind": "current_rtnclip_int4_probe_stats_jsonl",
        "source_path": relpath(path),
        "notes": "Current G128 RTNClip INT4 fixed-batch/fixed-direction probe; true gradient unavailable.",
    }
    return (
        aggregate_directional_records(
            records=records,
            meta=meta,
            h_field="h",
            fd_fields=("fd", "d_fd"),
            true_fields=("d_true",),
            key_fields=("k_dir",),
        ),
        warnings,
    )


def load_training_oracle() -> pd.DataFrame:
    frames = []
    fp = (
        REPO_ROOT
        / "experiments"
        / "main_latest"
        / "mezo"
        / "roberta-large"
        / "sst5"
        / "fp32_fp16_h_sweep_11h_seed16_bs64_ckpt1k_20260517"
        / "summaries"
        / "summary_all.csv"
    )
    if fp.exists():
        df = pd.read_csv(fp)
        df = df[df["precision_mode"].isin(["fp32", "fp16"])].copy()
        df["setting_hint"] = "roberta_sst5_current_dense_probe_ckpt1k"
        df["direction_family"] = "dense"
        df["sparse_p"] = math.nan
        frames.append(df.rename(columns={"precision_mode": "precision"}))

    legacy = (
        REPO_ROOT
        / "experiments"
        / "int8_update_sparse_plan"
        / "probe_window_h100_20260512"
        / "window_training_summary.csv"
    )
    if legacy.exists():
        df = pd.read_csv(legacy)
        df = df[df["precision_mode"].isin(["fp32", "fp16", "int8", "int4"])].copy()
        df["setting_hint"] = "legacy_probe_window_h100_20260512"
        frames.append(
            df.rename(
                columns={
                    "precision_mode": "precision",
                    "direction_type": "direction_family",
                    "best_acc": "best_eval_acc",
                }
            )
        )

    current_int8 = REPO_ROOT / "outputs" / "rtnclip_lowbit_roberta_sst5_seed16_20260519_batch" / "int8_hsearch_summary.csv"
    if current_int8.exists():
        df = pd.read_csv(current_int8)
        if "best_eval_acc" in df.columns:
            df = df[df["best_eval_acc"].notna()].copy()
            df["precision"] = "int8"
            df["direction_family"] = "dense"
            df["sparse_p"] = math.nan
            df["setting_hint"] = "rtnclip_g128_int8_training_diagnostics_geometry_only"
            frames.append(df)

    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def visibility_defect(row: pd.Series, th: Thresholds) -> float:
    defect = 0.0
    align = as_float(row.get("alignment"))
    if finite(align):
        defect += max(0.0, th.tau_align - align) / max(th.tau_align, EPS)
    rho = as_float(row.get("norm_ratio"))
    if finite(rho):
        if rho < th.tau_rho_low:
            defect += (th.tau_rho_low - rho) / max(th.tau_rho_low, EPS)
        elif rho > th.tau_rho_high:
            defect += (rho - th.tau_rho_high) / max(th.tau_rho_high, EPS)
    code = as_float(row.get("code_change_frac"))
    active = as_float(row.get("active_frac"))
    code_metric = code if finite(code) else active
    if finite(code_metric):
        defect += max(0.0, th.tau_code - code_metric) / max(th.tau_code, EPS)
    if finite(active):
        defect += max(0.0, th.tau_active - active) / max(th.tau_active, EPS)
    return float(defect)


def boundary_penalty(row: pd.Series) -> float:
    vals = [as_float(row.get("clip_frac")), as_float(row.get("saturation_frac"))]
    vals = [v for v in vals if finite(v)]
    if not vals:
        return 0.0
    return float(max(0.0, max(vals) - 0.05) / 0.05)


def geometry_visible(row: pd.Series, th: Thresholds) -> bool:
    align = as_float(row.get("alignment"))
    rho = as_float(row.get("norm_ratio"))
    code = as_float(row.get("code_change_frac"))
    active = as_float(row.get("active_frac"))
    code_metric = code if finite(code) else active
    checks = []
    if finite(align):
        checks.append(align >= th.tau_align)
    if finite(rho):
        checks.append(th.tau_rho_low <= rho <= th.tau_rho_high)
    if finite(code_metric):
        checks.append(code_metric >= th.tau_code)
    if finite(active):
        checks.append(active >= th.tau_active)
    if not checks:
        return False
    return bool(all(checks))


def apply_estimators(df: pd.DataFrame, th: Thresholds) -> pd.DataFrame:
    out = df.copy()
    if out.empty:
        return out
    for col in RESULT_COLUMNS:
        if col not in out.columns:
            out[col] = math.nan
    geom = []
    fd_local = []
    fd_available = []
    valid = []
    vdefs = []
    bps = []
    scores = []
    failures = []
    for _, row in out.iterrows():
        g = geometry_visible(row, th)
        rel = as_float(row.get("richardson_relerr"))
        favail = finite(rel)
        flocal = bool(favail and rel <= th.tau_richardson)
        vdef = visibility_defect(row, th)
        bp = boundary_penalty(row)
        score = th.lambda_v * vdef + th.lambda_l * (rel if favail else 1.0) + th.lambda_s * bp
        if g and flocal:
            failure = "ok"
        elif not g:
            failure = "too_small_visibility"
        elif favail:
            failure = "too_large_locality"
        else:
            failure = "locality_unavailable"
        geom.append(g)
        fd_local.append(flocal)
        fd_available.append(favail)
        valid.append(bool(g and flocal))
        vdefs.append(vdef)
        bps.append(bp)
        scores.append(float(score))
        failures.append(failure)
    out["geometry_visible"] = geom
    out["fd_local"] = fd_local
    out["fd_local_available"] = fd_available
    out["valid"] = valid
    out["visibility_defect"] = vdefs
    out["boundary_penalty"] = bps
    out["score"] = scores
    out["failure_mode"] = failures
    return out


def h_scale_col(group: pd.DataFrame) -> str:
    direction = str(group["direction_family"].iloc[0])
    return "h_active" if direction == "sparse" else "h"


def format_h(value) -> str:
    v = as_float(value)
    if not finite(v):
        return "NA"
    return f"{v:.4g}"


def interval_text(values: Sequence[float]) -> str:
    vals = [as_float(v) for v in values if finite(v)]
    if not vals:
        return "none"
    return f"[{min(vals):.4g}, {max(vals):.4g}]"


def select_windows(df: pd.DataFrame, th: Thresholds) -> pd.DataFrame:
    rows = []
    if df.empty:
        return pd.DataFrame()
    group_cols = ["setting", "precision", "quantizer", "direction_family", "sparse_p"]
    for key, group in df.groupby(group_cols, dropna=False):
        group = group.sort_values(["h_active" if str(key[3]) == "sparse" else "h", "h"])
        scale_col = h_scale_col(group)
        geom_group = group[group["geometry_visible"] == True]
        local_group = group[group["fd_local"] == True]
        valid_group = group[group["valid"] == True]
        h_vis = as_float(geom_group[scale_col].min()) if not geom_group.empty else math.nan
        h_loc = as_float(local_group[scale_col].max()) if not local_group.empty else math.nan
        common = {
            "setting": key[0],
            "precision": key[1],
            "quantizer": key[2],
            "direction_family": key[3],
            "sparse_p": key[4],
            "h_vis_min": h_vis,
            "h_loc_max": h_loc,
            "valid_window": interval_text(valid_group[scale_col].to_list()),
            "default_h_valid": bool(((group["h"] - 1e-3).abs() < 1e-12).any() and group.loc[(group["h"] - 1e-3).abs() < 1e-12, "valid"].any()),
            "geometry_rows": int(geom_group.shape[0]),
            "locality_rows": int(local_group.shape[0]),
            "valid_rows": int(valid_group.shape[0]),
        }

        def add_policy(policy: str, row: Optional[pd.Series], status: str, notes: str = "") -> None:
            selected_h = math.nan if row is None else as_float(row.get("h"))
            selected_h_active = math.nan if row is None else as_float(row.get("h_active"))
            rows.append(
                {
                    **common,
                    "policy": policy,
                    "selected_h": selected_h,
                    "selected_h_active": selected_h_active,
                    "selection_status": status,
                    "alignment": math.nan if row is None else as_float(row.get("alignment")),
                    "norm_ratio": math.nan if row is None else as_float(row.get("norm_ratio")),
                    "code_change_frac": math.nan if row is None else as_float(row.get("code_change_frac")),
                    "richardson_relerr": math.nan if row is None else as_float(row.get("richardson_relerr")),
                    "corr_fd_true": math.nan if row is None else as_float(row.get("corr_fd_true")),
                    "nMSE_fd_true": math.nan if row is None else as_float(row.get("nMSE_fd_true")),
                    "score": math.nan if row is None else as_float(row.get("score")),
                    "failure_mode": "" if row is None else str(row.get("failure_mode", "")),
                    "notes": notes,
                }
            )

        add_policy(
            "geometry_lower_bound",
            geom_group.sort_values(scale_col).iloc[0] if not geom_group.empty else None,
            "selected" if not geom_group.empty else "no_geometry_visible",
            "Estimator 1/2 lower-bound only; ignores locality.",
        )
        add_policy(
            "richardson_upper_bound",
            local_group.sort_values(scale_col).iloc[-1] if not local_group.empty else None,
            "selected" if not local_group.empty else "no_fd_local",
            "Estimator 3 upper-bound only; ignores quantization visibility.",
        )
        add_policy(
            "smallest_valid",
            valid_group.sort_values(scale_col).iloc[0] if not valid_group.empty else None,
            "selected" if not valid_group.empty else "no_valid_window",
        )
        if not valid_group.empty:
            hmin = as_float(valid_group[scale_col].min())
            hmax = as_float(valid_group[scale_col].max())
            midpoint = math.sqrt(hmin * hmax)
            vg = valid_group.copy()
            vg["_mid_dist"] = (np.log(vg[scale_col].astype(float)) - math.log(midpoint)).abs()
            add_policy("log_midpoint_valid", vg.sort_values(["_mid_dist", scale_col]).iloc[0], "selected")
            add_policy("score_min_valid", valid_group.sort_values(["score", scale_col]).iloc[0], "selected")
        else:
            add_policy("log_midpoint_valid", None, "no_valid_window")
            add_policy("score_min_valid", None, "no_valid_window")

        if group["corr_fd_true"].notna().any():
            add_policy(
                "probe_best_corr_fd_true",
                group.sort_values(["corr_fd_true", scale_col], ascending=[False, True]).iloc[0],
                "retrospective_only",
                "Calibration/evaluation only; not deployable.",
            )
        if group["nMSE_fd_true"].notna().any():
            add_policy(
                "probe_best_nMSE_fd_true",
                group.sort_values(["nMSE_fd_true", scale_col], ascending=[True, True]).iloc[0],
                "retrospective_only",
                "Calibration/evaluation only; not deployable.",
            )
    return pd.DataFrame(rows)


def threshold_sensitivity(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if df.empty:
        return pd.DataFrame()
    grids = {
        "tau_align": [0.7, 0.8, 0.9, 0.95],
        "tau_rho_low": [0.5, 0.7, 0.8],
        "tau_rho_high": [1.2, 1.5, 2.0],
        "tau_code": [1e-4, 1e-3, 1e-2, 5e-2],
        "tau_richardson": [0.1, 0.2, 0.3, 0.5],
    }
    group_cols = ["setting", "precision", "quantizer", "direction_family", "sparse_p"]
    for tau_align in grids["tau_align"]:
        for tau_rho_low in grids["tau_rho_low"]:
            for tau_rho_high in grids["tau_rho_high"]:
                if tau_rho_low >= tau_rho_high:
                    continue
                for tau_code in grids["tau_code"]:
                    for tau_richardson in grids["tau_richardson"]:
                        th = Thresholds(
                            tau_align=tau_align,
                            tau_rho_low=tau_rho_low,
                            tau_rho_high=tau_rho_high,
                            tau_code=tau_code,
                            tau_active=tau_code,
                            tau_richardson=tau_richardson,
                        )
                        cur = apply_estimators(df, th)
                        for key, group in cur.groupby(group_cols, dropna=False):
                            scale_col = h_scale_col(group)
                            valid_group = group[group["valid"] == True]
                            geom_group = group[group["geometry_visible"] == True]
                            local_group = group[group["fd_local"] == True]
                            rows.append(
                                {
                                    "setting": key[0],
                                    "precision": key[1],
                                    "quantizer": key[2],
                                    "direction_family": key[3],
                                    "sparse_p": key[4],
                                    "tau_align": tau_align,
                                    "tau_rho_low": tau_rho_low,
                                    "tau_rho_high": tau_rho_high,
                                    "tau_code": tau_code,
                                    "tau_richardson": tau_richardson,
                                    "h_vis_min": as_float(geom_group[scale_col].min()) if not geom_group.empty else math.nan,
                                    "h_loc_max": as_float(local_group[scale_col].max()) if not local_group.empty else math.nan,
                                    "valid_window": interval_text(valid_group[scale_col].to_list()),
                                    "smallest_valid": as_float(valid_group.sort_values(scale_col).iloc[0][scale_col])
                                    if not valid_group.empty
                                    else math.nan,
                                    "valid_rows": int(valid_group.shape[0]),
                                }
                            )
    return pd.DataFrame(rows)


def annotate_training_oracle(selected: pd.DataFrame, training: pd.DataFrame) -> pd.DataFrame:
    out = selected.copy()
    if out.empty:
        return out
    for col in ["training_best_h", "training_best_h_active", "training_best_acc", "factor_to_training_best"]:
        out[col] = math.nan
    if training.empty:
        out["training_notes"] = "training summary unavailable"
        return out
    notes = []
    for idx, row in out.iterrows():
        candidates = training[training["precision"].astype(str).str.lower() == str(row["precision"]).lower()]
        if "direction_family" in candidates.columns:
            candidates = candidates[candidates["direction_family"].astype(str).str.lower() == str(row["direction_family"]).lower()]
        if finite(row.get("sparse_p")) and "sparse_rate" in candidates.columns:
            candidates = candidates[np.isclose(candidates["sparse_rate"].astype(float), as_float(row.get("sparse_p")), atol=1e-9)]
        if candidates.empty or "best_eval_acc" not in candidates.columns:
            notes.append("training oracle unavailable")
            continue
        candidates = candidates[candidates["best_eval_acc"].notna()]
        if candidates.empty:
            notes.append("training oracle unavailable")
            continue
        best = candidates.sort_values("best_eval_acc", ascending=False).iloc[0]
        out.loc[idx, "training_best_h"] = as_float(best.get("h"))
        out.loc[idx, "training_best_h_active"] = as_float(best.get("h_active", best.get("h")))
        out.loc[idx, "training_best_acc"] = as_float(best.get("best_eval_acc"))
        sel = as_float(row.get("selected_h_active" if str(row.get("direction_family")) == "sparse" else "selected_h"))
        oracle = as_float(best.get("h_active", best.get("h"))) if str(row.get("direction_family")) == "sparse" else as_float(best.get("h"))
        if finite(sel) and finite(oracle) and sel > 0 and oracle > 0:
            out.loc[idx, "factor_to_training_best"] = max(sel / oracle, oracle / sel)
        notes.append("training oracle retrospective only")
    out["training_notes"] = notes
    return out


def md_table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    def fmt(x) -> str:
        if isinstance(x, float):
            return format_h(x)
        if x is None:
            return "NA"
        text = str(x)
        return text.replace("\n", " ")

    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(fmt(x) for x in row) + " |")
    return "\n".join(lines)


def make_methods_md(path: Path, th: Thresholds, diagnostics: Dict[str, object]) -> None:
    text = f"""# Precision-aware h-window estimator methods

This package is offline analysis only. It reads existing probe outputs and does
not launch training, submit jobs, or change quantizer semantics.

## Estimator 1: quantization-geometry lower bound

For low-bit G128 RTNClip probes, the lower-bound question is whether `h u` is
visible through the shared quantization grid built from the unperturbed FP16
master `w_t`. The analysis uses existing `active_frac` / `code_change_frac`,
effective displacement alignment, and norm-ratio diagnostics.

Default geometry thresholds:

- `tau_align = {th.tau_align}`
- `tau_rho_low = {th.tau_rho_low}`
- `tau_rho_high = {th.tau_rho_high}`
- `tau_code = {th.tau_code}`
- `tau_active = {th.tau_active}`

The deployable lower-bound estimate is `h_vis_min`, the smallest grid point
passing these geometry checks. This does not use true gradients.

## Estimator 2: effective-displacement diagnostics

The effective displacement is

`Delta_Q(h,u) = Q_t(w_t + h u) - Q_t(w_t - h u)`.

It is compared to `Delta_ideal = 2 h u` using alignment, norm ratio, code
change fraction, clip fraction, and saturation fraction. These diagnostics
are deployable for quantized oracles because they depend on the quantizer and
directions, not on true gradients.

## Estimator 3: Richardson/self-consistency locality

For each matched direction, the analysis compares

`d_h = [L(w + h u) - L(w - h u)] / (2h)`

with a smaller-scale estimate. Exact `h/2` pairs are preferred; when the
existing grid does not contain `h/2`, the script records an `h/3` or nearest
lower-scale smoothness pair in `richardson_pair_type`.

`richardson_relerr(h) = |d_h - d_ref| / max(|d_ref|, eps)`.

Default locality threshold:

- `tau_richardson = {th.tau_richardson}`

This estimator does not require true gradients. Rows without matched
per-direction finite differences are marked `locality_unavailable` and are not
treated as valid by the hybrid estimator.

## Estimator 4: true-direction calibration

When existing probe files contain `d_true = grad(w)^T u`, the analysis reports
`corr_fd_true` and `nMSE_fd_true`. These are retrospective calibration metrics
only and are not used to define the deployable valid window.

## Estimator 5: loss-SNR floor baseline

The requested repeated-base-loss baseline requires repeated evaluations of
`L(w_t)` on the same batch. No complete repeated-base-loss artifact was found
in the inspected files, so `loss_snr_visible` is left unavailable. The summary
lists the missing artifact and a probe-only command template.

## Estimator 6: hybrid precision-aware window

The main deployable rule is:

`valid(h) = geometry_visible(h) AND fd_local(h)`.

The script reports:

- `h_vis_min`: smallest h passing geometry visibility.
- `h_loc_max`: largest h passing Richardson/self-consistency.
- `valid_window`: grid interval satisfying both.
- `smallest_valid`: smallest valid h.
- `log_midpoint_valid`: grid h closest to the geometric midpoint of the valid interval.
- `score_min_valid`: valid h minimizing a weighted defect score.

For sparse directions, raw `h` and active-coordinate `h_active = h / sqrt(p)`
are both reported; selection and interval comparisons use `h_active`.

## Data-source notes

{json.dumps(diagnostics.get("source_summary", {}), indent=2, sort_keys=True, default=json_default)}
"""
    path.write_text(text, encoding="utf-8")


def build_summary_md(results: pd.DataFrame, selected: pd.DataFrame, sensitivity: pd.DataFrame, diagnostics: Dict[str, object]) -> str:
    lines = ["# h-window estimator summary", ""]
    th = diagnostics["default_thresholds"]
    lines.append(
        f"Default thresholds: align >= {th['tau_align']}, norm_ratio in [{th['tau_rho_low']}, {th['tau_rho_high']}], "
        f"code/active >= {th['tau_code']}, Richardson <= {th['tau_richardson']}."
    )
    lines.append("")

    keep_policies = ["geometry_lower_bound", "richardson_upper_bound", "smallest_valid", "log_midpoint_valid", "score_min_valid", "probe_best_corr_fd_true"]
    table = selected[selected["policy"].isin(keep_policies)].copy()
    rows = []
    for _, r in table.sort_values(["precision", "direction_family", "sparse_p", "setting", "policy"]).iterrows():
        h_show = r["selected_h_active"] if str(r["direction_family"]) == "sparse" else r["selected_h"]
        rows.append(
            [
                r["precision"],
                r["direction_family"],
                "" if not finite(r["sparse_p"]) else r["sparse_p"],
                r["quantizer"],
                r["policy"],
                h_show,
                r["h_vis_min"],
                r["h_loc_max"],
                r["valid_window"],
                r["selection_status"],
            ]
        )
    lines.append("## Selected h per method")
    lines.append("")
    lines.append(
        md_table(
            ["Precision", "Direction", "p", "Quantizer", "Policy", "Selected h-scale", "h_vis_min", "h_loc_max", "Valid window", "Status"],
            rows,
        )
    )
    lines.append("")

    rows = []
    for key, g in results.groupby(["setting", "precision", "quantizer", "direction_family", "sparse_p"], dropna=False):
        scale_col = h_scale_col(g)
        default_rows = g[(g["h"] - 1e-3).abs() < 1e-12]
        default_status = "missing"
        if not default_rows.empty:
            dr = default_rows.iloc[0]
            default_status = "valid" if bool(dr["valid"]) else str(dr["failure_mode"])
        probe_best_corr = math.nan
        if g["corr_fd_true"].notna().any():
            probe_best_corr = as_float(g.sort_values("corr_fd_true", ascending=False).iloc[0][scale_col])
        rows.append(
            [
                key[1],
                key[3],
                "" if not finite(key[4]) else key[4],
                key[2],
                interval_text(g[g["geometry_visible"] == True][scale_col].to_list()),
                interval_text(g[g["fd_local"] == True][scale_col].to_list()),
                interval_text(g[g["valid"] == True][scale_col].to_list()),
                default_status,
                probe_best_corr,
            ]
        )
    lines.append("## Windows by precision")
    lines.append("")
    lines.append(
        md_table(
            ["Precision", "Direction", "p", "Quantizer", "Geometry-visible", "FD-local", "Hybrid valid", "h=1e-3", "Probe-best corr h"],
            rows,
        )
    )
    lines.append("")

    lines.append("## Interpretation")
    lines.append("")
    lines.extend(make_interpretation(results, selected, sensitivity, diagnostics))
    lines.append("")
    lines.append("## Threshold sensitivity")
    lines.append("")
    if sensitivity.empty:
        lines.append("Threshold sensitivity was not computed.")
    else:
        rows = []
        focus = [
            ("roberta_sst5_current_dense_probe_ckpt1k", "fp32", "dense", math.nan),
            ("roberta_sst5_current_dense_probe_ckpt1k", "fp16", "dense", math.nan),
            ("historical_groupwise256_int8_probe", "int8", "dense", math.nan),
            ("historical_groupwise256_int8_probe", "int8", "sparse", 0.01),
            ("historical_groupwise256_int8_probe", "int8", "sparse", 0.003),
            ("rtnclip_g128_int4_current_probe", "int4", "dense", math.nan),
        ]
        for setting, precision, direction, sparse_p in focus:
            sub = sensitivity[
                (sensitivity["setting"] == setting)
                & (sensitivity["precision"] == precision)
                & (sensitivity["direction_family"] == direction)
            ]
            if finite(sparse_p):
                sub = sub[np.isclose(sub["sparse_p"].astype(float), sparse_p, atol=1e-12)]
            else:
                sub = sub[sub["sparse_p"].isna()]
            if sub.empty:
                continue
            top = sub["valid_window"].value_counts().head(3)
            rows.append(
                [
                    precision,
                    direction if not finite(sparse_p) else f"{direction} p={sparse_p:g}",
                    setting,
                    int((sub["valid_rows"] > 0).sum()),
                    int(sub.shape[0]),
                    "; ".join(f"{idx}: {cnt}" for idx, cnt in top.items()),
                ]
            )
        lines.append(md_table(["Precision", "Direction", "Setting", "Nonempty combos", "Total combos", "Most common windows"], rows))
        lines.append("")
        lines.append("The final recommendation uses the default threshold set above; the grid shows where a conclusion depends on relaxing locality or visibility.")
    lines.append("")
    lines.append("## Missing artifacts and probe-only commands")
    lines.append("")
    missing = diagnostics.get("missing_artifacts", [])
    if missing:
        for item in missing:
            lines.append(f"- {item}")
    else:
        lines.append("- No required artifact gaps were detected for the loaded offline analysis.")
    lines.append("")
    lines.append("Suggested probe-only extensions, if the missing G128 RTNClip grids are needed:")
    lines.append("")
    lines.append("```bash")
    lines.append("# INT8 G128 RTNClip fixed-batch/fixed-direction h-grid with Richardson pairs")
    lines.append("CUDA_VISIBLE_DEVICES=0 DATALOADER_SHUFFLE=True python tools/rtnclip_roberta_sst5_batch.py --output_root outputs/rtnclip_lowbit_roberta_sst5_seed16_20260519_batch --bitwidth 8 --probe_dirs 16 probe-int4")
    lines.append("")
    lines.append("# INT4 G128 RTNClip fixed-batch/fixed-direction h-grid, using the same generic probe-only path")
    lines.append("CUDA_VISIBLE_DEVICES=0 DATALOADER_SHUFFLE=True python tools/rtnclip_roberta_sst5_batch.py --output_root outputs/rtnclip_lowbit_roberta_sst5_seed16_20260519_batch --bitwidth 4 --probe_dirs 16 probe-int4")
    lines.append("")
    lines.append("# Sparse G128 RTNClip probe-only extension for p in {0.01, 0.003}; exact CLI may need adding to the smoke/probe harness")
    lines.append("CUDA_VISIBLE_DEVICES=0 DATALOADER_SHUFFLE=True python tools/window_estimation/estimate_h_window.py --suggest-sparse-rtnclip-probe")
    lines.append("```")
    return "\n".join(lines) + "\n"


def make_interpretation(results: pd.DataFrame, selected: pd.DataFrame, sensitivity: pd.DataFrame, diagnostics: Dict[str, object]) -> List[str]:
    lines: List[str] = []

    def group_has(precision: str, quantizer_contains: str) -> pd.DataFrame:
        return results[
            (results["precision"] == precision)
            & (results["quantizer"].astype(str).str.contains(quantizer_contains, regex=False))
        ]

    int8_current = group_has("int8", "G128_groupwise_RTNClip")
    int4_current = group_has("int4", "G128_groupwise_RTNClip")
    legacy_int8 = results[(results["precision"] == "int8") & (results["setting"].astype(str).str.contains("legacy"))]

    if not int8_current.empty:
        visible = int8_current[int8_current["geometry_visible"] == True]
        lines.append(
            f"1. Geometry-only diagnostics can estimate the visibility lower bound when effective displacement data exists. "
            f"Current G128 INT8 geometry rows first pass at h={format_h(visible['h'].min()) if not visible.empty else 'none'}; "
            "locality is unavailable because the current INT8 G128 artifact lacks fixed-direction h/h2 finite-difference pairs."
        )
    else:
        lines.append("1. Current G128 INT8 geometry artifacts were not found; historical INT8 probes are used only as calibration references.")

    if not int4_current.empty:
        vg = int4_current[int4_current["valid"] == True]
        gg = int4_current[int4_current["geometry_visible"] == True]
        lg = int4_current[int4_current["fd_local"] == True]
        lines.append(
            f"2. Current G128 INT4 visibility starts around h={format_h(gg['h'].min()) if not gg.empty else 'none'} under default thresholds, "
            f"while Richardson locality passes at h={interval_text(lg['h'].to_list())}. "
            f"The hybrid intersection is {interval_text(vg['h'].to_list())}, so INT4 shows the expected visibility/locality collision risk."
        )

    fp = results[results["precision"].isin(["fp32", "fp16"])]
    if not fp.empty:
        valid_by_precision = []
        for precision, g in fp.groupby("precision"):
            valid_by_precision.append(f"{precision}: {interval_text(g[g['valid'] == True]['h'].to_list())}")
        lines.append(
            "3. For FP32/FP16, geometry is essentially always visible on the loaded grid; the useful upper side is controlled by "
            f"self-consistency/locality. Hybrid windows: {', '.join(valid_by_precision)}."
        )

    if not legacy_int8.empty:
        dense = legacy_int8[legacy_int8["direction_family"] == "dense"]
        if not dense.empty:
            corr_best = dense.sort_values("corr_fd_true", ascending=False).iloc[0]
            geom = dense[dense["geometry_visible"] == True]
            lines.append(
                f"4. Historical dense INT8 calibration supports a quantization lower bound near h={format_h(geom['h'].min()) if not geom.empty else 'none'} "
                f"and probe-best corr near h={format_h(corr_best['h'])}; this explains why h=1e-3 can be good but is not universal."
            )

    sparse = results[(results["direction_family"] == "sparse") & (results["sparse_p"].notna())]
    if not sparse.empty:
        rows = []
        for p, g in sparse.groupby("sparse_p"):
            rows.append(f"p={p:g}: raw {interval_text(g[g['geometry_visible'] == True]['h'].to_list())}, active {interval_text(g[g['geometry_visible'] == True]['h_active'].to_list())}")
        lines.append(
            "5. Sparse probes are reported in both raw h and active-coordinate h_active. The active-coordinate windows are more comparable across p: "
            + "; ".join(rows)
            + "."
        )

    lines.append(
        "6. Between smallest_valid and log_midpoint_valid, log_midpoint_valid is the less brittle policy when the valid interval spans several grid points; "
        "smallest_valid is useful as a conservative lower-cost choice but can sit on the visibility boundary for INT8/INT4."
    )
    lines.append(
        "7. Minimal deployable diagnostics are effective displacement geometry plus matched-direction finite differences at h and a smaller reference h. "
        "True gradients and training accuracy are useful for validation, not for selecting h."
    )
    lines.append(
        "8. The loss-SNR baseline could not be evaluated from existing artifacts because repeated same-batch base-loss evaluations were not present."
    )
    return lines


def make_paper_table(results: pd.DataFrame, selected: pd.DataFrame) -> str:
    rows = []
    def include_for_paper(setting: str, precision: str, quantizer: str) -> bool:
        if setting == "roberta_sst5_current_dense_probe_ckpt1k":
            return precision in {"fp32", "fp16"}
        if setting == "historical_groupwise256_int8_probe":
            return True
        if setting in {"rtnclip_g128_int4_current_probe", "rtnclip_g128_int8_training_diagnostics_geometry_only"}:
            return True
        return False

    def direction_label(setting: str, direction: str, sparse_p: float, quantizer: str) -> str:
        parts = [direction]
        if finite(sparse_p):
            parts.append(f"p={sparse_p:g}")
        if "G128_groupwise_RTNClip" in quantizer:
            parts.append("G128 RTNClip")
        elif "groupwise_int8_block256" in quantizer:
            parts.append("hist groupwise256")
        return " ".join(parts)

    for key, g in results.groupby(["setting", "precision", "direction_family", "sparse_p", "quantizer"], dropna=False):
        setting = str(key[0])
        precision = str(key[1])
        direction_family = str(key[2])
        sparse_p = as_float(key[3])
        quantizer = str(key[4])
        if not include_for_paper(setting, precision, quantizer):
            continue
        scale_col = h_scale_col(g)
        sel = selected[
            (selected["setting"] == setting)
            & (selected["precision"] == precision)
            & (selected["direction_family"] == direction_family)
            & (selected["quantizer"] == quantizer)
            & (selected["policy"] == "log_midpoint_valid")
        ]
        if finite(sparse_p):
            sel = sel[np.isclose(sel["sparse_p"].astype(float), sparse_p, atol=1e-12)]
        selected_h = math.nan if sel.empty else as_float(sel.iloc[0]["selected_h_active" if direction_family == "sparse" else "selected_h"])
        default_rows = g[(g["h"] - 1e-3).abs() < 1e-12]
        default_valid = "missing"
        if not default_rows.empty:
            default_valid = "yes" if bool(default_rows.iloc[0]["valid"]) else "no"
        small_fail = "yes" if (g.sort_values(scale_col).head(1)["geometry_visible"] == False).any() else "no"
        large_fail = "yes" if (g.sort_values(scale_col).tail(1)["fd_local"] == False).any() else "no"
        direction = direction_label(setting, direction_family, sparse_p, quantizer)
        rows.append(
            [
                precision,
                direction,
                interval_text(g[g["valid"] == True][scale_col].to_list()),
                selected_h,
                default_valid,
                small_fail,
                large_fail,
            ]
        )
    return (
        "# Recommended paper table\n\n"
        + md_table(
            [
                "Precision",
                "Direction",
                "Estimated window",
                "Selected h",
                "Default h valid?",
                "Failure at small h",
                "Failure at large h",
            ],
            rows,
        )
        + "\n"
    )


def svg_escape(text: object) -> str:
    return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def svg_palette(i: int) -> str:
    colors = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ]
    return colors[i % len(colors)]


def write_svg_line_plot(
    path: Path,
    series: Sequence[Tuple[str, Sequence[float], Sequence[float]]],
    title: str,
    y_label: str,
    y_log: bool = False,
) -> None:
    width, height = 900, 560
    left, right, top, bottom = 86, 240, 46, 72
    plot_w = width - left - right
    plot_h = height - top - bottom
    clean = []
    for label, xs, ys in series:
        pairs = []
        for x, y in zip(xs, ys):
            x = as_float(x)
            y = as_float(y)
            if finite(x) and finite(y) and x > 0 and (not y_log or y > 0):
                pairs.append((x, y))
        if pairs:
            clean.append((label, pairs))
    if not clean:
        return
    all_x = [x for _, pairs in clean for x, _ in pairs]
    all_y = [y for _, pairs in clean for _, y in pairs]
    lx_min, lx_max = math.log10(min(all_x)), math.log10(max(all_x))
    if abs(lx_max - lx_min) < 1e-12:
        lx_min -= 1
        lx_max += 1
    if y_log:
        ly_min, ly_max = math.log10(min(all_y)), math.log10(max(all_y))
    else:
        ly_min, ly_max = min(all_y), max(all_y)
    if abs(ly_max - ly_min) < 1e-12:
        ly_min -= 1
        ly_max += 1
    y_pad = 0.08 * (ly_max - ly_min)
    ly_min -= y_pad
    ly_max += y_pad

    def sx(x: float) -> float:
        return left + (math.log10(x) - lx_min) / (lx_max - lx_min) * plot_w

    def sy(y: float) -> float:
        yy = math.log10(y) if y_log else y
        return top + (ly_max - yy) / (ly_max - ly_min) * plot_h

    elems = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2:.1f}" y="24" text-anchor="middle" font-family="Arial" font-size="18">{svg_escape(title)}</text>',
        f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="none" stroke="#222" stroke-width="1"/>',
        f'<text x="{left + plot_w/2:.1f}" y="{height-22}" text-anchor="middle" font-family="Arial" font-size="13">h_active for sparse, raw h for dense (log scale)</text>',
        f'<text transform="translate(20,{top + plot_h/2:.1f}) rotate(-90)" text-anchor="middle" font-family="Arial" font-size="13">{svg_escape(y_label)}</text>',
    ]
    for frac in np.linspace(0, 1, 6):
        x = left + frac * plot_w
        y = top + frac * plot_h
        elems.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top+plot_h}" stroke="#ddd" stroke-width="1"/>')
        elems.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left+plot_w}" y2="{y:.1f}" stroke="#eee" stroke-width="1"/>')
    for i in range(6):
        lx = lx_min + (lx_max - lx_min) * i / 5
        val = 10**lx
        x = left + plot_w * i / 5
        elems.append(f'<text x="{x:.1f}" y="{top+plot_h+20}" text-anchor="middle" font-family="Arial" font-size="10">{val:.1e}</text>')
        ly = ly_min + (ly_max - ly_min) * i / 5
        val_y = 10**ly if y_log else ly
        y = top + plot_h - plot_h * i / 5
        elems.append(f'<text x="{left-8}" y="{y+4:.1f}" text-anchor="end" font-family="Arial" font-size="10">{val_y:.2g}</text>')

    for idx, (label, pairs) in enumerate(clean):
        color = svg_palette(idx)
        points = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in sorted(pairs))
        elems.append(f'<polyline fill="none" stroke="{color}" stroke-width="2" points="{points}"/>')
        for x, y in pairs:
            elems.append(f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="3" fill="{color}"/>')
        legend_y = top + 18 + idx * 18
        elems.append(f'<line x1="{left+plot_w+22}" y1="{legend_y-4}" x2="{left+plot_w+42}" y2="{legend_y-4}" stroke="{color}" stroke-width="2"/>')
        elems.append(f'<text x="{left+plot_w+48}" y="{legend_y}" font-family="Arial" font-size="11">{svg_escape(label[:44])}</text>')
    elems.append("</svg>")
    path.write_text("\n".join(elems) + "\n", encoding="utf-8")


def make_svg_plots(results: pd.DataFrame, selected: pd.DataFrame, plot_dir: Path) -> List[str]:
    plot_dir.mkdir(parents=True, exist_ok=True)
    warnings = ["matplotlib unavailable; generated SVG fallback plots"]
    group_cols = ["setting", "precision", "quantizer", "direction_family", "sparse_p"]
    metrics = [
        ("alignment", "Alignment vs h", False),
        ("norm_ratio", "Norm ratio vs h", True),
        ("code_change_frac", "Code change / active fraction vs h", False),
        ("richardson_relerr", "Richardson/self-consistency relative error vs h", True),
        ("corr_fd_true", "corr_fd_true vs h", False),
        ("nMSE_fd_true", "nMSE_fd_true vs h", True),
    ]
    for metric, title, y_log in metrics:
        series = []
        for key, group in results.groupby(group_cols, dropna=False):
            y = pd.to_numeric(group[metric], errors="coerce")
            if not y.notna().any():
                continue
            xcol = h_scale_col(group)
            x = pd.to_numeric(group[xcol], errors="coerce")
            label = f"{key[1]} {key[3]}"
            if finite(key[4]):
                label += f" p={key[4]:g}"
            label += f" {key[2]}"
            series.append((label, x.to_list(), y.to_list()))
        write_svg_line_plot(plot_dir / f"{metric}_vs_h.svg", series, title, metric, y_log=y_log)

    # Valid-window overlay as a compact SVG strip plot.
    width, height = 980, max(260, 90 + 28 * results.groupby(group_cols, dropna=False).ngroups)
    left, right, top, bottom = 92, 260, 40, 52
    plot_w = width - left - right
    all_x = [x for x in pd.to_numeric(results["h_active"].where(results["direction_family"] == "sparse", results["h"]), errors="coerce") if finite(x) and x > 0]
    if all_x:
        lx_min, lx_max = math.log10(min(all_x)), math.log10(max(all_x))
        if abs(lx_max - lx_min) < 1e-12:
            lx_min -= 1
            lx_max += 1

        def sx(x: float) -> float:
            return left + (math.log10(x) - lx_min) / (lx_max - lx_min) * plot_w

        elems = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
            '<rect width="100%" height="100%" fill="white"/>',
            f'<text x="{width/2:.1f}" y="24" text-anchor="middle" font-family="Arial" font-size="18">Valid-window overlay</text>',
        ]
        y = top + 24
        for key, group in results.groupby(group_cols, dropna=False):
            xcol = h_scale_col(group)
            label = f"{key[1]} {key[3]} {key[2]}"
            if finite(key[4]):
                label += f" p={key[4]:g}"
            elems.append(f'<text x="{left-10}" y="{y+4}" text-anchor="end" font-family="Arial" font-size="10">{svg_escape(label[:42])}</text>')
            elems.append(f'<line x1="{left}" y1="{y}" x2="{left+plot_w}" y2="{y}" stroke="#ddd"/>')
            for _, row in group.iterrows():
                x = as_float(row[xcol])
                if not finite(x) or x <= 0:
                    continue
                color = "#2ca02c" if bool(row["valid"]) else ("#ff7f0e" if bool(row["geometry_visible"]) else "#d62728")
                elems.append(f'<circle cx="{sx(x):.1f}" cy="{y}" r="5" fill="{color}"/>')
            y += 28
        elems.append(f'<text x="{left+plot_w/2:.1f}" y="{height-18}" text-anchor="middle" font-family="Arial" font-size="13">green valid, orange geometry-only, red not visible/locality unavailable</text>')
        elems.append("</svg>")
        (plot_dir / "valid_window_overlay.svg").write_text("\n".join(elems) + "\n", encoding="utf-8")

    sparse = results[results["direction_family"] == "sparse"]
    if not sparse.empty:
        series = []
        for key, group in sparse.groupby(["precision", "quantizer", "sparse_p"], dropna=False):
            label = f"{key[0]} p={key[2]:g} geometry"
            series.append((label, group["h_active"].to_list(), group["geometry_visible"].astype(int).to_list()))
            series.append((f"{key[0]} p={key[2]:g} valid", group["h_active"].to_list(), group["valid"].astype(int).to_list()))
        write_svg_line_plot(plot_dir / "dense_vs_sparse_h_active_window.svg", series, "Sparse h_active window comparison", "boolean", y_log=False)

    # Selected-vs-oracle scatter fallback as CSV-like SVG if data exists.
    probe = selected[selected["policy"].isin(["log_midpoint_valid", "probe_best_corr_fd_true"])]
    scatter = []
    for key, g in probe.groupby(["setting", "precision", "quantizer", "direction_family", "sparse_p"], dropna=False):
        vals = {}
        for _, row in g.iterrows():
            vals[row["policy"]] = as_float(row["selected_h_active" if key[3] == "sparse" else "selected_h"])
        if finite(vals.get("log_midpoint_valid")) and finite(vals.get("probe_best_corr_fd_true")):
            scatter.append((f"{key[1]} {key[3]}", [vals["probe_best_corr_fd_true"]], [vals["log_midpoint_valid"]]))
    write_svg_line_plot(plot_dir / "selected_h_vs_probe_best_scatter.svg", scatter, "Selected h vs probe-best h", "selected h", y_log=True)
    return warnings


def make_plots(results: pd.DataFrame, selected: pd.DataFrame, plot_dir: Path) -> List[str]:
    plot_dir.mkdir(parents=True, exist_ok=True)
    if plt is None:
        return make_svg_plots(results, selected, plot_dir)
    warnings: List[str] = []
    group_cols = ["setting", "precision", "quantizer", "direction_family", "sparse_p"]
    metrics = [
        ("alignment", "Alignment vs h"),
        ("norm_ratio", "Norm ratio vs h"),
        ("code_change_frac", "Code change / active fraction vs h"),
        ("richardson_relerr", "Richardson/self-consistency relative error vs h"),
        ("corr_fd_true", "corr_fd_true vs h"),
        ("nMSE_fd_true", "nMSE_fd_true vs h"),
    ]
    for metric, title in metrics:
        fig, ax = plt.subplots(figsize=(8, 5))
        any_line = False
        for key, group in results.groupby(group_cols, dropna=False):
            y = pd.to_numeric(group[metric], errors="coerce")
            if not y.notna().any():
                continue
            xcol = h_scale_col(group)
            x = pd.to_numeric(group[xcol], errors="coerce")
            label = f"{key[1]} {key[3]}"
            if finite(key[4]):
                label += f" p={key[4]:g}"
            label += f" {key[2]}"
            ax.plot(x, y, marker="o", linewidth=1.5, label=label)
            any_line = True
        if not any_line:
            plt.close(fig)
            continue
        ax.set_xscale("log")
        if metric in {"norm_ratio", "richardson_relerr", "nMSE_fd_true"}:
            ax.set_yscale("log")
        ax.set_xlabel("h_active for sparse, raw h for dense")
        ax.set_ylabel(metric)
        ax.set_title(title)
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(plot_dir / f"{metric}_vs_h.png", dpi=180)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ybase = 0
    yticks = []
    ylabels = []
    for key, group in results.groupby(group_cols, dropna=False):
        xcol = h_scale_col(group)
        group = group.sort_values(xcol)
        x = pd.to_numeric(group[xcol], errors="coerce")
        valid = group["valid"].astype(bool).to_numpy()
        geom = group["geometry_visible"].astype(bool).to_numpy()
        ax.scatter(x, np.full(len(group), ybase), c=np.where(valid, "tab:green", np.where(geom, "tab:orange", "tab:red")), s=45)
        yticks.append(ybase)
        label = f"{key[1]} {key[3]}"
        if finite(key[4]):
            label += f" p={key[4]:g}"
        ylabels.append(label[:80])
        ybase += 1
    if ybase:
        ax.set_xscale("log")
        ax.set_yticks(yticks)
        ax.set_yticklabels(ylabels, fontsize=7)
        ax.set_xlabel("h_active for sparse, raw h for dense")
        ax.set_title("Valid-window overlay: green valid, orange geometry-only, red not visible")
        ax.grid(True, axis="x", which="both", alpha=0.25)
        fig.tight_layout()
        fig.savefig(plot_dir / "valid_window_overlay.png", dpi=180)
    plt.close(fig)

    probe = selected[selected["policy"].isin(["log_midpoint_valid", "probe_best_corr_fd_true"])]
    if not probe.empty:
        pivot_rows = []
        for key, g in probe.groupby(["setting", "precision", "quantizer", "direction_family", "sparse_p"], dropna=False):
            rec = {"label": f"{key[1]} {key[3]} {key[4] if finite(key[4]) else ''}"}
            for _, row in g.iterrows():
                rec[row["policy"]] = as_float(row["selected_h_active" if key[3] == "sparse" else "selected_h"])
            if finite(rec.get("log_midpoint_valid")) and finite(rec.get("probe_best_corr_fd_true")):
                pivot_rows.append(rec)
        if pivot_rows:
            fig, ax = plt.subplots(figsize=(6, 5))
            x = [r["probe_best_corr_fd_true"] for r in pivot_rows]
            y = [r["log_midpoint_valid"] for r in pivot_rows]
            ax.scatter(x, y)
            lims = [min(x + y) * 0.7, max(x + y) * 1.3]
            ax.plot(lims, lims, "k--", linewidth=1)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel("probe-best corr h")
            ax.set_ylabel("log_midpoint_valid selected h")
            ax.set_title("Selected h vs probe-best h")
            ax.grid(True, which="both", alpha=0.25)
            fig.tight_layout()
            fig.savefig(plot_dir / "selected_h_vs_probe_best_scatter.png", dpi=180)
            plt.close(fig)

    sparse = results[results["direction_family"] == "sparse"]
    if not sparse.empty:
        fig, ax = plt.subplots(figsize=(8, 5))
        for key, g in sparse.groupby(["precision", "quantizer", "sparse_p"], dropna=False):
            g = g.sort_values("h_active")
            ax.plot(g["h_active"], g["geometry_visible"].astype(int), marker="o", label=f"{key[0]} p={key[2]:g} geometry")
            ax.plot(g["h_active"], g["valid"].astype(int) + 0.05, marker="x", linestyle="--", label=f"{key[0]} p={key[2]:g} valid")
        ax.set_xscale("log")
        ax.set_xlabel("h_active")
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["false", "true"])
        ax.set_title("Sparse h_active window comparison")
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(plot_dir / "dense_vs_sparse_h_active_window.png", dpi=180)
        plt.close(fig)
    return warnings


def collect_inputs() -> Tuple[pd.DataFrame, Dict[str, object]]:
    frames = []
    warnings: List[str] = []
    source_counts: Dict[str, int] = {}

    loaders = [
        ("current_fp32_fp16", load_current_fp32_fp16),
        ("legacy_dense_sparse", load_legacy_dense_sparse_summaries),
        ("historical_groupwise256_int8", load_groupwise256_int8_jsonl),
        ("current_rtnclip_int8_geometry", load_rtnclip_int8_geometry),
        ("current_rtnclip_int4_probe", load_rtnclip_int4_probe),
    ]
    for name, loader in loaders:
        df, w = loader()
        warnings.extend(w)
        source_counts[name] = int(df.shape[0])
        if not df.empty:
            frames.append(df)

    missing_artifacts = []
    if source_counts.get("current_rtnclip_int8_geometry", 0) < 11:
        missing_artifacts.append(
            "Current G128 RTNClip INT8 does not have a complete fixed-batch/fixed-direction h-grid with Richardson pairs; only geometry diagnostics/smoke rows were found."
        )
    missing_artifacts.append(
        "Repeated same-batch base-loss evaluations for the loss-SNR floor baseline were not found."
    )
    missing_artifacts.append(
        "Current G128 RTNClip sparse p=0.01 and p=0.003 probe-only h-grids were not found; historical groupwise256 sparse INT8 probes are included as reference only."
    )

    diagnostics = {
        "repo_root": str(REPO_ROOT),
        "git_commit": git_commit(),
        "hostname": socket.gethostname(),
        "python": sys.executable,
        "source_summary": source_counts,
        "warnings": warnings,
        "missing_artifacts": missing_artifacts,
        "inspected_docs_and_code": [
            "tools/rtnclip_roberta_sst5_batch.py",
            "tools/smoke_rtnclip_roberta_sst5.py",
            "medium_models/src/trainer.py",
            "docs/sparse_mezo_h100_experiments.md",
            "experiments/int8_update_sparse_plan/probe_window_h100_20260512/dense_probe_summary.md",
            "experiments/int8_update_sparse_plan/probe_window_h100_20260512/sparse_probe_summary.md",
        ],
        "guardrails": {
            "launched_training": False,
            "submitted_jobs": False,
            "used_gptq": False,
            "used_residual_grid": False,
            "changed_quantizer_semantics": False,
        },
    }
    return (pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame(), diagnostics)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--tau-align", type=float, default=0.70)
    parser.add_argument("--tau-rho-low", type=float, default=0.70)
    parser.add_argument("--tau-rho-high", type=float, default=1.50)
    parser.add_argument("--tau-code", type=float, default=1e-2)
    parser.add_argument("--tau-active", type=float, default=None)
    parser.add_argument("--tau-richardson", type=float, default=0.30)
    parser.add_argument("--suggest-sparse-rtnclip-probe", action="store_true", help="Print probe-only sparse extension note and exit.")
    args = parser.parse_args(argv)

    if args.suggest_sparse_rtnclip_probe:
        print("Sparse G128 RTNClip probe-only support is not present in the current harness.")
        print("Add a probe-only mode that accepts --bitwidth {8,4}, --sparse-p {0.01,0.003}, fixed batch, fixed direction seeds, and writes probe_stats.jsonl.")
        return 0

    th = Thresholds(
        tau_align=args.tau_align,
        tau_rho_low=args.tau_rho_low,
        tau_rho_high=args.tau_rho_high,
        tau_code=args.tau_code,
        tau_active=args.tau_active if args.tau_active is not None else args.tau_code,
        tau_richardson=args.tau_richardson,
    )

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = out_dir / "window_estimator_plots"

    raw, diagnostics = collect_inputs()
    diagnostics["default_thresholds"] = th.__dict__

    if raw.empty:
        (out_dir / "failure_report.txt").write_text("FAILED: no probe inputs were found.\n", encoding="utf-8")
        return 2

    results = apply_estimators(raw, th)
    for col in RESULT_COLUMNS:
        if col not in results.columns:
            results[col] = math.nan
    results = results[RESULT_COLUMNS]
    results = results.sort_values(["precision", "direction_family", "sparse_p", "setting", "quantizer", "h_active", "h"])
    results.to_csv(out_dir / "window_estimator_results.csv", index=False)

    selected = select_windows(results, th)
    selected = annotate_training_oracle(selected, load_training_oracle())
    selected.to_csv(out_dir / "window_estimator_selected.csv", index=False)

    sensitivity = threshold_sensitivity(raw)
    sensitivity.to_csv(out_dir / "threshold_sensitivity.csv", index=False)

    plot_warnings = make_plots(results, selected, plot_dir)
    diagnostics["warnings"].extend(plot_warnings)
    diagnostics["num_result_rows"] = int(results.shape[0])
    diagnostics["num_selected_rows"] = int(selected.shape[0])
    diagnostics["num_sensitivity_rows"] = int(sensitivity.shape[0])
    (out_dir / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2, sort_keys=True, default=json_default) + "\n", encoding="utf-8")

    make_methods_md(out_dir / "window_estimator_methods.md", th, diagnostics)
    (out_dir / "window_estimator_summary.md").write_text(
        build_summary_md(results, selected, sensitivity, diagnostics),
        encoding="utf-8",
    )
    (out_dir / "recommended_paper_table.md").write_text(make_paper_table(results, selected), encoding="utf-8")

    print(f"Analysis output directory: {out_dir}")
    print(f"Rows: results={results.shape[0]}, selections={selected.shape[0]}, sensitivity={sensitivity.shape[0]}")
    for precision in ("fp32", "fp16", "int8", "int4"):
        sub = selected[(selected["precision"] == precision) & (selected["policy"].isin(["smallest_valid", "log_midpoint_valid", "score_min_valid"]))]
        if sub.empty:
            continue
        print(f"\n{precision.upper()}:")
        for _, r in sub.iterrows():
            h_show = r["selected_h_active"] if str(r["direction_family"]) == "sparse" else r["selected_h"]
            print(
                f"  {r['direction_family']} {'' if not finite(r['sparse_p']) else 'p='+format_h(r['sparse_p'])} "
                f"{r['quantizer']} {r['policy']}: h={format_h(h_show)}, window={r['valid_window']}, status={r['selection_status']}"
            )
    if diagnostics["warnings"]:
        print("\nWarnings:")
        for w in diagnostics["warnings"]:
            print(f"  - {w}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
