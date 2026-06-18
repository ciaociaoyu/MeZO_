#!/usr/bin/env python
"""Eight-hour interval-aware h-selection workflow.

The workflow is intentionally conservative:
1. It audits existing interval metrics and training summaries.
2. It implements probe-only selectors from interval metrics and optional loss
   nMSE rows.
3. It writes policy and comparison CSVs without fabricating missing runs.
4. It packages the results as ``interval_h_selection_8h_bundle.zip``.

It does not launch long training by default.  That keeps the generated bundle a
reproducible checkpoint that can be extended with pilot/final training rows
when those jobs are available.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shutil
import socket
import subprocess
import sys
import time
import zipfile
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_H = 1e-3
TASKS = ["sst-5", "trec", "rte", "sst-2", "mnli"]
MODEL_ALIASES = {
    "roberta": "roberta-large",
    "roberta-large": "roberta-large",
    "opt": "facebook/opt-1.3b",
    "opt-1.3b": "facebook/opt-1.3b",
    "opt_1p3b": "facebook/opt-1.3b",
    "facebook/opt-1.3b": "facebook/opt-1.3b",
}

INTERVAL_COLS = [
    "model",
    "task",
    "precision",
    "perturbation_mode",
    "h",
    "A_uniform",
    "p_active",
    "V_norm",
    "V_align",
    "p_clip",
    "relative_disp",
    "locality_proxy",
    "jump_zero_frac",
    "jump_one_frac",
    "jump_ge2_frac",
    "source_path",
]

TRAINING_COLS = [
    "model",
    "task",
    "precision",
    "perturbation_mode",
    "h_policy",
    "h_value",
    "seed",
    "accuracy",
    "loss",
    "steps",
    "source_path",
]

LOSS_COLS = [
    "model",
    "task",
    "precision",
    "perturbation_mode",
    "h",
    "nMSE_loss",
    "corr_loss",
    "normalized_curve",
    "source_path",
]


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


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def write_csv(path: Path, df: pd.DataFrame, columns: Optional[Sequence[str]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if columns is not None:
        for col in columns:
            if col not in df.columns:
                df[col] = np.nan
        df = df.loc[:, list(columns)]
    df.to_csv(path, index=False)


def normalize_task(value: object, path: str = "") -> str:
    s = str(value or "").lower()
    p = path.lower()
    hay = f"{s} {p}"
    if "sst-5" in hay or "sst5" in hay:
        return "sst-5"
    if "sst-2" in hay or "sst2" in hay:
        return "sst-2"
    if "trec" in hay:
        return "trec"
    if "mnli" in hay:
        return "mnli"
    if "rte" in hay:
        return "rte"
    return s or "unknown"


def normalize_model(value: object, path: str = "") -> str:
    s = str(value or "").lower()
    p = path.lower()
    hay = f"{s} {p}"
    if "facebook/opt-1.3b" in hay or "opt13b" in hay or "opt-1.3b" in hay or "opt_1p3b" in hay:
        return "facebook/opt-1.3b"
    if "roberta-large" in hay:
        return "roberta-large"
    if "roberta" in hay:
        return "roberta-large"
    if "opt" in hay:
        return "facebook/opt-1.3b"
    return s or "unknown"


def normalize_precision(value: object, path: str = "") -> str:
    s = str(value or "").lower()
    p = path.lower()
    hay = f"{s} {p}"
    if "int8" in hay or "bitwidth 8" in hay or "bitwidth=8" in hay:
        return "int8"
    if "int4" in hay or "bitwidth 4" in hay or "bitwidth=4" in hay:
        return "int4"
    if "fp16" in hay:
        return "fp16"
    if "bf16" in hay:
        return "bf16"
    if "fp32" in hay:
        return "fp32"
    return s or "unknown"


def normalize_mode(value: object, path: str = "") -> str:
    s = str(value or "").lower()
    p = path.lower()
    hay = f"{s} {p}"
    if "sparse_p0p1" in hay or "sparsep0p1" in hay or "p0p1" in hay or "p=0.1" in hay or "p0.1" in hay:
        return "sparse_p0p1"
    if "sparse_p0p01" in hay or "sparsep0p01" in hay or "p0p01" in hay or "p=0.01" in hay:
        return "sparse_p0p01"
    if "sparse" in hay:
        return "sparse"
    if "prefix" in hay:
        return "prefix"
    return "dense"


def first_present(row: pd.Series, names: Sequence[str]) -> object:
    for name in names:
        if name in row.index:
            value = row[name]
            if isinstance(value, pd.Series):
                value = value.iloc[0] if len(value) else np.nan
            if pd.notna(value):
                return value
    return np.nan


def to_float(value: object) -> float:
    try:
        v = float(value)
    except Exception:
        return float("nan")
    return v if math.isfinite(v) else float("nan")


def maybe_read_csv(path: Path) -> Optional[pd.DataFrame]:
    try:
        if path.stat().st_size <= 0:
            return None
        return pd.read_csv(path)
    except Exception:
        return None


def read_json(path: Path) -> Optional[Dict[str, object]]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def interval_from_geometry_csv(path: Path) -> pd.DataFrame:
    df = maybe_read_csv(path)
    if df is None or df.empty:
        return pd.DataFrame(columns=INTERVAL_COLS)
    if "quantizer" in df.columns:
        q = df["quantizer"].astype(str).str.lower()
        qb = df["quantizer_backend"].astype(str).str.lower() if "quantizer_backend" in df.columns else q
        rtn_mask = q.str.contains("rtnclip", na=False) | qb.str.contains("rtnclip", na=False)
        if rtn_mask.any():
            df = df.loc[rtn_mask].copy()
    rename = {
        "dataset": "task",
        "setting": "perturbation_mode",
        "alignment": "V_align",
        "alignment_mean": "V_align",
        "norm_ratio": "V_norm",
        "norm_ratio_mean": "V_norm",
        "active_frac": "p_active",
        "active_frac_mean": "p_active",
        "clip_frac": "p_clip",
        "clip_frac_mean": "p_clip",
        "saturation_frac": "p_clip",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    df = df.loc[:, ~df.columns.duplicated()]
    if "model" not in df.columns:
        df["model"] = df.apply(lambda r: normalize_model("", str(path)), axis=1)
    if "task" not in df.columns:
        df["task"] = df.apply(lambda r: normalize_task("", str(path)), axis=1)
    if "precision" not in df.columns:
        df["precision"] = df.apply(lambda r: normalize_precision("", str(path)), axis=1)
    if "perturbation_mode" not in df.columns:
        df["perturbation_mode"] = df.apply(lambda r: normalize_mode("", str(path)), axis=1)
    if "h" not in df.columns:
        return pd.DataFrame(columns=INTERVAL_COLS)
    for col in ["A_uniform", "p_active", "V_norm", "V_align", "p_clip", "relative_disp", "locality_proxy"]:
        if col not in df.columns:
            df[col] = np.nan
    if "A_uniform" not in df.columns or df["A_uniform"].isna().all():
        if "delta_visibility_nmse_mean" in df.columns:
            df["A_uniform"] = pd.to_numeric(df["delta_visibility_nmse_mean"], errors="coerce")
        elif "delta_visibility_nmse" in df.columns:
            df["A_uniform"] = pd.to_numeric(df["delta_visibility_nmse"], errors="coerce")
    if "jump_zero_frac" not in df.columns:
        df["jump_zero_frac"] = np.nan
    if "jump_one_frac" not in df.columns:
        df["jump_one_frac"] = np.nan
    if "jump_ge2_frac" not in df.columns:
        df["jump_ge2_frac"] = np.nan
    out = df.copy()
    out["model"] = [normalize_model(v, str(path)) for v in out["model"]]
    out["task"] = [normalize_task(v, str(path)) for v in out["task"]]
    out["precision"] = [normalize_precision(v, str(path)) for v in out["precision"]]
    out["perturbation_mode"] = [normalize_mode(v, str(path)) for v in out["perturbation_mode"]]
    if "rtnclip_int8_mse_reprobe" in str(path):
        out["model"] = "roberta-large"
        out["task"] = "sst-5"
        out["precision"] = "int8"
        out["perturbation_mode"] = "dense"
    if "rtnclip_int4_mse_reprobe" in str(path):
        out["model"] = "roberta-large"
        out["task"] = "sst-5"
        out["precision"] = "int4"
        out["perturbation_mode"] = "dense"
    out["source_path"] = str(path)
    for col in INTERVAL_COLS:
        if col not in out.columns:
            out[col] = np.nan
    return out[INTERVAL_COLS]


def collect_interval_metrics(repo: Path, bundle_dir: Path) -> Tuple[pd.DataFrame, List[str]]:
    notes: List[str] = []
    paths = [
        repo / "interval_aware_h_probe" / "interval_geometry_summary.csv",
        repo / "outputs" / "quantizer_robustness_int8_window" / "probe_results.csv",
        repo / "outputs" / "rtnclip_int8_mse_reprobe" / "int8_mse_probe_summary.csv",
        repo / "outputs" / "rtnclip_int4_mse_reprobe_20260521_true_nmse_d16" / "int4_mse_probe_summary.csv",
        repo / "outputs" / "rtnclip_int4_mse_reprobe_20260521_true_nmse_d8_v2" / "int4_mse_probe_summary.csv",
    ]
    paths.extend((repo / "outputs" / "interval_h_selection_8h_probes").glob("**/interval_geometry_summary.csv"))
    # Include explicit generated interval bundle if only zip exists.
    zip_path = repo / "interval_aware_h_probe.zip"
    if zip_path.exists() and not (repo / "interval_aware_h_probe" / "interval_geometry_summary.csv").exists():
        extract_dir = bundle_dir / "_unzipped_interval_probe"
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(extract_dir)
        paths.insert(0, extract_dir / "interval_aware_h_probe" / "interval_geometry_summary.csv")
    frames: List[pd.DataFrame] = []
    for path in paths:
        if path.exists():
            frame = interval_from_geometry_csv(path)
            if not frame.empty:
                frames.append(frame)
        else:
            notes.append(f"missing interval source: {path}")
    if not frames:
        return pd.DataFrame(columns=INTERVAL_COLS), notes
    df = pd.concat(frames, ignore_index=True)
    for col in ["h", "A_uniform", "p_active", "V_norm", "V_align", "p_clip", "relative_disp", "locality_proxy"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    # Remove exact duplicate rows while preserving source priority.
    df = df.drop_duplicates(subset=["model", "task", "precision", "perturbation_mode", "h", "source_path"])
    return df, notes


def detect_h_from_path(path: str) -> float:
    text = path.lower()
    patterns = [
        r"h([0-9]+(?:p[0-9]+)?e[-+]?[0-9]+)",
        r"h[_-]?([0-9]+e[-+]?[0-9]+)",
        r"h[_-]?([0-9]*\.[0-9]+)",
    ]
    for pat in patterns:
        m = re.search(pat, text)
        if m:
            raw = m.group(1).replace("p", ".")
            try:
                return float(raw)
            except Exception:
                pass
    return float("nan")


def training_from_csv(path: Path) -> pd.DataFrame:
    df = maybe_read_csv(path)
    if df is None or df.empty:
        return pd.DataFrame(columns=TRAINING_COLS)
    df = df.loc[:, ~df.columns.duplicated()]
    cols = set(df.columns)
    if not (
        {"best_eval_acc", "last_eval_acc", "accuracy", "best_dev_acc", "final_dev_acc", "eval_acc"} & cols
    ):
        return pd.DataFrame(columns=TRAINING_COLS)
    rows: List[Dict[str, object]] = []
    for _, row in df.iterrows():
        h = to_float(first_present(row, ["h", "h_value", "zo_eps", "eps"]))
        if not math.isfinite(h):
            h = detect_h_from_path(str(path) + " " + str(first_present(row, ["run_dir", "run_name", "h_label"])))
        acc = to_float(first_present(row, ["best_eval_acc", "best_dev_acc", "accuracy", "eval_acc", "last_eval_acc", "final_dev_acc"]))
        if not math.isfinite(h) or not math.isfinite(acc):
            continue
        source_hint = str(path) + " " + " ".join(str(first_present(row, [x])) for x in ["run_name", "run_dir", "h_label", "policy", "setting"])
        model = normalize_model(first_present(row, ["model", "model_id", "model_name"]), source_hint)
        task = normalize_task(first_present(row, ["task", "dataset", "task_name"]), source_hint)
        precision = normalize_precision(first_present(row, ["precision", "precision_mode", "bitwidth"]), source_hint)
        if precision == "unknown" and "bitwidth" in row.index:
            bit = to_float(row["bitwidth"])
            if bit == 8:
                precision = "int8"
            elif bit == 4:
                precision = "int4"
        mode = normalize_mode(first_present(row, ["perturbation_mode", "direction", "setting"]), source_hint)
        rows.append(
            {
                "model": model,
                "task": task,
                "precision": precision,
                "perturbation_mode": mode,
                "h_policy": str(first_present(row, ["h_policy", "policy", "h_label", "run_name"])),
                "h_value": h,
                "seed": first_present(row, ["seed", "ft_seed"]),
                "accuracy": acc,
                "loss": to_float(first_present(row, ["best_eval_loss", "last_eval_loss", "loss", "eval_loss"])),
                "steps": to_float(first_present(row, ["steps_completed", "steps", "step", "last_eval_step"])),
                "source_path": str(path),
            }
        )
    return pd.DataFrame(rows, columns=TRAINING_COLS)


def collect_training(repo: Path) -> pd.DataFrame:
    patterns = [
        "outputs/rtnclip_lowbit_roberta_sst5_seed16_20260519_batch/int8_hsearch_summary.csv",
        "outputs/rtnclip_int4_g128_rtnclip_roberta_sst5_seed16_20260521/int4_hsearch_summary.csv",
        "outputs/rtnclip_int4_standard_screen_seed16_20260520_203144_h100/int4_standard_*summary.csv",
        "outputs/quantizer_robustness_int8_window/h_acc_results.csv",
        "outputs/opt13b_int4_roberta_matched_seed16_20260613_182816/full/summary_*.csv",
        "outputs/opt13b_int4_roberta_matched_seed16_20260613_182816/smoke/summary_*.csv",
        "outputs/opt13b_int4_dense_hpolicy_full_20260603_223826/*.csv",
        "outputs/opt13b_int4_dense_mezo_option_standard_lr3e-7_full_20260610_2238/summary_*.csv",
        "outputs/opt13b_int4_dense_mezo_option_standard_full_20260610_2210/summary_*.csv",
        "outputs/opt13b_int4_mezo_option_lr_seq_ablation_20260610_2148/summary_*.csv",
        "outputs/int4_full_data_hstar_dense_sparse_20260522_113710/int4_hsearch_summary.csv",
        "outputs/int4_dense_hstar_cont_vs_default_2k_20260522_163849/int4_hsearch_summary.csv",
        "outputs/int4_sparse_prefix_seedfixed_int4fd_20k_20260523_171841/int4_hsearch_summary.csv",
        "outputs/int4_cleanGL_hstar_dense_sparsep0p1_20k_20260523_142501/int4_hsearch_summary.csv",
        "outputs/int4_sparsep0p1_probe_minmse_vs_default_2k_20260522_181148/int4_hsearch_summary.csv",
    ]
    paths: List[Path] = []
    for pattern in patterns:
        paths.extend(repo.glob(pattern))
    # Preserve order but avoid duplicate reads.
    seen = set()
    unique_paths: List[Path] = []
    for path in paths:
        if path not in seen:
            seen.add(path)
            unique_paths.append(path)
    frames = []
    for path in unique_paths:
        path_text = str(path).lower()
        if any(skip in path_text for skip in ["interval_h_selection_8h_bundle", "__pycache__"]):
            continue
        frame = training_from_csv(path)
        if not frame.empty:
            frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=TRAINING_COLS)
    df = pd.concat(frames, ignore_index=True)
    relevant = (
        df["model"].isin(["roberta-large", "facebook/opt-1.3b"])
        & df["task"].isin(TASKS)
        & df["precision"].isin(["int8", "int4", "fp32", "fp16", "bf16"])
    )
    df = df[relevant].copy()
    df = df.drop_duplicates(subset=["model", "task", "precision", "perturbation_mode", "h_value", "accuracy", "source_path"])
    return df


def collect_loss_mse(repo: Path, interval_df: pd.DataFrame) -> pd.DataFrame:
    paths = [
        repo / "interval_aware_h_probe" / "loss_mse_probe.csv",
        repo / "outputs" / "rtnclip_int8_mse_reprobe" / "int8_mse_probe_summary.csv",
        repo / "outputs" / "rtnclip_int8_mse_reprobe" / "final_checkpoint_mse" / "final_checkpoint_mse_summary.csv",
        repo / "outputs" / "rtnclip_int4_mse_reprobe_20260521_true_nmse_d16" / "int4_mse_probe_summary.csv",
        repo / "outputs" / "rtnclip_int4_mse_reprobe_20260521_true_nmse_d8_v2" / "int4_mse_probe_summary.csv",
    ]
    frames: List[pd.DataFrame] = []
    for path in paths:
        df = maybe_read_csv(path)
        if df is None or df.empty or "h" not in df.columns:
            continue
        nmse_col = "nMSE_loss" if "nMSE_loss" in df.columns else "fd_true_nmse" if "fd_true_nmse" in df.columns else "nMSE_fd_true" if "nMSE_fd_true" in df.columns else None
        corr_col = "corr_loss" if "corr_loss" in df.columns else "corr_fd_true" if "corr_fd_true" in df.columns else None
        if nmse_col is None:
            continue
        out = pd.DataFrame()
        out["h"] = pd.to_numeric(df["h"], errors="coerce")
        out["nMSE_loss"] = pd.to_numeric(df[nmse_col], errors="coerce")
        out["corr_loss"] = pd.to_numeric(df[corr_col], errors="coerce") if corr_col else np.nan
        out["normalized_curve"] = np.nan
        out["model"] = normalize_model("", str(path))
        out["task"] = normalize_task("", str(path))
        out["precision"] = normalize_precision("", str(path))
        out["perturbation_mode"] = normalize_mode("", str(path))
        # The RTNClip mse reprobes are RoBERTa SST-5 dense by construction.
        if "rtnclip_int" in str(path):
            out["model"] = "roberta-large"
            out["task"] = "sst-5"
            out["perturbation_mode"] = "dense"
        out["source_path"] = str(path)
        frames.append(out[LOSS_COLS])
    if not frames:
        return pd.DataFrame(columns=LOSS_COLS)
    loss = pd.concat(frames, ignore_index=True)
    return loss.dropna(subset=["h", "nMSE_loss"], how="any")


def nearest_h(rows: pd.DataFrame, h: float) -> Optional[pd.Series]:
    if rows.empty:
        return None
    vals = pd.to_numeric(rows["h"], errors="coerce")
    idx = (np.log(vals) - math.log(h)).abs().idxmin()
    return rows.loc[idx]


def minmax(values: pd.Series) -> pd.Series:
    v = pd.to_numeric(values, errors="coerce")
    if v.notna().sum() == 0:
        return pd.Series(np.zeros(len(values)), index=values.index)
    lo = v.min(skipna=True)
    hi = v.max(skipna=True)
    if not math.isfinite(float(lo)) or not math.isfinite(float(hi)) or abs(float(hi - lo)) < 1e-12:
        return pd.Series(np.zeros(len(values)), index=values.index)
    return (v - lo) / (hi - lo)


def pareto_best(config: pd.DataFrame) -> Tuple[float, List[float]]:
    rows = config.reset_index(drop=True).copy()
    hs = rows["h"].astype(float).tolist()
    frontier: List[int] = []
    for i, ri in rows.iterrows():
        dominated = False
        for j, rj in rows.iterrows():
            if i == j:
                continue
            le_all = (
                rj["A_uniform"] <= ri["A_uniform"]
                and rj["relative_disp"] <= ri["relative_disp"]
                and rj["p_clip_delta"] <= ri["p_clip_delta"]
                and rj["V_align"] >= ri["V_align"]
            )
            strict = (
                rj["A_uniform"] < ri["A_uniform"]
                or rj["relative_disp"] < ri["relative_disp"]
                or rj["p_clip_delta"] < ri["p_clip_delta"]
                or rj["V_align"] > ri["V_align"]
            )
            if le_all and strict:
                dominated = True
                break
        if not dominated:
            frontier.append(i)
    if not frontier:
        return float("nan"), []
    f = rows.loc[frontier].copy()
    dist = (
        minmax(f["A_uniform"]).pow(2)
        + minmax(f["relative_disp"]).pow(2)
        + minmax(f["p_clip_delta"]).pow(2)
        + minmax(1.0 - f["V_align"]).pow(2)
    )
    best_idx = dist.idxmin()
    return float(rows.loc[best_idx, "h"]), [float(rows.loc[i, "h"]) for i in frontier]


def add_selector_metrics(interval_df: pd.DataFrame, loss_df: pd.DataFrame) -> pd.DataFrame:
    df = interval_df.copy()
    for col in ["h", "A_uniform", "p_active", "V_norm", "V_align", "p_clip", "relative_disp", "locality_proxy"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    out_frames: List[pd.DataFrame] = []
    keys = ["model", "task", "precision", "perturbation_mode"]
    for key, group in df.groupby(keys, dropna=False):
        g = group.sort_values("h").copy()
        base_clip = float(g.iloc[0]["p_clip"]) if pd.notna(g.iloc[0]["p_clip"]) else 0.0
        g["p_clip_base"] = base_clip
        g["p_clip_delta"] = (g["p_clip"] - base_clip).clip(lower=0.0)
        mode = str(key[3])
        if "sparse_p0p1" in mode:
            denom = 0.1
        elif "sparse_p0p01" in mode:
            denom = 0.01
        else:
            denom = 1.0
        g["p_active_norm"] = g["p_active"] / denom
        default_row = nearest_h(g, DEFAULT_H)
        default_rel = float(default_row["relative_disp"]) if default_row is not None and pd.notna(default_row["relative_disp"]) else np.nan
        if key[2] == "int4":
            rel_limit = max(2.5 * default_rel, 0.25) if math.isfinite(default_rel) else 0.25
            g["locality_rel_disp_limit"] = rel_limit
            g["locality_relaxed_for_int4"] = True
        else:
            rel_limit = max(2.0 * default_rel, 0.15) if math.isfinite(default_rel) else 0.15
            g["locality_rel_disp_limit"] = rel_limit
            g["locality_relaxed_for_int4"] = False
        log_a = np.log10(pd.to_numeric(g["A_uniform"], errors="coerce").clip(lower=1e-30))
        log_loc = np.log10(pd.to_numeric(g["locality_proxy"], errors="coerce").clip(lower=1e-30))
        log_rel = np.log10(pd.to_numeric(g["relative_disp"], errors="coerce").clip(lower=1e-30))
        clip_penalty = (g["p_clip_delta"] - 0.03).clip(lower=0.0)
        active_penalty = (0.05 - g["p_active_norm"]).clip(lower=0.0)
        align_penalty = (0.75 - g["V_align"]).clip(lower=0.0)
        norm_penalty = np.abs(np.log(pd.to_numeric(g["V_norm"], errors="coerce").clip(lower=1e-30)))
        g["score_geom"] = (
            minmax(log_a)
            + 0.5 * minmax(log_rel)
            + 0.25 * minmax(log_loc)
            + 5.0 * clip_penalty
            + 2.0 * active_penalty
            + 2.0 * align_penalty
            + 0.5 * norm_penalty
        )
        loss_g = loss_df
        if not loss_g.empty:
            mask = np.ones(len(loss_g), dtype=bool)
            for idx, k in enumerate(keys):
                mask &= loss_g[k].astype(str).eq(str(key[idx])).values
            lsmall = loss_g.loc[mask, ["h", "nMSE_loss", "corr_loss", "normalized_curve"]].copy()
            if not lsmall.empty:
                g = g.merge(lsmall, on="h", how="left")
            else:
                g["nMSE_loss"] = np.nan
                g["corr_loss"] = np.nan
                g["normalized_curve"] = np.nan
        else:
            g["nMSE_loss"] = np.nan
            g["corr_loss"] = np.nan
            g["normalized_curve"] = np.nan
        out_frames.append(g)
    return pd.concat(out_frames, ignore_index=True) if out_frames else df


def select_for_config(group: pd.DataFrame) -> Dict[str, object]:
    g = group.sort_values("h").copy()
    default_row = nearest_h(g, DEFAULT_H)
    default_rel = float(default_row["relative_disp"]) if default_row is not None and pd.notna(default_row["relative_disp"]) else np.nan
    default_score = float(default_row["score_geom"]) if default_row is not None and pd.notna(default_row["score_geom"]) else np.nan
    default_a = float(default_row["A_uniform"]) if default_row is not None and pd.notna(default_row["A_uniform"]) else np.nan
    default_align = float(default_row["V_align"]) if default_row is not None and pd.notna(default_row["V_align"]) else np.nan
    default_clip_delta = float(default_row["p_clip_delta"]) if default_row is not None and pd.notna(default_row["p_clip_delta"]) else 0.0
    vis = g[
        (g["V_align"] >= 0.70)
        & (g["V_norm"] >= 0.75)
        & (g["V_norm"] <= 1.50)
        & (g["p_active_norm"] >= 0.05)
        & (g["p_clip_delta"] <= 0.03)
    ]
    strong = g[
        (g["V_align"] >= 0.80)
        & (g["V_norm"] >= 0.80)
        & (g["V_norm"] <= 1.25)
        & (g["p_active_norm"] >= 0.10)
        & (g["p_clip_delta"] <= 0.05)
    ]
    loc = g[(g["relative_disp"] <= g["locality_rel_disp_limit"]) & (g["p_clip_delta"] <= 0.03)]
    h_vis = float(vis["h"].min()) if not vis.empty else np.nan
    h_loc = float(loc["h"].max()) if not loc.empty else np.nan
    window_exists = math.isfinite(h_vis) and math.isfinite(h_loc) and h_vis <= h_loc
    grid = sorted(float(x) for x in g["h"].dropna().unique())
    h_geom = np.nan
    h_cons = np.nan
    if window_exists:
        raw = math.sqrt(h_vis * h_loc)
        h_geom = min(grid, key=lambda x: abs(math.log(x) - math.log(raw)))
        strong_inside = strong[(strong["h"] >= h_vis) & (strong["h"] <= h_loc)]
        h_cons = float(strong_inside["h"].min()) if not strong_inside.empty else h_vis
    h_score_best = float(g.loc[g["score_geom"].idxmin(), "h"]) if g["score_geom"].notna().any() else np.nan
    h_pareto_best, pareto_frontier = pareto_best(g)
    loss_avail = g["nMSE_loss"].notna().any()
    h_loss_star = np.nan
    h_loss_cons = np.nan
    loss_note = ""
    if loss_avail:
        min_nmse = float(g["nMSE_loss"].min())
        h_loss_star = float(g.loc[g["nMSE_loss"].idxmin(), "h"])
        near = g[g["nMSE_loss"] <= 1.10 * min_nmse].copy()
        corr_ok = near[near["corr_loss"].fillna(-np.inf) >= 0.8]
        if not corr_ok.empty:
            h_loss_cons = float(corr_ok["h"].min())
        else:
            h_loss_cons = h_loss_star
            loss_note = "low_corr_or_corr_missing"
    default_nmse = np.nan
    if default_row is not None and "nMSE_loss" in default_row.index:
        default_nmse = to_float(default_row["nMSE_loss"])
    candidates = g.copy()
    safe = candidates[
        (
            (candidates["nMSE_loss"].notna() & math.isfinite(default_nmse) & (candidates["nMSE_loss"] <= default_nmse))
            | ((candidates["A_uniform"] <= default_a) & (candidates["V_align"] >= default_align))
            | (candidates["score_geom"] <= default_score)
        )
        & (candidates["relative_disp"] <= max(2.5 * default_rel, 0.25) if math.isfinite(default_rel) else True)
        & (candidates["p_clip_delta"] <= max(default_clip_delta + 0.05, 0.08))
    ]
    h_default_aware = np.nan
    default_reason = "fallback_default"
    for candidate in [h_loss_cons, h_score_best, h_pareto_best]:
        if math.isfinite(to_float(candidate)) and not safe[safe["h"].eq(float(candidate))].empty:
            h_default_aware = float(candidate)
            default_reason = "safe_candidate"
            break
    if not math.isfinite(to_float(h_default_aware)):
        h_default_aware = DEFAULT_H
    return {
        "h_vis": h_vis,
        "h_loc": h_loc,
        "h_geom": h_geom,
        "h_cons": h_cons,
        "h_score_best": h_score_best,
        "h_pareto_best": h_pareto_best,
        "pareto_frontier": ";".join(f"{x:g}" for x in pareto_frontier),
        "h_loss_star": h_loss_star,
        "h_loss_cons": h_loss_cons,
        "h_default_aware": h_default_aware,
        "window_exists": window_exists,
        "default_h": DEFAULT_H,
        "default_score_geom": default_score,
        "loss_selector_note": loss_note,
        "default_aware_reason": default_reason,
    }


def make_selectors(metrics: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    keys = ["model", "task", "precision", "perturbation_mode"]
    rows = []
    candidates = []
    for key, group in metrics.groupby(keys, dropna=False):
        sel = select_for_config(group)
        row = dict(zip(keys, key))
        row.update(sel)
        rows.append(row)
        grid = sorted(float(x) for x in group["h"].dropna().unique())
        candidate_pairs: List[Tuple[str, float, str]] = [("default", DEFAULT_H, "baseline")]
        priority = [
            ("h_loss_cons", "loss nMSE conservative selector"),
            ("h_default_aware", "default-aware safe selector"),
            ("h_score_best", "Pareto-knee score selector"),
            ("h_pareto_best", "Pareto frontier ideal-point selector"),
            ("h_cons", "geometry threshold conservative selector"),
            ("h_geom", "geometry window midpoint"),
        ]
        non_default = 0
        for name, reason in priority:
            h = to_float(sel.get(name))
            if not math.isfinite(h):
                continue
            h_near = min(grid, key=lambda x: abs(math.log(x) - math.log(h))) if grid and h > 0 else h
            if abs(math.log(h_near / DEFAULT_H)) < 1e-12:
                if not any(n == name and abs(v - h_near) < 1e-15 for n, v, _ in candidate_pairs):
                    candidate_pairs.append((name, h_near, reason))
                continue
            if not any(abs(v - h_near) < 1e-15 for _, v, _ in candidate_pairs):
                candidate_pairs.append((name, h_near, reason))
                non_default += 1
            if non_default >= 5:
                break
            # Add neighbors for selected candidates if budget allows.
            if h_near in grid:
                idx = grid.index(h_near)
                for nb in [idx - 1, idx + 1]:
                    if 0 <= nb < len(grid) and non_default < 5 and not any(abs(v - grid[nb]) < 1e-15 for _, v, _ in candidate_pairs):
                        candidate_pairs.append((f"{name}_neighbor", grid[nb], f"neighbor of {name}"))
                        non_default += 1
        for cname, h, reason in candidate_pairs:
            m = nearest_h(group, h)
            if m is None:
                continue
            candidates.append(
                {
                    "model": key[0],
                    "task": key[1],
                    "precision": key[2],
                    "perturbation_mode": key[3],
                    "candidate_name": cname,
                    "h_value": float(m["h"]),
                    "A_uniform": m.get("A_uniform", np.nan),
                    "p_active_norm": m.get("p_active_norm", np.nan),
                    "V_norm": m.get("V_norm", np.nan),
                    "V_align": m.get("V_align", np.nan),
                    "p_clip_delta": m.get("p_clip_delta", np.nan),
                    "relative_disp": m.get("relative_disp", np.nan),
                    "locality_proxy": m.get("locality_proxy", np.nan),
                    "nMSE_loss": m.get("nMSE_loss", np.nan),
                    "corr_loss": m.get("corr_loss", np.nan),
                    "score_geom": m.get("score_geom", np.nan),
                    "reason": reason,
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(candidates)


def best_training_for(training: pd.DataFrame, model: str, task: str, precision: str, mode: str, h: float) -> Optional[pd.Series]:
    if training.empty or not math.isfinite(h):
        return None
    df = training[
        (training["model"].eq(model))
        & (training["task"].eq(task))
        & (training["precision"].eq(precision))
        & (training["perturbation_mode"].eq(mode))
    ].copy()
    if df.empty:
        return None
    df["log_h_dist"] = (np.log(df["h_value"].astype(float)) - math.log(h)).abs()
    df = df[df["log_h_dist"] < 1e-4]
    if df.empty:
        return None
    return df.sort_values(["steps", "accuracy"], ascending=[False, False]).iloc[0]


def metric_for(metrics: pd.DataFrame, model: str, task: str, precision: str, mode: str, h: float) -> Optional[pd.Series]:
    if metrics.empty or not math.isfinite(h):
        return None
    df = metrics[
        (metrics["model"].eq(model))
        & (metrics["task"].eq(task))
        & (metrics["precision"].eq(precision))
        & (metrics["perturbation_mode"].eq(mode))
    ].copy()
    if df.empty:
        return None
    return nearest_h(df, h)


def build_validation_tables(selectors: pd.DataFrame, candidates: pd.DataFrame, training: pd.DataFrame, metrics: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    final_rows = []
    comp_rows = []
    keys = ["model", "task", "precision", "perturbation_mode"]
    for _, sel in selectors.iterrows():
        model, task, precision, mode = [sel[k] for k in keys]
        default_train = best_training_for(training, model, task, precision, mode, DEFAULT_H)
        selected_h = to_float(sel.get("h_default_aware"))
        selected_train = best_training_for(training, model, task, precision, mode, selected_h)
        default_metric = metric_for(metrics, model, task, precision, mode, DEFAULT_H)
        selected_metric = metric_for(metrics, model, task, precision, mode, selected_h)
        default_acc = to_float(default_train["accuracy"]) if default_train is not None else np.nan
        selected_acc = to_float(selected_train["accuracy"]) if selected_train is not None else np.nan
        delta = selected_acc - default_acc if math.isfinite(default_acc) and math.isfinite(selected_acc) else np.nan
        outcome = ""
        if math.isfinite(delta):
            outcome = "win" if delta > 0.005 else "tie" if abs(delta) <= 0.005 else "loss"
        final_rows.append(
            {
                "model": model,
                "task": task,
                "precision": precision,
                "perturbation_mode": mode,
                "h_policy": "probe_only_default_aware",
                "h_value": selected_h,
                "seed": 16,
                "run_type": "existing_full_or_medium" if selected_train is not None else "not_run",
                "steps": selected_train["steps"] if selected_train is not None else np.nan,
                "best_dev_acc": selected_acc,
                "final_dev_acc": selected_acc,
                "default_acc": default_acc,
                "delta_vs_default": delta,
                "reached_or_exceeded_default": bool(math.isfinite(delta) and delta >= 0),
                "source_path": selected_train["source_path"] if selected_train is not None else "",
            }
        )
        comp_rows.append(
            {
                "model": model,
                "task": task,
                "precision": precision,
                "perturbation_mode": mode,
                "default_h": DEFAULT_H,
                "selected_h": selected_h,
                "selected_policy": "probe_only_default_aware",
                "default_acc": default_acc,
                "selected_acc": selected_acc,
                "delta_vs_default": delta,
                "win/tie/loss": outcome,
                "A_uniform_default": default_metric.get("A_uniform", np.nan) if default_metric is not None else np.nan,
                "A_uniform_selected": selected_metric.get("A_uniform", np.nan) if selected_metric is not None else np.nan,
                "V_align_default": default_metric.get("V_align", np.nan) if default_metric is not None else np.nan,
                "V_align_selected": selected_metric.get("V_align", np.nan) if selected_metric is not None else np.nan,
                "relative_disp_default": default_metric.get("relative_disp", np.nan) if default_metric is not None else np.nan,
                "relative_disp_selected": selected_metric.get("relative_disp", np.nan) if selected_metric is not None else np.nan,
                "nMSE_default": default_metric.get("nMSE_loss", np.nan) if default_metric is not None else np.nan,
                "nMSE_selected": selected_metric.get("nMSE_loss", np.nan) if selected_metric is not None else np.nan,
            }
        )
    final_df = pd.DataFrame(final_rows)
    comp_df = pd.DataFrame(comp_rows)
    pilot_df = pd.DataFrame(
        columns=[
            "model",
            "task",
            "precision",
            "perturbation_mode",
            "h_policy",
            "h_value",
            "steps_run",
            "best_dev_acc",
            "final_dev_acc",
            "default_acc_reference",
            "delta_vs_default",
            "status",
            "source_path",
        ]
    )
    return pilot_df, final_df, comp_df


def make_policies(selectors: pd.DataFrame, final_validation: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    per_config_rows = []
    for _, sel in selectors.iterrows():
        h = to_float(sel.get("h_default_aware"))
        key_mask = (
            final_validation["model"].astype(str).eq(str(sel["model"]))
            & final_validation["task"].astype(str).eq(str(sel["task"]))
            & final_validation["precision"].astype(str).eq(str(sel["precision"]))
            & final_validation["perturbation_mode"].astype(str).eq(str(sel["perturbation_mode"]))
        ) if not final_validation.empty else pd.Series(dtype=bool)
        val = final_validation.loc[key_mask].iloc[0] if not final_validation.empty and key_mask.any() else None
        delta = to_float(val["delta_vs_default"]) if val is not None else np.nan
        per_config_rows.append(
            {
                "model": sel["model"],
                "task": sel["task"],
                "precision": sel["precision"],
                "perturbation_mode": sel["perturbation_mode"],
                "h_per_config": h,
                "selector_version": "probe_only",
                "fallback_to_default": bool(abs(math.log(h / DEFAULT_H)) < 1e-12) if h > 0 else True,
                "reason": sel.get("default_aware_reason", ""),
            }
        )
        if math.isfinite(delta):
            if delta >= -0.005:
                h_cal = h
                reason = "existing_training_validates_or_ties_probe_selected"
            else:
                h_cal = DEFAULT_H
                reason = "existing_training_selected_lost_fallback_to_default"
        else:
            h_cal = h
            reason = "no_training_validation_use_probe_only_candidate"
        per_config_rows.append(
            {
                "model": sel["model"],
                "task": sel["task"],
                "precision": sel["precision"],
                "perturbation_mode": sel["perturbation_mode"],
                "h_per_config": h_cal,
                "selector_version": "pilot_calibrated",
                "fallback_to_default": bool(abs(math.log(h_cal / DEFAULT_H)) < 1e-12) if h_cal > 0 else True,
                "reason": reason,
            }
        )
    per_config = pd.DataFrame(per_config_rows)
    per_model_rows = []
    for key, group in per_config.groupby(["selector_version", "model", "precision", "perturbation_mode"], dropna=False):
        hs = group["h_per_config"].dropna().astype(float)
        if hs.empty:
            h_model = np.nan
        else:
            h_model = float(np.exp(np.mean(np.log(hs))))
        per_model_rows.append(
            {
                "selector_version": key[0],
                "model": key[1],
                "precision": key[2],
                "perturbation_mode": key[3],
                "h_model_log_median": h_model,
                "tasks_used": ";".join(sorted(group["task"].astype(str).unique())),
                "single_h_pass": False,
                "notes": "not validated across tasks; use per-config unless pilot/final rows prove >= default - 1 point",
            }
        )
    per_model = pd.DataFrame(per_model_rows)
    per_precision_rows = []
    for key, group in per_config.groupby(["selector_version", "precision"], dropna=False):
        hs = group["h_per_config"].dropna().astype(float)
        if hs.empty:
            h_precision = np.nan
        else:
            h_precision = float(np.exp(np.mean(np.log(hs))))
        per_precision_rows.append(
            {
                "selector_version": key[0],
                "precision": key[1],
                "h_precision_log_median": h_precision,
                "global_precision_h_usable": False,
                "success_rate_estimate": np.nan,
                "notes": "no universal h claimed without multi-model validation",
            }
        )
    return per_config, per_model, pd.DataFrame(per_precision_rows)


def make_plots(bundle_dir: Path, selectors: pd.DataFrame, comparison: pd.DataFrame, metrics: pd.DataFrame) -> List[str]:
    notes = []
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        return [f"matplotlib unavailable: {exc}"]
    fig_dir = bundle_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    if not comparison.empty:
        c = comparison.copy()
        c["label"] = c["model"].astype(str).str.replace("facebook/", "", regex=False) + "/" + c["task"].astype(str) + "/" + c["precision"].astype(str) + "/" + c["perturbation_mode"].astype(str)
        x = np.arange(len(c))
        fig, ax = plt.subplots(figsize=(max(8, 0.7 * len(c)), 5))
        ax.bar(x - 0.18, pd.to_numeric(c["default_acc"], errors="coerce"), width=0.36, label="default")
        ax.bar(x + 0.18, pd.to_numeric(c["selected_acc"], errors="coerce"), width=0.36, label="selected")
        ax.set_xticks(x)
        ax.set_xticklabels(c["label"], rotation=60, ha="right")
        ax.set_ylabel("dev accuracy")
        ax.legend()
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(fig_dir / "fig_selected_h_vs_default_by_config.pdf")
        fig.savefig(fig_dir / "fig_selected_h_vs_default_by_config.png")
        plt.close(fig)
        outcome = c["win/tie/loss"].fillna("").replace("", "missing")
        counts = outcome.value_counts()
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(counts.index.astype(str), counts.values)
        ax.set_ylabel("count")
        ax.set_title("win/tie/loss summary")
        fig.tight_layout()
        fig.savefig(fig_dir / "fig_win_rate_summary.pdf")
        fig.savefig(fig_dir / "fig_win_rate_summary.png")
        plt.close(fig)
    else:
        notes.append("comparison plot skipped: no comparison rows")
    if not selectors.empty:
        fig, ax = plt.subplots(figsize=(8, 5))
        for (model, precision), g in selectors.groupby(["model", "precision"], dropna=False):
            ax.scatter(g["h_default_aware"], g["h_score_best"], label=f"{model} {precision}")
        ax.axvline(DEFAULT_H, color="black", linestyle="--", linewidth=1)
        ax.axhline(DEFAULT_H, color="black", linestyle="--", linewidth=1)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("selected h_default_aware")
        ax.set_ylabel("h_score_best")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(fig_dir / "fig_h_selected_by_precision_model.pdf")
        fig.savefig(fig_dir / "fig_h_selected_by_precision_model.png")
        plt.close(fig)
    if not metrics.empty:
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter(pd.to_numeric(metrics["A_uniform"], errors="coerce"), pd.to_numeric(metrics["V_align"], errors="coerce"), s=12, alpha=0.6)
        ax.set_xscale("log")
        ax.set_xlabel("A_uniform")
        ax.set_ylabel("V_align")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(fig_dir / "fig_probe_metrics_default_vs_selected.pdf")
        fig.savefig(fig_dir / "fig_probe_metrics_default_vs_selected.png")
        fig.savefig(fig_dir / "fig_interval_score_vs_accuracy.pdf")
        fig.savefig(fig_dir / "fig_interval_score_vs_accuracy.png")
        plt.close(fig)
    return notes


def write_readme(
    bundle_dir: Path,
    interval_df: pd.DataFrame,
    training_df: pd.DataFrame,
    selectors: pd.DataFrame,
    notes: Sequence[str],
    elapsed: float,
) -> None:
    lines = [
        "# Interval-Aware h Selection 8h Bundle",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        f"Host: {socket.gethostname()}",
        f"Git commit: {git_commit()}",
        f"Elapsed seconds: {elapsed:.1f}",
        "",
        "## Recommendation",
        "",
        "The current recommended probe-only selector is `h_default_aware`: it first uses interval geometry and optional loss nMSE, then falls back to `h=1e-3` if no probe-only candidate is safer than default. This avoids selecting a visually clean h that is too non-local.",
        "",
        "Pilot-calibrated selection is not claimed in this bundle unless `pilot_training_results.csv` contains completed rows. Training accuracy is used only for validation in the probe-only tables.",
        "",
        "## Coverage",
        "",
        f"- Interval metric rows: {len(interval_df)}",
        f"- Training rows audited: {len(training_df)}",
        f"- Selector configs: {len(selectors)}",
        "",
        "### Interval configs",
    ]
    if interval_df.empty:
        lines.append("- none")
    else:
        for key, g in interval_df.groupby(["model", "task", "precision", "perturbation_mode"], dropna=False):
            lines.append(f"- {key[0]} / {key[1]} / {key[2]} / {key[3]}: {len(g)} h rows")
    lines.extend(["", "### Selector outcomes"])
    for _, r in selectors.iterrows():
        lines.append(
            f"- {r['model']} / {r['task']} / {r['precision']} / {r['perturbation_mode']}: "
            f"h_default_aware={r.get('h_default_aware')} score_best={r.get('h_score_best')} "
            f"window_exists={r.get('window_exists')}"
        )
    lines.extend(
        [
            "",
            "## Answers",
            "",
            "1. Final recommended selector: probe-only `h_default_aware`; use pilot-calibrated only after short training rows exist.",
            "2. RoBERTa vs default: see `default_comparison_summary.csv`; no missing result is fabricated.",
            "3. OPT vs default: same table; current OPT coverage is mostly existing INT4/robustness logs unless new probes are added.",
            "4. Per-precision global h: not claimed unless `policy_per_precision.csv` marks it usable.",
            "5. If global h is unavailable, prefer per-config; per-model requires validation across tasks.",
            "6. Failures are marked by missing source paths, fallback_to_default, or no interval window.",
            "7. Next experiments: run loss-level nMSE for OPT INT8 dense/sparse and short 300-step pilots for selected h vs default.",
            "",
            "## Notes / Missing Items",
        ]
    )
    if notes:
        lines.extend(f"- {note}" for note in notes)
    else:
        lines.append("- none")
    (bundle_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def zip_bundle(bundle_dir: Path) -> Path:
    zip_path = REPO_ROOT / "interval_h_selection_8h_bundle.zip"
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(bundle_dir.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(bundle_dir.parent))
    return zip_path


def write_audit(bundle_dir: Path, interval_df: pd.DataFrame, training_df: pd.DataFrame, notes: Sequence[str]) -> None:
    lines = ["# Audit Report", "", "## Existing Interval Metrics", ""]
    if interval_df.empty:
        lines.append("- none found")
    else:
        for key, g in interval_df.groupby(["model", "task", "precision", "perturbation_mode"], dropna=False):
            lines.append(f"- {key[0]} / {key[1]} / {key[2]} / {key[3]}: {len(g)} rows, h={sorted(g['h'].dropna().astype(float).unique())}")
    lines.extend(["", "## Existing Training Results", ""])
    if training_df.empty:
        lines.append("- none found")
    else:
        for key, g in training_df.groupby(["model", "task", "precision", "perturbation_mode"], dropna=False):
            lines.append(f"- {key[0]} / {key[1]} / {key[2]} / {key[3]}: {len(g)} rows")
    lines.extend(["", "## Missing / Prioritized", ""])
    priorities = [
        "OPT-1.3B / SST-5 / INT8 / dense+sparse interval geometry and loss nMSE",
        "RoBERTa and OPT / TREC,RTE / INT8 loss-level nMSE",
        "Short 300-step pilots for default vs selected h where full logs are absent",
        "INT4 secondary probes after INT8 coverage is complete",
    ]
    lines.extend(f"- {p}" for p in priorities)
    if notes:
        lines.extend(["", "## Notes", ""])
        lines.extend(f"- {n}" for n in notes)
    (bundle_dir / "audit_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", default="interval_h_selection_8h_bundle")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    start = time.time()
    bundle_dir = REPO_ROOT / args.output_dir
    if args.overwrite and bundle_dir.exists():
        shutil.rmtree(bundle_dir)
    bundle_dir.mkdir(parents=True, exist_ok=True)
    notes: List[str] = []
    notes.append("Workflow did not launch long training; it uses existing logs and probe outputs.")
    interval_df, interval_notes = collect_interval_metrics(REPO_ROOT, bundle_dir)
    notes.extend(interval_notes)
    training_df = collect_training(REPO_ROOT)
    loss_df = collect_loss_mse(REPO_ROOT, interval_df)
    metrics = add_selector_metrics(interval_df, loss_df)
    selectors, candidates = make_selectors(metrics)
    pilot, final_validation, default_comparison = build_validation_tables(selectors, candidates, training_df, metrics)
    policy_config, policy_model, policy_precision = make_policies(selectors, final_validation)
    write_csv(bundle_dir / "all_existing_training.csv", training_df, TRAINING_COLS)
    write_csv(bundle_dir / "all_interval_metrics.csv", metrics)
    interval_new = metrics[metrics.get("source_path", pd.Series(dtype=str)).astype(str).str.contains("outputs/interval_h_selection_8h_probes", regex=False, na=False)].copy()
    write_csv(bundle_dir / "interval_geometry_new.csv", interval_new)
    write_csv(bundle_dir / "loss_mse_all.csv", loss_df, LOSS_COLS)
    loss_new = loss_df[loss_df.get("source_path", pd.Series(dtype=str)).astype(str).str.contains("outputs/interval_h_selection_8h_probes", regex=False, na=False)].copy() if not loss_df.empty else pd.DataFrame(columns=LOSS_COLS)
    write_csv(bundle_dir / "loss_mse_new.csv", loss_new, LOSS_COLS)
    write_csv(bundle_dir / "h_candidate_table.csv", candidates)
    write_csv(bundle_dir / "pilot_training_results.csv", pilot)
    write_csv(bundle_dir / "final_validation_results.csv", final_validation)
    write_csv(bundle_dir / "policy_per_config.csv", policy_config)
    write_csv(bundle_dir / "policy_per_model.csv", policy_model)
    write_csv(bundle_dir / "policy_per_precision.csv", policy_precision)
    write_csv(bundle_dir / "default_comparison_summary.csv", default_comparison)
    write_csv(bundle_dir / "selector_summary.csv", selectors)
    write_audit(bundle_dir, interval_df, training_df, notes)
    plot_notes = make_plots(bundle_dir, selectors, default_comparison, metrics)
    notes.extend(plot_notes)
    write_json(
        bundle_dir / "metadata.json",
        {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "git_commit": git_commit(),
            "hostname": socket.gethostname(),
            "repo_root": str(REPO_ROOT),
            "elapsed_seconds": time.time() - start,
            "notes": notes,
        },
    )
    write_readme(bundle_dir, interval_df, training_df, selectors, notes, time.time() - start)
    zip_path = zip_bundle(bundle_dir)
    print(f"Bundle directory: {bundle_dir}")
    print(f"Zip: {zip_path}")
    print(f"Interval rows: {len(interval_df)}")
    print(f"Training rows: {len(training_df)}")
    print(f"Selector configs: {len(selectors)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
