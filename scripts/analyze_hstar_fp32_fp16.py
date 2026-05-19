#!/usr/bin/env python3
"""Offline h-star analysis for the FP32/FP16 RoBERTa-large SST-5 h-sweep.

This script is intentionally read-only with respect to training artifacts.  It
discovers completed FP32/FP16 runs, reuses the cached per-direction checkpoint
probe JSONL files, estimates Delta/G/L variants, and writes h-star summaries.
It does not launch training or run new forward/backward passes.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import os
import statistics
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


H_GRID = [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 1.5e-3, 2e-3, 3e-3, 4e-3, 5e-3, 1e-2]
PRECISIONS = {"fp32", "fp16"}
EPS_NUM = 1e-12


HSTAR_FIELDS = [
    "precision",
    "checkpoint_name",
    "checkpoint_path",
    "reference_run_h",
    "dtype",
    "d_trainable",
    "direction_normalization",
    "selector_name",
    "delta_mode",
    "Delta_value",
    "G_method",
    "G_hat",
    "G_h_used",
    "G_snr",
    "G_stability",
    "L_method",
    "L_hat_q",
    "L_hat",
    "L_h2_used",
    "L_snr2",
    "hstar_cont",
    "hstar_clipped",
    "hstar_nearest_grid",
    "hstar_out_of_grid_flag",
    "fallback_flags",
    "mse_cont",
    "nmse_cont",
    "corr_cont",
    "mse_nearest",
    "nmse_nearest",
    "corr_nearest",
    "empirical_min_mse_h",
    "empirical_min_nmse",
    "empirical_max_corr_h",
    "empirical_max_corr",
]


GRID_FIELDS = [
    "precision",
    "checkpoint",
    "h",
    "mse",
    "nmse",
    "corr",
    "bias",
    "mae",
    "median_abs_error",
    "active_frac",
    "alignment",
    "norm_ratio",
    "fd_zero_ratio",
    "d_fd_mean",
    "d_fd_std",
    "d_true_mean",
    "d_true_std",
    "num_directions",
    "delta_q_norm_mean",
    "nominal_delta_norm_mean",
    "snapping_norm_error_rms_proxy",
]


@dataclass
class RunInfo:
    precision: str
    h: float
    run_name: str
    run_dir: Path
    config_path: Path
    probe_path: Path
    config: Dict[str, Any]


def now_stamp() -> str:
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def safe_float(value: Any) -> Optional[float]:
    try:
        if value in (None, ""):
            return None
        out = float(value)
    except Exception:
        return None
    return out if math.isfinite(out) else None


def h_key(h: float) -> str:
    return f"{float(h):.12g}"


def h_close(a: float, b: float, rtol: float = 1e-9, atol: float = 1e-12) -> bool:
    return abs(float(a) - float(b)) <= max(atol, rtol * max(abs(float(a)), abs(float(b)), 1.0))


def find_h(value: float, grid: Sequence[float] = H_GRID) -> Optional[float]:
    for h in grid:
        if h_close(value, h):
            return float(h)
    return None


def read_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def write_csv(path: Path, rows: Iterable[Dict[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fields), extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or y.size < 2:
        return float("nan")
    sx = float(np.std(x))
    sy = float(np.std(y))
    if sx <= 0.0 or sy <= 0.0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def metrics(fd: np.ndarray, true: np.ndarray) -> Dict[str, float]:
    mask = np.isfinite(fd) & np.isfinite(true)
    fd = fd[mask]
    true = true[mask]
    if fd.size == 0:
        return {
            "mse": float("nan"),
            "nmse": float("nan"),
            "corr": float("nan"),
            "bias": float("nan"),
            "mae": float("nan"),
            "median_abs_error": float("nan"),
        }
    diff = fd - true
    mse = float(np.mean(diff * diff))
    denom = float(np.mean(true * true)) + EPS_NUM
    return {
        "mse": mse,
        "nmse": float(mse / denom),
        "corr": pearson(fd, true),
        "bias": float(np.mean(diff)),
        "mae": float(np.mean(np.abs(diff))),
        "median_abs_error": float(np.median(np.abs(diff))),
    }


def mean_or_nan(values: Sequence[Any]) -> float:
    vals = [float(v) for v in values if safe_float(v) is not None]
    return float(np.mean(vals)) if vals else float("nan")


def std_or_nan(values: Sequence[Any]) -> float:
    vals = [float(v) for v in values if safe_float(v) is not None]
    return float(np.std(vals)) if vals else float("nan")


def discover_runs(search_roots: Sequence[Path], diagnostics: Dict[str, Any]) -> List[RunInfo]:
    runs: List[RunInfo] = []
    config_paths: List[Path] = []
    for root in search_roots:
        if root.exists():
            config_paths.extend(root.rglob("run_config.json"))
    diagnostics["run_config_paths_seen"] = len(config_paths)

    for cfg_path in sorted(config_paths):
        cfg = read_json(cfg_path)
        precision = str(cfg.get("precision_mode", "")).lower()
        if precision not in PRECISIONS:
            continue
        model = str(cfg.get("model", cfg.get("model_name_or_path", ""))).lower()
        dataset = str(cfg.get("dataset", cfg.get("task_name", ""))).lower()
        if model != "roberta-large":
            continue
        if dataset not in {"sst-5", "sst5"}:
            continue
        if int(cfg.get("seed", -1)) != 16 or int(cfg.get("data_seed", -1)) != 16:
            continue
        if str(cfg.get("dataset_mode", "")).lower() != "full":
            continue
        if int(cfg.get("batch_size", cfg.get("per_device_train_batch_size", -1))) != 64:
            continue
        if not bool(cfg.get("dataloader_shuffle", False)):
            continue
        if str(cfg.get("direction_type", "")).lower() != "dense":
            continue
        h_val = safe_float(cfg.get("h"))
        if h_val is None or find_h(h_val) is None:
            continue
        run_dir = cfg_path.parent
        probe_path = run_dir / "checkpoint_probe_stats.jsonl"
        runs.append(
            RunInfo(
                precision=precision,
                h=float(find_h(h_val) or h_val),
                run_name=str(cfg.get("run_name") or run_dir.parent.name),
                run_dir=run_dir,
                config_path=cfg_path,
                probe_path=probe_path,
                config=cfg,
            )
        )

    runs = dedupe_runs(runs)
    diagnostics["runs_discovered"] = [
        {
            "precision": r.precision,
            "h": r.h,
            "run_name": r.run_name,
            "run_dir": str(r.run_dir),
            "probe_exists": r.probe_path.exists(),
        }
        for r in runs
    ]
    return runs


def dedupe_runs(runs: Sequence[RunInfo]) -> List[RunInfo]:
    by_key: Dict[Tuple[str, str], RunInfo] = {}
    for run in runs:
        key = (run.precision, h_key(run.h))
        existing = by_key.get(key)
        if existing is None:
            by_key[key] = run
            continue
        # Prefer the real main sweep over packaged copies or smoke runs.
        p = str(run.run_dir)
        q = str(existing.run_dir)
        score = (0 if "_package" in p else 1, 0 if "/smoke/" in p else 1, len(p))
        old_score = (0 if "_package" in q else 1, 0 if "/smoke/" in q else 1, len(q))
        if score > old_score:
            by_key[key] = run
    return [by_key[k] for k in sorted(by_key, key=lambda x: (x[0], float(x[1])))]


def load_probe_rows(run: RunInfo) -> List[Dict[str, Any]]:
    if not run.probe_path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with run.probe_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            h = safe_float(row.get("h_raw"))
            if h is None or not h_close(h, run.h):
                continue
            if str(row.get("direction_type", "dense")).lower() != "dense":
                continue
            if row.get("d_fd") is None or row.get("d_true") is None:
                continue
            rows.append(row)
    return rows


def direction_key(row: Dict[str, Any]) -> Tuple[int, int, int]:
    return (
        int(row.get("batch_index", 0) or 0),
        int(row.get("direction_index", 0) or 0),
        int(row.get("seed", 0) or 0),
    )


def build_probe_sets(runs: Sequence[RunInfo], diagnostics: Dict[str, Any]) -> Dict[Tuple[str, int], Dict[float, List[Dict[str, Any]]]]:
    grouped: Dict[Tuple[str, int], Dict[float, List[Dict[str, Any]]]] = {}
    skipped = []
    for run in runs:
        rows = load_probe_rows(run)
        if not rows:
            skipped.append({"run_name": run.run_name, "reason": "missing_or_empty_checkpoint_probe_stats"})
            continue
        by_step: Dict[int, List[Dict[str, Any]]] = {}
        for row in rows:
            step = int(row.get("checkpoint_step", row.get("global_step", 0)) or 0)
            by_step.setdefault(step, []).append(row)
        for step, step_rows in by_step.items():
            grouped.setdefault((run.precision, step), {})[run.h] = step_rows
    diagnostics["skipped_probe_runs"] = skipped
    return grouped


def align_probe_set(h_to_rows: Dict[float, List[Dict[str, Any]]]) -> Dict[float, List[Dict[str, Any]]]:
    key_sets = []
    by_h_keyed: Dict[float, Dict[Tuple[int, int, int], Dict[str, Any]]] = {}
    for h, rows in h_to_rows.items():
        keyed = {direction_key(row): row for row in rows}
        by_h_keyed[h] = keyed
        key_sets.append(set(keyed))
    common = set.intersection(*key_sets) if key_sets else set()
    ordered_keys = sorted(common)
    return {h: [by_h_keyed[h][k] for k in ordered_keys] for h in sorted(by_h_keyed)}


def arrays_for(rows: Sequence[Dict[str, Any]], field: str) -> np.ndarray:
    vals = [safe_float(row.get(field)) for row in rows]
    return np.asarray([float(v) for v in vals if v is not None], dtype=np.float64)


def compute_grid_metrics(
    precision: str,
    checkpoint_name: str,
    h_to_rows: Dict[float, List[Dict[str, Any]]],
) -> Tuple[List[Dict[str, Any]], Dict[float, Dict[str, Any]]]:
    rows_out: List[Dict[str, Any]] = []
    by_h: Dict[float, Dict[str, Any]] = {}
    for h, rows in sorted(h_to_rows.items()):
        fd = arrays_for(rows, "d_fd")
        true = arrays_for(rows, "d_true")
        m = metrics(fd, true)
        active = mean_or_nan([r.get("probe_active_frac") for r in rows])
        align = mean_or_nan([r.get("probe_alignment") for r in rows])
        norm_ratio = mean_or_nan([r.get("probe_norm_ratio") for r in rows])
        zero_ratio = float(np.mean(np.abs(fd) < EPS_NUM)) if fd.size else float("nan")
        delta_q = arrays_for(rows, "delta_q_norm")
        nominal = arrays_for(rows, "nominal_delta_norm")
        snap_proxy = float("nan")
        if delta_q.size and nominal.size and delta_q.size == nominal.size:
            snap_proxy = float(np.sqrt(np.mean((delta_q - nominal) ** 2)))
        item = {
            "precision": precision,
            "checkpoint": checkpoint_name,
            "h": h,
            **m,
            "active_frac": active,
            "alignment": align,
            "norm_ratio": norm_ratio,
            "fd_zero_ratio": zero_ratio,
            "d_fd_mean": float(np.mean(fd)) if fd.size else float("nan"),
            "d_fd_std": float(np.std(fd)) if fd.size else float("nan"),
            "d_true_mean": float(np.mean(true)) if true.size else float("nan"),
            "d_true_std": float(np.std(true)) if true.size else float("nan"),
            "num_directions": int(min(fd.size, true.size)),
            "delta_q_norm_mean": float(np.mean(delta_q)) if delta_q.size else float("nan"),
            "nominal_delta_norm_mean": float(np.mean(nominal)) if nominal.size else float("nan"),
            "snapping_norm_error_rms_proxy": snap_proxy,
        }
        rows_out.append(item)
        by_h[h] = item
    return rows_out, by_h


def infer_d_trainable(h_to_rows: Dict[float, List[Dict[str, Any]]]) -> int:
    vals = []
    for h, rows in h_to_rows.items():
        for row in rows:
            nominal = safe_float(row.get("nominal_delta_norm"))
            if nominal is None or h <= 0:
                continue
            u_norm = nominal / (2.0 * float(h))
            vals.append(u_norm * u_norm)
    if not vals:
        return 0
    return int(round(float(np.mean(vals))))


def choose_reference_run(runs: Sequence[RunInfo], precision: str, h_ref: float = 1e-3) -> Optional[RunInfo]:
    exact = [r for r in runs if r.precision == precision and h_close(r.h, h_ref)]
    if exact:
        return exact[0]
    same = [r for r in runs if r.precision == precision]
    if not same:
        return None
    return min(same, key=lambda r: abs(math.log(r.h) - math.log(h_ref)))


def checkpoint_model_path(reference_run: Optional[RunInfo], checkpoint_step: int) -> Tuple[str, Optional[Path], List[str]]:
    warnings: List[str] = []
    if reference_run is None:
        return "", None, ["no_reference_run_for_delta"]
    ckpt_root = reference_run.run_dir / "checkpoints"
    candidates: List[Path] = []
    if checkpoint_step > 0:
        candidates.append(ckpt_root / f"step_{checkpoint_step}" / "model.safetensors")
    else:
        warnings.append("no_on_disk_step0_checkpoint; using reference h=1e-3 step_1000 checkpoint for Delta ULP only")
        candidates.append(ckpt_root / "step_1000" / "model.safetensors")
    candidates.append(ckpt_root / "final" / "model.safetensors")
    for path in candidates:
        if path.exists():
            return str(path), path, warnings
    warnings.append("no_model_safetensors_found_for_delta")
    return "", None, warnings


def estimate_ulp_stats(model_path: Optional[Path], precision: str, diagnostics: Dict[str, Any]) -> Dict[str, float]:
    if model_path is None or not model_path.exists():
        return {}
    try:
        import torch
        from safetensors import safe_open
    except Exception as exc:
        diagnostics.setdefault("warnings", []).append(f"ULP estimation skipped: {type(exc).__name__}: {exc}")
        return {}

    target_dtype = torch.float16 if precision == "fp16" else torch.float32
    sum_sq = 0.0
    count = 0
    finite_count = 0
    nonfinite_count = 0
    zero_spacing_count = 0
    samples: List[np.ndarray] = []
    per_tensor_sample_cap = 8192
    dtype_seen: Dict[str, int] = {}

    with safe_open(str(model_path), framework="pt", device="cpu") as f:
        for key in f.keys():
            try:
                tensor = f.get_tensor(key)
            except Exception:
                continue
            if not torch.is_floating_point(tensor):
                continue
            dtype_seen[str(tensor.dtype)] = dtype_seen.get(str(tensor.dtype), 0) + 1
            cast = tensor.detach().to(dtype=target_dtype)
            inf = torch.full_like(cast, float("inf"))
            spacing = (torch.nextafter(cast, inf) - cast).abs().to(dtype=torch.float32).reshape(-1)
            finite = torch.isfinite(spacing)
            n_total = int(spacing.numel())
            n_finite = int(finite.sum().item())
            count += n_total
            finite_count += n_finite
            nonfinite_count += n_total - n_finite
            if n_finite <= 0:
                continue
            vals = spacing[finite]
            zero_spacing_count += int((vals == 0).sum().item())
            sum_sq += float(torch.sum(vals * vals).item())
            n_sample = min(per_tensor_sample_cap, int(vals.numel()))
            if n_sample > 0:
                if int(vals.numel()) == n_sample:
                    sample = vals
                else:
                    idx = torch.linspace(0, int(vals.numel()) - 1, steps=n_sample, dtype=torch.long)
                    sample = vals[idx]
                samples.append(sample.cpu().numpy().astype(np.float64, copy=False))
            del tensor, cast, inf, spacing

    out: Dict[str, float] = {
        "count": float(count),
        "finite_count": float(finite_count),
        "nonfinite_count": float(nonfinite_count),
        "zero_spacing_count": float(zero_spacing_count),
    }
    if finite_count > 0 and sum_sq > 0.0:
        out["ulp_rms"] = float(math.sqrt(sum_sq / float(finite_count)))
    if samples:
        arr = np.concatenate(samples)
        arr = arr[np.isfinite(arr)]
        if arr.size:
            out["ulp_median"] = float(np.quantile(arr, 0.50))
            out["ulp_p90"] = float(np.quantile(arr, 0.90))
            out["ulp_p95"] = float(np.quantile(arr, 0.95))
            out["ulp_sample_count"] = float(arr.size)
    diagnostics.setdefault("ulp_dtype_seen", {})[precision] = dtype_seen
    return out


def stability_against_double(h_to_rows: Dict[float, List[Dict[str, Any]]], h: float) -> Tuple[float, float, bool]:
    h2 = find_h(2.0 * h, h_to_rows.keys())
    if h2 is None:
        return float("nan"), float("nan"), False
    a = arrays_for(h_to_rows[h], "d_fd")
    b = arrays_for(h_to_rows[h2], "d_fd")
    n = min(a.size, b.size)
    if n == 0:
        return float("nan"), float("nan"), False
    a = a[:n]
    b = b[:n]
    corr = pearson(a, b)
    sign_flip = float(np.mean(np.sign(a) != np.sign(b)))
    return corr, sign_flip, True


def select_code_g(h_to_rows: Dict[float, List[Dict[str, Any]]]) -> Dict[str, Any]:
    h_ref = find_h(1e-3, h_to_rows.keys())
    flags: List[str] = []
    if h_ref is None:
        h_ref = min(h_to_rows, key=lambda h: abs(math.log(h) - math.log(1e-3)))
        flags.append("fallback_codeG_h_ref_nearest_1e-3")
    vals = np.abs(arrays_for(h_to_rows[h_ref], "d_fd"))
    g = float(math.sqrt(math.pi / 2.0) * np.mean(vals)) if vals.size else float("nan")
    return {
        "method": "codeG",
        "G_hat": g,
        "h_used": h_ref,
        "snr": float("nan"),
        "stability": float("nan"),
        "flags": flags,
    }


def select_abs_g(h_to_rows: Dict[float, List[Dict[str, Any]]]) -> Dict[str, Any]:
    candidates = []
    for h in sorted(h_to_rows):
        fd = arrays_for(h_to_rows[h], "d_fd")
        if fd.size == 0:
            continue
        corr2, sign_flip, has_double = stability_against_double(h_to_rows, h)
        align = mean_or_nan([r.get("probe_alignment") for r in h_to_rows[h]])
        norm = mean_or_nan([r.get("probe_norm_ratio") for r in h_to_rows[h]])
        med_abs = float(np.median(np.abs(fd)))
        snr = float(med_abs / (float(np.std(fd - np.mean(fd))) + EPS_NUM))
        ok = (
            med_abs > EPS_NUM
            and (not has_double or (corr2 >= 0.90 and sign_flip <= 0.10))
            and (not math.isfinite(align) or align > 0.5)
            and (not math.isfinite(norm) or (0.3 <= norm <= 3.0))
        )
        score = (
            (corr2 if math.isfinite(corr2) else -1.0)
            - sign_flip
            + min(1.0, med_abs / (np.median(np.abs(fd)) + EPS_NUM))
        )
        candidates.append((ok, score, h, med_abs, corr2, sign_flip, snr))
    flags: List[str] = []
    passing = [c for c in candidates if c[0]]
    if passing:
        chosen = min(passing, key=lambda c: c[2])
    else:
        chosen = max(candidates, key=lambda c: c[1])
        flags.append("fallback_absG_best_stability_score")
    _, _, h, _med_abs, corr2, sign_flip, snr = chosen
    fd = np.abs(arrays_for(h_to_rows[h], "d_fd"))
    g = float(math.sqrt(math.pi / 2.0) * np.mean(fd)) if fd.size else float("nan")
    return {
        "method": "absG",
        "G_hat": g,
        "h_used": h,
        "snr": snr,
        "stability": corr2,
        "sign_flip": sign_flip,
        "flags": flags,
        "candidates": [
            {
                "h": c[2],
                "passed": c[0],
                "score": c[1],
                "median_abs": c[3],
                "corr_d2_2h": c[4],
                "sign_flip_2h": c[5],
                "snr": c[6],
            }
            for c in candidates
        ],
    }


def richardson_values(h_to_rows: Dict[float, List[Dict[str, Any]]], h: float) -> Optional[np.ndarray]:
    h2 = find_h(2.0 * h, h_to_rows.keys())
    if h2 is None:
        return None
    d1 = arrays_for(h_to_rows[h], "d_fd")
    d2 = arrays_for(h_to_rows[h2], "d_fd")
    n = min(d1.size, d2.size)
    if n == 0:
        return None
    return (4.0 * d1[:n] - d2[:n]) / 3.0


def select_richardson_g(h_to_rows: Dict[float, List[Dict[str, Any]]]) -> Dict[str, Any]:
    candidates = []
    for h in sorted(h_to_rows):
        dr = richardson_values(h_to_rows, h)
        if dr is None or dr.size == 0 or not np.all(np.isfinite(dr)):
            continue
        dr2 = richardson_values(h_to_rows, 2.0 * h)
        if dr2 is not None and dr2.size:
            n = min(dr.size, dr2.size)
            err = np.abs(dr[:n] - dr2[:n])
            denom = float(np.median(np.abs(dr[:n]))) + EPS_NUM
            stability = float(np.median(err) / denom)
            snr = float(np.median(np.abs(dr[:n])) / (float(np.median(err)) + EPS_NUM))
            sign_flip = float(np.mean(np.sign(dr[:n]) != np.sign(dr2[:n])))
            ok = snr >= 3.0 and stability <= 0.33 and sign_flip <= 0.10
        else:
            d2h = arrays_for(h_to_rows[h], "d_fd")
            n = min(dr.size, d2h.size)
            correction = float(np.median(np.abs(dr[:n] - d2h[:n])) / (float(np.median(np.abs(dr[:n]))) + EPS_NUM))
            stability = correction
            snr = float(1.0 / (correction + EPS_NUM))
            sign_flip = float(np.mean(np.sign(dr[:n]) != np.sign(d2h[:n])))
            ok = correction <= 0.5
        candidates.append((ok, snr, stability, sign_flip, h))
    flags: List[str] = []
    if not candidates:
        return {
            "method": "richardsonG",
            "G_hat": float("nan"),
            "h_used": float("nan"),
            "snr": float("nan"),
            "stability": float("nan"),
            "flags": ["richardsonG_unavailable_no_2h_pairs"],
            "candidates": [],
        }
    passing = [c for c in candidates if c[0]]
    if passing:
        chosen = min(passing, key=lambda c: c[4])
    else:
        chosen = max(candidates, key=lambda c: c[1])
        flags.append("fallback_richardsonG_largest_snr")
    _, snr, stability, sign_flip, h = chosen
    dr = richardson_values(h_to_rows, h)
    g = float(math.sqrt(math.pi / 2.0) * np.mean(np.abs(dr))) if dr is not None and dr.size else float("nan")
    return {
        "method": "richardsonG",
        "G_hat": g,
        "h_used": h,
        "snr": snr,
        "stability": stability,
        "sign_flip": sign_flip,
        "flags": flags,
        "candidates": [
            {"h": c[4], "passed": c[0], "snr": c[1], "stability": c[2], "sign_flip": c[3]}
            for c in candidates
        ],
    }


def median_abs_deviation(x: np.ndarray) -> float:
    if x.size == 0:
        return float("nan")
    med = float(np.median(x))
    return float(np.median(np.abs(x - med)))


def estimate_l(h_to_rows: Dict[float, List[Dict[str, Any]]]) -> Dict[str, Any]:
    candidates = []
    by_h_lambdas: Dict[float, np.ndarray] = {}
    for h, rows in sorted(h_to_rows.items()):
        ks = []
        lambdas = []
        for row in rows:
            lp = safe_float(row.get("loss_plus"))
            lm = safe_float(row.get("loss_minus"))
            lb = safe_float(row.get("loss_base"))
            nominal = safe_float(row.get("nominal_delta_norm"))
            if lp is None or lm is None or lb is None or nominal is None or h <= 0:
                continue
            u_norm_sq = (nominal / (2.0 * h)) ** 2
            if u_norm_sq <= 0:
                continue
            k = (lp - 2.0 * lb + lm) / (h ** 2)
            lam = abs(k) / (u_norm_sq + EPS_NUM)
            if math.isfinite(k) and math.isfinite(lam):
                ks.append(k)
                lambdas.append(lam)
        k_arr = np.asarray(ks, dtype=np.float64)
        lam_arr = np.asarray(lambdas, dtype=np.float64)
        if k_arr.size == 0 or lam_arr.size == 0:
            continue
        by_h_lambdas[h] = lam_arr
        mad = median_abs_deviation(k_arr)
        snr2 = float(np.median(np.abs(k_arr)) / (1.4826 * mad + EPS_NUM))
        stability = float("nan")
        h2 = find_h(2.0 * h, by_h_lambdas.keys())
        if h2 is not None:
            denom = float(np.median(lam_arr)) + EPS_NUM
            stability = float(abs(float(np.median(lam_arr)) - float(np.median(by_h_lambdas[h2]))) / denom)
        ok = math.isfinite(snr2) and snr2 >= 2.0 and lam_arr.size > 0
        candidates.append((ok, snr2, stability, h, lam_arr))
    flags: List[str] = []
    passing = [c for c in candidates if c[0]]
    if passing:
        stable = [c for c in passing if (not math.isfinite(c[2])) or c[2] <= 0.5]
        chosen = min(stable or passing, key=lambda c: c[3])
    else:
        chosen = max(candidates, key=lambda c: c[1])
        flags.append("fallback_L_max_snr2")
    _, snr2, stability, h, lam_arr = chosen
    return {
        "method": "cached_symmetric_curvature_proxy",
        "h_used": h,
        "snr2": snr2,
        "stability": stability,
        "q50": float(np.quantile(lam_arr, 0.50)),
        "q90": float(np.quantile(lam_arr, 0.90)),
        "q95": float(np.quantile(lam_arr, 0.95)),
        "flags": flags,
        "candidates": [
            {
                "h": c[3],
                "passed": c[0],
                "snr2": c[1],
                "stability": c[2],
                "lambda_q50": float(np.quantile(c[4], 0.50)),
                "lambda_q90": float(np.quantile(c[4], 0.90)),
                "lambda_q95": float(np.quantile(c[4], 0.95)),
            }
            for c in candidates
        ],
    }


def nearest_grid_h(h: float) -> float:
    return min(H_GRID, key=lambda x: abs(math.log(float(x)) - math.log(float(h))))


def hstar_value(delta: float, g: float, l: float, d_dim: int) -> float:
    if delta <= 0 or g <= 0 or l <= 0 or d_dim <= 0:
        return float("nan")
    return float(0.5 * math.sqrt((delta * g) / (l * math.sqrt(float(d_dim) * float(d_dim + 2)))))


def selector_name(g_method: str, delta_mode: str, l_q: str) -> str:
    d_name = {
        "ulp_rms": "deltaUlp",
        "ulp_median": "deltaUlpMedian",
        "ulp_p90": "deltaUlpP90",
        "empirical_snap_rms": "deltaEmpSnap",
    }.get(delta_mode, delta_mode)
    return f"old_hstar_{g_method}_{d_name}_L{l_q}"


def estimate_rows_for_set(
    precision: str,
    checkpoint_step: int,
    h_to_rows: Dict[float, List[Dict[str, Any]]],
    runs: Sequence[RunInfo],
    diagnostics: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    checkpoint_name = f"checkpoint_step_{checkpoint_step}"
    if checkpoint_step == 0:
        checkpoint_name = "initial_cached_step0"
    aligned = align_probe_set(h_to_rows)
    grid_rows, grid_by_h = compute_grid_metrics(precision, checkpoint_name, aligned)
    d_dim = infer_d_trainable(aligned)
    ref_run = choose_reference_run(runs, precision, 1e-3)
    checkpoint_path_text, model_path, ckpt_warnings = checkpoint_model_path(ref_run, checkpoint_step)
    for w in ckpt_warnings:
        diagnostics.setdefault("warnings", []).append(f"{precision}/{checkpoint_name}: {w}")
    ulp_stats = estimate_ulp_stats(model_path, precision, diagnostics)
    if not ulp_stats:
        diagnostics.setdefault("warnings", []).append(f"{precision}/{checkpoint_name}: Delta ULP estimation failed")

    g_estimators = [select_code_g(aligned), select_abs_g(aligned), select_richardson_g(aligned)]
    l_est = estimate_l(aligned)
    for flag in l_est.get("flags", []):
        diagnostics.setdefault("warnings", []).append(f"{precision}/{checkpoint_name}: {flag}")

    delta_modes = {
        "ulp_rms": ulp_stats.get("ulp_rms"),
        "ulp_median": ulp_stats.get("ulp_median"),
        "ulp_p90": ulp_stats.get("ulp_p90"),
    }
    delta_modes = {k: v for k, v in delta_modes.items() if v is not None and math.isfinite(float(v)) and float(v) > 0.0}
    if "empirical_snap_rms" not in delta_modes:
        diagnostics.setdefault("warnings", []).append(
            f"{precision}/{checkpoint_name}: empirical_snap_rms skipped because cached probes do not contain per-coordinate actual_delta-intended_delta"
        )

    nmse_vals = [(h, row.get("nmse")) for h, row in grid_by_h.items() if math.isfinite(float(row.get("nmse", float("nan"))))]
    corr_vals = [(h, row.get("corr")) for h, row in grid_by_h.items() if math.isfinite(float(row.get("corr", float("nan"))))]
    emp_min_h, emp_min_nmse = min(nmse_vals, key=lambda x: float(x[1])) if nmse_vals else (float("nan"), float("nan"))
    emp_corr_h, emp_corr = max(corr_vals, key=lambda x: float(x[1])) if corr_vals else (float("nan"), float("nan"))

    estimate_rows: List[Dict[str, Any]] = []
    for delta_mode, delta_val in delta_modes.items():
        for g_info in g_estimators:
            for l_q in ["q50", "q90", "q95"]:
                l_val = safe_float(l_est.get(l_q))
                g_val = safe_float(g_info.get("G_hat"))
                if l_val is None or g_val is None:
                    continue
                h_cont = hstar_value(float(delta_val), float(g_val), float(l_val), d_dim)
                if not math.isfinite(h_cont):
                    continue
                h_clipped = min(max(h_cont, min(H_GRID)), max(H_GRID))
                h_nearest = nearest_grid_h(h_cont)
                out_of_grid = bool(h_cont < min(H_GRID) or h_cont > max(H_GRID))
                if out_of_grid and delta_mode == "ulp_rms" and l_q == "q90":
                    diagnostics.setdefault("warnings", []).append(
                        f"{precision}/{checkpoint_name}/{g_info['method']}: "
                        f"hstar_cont={h_cont:.6g} outside h-grid [{min(H_GRID):.6g}, {max(H_GRID):.6g}]"
                    )
                nearest_metrics = grid_by_h.get(h_nearest, {})
                cont_metrics = grid_by_h.get(h_clipped, {}) if h_clipped in grid_by_h else {}
                flags = []
                flags.extend(g_info.get("flags", []))
                flags.extend(l_est.get("flags", []))
                flags.extend(ckpt_warnings)
                if not cont_metrics:
                    flags.append("mse_cont_not_evaluated_offline; using nearest_grid for reported selector quality")
                if delta_mode != "ulp_rms" or l_q != "q90":
                    flags.append("sensitivity_row")
                estimate_rows.append(
                    {
                        "precision": precision,
                        "checkpoint_name": checkpoint_name,
                        "checkpoint_path": checkpoint_path_text or "cached_probe_no_model_path",
                        "reference_run_h": ref_run.h if ref_run else "",
                        "dtype": "float16" if precision == "fp16" else "float32",
                        "d_trainable": d_dim,
                        "direction_normalization": "raw Gaussian unnormalized; d inferred from nominal_delta_norm/(2h)",
                        "selector_name": selector_name(str(g_info["method"]), delta_mode, l_q),
                        "delta_mode": delta_mode,
                        "Delta_value": float(delta_val),
                        "G_method": g_info["method"],
                        "G_hat": g_val,
                        "G_h_used": g_info.get("h_used"),
                        "G_snr": g_info.get("snr"),
                        "G_stability": g_info.get("stability"),
                        "L_method": l_est.get("method"),
                        "L_hat_q": l_q,
                        "L_hat": l_val,
                        "L_h2_used": l_est.get("h_used"),
                        "L_snr2": l_est.get("snr2"),
                        "hstar_cont": h_cont,
                        "hstar_clipped": h_clipped,
                        "hstar_nearest_grid": h_nearest,
                        "hstar_out_of_grid_flag": out_of_grid,
                        "fallback_flags": ";".join(flags),
                        "mse_cont": cont_metrics.get("mse", ""),
                        "nmse_cont": cont_metrics.get("nmse", ""),
                        "corr_cont": cont_metrics.get("corr", ""),
                        "mse_nearest": nearest_metrics.get("mse", ""),
                        "nmse_nearest": nearest_metrics.get("nmse", ""),
                        "corr_nearest": nearest_metrics.get("corr", ""),
                        "empirical_min_mse_h": emp_min_h,
                        "empirical_min_nmse": emp_min_nmse,
                        "empirical_max_corr_h": emp_corr_h,
                        "empirical_max_corr": emp_corr,
                    }
                )

    detail = {
        "precision": precision,
        "checkpoint_name": checkpoint_name,
        "d_trainable": d_dim,
        "reference_run": ref_run.run_name if ref_run else None,
        "delta_model_path": str(model_path) if model_path else None,
        "ulp_stats": ulp_stats,
        "G_estimators": g_estimators,
        "L_estimator": l_est,
        "empirical_min_mse_h": emp_min_h,
        "empirical_min_nmse": emp_min_nmse,
        "empirical_max_corr_h": emp_corr_h,
        "empirical_max_corr": emp_corr,
        "aligned_num_directions": {h_key(h): len(rows) for h, rows in aligned.items()},
    }
    return estimate_rows, grid_rows, detail


def git_commit(repo_root: Path) -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(repo_root), text=True).strip()
    except Exception:
        return ""


def make_plots(out_dir: Path, grid_rows: Sequence[Dict[str, Any]], estimate_rows: Sequence[Dict[str, Any]]) -> None:
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        (plot_dir / "README.md").write_text(
            "# H-Star Plots\n\n"
            f"PNG plots were not generated because matplotlib is unavailable: `{type(exc).__name__}: {exc}`.\n"
            "Use `hstar_grid_mse.csv` and `hstar_estimates.csv` to generate plots in an environment with matplotlib.\n",
            encoding="utf-8",
        )
        return
    by_pc: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for row in grid_rows:
        by_pc.setdefault((str(row["precision"]), str(row["checkpoint"])), []).append(row)
    for (precision, checkpoint), rows in by_pc.items():
        rows = sorted(rows, key=lambda r: float(r["h"]))
        ests = [
            r
            for r in estimate_rows
            if str(r["precision"]) == precision
            and str(r["checkpoint_name"]) == checkpoint
            and str(r["delta_mode"]) == "ulp_rms"
            and str(r["L_hat_q"]) == "q90"
        ]
        for metric_name, ylabel in [("nmse", "nMSE"), ("corr", "corr")]:
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.plot([float(r["h"]) for r in rows], [float(r[metric_name]) for r in rows], marker="o", label=metric_name)
            for est in ests:
                h = safe_float(est.get("hstar_nearest_grid"))
                if h is None:
                    continue
                ax.axvline(h, linestyle="--", alpha=0.4, label=str(est.get("G_method")))
            ax.set_xscale("log")
            ax.set_xlabel("h")
            ax.set_ylabel(ylabel)
            ax.set_title(f"{precision} {checkpoint} {metric_name}")
            ax.grid(True, alpha=0.3)
            handles, labels = ax.get_legend_handles_labels()
            seen = set()
            unique = []
            for handle, label in zip(handles, labels):
                if label in seen:
                    continue
                seen.add(label)
                unique.append((handle, label))
            if unique:
                ax.legend([h for h, _ in unique], [l for _, l in unique], fontsize=8)
            fig.tight_layout()
            fig.savefig(plot_dir / f"{precision}_{checkpoint}_{metric_name}_vs_h.png")
            plt.close(fig)


def write_summary_md(out_dir: Path, estimate_rows: Sequence[Dict[str, Any]], grid_rows: Sequence[Dict[str, Any]], diagnostics: Dict[str, Any]) -> None:
    main = [
        r
        for r in estimate_rows
        if str(r.get("delta_mode")) == "ulp_rms"
        and str(r.get("L_hat_q")) == "q90"
        and str(r.get("G_method")) in {"codeG", "absG", "richardsonG"}
    ]
    lines = [
        "# Offline H-Star Analysis: FP32/FP16 RoBERTa-large SST-5",
        "",
        f"Analysis directory: `{out_dir}`",
        "",
        "This is an offline analysis over cached checkpoint probe JSONL files. It did not launch training.",
        "",
        "## Main Selector Table",
        "",
        "| precision | checkpoint | selector | hstar_cont | nearest h | nmse | corr | empirical best h | empirical best nmse |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(main, key=lambda r: (str(r["precision"]), str(r["checkpoint_name"]), str(r["G_method"]))):
        lines.append(
            "| {precision} | {checkpoint_name} | {selector_name} | {hstar_cont:.4g} | {hstar_nearest_grid:.4g} | {nmse_nearest:.4g} | {corr_nearest:.4g} | {empirical_min_mse_h:.4g} | {empirical_min_nmse:.4g} |".format(
                **{k: (float(v) if k not in {"precision", "checkpoint_name", "selector_name"} else v) for k, v in row.items() if k in {
                    "precision",
                    "checkpoint_name",
                    "selector_name",
                    "hstar_cont",
                    "hstar_nearest_grid",
                    "nmse_nearest",
                    "corr_nearest",
                    "empirical_min_mse_h",
                    "empirical_min_nmse",
                }}
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- The saved checkpoint probes cover `checkpoint_step=0` for every h. Step-1000/final trajectory probe curves were not cached, so they are listed as skipped rather than recomputed.",
            "- `codeG` is the existing two-point estimator formula found in `medium_models/src/trainer.py`: `sqrt(pi/2) * mean(abs(d_hat))`; it is not a signed mean.",
            "- `absG` adds an h-stability selector around the same absolute-moment G estimator.",
            "- `richardsonG` uses cached `d2(h)` and `d2(2h)` pairs to form `(4*d2(h)-d2(2h))/3`; this is empirical and separate from the old theorem.",
            "- `L` uses a cached symmetric-curvature proxy `(loss_plus - 2*loss_base + loss_minus)/h^2` because the shared-step `theta+delta, theta+2delta` losses were not cached.",
            "- Continuous h-star values outside the grid are clipped only for the `hstar_clipped` column; the quality table reports nearest-grid cached MSE/corr.",
            "",
            "## Warnings",
        ]
    )
    warnings = diagnostics.get("warnings", [])
    if warnings:
        for warning in warnings:
            lines.append(f"- {warning}")
    else:
        lines.append("- None.")
    lines.extend(
        [
            "",
            "## Output Files",
            "",
            "- `hstar_estimates.csv`",
            "- `hstar_grid_mse.csv`",
            "- `hstar_diagnostics.json`",
            "- `plots/`",
        ]
    )
    (out_dir / "hstar_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--search-root",
        action="append",
        default=[],
        help="Root(s) to search for run_config.json. Defaults to experiments/main_latest.",
    )
    parser.add_argument("--output-dir", default="", help="Output directory. Defaults to analysis/hstar_fp32_fp16_<timestamp>.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path.cwd().resolve()
    search_roots = [Path(p).resolve() for p in args.search_root] if args.search_root else [repo_root / "experiments" / "main_latest"]
    out_dir = Path(args.output_dir).resolve() if args.output_dir else repo_root / "analysis" / f"hstar_fp32_fp16_{now_stamp()}"
    out_dir.mkdir(parents=True, exist_ok=False)

    diagnostics: Dict[str, Any] = {
        "analysis_output_dir": str(out_dir),
        "search_roots": [str(p) for p in search_roots],
        "git_commit": git_commit(repo_root),
        "h_grid": H_GRID,
        "codeG_formula_found": "medium_models/src/trainer.py::_estimate_two_point_g_raw returns sqrt(pi/2) * mean(abs(d_hat)) over two-point finite differences; no signed mean is used.",
        "direction_convention_found": "raw Gaussian torch.normal(mean=0,std=1) per trainable parameter, unnormalized; probe_window uses _zo_materialize_random_vector and efficient_perturb_parameters.",
        "warnings": [],
        "skipped_checkpoints": [],
    }

    runs = discover_runs(search_roots, diagnostics)
    probe_sets = build_probe_sets(runs, diagnostics)

    estimate_rows: List[Dict[str, Any]] = []
    grid_rows: List[Dict[str, Any]] = []
    details: List[Dict[str, Any]] = []
    for precision in ["fp32", "fp16"]:
        for checkpoint_step in sorted(step for (prec, step) in probe_sets if prec == precision):
            h_to_rows = probe_sets[(precision, checkpoint_step)]
            missing = [h for h in H_GRID if find_h(h, h_to_rows.keys()) is None]
            if missing:
                diagnostics.setdefault("skipped_checkpoints", []).append(
                    {
                        "precision": precision,
                        "checkpoint_step": checkpoint_step,
                        "reason": f"missing h values: {missing}",
                    }
                )
                continue
            est, grid, detail = estimate_rows_for_set(precision, checkpoint_step, h_to_rows, runs, diagnostics)
            estimate_rows.extend(est)
            grid_rows.extend(grid)
            details.append(detail)

    for precision in ["fp32", "fp16"]:
        steps = sorted(step for (prec, step) in probe_sets if prec == precision)
        if 1000 not in steps:
            diagnostics.setdefault("skipped_checkpoints", []).append(
                {"precision": precision, "checkpoint": "step_1000", "reason": "no cached probe JSONL rows at checkpoint_step=1000"}
            )
        if 20000 not in steps:
            diagnostics.setdefault("skipped_checkpoints", []).append(
                {"precision": precision, "checkpoint": "final_or_step_20000", "reason": "no cached probe JSONL rows at checkpoint_step=20000/final"}
            )

    diagnostics["analysis_details"] = details
    diagnostics["run_discovery_summary"] = {
        "runs_used": len(runs),
        "probe_sets": [f"{prec}:{step}" for (prec, step) in sorted(probe_sets)],
        "estimate_rows": len(estimate_rows),
        "grid_rows": len(grid_rows),
    }
    diagnostics["compute_environment"] = {
        "cwd": str(repo_root),
        "python": os.environ.get("CONDA_PREFIX", ""),
        "hostname": os.uname().nodename if hasattr(os, "uname") else "",
    }

    write_csv(out_dir / "hstar_estimates.csv", estimate_rows, HSTAR_FIELDS)
    write_csv(out_dir / "hstar_grid_mse.csv", grid_rows, GRID_FIELDS)
    (out_dir / "hstar_diagnostics.json").write_text(json.dumps(diagnostics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_summary_md(out_dir, estimate_rows, grid_rows, diagnostics)
    make_plots(out_dir, grid_rows, estimate_rows)

    print(f"Analysis output directory: {out_dir}")
    for precision in ["fp32", "fp16"]:
        print(f"{precision.upper()}:")
        main_rows = [
            r
            for r in estimate_rows
            if r.get("precision") == precision and r.get("checkpoint_name") == "initial_cached_step0" and r.get("delta_mode") == "ulp_rms" and r.get("L_hat_q") == "q90"
        ]
        emp = next(iter(main_rows), None)
        if emp:
            print(f"  empirical min-MSE h: {float(emp['empirical_min_mse_h']):.6g} (nMSE={float(emp['empirical_min_nmse']):.6g})")
        for method in ["codeG", "absG", "richardsonG"]:
            row = next((r for r in main_rows if r.get("G_method") == method), None)
            if not row:
                continue
            print(
                f"  old_hstar_{method} nearest h: {float(row['hstar_nearest_grid']):.6g}, "
                f"nmse: {float(row['nmse_nearest']):.6g}, corr: {float(row['corr_nearest']):.6g}"
            )
    warnings = diagnostics.get("warnings", [])
    if warnings:
        print("Warnings:")
        for warning in warnings[:20]:
            print(f"  - {warning}")
        if len(warnings) > 20:
            print(f"  - ... {len(warnings) - 20} more warnings in hstar_diagnostics.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
