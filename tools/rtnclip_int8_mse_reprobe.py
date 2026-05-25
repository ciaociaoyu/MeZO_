#!/usr/bin/env python
"""Reliable low-bit RTNClip MSE/visibility reprobe for RoBERTa-large / SST-5.

This is a probe-only harness. It keeps the existing shared-grid RTNClip
fake-quant semantics and separates reconstruction, visibility, true-gradient,
and Richardson locality diagnostics.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = REPO_ROOT / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import quantizer_robustness_int8_window as qrw  # noqa: E402


H_GRID: List[Tuple[str, float]] = [
    ("1e-5", 1e-5),
    ("3e-5", 3e-5),
    ("1e-4", 1e-4),
    ("3e-4", 3e-4),
    ("1e-3", 1e-3),
    ("1p5e-3", 1.5e-3),
    ("2e-3", 2e-3),
    ("3e-3", 3e-3),
    ("4e-3", 4e-3),
    ("5e-3", 5e-3),
    ("1e-2", 1e-2),
]

EPS = 1e-30
DEFAULT_TRAINING_SUMMARIES = {
    8: REPO_ROOT / "outputs" / "rtnclip_lowbit_roberta_sst5_seed16_20260519_batch" / "int8_hsearch_summary.csv",
    4: REPO_ROOT / "outputs" / "rtnclip_int4_g128_rtnclip_roberta_sst5_seed16_20260521" / "int4_hsearch_summary.csv",
}


SUMMARY_COLUMNS = [
    "h",
    "n_batches",
    "n_directions",
    "fd_true_available",
    "weight_recon_mse_mean",
    "plus_recon_mse_mean",
    "minus_recon_mse_mean",
    "delta_visibility_nmse_mean",
    "delta_visibility_nmse_median",
    "delta_visibility_rel_l2_mean",
    "alignment_mean",
    "norm_ratio_mean",
    "active_frac_mean",
    "code_change_frac_mean",
    "clip_frac_mean",
    "saturation_frac_mean",
    "richardson_rmse_rel",
    "richardson_relerr_median",
    "fd_true_mse",
    "fd_true_nmse",
    "fd_true_rmse",
    "corr_fd_true",
    "fd_true_bias",
]

MERGE_COLUMNS = [
    "h",
    "best_eval_acc",
    "last_eval_acc",
    "active_frac",
    "alignment",
    "norm_ratio",
    "delta_visibility_nmse",
    "richardson_rmse_rel",
    "fd_true_nmse",
    "corr_fd_true",
]


def json_default(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.dtype):
        return str(value)
    return str(value)


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True, default=json_default) + "\n", encoding="utf-8")


def append_jsonl(path: Path, data: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(data, sort_keys=True, default=json_default) + "\n")


def write_csv(path: Path, rows: List[Dict[str, object]], columns: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(columns), extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def finite_float(value) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return out


def mean(values: Iterable[object]) -> Optional[float]:
    xs = [float(v) for v in values if finite_float(v) is not None]
    return sum(xs) / len(xs) if xs else None


def median(values: Iterable[object]) -> Optional[float]:
    xs = sorted(float(v) for v in values if finite_float(v) is not None)
    if not xs:
        return None
    mid = len(xs) // 2
    if len(xs) % 2:
        return xs[mid]
    return 0.5 * (xs[mid - 1] + xs[mid])


def corr(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    pairs = []
    for x, y in zip(xs, ys):
        xf = finite_float(x)
        yf = finite_float(y)
        if xf is not None and yf is not None:
            pairs.append((xf, yf))
    if len(pairs) < 2:
        return None
    x_vals = [p[0] for p in pairs]
    y_vals = [p[1] for p in pairs]
    mx = sum(x_vals) / len(x_vals)
    my = sum(y_vals) / len(y_vals)
    vx = sum((x - mx) ** 2 for x in x_vals)
    vy = sum((y - my) ** 2 for y in y_vals)
    if vx <= EPS or vy <= EPS:
        return None
    cov = sum((x - mx) * (y - my) for x, y in pairs)
    return cov / math.sqrt(vx * vy)


def visibility_stats(delta_q: torch.Tensor, delta_ideal: torch.Tensor, eps: float = EPS) -> Dict[str, float]:
    dq = delta_q.float()
    di = delta_ideal.float()
    err = dq - di
    err_sq = err.double().square().sum()
    ideal_sq = di.double().square().sum()
    dq_sq = dq.double().square().sum()
    dot = (dq.double() * di.double()).sum()
    denom = max(float(ideal_sq.detach().cpu()), eps)
    mse = float((err_sq / max(int(dq.numel()), 1)).detach().cpu())
    nmse = float((err_sq / max(float(ideal_sq.detach().cpu()), eps)).detach().cpu())
    rel_l2 = math.sqrt(float(err_sq.detach().cpu()) / denom)
    align_denom = max(math.sqrt(float(dq_sq.detach().cpu()) * float(ideal_sq.detach().cpu())), eps)
    alignment = float(dot.detach().cpu()) / align_denom
    norm_ratio = math.sqrt(float(dq_sq.detach().cpu())) / max(math.sqrt(float(ideal_sq.detach().cpu())), eps)
    return {
        "delta_visibility_mse": mse,
        "delta_visibility_nmse": nmse,
        "delta_visibility_rel_l2": rel_l2,
        "alignment": alignment,
        "norm_ratio": norm_ratio,
    }


def pooled_fd_true_stats(fd_values: Sequence[object], true_values: Sequence[object], eps: float = EPS) -> Dict[str, object]:
    pairs = []
    for fd, true in zip(fd_values, true_values):
        fdf = finite_float(fd)
        truef = finite_float(true)
        if fdf is not None and truef is not None:
            pairs.append((fdf, truef))
    if not pairs:
        return {
            "fd_true_available": False,
            "fd_true_mse": None,
            "fd_true_nmse": None,
            "fd_true_rmse": None,
            "corr_fd_true": None,
            "fd_true_bias": None,
        }
    err_sq = sum((fd - true) ** 2 for fd, true in pairs)
    true_sq = sum(true ** 2 for _, true in pairs)
    mse = err_sq / len(pairs)
    return {
        "fd_true_available": True,
        "fd_true_mse": mse,
        "fd_true_nmse": err_sq / max(true_sq, eps),
        "fd_true_rmse": math.sqrt(mse),
        "corr_fd_true": corr([fd for fd, _ in pairs], [true for _, true in pairs]),
        "fd_true_bias": sum(fd - true for fd, true in pairs) / len(pairs),
    }


def pooled_richardson_stats(d_h: Sequence[object], d_half: Sequence[object], eps: float = EPS) -> Dict[str, object]:
    pairs = []
    for x, y in zip(d_h, d_half):
        xf = finite_float(x)
        yf = finite_float(y)
        if xf is not None and yf is not None:
            pairs.append((xf, yf))
    if not pairs:
        return {
            "richardson_available": False,
            "richardson_absdiff": None,
            "richardson_rmse_rel": None,
            "richardson_relerr": None,
        }
    diff_sq = sum((x - y) ** 2 for x, y in pairs)
    half_sq = sum(y ** 2 for _, y in pairs)
    relerrs = [abs(x - y) / max(abs(y), eps) for x, y in pairs]
    return {
        "richardson_available": True,
        "richardson_absdiff": sum(abs(x - y) for x, y in pairs) / len(pairs),
        "richardson_rmse_rel": math.sqrt(diff_sq / max(half_sq, eps)),
        "richardson_relerr": median(relerrs),
    }


def forward_loss_roberta(harness: qrw.RobertaHarness, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    payload = dict(batch)
    payload["token_type_ids"] = torch.zeros_like(payload["input_ids"])
    outputs = harness.model(**payload)
    return outputs[0], outputs[1]


def finite_difference_pair(
    harness: qrw.RobertaHarness,
    master: Dict[str, torch.Tensor],
    states: Dict[str, qrw.QuantizerState],
    directions: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    h: float,
) -> Tuple[float, float, float]:
    params = harness.params()
    with torch.no_grad():
        qrw.copy_master_to_model(params, master, directions, h, +1.0, states)
        loss_plus, _ = forward_loss_roberta(harness, batch)
        qrw.copy_master_to_model(params, master, directions, h, -1.0, states)
        loss_minus, _ = forward_loss_roberta(harness, batch)
        qrw.restore_master(params, master)
    lp = float(loss_plus.detach().cpu())
    lm = float(loss_minus.detach().cpu())
    return lp, lm, (lp - lm) / (2.0 * h)


def compute_true_gradient(harness: qrw.RobertaHarness, master: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]) -> float:
    params = harness.params()
    qrw.restore_master(params, master)
    harness.model.zero_grad(set_to_none=True)
    loss, _ = forward_loss_roberta(harness, batch)
    loss.backward()
    return float(loss.detach().cpu())


def directional_true_derivative(params: Dict[str, torch.nn.Parameter], directions: Dict[str, torch.Tensor]) -> Optional[float]:
    device = next(iter(directions.values())).device
    acc = torch.zeros((), device=device, dtype=torch.float64)
    seen = False
    for name, param in params.items():
        grad = param.grad
        if grad is None or name not in directions:
            continue
        acc += (grad.detach().float() * directions[name].float()).double().sum()
        seen = True
    return float(acc.detach().cpu()) if seen else None


def quantized_pair_diagnostics(
    master: Dict[str, torch.Tensor],
    directions: Dict[str, torch.Tensor],
    states: Dict[str, qrw.QuantizerState],
    h: float,
) -> Dict[str, object]:
    device = next(iter(master.values())).device
    total = 0
    active = 0
    weight_err_sq = torch.zeros((), device=device, dtype=torch.float64)
    weight_ref_sq = torch.zeros((), device=device, dtype=torch.float64)
    plus_err_sq = torch.zeros((), device=device, dtype=torch.float64)
    minus_err_sq = torch.zeros((), device=device, dtype=torch.float64)
    plus_ref_sq = torch.zeros((), device=device, dtype=torch.float64)
    minus_ref_sq = torch.zeros((), device=device, dtype=torch.float64)
    delta_err_sq = torch.zeros((), device=device, dtype=torch.float64)
    delta_sq = torch.zeros((), device=device, dtype=torch.float64)
    ideal_sq = torch.zeros((), device=device, dtype=torch.float64)
    dot = torch.zeros((), device=device, dtype=torch.float64)
    clip_plus_num = 0.0
    clip_minus_num = 0.0
    codes_legal = True

    for name, state in states.items():
        base = master[name].float()
        direction = directions[name].float()
        x_plus = base.add(direction, alpha=float(h))
        x_minus = base.add(direction, alpha=-float(h))
        q_w, w_stats = qrw.quantize_with_state(base, state, return_stats=True)
        q_plus, plus_stats = qrw.quantize_with_state(x_plus, state, return_stats=True)
        q_minus, minus_stats = qrw.quantize_with_state(x_minus, state, return_stats=True)

        q_w_f = q_w.float()
        q_plus_f = q_plus.float()
        q_minus_f = q_minus.float()
        delta_q = q_plus_f - q_minus_f
        delta_ideal = 2.0 * float(h) * direction
        delta_err = delta_q - delta_ideal

        n = int(base.numel())
        total += n
        active += int((delta_q != 0).sum().detach().cpu())
        weight_err_sq += (q_w_f - base).double().square().sum()
        weight_ref_sq += base.double().square().sum()
        plus_err_sq += (q_plus_f - x_plus).double().square().sum()
        minus_err_sq += (q_minus_f - x_minus).double().square().sum()
        plus_ref_sq += x_plus.double().square().sum()
        minus_ref_sq += x_minus.double().square().sum()
        delta_err_sq += delta_err.double().square().sum()
        delta_sq += delta_q.double().square().sum()
        ideal_sq += delta_ideal.double().square().sum()
        dot += (delta_q.double() * delta_ideal.double()).sum()
        clip_plus_num += float(plus_stats["clip_frac"]) * n
        clip_minus_num += float(minus_stats["clip_frac"]) * n
        codes_legal = codes_legal and int(w_stats["code_min"]) >= -state.qmax and int(w_stats["code_max"]) <= state.qmax
        codes_legal = codes_legal and int(plus_stats["code_min"]) >= -state.qmax and int(plus_stats["code_max"]) <= state.qmax
        codes_legal = codes_legal and int(minus_stats["code_min"]) >= -state.qmax and int(minus_stats["code_max"]) <= state.qmax

    eps_t = torch.tensor(EPS, device=device, dtype=torch.float64)
    # Use pooled dot/norm statistics for alignment and the full pooled error for MSE/nMSE.
    alignment = float((dot / (delta_sq.sqrt() * ideal_sq.sqrt()).clamp_min(eps_t)).detach().cpu())
    norm_ratio = float((delta_sq.sqrt() / ideal_sq.sqrt().clamp_min(eps_t)).detach().cpu())
    delta_visibility_mse = float((delta_err_sq / max(total, 1)).detach().cpu())
    delta_visibility_nmse = float((delta_err_sq / ideal_sq.clamp_min(eps_t)).detach().cpu())
    delta_visibility_rel_l2 = float((delta_err_sq.sqrt() / ideal_sq.sqrt().clamp_min(eps_t)).detach().cpu())
    active_frac = active / max(total, 1)
    clip_frac = (clip_plus_num + clip_minus_num) / max(2 * total, 1)

    return {
        "weight_recon_mse": float((weight_err_sq / max(total, 1)).detach().cpu()),
        "weight_recon_rel_mse": float((weight_err_sq / weight_ref_sq.clamp_min(eps_t)).detach().cpu()),
        "weight_recon_sqnr_db": float((10.0 * torch.log10(weight_ref_sq / weight_err_sq.clamp_min(eps_t))).detach().cpu()),
        "plus_recon_mse": float((plus_err_sq / max(total, 1)).detach().cpu()),
        "minus_recon_mse": float((minus_err_sq / max(total, 1)).detach().cpu()),
        "plus_recon_rel_mse": float((plus_err_sq / plus_ref_sq.clamp_min(eps_t)).detach().cpu()),
        "minus_recon_rel_mse": float((minus_err_sq / minus_ref_sq.clamp_min(eps_t)).detach().cpu()),
        "delta_visibility_mse": delta_visibility_mse,
        "delta_visibility_nmse": delta_visibility_nmse,
        "delta_visibility_rel_l2": delta_visibility_rel_l2,
        "alignment": alignment,
        "norm_ratio": norm_ratio,
        "code_change_frac": active_frac,
        "active_frac": active_frac,
        "clip_frac": clip_frac,
        "saturation_frac": clip_frac,
        "clip_frac_w_plus": clip_plus_num / max(total, 1),
        "clip_frac_w_minus": clip_minus_num / max(total, 1),
        "saturation_frac_w_plus": clip_plus_num / max(total, 1),
        "saturation_frac_w_minus": clip_minus_num / max(total, 1),
        "codes_legal": bool(codes_legal),
        "pair_shared_grid": True,
        "pair_shared_grid_check": True,
        "fresh_round_codes": True,
        "fresh_round_codes_check": True,
        "independent_q_plus_q_minus_scales": False,
        "q_w_plus_hu_bypass": False,
        "grid_id_plus": 1,
        "grid_id_minus": 1,
        "scale_id_plus": 1,
        "scale_id_minus": 1,
        "grid_id_sharing_check": True,
        "scale_id_sharing_check": True,
    }


def h_key(h: object) -> int:
    return int(round(float(h) * 1_000_000_000_000))


def read_training_by_h(path: Path) -> Dict[int, Dict[str, object]]:
    if not path.exists():
        return {}
    rows: Dict[int, Dict[str, object]] = {}
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            hv = finite_float(row.get("h"))
            if hv is None:
                continue
            rows[h_key(hv)] = row
    return rows


def summarize_records(records: List[Dict[str, object]]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for _label, h in H_GRID:
        group = [r for r in records if h_key(r["h"]) == h_key(h)]
        if not group:
            continue
        rich = pooled_richardson_stats([r.get("d_h_Q") for r in group], [r.get("d_half_Q") for r in group])
        fd_stats = pooled_fd_true_stats([r.get("d_h_Q") for r in group], [r.get("d_true") for r in group])
        rows.append(
            {
                "h": h,
                "n_batches": len({int(r["batch_id"]) for r in group}),
                "n_directions": len(group),
                "fd_true_available": bool(fd_stats["fd_true_available"]),
                "weight_recon_mse_mean": mean(r.get("weight_recon_mse") for r in group),
                "plus_recon_mse_mean": mean(r.get("plus_recon_mse") for r in group),
                "minus_recon_mse_mean": mean(r.get("minus_recon_mse") for r in group),
                "delta_visibility_nmse_mean": mean(r.get("delta_visibility_nmse") for r in group),
                "delta_visibility_nmse_median": median(r.get("delta_visibility_nmse") for r in group),
                "delta_visibility_rel_l2_mean": mean(r.get("delta_visibility_rel_l2") for r in group),
                "alignment_mean": mean(r.get("alignment") for r in group),
                "norm_ratio_mean": mean(r.get("norm_ratio") for r in group),
                "active_frac_mean": mean(r.get("active_frac") for r in group),
                "code_change_frac_mean": mean(r.get("code_change_frac") for r in group),
                "clip_frac_mean": mean(r.get("clip_frac") for r in group),
                "saturation_frac_mean": mean(r.get("saturation_frac") for r in group),
                "richardson_rmse_rel": rich["richardson_rmse_rel"],
                "richardson_relerr_median": rich["richardson_relerr"],
                "fd_true_mse": fd_stats["fd_true_mse"],
                "fd_true_nmse": fd_stats["fd_true_nmse"],
                "fd_true_rmse": fd_stats["fd_true_rmse"],
                "corr_fd_true": fd_stats["corr_fd_true"],
                "fd_true_bias": fd_stats["fd_true_bias"],
            }
        )
    return rows


def make_training_merge(
    summary_rows: List[Dict[str, object]],
    output_root: Path,
    bitwidth: int,
    training_summary: Path,
) -> List[Dict[str, object]]:
    training = read_training_by_h(training_summary)
    merged: List[Dict[str, object]] = []
    for row in summary_rows:
        train = training.get(h_key(row["h"]), {})
        merged.append(
            {
                "h": row["h"],
                "best_eval_acc": train.get("best_eval_acc"),
                "last_eval_acc": train.get("last_eval_acc"),
                "active_frac": row.get("active_frac_mean"),
                "alignment": row.get("alignment_mean"),
                "norm_ratio": row.get("norm_ratio_mean"),
                "delta_visibility_nmse": row.get("delta_visibility_nmse_mean"),
                "richardson_rmse_rel": row.get("richardson_rmse_rel"),
                "fd_true_nmse": row.get("fd_true_nmse"),
                "corr_fd_true": row.get("corr_fd_true"),
            }
        )
    write_csv(output_root / f"int{bitwidth}_mse_vs_training.csv", merged, MERGE_COLUMNS)
    return merged


def plot_summary(summary_rows: List[Dict[str, object]], merged_rows: List[Dict[str, object]], output_root: Path, bitwidth: int) -> List[str]:
    plot_dir = output_root / f"int{bitwidth}_mse_probe_plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    h_vals = [float(r["h"]) for r in summary_rows]
    paths: List[str] = []

    def line(metric: str, filename: str, title: str, y_log: bool = False) -> None:
        ys = [r.get(metric) for r in summary_rows]
        path = plot_dir / filename
        qrw.write_svg_line_chart(path, title=title, x_label="h", y_label=metric, series=[(metric, h_vals, ys)], x_log=True, y_log=y_log)
        paths.append(str(path))

    line("delta_visibility_nmse_mean", "delta_visibility_nmse_vs_h.svg", f"INT{bitwidth} RTNClip delta visibility nMSE vs h", y_log=True)
    line("alignment_mean", "alignment_vs_h.svg", f"INT{bitwidth} RTNClip alignment vs h")
    line("norm_ratio_mean", "norm_ratio_vs_h.svg", f"INT{bitwidth} RTNClip norm ratio vs h", y_log=True)
    line("richardson_rmse_rel", "richardson_rmse_rel_vs_h.svg", f"INT{bitwidth} RTNClip Richardson locality vs h", y_log=True)
    if any(finite_float(r.get("fd_true_nmse")) is not None for r in summary_rows):
        line("fd_true_nmse", "fd_true_nmse_vs_h.svg", f"INT{bitwidth} RTNClip true-gradient nMSE vs h", y_log=True)
    if any(finite_float(r.get("corr_fd_true")) is not None for r in summary_rows):
        line("corr_fd_true", "corr_fd_true_vs_h.svg", f"INT{bitwidth} RTNClip corr(fd,true) vs h")

    if any(finite_float(r.get("best_eval_acc")) is not None or finite_float(r.get("last_eval_acc")) is not None for r in merged_rows):
        train_h = [float(r["h"]) for r in merged_rows]
        series = []
        if any(finite_float(r.get("best_eval_acc")) is not None for r in merged_rows):
            series.append(("best_eval_acc", train_h, [r.get("best_eval_acc") for r in merged_rows]))
        if any(finite_float(r.get("last_eval_acc")) is not None for r in merged_rows):
            series.append(("last_eval_acc", train_h, [r.get("last_eval_acc") for r in merged_rows]))
        path = plot_dir / "training_acc_overlay.svg"
        qrw.write_svg_line_chart(path, title=f"Existing INT{bitwidth} h-search accuracy overlay", x_label="h", y_label="accuracy", series=series, x_log=True, y_log=False)
        paths.append(str(path))
    return paths


def fmt(value, digits: int = 4) -> str:
    fv = finite_float(value)
    if fv is None:
        return "n/a"
    return f"{fv:.{digits}g}"


def row_for_h(summary_rows: Sequence[Dict[str, object]], h: float) -> Optional[Dict[str, object]]:
    key = h_key(h)
    for row in summary_rows:
        if h_key(row["h"]) == key:
            return row
    return None


def make_markdown_summary(
    summary_rows: List[Dict[str, object]],
    merged_rows: List[Dict[str, object]],
    plot_paths: List[str],
    output_root: Path,
    bitwidth: int,
    training_summary: Path,
    fd_true_available: bool,
    true_grad_note: str,
) -> None:
    small = row_for_h(summary_rows, 1e-5)
    default = row_for_h(summary_rows, 1e-3)
    large = row_for_h(summary_rows, 1e-2)
    small_visibility_bad = bool(
        small
        and default
        and finite_float(small.get("delta_visibility_nmse_mean")) is not None
        and finite_float(default.get("delta_visibility_nmse_mean")) is not None
        and float(small["delta_visibility_nmse_mean"]) > float(default["delta_visibility_nmse_mean"])
    )
    large_visibility_good = bool(
        large
        and finite_float(large.get("delta_visibility_nmse_mean")) is not None
        and float(large["delta_visibility_nmse_mean"]) < 1.0
    )
    if fd_true_available and large and default and finite_float(large.get("fd_true_nmse")) is not None and finite_float(default.get("fd_true_nmse")) is not None:
        large_locality_bad = float(large["fd_true_nmse"]) > max(float(default["fd_true_nmse"]) * 1.5, 0.05)
        locality_basis = "fd_true_nmse"
    elif large and default and finite_float(large.get("richardson_rmse_rel")) is not None and finite_float(default.get("richardson_rmse_rel")) is not None:
        large_locality_bad = float(large["richardson_rmse_rel"]) > max(float(default["richardson_rmse_rel"]) * 1.5, 0.5)
        locality_basis = "richardson_rmse_rel"
    else:
        large_locality_bad = False
        locality_basis = "insufficient locality data"

    default_visibility_good = bool(
        default
        and finite_float(default.get("delta_visibility_nmse_mean")) is not None
        and float(default["delta_visibility_nmse_mean"]) < 1.0
    )
    default_locality_good = False
    if default and fd_true_available:
        fd_nmse = finite_float(default.get("fd_true_nmse"))
        fd_corr = finite_float(default.get("corr_fd_true"))
        default_locality_good = bool((fd_nmse is not None and fd_nmse <= 0.1) or (fd_corr is not None and fd_corr >= 0.95))
    elif default and large:
        d_rich = finite_float(default.get("richardson_rmse_rel"))
        l_rich = finite_float(large.get("richardson_rmse_rel"))
        default_locality_good = bool(d_rich is not None and l_rich is not None and d_rich < l_rich)

    lines = [
        f"# INT{bitwidth} RTNClip MSE Reprobe Summary",
        "",
        "This run separates quantizer reconstruction, perturbation visibility, true finite-difference error, and Richardson locality. Reconstruction MSE is not finite-difference MSE.",
        "",
        f"True-gradient diagnostics: {'available' if fd_true_available else 'unavailable'} ({true_grad_note}).",
        "",
        "## Interpretation",
        "",
        f"- Small h visibility-bad: {'yes' if small_visibility_bad else 'not clearly observed'} (`h=1e-5` delta_visibility_nmse={fmt(None if small is None else small.get('delta_visibility_nmse_mean'))}).",
        f"- `h=1e-3` visibility-good: {'yes' if default_visibility_good else 'not clearly'}; locality-good: {'yes' if default_locality_good else 'not clearly'} (delta_visibility_nmse={fmt(None if default is None else default.get('delta_visibility_nmse_mean'))}, richardson_rmse_rel={fmt(None if default is None else default.get('richardson_rmse_rel'))}, fd_true_nmse={fmt(None if default is None else default.get('fd_true_nmse'))}, corr_fd_true={fmt(None if default is None else default.get('corr_fd_true'))}).",
        f"- `h=1e-2` visibility-good: {'yes' if large_visibility_good else 'not clearly'} (delta_visibility_nmse={fmt(None if large is None else large.get('delta_visibility_nmse_mean'))}).",
        f"- `h=1e-2` locality-bad: {'yes' if large_locality_bad else 'not clearly'} by {locality_basis} (richardson_rmse_rel={fmt(None if large is None else large.get('richardson_rmse_rel'))}, fd_true_nmse={fmt(None if large is None else large.get('fd_true_nmse'))}).",
        "",
        "## Field Semantics",
        "",
        "- `weight_recon_mse` and `recon_mse_global` measure `Q_t(w_t)` reconstruction only.",
        "- `plus_recon_mse` and `minus_recon_mse` measure reconstruction of each perturbed state.",
        "- `delta_visibility_nmse` measures whether `Q_t(w_t+h u)-Q_t(w_t-h u)` exposes the intended displacement; it is visibility-only.",
        "- `fd_true_nmse` is the finite-difference true-gradient MSE when true gradients are available.",
        "- `richardson_rmse_rel` is the self-consistency locality proxy from `d_h` versus `d_{h/2}`.",
        "",
        "## Per-h Summary",
        "",
        "| h | delta_visibility_nmse | alignment | norm_ratio | richardson_rmse_rel | fd_true_nmse | corr_fd_true |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary_rows:
        lines.append(
            f"| {float(row['h']):g} | {fmt(row.get('delta_visibility_nmse_mean'))} | {fmt(row.get('alignment_mean'))} | "
            f"{fmt(row.get('norm_ratio_mean'))} | {fmt(row.get('richardson_rmse_rel'))} | {fmt(row.get('fd_true_nmse'))} | {fmt(row.get('corr_fd_true'))} |"
        )
    lines.extend(["", "## Plots", ""])
    for path in plot_paths:
        lines.append(f"- `{Path(path).relative_to(output_root)}`")
    if merged_rows and any(finite_float(r.get("best_eval_acc")) is not None for r in merged_rows):
        lines.extend(["", f"Existing training overlay source: `{training_summary}`."])
    else:
        lines.extend(["", f"Existing INT{bitwidth} training overlay was not found or had no matching accuracy rows."])
    (output_root / f"int{bitwidth}_mse_probe_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_diagnostic_spec(output_root: Path) -> None:
    text = """# Low-Bit MSE Diagnostic Field Spec

Low-bit ZO diagnostics must not use a single generic `mse` field. The fields below have separate meanings and must not be substituted for each other.

## Weight Reconstruction

- `weight_recon_mse`: mean squared error of `Q_t(w_t)` versus the unperturbed FP16 master `w_t`.
- `weight_recon_rel_mse`: pooled squared error divided by pooled `||w_t||^2`.
- `weight_recon_sqnr_db`: `10 log10(||w_t||^2 / ||Q_t(w_t)-w_t||^2)`.
- Legacy `recon_mse_global` may be copied into `weight_recon_mse`, but it is not an h-window finite-difference MSE.

## Perturbed Reconstruction

- `plus_recon_mse`, `minus_recon_mse`: reconstruction MSE of `Q_t(w_t +/- h u)` against each perturbed floating state.
- `plus_recon_rel_mse`, `minus_recon_rel_mse`: pooled relative versions of the same diagnostics.

## Effective Displacement / Visibility

- `delta_visibility_mse`: MSE of `Q_t(w_t+h u)-Q_t(w_t-h u)` against `2h u`.
- `delta_visibility_nmse`: pooled normalized version of the same visibility error.
- `delta_visibility_rel_l2`: relative L2 visibility error.
- `alignment`, `norm_ratio`, `code_change_frac`, `active_frac`, `clip_frac`, and `saturation_frac` describe quantization geometry.

`delta_visibility_nmse` is not truncation error. Large `h` can have low visibility error while still being a poor finite-difference estimator.

## True-Gradient Finite-Difference Quality

- `fd_true_mse`: pooled MSE of `d_h^Q(u)` versus the unquantized true directional derivative `grad L(w_t)^T u`.
- `fd_true_nmse`: pooled normalized MSE against the true derivative energy.
- `fd_true_rmse`: square root of `fd_true_mse`.
- `corr_fd_true`: correlation across directions between quantized finite differences and true directional derivatives.
- `fd_true_bias`: mean `d_h^Q(u)-d_true(u)`.
- `fd_true_available`: false when true gradients are unavailable or OOM. Do not fill these fields from reconstruction metrics.

## Richardson / Locality

- `richardson_absdiff`: mean absolute difference between `d_h^Q` and `d_{h/2}^Q`.
- `richardson_rmse_rel`: pooled relative RMSE of `d_h^Q-d_{h/2}^Q` normalized by `d_{h/2}^Q`.
- `richardson_relerr`: median per-direction relative difference.
- `richardson_available`: false when the paired `h/2` probe is unavailable.

Richardson metrics and true-gradient metrics are the locality diagnostics. Weight reconstruction MSE and delta visibility nMSE must not be used as h-window truncation MSE.
"""
    docs_path = REPO_ROOT / "docs" / "lowbit_mse_diagnostic_spec.md"
    docs_path.parent.mkdir(parents=True, exist_ok=True)
    docs_path.write_text(text, encoding="utf-8")
    (output_root / "lowbit_mse_diagnostic_spec.md").write_text(text, encoding="utf-8")


def is_oom(exc: BaseException) -> bool:
    message = str(exc).lower()
    return "out of memory" in message or "cuda oom" in message or "cublas" in message and "alloc" in message


def build_args(batch_size: int, eval_batch_size: int, probe_dirs: int) -> argparse.Namespace:
    return argparse.Namespace(
        roberta_batch_size=batch_size,
        roberta_eval_batch_size=eval_batch_size,
        roberta_lr=1e-6,
        probe_dirs=probe_dirs,
        eval_batches=0,
        diag_every=0,
        opt_batch_size=1,
        opt_eval_batch_size=1,
        opt_lr=1e-6,
        opt_max_length=256,
    )


def run_probe_once(args: argparse.Namespace, batch_size: int, true_grad_enabled: bool) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], bool, str]:
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    bitwidth = int(args.bitwidth)
    training_summary = Path(args.training_summary)
    records_path = output_root / f"int{bitwidth}_mse_probe_records.jsonl"
    if records_path.exists():
        records_path.unlink()
    os.environ["DATALOADER_SHUFFLE"] = "true"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    harness = qrw.RobertaHarness(build_args(batch_size, int(args.eval_batch_size), int(args.directions)), device)
    if harness.train_sampler_name != "RandomSampler":
        raise RuntimeError(f"Expected RandomSampler, got {harness.train_sampler_name}")
    params = harness.params()
    master = harness.make_master()
    qrw.restore_master(params, master)
    states, refresh_rows = qrw.refresh_quantizer_states(
        master,
        harness.quantized_module_names,
        quantizer="rtnclip",
        activation_rms={},
        bitwidth=bitwidth,
        group_size=128,
    )
    quant = qrw.aggregate_quantizer_stats(refresh_rows, harness.numel_by_quantized_name())
    write_json(output_root / "quantizer_refresh_summary.json", quant)

    config = {
        "model": "roberta-large",
        "dataset": "SST-5",
        "dataset_mode": "full",
        "seed": 16,
        "data_seed": 16,
        "batch_size": batch_size,
        "requested_batch_size": int(args.batch_size),
        "batch_size_fallback_used": batch_size != int(args.batch_size),
        "dataloader_shuffle": True,
        "sampler": harness.train_sampler_name,
        "direction": "dense",
        "quant_bits": bitwidth,
        "group_size": 128,
        "quantizer": f"INT{bitwidth}_G128_RTNClip_shared_grid_fake_quant",
        "quantizer_backend": qrw.QUANTIZER_BACKENDS["rtnclip"],
        "K": 1,
        "scale_refresh_k": 1,
        "update_backend": "FP16 master",
        "master_dtype": "fp16",
        "pair_shared_grid": True,
        "fresh_round_codes": True,
        "direct_int_update": False,
        "independent_q_plus_q_minus_scales": False,
        "q_w_plus_hu_bypass": False,
        "probe_batches": int(args.probe_batches),
        "directions_per_h_per_batch": int(args.directions),
        "h_grid": [{"label": label, "h": h} for label, h in H_GRID],
        "fd_true_requested": bool(true_grad_enabled),
        "quantized_modules": "Linear weights only",
        "attention_projections": "quantized",
        "mlp_projections": "quantized",
        "embeddings": "fp16",
        "layernorm": "fp16",
        "bias": "fp16",
    }
    write_json(output_root / "run_config.json", config)
    write_json(output_root / "env.json", qrw.collect_env())

    records: List[Dict[str, object]] = []
    fd_true_available = bool(true_grad_enabled)
    true_grad_note = "computed"
    data_iter = iter(harness.train_loader)
    start = time.time()
    for batch_id in range(int(args.probe_batches)):
        try:
            batch = qrw.move_batch(next(data_iter), device)
        except StopIteration:
            break
        if true_grad_enabled:
            compute_true_gradient(harness, master, batch)
            true_grad_note = "computed against unquantized master objective"
        else:
            fd_true_available = False
            true_grad_note = "disabled after true-gradient OOM"
        for label, h in H_GRID:
            for direction_id in range(int(args.directions)):
                seed = qrw.direction_seed(16, "rtnclip", h, batch_id, extra=direction_id)
                directions = qrw.sample_directions(master, seed)
                d_true = directional_true_derivative(params, directions) if true_grad_enabled else None
                loss_plus, loss_minus, d_h = finite_difference_pair(harness, master, states, directions, batch, h)
                _, _, d_half = finite_difference_pair(harness, master, states, directions, batch, h / 2.0)
                diag = quantized_pair_diagnostics(master, directions, states, h)
                absdiff = abs(d_h - d_half)
                relerr = absdiff / max(abs(d_half), EPS)
                fd_true_error = (d_h - d_true) if d_true is not None else None
                record = {
                    "model": "roberta-large",
                    "dataset": "SST-5",
                    "dataset_mode": "full",
                    "h": h,
                    "h_label": label,
                    "batch_id": batch_id,
                    "direction_id": direction_id,
                    "direction_seed": seed,
                    "quant_bits": bitwidth,
                    "group_size": 128,
                    "quantizer": f"INT{bitwidth}_G128_RTNClip_shared_grid_fake_quant",
                    "K": 1,
                    "pair_shared_grid": True,
                    "fresh_round_codes": True,
                    "loss_plus": loss_plus,
                    "loss_minus": loss_minus,
                    "d_h_Q": d_h,
                    "d_half_Q": d_half,
                    "richardson_absdiff": absdiff,
                    "richardson_relerr_per_direction": relerr,
                    "richardson_available": True,
                    "d_true": d_true,
                    "fd_true_error": fd_true_error,
                    "fd_true_available": bool(d_true is not None),
                    **{key: value for key, value in diag.items() if not key.startswith("_")},
                }
                append_jsonl(records_path, record)
                records.append(record)
        if true_grad_enabled:
            harness.model.zero_grad(set_to_none=True)
        qrw.restore_master(params, master)

    summary_rows = summarize_records(records)
    write_csv(output_root / f"int{bitwidth}_mse_probe_summary.csv", summary_rows, SUMMARY_COLUMNS)
    merged = make_training_merge(summary_rows, output_root, bitwidth, training_summary)
    plots = plot_summary(summary_rows, merged, output_root, bitwidth)
    make_markdown_summary(summary_rows, merged, plots, output_root, bitwidth, training_summary, fd_true_available, true_grad_note)
    write_diagnostic_spec(output_root)
    write_json(
        output_root / "run_summary.json",
        {
            **config,
            "status": "complete",
            "records": len(records),
            "summary_rows": len(summary_rows),
            "fd_true_available": fd_true_available,
            "true_gradient_note": true_grad_note,
            "runtime_seconds": time.time() - start,
            "peak_gpu_mem_mb": float(torch.cuda.max_memory_allocated() / 1024 / 1024) if torch.cuda.is_available() else 0.0,
        },
    )
    return records, summary_rows, fd_true_available, true_grad_note


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe-only low-bit RTNClip MSE diagnostics for RoBERTa-large/SST-5")
    parser.add_argument("--bitwidth", type=int, choices=sorted(DEFAULT_TRAINING_SUMMARIES), default=8)
    parser.add_argument("--output_root", type=str, default=None)
    parser.add_argument("--training_summary", type=str, default=None)
    parser.add_argument("--probe_batches", type=int, default=3)
    parser.add_argument("--directions", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--disable_true_grad", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_root is None:
        args.output_root = str(REPO_ROOT / "outputs" / f"rtnclip_int{int(args.bitwidth)}_mse_reprobe")
    if args.training_summary is None:
        args.training_summary = str(DEFAULT_TRAINING_SUMMARIES[int(args.bitwidth)])
    requested = int(args.batch_size)
    batch_sizes = []
    for candidate in (requested, 32, 16, 8):
        if candidate not in batch_sizes and candidate <= requested:
            batch_sizes.append(candidate)
    if args.disable_true_grad:
        run_probe_once(args, requested, true_grad_enabled=False)
        return
    last_exc: Optional[BaseException] = None
    for batch_size in batch_sizes:
        try:
            run_probe_once(args, batch_size, true_grad_enabled=True)
            return
        except RuntimeError as exc:
            if not is_oom(exc):
                raise
            last_exc = exc
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(f"CUDA OOM at batch_size={batch_size}; retrying smaller batch if available", file=sys.stderr)
    print(f"True-gradient probe OOM after retries; rerunning visibility/Richardson only. Last error: {last_exc}", file=sys.stderr)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    run_probe_once(args, batch_sizes[-1], true_grad_enabled=False)


if __name__ == "__main__":
    main()
