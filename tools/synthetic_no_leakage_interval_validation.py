#!/usr/bin/env python
"""No-leakage synthetic directional-MSE validation for h-window selectors.

The target is always the loss-level directional MSE

    A_true(h) = E[(d_Q(h,u) - <grad F(w), u>)^2] / E[<grad F(w),u>^2].

Predictors are restricted to:
  - old envelopes: alpha / h^2 + beta h^2 + gamma, and h^4 variant;
  - interval geometry A_cross computed from Q(w+h u)-Q(w-h u);
  - an independent full-precision locality proxy A_loc_FP.

No predictor uses d_Q or the target residual.  Output is written under
synthetic_no_leakage_interval/ by default.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import socket
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_H_GRID = np.array(
    [
        1e-7,
        3e-7,
        1e-6,
        3e-6,
        1e-5,
        3e-5,
        1e-4,
        3e-4,
        1e-3,
        1.5e-3,
        2e-3,
        3e-3,
        4e-3,
        5e-3,
        1e-2,
        3e-2,
    ],
    dtype=np.float64,
)
EPS = 1e-30


@dataclass(frozen=True)
class BaseConfig:
    config_id: str
    split: str
    d: int
    qbits: int
    group_size: int
    teacher_norm: float
    label_noise: float
    cond: float
    seed: int


@dataclass
class CheckpointState:
    checkpoint: str
    step: int
    w: torch.Tensor
    train_loss: float
    train_acc: float


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    except Exception:
        return ""


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def get_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def covariance_diag(d: int, cond: float, device: torch.device) -> torch.Tensor:
    if cond <= 1:
        return torch.ones(d, device=device)
    # Log-spaced diagonal with mean normalized to 1.
    exponents = torch.linspace(0.0, 1.0, d, device=device)
    diag = cond**exponents
    return diag / diag.mean()


def make_teacher(d: int, norm: float, device: torch.device, gen: torch.Generator) -> torch.Tensor:
    w = torch.randn(d, device=device, generator=gen)
    # With x ~ N(0, Sigma/d), norm*sqrt(d) gives O(norm) logits.
    return w / w.norm().clamp_min(1e-12) * (norm * math.sqrt(d))


def sample_batch(
    n: int,
    d: int,
    cov_diag: torch.Tensor,
    teacher: torch.Tensor,
    label_noise: float,
    device: torch.device,
    gen: torch.Generator,
) -> Tuple[torch.Tensor, torch.Tensor]:
    x = torch.randn(n, d, device=device, generator=gen)
    x = x * torch.sqrt(cov_diag).unsqueeze(0) / math.sqrt(d)
    logits = x @ teacher
    probs = torch.sigmoid(logits)
    y = torch.bernoulli(probs, generator=gen)
    if label_noise > 0:
        flips = torch.rand(n, device=device, generator=gen) < label_noise
        y = torch.where(flips, 1.0 - y, y)
    return x, y


def logistic_loss_from_logits(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    if logits.ndim == 1:
        return torch.nn.functional.binary_cross_entropy_with_logits(logits, y)
    y2 = y.unsqueeze(1).expand_as(logits)
    return torch.nn.functional.binary_cross_entropy_with_logits(logits, y2, reduction="none").mean(dim=0)


def loss_for_weights(x: torch.Tensor, y: torch.Tensor, w_batch: torch.Tensor) -> torch.Tensor:
    # w_batch: [k, d], returns [k]
    logits = x @ w_batch.t()
    return logistic_loss_from_logits(logits, y)


def grad_for_weight(x: torch.Tensor, y: torch.Tensor, w: torch.Tensor) -> Tuple[torch.Tensor, float, float]:
    logits = x @ w
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y)
    prob = torch.sigmoid(logits)
    grad = x.t() @ (prob - y) / x.shape[0]
    acc = ((prob >= 0.5).float() == y).float().mean().item()
    return grad, float(loss.item()), float(acc)


def train_checkpoints(
    cfg: BaseConfig,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    device: torch.device,
    steps: int,
    lr: float,
    gen: torch.Generator,
) -> List[CheckpointState]:
    w = 0.01 * torch.randn(cfg.d, device=device, generator=gen)
    wanted = {
        "initial": 0,
        "25pct": max(1, steps // 4),
        "50pct": max(1, steps // 2),
        "75pct": max(1, (3 * steps) // 4),
        "converged": steps,
    }
    states: List[CheckpointState] = []

    def capture(name: str, step: int) -> None:
        _, loss, acc = grad_for_weight(x_train, y_train, w)
        states.append(CheckpointState(name, step, w.detach().clone(), loss, acc))

    capture("initial", 0)
    for step in range(1, steps + 1):
        grad, _, _ = grad_for_weight(x_train, y_train, w)
        w = w - lr * grad
        for name, s in wanted.items():
            if s == step and name != "initial":
                capture(name, step)
    return states


def group_scales(w: torch.Tensor, group_size: int, qbits: int) -> Tuple[torch.Tensor, int, int]:
    qmax = 2 ** (qbits - 1) - 1
    qmin = -(2 ** (qbits - 1))
    d = w.numel()
    n_groups = math.ceil(d / group_size)
    pad = n_groups * group_size - d
    if pad:
        wp = torch.nn.functional.pad(w, (0, pad))
    else:
        wp = w
    groups = wp.view(n_groups, group_size)
    max_abs = groups.abs().amax(dim=1).clamp_min(1e-12)
    scale_g = max_abs / max(abs(qmin), abs(qmax))
    scale = scale_g.repeat_interleave(group_size)[:d]
    return scale, qmin, qmax


def quantize_shared(x: torch.Tensor, scale: torch.Tensor, qmin: int, qmax: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    code = torch.round(x / scale)
    clip = (code <= qmin) | (code >= qmax)
    code = code.clamp(qmin, qmax)
    return code * scale, code, clip


def direction_chunks(n_dirs: int, d: int, chunk: int, device: torch.device, gen: torch.Generator) -> Iterable[torch.Tensor]:
    seen = 0
    while seen < n_dirs:
        cur = min(chunk, n_dirs - seen)
        yield torch.randn(cur, d, device=device, generator=gen)
        seen += cur


def compute_target_and_predictors(
    cfg: BaseConfig,
    ckpt: CheckpointState,
    h_grid: np.ndarray,
    x_eval: torch.Tensor,
    y_eval: torch.Tensor,
    x_cal: torch.Tensor,
    y_cal: torch.Tensor,
    n_dirs_eval: int,
    n_dirs_cal: int,
    device: torch.device,
    seed: int,
) -> pd.DataFrame:
    w = ckpt.w.to(device)
    scale, qmin, qmax = group_scales(w, cfg.group_size, cfg.qbits)
    g_eval, eval_loss, eval_acc = grad_for_weight(x_eval, y_eval, w)
    g_cal, cal_loss, cal_acc = grad_for_weight(x_cal, y_cal, w)
    g_eval_norm_sq = torch.sum(g_eval * g_eval).item()
    g_cal_norm_sq = torch.sum(g_cal * g_cal).item()
    w_norm = w.norm().item()
    rows: List[Dict[str, Any]] = []
    eval_gen = torch.Generator(device=device)
    cal_gen = torch.Generator(device=device)
    eval_gen.manual_seed(seed + 1009)
    cal_gen.manual_seed(seed + 9001)
    chunk = 4 if cfg.d >= 100_000 else 8

    for h in h_grid:
        h = float(h)
        target_accum = {"mse": 0.0, "dstar2": 0.0}
        for u in direction_chunks(n_dirs_eval, cfg.d, chunk, device, eval_gen):
            plus = w.unsqueeze(0) + h * u
            minus = w.unsqueeze(0) - h * u
            q_plus, _, _ = quantize_shared(plus, scale.unsqueeze(0), qmin, qmax)
            q_minus, _, _ = quantize_shared(minus, scale.unsqueeze(0), qmin, qmax)
            d_q = (loss_for_weights(x_eval, y_eval, q_plus) - loss_for_weights(x_eval, y_eval, q_minus)) / (2.0 * h)
            d_star = torch.sum(g_eval.unsqueeze(0) * u, dim=1)
            err = d_q - d_star
            target_accum["mse"] += float(torch.sum(err * err).item())
            target_accum["dstar2"] += float(torch.sum(d_star * d_star).item())

        pred_accum = {
            "cross_uniform": 0.0,
            "cross_grad": 0.0,
            "loc_fp": 0.0,
            "dstar2_cal": 0.0,
            "active": 0.0,
            "clip": 0.0,
            "align": 0.0,
            "norm_ratio": 0.0,
            "jump0": 0.0,
            "jump1": 0.0,
            "jumpge2": 0.0,
            "count": 0,
        }
        for u in direction_chunks(n_dirs_cal, cfg.d, chunk, device, cal_gen):
            plus = w.unsqueeze(0) + h * u
            minus = w.unsqueeze(0) - h * u
            q_plus, code_plus, clip_plus = quantize_shared(plus, scale.unsqueeze(0), qmin, qmax)
            q_minus, code_minus, clip_minus = quantize_shared(minus, scale.unsqueeze(0), qmin, qmax)
            delta_q = q_plus - q_minus
            b = delta_q / (2.0 * h)
            interval_err = b - u
            d_fp = (loss_for_weights(x_cal, y_cal, plus) - loss_for_weights(x_cal, y_cal, minus)) / (2.0 * h)
            d_star_cal = torch.sum(g_cal.unsqueeze(0) * u, dim=1)
            loc_err = d_fp - d_star_cal
            jump = torch.abs(code_plus - code_minus)
            intended = 2.0 * h * u
            norm_dq = torch.linalg.vector_norm(delta_q, dim=1)
            norm_int = torch.linalg.vector_norm(intended, dim=1).clamp_min(1e-30)
            align = torch.sum(delta_q * intended, dim=1) / (norm_dq.clamp_min(1e-30) * norm_int)
            pred_accum["cross_uniform"] += float(torch.sum(interval_err * interval_err).item()) / cfg.d
            pred_accum["cross_grad"] += float(torch.sum(interval_err * interval_err * (g_cal.unsqueeze(0) ** 2)).item()) / max(g_cal_norm_sq, 1e-30)
            pred_accum["loc_fp"] += float(torch.sum(loc_err * loc_err).item())
            pred_accum["dstar2_cal"] += float(torch.sum(d_star_cal * d_star_cal).item())
            pred_accum["active"] += float(torch.sum(jump > 0).item()) / cfg.d
            pred_accum["clip"] += float(torch.sum(clip_plus | clip_minus).item()) / cfg.d
            pred_accum["align"] += float(torch.sum(align).item())
            pred_accum["norm_ratio"] += float(torch.sum(norm_dq / norm_int).item())
            pred_accum["jump0"] += float(torch.sum(jump == 0).item()) / cfg.d
            pred_accum["jump1"] += float(torch.sum(jump == 1).item()) / cfg.d
            pred_accum["jumpge2"] += float(torch.sum(jump >= 2).item()) / cfg.d
            pred_accum["count"] += u.shape[0]

        n_eval = n_dirs_eval
        n_cal = pred_accum["count"]
        rows.append(
            {
                **asdict(cfg),
                "checkpoint": ckpt.checkpoint,
                "checkpoint_step": ckpt.step,
                "h": h,
                "target_kind": "paper_directional_nmse:d_h_minus_grad_dot_u",
                "target_is_paper_directional_mse": True,
                "A_true": target_accum["mse"] / max(target_accum["dstar2"], 1e-30),
                "raw_mse": target_accum["mse"] / max(n_eval, 1),
                "dstar2_eval_mean": target_accum["dstar2"] / max(n_eval, 1),
                "A_cross_uniform": pred_accum["cross_uniform"] / max(n_cal, 1),
                "A_cross_grad": pred_accum["cross_grad"] / max(n_cal, 1),
                "A_loc_FP": pred_accum["loc_fp"] / max(pred_accum["dstar2_cal"], 1e-30),
                "p_active": pred_accum["active"] / max(n_cal, 1),
                "p_clip": pred_accum["clip"] / max(n_cal, 1),
                "V_align": pred_accum["align"] / max(n_cal, 1),
                "V_norm": pred_accum["norm_ratio"] / max(n_cal, 1),
                "jump_zero_frac": pred_accum["jump0"] / max(n_cal, 1),
                "jump_one_frac": pred_accum["jump1"] / max(n_cal, 1),
                "jump_ge2_frac": pred_accum["jumpge2"] / max(n_cal, 1),
                "eval_loss": eval_loss,
                "eval_acc": eval_acc,
                "cal_loss": cal_loss,
                "cal_acc": cal_acc,
                "train_loss_checkpoint": ckpt.train_loss,
                "train_acc_checkpoint": ckpt.train_acc,
                "g_eval_norm": math.sqrt(max(g_eval_norm_sq, 0.0)),
                "g_cal_norm": math.sqrt(max(g_cal_norm_sq, 0.0)),
                "w_norm": w_norm,
                "n_dirs_eval": n_dirs_eval,
                "n_dirs_cal": n_dirs_cal,
            }
        )
    return pd.DataFrame(rows)


def nnls_enum(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Small nonnegative least-squares by active-set enumeration."""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    p = X.shape[1]
    best: Optional[Tuple[float, np.ndarray]] = None
    for mask in range(1, 1 << p):
        idx = [i for i in range(p) if mask & (1 << i)]
        Xi = X[:, idx]
        try:
            ci = np.linalg.lstsq(Xi, y, rcond=None)[0]
        except np.linalg.LinAlgError:
            continue
        if np.all(ci >= -1e-12):
            c = np.zeros(p, dtype=np.float64)
            c[idx] = np.maximum(ci, 0.0)
            rss = float(np.sum((X @ c - y) ** 2))
            if best is None or rss < best[0]:
                best = (rss, c)
    if best is not None:
        return best[1]
    return np.maximum(np.linalg.lstsq(X, y, rcond=None)[0], 0.0)


def feature_matrix(df: pd.DataFrame, model: str) -> np.ndarray:
    h = df["h"].to_numpy(dtype=np.float64)
    if model == "M2":
        return np.column_stack([1.0 / np.maximum(h, EPS) ** 2, h**2, np.ones_like(h)])
    if model == "M4":
        return np.column_stack([1.0 / np.maximum(h, EPS) ** 2, h**4, np.ones_like(h)])
    if model == "S_IA_uniform":
        return np.column_stack([df["A_cross_uniform"].to_numpy(dtype=np.float64), df["A_loc_FP"].to_numpy(dtype=np.float64), np.ones_like(h)])
    if model == "S_IA_grad":
        return np.column_stack([df["A_cross_grad"].to_numpy(dtype=np.float64), df["A_loc_FP"].to_numpy(dtype=np.float64), np.ones_like(h)])
    raise ValueError(model)


def fit_predictors(raw: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    models = ["M2", "M4", "S_IA_uniform", "S_IA_grad"]
    train = raw[raw["split"].eq("train")].copy()
    coeff_rows = []
    pred_frames = []
    y_train = train["A_true"].to_numpy(dtype=np.float64)
    for model in models:
        X = feature_matrix(train, model)
        coef = nnls_enum(X, y_train)
        coeff_rows.append({"model": model, "coef_json": json.dumps(coef.tolist()), "n_train_rows": len(train)})
        pred = raw.copy()
        pred["predictor_model"] = model
        pred["A_pred"] = feature_matrix(raw, model) @ coef
        pred_frames.append(pred)
    return pd.DataFrame(coeff_rows), pd.concat(pred_frames, ignore_index=True)


def rank_corr(x: np.ndarray, y: np.ndarray) -> float:
    rx = pd.Series(x).rank(method="average").to_numpy()
    ry = pd.Series(y).rank(method="average").to_numpy()
    return float(np.corrcoef(rx, ry)[0, 1]) if len(x) > 1 else np.nan


def evaluate_predictors(pred: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    metric_rows = []
    regret_rows = []
    group_cols = ["config_id", "checkpoint", "split", "qbits", "d", "teacher_norm", "label_noise", "cond"]
    for (model, *gkey), g in pred.groupby(["predictor_model"] + group_cols):
        g = g.sort_values("h")
        y = np.maximum(g["A_true"].to_numpy(dtype=np.float64), EPS)
        p = np.maximum(g["A_pred"].to_numpy(dtype=np.float64), EPS)
        pear = float(np.corrcoef(np.log10(p), np.log10(y))[0, 1]) if len(g) > 1 else np.nan
        spear = rank_corr(np.log10(p), np.log10(y))
        rmse = float(np.sqrt(np.mean((np.log10(p) - np.log10(y)) ** 2)))
        h_values = g["h"].to_numpy(dtype=np.float64)
        h_sel = float(h_values[int(np.argmin(p))])
        h_true = float(h_values[int(np.argmin(y))])
        true_min = float(np.min(y))
        true_at_sel = float(y[int(np.argmin(p))])
        window = set(h_values[y <= 1.1 * true_min])
        metric_rows.append(
            {
                "predictor_model": model,
                **dict(zip(group_cols, gkey)),
                "pearson_log": pear,
                "spearman_log": spear,
                "log_rmse": rmse,
                "selected_h": h_sel,
                "true_opt_h": h_true,
                "log10_h_distance": abs(math.log10(h_sel) - math.log10(h_true)),
                "mse_regret": true_at_sel / true_min,
                "inside_true_1p1_window": h_sel in window,
                "true_window_width": len(window),
            }
        )
        regret_rows.append(metric_rows[-1].copy())
    return pd.DataFrame(metric_rows), pd.DataFrame(regret_rows)


def zo_train_validate(
    raw: pd.DataFrame,
    pred: pd.DataFrame,
    configs: Dict[str, Tuple[BaseConfig, List[CheckpointState], Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]],
    device: torch.device,
    out_dir: Path,
    seed: int,
    steps: int,
    batch_size: int,
    lr: float,
) -> pd.DataFrame:
    rows = []
    test_groups = raw[raw["split"].eq("test")].groupby(["config_id", "checkpoint"])
    # Keep validation small and deterministic.
    selected_groups = list(test_groups.groups.keys())[:2]
    for config_id, checkpoint in selected_groups:
        cfg, states, data = configs[config_id]
        state = next(s for s in states if s.checkpoint == checkpoint)
        x_train, y_train, x_eval, y_eval = data
        g = raw[(raw["config_id"].eq(config_id)) & (raw["checkpoint"].eq(checkpoint))]
        if g.empty:
            continue
        true_opt_h = float(g.loc[g["A_true"].idxmin(), "h"])
        policy_h = {"default": 1e-3, "oracle_true_mse": true_opt_h}
        for model in ["M2", "M4", "S_IA_grad"]:
            pg = pred[(pred["config_id"].eq(config_id)) & (pred["checkpoint"].eq(checkpoint)) & (pred["predictor_model"].eq(model))]
            if not pg.empty:
                policy_h[model] = float(pg.loc[pg["A_pred"].idxmin(), "h"])
        for policy, h in policy_h.items():
            w = state.w.detach().clone().to(device)
            scale, qmin, qmax = group_scales(w, cfg.group_size, cfg.qbits)
            gen = torch.Generator(device=device)
            gen.manual_seed(seed + abs(hash((config_id, checkpoint, policy))) % 1_000_000)
            _, start_loss, start_acc = grad_for_weight(x_eval, y_eval, w)
            for _ in range(steps):
                idx = torch.randint(0, x_train.shape[0], (min(batch_size, x_train.shape[0]),), device=device, generator=gen)
                xb, yb = x_train[idx], y_train[idx]
                u = torch.randn(cfg.d, device=device, generator=gen)
                q_plus, _, _ = quantize_shared((w + h * u).unsqueeze(0), scale.unsqueeze(0), qmin, qmax)
                q_minus, _, _ = quantize_shared((w - h * u).unsqueeze(0), scale.unsqueeze(0), qmin, qmax)
                d_q = (loss_for_weights(xb, yb, q_plus) - loss_for_weights(xb, yb, q_minus))[0] / (2.0 * h)
                w = w - lr * d_q * u / math.sqrt(cfg.d)
                scale, qmin, qmax = group_scales(w, cfg.group_size, cfg.qbits)
            _, end_loss, end_acc = grad_for_weight(x_eval, y_eval, w)
            rows.append(
                {
                    "config_id": config_id,
                    "checkpoint": checkpoint,
                    "policy": policy,
                    "h": h,
                    "steps": steps,
                    "start_loss": start_loss,
                    "end_loss": end_loss,
                    "delta_loss": end_loss - start_loss,
                    "start_acc": start_acc,
                    "end_acc": end_acc,
                    "delta_acc": end_acc - start_acc,
                    "true_opt_h": true_opt_h,
                    "selector_regret": float(g.loc[np.isclose(g["h"], h), "A_true"].iloc[0] / g["A_true"].min()) if np.any(np.isclose(g["h"], h)) else np.nan,
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(out_dir / "synthetic_training_validation.csv", index=False)
    return out


def make_plots(raw: pd.DataFrame, pred: pd.DataFrame, metrics: pd.DataFrame, regret: pd.DataFrame, out_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Representative held-out curve.
    test = raw[raw["split"].eq("test")]
    if not test.empty:
        key = test.groupby(["config_id", "checkpoint"]).size().sort_values(ascending=False).index[0]
        g = test[(test["config_id"].eq(key[0])) & (test["checkpoint"].eq(key[1]))].sort_values("h")
        plt.figure(figsize=(7, 5))
        plt.loglog(g["h"], g["A_true"], "o-", label="A_true")
        for model in ["M2", "M4", "S_IA_grad"]:
            pg = pred[(pred["config_id"].eq(key[0])) & (pred["checkpoint"].eq(key[1])) & (pred["predictor_model"].eq(model))].sort_values("h")
            if not pg.empty:
                plt.loglog(pg["h"], np.maximum(pg["A_pred"], EPS), "--", label=model)
        plt.xlabel("h")
        plt.ylabel("directional nMSE / predictor")
        plt.title(f"A_true and predictors: {key[0]} {key[1]}")
        plt.legend()
        plt.tight_layout()
        for ext in ["png", "pdf"]:
            plt.savefig(fig_dir / f"fig_true_vs_predictors.{ext}")
        plt.close()

    plt.figure(figsize=(7, 4))
    test_metrics = metrics[metrics["split"].eq("test")]
    if not test_metrics.empty:
        labels = sorted(test_metrics["predictor_model"].unique())
        data = [test_metrics[test_metrics["predictor_model"].eq(m)]["spearman_log"].dropna().to_numpy() for m in labels]
        try:
            plt.boxplot(data, tick_labels=labels, showmeans=True)
        except TypeError:
            plt.boxplot(data, showmeans=True)
            plt.xticks(range(1, len(labels) + 1), labels)
        plt.xticks(rotation=20)
        plt.ylabel("held-out Spearman(log)")
        plt.tight_layout()
        for ext in ["png", "pdf"]:
            plt.savefig(fig_dir / f"fig_heldout_spearman.{ext}")
    plt.close()

    if not test_metrics.empty:
        plt.figure(figsize=(6, 5))
        for model, g in test_metrics.groupby("predictor_model"):
            plt.scatter(g["true_opt_h"], g["selected_h"], label=model, alpha=0.8)
        lo, hi = raw["h"].min(), raw["h"].max()
        plt.plot([lo, hi], [lo, hi], "k--", lw=1)
        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel("true optimal h")
        plt.ylabel("selected h")
        plt.legend()
        plt.tight_layout()
        for ext in ["png", "pdf"]:
            plt.savefig(fig_dir / f"fig_selected_vs_true_h.{ext}")
        plt.close()

        plt.figure(figsize=(7, 4))
        data = [test_metrics[test_metrics["predictor_model"].eq(m)]["mse_regret"].dropna().to_numpy() for m in labels]
        try:
            plt.boxplot(data, tick_labels=labels, showmeans=True)
        except TypeError:
            plt.boxplot(data, showmeans=True)
            plt.xticks(range(1, len(labels) + 1), labels)
        plt.yscale("log")
        plt.xticks(rotation=20)
        plt.ylabel("A_true(h_selected) / min A_true")
        plt.tight_layout()
        for ext in ["png", "pdf"]:
            plt.savefig(fig_dir / f"fig_mse_regret_by_model.{ext}")
        plt.close()

    # Correlation/window width vs dimension.
    if not test_metrics.empty:
        plt.figure(figsize=(7, 4))
        for model, g in test_metrics.groupby("predictor_model"):
            med = g.groupby("d")["spearman_log"].median()
            plt.plot(med.index, med.values, "o-", label=model)
        plt.xscale("log")
        plt.xlabel("dimension")
        plt.ylabel("median held-out Spearman")
        plt.legend()
        plt.tight_layout()
        for ext in ["png", "pdf"]:
            plt.savefig(fig_dir / f"fig_correlation_vs_dimension.{ext}")
        plt.close()


def default_configs(smoke: bool) -> List[BaseConfig]:
    if smoke:
        return [
            BaseConfig("train_d1e3_b8_norm1_noise0_cond1_g128", "train", 1_000, 8, 128, 1.0, 0.0, 1.0, 11),
            BaseConfig("test_d1e4_b4_norm2_noise03_cond100_g128", "test", 10_000, 4, 128, 2.0, 0.3, 100.0, 12),
        ]
    return [
        BaseConfig("train_d1e3_b8_norm1_noise0_cond1_g128", "train", 1_000, 8, 128, 1.0, 0.0, 1.0, 11),
        BaseConfig("train_d1e3_b4_norm2_noise01_cond10_g64", "train", 1_000, 4, 64, 2.0, 0.1, 10.0, 12),
        BaseConfig("train_d1e4_b8_norm05_noise01_cond10_g128", "train", 10_000, 8, 128, 0.5, 0.1, 10.0, 13),
        BaseConfig("train_d1e4_b4_norm1_noise0_cond1_g64", "train", 10_000, 4, 64, 1.0, 0.0, 1.0, 14),
        BaseConfig("test_d1e5_b4_norm2_noise03_cond100_g128", "test", 100_000, 4, 128, 2.0, 0.3, 100.0, 21),
        BaseConfig("test_d1e5_b8_norm05_noise03_cond100_g64", "test", 100_000, 8, 64, 0.5, 0.3, 100.0, 22),
    ]


def write_readme(
    out_dir: Path,
    raw: pd.DataFrame,
    metrics: pd.DataFrame,
    regret: pd.DataFrame,
    train_val: pd.DataFrame,
    coeffs: pd.DataFrame,
) -> None:
    def markdown_table(df: pd.DataFrame) -> str:
        if df.empty:
            return ""
        cols = list(df.columns)
        lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
        for _, row in df.iterrows():
            vals = []
            for c in cols:
                v = row[c]
                if isinstance(v, float):
                    vals.append(f"{v:.6g}")
                else:
                    vals.append(str(v))
            lines.append("| " + " | ".join(vals) + " |")
        return "\n".join(lines)

    test = metrics[metrics["split"].eq("test")]
    summary = []
    summary.append("# No-leakage synthetic interval-aware directional-MSE validation\n")
    summary.append("## Target and leakage control\n")
    summary.append(
        "The only fit target is `A_true = E[(d_Q - <grad F(w),u>)^2] / E[<grad F(w),u>^2]`. "
        "Predictors use separate calibration batches and direction seeds. `M_loc_true` / `d_Q-<grad,b_h>` is not used.\n"
    )
    summary.append("## Held-out selector summary\n")
    if not test.empty:
        tab = (
            test.groupby("predictor_model")
            .agg(
                n=("mse_regret", "size"),
                pearson_median=("pearson_log", "median"),
                spearman_median=("spearman_log", "median"),
                log_rmse_median=("log_rmse", "median"),
                regret_median=("mse_regret", "median"),
                h_distance_median=("log10_h_distance", "median"),
                window_coverage=("inside_true_1p1_window", "mean"),
            )
            .reset_index()
        )
        summary.append(markdown_table(tab))
        summary.append("\n")
    summary.append("## Answers\n")
    if not test.empty:
        best = test.groupby("predictor_model")["mse_regret"].median().sort_values()
        best_model = best.index[0]
        ia = test[test["predictor_model"].str.startswith("S_IA")]
        m2 = test[test["predictor_model"].eq("M2")]
        ia_corr = ia["spearman_log"].median() if not ia.empty else np.nan
        m2_corr = m2["spearman_log"].median() if not m2.empty else np.nan
        summary.append(f"1. Interval geometry without target leakage has held-out median Spearman around `{ia_corr:.3g}` in this run.\n")
        summary.append("2. The held-out split includes d=1e5 and INT4, so the reported test rows exercise unseen dimension/noise/checkpoints.\n")
        summary.append(f"3. Lowest median h-selection regret in this run: `{best_model}`.\n")
        summary.append("4. The interval predictor is evaluated out-of-sample; inspect `synthetic_heldout_correlation.csv` for per-config failures.\n")
        summary.append(f"5. For the paper main method, prefer the lowest-regret held-out selector if stable; in this run that is `{best_model}`, with default-aware fallback still recommended.\n")
        summary.append(f"\nM2 median Spearman: `{m2_corr:.3g}`; interval median Spearman: `{ia_corr:.3g}`.\n")
    else:
        summary.append("No held-out metrics were generated.\n")
    summary.append("\n## Files\n")
    for name in [
        "synthetic_true_mse_raw.csv",
        "synthetic_predictor_metrics.csv",
        "synthetic_heldout_correlation.csv",
        "synthetic_h_selection_regret.csv",
        "synthetic_training_validation.csv",
    ]:
        summary.append(f"- `{name}`\n")
    if not train_val.empty:
        summary.append("\n## ZO training validation preview\n")
        summary.append(markdown_table(train_val))
        summary.append("\n")
    (out_dir / "README.md").write_text("\n".join(summary), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir", default="synthetic_no_leakage_interval")
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=20260625)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--train_steps", type=int, default=80)
    p.add_argument("--batch_train", type=int, default=256)
    p.add_argument("--batch_eval", type=int, default=128)
    p.add_argument("--n_dirs_eval_small", type=int, default=24)
    p.add_argument("--n_dirs_eval_large", type=int, default=8)
    p.add_argument("--n_dirs_cal_small", type=int, default=24)
    p.add_argument("--n_dirs_cal_large", type=int, default=8)
    p.add_argument("--zo_validation_steps", type=int, default=20)
    p.add_argument("--skip_training_validation", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = get_device(args.device)
    h_grid = DEFAULT_H_GRID if not args.smoke else DEFAULT_H_GRID[::3]
    configs = default_configs(args.smoke)
    write_json(
        out_dir / "metadata.json",
        {
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "git_commit": git_commit(),
            "hostname": socket.gethostname(),
            "python": sys.version,
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "device": str(device),
            "h_grid": h_grid.tolist(),
            "target": "A_true=E[(d_Q-<grad,u>)^2]/E[<grad,u>^2]",
            "leakage_control": "predictors use independent calibration batch and direction seed; no M_loc_true/d_Q residual predictor",
            "configs": [asdict(c) for c in configs],
        },
    )

    raw_frames: List[pd.DataFrame] = []
    config_store: Dict[str, Tuple[BaseConfig, List[CheckpointState], Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]] = {}
    for cfg in configs:
        gen = torch.Generator(device=device)
        gen.manual_seed(args.seed + cfg.seed)
        cov = covariance_diag(cfg.d, cfg.cond, device)
        teacher = make_teacher(cfg.d, cfg.teacher_norm, device, gen)
        x_train, y_train = sample_batch(args.batch_train, cfg.d, cov, teacher, cfg.label_noise, device, gen)
        x_eval, y_eval = sample_batch(args.batch_eval, cfg.d, cov, teacher, cfg.label_noise, device, gen)
        x_cal, y_cal = sample_batch(args.batch_eval, cfg.d, cov, teacher, cfg.label_noise, device, gen)
        lr = 5.0
        states = train_checkpoints(cfg, x_train, y_train, device, args.train_steps, lr, gen)
        config_store[cfg.config_id] = (cfg, states, (x_train, y_train, x_eval, y_eval))
        if cfg.split == "train":
            ckpts = [s for s in states if s.checkpoint in {"initial", "25pct", "50pct"}]
        else:
            ckpts = [s for s in states if s.checkpoint in {"75pct", "converged"}]
        for state in ckpts:
            n_eval = args.n_dirs_eval_large if cfg.d >= 100_000 else args.n_dirs_eval_small
            n_cal = args.n_dirs_cal_large if cfg.d >= 100_000 else args.n_dirs_cal_small
            df = compute_target_and_predictors(
                cfg,
                state,
                h_grid,
                x_eval,
                y_eval,
                x_cal,
                y_cal,
                n_eval,
                n_cal,
                device,
                args.seed + state.step + cfg.seed * 17,
            )
            raw_frames.append(df)
            pd.concat(raw_frames, ignore_index=True).to_csv(out_dir / "synthetic_true_mse_raw.csv", index=False)
            print(f"[done] {cfg.config_id} {state.checkpoint} rows={len(df)}", flush=True)

    raw = pd.concat(raw_frames, ignore_index=True)
    raw.to_csv(out_dir / "synthetic_true_mse_raw.csv", index=False)
    coeffs, pred = fit_predictors(raw)
    coeffs.to_csv(out_dir / "synthetic_predictor_coefficients.csv", index=False)
    pred.to_csv(out_dir / "synthetic_predictor_metrics.csv", index=False)
    metrics, regret = evaluate_predictors(pred)
    metrics.to_csv(out_dir / "synthetic_heldout_correlation.csv", index=False)
    regret.to_csv(out_dir / "synthetic_h_selection_regret.csv", index=False)

    if args.skip_training_validation:
        train_val = pd.DataFrame()
        train_val.to_csv(out_dir / "synthetic_training_validation.csv", index=False)
    else:
        train_val = zo_train_validate(
            raw,
            pred,
            config_store,
            device,
            out_dir,
            args.seed,
            args.zo_validation_steps if not args.smoke else 3,
            min(64, args.batch_train),
            lr=0.05,
        )
    make_plots(raw, pred, metrics, regret, out_dir)
    write_readme(out_dir, raw, metrics, regret, train_val, coeffs)
    shutil.make_archive(str(out_dir), "zip", out_dir)
    print(f"Output directory: {out_dir}")
    print(f"Zip: {out_dir}.zip")
    if not metrics.empty:
        test = metrics[metrics["split"].eq("test")]
        print(test.groupby("predictor_model")[["spearman_log", "log_rmse", "mse_regret"]].median().to_string())


if __name__ == "__main__":
    main()
