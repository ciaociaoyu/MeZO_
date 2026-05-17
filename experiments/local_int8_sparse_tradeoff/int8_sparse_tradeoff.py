#!/usr/bin/env python3
"""Probe the int8 sparse-coordinate perturbation trade-off.

The script first tries to use locally cached RoBERTa weights as the parameter
source. If transformers or the local weights are unavailable, it falls back to a
small synthetic torch-only parameter vector. The objective is a synthetic binary
classification loss so that we can compare a two-sided finite-difference loss
direction estimate against the exact autograd gradient projection.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

# This local probe can import torch, numpy, and matplotlib in the same process on
# macOS, where mixed OpenMP runtimes are common in ad-hoc environments.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


def symmetric_quantize(
    x: torch.Tensor,
    bits: int,
    *,
    stochastic: bool = False,
    seed: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    if bits < 2:
        raise ValueError("bits must be >= 2")

    qmax = (1 << (bits - 1)) - 1
    max_abs = float(x.detach().abs().max().item()) if x.numel() else 0.0
    if (not math.isfinite(max_abs)) or max_abs <= 0.0:
        codes = torch.zeros_like(x, dtype=torch.int16)
        return torch.zeros_like(x), codes, 1.0

    scale = max_abs / float(qmax)
    y = torch.clamp(x.detach() / scale, -float(qmax), float(qmax))
    if stochastic:
        lower = torch.floor(y)
        prob = torch.clamp(y - lower, 0.0, 1.0)
        generator = torch.Generator(device=x.device.type)
        generator.manual_seed(int(seed or 0))
        rnd = torch.rand(y.size(), device=y.device, dtype=torch.float32, generator=generator)
        q = lower + (rnd < prob).to(dtype=y.dtype)
    else:
        q = torch.round(y)

    q = torch.clamp(q, -float(qmax), float(qmax))
    return (q * scale).to(dtype=x.dtype), q.to(dtype=torch.int16), float(scale)


def try_load_roberta_vector(
    model_name: str,
    dim: int,
    *,
    allow_download: bool,
) -> Tuple[Optional[torch.Tensor], Dict[str, Any]]:
    info: Dict[str, Any] = {
        "requested_model": model_name,
        "allow_download": bool(allow_download),
        "loaded": False,
    }
    try:
        from transformers import AutoModel  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on local env
        info["failure"] = f"transformers import failed: {type(exc).__name__}: {exc}"
        return None, info

    try:
        model = AutoModel.from_pretrained(model_name, local_files_only=not allow_download)
        model.eval()
        chunks = []
        seen = 0
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.ndim < 2:
                    continue
                flat = param.detach().cpu().float().reshape(-1)
                take = min(int(flat.numel()), dim - seen)
                if take <= 0:
                    break
                chunks.append(flat[:take])
                seen += take
                if seen >= dim:
                    break
        if seen <= 0:
            info["failure"] = "model loaded, but no matrix-like parameters were found"
            return None, info
        theta = torch.cat(chunks)
        if theta.numel() < dim:
            repeats = int(math.ceil(float(dim) / float(theta.numel())))
            theta = theta.repeat(repeats)[:dim].clone()
        info.update({"loaded": True, "source": "roberta_weights", "used_dim": int(theta.numel())})
        return theta[:dim].contiguous(), info
    except Exception as exc:  # pragma: no cover - depends on local env
        info["failure"] = f"model load failed: {type(exc).__name__}: {exc}"
        return None, info


def make_synthetic_theta(dim: int, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    theta = 0.35 * torch.randn(dim, generator=generator)
    outlier_count = max(1, dim // 128)
    outlier_idx = torch.randperm(dim, generator=generator)[:outlier_count]
    theta[outlier_idx] *= 4.0
    return theta.float()


def make_dataset(dim: int, n_examples: int, seed: int) -> Tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    x = torch.randn(n_examples, dim, generator=generator) / math.sqrt(float(dim))
    teacher = 1.3 * torch.randn(dim, generator=generator)
    logits = x.matmul(teacher)
    probs = torch.sigmoid(logits)
    y = torch.bernoulli(probs, generator=generator).float()
    return x.float(), y.float()


def bce_loss(theta: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(x.matmul(theta), y, reduction="mean")


def bce_loss_vector(theta: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(x.matmul(theta), y, reduction="none")


def lowest_magnitude_mask(theta: torch.Tensor, active_fraction: float) -> torch.Tensor:
    dim = int(theta.numel())
    k = max(1, min(dim, int(round(float(active_fraction) * dim))))
    if k >= dim:
        return torch.ones(dim, dtype=torch.bool)
    order = torch.argsort(theta.detach().abs())
    mask = torch.zeros(dim, dtype=torch.bool)
    mask[order[:k]] = True
    return mask


def random_sign_delta(
    dim: int,
    *,
    mask: torch.Tensor,
    int8_scale: float,
    code_step: int,
    seed: int,
) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    signs = torch.randint(0, 2, (dim,), generator=generator).float().mul_(2.0).sub_(1.0)
    codes = signs * float(code_step)
    codes = torch.where(mask, codes, torch.zeros_like(codes))
    return codes * float(int8_scale)


def summarize(values: Iterable[float]) -> Dict[str, float]:
    arr = np.asarray(list(values), dtype=float)
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p25": float(np.percentile(arr, 25)),
        "p75": float(np.percentile(arr, 75)),
    }


def run_probe(args: argparse.Namespace) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    roberta_theta, model_info = try_load_roberta_vector(
        args.model_name,
        args.dim,
        allow_download=args.allow_download,
    )
    if roberta_theta is None:
        theta0 = make_synthetic_theta(args.dim, args.seed)
        model_info.update({"source": "synthetic_torch_vector", "used_dim": int(args.dim)})
    else:
        theta0 = roberta_theta

    theta_q, theta_codes, int8_scale = symmetric_quantize(theta0, 8, stochastic=False)
    x, y = make_dataset(theta_q.numel(), args.n_examples, args.seed + 101)

    theta_var = theta_q.detach().clone().requires_grad_(True)
    base_loss = bce_loss(theta_var, x, y)
    base_loss.backward()
    grad = theta_var.grad.detach().clone()

    rows = []
    for ratio in args.sparse_ratios:
        mask = lowest_magnitude_mask(theta_q, ratio)
        active_fraction = float(mask.float().mean().item())
        for rep in range(args.repeats):
            delta = random_sign_delta(
                theta_q.numel(),
                mask=mask,
                int8_scale=int8_scale,
                code_step=args.code_step,
                seed=args.seed + 1009 * rep + int(round(1000000 * ratio)),
            )
            h = float(args.code_step) * float(int8_scale)
            if h <= 0.0:
                raise ValueError("non-positive perturbation h")
            direction = delta / h
            plus_vec = bce_loss_vector(theta_q + delta, x, y)
            minus_vec = bce_loss_vector(theta_q - delta, x, y)
            gap_vec = plus_vec - minus_vec
            gap = float(gap_vec.mean().item())
            true_projection = float(torch.dot(grad, direction).item())
            fd_projection = gap / (2.0 * h)
            projection_error = fd_projection - true_projection
            projection_error_sq = projection_error * projection_error

            delta_norm = float(delta.norm().item())
            if delta_norm <= 0.0:
                unit_true_projection = 0.0
                unit_fd_projection = 0.0
            else:
                unit_direction = delta / delta_norm
                unit_true_projection = float(torch.dot(grad, unit_direction).item())
                unit_fd_projection = gap / (2.0 * delta_norm)
            unit_projection_error = unit_fd_projection - unit_true_projection
            unit_projection_error_sq = unit_projection_error * unit_projection_error

            linear_gap = float((2.0 * h * true_projection))
            trunc_abs = abs(gap - linear_gap)
            denom = max(abs(linear_gap), 1e-12)
            trunc_rel = trunc_abs / denom
            minibatch_gap_std = float(gap_vec.std(unbiased=True).item() / math.sqrt(float(args.batch_size)))
            minibatch_signal_quality_snr = abs(gap) / max(minibatch_gap_std, 1e-12)
            loss_gap_floor_snr = abs(gap) / max(float(args.loss_noise_floor), 1e-12)

            rows.append(
                {
                    "configured_sparse_ratio": float(ratio),
                    "active_fraction": active_fraction,
                    "repeat": int(rep),
                    "active_coordinates": int(mask.sum().item()),
                    "int8_scale": float(int8_scale),
                    "code_step": int(args.code_step),
                    "h": float(h),
                    "delta_norm": delta_norm,
                    "loss_gap": gap,
                    "abs_loss_gap": abs(gap),
                    "fd_projection": fd_projection,
                    "true_projection": true_projection,
                    "projection_error": projection_error,
                    "projection_error_sq": projection_error_sq,
                    "unit_fd_projection": unit_fd_projection,
                    "unit_true_projection": unit_true_projection,
                    "unit_projection_error": unit_projection_error,
                    "unit_projection_error_sq": unit_projection_error_sq,
                    "linear_gap": linear_gap,
                    "truncation_abs": trunc_abs,
                    "truncation_rel": trunc_rel,
                    "minibatch_gap_std": minibatch_gap_std,
                    "minibatch_signal_quality_snr": minibatch_signal_quality_snr,
                    "loss_noise_floor": float(args.loss_noise_floor),
                    "loss_gap_floor_snr": loss_gap_floor_snr,
                }
            )

    df = pd.DataFrame(rows)
    summary_rows = []
    for ratio, group in df.groupby("configured_sparse_ratio", sort=True):
        projection_error_sq = summarize(group["projection_error_sq"])
        projection_abs_error = summarize(group["projection_error"].abs())
        unit_projection_error_sq = summarize(group["unit_projection_error_sq"])
        trunc_abs = summarize(group["truncation_abs"])
        trunc_rel = summarize(group["truncation_rel"])
        abs_loss_gap = summarize(group["abs_loss_gap"])
        floor_snr = summarize(group["loss_gap_floor_snr"])
        minibatch_snr = summarize(group["minibatch_signal_quality_snr"])
        delta_norm = summarize(group["delta_norm"])
        summary_rows.append(
            {
                "configured_sparse_ratio": float(ratio),
                "active_fraction": float(group["active_fraction"].iloc[0]),
                "active_coordinates": int(group["active_coordinates"].iloc[0]),
                "projection_mse": projection_error_sq["mean"],
                "projection_error_sq_median": projection_error_sq["median"],
                "projection_error_sq_p25": projection_error_sq["p25"],
                "projection_error_sq_p75": projection_error_sq["p75"],
                "projection_abs_error_median": projection_abs_error["median"],
                "unit_projection_mse": unit_projection_error_sq["mean"],
                "truncation_abs_mean": trunc_abs["mean"],
                "truncation_abs_median": trunc_abs["median"],
                "truncation_abs_p25": trunc_abs["p25"],
                "truncation_abs_p75": trunc_abs["p75"],
                "truncation_rel_median": trunc_rel["median"],
                "abs_loss_gap_median": abs_loss_gap["median"],
                "loss_gap_floor_snr_mean": floor_snr["mean"],
                "loss_gap_floor_snr_median": floor_snr["median"],
                "loss_gap_floor_snr_p25": floor_snr["p25"],
                "loss_gap_floor_snr_p75": floor_snr["p75"],
                "minibatch_signal_quality_snr_median": minibatch_snr["median"],
                "delta_norm_median": delta_norm["median"],
            }
        )
    summary = pd.DataFrame(summary_rows).sort_values("active_fraction").reset_index(drop=True)

    meta = {
        "created_at_unix": time.time(),
        "bits": 8,
        "dim": int(theta_q.numel()),
        "n_examples": int(args.n_examples),
        "batch_size_for_snr": int(args.batch_size),
        "repeats": int(args.repeats),
        "code_step": int(args.code_step),
        "base_loss": float(base_loss.detach().item()),
        "theta_int8_scale": float(int8_scale),
        "theta_code_min": int(theta_codes.min().item()),
        "theta_code_max": int(theta_codes.max().item()),
        "loss_noise_floor": float(args.loss_noise_floor),
        "model_info": model_info,
        "notes": (
            "Perturbations are integer code-step moves on the int8 parameter grid. "
            "Sparse ratio controls how many coordinates receive a nonzero move. "
            "projection_mse is mean((finite_difference_projection - true_gradient_projection)^2), "
            "where finite_difference_projection=(f(w+h*z)-f(w-h*z))/(2*h) and true_gradient_projection=<grad,z>."
        ),
    }
    return df, summary, meta


def choose_sweet_spot(summary: pd.DataFrame, target_snr: float) -> pd.Series:
    ok = summary[summary["loss_gap_floor_snr_median"] >= float(target_snr)]
    if len(ok) > 0:
        return ok.sort_values(["projection_mse", "active_fraction"]).iloc[0]

    score = np.log10(summary["loss_gap_floor_snr_median"].clip(lower=1e-12))
    penalty = np.log10(summary["projection_mse"].clip(lower=1e-30))
    norm_score = (score - score.min()) / max(float(score.max() - score.min()), 1e-12)
    norm_penalty = (penalty - penalty.min()) / max(float(penalty.max() - penalty.min()), 1e-12)
    return summary.assign(_score=norm_score - norm_penalty).sort_values("_score", ascending=False).iloc[0]


def plot_summary(summary: pd.DataFrame, meta: Dict[str, Any], output_path: Path, target_snr: float) -> None:
    sweet = choose_sweet_spot(summary, target_snr)
    x = summary["active_fraction"].to_numpy(dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), dpi=160)
    fig.suptitle("Int8 sparse-coordinate perturbation trade-off", fontsize=13, fontweight="bold")

    ax = axes[0]
    color_err = "#C44E52"
    color_snr = "#4C78A8"
    projection_line, = ax.plot(
        x,
        summary["projection_mse"],
        marker="o",
        color=color_err,
        label="Projection MSE: FD projection vs true grad projection",
    )
    ax.fill_between(
        x,
        summary["projection_error_sq_p25"],
        summary["projection_error_sq_p75"],
        color=color_err,
        alpha=0.16,
        linewidth=0,
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("active coordinate fraction controlled by sparse ratio")
    ax.set_ylabel("MSE(FD projection - true grad projection)", color=color_err)
    ax.tick_params(axis="y", labelcolor=color_err)
    ax.grid(True, which="both", linewidth=0.5, alpha=0.28)

    ax2 = ax.twinx()
    snr_line, = ax2.plot(
        x,
        summary["loss_gap_floor_snr_median"],
        marker="s",
        color=color_snr,
        label="Loss-gap SNR: signal quality",
    )
    ax2.fill_between(
        x,
        summary["loss_gap_floor_snr_p25"],
        summary["loss_gap_floor_snr_p75"],
        color=color_snr,
        alpha=0.14,
        linewidth=0,
    )
    target_line = ax2.axhline(
        target_snr,
        color=color_snr,
        linestyle="--",
        linewidth=1.0,
        alpha=0.55,
        label=f"Target SNR={target_snr:g}",
    )
    ax2.set_yscale("log")
    ax2.set_ylabel("loss-gap SNR vs fixed noise floor", color=color_snr)
    ax2.tick_params(axis="y", labelcolor=color_snr)
    legend = ax.legend(
        handles=[projection_line, snr_line, target_line],
        loc="upper left",
        frameon=True,
        fontsize=8.4,
        borderpad=0.5,
        handlelength=2.4,
    )
    legend.get_frame().set_alpha(0.88)
    legend.get_frame().set_linewidth(0.4)

    ax.axvline(float(sweet["active_fraction"]), color="#333333", linestyle=":", linewidth=1.2)
    ax.text(
        float(sweet["active_fraction"]),
        float(summary["projection_mse"].max()),
        f"  selected ratio={sweet['configured_sparse_ratio']:.3g}",
        fontsize=8.5,
        va="top",
        ha="left",
        color="#333333",
    )

    axp = axes[1]
    scatter = axp.scatter(
        summary["projection_mse"],
        summary["loss_gap_floor_snr_median"],
        c=summary["active_fraction"],
        cmap="viridis",
        s=80,
        edgecolor="#222222",
        linewidth=0.5,
    )
    axp.scatter(
        [sweet["projection_mse"]],
        [sweet["loss_gap_floor_snr_median"]],
        marker="*",
        s=180,
        color="#E6AB02",
        edgecolor="#222222",
        linewidth=0.7,
        label="selected",
        zorder=4,
    )
    for _, row in summary.iterrows():
        if row["configured_sparse_ratio"] in {summary.iloc[0]["configured_sparse_ratio"], summary.iloc[-1]["configured_sparse_ratio"]}:
            axp.annotate(
                f"{row['configured_sparse_ratio']:.3g}",
                (row["projection_mse"], row["loss_gap_floor_snr_median"]),
                xytext=(5, 4),
                textcoords="offset points",
                fontsize=8,
            )
    axp.set_xscale("log")
    axp.set_yscale("log")
    axp.set_xlabel("projection MSE")
    axp.set_ylabel("median loss-gap SNR")
    axp.set_title("Pareto view")
    axp.grid(True, which="both", linewidth=0.5, alpha=0.28)
    axp.legend(frameon=False, loc="best")
    cbar = fig.colorbar(scatter, ax=axp, fraction=0.046, pad=0.04)
    cbar.set_label("active fraction")

    source = meta["model_info"].get("source", "unknown")
    loaded = meta["model_info"].get("loaded", False)
    footer = (
        f"source={source}, roberta_loaded={loaded}, dim={meta['dim']}, "
        f"int8_scale={meta['theta_int8_scale']:.3e}, code_step={meta['code_step']}, "
        f"noise_floor={meta['loss_noise_floor']:.1e}"
    )
    fig.text(0.5, 0.01, footer, ha="center", va="bottom", fontsize=8.3, color="#555555")
    fig.tight_layout(rect=(0, 0.035, 1, 0.93))
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def parse_sparse_ratios(text: str) -> Tuple[float, ...]:
    values = tuple(float(item.strip()) for item in text.split(",") if item.strip())
    if not values:
        raise argparse.ArgumentTypeError("at least one ratio is required")
    for value in values:
        if value <= 0.0 or value > 1.0:
            raise argparse.ArgumentTypeError("ratios must be in (0, 1]")
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/local_int8_sparse_tradeoff/results"))
    parser.add_argument("--model-name", default="roberta-large")
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--dim", type=int, default=2048)
    parser.add_argument("--n-examples", type=int, default=1536)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=32)
    parser.add_argument("--code-step", type=int, default=8)
    parser.add_argument("--loss-noise-floor", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=20260428)
    parser.add_argument(
        "--sparse-ratios",
        type=parse_sparse_ratios,
        default=parse_sparse_ratios("0.005,0.01,0.02,0.05,0.1,0.2,0.35,0.5,0.75,1.0"),
    )
    parser.add_argument("--target-snr", type=float, default=3.0)
    args = parser.parse_args()

    if args.dim <= 0:
        raise ValueError("--dim must be positive")
    if args.n_examples <= 1:
        raise ValueError("--n-examples must be > 1")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.repeats <= 0:
        raise ValueError("--repeats must be positive")
    if args.code_step <= 0 or args.code_step > 127:
        raise ValueError("--code-step must be in [1, 127]")
    if args.loss_noise_floor <= 0.0:
        raise ValueError("--loss-noise-floor must be positive")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    df, summary, meta = run_probe(args)
    raw_path = args.output_dir / "int8_sparse_tradeoff_raw.csv"
    summary_path = args.output_dir / "int8_sparse_tradeoff_summary.csv"
    meta_path = args.output_dir / "int8_sparse_tradeoff_meta.json"
    plot_path = args.output_dir / "int8_sparse_tradeoff.png"

    df.to_csv(raw_path, index=False)
    summary.to_csv(summary_path, index=False)
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
    plot_summary(summary, meta, plot_path, args.target_snr)

    print(f"raw_csv={raw_path}")
    print(f"summary_csv={summary_path}")
    print(f"meta_json={meta_path}")
    print(f"plot_png={plot_path}")
    print("model_info=" + json.dumps(meta["model_info"], sort_keys=True))


if __name__ == "__main__":
    main()
