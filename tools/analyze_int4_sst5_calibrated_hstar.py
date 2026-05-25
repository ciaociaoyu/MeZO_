#!/usr/bin/env python
"""Offline simple2pt_corrected h-star check for RTNClip RoBERTa-large/SST-5.

This is an analysis-only harness. It reuses the existing SST-5 RTNClip fake
quantization path, estimates the h4 selector components, and compares the
result against the existing low-bit h-sweep summary. It does not train.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import socket
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, List, Optional, Tuple

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = REPO_ROOT / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import smoke_rtnclip_roberta_sst5 as smoke  # noqa: E402


H_GRID = [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 1.5e-3, 2e-3, 3e-3, 4e-3, 5e-3, 1e-2]
EPS = 1e-12


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: List[Dict[str, object]], fieldnames: Optional[List[str]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys: List[str] = []
        for row in rows:
            for key in row:
                if key not in keys:
                    keys.append(key)
        fieldnames = keys
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT), text=True).strip()
    except Exception:
        return ""


def env_report() -> Dict[str, object]:
    info: Dict[str, object] = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "hostname": socket.gethostname(),
        "pwd": str(REPO_ROOT),
        "python": sys.executable,
        "python_version": sys.version.replace("\n", " "),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "conda_default_env": os.environ.get("CONDA_DEFAULT_ENV", ""),
        "torch_version": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "git_commit": git_commit(),
    }
    if torch.cuda.is_available():
        info["device_name"] = torch.cuda.get_device_name(0)
        info["device_count"] = torch.cuda.device_count()
    return info


def make_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        repo_root=REPO_ROOT,
        model_id=args.model_id,
        dataset_mode="full",
        data_seed=args.data_seed,
        num_k=16,
        data_dir=None,
        batch_size=args.batch_size,
        eval_batch_size=args.batch_size,
    )


def direction_norm_sq(directions: Dict[str, torch.Tensor], names: Optional[Iterable[str]] = None) -> float:
    selected = names if names is not None else directions.keys()
    total = torch.zeros((), device=next(iter(directions.values())).device, dtype=torch.float64)
    for name in selected:
        tensor = directions[name].float()
        total += tensor.double().square().sum()
    return float(total.detach().cpu())


def copy_clean_to_model(
    params: Dict[str, torch.nn.Parameter],
    master32: Dict[str, torch.Tensor],
    directions: Optional[Dict[str, torch.Tensor]],
    h: float,
    sign: float,
) -> None:
    with torch.no_grad():
        for name, param in params.items():
            base = master32[name]
            if directions is not None:
                value = base.add(directions[name].float(), alpha=sign * h)
            else:
                value = base
            param.copy_(value.to(dtype=param.dtype))


def loss_value(model, batch: Dict[str, torch.Tensor]) -> float:
    loss, _ = smoke.forward_loss_and_logits(model, batch)
    return float(loss.detach().float().cpu())


def quantized_d2(
    model,
    params: Dict[str, torch.nn.Parameter],
    master: Dict[str, torch.Tensor],
    states: Dict[str, smoke.RTNClipState],
    batch: Dict[str, torch.Tensor],
    directions: Dict[str, torch.Tensor],
    h: float,
) -> float:
    smoke.copy_master_to_model(params, master, directions, h, 1.0, states)
    lp = loss_value(model, batch)
    smoke.copy_master_to_model(params, master, directions, h, -1.0, states)
    lm = loss_value(model, batch)
    smoke.restore_master(params, master)
    return (lp - lm) / (2.0 * h)


def clean32_d2(
    model,
    params: Dict[str, torch.nn.Parameter],
    master32: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    directions: Dict[str, torch.Tensor],
    h: float,
) -> float:
    copy_clean_to_model(params, master32, directions, h, 1.0)
    lp = loss_value(model, batch)
    copy_clean_to_model(params, master32, directions, h, -1.0)
    lm = loss_value(model, batch)
    copy_clean_to_model(params, master32, None, 0.0, 0.0)
    return (lp - lm) / (2.0 * h)


def pearson(xs: List[float], ys: List[float]) -> float:
    pairs = [(x, y) for x, y in zip(xs, ys) if math.isfinite(x) and math.isfinite(y)]
    if len(pairs) < 2:
        return float("nan")
    mx = sum(x for x, _ in pairs) / len(pairs)
    my = sum(y for _, y in pairs) / len(pairs)
    vx = sum((x - mx) ** 2 for x, _ in pairs)
    vy = sum((y - my) ** 2 for _, y in pairs)
    if vx <= 0 or vy <= 0:
        return float("nan")
    cov = sum((x - mx) * (y - my) for x, y in pairs)
    return cov / math.sqrt(vx * vy)


def sign_flip_rate(xs: List[float], ys: List[float]) -> float:
    pairs = [(x, y) for x, y in zip(xs, ys) if math.isfinite(x) and math.isfinite(y)]
    if not pairs:
        return float("nan")
    flips = sum(1 for x, y in pairs if (x > 0) != (y > 0) and abs(x) > 0 and abs(y) > 0)
    return flips / len(pairs)


def weighted_int4_delta(states: Dict[str, smoke.RTNClipState]) -> Dict[str, float]:
    scale_sq_sum = 0.0
    scale_sum = 0.0
    values = 0
    flat_scales: List[torch.Tensor] = []
    for state in states.values():
        lengths = state.lengths.view(1, -1, 1).double()
        scales = state.scales.double()
        scale_sq_sum += float((scales.square() * lengths).sum().detach().cpu())
        scale_sum += float((scales * lengths).sum().detach().cpu())
        values += int(lengths.sum().detach().cpu()) * int(state.shape[0])
        flat_scales.append(state.scales.detach().float().reshape(-1).cpu())
    all_scales = torch.cat(flat_scales) if flat_scales else torch.empty(0)
    return {
        "delta_int4_rtnclip_scale_rms": math.sqrt(scale_sq_sum / max(values, 1)),
        "delta_int4_rtnclip_scale_mean": scale_sum / max(values, 1),
        "scale_median_unweighted": float(all_scales.median()) if all_scales.numel() else float("nan"),
        "scale_p90_unweighted": float(torch.quantile(all_scales, 0.90)) if all_scales.numel() else float("nan"),
        "scale_p95_unweighted": float(torch.quantile(all_scales, 0.95)) if all_scales.numel() else float("nan"),
        "num_quantized_values_for_delta": values,
    }


def hstar(delta: float, g: float, lval: float, d_dim: int) -> float:
    vals = [delta, g, lval, float(d_dim)]
    if min(vals) <= 0 or not all(math.isfinite(v) for v in vals):
        return float("nan")
    return (delta * delta * g * g / (16.0 * lval * lval * float(d_dim) * float(d_dim + 2))) ** 0.25


def _positive_or_nan(value: object) -> float:
    try:
        out = float(value)
    except Exception:
        return float("nan")
    return out if math.isfinite(out) and out > 0.0 else float("nan")


def simple2pt_corrected(
    precision: str,
    d_trainable: int,
    l_clean32_q90: float,
    *,
    scale_rms: Optional[float] = None,
    clean32_g_median: Optional[float] = None,
    clean32_g_h3e4: Optional[float] = None,
    selected_g: Optional[float] = None,
    selected_g_mode: Optional[str] = None,
) -> Dict[str, object]:
    """修正简单两点法.

    Default retained selector:
      Delta = RTNClip group scale RMS / sqrt(6)
      G     = clean FP32 two-point absG median over {1e-4, 3e-4, 1e-3}
      L     = clean FP32 shared-step curvature q90

    Callers may pass selected_g/selected_g_mode to use a precision-aware
    G estimate, e.g. a low-bit shared-grid finite-difference absG. L remains
    the clean32 curvature term in this selector.

    Deprecated exploratory options are intentionally disabled here:
      # Delta = raw scale RMS
      # Delta = empirical snap RMS
      # G = raw low-bit absG
      # L = q50/q95 or oracle-precision L
    """
    _ = precision
    g_selected = _positive_or_nan(selected_g)
    if math.isfinite(g_selected):
        g_mode = selected_g_mode or "precision_aware_G"
        g_value = g_selected
        g_note = g_mode
    else:
        g_clean = _positive_or_nan(clean32_g_median)
        if not math.isfinite(g_clean):
            g_clean = _positive_or_nan(clean32_g_h3e4)
        g_mode = "clean32_absG_median_1e-4_3e-4_1e-3"
        g_value = g_clean
        g_note = "clean32 absG median"
    scale = _positive_or_nan(scale_rms)
    delta_mode = "scale_rms_over_sqrt6"
    delta = scale / math.sqrt(6.0) if math.isfinite(scale) else float("nan")
    notes = f"single selector: scale RMS / sqrt(6) Delta; {g_note}; L_clean32 q90"

    l_value = _positive_or_nan(l_clean32_q90)
    hs = hstar(delta, g_value, l_value, d_trainable)
    return {
        "selector_name": "simple2pt_corrected",
        "precision": precision,
        "Delta_mode": delta_mode,
        "Delta_value": delta,
        "G_mode": g_mode,
        "G_value": g_value,
        "L_mode": "L_clean32",
        "L_q": "q90",
        "L_hat": l_value,
        "hstar_cont": hs,
        "hstar_nearest_grid": nearest_grid(hs),
        "notes": notes,
    }


def nearest_grid(h: float) -> float:
    if not math.isfinite(h) or h <= 0:
        return float("nan")
    return min(H_GRID, key=lambda x: abs(math.log(x) - math.log(h)))


def read_sweep_summary(path: Path) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    if not path.exists():
        return [], {}
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    parsed: List[Dict[str, object]] = []
    for row in rows:
        out: Dict[str, object] = dict(row)
        for key in (
            "h",
            "best_eval_acc",
            "last_eval_acc",
            "lowbit_true_nmse",
            "lowbit_true_corr",
            "alignment",
            "norm_ratio",
            "active_frac",
            "weight_recon_mse",
        ):
            try:
                out[key] = float(row.get(key, "nan"))
            except Exception:
                out[key] = float("nan")
        parsed.append(out)
    complete = [r for r in parsed if r.get("status") == "complete"]
    best_acc = max(complete or parsed, key=lambda r: float(r.get("best_eval_acc", float("nan"))))
    best_last = max(complete or parsed, key=lambda r: float(r.get("last_eval_acc", float("nan"))))
    best_vis = min(complete or parsed, key=lambda r: float(r.get("lowbit_true_nmse", float("inf"))))
    return parsed, {
        "sweep_path": str(path),
        "best_acc_h": best_acc.get("h"),
        "best_acc": best_acc.get("best_eval_acc"),
        "best_last_h": best_last.get("h"),
        "best_last_acc": best_last.get("last_eval_acc"),
        "best_visibility_h": best_vis.get("h"),
        "best_visibility_lowbit_true_nmse": best_vis.get("lowbit_true_nmse"),
    }


def choose_l_plateau(rows: List[Dict[str, object]]) -> Tuple[Dict[str, object], str]:
    finite = [r for r in rows if math.isfinite(float(r["lambda_q90"])) and float(r["lambda_q90"]) > 0]
    if not finite:
        return {}, "no_finite_L"
    by_h = {float(r["h2"]): r for r in finite}
    for idx, row in enumerate(finite):
        q = float(row["lambda_q90"])
        next_q = float(finite[idx + 1]["lambda_q90"]) if idx + 1 < len(finite) else float("nan")
        prev_q = float(finite[idx - 1]["lambda_q90"]) if idx > 0 else float("nan")
        st_next = abs(q - next_q) / (q + EPS) if math.isfinite(next_q) else float("nan")
        st_prev = abs(q - prev_q) / (q + EPS) if math.isfinite(prev_q) else float("nan")
        row["stability_q90_next"] = st_next
        row["stability_q90_prev"] = st_prev
        low_noise = bool(math.isfinite(next_q) and (q / max(next_q, EPS) >= 5.0))
        row["low_h2_noise_suspected"] = low_noise
        if not low_noise and ((math.isfinite(st_next) and st_next <= 0.5) or (math.isfinite(st_prev) and st_prev <= 0.5)):
            return row, "plateau_q90_primary"
    chosen = min(finite, key=lambda r: min(
        float(r.get("stability_q90_next", float("inf"))) if math.isfinite(float(r.get("stability_q90_next", float("nan")))) else float("inf"),
        float(r.get("stability_q90_prev", float("inf"))) if math.isfinite(float(r.get("stability_q90_prev", float("nan")))) else float("inf"),
    ))
    return chosen, "fallback_best_adjacent_stability"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="")
    parser.add_argument("--sweep_summary", default=str(REPO_ROOT / "outputs/rtnclip_int4_g128_rtnclip_roberta_sst5_seed16_20260521/int4_hsearch_summary.csv"))
    parser.add_argument("--model_id", default="roberta-large")
    parser.add_argument("--seed", type=int, default=16)
    parser.add_argument("--data_seed", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--group_size", type=int, default=128)
    parser.add_argument("--bitwidth", type=int, default=4)
    parser.add_argument("--m_g", type=int, default=8)
    parser.add_argument("--m_l", type=int, default=4)
    # Deprecated raw-lowbit G selection is intentionally disabled:
    # parser.add_argument("--h_g", type=float, default=1e-3)
    # Deprecated exploratory selector output intentionally disabled:
    # parser.add_argument("--write_legacy_variants", action="store_true", help="Also write the old exploratory Delta/G variant grid.")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this analysis.")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir) if args.output_dir else REPO_ROOT / "analysis" / f"int{args.bitwidth}_sst5_simple2pt_corrected_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "env_report.json", env_report())
    (out_dir / "env_report.txt").write_text("\n".join(f"{k}: {v}" for k, v in env_report().items()) + "\n", encoding="utf-8")

    os.environ.setdefault("DATALOADER_SHUFFLE", "True")
    torch.manual_seed(args.seed)
    device = torch.device("cuda:0")
    smoke_args = make_args(args)
    orig_torch_load = torch.load

    def _compat_torch_load(*load_args_, **load_kwargs_):
        load_kwargs_.setdefault("weights_only", False)
        return orig_torch_load(*load_args_, **load_kwargs_)

    torch.load = _compat_torch_load
    try:
        model, train_loader, _dev_loader, _data_args, sampler = smoke.load_prompt_model_and_data(smoke_args, device)
    finally:
        torch.load = orig_torch_load
    batch = smoke.move_batch(next(iter(train_loader)), device)
    params = smoke.named_parameter_map(model)
    master = {name: p.detach().clone().to(device=device, dtype=torch.float16) for name, p in params.items() if p.is_floating_point()}
    master32 = {name: tensor.detach().float().clone() for name, tensor in master.items()}
    q_names = smoke.linear_weight_names(model)
    states, quant_rows = smoke.refresh_quantizer_states(master, q_names, args.bitwidth, args.group_size)
    qstats = smoke.aggregate_quantizer_stats(quant_rows, {name: params[name].numel() for name in q_names})
    d_trainable = sum(int(t.numel()) for t in master.values())
    d_quantized = sum(int(master[name].numel()) for name in q_names)
    delta_stats = weighted_int4_delta(states)

    sweep_rows, sweep_summary = read_sweep_summary(Path(args.sweep_summary))
    write_csv(out_dir / "lowbit_sweep_reference.csv", sweep_rows)

    visibility_rows: List[Dict[str, object]] = []
    d2_by_h: Dict[float, List[float]] = {}
    g = torch.Generator(device=device)
    direction_seeds = [args.seed * 1000003 + i for i in range(max(args.m_g, args.m_l))]

    model.half()
    for h in H_GRID:
        vals: List[float] = []
        vis_acc: Dict[str, List[float]] = {}
        for i in range(args.m_g):
            g.manual_seed(direction_seeds[i])
            directions = smoke.sample_directions(master, g)
            if i < min(args.m_g, 4):
                vis = smoke.perturbation_metrics(master, directions, states, h)
                for key in ("active_frac", "alignment", "norm_ratio", "zero_effective_displacement_frac", "delta_visibility_mse", "delta_visibility_nmse"):
                    vis_acc.setdefault(key, []).append(float(vis[key]))
            vals.append(quantized_d2(model, params, master, states, batch, directions, h))
        d2_by_h[h] = vals
        mean_delta_visibility_mse = sum(vis_acc.get("delta_visibility_mse", [float("nan")])) / max(len(vis_acc.get("delta_visibility_mse", [])), 1)
        visibility_rows.append(
            {
                "h": h,
                "G_abs": math.sqrt(math.pi / 2.0) * (sum(abs(v) for v in vals) / max(len(vals), 1)),
                "G_rms": math.sqrt(sum(v * v for v in vals) / max(len(vals), 1)),
                "d2_mean": sum(vals) / max(len(vals), 1),
                "d2_abs_mean": sum(abs(v) for v in vals) / max(len(vals), 1),
                "alignment_eff": sum(vis_acc.get("alignment", [float("nan")])) / max(len(vis_acc.get("alignment", [])), 1),
                "norm_ratio_eff": sum(vis_acc.get("norm_ratio", [float("nan")])) / max(len(vis_acc.get("norm_ratio", [])), 1),
                "active_frac_eff": sum(vis_acc.get("active_frac", [float("nan")])) / max(len(vis_acc.get("active_frac", [])), 1),
                "zero_coord_frac_eff": sum(vis_acc.get("zero_effective_displacement_frac", [float("nan")])) / max(len(vis_acc.get("zero_effective_displacement_frac", [])), 1),
                "delta_visibility_mse": mean_delta_visibility_mse,
                # Deprecated as a Delta selector:
                # "empirical_snap_rms": math.sqrt(mean_delta_visibility_mse) if math.isfinite(mean_delta_visibility_mse) and mean_delta_visibility_mse >= 0.0 else float("nan"),
                "lowbit_visibility_nmse": sum(vis_acc.get("delta_visibility_nmse", [float("nan")])) / max(len(vis_acc.get("delta_visibility_nmse", [])), 1),
            }
        )

    for row in visibility_rows:
        h = float(row["h"])
        if 2.0 * h in d2_by_h:
            row["corr_d2_h_2h"] = pearson(d2_by_h[h], d2_by_h[2.0 * h])
            row["sign_flip_rate_h_2h"] = sign_flip_rate(d2_by_h[h], d2_by_h[2.0 * h])
        else:
            row["corr_d2_h_2h"] = ""
            row["sign_flip_rate_h_2h"] = ""
        strict_pass = (
            float(row["alignment_eff"]) >= 0.99
            and 0.9 <= float(row["norm_ratio_eff"]) <= 1.1
            and float(row["zero_coord_frac_eff"]) <= 0.10
        )
        lowbit_pass = (
            float(row["active_frac_eff"]) >= 0.01
            and float(row["alignment_eff"]) >= 0.30
            and 0.3 <= float(row["norm_ratio_eff"]) <= 3.0
        )
        row["strict_fp16_visibility_pass"] = strict_pass
        row["lowbit_window_visibility_pass"] = lowbit_pass

    strict_candidates = [r for r in visibility_rows if r["strict_fp16_visibility_pass"]]
    lowbit_candidates = [r for r in visibility_rows if r["lowbit_window_visibility_pass"]]

    clean_g_rows: List[Dict[str, object]] = []
    l_rows: List[Dict[str, object]] = []
    model.float()
    copy_clean_to_model(params, master32, None, 0.0, 0.0)
    with torch.no_grad():
        base_loss = loss_value(model, batch)
    old_tf32_matmul = torch.backends.cuda.matmul.allow_tf32
    old_tf32_cudnn = torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        for h_clean in H_GRID:
            vals: List[float] = []
            for i in range(args.m_g):
                g.manual_seed(direction_seeds[i])
                directions = smoke.sample_directions(master, g)
                vals.append(clean32_d2(model, params, master32, batch, directions, h_clean))
            clean_g_rows.append(
                {
                    "h": h_clean,
                    "G_clean32_abs": math.sqrt(math.pi / 2.0) * (sum(abs(v) for v in vals) / max(len(vals), 1)),
                    "G_clean32_rms": math.sqrt(sum(v * v for v in vals) / max(len(vals), 1)),
                    "d2_clean32_mean": sum(vals) / max(len(vals), 1),
                    "d2_clean32_abs_mean": sum(abs(v) for v in vals) / max(len(vals), 1),
                }
            )
        for h2 in H_GRID:
            lambdas: List[float] = []
            ks: List[float] = []
            for i in range(args.m_l):
                g.manual_seed(direction_seeds[i])
                directions = smoke.sample_directions(master, g)
                copy_clean_to_model(params, master32, directions, h2, 1.0)
                l1 = loss_value(model, batch)
                copy_clean_to_model(params, master32, directions, 2.0 * h2, 1.0)
                l2 = loss_value(model, batch)
                copy_clean_to_model(params, master32, None, 0.0, 0.0)
                k = (l2 - 2.0 * l1 + base_loss) / (h2 * h2)
                norm_sq = direction_norm_sq(directions)
                lambdas.append(abs(k) / (norm_sq + EPS))
                ks.append(k)
            t = torch.tensor(lambdas, dtype=torch.float64)
            kt = torch.tensor(ks, dtype=torch.float64)
            med = torch.median(kt)
            mad = torch.median((kt - med).abs())
            row = {
                "h2": h2,
                "lambda_q50": float(torch.quantile(t, 0.50)),
                "lambda_q90": float(torch.quantile(t, 0.90)),
                "lambda_q95": float(torch.quantile(t, 0.95)),
                "median_abs_K": float(torch.median(kt.abs())),
                "MAD_K": float(mad),
                "SNR2": float(torch.median(kt.abs()) / (1.4826 * mad + EPS)),
                "finite_rate": float(torch.isfinite(t).float().mean()),
            }
            l_rows.append(row)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_tf32_matmul
        torch.backends.cudnn.allow_tf32 = old_tf32_cudnn
        model.half()
        smoke.restore_master(params, master)

    l_selected, l_status = choose_l_plateau(l_rows)
    l_hat = float(l_selected.get("lambda_q90", float("nan")))
    by_h = {float(r.get("h")): r for r in sweep_rows if r.get("h") != ""}

    clean_g_by_h = {float(r["h"]): r for r in clean_g_rows}
    clean_g_primary_h = 3e-4 if 3e-4 in clean_g_by_h else H_GRID[0]
    clean_g_primary = float(clean_g_by_h[clean_g_primary_h]["G_clean32_abs"])
    clean_stable_vals = [
        float(clean_g_by_h[h]["G_clean32_abs"])
        for h in (1e-4, 3e-4, 1e-3)
        if h in clean_g_by_h and math.isfinite(float(clean_g_by_h[h]["G_clean32_abs"]))
    ]
    clean_g_median = sorted(clean_stable_vals)[len(clean_stable_vals) // 2] if clean_stable_vals else clean_g_primary
    delta_scale = float(delta_stats["delta_int4_rtnclip_scale_rms"])
    delta_sqrt6 = delta_scale / math.sqrt(6.0)

    precision_name = "int8" if args.bitwidth == 8 else "int4" if args.bitwidth == 4 else f"int{args.bitwidth}"
    corrected = simple2pt_corrected(
        precision_name,
        d_trainable,
        l_hat,
        scale_rms=delta_scale,
        clean32_g_median=clean_g_median,
        clean32_g_h3e4=clean_g_primary,
    )
    h_cont = float(corrected["hstar_cont"])
    h_grid = float(corrected["hstar_nearest_grid"])

    # Deprecated exploratory h-star grids are intentionally disabled. The old
    # options were:
    #   Delta in {raw scale RMS, empirical snap RMS, scale RMS / sqrt(6)}
    #   G in {raw low-bit absG, clean32 fixed-h absG, clean32 median absG}
    # Keep the implementation single-path so generated summaries cannot mix
    # selector variants.

    nearest_ref = by_h.get(h_grid, {})
    hstar_eval = [
        {
            "selector_name": corrected["selector_name"],
            "model": "roberta-large",
            "dataset": "sst-5",
            "seed": args.seed,
            "precision_oracle": f"rtnclip_{precision_name}_g128_fake_quant_forward",
            "Delta_mode": corrected["Delta_mode"],
            "Delta_value": corrected["Delta_value"],
            "G_method": corrected["G_mode"],
            "G_hat": corrected["G_value"],
            "h_G": "median_1e-4_3e-4_1e-3",
            "G_selection_status": "clean32_absG_median_only",
            "L_mode": corrected["L_mode"],
            "L_q": corrected["L_q"],
            "L_hat": corrected["L_hat"],
            "h2_L": l_selected.get("h2", ""),
            "L_selection_status": l_status,
            "d_trainable": d_trainable,
            "d_quantized_linear": d_quantized,
            "hstar_cont": h_cont,
            "hstar_nearest_grid": h_grid,
            "nearest_grid_best_eval_acc": nearest_ref.get("best_eval_acc", ""),
            "nearest_grid_last_eval_acc": nearest_ref.get("last_eval_acc", ""),
            "nearest_grid_lowbit_true_nmse": nearest_ref.get("lowbit_true_nmse", ""),
            "nearest_grid_lowbit_true_corr": nearest_ref.get("lowbit_true_corr", ""),
            "sweep_best_acc_h": sweep_summary.get("best_acc_h", ""),
            "sweep_best_acc": sweep_summary.get("best_acc", ""),
            "sweep_best_last_h": sweep_summary.get("best_last_h", ""),
            "sweep_best_last_acc": sweep_summary.get("best_last_acc", ""),
            "sweep_best_visibility_h": sweep_summary.get("best_visibility_h", ""),
            "hstar_over_best_acc_h": h_cont / float(sweep_summary["best_acc_h"]) if sweep_summary.get("best_acc_h") else "",
            "nearest_over_best_acc_h": h_grid / float(sweep_summary["best_acc_h"]) if sweep_summary.get("best_acc_h") else "",
            "notes": corrected["notes"],
        }
    ]

    components = [
        {
            "model": "roberta-large",
            "dataset": "sst-5",
            "seed": args.seed,
            "bitwidth": args.bitwidth,
            "group_size": args.group_size,
            "batch_size": args.batch_size,
            "m_g": args.m_g,
            "m_l": args.m_l,
            "sampler_name": type(sampler).__name__,
            "d_trainable": d_trainable,
            "d_quantized_linear": d_quantized,
            **delta_stats,
            **qstats,
            "Delta_mode": corrected["Delta_mode"],
            "Delta_value": corrected["Delta_value"],
            "G_method": corrected["G_mode"],
            "G_hat": corrected["G_value"],
            "h_G": "median_1e-4_3e-4_1e-3",
            "G_selection_status": "clean32_absG_median_only",
            "G_clean32_abs_h3e-4": clean_g_by_h.get(3e-4, {}).get("G_clean32_abs", ""),
            "G_clean32_abs_h1e-3": clean_g_by_h.get(1e-3, {}).get("G_clean32_abs", ""),
            "G_clean32_abs_median_1e-4_3e-4_1e-3": clean_g_median,
            "delta_scale_rms_over_sqrt6": delta_sqrt6,
            # Deprecated Delta diagnostics are intentionally not used by h-star:
            # "delta_empirical_snap_rms_h1e-3": ...,
            # "delta_empirical_snap_rms_first_lowbit_pass": ...,
            "L_clean32_q90": l_hat,
            "L_mode": corrected["L_mode"],
            "L_q": corrected["L_q"],
            "L_hat": corrected["L_hat"],
            "h2_L": l_selected.get("h2", ""),
            "L_selection_status": l_status,
            "hstar_cont": h_cont,
            "hstar_nearest_grid": h_grid,
        }
    ]

    write_csv(out_dir / "visibility_and_G_candidates.csv", visibility_rows)
    write_csv(out_dir / "clean32_G_candidates.csv", clean_g_rows)
    write_csv(out_dir / "L_candidates.csv", l_rows)
    write_csv(out_dir / "hstar_components.csv", components)
    write_csv(out_dir / "hstar_eval_against_sweep.csv", hstar_eval)
    write_json(
        out_dir / "diagnostics.json",
        {
            "env": env_report(),
            "sweep_summary": sweep_summary,
            "strict_fp16_visibility_candidates": [r["h"] for r in strict_candidates],
            "lowbit_window_visibility_candidates": [r["h"] for r in lowbit_candidates],
            "default_selector": corrected,
            "legacy_variants_written": False,
            "notes": [
                "Analysis-only; no training launched.",
                "Default selector is simple2pt_corrected.",
                "Only one Delta/G/L path is enabled: scale RMS / sqrt(6), clean32 absG median, L_clean32 q90.",
                "Legacy exploratory Delta/G/L variants are commented out and are not written.",
                "Strict FP16 visibility gate is reported separately because low-bit displacement geometry is intentionally quantized.",
            ],
        },
    )

    lines = [
        f"# INT{args.bitwidth} SST-5 simple2pt_corrected h-star check",
        "",
        f"Output directory: `{out_dir}`",
        "",
        "Selector tested: `simple2pt_corrected` (修正简单两点法).",
        "",
        "| quantity | value |",
        "|---|---:|",
        f"| precision branch | {precision_name} |",
        f"| Delta RTNClip scale RMS | {delta_stats['delta_int4_rtnclip_scale_rms']:.6g} |",
        f"| selected Delta mode | {corrected['Delta_mode']} |",
        f"| selected Delta | {float(corrected['Delta_value']):.6g} |",
        f"| G clean32 absG at h=3e-4 | {clean_g_primary:.6g} |",
        f"| G clean32 median absG | {clean_g_median:.6g} |",
        f"| selected G mode | {corrected['G_mode']} |",
        f"| selected G | {float(corrected['G_value']):.6g} |",
        f"| L clean32 q90 | {l_hat:.6g} |",
        f"| h2_L | {float(l_selected.get('h2', float('nan'))):.6g} |",
        f"| L selection | {l_status} |",
        f"| d trainable | {d_trainable} |",
        f"| hstar_cont | {h_cont:.6g} |",
        f"| nearest grid h | {h_grid:.6g} |",
        "",
        f"## Comparison to current INT{args.bitwidth} sweep",
        "",
        "| reference | h | metric |",
        "|---|---:|---:|",
        f"| sweep best best_eval_acc | {float(sweep_summary.get('best_acc_h', float('nan'))):.6g} | {float(sweep_summary.get('best_acc', float('nan'))):.6g} |",
        f"| sweep best last_eval_acc | {float(sweep_summary.get('best_last_h', float('nan'))):.6g} | {float(sweep_summary.get('best_last_acc', float('nan'))):.6g} |",
        f"| sweep best lowbit visibility nMSE | {float(sweep_summary.get('best_visibility_h', float('nan'))):.6g} | {float(sweep_summary.get('best_visibility_lowbit_true_nmse', float('nan'))):.6g} |",
        f"| hstar_cont / best-training-h | {h_cont / float(sweep_summary.get('best_acc_h', float('nan'))):.6g} | |",
        f"| nearest-grid / best-training-h | {h_grid / float(sweep_summary.get('best_acc_h', float('nan'))):.6g} | |",
        "",
        "Interpretation: `simple2pt_corrected` is now the only enabled h-star selector: scale RMS / sqrt(6) Delta, clean32 absG median G, and L_clean32 q90.",
        "",
    ]
    (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"Analysis output directory: {out_dir}")
    print(f"hstar_cont={h_cont:.6g}, nearest_grid={h_grid:.6g}")
    print(f"sweep_best_acc_h={sweep_summary.get('best_acc_h')}, sweep_best_visibility_h={sweep_summary.get('best_visibility_h')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
