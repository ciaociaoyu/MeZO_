#!/usr/bin/env python
"""Estimate INT4 RTNClip simple2pt_corrected h-star on full-data tasks.

This is a planning/preflight utility. It uses the existing RTNClip shared-grid
forward implementation and the retained simple2pt_corrected selector from
``analyze_int4_sst5_calibrated_hstar.py``. It does not train.
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
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = REPO_ROOT / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import smoke_rtnclip_roberta_sst5 as smoke  # noqa: E402
from analyze_int4_sst5_calibrated_hstar import (  # noqa: E402
    H_GRID,
    EPS,
    choose_l_plateau,
    clean32_d2,
    copy_clean_to_model,
    direction_norm_sq,
    loss_value,
    nearest_grid,
    quantized_d2,
    simple2pt_corrected,
    weighted_int4_delta,
)
from rtnclip_roberta_sst5_batch import (  # noqa: E402
    CURRENT_SPARSE_MASK_STRATEGY,
    LEGACY_ABS_SPARSE_MASK_STRATEGIES,
    build_sparse_masks,
    build_task_grad_sparse_masks,
    inject_prefix_for_training,
    reset_run_seed,
)


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: List[Dict[str, object]], fieldnames: Optional[Sequence[str]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys: List[str] = []
        for row in rows:
            for key in row:
                if key not in keys:
                    keys.append(key)
        fieldnames = keys
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT), text=True).strip()
    except Exception:
        return ""


def env_report() -> Dict[str, object]:
    out: Dict[str, object] = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "hostname": socket.gethostname(),
        "python": sys.executable,
        "python_version": sys.version.replace("\n", " "),
        "conda_default_env": os.environ.get("CONDA_DEFAULT_ENV", ""),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "torch_version": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "git_commit": git_commit(),
    }
    if torch.cuda.is_available():
        out["gpu_name"] = torch.cuda.get_device_name(0)
        out["gpu_count"] = torch.cuda.device_count()
    return out


def make_loader_args(task_name: str, args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        repo_root=REPO_ROOT,
        model_id=args.model_id,
        task_name=task_name,
        seed=args.seed,
        data_seed=args.data_seed,
        batch_size=args.batch_size,
        eval_batch_size=args.batch_size,
        dataset_mode="full",
        data_dir=None,
        num_k=16,
    )


def sample_direction(
    master: Dict[str, torch.Tensor],
    seed: int,
    masks: Optional[Dict[str, torch.Tensor]],
    active_names: Optional[Iterable[str]] = None,
) -> Dict[str, torch.Tensor]:
    first = next(iter(master.values()))
    gen = torch.Generator(device=first.device).manual_seed(int(seed))
    directions = smoke.sample_directions(master, gen)
    if active_names is not None:
        active = set(active_names)
        for name in list(directions.keys()):
            if name not in active:
                directions[name] = torch.zeros_like(directions[name])
    if masks is not None:
        for name, mask in masks.items():
            if name in directions:
                directions[name] = directions[name] * mask.to(device=directions[name].device, dtype=directions[name].dtype)
    return directions


def prefix_lattice_delta(master: Dict[str, torch.Tensor], prefix_names: Iterable[str], dtype_name: str) -> Dict[str, float]:
    """RMS trainable-prefix lattice spacing.

    Prefix parameters are not INT4-quantized in the main prefix setting.  Their
    visibility scale therefore comes from the trainable prefix dtype, not from
    the frozen base INT4 RTNClip scale.
    """
    chunks: List[torch.Tensor] = []
    dtype_key = str(dtype_name or "fp16").strip().lower()
    for name in prefix_names:
        tensor = master.get(name)
        if tensor is None or tensor.numel() == 0:
            continue
        if dtype_key in {"fp32", "float32", "32"}:
            t32 = tensor.detach().float()
            step = torch.nextafter(t32, torch.full_like(t32, float("inf"))) - t32
            step = step.abs().clamp_min(torch.finfo(torch.float32).tiny)
            chunks.append(step.reshape(-1).cpu())
            continue
        # CUDA does not implement nextafter for fp16 on all deployed PyTorch
        # builds.  Compute the binary16 spacing directly from the exponent:
        # normal spacing is 2^(floor(log2(abs(x))) - 10); subnormal spacing is
        # 2^-24.  This is the prefix trainable lattice, not the base INT4 grid.
        t = tensor.detach().abs().float()
        min_normal = 2.0 ** -14
        min_subnormal = 2.0 ** -24
        safe = torch.clamp(t, min=min_normal)
        spacing = torch.pow(torch.full_like(safe, 2.0), torch.floor(torch.log2(safe)) - 10.0)
        step = torch.where(t < min_normal, torch.full_like(t, min_subnormal), spacing)
        chunks.append(step.reshape(-1).cpu())
    if not chunks:
        return {
            "delta_int4_rtnclip_scale_rms": float("nan"),
            "delta_int4_rtnclip_scale_mean": float("nan"),
            "prefix_lattice_delta_ulp_rms": float("nan"),
            "prefix_lattice_delta_ulp_mean": float("nan"),
            "prefix_lattice_delta_ulp_median": float("nan"),
            "prefix_lattice_delta_ulp_p90": float("nan"),
            "prefix_lattice_delta_ulp_p95": float("nan"),
            "num_quantized_values_for_delta": 0,
        }
    vals = torch.cat(chunks).double()
    rms = float(torch.sqrt((vals * vals).mean()))
    return {
        "delta_int4_rtnclip_scale_rms": rms,
        "delta_int4_rtnclip_scale_mean": float(vals.mean()),
        "prefix_lattice_dtype": dtype_key,
        "prefix_lattice_delta_ulp_rms": rms,
        "prefix_lattice_delta_ulp_mean": float(vals.mean()),
        "prefix_lattice_delta_ulp_median": float(torch.quantile(vals, 0.50)),
        "prefix_lattice_delta_ulp_p90": float(torch.quantile(vals, 0.90)),
        "prefix_lattice_delta_ulp_p95": float(torch.quantile(vals, 0.95)),
        "num_quantized_values_for_delta": int(vals.numel()),
    }


def grouped_mask_counts(mask: torch.Tensor, group_size: int) -> torch.Tensor:
    if mask.ndim != 2:
        raise ValueError(f"Expected 2D Linear mask, got {tuple(mask.shape)}")
    out_features, in_features = mask.shape
    num_groups = int(math.ceil(in_features / group_size))
    padded_cols = num_groups * group_size
    pad_cols = padded_cols - in_features
    m = mask.to(dtype=torch.float32)
    if pad_cols:
        m = F.pad(m, (0, pad_cols))
    return m.reshape(out_features, num_groups, group_size).sum(dim=-1).double().unsqueeze(-1)


def weighted_delta_with_optional_masks(
    states: Dict[str, smoke.RTNClipState],
    masks: Optional[Dict[str, torch.Tensor]],
) -> Dict[str, float]:
    if masks is None:
        return weighted_int4_delta(states)
    scale_sq_sum = 0.0
    scale_sum = 0.0
    values = 0.0
    flat_scales: List[torch.Tensor] = []
    for name, state in states.items():
        mask = masks.get(name)
        if mask is None:
            continue
        counts = grouped_mask_counts(mask, state.group_size)
        scales = state.scales.double()
        scale_sq_sum += float((scales.square() * counts).sum().detach().cpu())
        scale_sum += float((scales * counts).sum().detach().cpu())
        values += float(counts.sum().detach().cpu())
        selected = state.scales.detach().float()[counts.squeeze(-1).bool()]
        if selected.numel():
            flat_scales.append(selected.reshape(-1).cpu())
    all_scales = torch.cat(flat_scales) if flat_scales else torch.empty(0)
    denom = max(values, 1.0)
    return {
        "delta_int4_rtnclip_scale_rms": math.sqrt(scale_sq_sum / denom),
        "delta_int4_rtnclip_scale_mean": scale_sum / denom,
        "scale_median_unweighted": float(all_scales.median()) if all_scales.numel() else float("nan"),
        "scale_p90_unweighted": float(torch.quantile(all_scales, 0.90)) if all_scales.numel() else float("nan"),
        "scale_p95_unweighted": float(torch.quantile(all_scales, 0.95)) if all_scales.numel() else float("nan"),
        "num_quantized_values_for_delta": int(values),
    }


def summarize_abs_g(vals: List[float]) -> Tuple[float, float, float, float]:
    finite = [float(v) for v in vals if math.isfinite(float(v))]
    if not finite:
        return float("nan"), float("nan"), float("nan"), float("nan")
    abs_mean = sum(abs(v) for v in finite) / len(finite)
    return (
        math.sqrt(math.pi / 2.0) * abs_mean,
        math.sqrt(sum(v * v for v in finite) / len(finite)),
        sum(finite) / len(finite),
        abs_mean,
    )


def estimate_one(task_name: str, direction_mode: str, args: argparse.Namespace, out_dir: Path) -> Dict[str, object]:
    t0 = time.time()
    device = torch.device("cuda:0")
    os.environ.setdefault("DATALOADER_SHUFFLE", "True")
    reset_run_seed(args.seed)

    load_args = make_loader_args(task_name, args)
    orig_torch_load = torch.load

    def _compat_torch_load(*load_args_, **load_kwargs_):
        load_kwargs_.setdefault("weights_only", False)
        return orig_torch_load(*load_args_, **load_kwargs_)

    torch.load = _compat_torch_load
    try:
        model, train_loader, _dev_loader, data_args, sampler = smoke.load_prompt_model_and_data(load_args, device)
    finally:
        torch.load = orig_torch_load

    batch = smoke.move_batch(next(iter(train_loader)), device)
    prefix_names: List[str] = []
    prefix_status = ""
    if direction_mode == "prefix":
        reset_run_seed(args.seed)
        prefix_names, prefix_status = inject_prefix_for_training(
            model,
            num_prefix=5,
            prefix_precision=args.prefix_precision,
            init_strategy=args.prefix_init_strategy,
        )

    params = smoke.named_parameter_map(model)
    master_dtype = torch.float32 if str(args.master_dtype).lower() == "fp32" else torch.float16
    master = {name: p.detach().clone().to(device=device, dtype=master_dtype) for name, p in params.items() if p.is_floating_point()}
    master32 = {name: tensor.detach().float().clone() for name, tensor in master.items()}
    q_names = [name for name in smoke.linear_weight_names(model) if name in params and "prefix" not in name]
    states, quant_rows = smoke.refresh_quantizer_states(master, q_names, args.bitwidth, args.group_size)
    qstats = smoke.aggregate_quantizer_stats(quant_rows, {name: params[name].numel() for name in q_names})

    masks: Optional[Dict[str, torch.Tensor]] = None
    mask_stats: Dict[str, object] = {
        "direction_mode": direction_mode,
        "sparse_ratio": "",
        "sparse_mask_strategy": "",
        "sparse_rescale": "",
        "active_params_all": sum(int(t.numel()) for t in master.values()),
        "mask_active_frac_all": 1.0,
        "active_params_quantized_linear": sum(int(master[name].numel()) for name in q_names),
        "mask_active_frac_quantized_linear": 1.0,
    }
    if direction_mode == "sparse":
        sparse_strategy = str(args.sparse_mask_strategy).strip().lower()
        if sparse_strategy in LEGACY_ABS_SPARSE_MASK_STRATEGIES:
            sparse_strategy = CURRENT_SPARSE_MASK_STRATEGY
        if sparse_strategy == CURRENT_SPARSE_MASK_STRATEGY:
            masks, mask_stats = build_task_grad_sparse_masks(
                model,
                params,
                master,
                train_loader,
                device,
                sparse_ratio=args.sparse_ratio,
                quantized_names=q_names,
                mask_batches=int(args.sparse_mask_batches),
                mask_scope=str(args.sparse_mask_scope),
            )
        else:
            masks, mask_stats = build_sparse_masks(
                master,
                sparse_ratio=args.sparse_ratio,
                quantized_names=q_names,
                mask_strategy=sparse_strategy,
            )
    elif direction_mode == "prefix":
        active = sum(int(master[name].numel()) for name in prefix_names if name in master)
        total = sum(int(t.numel()) for t in master.values())
        mask_stats = {
            "direction_mode": direction_mode,
            "sparse_ratio": "",
            "sparse_mask_strategy": "",
            "sparse_rescale": "",
            "active_params_all": active,
            "mask_active_frac_all": active / max(total, 1),
            "active_params_quantized_linear": 0,
            "mask_active_frac_quantized_linear": 0.0,
            "prefix_num": 5,
            "prefix_status": prefix_status,
            "prefix_param_count": active,
            "prefix_param_names": ";".join(prefix_names),
        }
    d_trainable = int(mask_stats["active_params_all"])
    d_quantized = int(mask_stats["active_params_quantized_linear"])
    delta_stats = prefix_lattice_delta(master, prefix_names, args.master_dtype) if direction_mode == "prefix" else weighted_delta_with_optional_masks(states, masks)

    direction_seeds = [args.seed * 1_000_003 + i for i in range(max(args.m_g, args.m_l))]
    lowbit_g_rows: List[Dict[str, object]] = []
    lowbit_l_rows: List[Dict[str, object]] = []
    clean_g_rows: List[Dict[str, object]] = []
    l_rows: List[Dict[str, object]] = []

    # Precision-aware low-bit G diagnostics: use the same shared-grid
    # quantized forward oracle as training/probing. The grid comes from the
    # unperturbed FP16 master weights in ``states``; +h and -h fresh-round on
    # that same grid inside quantized_d2.
    smoke.copy_master_to_model(params, master, None, 0.0, 0.0, states)
    with torch.no_grad():
        base_lowbit_loss = loss_value(model, batch)
    for h_lowbit in H_GRID:
        vals = []
        for i in range(args.m_g):
            directions = sample_direction(master, direction_seeds[i], masks, prefix_names if direction_mode == "prefix" else None)
            vals.append(quantized_d2(model, params, master, states, batch, directions, h_lowbit))
        g_abs, g_rms, d2_mean, d2_abs_mean = summarize_abs_g(vals)
        lowbit_g_rows.append(
            {
                "task_name": task_name,
                "direction_mode": direction_mode,
                "h": h_lowbit,
                "G_lowbit_abs": g_abs,
                "G_lowbit_rms": g_rms,
                "d2_lowbit_mean": d2_mean,
                "d2_lowbit_abs_mean": d2_abs_mean,
                "G_oracle": f"rtnclip_int{args.bitwidth}_shared_grid_fd",
            }
        )
    for h2 in H_GRID:
        lambdas = []
        ks = []
        for i in range(args.m_l):
            directions = sample_direction(master, direction_seeds[i], masks, prefix_names if direction_mode == "prefix" else None)
            smoke.copy_master_to_model(params, master, directions, h2, 1.0, states)
            l1 = loss_value(model, batch)
            smoke.copy_master_to_model(params, master, directions, 2.0 * h2, 1.0, states)
            l2 = loss_value(model, batch)
            smoke.restore_master(params, master)
            k = (l2 - 2.0 * l1 + base_lowbit_loss) / (h2 * h2)
            norm_sq = direction_norm_sq(directions)
            lambdas.append(abs(k) / (norm_sq + EPS))
            ks.append(k)
        t = torch.tensor(lambdas, dtype=torch.float64)
        kt = torch.tensor(ks, dtype=torch.float64)
        med = torch.median(kt)
        mad = torch.median((kt - med).abs())
        lowbit_l_rows.append(
            {
                "task_name": task_name,
                "direction_mode": direction_mode,
                "h2": h2,
                "lambda_q50": float(torch.quantile(t, 0.50)),
                "lambda_q90": float(torch.quantile(t, 0.90)),
                "lambda_q95": float(torch.quantile(t, 0.95)),
                "median_abs_K": float(torch.median(kt.abs())),
                "MAD_K": float(mad),
                "SNR2": float(torch.median(kt.abs()) / (1.4826 * mad + EPS)),
                "finite_rate": float(torch.isfinite(t).float().mean()),
                "L_oracle": f"rtnclip_int{args.bitwidth}_shared_grid_forward_second_diff",
            }
        )
    smoke.restore_master(params, master)

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
                directions = sample_direction(master, direction_seeds[i], masks, prefix_names if direction_mode == "prefix" else None)
                vals.append(clean32_d2(model, params, master32, batch, directions, h_clean))
            clean_g_rows.append(
                {
                    "task_name": task_name,
                    "direction_mode": direction_mode,
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
                directions = sample_direction(master, direction_seeds[i], masks, prefix_names if direction_mode == "prefix" else None)
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
            l_rows.append(
                {
                    "task_name": task_name,
                    "direction_mode": direction_mode,
                    "h2": h2,
                    "lambda_q50": float(torch.quantile(t, 0.50)),
                    "lambda_q90": float(torch.quantile(t, 0.90)),
                    "lambda_q95": float(torch.quantile(t, 0.95)),
                    "median_abs_K": float(torch.median(kt.abs())),
                    "MAD_K": float(mad),
                    "SNR2": float(torch.median(kt.abs()) / (1.4826 * mad + EPS)),
                    "finite_rate": float(torch.isfinite(t).float().mean()),
                }
            )
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_tf32_matmul
        torch.backends.cudnn.allow_tf32 = old_tf32_cudnn
        model.half()
        smoke.restore_master(params, master)

    l_selected, l_status = choose_l_plateau(l_rows)
    l_hat = float(l_selected.get("lambda_q90", float("nan")))
    lowbit_l_selected, lowbit_l_status = choose_l_plateau(lowbit_l_rows)
    lowbit_l_hat = float(lowbit_l_selected.get("lambda_q90", float("nan")))
    clean_g_by_h = {float(r["h"]): r for r in clean_g_rows}
    lowbit_g_by_h = {float(r["h"]): r for r in lowbit_g_rows}
    clean_g_primary_h = 3e-4 if 3e-4 in clean_g_by_h else H_GRID[0]
    clean_g_primary = float(clean_g_by_h[clean_g_primary_h]["G_clean32_abs"])
    clean_stable_vals = [
        float(clean_g_by_h[h]["G_clean32_abs"])
        for h in (1e-4, 3e-4, 1e-3)
        if h in clean_g_by_h and math.isfinite(float(clean_g_by_h[h]["G_clean32_abs"]))
    ]
    clean_g_median = sorted(clean_stable_vals)[len(clean_stable_vals) // 2] if clean_stable_vals else clean_g_primary
    lowbit_stable_vals = [
        float(lowbit_g_by_h[h]["G_lowbit_abs"])
        for h in (1e-4, 3e-4, 1e-3)
        if h in lowbit_g_by_h and math.isfinite(float(lowbit_g_by_h[h]["G_lowbit_abs"]))
    ]
    lowbit_g_median = sorted(lowbit_stable_vals)[len(lowbit_stable_vals) // 2] if lowbit_stable_vals else float("nan")
    selected_g = lowbit_g_median if math.isfinite(lowbit_g_median) and lowbit_g_median > 0.0 else clean_g_median
    selected_g_mode = (
        f"rtnclip_int{args.bitwidth}_shared_grid_absG_median_1e-4_3e-4_1e-3"
        if math.isfinite(lowbit_g_median) and lowbit_g_median > 0.0
        else "clean32_absG_median_1e-4_3e-4_1e-3_fallback"
    )
    g_selection_status = "precision_aware_lowbit_G" if selected_g_mode.startswith("rtnclip_") else "clean32_G_fallback"
    delta_scale = float(delta_stats["delta_int4_rtnclip_scale_rms"])
    l_for_selector = lowbit_l_hat if direction_mode == "prefix" and math.isfinite(lowbit_l_hat) and lowbit_l_hat > 0.0 else l_hat
    # simple2pt_corrected reports Delta = scale_rms / sqrt(6).  For prefix
    # mode, delta_scale is already the FP16 ULP RMS target Delta, so pass the
    # equivalent scale value to avoid using frozen-base INT4 scales.
    scale_for_selector = delta_scale * math.sqrt(6.0) if direction_mode == "prefix" else delta_scale
    corrected = simple2pt_corrected(
        "int4",
        d_trainable,
        l_for_selector,
        scale_rms=scale_for_selector,
        clean32_g_median=clean_g_median,
        clean32_g_h3e4=clean_g_primary,
        selected_g=selected_g,
        selected_g_mode=selected_g_mode,
    )
    if direction_mode == "prefix":
        corrected["Delta_mode"] = f"prefix_{args.master_dtype}_delta_ulp_rms"
        corrected["Delta_value"] = delta_scale
        corrected["L_mode"] = "L_lowbit_q90_prefix_quantized_base"
        corrected["L_q"] = "q90_lowbit"
        corrected["L_hat"] = l_for_selector
        corrected["notes"] = (
            f"prefix selector: Delta = prefix {args.master_dtype} ULP RMS; G = prefix-only "
            "low-bit shared-grid absG median; L = prefix-only low-bit "
            "quantized-base second-diff q90; frozen base excluded from G/L/d"
        )
    h_cont = float(corrected["hstar_cont"])
    h_grid = float(corrected["hstar_nearest_grid"])
    row = {
        "dataset": smoke.normalize_task_name(task_name),
        "task_name": smoke.normalize_task_name(task_name),
        "model": "roberta-large",
        "dataset_mode": "full",
        "seed": args.seed,
        "data_seed": args.data_seed,
        "batch_size": args.batch_size,
        "master_dtype": args.master_dtype,
        "prefix_precision": args.prefix_precision if direction_mode == "prefix" else "",
        "prefix_init_strategy": args.prefix_init_strategy if direction_mode == "prefix" else "",
        "precision": "int4",
        "quantizer": "G128_RTNClip_shared_grid_fake_quant",
        "bitwidth": args.bitwidth,
        "group_size": args.group_size,
        "direction_mode": direction_mode,
        "selector_name": corrected["selector_name"],
        "Delta_mode": corrected["Delta_mode"],
        "Delta_value": corrected["Delta_value"],
        "delta_int4_rtnclip_scale_rms": delta_scale,
        "delta_scale_rms_over_sqrt6": delta_scale / math.sqrt(6.0),
        "G_mode": corrected["G_mode"],
        "G_value": corrected["G_value"],
        "G_clean32_abs_h3e-4": clean_g_primary,
        "G_clean32_abs_median_1e-4_3e-4_1e-3": clean_g_median,
        "G_lowbit_abs_h1e-4": lowbit_g_by_h.get(1e-4, {}).get("G_lowbit_abs", ""),
        "G_lowbit_abs_h3e-4": lowbit_g_by_h.get(3e-4, {}).get("G_lowbit_abs", ""),
        "G_lowbit_abs_h1e-3": lowbit_g_by_h.get(1e-3, {}).get("G_lowbit_abs", ""),
        "G_lowbit_abs_median_1e-4_3e-4_1e-3": lowbit_g_median,
        "G_selection_status": g_selection_status,
        "L_mode": corrected["L_mode"],
        "L_q": corrected["L_q"],
        "L_hat": corrected["L_hat"],
        "h2_L": l_selected.get("h2", ""),
        "L_selection_status": l_status,
        "L_lowbit_q90": lowbit_l_hat,
        "L_lowbit_h2": lowbit_l_selected.get("h2", ""),
        "L_lowbit_selection_status": lowbit_l_status,
        "L_lowbit_over_clean32": lowbit_l_hat / l_hat if math.isfinite(lowbit_l_hat) and math.isfinite(l_hat) and l_hat > 0.0 else "",
        "L_lowbit_diagnostic_only": True,
        "d_trainable": d_trainable,
        "d_quantized_linear": d_quantized,
        "hstar_cont": h_cont,
        "hstar_nearest_grid": h_grid,
        "hstar_nearest_grid_label": label_h(h_grid),
        "sampler_name": type(sampler).__name__,
        "resolved_data_dir": getattr(data_args, "data_dir", ""),
        "train_size": len(train_loader.dataset),
        "base_loss": base_loss,
        "runtime_sec": time.time() - t0,
        **delta_stats,
        **qstats,
        **mask_stats,
        "notes": corrected["notes"],
    }
    task_dir = out_dir / f"{row['task_name']}_{direction_mode}"
    write_csv(task_dir / "lowbit_G_candidates.csv", lowbit_g_rows)
    write_csv(task_dir / "lowbit_L_candidates.csv", lowbit_l_rows)
    write_csv(task_dir / "clean32_G_candidates.csv", clean_g_rows)
    write_csv(task_dir / "L_candidates.csv", l_rows)
    write_json(task_dir / "hstar_summary.json", row)
    write_csv(task_dir / "hstar_summary.csv", [row])

    del model, train_loader, batch, params, master, master32, states
    torch.cuda.empty_cache()
    return row


def label_h(h: float) -> str:
    if not math.isfinite(float(h)):
        return "nan"
    for raw, label in (
        (1e-5, "1e-5"),
        (3e-5, "3e-5"),
        (1e-4, "1e-4"),
        (3e-4, "3e-4"),
        (1e-3, "1e-3"),
        (1.5e-3, "1p5e-3"),
        (2e-3, "2e-3"),
        (3e-3, "3e-3"),
        (4e-3, "4e-3"),
        (5e-3, "5e-3"),
        (1e-2, "1e-2"),
    ):
        if abs(float(h) - raw) <= max(abs(raw) * 1e-9, 1e-15):
            return label
    return f"{h:g}".replace(".", "p")


def fmt_float(value: object) -> str:
    try:
        v = float(value)
    except Exception:
        return "NA"
    return f"{v:.6g}" if math.isfinite(v) else "NA"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_id", default="roberta-large")
    parser.add_argument("--tasks", nargs="+", default=["sst-2", "sst-5", "rte", "mnli", "trec"])
    parser.add_argument("--directions", nargs="+", choices=["dense", "sparse", "prefix"], default=["dense", "sparse"])
    parser.add_argument("--seed", type=int, default=16)
    parser.add_argument("--data_seed", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--bitwidth", type=int, default=4)
    parser.add_argument("--group_size", type=int, default=128)
    parser.add_argument("--m_g", type=int, default=8)
    parser.add_argument("--m_l", type=int, default=4)
    parser.add_argument("--sparse_ratio", type=float, default=0.1)
    parser.add_argument("--sparse_mask_strategy", choices=["highest_abs", "lowest_abs", "task_grad_static"], default=CURRENT_SPARSE_MASK_STRATEGY)
    parser.add_argument("--sparse_mask_batches", type=int, default=1)
    parser.add_argument("--sparse_mask_scope", choices=["linear_weight", "all_floating"], default="linear_weight")
    parser.add_argument("--prefix_precision", choices=["fp16", "fp32"], default="fp16")
    parser.add_argument("--prefix_init_strategy", choices=["random", "real_act", "real_act_with_random_fallback"], default="random")
    parser.add_argument("--master_dtype", choices=["fp16", "fp32"], default="fp16")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for h-star estimation.")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "env_report.json", env_report())

    rows: List[Dict[str, object]] = []
    failures: List[Dict[str, object]] = []
    for task_name in args.tasks:
        task = smoke.normalize_task_name(task_name)
        for direction_mode in args.directions:
            try:
                row = estimate_one(task, direction_mode, args, out_dir)
                rows.append(row)
                print(
                    f"{task} {direction_mode}: hstar_cont={row['hstar_cont']:.6g} "
                    f"nearest={row['hstar_nearest_grid_label']} G={row['G_value']:.6g} L={row['L_hat']:.6g}",
                    flush=True,
                )
            except Exception as exc:
                failure = {"task_name": task, "direction_mode": direction_mode, "error": repr(exc)}
                failures.append(failure)
                write_json(out_dir / f"failure_{task}_{direction_mode}.json", failure)
                print(f"FAILED {task} {direction_mode}: {exc}", flush=True)
                torch.cuda.empty_cache()
    write_csv(out_dir / "hstar_full_data_summary.csv", rows)
    write_json(out_dir / "failures.json", failures)
    md = [
        "# INT4 Full-Data h-star Summary",
        "",
        "Selector: `simple2pt_corrected`; dense/sparse Delta = RTNClip scale RMS / sqrt(6), G = precision-aware RTNClip shared-grid absG median over 1e-4/3e-4/1e-3. Prefix Delta = prefix trainable-dtype ULP RMS, G/L/d are prefix-only with INT4 base forward.",
        "",
        "Low-bit L is written as a diagnostic (`lowbit_L_candidates.csv`) but is not selected by default because the quantized-forward objective is staircase-like and its second differences can include snap/jump artifacts.",
        "",
        "| dataset | direction | hstar_cont | nearest | G | G mode | clean32 G | lowbit L / clean32 L | L | Delta | d_trainable |",
        "| --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in rows:
        md.append(
            f"| {r['task_name']} | {r['direction_mode']} | {fmt_float(r['hstar_cont'])} | "
            f"{r['hstar_nearest_grid_label']} | {fmt_float(r['G_value'])} | {r['G_mode']} | "
            f"{fmt_float(r['G_clean32_abs_median_1e-4_3e-4_1e-3'])} | "
            f"{fmt_float(r['L_lowbit_over_clean32'])} | {fmt_float(r['L_hat'])} | "
            f"{fmt_float(r['Delta_value'])} | {r['d_trainable']} |"
        )
    if failures:
        md.extend(["", "## Failures", ""])
        for f in failures:
            md.append(f"- {f['task_name']} {f['direction_mode']}: `{f['error']}`")
    (out_dir / "hstar_full_data_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
