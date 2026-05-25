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
    simple2pt_corrected,
    weighted_int4_delta,
)
from rtnclip_roberta_sst5_batch import build_sparse_masks  # noqa: E402


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


def sample_direction(master: Dict[str, torch.Tensor], seed: int, masks: Optional[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    first = next(iter(master.values()))
    gen = torch.Generator(device=first.device).manual_seed(int(seed))
    directions = smoke.sample_directions(master, gen)
    if masks is not None:
        for name, mask in masks.items():
            if name in directions:
                directions[name] = directions[name] * mask.to(device=directions[name].device, dtype=directions[name].dtype)
    return directions


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


def estimate_one(task_name: str, direction_mode: str, args: argparse.Namespace, out_dir: Path) -> Dict[str, object]:
    t0 = time.time()
    device = torch.device("cuda:0")
    os.environ.setdefault("DATALOADER_SHUFFLE", "True")
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

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
    params = smoke.named_parameter_map(model)
    master = {name: p.detach().clone().to(device=device, dtype=torch.float16) for name, p in params.items() if p.is_floating_point()}
    master32 = {name: tensor.detach().float().clone() for name, tensor in master.items()}
    q_names = smoke.linear_weight_names(model)
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
        masks, mask_stats = build_sparse_masks(
            master,
            sparse_ratio=args.sparse_ratio,
            quantized_names=q_names,
            mask_strategy=args.sparse_mask_strategy,
        )
    d_trainable = int(mask_stats["active_params_all"])
    d_quantized = int(mask_stats["active_params_quantized_linear"])
    delta_stats = weighted_delta_with_optional_masks(states, masks)

    direction_seeds = [args.seed * 1_000_003 + i for i in range(max(args.m_g, args.m_l))]
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
                directions = sample_direction(master, direction_seeds[i], masks)
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
                directions = sample_direction(master, direction_seeds[i], masks)
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
    corrected = simple2pt_corrected(
        "int4",
        d_trainable,
        l_hat,
        scale_rms=delta_scale,
        clean32_g_median=clean_g_median,
        clean32_g_h3e4=clean_g_primary,
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
        "L_mode": corrected["L_mode"],
        "L_q": corrected["L_q"],
        "L_hat": corrected["L_hat"],
        "h2_L": l_selected.get("h2", ""),
        "L_selection_status": l_status,
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_id", default="roberta-large")
    parser.add_argument("--tasks", nargs="+", default=["sst-2", "sst-5", "rte", "mnli", "trec"])
    parser.add_argument("--directions", nargs="+", choices=["dense", "sparse"], default=["dense", "sparse"])
    parser.add_argument("--seed", type=int, default=16)
    parser.add_argument("--data_seed", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--bitwidth", type=int, default=4)
    parser.add_argument("--group_size", type=int, default=128)
    parser.add_argument("--m_g", type=int, default=8)
    parser.add_argument("--m_l", type=int, default=4)
    parser.add_argument("--sparse_ratio", type=float, default=0.1)
    parser.add_argument("--sparse_mask_strategy", choices=["highest_abs", "lowest_abs"], default="highest_abs")
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
        "Selector: `simple2pt_corrected`; Delta = RTNClip scale RMS / sqrt(6), G = clean32 absG median over 1e-4/3e-4/1e-3, L = clean32 q90.",
        "",
        "| dataset | direction | hstar_cont | nearest | G | L | Delta | d_trainable |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in rows:
        md.append(
            f"| {r['task_name']} | {r['direction_mode']} | {float(r['hstar_cont']):.6g} | "
            f"{r['hstar_nearest_grid_label']} | {float(r['G_value']):.6g} | {float(r['L_hat']):.6g} | "
            f"{float(r['Delta_value']):.6g} | {r['d_trainable']} |"
        )
    if failures:
        md.extend(["", "## Failures", ""])
        for f in failures:
            md.append(f"- {f['task_name']} {f['direction_mode']}: `{f['error']}`")
    (out_dir / "hstar_full_data_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
