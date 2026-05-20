#!/usr/bin/env python
"""Phase D INT4 effective-displacement update experiment.

This runner is intentionally isolated from the production Trainer. It reuses the
existing RoBERTa/SST-5 RTNClip fake-quantized forward oracle and compares update
directions in the quantized Linear.weight space.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = REPO_ROOT / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import smoke_rtnclip_roberta_sst5 as smoke  # noqa: E402


H_GRID: List[Tuple[str, float]] = [
    ("5e-4", 5e-4),
    ("1e-3", 1e-3),
    ("2e-3", 2e-3),
    ("3e-3", 3e-3),
]

VARIANTS = ("standard", "effdir_global", "effdir_active", "effdir_secant")

SUMMARY_COLUMNS = [
    "run_name",
    "variant",
    "h_label",
    "h",
    "status",
    "steps_completed",
    "best_eval_acc",
    "best_eval_step",
    "last_eval_acc",
    "last_eval_step",
    "best_eval_loss",
    "last_eval_loss",
    "final_train_loss",
    "d_h_finite_rate",
    "skip_update_frac",
    "delta_q_norm_last",
    "ideal_displacement_norm_last",
    "norm_ratio_last",
    "alignment_last",
    "active_frac_last",
    "active_set_norm_ratio_last",
    "update_norm_last",
    "update_norm_ratio_vs_standard_last",
    "clipped_frac",
    "richardson_relerr_last",
    "seconds_per_step",
    "peak_gpu_mem",
    "run_dir",
    "notes",
]


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def append_jsonl(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(data, sort_keys=True, default=str) + "\n")


def write_csv(path: Path, rows: List[Dict[str, object]], columns: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def log_to(run_dir: Path, message: str) -> None:
    line = f"[{datetime.now().isoformat(timespec='seconds')}] {message}"
    print(line, flush=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "train.log").open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def stable_h_key(h: float) -> int:
    return int(round(float(h) * 1_000_000_000_000))


def sample_quantized_directions(
    master: Dict[str, torch.Tensor],
    q_names: Iterable[str],
    seed: int,
    h: float,
    step: int,
    variant: str,
) -> Dict[str, torch.Tensor]:
    first = next(iter(master.values()))
    gen = torch.Generator(device=first.device)
    variant_offset = sum(ord(c) for c in variant) * 997
    gen.manual_seed(int(seed) + stable_h_key(h) + step * 1_000_003 + variant_offset)
    return {
        name: torch.randn(master[name].shape, device=first.device, generator=gen, dtype=torch.float16)
        for name in q_names
    }


def copy_master_to_model_qspace(
    params: Dict[str, torch.nn.Parameter],
    master: Dict[str, torch.Tensor],
    directions: Optional[Dict[str, torch.Tensor]],
    h: float,
    sign: float,
    states: Dict[str, smoke.RTNClipState],
) -> None:
    with torch.no_grad():
        for name, param in params.items():
            if directions is not None and name in directions:
                value = master[name].float().add(directions[name].float(), alpha=sign * h)
            else:
                value = master[name]
            if name in states:
                value = smoke.quantize_with_state(value, states[name])
            param.copy_(value.to(dtype=param.dtype))


def evaluate_quantized(
    model,
    params: Dict[str, torch.nn.Parameter],
    master: Dict[str, torch.Tensor],
    q_names: List[str],
    bitwidth: int,
    group_size: int,
    dev_loader,
    device: torch.device,
    max_batches: int,
) -> Tuple[Optional[float], Optional[float]]:
    if max_batches == 0:
        return None, None
    states, _ = smoke.refresh_quantizer_states(master, q_names, bitwidth, group_size)
    copy_master_to_model_qspace(params, master, None, 0.0, 0.0, states)
    total_loss = 0.0
    total_correct = 0
    total_items = 0
    for idx, batch in enumerate(dev_loader):
        if max_batches > 0 and idx >= max_batches:
            break
        batch = smoke.move_batch(batch, device)
        loss, logits = smoke.forward_loss_and_logits(model, batch)
        labels = batch["labels"]
        total_loss += float(loss.detach().cpu()) * int(labels.numel())
        total_correct += int((logits.argmax(dim=-1) == labels).sum().detach().cpu())
        total_items += int(labels.numel())
    smoke.restore_master(params, master)
    if total_items == 0:
        return None, None
    return total_loss / total_items, total_correct / total_items


def compute_effective_stats(
    master: Dict[str, torch.Tensor],
    directions: Dict[str, torch.Tensor],
    states: Dict[str, smoke.RTNClipState],
    h: float,
) -> Dict[str, object]:
    device = next(iter(master.values())).device
    dot = torch.zeros((), device=device, dtype=torch.float64)
    delta_sq = torch.zeros_like(dot)
    intended_sq = torch.zeros_like(dot)
    u_sq = torch.zeros_like(dot)
    active_u_sq = torch.zeros_like(dot)
    active_count = 0
    total_count = 0
    clip_plus_num = 0.0
    clip_minus_num = 0.0
    legal = True

    for name, direction in directions.items():
        state = states[name]
        u = direction.float()
        plus, plus_stats = smoke.quantize_with_state(master[name].float().add(u, alpha=h), state, True)
        minus, minus_stats = smoke.quantize_with_state(master[name].float().add(u, alpha=-h), state, True)
        delta = plus.float() - minus.float()
        intended = u.mul(2.0 * h)
        active = delta != 0
        active_count += int(active.sum().detach().cpu())
        total_count += int(delta.numel())
        dot += (delta.double() * intended.double()).sum()
        delta_sq += (delta.double() * delta.double()).sum()
        intended_sq += (intended.double() * intended.double()).sum()
        u_sq += (u.double() * u.double()).sum()
        active_u_sq += (u.double()[active] * u.double()[active]).sum()
        clip_plus_num += float(plus_stats["clip_frac"]) * delta.numel()
        clip_minus_num += float(minus_stats["clip_frac"]) * delta.numel()
        legal = legal and plus_stats["code_min"] >= -state.qmax and plus_stats["code_max"] <= state.qmax
        legal = legal and minus_stats["code_min"] >= -state.qmax and minus_stats["code_max"] <= state.qmax

    eps = 1e-12
    delta_norm = delta_sq.sqrt()
    intended_norm = intended_sq.sqrt()
    u_norm = u_sq.sqrt()
    active_u_norm = active_u_sq.sqrt()
    active_frac = active_count / max(total_count, 1)
    alignment = dot / (delta_norm * intended_norm + eps)
    return {
        "delta_q_norm": float(delta_norm.detach().cpu()),
        "delta_q_sq": float(delta_sq.detach().cpu()),
        "ideal_displacement_norm": float(intended_norm.detach().cpu()),
        "u_norm": float(u_norm.detach().cpu()),
        "u_sq": float(u_sq.detach().cpu()),
        "active_u_norm": float(active_u_norm.detach().cpu()),
        "active_set_norm_ratio": float((active_u_norm / (u_norm + eps)).detach().cpu()),
        "norm_ratio": float((delta_norm / (intended_norm + eps)).detach().cpu()),
        "alignment": float(alignment.detach().cpu()),
        "active_frac": active_frac,
        "zero_effective_displacement_frac": 1.0 - active_frac,
        "saturation_frac_w_plus": clip_plus_num / max(total_count, 1),
        "saturation_frac_w_minus": clip_minus_num / max(total_count, 1),
        "codes_legal": bool(legal),
    }


def compute_richardson(
    model,
    params: Dict[str, torch.nn.Parameter],
    master: Dict[str, torch.Tensor],
    directions: Dict[str, torch.Tensor],
    states: Dict[str, smoke.RTNClipState],
    batch: Dict[str, torch.Tensor],
    h: float,
    d_h: float,
) -> Tuple[Optional[float], Optional[float]]:
    half_h = 0.5 * h
    copy_master_to_model_qspace(params, master, directions, half_h, +1.0, states)
    loss_plus, _ = smoke.forward_loss_and_logits(model, batch)
    copy_master_to_model_qspace(params, master, directions, half_h, -1.0, states)
    loss_minus, _ = smoke.forward_loss_and_logits(model, batch)
    smoke.restore_master(params, master)
    lp = float(loss_plus.detach().cpu())
    lm = float(loss_minus.detach().cpu())
    d_half = (lp - lm) / (2.0 * half_h)
    if not math.isfinite(d_h) or not math.isfinite(d_half):
        return d_half, None
    relerr = abs(d_h - d_half) / (abs(d_half) + 1e-12)
    return d_half, relerr


def update_norm_for_variant(
    master: Dict[str, torch.Tensor],
    directions: Dict[str, torch.Tensor],
    states: Dict[str, smoke.RTNClipState],
    h: float,
    variant: str,
    lr: float,
    delta_l: float,
    d_h: float,
    stats: Dict[str, object],
) -> float:
    if variant == "standard":
        coeff = -lr * d_h
        sq = torch.zeros((), device=next(iter(master.values())).device, dtype=torch.float64)
        for direction in directions.values():
            update = direction.float().mul(coeff)
            sq += (update.double() * update.double()).sum()
        return float(sq.sqrt().detach().cpu())

    eps = 1e-12
    delta_norm = float(stats["delta_q_norm"])
    u_norm = float(stats["u_norm"])
    active_u_norm = float(stats["active_u_norm"])
    delta_q_sq = float(stats["delta_q_sq"])
    if variant == "effdir_global":
        coeff = -lr * d_h * (u_norm / (delta_norm + eps))
    elif variant == "effdir_active":
        coeff = -lr * d_h * (active_u_norm / (delta_norm + eps))
    elif variant == "effdir_secant":
        coeff = -lr * delta_l * (float(stats["u_sq"]) / (delta_q_sq + eps))
    else:
        raise ValueError(f"unknown variant {variant}")

    sq = torch.zeros((), device=next(iter(master.values())).device, dtype=torch.float64)
    for name, direction in directions.items():
        state = states[name]
        plus = smoke.quantize_with_state(master[name].float().add(direction.float(), alpha=h), state)
        minus = smoke.quantize_with_state(master[name].float().add(direction.float(), alpha=-h), state)
        delta = plus.float() - minus.float()
        update = delta.mul(coeff)
        sq += (update.double() * update.double()).sum()
    return float(sq.sqrt().detach().cpu())


def apply_variant_update(
    master: Dict[str, torch.Tensor],
    directions: Dict[str, torch.Tensor],
    states: Dict[str, smoke.RTNClipState],
    h: float,
    variant: str,
    lr: float,
    delta_l: float,
    d_h: float,
    stats: Dict[str, object],
    update_norm_clip: float,
) -> Dict[str, object]:
    standard_norm = max(update_norm_for_variant(master, directions, states, h, "standard", lr, delta_l, d_h, stats), 0.0)
    update_norm = update_norm_for_variant(master, directions, states, h, variant, lr, delta_l, d_h, stats)
    clip_factor = 1.0
    if variant == "effdir_secant" and update_norm_clip > 0.0 and update_norm > update_norm_clip:
        clip_factor = float(update_norm_clip / (update_norm + 1e-12))
    applied_norm = update_norm * clip_factor

    if variant == "standard":
        coeff = -lr * d_h * clip_factor
        with torch.no_grad():
            for name, direction in directions.items():
                update = direction.float().mul(coeff)
                master[name].copy_(master[name].float().add(update).to(dtype=master[name].dtype))
    else:
        eps = 1e-12
        delta_norm = float(stats["delta_q_norm"])
        active_u_norm = float(stats["active_u_norm"])
        u_norm = float(stats["u_norm"])
        delta_q_sq = float(stats["delta_q_sq"])
        if variant == "effdir_global":
            coeff = -lr * d_h * (u_norm / (delta_norm + eps)) * clip_factor
        elif variant == "effdir_active":
            coeff = -lr * d_h * (active_u_norm / (delta_norm + eps)) * clip_factor
        elif variant == "effdir_secant":
            coeff = -lr * delta_l * (float(stats["u_sq"]) / (delta_q_sq + eps)) * clip_factor
        else:
            raise ValueError(f"unknown variant {variant}")
        with torch.no_grad():
            for name, direction in directions.items():
                state = states[name]
                plus = smoke.quantize_with_state(master[name].float().add(direction.float(), alpha=h), state)
                minus = smoke.quantize_with_state(master[name].float().add(direction.float(), alpha=-h), state)
                delta = plus.float() - minus.float()
                update = delta.mul(coeff)
                master[name].copy_(master[name].float().add(update).to(dtype=master[name].dtype))

    return {
        "standard_update_norm_ref": standard_norm,
        "update_norm": applied_norm,
        "update_norm_unclipped": update_norm,
        "update_norm_ratio_vs_standard": applied_norm / (standard_norm + 1e-12),
        "clipped": bool(clip_factor < 1.0),
        "clip_factor": clip_factor,
        "clipping_fraction": 1.0 - clip_factor,
    }


def make_run_config(args, run_dir: Path, variant: str, h_label: str, h: float, steps: int) -> Dict[str, object]:
    return {
        "run_name": f"int4_effdir_{variant}_h{h_label}_step{steps}",
        "phase": "phase_d_int4_effective_displacement_update",
        "model": "roberta-large",
        "dataset": "SST-5",
        "dataset_mode": "full",
        "seed": 16,
        "data_seed": 16,
        "batch_size": 64,
        "shuffle": True,
        "DATALOADER_SHUFFLE": os.environ.get("DATALOADER_SHUFFLE", ""),
        "sampler_required": "RandomSampler",
        "direction": "dense_gaussian_quantized_linear_weight_space",
        "update_scope": "quantized_linear_weights_only",
        "variant": variant,
        "h": h,
        "h_label": h_label,
        "steps": steps,
        "lr": float(args.lr),
        "update_norm_clip": float(args.update_norm_clip),
        "precision": "INT4 fake quantized forward",
        "quantizer_backend": "G128_groupwise_RTNClip_fake_quant",
        "quantizer": "group_rtn_clip",
        "bitwidth": 4,
        "group_size": 128,
        "scale_refresh_k": 1,
        "pair_shared_grid": True,
        "fresh_round_codes": True,
        "grid_source": "unperturbed_fp16_master_weight",
        "scale_source": "unperturbed_master_w_t",
        "update_backend": "fp16_master",
        "master_dtype": "fp16",
        "direct_int_update": False,
        "residual_grid": False,
        "GPTQ": False,
        "run_dir": str(run_dir),
    }


def train_one(
    args,
    model,
    train_loader,
    dev_loader,
    train_sampler,
    params: Dict[str, torch.nn.Parameter],
    q_names: List[str],
    initial_master_cpu: Dict[str, torch.Tensor],
    variant: str,
    h_label: str,
    h: float,
    steps: int,
    output_root: Path,
) -> Dict[str, object]:
    run_name = f"int4_effdir_{variant}_h{h_label}_step{steps}"
    run_dir = output_root / "runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    for stale in ("metrics.csv", "diagnostics.jsonl", "eval_metrics.jsonl", "quantizer_diagnostics.jsonl", "train.log"):
        path = run_dir / stale
        if path.exists():
            path.unlink()

    config = make_run_config(args, run_dir, variant, h_label, h, steps)
    config["run_name"] = run_name
    config["sampler_name"] = type(train_sampler).__name__
    config["quantized_modules"] = q_names
    config["env"] = smoke.collect_env(REPO_ROOT)
    write_json(run_dir / "run_config.json", config)
    write_json(run_dir / "run_manifest_row.json", config)

    if os.environ.get("DATALOADER_SHUFFLE") != "True" or type(train_sampler).__name__ != "RandomSampler":
        summary = {**config, "status": "failed", "steps_completed": 0, "error_message": "shuffle/sampler contract failed"}
        write_json(run_dir / "run_summary.json", summary)
        return summary

    device = next(model.parameters()).device
    master = {name: tensor.to(device=device, dtype=torch.float16).clone() for name, tensor in initial_master_cpu.items()}
    smoke.restore_master(params, master)
    batch_iter = smoke.cycle(train_loader)
    best = {"best_eval_acc": None, "best_eval_step": None, "best_eval_loss": None, "best_eval_loss_step": None}
    numel_by_name = {name: params[name].numel() for name in q_names}
    finite_count = 0
    skip_count = 0
    clipped_count = 0
    last_train_loss = None
    last_diag: Dict[str, object] = {}
    last_update: Dict[str, object] = {}
    last_eval_acc = None
    last_eval_loss = None
    last_eval_step = None
    status = "complete"
    error_message = ""
    total_start = time.time()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    metrics_columns = [
        "step",
        "loss_plus",
        "loss_minus",
        "delta_l",
        "d_h",
        "d_h_finite",
        "train_loss",
        "delta_q_norm",
        "ideal_displacement_norm",
        "norm_ratio",
        "alignment",
        "active_frac",
        "active_set_norm_ratio",
        "skip_update",
        "update_norm",
        "standard_update_norm_ref",
        "update_norm_ratio_vs_standard",
        "clipped",
        "clip_factor",
        "clipping_fraction",
        "richardson_d_half",
        "richardson_relerr",
        "eval_loss",
        "eval_acc",
        "seconds",
        "nan_flag",
    ]
    with (run_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=metrics_columns)
        writer.writeheader()
        for step_idx in range(steps):
            step_start = time.time()
            states, refresh_stats = smoke.refresh_quantizer_states(master, q_names, 4, 128)
            quant = smoke.aggregate_quantizer_stats(refresh_stats, numel_by_name)
            if step_idx % int(args.quant_log_every) == 0:
                append_jsonl(run_dir / "quantizer_diagnostics.jsonl", {"step": step_idx, "record_type": "refresh_summary", **quant})

            directions = sample_quantized_directions(master, q_names, 16, h, step_idx, variant)
            batch = smoke.move_batch(next(batch_iter), device)
            copy_master_to_model_qspace(params, master, directions, h, +1.0, states)
            loss_plus, _ = smoke.forward_loss_and_logits(model, batch)
            copy_master_to_model_qspace(params, master, directions, h, -1.0, states)
            loss_minus, _ = smoke.forward_loss_and_logits(model, batch)
            smoke.restore_master(params, master)

            loss_plus_f = float(loss_plus.detach().cpu())
            loss_minus_f = float(loss_minus.detach().cpu())
            delta_l = loss_plus_f - loss_minus_f
            d_h = delta_l / (2.0 * h)
            finite = math.isfinite(loss_plus_f) and math.isfinite(loss_minus_f) and math.isfinite(d_h)
            diag = compute_effective_stats(master, directions, states, h)
            richardson_d_half = None
            richardson_relerr = None
            if step_idx == 0 or (step_idx + 1) % int(args.richardson_every) == 0:
                richardson_d_half, richardson_relerr = compute_richardson(model, params, master, directions, states, batch, h, d_h)

            skip_update = bool(diag["delta_q_norm"] <= 0.0 or diag["active_frac"] < 1e-3)
            update = {
                "standard_update_norm_ref": None,
                "update_norm": 0.0,
                "update_norm_unclipped": 0.0,
                "update_norm_ratio_vs_standard": None,
                "clipped": False,
                "clip_factor": 1.0,
                "clipping_fraction": 0.0,
            }
            if finite and not skip_update:
                finite_count += 1
                update = apply_variant_update(
                    master,
                    directions,
                    states,
                    h,
                    variant,
                    float(args.lr),
                    delta_l,
                    d_h,
                    diag,
                    float(args.update_norm_clip),
                )
                smoke.restore_master(params, master)
                clipped_count += int(bool(update["clipped"]))
            else:
                skip_count += 1

            completed_step = step_idx + 1
            eval_loss = None
            eval_acc = None
            if completed_step % int(args.eval_every) == 0 or completed_step == steps:
                eval_loss, eval_acc = evaluate_quantized(model, params, master, q_names, 4, 128, dev_loader, device, int(args.eval_batches))
                append_jsonl(run_dir / "eval_metrics.jsonl", {"step": completed_step, "eval_loss": eval_loss, "eval_acc": eval_acc})
                last_eval_loss = eval_loss
                last_eval_acc = eval_acc
                last_eval_step = completed_step
                if eval_acc is not None and (best["best_eval_acc"] is None or eval_acc > best["best_eval_acc"]):
                    best["best_eval_acc"] = eval_acc
                    best["best_eval_step"] = completed_step
                if eval_loss is not None and (best["best_eval_loss"] is None or eval_loss < best["best_eval_loss"]):
                    best["best_eval_loss"] = eval_loss
                    best["best_eval_loss_step"] = completed_step

            last_train_loss = (loss_plus_f + loss_minus_f) / 2.0
            nan_flag = (not finite) or (not bool(diag["codes_legal"])) or (not math.isfinite(float(update["update_norm"])))
            row = {
                "step": completed_step,
                "loss_plus": loss_plus_f,
                "loss_minus": loss_minus_f,
                "delta_l": delta_l,
                "d_h": d_h,
                "d_h_finite": finite,
                "train_loss": last_train_loss,
                "delta_q_norm": diag["delta_q_norm"],
                "ideal_displacement_norm": diag["ideal_displacement_norm"],
                "norm_ratio": diag["norm_ratio"],
                "alignment": diag["alignment"],
                "active_frac": diag["active_frac"],
                "active_set_norm_ratio": diag["active_set_norm_ratio"],
                "skip_update": skip_update,
                "update_norm": update["update_norm"],
                "standard_update_norm_ref": update["standard_update_norm_ref"],
                "update_norm_ratio_vs_standard": update["update_norm_ratio_vs_standard"],
                "clipped": update["clipped"],
                "clip_factor": update["clip_factor"],
                "clipping_fraction": update["clipping_fraction"],
                "richardson_d_half": richardson_d_half,
                "richardson_relerr": richardson_relerr,
                "eval_loss": eval_loss,
                "eval_acc": eval_acc,
                "seconds": time.time() - step_start,
                "nan_flag": nan_flag,
            }
            writer.writerow(row)
            f.flush()
            append_jsonl(run_dir / "diagnostics.jsonl", {"step": completed_step, **diag, **update, "skip_update": skip_update, "richardson_relerr": richardson_relerr})
            last_diag = diag
            last_update = update

            if nan_flag:
                status = "failed"
                error_message = f"non-finite/check failure at step {completed_step}"
                log_to(run_dir, error_message)
                break
            if completed_step % int(args.log_every) == 0 or completed_step == 1:
                log_to(
                    run_dir,
                    f"{run_name} step={completed_step}/{steps} loss={last_train_loss:.6g} "
                    f"eval_acc={eval_acc} active={diag['active_frac']:.4g} align={diag['alignment']:.4g} "
                    f"norm_ratio={diag['norm_ratio']:.4g} update_norm={update['update_norm']:.4g} skip={skip_update}",
                )

    steps_completed = 0
    with (run_dir / "metrics.csv").open(newline="", encoding="utf-8") as mf:
        rows = list(csv.DictReader(mf))
        if rows:
            steps_completed = int(float(rows[-1]["step"]))
    peak_mem = float(torch.cuda.max_memory_allocated() / 1024 / 1024) if torch.cuda.is_available() else 0.0
    total_runtime = time.time() - total_start
    summary = {
        **config,
        "status": status,
        "error_message": error_message,
        "steps_completed": steps_completed,
        "best_eval_acc": best["best_eval_acc"],
        "best_eval_step": best["best_eval_step"],
        "last_eval_acc": last_eval_acc,
        "last_eval_step": last_eval_step,
        "best_eval_loss": best["best_eval_loss"],
        "best_eval_loss_step": best["best_eval_loss_step"],
        "last_eval_loss": last_eval_loss,
        "last_eval_loss_step": last_eval_step,
        "final_train_loss": last_train_loss,
        "d_h_finite_rate": finite_count / max(steps_completed, 1),
        "skip_update_frac": skip_count / max(steps_completed, 1),
        "delta_q_norm_last": last_diag.get("delta_q_norm"),
        "ideal_displacement_norm_last": last_diag.get("ideal_displacement_norm"),
        "norm_ratio_last": last_diag.get("norm_ratio"),
        "alignment_last": last_diag.get("alignment"),
        "active_frac_last": last_diag.get("active_frac"),
        "active_set_norm_ratio_last": last_diag.get("active_set_norm_ratio"),
        "update_norm_last": last_update.get("update_norm"),
        "standard_update_norm_ref_last": last_update.get("standard_update_norm_ref"),
        "update_norm_ratio_vs_standard_last": last_update.get("update_norm_ratio_vs_standard"),
        "clipped_frac": clipped_count / max(steps_completed, 1),
        "richardson_relerr_last": None if not rows else rows[-1].get("richardson_relerr"),
        "seconds_per_step": total_runtime / max(steps_completed, 1),
        "peak_gpu_mem": peak_mem,
        "run_dir": str(run_dir),
        "notes": "Dense directions and updates are restricted to INT4-quantized Linear.weight tensors.",
    }
    write_json(run_dir / "run_summary.json", summary)
    log_to(run_dir, f"finished {run_name} status={status} steps={steps_completed}")
    return summary


def summarize(output_root: Path) -> List[Dict[str, object]]:
    rows = []
    for path in sorted((output_root / "runs").glob("*/run_summary.json")):
        rows.append(json.loads(path.read_text(encoding="utf-8")))
    if not rows:
        return []
    write_csv(output_root / "summary_phaseD.csv", rows, SUMMARY_COLUMNS)

    def fnum(value, digits=4):
        if value in (None, ""):
            return "n/a"
        try:
            return f"{float(value):.{digits}g}"
        except Exception:
            return str(value)

    best = max((r for r in rows if r.get("best_eval_acc") is not None), key=lambda r: float(r["best_eval_acc"]), default=None)
    by_variant: Dict[str, List[Dict[str, object]]] = {}
    for row in rows:
        by_variant.setdefault(str(row.get("variant")), []).append(row)
    lines = [
        "# Phase D INT4 Effective-Displacement Update Summary",
        "",
        f"Output root: `{output_root}`",
        "Scope: dense directions and updates over INT4-quantized `Linear.weight` tensors only.",
        f"Runs completed: {sum(1 for r in rows if r.get('status') == 'complete')} / {len(rows)}",
        f"Best run: `{best['run_name']}` best_eval_acc={fnum(best.get('best_eval_acc'))} last_eval_acc={fnum(best.get('last_eval_acc'))}" if best else "Best run: n/a",
        "",
        "| variant | h | status | steps | best_acc | last_acc | last_loss | active | align | norm_ratio | active_u/u | upd/std | skip | clip |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row.get('variant')} | {row.get('h_label')} | {row.get('status')} | {row.get('steps_completed')} | "
            f"{fnum(row.get('best_eval_acc'))} | {fnum(row.get('last_eval_acc'))} | {fnum(row.get('last_eval_loss'))} | "
            f"{fnum(row.get('active_frac_last'))} | {fnum(row.get('alignment_last'))} | {fnum(row.get('norm_ratio_last'))} | "
            f"{fnum(row.get('active_set_norm_ratio_last'))} | {fnum(row.get('update_norm_ratio_vs_standard_last'))} | "
            f"{fnum(row.get('skip_update_frac'))} | {fnum(row.get('clipped_frac'))} |"
        )
    lines.extend(["", "## Questions", ""])
    std_rows = by_variant.get("standard", [])
    best_std = max((r for r in std_rows if r.get("best_eval_acc") is not None), key=lambda r: float(r["best_eval_acc"]), default=None)
    lines.append(f"- Best standard dense INT4 h in this Phase D scope: `{best_std.get('h_label')}` with best_eval_acc={fnum(best_std.get('best_eval_acc'))}." if best_std else "- Best standard dense INT4 h: unavailable.")
    small_h = {"5e-4", "1e-3"}
    small_eff = [r for r in rows if r.get("variant") != "standard" and r.get("h_label") in small_h and r.get("best_eval_acc") is not None]
    small_std = [r for r in std_rows if r.get("h_label") in small_h and r.get("best_eval_acc") is not None]
    if small_eff and small_std:
        best_eff = max(small_eff, key=lambda r: float(r["best_eval_acc"]))
        best_small_std = max(small_std, key=lambda r: float(r["best_eval_acc"]))
        improved = float(best_eff["best_eval_acc"]) > float(best_small_std["best_eval_acc"])
        lines.append(f"- Effdir improves small-h INT4 in this short run: {'yes' if improved else 'no'}; best effdir `{best_eff['run_name']}` vs best small standard `{best_small_std['run_name']}`.")
    else:
        lines.append("- Effdir small-h comparison: unavailable.")
    variant_best = []
    for variant, group in by_variant.items():
        cand = max((r for r in group if r.get("best_eval_acc") is not None), key=lambda r: float(r["best_eval_acc"]), default=None)
        if cand:
            variant_best.append((variant, cand))
    if variant_best:
        stable = sorted(variant_best, key=lambda item: (-float(item[1]["best_eval_acc"]), float(item[1].get("skip_update_frac") or 0.0)))[0]
        lines.append(f"- Most stable/best scaling by short-run accuracy: `{stable[0]}` at h `{stable[1].get('h_label')}`.")
    extend = []
    if best and best.get("status") == "complete" and best.get("best_eval_acc") is not None and float(best["best_eval_acc"]) >= 0.35:
        extend.append(best)
    if extend:
        lines.append(f"- Recommended extension candidate: `{extend[0]['run_name']}` to 1k/2k before any 5k run.")
    else:
        lines.append("- Recommended extension candidate: none from this short run.")
    (output_root / "summary_phaseD.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return rows


def run_matrix(args, steps: int, suffix: str = "") -> List[Dict[str, object]]:
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    write_json(output_root / "env.json", smoke.collect_env(REPO_ROOT))
    write_json(
        output_root / "config_manifest.json",
        {
            "phase": "D",
            "description": "INT4 effective-displacement direction update on small h",
            "h_grid": [{"label": label, "h": h} for label, h in H_GRID],
            "variants": list(VARIANTS),
            "steps": steps,
            "lr": args.lr,
            "update_norm_clip": args.update_norm_clip,
            "seed": 16,
            "data_seed": 16,
            "batch_size": 64,
            "shuffle": True,
            "scope": "quantized_linear_weights_only",
        },
    )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for RoBERTa-large Phase D.")
    if os.environ.get("DATALOADER_SHUFFLE") != "True":
        raise RuntimeError("DATALOADER_SHUFFLE=True must be exported.")

    device = torch.device("cuda")
    model, train_loader, dev_loader, _, train_sampler = smoke.load_prompt_model_and_data(
        argparse.Namespace(repo_root=REPO_ROOT, model_id="roberta-large", seed=16, data_seed=16, batch_size=64, eval_batch_size=args.eval_batch_size),
        device,
    )
    params = smoke.named_parameter_map(model)
    q_names = smoke.linear_weight_names(model)
    initial_master_cpu = {
        name: p.detach().cpu().to(dtype=torch.float16).clone()
        for name, p in params.items()
        if p.detach().is_floating_point()
    }
    requested_variants = set(args.variants.split(",")) if args.variants else set(VARIANTS)
    requested_h_labels = set(args.h_labels.split(",")) if args.h_labels else {label for label, _ in H_GRID}
    unknown_variants = requested_variants.difference(VARIANTS)
    if unknown_variants:
        raise ValueError(f"unknown variants: {sorted(unknown_variants)}")
    unknown_h = requested_h_labels.difference({label for label, _ in H_GRID})
    if unknown_h:
        raise ValueError(f"unknown h labels: {sorted(unknown_h)}")

    rows = []
    for variant in VARIANTS:
        if variant not in requested_variants:
            continue
        for h_label, h in H_GRID:
            if h_label not in requested_h_labels:
                continue
            label = f"{h_label}{suffix}"
            rows.append(train_one(args, model, train_loader, dev_loader, train_sampler, params, q_names, initial_master_cpu, variant, label, h, steps, output_root))
            summarize(output_root)
    return rows


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_root", default=str(REPO_ROOT / "outputs" / "rtnclip_int4_effdir_phaseD_seed16"))
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--eval_every", type=int, default=100)
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--eval_batches", type=int, default=-1)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--quant_log_every", type=int, default=100)
    parser.add_argument("--richardson_every", type=int, default=100)
    parser.add_argument("--update_norm_clip", type=float, default=1.0)
    parser.add_argument("--variants", default="", help="Comma-separated subset of variants to run.")
    parser.add_argument("--h_labels", default="", help="Comma-separated subset of h labels to run, e.g. 5e-4,1e-3.")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("run")
    sub.add_parser("summarize")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.cmd == "run":
        run_matrix(args, int(args.steps))
        summarize(Path(args.output_root))
    elif args.cmd == "summarize":
        summarize(Path(args.output_root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
