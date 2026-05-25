#!/usr/bin/env python
"""Local RoBERTa-large/SST-5 INT4 RTNClip residual-grid update runner.

This is intentionally isolated from the main RTNClip FP16-master runner.  It
keeps a fixed G128 RTNClip grid computed once from the initial unperturbed
weights, uses that grid for the two-point INT4 forward oracle, and commits the
ZO update to INT4 code steps with error feedback:

    acc_t = residual_t - lr * d_h * u_t
    k_t   = round(acc_t / scale)
    q     = clamp(q + k_t)
    residual_{t+1} = acc_t - (q_t - q_{t-1}) * scale

The implementation follows the standard error-feedback pattern used for biased
compression: accumulate the compression error and add it to the next update.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import shutil
import socket
import subprocess
import sys
import time
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
import rtnclip_roberta_sst5_batch as rtn_batch  # noqa: E402


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


def append_jsonl(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, sort_keys=True, default=str) + "\n")


def append_log(path: Path, message: str) -> None:
    line = f"[{datetime.now().isoformat(timespec='seconds')}] {message}"
    print(line, flush=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def cycle(loader):
    while True:
        for batch in loader:
            yield batch


def expand_scales(state: smoke.RTNClipState, weight: torch.Tensor) -> torch.Tensor:
    if weight.ndim != 2:
        raise ValueError(f"RTNClip residual grid expects 2D Linear.weight, got {tuple(weight.shape)}")
    out_features, in_features = weight.shape
    group_size = int(state.group_size)
    scales = state.scales.to(device=weight.device, dtype=torch.float32).squeeze(-1)
    expanded = scales.repeat_interleave(group_size, dim=1)[:, :in_features]
    return expanded.reshape(out_features, in_features)


def quantize_codes(weight: torch.Tensor, state: smoke.RTNClipState) -> torch.Tensor:
    scale = expand_scales(state, weight)
    q = torch.round(weight.detach().float() / scale).clamp(-state.qmax, state.qmax)
    return q.to(dtype=torch.int8)


def dequantize_codes(codes: torch.Tensor, state: smoke.RTNClipState, dtype: torch.dtype) -> torch.Tensor:
    dummy = torch.empty(tuple(state.shape), device=codes.device, dtype=torch.float32)
    scale = expand_scales(state, dummy)
    return (codes.float() * scale).to(dtype=dtype)


def sample_linear_only_directions(
    master: Dict[str, torch.Tensor],
    q_names: Iterable[str],
    *,
    seed: int,
    h: float,
    step: int,
) -> Dict[str, torch.Tensor]:
    q_set = set(q_names)
    first = next(iter(master.values()))
    gen = torch.Generator(device=first.device)
    gen.manual_seed(int(seed) + 4 * 1009 + 1 * 9176 + rtn_batch.stable_h_key(h) + int(step) * 1_000_003)
    directions: Dict[str, torch.Tensor] = {}
    for name, tensor in master.items():
        if name in q_set:
            directions[name] = torch.randn(tensor.shape, device=tensor.device, generator=gen, dtype=tensor.dtype)
        elif tensor.is_floating_point():
            directions[name] = torch.zeros_like(tensor)
    return directions


def init_fixed_int4_grid(
    master: Dict[str, torch.Tensor],
    q_names: List[str],
    *,
    bitwidth: int,
    group_size: int,
) -> Tuple[Dict[str, smoke.RTNClipState], List[Dict[str, object]], Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    states, rows = smoke.refresh_quantizer_states(master, q_names, bitwidth, group_size)
    codes: Dict[str, torch.Tensor] = {}
    residuals: Dict[str, torch.Tensor] = {}
    with torch.no_grad():
        for name in q_names:
            code = quantize_codes(master[name], states[name])
            codes[name] = code
            master[name].copy_(dequantize_codes(code, states[name], master[name].dtype))
            residuals[name] = torch.zeros_like(master[name], dtype=torch.float32)
    return states, rows, codes, residuals


def residual_commit_update(
    master: Dict[str, torch.Tensor],
    states: Dict[str, smoke.RTNClipState],
    codes: Dict[str, torch.Tensor],
    residuals: Dict[str, torch.Tensor],
    directions: Dict[str, torch.Tensor],
    *,
    lr: float,
    d_h: float,
    max_code_step: int,
) -> Dict[str, float]:
    device = next(iter(master.values())).device
    totals = {
        "numel": 0.0,
        "active_count": 0.0,
        "candidate_active_count": 0.0,
        "saturation_count": 0.0,
        "intended_sq": 0.0,
        "acc_sq": 0.0,
        "actual_sq": 0.0,
        "residual_sq": 0.0,
        "dot_intended_actual": 0.0,
        "dot_acc_actual": 0.0,
        "ef_error_sq": 0.0,
        "ef_error_max": 0.0,
        "residual_over_scale_max": 0.0,
        "residual_bound_violation_count": 0.0,
        "grid_error_sq": 0.0,
        "grid_error_max": 0.0,
    }
    with torch.no_grad():
        for name, state in states.items():
            direction = directions[name].float()
            scale = expand_scales(state, master[name])
            desired = torch.nan_to_num(-float(lr) * float(d_h) * direction, nan=0.0, posinf=0.0, neginf=0.0)
            residual_before = residuals[name]
            acc = torch.nan_to_num(residual_before + desired, nan=0.0, posinf=0.0, neginf=0.0)
            k = torch.round(acc / scale)
            if max_code_step > 0:
                k = torch.clamp(k, -float(max_code_step), float(max_code_step))
            k = torch.nan_to_num(k, nan=0.0, posinf=float(max_code_step or state.qmax), neginf=-float(max_code_step or state.qmax))
            candidate_active = int(torch.count_nonzero(k != 0).item())

            q_old = codes[name].float()
            q_new = torch.clamp(q_old + k, -float(state.qmax), float(state.qmax))
            q_delta = q_new - q_old
            actual = q_delta * scale
            new_residual = torch.nan_to_num(acc - actual, nan=0.0, posinf=0.0, neginf=0.0)

            codes[name] = q_new.to(dtype=torch.int8)
            residuals[name].copy_(new_residual)
            master[name].copy_(dequantize_codes(codes[name], state, master[name].dtype))

            ef_error = new_residual - (acc - actual)
            grid_q = quantize_codes(master[name], state).float()
            grid_err = master[name].float() - grid_q * scale
            residual_over_scale = torch.abs(new_residual / scale)

            totals["numel"] += float(actual.numel())
            totals["active_count"] += float(torch.count_nonzero(q_delta != 0).item())
            totals["candidate_active_count"] += float(candidate_active)
            totals["saturation_count"] += float(torch.count_nonzero((q_new == -state.qmax) | (q_new == state.qmax)).item())
            totals["intended_sq"] += float(torch.sum(desired.double() * desired.double()).item())
            totals["acc_sq"] += float(torch.sum(acc.double() * acc.double()).item())
            totals["actual_sq"] += float(torch.sum(actual.double() * actual.double()).item())
            totals["residual_sq"] += float(torch.sum(new_residual.double() * new_residual.double()).item())
            totals["dot_intended_actual"] += float(torch.sum(desired.double() * actual.double()).item())
            totals["dot_acc_actual"] += float(torch.sum(acc.double() * actual.double()).item())
            totals["ef_error_sq"] += float(torch.sum(ef_error.double() * ef_error.double()).item())
            totals["ef_error_max"] = max(totals["ef_error_max"], float(torch.max(torch.abs(ef_error)).item()) if ef_error.numel() else 0.0)
            totals["residual_over_scale_max"] = max(totals["residual_over_scale_max"], float(torch.max(residual_over_scale).item()) if residual_over_scale.numel() else 0.0)
            totals["residual_bound_violation_count"] += float(torch.count_nonzero(residual_over_scale > 0.5001).item())
            totals["grid_error_sq"] += float(torch.sum(grid_err.double() * grid_err.double()).item())
            totals["grid_error_max"] = max(totals["grid_error_max"], float(torch.max(torch.abs(grid_err)).item()) if grid_err.numel() else 0.0)

    eps = 1e-12
    intended_norm = math.sqrt(max(totals["intended_sq"], 0.0))
    acc_norm = math.sqrt(max(totals["acc_sq"], 0.0))
    actual_norm = math.sqrt(max(totals["actual_sq"], 0.0))
    residual_norm = math.sqrt(max(totals["residual_sq"], 0.0))
    ef_error_norm = math.sqrt(max(totals["ef_error_sq"], 0.0))
    grid_error_norm = math.sqrt(max(totals["grid_error_sq"], 0.0))
    numel = max(totals["numel"], 1.0)
    return {
        "numel": totals["numel"],
        "candidate_active_frac": totals["candidate_active_count"] / numel,
        "active_frac": totals["active_count"] / numel,
        "saturation_frac": totals["saturation_count"] / numel,
        "intended_update_norm": intended_norm,
        "acc_update_norm": acc_norm,
        "actual_update_norm": actual_norm,
        "residual_norm": residual_norm,
        "cos_intended_actual": totals["dot_intended_actual"] / (intended_norm * actual_norm + eps),
        "acc_actual_cos": totals["dot_acc_actual"] / (acc_norm * actual_norm + eps),
        "actual_over_intended_norm_ratio": actual_norm / (intended_norm + eps),
        "actual_over_acc_norm_ratio": actual_norm / (acc_norm + eps),
        "ef_error_norm": ef_error_norm,
        "ef_error_max": totals["ef_error_max"],
        "residual_over_scale_max": totals["residual_over_scale_max"],
        "residual_bound_violation_frac": totals["residual_bound_violation_count"] / numel,
        "grid_error_norm": grid_error_norm,
        "grid_error_max": totals["grid_error_max"],
        "scale_drift_max": 0.0,
        "update_norm": actual_norm,
        "device_index": torch.cuda.current_device() if device.type == "cuda" else None,
    }


def state_to_cpu_dict(state: smoke.RTNClipState) -> Dict[str, object]:
    return {
        "name": state.name,
        "shape": tuple(state.shape),
        "group_size": int(state.group_size),
        "bitwidth": int(state.bitwidth),
        "qmax": int(state.qmax),
        "scales": state.scales.detach().cpu(),
        "alpha_idx": state.alpha_idx.detach().cpu(),
        "alpha_values": state.alpha_values.detach().cpu(),
        "lengths": state.lengths.detach().cpu(),
    }


def save_checkpoint(
    path: Path,
    *,
    step: int,
    master: Dict[str, torch.Tensor],
    codes: Dict[str, torch.Tensor],
    residuals: Dict[str, torch.Tensor],
    states: Dict[str, smoke.RTNClipState],
    best: Dict[str, object],
    config: Dict[str, object],
) -> None:
    path.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": int(step),
            "master": {k: v.detach().cpu() for k, v in master.items()},
            "int4_codes": {k: v.detach().cpu() for k, v in codes.items()},
            "residuals": {k: v.detach().cpu() for k, v in residuals.items()},
            "fixed_rtnclip_states": {k: state_to_cpu_dict(v) for k, v in states.items()},
            "best": best,
            "config": config,
        },
        path / "state.pt",
    )
    write_json(
        path / "checkpoint_manifest.json",
        {
            "step": int(step),
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "has_int4_codes": True,
            "has_residuals": True,
            "fixed_scale": True,
            "quantized_tensors": len(codes),
        },
    )


def copy_checkpoint(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def synthetic_residual_check(device: torch.device) -> Dict[str, object]:
    state = SimpleNamespace(qmax=7, group_size=4, bitwidth=4, shape=(2, 4), scales=torch.full((2, 1, 1), 0.1, device=device))
    master = {"w": torch.zeros((2, 4), device=device, dtype=torch.float16)}
    states = {"w": state}
    codes = {"w": torch.zeros((2, 4), device=device, dtype=torch.int8)}
    residuals = {"w": torch.zeros((2, 4), device=device, dtype=torch.float32)}
    directions = {"w": torch.ones((2, 4), device=device, dtype=torch.float16)}
    stats1 = residual_commit_update(master, states, codes, residuals, directions, lr=0.04, d_h=1.0, max_code_step=1)
    ok1 = stats1["active_frac"] == 0.0 and abs(float(residuals["w"].flatten()[0]) + 0.04) < 1e-6
    stats2 = residual_commit_update(master, states, codes, residuals, directions, lr=0.04, d_h=1.0, max_code_step=1)
    ok2 = stats2["active_frac"] == 1.0 and torch.all(codes["w"] == -1).item()
    return {
        "status": "pass" if ok1 and ok2 and stats2["ef_error_norm"] == 0.0 else "fail",
        "step1_active_frac": stats1["active_frac"],
        "step2_active_frac": stats2["active_frac"],
        "step2_ef_error_norm": stats2["ef_error_norm"],
        "step2_residual_over_scale_max": stats2["residual_over_scale_max"],
    }


def run(args: argparse.Namespace) -> Dict[str, object]:
    if os.environ.get("DATALOADER_SHUFFLE") != "True":
        raise RuntimeError("DATALOADER_SHUFFLE=True must be exported.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this local RoBERTa-large run.")

    run_dir = Path(args.output_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "train.log"
    device = torch.device("cuda")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    rtn_batch.reset_run_seed(args.seed)

    env = smoke.collect_env(REPO_ROOT)
    env.update({"hostname": socket.gethostname(), "git_commit": git_commit(), "command": " ".join(sys.argv)})
    write_json(run_dir / "env.json", env)
    synth = synthetic_residual_check(device)
    write_json(run_dir / "synthetic_residual_check.json", synth)
    if synth["status"] != "pass":
        raise RuntimeError(f"synthetic residual check failed: {synth}")

    load_args = SimpleNamespace(
        repo_root=REPO_ROOT,
        model_id="roberta-large",
        task_name="sst-5",
        dataset_mode=args.dataset_mode,
        data_seed=args.data_seed,
        num_k=args.num_k,
        data_dir=args.data_dir or None,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
    )
    model, train_loader, dev_loader, data_args, train_sampler = smoke.load_prompt_model_and_data(load_args, device)
    sampler_name = type(train_sampler).__name__
    params = smoke.named_parameter_map(model)
    q_names = [name for name in smoke.linear_weight_names(model) if name in params]
    numel_by_name = {name: int(params[name].numel()) for name in q_names}

    q_name_set = set(q_names)
    # Keep quantized Linear weights as fp32 dequantized code values so the
    # stored master is exactly on the fixed RTN grid. The model forward still
    # receives fp16 tensors because the RoBERTa module itself is fp16.
    master = {
        name: p.detach().clone().to(device=device, dtype=(torch.float32 if name in q_name_set else torch.float16))
        for name, p in params.items()
        if p.detach().is_floating_point()
    }
    states, refresh_rows, codes, residuals = init_fixed_int4_grid(master, q_names, bitwidth=4, group_size=args.group_size)
    smoke.restore_master(params, master)
    quant_agg = smoke.aggregate_quantizer_stats(refresh_rows, numel_by_name)
    append_jsonl(run_dir / "quantizer_diagnostics.jsonl", {"step": 0, "record_type": "fixed_initial_grid", "num_scale_refreshes": 1, **quant_agg})

    config = {
        "run_name": args.run_name,
        "model": "roberta-large",
        "dataset": "sst-5",
        "dataset_mode": args.dataset_mode,
        "seed": args.seed,
        "data_seed": args.data_seed,
        "batch_size": args.batch_size,
        "sampler_name": sampler_name,
        "h": args.h,
        "lr": args.lr,
        "steps": args.steps,
        "eval_every": args.eval_every,
        "checkpoint_steps": args.checkpoint_steps,
        "quantizer": "INT4_G128_RTNClip_fixed_grid",
        "quant_bits": 4,
        "group_size": args.group_size,
        "scale_freeze": True,
        "scale_refresh_k": 0,
        "grid_source": "initial_unperturbed_fp16_master_weight",
        "pair_shared_grid": True,
        "fresh_round_codes": True,
        "update_backend": "int4_residual_grid_error_feedback",
        "residual_dtype": "fp32",
        "commit_mode": "round",
        "max_code_step": args.max_code_step,
        "perturbed_parameter_scope": "Linear.weight_only",
        "quantized_forward_scope": "Linear.weight_only",
        "non_linear_parameters": "frozen",
        "quantization_seed": "not_used_deterministic_RTNClip",
        "direction_seed_formula": "seed + bitwidth*1009 + stable_h_key(h) + step*1000003",
        "error_feedback_reference": "Karimireddy et al. 2019 EF-SGD / Stich & Karimireddy 2020 EF framework",
        "command": " ".join(sys.argv),
    }
    write_json(run_dir / "run_config.json", config)
    write_json(run_dir / "run_manifest_row.json", config)
    (run_dir / "resume_command.txt").write_text(
        "CUDA_VISIBLE_DEVICES=0 DATALOADER_SHUFFLE=True "
        + " ".join(sys.argv)
        + "\n",
        encoding="utf-8",
    )

    best: Dict[str, object] = {
        "best_eval_acc": None,
        "best_eval_step": None,
        "best_eval_loss": None,
        "best_eval_loss_step": None,
    }
    last_eval_acc = None
    last_eval_loss = None
    last_eval_step = None
    last_update_stats: Dict[str, float] = {}
    last_pert: Dict[str, object] = {}
    train_loss = None
    finite_count = 0
    status = "running"
    error_message = ""
    batch_iter = cycle(train_loader)
    metrics_path = run_dir / "metrics.csv"
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    total_start = time.time()

    fields = [
        "step",
        "loss_plus",
        "loss_minus",
        "train_loss",
        "d_h",
        "d_h_finite",
        "update_norm",
        "active_frac",
        "candidate_active_frac",
        "cos_intended_actual",
        "acc_actual_cos",
        "actual_over_intended_norm_ratio",
        "actual_over_acc_norm_ratio",
        "residual_norm",
        "residual_over_scale_max",
        "residual_bound_violation_frac",
        "grid_error_norm",
        "ef_error_norm",
        "saturation_frac",
        "eval_loss",
        "eval_acc",
        "seconds",
        "nan_flag",
    ]
    with metrics_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for step_idx in range(args.steps):
            step_start = time.time()
            directions = sample_linear_only_directions(master, q_names, seed=args.seed, h=args.h, step=step_idx)
            batch = smoke.move_batch(next(batch_iter), device)

            smoke.copy_master_to_model(params, master, directions, args.h, +1.0, states)
            loss_plus, _ = smoke.forward_loss_and_logits(model, batch)
            smoke.copy_master_to_model(params, master, directions, args.h, -1.0, states)
            loss_minus, _ = smoke.forward_loss_and_logits(model, batch)
            restore_diff = smoke.restore_master(params, master)

            loss_plus_f = float(loss_plus.detach().cpu())
            loss_minus_f = float(loss_minus.detach().cpu())
            d_h = (loss_plus_f - loss_minus_f) / (2.0 * args.h)
            finite = math.isfinite(loss_plus_f) and math.isfinite(loss_minus_f) and math.isfinite(d_h)
            if finite:
                finite_count += 1
                last_update_stats = residual_commit_update(
                    master,
                    states,
                    codes,
                    residuals,
                    directions,
                    lr=args.lr,
                    d_h=d_h,
                    max_code_step=args.max_code_step,
                )
                smoke.restore_master(params, master)
            train_loss = (loss_plus_f + loss_minus_f) / 2.0

            completed_step = step_idx + 1
            if completed_step % args.diag_every == 0 or completed_step == 1 or completed_step == args.steps:
                last_pert = smoke.perturbation_metrics(master, directions, states, args.h)
                last_pert["code_change_frac"] = last_pert["active_frac"]
                last_pert["grid_id_plus"] = 1
                last_pert["grid_id_minus"] = 1
                last_pert["scale_id_plus"] = 1
                last_pert["scale_id_minus"] = 1
                append_jsonl(run_dir / "perturbation_diagnostics.jsonl", {"step": completed_step, **last_pert})
                append_jsonl(run_dir / "update_diagnostics.jsonl", {"step": completed_step, **last_update_stats})

            eval_loss = None
            eval_acc = None
            if completed_step % args.eval_every == 0 or completed_step == args.steps:
                eval_loss, eval_acc = rtn_batch.evaluate_full(model, params, master, states, dev_loader, device, args.eval_batches)
                last_eval_loss = eval_loss
                last_eval_acc = eval_acc
                last_eval_step = completed_step
                append_jsonl(run_dir / "eval_metrics.jsonl", {"step": completed_step, "eval_loss": eval_loss, "eval_acc": eval_acc})
                if eval_acc is not None and (best["best_eval_acc"] is None or eval_acc > best["best_eval_acc"]):
                    best["best_eval_acc"] = eval_acc
                    best["best_eval_step"] = completed_step
                    save_checkpoint(run_dir / "checkpoints" / f"step_{completed_step}", step=completed_step, master=master, codes=codes, residuals=residuals, states=states, best=best, config=config)
                    copy_checkpoint(run_dir / "checkpoints" / f"step_{completed_step}", run_dir / "checkpoints" / "best_acc")
                if eval_loss is not None and (best["best_eval_loss"] is None or eval_loss < best["best_eval_loss"]):
                    best["best_eval_loss"] = eval_loss
                    best["best_eval_loss_step"] = completed_step
                    save_checkpoint(run_dir / "checkpoints" / f"step_{completed_step}", step=completed_step, master=master, codes=codes, residuals=residuals, states=states, best=best, config=config)
                    copy_checkpoint(run_dir / "checkpoints" / f"step_{completed_step}", run_dir / "checkpoints" / "best_loss")

            if completed_step % args.checkpoint_steps == 0 or completed_step == args.steps:
                save_checkpoint(run_dir / "checkpoints" / f"step_{completed_step}", step=completed_step, master=master, codes=codes, residuals=residuals, states=states, best=best, config=config)

            nan_flag = (
                not finite
                or restore_diff > 1e-3
                or not math.isfinite(float(last_update_stats.get("update_norm", 0.0)))
                or float(last_update_stats.get("ef_error_norm", 0.0)) != 0.0
                or float(last_update_stats.get("grid_error_norm", 0.0)) > 1e-5
            )
            writer.writerow(
                {
                    "step": completed_step,
                    "loss_plus": loss_plus_f,
                    "loss_minus": loss_minus_f,
                    "train_loss": train_loss,
                    "d_h": d_h,
                    "d_h_finite": finite,
                    "update_norm": last_update_stats.get("update_norm"),
                    "active_frac": last_update_stats.get("active_frac"),
                    "candidate_active_frac": last_update_stats.get("candidate_active_frac"),
                    "cos_intended_actual": last_update_stats.get("cos_intended_actual"),
                    "acc_actual_cos": last_update_stats.get("acc_actual_cos"),
                    "actual_over_intended_norm_ratio": last_update_stats.get("actual_over_intended_norm_ratio"),
                    "actual_over_acc_norm_ratio": last_update_stats.get("actual_over_acc_norm_ratio"),
                    "residual_norm": last_update_stats.get("residual_norm"),
                    "residual_over_scale_max": last_update_stats.get("residual_over_scale_max"),
                    "residual_bound_violation_frac": last_update_stats.get("residual_bound_violation_frac"),
                    "grid_error_norm": last_update_stats.get("grid_error_norm"),
                    "ef_error_norm": last_update_stats.get("ef_error_norm"),
                    "saturation_frac": last_update_stats.get("saturation_frac"),
                    "eval_loss": eval_loss,
                    "eval_acc": eval_acc,
                    "seconds": time.time() - step_start,
                    "nan_flag": nan_flag,
                }
            )
            f.flush()
            if completed_step == 1 or completed_step % args.log_every == 0:
                append_log(
                    log_path,
                    f"step={completed_step}/{args.steps} loss={train_loss:.6g} d_h={d_h:.6g} "
                    f"eval_acc={eval_acc} commit_active={last_update_stats.get('active_frac')} "
                    f"acc_cos={last_update_stats.get('acc_actual_cos')} actual/acc={last_update_stats.get('actual_over_acc_norm_ratio')}",
                )
            if nan_flag:
                status = "failed"
                error_message = f"nan/grid/EF invariant failure at step {completed_step}"
                append_log(log_path, error_message)
                break

    steps_completed = 0
    rows: List[Dict[str, str]] = []
    with metrics_path.open(newline="", encoding="utf-8") as mf:
        rows = list(csv.DictReader(mf))
        if rows:
            steps_completed = int(float(rows[-1]["step"]))
    save_checkpoint(run_dir / "checkpoints" / "final", step=steps_completed, master=master, codes=codes, residuals=residuals, states=states, best=best, config=config)
    if not (run_dir / "checkpoints" / "best_acc").exists():
        copy_checkpoint(run_dir / "checkpoints" / "final", run_dir / "checkpoints" / "best_acc")
    if not (run_dir / "checkpoints" / "best_loss").exists():
        copy_checkpoint(run_dir / "checkpoints" / "final", run_dir / "checkpoints" / "best_loss")

    if status != "failed":
        status = "complete" if steps_completed >= args.steps else "partial"
    total_runtime = time.time() - total_start
    peak_mem = float(torch.cuda.max_memory_allocated() / 1024 / 1024) if torch.cuda.is_available() else 0.0
    summary = {
        **config,
        "status": status,
        "error_message": error_message,
        "steps_completed": steps_completed,
        "best_eval_acc": best.get("best_eval_acc"),
        "best_eval_step": best.get("best_eval_step"),
        "best_eval_loss": best.get("best_eval_loss"),
        "best_eval_loss_step": best.get("best_eval_loss_step"),
        "last_eval_acc": last_eval_acc,
        "last_eval_loss": last_eval_loss,
        "last_eval_step": last_eval_step,
        "final_train_loss": train_loss,
        "d_h_finite_rate": finite_count / max(steps_completed, 1),
        "num_scale_refreshes": 1,
        "scale_drift_max": 0.0,
        "seconds_per_step": total_runtime / max(steps_completed, 1),
        "total_runtime": total_runtime,
        "peak_gpu_mem": peak_mem,
        "quantizer_initial": quant_agg,
        "update_last": last_update_stats,
        "perturbation_last": last_pert,
    }
    write_json(run_dir / "run_summary.json", summary)
    write_report(run_dir, summary, synth)
    append_log(log_path, f"finished status={status} steps={steps_completed} best_acc={best.get('best_eval_acc')} last_acc={last_eval_acc}")
    return summary


def write_report(run_dir: Path, summary: Dict[str, object], synth: Dict[str, object]) -> None:
    update = summary.get("update_last", {}) or {}
    pert = summary.get("perturbation_last", {}) or {}
    lines = [
        "# INT4 RTNClip Residual-Grid Local Report",
        "",
        "This run uses fixed G128 RTNClip INT4 scales computed once from the initial unperturbed RoBERTa-large SST-5 weights.",
        "Linear weights are stored/updated as INT4 lattice codes plus FP32 residual buffers; non-linear parameters are frozen.",
        "",
        "References used for the update rule:",
        "- Karimireddy et al., 2019, Error Feedback Fixes SignSGD and other Gradient Compression Schemes.",
        "- Stich and Karimireddy, 2020, The Error-Feedback framework: SGD with Delayed Gradients.",
        "",
        f"Synthetic residual check: `{synth.get('status')}`.",
        "",
        "| metric | value |",
        "|---|---:|",
        f"| status | {summary.get('status')} |",
        f"| steps_completed | {summary.get('steps_completed')} |",
        f"| h | {summary.get('h')} |",
        f"| lr | {summary.get('lr')} |",
        f"| best_eval_acc | {summary.get('best_eval_acc')} |",
        f"| last_eval_acc | {summary.get('last_eval_acc')} |",
        f"| final_train_loss | {summary.get('final_train_loss')} |",
        f"| commit_active_frac_last | {update.get('active_frac')} |",
        f"| perturb_active_frac_last | {pert.get('active_frac')} |",
        f"| acc_actual_cos_last | {update.get('acc_actual_cos')} |",
        f"| actual_over_acc_norm_ratio_last | {update.get('actual_over_acc_norm_ratio')} |",
        f"| residual_over_scale_max_last | {update.get('residual_over_scale_max')} |",
        f"| residual_bound_violation_frac_last | {update.get('residual_bound_violation_frac')} |",
        f"| grid_error_norm_last | {update.get('grid_error_norm')} |",
        f"| ef_error_norm_last | {update.get('ef_error_norm')} |",
        f"| scale_drift_max | {summary.get('scale_drift_max')} |",
    ]
    (run_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RoBERTa-large SST-5 INT4 RTNClip fixed-grid residual update local runner")
    parser.add_argument("--output_dir", default=str(REPO_ROOT / "outputs" / f"int4_rtnclip_residual_grid_sst5_{datetime.now().strftime('%Y%m%d_%H%M%S')}"))
    parser.add_argument("--run_name", default="int4_rtnclip_fixedscale_residual_grid_sst5_h1e-3")
    parser.add_argument("--dataset_mode", choices=["full", "fewshot"], default="full")
    parser.add_argument("--data_dir", default="")
    parser.add_argument("--num_k", type=int, default=16)
    parser.add_argument("--seed", type=int, default=16)
    parser.add_argument("--data_seed", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--eval_batches", type=int, default=-1)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--h", type=float, default=1e-3)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--group_size", type=int, default=128)
    parser.add_argument("--max_code_step", type=int, default=1)
    parser.add_argument("--eval_every", type=int, default=100)
    parser.add_argument("--checkpoint_steps", type=int, default=100)
    parser.add_argument("--diag_every", type=int, default=25)
    parser.add_argument("--log_every", type=int, default=25)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
