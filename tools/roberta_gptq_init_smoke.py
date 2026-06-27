#!/usr/bin/env python
"""RoBERTa Hessian-GPTQ-initialized shared-grid ZO smoke.

This is a local proof-of-concept runner.  It performs one offline GPTQ-style
Hessian reconstruction pass for selected RoBERTa Linear weights, freezes the
resulting groupwise scale/zero grid, then runs a short two-point MeZO smoke
using fresh rounding on that fixed grid.

It intentionally does not call this path "full GPTQ training": GPTQ is only the
initialization/calibration step; the ZO probes use a fixed shared fake-quant
grid derived from the GPTQ-reconstructed base weights.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import re
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = REPO_ROOT / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import smoke_rtnclip_roberta_sst5 as smoke  # noqa: E402


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def append_jsonl(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, sort_keys=True, default=str) + "\n")


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    except Exception:
        return ""


@dataclass
class GPTQGridState:
    name: str
    shape: Tuple[int, int]
    bits: int
    group_size: int
    qmin: int
    qmax: int
    scales: torch.Tensor
    zeros: torch.Tensor
    codes: torch.Tensor
    lengths: torch.Tensor
    valid: torch.Tensor


def _group_view_2d(weight: torch.Tensor, group_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if weight.ndim != 2:
        raise ValueError(f"Linear weight must be 2D, got shape={tuple(weight.shape)}")
    out_features, in_features = weight.shape
    num_groups = int(math.ceil(in_features / group_size))
    padded_cols = num_groups * group_size
    pad_cols = padded_cols - in_features
    w = weight.float()
    if pad_cols:
        w = F.pad(w, (0, pad_cols))
    groups = w.reshape(out_features, num_groups, group_size)
    lengths = torch.full((num_groups,), group_size, device=weight.device, dtype=torch.long)
    if pad_cols:
        lengths[-1] = group_size - pad_cols
    valid = torch.arange(group_size, device=weight.device).view(1, 1, group_size) < lengths.view(1, num_groups, 1)
    return groups, lengths, valid


def compute_asym_group_grid(weight: torch.Tensor, *, bits: int, group_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    qmin = 0
    qmax = (1 << bits) - 1
    groups, lengths, valid = _group_view_2d(weight, group_size)
    masked = groups.masked_fill(~valid, 0.0)
    w_min = masked.masked_fill(~valid, float("inf")).amin(dim=-1)
    w_max = masked.masked_fill(~valid, float("-inf")).amax(dim=-1)
    zero_groups = ~torch.isfinite(w_min) | ~torch.isfinite(w_max) | ((w_max - w_min).abs() <= 0)
    w_min = torch.where(zero_groups, torch.zeros_like(w_min), w_min)
    w_max = torch.where(zero_groups, torch.ones_like(w_max), w_max)
    scales = ((w_max - w_min) / float(qmax - qmin)).clamp_min(1e-12)
    zeros = torch.round(float(qmin) - w_min / scales).clamp(qmin, qmax)
    return scales.detach(), zeros.detach(), lengths.detach(), valid.detach(), zero_groups.detach()


def quantize_with_gptq_state(weight: torch.Tensor, state: GPTQGridState, return_stats: bool = False):
    groups, _, valid = _group_view_2d(weight, state.group_size)
    scales = state.scales.unsqueeze(-1)
    zeros = state.zeros.unsqueeze(-1)
    q = torch.round(groups / scales + zeros).clamp(state.qmin, state.qmax)
    q = q.masked_fill(~valid, 0.0)
    wq = ((q - zeros) * scales).reshape(weight.shape[0], -1)[:, : weight.shape[1]].reshape_as(weight)
    wq = wq.to(dtype=weight.dtype)
    if not return_stats:
        return wq
    valid_q = q[valid.expand_as(q)]
    stats = {
        "code_min": int(valid_q.min().detach().cpu()) if valid_q.numel() else 0,
        "code_max": int(valid_q.max().detach().cpu()) if valid_q.numel() else 0,
        "clip_frac": float(((valid_q <= state.qmin) | (valid_q >= state.qmax)).float().mean().detach().cpu()) if valid_q.numel() else 0.0,
    }
    return wq, stats


def dequantize_state_codes(state: GPTQGridState, *, dtype: torch.dtype = torch.float16) -> torch.Tensor:
    """Dequantize the committed GPTQ initialization codes."""
    in_features = state.shape[1]
    scale_full = state.scales.repeat_interleave(state.group_size, dim=1)[:, :in_features]
    zero_full = state.zeros.repeat_interleave(state.group_size, dim=1)[:, :in_features]
    return ((state.codes[:, :in_features].float() - zero_full) * scale_full).to(dtype=dtype)


def _quantize_column_asym(col: torch.Tensor, scales: torch.Tensor, zeros: torch.Tensor, group_idx: int, qmin: int, qmax: int) -> Tuple[torch.Tensor, torch.Tensor]:
    scale = scales[:, group_idx]
    zero = zeros[:, group_idx]
    q = torch.round(col / scale + zero).clamp(qmin, qmax)
    deq = (q - zero) * scale
    return q, deq


def hessian_gptq_reconstruct(
    name: str,
    weight: torch.Tensor,
    hessian: Optional[torch.Tensor],
    *,
    bits: int,
    group_size: int,
    damp_percent: float,
    mode: str,
) -> Tuple[GPTQGridState, Dict[str, object]]:
    """Sequential GPTQ-style reconstruction with fixed asymmetric group grids."""
    if weight.ndim != 2:
        raise ValueError(f"{name}: expected 2D Linear weight, got {tuple(weight.shape)}")
    device = weight.device
    qmin = 0
    qmax = (1 << bits) - 1
    orig = weight.detach().float()
    out_features, in_features = orig.shape
    scales, zeros, lengths, valid, _ = compute_asym_group_grid(orig, bits=bits, group_size=group_size)

    if mode == "rtn":
        q_groups = torch.round(_group_view_2d(orig, group_size)[0] / scales.unsqueeze(-1) + zeros.unsqueeze(-1)).clamp(qmin, qmax)
        q_groups = q_groups.masked_fill(~valid, 0.0)
        codes = q_groups.reshape(out_features, -1)[:, :in_features].to(torch.int16)
    else:
        if hessian is None:
            hessian = torch.eye(in_features, device=device, dtype=torch.float32)
        H = hessian.detach().float().to(device)
        if H.shape != (in_features, in_features):
            raise ValueError(f"{name}: Hessian shape {tuple(H.shape)} does not match in_features={in_features}")
        diag = torch.diag(H)
        dead = diag <= 0
        if bool(dead.any()):
            H = H.clone()
            H[dead, dead] = 1.0
        damp = float(damp_percent) * float(torch.mean(torch.diag(H)).detach().cpu())
        H = H.clone()
        H[torch.arange(in_features, device=device), torch.arange(in_features, device=device)] += max(damp, 1e-8)
        try:
            chol = torch.linalg.cholesky(H)
            Hinv = torch.cholesky_inverse(chol)
        except Exception:
            # Fall back to pinv for numerically awkward tiny smoke cases.
            Hinv = torch.linalg.pinv(H)
        if mode == "diag":
            Hinv = torch.diag(torch.diag(Hinv))
        W = orig.clone()
        if bool(dead.any()):
            W[:, dead] = 0.0
        codes = torch.empty((out_features, in_features), device=device, dtype=torch.int16)
        for col_idx in range(in_features):
            group_idx = col_idx // group_size
            q_col, deq_col = _quantize_column_asym(W[:, col_idx], scales, zeros, group_idx, qmin, qmax)
            codes[:, col_idx] = q_col.to(torch.int16)
            denom = Hinv[col_idx, col_idx].clamp_min(1e-12)
            err = (W[:, col_idx] - deq_col) / denom
            if col_idx + 1 < in_features:
                W[:, col_idx + 1 :] -= err.unsqueeze(1) * Hinv[col_idx, col_idx + 1 :].unsqueeze(0)

    padded_cols = int(math.ceil(in_features / group_size)) * group_size
    if padded_cols > in_features:
        pad = torch.zeros((out_features, padded_cols - in_features), device=device, dtype=codes.dtype)
        codes_padded = torch.cat([codes, pad], dim=1)
    else:
        codes_padded = codes
    state = GPTQGridState(
        name=name,
        shape=tuple(weight.shape),
        bits=int(bits),
        group_size=int(group_size),
        qmin=qmin,
        qmax=qmax,
        scales=scales.detach(),
        zeros=zeros.detach(),
        codes=codes_padded.detach(),
        lengths=lengths.detach(),
        valid=valid.detach(),
    )
    deq, q_stats = quantize_with_gptq_state(orig, state, return_stats=True)
    # Use recorded GPTQ codes for the reconstruction error, not a re-rounded RTN
    # copy.  This captures the sequential Hessian correction committed at init.
    scale_full = state.scales.repeat_interleave(group_size, dim=1)[:, :in_features]
    zero_full = state.zeros.repeat_interleave(group_size, dim=1)[:, :in_features]
    deq_from_codes = ((state.codes[:, :in_features].float() - zero_full) * scale_full).to(orig.dtype)
    diff = deq_from_codes - orig
    err_sq = diff.double().square().sum()
    ref_sq = orig.double().square().sum().clamp_min(1e-30)
    stats = {
        "module_name": name,
        "bits": int(bits),
        "group_size": int(group_size),
        "mode": mode,
        "qmin": qmin,
        "qmax": qmax,
        "num_groups": int(scales.numel()),
        "scale_min": float(scales.min().detach().cpu()),
        "scale_median": float(scales.median().detach().cpu()),
        "scale_max": float(scales.max().detach().cpu()),
        "zero_min": float(zeros.min().detach().cpu()),
        "zero_median": float(zeros.median().detach().cpu()),
        "zero_max": float(zeros.max().detach().cpu()),
        "clip_frac_reround": q_stats["clip_frac"],
        "weight_recon_mse": float((err_sq / max(orig.numel(), 1)).detach().cpu()),
        "weight_recon_rel_mse": float((err_sq / ref_sq).detach().cpu()),
        "weight_recon_sqnr_db": float((10.0 * torch.log10(ref_sq / err_sq.clamp_min(1e-30))).detach().cpu()),
        "hessian_trace": float(torch.trace(hessian.detach().float()).detach().cpu()) if hessian is not None else None,
        "hessian_damp_percent": float(damp_percent),
    }
    return state, stats


def rtn_symmetric_reconstruct(
    name: str,
    weight: torch.Tensor,
    *,
    bits: int,
    group_size: int,
) -> Tuple[GPTQGridState, Dict[str, object]]:
    """Plain groupwise signed symmetric RTN state, used as a control."""
    if weight.ndim != 2:
        raise ValueError(f"{name}: expected 2D Linear weight, got {tuple(weight.shape)}")
    qmax = (1 << (bits - 1)) - 1
    qmin = -qmax
    orig = weight.detach().float()
    out_features, in_features = orig.shape
    groups, lengths, valid = _group_view_2d(orig, group_size)
    max_abs = groups.abs().masked_fill(~valid, 0.0).amax(dim=-1)
    scales = (max_abs / float(qmax)).clamp_min(1e-12)
    zeros = torch.zeros_like(scales)
    q_groups = torch.round(groups / scales.unsqueeze(-1)).clamp(qmin, qmax)
    q_groups = q_groups.masked_fill(~valid, 0.0)
    padded_cols = int(math.ceil(in_features / group_size)) * group_size
    codes = q_groups.reshape(out_features, padded_cols).to(torch.int16)
    state = GPTQGridState(
        name=name,
        shape=tuple(weight.shape),
        bits=int(bits),
        group_size=int(group_size),
        qmin=qmin,
        qmax=qmax,
        scales=scales.detach(),
        zeros=zeros.detach(),
        codes=codes.detach(),
        lengths=lengths.detach(),
        valid=valid.detach(),
    )
    deq = dequantize_state_codes(state, dtype=torch.float32)
    diff = deq - orig
    err_sq = diff.double().square().sum()
    ref_sq = orig.double().square().sum().clamp_min(1e-30)
    valid_q = q_groups[valid.expand_as(q_groups)]
    stats = {
        "module_name": name,
        "bits": int(bits),
        "group_size": int(group_size),
        "mode": "rtn_symmetric",
        "qmin": qmin,
        "qmax": qmax,
        "num_groups": int(scales.numel()),
        "scale_min": float(scales.min().detach().cpu()),
        "scale_median": float(scales.median().detach().cpu()),
        "scale_max": float(scales.max().detach().cpu()),
        "zero_min": 0.0,
        "zero_median": 0.0,
        "zero_max": 0.0,
        "clip_frac_reround": float((valid_q.abs() >= qmax).float().mean().detach().cpu()) if valid_q.numel() else 0.0,
        "weight_recon_mse": float((err_sq / max(orig.numel(), 1)).detach().cpu()),
        "weight_recon_rel_mse": float((err_sq / ref_sq).detach().cpu()),
        "weight_recon_sqnr_db": float((10.0 * torch.log10(ref_sq / err_sq.clamp_min(1e-30))).detach().cpu()),
        "hessian_trace": None,
        "hessian_damp_percent": None,
    }
    return state, stats


def select_linear_names(model: nn.Module, scope: str) -> List[str]:
    all_names = [f"{name}.weight" for name, module in model.named_modules() if isinstance(module, nn.Linear)]
    if scope == "all_linear":
        return all_names
    if scope == "classifier":
        selected = [name for name in all_names if "classifier" in name or "lm_head" in name]
        return selected or all_names[-2:]

    layer_ids = []
    for name in all_names:
        m = re.search(r"roberta\.encoder\.layer\.(\d+)\.", name)
        if m:
            layer_ids.append(int(m.group(1)))
    if not layer_ids:
        return all_names[-2:]
    last = max(layer_ids)
    if scope == "last_block_all_linear":
        return [name for name in all_names if f"roberta.encoder.layer.{last}." in name]
    if scope == "last_mlp":
        needles = (
            f"roberta.encoder.layer.{last}.intermediate.dense.weight",
            f"roberta.encoder.layer.{last}.output.dense.weight",
        )
        selected = [name for name in all_names if name in needles]
        return selected or [name for name in all_names if f"roberta.encoder.layer.{last}." in name][-2:]
    raise ValueError(f"Unsupported module_scope={scope!r}")


def collect_hessians(
    model: nn.Module,
    train_loader,
    module_names: List[str],
    *,
    device: torch.device,
    calib_batches: int,
    max_rows_per_module_per_batch: int,
    seed: int,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, object]]:
    modules = dict(model.named_modules())
    module_base_names = [name.rsplit(".weight", 1)[0] for name in module_names]
    hessians: Dict[str, torch.Tensor] = {}
    row_counts: Dict[str, int] = {name: 0 for name in module_names}
    gen = torch.Generator(device=device).manual_seed(seed + 9173)
    hooks = []

    def make_hook(weight_name: str):
        def hook(_module, inputs):
            if not inputs:
                return
            x = inputs[0].detach().float()
            if x.numel() == 0:
                return
            X = x.reshape(-1, x.shape[-1])
            if X.shape[0] > max_rows_per_module_per_batch:
                idx = torch.randperm(X.shape[0], device=X.device, generator=gen)[:max_rows_per_module_per_batch]
                X = X.index_select(0, idx)
            H = X.t().matmul(X)
            if weight_name not in hessians:
                hessians[weight_name] = H
            else:
                hessians[weight_name].add_(H)
            row_counts[weight_name] += int(X.shape[0])

        return hook

    for base_name, weight_name in zip(module_base_names, module_names):
        hooks.append(modules[base_name].register_forward_pre_hook(make_hook(weight_name)))
    batch_iter = iter(train_loader)
    model.eval()
    with torch.no_grad():
        for _ in range(calib_batches):
            try:
                batch = next(batch_iter)
            except StopIteration:
                break
            batch = smoke.move_batch(batch, device)
            _ = smoke.forward_loss_and_logits(model, batch)
    for hook in hooks:
        hook.remove()
    for name, H in list(hessians.items()):
        rows = max(row_counts.get(name, 0), 1)
        hessians[name] = (H / float(rows)).detach()
    meta = {
        "calib_batches": int(calib_batches),
        "max_rows_per_module_per_batch": int(max_rows_per_module_per_batch),
        "row_counts": row_counts,
        "hessian_modules_collected": len(hessians),
    }
    return hessians, meta


def copy_branch_to_model(
    params: Dict[str, nn.Parameter],
    master: Dict[str, torch.Tensor],
    directions: Optional[Dict[str, torch.Tensor]],
    h: float,
    sign: float,
    states: Dict[str, GPTQGridState],
) -> None:
    with torch.no_grad():
        for name, tensor in master.items():
            value = tensor if directions is None else tensor.float().add(directions[name].float(), alpha=sign * h)
            if name in states:
                value = quantize_with_gptq_state(value, states[name])
            params[name].copy_(value.to(dtype=params[name].dtype))


def restore_master_subset(params: Dict[str, nn.Parameter], master: Dict[str, torch.Tensor]) -> float:
    max_diff = 0.0
    with torch.no_grad():
        for name, tensor in master.items():
            params[name].copy_(tensor.to(dtype=params[name].dtype))
            diff = (params[name].detach().float() - tensor.float()).abs().max()
            max_diff = max(max_diff, float(diff.detach().cpu()))
    return max_diff


def sample_directions(master: Dict[str, torch.Tensor], gen: torch.Generator) -> Dict[str, torch.Tensor]:
    return {name: torch.randn(tensor.shape, device=tensor.device, dtype=tensor.dtype, generator=gen) for name, tensor in master.items()}


def update_master(master: Dict[str, torch.Tensor], directions: Dict[str, torch.Tensor], lr: float, d_h: float) -> float:
    sq = torch.zeros((), device=next(iter(master.values())).device, dtype=torch.float64)
    with torch.no_grad():
        for name, tensor in master.items():
            update = directions[name].float().mul(-float(lr) * float(d_h))
            sq += update.double().square().sum()
            tensor.copy_(tensor.float().add(update).to(dtype=tensor.dtype))
    return float(sq.sqrt().detach().cpu())


def perturbation_metrics(master: Dict[str, torch.Tensor], directions: Dict[str, torch.Tensor], states: Dict[str, GPTQGridState], h: float) -> Dict[str, object]:
    active = 0
    total = 0
    dot = torch.zeros((), device=next(iter(master.values())).device, dtype=torch.float64)
    delta_sq = torch.zeros_like(dot)
    intended_sq = torch.zeros_like(dot)
    err_sq = torch.zeros_like(dot)
    clip_plus = 0.0
    clip_minus = 0.0
    legal = True
    values = 0
    for name, state in states.items():
        intended = 2.0 * h * directions[name].float()
        plus, ps = quantize_with_gptq_state(master[name].float().add(directions[name].float(), alpha=h), state, True)
        minus, ms = quantize_with_gptq_state(master[name].float().add(directions[name].float(), alpha=-h), state, True)
        delta = plus.float() - minus.float()
        active += int((delta != 0).sum().detach().cpu())
        total += delta.numel()
        dot += (delta.double() * intended.double()).sum()
        delta_sq += delta.double().square().sum()
        intended_sq += intended.double().square().sum()
        err_sq += (delta.float() - intended).double().square().sum()
        clip_plus += ps["clip_frac"] * delta.numel()
        clip_minus += ms["clip_frac"] * delta.numel()
        values += delta.numel()
        legal = legal and ps["code_min"] >= state.qmin and ps["code_max"] <= state.qmax
        legal = legal and ms["code_min"] >= state.qmin and ms["code_max"] <= state.qmax
    eps = 1e-12
    return {
        "active_frac": active / max(total, 1),
        "zero_effective_displacement_frac": 1.0 - active / max(total, 1),
        "alignment": float((dot / (delta_sq.sqrt() * intended_sq.sqrt() + eps)).detach().cpu()),
        "norm_ratio": float((delta_sq.sqrt() / (intended_sq.sqrt() + eps)).detach().cpu()),
        "delta_visibility_nmse": float((err_sq / intended_sq.clamp_min(eps)).detach().cpu()),
        "delta_visibility_mse": float((err_sq / max(total, 1)).detach().cpu()),
        "saturation_frac_w_plus": clip_plus / max(values, 1),
        "saturation_frac_w_minus": clip_minus / max(values, 1),
        "codes_legal": bool(legal),
    }


def evaluate_quantized(model, params, master, states, dev_loader, device, max_batches: int):
    if max_batches <= 0:
        return None, None
    copy_branch_to_model(params, master, None, 0.0, 0.0, states)
    total_loss = 0.0
    total_correct = 0
    total_items = 0
    for idx, batch in enumerate(dev_loader):
        if idx >= max_batches:
            break
        batch = smoke.move_batch(batch, device)
        loss, logits = smoke.forward_loss_and_logits(model, batch)
        labels = batch["labels"]
        total_loss += float(loss.detach().cpu()) * int(labels.numel())
        total_correct += int((logits.argmax(dim=-1) == labels).sum().detach().cpu())
        total_items += int(labels.numel())
    restore_master_subset(params, master)
    if total_items == 0:
        return None, None
    return total_loss / total_items, total_correct / total_items


def run(args: argparse.Namespace) -> Dict[str, object]:
    if os.environ.get("DATALOADER_SHUFFLE") != "True":
        raise RuntimeError("DATALOADER_SHUFFLE=True is required.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this RoBERTa-large smoke.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_dir = args.output_dir / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    env = smoke.collect_env(REPO_ROOT)
    write_json(run_dir / "env.json", env)

    # medium_models checkpoints were written with older torch semantics in a few
    # places; this keeps dataset/model loading compatible across environments.
    orig_torch_load = torch.load

    def _compat_torch_load(*load_args, **load_kwargs):
        load_kwargs.setdefault("weights_only", False)
        return orig_torch_load(*load_args, **load_kwargs)

    torch.load = _compat_torch_load
    try:
        model, train_loader, dev_loader, data_args, train_sampler = smoke.load_prompt_model_and_data(args, device)
    finally:
        torch.load = orig_torch_load
    model.eval()
    if type(train_sampler).__name__ != "RandomSampler":
        raise RuntimeError(f"Expected RandomSampler, got {type(train_sampler).__name__}")
    params = dict(model.named_parameters())
    selected = select_linear_names(model, args.module_scope)
    if args.max_modules > 0:
        selected = selected[: args.max_modules]
    if args.init_quantizer == "gptq_asym":
        hessians, hmeta = collect_hessians(
            model,
            train_loader,
            selected,
            device=device,
            calib_batches=args.calib_batches,
            max_rows_per_module_per_batch=args.max_rows_per_module_per_batch,
            seed=args.seed,
        )
    else:
        hessians, hmeta = {}, {"calib_batches": 0, "hessian_modules_collected": 0, "note": "RTN symmetric control does not use Hessian."}
    states: Dict[str, GPTQGridState] = {}
    quant_rows: List[Dict[str, object]] = []
    t_quant = time.time()
    for name in selected:
        if args.init_quantizer == "gptq_asym":
            state, stats = hessian_gptq_reconstruct(
                name,
                params[name].detach(),
                hessians.get(name),
                bits=args.bits,
                group_size=args.group_size,
                damp_percent=args.damp_percent,
                mode=args.gptq_mode,
            )
        else:
            state, stats = rtn_symmetric_reconstruct(
                name,
                params[name].detach(),
                bits=args.bits,
                group_size=args.group_size,
            )
        states[name] = state
        quant_rows.append(stats)
        append_jsonl(run_dir / "gptq_init_diagnostics.jsonl", stats)
    quant_seconds = time.time() - t_quant
    if args.train_scope == "quantized_linear_weights_only":
        train_names = list(selected)
    elif args.train_scope == "all_float_params":
        train_names = [name for name, param in params.items() if param.detach().is_floating_point()]
    else:
        raise ValueError(f"Unsupported train_scope={args.train_scope!r}")
    master = {name: params[name].detach().clone().to(device=device, dtype=torch.float16) for name in train_names}
    # Start the trainable low-bit state from the committed GPTQ-dequantized
    # weights, then keep scale/zero fixed for all later probes.
    with torch.no_grad():
        for name, state in states.items():
            master[name].copy_(dequantize_state_codes(state, dtype=torch.float16))
    restore_master_subset(params, master)

    run_config = {
        "method": "hessian_gptq_initialized_fixed_grid_mezo_smoke",
        "init_quantizer": args.init_quantizer,
        "not_full_gptq_training": True,
        "gptq_usage": "offline_initialization_only" if args.init_quantizer == "gptq_asym" else "not_used_rtn_control",
        "model": args.model_id,
        "task_name": args.task_name,
        "dataset_mode": args.dataset_mode,
        "data_dir": getattr(data_args, "data_dir", ""),
        "seed": args.seed,
        "data_seed": args.data_seed,
        "batch_size": args.batch_size,
        "sampler_name": type(train_sampler).__name__,
        "bits": args.bits,
        "group_size": args.group_size,
        "gptq_mode": args.gptq_mode,
        "damp_percent": args.damp_percent,
        "module_scope": args.module_scope,
        "quantized_modules": selected,
        "train_scope": args.train_scope,
        "trainable_master_tensors": train_names,
        "scale_zero_source": "hessian_gptq_reconstructed_initial_weight" if args.init_quantizer == "gptq_asym" else "plain_groupwise_symmetric_rtn_initial_weight",
        "scale_zero_frozen_after_init": True,
        "pair_shared_grid": True,
        "fresh_round_codes": True,
        "h": args.h,
        "lr": args.lr,
        "steps": args.steps,
        "calibration": hmeta,
        "gptq_init_seconds": quant_seconds,
        "git_commit": git_commit(),
        "hostname": socket.gethostname(),
        "output_dir": str(run_dir),
    }
    write_json(run_dir / "run_config.json", run_config)
    write_json(run_dir / "run_manifest_row.json", run_config)

    quant_agg = {
        "num_modules": len(quant_rows),
        "weight_recon_mse_mean": sum(float(r["weight_recon_mse"]) for r in quant_rows) / max(len(quant_rows), 1),
        "weight_recon_rel_mse_mean": sum(float(r["weight_recon_rel_mse"]) for r in quant_rows) / max(len(quant_rows), 1),
        "weight_recon_sqnr_db_mean": sum(float(r["weight_recon_sqnr_db"]) for r in quant_rows) / max(len(quant_rows), 1),
        "scale_min": min(float(r["scale_min"]) for r in quant_rows) if quant_rows else None,
        "scale_max": max(float(r["scale_max"]) for r in quant_rows) if quant_rows else None,
    }
    write_json(run_dir / "gptq_init_summary.json", {"config": run_config, "aggregate": quant_agg, "modules": quant_rows})

    metrics_path = run_dir / "metrics.csv"
    pert_path = run_dir / "perturbation_diagnostics.jsonl"
    gen = torch.Generator(device=device).manual_seed(args.seed + 314159)
    batch_iter = smoke.cycle(train_loader)
    d_h_finite = 0
    final_train_loss = float("nan")
    update_norm_last = float("nan")
    last_pert: Dict[str, object] = {}
    status = "pass"
    error_message = ""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    start = time.time()
    with metrics_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["step", "loss_plus", "loss_minus", "d_h", "update_norm", "seconds", "nan_flag"])
        writer.writeheader()
        for step in range(args.steps):
            step_start = time.time()
            directions = sample_directions(master, gen)
            batch = smoke.move_batch(next(batch_iter), device)
            copy_branch_to_model(params, master, directions, args.h, +1.0, states)
            loss_plus, _ = smoke.forward_loss_and_logits(model, batch)
            copy_branch_to_model(params, master, directions, args.h, -1.0, states)
            loss_minus, _ = smoke.forward_loss_and_logits(model, batch)
            restore_master_subset(params, master)
            lp = float(loss_plus.detach().cpu())
            lm = float(loss_minus.detach().cpu())
            d_h = (lp - lm) / (2.0 * args.h)
            finite = math.isfinite(lp) and math.isfinite(lm) and math.isfinite(d_h)
            if finite:
                d_h_finite += 1
                update_norm_last = update_master(master, directions, args.lr, d_h)
                restore_master_subset(params, master)
            else:
                status = "failed"
                error_message = f"non-finite loss/d_h at step {step + 1}"
            last_pert = perturbation_metrics(master, directions, states, args.h)
            append_jsonl(pert_path, {"step": step + 1, **last_pert})
            final_train_loss = (lp + lm) / 2.0
            row = {
                "step": step + 1,
                "loss_plus": lp,
                "loss_minus": lm,
                "d_h": d_h,
                "update_norm": update_norm_last,
                "seconds": time.time() - step_start,
                "nan_flag": (not finite) or (not last_pert.get("codes_legal", False)),
            }
            writer.writerow(row)
            f.flush()
            print(
                f"[{datetime.now().isoformat(timespec='seconds')}] step={step+1}/{args.steps} "
                f"loss={final_train_loss:.6g} d_h={d_h:.6g} "
                f"active={last_pert['active_frac']:.4f} align={last_pert['alignment']:.4f} "
                f"norm_ratio={last_pert['norm_ratio']:.4f}",
                flush=True,
            )
            if status != "pass":
                break
    eval_loss, eval_acc = evaluate_quantized(model, params, master, states, dev_loader, device, args.eval_batches)
    append_jsonl(run_dir / "eval_metrics.jsonl", {"step": d_h_finite, "eval_loss": eval_loss, "eval_acc": eval_acc})
    elapsed = time.time() - start
    steps_completed = sum(1 for _ in metrics_path.open(encoding="utf-8")) - 1
    summary = {
        **run_config,
        **quant_agg,
        "steps_completed": steps_completed,
        "status": status,
        "error_message": error_message,
        "seconds_per_step": elapsed / max(steps_completed, 1),
        "peak_gpu_mem_mb": float(torch.cuda.max_memory_allocated() / 1024 / 1024) if torch.cuda.is_available() else 0.0,
        "final_train_loss": final_train_loss,
        "final_eval_loss": eval_loss,
        "final_eval_acc": eval_acc,
        "d_h_finite_rate": d_h_finite / max(steps_completed, 1),
        "update_norm_last": update_norm_last,
        **last_pert,
    }
    write_json(run_dir / "run_summary.json", summary)
    return summary


def unit_tests() -> Dict[str, object]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gen = torch.Generator(device=device).manual_seed(123)
    w = torch.randn(8, 33, device=device, generator=gen) * 0.2
    x = torch.randn(64, 33, device=device, generator=gen)
    H = x.t().matmul(x) / x.shape[0]
    state, stats = hessian_gptq_reconstruct("unit.linear.weight", w, H, bits=4, group_size=16, damp_percent=0.01, mode="full")
    direction = torch.randn(w.shape, device=device, generator=gen)
    plus, ps = quantize_with_gptq_state(w + 1e-3 * direction, state, True)
    minus, ms = quantize_with_gptq_state(w - 1e-3 * direction, state, True)
    assert torch.isfinite(plus).all() and torch.isfinite(minus).all()
    assert ps["code_min"] >= state.qmin and ps["code_max"] <= state.qmax
    assert ms["code_min"] >= state.qmin and ms["code_max"] <= state.qmax
    assert state.scales.data_ptr() == state.scales.data_ptr()
    assert bool(((plus - minus) != 0).any())
    return {
        "status": "pass",
        "weight_recon_mse": stats["weight_recon_mse"],
        "weight_recon_rel_mse": stats["weight_recon_rel_mse"],
        "code_range": [state.qmin, state.qmax],
        "pair_shared_grid_check": True,
        "fresh_perturbed_code_check": True,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", default="roberta-large")
    parser.add_argument("--task_name", default="sst-5")
    parser.add_argument("--dataset_mode", default="full")
    parser.add_argument("--data_dir", default=None)
    parser.add_argument("--num_k", type=int, default=16)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--run_name", default="roberta_sst5_int4_gptqinit_lastmlp_smoke")
    parser.add_argument("--seed", type=int, default=16)
    parser.add_argument("--data_seed", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--eval_batch_size", type=int, default=8)
    parser.add_argument("--eval_batches", type=int, default=8)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--h", type=float, default=1e-3)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--group_size", type=int, default=128)
    parser.add_argument("--damp_percent", type=float, default=0.01)
    parser.add_argument("--gptq_mode", choices=["full", "diag", "rtn"], default="full")
    parser.add_argument("--init_quantizer", choices=["gptq_asym", "rtn_symmetric"], default="gptq_asym")
    parser.add_argument("--module_scope", choices=["last_mlp", "last_block_all_linear", "classifier", "all_linear"], default="last_mlp")
    parser.add_argument("--train_scope", choices=["quantized_linear_weights_only", "all_float_params"], default="quantized_linear_weights_only")
    parser.add_argument("--max_modules", type=int, default=0)
    parser.add_argument("--calib_batches", type=int, default=2)
    parser.add_argument("--max_rows_per_module_per_batch", type=int, default=4096)
    parser.add_argument("--unit_only", action="store_true")
    args = parser.parse_args()
    args.repo_root = REPO_ROOT
    return args


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    unit = unit_tests()
    write_json(args.output_dir / "unit_tests.json", unit)
    if args.unit_only:
        print(json.dumps(unit, indent=2))
        return 0
    summary = run(args)
    print(f"GPTQ-init smoke output directory: {args.output_dir / args.run_name}")
    print(
        "status={status} steps={steps_completed} eval_acc={final_eval_acc} "
        "loss={final_train_loss} active={active_frac} align={alignment} norm_ratio={norm_ratio}".format(**summary)
    )
    return 0 if summary.get("status") == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
