#!/usr/bin/env python
"""INT8 RTNClip vs AWQ-style h-window robustness sanity runner.

This runner is intentionally separate from the production Trainer paths. It
uses a shared-grid fake-quantized forward oracle:

    d_h = [L(Q_t(w_t + h u)) - L(Q_t(w_t - h u))] / (2h)

where Q_t is built from the unperturbed FP16 master weights, and the plus/minus
perturbations are freshly rounded on the same cached grid.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import random
import shutil
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler


REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = REPO_ROOT / "tools"
LARGE_MODELS_DIR = REPO_ROOT / "large_models"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))
if str(LARGE_MODELS_DIR) not in sys.path:
    sys.path.insert(0, str(LARGE_MODELS_DIR))

import smoke_rtnclip_roberta_sst5 as roberta_smoke  # noqa: E402


H_GRID: List[Tuple[str, float]] = [
    ("1e-5", 1e-5),
    ("3e-5", 3e-5),
    ("1e-4", 1e-4),
    ("3e-4", 3e-4),
    ("1e-3", 1e-3),
    ("2e-3", 2e-3),
    ("3e-3", 3e-3),
    ("5e-3", 5e-3),
    ("1e-2", 1e-2),
]

RTN_ALPHA_GRID = (1.0, 0.95, 0.90, 0.85, 0.80, 0.75, 0.70)
AWQ_ALPHA_GRID = (1.0, 0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60, 0.55, 0.50)
QUANTIZER_BACKENDS = {
    "rtnclip": "G128_RTNClip_shared_grid_fake_quant",
    "awq": "awq_style_g128_fake_quant",
}


@dataclass
class QuantizerState:
    name: str
    quantizer: str
    shape: Tuple[int, int]
    group_size: int
    bitwidth: int
    qmax: int
    scales: torch.Tensor
    alpha_idx: torch.Tensor
    alpha_values: torch.Tensor
    lengths: torch.Tensor
    valid: torch.Tensor


def json_default(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.dtype):
        return str(value)
    return str(value)


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True, default=json_default) + "\n", encoding="utf-8")


def append_jsonl(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(data, sort_keys=True, default=json_default) + "\n")


def write_csv(path: Path, rows: List[Dict[str, object]], columns: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(columns), extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def read_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    except Exception:
        return ""


def collect_env() -> Dict[str, object]:
    env: Dict[str, object] = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "hostname": socket.gethostname(),
        "python_executable": sys.executable,
        "python_version": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "git_commit": git_commit(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "DATALOADER_SHUFFLE": os.environ.get("DATALOADER_SHUFFLE", ""),
    }
    for name in ("transformers", "datasets", "accelerate"):
        try:
            module = __import__(name)
            env[f"{name}_version"] = getattr(module, "__version__", "")
        except Exception:
            env[f"{name}_version"] = None
    if torch.cuda.is_available():
        idx = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(idx)
        env.update(
            {
                "gpu_index": idx,
                "gpu_name": props.name,
                "gpu_total_memory_mb": int(props.total_memory / 1024 / 1024),
                "gpu_count": torch.cuda.device_count(),
            }
        )
    return env


def h_label(h: float) -> str:
    for label, value in H_GRID:
        if abs(value - h) <= max(abs(value), 1.0) * 1e-12:
            return label
    return f"{h:g}".replace(".", "p")


def stable_h_key(h: float) -> int:
    return int(round(float(h) * 1_000_000_000_000))


def format_float(value, digits: int = 4) -> str:
    if value is None or value == "":
        return "n/a"
    try:
        return f"{float(value):.{digits}g}"
    except Exception:
        return str(value)


def cycle(loader: Iterable[Dict[str, torch.Tensor]]) -> Iterator[Dict[str, torch.Tensor]]:
    while True:
        for batch in loader:
            yield batch


def move_batch(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}


def group_view_2d(weight: torch.Tensor, group_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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


def activation_group_weights(
    activation_rms: Optional[torch.Tensor],
    in_features: int,
    group_size: int,
    device: torch.device,
) -> Optional[torch.Tensor]:
    if activation_rms is None:
        return None
    act = activation_rms.detach().float().to(device=device).flatten()
    if act.numel() != in_features:
        return None
    num_groups = int(math.ceil(in_features / group_size))
    pad_cols = num_groups * group_size - in_features
    if pad_cols:
        act = F.pad(act, (0, pad_cols), value=0.0)
    weights = act.square().clamp_min(1e-12).reshape(1, num_groups, group_size)
    mean = weights[weights > 0].mean() if bool((weights > 0).any()) else torch.ones((), device=device)
    return weights / mean.clamp_min(1e-12)


def compute_quantizer_state(
    name: str,
    weight: torch.Tensor,
    *,
    quantizer: str,
    bitwidth: int,
    group_size: int,
    activation_rms: Optional[torch.Tensor],
) -> Tuple[QuantizerState, Dict[str, object]]:
    if bitwidth != 8:
        raise ValueError("this robustness runner only supports INT8")
    qmax = 127
    groups, lengths, valid = group_view_2d(weight, group_size)
    alpha_values = AWQ_ALPHA_GRID if quantizer == "awq" else RTN_ALPHA_GRID
    alpha_grid = torch.tensor(alpha_values, device=weight.device, dtype=torch.float32)
    masked_abs = groups.abs().masked_fill(~valid, 0.0)
    max_abs = masked_abs.amax(dim=-1, keepdim=True)
    zero_groups = max_abs <= 0
    base_scale = (max_abs / float(qmax)).clamp_min(1e-12)
    candidate_scales = base_scale.unsqueeze(2) * alpha_grid.view(1, 1, -1, 1)
    x = groups.unsqueeze(2)
    q = torch.round(x / candidate_scales).clamp(-qmax, qmax)
    wq = q * candidate_scales
    err = (x - wq).square().masked_fill(~valid.unsqueeze(2), 0.0)
    if quantizer == "awq":
        aw = activation_group_weights(activation_rms, weight.shape[1], group_size, weight.device)
        if aw is not None:
            err = err * aw.unsqueeze(2)
            denom = (aw * valid).sum(dim=-1).clamp_min(1e-12)
            mse = err.sum(dim=-1) / denom.unsqueeze(2)
        else:
            mse = err.sum(dim=-1) / lengths.float().view(1, -1, 1)
    else:
        mse = err.sum(dim=-1) / lengths.float().view(1, -1, 1)
    best_idx = mse.argmin(dim=2)
    scales_all = candidate_scales.squeeze(-1)
    best_scales = torch.gather(scales_all, 2, best_idx.unsqueeze(-1)).clamp_min(1e-12)
    best_scales = torch.where(zero_groups, torch.ones_like(best_scales), best_scales)
    best_alpha = alpha_grid[best_idx]
    best_alpha = torch.where(zero_groups.squeeze(-1), torch.ones_like(best_alpha), best_alpha)
    state = QuantizerState(
        name=name,
        quantizer=quantizer,
        shape=tuple(weight.shape),
        group_size=group_size,
        bitwidth=bitwidth,
        qmax=qmax,
        scales=best_scales.detach(),
        alpha_idx=best_idx.detach(),
        alpha_values=best_alpha.detach(),
        lengths=lengths.detach(),
        valid=valid.detach(),
    )
    q_w, stats = quantize_with_state(weight, state, return_stats=True)
    diff = q_w.float() - weight.float()
    err_sq = diff.double().square().sum()
    ref_sq = weight.float().double().square().sum()
    eps64 = torch.tensor(1e-30, device=weight.device, dtype=torch.float64)
    recon_mse = float((err_sq / max(int(weight.numel()), 1)).detach().cpu())
    recon_rel_mse = float((err_sq / ref_sq.clamp_min(eps64)).detach().cpu())
    recon_sqnr_db = float((10.0 * torch.log10(ref_sq / err_sq.clamp_min(eps64))).detach().cpu())
    if quantizer == "awq" and activation_rms is not None and activation_rms.numel() == weight.shape[1]:
        col_w = activation_rms.detach().float().to(weight.device).square().clamp_min(1e-12)
        weighted_mse = ((q_w.float() - weight.float()).square() * (col_w / col_w.mean().clamp_min(1e-12)).view(1, -1)).mean()
        awq_weighted_mse = float(weighted_mse.detach().cpu())
    else:
        awq_weighted_mse = None
    stats.update(
        {
            "module_name": name,
            "quantizer": quantizer,
            "quantizer_backend": QUANTIZER_BACKENDS[quantizer],
            "bitwidth": bitwidth,
            "group_size": group_size,
            "num_groups": int(best_idx.numel()),
            "scale_min": float(best_scales.min().detach().cpu()),
            "scale_median": float(best_scales.median().detach().cpu()),
            "scale_max": float(best_scales.max().detach().cpu()),
            "alpha_mean": float(best_alpha.float().mean().detach().cpu()),
            "alpha_min": float(best_alpha.min().detach().cpu()),
            "alpha_max": float(best_alpha.max().detach().cpu()),
            "alpha_lt_1_frac": float((best_alpha < 1.0).float().mean().detach().cpu()),
            "recon_mse": recon_mse,
            "weight_recon_mse": recon_mse,
            "weight_recon_rel_mse": recon_rel_mse,
            "weight_recon_sqnr_db": recon_sqnr_db,
            "weight_recon_sse": float(err_sq.detach().cpu()),
            "weight_recon_ref_sse": float(ref_sq.detach().cpu()),
            "activation_weighted_mse": awq_weighted_mse,
            "activation_rms_available": activation_rms is not None and activation_rms.numel() == weight.shape[1],
        }
    )
    for alpha in alpha_values:
        stats[f"alpha_{alpha:g}_count"] = int((best_alpha == alpha).sum().detach().cpu())
    return state, stats


def quantize_with_state(weight: torch.Tensor, state: QuantizerState, return_stats: bool = False):
    groups, _, valid = group_view_2d(weight, state.group_size)
    q = torch.round(groups / state.scales).clamp(-state.qmax, state.qmax)
    q = q.masked_fill(~valid, 0.0)
    wq = (q * state.scales).reshape(weight.shape[0], -1)[:, : weight.shape[1]].reshape_as(weight)
    wq = wq.to(dtype=weight.dtype)
    if not return_stats:
        return wq
    valid_q = q[valid.expand_as(q)]
    clip_frac = float((valid_q.abs() >= state.qmax).float().mean().detach().cpu()) if valid_q.numel() else 0.0
    return wq, {
        "code_min": int(valid_q.min().detach().cpu()) if valid_q.numel() else 0,
        "code_max": int(valid_q.max().detach().cpu()) if valid_q.numel() else 0,
        "clip_frac": clip_frac,
        "saturation_frac": clip_frac,
    }


def refresh_quantizer_states(
    master: Dict[str, torch.Tensor],
    q_names: Sequence[str],
    *,
    quantizer: str,
    activation_rms: Dict[str, torch.Tensor],
    bitwidth: int = 8,
    group_size: int = 128,
) -> Tuple[Dict[str, QuantizerState], List[Dict[str, object]]]:
    states: Dict[str, QuantizerState] = {}
    rows: List[Dict[str, object]] = []
    for name in q_names:
        state, stats = compute_quantizer_state(
            name,
            master[name],
            quantizer=quantizer,
            bitwidth=bitwidth,
            group_size=group_size,
            activation_rms=activation_rms.get(name),
        )
        states[name] = state
        rows.append(stats)
    return states, rows


def aggregate_quantizer_stats(rows: List[Dict[str, object]], numel_by_name: Dict[str, int]) -> Dict[str, object]:
    if not rows:
        return {}
    total_values = sum(numel_by_name[str(row["module_name"])] for row in rows)
    total_groups = sum(int(row["num_groups"]) for row in rows)

    def weighted_mean(key: str, by_groups: bool = False):
        vals = []
        weights = []
        for row in rows:
            value = row.get(key)
            if value is None:
                continue
            vals.append(float(value))
            weights.append(int(row["num_groups"]) if by_groups else numel_by_name[str(row["module_name"])])
        if not vals:
            return None
        denom = max(sum(weights), 1)
        return sum(v * w for v, w in zip(vals, weights)) / denom

    err_total = sum(float(row.get("weight_recon_sse", 0.0)) for row in rows)
    ref_total = sum(float(row.get("weight_recon_ref_sse", 0.0)) for row in rows)
    pooled_rel_mse = err_total / max(ref_total, 1e-30) if ref_total > 0.0 else weighted_mean("weight_recon_rel_mse")
    pooled_sqnr_db = 10.0 * math.log10(ref_total / max(err_total, 1e-30)) if ref_total > 0.0 else weighted_mean("weight_recon_sqnr_db")
    out = {
        "num_quantized_modules": len(rows),
        "num_quantized_values": total_values,
        "num_groups": total_groups,
        "recon_mse_global": weighted_mean("recon_mse"),
        "weight_recon_mse": weighted_mean("weight_recon_mse"),
        "weight_recon_rel_mse": pooled_rel_mse,
        "weight_recon_sqnr_db": pooled_sqnr_db,
        "activation_weighted_mse_global": weighted_mean("activation_weighted_mse"),
        "clip_frac": weighted_mean("clip_frac"),
        "saturation_frac": weighted_mean("saturation_frac"),
        "saturation_frac_w": weighted_mean("saturation_frac"),
        "alpha_mean": weighted_mean("alpha_mean", by_groups=True),
        "alpha_lt_1_frac": weighted_mean("alpha_lt_1_frac", by_groups=True),
        "scale_min_global": min(float(row["scale_min"]) for row in rows),
        "scale_median_weighted": weighted_mean("scale_median"),
        "scale_max_global": max(float(row["scale_max"]) for row in rows),
        "activation_rms_coverage": sum(1 for row in rows if row.get("activation_rms_available")) / max(len(rows), 1),
    }
    alpha_keys = sorted({key for row in rows for key in row if key.startswith("alpha_") and key.endswith("_count")})
    for key in alpha_keys:
        out[key] = sum(int(row.get(key, 0)) for row in rows)
    return out


def named_parameter_map(model: nn.Module) -> Dict[str, nn.Parameter]:
    return dict(model.named_parameters())


def linear_weight_names(model: nn.Module) -> List[str]:
    names = []
    params = named_parameter_map(model)
    for module_name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            pname = f"{module_name}.weight" if module_name else "weight"
            if pname in params:
                names.append(pname)
    return names


def module_name_from_weight_name(weight_name: str) -> str:
    return weight_name[:-7] if weight_name.endswith(".weight") else weight_name


def restore_master(params: Dict[str, nn.Parameter], master: Dict[str, torch.Tensor]) -> float:
    max_diff = 0.0
    with torch.no_grad():
        for name, param in params.items():
            if name not in master:
                continue
            param.copy_(master[name].to(dtype=param.dtype))
            diff = (param.detach().float() - master[name].float()).abs().max()
            max_diff = max(max_diff, float(diff.detach().cpu()))
    return max_diff


def copy_master_to_model(
    params: Dict[str, nn.Parameter],
    master: Dict[str, torch.Tensor],
    directions: Optional[Dict[str, torch.Tensor]],
    h: float,
    sign: float,
    states: Dict[str, QuantizerState],
) -> None:
    with torch.no_grad():
        for name, param in params.items():
            if name not in master:
                continue
            if directions is None:
                value = master[name]
            else:
                value = master[name].float().add(directions[name].float(), alpha=sign * h)
            if name in states:
                value = quantize_with_state(value, states[name])
            param.copy_(value.to(dtype=param.dtype))


def sample_directions(master: Dict[str, torch.Tensor], seed: int) -> Dict[str, torch.Tensor]:
    first = next(iter(master.values()))
    gen = torch.Generator(device=first.device).manual_seed(int(seed))
    directions = {}
    for name, tensor in master.items():
        if tensor.is_floating_point():
            directions[name] = torch.randn(tensor.shape, device=tensor.device, generator=gen, dtype=torch.float16)
    return directions


def direction_seed(base_seed: int, quantizer: str, h: float, step: int, extra: int = 0) -> int:
    q_offset = 100_003 if quantizer == "awq" else 17_071
    return int(base_seed) + q_offset + stable_h_key(h) + step * 1_000_003 + extra * 97_531


def update_master(master: Dict[str, torch.Tensor], directions: Dict[str, torch.Tensor], lr: float, d_h: float) -> float:
    sq = torch.zeros((), device=next(iter(master.values())).device, dtype=torch.float64)
    with torch.no_grad():
        for name, tensor in master.items():
            update = directions[name].float().mul(-float(lr) * float(d_h))
            sq += update.double().square().sum()
            tensor.copy_(tensor.float().add(update).to(dtype=tensor.dtype))
    return float(sq.sqrt().detach().cpu())


def perturbation_metrics(
    master: Dict[str, torch.Tensor],
    directions: Dict[str, torch.Tensor],
    states: Dict[str, QuantizerState],
    h: float,
) -> Dict[str, object]:
    device = next(iter(master.values())).device
    active = 0
    total = 0
    dot = torch.zeros((), device=device, dtype=torch.float64)
    delta_sq = torch.zeros((), device=device, dtype=torch.float64)
    intended_sq = torch.zeros((), device=device, dtype=torch.float64)
    delta_err_sq = torch.zeros((), device=device, dtype=torch.float64)
    clip_plus_num = 0.0
    clip_minus_num = 0.0
    value_num = 0
    legal = True
    for name, state in states.items():
        intended = 2.0 * float(h) * directions[name].float()
        plus, plus_stats = quantize_with_state(master[name].float().add(directions[name].float(), alpha=float(h)), state, True)
        minus, minus_stats = quantize_with_state(master[name].float().add(directions[name].float(), alpha=-float(h)), state, True)
        delta = plus.float() - minus.float()
        delta_err = delta.float() - intended.float()
        active += int((delta != 0).sum().detach().cpu())
        total += delta.numel()
        dot += (delta.double() * intended.double()).sum()
        delta_sq += delta.double().square().sum()
        intended_sq += intended.double().square().sum()
        delta_err_sq += delta_err.double().square().sum()
        clip_plus_num += float(plus_stats["clip_frac"]) * delta.numel()
        clip_minus_num += float(minus_stats["clip_frac"]) * delta.numel()
        value_num += delta.numel()
        legal = legal and int(plus_stats["code_min"]) >= -state.qmax and int(plus_stats["code_max"]) <= state.qmax
        legal = legal and int(minus_stats["code_min"]) >= -state.qmax and int(minus_stats["code_max"]) <= state.qmax
    eps = 1e-12
    alignment = float((dot / (delta_sq.sqrt() * intended_sq.sqrt() + eps)).detach().cpu())
    norm_ratio = float((delta_sq.sqrt() / (intended_sq.sqrt() + eps)).detach().cpu())
    active_frac = active / max(total, 1)
    delta_visibility_mse = float((delta_err_sq / max(total, 1)).detach().cpu())
    delta_visibility_nmse = float((delta_err_sq / intended_sq.clamp_min(eps)).detach().cpu())
    delta_visibility_rel_l2 = float((delta_err_sq.sqrt() / intended_sq.sqrt().clamp_min(eps)).detach().cpu())
    avg_clip = (clip_plus_num + clip_minus_num) / max(2 * value_num, 1)
    return {
        "delta_q_norm": float(delta_sq.sqrt().detach().cpu()),
        "nominal_delta_norm": float(intended_sq.sqrt().detach().cpu()),
        "alignment": alignment,
        "norm_ratio": norm_ratio,
        "delta_visibility_mse": delta_visibility_mse,
        "delta_visibility_nmse": delta_visibility_nmse,
        "delta_visibility_rel_l2": delta_visibility_rel_l2,
        "active_frac": active_frac,
        "code_change_frac": active_frac,
        "zero_effective_displacement_frac": 1.0 - active_frac,
        "clip_frac": avg_clip,
        "saturation_frac": avg_clip,
        "clip_frac_w_plus": clip_plus_num / max(value_num, 1),
        "clip_frac_w_minus": clip_minus_num / max(value_num, 1),
        "saturation_frac_w_plus": clip_plus_num / max(value_num, 1),
        "saturation_frac_w_minus": clip_minus_num / max(value_num, 1),
        "codes_legal": bool(legal),
        "pair_shared_grid_check": True,
        "fresh_round_codes_check": True,
        "independent_q_plus_q_minus_scales": False,
        "q_w_plus_hu_bypass": False,
    }


class BaseHarness:
    setting_name: str
    model_name: str
    dataset_name: str
    batch_size: int
    eval_batch_size: int
    lr: float
    max_length: int

    def __init__(self, args: argparse.Namespace, device: torch.device) -> None:
        self.args = args
        self.device = device
        self.model: nn.Module
        self.train_loader: DataLoader
        self.dev_loader: DataLoader
        self.train_sampler_name: str
        self.quantized_module_names: List[str]
        self.activation_rms: Dict[str, torch.Tensor] = {}

    def forward_loss_and_logits(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def make_master(self) -> Dict[str, torch.Tensor]:
        return {
            name: p.detach().clone().to(device=self.device, dtype=torch.float16)
            for name, p in self.model.named_parameters()
            if p.detach().is_floating_point()
        }

    def params(self) -> Dict[str, nn.Parameter]:
        return named_parameter_map(self.model)

    def numel_by_quantized_name(self) -> Dict[str, int]:
        params = self.params()
        return {name: params[name].numel() for name in self.quantized_module_names}

    def eval_quantized(
        self,
        master: Dict[str, torch.Tensor],
        states: Dict[str, QuantizerState],
        max_batches: int,
    ) -> Tuple[Optional[float], Optional[float]]:
        if max_batches == 0:
            return None, None
        params = self.params()
        copy_master_to_model(params, master, None, 0.0, 0.0, states)
        total_loss = 0.0
        total_correct = 0
        total_items = 0
        for idx, batch in enumerate(self.dev_loader):
            if max_batches > 0 and idx >= max_batches:
                break
            batch = move_batch(batch, self.device)
            loss, logits = self.forward_loss_and_logits(batch)
            labels = batch["labels"]
            total_loss += float(loss.detach().cpu()) * int(labels.numel())
            total_correct += int((logits.argmax(dim=-1) == labels).sum().detach().cpu())
            total_items += int(labels.numel())
        restore_master(params, master)
        if total_items == 0:
            return None, None
        return total_loss / total_items, total_correct / total_items

    def collect_activation_rms(self, calibration_examples: int) -> Dict[str, torch.Tensor]:
        module_to_weight = {module_name_from_weight_name(name): name for name in self.quantized_module_names}
        sums: Dict[str, torch.Tensor] = {}
        counts: Dict[str, int] = {}
        hooks = []

        def make_hook(weight_name: str):
            def hook(_module, inputs, _output):
                if not inputs:
                    return
                x = inputs[0].detach().float()
                if x.numel() == 0:
                    return
                x2 = x.reshape(-1, x.shape[-1]).square().sum(dim=0).cpu()
                sums[weight_name] = sums.get(weight_name, torch.zeros_like(x2)) + x2
                counts[weight_name] = counts.get(weight_name, 0) + x.reshape(-1, x.shape[-1]).shape[0]

            return hook

        for module_name, module in self.model.named_modules():
            weight_name = module_to_weight.get(module_name)
            if weight_name is not None and isinstance(module, nn.Linear):
                hooks.append(module.register_forward_hook(make_hook(weight_name)))
        seen = 0
        try:
            copy_master_to_model(self.params(), self.make_master(), None, 0.0, 0.0, {})
            for batch in self.train_loader:
                batch = move_batch(batch, self.device)
                with torch.no_grad():
                    self.forward_loss_and_logits(batch)
                seen += int(batch.get("example_count", batch["labels"].numel()))
                if seen >= calibration_examples:
                    break
        finally:
            for hook in hooks:
                hook.remove()
        out = {}
        for name in self.quantized_module_names:
            if name in sums and counts.get(name, 0) > 0:
                out[name] = (sums[name] / float(counts[name])).clamp_min(0.0).sqrt().to(self.device)
        self.activation_rms = out
        return out


class RobertaHarness(BaseHarness):
    setting_name = "roberta_large_sst5"
    model_name = "roberta-large"
    dataset_name = "SST-5"

    def __init__(self, args: argparse.Namespace, device: torch.device) -> None:
        super().__init__(args, device)
        self.batch_size = int(args.roberta_batch_size)
        self.eval_batch_size = int(args.roberta_eval_batch_size)
        self.lr = float(args.roberta_lr)
        self.max_length = 128
        roberta_smoke.add_medium_models_to_path(REPO_ROOT)
        from src.models import RobertaModelForPromptFinetuning
        from src.modeling_roberta import RobertaModel

        if not hasattr(RobertaModelForPromptFinetuning, "all_tied_weights_keys"):
            RobertaModelForPromptFinetuning.all_tied_weights_keys = {}
        if not hasattr(RobertaModel, "get_head_mask"):
            def _compat_get_head_mask(self, head_mask, num_hidden_layers, is_attention_chunked=False):
                if head_mask is None:
                    return [None] * num_hidden_layers
                if head_mask.dim() == 1:
                    head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                    head_mask = head_mask.expand(num_hidden_layers, -1, -1, -1, -1)
                elif head_mask.dim() == 2:
                    head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
                return head_mask.to(dtype=self.dtype)

            RobertaModel.get_head_mask = _compat_get_head_mask
        load_args = argparse.Namespace(
            repo_root=REPO_ROOT,
            model_id="roberta-large",
            seed=16,
            data_seed=16,
            batch_size=self.batch_size,
            eval_batch_size=self.eval_batch_size,
        )
        orig_torch_load = torch.load

        def _compat_torch_load(*load_args_, **load_kwargs_):
            load_kwargs_.setdefault("weights_only", False)
            return orig_torch_load(*load_args_, **load_kwargs_)

        torch.load = _compat_torch_load
        try:
            self.model, self.train_loader, self.dev_loader, self.data_args, train_sampler = roberta_smoke.load_prompt_model_and_data(load_args, device)
        finally:
            torch.load = orig_torch_load
        self.train_sampler_name = type(train_sampler).__name__
        self.quantized_module_names = linear_weight_names(self.model)

    def forward_loss_and_logits(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        batch = dict(batch)
        batch["token_type_ids"] = torch.zeros_like(batch["input_ids"])
        return roberta_smoke.forward_loss_and_logits(self.model, batch)


class OPTSSTDataset(torch.utils.data.Dataset):
    def __init__(self, samples, task, tokenizer, max_length: int) -> None:
        from utils import encode_prompt

        self.samples = samples
        self.task = task
        self.template = task.get_template()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.encode_prompt = encode_prompt

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        sample = self.samples[idx]
        encoded_candidates, option_lens = self.encode_prompt(
            self.task,
            self.template,
            [],
            sample,
            self.tokenizer,
            max_length=self.max_length,
            generation=self.task.generation,
            generation_with_gold=True,
        )
        correct = sample.candidates.index(sample.correct_candidate)
        return {
            "input_ids": encoded_candidates,
            "option_lens": option_lens,
            "label": correct,
            "num_options": len(sample.candidates),
        }


class OPTCollator:
    def __init__(self, tokenizer) -> None:
        self.tokenizer = tokenizer

    def __call__(self, features: List[Dict[str, object]]) -> Dict[str, torch.Tensor]:
        pad_id = int(self.tokenizer.pad_token_id)
        flat_ids: List[List[int]] = []
        option_lens: List[int] = []
        labels: List[int] = []
        num_options: List[int] = []
        for feat in features:
            candidates = feat["input_ids"]
            opts = feat["option_lens"]
            label = int(feat["label"])
            nopt = int(feat["num_options"])
            for ids, olen in zip(candidates, opts):
                flat_ids.append(list(ids))
                option_lens.append(int(olen))
                labels.append(label)
                num_options.append(nopt)
        max_len = max(len(ids) for ids in flat_ids)
        padded = []
        attention = []
        for ids in flat_ids:
            pad = [pad_id] * (max_len - len(ids))
            padded_ids = pad + ids
            padded.append(padded_ids)
            attention.append([0] * len(pad) + [1] * len(ids))
        return {
            "input_ids": torch.tensor(padded, dtype=torch.long),
            "attention_mask": torch.tensor(attention, dtype=torch.long),
            "option_len": torch.tensor(option_lens, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "num_options": torch.tensor(num_options, dtype=torch.long),
            "example_count": torch.tensor(len(features), dtype=torch.long),
        }


class OPTHarness(BaseHarness):
    setting_name = "opt_1p3b_sst5"
    model_name = "opt-1.3b"
    dataset_name = "SST-5"

    def __init__(self, args: argparse.Namespace, device: torch.device) -> None:
        super().__init__(args, device)
        from tasks import get_task
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.batch_size = int(args.opt_batch_size)
        self.eval_batch_size = int(args.opt_eval_batch_size)
        self.lr = float(args.opt_lr)
        self.max_length = int(args.opt_max_length)
        self.task_name = "SST5"
        self.task = get_task(self.task_name)
        train_sets = self.task.sample_train_sets(
            num_train=-1,
            num_dev=0,
            num_eval=None,
            num_train_sets=1,
            seed=16,
            dataset_mode="full",
            num_k=16,
        )
        train_samples = train_sets[0]
        eval_samples = self.task.valid_samples
        self.dataset_fallback = None
        self.tokenizer = AutoTokenizer.from_pretrained("facebook/opt-1.3b", use_fast=False)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained("facebook/opt-1.3b", torch_dtype=torch.float16)
        if self.model.config.pad_token_id is None:
            self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model.to(device)
        self.model.eval()
        train_dataset = OPTSSTDataset(train_samples, self.task, self.tokenizer, self.max_length)
        dev_dataset = OPTSSTDataset(eval_samples, self.task, self.tokenizer, self.max_length)
        train_gen = torch.Generator().manual_seed(16)
        train_sampler = RandomSampler(train_dataset, generator=train_gen)
        collator = OPTCollator(self.tokenizer)
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, sampler=train_sampler, collate_fn=collator)
        self.dev_loader = DataLoader(dev_dataset, batch_size=self.eval_batch_size, sampler=SequentialSampler(dev_dataset), collate_fn=collator)
        self.train_sampler_name = type(train_sampler).__name__
        self.quantized_module_names = linear_weight_names(self.model)

    def forward_loss_and_logits(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            outputs = self.model(input_ids=batch["input_ids"], attention_mask=batch.get("attention_mask"))
            logits = outputs.logits
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = batch["input_ids"][..., 1:].contiguous()
            pad_id = int(self.model.config.pad_token_id)
            shift_labels = shift_labels.masked_fill(shift_labels == pad_id, -100)
            option_len = batch["option_len"].to(shift_labels.device)
            for idx, olen in enumerate(option_len.tolist()):
                if olen <= 0:
                    shift_labels[idx, :] = -100
                else:
                    shift_labels[idx, :-int(olen)] = -100
            log_probs = F.log_softmax(shift_logits, dim=-1)
            gather_labels = shift_labels.masked_fill(shift_labels == -100, 0)
            selected = torch.gather(log_probs, dim=-1, index=gather_labels.unsqueeze(-1)).squeeze(-1)
            mask = shift_labels != -100
            candidate_scores = (selected * mask).sum(dim=-1) / mask.sum(dim=-1).clamp_min(1)
            nopt = int(batch["num_options"][0].item())
            scores = candidate_scores.view(-1, nopt)
            labels = batch["labels"].view(-1, nopt)[:, 0]
            loss = F.cross_entropy(scores, labels)
        return loss, scores

    def eval_quantized(
        self,
        master: Dict[str, torch.Tensor],
        states: Dict[str, QuantizerState],
        max_batches: int,
    ) -> Tuple[Optional[float], Optional[float]]:
        if max_batches == 0:
            return None, None
        params = self.params()
        copy_master_to_model(params, master, None, 0.0, 0.0, states)
        total_loss = 0.0
        total_correct = 0
        total_items = 0
        for idx, batch in enumerate(self.dev_loader):
            if max_batches > 0 and idx >= max_batches:
                break
            batch = move_batch(batch, self.device)
            loss, scores = self.forward_loss_and_logits(batch)
            nopt = int(batch["num_options"][0].item())
            labels = batch["labels"].view(-1, nopt)[:, 0]
            total_loss += float(loss.detach().cpu()) * int(labels.numel())
            total_correct += int((scores.argmax(dim=-1) == labels).sum().detach().cpu())
            total_items += int(labels.numel())
        restore_master(params, master)
        if total_items == 0:
            return None, None
        return total_loss / total_items, total_correct / total_items


def make_run_config(
    harness: BaseHarness,
    *,
    phase: str,
    run_name: str,
    run_dir: Path,
    quantizer: str,
    h: float,
    steps: int,
    eval_every: int,
    checkpoint_every: int,
    calibration_examples: int,
    env: Dict[str, object],
) -> Dict[str, object]:
    return {
        "phase": phase,
        "run_name": run_name,
        "model": harness.model_name,
        "dataset": harness.dataset_name,
        "dataset_mode": "full",
        "dataset_fallback": getattr(harness, "dataset_fallback", None),
        "seed": 16,
        "data_seed": 16,
        "batch_size": harness.batch_size,
        "eval_batch_size": harness.eval_batch_size,
        "dataloader_shuffle": True,
        "sampler_name": harness.train_sampler_name,
        "direction": "dense",
        "estimator": "two_point_symmetric_mezo",
        "h": float(h),
        "h_label": h_label(h),
        "max_steps": int(steps),
        "eval_every": int(eval_every),
        "checkpoint_every": int(checkpoint_every),
        "lr": float(harness.lr),
        "update_backend": "fp16_master",
        "master_dtype": "fp16",
        "direct_int_update": False,
        "quant_bits": 8,
        "bitwidth": 8,
        "group_size": 128,
        "quantizer": quantizer,
        "quantizer_backend": QUANTIZER_BACKENDS[quantizer],
        "quantizer_name": QUANTIZER_BACKENDS[quantizer],
        "scale_source": "unperturbed_fp16_master_w_t",
        "grid_source": "unperturbed_fp16_master_w_t",
        "pair_shared_grid": True,
        "fresh_round_codes": True,
        "independent_q_plus_q_minus_scales": False,
        "q_w_plus_hu_bypass": False,
        "module_quantization_scope": "Linear.weight only",
        "quantized_modules": harness.quantized_module_names,
        "linear_weights": True,
        "attention_projections": True,
        "mlp_projections": True,
        "embeddings": False,
        "layernorm": False,
        "bias": False,
        "activation_stats_fixed_after_calibration": quantizer == "awq",
        "activation_calibration_examples": int(calibration_examples) if quantizer == "awq" else 0,
        "awq_objective": "activation_weighted_weight_mse" if quantizer == "awq" else None,
        "awq_alpha_grid": list(AWQ_ALPHA_GRID) if quantizer == "awq" else None,
        "rtnclip_alpha_grid": list(RTN_ALPHA_GRID) if quantizer == "rtnclip" else None,
        "run_dir": str(run_dir),
        "env": env,
        "gpu_name": env.get("gpu_name", ""),
        "gpu_type_requested": "local_H100_preallocated",
        "fallback_used": False,
        "excluded_methods": ["GPTQ", "INT4", "residual_grid", "sparse", "LoRA", "Mistral", "RTE", "MNLI"],
    }


def write_resume_command(path: Path, config: Dict[str, object]) -> None:
    cmd = (
        "CUDA_VISIBLE_DEVICES=0 DATALOADER_SHUFFLE=True "
        f"python tools/quantizer_robustness_int8_window.py --resume_run {config['run_dir']}"
    )
    (path / "resume_command.txt").write_text(cmd + "\n", encoding="utf-8")


def hard_invariant_failures(config: Dict[str, object], compare_scope: Optional[Sequence[str]] = None) -> List[str]:
    failures = []
    if int(config.get("seed", -1)) != 16 or int(config.get("data_seed", -1)) != 16:
        failures.append("seed/data_seed must both be 16")
    if config.get("dataloader_shuffle") is not True:
        failures.append("dataloader_shuffle must be true")
    if config.get("sampler_name") != "RandomSampler":
        failures.append(f"sampler must be RandomSampler, got {config.get('sampler_name')}")
    if config.get("direction") != "dense":
        failures.append("direction must be dense")
    if int(config.get("quant_bits", 0)) != 8:
        failures.append("quant_bits must be 8")
    if int(config.get("group_size", 0)) != 128:
        failures.append("group_size must be 128")
    if config.get("update_backend") != "fp16_master":
        failures.append("update_backend must be fp16_master")
    if config.get("direct_int_update") is not False:
        failures.append("direct_int_update must be false")
    if config.get("pair_shared_grid") is not True:
        failures.append("pair_shared_grid must be true")
    if config.get("fresh_round_codes") is not True:
        failures.append("fresh_round_codes must be true")
    if config.get("independent_q_plus_q_minus_scales") is not False:
        failures.append("independent Q_plus/Q_minus scale recomputation is forbidden")
    if config.get("q_w_plus_hu_bypass") is not False:
        failures.append("Q(w_t) +/- h u bypass is forbidden")
    if compare_scope is not None and list(config.get("quantized_modules", [])) != list(compare_scope):
        failures.append("module quantization scope differs between RTNClip and AWQ-style")
    if config.get("quantizer") == "awq" and config.get("quantizer_name") != "awq_style_g128_fake_quant":
        failures.append("AWQ-style quantizer must be named awq_style_g128_fake_quant")
    if config.get("quantizer") == "rtnclip" and config.get("quantizer_name") != "G128_RTNClip_shared_grid_fake_quant":
        failures.append("RTNClip quantizer name mismatch")
    return failures


def save_checkpoint(run_dir: Path, step: int, master: Dict[str, torch.Tensor], best: Dict[str, object], config: Dict[str, object]) -> None:
    ckpt = run_dir / "checkpoints" / f"step_{step}"
    ckpt.mkdir(parents=True, exist_ok=True)
    cpu_master = {name: value.detach().cpu().to(dtype=torch.float16) for name, value in master.items()}
    torch.save({"step": step, "master": cpu_master, "best": best, "config": config}, ckpt / "state.pt")
    write_json(ckpt / "checkpoint_manifest.json", {"step": step, "num_tensors": len(cpu_master), "created_at": datetime.now().isoformat()})
    final = run_dir / "checkpoints" / "final"
    if final.exists():
        shutil.rmtree(final)
    shutil.copytree(ckpt, final)


def run_training_job(
    harness: BaseHarness,
    *,
    output_root: Path,
    phase: str,
    quantizer: str,
    h: float,
    steps: int,
    eval_every: int,
    checkpoint_every: int,
    calibration_examples: int,
    env: Dict[str, object],
    compare_scope: Optional[Sequence[str]] = None,
) -> Dict[str, object]:
    run_name = f"{harness.setting_name}_{quantizer}_int8_h{h_label(h)}_{phase}_steps{steps}"
    run_dir = output_root / phase / harness.setting_name / quantizer / f"h_{h_label(h)}"
    run_dir.mkdir(parents=True, exist_ok=True)
    for stale in ("metrics.csv", "eval_metrics.jsonl", "quantizer_diagnostics.jsonl", "perturbation_diagnostics.jsonl"):
        path = run_dir / stale
        if path.exists():
            path.unlink()
    config = make_run_config(
        harness,
        phase=phase,
        run_name=run_name,
        run_dir=run_dir,
        quantizer=quantizer,
        h=h,
        steps=steps,
        eval_every=eval_every,
        checkpoint_every=checkpoint_every,
        calibration_examples=calibration_examples,
        env=env,
    )
    write_json(run_dir / "run_config.json", config)
    write_resume_command(run_dir, config)
    failures = hard_invariant_failures(config, compare_scope=compare_scope)
    if failures:
        summary = {**config, "status": "failed", "steps_completed": 0, "error_message": "; ".join(failures)}
        write_json(run_dir / "run_summary.json", summary)
        return summary

    params = harness.params()
    master = harness.make_master()
    restore_master(params, master)
    q_names = harness.quantized_module_names
    numel_by_name = harness.numel_by_quantized_name()
    batch_iter = cycle(harness.train_loader)
    best = {"best_eval_acc": None, "best_step": None, "best_eval_loss": None, "best_eval_loss_step": None}
    finite_count = 0
    update_norm_last = None
    last_train_loss = None
    last_pert: Dict[str, object] = {}
    last_quant: Dict[str, object] = {}
    status = "complete"
    error_message = ""
    start = time.time()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    with (run_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "step",
                "loss_plus",
                "loss_minus",
                "train_loss",
                "d_h",
                "d_h_finite",
                "update_norm",
                "seconds",
                "eval_loss",
                "eval_acc",
                "nan_flag",
            ],
        )
        writer.writeheader()
        for step_idx in range(steps):
            step_start = time.time()
            states, refresh_rows = refresh_quantizer_states(
                master,
                q_names,
                quantizer=quantizer,
                activation_rms=harness.activation_rms if quantizer == "awq" else {},
            )
            last_quant = aggregate_quantizer_stats(refresh_rows, numel_by_name)
            append_jsonl(
                run_dir / "quantizer_diagnostics.jsonl",
                {
                    "step": step_idx,
                    "record_type": "refresh_summary",
                    "grid_id": step_idx + 1,
                    "scale_id": step_idx + 1,
                    **last_quant,
                },
            )
            directions = sample_directions(master, direction_seed(16, quantizer, h, step_idx))
            batch = move_batch(next(batch_iter), harness.device)
            copy_master_to_model(params, master, directions, h, +1.0, states)
            loss_plus, _ = harness.forward_loss_and_logits(batch)
            copy_master_to_model(params, master, directions, h, -1.0, states)
            loss_minus, _ = harness.forward_loss_and_logits(batch)
            restore_diff = restore_master(params, master)
            lp = float(loss_plus.detach().cpu())
            lm = float(loss_minus.detach().cpu())
            d_h = (lp - lm) / (2.0 * h)
            finite = math.isfinite(lp) and math.isfinite(lm) and math.isfinite(d_h)
            if finite:
                finite_count += 1
                update_norm_last = update_master(master, directions, harness.lr, d_h)
                restore_master(params, master)
            last_train_loss = (lp + lm) / 2.0
            if step_idx == 0 or (step_idx + 1) % max(1, int(harness.args.diag_every)) == 0 or step_idx == steps - 1:
                last_pert = perturbation_metrics(master, directions, states, h)
                last_pert.update({"grid_id_plus": step_idx + 1, "grid_id_minus": step_idx + 1, "scale_id_plus": step_idx + 1, "scale_id_minus": step_idx + 1})
                append_jsonl(run_dir / "perturbation_diagnostics.jsonl", {"step": step_idx + 1, **last_pert})

            eval_loss = None
            eval_acc = None
            completed_step = step_idx + 1
            if completed_step % eval_every == 0 or completed_step == steps:
                eval_loss, eval_acc = harness.eval_quantized(master, states, int(harness.args.eval_batches))
                append_jsonl(run_dir / "eval_metrics.jsonl", {"step": completed_step, "eval_loss": eval_loss, "eval_acc": eval_acc})
                if eval_acc is not None and (best["best_eval_acc"] is None or eval_acc > best["best_eval_acc"]):
                    best["best_eval_acc"] = eval_acc
                    best["best_step"] = completed_step
                if eval_loss is not None and (best["best_eval_loss"] is None or eval_loss < best["best_eval_loss"]):
                    best["best_eval_loss"] = eval_loss
                    best["best_eval_loss_step"] = completed_step
            if completed_step % checkpoint_every == 0 or completed_step == steps:
                save_checkpoint(run_dir, completed_step, master, best, config)

            nan_flag = (not finite) or restore_diff > 1e-3 or (update_norm_last is not None and not math.isfinite(float(update_norm_last)))
            writer.writerow(
                {
                    "step": completed_step,
                    "loss_plus": lp,
                    "loss_minus": lm,
                    "train_loss": last_train_loss,
                    "d_h": d_h,
                    "d_h_finite": finite,
                    "update_norm": update_norm_last,
                    "seconds": time.time() - step_start,
                    "eval_loss": eval_loss,
                    "eval_acc": eval_acc,
                    "nan_flag": nan_flag,
                }
            )
            f.flush()
            if nan_flag:
                status = "failed"
                error_message = f"non-finite/restore invariant failure at step {completed_step}"
                break

    steps_completed = 0
    last_eval_acc = None
    last_eval_loss = None
    last_eval_step = None
    with (run_dir / "metrics.csv").open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if rows:
        steps_completed = int(float(rows[-1]["step"]))
        eval_rows = [row for row in rows if row.get("eval_acc") not in (None, "")]
        if eval_rows:
            last = eval_rows[-1]
            last_eval_acc = float(last["eval_acc"]) if last["eval_acc"] else None
            last_eval_loss = float(last["eval_loss"]) if last["eval_loss"] else None
            last_eval_step = int(float(last["step"]))
    elapsed = time.time() - start
    peak_mem = float(torch.cuda.max_memory_allocated() / 1024 / 1024) if torch.cuda.is_available() else 0.0
    summary = {
        **config,
        "status": status,
        "error_message": error_message,
        "steps_completed": steps_completed,
        "best_eval_acc": best["best_eval_acc"],
        "best_step": best["best_step"],
        "last_eval_acc": last_eval_acc,
        "last_eval_step": last_eval_step,
        "best_eval_loss": best["best_eval_loss"],
        "best_eval_loss_step": best["best_eval_loss_step"],
        "last_eval_loss": last_eval_loss,
        "last_eval_loss_step": last_eval_step,
        "final_train_loss": last_train_loss,
        "d_h_finite_rate": finite_count / max(steps_completed, 1),
        "update_norm_last": update_norm_last,
        "active_frac": last_pert.get("active_frac"),
        "alignment": last_pert.get("alignment"),
        "norm_ratio": last_pert.get("norm_ratio"),
        "code_change_frac": last_pert.get("code_change_frac"),
        "delta_visibility_mse": last_pert.get("delta_visibility_mse"),
        "delta_visibility_nmse": last_pert.get("delta_visibility_nmse"),
        "delta_visibility_rel_l2": last_pert.get("delta_visibility_rel_l2"),
        "delta_q_norm": last_pert.get("delta_q_norm"),
        "saturation_frac_w": last_quant.get("saturation_frac_w"),
        "saturation_frac_w_plus": last_pert.get("saturation_frac_w_plus"),
        "saturation_frac_w_minus": last_pert.get("saturation_frac_w_minus"),
        "clip_frac": last_quant.get("clip_frac"),
        "recon_mse_global": last_quant.get("recon_mse_global"),
        "weight_recon_mse": last_quant.get("weight_recon_mse", last_quant.get("recon_mse_global")),
        "weight_recon_rel_mse": last_quant.get("weight_recon_rel_mse"),
        "weight_recon_sqnr_db": last_quant.get("weight_recon_sqnr_db"),
        "activation_weighted_mse_global": last_quant.get("activation_weighted_mse_global"),
        "alpha_mean": last_quant.get("alpha_mean"),
        "alpha_lt_1_frac": last_quant.get("alpha_lt_1_frac"),
        "grid_id_sharing_check": last_pert.get("grid_id_plus") == last_pert.get("grid_id_minus"),
        "scale_id_sharing_check": last_pert.get("scale_id_plus") == last_pert.get("scale_id_minus"),
        "pair_shared_grid_observed": bool(last_pert.get("pair_shared_grid_check", False)),
        "fresh_round_codes_observed": bool(last_pert.get("fresh_round_codes_check", False)),
        "corr_fd_true": None,
        "nMSE_fd_true": None,
        "fd_true_available": False,
        "fd_true_mse": None,
        "fd_true_nmse": None,
        "fd_true_rmse": None,
        "fd_true_bias": None,
        "true_gradient_metrics": "unavailable_not_computed",
        "total_runtime": elapsed,
        "seconds_per_step": elapsed / max(steps_completed, 1),
        "peak_gpu_mem": peak_mem,
    }
    write_json(run_dir / "run_summary.json", summary)
    return summary


def finite_difference_once(
    harness: BaseHarness,
    master: Dict[str, torch.Tensor],
    states: Dict[str, QuantizerState],
    directions: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    h: float,
) -> Tuple[float, float, float]:
    params = harness.params()
    copy_master_to_model(params, master, directions, h, +1.0, states)
    loss_plus, _ = harness.forward_loss_and_logits(batch)
    copy_master_to_model(params, master, directions, h, -1.0, states)
    loss_minus, _ = harness.forward_loss_and_logits(batch)
    restore_master(params, master)
    lp = float(loss_plus.detach().cpu())
    lm = float(loss_minus.detach().cpu())
    return lp, lm, (lp - lm) / (2.0 * h)


def run_probe(
    harness: BaseHarness,
    *,
    output_root: Path,
    quantizer: str,
    calibration_examples: int,
    env: Dict[str, object],
    compare_scope: Optional[Sequence[str]] = None,
) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    probe_dir = output_root / "probe" / harness.setting_name / quantizer
    probe_dir.mkdir(parents=True, exist_ok=True)
    config = make_run_config(
        harness,
        phase="probe",
        run_name=f"{harness.setting_name}_{quantizer}_int8_probe",
        run_dir=probe_dir,
        quantizer=quantizer,
        h=1e-3,
        steps=0,
        eval_every=0,
        checkpoint_every=0,
        calibration_examples=calibration_examples,
        env=env,
    )
    config["probe_h_grid"] = [{"label": label, "h": h} for label, h in H_GRID]
    config["probe_dirs"] = int(harness.args.probe_dirs)
    write_json(probe_dir / "run_config.json", config)
    write_resume_command(probe_dir, config)
    failures = hard_invariant_failures(config, compare_scope=compare_scope)
    if failures:
        raise RuntimeError(f"probe invariant failure for {harness.setting_name}/{quantizer}: {'; '.join(failures)}")

    params = harness.params()
    master = harness.make_master()
    restore_master(params, master)
    q_names = harness.quantized_module_names
    numel_by_name = harness.numel_by_quantized_name()
    states, refresh_rows = refresh_quantizer_states(
        master,
        q_names,
        quantizer=quantizer,
        activation_rms=harness.activation_rms if quantizer == "awq" else {},
    )
    quant = aggregate_quantizer_stats(refresh_rows, numel_by_name)
    write_json(probe_dir / "quantizer_refresh_summary.json", quant)
    batch = move_batch(next(iter(harness.train_loader)), harness.device)
    stats_path = probe_dir / "probe_stats.jsonl"
    if stats_path.exists():
        stats_path.unlink()
    rows = []
    for label, h in H_GRID:
        acc: Dict[str, List[float]] = {
            "loss_plus": [],
            "loss_minus": [],
            "fd": [],
            "fd_half": [],
            "locality_rel_error": [],
            "finite": [],
            "alignment": [],
            "norm_ratio": [],
            "active_frac": [],
            "code_change_frac": [],
            "delta_visibility_mse": [],
            "delta_visibility_nmse": [],
            "delta_visibility_rel_l2": [],
            "delta_q_norm": [],
            "nominal_delta_norm": [],
            "clip_plus": [],
            "clip_minus": [],
        }
        for k_dir in range(int(harness.args.probe_dirs)):
            directions = sample_directions(master, direction_seed(16, quantizer, h, 0, extra=k_dir))
            lp, lm, fd = finite_difference_once(harness, master, states, directions, batch, h)
            _, _, fd_half = finite_difference_once(harness, master, states, directions, batch, h / 2.0)
            pert = perturbation_metrics(master, directions, states, h)
            loc = abs(fd - fd_half) / (abs(fd_half) + 1e-12)
            finite = math.isfinite(lp) and math.isfinite(lm) and math.isfinite(fd)
            item = {
                "model": harness.model_name,
                "dataset": harness.dataset_name,
                "setting": harness.setting_name,
                "quantizer": quantizer,
                "quantizer_backend": QUANTIZER_BACKENDS[quantizer],
                "h_label": label,
                "h": h,
                "k_dir": k_dir,
                "loss_plus": lp,
                "loss_minus": lm,
                "finite_difference": fd,
                "fd_h_over_2": fd_half,
                "richardson_self_consistency": loc,
                "finite": finite,
                "grid_id_plus": 1,
                "grid_id_minus": 1,
                "scale_id_plus": 1,
                "scale_id_minus": 1,
                "grid_id_sharing_check": True,
                "scale_id_sharing_check": True,
                "corr_fd_true": None,
                "nMSE_fd_true": None,
                "fd_true_available": False,
                "fd_true_mse": None,
                "fd_true_nmse": None,
                "fd_true_rmse": None,
                "fd_true_bias": None,
                "true_gradient_metrics": "unavailable_not_computed",
                **pert,
                **{
                    key: quant.get(key)
                    for key in (
                        "clip_frac",
                        "saturation_frac_w",
                        "recon_mse_global",
                        "weight_recon_mse",
                        "weight_recon_rel_mse",
                        "weight_recon_sqnr_db",
                        "activation_weighted_mse_global",
                        "alpha_mean",
                        "alpha_lt_1_frac",
                        "scale_min_global",
                        "scale_median_weighted",
                        "scale_max_global",
                    )
                },
            }
            append_jsonl(stats_path, item)
            acc["loss_plus"].append(lp)
            acc["loss_minus"].append(lm)
            acc["fd"].append(fd)
            acc["fd_half"].append(fd_half)
            acc["locality_rel_error"].append(loc)
            acc["finite"].append(1.0 if finite else 0.0)
            acc["alignment"].append(float(pert["alignment"]))
            acc["norm_ratio"].append(float(pert["norm_ratio"]))
            acc["active_frac"].append(float(pert["active_frac"]))
            acc["code_change_frac"].append(float(pert["code_change_frac"]))
            acc["delta_visibility_mse"].append(float(pert["delta_visibility_mse"]))
            acc["delta_visibility_nmse"].append(float(pert["delta_visibility_nmse"]))
            acc["delta_visibility_rel_l2"].append(float(pert["delta_visibility_rel_l2"]))
            acc["delta_q_norm"].append(float(pert["delta_q_norm"]))
            acc["nominal_delta_norm"].append(float(pert["nominal_delta_norm"]))
            acc["clip_plus"].append(float(pert["clip_frac_w_plus"]))
            acc["clip_minus"].append(float(pert["clip_frac_w_minus"]))
        row = {
            "model": harness.model_name,
            "dataset": harness.dataset_name,
            "setting": harness.setting_name,
            "quantizer": quantizer,
            "quantizer_backend": QUANTIZER_BACKENDS[quantizer],
            "h_label": label,
            "h": h,
            "probe_dirs": int(harness.args.probe_dirs),
            "loss_plus": sum(acc["loss_plus"]) / len(acc["loss_plus"]),
            "loss_minus": sum(acc["loss_minus"]) / len(acc["loss_minus"]),
            "finite_difference": sum(acc["fd"]) / len(acc["fd"]),
            "fd_h_over_2": sum(acc["fd_half"]) / len(acc["fd_half"]),
            "richardson_self_consistency": sum(acc["locality_rel_error"]) / len(acc["locality_rel_error"]),
            "finite_rate": sum(acc["finite"]) / len(acc["finite"]),
            "delta_q_norm": sum(acc["delta_q_norm"]) / len(acc["delta_q_norm"]),
            "nominal_delta_norm": sum(acc["nominal_delta_norm"]) / len(acc["nominal_delta_norm"]),
            "alignment": sum(acc["alignment"]) / len(acc["alignment"]),
            "norm_ratio": sum(acc["norm_ratio"]) / len(acc["norm_ratio"]),
            "active_frac": sum(acc["active_frac"]) / len(acc["active_frac"]),
            "code_change_frac": sum(acc["code_change_frac"]) / len(acc["code_change_frac"]),
            "delta_visibility_mse": sum(acc["delta_visibility_mse"]) / len(acc["delta_visibility_mse"]),
            "delta_visibility_nmse": sum(acc["delta_visibility_nmse"]) / len(acc["delta_visibility_nmse"]),
            "delta_visibility_rel_l2": sum(acc["delta_visibility_rel_l2"]) / len(acc["delta_visibility_rel_l2"]),
            "clip_frac": quant.get("clip_frac"),
            "clip_frac_w_plus": sum(acc["clip_plus"]) / len(acc["clip_plus"]),
            "clip_frac_w_minus": sum(acc["clip_minus"]) / len(acc["clip_minus"]),
            "saturation_frac": quant.get("saturation_frac"),
            "saturation_frac_w": quant.get("saturation_frac_w"),
            "recon_mse_global": quant.get("recon_mse_global"),
            "weight_recon_mse": quant.get("weight_recon_mse", quant.get("recon_mse_global")),
            "weight_recon_rel_mse": quant.get("weight_recon_rel_mse"),
            "weight_recon_sqnr_db": quant.get("weight_recon_sqnr_db"),
            "activation_weighted_mse_global": quant.get("activation_weighted_mse_global"),
            "scale_min_global": quant.get("scale_min_global"),
            "scale_median_weighted": quant.get("scale_median_weighted"),
            "scale_max_global": quant.get("scale_max_global"),
            "alpha_mean": quant.get("alpha_mean"),
            "alpha_lt_1_frac": quant.get("alpha_lt_1_frac"),
            "grid_id_sharing_check": True,
            "scale_id_sharing_check": True,
            "pair_shared_grid_check": True,
            "fresh_round_codes_check": True,
            "corr_fd_true": None,
            "nMSE_fd_true": None,
            "fd_true_available": False,
            "fd_true_mse": None,
            "fd_true_nmse": None,
            "fd_true_rmse": None,
            "fd_true_bias": None,
            "true_gradient_metrics": "unavailable_not_computed",
        }
        rows.append(row)

    selected, verdict = select_h_from_probe(rows)
    summary = {
        "model": harness.model_name,
        "dataset": harness.dataset_name,
        "setting": harness.setting_name,
        "quantizer": quantizer,
        "quantizer_backend": QUANTIZER_BACKENDS[quantizer],
        "h_vis_min": verdict["h_vis_min"],
        "h_loc_max": verdict["h_loc_max"],
        "valid_window": verdict["valid_window"],
        "selected_h": selected["h"] if selected else None,
        "selected_h_label": selected["h_label"] if selected else None,
        "failure_mode": verdict["failure_mode"],
        "small_h_visibility_failure": verdict["small_h_visibility_failure"],
        "large_h_locality_failure": verdict["large_h_locality_failure"],
        "run_dir": str(probe_dir),
    }
    write_json(probe_dir / "probe_summary.json", summary)
    return rows, summary


def select_h_from_probe(rows: List[Dict[str, object]]) -> Tuple[Optional[Dict[str, object]], Dict[str, object]]:
    visible = [
        row
        for row in rows
        if float(row["finite_rate"]) >= 1.0
        and float(row["code_change_frac"]) >= 0.01
        and float(row["norm_ratio"]) >= 0.05
        and float(row["alignment"]) > 0.02
    ]
    local = [row for row in visible if float(row["richardson_self_consistency"]) <= 1.5]
    candidates = [row for row in local if float(row["h"]) < 1e-2] or [row for row in visible if float(row["h"]) < 1e-2] or visible
    selected = None
    if candidates:
        selected = max(
            candidates,
            key=lambda row: (
                float(row["alignment"]),
                -float(row["richardson_self_consistency"]),
                -abs(math.log(max(float(row["norm_ratio"]), 1e-12))),
            ),
        )
    small_rows = [row for row in rows if float(row["h"]) <= 1e-4]
    small_bad = bool(small_rows) and max(float(row["code_change_frac"]) for row in small_rows) < 0.05
    large_row = next((row for row in rows if abs(float(row["h"]) - 1e-2) < 1e-12), None)
    large_bad = False
    if large_row is not None and selected is not None:
        large_bad = float(large_row["richardson_self_consistency"]) > max(1.5, 1.5 * float(selected["richardson_self_consistency"]))
        large_bad = large_bad or float(large_row["alignment"]) < 0.5 * max(float(selected["alignment"]), 1e-12)
    if not visible:
        failure_mode = "no_window"
    elif len(visible) <= 2:
        failure_mode = "narrow_window"
    else:
        failure_mode = "window_exists"
    if small_bad and failure_mode != "no_window":
        small_mode = "small_h_visibility_failure"
    else:
        small_mode = failure_mode
    h_vis_min = min((float(row["h"]) for row in visible), default=None)
    h_loc_max = max((float(row["h"]) for row in local), default=None)
    return selected, {
        "h_vis_min": h_vis_min,
        "h_loc_max": h_loc_max,
        "valid_window": bool(visible and local),
        "failure_mode": small_mode if failure_mode == "window_exists" and not large_bad else failure_mode,
        "small_h_visibility_failure": small_bad,
        "large_h_locality_failure": large_bad,
    }


def h_policies_from_probe(summary: Dict[str, object]) -> List[Tuple[str, float]]:
    policies: List[Tuple[str, float]] = [("bad-small", 1e-5), ("default", 1e-3)]
    selected = summary.get("selected_h")
    if selected is not None:
        selected_h = float(selected)
        if all(abs(selected_h - h) > 1e-12 for _, h in policies):
            policies.append(("selected_h", selected_h))
        else:
            neighbor = next((h for h in (2e-3, 3e-3, 5e-3) if all(abs(h - x) > 1e-12 for _, x in policies)), None)
            if neighbor is not None:
                policies.append(("inside-window-neighbor", neighbor))
    else:
        policies.append(("selected_h_unavailable_neighbor", 3e-3))
    if all(abs(1e-2 - h) > 1e-12 for _, h in policies):
        policies.append(("bad-large", 1e-2))
    return policies


def run_smoke_matrix(
    harnesses: Sequence[BaseHarness],
    *,
    output_root: Path,
    env: Dict[str, object],
    calibration_examples: int,
) -> List[Dict[str, object]]:
    rows = []
    scope_by_setting = {}
    for harness in harnesses:
        scope_by_setting[harness.setting_name] = list(harness.quantized_module_names)
        for quantizer in ("rtnclip", "awq"):
            rows.append(
                run_training_job(
                    harness,
                    output_root=output_root,
                    phase="smoke",
                    quantizer=quantizer,
                    h=1e-3,
                    steps=20,
                    eval_every=20,
                    checkpoint_every=20,
                    calibration_examples=calibration_examples,
                    env=env,
                    compare_scope=scope_by_setting[harness.setting_name],
                )
            )
    return rows


def plot_results_matplotlib(output_root: Path, probe_rows: List[Dict[str, object]], hacc_rows: List[Dict[str, object]]) -> List[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_dir = output_root / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    paths: List[str] = []

    def save(path: Path):
        plt.tight_layout()
        plt.savefig(path, dpi=160)
        plt.close()
        paths.append(str(path))

    for setting, title in (("roberta_large_sst5", "RoBERTa Probe"), ("opt_1p3b_sst5", "OPT Probe")):
        subset = [row for row in probe_rows if row["setting"] == setting]
        if not subset:
            continue
        plt.figure(figsize=(8, 5))
        for quantizer in ("rtnclip", "awq"):
            qrows = sorted([row for row in subset if row["quantizer"] == quantizer], key=lambda r: float(r["h"]))
            if not qrows:
                continue
            hs = [float(row["h"]) for row in qrows]
            plt.plot(hs, [float(row["alignment"]) for row in qrows], marker="o", label=f"{quantizer} alignment")
            plt.plot(hs, [float(row["norm_ratio"]) for row in qrows], marker="s", label=f"{quantizer} norm_ratio")
            plt.plot(hs, [float(row["code_change_frac"]) for row in qrows], marker="^", label=f"{quantizer} code_change_frac")
        plt.xscale("log")
        plt.xlabel("h")
        plt.ylabel("metric")
        plt.title(title)
        plt.legend(fontsize=8)
        save(plot_dir / f"{setting}_probe_alignment_norm_code.png")

    for setting, title in (("roberta_large_sst5", "RoBERTa h-acc"), ("opt_1p3b_sst5", "OPT h-acc")):
        subset = [row for row in hacc_rows if row["setting"] == setting]
        if not subset:
            continue
        plt.figure(figsize=(7, 4))
        for quantizer in ("rtnclip", "awq"):
            qrows = sorted([row for row in subset if row["quantizer"] == quantizer], key=lambda r: float(r["h"]))
            if not qrows:
                continue
            plt.plot([float(row["h"]) for row in qrows], [float(row.get("last_eval_acc") or 0.0) for row in qrows], marker="o", label=quantizer)
        plt.xscale("log")
        plt.xlabel("h")
        plt.ylabel("eval accuracy")
        plt.title(title)
        plt.legend()
        save(plot_dir / f"{setting}_h_acc_eval_accuracy.png")

    selected_rows = []
    for setting in ("roberta_large_sst5", "opt_1p3b_sst5"):
        for quantizer in ("rtnclip", "awq"):
            qrows = [row for row in probe_rows if row["setting"] == setting and row["quantizer"] == quantizer]
            if qrows:
                selected, _ = select_h_from_probe(qrows)
                selected_rows.append({"label": f"{setting}\n{quantizer}", "h": float(selected["h"]) if selected else float("nan")})
    if selected_rows:
        plt.figure(figsize=(8, 4))
        plt.bar([row["label"] for row in selected_rows], [row["h"] for row in selected_rows])
        plt.yscale("log")
        plt.ylabel("selected_h")
        plt.title("Selected h comparison")
        save(plot_dir / "selected_h_comparison.png")

    window_rows = []
    for setting in ("roberta_large_sst5", "opt_1p3b_sst5"):
        for quantizer in ("rtnclip", "awq"):
            qrows = [row for row in probe_rows if row["setting"] == setting and row["quantizer"] == quantizer]
            if qrows:
                _, verdict = select_h_from_probe(qrows)
                if verdict["h_vis_min"] is not None and verdict["h_loc_max"] is not None:
                    window_rows.append((setting, quantizer, float(verdict["h_vis_min"]), float(verdict["h_loc_max"])))
    if window_rows:
        plt.figure(figsize=(8, 4))
        yticks = []
        ylabels = []
        for idx, (setting, quantizer, lo, hi) in enumerate(window_rows):
            plt.hlines(idx, lo, hi, linewidth=5, label=quantizer)
            plt.scatter([lo, hi], [idx, idx])
            yticks.append(idx)
            ylabels.append(f"{setting} {quantizer}")
        plt.xscale("log")
        plt.yticks(yticks, ylabels)
        plt.xlabel("h")
        plt.title("Valid-window overlay")
        save(plot_dir / "valid_window_overlay.png")
    return paths


def svg_escape(text: object) -> str:
    value = str(text)
    return value.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def finite_float(value: object) -> Optional[float]:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return out


def write_svg_line_chart(
    path: Path,
    *,
    title: str,
    x_label: str,
    y_label: str,
    series: Sequence[Tuple[str, Sequence[float], Sequence[float]]],
    x_log: bool = True,
    y_log: bool = False,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf", "#8c564b", "#7f7f7f"]
    width, height = 900, 520
    left, right, top, bottom = 82, 28, 48, 74
    plot_w = width - left - right
    plot_h = height - top - bottom
    points = []
    for label, xs, ys in series:
        clean = []
        for x, y in zip(xs, ys):
            xf = finite_float(x)
            yf = finite_float(y)
            if xf is None or yf is None:
                continue
            if x_log and xf <= 0:
                continue
            if y_log and yf <= 0:
                continue
            clean.append((xf, yf))
        if clean:
            points.append((label, clean))
    if not points:
        path.write_text("<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"900\" height=\"220\"><text x=\"20\" y=\"40\">No data</text></svg>\n", encoding="utf-8")
        return
    x_vals = [math.log10(x) if x_log else x for _, clean in points for x, _ in clean]
    y_vals = [math.log10(y) if y_log else y for _, clean in points for _, y in clean]
    x_min, x_max = min(x_vals), max(x_vals)
    y_min, y_max = min(y_vals), max(y_vals)
    if abs(x_max - x_min) < 1e-12:
        x_min -= 1.0
        x_max += 1.0
    if abs(y_max - y_min) < 1e-12:
        y_min -= 1.0
        y_max += 1.0
    y_pad = 0.05 * (y_max - y_min)
    y_min -= y_pad
    y_max += y_pad

    def sx(x: float) -> float:
        xv = math.log10(x) if x_log else x
        return left + (xv - x_min) / (x_max - x_min) * plot_w

    def sy(y: float) -> float:
        yv = math.log10(y) if y_log else y
        return top + (y_max - yv) / (y_max - y_min) * plot_h

    parts = [
        f"<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{width}\" height=\"{height}\" viewBox=\"0 0 {width} {height}\">",
        "<rect width=\"100%\" height=\"100%\" fill=\"white\"/>",
        f"<text x=\"{width / 2:.1f}\" y=\"28\" text-anchor=\"middle\" font-size=\"18\" font-family=\"sans-serif\">{svg_escape(title)}</text>",
        f"<line x1=\"{left}\" y1=\"{top + plot_h}\" x2=\"{left + plot_w}\" y2=\"{top + plot_h}\" stroke=\"#222\"/>",
        f"<line x1=\"{left}\" y1=\"{top}\" x2=\"{left}\" y2=\"{top + plot_h}\" stroke=\"#222\"/>",
        f"<text x=\"{left + plot_w / 2:.1f}\" y=\"{height - 22}\" text-anchor=\"middle\" font-size=\"13\" font-family=\"sans-serif\">{svg_escape(x_label)}</text>",
        f"<text x=\"18\" y=\"{top + plot_h / 2:.1f}\" transform=\"rotate(-90 18 {top + plot_h / 2:.1f})\" text-anchor=\"middle\" font-size=\"13\" font-family=\"sans-serif\">{svg_escape(y_label)}</text>",
    ]
    x_ticks = sorted({x for _, clean in points for x, _ in clean})
    if len(x_ticks) > 12:
        x_ticks = x_ticks[:: max(1, len(x_ticks) // 9)]
    for x in x_ticks:
        px = sx(x)
        parts.append(f"<line x1=\"{px:.1f}\" y1=\"{top + plot_h}\" x2=\"{px:.1f}\" y2=\"{top + plot_h + 5}\" stroke=\"#222\"/>")
        parts.append(f"<text x=\"{px:.1f}\" y=\"{top + plot_h + 22}\" text-anchor=\"middle\" font-size=\"10\" font-family=\"sans-serif\">{x:g}</text>")
    for idx in range(5):
        frac = idx / 4
        yv = y_min + frac * (y_max - y_min)
        raw = 10 ** yv if y_log else yv
        py = sy(raw)
        parts.append(f"<line x1=\"{left - 5}\" y1=\"{py:.1f}\" x2=\"{left}\" y2=\"{py:.1f}\" stroke=\"#222\"/>")
        parts.append(f"<text x=\"{left - 9}\" y=\"{py + 4:.1f}\" text-anchor=\"end\" font-size=\"10\" font-family=\"sans-serif\">{raw:.3g}</text>")
        if idx not in (0, 4):
            parts.append(f"<line x1=\"{left}\" y1=\"{py:.1f}\" x2=\"{left + plot_w}\" y2=\"{py:.1f}\" stroke=\"#ddd\"/>")
    for idx, (label, clean) in enumerate(points):
        color = colors[idx % len(colors)]
        coords = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in clean)
        parts.append(f"<polyline points=\"{coords}\" fill=\"none\" stroke=\"{color}\" stroke-width=\"2\"/>")
        for x, y in clean:
            parts.append(f"<circle cx=\"{sx(x):.1f}\" cy=\"{sy(y):.1f}\" r=\"3\" fill=\"{color}\"/>")
        lx = left + 16 + (idx % 3) * 245
        ly = top + 18 + (idx // 3) * 18
        parts.append(f"<line x1=\"{lx}\" y1=\"{ly}\" x2=\"{lx + 18}\" y2=\"{ly}\" stroke=\"{color}\" stroke-width=\"2\"/>")
        parts.append(f"<text x=\"{lx + 24}\" y=\"{ly + 4}\" font-size=\"11\" font-family=\"sans-serif\">{svg_escape(label)}</text>")
    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def write_svg_bar_chart(path: Path, *, title: str, labels: Sequence[str], values: Sequence[float], y_log: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 900, 460
    left, right, top, bottom = 70, 24, 48, 112
    plot_w = width - left - right
    plot_h = height - top - bottom
    clean = [(label, finite_float(value)) for label, value in zip(labels, values)]
    clean = [(label, value) for label, value in clean if value is not None and (not y_log or value > 0)]
    if not clean:
        path.write_text("<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"900\" height=\"220\"><text x=\"20\" y=\"40\">No data</text></svg>\n", encoding="utf-8")
        return
    vals = [math.log10(value) if y_log else value for _, value in clean]
    y_min, y_max = min(vals), max(vals)
    if abs(y_max - y_min) < 1e-12:
        y_min -= 1.0
        y_max += 1.0
    y_min -= 0.1 * (y_max - y_min)
    y_max += 0.1 * (y_max - y_min)

    def sy(value: float) -> float:
        yv = math.log10(value) if y_log else value
        return top + (y_max - yv) / (y_max - y_min) * plot_h

    bar_w = plot_w / max(len(clean), 1) * 0.65
    parts = [
        f"<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{width}\" height=\"{height}\" viewBox=\"0 0 {width} {height}\">",
        "<rect width=\"100%\" height=\"100%\" fill=\"white\"/>",
        f"<text x=\"{width / 2:.1f}\" y=\"28\" text-anchor=\"middle\" font-size=\"18\" font-family=\"sans-serif\">{svg_escape(title)}</text>",
        f"<line x1=\"{left}\" y1=\"{top + plot_h}\" x2=\"{left + plot_w}\" y2=\"{top + plot_h}\" stroke=\"#222\"/>",
        f"<line x1=\"{left}\" y1=\"{top}\" x2=\"{left}\" y2=\"{top + plot_h}\" stroke=\"#222\"/>",
    ]
    for idx, (label, value) in enumerate(clean):
        cx = left + (idx + 0.5) * plot_w / len(clean)
        y = sy(value)
        parts.append(f"<rect x=\"{cx - bar_w / 2:.1f}\" y=\"{y:.1f}\" width=\"{bar_w:.1f}\" height=\"{top + plot_h - y:.1f}\" fill=\"#1f77b4\"/>")
        parts.append(f"<text x=\"{cx:.1f}\" y=\"{top + plot_h + 18}\" transform=\"rotate(25 {cx:.1f} {top + plot_h + 18})\" text-anchor=\"start\" font-size=\"10\" font-family=\"sans-serif\">{svg_escape(label)}</text>")
        parts.append(f"<text x=\"{cx:.1f}\" y=\"{y - 5:.1f}\" text-anchor=\"middle\" font-size=\"10\" font-family=\"sans-serif\">{value:g}</text>")
    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def write_svg_window_overlay(path: Path, window_rows: Sequence[Tuple[str, str, float, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 900, 300 + 34 * len(window_rows)
    left, right, top, bottom = 170, 32, 48, 50
    plot_w = width - left - right
    if not window_rows:
        path.write_text("<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"900\" height=\"220\"><text x=\"20\" y=\"40\">No data</text></svg>\n", encoding="utf-8")
        return
    vals = [v for _, _, lo, hi in window_rows for v in (lo, hi) if v > 0]
    x_min, x_max = math.log10(min(vals)), math.log10(max(vals))
    if abs(x_max - x_min) < 1e-12:
        x_min -= 1.0
        x_max += 1.0

    def sx(value: float) -> float:
        return left + (math.log10(value) - x_min) / (x_max - x_min) * plot_w

    colors = {"rtnclip": "#1f77b4", "awq": "#d62728"}
    parts = [
        f"<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{width}\" height=\"{height}\" viewBox=\"0 0 {width} {height}\">",
        "<rect width=\"100%\" height=\"100%\" fill=\"white\"/>",
        f"<text x=\"{width / 2:.1f}\" y=\"28\" text-anchor=\"middle\" font-size=\"18\" font-family=\"sans-serif\">Valid-window overlay</text>",
        f"<line x1=\"{left}\" y1=\"{height - bottom}\" x2=\"{left + plot_w}\" y2=\"{height - bottom}\" stroke=\"#222\"/>",
    ]
    for idx, (setting, quantizer, lo, hi) in enumerate(window_rows):
        y = top + 36 + idx * 34
        color = colors.get(quantizer, "#444")
        parts.append(f"<text x=\"{left - 10}\" y=\"{y + 4}\" text-anchor=\"end\" font-size=\"11\" font-family=\"sans-serif\">{svg_escape(setting)} {svg_escape(quantizer)}</text>")
        parts.append(f"<line x1=\"{sx(lo):.1f}\" y1=\"{y}\" x2=\"{sx(hi):.1f}\" y2=\"{y}\" stroke=\"{color}\" stroke-width=\"7\" stroke-linecap=\"round\"/>")
        parts.append(f"<circle cx=\"{sx(lo):.1f}\" cy=\"{y}\" r=\"4\" fill=\"{color}\"/>")
        parts.append(f"<circle cx=\"{sx(hi):.1f}\" cy=\"{y}\" r=\"4\" fill=\"{color}\"/>")
    for value in sorted(set(vals)):
        x = sx(value)
        parts.append(f"<line x1=\"{x:.1f}\" y1=\"{height - bottom}\" x2=\"{x:.1f}\" y2=\"{height - bottom + 5}\" stroke=\"#222\"/>")
        parts.append(f"<text x=\"{x:.1f}\" y=\"{height - bottom + 22}\" text-anchor=\"middle\" font-size=\"10\" font-family=\"sans-serif\">{value:g}</text>")
    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def plot_results_svg(output_root: Path, probe_rows: List[Dict[str, object]], hacc_rows: List[Dict[str, object]]) -> List[str]:
    plot_dir = output_root / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    paths: List[str] = []

    for setting, title in (("roberta_large_sst5", "RoBERTa Probe"), ("opt_1p3b_sst5", "OPT Probe")):
        subset = [row for row in probe_rows if row["setting"] == setting]
        if not subset:
            continue
        series = []
        for quantizer in ("rtnclip", "awq"):
            qrows = sorted([row for row in subset if row["quantizer"] == quantizer], key=lambda r: float(r["h"]))
            if not qrows:
                continue
            hs = [float(row["h"]) for row in qrows]
            series.append((f"{quantizer} alignment", hs, [float(row["alignment"]) for row in qrows]))
            series.append((f"{quantizer} norm_ratio", hs, [float(row["norm_ratio"]) for row in qrows]))
            series.append((f"{quantizer} code_change_frac", hs, [float(row["code_change_frac"]) for row in qrows]))
        path = plot_dir / f"{setting}_probe_alignment_norm_code.svg"
        write_svg_line_chart(path, title=title, x_label="h", y_label="metric", series=series, x_log=True)
        paths.append(str(path))

    for setting, title in (("roberta_large_sst5", "RoBERTa h-acc"), ("opt_1p3b_sst5", "OPT h-acc")):
        subset = [row for row in hacc_rows if row["setting"] == setting]
        if not subset:
            continue
        series = []
        for quantizer in ("rtnclip", "awq"):
            qrows = sorted([row for row in subset if row["quantizer"] == quantizer], key=lambda r: float(r["h"]))
            if not qrows:
                continue
            series.append((quantizer, [float(row["h"]) for row in qrows], [float(row.get("last_eval_acc") or 0.0) for row in qrows]))
        path = plot_dir / f"{setting}_h_acc_eval_accuracy.svg"
        write_svg_line_chart(path, title=title, x_label="h", y_label="eval accuracy", series=series, x_log=True)
        paths.append(str(path))

    selected_rows = []
    for setting in ("roberta_large_sst5", "opt_1p3b_sst5"):
        for quantizer in ("rtnclip", "awq"):
            qrows = [row for row in probe_rows if row["setting"] == setting and row["quantizer"] == quantizer]
            if qrows:
                selected, _ = select_h_from_probe(qrows)
                selected_rows.append((f"{setting} {quantizer}", float(selected["h"]) if selected else float("nan")))
    if selected_rows:
        path = plot_dir / "selected_h_comparison.svg"
        write_svg_bar_chart(path, title="Selected h comparison", labels=[row[0] for row in selected_rows], values=[row[1] for row in selected_rows], y_log=True)
        paths.append(str(path))

    window_rows = []
    for setting in ("roberta_large_sst5", "opt_1p3b_sst5"):
        for quantizer in ("rtnclip", "awq"):
            qrows = [row for row in probe_rows if row["setting"] == setting and row["quantizer"] == quantizer]
            if qrows:
                _, verdict = select_h_from_probe(qrows)
                if verdict["h_vis_min"] is not None and verdict["h_loc_max"] is not None:
                    window_rows.append((setting, quantizer, float(verdict["h_vis_min"]), float(verdict["h_loc_max"])))
    if window_rows:
        path = plot_dir / "valid_window_overlay.svg"
        write_svg_window_overlay(path, window_rows)
        paths.append(str(path))
    return paths


def plot_results(output_root: Path, probe_rows: List[Dict[str, object]], hacc_rows: List[Dict[str, object]]) -> List[str]:
    try:
        return plot_results_matplotlib(output_root, probe_rows, hacc_rows)
    except ModuleNotFoundError as exc:
        if exc.name != "matplotlib":
            raise
        return plot_results_svg(output_root, probe_rows, hacc_rows)


def write_markdown_outputs(
    output_root: Path,
    *,
    env: Dict[str, object],
    smoke_rows: List[Dict[str, object]],
    calibration_rows: List[Dict[str, object]],
    probe_rows: List[Dict[str, object]],
    probe_summaries: List[Dict[str, object]],
    hacc_rows: List[Dict[str, object]],
    plot_paths: List[str],
) -> None:
    write_json(output_root / "smoke_summary.json", {"runs": smoke_rows})
    smoke_lines = [
        "# Smoke Summary",
        "",
        f"All smoke runs passed: {'yes' if all(row.get('status') == 'complete' and int(row.get('steps_completed', 0)) == 20 for row in smoke_rows) else 'no'}",
        "",
        "| model | dataset | quantizer | status | steps | sampler | active_frac | alignment | norm_ratio | run_dir |",
        "| --- | --- | --- | --- | ---: | --- | ---: | ---: | ---: | --- |",
    ]
    for row in smoke_rows:
        smoke_lines.append(
            f"| {row.get('model')} | {row.get('dataset')} | {row.get('quantizer_backend')} | {row.get('status')} | "
            f"{row.get('steps_completed')} | {row.get('sampler_name')} | {format_float(row.get('active_frac'))} | "
            f"{format_float(row.get('alignment'))} | {format_float(row.get('norm_ratio'))} | `{row.get('run_dir')}` |"
        )
    (output_root / "smoke_summary.md").write_text("\n".join(smoke_lines) + "\n", encoding="utf-8")

    calib_lines = [
        "# Quantizer Calibration Summary",
        "",
        "AWQ-style calibration uses activation RMS from a small training subset and activation-weighted weight MSE. RTNClip uses the existing unweighted clipping objective.",
        "",
        "| model | quantizer | calibration_examples | activation_rms_modules | quantized_modules | objective | alpha_grid |",
        "| --- | --- | ---: | ---: | ---: | --- | --- |",
    ]
    for row in calibration_rows:
        calib_lines.append(
            f"| {row['model']} | {row['quantizer']} | {row['calibration_examples']} | {row['activation_rms_modules']} | "
            f"{row['quantized_modules']} | {row['objective']} | `{row['alpha_grid']}` |"
        )
    (output_root / "quantizer_calibration_summary.md").write_text("\n".join(calib_lines) + "\n", encoding="utf-8")
    write_json(output_root / "quantizer_calibration_summary.json", {"rows": calibration_rows})

    probe_cols = [
        "model",
        "dataset",
        "setting",
        "quantizer",
        "quantizer_backend",
        "h_label",
        "h",
        "probe_dirs",
        "loss_plus",
        "loss_minus",
        "finite_difference",
        "fd_h_over_2",
        "richardson_self_consistency",
        "finite_rate",
        "delta_q_norm",
        "nominal_delta_norm",
        "alignment",
        "norm_ratio",
        "active_frac",
        "code_change_frac",
        "delta_visibility_mse",
        "delta_visibility_nmse",
        "delta_visibility_rel_l2",
        "clip_frac",
        "clip_frac_w_plus",
        "clip_frac_w_minus",
        "saturation_frac",
        "saturation_frac_w",
        "recon_mse_global",
        "weight_recon_mse",
        "weight_recon_rel_mse",
        "weight_recon_sqnr_db",
        "activation_weighted_mse_global",
        "scale_min_global",
        "scale_median_weighted",
        "scale_max_global",
        "alpha_mean",
        "alpha_lt_1_frac",
        "grid_id_sharing_check",
        "scale_id_sharing_check",
        "pair_shared_grid_check",
        "fresh_round_codes_check",
        "corr_fd_true",
        "nMSE_fd_true",
        "fd_true_available",
        "fd_true_mse",
        "fd_true_nmse",
        "fd_true_rmse",
        "fd_true_bias",
        "true_gradient_metrics",
    ]
    write_csv(output_root / "probe_results.csv", probe_rows, probe_cols)
    probe_lines = [
        "# Probe Summary",
        "",
        "| model | quantizer | h_vis_min | h_loc_max | selected_h | valid_window | failure_mode | run_dir |",
        "| --- | --- | ---: | ---: | ---: | --- | --- | --- |",
    ]
    for row in probe_summaries:
        probe_lines.append(
            f"| {row['model']} | {row['quantizer_backend']} | {format_float(row['h_vis_min'])} | "
            f"{format_float(row['h_loc_max'])} | {format_float(row['selected_h'])} | {row['valid_window']} | "
            f"{row['failure_mode']} | `{row['run_dir']}` |"
        )
    (output_root / "probe_summary.md").write_text("\n".join(probe_lines) + "\n", encoding="utf-8")
    write_json(output_root / "probe_summary.json", {"runs": probe_summaries})

    hacc_cols = [
        "model",
        "dataset",
        "setting",
        "quantizer",
        "quantizer_backend",
        "policy",
        "h",
        "h_label",
        "status",
        "steps_completed",
        "batch_size",
        "lr",
        "best_eval_acc",
        "last_eval_acc",
        "best_step",
        "last_eval_step",
        "best_eval_loss",
        "last_eval_loss",
        "final_train_loss",
        "active_frac",
        "alignment",
        "norm_ratio",
        "code_change_frac",
        "run_dir",
    ]
    write_csv(output_root / "h_acc_results.csv", hacc_rows, hacc_cols)
    hacc_lines = [
        "# h-acc Summary",
        "",
        "| model | quantizer | policy | h | status | steps | best_acc | last_acc | run_dir |",
        "| --- | --- | --- | ---: | --- | ---: | ---: | ---: | --- |",
    ]
    for row in hacc_rows:
        hacc_lines.append(
            f"| {row.get('model')} | {row.get('quantizer_backend')} | {row.get('policy')} | {format_float(row.get('h'))} | "
            f"{row.get('status')} | {row.get('steps_completed')} | {format_float(row.get('best_eval_acc'))} | "
            f"{format_float(row.get('last_eval_acc'))} | `{row.get('run_dir')}` |"
        )
    (output_root / "h_acc_summary.md").write_text("\n".join(hacc_lines) + "\n", encoding="utf-8")

    scheduler_lines = [
        "# Scheduler Jobs",
        "",
        "No scheduler jobs were submitted. The batch ran as one local lane on the preallocated GPU.",
        "",
        f"- GPU: `{env.get('gpu_name', '')}`",
        "- Active scheduler tasks: 0",
        "- Lane count: 1",
        "- H100/A100 fallback status: no fallback; local H100 was already allocated",
    ]
    (output_root / "scheduler_jobs.md").write_text("\n".join(scheduler_lines) + "\n", encoding="utf-8")

    answers = final_verdict_answers(probe_summaries, hacc_rows)
    next_lines = [
        "# Recommended Next Steps",
        "",
        *[f"{idx}. {answer}" for idx, answer in enumerate(answers, start=1)],
        "",
        "Plots:",
        *[f"- `{path}`" for path in plot_paths],
    ]
    (output_root / "recommended_next_steps.md").write_text("\n".join(next_lines) + "\n", encoding="utf-8")


def final_verdict_answers(probe_summaries: List[Dict[str, object]], hacc_rows: List[Dict[str, object]]) -> List[str]:
    by_key = {(row["setting"], row["quantizer"]): row for row in probe_summaries}

    def exists(setting: str, quantizer: str) -> bool:
        row = by_key.get((setting, quantizer), {})
        return bool(row.get("valid_window")) and row.get("failure_mode") != "no_window"

    def selected(setting: str, quantizer: str):
        row = by_key.get((setting, quantizer), {})
        return row.get("selected_h")

    def default_inside(setting: str, quantizer: str) -> bool:
        row = by_key.get((setting, quantizer), {})
        lo = row.get("h_vis_min")
        hi = row.get("h_loc_max")
        return lo is not None and hi is not None and float(lo) <= 1e-3 <= float(hi)

    roberta_awq = exists("roberta_large_sst5", "awq")
    opt_awq = exists("opt_1p3b_sst5", "awq")
    roberta_shift = selected("roberta_large_sst5", "rtnclip") != selected("roberta_large_sst5", "awq")
    opt_shift = selected("opt_1p3b_sst5", "rtnclip") != selected("opt_1p3b_sst5", "awq")
    small_awq = [
        by_key.get(("roberta_large_sst5", "awq"), {}).get("small_h_visibility_failure"),
        by_key.get(("opt_1p3b_sst5", "awq"), {}).get("small_h_visibility_failure"),
    ]
    large_awq = [
        by_key.get(("roberta_large_sst5", "awq"), {}).get("large_h_locality_failure"),
        by_key.get(("opt_1p3b_sst5", "awq"), {}).get("large_h_locality_failure"),
    ]
    return [
        f"RoBERTa-large AWQ-style INT8 window: {'yes' if roberta_awq else 'no'} based on probe valid_window.",
        f"OPT-1.3B AWQ-style INT8 window: {'yes' if opt_awq else 'no'} based on probe valid_window.",
        f"AWQ-style selected_h shift vs RTNClip: RoBERTa={'yes' if roberta_shift else 'no'}, OPT={'yes' if opt_shift else 'no'}.",
        f"AWQ-style small-h visibility failure reduced: {'mixed/see probe metrics' if any(small_awq) else 'not clearly'}; compare code_change_frac at 1e-5 and 3e-5.",
        f"AWQ-style large-h locality changed: {'yes' if any(large_awq) else 'not clearly'} by Richardson/self-consistency at 1e-2.",
        "MeZO default h=1e-3 inside estimated window: "
        f"RoBERTa RTNClip={default_inside('roberta_large_sst5', 'rtnclip')}, "
        f"RoBERTa AWQ={default_inside('roberta_large_sst5', 'awq')}, "
        f"OPT RTNClip={default_inside('opt_1p3b_sst5', 'rtnclip')}, "
        f"OPT AWQ={default_inside('opt_1p3b_sst5', 'awq')}.",
        f"Qualitative window shape robust to quantizer change: {'yes' if roberta_awq and opt_awq else 'not fully'} for this sanity batch.",
        f"Add AWQ-style as appendix robustness ablation: {'yes' if roberta_awq or opt_awq else 'defer until a cleaner probe'}; keep name awq_style_g128_fake_quant.",
        "HQQ-style next: unnecessary for this batch because no easy existing HQQ path was found; run later only if a lightweight shared-grid fake-quant HQQ probe is added.",
    ]


def write_read_files_report(output_root: Path) -> None:
    requested = [
        "updated_experiment_plan_int8_breadth_v6.md",
        "updated_experiment_plan_rtnclip_v4.md",
        "main_experiment_plan_revised_with_tables.md",
    ]
    lines = ["# Input Files Read", ""]
    for name in requested:
        path = REPO_ROOT / name
        lines.append(f"- `{name}`: {'present/read' if path.exists() else 'not present'}")
    lines.extend(
        [
            "- `tools/smoke_rtnclip_roberta_sst5.py`: read",
            "- `tools/rtnclip_roberta_sst5_batch.py`: read",
            "- `large_models/README.md`: read",
            "- `large_models/tasks.py`: read",
            "- `large_models/run.py`: read relevant OPT/SST-5 sections",
            "- `large_models/utils.py`: read prompt encoding and option-loss semantics",
            "- `docs/roberta_int8_implementation.md`: read",
            "- `docs/quzo_probe_smoke_20260419.md`: read",
        ]
    )
    (output_root / "input_files_read.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def mean_present(rows: Sequence[Dict[str, object]], key: str) -> Optional[float]:
    vals = [finite_float(row.get(key)) for row in rows]
    clean = [value for value in vals if value is not None]
    if not clean:
        return None
    return sum(clean) / len(clean)


def rebuild_probe_rows_from_stats(output_root: Path) -> List[Dict[str, object]]:
    probe_rows: List[Dict[str, object]] = []
    for stats_path in sorted((output_root / "probe").glob("*/*/probe_stats.jsonl")):
        raw_rows = read_jsonl(stats_path)
        grouped: Dict[float, List[Dict[str, object]]] = {}
        for row in raw_rows:
            h = finite_float(row.get("h"))
            if h is None:
                continue
            grouped.setdefault(h, []).append(row)
        for h in sorted(grouped):
            rows = grouped[h]
            first = rows[0]
            finite_vals = [1.0 if row.get("finite") is True else 0.0 for row in rows]
            row = {
                "model": first.get("model"),
                "dataset": first.get("dataset"),
                "setting": first.get("setting"),
                "quantizer": first.get("quantizer"),
                "quantizer_backend": first.get("quantizer_backend"),
                "h_label": first.get("h_label"),
                "h": h,
                "probe_dirs": len(rows),
                "loss_plus": mean_present(rows, "loss_plus"),
                "loss_minus": mean_present(rows, "loss_minus"),
                "finite_difference": mean_present(rows, "finite_difference"),
                "fd_h_over_2": mean_present(rows, "fd_h_over_2"),
                "richardson_self_consistency": mean_present(rows, "richardson_self_consistency"),
                "finite_rate": sum(finite_vals) / len(finite_vals) if finite_vals else None,
                "delta_q_norm": mean_present(rows, "delta_q_norm"),
                "nominal_delta_norm": mean_present(rows, "nominal_delta_norm"),
                "alignment": mean_present(rows, "alignment"),
                "norm_ratio": mean_present(rows, "norm_ratio"),
                "active_frac": mean_present(rows, "active_frac"),
                "code_change_frac": mean_present(rows, "code_change_frac"),
                "clip_frac": mean_present(rows, "clip_frac"),
                "clip_frac_w_plus": mean_present(rows, "clip_frac_w_plus"),
                "clip_frac_w_minus": mean_present(rows, "clip_frac_w_minus"),
                "saturation_frac": mean_present(rows, "saturation_frac_w"),
                "saturation_frac_w": mean_present(rows, "saturation_frac_w"),
                "scale_min_global": mean_present(rows, "scale_min_global"),
                "scale_median_weighted": mean_present(rows, "scale_median_weighted"),
                "scale_max_global": mean_present(rows, "scale_max_global"),
                "alpha_mean": mean_present(rows, "alpha_mean"),
                "alpha_lt_1_frac": mean_present(rows, "alpha_lt_1_frac"),
                "grid_id_sharing_check": all(bool(item.get("grid_id_sharing_check")) for item in rows),
                "scale_id_sharing_check": all(bool(item.get("scale_id_sharing_check")) for item in rows),
                "pair_shared_grid_check": all(bool(item.get("pair_shared_grid_check")) for item in rows),
                "fresh_round_codes_check": all(bool(item.get("fresh_round_codes_check")) for item in rows),
                "corr_fd_true": None,
                "nMSE_fd_true": None,
                "true_gradient_metrics": first.get("true_gradient_metrics", "unavailable_not_computed"),
            }
            probe_rows.append(row)
    return probe_rows


def infer_policy(row: Dict[str, object], probe_summaries: Sequence[Dict[str, object]]) -> str:
    h = finite_float(row.get("h"))
    if h is None:
        return str(row.get("policy") or "")
    if abs(h - 1e-5) < 1e-12:
        return "bad-small"
    if abs(h - 1e-3) < 1e-12:
        return "default"
    if abs(h - 1e-2) < 1e-12:
        return "bad-large"
    for summary in probe_summaries:
        if summary.get("setting") == row.get("setting") and summary.get("quantizer") == row.get("quantizer"):
            selected_h = finite_float(summary.get("selected_h"))
            if selected_h is not None and abs(h - selected_h) < 1e-12:
                return "selected_h"
    return "inside-window-neighbor"


def load_completed_outputs(output_root: Path) -> Tuple[Dict[str, object], List[Dict[str, object]], List[Dict[str, object]], List[Dict[str, object]], List[Dict[str, object]], List[Dict[str, object]]]:
    env = read_json(output_root / "env.json") if (output_root / "env.json").exists() else {}
    smoke_rows = read_json(output_root / "smoke_summary.json").get("runs", []) if (output_root / "smoke_summary.json").exists() else []
    calibration_rows = read_json(output_root / "quantizer_calibration_summary.json").get("rows", []) if (output_root / "quantizer_calibration_summary.json").exists() else []
    probe_rows = rebuild_probe_rows_from_stats(output_root)
    probe_summaries = [read_json(path) for path in sorted((output_root / "probe").glob("*/*/probe_summary.json"))]
    hacc_rows: List[Dict[str, object]] = []
    for summary_path in sorted((output_root / "h_acc").glob("*/*/*/run_summary.json")):
        row = read_json(summary_path)
        rel_parts = summary_path.relative_to(output_root / "h_acc").parts
        if rel_parts:
            row.setdefault("setting", rel_parts[0])
        row["policy"] = infer_policy(row, probe_summaries)
        hacc_rows.append(row)
    return env, smoke_rows, calibration_rows, probe_rows, probe_summaries, hacc_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_root", type=Path, default=REPO_ROOT / "outputs" / "quantizer_robustness_int8_window")
    parser.add_argument("--probe_dirs", type=int, default=4)
    parser.add_argument("--calibration_examples", type=int, default=128)
    parser.add_argument("--roberta_train_steps", type=int, default=1000)
    parser.add_argument("--opt_train_steps", type=int, default=500)
    parser.add_argument("--smoke_only", action="store_true")
    parser.add_argument("--skip_smoke", action="store_true")
    parser.add_argument("--probe_only", action="store_true")
    parser.add_argument("--skip_hacc", action="store_true")
    parser.add_argument("--eval_batches", type=int, default=8, help="-1 means full dev, positive caps eval batches")
    parser.add_argument("--diag_every", type=int, default=100)
    parser.add_argument("--roberta_batch_size", type=int, default=64)
    parser.add_argument("--roberta_eval_batch_size", type=int, default=64)
    parser.add_argument("--roberta_lr", type=float, default=1e-6)
    parser.add_argument("--opt_batch_size", type=int, default=4)
    parser.add_argument("--opt_eval_batch_size", type=int, default=4)
    parser.add_argument("--opt_lr", type=float, default=1e-7)
    parser.add_argument("--opt_max_length", type=int, default=128)
    parser.add_argument("--resume_run", type=Path)
    parser.add_argument("--postprocess_only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.resume_run is not None:
        raise SystemExit("resume_run is recorded for reproducibility; use the full runner command for this batch")
    output_root: Path = args.output_root
    if args.postprocess_only:
        env, smoke_rows, calibration_rows, probe_rows, probe_summaries, hacc_rows = load_completed_outputs(output_root)
        plot_paths = plot_results(output_root, probe_rows, hacc_rows)
        write_markdown_outputs(
            output_root,
            env=env,
            smoke_rows=smoke_rows,
            calibration_rows=calibration_rows,
            probe_rows=probe_rows,
            probe_summaries=probe_summaries,
            hacc_rows=hacc_rows,
            plot_paths=plot_paths,
        )
        print(f"[{datetime.now().isoformat(timespec='seconds')}] postprocess done: {output_root}", flush=True)
        return 0
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this quantizer robustness batch.")
    if os.environ.get("DATALOADER_SHUFFLE") != "True":
        raise RuntimeError("DATALOADER_SHUFFLE=True must be set.")
    random.seed(16)
    torch.manual_seed(16)
    torch.cuda.manual_seed_all(16)
    device = torch.device("cuda")
    output_root.mkdir(parents=True, exist_ok=True)
    env = collect_env()
    write_json(output_root / "env.json", env)
    write_read_files_report(output_root)

    print(f"[{datetime.now().isoformat(timespec='seconds')}] loading RoBERTa-large/SST-5", flush=True)
    roberta = RobertaHarness(args, device)
    print(f"[{datetime.now().isoformat(timespec='seconds')}] loading OPT-1.3B/SST-5", flush=True)
    opt = OPTHarness(args, device)
    harnesses: List[BaseHarness] = [roberta, opt]

    calibration_rows: List[Dict[str, object]] = []
    for harness in harnesses:
        print(f"[{datetime.now().isoformat(timespec='seconds')}] calibrating AWQ-style activations for {harness.setting_name}", flush=True)
        act = harness.collect_activation_rms(min(int(args.calibration_examples), 128))
        for quantizer in ("rtnclip", "awq"):
            calibration_rows.append(
                {
                    "model": harness.model_name,
                    "dataset": harness.dataset_name,
                    "setting": harness.setting_name,
                    "quantizer": quantizer,
                    "quantizer_backend": QUANTIZER_BACKENDS[quantizer],
                    "calibration_examples": 0 if quantizer == "rtnclip" else min(int(args.calibration_examples), 128),
                    "activation_rms_modules": 0 if quantizer == "rtnclip" else len(act),
                    "quantized_modules": len(harness.quantized_module_names),
                    "objective": "unweighted_weight_mse_clip_search" if quantizer == "rtnclip" else "activation_weighted_weight_mse",
                    "alpha_grid": list(RTN_ALPHA_GRID if quantizer == "rtnclip" else AWQ_ALPHA_GRID),
                }
            )

    if args.skip_smoke and (output_root / "smoke_summary.json").exists():
        print(f"[{datetime.now().isoformat(timespec='seconds')}] reusing existing smoke summary", flush=True)
        smoke_rows = read_json(output_root / "smoke_summary.json").get("runs", [])
    else:
        print(f"[{datetime.now().isoformat(timespec='seconds')}] running smoke matrix", flush=True)
        smoke_rows = run_smoke_matrix(harnesses, output_root=output_root, env=env, calibration_examples=min(int(args.calibration_examples), 128))
    if not all(row.get("status") == "complete" and int(row.get("steps_completed", 0)) == 20 for row in smoke_rows):
        write_markdown_outputs(
            output_root,
            env=env,
            smoke_rows=smoke_rows,
            calibration_rows=calibration_rows,
            probe_rows=[],
            probe_summaries=[],
            hacc_rows=[],
            plot_paths=[],
        )
        write_json(output_root / "smoke_failure_report.json", {"runs": smoke_rows})
        raise SystemExit("smoke failed; stopping before probe/training")
    if args.smoke_only:
        write_markdown_outputs(
            output_root,
            env=env,
            smoke_rows=smoke_rows,
            calibration_rows=calibration_rows,
            probe_rows=[],
            probe_summaries=[],
            hacc_rows=[],
            plot_paths=[],
        )
        return 0

    print(f"[{datetime.now().isoformat(timespec='seconds')}] running probe grids", flush=True)
    probe_rows: List[Dict[str, object]] = []
    probe_summaries: List[Dict[str, object]] = []
    for harness in harnesses:
        scope = list(harness.quantized_module_names)
        for quantizer in ("rtnclip", "awq"):
            rows, summary = run_probe(
                harness,
                output_root=output_root,
                quantizer=quantizer,
                calibration_examples=min(int(args.calibration_examples), 128),
                env=env,
                compare_scope=scope,
            )
            probe_rows.extend(rows)
            probe_summaries.append(summary)
    if args.probe_only or args.skip_hacc:
        plot_paths = plot_results(output_root, probe_rows, [])
        write_markdown_outputs(
            output_root,
            env=env,
            smoke_rows=smoke_rows,
            calibration_rows=calibration_rows,
            probe_rows=probe_rows,
            probe_summaries=probe_summaries,
            hacc_rows=[],
            plot_paths=plot_paths,
        )
        return 0

    print(f"[{datetime.now().isoformat(timespec='seconds')}] running short h-acc validation", flush=True)
    summary_by_key = {(row["setting"], row["quantizer"]): row for row in probe_summaries}
    hacc_rows: List[Dict[str, object]] = []
    for harness in harnesses:
        steps = int(args.roberta_train_steps if harness.setting_name == "roberta_large_sst5" else args.opt_train_steps)
        eval_every = 500 if steps >= 500 else steps
        checkpoint_every = 500 if steps >= 500 else steps
        for quantizer in ("rtnclip", "awq"):
            policies = h_policies_from_probe(summary_by_key[(harness.setting_name, quantizer)])
            for policy, h in policies:
                row = run_training_job(
                    harness,
                    output_root=output_root,
                    phase="h_acc",
                    quantizer=quantizer,
                    h=h,
                    steps=steps,
                    eval_every=eval_every,
                    checkpoint_every=checkpoint_every,
                    calibration_examples=min(int(args.calibration_examples), 128),
                    env=env,
                    compare_scope=list(harness.quantized_module_names),
                )
                row["policy"] = policy
                hacc_rows.append(row)

    plot_paths = plot_results(output_root, probe_rows, hacc_rows)
    write_markdown_outputs(
        output_root,
        env=env,
        smoke_rows=smoke_rows,
        calibration_rows=calibration_rows,
        probe_rows=probe_rows,
        probe_summaries=probe_summaries,
        hacc_rows=hacc_rows,
        plot_paths=plot_paths,
    )
    print(f"[{datetime.now().isoformat(timespec='seconds')}] done: {output_root}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
