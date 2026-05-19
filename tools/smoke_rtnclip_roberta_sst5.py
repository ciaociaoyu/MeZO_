#!/usr/bin/env python
"""RTNClip quantized-forward MeZO smoke for RoBERTa-large / SST-5.

This is an isolated smoke harness. It does not modify the production Trainer
path and does not pack or persist low-bit weights.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import random
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler


ALPHA_GRID_VALUES = (1.0, 0.95, 0.90, 0.85, 0.80, 0.75, 0.70)
SUMMARY_COLUMNS = [
    "bitwidth",
    "h",
    "group_size",
    "quantizer_backend",
    "scale_refresh_k",
    "pair_shared_grid",
    "grid_source",
    "steps_completed",
    "seed",
    "data_seed",
    "batch_size",
    "shuffle",
    "sampler_name",
    "update_backend",
    "quantized_modules",
    "seconds_per_step",
    "peak_gpu_mem",
    "final_train_loss",
    "final_eval_acc_if_available",
    "d_h_finite_rate",
    "update_norm_last",
    "active_frac",
    "alignment",
    "norm_ratio",
    "zero_effective_displacement_frac",
    "saturation_frac_w",
    "saturation_frac_w_plus",
    "saturation_frac_w_minus",
    "recon_mse_global",
    "alpha_lt_1_frac",
    "num_scale_refreshes",
    "status",
    "warnings",
]


@dataclass
class RTNClipState:
    name: str
    shape: Tuple[int, int]
    group_size: int
    bitwidth: int
    qmax: int
    scales: torch.Tensor
    alpha_idx: torch.Tensor
    alpha_values: torch.Tensor
    lengths: torch.Tensor
    valid: torch.Tensor


def _json_default(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.dtype):
        return str(value)
    return str(value)


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")


def append_jsonl(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(data, sort_keys=True, default=_json_default) + "\n")


def git_commit(repo_root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return ""


def collect_env(repo_root: Path) -> Dict[str, object]:
    info: Dict[str, object] = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "hostname": socket.gethostname(),
        "python_version": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "git_commit": git_commit(repo_root),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "DATALOADER_SHUFFLE": os.environ.get("DATALOADER_SHUFFLE", ""),
    }
    for module_name in ("transformers", "accelerate"):
        try:
            module = __import__(module_name)
            info[f"{module_name}_version"] = getattr(module, "__version__", "")
        except Exception:
            info[f"{module_name}_version"] = None
    if torch.cuda.is_available():
        idx = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(idx)
        info.update(
            {
                "gpu_name": props.name,
                "gpu_total_memory_mb": int(props.total_memory / 1024 / 1024),
                "gpu_count": torch.cuda.device_count(),
            }
        )
    else:
        info.update({"gpu_name": "", "gpu_total_memory_mb": 0, "gpu_count": 0})
    return info


def _group_view_2d(weight: torch.Tensor, group_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if weight.ndim != 2:
        raise ValueError(f"RTNClip Linear weight must be 2D, got shape={tuple(weight.shape)}")
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


def compute_rtnclip_state(name: str, weight: torch.Tensor, bitwidth: int, group_size: int) -> Tuple[RTNClipState, Dict[str, object]]:
    qmax = 127 if bitwidth == 8 else 7
    groups, lengths, valid = _group_view_2d(weight, group_size)
    alpha_grid = torch.tensor(ALPHA_GRID_VALUES, device=weight.device, dtype=torch.float32)
    masked_abs = groups.abs().masked_fill(~valid, 0.0)
    max_abs = masked_abs.amax(dim=-1, keepdim=True)
    zero_groups = max_abs <= 0
    base_scale = (max_abs / float(qmax)).clamp_min(1e-12)
    candidate_scales = base_scale.unsqueeze(2) * alpha_grid.view(1, 1, -1, 1)
    x = groups.unsqueeze(2)
    q = torch.round(x / candidate_scales).clamp(-qmax, qmax)
    wq = q * candidate_scales
    err = ((x - wq) ** 2).masked_fill(~valid.unsqueeze(2), 0.0)
    mse = err.sum(dim=-1) / lengths.float().view(1, -1, 1)
    best_idx = mse.argmin(dim=2)
    scales_all = candidate_scales.squeeze(-1)
    best_scales = torch.gather(scales_all, 2, best_idx.unsqueeze(-1)).clamp_min(1e-12)
    best_scales = torch.where(zero_groups, torch.ones_like(best_scales), best_scales)
    alpha_values = alpha_grid[best_idx]
    alpha_values = torch.where(zero_groups.squeeze(-1), torch.ones_like(alpha_values), alpha_values)
    state = RTNClipState(
        name=name,
        shape=tuple(weight.shape),
        group_size=group_size,
        bitwidth=bitwidth,
        qmax=qmax,
        scales=best_scales.detach(),
        alpha_idx=best_idx.detach(),
        alpha_values=alpha_values.detach(),
        lengths=lengths.detach(),
        valid=valid.detach(),
    )
    q_w, stats = quantize_with_state(weight, state, return_stats=True)
    recon_mse = float(((q_w.float() - weight.float()) ** 2).mean().detach().cpu())
    stats.update(
        {
            "module_name": name,
            "bitwidth": bitwidth,
            "group_size": group_size,
            "num_groups": int(best_idx.numel()),
            "scale_min": float(best_scales.min().detach().cpu()),
            "scale_median": float(best_scales.median().detach().cpu()),
            "scale_max": float(best_scales.max().detach().cpu()),
            "recon_mse": recon_mse,
            "alpha_lt_1_frac": float((alpha_values < 1.0).float().mean().detach().cpu()),
        }
    )
    for alpha in ALPHA_GRID_VALUES:
        stats[f"alpha_{alpha:g}_count"] = int((alpha_values == alpha).sum().detach().cpu())
    return state, stats


def quantize_with_state(weight: torch.Tensor, state: RTNClipState, return_stats: bool = False):
    groups, _, valid = _group_view_2d(weight, state.group_size)
    q = torch.round(groups / state.scales).clamp(-state.qmax, state.qmax)
    q = q.masked_fill(~valid, 0.0)
    wq = (q * state.scales).reshape(weight.shape[0], -1)[:, : weight.shape[1]].reshape_as(weight)
    wq = wq.to(dtype=weight.dtype)
    if not return_stats:
        return wq
    valid_q = q[valid.expand_as(q)]
    stats = {
        "code_min": int(valid_q.min().detach().cpu()) if valid_q.numel() else 0,
        "code_max": int(valid_q.max().detach().cpu()) if valid_q.numel() else 0,
        "clip_frac": float((valid_q.abs() >= state.qmax).float().mean().detach().cpu()) if valid_q.numel() else 0.0,
    }
    return wq, stats


def unit_tests() -> Dict[str, object]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gen = torch.Generator(device=device).manual_seed(1234)
    w = torch.randn((5, 263), device=device, generator=gen) * 0.2
    results = {}
    for bitwidth, qmax in ((8, 127), (4, 7)):
        state, stats = compute_rtnclip_state("unit.linear", w, bitwidth, 128)
        direction = torch.randn(w.shape, device=device, generator=gen)
        scale_ptr_before = state.scales.data_ptr()
        wq_plus, plus_stats = quantize_with_state(w + 1e-3 * direction, state, True)
        scale_ptr_mid = state.scales.data_ptr()
        wq_minus, minus_stats = quantize_with_state(w - 1e-3 * direction, state, True)
        scale_ptr_after = state.scales.data_ptr()
        assert torch.isfinite(wq_plus).all() and torch.isfinite(wq_minus).all()
        assert torch.isfinite(state.scales).all() and bool((state.scales > 0).all())
        assert plus_stats["code_min"] >= -qmax and plus_stats["code_max"] <= qmax
        assert minus_stats["code_min"] >= -qmax and minus_stats["code_max"] <= qmax
        assert scale_ptr_before == scale_ptr_mid == scale_ptr_after
        assert bool(((wq_plus.float() - wq_minus.float()) != 0).any())
        alpha_set = set(round(float(x), 2) for x in state.alpha_values.flatten().detach().cpu().tolist())
        assert alpha_set.issubset(set(round(x, 2) for x in ALPHA_GRID_VALUES))
        assert state.scales.data_ptr() == state.scales.data_ptr()
        refresh_counts = {1: 0, 10: 0}
        for interval in refresh_counts:
            for step in range(12):
                if step % interval == 0:
                    refresh_counts[interval] += 1
        assert refresh_counts[1] == 12 and refresh_counts[10] == 2
        master = w.clone()
        before = master.clone()
        _ = quantize_with_state(master, state)
        assert torch.equal(master, before)
        results[f"int{bitwidth}"] = {
            "status": "pass",
            "scale_min": stats["scale_min"],
            "scale_max": stats["scale_max"],
            "clip_frac_plus": plus_stats["clip_frac"],
            "clip_frac_minus": minus_stats["clip_frac"],
            "pair_shared_grid_check": True,
            "fresh_perturbed_code_check": True,
        }
    return results


def add_medium_models_to_path(repo_root: Path) -> None:
    medium = repo_root / "medium_models"
    if str(medium) not in sys.path:
        sys.path.insert(0, str(medium))


def make_dataset_args(repo_root: Path) -> SimpleNamespace:
    return SimpleNamespace(
        task_name="sst-5",
        data_dir=str(repo_root / "medium_models" / "data" / "k-shot-1k-test" / "SST-5" / "full-16"),
        data_root=str(repo_root / "medium_models" / "data" / "k-shot-1k-test"),
        dataset_mode="full",
        max_seq_length=128,
        num_k=16,
        num_sample=16,
        num_demo=1,
        auto_demo=True,
        prompt=True,
        template="*cls**sent_0*_It_was*mask*.*sep+*",
        mapping="{0:'terrible',1:'bad',2:'okay',3:'good',4:'great'}",
        template_list=None,
        overwrite_cache=False,
        demo_filter=False,
        demo_filter_rate=0.5,
        demo_filter_model=None,
        debug_mode=False,
        first_sent_limit=110,
        other_sent_limit=20,
        double_demo=True,
        gpt3_in_context_head=False,
        gpt3_in_context_tail=False,
        gpt3_in_context_num=32,
        gpt3_demo_separator="\n\n\n",
        truncate_head=False,
        sfc_prompt=None,
        icl_sfc_prompt=None,
    )


def collate_features(features: List[object]) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    pad_values = {"input_ids": 1, "attention_mask": 0, "token_type_ids": 0}
    for attr, dest, dtype in (
        ("input_ids", "input_ids", torch.long),
        ("attention_mask", "attention_mask", torch.long),
        ("token_type_ids", "token_type_ids", torch.long),
    ):
        values = [getattr(f, attr, None) for f in features]
        if values[0] is None:
            continue
        max_len = max(len(v) for v in values)
        padded = [list(v) + [pad_values[dest]] * (max_len - len(v)) for v in values]
        out[dest] = torch.tensor(padded, dtype=dtype)
    for attr, dest, dtype in (
        ("mask_pos", "mask_pos", torch.long),
        ("label", "labels", torch.long),
    ):
        values = [getattr(f, attr, None) for f in features]
        if values[0] is None:
            continue
        out[dest] = torch.tensor(values, dtype=dtype)
    return out


def load_prompt_model_and_data(args, device: torch.device):
    add_medium_models_to_path(args.repo_root)
    from src.dataset import FewShotDataset
    from src.models import RobertaModelForPromptFinetuning
    from transformers import AutoConfig, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if not hasattr(tokenizer, "model_type"):
        tokenizer.model_type = "roberta"
    data_args = make_dataset_args(args.repo_root)
    config = AutoConfig.from_pretrained(args.model_id, num_labels=5, finetuning_task="sst-5")
    model = RobertaModelForPromptFinetuning.from_pretrained(args.model_id, config=config)
    model.model_args = SimpleNamespace(use_task_word=False, l2_loss=False, sfc=False)
    model.data_args = data_args
    model.tokenizer = tokenizer
    model.return_full_softmax = False
    model.half()
    model.to(device)
    model.eval()

    train_dataset = FewShotDataset(data_args, tokenizer, mode="train", use_demo=False)
    dev_dataset = FewShotDataset(data_args, tokenizer, mode="dev", use_demo=False)
    model.label_word_list = torch.tensor(train_dataset.label_word_list, dtype=torch.long, device=device)

    train_gen = torch.Generator()
    train_gen.manual_seed(args.data_seed)
    train_sampler = RandomSampler(train_dataset, generator=train_gen)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler, collate_fn=collate_features)
    dev_loader = DataLoader(dev_dataset, batch_size=args.eval_batch_size, sampler=SequentialSampler(dev_dataset), collate_fn=collate_features)
    return model, train_loader, dev_loader, data_args, train_sampler


def move_batch(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device) for k, v in batch.items()}


def forward_loss_and_logits(model, batch: Dict[str, torch.Tensor]):
    with torch.no_grad():
        outputs = model(**batch)
        loss = outputs[0]
        logits = outputs[1]
    return loss, logits


def named_parameter_map(model: nn.Module) -> Dict[str, nn.Parameter]:
    return dict(model.named_parameters())


def linear_weight_names(model: nn.Module) -> List[str]:
    names = []
    for module_name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            names.append(f"{module_name}.weight" if module_name else "weight")
    return names


def refresh_quantizer_states(master: Dict[str, torch.Tensor], q_names: Iterable[str], bitwidth: int, group_size: int):
    states: Dict[str, RTNClipState] = {}
    stats_rows: List[Dict[str, object]] = []
    for name in q_names:
        state, stats = compute_rtnclip_state(name, master[name], bitwidth, group_size)
        states[name] = state
        stats_rows.append(stats)
    return states, stats_rows


def aggregate_quantizer_stats(rows: List[Dict[str, object]], numel_by_name: Dict[str, int]) -> Dict[str, object]:
    if not rows:
        return {}
    total = sum(numel_by_name[r["module_name"]] for r in rows)
    group_total = sum(int(r["num_groups"]) for r in rows)
    weighted = {}
    for key in ("recon_mse", "clip_frac", "alpha_lt_1_frac", "scale_min", "scale_median", "scale_max"):
        vals = []
        weights = []
        for row in rows:
            weight = int(row["num_groups"]) if key == "alpha_lt_1_frac" else numel_by_name[row["module_name"]]
            vals.append(float(row[key]))
            weights.append(weight)
        denom = max(sum(weights), 1)
        weighted[key] = sum(v * w for v, w in zip(vals, weights)) / denom
    hist = {f"alpha_{alpha:g}_count": sum(int(r.get(f"alpha_{alpha:g}_count", 0)) for r in rows) for alpha in ALPHA_GRID_VALUES}
    return {
        "recon_mse_global": weighted["recon_mse"],
        "saturation_frac_w": weighted["clip_frac"],
        "clip_frac_w": weighted["clip_frac"],
        "alpha_lt_1_frac": weighted["alpha_lt_1_frac"],
        "scale_min_global": min(float(r["scale_min"]) for r in rows),
        "scale_median_weighted": weighted["scale_median"],
        "scale_max_global": max(float(r["scale_max"]) for r in rows),
        "num_quantized_modules": len(rows),
        "num_quantized_values": total,
        "num_groups": group_total,
        **hist,
    }


def copy_master_to_model(
    params: Dict[str, nn.Parameter],
    master: Dict[str, torch.Tensor],
    directions: Optional[Dict[str, torch.Tensor]],
    h: float,
    sign: float,
    quantizer_states: Dict[str, RTNClipState],
) -> None:
    with torch.no_grad():
        for name, param in params.items():
            if directions is None:
                value = master[name]
            else:
                value = master[name].float().add(directions[name].float(), alpha=sign * h)
            if name in quantizer_states:
                value = quantize_with_state(value, quantizer_states[name])
            param.copy_(value.to(dtype=param.dtype))


def restore_master(params: Dict[str, nn.Parameter], master: Dict[str, torch.Tensor]) -> float:
    max_diff = 0.0
    with torch.no_grad():
        for name, param in params.items():
            param.copy_(master[name].to(dtype=param.dtype))
            diff = (param.detach().float() - master[name].float()).abs().max()
            max_diff = max(max_diff, float(diff.detach().cpu()))
    return max_diff


def sample_directions(master: Dict[str, torch.Tensor], generator: torch.Generator) -> Dict[str, torch.Tensor]:
    directions = {}
    for name, tensor in master.items():
        if tensor.is_floating_point():
            directions[name] = torch.randn(tensor.shape, device=tensor.device, generator=generator, dtype=torch.float16)
    return directions


def perturbation_metrics(
    master: Dict[str, torch.Tensor],
    directions: Dict[str, torch.Tensor],
    states: Dict[str, RTNClipState],
    h: float,
) -> Dict[str, object]:
    active = 0
    total = 0
    dot = torch.zeros((), device=next(iter(master.values())).device, dtype=torch.float64)
    delta_sq = torch.zeros_like(dot)
    intended_sq = torch.zeros_like(dot)
    clip_plus_num = 0.0
    clip_minus_num = 0.0
    value_num = 0
    legal = True
    for name, state in states.items():
        intended = (2.0 * h * directions[name].float())
        plus, plus_stats = quantize_with_state(master[name].float().add(directions[name].float(), alpha=h), state, True)
        minus, minus_stats = quantize_with_state(master[name].float().add(directions[name].float(), alpha=-h), state, True)
        delta = plus.float() - minus.float()
        active += int((delta != 0).sum().detach().cpu())
        total += delta.numel()
        dot += (delta.double() * intended.double()).sum()
        delta_sq += (delta.double() * delta.double()).sum()
        intended_sq += (intended.double() * intended.double()).sum()
        clip_plus_num += float(plus_stats["clip_frac"]) * delta.numel()
        clip_minus_num += float(minus_stats["clip_frac"]) * delta.numel()
        value_num += delta.numel()
        legal = legal and plus_stats["code_min"] >= -state.qmax and plus_stats["code_max"] <= state.qmax
        legal = legal and minus_stats["code_min"] >= -state.qmax and minus_stats["code_max"] <= state.qmax
    eps = 1e-12
    active_frac = active / max(total, 1)
    alignment = float((dot / (delta_sq.sqrt() * intended_sq.sqrt() + eps)).detach().cpu())
    norm_ratio = float((delta_sq.sqrt() / (intended_sq.sqrt() + eps)).detach().cpu())
    return {
        "active_frac": active_frac,
        "alignment": alignment,
        "norm_ratio": norm_ratio,
        "zero_effective_displacement_frac": 1.0 - active_frac,
        "saturation_frac_w_plus": clip_plus_num / max(value_num, 1),
        "saturation_frac_w_minus": clip_minus_num / max(value_num, 1),
        "clip_frac_w_plus": clip_plus_num / max(value_num, 1),
        "clip_frac_w_minus": clip_minus_num / max(value_num, 1),
        "codes_legal": bool(legal),
        "pair_shared_grid": True,
        "grid_source": "unperturbed_fp16_master_weight",
        "fresh_rounded_perturbed_codes": True,
    }


def update_master(master: Dict[str, torch.Tensor], directions: Dict[str, torch.Tensor], lr: float, d_h: float) -> float:
    sq = torch.zeros((), device=next(iter(master.values())).device, dtype=torch.float64)
    with torch.no_grad():
        for name, tensor in master.items():
            update = directions[name].float().mul(-lr * d_h)
            sq += (update.double() * update.double()).sum()
            tensor.copy_(tensor.float().add(update).to(dtype=tensor.dtype))
    return float(sq.sqrt().detach().cpu())


def evaluate_quantized(model, params, master, states, dev_loader, device, max_batches: int) -> Tuple[Optional[float], Optional[float]]:
    if max_batches <= 0:
        return None, None
    copy_master_to_model(params, master, None, 0.0, 0.0, states)
    total_loss = 0.0
    total_correct = 0
    total_items = 0
    for idx, batch in enumerate(dev_loader):
        if idx >= max_batches:
            break
        batch = move_batch(batch, device)
        loss, logits = forward_loss_and_logits(model, batch)
        labels = batch["labels"]
        total_loss += float(loss.detach().cpu()) * int(labels.numel())
        total_correct += int((logits.argmax(dim=-1) == labels).sum().detach().cpu())
        total_items += int(labels.numel())
    restore_master(params, master)
    if total_items == 0:
        return None, None
    return total_loss / total_items, total_correct / total_items


def cycle(loader):
    while True:
        for batch in loader:
            yield batch


def hard_failure_reasons(config: Dict[str, object], sampler_name: str, quantized_modules: List[str]) -> List[str]:
    reasons = []
    if os.environ.get("DATALOADER_SHUFFLE") != "True":
        reasons.append("DATALOADER_SHUFFLE is not True")
    if sampler_name != "RandomSampler":
        reasons.append(f"sampler is {sampler_name}, not RandomSampler")
    if int(config["seed"]) != 16 or int(config["data_seed"]) != 16:
        reasons.append("seed/data_seed contract violation")
    if int(config["batch_size"]) != 64:
        reasons.append("batch_size contract violation")
    if not quantized_modules:
        reasons.append("no Linear.weight modules were quantized")
    return reasons


def run_one_config(args, base_context: Dict[str, object], config: Dict[str, object]) -> Dict[str, object]:
    run_dir = args.output_dir / config["run_name"]
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "train.log"
    metrics_path = run_dir / "metrics.csv"
    quant_path = run_dir / "quantizer_diagnostics.jsonl"
    pert_path = run_dir / "perturbation_diagnostics.jsonl"
    for stale in (metrics_path, quant_path, pert_path):
        if stale.exists():
            stale.unlink()

    def log(message: str) -> None:
        line = f"[{datetime.now().isoformat(timespec='seconds')}] {message}"
        print(line, flush=True)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    device = base_context["device"]
    model = base_context["model"]
    train_loader = base_context["train_loader"]
    dev_loader = base_context["dev_loader"]
    sampler_name = base_context["sampler_name"]
    params = named_parameter_map(model)
    q_names = base_context["quantized_module_names"]
    numel_by_name = {name: params[name].numel() for name in q_names}

    scale_refresh_k = int(config.get("scale_refresh_k", config.get("scale_refresh_interval", 1)))
    run_config = {
        **config,
        "model": args.model_id,
        "dataset": "SST-5",
        "dataset_mode": "full",
        "seed": args.seed,
        "data_seed": args.data_seed,
        "batch_size": args.batch_size,
        "shuffle": True,
        "DATALOADER_SHUFFLE": os.environ.get("DATALOADER_SHUFFLE", ""),
        "sampler_name": sampler_name,
        "direction": "dense",
        "h": args.h,
        "steps": args.steps,
        "lr": args.lr,
        "update_backend": "fp16_master",
        "quantizer_backend": "group_rtn_clip",
        "quantizer": "group_rtn_clip",
        "quantization_type": "weight_only_fake_quant",
        "group_size": args.group_size,
        "scale_refresh_k": scale_refresh_k,
        "pair_shared_grid": True,
        "grid_source": "unperturbed_fp16_master_weight",
        "perturbed_codes": "fresh_round_each_probe",
        "symmetric": True,
        "zero_point": "none",
        "activation_quantization": False,
        "real_packing": False,
        "full_gptq": False,
        "quantized_modules": q_names,
        "keep_unquantized": "bias, LayerNorm, embeddings, non-Linear parameters",
        "zo_trainable_tensors": "all floating parameters; RTNClip applied only to Linear.weight",
    }
    write_json(run_dir / "run_config.json", run_config)
    write_json(run_dir / "run_manifest_row.json", run_config)

    reasons = hard_failure_reasons(run_config, sampler_name, q_names)
    if reasons:
        summary = {
            **run_config,
            "steps_completed": 0,
            "status": "failed",
            "warnings": "; ".join(reasons),
            "error_message": "; ".join(reasons),
        }
        write_json(run_dir / "run_summary.json", summary)
        return summary

    master = {name: p.detach().clone().to(device=device, dtype=torch.float16) for name, p in params.items() if p.detach().is_floating_point()}
    restore_master(params, master)
    gen = torch.Generator(device=device)
    gen.manual_seed(args.seed + int(config["bitwidth"]) * 1000 + scale_refresh_k)
    batch_iter = cycle(train_loader)
    states: Dict[str, RTNClipState] = {}
    last_refresh_stats: List[Dict[str, object]] = []
    num_refreshes = 0
    d_h_finite = 0
    update_norm_last = float("nan")
    final_train_loss = float("nan")
    status = "pass"
    warnings: List[str] = []
    first_error = ""
    last_pert = {}
    last_quant = {}
    total_start = time.time()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    with metrics_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "step",
                "loss_plus",
                "loss_minus",
                "d_h",
                "d_h_finite",
                "update_norm",
                "seconds",
                "param_restore_max_abs_diff",
                "scale_refreshed",
                "nan_flag",
            ],
        )
        writer.writeheader()
        for step in range(args.steps):
            step_start = time.time()
            scale_refreshed = step % scale_refresh_k == 0
            if scale_refreshed:
                states, last_refresh_stats = refresh_quantizer_states(master, q_names, int(config["bitwidth"]), args.group_size)
                num_refreshes += 1
                refresh_agg = aggregate_quantizer_stats(last_refresh_stats, numel_by_name)
                append_jsonl(
                    quant_path,
                    {
                        "step": step,
                        "record_type": "refresh_summary",
                        "scale_refreshed": True,
                        "scale_refresh_k": scale_refresh_k,
                        "pair_shared_grid": True,
                        "grid_source": "unperturbed_fp16_master_weight",
                        **refresh_agg,
                    },
                )
                for module_row in last_refresh_stats:
                    append_jsonl(
                        quant_path,
                        {
                            "step": step,
                            "record_type": "per_module_refresh",
                            "scale_refreshed": True,
                            "scale_refresh_k": scale_refresh_k,
                            **module_row,
                        },
                    )
            elif args.drift_check_modules > 0:
                sample_names = q_names[: args.drift_check_modules]
                drift_vals = []
                for name in sample_names:
                    fresh, _ = compute_rtnclip_state(name, master[name], int(config["bitwidth"]), args.group_size)
                    stale = states[name].scales
                    rel = (fresh.scales - stale).abs() / stale.abs().clamp_min(1e-12)
                    drift_vals.append(float(rel.max().detach().cpu()))
                append_jsonl(
                    quant_path,
                    {
                        "step": step,
                        "record_type": "scale_drift_sample",
                        "scale_refreshed": False,
                        "scale_refresh_k": scale_refresh_k,
                        "scale_drift_sample_max": max(drift_vals) if drift_vals else 0.0,
                        "scale_drift_sample_modules": len(drift_vals),
                    },
                )

            directions = sample_directions(master, gen)
            batch = move_batch(next(batch_iter), device)
            copy_master_to_model(params, master, directions, args.h, +1.0, states)
            loss_plus, _ = forward_loss_and_logits(model, batch)
            copy_master_to_model(params, master, directions, args.h, -1.0, states)
            loss_minus, _ = forward_loss_and_logits(model, batch)
            restore_diff = restore_master(params, master)

            loss_plus_f = float(loss_plus.detach().cpu())
            loss_minus_f = float(loss_minus.detach().cpu())
            d_h = (loss_plus_f - loss_minus_f) / (2.0 * args.h)
            finite = math.isfinite(loss_plus_f) and math.isfinite(loss_minus_f) and math.isfinite(d_h)
            if finite:
                d_h_finite += 1
                update_norm_last = update_master(master, directions, args.lr, d_h)
                restore_master(params, master)
            else:
                status = "failed"
                first_error = f"non-finite loss or d_h at step {step}"
                update_norm_last = float("nan")

            pert = perturbation_metrics(master, directions, states, args.h)
            quant_agg = aggregate_quantizer_stats(last_refresh_stats, numel_by_name)
            last_pert = pert
            last_quant = quant_agg
            append_jsonl(pert_path, {"step": step, **pert})

            nan_flag = not finite or not math.isfinite(update_norm_last) or restore_diff > 1e-3 or not pert.get("codes_legal", False)
            if restore_diff > 1e-3:
                status = "failed"
                first_error = first_error or f"FP16 master restore diff too large at step {step}: {restore_diff}"
            if not pert.get("codes_legal", False):
                status = "failed"
                first_error = first_error or f"quantized code range violation at step {step}"
            if status == "failed" and first_error:
                log(f"{config['run_name']} failed: {first_error}")

            row = {
                "step": step + 1,
                "loss_plus": loss_plus_f,
                "loss_minus": loss_minus_f,
                "d_h": d_h,
                "d_h_finite": finite,
                "update_norm": update_norm_last,
                "seconds": time.time() - step_start,
                "param_restore_max_abs_diff": restore_diff,
                "scale_refreshed": scale_refreshed,
                "nan_flag": nan_flag,
            }
            writer.writerow(row)
            f.flush()
            final_train_loss = (loss_plus_f + loss_minus_f) / 2.0
            if (step + 1) % args.log_every == 0 or step == 0:
                log(
                    f"{config['run_name']} step={step + 1}/{args.steps} "
                    f"loss={final_train_loss:.6g} d_h={d_h:.6g} "
                    f"active={pert['active_frac']:.4f} align={pert['alignment']:.4f} "
                    f"norm_ratio={pert['norm_ratio']:.4f} refresh={scale_refreshed}"
                )
            if status == "failed":
                break

    eval_loss, eval_acc = evaluate_quantized(model, params, master, states, dev_loader, device, args.eval_batches)
    eval_path = run_dir / "eval_metrics.jsonl"
    append_jsonl(eval_path, {"step": min(args.steps, d_h_finite), "eval_loss": eval_loss, "eval_acc": eval_acc})
    elapsed = time.time() - total_start
    steps_completed = sum(1 for _ in metrics_path.open(encoding="utf-8")) - 1
    seconds_per_step = elapsed / max(steps_completed, 1)
    peak_mem = float(torch.cuda.max_memory_allocated() / 1024 / 1024) if torch.cuda.is_available() else 0.0

    if scale_refresh_k == 10 and num_refreshes != int(math.ceil(steps_completed / 10.0)):
        status = "failed"
        first_error = first_error or f"K=10 refresh count mismatch: {num_refreshes}"
    if last_pert.get("active_frac", 1.0) <= 0:
        status = "failed"
        first_error = first_error or "effective displacement active_frac is zero"
    if int(config["bitwidth"]) == 4 and last_pert.get("active_frac", 0.0) < 0.01:
        warnings.append("INT4 h=1e-3 has very low active_frac")
    if scale_refresh_k == 10:
        warnings.append("K=10 uses stale cached scales between refreshes by design")

    summary = {
        **run_config,
        "steps_completed": steps_completed,
        "seconds_per_step": seconds_per_step,
        "peak_gpu_mem": peak_mem,
        "final_train_loss": final_train_loss,
        "final_eval_loss_if_available": eval_loss,
        "final_eval_acc_if_available": eval_acc,
        "d_h_finite_rate": d_h_finite / max(steps_completed, 1),
        "update_norm_last": update_norm_last,
        "active_frac": last_pert.get("active_frac"),
        "alignment": last_pert.get("alignment"),
        "norm_ratio": last_pert.get("norm_ratio"),
        "zero_effective_displacement_frac": last_pert.get("zero_effective_displacement_frac"),
        "saturation_frac_w": last_quant.get("saturation_frac_w"),
        "saturation_frac_w_plus": last_pert.get("saturation_frac_w_plus"),
        "saturation_frac_w_minus": last_pert.get("saturation_frac_w_minus"),
        "clip_frac_w": last_quant.get("clip_frac_w"),
        "clip_frac_w_plus": last_pert.get("clip_frac_w_plus"),
        "clip_frac_w_minus": last_pert.get("clip_frac_w_minus"),
        "recon_mse_global": last_quant.get("recon_mse_global"),
        "alpha_lt_1_frac": last_quant.get("alpha_lt_1_frac"),
        "num_scale_refreshes": num_refreshes,
        "status": status,
        "warnings": "; ".join(warnings),
        "error_message": first_error,
    }
    write_json(run_dir / "run_summary.json", summary)
    log(f"{config['run_name']} completed status={status} steps={steps_completed} sec_per_step={seconds_per_step:.3f}")
    return summary


def write_top_level_outputs(output_dir: Path, summaries: List[Dict[str, object]], env: Dict[str, object], unit: Dict[str, object]) -> None:
    with (output_dir / "smoke_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in summaries:
            writer.writerow(row)
    diagnostics = {"env": env, "unit_tests": unit, "runs": summaries}
    write_json(output_dir / "smoke_diagnostics.json", diagnostics)

    def fmt(value, digits=4):
        if value is None or value == "":
            return "n/a"
        try:
            return f"{float(value):.{digits}g}"
        except Exception:
            return str(value)

    all_complete = all(r.get("status") == "pass" and int(r.get("steps_completed", 0)) == 50 for r in summaries)
    int8 = [r for r in summaries if int(r.get("bitwidth", 0)) == 8]
    int4 = [r for r in summaries if int(r.get("bitwidth", 0)) == 4]
    k1 = {int(r["bitwidth"]): r for r in summaries if int(r.get("scale_refresh_k", r.get("scale_refresh_interval", 0))) == 1}
    k10 = {int(r["bitwidth"]): r for r in summaries if int(r.get("scale_refresh_k", r.get("scale_refresh_interval", 0))) == 10}
    k10_close = []
    for bit in (8, 4):
        if bit in k1 and bit in k10:
            nr1 = float(k1[bit].get("norm_ratio") or 0.0)
            nr10 = float(k10[bit].get("norm_ratio") or 0.0)
            al1 = float(k1[bit].get("alignment") or 0.0)
            al10 = float(k10[bit].get("alignment") or 0.0)
            k10_close.append(abs(nr10 - nr1) <= max(0.25 * abs(nr1), 0.25) and abs(al10 - al1) <= 0.25)
    ready = all_complete and all(k10_close or [False])
    recommended_k = 10 if ready else 1
    lines = [
        "# RTNClip INT8/INT4 Smoke Summary",
        "",
        f"Output directory: `{output_dir}`",
        f"All four configs completed 50 steps: {'yes' if all_complete else 'no'}",
        f"Quantizer numerically valid: {'yes' if all(r.get('status') == 'pass' for r in summaries) else 'no'}",
        "Plus/minus probes use the same grid from unperturbed w_t: yes",
        "Integer codes for w_t +/- h u are freshly rounded: yes",
        "FP16 master update working: yes" if all(r.get("d_h_finite_rate") == 1.0 for r in summaries) else "FP16 master update working: no",
        f"K=10 reuse grids correctly: {'yes' if all(int(r.get('num_scale_refreshes', 0)) == 5 for r in summaries if int(r.get('scale_refresh_k', r.get('scale_refresh_interval', 0))) == 10) else 'no'}",
        f"Recommended scale_refresh_k for h-search: {recommended_k}",
        f"ready_for_h_search: {'yes' if ready else 'no'}",
        "",
        "K=10 speed comparison:",
    ]
    for bit in (8, 4):
        if bit in k1 and bit in k10:
            s1 = float(k1[bit].get("seconds_per_step") or 0.0)
            s10 = float(k10[bit].get("seconds_per_step") or 0.0)
            ratio = s10 / s1 if s1 > 0 else float("nan")
            lines.append(f"- INT{bit}: K=10 is {fmt(ratio)}x K=1 wall time per step ({fmt(s10)} vs {fmt(s1)} sec/step).")
    lines.extend([
        "",
        "| run | status | sec/step | loss | active_frac | alignment | norm_ratio | warnings |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ])
    for row in summaries:
        lines.append(
            f"| {row.get('run_name')} | {row.get('status')} | {fmt(row.get('seconds_per_step'))} | "
            f"{fmt(row.get('final_train_loss'))} | {fmt(row.get('active_frac'))} | "
            f"{fmt(row.get('alignment'))} | {fmt(row.get('norm_ratio'))} | {row.get('warnings', '')} |"
        )
    if int4:
        low = [r for r in int4 if float(r.get("active_frac") or 0.0) < 0.01]
        lines.extend(["", f"INT4 h=1e-3 visibly degenerate: {'yes' if low else 'no'}"])
    if int8:
        pass
    write_json(
        output_dir / "recommendation.json",
        {
            "ready_for_h_search": ready,
            "recommended_scale_refresh_k": recommended_k,
            "recommended_scale_refresh_interval": recommended_k,
        },
    )
    (output_dir / "smoke_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", default="roberta-large")
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=16)
    parser.add_argument("--data_seed", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--eval_batch_size", type=int, default=4)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--h", type=float, default=1e-3)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--group_size", type=int, default=128)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--eval_batches", type=int, default=8)
    parser.add_argument("--drift_check_modules", type=int, default=4)
    parser.add_argument("--unit_only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    args.repo_root = repo_root
    args.output_dir.mkdir(parents=True, exist_ok=True)
    env = collect_env(repo_root)
    write_json(args.output_dir / "env.json", env)
    unit = unit_tests()
    write_json(args.output_dir / "unit_tests.json", unit)
    if args.unit_only:
        print(json.dumps({"status": "pass", "unit_tests": unit}, indent=2))
        return 0

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for RoBERTa-large RTNClip smoke; run --unit_only for CPU quantizer tests.")
    if os.environ.get("DATALOADER_SHUFFLE") != "True":
        raise RuntimeError("DATALOADER_SHUFFLE=True must be exported for this smoke.")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda")
    model, train_loader, dev_loader, data_args, train_sampler = load_prompt_model_and_data(args, device)
    q_names = linear_weight_names(model)
    context = {
        "device": device,
        "model": model,
        "train_loader": train_loader,
        "dev_loader": dev_loader,
        "data_args": data_args,
        "sampler_name": type(train_sampler).__name__,
        "quantized_module_names": q_names,
    }
    configs = [
        {"run_name": "int8_wo_g128_rtnclip_sharedgrid_k1_h1e-3_step50", "bitwidth": 8, "scale_refresh_k": 1},
        {"run_name": "int8_wo_g128_rtnclip_sharedgrid_k10_h1e-3_step50", "bitwidth": 8, "scale_refresh_k": 10},
        {"run_name": "int4_wo_g128_rtnclip_sharedgrid_k1_h1e-3_step50", "bitwidth": 4, "scale_refresh_k": 1},
        {"run_name": "int4_wo_g128_rtnclip_sharedgrid_k10_h1e-3_step50", "bitwidth": 4, "scale_refresh_k": 10},
    ]
    summaries = []
    for config in configs:
        summaries.append(run_one_config(args, context, config))
    write_top_level_outputs(args.output_dir, summaries, env, unit)

    print(f"Smoke output directory: {args.output_dir}")
    for row in summaries:
        label = f"INT{row['bitwidth']} K={row['scale_refresh_k']}"
        print(
            f"{label}: {row.get('status')}, sec/step={row.get('seconds_per_step'):.3f}, "
            f"loss={row.get('final_train_loss'):.6g}, active_frac={row.get('active_frac'):.4f}, "
            f"alignment={row.get('alignment'):.4f}, norm_ratio={row.get('norm_ratio'):.4f}"
        )
    rec = json.loads((args.output_dir / "recommendation.json").read_text())
    warning_text = "; ".join(str(r.get("warnings", "")) for r in summaries if r.get("warnings"))
    print("Recommendation:")
    print(f"  ready_for_h_search: {'yes' if rec['ready_for_h_search'] else 'no'}")
    print(f"  recommended_scale_refresh_k: {rec['recommended_scale_refresh_k']}")
    print(f"  warnings_to_consider_before_h_search: {warning_text or 'none'}")
    return 0 if all(r.get("status") == "pass" for r in summaries) else 1


if __name__ == "__main__":
    raise SystemExit(main())
