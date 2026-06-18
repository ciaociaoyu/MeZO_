#!/usr/bin/env python
"""Interval-aware perturbation-radius probe for low-bit ZO/MeZO.

This tool measures the actual shared-grid low-bit displacement

    Delta_Q(h, u) = Q_t(w + h u) - Q_t(w - h u)

using the project's RTNClip quantizer state.  It is intentionally a probe:
it does not train, does not mutate model weights, and defaults to cached-only
model loading to avoid accidental large downloads.
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
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn


REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_ROOT = REPO_ROOT / "tools"
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

import smoke_rtnclip_roberta_sst5 as rtnclip  # noqa: E402


DEFAULT_H_GRID = [
    1e-8,
    3e-8,
    1e-7,
    3e-7,
    1e-6,
    3e-6,
    1e-5,
    3e-5,
    1e-4,
    3e-4,
    1e-3,
    3e-3,
    1e-2,
]

GEOMETRY_COLUMNS = [
    "model",
    "task",
    "precision",
    "perturbation_mode",
    "layer",
    "h",
    "A_uniform",
    "p_active",
    "V_norm",
    "V_align",
    "p_clip",
    "jump_mean",
    "jump_median",
    "jump_zero_frac",
    "jump_one_frac",
    "jump_ge2_frac",
    "relative_disp_layer",
]

SUMMARY_COLUMNS = [
    "model",
    "task",
    "precision",
    "perturbation_mode",
    "h",
    "A_uniform",
    "p_active",
    "V_norm",
    "V_align",
    "p_clip",
    "jump_mean",
    "jump_median",
    "jump_zero_frac",
    "jump_one_frac",
    "jump_ge2_frac",
    "disp_rms",
    "relative_disp",
    "locality_proxy",
]

SELECTION_COLUMNS = [
    "model",
    "task",
    "precision",
    "perturbation_mode",
    "h_vis",
    "h_loc",
    "h_geom",
    "h_cons",
    "h_loss_star",
    "h_loss_cons",
    "default_h",
    "default_in_window",
    "window_exists",
    "window_width_log10",
    "notes",
]


@dataclass
class LayerSample:
    name: str
    weight: torch.Tensor
    rows: torch.Tensor
    cols: torch.Tensor
    group_ids: torch.Tensor
    scales: Optional[torch.Tensor]
    qmax: Optional[int]
    sample_count: int
    total_numel: int
    weight_norm_sample: float


def str2bool(value: str) -> bool:
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected boolean, got {value!r}")


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


def collect_env() -> Dict[str, object]:
    env: Dict[str, object] = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "hostname": socket.gethostname(),
        "python": sys.executable,
        "python_version": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "git_commit": git_commit(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
    }
    for module_name in ("transformers", "accelerate", "datasets"):
        try:
            module = __import__(module_name)
            env[f"{module_name}_version"] = getattr(module, "__version__", "")
        except Exception as exc:
            env[f"{module_name}_version"] = None
            env[f"{module_name}_import_error"] = str(exc)
    if torch.cuda.is_available():
        idx = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(idx)
        env.update(
            {
                "gpu_name": props.name,
                "gpu_total_memory_mb": int(props.total_memory / 1024 / 1024),
                "gpu_count": torch.cuda.device_count(),
            }
        )
    else:
        env.update({"gpu_name": "", "gpu_total_memory_mb": 0, "gpu_count": 0})
    return env


def write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def append_rows(path: Path, fieldnames: Sequence[str], rows: Iterable[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)


def load_existing_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def parse_precision(precision: str) -> Tuple[str, Optional[int]]:
    p = precision.lower()
    if p in {"int4", "4", "w4"}:
        return "int4", 4
    if p in {"int8", "8", "w8"}:
        return "int8", 8
    if p in {"fp32", "float32"}:
        return "fp32", None
    if p in {"fp16", "float16", "half"}:
        return "fp16", None
    if p in {"bf16", "bfloat16"}:
        return "bf16", None
    raise ValueError(f"Unsupported precision: {precision}")


def parse_sparse_mode(mode: str) -> Tuple[str, float]:
    m = mode.lower()
    if m == "dense":
        return "dense", 1.0
    if m.startswith("sparse_p"):
        raw = m[len("sparse_p") :].replace("p", ".")
        return m, float(raw)
    if m.startswith("sparse:"):
        return m, float(m.split(":", 1)[1])
    raise ValueError(f"Unsupported perturbation mode: {mode}")


def resolve_model_id(model: str) -> str:
    m = model.lower()
    if m in {"roberta", "roberta-large"}:
        return "roberta-large"
    if m in {"roberta-base"}:
        return "roberta-base"
    if m in {"opt", "opt-1.3b", "opt1.3b", "facebook/opt-1.3b"}:
        return "facebook/opt-1.3b"
    if m in {"opt-125m", "facebook/opt-125m"}:
        return "facebook/opt-125m"
    return model


def load_model(model_id: str, device: torch.device, allow_download: bool) -> nn.Module:
    from transformers import AutoModelForCausalLM, AutoModelForMaskedLM

    kwargs = {"local_files_only": not allow_download}
    if device.type == "cuda":
        kwargs["torch_dtype"] = torch.float16
    if "opt" in model_id.lower():
        model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    elif "roberta" in model_id.lower():
        model = AutoModelForMaskedLM.from_pretrained(model_id, **kwargs)
    else:
        try:
            model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
        except Exception:
            model = AutoModelForMaskedLM.from_pretrained(model_id, **kwargs)
    model.eval()
    model.to(device)
    return model


def iter_linear_weights(model: nn.Module) -> Iterable[Tuple[str, torch.Tensor]]:
    for module_name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            name = f"{module_name}.weight" if module_name else "weight"
            yield name, module.weight.detach()


def build_layer_samples(
    model: nn.Module,
    precision: str,
    group_size: int,
    coords_per_tensor: int,
    max_coords: int,
    seed: int,
) -> Tuple[List[LayerSample], Dict[str, Dict[str, object]], List[str]]:
    _, bits = parse_precision(precision)
    linear_weights = [(name, w) for name, w in iter_linear_weights(model) if w.ndim == 2]
    total_linear = sum(int(w.numel()) for _, w in linear_weights)
    if total_linear <= 0:
        raise RuntimeError("No nn.Linear.weight tensors found for interval probe")

    gen = torch.Generator(device="cpu").manual_seed(seed)
    samples: List[LayerSample] = []
    state_stats: Dict[str, Dict[str, object]] = {}
    notes: List[str] = []
    per_layer_budget = int(coords_per_tensor)
    if per_layer_budget * max(len(linear_weights), 1) > int(max_coords):
        per_layer_budget = max(1, int(max_coords) // max(len(linear_weights), 1))
    for idx, (name, weight) in enumerate(linear_weights):
        numel = int(weight.numel())
        if numel <= 0:
            continue
        # Keep a floor per layer but never exceed the global cap.  This gives
        # small layers a voice while keeping OPT-size models cheap.
        layer_budget = min(per_layer_budget, numel)
        flat_idx = torch.randperm(numel, generator=gen)[:layer_budget]
        rows = torch.div(flat_idx, weight.shape[1], rounding_mode="floor").to(device=weight.device, dtype=torch.long)
        cols = (flat_idx % weight.shape[1]).to(device=weight.device, dtype=torch.long)
        group_ids = torch.div(cols, int(group_size), rounding_mode="floor")
        scales = None
        qmax = None
        if bits is not None:
            state, stats = rtnclip.compute_rtnclip_state(name, weight, bits, group_size)
            state_stats[name] = stats
            scales = state.scales[rows, group_ids, 0].detach()
            qmax = int(state.qmax)
            validate_sampled_quantizer(name, weight, state, rows, cols, scales, qmax)
        values = weight[rows, cols].detach().float()
        samples.append(
            LayerSample(
                name=name,
                weight=values,
                rows=rows,
                cols=cols,
                group_ids=group_ids,
                scales=scales.detach().float() if scales is not None else None,
                qmax=qmax,
                sample_count=int(layer_budget),
                total_numel=numel,
                weight_norm_sample=float(values.double().square().sum().sqrt().detach().cpu()),
            )
        )
    sampled_total = sum(s.sample_count for s in samples)
    if sampled_total < int(max_coords):
        notes.append(f"sampled_coords={sampled_total} below max_coords={max_coords}; per_layer_budget={per_layer_budget}")
    return samples, state_stats, notes


def validate_sampled_quantizer(
    name: str,
    weight: torch.Tensor,
    state: rtnclip.RTNClipState,
    rows: torch.Tensor,
    cols: torch.Tensor,
    scales: torch.Tensor,
    qmax: int,
) -> None:
    if rows.numel() == 0:
        return
    take = min(256, rows.numel())
    r = rows[:take]
    c = cols[:take]
    s = scales[:take]
    manual = torch.round(weight[r, c].float() / s.float()).clamp(-qmax, qmax) * s.float()
    manual = manual.to(dtype=weight.dtype).float()
    full = rtnclip.quantize_with_state(weight, state)[r, c].float()
    max_diff = float((manual - full).abs().max().detach().cpu())
    if max_diff > 2e-3:
        raise RuntimeError(f"Sampled quantizer mismatch for {name}: max_diff={max_diff}")


def quantize_sampled(
    values: torch.Tensor,
    precision: str,
    scales: Optional[torch.Tensor],
    qmax: Optional[int],
) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
    p, bits = parse_precision(precision)
    if bits is not None:
        assert scales is not None and qmax is not None
        codes = torch.round(values.float() / scales.float()).clamp(-int(qmax), int(qmax))
        deq = codes * scales.float()
        clipped = codes.abs() >= int(qmax)
        return deq.float(), codes.float(), clipped
    if p == "fp32":
        deq = values.float()
    elif p == "fp16":
        deq = values.to(torch.float16).float()
    elif p == "bf16":
        deq = values.to(torch.bfloat16).float()
    else:
        raise ValueError(p)
    return deq, None, torch.zeros_like(deq, dtype=torch.bool)


def make_direction(
    shape: torch.Size,
    mode: str,
    p_active: float,
    generator: torch.Generator,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    z = torch.randn(shape, generator=generator, device=device, dtype=dtype)
    if mode == "dense":
        return z
    mask = torch.rand(shape, generator=generator, device=device) < float(p_active)
    return z * mask.to(dtype=dtype)


def empty_accumulator() -> Dict[str, object]:
    return {
        "count": 0,
        "u_sq": 0.0,
        "err_sq": 0.0,
        "active": 0.0,
        "clip": 0.0,
        "jump_sum": 0.0,
        "jump_values": [],
        "jump_zero": 0.0,
        "jump_one": 0.0,
        "jump_ge2": 0.0,
        "delta_sq_by_dir": defaultdict(float),
        "target_sq_by_dir": defaultdict(float),
        "dot_by_dir": defaultdict(float),
        "disp_sq_by_dir": defaultdict(float),
        "disp_pair_sq_by_dir": defaultdict(float),
        "weight_sq": 0.0,
    }


def update_accumulator(
    acc: Dict[str, object],
    *,
    dir_idx: int,
    u: torch.Tensor,
    delta_q: torch.Tensor,
    target: torch.Tensor,
    b: torch.Tensor,
    clipped_plus: torch.Tensor,
    clipped_minus: torch.Tensor,
    jump: Optional[torch.Tensor],
    e_plus: torch.Tensor,
    e_minus: torch.Tensor,
    weight: torch.Tensor,
) -> None:
    count = int(u.numel())
    err = b - u.float()
    acc["count"] = int(acc["count"]) + count
    acc["u_sq"] = float(acc["u_sq"]) + float(u.float().double().square().sum().detach().cpu())
    acc["err_sq"] = float(acc["err_sq"]) + float(err.double().square().sum().detach().cpu())
    acc["active"] = float(acc["active"]) + float((delta_q != 0).double().sum().detach().cpu())
    acc["clip"] = float(acc["clip"]) + float((clipped_plus | clipped_minus).double().sum().detach().cpu())
    acc["delta_sq_by_dir"][dir_idx] += float(delta_q.double().square().sum().detach().cpu())
    acc["target_sq_by_dir"][dir_idx] += float(target.double().square().sum().detach().cpu())
    acc["dot_by_dir"][dir_idx] += float((delta_q.double() * target.double()).sum().detach().cpu())
    disp_pair = e_plus.double().square().sum() + e_minus.double().square().sum()
    acc["disp_sq_by_dir"][dir_idx] += float((0.5 * disp_pair).detach().cpu())
    acc["disp_pair_sq_by_dir"][dir_idx] += float(disp_pair.detach().cpu())
    acc["weight_sq"] = float(acc["weight_sq"]) + float(weight.double().square().sum().detach().cpu())
    if jump is None:
        return
    j = jump.detach().float()
    acc["jump_sum"] = float(acc["jump_sum"]) + float(j.double().sum().detach().cpu())
    acc["jump_zero"] = float(acc["jump_zero"]) + float((j == 0).double().sum().detach().cpu())
    acc["jump_one"] = float(acc["jump_one"]) + float((j == 1).double().sum().detach().cpu())
    acc["jump_ge2"] = float(acc["jump_ge2"]) + float((j >= 2).double().sum().detach().cpu())
    # Storing every jump for 5M*128 can be too much.  Keep a deterministic
    # prefix sample; the median is only a summary diagnostic.
    values: List[float] = acc["jump_values"]  # type: ignore[assignment]
    if len(values) < 250_000:
        values.extend(j[: max(0, 250_000 - len(values))].cpu().tolist())


def update_accumulator_batch(
    acc: Dict[str, object],
    *,
    u: torch.Tensor,
    delta_q: torch.Tensor,
    target: torch.Tensor,
    b: torch.Tensor,
    clipped_plus: torch.Tensor,
    clipped_minus: torch.Tensor,
    jump: Optional[torch.Tensor],
    e_plus: torch.Tensor,
    e_minus: torch.Tensor,
    weight: torch.Tensor,
) -> None:
    """Accumulate metrics for a [n_dirs, n_coords] sampled layer block."""
    if u.ndim != 2:
        raise ValueError(f"Expected batched direction tensor [n_dirs, n_coords], got {tuple(u.shape)}")
    n_dirs, n_coords = int(u.shape[0]), int(u.shape[1])
    count = n_dirs * n_coords
    err = b.float() - u.float()
    acc["count"] = int(acc["count"]) + count
    acc["u_sq"] = float(acc["u_sq"]) + float(u.float().double().square().sum().detach().cpu())
    acc["err_sq"] = float(acc["err_sq"]) + float(err.double().square().sum().detach().cpu())
    acc["active"] = float(acc["active"]) + float((delta_q != 0).double().sum().detach().cpu())
    acc["clip"] = float(acc["clip"]) + float((clipped_plus | clipped_minus).double().sum().detach().cpu())

    delta_sq = delta_q.double().square().sum(dim=1).detach().cpu().tolist()
    target_sq = target.double().square().sum(dim=1).detach().cpu().tolist()
    dots = (delta_q.double() * target.double()).sum(dim=1).detach().cpu().tolist()
    disp_pair = e_plus.double().square().sum(dim=1) + e_minus.double().square().sum(dim=1)
    disp_sq = (0.5 * disp_pair).detach().cpu().tolist()
    disp_pair_list = disp_pair.detach().cpu().tolist()
    for dir_idx in range(n_dirs):
        acc["delta_sq_by_dir"][dir_idx] += float(delta_sq[dir_idx])
        acc["target_sq_by_dir"][dir_idx] += float(target_sq[dir_idx])
        acc["dot_by_dir"][dir_idx] += float(dots[dir_idx])
        acc["disp_sq_by_dir"][dir_idx] += float(disp_sq[dir_idx])
        acc["disp_pair_sq_by_dir"][dir_idx] += float(disp_pair_list[dir_idx])

    acc["weight_sq"] = float(acc["weight_sq"]) + n_dirs * float(weight.double().square().sum().detach().cpu())
    if jump is None:
        return
    j = jump.detach().float()
    acc["jump_sum"] = float(acc["jump_sum"]) + float(j.double().sum().detach().cpu())
    acc["jump_zero"] = float(acc["jump_zero"]) + float((j == 0).double().sum().detach().cpu())
    acc["jump_one"] = float(acc["jump_one"]) + float((j == 1).double().sum().detach().cpu())
    acc["jump_ge2"] = float(acc["jump_ge2"]) + float((j >= 2).double().sum().detach().cpu())
    values: List[float] = acc["jump_values"]  # type: ignore[assignment]
    if len(values) < 250_000:
        flat = j.reshape(-1)
        values.extend(flat[: max(0, 250_000 - len(values))].cpu().tolist())


def finalize_accumulator(acc: Dict[str, object], h: float, n_dirs: int, layer: Optional[str] = None) -> Dict[str, float]:
    count = max(int(acc["count"]), 1)
    eps = 1e-30
    delta_norms: List[float] = []
    target_norms: List[float] = []
    alignments: List[float] = []
    for dir_idx in range(n_dirs):
        d2 = float(acc["delta_sq_by_dir"].get(dir_idx, 0.0))
        t2 = float(acc["target_sq_by_dir"].get(dir_idx, 0.0))
        dot = float(acc["dot_by_dir"].get(dir_idx, 0.0))
        delta_norms.append(math.sqrt(max(d2, 0.0)))
        target_norms.append(math.sqrt(max(t2, 0.0)))
        if d2 > 0 and t2 > 0:
            alignments.append(dot / (math.sqrt(d2) * math.sqrt(t2) + eps))
    ratios = [d / (t + eps) for d, t in zip(delta_norms, target_norms) if t > 0]
    jump_values = acc["jump_values"] if acc["jump_values"] else [0.0]
    jump_sorted = sorted(float(x) for x in jump_values)
    med = jump_sorted[len(jump_sorted) // 2]
    disp_vals = [float(acc["disp_sq_by_dir"].get(i, 0.0)) for i in range(n_dirs)]
    pair_vals = [float(acc["disp_pair_sq_by_dir"].get(i, 0.0)) for i in range(n_dirs)]
    mean_disp = sum(disp_vals) / max(len(disp_vals), 1)
    weight_sq = max(float(acc["weight_sq"]) / max(n_dirs, 1), eps)
    locality_proxy = sum(v * v for v in pair_vals) / max(len(pair_vals), 1) / (16.0 * h * h + eps)
    return {
        "A_uniform": float(acc["err_sq"]) / (float(acc["u_sq"]) + eps),
        "p_active": float(acc["active"]) / count,
        "V_norm": sum(ratios) / max(len(ratios), 1),
        "V_align": sum(alignments) / max(len(alignments), 1) if alignments else float("nan"),
        "p_clip": float(acc["clip"]) / count,
        "jump_mean": float(acc["jump_sum"]) / count,
        "jump_median": med,
        "jump_zero_frac": float(acc["jump_zero"]) / count,
        "jump_one_frac": float(acc["jump_one"]) / count,
        "jump_ge2_frac": float(acc["jump_ge2"]) / count,
        "disp_rms": math.sqrt(max(mean_disp / count * max(n_dirs, 1), 0.0)),
        "relative_disp": math.sqrt(max(mean_disp, 0.0)) / math.sqrt(weight_sq),
        "relative_disp_layer": math.sqrt(max(mean_disp, 0.0)) / math.sqrt(weight_sq),
        "locality_proxy": locality_proxy,
    }


def probe_config(
    *,
    model_label: str,
    task: str,
    precision: str,
    mode: str,
    samples: List[LayerSample],
    h_grid: Sequence[float],
    n_dirs: int,
    seed: int,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    mode_name, p_active = parse_sparse_mode(mode)
    device = samples[0].weight.device if samples else torch.device("cpu")
    geometry_rows: List[Dict[str, object]] = []
    summary_rows: List[Dict[str, object]] = []
    for h in h_grid:
        global_acc = empty_accumulator()
        layer_accs = {sample.name: empty_accumulator() for sample in samples}
        gen = torch.Generator(device=device).manual_seed(seed + int(abs(math.log10(float(h))) * 10_000))
        for sample in samples:
            weight = sample.weight
            u = make_direction((int(n_dirs), int(weight.numel())), mode_name, p_active, gen, device, weight.dtype)
            weight_b = weight.unsqueeze(0)
            plus_values = weight_b + float(h) * u
            minus_values = weight_b - float(h) * u
            q_plus, code_plus, clipped_plus = quantize_sampled(plus_values, precision, sample.scales, sample.qmax)
            q_minus, code_minus, clipped_minus = quantize_sampled(minus_values, precision, sample.scales, sample.qmax)
            delta_q = q_plus.float() - q_minus.float()
            target = 2.0 * float(h) * u.float()
            b = delta_q / max(2.0 * float(h), 1e-30)
            base_q, _, _ = quantize_sampled(weight, precision, sample.scales, sample.qmax)
            e_plus = q_plus.float() - base_q.float().unsqueeze(0)
            e_minus = q_minus.float() - base_q.float().unsqueeze(0)
            jump = None
            if code_plus is not None and code_minus is not None:
                jump = (code_plus - code_minus).abs()
            update_accumulator_batch(
                global_acc,
                u=u,
                delta_q=delta_q,
                target=target,
                b=b,
                clipped_plus=clipped_plus,
                clipped_minus=clipped_minus,
                jump=jump,
                e_plus=e_plus,
                e_minus=e_minus,
                weight=weight,
            )
            update_accumulator_batch(
                layer_accs[sample.name],
                u=u,
                delta_q=delta_q,
                target=target,
                b=b,
                clipped_plus=clipped_plus,
                clipped_minus=clipped_minus,
                jump=jump,
                e_plus=e_plus,
                e_minus=e_minus,
                weight=weight,
            )
        for sample in samples:
            stats = finalize_accumulator(layer_accs[sample.name], float(h), n_dirs, sample.name)
            row = {
                "model": model_label,
                "task": task,
                "precision": precision,
                "perturbation_mode": mode,
                "layer": sample.name,
                "h": float(h),
                **stats,
            }
            geometry_rows.append(row)
        stats = finalize_accumulator(global_acc, float(h), n_dirs)
        summary_rows.append(
            {
                "model": model_label,
                "task": task,
                "precision": precision,
                "perturbation_mode": mode,
                "h": float(h),
                **stats,
            }
        )
    return geometry_rows, summary_rows


def finite_float(value: object) -> Optional[float]:
    try:
        x = float(value)
    except Exception:
        return None
    if math.isfinite(x):
        return x
    return None


def nearest_grid_h(h_grid: Sequence[float], value: float) -> float:
    return min((float(h) for h in h_grid), key=lambda h: abs(math.log10(h) - math.log10(value)))


def select_h(summary_rows: List[Dict[str, object]], h_grid: Sequence[float]) -> List[Dict[str, object]]:
    groups: Dict[Tuple[str, str, str, str], List[Dict[str, object]]] = defaultdict(list)
    for row in summary_rows:
        groups[(str(row["model"]), str(row["task"]), str(row["precision"]), str(row["perturbation_mode"]))].append(row)
    out: List[Dict[str, object]] = []
    for key, rows in groups.items():
        rows = sorted(rows, key=lambda r: float(r["h"]))
        visible = [
            r
            for r in rows
            if finite_float(r.get("A_uniform")) is not None
            and float(r["A_uniform"]) <= 0.50
            and finite_float(r.get("V_align")) is not None
            and float(r["V_align"]) >= 0.70
            and finite_float(r.get("V_norm")) is not None
            and 0.50 <= float(r["V_norm"]) <= 1.50
            and finite_float(r.get("p_active")) is not None
            and float(r["p_active"]) >= 0.05
            and finite_float(r.get("p_clip")) is not None
            and float(r["p_clip"]) <= 0.01
        ]
        local = [
            r
            for r in rows
            if finite_float(r.get("relative_disp")) is not None
            and float(r["relative_disp"]) <= 0.01
            and finite_float(r.get("p_clip")) is not None
            and float(r["p_clip"]) <= 0.01
        ]
        h_vis = float(visible[0]["h"]) if visible else float("nan")
        h_loc = float(local[-1]["h"]) if local else float("nan")
        window_exists = math.isfinite(h_vis) and math.isfinite(h_loc) and h_vis <= h_loc
        h_geom = float("nan")
        h_cons = float("nan")
        notes: List[str] = []
        if window_exists:
            h_geom = math.sqrt(h_vis * h_loc)
            inside = [r for r in rows if h_vis <= float(r["h"]) <= h_loc]
            best_a = min(float(r["A_uniform"]) for r in inside)
            candidates = [r for r in inside if float(r["A_uniform"]) <= 1.10 * best_a]
            if candidates:
                h_cons = float(candidates[0]["h"])
            else:
                h_cons = nearest_grid_h([float(r["h"]) for r in inside], h_geom)
        else:
            notes.append("no_interval_window_under_default_thresholds")
            if not visible:
                notes.append("visibility_lower_bound_not_satisfied")
            if not local:
                notes.append("locality_upper_bound_not_satisfied")
        default_h = 1e-3
        default_in_window = bool(window_exists and h_vis <= default_h <= h_loc)
        out.append(
            {
                "model": key[0],
                "task": key[1],
                "precision": key[2],
                "perturbation_mode": key[3],
                "h_vis": h_vis,
                "h_loc": h_loc,
                "h_geom": h_geom,
                "h_cons": h_cons,
                "h_loss_star": "",
                "h_loss_cons": "",
                "default_h": default_h,
                "default_in_window": default_in_window,
                "window_exists": window_exists,
                "window_width_log10": math.log10(h_loc / h_vis) if window_exists else float("nan"),
                "notes": ";".join(notes),
            }
        )
    return out


def global_precision_h(selection_rows: List[Dict[str, object]], h_grid: Sequence[float]) -> List[Dict[str, object]]:
    by_precision: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in selection_rows:
        by_precision[str(row["precision"])].append(row)
    out: List[Dict[str, object]] = []
    for precision, rows in by_precision.items():
        good = [
            r
            for r in rows
            if str(r.get("window_exists")).lower() in {"true", "1"}
            and math.isfinite(float(r["h_vis"]))
            and math.isfinite(float(r["h_loc"]))
        ]
        failed = len(rows) - len(good)
        if good:
            h_vis_global = max(float(r["h_vis"]) for r in good)
            h_loc_global = min(float(r["h_loc"]) for r in good)
            exists = h_vis_global <= h_loc_global
            if exists:
                h_precision_global = nearest_grid_h(h_grid, math.sqrt(h_vis_global * h_loc_global))
                h_precision_cons = min(float(h) for h in h_grid if h >= h_vis_global and h <= h_loc_global)
            else:
                h_precision_global = float("nan")
                h_precision_cons = float("nan")
        else:
            h_vis_global = h_loc_global = h_precision_global = h_precision_cons = float("nan")
            exists = False
        out.append(
            {
                "precision": precision,
                "h_vis_global": h_vis_global,
                "h_loc_global": h_loc_global,
                "h_precision_global": h_precision_global,
                "h_precision_cons": h_precision_cons,
                "global_window_exists": exists,
                "configs_used": len(good),
                "configs_failed": failed,
            }
        )
    return out


def write_loss_mse_placeholder(output_dir: Path, configs: Sequence[Tuple[str, str, str, str, float]]) -> None:
    path = output_dir / "loss_mse_probe.csv"
    fieldnames = ["model", "task", "precision", "perturbation_mode", "h", "nMSE_loss", "corr_loss", "normalized_curve"]
    rows = [
        {
            "model": model,
            "task": task,
            "precision": precision,
            "perturbation_mode": mode,
            "h": h,
            "nMSE_loss": "",
            "corr_loss": "",
            "normalized_curve": "",
        }
        for model, task, precision, mode, h in configs
    ]
    append_rows(path, fieldnames, rows)


def try_training_comparison(output_dir: Path, summary_rows: List[Dict[str, object]]) -> List[str]:
    missing: List[str] = []
    comparison_path = output_dir / "training_comparison_if_available.csv"
    fields = [
        "model",
        "task",
        "precision",
        "perturbation_mode",
        "h_policy",
        "h_value",
        "accuracy",
        "seed",
        "A_uniform",
        "p_active",
        "V_align",
        "relative_disp",
        "nMSE_loss",
        "inside_interval_window",
    ]
    candidates = [
        REPO_ROOT / "outputs/rtnclip_lowbit_roberta_sst5_seed16" / "int8_hsearch_summary.csv",
        REPO_ROOT / "outputs/rtnclip_lowbit_roberta_sst5_seed16" / "int4_hsearch_summary.csv",
        REPO_ROOT / "outputs/int4_standard_screen" / "int4_standard_screen_1k_summary.csv",
    ]
    rows: List[Dict[str, object]] = []
    metric_by_key = {}
    selections = select_h(summary_rows, DEFAULT_H_GRID)
    windows = {
        (r["model"], r["task"], r["precision"], r["perturbation_mode"]): r
        for r in selections
    }
    for row in summary_rows:
        metric_by_key[(row["model"], row["task"], row["precision"], row["perturbation_mode"], float(row["h"]))] = row
    for path in candidates:
        if not path.exists():
            continue
        for r in load_existing_csv(path):
            h = finite_float(r.get("h") or r.get("h_value"))
            acc = finite_float(r.get("best_eval_acc") or r.get("last_eval_acc") or r.get("accuracy"))
            if h is None or acc is None:
                continue
            precision = str(r.get("precision") or r.get("precision_mode") or "")
            if precision in {"4", "int4_rtnclip"}:
                precision = "int4"
            if precision in {"8", "int8_rtnclip"}:
                precision = "int8"
            model = str(r.get("model") or r.get("model_name") or "roberta-large")
            task = str(r.get("task") or r.get("dataset") or "sst-5")
            mode = str(r.get("perturbation_mode") or r.get("direction") or "dense")
            metric = metric_by_key.get((model, task, precision, mode, h))
            if metric is None:
                continue
            window = windows.get((model, task, precision, mode))
            inside = ""
            if window and str(window.get("window_exists")).lower() in {"true", "1"}:
                inside = float(window["h_vis"]) <= h <= float(window["h_loc"])
            rows.append(
                {
                    "model": model,
                    "task": task,
                    "precision": precision,
                    "perturbation_mode": mode,
                    "h_policy": r.get("h_policy") or r.get("run_name") or path.parent.name,
                    "h_value": h,
                    "accuracy": acc,
                    "seed": r.get("seed", ""),
                    "A_uniform": metric.get("A_uniform", ""),
                    "p_active": metric.get("p_active", ""),
                    "V_align": metric.get("V_align", ""),
                    "relative_disp": metric.get("relative_disp", ""),
                    "nMSE_loss": "",
                    "inside_interval_window": inside,
                }
            )
    append_rows(comparison_path, fields, rows)
    if not rows:
        missing.append("No matching existing training summary rows were found for training_comparison_if_available.csv.")
    return missing


def plot_outputs(output_dir: Path, summary_rows: List[Dict[str, object]], selection_rows: List[Dict[str, object]]) -> List[str]:
    missing: List[str] = []
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        return [f"matplotlib unavailable: {exc}"]

    by_config: Dict[Tuple[str, str, str, str], List[Dict[str, object]]] = defaultdict(list)
    for row in summary_rows:
        by_config[(str(row["model"]), str(row["task"]), str(row["precision"]), str(row["perturbation_mode"]))].append(row)
    for key, rows in by_config.items():
        rows = sorted(rows, key=lambda r: float(r["h"]))
        h = [float(r["h"]) for r in rows]
        fig, axes = plt.subplots(2, 2, figsize=(11, 8))
        axes = axes.flatten()
        axes[0].plot(h, [float(r["A_uniform"]) for r in rows], marker="o")
        axes[0].set_ylabel("A_uniform")
        axes[1].plot(h, [float(r["p_active"]) for r in rows], marker="o")
        axes[1].set_ylabel("p_active")
        axes[2].plot(h, [float(r["V_align"]) for r in rows], marker="o", label="V_align")
        axes[2].plot(h, [float(r["V_norm"]) for r in rows], marker="s", label="V_norm")
        axes[2].legend()
        axes[3].plot(h, [float(r["relative_disp"]) for r in rows], marker="o", label="relative_disp")
        axes[3].plot(h, [float(r["p_clip"]) for r in rows], marker="s", label="p_clip")
        axes[3].legend()
        for ax in axes:
            ax.set_xscale("log")
            ax.set_xlabel("h")
            ax.grid(True, alpha=0.3)
        title = "_".join(part.replace("/", "-").replace(" ", "") for part in key)
        fig.suptitle(" / ".join(key))
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(output_dir / f"fig_interval_metrics_{title}.{ext}")
        plt.close(fig)

    if selection_rows:
        fig, ax = plt.subplots(figsize=(10, max(4, 0.35 * len(selection_rows))))
        labels = [f"{r['precision']} {r['model']} {r['perturbation_mode']}" for r in selection_rows]
        y = list(range(len(selection_rows)))
        for i, row in enumerate(selection_rows):
            h_vis = finite_float(row.get("h_vis"))
            h_loc = finite_float(row.get("h_loc"))
            h_cons = finite_float(row.get("h_cons"))
            if h_vis is not None and h_loc is not None:
                ax.plot([h_vis, h_loc], [i, i], marker="|", linewidth=3)
            if h_cons is not None and math.isfinite(h_cons):
                ax.scatter([h_cons], [i], marker="o", color="tab:red")
        ax.axvline(1e-3, color="black", linestyle="--", linewidth=1, label="default h=1e-3")
        ax.set_xscale("log")
        ax.set_yticks(y)
        ax.set_yticklabels(labels)
        ax.set_xlabel("h")
        ax.grid(True, axis="x", alpha=0.3)
        ax.legend()
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(output_dir / f"fig_h_window_by_precision.{ext}")
            fig.savefig(output_dir / f"fig_roberta_vs_opt_h_selection.{ext}")
        plt.close(fig)

    comparison = load_existing_csv(output_dir / "training_comparison_if_available.csv")
    if comparison:
        fig, ax = plt.subplots(figsize=(7, 5))
        xs = [finite_float(r.get("A_uniform")) for r in comparison]
        ys = [finite_float(r.get("accuracy")) for r in comparison]
        pairs = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
        if pairs:
            ax.scatter([x for x, _ in pairs], [y for _, y in pairs])
        ax.set_xlabel("A_uniform")
        ax.set_ylabel("final/best accuracy")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(output_dir / f"fig_training_vs_interval_metrics.{ext}")
        plt.close(fig)
    else:
        missing.append("fig_training_vs_interval_metrics not generated because no matched training comparison rows were available.")
    return missing


def zip_output(output_dir: Path) -> Path:
    zip_path = output_dir.with_suffix(".zip")
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(output_dir.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(output_dir.parent))
    return zip_path


def write_missing(output_dir: Path, items: Sequence[str]) -> None:
    lines = ["# Missing / unavailable items", ""]
    if not items:
        lines.append("- None.")
    else:
        for item in items:
            lines.append(f"- {item}")
    (output_dir / "missing_items.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def smoke_adjust_args(args: argparse.Namespace) -> None:
    if not args.smoke:
        return
    args.n_dirs = min(args.n_dirs, 4)
    args.coords_per_tensor = min(args.coords_per_tensor, 1024)
    args.max_coords = min(args.max_coords, 8192)
    if args.h_grid == DEFAULT_H_GRID:
        args.h_grid = [1e-5, 1e-4, 1e-3, 1e-2]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", default="interval_aware_h_probe")
    parser.add_argument("--models", nargs="+", default=["roberta-large"])
    parser.add_argument("--tasks", nargs="+", default=["sst-5"])
    parser.add_argument("--precisions", nargs="+", default=["int4", "int8"])
    parser.add_argument("--modes", nargs="+", default=["dense", "sparse_p0p1"])
    parser.add_argument("--h_grid", nargs="+", type=float, default=list(DEFAULT_H_GRID))
    parser.add_argument("--auto_extend_large_h", type=str2bool, default=True)
    parser.add_argument("--n_dirs", type=int, default=64)
    parser.add_argument("--coords_per_tensor", type=int, default=50_000)
    parser.add_argument("--max_coords", type=int, default=1_000_000)
    parser.add_argument("--group_size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=16)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--allow_download", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no_zip", action="store_true")
    args = parser.parse_args()
    smoke_adjust_args(args)

    if args.auto_extend_large_h:
        max_h = max(args.h_grid)
        if max_h < 3e-2:
            args.h_grid.append(3e-2)
        if max_h < 1e-1:
            args.h_grid.append(1e-1)
        args.h_grid = sorted(set(float(h) for h in args.h_grid))

    output_dir = Path(args.output_dir)
    if args.overwrite and output_dir.exists():
        shutil.rmtree(output_dir)
        zip_path = output_dir.with_suffix(".zip")
        if zip_path.exists():
            zip_path.unlink()
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "metadata.json", {"args": vars(args), "env": collect_env(), "h_grid": args.h_grid})

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    missing: List[str] = []
    all_geometry_rows: List[Dict[str, object]] = []
    all_summary_rows: List[Dict[str, object]] = []
    configs_for_loss_placeholder: List[Tuple[str, str, str, str, float]] = []
    quantizer_stats: Dict[str, object] = {}

    for model_arg in args.models:
        model_id = resolve_model_id(model_arg)
        try:
            t0 = time.time()
            model = load_model(model_id, device, args.allow_download)
            load_seconds = time.time() - t0
        except Exception as exc:
            missing.append(f"model_load_failed model={model_id}: {exc}")
            continue
        try:
            for precision_arg in args.precisions:
                precision, _bits = parse_precision(precision_arg)
                try:
                    samples, qstats, sample_notes = build_layer_samples(
                        model,
                        precision,
                        args.group_size,
                        args.coords_per_tensor,
                        args.max_coords,
                        args.seed,
                    )
                    quantizer_stats[f"{model_id}:{precision}"] = {
                        "num_layers_sampled": len(samples),
                        "num_coords_sampled": sum(s.sample_count for s in samples),
                        "state_stats": qstats,
                        "notes": sample_notes,
                        "model_load_seconds": load_seconds,
                    }
                except Exception as exc:
                    missing.append(f"sampling_or_quantizer_failed model={model_id} precision={precision}: {exc}")
                    continue
                for task in args.tasks:
                    for mode in args.modes:
                        try:
                            geometry_rows, summary_rows = probe_config(
                                model_label=model_id,
                                task=task,
                                precision=precision,
                                mode=mode,
                                samples=samples,
                                h_grid=args.h_grid,
                                n_dirs=args.n_dirs,
                                seed=args.seed,
                            )
                        except Exception as exc:
                            missing.append(
                                f"probe_failed model={model_id} task={task} precision={precision} mode={mode}: {exc}"
                            )
                            continue
                        append_rows(output_dir / "interval_geometry_metrics.csv", GEOMETRY_COLUMNS, geometry_rows)
                        append_rows(output_dir / "interval_geometry_summary.csv", SUMMARY_COLUMNS, summary_rows)
                        all_geometry_rows.extend(geometry_rows)
                        all_summary_rows.extend(summary_rows)
                        for row in summary_rows:
                            configs_for_loss_placeholder.append((model_id, task, precision, mode, float(row["h"])))
        finally:
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    write_json(output_dir / "quantizer_state_stats.json", quantizer_stats)
    write_loss_mse_placeholder(output_dir, configs_for_loss_placeholder)
    missing.append(
        "loss-level nMSE/corr and normalized_curve were not computed in this geometry-only run; use a task-loss probe extension for those fields."
    )
    missing.append(
        "dynamic-grid results were not generated; main outputs use shared-grid RTNClip states from unperturbed weights."
    )

    selection_rows = select_h(all_summary_rows, args.h_grid)
    append_rows(output_dir / "h_selection_summary.csv", SELECTION_COLUMNS, selection_rows)
    global_rows = global_precision_h(selection_rows, args.h_grid)
    append_rows(
        output_dir / "precision_global_h.csv",
        [
            "precision",
            "h_vis_global",
            "h_loc_global",
            "h_precision_global",
            "h_precision_cons",
            "global_window_exists",
            "configs_used",
            "configs_failed",
        ],
        global_rows,
    )
    missing.extend(try_training_comparison(output_dir, all_summary_rows))
    missing.extend(plot_outputs(output_dir, all_summary_rows, selection_rows))
    write_missing(output_dir, missing)

    zip_path = None
    if not args.no_zip:
        zip_path = zip_output(output_dir)

    print(f"Interval-aware output directory: {output_dir}")
    if zip_path is not None:
        print(f"Zip: {zip_path}")
    if selection_rows:
        print("Selections:")
        for row in selection_rows:
            print(
                f"  {row['model']} {row['task']} {row['precision']} {row['perturbation_mode']}: "
                f"h_vis={row['h_vis']} h_loc={row['h_loc']} h_cons={row['h_cons']} "
                f"default_in_window={row['default_in_window']}"
            )
    else:
        print("No successful interval-aware configs.")
    if missing:
        print("Missing/unavailable items were written to missing_items.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
