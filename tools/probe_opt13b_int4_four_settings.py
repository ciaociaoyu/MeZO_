#!/usr/bin/env python
"""Probe OPT-1.3B INT4 RTNClip finite-difference windows.

This is a diagnostic-only runner. It does not train, does not submit jobs, and
does not mutate stored model weights.  It uses the same low-bit oracle semantics
as the RoBERTa RTNClip probes:

    grid_t = RTNClipGrid(w_t)
    d_h = [L(Q_t(w_t + h u)) - L(Q_t(w_t - h u))] / (2 h)

The default nMSE reported here is the legacy/default window metric:
finite-difference d_h versus the true directional derivative g^T u.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import socket
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn


REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = REPO_ROOT / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import smoke_rtnclip_roberta_sst5 as rtn  # noqa: E402


TOY_TEXTS = [
    "Zero order optimization probes a model with two forward passes.",
    "Quantized perturbations should be visible at a useful step size.",
    "Sparse directions can change the finite-difference window.",
    "Prefix parameters are isolated from the quantized base model.",
]

SUMMARY_COLUMNS = [
    "model_id",
    "setting",
    "h",
    "k_dirs",
    "default_fd_true_nmse",
    "default_corr_fd_true",
    "default_true_direction",
    "d_h_mean",
    "d_true_mean",
    "d_h_abs_mean",
    "d_true_abs_mean",
    "active_frac_mean",
    "alignment_mean",
    "norm_ratio_mean",
    "delta_q_norm_mean",
    "ideal_displacement_norm_mean",
    "sparse_p",
    "mask_strategy",
    "prefix_impl",
    "perturb_scope",
    "quantized_modules",
    "status",
]


def json_default(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.dtype):
        return str(value)
    return str(value)


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True, default=json_default) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: List[Dict[str, object]], columns: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(columns), extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def append_jsonl(path: Path, row: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, sort_keys=True, default=json_default) + "\n")


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


def env_info() -> Dict[str, object]:
    info: Dict[str, object] = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "hostname": socket.gethostname(),
        "python": sys.executable,
        "python_version": platform.python_version(),
        "conda_env": os.environ.get("CONDA_DEFAULT_ENV", ""),
        "git_commit": git_commit(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
    }
    try:
        import transformers

        info["transformers_version"] = transformers.__version__
    except Exception:
        info["transformers_version"] = None
    if torch.cuda.is_available():
        idx = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(idx)
        info["gpu_name"] = props.name
        info["gpu_total_memory_mb"] = int(props.total_memory / 1024 / 1024)
    return info


def parse_h_grid(raw: Sequence[str]) -> List[float]:
    vals: List[float] = []
    for item in raw:
        for part in str(item).replace(",", " ").split():
            vals.append(float(part))
    return vals


def finite_float(value: object) -> Optional[float]:
    try:
        out = float(value)  # type: ignore[arg-type]
    except Exception:
        return None
    return out if math.isfinite(out) else None


def corr(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    pairs = [(float(x), float(y)) for x, y in zip(xs, ys) if math.isfinite(float(x)) and math.isfinite(float(y))]
    if len(pairs) < 2:
        return None
    xv = [x for x, _ in pairs]
    yv = [y for _, y in pairs]
    mx = sum(xv) / len(xv)
    my = sum(yv) / len(yv)
    vx = sum((x - mx) ** 2 for x in xv)
    vy = sum((y - my) ** 2 for y in yv)
    if vx <= 1e-30 or vy <= 1e-30:
        return None
    cov = sum((x - mx) * (y - my) for x, y in pairs)
    return cov / math.sqrt(vx * vy)


def load_model_and_inputs(args: argparse.Namespace, device: torch.device):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, local_files_only=args.local_files_only)
    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = getattr(tokenizer, "eos_token", None) or tokenizer.unk_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        local_files_only=args.local_files_only,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
    )
    model.config.use_cache = False
    model.eval()
    model.to(device)
    encoded = tokenizer(
        TOY_TEXTS[: max(1, min(args.batch_size, len(TOY_TEXTS)))],
        padding="max_length",
        truncation=True,
        max_length=args.max_seq_len,
        return_tensors="pt",
    )
    inputs = {k: v.to(device) for k, v in encoded.items()}
    labels = inputs["input_ids"].clone()
    pad_id = getattr(tokenizer, "pad_token_id", None)
    if pad_id is not None:
        labels = labels.masked_fill(inputs["input_ids"] == int(pad_id), -100)
    inputs["labels"] = labels
    return model, tokenizer, inputs


def params_map(model: nn.Module) -> Dict[str, nn.Parameter]:
    return {name: p for name, p in model.named_parameters() if p.detach().is_floating_point()}


def linear_weight_names(model: nn.Module, params: Dict[str, nn.Parameter]) -> List[str]:
    names: List[str] = []
    for module_name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            name = f"{module_name}.weight" if module_name else "weight"
            if name in params and params[name].ndim == 2:
                names.append(name)
    return names


def make_master(params: Dict[str, nn.Parameter]) -> Dict[str, torch.Tensor]:
    return {name: p.detach().clone().to(device=p.device, dtype=torch.float16) for name, p in params.items()}


def restore_master(params: Dict[str, nn.Parameter], master: Dict[str, torch.Tensor]) -> None:
    with torch.no_grad():
        for name, tensor in master.items():
            params[name].copy_(tensor.to(dtype=params[name].dtype))


def refresh_states(master: Dict[str, torch.Tensor], q_names: Iterable[str], bitwidth: int, group_size: int):
    states: Dict[str, rtn.RTNClipState] = {}
    rows: List[Dict[str, object]] = []
    for name in q_names:
        state, stats = rtn.compute_rtnclip_state(name, master[name], bitwidth, group_size)
        states[name] = state
        rows.append(stats)
    return states, rows


def apply_values(
    params: Dict[str, nn.Parameter],
    master: Dict[str, torch.Tensor],
    directions: Optional[Dict[str, torch.Tensor]],
    states: Dict[str, rtn.RTNClipState],
    h: float,
    sign: float,
) -> None:
    with torch.no_grad():
        for name, tensor in master.items():
            value = tensor.float()
            if directions is not None and name in directions:
                value = value.add(directions[name].float(), alpha=float(sign) * float(h))
            if name in states:
                value = rtn.quantize_with_state(value, states[name])
            params[name].copy_(value.to(dtype=params[name].dtype))


def forward_loss(model: nn.Module, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
    out = model(**inputs)
    return out.loss


def forward_loss_with_prefix(
    model: nn.Module,
    inputs: Dict[str, torch.Tensor],
    prefix: torch.Tensor,
) -> torch.Tensor:
    embed = model.get_input_embeddings()
    token_embeds = embed(inputs["input_ids"])
    batch = token_embeds.shape[0]
    prefix_batch = prefix.unsqueeze(0).expand(batch, -1, -1).to(dtype=token_embeds.dtype)
    inputs_embeds = torch.cat([prefix_batch, token_embeds], dim=1)
    prefix_mask = torch.ones((batch, prefix.shape[0]), dtype=inputs["attention_mask"].dtype, device=inputs["attention_mask"].device)
    attention_mask = torch.cat([prefix_mask, inputs["attention_mask"]], dim=1)
    prefix_labels = torch.full((batch, prefix.shape[0]), -100, dtype=inputs["labels"].dtype, device=inputs["labels"].device)
    labels = torch.cat([prefix_labels, inputs["labels"]], dim=1)
    out = model(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels)
    return out.loss


def compute_true_gradient(
    model: nn.Module,
    params: Dict[str, nn.Parameter],
    master: Dict[str, torch.Tensor],
    inputs: Dict[str, torch.Tensor],
    *,
    states: Optional[Dict[str, rtn.RTNClipState]] = None,
    prefix: Optional[torch.Tensor] = None,
) -> None:
    model.zero_grad(set_to_none=True)
    if prefix is not None and prefix.grad is not None:
        prefix.grad = None
    restore_master(params, master)
    if states is not None:
        apply_values(params, master, None, states, 0.0, 0.0)
    if prefix is None:
        loss = forward_loss(model, inputs)
    else:
        loss = forward_loss_with_prefix(model, inputs, prefix)
    loss.backward()


def build_highest_abs_masks(master: Dict[str, torch.Tensor], names: Sequence[str], sparse_p: float) -> Tuple[Dict[str, torch.Tensor], Dict[str, object]]:
    masks: Dict[str, torch.Tensor] = {}
    active = 0
    total = 0
    for name in names:
        tensor = master[name]
        n = int(tensor.numel())
        total += n
        if sparse_p >= 1.0:
            mask = torch.ones_like(tensor, dtype=torch.bool)
        else:
            k = max(1, int(math.floor(float(sparse_p) * n)))
            flat = tensor.detach().abs().float().reshape(-1)
            if k >= n:
                mask = torch.ones_like(tensor, dtype=torch.bool)
            else:
                threshold = torch.kthvalue(flat, n - k + 1).values.to(device=tensor.device, dtype=flat.dtype)
                mask = tensor.detach().abs() >= threshold
        masks[name] = mask
        active += int(mask.sum().detach().cpu())
    return masks, {
        "mask_strategy": "highest_abs_per_tensor",
        "sparse_p": float(sparse_p),
        "active_param_count": active,
        "total_param_count": total,
        "active_frac": active / max(total, 1),
    }


def sample_direction(
    master: Dict[str, torch.Tensor],
    names: Sequence[str],
    seed: int,
    masks: Optional[Dict[str, torch.Tensor]] = None,
) -> Dict[str, torch.Tensor]:
    first = master[names[0]]
    gen = torch.Generator(device=first.device).manual_seed(int(seed))
    out: Dict[str, torch.Tensor] = {}
    for name in names:
        z = torch.randn(master[name].shape, device=first.device, generator=gen, dtype=torch.float16)
        if masks is not None:
            z = z * masks[name].to(device=first.device, dtype=z.dtype)
        out[name] = z
    return out


def grad_dot_direction(params: Dict[str, nn.Parameter], directions: Dict[str, torch.Tensor]) -> float:
    total = torch.zeros((), device=next(iter(params.values())).device, dtype=torch.float64)
    for name, direction in directions.items():
        grad = params[name].grad
        if grad is None:
            continue
        total += (grad.detach().double() * direction.double()).sum()
    return float(total.detach().cpu())


def prefix_grad_dot(prefix: torch.Tensor, direction: torch.Tensor) -> float:
    if prefix.grad is None:
        return float("nan")
    return float((prefix.grad.detach().double() * direction.double()).sum().detach().cpu())


def finite_difference(
    model: nn.Module,
    params: Dict[str, nn.Parameter],
    master: Dict[str, torch.Tensor],
    inputs: Dict[str, torch.Tensor],
    states: Dict[str, rtn.RTNClipState],
    directions: Dict[str, torch.Tensor],
    h: float,
) -> Tuple[float, float, float]:
    with torch.no_grad():
        apply_values(params, master, directions, states, h, +1.0)
        loss_plus = float(forward_loss(model, inputs).detach().cpu())
        apply_values(params, master, directions, states, h, -1.0)
        loss_minus = float(forward_loss(model, inputs).detach().cpu())
        restore_master(params, master)
    return loss_plus, loss_minus, (loss_plus - loss_minus) / (2.0 * float(h))


def finite_difference_prefix(
    model: nn.Module,
    params: Dict[str, nn.Parameter],
    master: Dict[str, torch.Tensor],
    inputs: Dict[str, torch.Tensor],
    states: Dict[str, rtn.RTNClipState],
    prefix_base: torch.Tensor,
    prefix_direction: torch.Tensor,
    h: float,
) -> Tuple[float, float, float]:
    with torch.no_grad():
        apply_values(params, master, None, states, 0.0, 0.0)
        loss_plus = float(forward_loss_with_prefix(model, inputs, prefix_base + float(h) * prefix_direction).detach().cpu())
        loss_minus = float(forward_loss_with_prefix(model, inputs, prefix_base - float(h) * prefix_direction).detach().cpu())
        restore_master(params, master)
    return loss_plus, loss_minus, (loss_plus - loss_minus) / (2.0 * float(h))


def visibility_metrics(
    master: Dict[str, torch.Tensor],
    directions: Dict[str, torch.Tensor],
    states: Dict[str, rtn.RTNClipState],
    h: float,
) -> Dict[str, float]:
    active = 0
    total = 0
    dot = torch.zeros((), device=next(iter(master.values())).device, dtype=torch.float64)
    delta_sq = torch.zeros_like(dot)
    intended_sq = torch.zeros_like(dot)
    for name, state in states.items():
        if name not in directions:
            continue
        direction = directions[name].float()
        intended = 2.0 * float(h) * direction
        plus = rtn.quantize_with_state(master[name].float().add(direction, alpha=float(h)), state)
        minus = rtn.quantize_with_state(master[name].float().add(direction, alpha=-float(h)), state)
        delta = plus.float() - minus.float()
        active += int((delta != 0).sum().detach().cpu())
        total += int(delta.numel())
        dot += (delta.double() * intended.double()).sum()
        delta_sq += delta.double().square().sum()
        intended_sq += intended.double().square().sum()
    eps = 1e-30
    return {
        "active_frac": active / max(total, 1),
        "alignment": float((dot / (delta_sq.sqrt() * intended_sq.sqrt() + eps)).detach().cpu()) if float(intended_sq.detach().cpu()) > 0 else float("nan"),
        "norm_ratio": float((delta_sq.sqrt() / intended_sq.sqrt().clamp_min(eps)).detach().cpu()) if float(intended_sq.detach().cpu()) > 0 else float("nan"),
        "delta_q_norm": float(delta_sq.sqrt().detach().cpu()),
        "ideal_displacement_norm": float(intended_sq.sqrt().detach().cpu()),
    }


def prefix_visibility(prefix_direction: torch.Tensor) -> Dict[str, float]:
    # Prefix parameters are not quantized; the effective prefix displacement is
    # exactly the intended displacement. The base model is still INT4 RTNClip.
    return {
        "active_frac": 1.0,
        "alignment": 1.0,
        "norm_ratio": 1.0,
        "delta_q_norm": float((2.0 * prefix_direction.float()).norm().detach().cpu()),
        "ideal_displacement_norm": float((2.0 * prefix_direction.float()).norm().detach().cpu()),
    }


def aggregate(records: List[Dict[str, object]], h_grid: Sequence[float], config: Dict[str, object]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    settings = sorted({str(r["setting"]) for r in records})
    for setting in settings:
        for h in h_grid:
            group = [r for r in records if str(r["setting"]) == setting and abs(float(r["h"]) - float(h)) <= 1e-15]
            if not group:
                continue
            dh = [float(r["d_h"]) for r in group if finite_float(r.get("d_h")) is not None]
            dt = [float(r["d_true"]) for r in group if finite_float(r.get("d_true")) is not None]
            n = min(len(dh), len(dt))
            dh = dh[:n]
            dt = dt[:n]
            mse = sum((a - b) ** 2 for a, b in zip(dh, dt)) / max(n, 1)
            ref = sum(b ** 2 for b in dt) / max(n, 1)
            row = {
                "model_id": config["model_id"],
                "setting": setting,
                "h": h,
                "k_dirs": n,
                "default_fd_true_nmse": mse / max(ref, 1e-30),
                "default_corr_fd_true": corr(dh, dt),
                "default_true_direction": config["default_true_direction"],
                "d_h_mean": sum(dh) / max(n, 1),
                "d_true_mean": sum(dt) / max(n, 1),
                "d_h_abs_mean": sum(abs(x) for x in dh) / max(n, 1),
                "d_true_abs_mean": sum(abs(x) for x in dt) / max(n, 1),
                "active_frac_mean": mean_key(group, "active_frac"),
                "alignment_mean": mean_key(group, "alignment"),
                "norm_ratio_mean": mean_key(group, "norm_ratio"),
                "delta_q_norm_mean": mean_key(group, "delta_q_norm"),
                "ideal_displacement_norm_mean": mean_key(group, "ideal_displacement_norm"),
                "sparse_p": group[0].get("sparse_p", ""),
                "mask_strategy": group[0].get("mask_strategy", ""),
                "prefix_impl": group[0].get("prefix_impl", ""),
                "perturb_scope": group[0].get("perturb_scope", ""),
                "quantized_modules": config["quantized_modules"],
                "status": "complete",
            }
            rows.append(row)
    return rows


def mean_key(rows: Sequence[Dict[str, object]], key: str) -> Optional[float]:
    vals = [float(r[key]) for r in rows if finite_float(r.get(key)) is not None]
    return sum(vals) / len(vals) if vals else None


def write_report(output_dir: Path, summary_rows: List[Dict[str, object]], config: Dict[str, object]) -> None:
    lines = [
        "# OPT-1.3B INT4 RTNClip Four-Setting Probe",
        "",
        "This is probe-only. It uses cached `facebook/opt-1.3b`, synthetic causal-LM batches, INT4 G128 RTNClip fake-quantized Linear weights, shared grid from unperturbed weights, and fresh rounding for `w +/- h u`.",
        "",
        "Default nMSE is `default_dh_vs_gTu`: finite-difference `d_h` compared with `g^T u`.",
        "",
        f"- settings: `{config['settings']}`",
        f"- h grid: `{config['h_grid']}`",
        f"- directions: `{config['k_dirs']}`",
        f"- quantized Linear weights: `{config['quantized_modules']}`",
        "",
        "| setting | h | default_fd_true_nmse | corr | active_frac | alignment | norm_ratio | d_h_mean | d_true_mean |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['setting']} | {float(row['h']):.6g} | {fmt(row['default_fd_true_nmse'])} | "
            f"{fmt(row['default_corr_fd_true'])} | {fmt(row['active_frac_mean'])} | {fmt(row['alignment_mean'])} | "
            f"{fmt(row['norm_ratio_mean'])} | {fmt(row['d_h_mean'])} | {fmt(row['d_true_mean'])} |"
        )
    output_dir.joinpath("summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def fmt(value: object) -> str:
    vf = finite_float(value)
    return "NA" if vf is None else f"{vf:.6g}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", default="facebook/opt-1.3b")
    parser.add_argument("--output_dir", default="")
    parser.add_argument("--settings", nargs="+", default=["dense", "sparse_p0p1", "sparse_p0p01", "prefix"])
    parser.add_argument("--h_grid", nargs="+", default=["1e-5", "3e-5", "1e-4", "3e-4", "1e-3", "3e-3", "1e-2", "3e-2", "1e-1"])
    parser.add_argument("--k_dirs", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_seq_len", type=int, default=64)
    parser.add_argument("--group_size", type=int, default=128)
    parser.add_argument("--bitwidth", type=int, default=4)
    parser.add_argument("--seed", type=int, default=16)
    parser.add_argument("--prefix_len", type=int, default=16)
    parser.add_argument("--local_files_only", action="store_true", default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    h_grid = parse_h_grid(args.h_grid)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) if args.output_dir else REPO_ROOT / "outputs" / f"opt13b_int4_four_settings_probe_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    records_path = output_dir / "probe_records.jsonl"
    if records_path.exists():
        records_path.unlink()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("This OPT-1.3B probe is intended for the local H100 CUDA device.")
    torch.manual_seed(args.seed)
    env = env_info()
    write_json(output_dir / "env.json", env)
    print(json.dumps(env, indent=2, sort_keys=True), flush=True)

    model, tokenizer, inputs = load_model_and_inputs(args, device)
    params = params_map(model)
    q_names = linear_weight_names(model, params)
    perturb_names = list(params.keys())
    master = make_master(params)
    restore_master(params, master)
    states, q_rows = refresh_states(master, q_names, int(args.bitwidth), int(args.group_size))
    quant_agg = rtn.aggregate_quantizer_stats(q_rows, {name: int(params[name].numel()) for name in q_names})

    config = {
        "model_id": args.model_id,
        "settings": args.settings,
        "h_grid": h_grid,
        "k_dirs": args.k_dirs,
        "batch_size": args.batch_size,
        "max_seq_len": args.max_seq_len,
        "bitwidth": args.bitwidth,
        "group_size": args.group_size,
        "quantizer": "INT4_G128_RTNClip_shared_grid_fake_quant",
        "scale_source": "unperturbed_fp16_master_weight",
        "pair_shared_grid": True,
        "fresh_round_codes": True,
        "default_nmse_metric": "default_dh_vs_gTu",
        "default_true_direction": "gTu",
        "perturb_scope_dense_sparse": "all_floating_parameters",
        "sparse_mask_strategy": "highest_abs_per_tensor",
        "sparse_rescale": "none",
        "prefix_impl": "soft_prompt_prefix_inputs_embeds",
        "quantized_modules": len(q_names),
        "quantized_module_names": q_names,
        "quantizer_summary": quant_agg,
    }
    write_json(output_dir / "run_config.json", config)
    write_json(output_dir / "quantizer_summary.json", {"aggregate": quant_agg, "per_module": q_rows})

    records: List[Dict[str, object]] = []
    start = time.time()
    sparse_masks: Dict[str, Tuple[Dict[str, torch.Tensor], Dict[str, object]]] = {}
    for p in (0.1, 0.01):
        sparse_masks[f"sparse_p{p:g}".replace(".", "p")] = build_highest_abs_masks(master, perturb_names, p)

    for setting in args.settings:
        print(f"[{datetime.now().isoformat(timespec='seconds')}] setting={setting} true-gradient", flush=True)
        if setting == "prefix":
            apply_values(params, master, None, states, 0.0, 0.0)
            hidden = int(model.config.hidden_size)
            gen = torch.Generator(device=device).manual_seed(args.seed + 9000)
            prefix_base = (0.02 * torch.randn((args.prefix_len, hidden), device=device, generator=gen, dtype=torch.float16)).detach().requires_grad_(True)
            compute_true_gradient(model, params, master, inputs, states=states, prefix=prefix_base)
            direction_names: Sequence[str] = []
            masks = None
            mask_stats: Dict[str, object] = {}
        else:
            compute_true_gradient(model, params, master, inputs, states=None)
            prefix_base = None
            direction_names = perturb_names
            if setting == "dense":
                masks = None
                mask_stats = {"sparse_p": "", "mask_strategy": ""}
            elif setting == "sparse_p0p1":
                masks, mask_stats = sparse_masks["sparse_p0p1"]
            elif setting == "sparse_p0p01":
                masks, mask_stats = sparse_masks["sparse_p0p01"]
            else:
                raise ValueError(f"unknown setting {setting!r}")

        for direction_id in range(args.k_dirs):
            if setting == "prefix":
                assert prefix_base is not None
                gen = torch.Generator(device=device).manual_seed(args.seed + direction_id * 1009 + 12000)
                prefix_dir = torch.randn(prefix_base.shape, device=device, generator=gen, dtype=torch.float16)
                d_true = prefix_grad_dot(prefix_base, prefix_dir)
                directions = {}
            else:
                directions = sample_direction(master, direction_names, args.seed + direction_id * 1009 + 5000, masks=masks)
                d_true = grad_dot_direction(params, directions)
                prefix_dir = None
            for h in h_grid:
                if setting == "prefix":
                    assert prefix_base is not None and prefix_dir is not None
                    loss_plus, loss_minus, d_h = finite_difference_prefix(
                        model, params, master, inputs, states, prefix_base.detach(), prefix_dir, float(h)
                    )
                    vis = prefix_visibility(prefix_dir * float(h))
                else:
                    loss_plus, loss_minus, d_h = finite_difference(model, params, master, inputs, states, directions, float(h))
                    vis = visibility_metrics(master, directions, states, float(h))
                record = {
                    "setting": setting,
                    "h": float(h),
                    "direction_id": int(direction_id),
                    "loss_plus": loss_plus,
                    "loss_minus": loss_minus,
                    "d_h": d_h,
                    "d_true": d_true,
                    "fd_true_error": d_h - d_true,
                    "sparse_p": mask_stats.get("sparse_p", ""),
                    "mask_strategy": mask_stats.get("mask_strategy", ""),
                    "prefix_impl": "soft_prompt_prefix_inputs_embeds" if setting == "prefix" else "",
                    "perturb_scope": "prefix_parameters_only" if setting == "prefix" else "all_floating_parameters",
                    **vis,
                }
                append_jsonl(records_path, record)
                records.append(record)
            print(f"  direction={direction_id} d_true={d_true:.6g}", flush=True)
        model.zero_grad(set_to_none=True)
        restore_master(params, master)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary_rows = aggregate(records, h_grid, config)
    write_csv(output_dir / "summary.csv", summary_rows, SUMMARY_COLUMNS)
    write_report(output_dir, summary_rows, config)
    run_summary = {
        **config,
        "output_dir": str(output_dir),
        "status": "complete",
        "records": len(records),
        "summary_rows": len(summary_rows),
        "runtime_seconds": time.time() - start,
        "peak_gpu_memory_mb": float(torch.cuda.max_memory_allocated() / 1024 / 1024) if torch.cuda.is_available() else 0.0,
    }
    write_json(output_dir / "run_summary.json", run_summary)

    print(f"Output: {output_dir}", flush=True)
    for row in summary_rows:
        print(
            f"{row['setting']:>13} h={float(row['h']):.3g} "
            f"nmse={fmt(row['default_fd_true_nmse'])} corr={fmt(row['default_corr_fd_true'])} "
            f"active={fmt(row['active_frac_mean'])} align={fmt(row['alignment_mean'])} norm={fmt(row['norm_ratio_mean'])}",
            flush=True,
        )


if __name__ == "__main__":
    main()
