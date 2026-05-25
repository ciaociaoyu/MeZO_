#!/usr/bin/env python
"""QZO-paper-style scale-only ZO runner for RoBERTa / SST-5.

The official QZO implementation trains floating quantization scales in GPTQ/AQLM
modules with fixed low-bit codes. This runner ports that update semantics to
the local RoBERTa prompt-classification path. It is intentionally labeled as a
RoBERTa adaptation when exact GPTQModel quantization is unavailable.
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
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = REPO_ROOT / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import smoke_rtnclip_roberta_sst5 as smoke  # noqa: E402


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    except Exception:
        return ""


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def append_jsonl(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, sort_keys=True, default=str) + "\n")


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def scale_dtype_from_name(name: str) -> torch.dtype:
    norm = str(name).lower()
    if norm in {"fp16", "float16", "half"}:
        return torch.float16
    if norm in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported scale dtype {name!r}")


def _group_quantize_weight(
    weight: torch.Tensor,
    *,
    bits: int,
    group_size: int,
    scale_dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, object]]:
    if weight.ndim != 2:
        raise ValueError(f"QZO quantized Linear weight must be 2D, got {tuple(weight.shape)}")
    if bits != 4:
        raise ValueError("This runner currently implements the requested INT4 QZO setting only.")
    qmax = 2 ** (bits - 1) - 1
    out_features, in_features = weight.shape
    num_groups = int(math.ceil(in_features / group_size))
    padded_cols = num_groups * group_size
    pad_cols = padded_cols - in_features
    w = weight.detach().float()
    if pad_cols:
        w = F.pad(w, (0, pad_cols))
    grouped = w.reshape(out_features, num_groups, group_size)
    lengths = torch.full((num_groups,), group_size, device=w.device, dtype=torch.long)
    if pad_cols:
        lengths[-1] = group_size - pad_cols
    valid = torch.arange(group_size, device=w.device).view(1, 1, group_size) < lengths.view(1, num_groups, 1)
    max_abs = grouped.abs().masked_fill(~valid, 0.0).amax(dim=-1, keepdim=True)
    scales = (max_abs / float(qmax)).clamp_min(1e-7)
    codes = torch.round(grouped / scales).clamp(-qmax, qmax).to(torch.int8)
    codes = codes.masked_fill(~valid, 0)
    deq = (codes.float() * scales).reshape(out_features, padded_cols)[:, :in_features]
    diff = deq - weight.detach().float()
    ref_sq = weight.detach().float().square().sum().double().clamp_min(1e-30)
    err_sq = diff.square().sum().double()
    valid_codes = codes[valid.expand_as(codes)]
    stats = {
        "bits": bits,
        "qmin": -qmax,
        "qmax": qmax,
        "group_size": group_size,
        "num_groups": int(out_features * num_groups),
        "scale_dtype": str(scale_dtype).replace("torch.", ""),
        "scale_min": float(scales.min().detach().cpu()),
        "scale_median": float(scales.median().detach().cpu()),
        "scale_max": float(scales.max().detach().cpu()),
        "code_min": int(valid_codes.min().detach().cpu()) if valid_codes.numel() else 0,
        "code_max": int(valid_codes.max().detach().cpu()) if valid_codes.numel() else 0,
        "saturation_frac": float((valid_codes.abs() >= qmax).float().mean().detach().cpu()) if valid_codes.numel() else 0.0,
        "weight_recon_mse": float((err_sq / max(weight.numel(), 1)).detach().cpu()),
        "weight_recon_rel_mse": float((err_sq / ref_sq).detach().cpu()),
        "weight_recon_sqnr_db": float((10.0 * torch.log10(ref_sq / err_sq.clamp_min(1e-30))).detach().cpu()),
    }
    return codes.reshape(out_features, padded_cols), scales.squeeze(-1).to(dtype=scale_dtype), stats


class QZOScaleLinear(nn.Module):
    """Fixed-code low-bit Linear with trainable floating scales."""

    def __init__(
        self,
        linear: nn.Linear,
        *,
        name: str,
        bits: int,
        group_size: int,
        scale_dtype: torch.dtype,
    ) -> None:
        super().__init__()
        weight = linear.weight.detach()
        codes, scales, stats = _group_quantize_weight(
            weight,
            bits=bits,
            group_size=group_size,
            scale_dtype=scale_dtype,
        )
        self.name = name
        self.bits = bits
        self.group_size = group_size
        self.in_features = int(linear.in_features)
        self.out_features = int(linear.out_features)
        self.qmax = 2 ** (bits - 1) - 1
        self.register_buffer("codes", codes.contiguous(), persistent=True)
        self.scales = nn.Parameter(scales.contiguous())
        if linear.bias is not None:
            self.bias = nn.Parameter(linear.bias.detach().clone(), requires_grad=False)
        else:
            self.register_parameter("bias", None)
        self.quant_stats = stats

    def dequant_weight(self) -> torch.Tensor:
        scale = self.scales.clamp_min(1e-7).float().repeat_interleave(self.group_size, dim=1)
        weight = self.codes.float()[:, : self.in_features] * scale[:, : self.in_features]
        return weight

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        weight = self.dequant_weight().to(dtype=input.dtype, device=input.device)
        bias = None if self.bias is None else self.bias.to(dtype=input.dtype, device=input.device)
        return F.linear(input, weight, bias)


def _get_parent_module(root: nn.Module, module_name: str) -> Tuple[nn.Module, str]:
    parts = module_name.split(".")
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def replace_linear_with_qzo(
    model: nn.Module,
    *,
    bits: int,
    group_size: int,
    scale_dtype: torch.dtype,
) -> Tuple[List[QZOScaleLinear], List[Dict[str, object]]]:
    modules = [(name, module) for name, module in model.named_modules() if isinstance(module, nn.Linear)]
    qzo_modules: List[QZOScaleLinear] = []
    rows: List[Dict[str, object]] = []
    for name, module in modules:
        parent, attr = _get_parent_module(model, name)
        q_module = QZOScaleLinear(module, name=name, bits=bits, group_size=group_size, scale_dtype=scale_dtype)
        setattr(parent, attr, q_module)
        qzo_modules.append(q_module)
        rows.append({"module_name": name, **q_module.quant_stats})
    for param in model.parameters():
        param.requires_grad_(False)
    for module in qzo_modules:
        module.scales.requires_grad_(True)
    return qzo_modules, rows


def qzo_scale_params(modules: Iterable[QZOScaleLinear]) -> List[nn.Parameter]:
    return [module.scales for module in modules]


def perturb_scale_params(params: List[nn.Parameter], *, seed: int, eps: float, scaling_factor: float) -> None:
    torch.manual_seed(seed)
    with torch.no_grad():
        for param in params:
            z = torch.normal(mean=0.0, std=1.0, size=param.shape, device=param.device, dtype=param.dtype)
            param.add_(z, alpha=float(scaling_factor) * float(eps))


def update_scale_params(
    params: List[nn.Parameter],
    *,
    seed: int,
    lr: float,
    projected_grad: float,
    clip_zo_grad: bool,
) -> Tuple[float, float, bool]:
    torch.manual_seed(seed)
    pg = float(projected_grad)
    clipped = False
    if clip_zo_grad:
        new_pg = min(max(-100.0, pg), 100.0)
        clipped = new_pg != pg
        pg = new_pg
    sq = torch.zeros((), device=params[0].device, dtype=torch.float64)
    with torch.no_grad():
        for param in params:
            before = param.detach().float()
            z = torch.normal(mean=0.0, std=1.0, size=param.shape, device=param.device, dtype=param.dtype)
            update = z.float().mul(-float(lr) * pg)
            sq += update.double().square().sum()
            param.copy_(param.float().add(update).clamp_min(1e-7).to(dtype=param.dtype))
            if not torch.isfinite(param).all():
                raise FloatingPointError("non-finite QZO scale after update")
            if torch.equal(before, param.detach().float()):
                pass
    return pg, float(sq.sqrt().detach().cpu()), clipped


def forward_loss_and_logits(model: nn.Module, batch: Dict[str, torch.Tensor]):
    return smoke.forward_loss_and_logits(model, batch)


def evaluate(model, dev_loader, device: torch.device, max_batches: int) -> Tuple[Optional[float], Optional[float]]:
    if max_batches == 0:
        return None, None
    total_loss = 0.0
    total_correct = 0
    total_items = 0
    model.eval()
    for idx, batch in enumerate(dev_loader):
        if max_batches > 0 and idx >= max_batches:
            break
        batch = smoke.move_batch(batch, device)
        loss, logits = forward_loss_and_logits(model, batch)
        labels = batch["labels"]
        total_loss += float(loss.detach().cpu()) * int(labels.numel())
        total_correct += int((logits.argmax(dim=-1) == labels).sum().detach().cpu())
        total_items += int(labels.numel())
    if total_items == 0:
        return None, None
    return total_loss / total_items, total_correct / total_items


def aggregate_quant_stats(rows: List[Dict[str, object]]) -> Dict[str, object]:
    if not rows:
        return {}
    group_total = sum(int(r["num_groups"]) for r in rows)
    value_total = sum(int(r["num_groups"]) * int(r["group_size"]) for r in rows)
    weighted = {}
    for key in ("saturation_frac", "weight_recon_mse", "weight_recon_rel_mse", "weight_recon_sqnr_db"):
        weighted[key] = sum(float(r[key]) * int(r["num_groups"]) for r in rows) / max(group_total, 1)
    return {
        "quantized_module_count": len(rows),
        "quantized_group_count": group_total,
        "quantized_padded_value_count": value_total,
        "scale_min_global": min(float(r["scale_min"]) for r in rows),
        "scale_median_avg": sum(float(r["scale_median"]) * int(r["num_groups"]) for r in rows) / max(group_total, 1),
        "scale_max_global": max(float(r["scale_max"]) for r in rows),
        **weighted,
    }


def save_checkpoint(
    ckpt_dir: Path,
    *,
    step: int,
    modules: List[QZOScaleLinear],
    best: Dict[str, object],
    config: Dict[str, object],
) -> None:
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    state = {
        "step": step,
        "scales": {module.name: module.scales.detach().cpu() for module in modules},
        "best": best,
        "config": config,
        "rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    torch.save(state, ckpt_dir / "scale_state.pt")
    write_json(ckpt_dir / "checkpoint_manifest.json", {"step": step, "scale_tensors": len(modules)})


def copy_checkpoint(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    if src.exists():
        shutil.copytree(src, dst)


def latest_checkpoint(run_dir: Path) -> Optional[Path]:
    ckpt_root = run_dir / "checkpoints"
    if not ckpt_root.exists():
        return None
    best_step = -1
    best_path = None
    for path in ckpt_root.glob("step_*"):
        try:
            step = int(path.name.split("_", 1)[1])
        except Exception:
            continue
        if (path / "scale_state.pt").exists() and step > best_step:
            best_step = step
            best_path = path
    return best_path


def load_checkpoint(path: Path, modules: List[QZOScaleLinear], device: torch.device) -> Tuple[int, Dict[str, object]]:
    payload = torch.load(path / "scale_state.pt", map_location=device, weights_only=False)
    by_name = {module.name: module for module in modules}
    with torch.no_grad():
        for name, tensor in payload.get("scales", {}).items():
            if name in by_name:
                by_name[name].scales.copy_(tensor.to(device=device, dtype=by_name[name].scales.dtype))
    return int(payload.get("step", 0)), dict(payload.get("best", {}) or {})


def run(args: argparse.Namespace) -> Dict[str, object]:
    run_dir = Path(args.output_dir) / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    for name in ("metrics.csv", "eval_metrics.jsonl", "quantizer_diagnostics.jsonl"):
        (run_dir / name).touch(exist_ok=True)

    if os.environ.get("DATALOADER_SHUFFLE") != "True":
        raise RuntimeError("DATALOADER_SHUFFLE must be True for this run.")
    if not torch.cuda.is_available() and args.device == "cuda":
        raise RuntimeError("CUDA was requested but is not available.")
    device = torch.device("cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu")
    set_seed(int(args.seed))
    env = smoke.collect_env(REPO_ROOT)
    try:
        import importlib.util

        exact_gptq_available = importlib.util.find_spec("gptqmodel") is not None
        auto_gptq_available = importlib.util.find_spec("auto_gptq") is not None
    except Exception:
        exact_gptq_available = False
        auto_gptq_available = False
    config = {
        "method": "qzo_article_scale_only",
        "article_repo": "https://github.com/maifoundations/QZO",
        "article_reference_files": [
            "/tmp/QZO_official/large_language_models/trainer.py",
            "/tmp/QZO_official/large_language_models/quantization.py",
        ],
        "exact_qzo_status": "not_exact_gptq_roberta_adaptation",
        "fallback_reason": "Official QZO code uses GPTQModel/AQLM quantized LLM modules and does not directly provide RoBERTa/SST-5 prompt-classification GPTQ modules in this environment.",
        "gptqmodel_available": exact_gptq_available,
        "auto_gptq_available": auto_gptq_available,
        "actual_quantizer": "fixed_code_groupwise_symmetric_int4_scale_trainable",
        "not_quzo": True,
        "not_rtnclip": True,
        "not_direct_int_update": True,
        "fixed_integer_codes": True,
        "trainable_tensors": "quantization_scales_only",
        "scale_dtype": args.scale_dtype,
        "scale_min_clamp": 1e-7,
        "quant_bits": int(args.bits),
        "group_size": int(args.group_size),
        "model": args.model_id,
        "task_name": args.task_name,
        "dataset_mode": args.dataset_mode,
        "seed": int(args.seed),
        "data_seed": int(args.data_seed),
        "batch_size": int(args.batch_size),
        "eval_batch_size": int(args.eval_batch_size),
        "steps": int(args.steps),
        "h": float(args.h),
        "lr": float(args.lr),
        "clip_zo_grad": bool(args.clip_zo_grad),
        "zo_grad_clip_range": [-100.0, 100.0],
        "eval_every": int(args.eval_every),
        "checkpoint_steps": int(args.checkpoint_steps),
        "DATALOADER_SHUFFLE": os.environ.get("DATALOADER_SHUFFLE", ""),
        "git_commit": git_commit(),
        "hostname": socket.gethostname(),
        "env": env,
        "run_dir": str(run_dir),
    }
    write_json(run_dir / "run_config.json", config)
    write_json(run_dir / "run_manifest_row.json", config)
    (run_dir / "resume_command.txt").write_text(
        "CUDA_VISIBLE_DEVICES=0 DATALOADER_SHUFFLE=True "
        f"conda run -n ciao python tools/qzo_article_roberta_sst5.py "
        f"--output_dir {Path(args.output_dir)} --run_name {args.run_name} --steps {args.steps} "
        f"--h {args.h} --lr {args.lr} --scale_dtype {args.scale_dtype} --resume\n",
        encoding="utf-8",
    )
    (run_dir / "QZO_METHOD_NOTE.md").write_text(
        "# QZO Method Note\n\n"
        "This run uses the QZO paper update semantics: fixed low-bit integer codes and ZO updates only on floating quantization scales.\n\n"
        "It is not the earlier local QuZO path. Exact GPTQModel RoBERTa/SST-5 support was not available in this environment, so the quantizer is labeled as a fixed-code groupwise symmetric INT4 fallback rather than exact GPTQ.\n",
        encoding="utf-8",
    )

    orig_torch_load = torch.load

    def _compat_torch_load(*load_args, **load_kwargs):
        load_kwargs.setdefault("weights_only", False)
        return orig_torch_load(*load_args, **load_kwargs)

    torch.load = _compat_torch_load
    try:
        model, train_loader, dev_loader, data_args, train_sampler = smoke.load_prompt_model_and_data(
            argparse.Namespace(
                repo_root=REPO_ROOT,
                model_id=args.model_id,
                task_name=args.task_name,
                dataset_mode=args.dataset_mode,
                data_seed=int(args.data_seed),
                seed=int(args.seed),
                num_k=int(args.num_k),
                data_dir=args.data_dir or None,
                batch_size=int(args.batch_size),
                eval_batch_size=int(args.eval_batch_size),
            ),
            device,
        )
    finally:
        torch.load = orig_torch_load
    if type(train_sampler).__name__ != "RandomSampler":
        raise RuntimeError(f"Expected RandomSampler, got {type(train_sampler).__name__}")
    scale_dtype = scale_dtype_from_name(args.scale_dtype)
    modules, quant_rows = replace_linear_with_qzo(
        model,
        bits=int(args.bits),
        group_size=int(args.group_size),
        scale_dtype=scale_dtype,
    )
    scale_params = qzo_scale_params(modules)
    quant_agg = aggregate_quant_stats(quant_rows)
    config.update(
        {
            "data_dir_resolved": getattr(data_args, "data_dir", ""),
            "sampler_name": type(train_sampler).__name__,
            "quantized_module_count": len(modules),
            "scale_parameter_count": sum(int(p.numel()) for p in scale_params),
            "trainable_parameter_count": sum(int(p.numel()) for p in model.parameters() if p.requires_grad),
            **quant_agg,
        }
    )
    write_json(run_dir / "run_config.json", config)
    for row in quant_rows:
        append_jsonl(run_dir / "quantizer_diagnostics.jsonl", {"record_type": "module_quantization", **row})
    append_jsonl(run_dir / "quantizer_diagnostics.jsonl", {"record_type": "aggregate_quantization", **quant_agg})

    start_step = 0
    best: Dict[str, object] = {"best_eval_acc": None, "best_eval_step": None, "best_eval_loss": None, "best_eval_loss_step": None}
    if args.resume:
        ckpt = latest_checkpoint(run_dir)
        if ckpt is not None:
            start_step, best = load_checkpoint(ckpt, modules, device)
    batch_iter = smoke.cycle(train_loader)
    for _ in range(start_step):
        next(batch_iter)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    metrics_path = run_dir / "metrics.csv"
    write_header = metrics_path.stat().st_size == 0
    finite_count = 0
    last_train_loss = None
    last_eval_loss = None
    last_eval_acc = None
    last_eval_step = None
    last_projected_grad = None
    last_update_norm = None
    status = "running"
    error_message = ""
    total_start = time.time()
    with metrics_path.open("a", newline="", encoding="utf-8") as f:
        fieldnames = [
            "step",
            "loss_plus",
            "loss_minus",
            "train_loss",
            "projected_grad_raw",
            "projected_grad_used",
            "projected_grad_clipped",
            "update_norm",
            "scale_min",
            "scale_median",
            "scale_max",
            "eval_loss",
            "eval_acc",
            "seconds",
            "nan_flag",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        for step_idx in range(start_step, int(args.steps)):
            step_start = time.time()
            batch = smoke.move_batch(next(batch_iter), device)
            zo_seed = random.randint(0, 1_000_000_000)
            perturb_scale_params(scale_params, seed=zo_seed, eps=float(args.h), scaling_factor=1.0)
            loss_plus, _ = forward_loss_and_logits(model, batch)
            perturb_scale_params(scale_params, seed=zo_seed, eps=float(args.h), scaling_factor=-2.0)
            loss_minus, _ = forward_loss_and_logits(model, batch)
            projected_grad_raw = (float(loss_plus.detach().cpu()) - float(loss_minus.detach().cpu())) / (2.0 * float(args.h))
            perturb_scale_params(scale_params, seed=zo_seed, eps=float(args.h), scaling_factor=1.0)
            projected_grad_used, update_norm, clipped = update_scale_params(
                scale_params,
                seed=zo_seed,
                lr=float(args.lr),
                projected_grad=projected_grad_raw,
                clip_zo_grad=bool(args.clip_zo_grad),
            )
            scale_values = torch.cat([p.detach().float().reshape(-1) for p in scale_params])
            loss_plus_f = float(loss_plus.detach().cpu())
            loss_minus_f = float(loss_minus.detach().cpu())
            last_train_loss = (loss_plus_f + loss_minus_f) / 2.0
            last_projected_grad = projected_grad_used
            last_update_norm = update_norm
            finite = (
                math.isfinite(loss_plus_f)
                and math.isfinite(loss_minus_f)
                and math.isfinite(projected_grad_raw)
                and math.isfinite(projected_grad_used)
                and math.isfinite(update_norm)
                and bool(torch.isfinite(scale_values).all())
            )
            if finite:
                finite_count += 1
            completed = step_idx + 1
            eval_loss = None
            eval_acc = None
            if completed % int(args.eval_every) == 0 or completed == int(args.steps):
                eval_loss, eval_acc = evaluate(model, dev_loader, device, int(args.eval_batches))
                last_eval_loss = eval_loss
                last_eval_acc = eval_acc
                last_eval_step = completed
                append_jsonl(run_dir / "eval_metrics.jsonl", {"step": completed, "eval_loss": eval_loss, "eval_acc": eval_acc})
                if eval_acc is not None and (best.get("best_eval_acc") is None or float(eval_acc) > float(best["best_eval_acc"])):
                    best["best_eval_acc"] = eval_acc
                    best["best_eval_step"] = completed
                    save_checkpoint(run_dir / "checkpoints" / f"step_{completed}", step=completed, modules=modules, best=best, config=config)
                    copy_checkpoint(run_dir / "checkpoints" / f"step_{completed}", run_dir / "checkpoints" / "best_acc")
                if eval_loss is not None and (best.get("best_eval_loss") is None or float(eval_loss) < float(best["best_eval_loss"])):
                    best["best_eval_loss"] = eval_loss
                    best["best_eval_loss_step"] = completed
                    save_checkpoint(run_dir / "checkpoints" / f"step_{completed}", step=completed, modules=modules, best=best, config=config)
                    copy_checkpoint(run_dir / "checkpoints" / f"step_{completed}", run_dir / "checkpoints" / "best_loss")
            if completed % int(args.checkpoint_steps) == 0 or completed == int(args.steps):
                save_checkpoint(run_dir / "checkpoints" / f"step_{completed}", step=completed, modules=modules, best=best, config=config)
            nan_flag = not finite
            writer.writerow(
                {
                    "step": completed,
                    "loss_plus": loss_plus_f,
                    "loss_minus": loss_minus_f,
                    "train_loss": last_train_loss,
                    "projected_grad_raw": projected_grad_raw,
                    "projected_grad_used": projected_grad_used,
                    "projected_grad_clipped": clipped,
                    "update_norm": update_norm,
                    "scale_min": float(scale_values.min().detach().cpu()),
                    "scale_median": float(scale_values.median().detach().cpu()),
                    "scale_max": float(scale_values.max().detach().cpu()),
                    "eval_loss": eval_loss,
                    "eval_acc": eval_acc,
                    "seconds": time.time() - step_start,
                    "nan_flag": nan_flag,
                }
            )
            f.flush()
            if completed == 1 or completed % int(args.log_every) == 0:
                with (run_dir / "train.log").open("a", encoding="utf-8") as log:
                    log.write(
                        f"step={completed}/{args.steps} loss={last_train_loss:.6g} "
                        f"pg={projected_grad_used:.6g} update_norm={update_norm:.6g} eval_acc={eval_acc}\n"
                    )
            if nan_flag:
                status = "failed"
                error_message = f"non-finite QZO value at step {completed}"
                break

    steps_completed = start_step
    try:
        with metrics_path.open(newline="", encoding="utf-8") as mf:
            rows = list(csv.DictReader(mf))
            if rows:
                steps_completed = int(float(rows[-1]["step"]))
    except Exception:
        pass
    if status != "failed":
        status = "complete" if steps_completed >= int(args.steps) else "partial"
    save_checkpoint(run_dir / "checkpoints" / "final", step=steps_completed, modules=modules, best=best, config=config)
    if not (run_dir / "checkpoints" / "best_acc").exists():
        copy_checkpoint(run_dir / "checkpoints" / "final", run_dir / "checkpoints" / "best_acc")
    if not (run_dir / "checkpoints" / "best_loss").exists():
        copy_checkpoint(run_dir / "checkpoints" / "final", run_dir / "checkpoints" / "best_loss")
    runtime = time.time() - total_start
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
        "final_train_loss": last_train_loss,
        "last_projected_grad": last_projected_grad,
        "last_update_norm": last_update_norm,
        "d_h_finite_rate": finite_count / max(steps_completed - start_step, 1),
        "total_runtime": runtime,
        "seconds_per_step": runtime / max(steps_completed - start_step, 1),
        "peak_gpu_mem": float(torch.cuda.max_memory_allocated() / 1024 / 1024) if torch.cuda.is_available() else 0.0,
    }
    write_json(run_dir / "run_summary.json", summary)
    print(json.dumps(summary, indent=2, default=str))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="QZO paper scale-only RoBERTa/SST-5 adaptation runner")
    parser.add_argument("--output_dir", default=str(REPO_ROOT / "outputs" / f"qzo_article_scaleonly_roberta_sst5_seed16_{datetime.now().strftime('%Y%m%d_%H%M%S')}"))
    parser.add_argument("--run_name", default="qzo_article_scaleonly_int4_sst5_h1e-3_seed16_bs64")
    parser.add_argument("--model_id", default="roberta-large")
    parser.add_argument("--task_name", default="sst-5")
    parser.add_argument("--dataset_mode", default="full")
    parser.add_argument("--data_dir", default="")
    parser.add_argument("--num_k", type=int, default=16)
    parser.add_argument("--seed", type=int, default=16)
    parser.add_argument("--data_seed", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--eval_batches", type=int, default=-1)
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--eval_every", type=int, default=1000)
    parser.add_argument("--checkpoint_steps", type=int, default=1000)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--h", type=float, default=1e-3)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--group_size", type=int, default=128)
    parser.add_argument("--scale_dtype", default="fp16", choices=["fp16", "fp32"])
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu", "auto"])
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--clip_zo_grad", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
