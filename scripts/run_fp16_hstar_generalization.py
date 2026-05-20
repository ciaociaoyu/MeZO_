#!/usr/bin/env python3
"""FP16 h-star generalization analysis.

This is an offline probe/curvature analyzer. It loads initial prompt-tuning
models, materializes full data splits through the existing medium_models data
resolver, and evaluates FP16 two-point finite differences against FP32 true
directional derivatives. It does not launch or resume training.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import os
import platform
import socket
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
MEDIUM_ROOT = REPO_ROOT / "medium_models"
H_GRID = [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 1.5e-3, 2e-3, 3e-3, 4e-3, 5e-3, 1e-2]
EPS = 1e-12

PROBE_FIELDS = [
    "group",
    "model",
    "dataset",
    "seed",
    "data_seed",
    "checkpoint",
    "h",
    "num_probe_dirs",
    "mse",
    "nmse",
    "corr",
    "bias",
    "mae",
    "median_abs_error",
    "alignment_eff",
    "norm_ratio_eff",
    "zero_coord_frac_eff",
    "rms_snap_error",
    "fd_mean",
    "fd_std",
    "true_mean",
    "true_std",
    "truth_kind",
]

G_FIELDS = [
    "group",
    "model",
    "dataset",
    "seed",
    "G_method",
    "h_G",
    "G_hat",
    "alignment_eff",
    "norm_ratio_eff",
    "zero_coord_frac_eff",
    "corr_h_2h",
    "sign_flip_rate_h_2h",
    "stability_score",
    "fallback_flags",
]

L_CAND_FIELDS = [
    "group",
    "model",
    "dataset",
    "seed",
    "L_mode",
    "h2",
    "m_L",
    "lambda_q50",
    "lambda_q90",
    "lambda_q95",
    "median_abs_K",
    "MAD_K",
    "SNR2",
    "finite_rate",
    "zero_K_frac",
    "stability_q90_2x",
    "stability_q90_next",
    "stability_q90_prev",
    "log_slope_q90_next",
    "log_slope_q90_prev",
    "low_h2_noise_suspected",
    "large_h2_nonlocal_suspected",
    "alignment_eff",
    "norm_ratio_eff",
    "zero_coord_frac_eff",
    "rms_snap_error",
]

L_SEL_FIELDS = [
    "group",
    "model",
    "dataset",
    "seed",
    "L_mode",
    "selector",
    "selected_h2",
    "selected_L_q50",
    "selected_L_q90",
    "selected_L_q95",
    "selection_status",
    "flags",
]

HSTAR_FIELDS = [
    "group",
    "model",
    "dataset",
    "seed",
    "selector",
    "Delta_mode",
    "Delta_value",
    "G_method",
    "G_hat",
    "G_h_used",
    "L_mode",
    "L_q",
    "L_hat",
    "L_h2_used",
    "d_trainable",
    "hstar_cont",
    "hstar_nearest_grid",
    "nmse_at_selected_h",
    "corr_at_selected_h",
    "empirical_min_nmse_h",
    "empirical_min_nmse",
    "empirical_max_corr_h",
    "empirical_max_corr",
    "nmse_ratio",
    "corr_gap",
    "success_pass",
    "strict_success_pass",
    "notes",
]


@dataclass
class Setting:
    group: str
    model_label: str
    model_name: str
    dataset: str
    seed: int
    data_seed: int
    fallback: bool = False


@dataclass
class Context:
    setting: Setting
    model: Any
    tokenizer: Any
    batch: Dict[str, Any]
    named_params: List[Tuple[str, Any]]
    backups: List[Any]
    device: Any
    d_trainable: int
    data_info: Dict[str, Any]
    direction_dtype_name: str = "float32"
    forward_precision: str = "fp32"


def write_csv(path: Path, rows: Iterable[Dict[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fields), extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def finite_corr(a: Sequence[float], b: Sequence[float]) -> float:
    x = np.asarray(a, dtype=np.float64)
    y = np.asarray(b, dtype=np.float64)
    m = np.isfinite(x) & np.isfinite(y)
    if int(m.sum()) < 2:
        return float("nan")
    x = x[m]
    y = y[m]
    sx = float(np.std(x))
    sy = float(np.std(y))
    if sx <= 0.0 or sy <= 0.0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def safe_float(v: Any) -> Optional[float]:
    try:
        out = float(v)
    except Exception:
        return None
    return out if math.isfinite(out) else None


def h_key(h: float) -> str:
    return f"{float(h):.12g}"


def nearest_grid(h: float) -> float:
    if not math.isfinite(float(h)) or float(h) <= 0.0:
        return float("nan")
    return min(H_GRID, key=lambda x: abs(math.log(float(x)) - math.log(float(h))))


def run_cmd(args: Sequence[str], cwd: Path = REPO_ROOT) -> str:
    try:
        return subprocess.check_output(list(args), cwd=str(cwd), text=True, stderr=subprocess.STDOUT).strip()
    except Exception as exc:
        return f"ERROR: {exc}"


def env_report() -> str:
    lines = [
        f"hostname: {socket.gethostname()}",
        f"date: {dt.datetime.now().isoformat()}",
        f"pwd: {Path.cwd()}",
        f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', '')}",
        f"CONDA_DEFAULT_ENV: {os.environ.get('CONDA_DEFAULT_ENV', '')}",
        f"which python: {run_cmd(['which', 'python'])}",
        f"python --version: {run_cmd(['python', '--version'])}",
        "nvidia-smi:",
        run_cmd(["nvidia-smi"]),
    ]
    try:
        import torch

        lines.extend(
            [
                f"torch.__version__: {torch.__version__}",
                f"torch.version.cuda: {torch.version.cuda}",
                f"torch.cuda.is_available(): {torch.cuda.is_available()}",
                f"torch.cuda.get_device_name(0): {torch.cuda.get_device_name(0) if torch.cuda.is_available() else ''}",
            ]
        )
    except Exception as exc:
        lines.append(f"torch import failed: {exc}")
    return "\n".join(lines) + "\n"


def import_medium():
    if str(MEDIUM_ROOT) not in sys.path:
        sys.path.insert(0, str(MEDIUM_ROOT))
    from transformers import AutoConfig, AutoTokenizer
    from src.data_utils import resolve_and_prepare_data
    from src.dataset import FewShotDataset
    from src.models import MODEL_TYPES
    from src.processors import num_labels_mapping

    return AutoConfig, AutoTokenizer, FewShotDataset, MODEL_TYPES, num_labels_mapping, resolve_and_prepare_data


def task_defaults(task: str) -> Dict[str, Any]:
    task_l = task.lower()
    base = {
        "task_name": task_l,
        "max_seq_length": 128,
        "overwrite_cache": False,
        "num_k": 16,
        "num_sample": 16,
        "num_demo": 1,
        "auto_demo": True,
        "sfc_prompt": None,
        "template_path": None,
        "mapping_path": None,
        "prompt_path": None,
        "template_id": None,
        "mapping_id": None,
        "prompt_id": None,
        "top_n_template": None,
        "tag": "fp16_hstar_generalization",
        "demo_filter": False,
        "demo_filter_rate": 0.5,
        "demo_filter_model": None,
        "debug_mode": False,
        "double_demo": False,
        "first_sent_limit": None,
        "other_sent_limit": None,
        "use_full_length": None,
        "dataset_mode": "full",
        "data_root": "data/k-shot-1k-test",
        "full_dev_ratio": 0.1,
        "gpt3_in_context_head": False,
        "gpt3_in_context_tail": False,
        "gpt3_in_context_num": 32,
        "gpt3_demo_separator": "\n\n\n",
        "truncate_head": False,
        "prompt": True,
        "template_list": None,
    }
    if task_l == "sst-5":
        base.update(
            {
                "template": "*cls**sent_0*_It_was*mask*.*sep+*",
                "mapping": "{0:'terrible',1:'bad',2:'okay',3:'good',4:'great'}",
                "first_sent_limit": 110,
                "other_sent_limit": 20,
                "double_demo": True,
            }
        )
    elif task_l == "rte":
        base.update(
            {
                "template": "*cls**sent-_0*?*mask*,*+sentl_1**sep+*",
                "mapping": "{'not_entailment':'No','entailment':'Yes'}",
                "max_seq_length": 256,
                "first_sent_limit": 240,
            }
        )
    elif task_l in {"sst-2", "sst2"}:
        base.update(
            {
                "task_name": "sst-2",
                "template": "*cls**sent_0*_It_was*mask*.*sep+*",
                "mapping": "{'0':'terrible','1':'great'}",
            }
        )
    else:
        raise ValueError(f"unsupported task defaults for {task}")
    base["data_dir"] = str(MEDIUM_ROOT / "data/k-shot-1k-test" / task / "16-16")
    return base


def collate_with_padding(tokenizer: Any, features: Sequence[Any]) -> Dict[str, Any]:
    import torch

    items: List[Dict[str, Any]] = []
    mask_pos: List[Any] = []
    for item in features:
        row: Dict[str, Any] = {}
        for field in ["input_ids", "label", "attention_mask", "token_type_ids"]:
            value = getattr(item, field, None)
            if value is not None:
                row[field] = value
        items.append(row)
        mask_pos.append(getattr(item, "mask_pos", None))
    batch = tokenizer.pad(items, padding=True, return_tensors="pt")
    if any(x is not None for x in mask_pos):
        batch["mask_pos"] = torch.tensor(mask_pos)
    if "label" in batch:
        batch["labels"] = batch.pop("label")
    if "label_ids" in batch:
        batch["labels"] = batch.pop("label_ids")
    return batch


def set_seed(seed: int) -> None:
    import torch

    np.random.seed(int(seed) % (2**32))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def direction_seeds(seed: int, n: int, offset: int = 0) -> List[int]:
    rng = np.random.RandomState((int(seed) * 1000003 + int(offset) * 9176 + 97) % 2147483647)
    return [int(x) for x in rng.randint(0, 2147483647, size=max(1, int(n)))]


def load_context(setting: Setting, device: Any, batch_size: int) -> Context:
    import torch

    AutoConfig, AutoTokenizer, FewShotDataset, MODEL_TYPES, num_labels_mapping, resolve_and_prepare_data = import_medium()
    set_seed(setting.seed)

    data_args = SimpleNamespace(**task_defaults(setting.dataset))
    if "opt" in setting.model_name.lower():
        # OPT/GPT-style tokenizers do not have cls/mask tokens. The local
        # dataset converter uses the final token position as the prompt site
        # when no mask token exists, so avoid templates that inject None ids.
        if str(data_args.task_name).lower() in {"sst-2", "sst2"}:
            data_args.template = "*bos**sent_0*_It_was"
    train_args = SimpleNamespace(seed=setting.seed, data_seed=setting.data_seed)
    data_resolution = resolve_and_prepare_data(data_args, train_args)
    data_args.data_dir = data_resolution.resolved_data_dir
    data_args.dataset_mode = data_resolution.resolved_dataset_mode

    task_name = str(data_args.task_name).lower()
    config = AutoConfig.from_pretrained(
        setting.model_name,
        num_labels=int(num_labels_mapping[task_name]),
        finetuning_task=task_name,
    )
    # The repository's local OPT modules read these LoRA attributes directly
    # from config even when LoRA is disabled.
    if not hasattr(config, "apply_lora"):
        config.apply_lora = False
    if not hasattr(config, "lora_r"):
        config.lora_r = None
    if not hasattr(config, "lora_alpha"):
        config.lora_alpha = None
    tokenizer = AutoTokenizer.from_pretrained(setting.model_name)
    if "opt" in setting.model_name.lower():
        tokenizer.sep_token_id = tokenizer.eos_token_id
        tokenizer.bos_token_id = 0
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
            config.pad_token_id = tokenizer.pad_token_id
    tokenizer.model_type = config.model_type

    model_fn = MODEL_TYPES[config.model_type]
    model = model_fn.from_pretrained(setting.model_name, config=config)
    model_args = SimpleNamespace(
        model_name_or_path=setting.model_name,
        few_shot_type="prompt",
        random_segment=False,
        l2_loss=False,
        use_task_word=False,
        apply_lora=False,
        sfc=False,
        icl_sfc=False,
    )

    dataset = FewShotDataset(data_args, tokenizer=tokenizer, mode="train", use_demo=False)
    if getattr(dataset, "label_word_list", None) is not None:
        model.label_word_list = torch.tensor(dataset.label_word_list).long()
    model.model_args = model_args
    model.data_args = data_args
    model.tokenizer = tokenizer
    model.eval()
    model.float()
    model.to(device)
    if getattr(model, "label_word_list", None) is not None:
        model.label_word_list = model.label_word_list.to(device)

    gen = torch.Generator()
    gen.manual_seed(int(setting.data_seed))
    n = min(int(batch_size), len(dataset))
    idxs = torch.randperm(len(dataset), generator=gen)[:n].tolist()
    batch = collate_with_padding(tokenizer, [dataset[int(i)] for i in idxs])
    batch = {k: v.to(device) if hasattr(v, "to") else v for k, v in batch.items()}
    named = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    backups = [p.detach().clone() for _, p in named]
    d_trainable = int(sum(p.numel() for _, p in named))
    return Context(
        setting=setting,
        model=model,
        tokenizer=tokenizer,
        batch=batch,
        named_params=named,
        backups=backups,
        device=device,
        d_trainable=d_trainable,
        data_info={
            "resolved_data_dir": data_resolution.resolved_data_dir,
            "generated_data_split": bool(data_resolution.generated_data_split),
            "dataset_indices": [int(i) for i in idxs],
            "batch_size": int(n),
            "direction_normalization": "raw Gaussian unnormalized; torch normal per trainable parameter",
        },
    )


def restore(ctx: Context) -> None:
    import torch

    with torch.no_grad():
        for (_, p), b in zip(ctx.named_params, ctx.backups):
            p.data.copy_(b)


def reset_backups(ctx: Context) -> None:
    ctx.backups = [p.detach().clone() for _, p in ctx.named_params]


def restore_external_backups(ctx: Context, backups: Sequence[Any]) -> None:
    import torch

    with torch.no_grad():
        for (_, p), b in zip(ctx.named_params, backups):
            p.data.copy_(b.to(device=p.device, dtype=p.data.dtype))
    reset_backups(ctx)


def autocast_ctx(ctx: Context):
    import contextlib
    import torch

    if ctx.forward_precision == "fp16" and str(ctx.device).startswith("cuda"):
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return contextlib.nullcontext()


def compute_loss(ctx: Context, grad: bool = False) -> Any:
    if grad:
        with autocast_ctx(ctx):
            outputs = ctx.model(**ctx.batch)
            return outputs[0] if isinstance(outputs, (tuple, list)) else outputs.loss
    import torch

    with torch.no_grad():
        with autocast_ctx(ctx):
            outputs = ctx.model(**ctx.batch)
            loss = outputs[0] if isinstance(outputs, (tuple, list)) else outputs.loss
    return float(loss.detach().float().cpu().item())


def dtype_for_direction(ctx: Context):
    import torch

    return torch.float16 if ctx.direction_dtype_name == "float16" else torch.float32


def reset_rng(seed: int) -> None:
    import torch

    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def sample_z(param: Any, dtype: Any):
    import torch

    return torch.empty_like(param.data, dtype=dtype).normal_(0.0, 1.0)


def true_directional_from_grads(ctx: Context, grads: Sequence[Any], seed: int) -> float:
    import torch

    dtype = dtype_for_direction(ctx)
    total = 0.0
    reset_rng(seed)
    with torch.no_grad():
        for (_, p), g in zip(ctx.named_params, grads):
            z = sample_z(p, dtype)
            if g is not None:
                total += float(torch.sum(g.detach().float() * z.detach().float()).item())
            del z
    return float(total)


def compute_true_grads(ctx: Context) -> Tuple[float, List[Any], str]:
    import torch

    ctx.model.float()
    ctx.forward_precision = "fp32"
    ctx.direction_dtype_name = "float16"
    restore(ctx)
    ctx.model.zero_grad(set_to_none=True)
    loss = compute_loss(ctx, grad=True)
    loss.backward()
    grads = [p.grad.detach().clone().float() if p.grad is not None else None for _, p in ctx.named_params]
    loss_value = float(loss.detach().float().cpu().item())
    ctx.model.zero_grad(set_to_none=True)
    return loss_value, grads, "true_fp32_autograd"


def estimate_ulp(ctx: Context) -> Dict[str, Any]:
    import torch

    vals: List[np.ndarray] = []
    sum_sq = 0.0
    count = 0
    nonfinite = 0
    zero = 0
    sample_cap = 8192
    with torch.no_grad():
        for _, p in ctx.named_params:
            cast = p.detach().to(dtype=torch.float16).cpu()
            inf = torch.full_like(cast, float("inf"))
            spacing = (torch.nextafter(cast, inf) - cast).abs().to(torch.float32).reshape(-1)
            finite = torch.isfinite(spacing)
            count += int(spacing.numel())
            nonfinite += int((~finite).sum().item())
            if int(finite.sum().item()) == 0:
                continue
            x = spacing[finite]
            zero += int((x == 0).sum().item())
            sum_sq += float((x * x).sum().item())
            vals.append(x[: min(sample_cap, x.numel())].numpy().astype(np.float64, copy=False))
    arr = np.concatenate(vals) if vals else np.asarray([], dtype=np.float64)
    finite_count = count - nonfinite
    return {
        "delta_ulp_rms": float(math.sqrt(sum_sq / max(finite_count, 1))) if finite_count else float("nan"),
        "delta_ulp_median": float(np.quantile(arr, 0.50)) if arr.size else float("nan"),
        "delta_ulp_p90": float(np.quantile(arr, 0.90)) if arr.size else float("nan"),
        "delta_ulp_p95": float(np.quantile(arr, 0.95)) if arr.size else float("nan"),
        "count": int(count),
        "nonfinite_count": int(nonfinite),
        "zero_spacing_count": int(zero),
    }


def set_mode_fp16(ctx: Context) -> None:
    ctx.model.half()
    if getattr(ctx.model, "label_word_list", None) is not None:
        ctx.model.label_word_list = ctx.model.label_word_list.long().to(ctx.device)
    ctx.forward_precision = "fp16"
    ctx.direction_dtype_name = "float16"
    reset_backups(ctx)


def set_mode_fp32(ctx: Context) -> None:
    ctx.model.float()
    if getattr(ctx.model, "label_word_list", None) is not None:
        ctx.model.label_word_list = ctx.model.label_word_list.long().to(ctx.device)
    ctx.forward_precision = "fp32"
    ctx.direction_dtype_name = "float32"
    reset_backups(ctx)


def apply_signed(ctx: Context, seed: int, h: float, sign: float) -> None:
    import torch

    dtype = dtype_for_direction(ctx)
    reset_rng(seed)
    with torch.no_grad():
        for _, p in ctx.named_params:
            z = sample_z(p, dtype)
            delta = (z * (float(sign) * float(h))).to(dtype=p.data.dtype)
            p.data.add_(delta)
            del z, delta


def pair_effective_stats(ctx: Context, seed: int, h: float) -> Dict[str, float]:
    import torch

    dtype = dtype_for_direction(ctx)
    dot = eff_sq = tgt_sq = snap_sq = 0.0
    zero = total = 0
    reset_rng(seed)
    with torch.no_grad():
        for (_, p), b in zip(ctx.named_params, ctx.backups):
            z = sample_z(p, dtype)
            plus = (b + (z * float(h)).to(dtype=p.data.dtype)).to(dtype=p.data.dtype)
            minus = (b - (z * float(h)).to(dtype=p.data.dtype)).to(dtype=p.data.dtype)
            eff = (plus.float() - minus.float()).reshape(-1)
            tgt = (2.0 * float(h) * z.float()).reshape(-1)
            dot += float((eff * tgt).sum().item())
            eff_sq += float((eff * eff).sum().item())
            tgt_sq += float((tgt * tgt).sum().item())
            diff = eff - tgt
            snap_sq += float((diff * diff).sum().item())
            zero += int((eff == 0).sum().item())
            total += int(eff.numel())
            del z, plus, minus, eff, tgt
    eff_norm = math.sqrt(max(eff_sq, 0.0))
    tgt_norm = math.sqrt(max(tgt_sq, 0.0))
    return {
        "alignment_eff": float(dot / (eff_norm * tgt_norm + EPS)) if eff_norm > 0 and tgt_norm > 0 else float("nan"),
        "norm_ratio_eff": float(eff_norm / (tgt_norm + EPS)) if tgt_norm > 0 else float("nan"),
        "zero_coord_frac_eff": float(zero / total) if total else float("nan"),
        "rms_snap_error": float(math.sqrt(snap_sq / total)) if total else float("nan"),
    }


def two_point_fd(ctx: Context, seed: int, h: float) -> Tuple[float, float, float]:
    restore(ctx)
    apply_signed(ctx, seed, h, +1.0)
    loss_plus = compute_loss(ctx)
    restore(ctx)
    apply_signed(ctx, seed, h, -1.0)
    loss_minus = compute_loss(ctx)
    restore(ctx)
    return float((loss_plus - loss_minus) / (2.0 * float(h))), float(loss_plus), float(loss_minus)


def probe_grid(ctx: Context, grads: Sequence[Any], true_loss: float, seeds: Sequence[int]) -> Tuple[List[Dict[str, Any]], Dict[float, List[float]], List[float], Dict[float, Dict[str, float]]]:
    import torch

    set_mode_fp16(ctx)
    d_true = [true_directional_from_grads(ctx, grads, int(s)) for s in seeds]
    rows: List[Dict[str, Any]] = []
    fd_by_h: Dict[float, List[float]] = {}
    eff_by_h: Dict[float, Dict[str, float]] = {}
    for h in H_GRID:
        fds: List[float] = []
        eff_rows: List[Dict[str, float]] = []
        t0 = time.time()
        for i, seed in enumerate(seeds):
            fd, _, _ = two_point_fd(ctx, int(seed), float(h))
            fds.append(fd)
            if i < min(4, len(seeds)):
                eff_rows.append(pair_effective_stats(ctx, int(seed), float(h)))
        arr = np.asarray(fds, dtype=np.float64)
        tru = np.asarray(d_true, dtype=np.float64)
        err = arr - tru
        mse = float(np.mean(err * err))
        nmse = float(mse / (float(np.mean(tru * tru)) + EPS))
        eff = {k: float(np.nanmean([x[k] for x in eff_rows])) if eff_rows else float("nan") for k in ["alignment_eff", "norm_ratio_eff", "zero_coord_frac_eff", "rms_snap_error"]}
        row = {
            "group": ctx.setting.group,
            "model": ctx.setting.model_label,
            "dataset": ctx.setting.dataset,
            "seed": ctx.setting.seed,
            "data_seed": ctx.setting.data_seed,
            "checkpoint": "initial_pretrained",
            "h": float(h),
            "num_probe_dirs": len(seeds),
            "mse": mse,
            "nmse": nmse,
            "corr": finite_corr(arr, tru),
            "bias": float(np.mean(err)),
            "mae": float(np.mean(np.abs(err))),
            "median_abs_error": float(np.median(np.abs(err))),
            "fd_mean": float(np.mean(arr)),
            "fd_std": float(np.std(arr)),
            "true_mean": float(np.mean(tru)),
            "true_std": float(np.std(tru)),
            "truth_kind": "true_fp32_autograd",
            **eff,
        }
        rows.append(row)
        fd_by_h[float(h)] = fds
        eff_by_h[float(h)] = eff
        print(
            f"[probe] {ctx.setting.group}/{ctx.setting.model_label}/{ctx.setting.dataset}/seed{ctx.setting.seed} "
            f"h={h:g} nmse={nmse:.4g} corr={row['corr']:.4g} elapsed={time.time()-t0:.1f}s",
            flush=True,
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return rows, fd_by_h, d_true, eff_by_h


def sign_flip_rate(a: Sequence[float], b: Sequence[float]) -> float:
    x = np.asarray(a, dtype=np.float64)
    y = np.asarray(b, dtype=np.float64)
    m = np.isfinite(x) & np.isfinite(y)
    if int(m.sum()) == 0:
        return float("nan")
    return float(np.mean(np.sign(x[m]) != np.sign(y[m])))


def estimate_g(ctx: Context, probe_rows: List[Dict[str, Any]], fd_by_h: Dict[float, List[float]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    by_h = {float(r["h"]): r for r in probe_rows}
    for h in H_GRID:
        fd = np.asarray(fd_by_h[float(h)], dtype=np.float64)
        ghat = float(math.sqrt(math.pi / 2.0) * np.mean(np.abs(fd[np.isfinite(fd)]))) if np.isfinite(fd).any() else float("nan")
        h2 = next((x for x in H_GRID if abs(x - 2 * h) <= max(1e-12, 1e-9 * abs(2 * h))), None)
        corr2 = finite_corr(fd_by_h[h], fd_by_h[h2]) if h2 is not None and h2 in fd_by_h else float("nan")
        flip = sign_flip_rate(fd_by_h[h], fd_by_h[h2]) if h2 is not None and h2 in fd_by_h else float("nan")
        prow = by_h[h]
        passing = (
            safe_float(prow.get("alignment_eff")) is not None
            and float(prow["alignment_eff"]) >= 0.99
            and safe_float(prow.get("norm_ratio_eff")) is not None
            and 0.9 <= float(prow["norm_ratio_eff"]) <= 1.1
            and safe_float(prow.get("zero_coord_frac_eff")) is not None
            and float(prow["zero_coord_frac_eff"]) <= 0.10
            and math.isfinite(ghat)
            and (h2 is None or not math.isfinite(corr2) or (corr2 >= 0.90 and flip <= 0.10))
        )
        score = 0.0
        if safe_float(prow.get("alignment_eff")) is not None:
            score += max(0.0, 0.99 - float(prow["alignment_eff"]))
        if safe_float(prow.get("norm_ratio_eff")) is not None:
            score += abs(float(prow["norm_ratio_eff"]) - 1.0)
        if safe_float(prow.get("zero_coord_frac_eff")) is not None:
            score += float(prow["zero_coord_frac_eff"])
        if math.isfinite(corr2):
            score += max(0.0, 0.90 - corr2)
        if math.isfinite(flip):
            score += max(0.0, flip - 0.10)
        rows.append(
            {
                "group": ctx.setting.group,
                "model": ctx.setting.model_label,
                "dataset": ctx.setting.dataset,
                "seed": ctx.setting.seed,
                "G_method": "absG_candidate",
                "h_G": h,
                "G_hat": ghat,
                "alignment_eff": prow.get("alignment_eff"),
                "norm_ratio_eff": prow.get("norm_ratio_eff"),
                "zero_coord_frac_eff": prow.get("zero_coord_frac_eff"),
                "corr_h_2h": corr2,
                "sign_flip_rate_h_2h": flip,
                "stability_score": score,
                "fallback_flags": "" if passing else "candidate_not_primary_pass",
                "_passing": passing,
            }
        )
    passing = [r for r in rows if r.get("_passing")]
    if passing:
        chosen = min(passing, key=lambda r: float(r["h_G"]))
        flags = ""
    else:
        chosen = min(rows, key=lambda r: float(r["stability_score"]))
        flags = "fallback_G"
    primary = dict(chosen)
    primary["G_method"] = "absG"
    primary["fallback_flags"] = flags
    rows.append(primary)

    for fixed in [3e-4, 1e-3]:
        if fixed in fd_by_h:
            fd = np.asarray(fd_by_h[fixed], dtype=np.float64)
            rows.append(
                {
                    "group": ctx.setting.group,
                    "model": ctx.setting.model_label,
                    "dataset": ctx.setting.dataset,
                    "seed": ctx.setting.seed,
                    "G_method": f"absG_fixed_{h_key(fixed)}",
                    "h_G": fixed,
                    "G_hat": float(math.sqrt(math.pi / 2.0) * np.mean(np.abs(fd))),
                    "fallback_flags": "diagnostic_only",
                }
            )
    for h in H_GRID:
        h2 = next((x for x in H_GRID if abs(x - 2 * h) <= max(1e-12, 1e-9 * abs(2 * h))), None)
        if h2 is not None and h2 in fd_by_h:
            dR = (4.0 * np.asarray(fd_by_h[h], dtype=np.float64) - np.asarray(fd_by_h[h2], dtype=np.float64)) / 3.0
            rows.append(
                {
                    "group": ctx.setting.group,
                    "model": ctx.setting.model_label,
                    "dataset": ctx.setting.dataset,
                    "seed": ctx.setting.seed,
                    "G_method": "richardsonG_candidate",
                    "h_G": h,
                    "G_hat": float(math.sqrt(math.pi / 2.0) * np.mean(np.abs(dR))),
                    "fallback_flags": "diagnostic_only",
                }
            )
            break
    return rows


def direction_norm_sq(ctx: Context, seed: int) -> float:
    import torch

    dtype = dtype_for_direction(ctx)
    total = 0.0
    reset_rng(seed)
    with torch.no_grad():
        for _, p in ctx.named_params:
            z = sample_z(p, dtype)
            total += float((z.float() * z.float()).sum().item())
            del z
    return total


def second_order_one(ctx: Context, seed: int, h2: float, base_loss: float) -> Tuple[float, float, Dict[str, float]]:
    restore(ctx)
    norm_sq = direction_norm_sq(ctx, seed)
    apply_signed(ctx, seed, h2, +1.0)
    loss1 = compute_loss(ctx)
    apply_signed(ctx, seed, h2, +1.0)
    loss2 = compute_loss(ctx)
    restore(ctx)
    k = (loss2 - 2.0 * loss1 + base_loss) / max(float(h2) ** 2, 1e-30)
    lam = abs(float(k)) / (norm_sq + EPS)
    eff = pair_effective_stats(ctx, seed, h2)
    return float(k), float(lam), eff


def summarize_l_rows(rows: List[Dict[str, Any]]) -> None:
    rows.sort(key=lambda r: float(r["h2"]))
    hs = [float(r["h2"]) for r in rows]
    by_h = {float(r["h2"]): r for r in rows}
    for i, r in enumerate(rows):
        h = float(r["h2"])
        h2 = next((x for x in hs if abs(x - 2 * h) <= max(1e-12, 1e-9 * abs(2 * h))), None)
        if h2 is not None:
            other = by_h[h2]
            a = safe_float(r.get("lambda_q90"))
            b = safe_float(other.get("lambda_q90"))
            if a is not None and b is not None:
                r["stability_q90_2x"] = abs(a - b) / (abs(a) + EPS)
        if i + 1 < len(rows):
            nxt = rows[i + 1]
            a = safe_float(r.get("lambda_q90"))
            b = safe_float(nxt.get("lambda_q90"))
            if a is not None and b is not None:
                r["stability_q90_next"] = abs(a - b) / (abs(a) + EPS)
                r["log_slope_q90_next"] = abs(math.log(b + EPS) - math.log(a + EPS)) / abs(math.log(float(nxt["h2"])) - math.log(h))
        if i > 0:
            prv = rows[i - 1]
            a = safe_float(r.get("lambda_q90"))
            b = safe_float(prv.get("lambda_q90"))
            if a is not None and b is not None:
                r["stability_q90_prev"] = abs(a - b) / (abs(a) + EPS)
                r["log_slope_q90_prev"] = abs(math.log(a + EPS) - math.log(b + EPS)) / abs(math.log(h) - math.log(float(prv["h2"])))
    for i, r in enumerate(rows):
        low = False
        q90 = safe_float(r.get("lambda_q90"))
        if q90 is not None:
            larger = [float(x["lambda_q90"]) for x in rows if float(x["h2"]) > float(r["h2"]) and safe_float(x.get("lambda_q90")) is not None]
            if larger and q90 / (float(np.median(larger)) + EPS) >= 5.0:
                low = True
            if i + 1 < len(rows) and safe_float(rows[i + 1].get("lambda_q90")) is not None and q90 / (float(rows[i + 1]["lambda_q90"]) + EPS) >= 5.0:
                low = True
        if safe_float(r.get("alignment_eff")) is not None and float(r["alignment_eff"]) < 0.99:
            low = True
        if safe_float(r.get("norm_ratio_eff")) is not None and not (0.9 <= float(r["norm_ratio_eff"]) <= 1.1):
            low = True
        if safe_float(r.get("zero_coord_frac_eff")) is not None and float(r["zero_coord_frac_eff"]) > 0.10:
            low = True
        if safe_float(r.get("finite_rate")) is not None and float(r["finite_rate"]) < 0.95:
            low = True
        r["low_h2_noise_suspected"] = bool(low)
        r["large_h2_nonlocal_suspected"] = bool(i >= len(rows) - 2 and safe_float(r.get("log_slope_q90_prev")) is not None and float(r["log_slope_q90_prev"]) > 1.5)


def l_candidates(ctx: Context, mode: str, seeds: Sequence[int]) -> List[Dict[str, Any]]:
    import torch

    if mode == "L_clean32":
        set_mode_fp32(ctx)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    elif mode == "L_oracle_precision":
        set_mode_fp16(ctx)
    else:
        raise ValueError(mode)
    base_loss = compute_loss(ctx)
    rows: List[Dict[str, Any]] = []
    for h2 in H_GRID:
        ks: List[float] = []
        lams: List[float] = []
        eff_rows: List[Dict[str, float]] = []
        for i, seed in enumerate(seeds):
            k, lam, eff = second_order_one(ctx, int(seed), float(h2), float(base_loss))
            ks.append(k)
            lams.append(lam)
            if i < min(4, len(seeds)):
                eff_rows.append(eff)
        k_arr = np.asarray(ks, dtype=np.float64)
        l_arr = np.asarray(lams, dtype=np.float64)
        fin_k = k_arr[np.isfinite(k_arr)]
        fin_l = l_arr[np.isfinite(l_arr)]
        med = float(np.median(fin_k)) if fin_k.size else float("nan")
        mad = float(np.median(np.abs(fin_k - med))) if fin_k.size else float("nan")
        med_abs = float(np.median(np.abs(fin_k))) if fin_k.size else float("nan")
        eff = {k: float(np.nanmean([x[k] for x in eff_rows])) if eff_rows else float("nan") for k in ["alignment_eff", "norm_ratio_eff", "zero_coord_frac_eff", "rms_snap_error"]}
        rows.append(
            {
                "group": ctx.setting.group,
                "model": ctx.setting.model_label,
                "dataset": ctx.setting.dataset,
                "seed": ctx.setting.seed,
                "L_mode": mode,
                "h2": float(h2),
                "m_L": len(seeds),
                "lambda_q50": float(np.quantile(fin_l, 0.50)) if fin_l.size else float("nan"),
                "lambda_q90": float(np.quantile(fin_l, 0.90)) if fin_l.size else float("nan"),
                "lambda_q95": float(np.quantile(fin_l, 0.95)) if fin_l.size else float("nan"),
                "median_abs_K": med_abs,
                "MAD_K": mad,
                "SNR2": float(med_abs / (1.4826 * mad + EPS)) if math.isfinite(med_abs) and math.isfinite(mad) else float("nan"),
                "finite_rate": float(np.mean(np.isfinite(k_arr))) if k_arr.size else 0.0,
                "zero_K_frac": float(np.mean(np.abs(fin_k) < EPS)) if fin_k.size else float("nan"),
                "stability_q90_2x": float("nan"),
                "stability_q90_next": float("nan"),
                "stability_q90_prev": float("nan"),
                "log_slope_q90_next": float("nan"),
                "log_slope_q90_prev": float("nan"),
                "low_h2_noise_suspected": False,
                "large_h2_nonlocal_suspected": False,
                **eff,
            }
        )
        print(f"[L] {ctx.setting.group}/{ctx.setting.model_label}/{ctx.setting.dataset}/seed{ctx.setting.seed} {mode} h2={h2:g}", flush=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    summarize_l_rows(rows)
    return rows


def select_l(rows: List[Dict[str, Any]], selector: str) -> Dict[str, Any]:
    rows = sorted(rows, key=lambda r: float(r["h2"]))
    candidates: List[Dict[str, Any]] = []
    for r in rows:
        q90 = safe_float(r.get("lambda_q90"))
        if q90 is None:
            continue
        finite_rate = safe_float(r.get("finite_rate"))
        if finite_rate is None or finite_rate < 0.95:
            continue
        if bool(r.get("low_h2_noise_suspected")):
            continue
        stab_vals = [safe_float(r.get(k)) for k in ["stability_q90_2x", "stability_q90_next", "stability_q90_prev"]]
        stab_vals = [x for x in stab_vals if x is not None]
        slope_vals = [safe_float(r.get(k)) for k in ["log_slope_q90_next", "log_slope_q90_prev"]]
        slope_vals = [x for x in slope_vals if x is not None]
        if stab_vals and min(stab_vals) > 0.5:
            continue
        if slope_vals and min(slope_vals) > 1.0:
            continue
        candidates.append(r)
    if candidates:
        chosen = candidates[0]
        status = "selected"
        flags = ""
    else:
        chosen = min(rows, key=lambda r: (bool(r.get("low_h2_noise_suspected")), safe_float(r.get("stability_q90_next")) or float("inf"), float(r["h2"])))
        status = "fallback_unreliable"
        flags = "fallback_plateau"
    return {
        "group": chosen["group"],
        "model": chosen["model"],
        "dataset": chosen["dataset"],
        "seed": chosen["seed"],
        "L_mode": chosen["L_mode"],
        "selector": selector,
        "selected_h2": chosen["h2"],
        "selected_L_q50": chosen["lambda_q50"],
        "selected_L_q90": chosen["lambda_q90"],
        "selected_L_q95": chosen["lambda_q95"],
        "selection_status": status,
        "flags": flags,
    }


def old_snr_l(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    finite = [r for r in rows if safe_float(r.get("SNR2")) is not None]
    passing = [r for r in finite if float(r["SNR2"]) >= 2.0]
    chosen = passing[0] if passing else max(finite, key=lambda r: float(r["SNR2"]))
    return {
        "group": chosen["group"],
        "model": chosen["model"],
        "dataset": chosen["dataset"],
        "seed": chosen["seed"],
        "L_mode": chosen["L_mode"],
        "selector": "old_snr_max_fallback_ablation",
        "selected_h2": chosen["h2"],
        "selected_L_q50": chosen["lambda_q50"],
        "selected_L_q90": chosen["lambda_q90"],
        "selected_L_q95": chosen["lambda_q95"],
        "selection_status": "selected",
        "flags": "ablation_only" if passing else "ablation_only;fallback_max_snr",
    }


def hstar(delta: float, g: float, lval: float, d: int) -> float:
    if min(delta, g, lval, float(d)) <= 0 or not all(math.isfinite(x) for x in [delta, g, lval]):
        return float("nan")
    return float((delta * delta * g * g / (16.0 * lval * lval * float(d) * float(d + 2))) ** 0.25)


def compute_hstar_rows(
    ctx: Context,
    ulp: Dict[str, Any],
    probe_rows: List[Dict[str, Any]],
    g_rows: List[Dict[str, Any]],
    l_selected: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    by_h = {float(r["h"]): r for r in probe_rows}
    empirical_min = min(probe_rows, key=lambda r: float(r["nmse"]) if safe_float(r.get("nmse")) is not None else float("inf"))
    empirical_corr = max(probe_rows, key=lambda r: float(r["corr"]) if safe_float(r.get("corr")) is not None else -float("inf"))
    primary_g = next((r for r in g_rows if r["G_method"] == "absG"), None)
    rich_g = next((r for r in g_rows if r["G_method"] == "richardsonG_candidate"), None)
    l_by = {(r["L_mode"], r["selector"]): r for r in l_selected}
    rows: List[Dict[str, Any]] = []

    def add_selector(name: str, delta_mode: str, g_row: Dict[str, Any], l_row: Dict[str, Any], q: str, notes: str = ""):
        delta = float(ulp.get(delta_mode, float("nan")))
        lval = float(l_row.get(f"selected_L_{q}", float("nan")))
        ghat = float(g_row.get("G_hat", float("nan")))
        hs = hstar(delta, ghat, lval, ctx.d_trainable)
        ng = nearest_grid(hs)
        metric = by_h.get(float(ng), {})
        nmse_sel = safe_float(metric.get("nmse"))
        corr_sel = safe_float(metric.get("corr"))
        min_nmse = float(empirical_min["nmse"])
        max_corr = float(empirical_corr["corr"])
        nmse_ratio = float(nmse_sel / (min_nmse + EPS)) if nmse_sel is not None else float("nan")
        corr_gap = float(max_corr - corr_sel) if corr_sel is not None and math.isfinite(max_corr) else float("nan")
        rows.append(
            {
                "group": ctx.setting.group,
                "model": ctx.setting.model_label,
                "dataset": ctx.setting.dataset,
                "seed": ctx.setting.seed,
                "selector": name,
                "Delta_mode": delta_mode,
                "Delta_value": delta,
                "G_method": g_row.get("G_method"),
                "G_hat": ghat,
                "G_h_used": g_row.get("h_G"),
                "L_mode": l_row.get("L_mode"),
                "L_q": q,
                "L_hat": lval,
                "L_h2_used": l_row.get("selected_h2"),
                "d_trainable": ctx.d_trainable,
                "hstar_cont": hs,
                "hstar_nearest_grid": ng,
                "nmse_at_selected_h": nmse_sel,
                "corr_at_selected_h": corr_sel,
                "empirical_min_nmse_h": empirical_min["h"],
                "empirical_min_nmse": min_nmse,
                "empirical_max_corr_h": empirical_corr["h"],
                "empirical_max_corr": max_corr,
                "nmse_ratio": nmse_ratio,
                "corr_gap": corr_gap,
                "success_pass": bool((math.isfinite(nmse_ratio) and nmse_ratio <= 1.25) or (math.isfinite(corr_gap) and corr_gap <= 0.01)),
                "strict_success_pass": bool(math.isfinite(nmse_ratio) and nmse_ratio <= 1.10),
                "notes": notes,
            }
        )

    if primary_g:
        clean = l_by.get(("L_clean32", "plateau_q90_primary"))
        oracle = l_by.get(("L_oracle_precision", "plateau_q90_primary"))
        old = l_by.get(("L_oracle_precision", "old_snr_max_fallback_ablation"))
        if clean:
            add_selector("calibrated_hstar_absG_Lclean32_q90", "delta_ulp_rms", primary_g, clean, "q90")
            add_selector("calibrated_hstar_absG_Lclean32_q50", "delta_ulp_rms", primary_g, clean, "q50")
            add_selector("calibrated_hstar_absG_Lclean32_q95", "delta_ulp_rms", primary_g, clean, "q95")
        if oracle:
            add_selector("calibrated_hstar_absG_Loracle_q90", "delta_ulp_rms", primary_g, oracle, "q90", "diagnostic oracle-precision L")
        if old:
            add_selector("oldSNR_L_ablation", "delta_ulp_rms", primary_g, old, "q90", "old SNR L ablation only")
    if rich_g and l_by.get(("L_clean32", "plateau_q90_primary")):
        add_selector("calibrated_hstar_richardsonG_Lclean32_q90", "delta_ulp_rms", rich_g, l_by[("L_clean32", "plateau_q90_primary")], "q90")

    for name, hval in [("mezo_default_h_1e-3", 1e-3), ("empirical_probe_min_nmse", float(empirical_min["h"])), ("empirical_probe_max_corr", float(empirical_corr["h"]))]:
        metric = by_h[hval]
        rows.append(
            {
                "group": ctx.setting.group,
                "model": ctx.setting.model_label,
                "dataset": ctx.setting.dataset,
                "seed": ctx.setting.seed,
                "selector": name,
                "Delta_mode": "",
                "Delta_value": "",
                "G_method": "",
                "G_hat": "",
                "G_h_used": "",
                "L_mode": "",
                "L_q": "",
                "L_hat": "",
                "L_h2_used": "",
                "d_trainable": ctx.d_trainable,
                "hstar_cont": hval,
                "hstar_nearest_grid": hval,
                "nmse_at_selected_h": metric["nmse"],
                "corr_at_selected_h": metric["corr"],
                "empirical_min_nmse_h": empirical_min["h"],
                "empirical_min_nmse": empirical_min["nmse"],
                "empirical_max_corr_h": empirical_corr["h"],
                "empirical_max_corr": empirical_corr["corr"],
                "nmse_ratio": float(float(metric["nmse"]) / (float(empirical_min["nmse"]) + EPS)),
                "corr_gap": float(float(empirical_corr["corr"]) - float(metric["corr"])),
                "success_pass": True,
                "strict_success_pass": True,
                "notes": "reference selector",
            }
        )
    return rows


def suggested_runs(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    by_setting: Dict[Tuple[str, str, int], List[Dict[str, Any]]] = {}
    for r in rows:
        by_setting.setdefault((str(r["model"]), str(r["dataset"]), int(r["seed"])), []).append(r)
    for (model, dataset, seed), items in by_setting.items():
        hvals = {1e-3, 3e-5}
        for sel in ["calibrated_hstar_absG_Lclean32_q90", "empirical_probe_min_nmse"]:
            row = next((x for x in items if x["selector"] == sel), None)
            if row and safe_float(row.get("hstar_nearest_grid")) is not None:
                hvals.add(float(row["hstar_nearest_grid"]))
        out.append({"model": model, "dataset": dataset, "seed": seed, "candidate_h": sorted(hvals)})
    return out


def plot_setting(out_dir: Path, setting: Setting, probe_rows: List[Dict[str, Any]], hstar_rows: List[Dict[str, Any]], l_rows: List[Dict[str, Any]]) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    plot_dir = out_dir / "plots" / f"{setting.group}_{setting.model_label}_{setting.dataset}_seed{setting.seed}".replace("/", "_")
    plot_dir.mkdir(parents=True, exist_ok=True)
    hs = np.asarray([float(r["h"]) for r in probe_rows], dtype=np.float64)
    nmse = np.asarray([float(r["nmse"]) for r in probe_rows], dtype=np.float64)
    corr = np.asarray([float(r["corr"]) for r in probe_rows], dtype=np.float64)
    marks = [r for r in hstar_rows if r["selector"] in {"calibrated_hstar_absG_Lclean32_q90", "mezo_default_h_1e-3", "empirical_probe_min_nmse"}]

    for y, name, ylabel in [(nmse, "nmse_vs_h.png", "nMSE"), (corr, "corr_vs_h.png", "corr")]:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(hs, y, marker="o")
        for r in marks:
            h = safe_float(r.get("hstar_nearest_grid"))
            if h is not None:
                ax.axvline(h, linestyle="--", alpha=0.5, label=str(r["selector"]))
        ax.set_xscale("log")
        ax.set_xlabel("h")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(plot_dir / name, dpi=160)
        plt.close(fig)

    for mode in ["L_clean32", "L_oracle_precision"]:
        rows = [r for r in l_rows if r["L_mode"] == mode]
        if not rows:
            continue
        fig, ax = plt.subplots(figsize=(7, 4))
        x = np.asarray([float(r["h2"]) for r in rows], dtype=np.float64)
        for q in ["lambda_q50", "lambda_q90", "lambda_q95"]:
            ax.plot(x, [float(r[q]) for r in rows], marker="o", label=q)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("h2")
        ax.set_ylabel("lambda")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(plot_dir / f"{mode}_lambda_vs_h2.png", dpi=160)
        plt.close(fig)


def markdown_summary(hstar_rows: List[Dict[str, Any]], diagnostics: Dict[str, Any]) -> str:
    primary = [r for r in hstar_rows if r["selector"] == "calibrated_hstar_absG_Lclean32_q90"]
    lines = [
        "# FP16 h-star generalization summary",
        "",
        f"Generated: {dt.datetime.now().isoformat()}",
        "",
        "Primary selector: `calibrated_hstar_absG_Lclean32_q90`.",
        "",
    ]
    for title, group in [("Seed robustness: RoBERTa-large/SST-5 FP16", "A_seed"), ("Task robustness: RoBERTa-large/RTE FP16", "B_task"), ("Model robustness: OPT/SST-2 FP16", "C_model")]:
        lines += [f"## {title}", "", "| model | dataset | seed | hstar h | empirical min-MSE h | nmse ratio | corr gap | pass | L_h2 | G_h |", "|---|---|---:|---:|---:|---:|---:|---|---:|---:|"]
        for r in [x for x in primary if x["group"] == group]:
            lines.append(
                f"| {r['model']} | {r['dataset']} | {r['seed']} | {float(r['hstar_nearest_grid']):.6g} | "
                f"{float(r['empirical_min_nmse_h']):.6g} | {float(r['nmse_ratio']):.4g} | {float(r['corr_gap']):.4g} | "
                f"{r['success_pass']} | {float(r['L_h2_used']):.6g} | {float(r['G_h_used']):.6g} |"
            )
        if not [x for x in primary if x["group"] == group]:
            lines.append("| _none completed_ | | | | | | | | | |")
        lines.append("")
    if diagnostics.get("skipped_settings"):
        lines += ["## Skipped settings", ""]
        for item in diagnostics["skipped_settings"]:
            lines.append(f"- {item.get('group')} {item.get('model')} {item.get('dataset')} seed {item.get('seed')}: {item.get('reason')}")
        lines.append("")
    return "\n".join(lines) + "\n"


def resolve_settings(include_opt: bool) -> Tuple[List[Setting], List[Dict[str, Any]]]:
    settings: List[Setting] = []
    skipped: List[Dict[str, Any]] = []
    for seed in [16, 17, 18, 19]:
        settings.append(Setting("A_seed", "roberta-large", "roberta-large", "sst-5", seed, seed))
    for seed in [16, 17, 18]:
        settings.append(Setting("B_task", "roberta-large", "roberta-large", "rte", seed, seed))
    if include_opt:
        candidates = [("OPT-1.3B", "facebook/opt-1.3b", False), ("fallback_OPT-350M", "facebook/opt-350m", True), ("fallback_OPT-125M", "facebook/opt-125m", True)]
        usable: Optional[Tuple[str, str, bool]] = None
        for label, name, fallback in candidates:
            try:
                # Config load is a cheap support/cache check; model load happens later.
                if str(MEDIUM_ROOT) not in sys.path:
                    sys.path.insert(0, str(MEDIUM_ROOT))
                from transformers import AutoConfig

                AutoConfig.from_pretrained(name)
                usable = (label, name, fallback)
                break
            except Exception as exc:
                skipped.append({"group": "C_model", "model": label, "dataset": "sst-2", "seed": "", "reason": f"config unavailable: {exc}"})
        if usable is None:
            skipped.append({"group": "C_model", "model": "OPT", "dataset": "sst-2", "seed": "", "reason": "no OPT-1.3B/350M/125M config available"})
        else:
            label, name, fallback = usable
            for seed in [16, 17]:
                settings.append(Setting("C_model", label, name, "sst-2", seed, seed, fallback=fallback))
    return settings, skipped


def analyze_setting(
    setting: Setting,
    args: argparse.Namespace,
    out_dir: Path,
    diagnostics: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    import torch

    device = torch.device("cuda:0")
    t0 = time.time()
    print(f"[setting] start {setting.group}/{setting.model_label}/{setting.dataset}/seed{setting.seed}", flush=True)
    ctx = load_context(setting, device, args.batch_size)
    diagnostics.setdefault("settings", []).append(
        {
            "group": setting.group,
            "model_label": setting.model_label,
            "model_name": setting.model_name,
            "dataset": setting.dataset,
            "seed": setting.seed,
            "data_seed": setting.data_seed,
            "data_info": ctx.data_info,
            "d_trainable": ctx.d_trainable,
            "fallback": setting.fallback,
        }
    )
    ulp = estimate_ulp(ctx)
    diagnostics.setdefault("Delta_estimates", {})[f"{setting.group}/{setting.model_label}/{setting.dataset}/seed{setting.seed}"] = ulp
    true_loss, grads, truth_kind = compute_true_grads(ctx)
    fp32_initial_backups = [b.detach().clone() for b in ctx.backups]
    diagnostics.setdefault("true_grad", {})[f"{setting.group}/{setting.model_label}/{setting.dataset}/seed{setting.seed}"] = {
        "base_loss": true_loss,
        "truth_kind": truth_kind,
    }
    p_seeds = direction_seeds(setting.seed, args.num_probe_dirs, 0)
    l_seeds = direction_seeds(setting.seed, args.num_L_dirs, 1)
    probe_rows, fd_by_h, _, _ = probe_grid(ctx, grads, true_loss, p_seeds)
    g_rows = estimate_g(ctx, probe_rows, fd_by_h)

    l_rows: List[Dict[str, Any]] = []
    l_sel: List[Dict[str, Any]] = []
    ctx.model.float()
    ctx.forward_precision = "fp32"
    ctx.direction_dtype_name = "float32"
    restore_external_backups(ctx, fp32_initial_backups)
    clean_rows = l_candidates(ctx, "L_clean32", l_seeds)
    ctx.model.float()
    ctx.forward_precision = "fp32"
    ctx.direction_dtype_name = "float32"
    restore_external_backups(ctx, fp32_initial_backups)
    oracle_rows = l_candidates(ctx, "L_oracle_precision", l_seeds)
    l_rows.extend(clean_rows)
    l_rows.extend(oracle_rows)
    l_sel.append(select_l(clean_rows, "plateau_q90_primary"))
    l_sel.append(select_l(oracle_rows, "plateau_q90_primary"))
    l_sel.append(old_snr_l(oracle_rows))
    hstar_rows = compute_hstar_rows(ctx, ulp, probe_rows, g_rows, l_sel)
    plot_setting(out_dir, setting, probe_rows, hstar_rows, l_rows)
    diagnostics.setdefault("runtime_sec_by_setting", {})[f"{setting.group}/{setting.model_label}/{setting.dataset}/seed{setting.seed}"] = time.time() - t0
    del ctx, grads
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"[setting] done {setting.group}/{setting.model_label}/{setting.dataset}/seed{setting.seed} elapsed={time.time()-t0:.1f}s", flush=True)
    return probe_rows, g_rows, l_rows, l_sel, hstar_rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="")
    parser.add_argument("--num_probe_dirs", type=int, default=32)
    parser.add_argument("--num_L_dirs", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--skip_opt", action="store_true")
    parser.add_argument("--max_settings", type=int, default=0)
    parser.add_argument("--only_group", default="", help="Optional group filter, e.g. A_seed, B_task, or C_model.")
    args = parser.parse_args()

    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir) if args.output_dir else REPO_ROOT / "analysis" / f"fp16_hstar_generalization_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=False)
    (out_dir / "plots").mkdir(parents=True, exist_ok=True)
    report = env_report()
    (out_dir / "env_report.txt").write_text(report, encoding="utf-8")

    diagnostics: Dict[str, Any] = {
        "start_time": dt.datetime.now().isoformat(),
        "git_commit": run_cmd(["git", "rev-parse", "HEAD"]),
        "env": report,
        "h_grid": H_GRID,
        "num_probe_dirs": args.num_probe_dirs,
        "num_L_dirs": args.num_L_dirs,
        "warnings": [],
        "skipped_settings": [],
    }
    if args.num_probe_dirs < 64:
        diagnostics["warnings"].append(f"m_probe reduced to {args.num_probe_dirs} for runtime; requested rule allows reduction when 64 is not feasible.")
    if args.num_L_dirs < 32:
        diagnostics["warnings"].append(f"m_L reduced to {args.num_L_dirs} for runtime.")

    try:
        import torch

        if not torch.cuda.is_available():
            (out_dir / "failure_report.txt").write_text("CUDA unavailable; refusing to write empty scientific CSVs.\n", encoding="utf-8")
            write_json(out_dir / "diagnostics.json", diagnostics)
            return 2
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    except Exception as exc:
        (out_dir / "failure_report.txt").write_text(f"torch/CUDA startup failed: {exc}\n", encoding="utf-8")
        write_json(out_dir / "diagnostics.json", diagnostics)
        return 2

    settings, skipped = resolve_settings(include_opt=not args.skip_opt)
    if args.only_group:
        settings = [s for s in settings if s.group == args.only_group]
    diagnostics["skipped_settings"].extend(skipped)
    if args.max_settings > 0:
        settings = settings[: args.max_settings]
    all_probe: List[Dict[str, Any]] = []
    all_g: List[Dict[str, Any]] = []
    all_l: List[Dict[str, Any]] = []
    all_l_sel: List[Dict[str, Any]] = []
    all_hstar: List[Dict[str, Any]] = []
    for setting in settings:
        try:
            probe_rows, g_rows, l_rows, l_sel, hstar_rows = analyze_setting(setting, args, out_dir, diagnostics)
            all_probe.extend(probe_rows)
            all_g.extend([{k: v for k, v in r.items() if not k.startswith("_")} for r in g_rows])
            all_l.extend(l_rows)
            all_l_sel.extend(l_sel)
            all_hstar.extend(hstar_rows)
            write_csv(out_dir / "probe_grid_metrics.csv", all_probe, PROBE_FIELDS)
            write_csv(out_dir / "G_estimates.csv", all_g, G_FIELDS)
            write_csv(out_dir / "L_candidates.csv", all_l, L_CAND_FIELDS)
            write_csv(out_dir / "L_selected.csv", all_l_sel, L_SEL_FIELDS)
            write_csv(out_dir / "hstar_estimates.csv", all_hstar, HSTAR_FIELDS)
            write_json(out_dir / "diagnostics.json", diagnostics)
        except Exception as exc:
            tb = traceback.format_exc()
            diagnostics.setdefault("skipped_settings", []).append(
                {
                    "group": setting.group,
                    "model": setting.model_label,
                    "dataset": setting.dataset,
                    "seed": setting.seed,
                    "reason": repr(exc),
                    "traceback": tb,
                }
            )
            print(f"[setting] skipped {setting.group}/{setting.model_label}/{setting.dataset}/seed{setting.seed}: {exc}\n{tb}", flush=True)
            write_json(out_dir / "diagnostics.json", diagnostics)

    write_csv(out_dir / "probe_grid_metrics.csv", all_probe, PROBE_FIELDS)
    write_csv(out_dir / "G_estimates.csv", all_g, G_FIELDS)
    write_csv(out_dir / "L_candidates.csv", all_l, L_CAND_FIELDS)
    write_csv(out_dir / "L_selected.csv", all_l_sel, L_SEL_FIELDS)
    write_csv(out_dir / "hstar_estimates.csv", all_hstar, HSTAR_FIELDS)
    write_json(out_dir / "suggested_short_training_runs.json", suggested_runs(all_hstar))
    diagnostics["end_time"] = dt.datetime.now().isoformat()
    diagnostics["settings_attempted"] = len(settings)
    completed_keys = {(r["group"], r["model"], r["dataset"], int(r["seed"])) for r in all_hstar if r["selector"] == "calibrated_hstar_absG_Lclean32_q90"}
    diagnostics["settings_completed"] = len(completed_keys)
    write_json(out_dir / "diagnostics.json", diagnostics)
    (out_dir / "hstar_generalization_summary.md").write_text(markdown_summary(all_hstar, diagnostics), encoding="utf-8")

    primary = [r for r in all_hstar if r["selector"] == "calibrated_hstar_absG_Lclean32_q90"]
    pass_rate = float(np.mean([bool(r["success_pass"]) for r in primary])) if primary else float("nan")
    print(f"Analysis output directory: {out_dir}")
    print(f"Total settings attempted: {len(settings)}")
    print(f"Total settings completed: {len(completed_keys)}")
    print(f"Pass rate calibrated_hstar_absG_Lclean32_q90: {pass_rate:.4g}")
    for group in ["A_seed", "B_task", "C_model"]:
        rows = [r for r in primary if r["group"] == group]
        rate = float(np.mean([bool(r["success_pass"]) for r in rows])) if rows else float("nan")
        print(f"{group} pass rate: {rate:.4g} ({len(rows)} settings)")
    if diagnostics.get("skipped_settings"):
        print("Skipped settings:")
        for item in diagnostics["skipped_settings"]:
            print(f"  {item}")
    print(f"Suggested short-training h values: {out_dir / 'suggested_short_training_runs.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
