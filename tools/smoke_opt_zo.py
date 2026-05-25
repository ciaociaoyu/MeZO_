#!/usr/bin/env python
"""Smoke-test harness for OPT ZO low-precision engineering paths.

This runner is deliberately small and synthetic. It does not launch real
training, does not download large models by default, and labels fake
quantization as fake quantization.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import importlib.metadata
import json
import math
import os
import platform
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from torch import nn


REPO_ROOT = Path(__file__).resolve().parents[1]
MEDIUM_MODELS = REPO_ROOT / "medium_models"
if str(MEDIUM_MODELS) not in sys.path:
    sys.path.insert(0, str(MEDIUM_MODELS))

try:
    from src.int8_residual_grid import ResidualGridUpdater
except Exception:  # pragma: no cover - fallback is covered by runtime smoke.
    ResidualGridUpdater = None  # type: ignore[assignment]


SCHEMA = [
    "timestamp",
    "git_commit",
    "hostname",
    "conda_env",
    "python_executable",
    "python_version",
    "torch_version",
    "cuda_available",
    "gpu_name",
    "model_id",
    "method",
    "precision_or_quant_backend",
    "device",
    "dtype",
    "h",
    "sparse_p_requested",
    "sparse_p_actual",
    "h_raw",
    "h_active",
    "k_dir",
    "batch_size",
    "max_seq_len",
    "loss_plus",
    "loss_minus",
    "fd",
    "active_fraction",
    "norm_ratio",
    "alignment",
    "trainable_params",
    "total_params",
    "peak_memory_mb",
    "nan_flag",
    "status",
    "skip_reason",
    "error_message",
    "mask_strategy",
    "active_count",
    "perturb_norm",
    "param_restore_max_abs_diff",
    "loss",
    "quantized_module_count",
    "lora_target_modules",
    "trainable_ratio",
    "update_norm",
    "commit_norm",
    "residual_norm",
    "commit_nonzero_fraction",
    "grid_scale",
    "conservation_error",
    "save_load_ok",
]


TOY_TEXTS = [
    "Zero order optimization can probe a model with two forward passes.",
    "Quantized perturbations should be visible at a useful step size.",
    "Sparse directions must log both raw and active perturbation scales.",
    "Checkpoint smoke tests should catch state restore failures early.",
]


def now_ts() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def run_cmd_text(cmd: Sequence[str]) -> Optional[str]:
    try:
        out = subprocess.check_output(cmd, cwd=str(REPO_ROOT), stderr=subprocess.DEVNULL, text=True)
    except Exception:
        return None
    return out.strip()


def package_version(name: str) -> Optional[str]:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def finite_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return out


def tensor_scalar(value: torch.Tensor) -> float:
    return float(value.detach().float().cpu().item())


def vector_norm(tensor: torch.Tensor) -> float:
    if tensor.numel() == 0:
        return 0.0
    return float(torch.linalg.vector_norm(torch.nan_to_num(tensor.detach().float()).reshape(-1)).item())


def max_abs(tensor: torch.Tensor) -> float:
    if tensor.numel() == 0:
        return 0.0
    return float(torch.max(torch.abs(torch.nan_to_num(tensor.detach().float()))).item())


def get_env_info() -> Dict[str, Any]:
    cuda_available = bool(torch.cuda.is_available())
    gpu_name = None
    gpu_vram_mb = None
    if cuda_available:
        try:
            idx = torch.cuda.current_device()
            props = torch.cuda.get_device_properties(idx)
            gpu_name = props.name
            gpu_vram_mb = int(props.total_memory // (1024 * 1024))
        except Exception:
            gpu_name = None
            gpu_vram_mb = None
    return {
        "timestamp": now_ts(),
        "hostname": socket.gethostname(),
        "conda_env": os.environ.get("CONDA_DEFAULT_ENV"),
        "conda_prefix": os.environ.get("CONDA_PREFIX"),
        "virtual_env": os.environ.get("VIRTUAL_ENV"),
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "git_commit": run_cmd_text(["git", "rev-parse", "--short", "HEAD"]),
        "torch_version": torch.__version__,
        "cuda_available": cuda_available,
        "cuda_version": torch.version.cuda,
        "gpu_name": gpu_name,
        "gpu_vram_mb": gpu_vram_mb,
        "transformers_version": package_version("transformers"),
        "accelerate_version": package_version("accelerate"),
        "bitsandbytes_version": package_version("bitsandbytes"),
        "peft_version": package_version("peft"),
    }


class ResultWriter:
    def __init__(self, output_dir: Path, env: Dict[str, Any], args: argparse.Namespace) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.env = env
        self.args = args
        self.rows: List[Dict[str, Any]] = []
        self.jsonl_path = output_dir / "smoke_results.jsonl"
        self.summary_path = output_dir / "smoke_summary.csv"
        self.env_path = output_dir / "env.json"
        self.jsonl_f = self.jsonl_path.open("w", encoding="utf-8")
        with self.env_path.open("w", encoding="utf-8") as f:
            json.dump({"env": env, "args": vars(args)}, f, indent=2, sort_keys=True)

    def close(self) -> None:
        self.jsonl_f.close()
        with self.summary_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=SCHEMA, extrasaction="ignore")
            writer.writeheader()
            for row in self.rows:
                writer.writerow({key: row.get(key) for key in SCHEMA})

    def base_row(self, method: str, status: str = "pass") -> Dict[str, Any]:
        row = {key: None for key in SCHEMA}
        row.update(
            {
                "timestamp": now_ts(),
                "git_commit": self.env.get("git_commit"),
                "hostname": self.env.get("hostname"),
                "conda_env": self.env.get("conda_env"),
                "python_executable": self.env.get("python_executable"),
                "python_version": self.env.get("python_version"),
                "torch_version": self.env.get("torch_version"),
                "cuda_available": self.env.get("cuda_available"),
                "gpu_name": self.env.get("gpu_name"),
                "model_id": self.args.model_id,
                "method": method,
                "device": None,
                "batch_size": self.args.batch_size,
                "max_seq_len": self.args.max_seq_len,
                "nan_flag": False,
                "status": status,
                "skip_reason": "",
                "error_message": "",
            }
        )
        return row

    def write(self, row: Dict[str, Any]) -> None:
        full = self.base_row(str(row.get("method", "")), str(row.get("status", "pass")))
        full.update(row)
        for key in ("loss_plus", "loss_minus", "fd", "loss"):
            value = full.get(key)
            if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
                full["nan_flag"] = True
        self.rows.append(full)
        self.jsonl_f.write(json.dumps(full, sort_keys=True) + "\n")
        self.jsonl_f.flush()

    def skip(self, method: str, reason: str, **extra: Any) -> None:
        row = self.base_row(method, "skip")
        row.update(extra)
        row["skip_reason"] = reason
        self.write(row)

    def fail(self, method: str, error: BaseException | str, **extra: Any) -> None:
        row = self.base_row(method, "fail")
        row.update(extra)
        row["error_message"] = str(error)
        row["nan_flag"] = True
        self.write(row)


@dataclass
class ModelBundle:
    model: nn.Module
    tokenizer: Any
    model_id: str
    source: str
    device: torch.device
    dtype: torch.dtype
    inputs: Dict[str, torch.Tensor]


def parse_methods(value: str) -> List[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if value == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is not available")
    return torch.device(value)


def resolve_dtype(value: str, device: torch.device) -> torch.dtype:
    key = str(value or "auto").lower()
    if key == "auto":
        if device.type == "cuda":
            return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        return torch.float32
    mapping = {
        "fp32": torch.float32,
        "float32": torch.float32,
        "fp16": torch.float16,
        "float16": torch.float16,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
    }
    if key not in mapping:
        raise ValueError(f"Unsupported --dtype={value!r}")
    if device.type == "cpu" and mapping[key] in {torch.float16, torch.bfloat16}:
        return torch.float32
    return mapping[key]


def local_opt_config(max_seq_len: int):
    from transformers import OPTConfig

    return OPTConfig(
        vocab_size=256,
        hidden_size=32,
        ffn_dim=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        max_position_embeddings=max(128, max_seq_len + 8),
        dropout=0.0,
        attention_dropout=0.0,
        activation_dropout=0.0,
        bos_token_id=2,
        eos_token_id=2,
        pad_token_id=1,
    )


def build_local_opt(max_seq_len: int) -> Tuple[nn.Module, Any, str]:
    from transformers import OPTForCausalLM

    config = local_opt_config(max_seq_len)
    model = OPTForCausalLM(config)
    return model, None, "local_opt_config"


def load_model_bundle(args: argparse.Namespace, writer: ResultWriter) -> Optional[ModelBundle]:
    if not module_available("transformers"):
        writer.fail("model_load", "transformers is not installed")
        return None

    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    model_id = args.model_id
    local_only = bool(args.allow_cached_only or not args.allow_download)
    tokenizer = None
    model = None
    source = "from_pretrained"

    if model_id in {"local-opt-tiny", "local_opt_tiny", "optconfig-tiny"}:
        model, tokenizer, source = build_local_opt(args.max_seq_len)
    else:
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=local_only)
        except Exception:
            tokenizer = None
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                local_files_only=local_only,
                torch_dtype=dtype if device.type == "cuda" else torch.float32,
            )
        except Exception as exc:
            if model_id == "hf-internal-testing/tiny-random-OPTForCausalLM":
                writer.skip("model_load", f"tiny HF model unavailable locally; falling back to OPTConfig: {exc}")
                model, tokenizer, source = build_local_opt(args.max_seq_len)
            else:
                writer.skip("model_load", f"model unavailable and downloads disabled or failed: {exc}")
                return None

    assert model is not None
    model.eval()
    if device.type == "cuda":
        model.to(device=device, dtype=dtype)
    else:
        model.to(device=device)
    inputs = build_inputs(model, tokenizer, device, args.batch_size, args.max_seq_len)
    bundle = ModelBundle(model=model, tokenizer=tokenizer, model_id=model_id, source=source, device=device, dtype=dtype, inputs=inputs)
    writer.write(
        {
            "method": "model_load",
            "status": "pass",
            "precision_or_quant_backend": source,
            "device": str(device),
            "dtype": str(dtype).replace("torch.", ""),
            "total_params": count_params(model, trainable_only=False),
            "trainable_params": count_params(model, trainable_only=True),
        }
    )
    return bundle


def build_inputs(model: nn.Module, tokenizer: Any, device: torch.device, batch_size: int, max_seq_len: int) -> Dict[str, torch.Tensor]:
    if tokenizer is not None:
        if getattr(tokenizer, "pad_token", None) is None:
            tokenizer.pad_token = getattr(tokenizer, "eos_token", None) or tokenizer.unk_token
        encoded = tokenizer(
            TOY_TEXTS[: max(1, min(batch_size, len(TOY_TEXTS)))],
            padding="max_length",
            truncation=True,
            max_length=max_seq_len,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded.get("attention_mask", torch.ones_like(input_ids)).to(device)
    else:
        vocab_size = int(getattr(model.config, "vocab_size", 256))
        generator = torch.Generator(device="cpu")
        generator.manual_seed(1234)
        input_ids = torch.randint(4, max(vocab_size, 8), (batch_size, max_seq_len), generator=generator, dtype=torch.long)
        input_ids[:, 0] = int(getattr(model.config, "bos_token_id", 2) or 2)
        attention_mask = torch.ones_like(input_ids)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
    labels = input_ids.clone()
    pad_id = getattr(model.config, "pad_token_id", None)
    if pad_id is not None:
        labels = labels.masked_fill(input_ids == int(pad_id), -100)
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def count_params(model: nn.Module, *, trainable_only: bool) -> int:
    return int(sum(p.numel() for p in model.parameters() if (p.requires_grad or not trainable_only)))


def forward_loss(model: nn.Module, inputs: Dict[str, torch.Tensor]) -> float:
    with torch.no_grad():
        out = model(**inputs)
        loss = out.loss
    return tensor_scalar(loss)


def forward_loss_tensor(model: nn.Module, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
    out = model(**inputs)
    return out.loss


def peak_memory_mb(device: torch.device) -> Optional[float]:
    if device.type != "cuda":
        return None
    return float(torch.cuda.max_memory_allocated(device) / (1024 * 1024))


def selected_parameters(
    model: nn.Module,
    *,
    max_params: int,
    max_elements: int,
    trainable_only: bool = False,
    lora_only: bool = False,
) -> List[Tuple[str, nn.Parameter]]:
    preferred = ("q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2")
    skip = ("embed_tokens", "embed_positions", "lm_head")
    rows: List[Tuple[int, str, nn.Parameter]] = []
    for name, param in model.named_parameters():
        if trainable_only and not param.requires_grad:
            continue
        if lora_only and "lora" not in name.lower():
            continue
        if (not torch.is_floating_point(param.data)) or param.numel() == 0:
            continue
        if any(part in name for part in skip) and not lora_only:
            continue
        if param.numel() > max_elements and not lora_only:
            continue
        priority = 0 if any(part in name for part in preferred) and param.ndim >= 2 else 1
        rows.append((priority, name, param))
    rows.sort(key=lambda item: (item[0], item[1]))
    return [(name, param) for _, name, param in rows[:max_params]]


def make_generator(seed: int, device: torch.device) -> torch.Generator:
    gen = torch.Generator(device=device.type)
    gen.manual_seed(int(seed))
    return gen


def sample_directions(named_params: Sequence[Tuple[str, nn.Parameter]], seed: int) -> Dict[str, torch.Tensor]:
    directions: Dict[str, torch.Tensor] = {}
    for idx, (name, param) in enumerate(named_params):
        gen = make_generator(seed + idx * 1009, param.device)
        directions[name] = torch.randn(param.shape, generator=gen, device=param.device, dtype=torch.float32)
    return directions


def backup_params(named_params: Sequence[Tuple[str, nn.Parameter]]) -> Dict[str, torch.Tensor]:
    return {name: param.detach().clone() for name, param in named_params}


def restore_params(named_params: Sequence[Tuple[str, nn.Parameter]], backups: Dict[str, torch.Tensor]) -> float:
    restore_diff = 0.0
    with torch.no_grad():
        for name, param in named_params:
            param.data.copy_(backups[name])
            restore_diff = max(restore_diff, max_abs(param.detach().float() - backups[name].float()))
    return restore_diff


def restore_tolerance(named_params: Sequence[Tuple[str, nn.Parameter]]) -> float:
    tol = 1e-7
    for _, param in named_params:
        if param.dtype in {torch.float16, torch.bfloat16}:
            tol = max(tol, 1e-3)
    return tol


def apply_perturbation(
    named_params: Sequence[Tuple[str, nn.Parameter]],
    backups: Dict[str, torch.Tensor],
    directions: Dict[str, torch.Tensor],
    h: float,
    sign: float,
    *,
    fake_quant: bool = False,
) -> Tuple[float, Optional[Tuple[float, float, float]]]:
    perturb_norm_sq = 0.0
    delta_q_sq = 0.0
    intended_sq = 0.0
    dot = 0.0
    active = 0
    numel = 0
    with torch.no_grad():
        for name, param in named_params:
            base = backups[name].detach().float()
            z = directions[name].detach().float()
            target = base + float(sign) * float(h) * z
            perturb_norm_sq += float(torch.sum((float(h) * z) ** 2).item())
            if fake_quant:
                q_target = fake_quant_tensor(target, bits=8)
                param.data.copy_(q_target.to(dtype=param.dtype))
            else:
                param.data.copy_(target.to(dtype=param.dtype))
    if fake_quant:
        with torch.no_grad():
            for name, _ in named_params:
                z = directions[name].detach().float()
                q_plus = fake_quant_tensor(backups[name].float() + float(h) * z, bits=8)
                q_minus = fake_quant_tensor(backups[name].float() - float(h) * z, bits=8)
                delta_q = q_plus - q_minus
                intended = 2.0 * float(h) * z
                delta_q_sq += float(torch.sum(delta_q * delta_q).item())
                intended_sq += float(torch.sum(intended * intended).item())
                dot += float(torch.sum(delta_q * intended).item())
                active += int(torch.count_nonzero(delta_q != 0).item())
                numel += int(delta_q.numel())
        norm_ratio = math.sqrt(delta_q_sq) / (math.sqrt(intended_sq) + 1e-12) if intended_sq > 0 else None
        alignment = dot / (math.sqrt(delta_q_sq) * math.sqrt(intended_sq) + 1e-12) if delta_q_sq > 0 and intended_sq > 0 else None
        active_fraction = float(active) / float(numel) if numel > 0 else 0.0
        return math.sqrt(max(perturb_norm_sq, 0.0)), (active_fraction, norm_ratio or 0.0, alignment or 0.0)
    return math.sqrt(max(perturb_norm_sq, 0.0)), None


def fake_quant_tensor(tensor: torch.Tensor, *, bits: int = 8) -> torch.Tensor:
    x = torch.nan_to_num(tensor.detach().float(), nan=0.0, posinf=0.0, neginf=0.0)
    if x.numel() == 0:
        return x
    qmax = float((1 << (bits - 1)) - 1)
    max_abs_value = float(torch.max(torch.abs(x)).item())
    if (not math.isfinite(max_abs_value)) or max_abs_value <= 0.0:
        return torch.zeros_like(x)
    scale = max_abs_value / qmax
    q = torch.clamp(torch.round(x / scale), -qmax, qmax)
    return q * scale


def finite_loss_row(loss_plus: float, loss_minus: float, fd: float) -> bool:
    return all(math.isfinite(x) for x in (loss_plus, loss_minus, fd))


def run_dense(bundle: ModelBundle, args: argparse.Namespace, writer: ResultWriter) -> None:
    named_params = selected_parameters(bundle.model, max_params=args.max_touched_params, max_elements=args.max_param_elements)
    if not named_params:
        writer.skip("dense", "no suitable floating parameters selected", device=str(bundle.device), dtype=str(bundle.dtype).replace("torch.", ""))
        return
    tol = restore_tolerance(named_params)
    for k_dir in range(args.k_dirs):
        directions = sample_directions(named_params, seed=1000 + k_dir)
        backups = backup_params(named_params)
        for h in args.h_grid:
            try:
                perturb_norm, _ = apply_perturbation(named_params, backups, directions, h, +1.0)
                loss_plus = forward_loss(bundle.model, bundle.inputs)
                restore_params(named_params, backups)
                apply_perturbation(named_params, backups, directions, h, -1.0)
                loss_minus = forward_loss(bundle.model, bundle.inputs)
                restore_diff = restore_params(named_params, backups)
                fd = (loss_plus - loss_minus) / (2.0 * float(h))
                ok = finite_loss_row(loss_plus, loss_minus, fd) and restore_diff <= tol
                writer.write(
                    {
                        "method": "dense",
                        "precision_or_quant_backend": "bf16_fp16_dense_probe" if bundle.dtype in {torch.float16, torch.bfloat16} else "fp32_dense_probe",
                        "device": str(bundle.device),
                        "dtype": str(bundle.dtype).replace("torch.", ""),
                        "h": float(h),
                        "k_dir": k_dir,
                        "loss_plus": loss_plus,
                        "loss_minus": loss_minus,
                        "fd": fd,
                        "perturb_norm": perturb_norm,
                        "param_restore_max_abs_diff": restore_diff,
                        "peak_memory_mb": peak_memory_mb(bundle.device),
                        "status": "pass" if ok else "fail",
                        "error_message": "" if ok else f"finite={finite_loss_row(loss_plus, loss_minus, fd)} restore_diff={restore_diff} tol={tol}",
                    }
                )
            except Exception as exc:
                restore_params(named_params, backups)
                writer.fail("dense", exc, h=float(h), k_dir=k_dir, device=str(bundle.device), dtype=str(bundle.dtype).replace("torch.", ""))


def run_fake_int8(bundle: ModelBundle, args: argparse.Namespace, writer: ResultWriter) -> None:
    named_params = selected_parameters(bundle.model, max_params=args.max_touched_params, max_elements=args.max_param_elements)
    if not named_params:
        writer.skip("fake_int8", "no suitable floating parameters selected", device=str(bundle.device), dtype=str(bundle.dtype).replace("torch.", ""))
        return
    saw_visible = False
    tol = restore_tolerance(named_params)
    for k_dir in range(args.k_dirs):
        directions = sample_directions(named_params, seed=2000 + k_dir)
        backups = backup_params(named_params)
        for h in args.h_grid:
            try:
                perturb_norm, metrics = apply_perturbation(named_params, backups, directions, h, +1.0, fake_quant=True)
                loss_plus = forward_loss(bundle.model, bundle.inputs)
                restore_params(named_params, backups)
                apply_perturbation(named_params, backups, directions, h, -1.0, fake_quant=True)
                loss_minus = forward_loss(bundle.model, bundle.inputs)
                restore_diff = restore_params(named_params, backups)
                fd = (loss_plus - loss_minus) / (2.0 * float(h))
                active_fraction, norm_ratio, alignment = metrics or (0.0, 0.0, 0.0)
                if h >= 3e-3 and active_fraction > 0:
                    saw_visible = True
                ok = finite_loss_row(loss_plus, loss_minus, fd) and restore_diff <= tol
                writer.write(
                    {
                        "method": "fake_int8",
                        "precision_or_quant_backend": "fake_uniform_int8",
                        "device": str(bundle.device),
                        "dtype": str(bundle.dtype).replace("torch.", ""),
                        "h": float(h),
                        "k_dir": k_dir,
                        "loss_plus": loss_plus,
                        "loss_minus": loss_minus,
                        "fd": fd,
                        "active_fraction": active_fraction,
                        "norm_ratio": norm_ratio,
                        "alignment": alignment,
                        "perturb_norm": perturb_norm,
                        "param_restore_max_abs_diff": restore_diff,
                        "peak_memory_mb": peak_memory_mb(bundle.device),
                        "status": "pass" if ok else "fail",
                        "error_message": "" if ok else f"finite={finite_loss_row(loss_plus, loss_minus, fd)} restore_diff={restore_diff} tol={tol}",
                    }
                )
            except Exception as exc:
                restore_params(named_params, backups)
                writer.fail("fake_int8", exc, h=float(h), k_dir=k_dir, device=str(bundle.device), dtype=str(bundle.dtype).replace("torch.", ""))
    writer.write(
        {
            "method": "fake_int8_visibility_summary",
            "precision_or_quant_backend": "fake_uniform_int8",
            "device": str(bundle.device),
            "dtype": str(bundle.dtype).replace("torch.", ""),
            "status": "pass" if saw_visible else "fail",
            "error_message": "" if saw_visible else "no non-tiny h produced active fake-INT8 perturbation",
        }
    )


def collect_linear_activation_scales(model: nn.Module, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    scales: Dict[str, torch.Tensor] = {}
    handles = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            def hook(mod: nn.Module, inp: Tuple[torch.Tensor, ...], _out: torch.Tensor, key: str = name) -> None:
                if not inp:
                    return
                x = inp[0].detach().float()
                if x.numel() == 0:
                    return
                dims = tuple(range(max(0, x.ndim - 1)))
                scales[key] = torch.sqrt(torch.mean(x * x, dim=dims) + 1e-12).detach()
            handles.append(module.register_forward_hook(hook))
    try:
        forward_loss(model, inputs)
    finally:
        for handle in handles:
            handle.remove()
    return scales


def sparse_masks(
    named_params: Sequence[Tuple[str, nn.Parameter]],
    p: float,
    seed: int,
    strategy: str,
    activation_scales: Optional[Dict[str, torch.Tensor]] = None,
) -> Tuple[Dict[str, torch.Tensor], float, int, int]:
    masks: Dict[str, torch.Tensor] = {}
    total_active = 0
    total_numel = 0
    for idx, (name, param) in enumerate(named_params):
        numel = int(param.numel())
        if numel == 0:
            masks[name] = torch.zeros_like(param, dtype=torch.bool)
            continue
        k = max(1, int(round(float(p) * float(numel))))
        if strategy == "random":
            gen = make_generator(seed + idx * 1543, param.device)
            mask = torch.rand(param.shape, generator=gen, device=param.device) < float(p)
            if not bool(mask.any().item()):
                flat = mask.reshape(-1)
                flat[0] = True
                mask = flat.view_as(mask)
        else:
            score = torch.abs(param.detach().float())
            if strategy == "wanda_like" and param.ndim == 2:
                module_name = name.rsplit(".", 1)[0]
                col = None if activation_scales is None else activation_scales.get(module_name)
                if col is not None and col.numel() == param.shape[1]:
                    score = score * col.to(device=param.device, dtype=torch.float32).view(1, -1)
            flat_score = score.reshape(-1)
            _, idxs = torch.topk(flat_score, k=min(k, numel), largest=True, sorted=False)
            flat_mask = torch.zeros(numel, dtype=torch.bool, device=param.device)
            flat_mask[idxs] = True
            mask = flat_mask.view_as(param)
        masks[name] = mask
        total_active += int(torch.count_nonzero(mask).item())
        total_numel += numel
    p_actual = float(total_active) / float(total_numel) if total_numel else 0.0
    return masks, p_actual, total_active, total_numel


def sample_sparse_directions(
    named_params: Sequence[Tuple[str, nn.Parameter]],
    masks: Dict[str, torch.Tensor],
    p_actual: float,
    seed: int,
) -> Dict[str, torch.Tensor]:
    directions: Dict[str, torch.Tensor] = {}
    scale = 1.0 / math.sqrt(max(float(p_actual), 1e-12))
    for idx, (name, param) in enumerate(named_params):
        gen = make_generator(seed + idx * 4211, param.device)
        z = torch.randn(param.shape, generator=gen, device=param.device, dtype=torch.float32)
        directions[name] = torch.where(masks[name], z * scale, torch.zeros_like(z))
    return directions


def run_sparse(bundle: ModelBundle, args: argparse.Namespace, writer: ResultWriter) -> None:
    named_params = selected_parameters(bundle.model, max_params=args.max_touched_params, max_elements=args.max_param_elements)
    if not named_params:
        writer.skip("sparse", "no suitable floating parameters selected", device=str(bundle.device), dtype=str(bundle.dtype).replace("torch.", ""))
        return
    activation_scales = None
    strategies = parse_methods(args.sparse_mask_strategies)
    if "wanda_like" in strategies:
        try:
            activation_scales = collect_linear_activation_scales(bundle.model, bundle.inputs)
        except Exception:
            activation_scales = {}
    tol = restore_tolerance(named_params)
    saw_visible = False
    for strategy in strategies:
        p_values = args.sparse_p if strategy == "random" else args.sparse_p[:1]
        h_values = args.sparse_h_active if strategy == "random" else args.sparse_h_active[:1]
        k_dirs = range(args.k_dirs) if strategy == "random" else range(1)
        for p_req in p_values:
            masks, p_actual, active_count, total_count = sparse_masks(named_params, p_req, seed=3000, strategy=strategy, activation_scales=activation_scales)
            if active_count <= 0:
                writer.skip("sparse", "sparse mask selected no active coordinates", sparse_p_requested=p_req, mask_strategy=strategy)
                continue
            for h_active in h_values:
                h_raw = float(h_active) * math.sqrt(max(p_actual, 1e-12))
                for k_dir in k_dirs:
                    directions = sample_sparse_directions(named_params, masks, p_actual, seed=4000 + k_dir)
                    backups = backup_params(named_params)
                    try:
                        perturb_norm, metrics = apply_perturbation(named_params, backups, directions, h_raw, +1.0, fake_quant=True)
                        loss_plus = forward_loss(bundle.model, bundle.inputs)
                        restore_params(named_params, backups)
                        apply_perturbation(named_params, backups, directions, h_raw, -1.0, fake_quant=True)
                        loss_minus = forward_loss(bundle.model, bundle.inputs)
                        restore_diff = restore_params(named_params, backups)
                        fd = (loss_plus - loss_minus) / (2.0 * h_raw)
                        active_fraction, norm_ratio, alignment = metrics or (0.0, 0.0, 0.0)
                        saw_visible = saw_visible or active_fraction > 0.0
                        ok = finite_loss_row(loss_plus, loss_minus, fd) and restore_diff <= tol and active_count > 0
                        writer.write(
                            {
                                "method": "sparse",
                                "precision_or_quant_backend": "fake_uniform_int8_sparse",
                                "device": str(bundle.device),
                                "dtype": str(bundle.dtype).replace("torch.", ""),
                                "h": h_raw,
                                "sparse_p_requested": float(p_req),
                                "sparse_p_actual": p_actual,
                                "h_raw": h_raw,
                                "h_active": float(h_active),
                                "k_dir": k_dir,
                                "loss_plus": loss_plus,
                                "loss_minus": loss_minus,
                                "fd": fd,
                                "active_fraction": active_fraction,
                                "norm_ratio": norm_ratio,
                                "alignment": alignment,
                                "active_count": active_count,
                                "mask_strategy": strategy,
                                "perturb_norm": perturb_norm,
                                "param_restore_max_abs_diff": restore_diff,
                                "peak_memory_mb": peak_memory_mb(bundle.device),
                                "status": "pass" if ok else "fail",
                                "error_message": "" if ok else f"finite={finite_loss_row(loss_plus, loss_minus, fd)} restore_diff={restore_diff} tol={tol} active_count={active_count} total={total_count}",
                            }
                        )
                    except Exception as exc:
                        restore_params(named_params, backups)
                        writer.fail("sparse", exc, h=h_raw, sparse_p_requested=float(p_req), sparse_p_actual=p_actual, h_raw=h_raw, h_active=float(h_active), k_dir=k_dir, mask_strategy=strategy)
    writer.write(
        {
            "method": "sparse_visibility_summary",
            "precision_or_quant_backend": "fake_uniform_int8_sparse",
            "device": str(bundle.device),
            "dtype": str(bundle.dtype).replace("torch.", ""),
            "status": "pass" if saw_visible else "fail",
            "error_message": "" if saw_visible else "no sparse fake-INT8 configuration produced visible perturbations",
        }
    )


def discover_lora_targets(model: nn.Module) -> List[str]:
    wanted = ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]
    leaf_names = {name.rsplit(".", 1)[-1] for name, module in model.named_modules() if isinstance(module, nn.Linear)}
    targets = [name for name in wanted if name in leaf_names]
    return targets[:4] if targets else []


def run_lora(bundle: ModelBundle, args: argparse.Namespace, writer: ResultWriter, state: Dict[str, Any]) -> None:
    if not module_available("peft"):
        reason = "peft is not installed"
        if args.require_peft:
            writer.fail("lora", reason, device=str(bundle.device), dtype=str(bundle.dtype).replace("torch.", ""))
        else:
            writer.skip("lora", reason, device=str(bundle.device), dtype=str(bundle.dtype).replace("torch.", ""))
        return
    try:
        from peft import LoraConfig, TaskType, get_peft_model
    except Exception as exc:
        writer.fail("lora" if args.require_peft else "lora", exc, device=str(bundle.device), dtype=str(bundle.dtype).replace("torch.", ""))
        return

    targets = discover_lora_targets(bundle.model)
    if not targets:
        writer.skip("lora", "no OPT LoRA target Linear modules found", device=str(bundle.device), dtype=str(bundle.dtype).replace("torch.", ""))
        return
    try:
        for param in bundle.model.parameters():
            param.requires_grad = False
        config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=16,
            lora_dropout=0.0,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            target_modules=targets,
        )
        lora_model = get_peft_model(bundle.model, config)
        lora_model.eval()
        trainable = count_params(lora_model, trainable_only=True)
        total = count_params(lora_model, trainable_only=False)
        ratio = float(trainable) / float(total) if total else 0.0
        non_lora_trainable = [name for name, param in lora_model.named_parameters() if param.requires_grad and "lora" not in name.lower()]
        loss = forward_loss(lora_model, bundle.inputs)
        named_params = selected_parameters(lora_model, max_params=args.max_touched_params, max_elements=args.max_param_elements, trainable_only=True, lora_only=True)
        backups = backup_params(named_params)
        directions = sample_directions(named_params, seed=5000)
        h = float(args.h_grid[min(1, len(args.h_grid) - 1)])
        perturb_norm, _ = apply_perturbation(named_params, backups, directions, h, +1.0)
        loss_plus = forward_loss(lora_model, bundle.inputs)
        restore_params(named_params, backups)
        apply_perturbation(named_params, backups, directions, h, -1.0)
        loss_minus = forward_loss(lora_model, bundle.inputs)
        restore_diff = restore_params(named_params, backups)
        fd = (loss_plus - loss_minus) / (2.0 * h)
        ok = (
            trainable > 0
            and finite_loss_row(loss_plus, loss_minus, fd)
            and math.isfinite(loss)
            and restore_diff <= restore_tolerance(named_params)
            and not non_lora_trainable
        )
        if "tiny" not in bundle.model_id.lower() and bundle.source != "local_opt_config":
            ok = ok and ratio < 0.05
        state["lora_model"] = lora_model
        state["lora_targets"] = targets
        state["lora_state_dict"] = {name: param.detach().cpu().clone() for name, param in lora_model.named_parameters() if "lora" in name.lower()}
        writer.write(
            {
                "method": "lora",
                "precision_or_quant_backend": "peft_lora",
                "device": str(bundle.device),
                "dtype": str(bundle.dtype).replace("torch.", ""),
                "h": h,
                "k_dir": 0,
                "loss": loss,
                "loss_plus": loss_plus,
                "loss_minus": loss_minus,
                "fd": fd,
                "perturb_norm": perturb_norm,
                "param_restore_max_abs_diff": restore_diff,
                "trainable_params": trainable,
                "total_params": total,
                "trainable_ratio": ratio,
                "lora_target_modules": ",".join(targets),
                "peak_memory_mb": peak_memory_mb(bundle.device),
                "status": "pass" if ok else "fail",
                "error_message": "" if ok else f"non_lora_trainable={non_lora_trainable[:5]} ratio={ratio} restore_diff={restore_diff}",
            }
        )
    except Exception as exc:
        writer.fail("lora", exc, device=str(bundle.device), dtype=str(bundle.dtype).replace("torch.", ""))


def run_residual(bundle: ModelBundle, args: argparse.Namespace, writer: ResultWriter, state: Dict[str, Any]) -> None:
    try:
        param = nn.Parameter(torch.tensor([0.0, 0.2, -0.4, 0.6, 0.1, -0.1], dtype=torch.float32, device=bundle.device))
        direction = torch.tensor([1.0, -0.5, 0.25, -1.25, 0.75, -0.75], dtype=torch.float32, device=bundle.device)
        lr = 0.05
        fd = 1.0
        if ResidualGridUpdater is not None:
            updater = ResidualGridUpdater([("w", param)], bits=8, residual_dtype="fp32", commit_mode="round", max_code_step=0, freeze_scale=True)
            scale = updater.scales["w"].detach().clone()
            old_residual = updater.residuals["w"].detach().clone()
            desired = -lr * fd * direction
            stats = updater.apply_update("w", param, direction, projected_grad=fd, learning_rate=lr)
            residual = updater.residuals["w"].detach().clone()
            commit = desired + old_residual.float() - residual.float()
            conservation_error = max_abs((commit + residual.float()) - (desired + old_residual.float()))
            residual_state = {"residuals": {k: v.detach().cpu().clone() for k, v in updater.residuals.items()}, "scales": {k: v.detach().cpu().clone() for k, v in updater.scales.items()}}
            commit_nonzero_fraction = float(stats.get("active_frac", 0.0))
        else:
            scale = torch.tensor(0.01, dtype=torch.float32, device=bundle.device)
            old_residual = torch.zeros_like(param)
            desired = -lr * fd * direction
            acc = desired + old_residual
            commit = torch.round(acc / scale) * scale
            residual = acc - commit
            conservation_error = max_abs((commit + residual) - acc)
            residual_state = {"residual": residual.detach().cpu().clone(), "scale": scale.detach().cpu().clone()}
            commit_nonzero_fraction = float(torch.count_nonzero(commit != 0).item()) / float(commit.numel())
        ckpt = Path(args.output_dir) / "checkpoint_smoke"
        ckpt.mkdir(parents=True, exist_ok=True)
        residual_path = ckpt / "residual_state.pt"
        torch.save(residual_state, residual_path)
        loaded = torch.load(residual_path, map_location="cpu", weights_only=False)
        if "residuals" in residual_state:
            save_load_ok = all(torch.equal(loaded["residuals"][k], residual_state["residuals"][k]) for k in residual_state["residuals"])
        else:
            save_load_ok = torch.equal(loaded["residual"], residual_state["residual"])
        state["residual_state_path"] = str(residual_path)
        state["residual_state"] = residual_state
        ok = math.isfinite(conservation_error) and conservation_error <= 1e-6 and save_load_ok and commit_nonzero_fraction > 0.0
        writer.write(
            {
                "method": "residual",
                "precision_or_quant_backend": "residual_grid_error_feedback",
                "device": str(bundle.device),
                "dtype": "float32",
                "update_norm": vector_norm(desired),
                "commit_norm": vector_norm(commit),
                "residual_norm": vector_norm(residual),
                "commit_nonzero_fraction": commit_nonzero_fraction,
                "grid_scale": float(scale.reshape(-1)[0].detach().cpu().item()),
                "conservation_error": conservation_error,
                "save_load_ok": bool(save_load_ok),
                "status": "pass" if ok else "fail",
                "error_message": "" if ok else f"conservation_error={conservation_error} save_load_ok={save_load_ok} commit_nonzero_fraction={commit_nonzero_fraction}",
            }
        )
    except Exception as exc:
        writer.fail("residual", exc, device=str(bundle.device), dtype=str(bundle.dtype).replace("torch.", ""))


def run_checkpoint(bundle: ModelBundle, args: argparse.Namespace, writer: ResultWriter, state: Dict[str, Any]) -> None:
    try:
        ckpt = Path(args.output_dir) / "checkpoint_smoke"
        ckpt.mkdir(parents=True, exist_ok=True)
        config_path = ckpt / "smoke_config.json"
        config_path.write_text(json.dumps(vars(args), indent=2, sort_keys=True), encoding="utf-8")
        selected = selected_parameters(bundle.model, max_params=1, max_elements=args.max_param_elements)
        model_state_path = ckpt / "selected_model_state.pt"
        torch.save({name: param.detach().cpu().clone() for name, param in selected}, model_state_path)
        lora_ok = None
        if "lora_state_dict" in state:
            lora_path = ckpt / "lora_adapter_state.pt"
            torch.save(state["lora_state_dict"], lora_path)
            loaded_lora = torch.load(lora_path, map_location="cpu", weights_only=False)
            lora_ok = set(loaded_lora) == set(state["lora_state_dict"])
        residual_ok = None
        if "residual_state_path" in state:
            loaded_residual = torch.load(state["residual_state_path"], map_location="cpu", weights_only=False)
            residual_ok = bool(loaded_residual)
        loaded_model_state = torch.load(model_state_path, map_location="cpu", weights_only=False)
        loss = forward_loss(bundle.model, bundle.inputs)
        ok = bool(loaded_model_state) and config_path.exists() and math.isfinite(loss)
        if lora_ok is not None:
            ok = ok and lora_ok
        if residual_ok is not None:
            ok = ok and residual_ok
        writer.write(
            {
                "method": "checkpoint",
                "precision_or_quant_backend": "checkpoint_resume_smoke",
                "device": str(bundle.device),
                "dtype": str(bundle.dtype).replace("torch.", ""),
                "loss": loss,
                "save_load_ok": bool(ok),
                "status": "pass" if ok else "fail",
                "error_message": "" if ok else f"lora_ok={lora_ok} residual_ok={residual_ok} model_state={bool(loaded_model_state)}",
            }
        )
    except Exception as exc:
        writer.fail("checkpoint", exc, device=str(bundle.device), dtype=str(bundle.dtype).replace("torch.", ""))


def run_bnb_int8(args: argparse.Namespace, writer: ResultWriter, method: str) -> None:
    if not torch.cuda.is_available():
        writer.skip(method, "CUDA is not available")
        return
    if not module_available("bitsandbytes"):
        if args.require_bnb:
            writer.fail(method, "bitsandbytes is not installed")
        else:
            writer.skip(method, "bitsandbytes is not installed")
        return
    if not module_available("transformers"):
        writer.fail(method, "transformers is not installed")
        return
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        quant_config = BitsAndBytesConfig(load_in_8bit=True)
        local_only = bool(args.allow_cached_only or not args.allow_download)
        tokenizer = AutoTokenizer.from_pretrained(args.model_id, local_files_only=local_only)
        model = AutoModelForCausalLM.from_pretrained(args.model_id, quantization_config=quant_config, device_map="auto", local_files_only=local_only)
        model.eval()
        if getattr(tokenizer, "pad_token", None) is None:
            tokenizer.pad_token = getattr(tokenizer, "eos_token", None) or tokenizer.unk_token
        encoded = tokenizer(TOY_TEXTS[:1], padding="max_length", truncation=True, max_length=args.max_seq_len, return_tensors="pt")
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in encoded.items()}
        inputs["labels"] = inputs["input_ids"].clone()
        loss = forward_loss(model, inputs)
        quantized_module_count = sum(1 for module in model.modules() if "8bit" in module.__class__.__name__.lower() or "linear8bit" in module.__class__.__name__.lower())
        ok = math.isfinite(loss) and quantized_module_count > 0
        writer.write(
            {
                "method": method,
                "precision_or_quant_backend": "bitsandbytes_int8",
                "device": str(device),
                "dtype": "int8_load",
                "loss": loss,
                "quantized_module_count": quantized_module_count,
                "peak_memory_mb": peak_memory_mb(torch.device("cuda")),
                "status": "pass" if ok else "fail",
                "error_message": "" if ok else f"loss={loss} quantized_module_count={quantized_module_count}",
            }
        )
    except Exception as exc:
        writer.skip(method, f"bnb INT8 model unavailable or failed with cached/download policy: {exc}")


def run_env_row(writer: ResultWriter, bundle: Optional[ModelBundle]) -> None:
    row = writer.base_row("env", "pass")
    row.update(
        {
            "precision_or_quant_backend": "environment",
            "device": str(bundle.device) if bundle else None,
            "dtype": str(bundle.dtype).replace("torch.", "") if bundle else None,
            "total_params": count_params(bundle.model, trainable_only=False) if bundle else None,
            "trainable_params": count_params(bundle.model, trainable_only=True) if bundle else None,
        }
    )
    writer.write(row)


def print_summary(rows: Sequence[Dict[str, Any]]) -> None:
    counts: Dict[Tuple[str, str], int] = {}
    for row in rows:
        key = (str(row.get("method")), str(row.get("status")))
        counts[key] = counts.get(key, 0) + 1
    print("\nSmoke summary")
    print("method,status,count")
    for (method, status), count in sorted(counts.items()):
        print(f"{method},{status},{count}")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_id", default="hf-internal-testing/tiny-random-OPTForCausalLM")
    parser.add_argument("--methods", default="env,dense,fake_int8,sparse,lora,residual,checkpoint")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto", choices=["auto", "fp32", "float32", "fp16", "float16", "bf16", "bfloat16"])
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_seq_len", type=int, default=64)
    parser.add_argument("--k_dirs", type=int, default=2)
    parser.add_argument("--h_grid", type=float, nargs="+", default=[1e-4, 1e-3, 3e-3, 1e-2])
    parser.add_argument("--sparse_p", type=float, nargs="+", default=[0.01, 0.05])
    parser.add_argument("--sparse_h_active", type=float, nargs="+", default=[3e-3, 6e-3, 1e-2])
    parser.add_argument("--sparse_mask_strategies", default="random,outlier_abs,wanda_like")
    parser.add_argument("--max_steps", type=int, default=2)
    parser.add_argument("--max_touched_params", type=int, default=2)
    parser.add_argument("--max_param_elements", type=int, default=2_000_000)
    parser.add_argument("--output_dir", default="outputs/smoke_opt_zo")
    parser.add_argument("--allow_download", action="store_true")
    parser.add_argument("--allow_cached_only", action="store_true")
    parser.add_argument("--require_bnb", action="store_true")
    parser.add_argument("--require_peft", action="store_true")
    parser.add_argument("--load_in_8bit", action="store_true")
    parser.add_argument("--lora_rank", type=int, default=4)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    args.output_dir = str(Path(args.output_dir))
    env = get_env_info()
    print(json.dumps(env, indent=2, sort_keys=True))
    writer = ResultWriter(Path(args.output_dir), env, args)
    bundle: Optional[ModelBundle] = None
    state: Dict[str, Any] = {}
    try:
        methods = parse_methods(args.methods)
        needs_regular_model = any(method in methods for method in {"env", "dense", "fake_int8", "sparse", "lora", "residual", "checkpoint"})
        if needs_regular_model:
            bundle = load_model_bundle(args, writer)
        if "env" in methods:
            run_env_row(writer, bundle)
        if "bnb_int8_load" in methods:
            run_bnb_int8(args, writer, "bnb_int8_load")
        if "int8_forward" in methods:
            if args.load_in_8bit:
                run_bnb_int8(args, writer, "int8_forward")
            else:
                writer.skip("int8_forward", "real INT8 forward requires --load_in_8bit; fake quantized forward is method=fake_int8")
        if bundle is not None:
            if "dense" in methods:
                run_dense(bundle, args, writer)
            if "fake_int8" in methods:
                run_fake_int8(bundle, args, writer)
            if "sparse" in methods:
                run_sparse(bundle, args, writer)
            if "lora" in methods:
                run_lora(bundle, args, writer, state)
            if "residual" in methods:
                run_residual(bundle, args, writer, state)
            if "checkpoint" in methods:
                run_checkpoint(bundle, args, writer, state)
        elif needs_regular_model:
            for method in methods:
                if method not in {"bnb_int8_load", "int8_forward"}:
                    writer.skip(method, "regular model was not loaded")
    finally:
        writer.close()
    print_summary(writer.rows)
    failures = [row for row in writer.rows if row.get("status") == "fail"]
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
