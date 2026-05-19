#!/usr/bin/env python3
"""Offline L-curvature analysis for RoBERTa-large/SST-5 FP32 and FP16 runs.

This script is intentionally read-only with respect to training artifacts. It
loads existing checkpoint directories, evaluates a fixed probe batch with fixed
raw Gaussian directions over an h2 grid, and writes L-estimation diagnostics.
It does not launch or resume training.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import datetime as dt
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
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
MEDIUM_ROOT = REPO_ROOT / "medium_models"
FIXED_H2_GRID = [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 1.5e-3, 2e-3, 3e-3, 4e-3, 5e-3, 1e-2]
FIXED_H2_DIAGNOSTIC = [3e-4, 1e-3, 2e-3, 3e-3, 4e-3]
TARGET_PRECISIONS = {"fp32", "fp16"}
EPS_NUM = 1e-12
TAU_SNR = 2.0


L_CANDIDATE_FIELDS = [
    "precision",
    "checkpoint_name",
    "checkpoint_path",
    "target_run_h",
    "L_mode",
    "h2",
    "m_L",
    "d_trainable",
    "direction_normalization",
    "forward_precision",
    "loss_dtype",
    "tf32_enabled",
    "autocast_enabled",
    "autocast_dtype",
    "delta_mode",
    "base_loss",
    "median_abs_K",
    "mean_K",
    "std_K",
    "MAD_K",
    "SNR2",
    "finite_rate",
    "zero_K_frac",
    "lambda_q50",
    "lambda_q90",
    "lambda_q95",
    "lambda_mean",
    "lambda_std",
    "stability_q50_2x",
    "stability_q90_2x",
    "stability_q95_2x",
    "stability_q90_next",
    "stability_q90_prev",
    "log_slope_q90_next",
    "log_slope_q90_prev",
    "low_h2_noise_suspected",
    "large_h2_nonlocal_suspected",
    "alignment_eff_1",
    "norm_ratio_eff_1",
    "zero_coord_frac_eff_1",
    "rms_snap_error_1",
    "alignment_eff_2",
    "norm_ratio_eff_2",
    "zero_coord_frac_eff_2",
    "rms_snap_error_2",
    "warnings",
]


L_SELECTED_FIELDS = [
    "precision",
    "checkpoint_name",
    "checkpoint_path",
    "L_mode",
    "selector_name",
    "selected_h2",
    "selected_L_q",
    "selected_L_value",
    "selected_L_q50",
    "selected_L_q90",
    "selected_L_q95",
    "SNR2",
    "stability_q90_2x",
    "stability_q90_next",
    "stability_q90_prev",
    "low_h2_noise_suspected",
    "large_h2_nonlocal_suspected",
    "selection_status",
    "fallback_flags",
    "notes",
]


@dataclass
class RunInfo:
    precision: str
    h: float
    run_name: str
    run_dir: Path
    run_config_path: Path
    raw_config: Dict[str, Any]
    summary_config: Dict[str, Any]
    metadata: Dict[str, Any]


@dataclass
class CheckpointInfo:
    precision: str
    checkpoint_name: str
    checkpoint_dir: Path
    target_run_h: float
    run: RunInfo


@dataclass
class EvalContext:
    model: Any
    batch: Dict[str, Any]
    named_params: List[Tuple[str, Any]]
    backup_params: List[Any]
    d_trainable: int
    device: Any
    forward_precision: str
    delta_mode: str
    direction_dtype_name: str
    tf32_enabled: str
    autocast_enabled: bool
    autocast_dtype: str
    loss_dtype: str = ""


def now_stamp() -> str:
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def read_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def write_csv(path: Path, rows: Iterable[Dict[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fields), extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})


def fail_analysis(out_dir: Path, diagnostics: Dict[str, Any], message: str, code: int) -> int:
    diagnostics.setdefault("warnings", []).append(message)
    diagnostics["failure"] = message
    diagnostics["end_time"] = dt.datetime.now().isoformat()
    write_json(out_dir / "L_diagnostics.json", diagnostics)
    (out_dir / "failure_report.txt").write_text(f"FAILED: {message}\n", encoding="utf-8")
    print(f"FAILED: {message}", file=sys.stderr)
    return int(code)


def safe_float(value: Any) -> Optional[float]:
    try:
        if value in (None, ""):
            return None
        out = float(value)
    except Exception:
        return None
    return out if math.isfinite(out) else None


def h_key(value: float) -> str:
    return f"{float(value):.12g}"


def h_close(a: float, b: float, rtol: float = 1e-9, atol: float = 1e-12) -> bool:
    return abs(float(a) - float(b)) <= max(atol, rtol * max(abs(float(a)), abs(float(b)), 1.0))


def find_close(value: float, grid: Iterable[float]) -> Optional[float]:
    for item in grid:
        if h_close(value, float(item)):
            return float(item)
    return None


def sorted_unique(values: Iterable[float]) -> List[float]:
    out: List[float] = []
    for value in sorted(float(v) for v in values if math.isfinite(float(v)) and float(v) > 0.0):
        if find_close(value, out) is None:
            out.append(value)
    return out


def quantile(values: Sequence[float], q: float) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(np.quantile(arr, q))


def finite_mean(values: Sequence[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr)) if arr.size else float("nan")


def finite_std(values: Sequence[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(np.std(arr)) if arr.size else float("nan")


def boolish(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT), text=True).strip()
    except Exception:
        return ""


def discover_runs(search_roots: Sequence[Path], diagnostics: Dict[str, Any]) -> List[RunInfo]:
    config_paths: List[Path] = []
    for root in search_roots:
        if root.exists():
            config_paths.extend(root.rglob("run_config.json"))
    diagnostics["run_config_paths_seen"] = len(config_paths)

    runs: List[RunInfo] = []
    for cfg_path in sorted(config_paths):
        raw = read_json(cfg_path)
        precision = str(raw.get("precision_mode", "")).strip().lower()
        if precision not in TARGET_PRECISIONS:
            continue
        model = str(raw.get("model", raw.get("model_name_or_path", ""))).strip().lower()
        dataset = str(raw.get("dataset", raw.get("task_name", ""))).strip().lower()
        if model != "roberta-large":
            continue
        if dataset not in {"sst-5", "sst5"}:
            continue
        if int(raw.get("seed", -1)) != 16 or int(raw.get("data_seed", -1)) != 16:
            continue
        if str(raw.get("dataset_mode", "")).strip().lower() != "full":
            continue
        if int(raw.get("batch_size", raw.get("per_device_train_batch_size", -1))) != 64:
            continue
        if not boolish(raw.get("dataloader_shuffle", False)):
            continue
        if str(raw.get("direction_type", "")).strip().lower() != "dense":
            continue
        h_val = safe_float(raw.get("h", raw.get("zero_order_eps")))
        if h_val is None:
            continue
        run_dir = cfg_path.parent
        raw_summary = read_json(run_dir / "run_summary_raw.json")
        summary_config = raw_summary.get("config", {}) if isinstance(raw_summary.get("config", {}), dict) else {}
        metadata = read_json(run_dir / "run_metadata.json")
        runs.append(
            RunInfo(
                precision=precision,
                h=float(h_val),
                run_name=str(raw.get("run_name") or run_dir.parent.name),
                run_dir=run_dir,
                run_config_path=cfg_path,
                raw_config=raw,
                summary_config=summary_config,
                metadata=metadata,
            )
        )

    runs = dedupe_runs(runs)
    diagnostics["runs_discovered"] = [
        {
            "precision": r.precision,
            "h": r.h,
            "run_name": r.run_name,
            "run_dir": str(r.run_dir),
            "has_run_summary_raw": bool(r.summary_config),
            "storage_dtype": r.metadata.get("storage_dtype"),
            "compute_dtype": r.metadata.get("compute_dtype"),
            "zo_quantization": r.metadata.get("zo_quantization"),
        }
        for r in runs
    ]
    return runs


def dedupe_runs(runs: Sequence[RunInfo]) -> List[RunInfo]:
    by_key: Dict[Tuple[str, str], RunInfo] = {}
    for run in runs:
        key = (run.precision, h_key(run.h))
        old = by_key.get(key)
        if old is None:
            by_key[key] = run
            continue
        p = str(run.run_dir)
        q = str(old.run_dir)
        score = (
            1 if "/experiments/main_latest/" in p else 0,
            0 if "/smoke/" in p else 1,
            1 if "h_sweep_11h" in p else 0,
            len(p),
        )
        old_score = (
            1 if "/experiments/main_latest/" in q else 0,
            0 if "/smoke/" in q else 1,
            1 if "h_sweep_11h" in q else 0,
            len(q),
        )
        if score > old_score:
            by_key[key] = run
    return [by_key[key] for key in sorted(by_key, key=lambda x: (x[0], float(x[1])))]


def choose_reference_runs(runs: Sequence[RunInfo], diagnostics: Dict[str, Any]) -> Dict[str, RunInfo]:
    out: Dict[str, RunInfo] = {}
    for precision in sorted(TARGET_PRECISIONS):
        candidates = [r for r in runs if r.precision == precision]
        exact = [r for r in candidates if h_close(r.h, 1e-3)]
        if exact:
            out[precision] = exact[0]
        elif candidates:
            out[precision] = min(candidates, key=lambda r: abs(math.log(max(r.h, 1e-30)) - math.log(1e-3)))
            diagnostics.setdefault("warnings", []).append(
                f"{precision}: h=1e-3 reference run missing; using nearest h={out[precision].h:g}"
            )
        else:
            diagnostics.setdefault("warnings", []).append(f"{precision}: no target runs discovered")
    return out


def checkpoint_has_model(path: Path) -> bool:
    return (path / "model.safetensors").exists() or (path / "pytorch_model.bin").exists()


def select_checkpoints(reference_runs: Dict[str, RunInfo], diagnostics: Dict[str, Any]) -> List[CheckpointInfo]:
    selected: List[CheckpointInfo] = []
    skipped: List[Dict[str, Any]] = []
    for precision, run in sorted(reference_runs.items()):
        ckpt_root = run.run_dir / "checkpoints"
        initial_candidates = [ckpt_root / "initial", ckpt_root / "step_0"]
        initial = next((p for p in initial_candidates if checkpoint_has_model(p)), None)
        if initial is not None:
            selected.append(CheckpointInfo(precision, "initial", initial, run.h, run))
        else:
            skipped.append(
                {
                    "precision": precision,
                    "checkpoint": "initial",
                    "reason": "no initial/step_0 checkpoint directory with model weights",
                }
            )

        step_1000 = ckpt_root / "step_1000"
        if checkpoint_has_model(step_1000):
            selected.append(CheckpointInfo(precision, "step_1000", step_1000, run.h, run))
        else:
            step_dirs = sorted(
                [p for p in ckpt_root.glob("step_*") if checkpoint_has_model(p)],
                key=lambda p: int(p.name.split("_", 1)[1]) if p.name.split("_", 1)[1].isdigit() else 10**18,
            )
            if step_dirs:
                selected.append(CheckpointInfo(precision, f"{step_dirs[0].name}_earliest", step_dirs[0], run.h, run))
                skipped.append(
                    {
                        "precision": precision,
                        "checkpoint": "step_1000",
                        "reason": f"missing; used earliest available {step_dirs[0].name}",
                    }
                )
            else:
                skipped.append(
                    {
                        "precision": precision,
                        "checkpoint": "step_1000",
                        "reason": "missing and no step_* checkpoint with model weights",
                    }
                )

        final = ckpt_root / "final"
        step_20000 = ckpt_root / "step_20000"
        if checkpoint_has_model(final):
            selected.append(CheckpointInfo(precision, "final", final, run.h, run))
        elif checkpoint_has_model(step_20000):
            selected.append(CheckpointInfo(precision, "step_20000", step_20000, run.h, run))
        else:
            skipped.append(
                {
                    "precision": precision,
                    "checkpoint": "final_or_step_20000",
                    "reason": "missing",
                }
            )

    diagnostics["selected_checkpoints"] = [
        {
            "precision": c.precision,
            "checkpoint_name": c.checkpoint_name,
            "checkpoint_dir": str(c.checkpoint_dir),
            "target_run_h": c.target_run_h,
            "reference_run": c.run.run_name,
            "reference_run_dir": str(c.run.run_dir),
        }
        for c in selected
    ]
    diagnostics["skipped_checkpoints"] = skipped
    return selected


def import_medium_modules():
    if str(MEDIUM_ROOT) not in sys.path:
        sys.path.insert(0, str(MEDIUM_ROOT))
    from transformers import AutoConfig, AutoTokenizer
    from src.dataset import FewShotDataset
    from src.models import MODEL_TYPES
    from src.processors import num_labels_mapping, output_modes_mapping

    return {
        "AutoConfig": AutoConfig,
        "AutoTokenizer": AutoTokenizer,
        "FewShotDataset": FewShotDataset,
        "MODEL_TYPES": MODEL_TYPES,
        "num_labels_mapping": num_labels_mapping,
        "output_modes_mapping": output_modes_mapping,
    }


def collate_with_project_padding(tokenizer: Any, features: Sequence[Any]) -> Dict[str, Any]:
    """Mirror medium_models.run.MyDataCollatorWithPadding without importing run.py."""
    import torch

    standard_features: List[Dict[str, Any]] = []
    mask_pos: List[Any] = []
    has_sfc = bool(features and getattr(features[0], "sfc_input_ids", None) is not None)
    sfc_features: List[Dict[str, Any]] = []
    sfc_mask_pos: List[Any] = []

    for item in features:
        standard_item: Dict[str, Any] = {}
        for field in ["input_ids", "label", "attention_mask", "token_type_ids"]:
            value = getattr(item, field, None)
            if value is not None:
                standard_item[field] = value
        standard_features.append(standard_item)
        mask_pos.append(getattr(item, "mask_pos", None))

        if has_sfc:
            sfc_item: Dict[str, Any] = {}
            if getattr(item, "sfc_input_ids", None) is not None:
                sfc_item["input_ids"] = getattr(item, "sfc_input_ids")
            if getattr(item, "sfc_attention_mask", None) is not None:
                sfc_item["attention_mask"] = getattr(item, "sfc_attention_mask")
            sfc_features.append(sfc_item)
            sfc_mask_pos.append(getattr(item, "sfc_mask_pos", None))

    batch = tokenizer.pad(standard_features, padding=True, return_tensors="pt")
    if any(mask_pos):
        batch["mask_pos"] = torch.tensor(mask_pos)
    if "label" in batch:
        batch["labels"] = batch["label"]
        del batch["label"]
    if "label_ids" in batch:
        batch["labels"] = batch["label_ids"]
        del batch["label_ids"]

    if has_sfc:
        sfc_batch = tokenizer.pad(sfc_features, padding=True, return_tensors="pt")
        batch["sfc_input_ids"] = sfc_batch["input_ids"]
        batch["sfc_attention_mask"] = sfc_batch["attention_mask"]
        if any(sfc_mask_pos):
            batch["sfc_mask_pos"] = torch.tensor(sfc_mask_pos)
    return batch


def namespace_from_config(cfg: Dict[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(**dict(cfg))


def resolve_mode_configs(run: RunInfo) -> Tuple[SimpleNamespace, SimpleNamespace, SimpleNamespace]:
    cfg = run.summary_config
    if not cfg:
        raise RuntimeError(f"{run.run_dir}: run_summary_raw.json config missing; cannot rebuild model/data context")
    model_args = namespace_from_config(cfg.get("model_args", {}))
    data_args = namespace_from_config(cfg.get("data_args", {}))
    training_args = namespace_from_config(cfg.get("training_args", {}))
    return model_args, data_args, training_args


def get_device(device_arg: str, diagnostics: Dict[str, Any]):
    import torch

    if device_arg == "auto":
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_arg)
    diagnostics.setdefault("compute_environment", {})["selected_device"] = str(device)
    return device


def current_tf32_state() -> Dict[str, bool]:
    import torch

    return {
        "matmul": bool(getattr(torch.backends.cuda.matmul, "allow_tf32", False)),
        "cudnn": bool(getattr(torch.backends.cudnn, "allow_tf32", False)),
    }


def set_tf32_state(matmul: bool, cudnn: bool) -> Dict[str, bool]:
    import torch

    torch.backends.cuda.matmul.allow_tf32 = bool(matmul)
    torch.backends.cudnn.allow_tf32 = bool(cudnn)
    return current_tf32_state()


def set_all_seeds(seed: int) -> None:
    import torch

    np.random.seed(int(seed) % (2**32))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def direction_seed_list(base_seed: int, m_l: int, batch_index: int = 0) -> List[int]:
    mixed_seed = (int(base_seed) * 1000003 + int(batch_index) * 9176 + 97) % 2147483647
    rng = np.random.RandomState(int(mixed_seed))
    return [int(x) for x in rng.randint(0, 2147483647, size=max(1, int(m_l)))]


def load_eval_context(
    checkpoint: CheckpointInfo,
    *,
    mode_name: str,
    device: Any,
    diagnostics: Dict[str, Any],
) -> EvalContext:
    import torch

    mods = import_medium_modules()
    AutoConfig = mods["AutoConfig"]
    AutoTokenizer = mods["AutoTokenizer"]
    FewShotDataset = mods["FewShotDataset"]
    MODEL_TYPES = mods["MODEL_TYPES"]
    num_labels_mapping = mods["num_labels_mapping"]

    model_args, data_args, training_args = resolve_mode_configs(checkpoint.run)
    data_args.task_name = str(getattr(data_args, "task_name", "sst-5")).lower()
    data_args.prompt = bool(getattr(data_args, "prompt", True))
    training_args.dataloader_shuffle = True

    num_labels = int(num_labels_mapping[data_args.task_name])
    config = AutoConfig.from_pretrained(
        str(checkpoint.checkpoint_dir),
        num_labels=num_labels,
        finetuning_task=data_args.task_name,
        local_files_only=True,
    )
    model_fn = MODEL_TYPES[config.model_type]
    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint.checkpoint_dir), local_files_only=True)
    tokenizer.model_type = config.model_type

    model = model_fn.from_pretrained(str(checkpoint.checkpoint_dir), config=config, local_files_only=True)
    label_source = FewShotDataset(data_args, tokenizer=tokenizer, mode="train", use_demo=("demo" in str(getattr(model_args, "few_shot_type", ""))))
    if getattr(label_source, "label_word_list", None) is not None:
        model.label_word_list = torch.tensor(label_source.label_word_list).long().to(device)
    model.model_args = model_args
    model.data_args = data_args
    model.tokenizer = tokenizer

    forward_precision = "fp32"
    delta_mode = "identity_fp32"
    direction_dtype_name = "float32"
    tf32_defaults = diagnostics.get("tf32_default_state") or current_tf32_state()
    if mode_name == "L_clean32":
        tf32_state = set_tf32_state(False, False)
        model.float()
        forward_precision = "fp32"
        delta_mode = "identity_fp32"
        direction_dtype_name = "float32"
    elif mode_name in {"L_oracle_precision", "L_oracle_oldSNR"}:
        tf32_state = set_tf32_state(bool(tf32_defaults.get("matmul", False)), bool(tf32_defaults.get("cudnn", False)))
        if checkpoint.precision == "fp16":
            model.half()
            forward_precision = "fp16"
            delta_mode = "fp16_delta_snap"
            direction_dtype_name = "float16"
        else:
            model.float()
            forward_precision = "fp32"
            delta_mode = "identity_fp32"
            direction_dtype_name = "float32"
    else:
        raise ValueError(f"unsupported mode_name={mode_name}")

    model.to(device)
    model.eval()

    generator = torch.Generator()
    generator.manual_seed(int(getattr(training_args, "data_seed", 16)))
    batch_size = int(getattr(training_args, "per_device_train_batch_size", 64))
    batch_indices = torch.randperm(len(label_source), generator=generator)[:batch_size].tolist()
    batch = collate_with_project_padding(tokenizer, [label_source[int(i)] for i in batch_indices])
    batch = {k: v.to(device) if hasattr(v, "to") else v for k, v in batch.items()}

    named_params = [(name, param) for name, param in model.named_parameters() if param.requires_grad]
    d_trainable = int(sum(int(p.numel()) for _, p in named_params))
    backup_params = [p.detach().clone() for _, p in named_params]

    diagnostics.setdefault("batch_info", {})[f"{checkpoint.precision}/{checkpoint.checkpoint_name}/{mode_name}"] = {
        "batch_size": int(batch["input_ids"].shape[0]) if "input_ids" in batch else None,
        "input_ids_shape": list(batch["input_ids"].shape) if "input_ids" in batch else None,
        "dataset_indices": [int(i) for i in batch_indices],
        "labels": batch.get("labels", batch.get("label", None)).detach().cpu().tolist()
        if hasattr(batch.get("labels", batch.get("label", None)), "detach")
        else None,
        "sampler": "torch.randperm first batch matching RandomSampler convention",
        "sampler_seed": int(getattr(training_args, "data_seed", 16)),
    }

    return EvalContext(
        model=model,
        batch=batch,
        named_params=named_params,
        backup_params=backup_params,
        d_trainable=d_trainable,
        device=device,
        forward_precision=forward_precision,
        delta_mode=delta_mode,
        direction_dtype_name=direction_dtype_name,
        tf32_enabled=f"matmul={tf32_state['matmul']};cudnn={tf32_state['cudnn']}",
        autocast_enabled=bool(forward_precision == "fp16" and str(device).startswith("cuda")),
        autocast_dtype="torch.float16" if forward_precision == "fp16" and str(device).startswith("cuda") else "",
    )


def restore_params(ctx: EvalContext) -> None:
    with no_grad():
        for (_, param), backup in zip(ctx.named_params, ctx.backup_params):
            param.data.copy_(backup)


def no_grad():
    import torch

    return torch.no_grad()


def autocast_context(ctx: EvalContext):
    import contextlib
    import torch

    if ctx.forward_precision == "fp16" and str(ctx.device).startswith("cuda"):
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return contextlib.nullcontext()


def compute_loss(ctx: EvalContext) -> Tuple[float, str]:
    with no_grad():
        with autocast_context(ctx):
            outputs = ctx.model(**ctx.batch)
            loss = outputs[0] if isinstance(outputs, (tuple, list)) else outputs.loss
    ctx.loss_dtype = str(loss.dtype)
    return float(loss.detach().float().cpu().item()), str(loss.dtype)


def direction_dtype(ctx: EvalContext):
    import torch

    return torch.float16 if ctx.direction_dtype_name == "float16" else torch.float32


def reset_rng_for_direction(seed: int) -> None:
    import torch

    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def sample_direction_like(param: Any, dtype: Any):
    import torch

    return torch.empty_like(param.data, dtype=dtype).normal_(mean=0.0, std=1.0)


def build_delta(z: Any, h2: float, ctx: EvalContext, param: Any):
    import torch

    if ctx.delta_mode == "fp16_delta_snap":
        return (z * float(h2)).detach().to(dtype=torch.float16).to(dtype=param.data.dtype)
    return (z * float(h2)).to(dtype=param.data.dtype)


def update_eff_stats(
    accum: Dict[str, float],
    effective: Any,
    intended: Any,
) -> None:
    import torch

    eff = torch.nan_to_num(effective.detach().float(), nan=0.0, posinf=0.0, neginf=0.0)
    tgt = torch.nan_to_num(intended.detach().float(), nan=0.0, posinf=0.0, neginf=0.0)
    accum["dot"] += float(torch.sum(eff * tgt).item())
    accum["eff_sq"] += float(torch.sum(eff * eff).item())
    accum["tgt_sq"] += float(torch.sum(tgt * tgt).item())
    diff = eff - tgt
    accum["snap_sq"] += float(torch.sum(diff * diff).item())
    accum["zero"] += int(torch.count_nonzero(eff == 0).item())
    accum["total"] += int(eff.numel())


def finalize_eff_stats(accum: Dict[str, float]) -> Dict[str, float]:
    eff_norm = math.sqrt(max(float(accum.get("eff_sq", 0.0)), 0.0))
    tgt_norm = math.sqrt(max(float(accum.get("tgt_sq", 0.0)), 0.0))
    denom = eff_norm * tgt_norm
    total = int(accum.get("total", 0))
    return {
        "alignment": float(accum["dot"] / denom) if denom > 0 else float("nan"),
        "norm_ratio": float(eff_norm / (tgt_norm + EPS_NUM)) if tgt_norm > 0 else float("nan"),
        "zero_coord_frac": float(accum["zero"] / total) if total > 0 else float("nan"),
        "rms_snap_error": float(math.sqrt(max(accum["snap_sq"], 0.0) / total)) if total > 0 else float("nan"),
    }


def apply_once(
    ctx: EvalContext,
    *,
    seed: int,
    h2: float,
    collect_eff: bool,
    eff_accum: Optional[Dict[str, float]],
    intended_multiplier: float,
) -> float:
    import torch

    dtype = direction_dtype(ctx)
    norm_sq = 0.0
    reset_rng_for_direction(seed)
    with no_grad():
        for idx, (_, param) in enumerate(ctx.named_params):
            z = sample_direction_like(param, dtype=dtype)
            norm_sq += float(torch.sum(z.detach().float() * z.detach().float()).item())
            delta = build_delta(z, h2, ctx, param)
            param.data.add_(delta)
            if collect_eff and eff_accum is not None:
                effective = param.data.detach().float() - ctx.backup_params[idx].detach().float()
                intended = (float(intended_multiplier) * float(h2)) * z.detach().float()
                update_eff_stats(eff_accum, effective, intended)
            del z, delta
    return norm_sq


def direction_norm_sq(ctx: EvalContext, seed: int) -> float:
    dtype = direction_dtype(ctx)
    norm_sq = 0.0
    reset_rng_for_direction(seed)
    with no_grad():
        for _, param in ctx.named_params:
            z = sample_direction_like(param, dtype=dtype)
            norm_sq += float((z.detach().float() * z.detach().float()).sum().item())
            del z
    return norm_sq


def second_order_probe_direction(
    ctx: EvalContext,
    *,
    seed: int,
    h2: float,
    base_loss: float,
    collect_eff: bool,
) -> Dict[str, Any]:
    restore_params(ctx)
    eff1 = {"dot": 0.0, "eff_sq": 0.0, "tgt_sq": 0.0, "snap_sq": 0.0, "zero": 0, "total": 0}
    eff2 = {"dot": 0.0, "eff_sq": 0.0, "tgt_sq": 0.0, "snap_sq": 0.0, "zero": 0, "total": 0}
    try:
        norm_sq = apply_once(ctx, seed=seed, h2=h2, collect_eff=collect_eff, eff_accum=eff1, intended_multiplier=1.0)
        loss1, loss_dtype1 = compute_loss(ctx)
        _ = apply_once(ctx, seed=seed, h2=h2, collect_eff=collect_eff, eff_accum=eff2, intended_multiplier=2.0)
        loss2, loss_dtype2 = compute_loss(ctx)
    finally:
        restore_params(ctx)
    k_sh = (float(loss2) - 2.0 * float(loss1) + float(base_loss)) / max(float(h2) ** 2, 1e-30)
    lam = abs(float(k_sh)) / (float(norm_sq) + EPS_NUM)
    out = {
        "seed": int(seed),
        "h2": float(h2),
        "norm_sq": float(norm_sq),
        "loss1": float(loss1),
        "loss2": float(loss2),
        "loss_dtype": loss_dtype2 or loss_dtype1,
        "K_sh": float(k_sh),
        "lambda": float(lam),
    }
    if collect_eff:
        out["eff1"] = finalize_eff_stats(eff1)
        out["eff2"] = finalize_eff_stats(eff2)
    return out


def estimate_ulp_stats(ctx: EvalContext, dtype_name: str) -> Dict[str, Any]:
    import torch

    target_dtype = torch.float16 if dtype_name == "fp16" else torch.float32
    samples: List[np.ndarray] = []
    sum_sq = 0.0
    count = 0
    finite_count = 0
    nonfinite_count = 0
    zero_spacing_count = 0
    dtype_seen: Dict[str, int] = {}
    sample_cap = 8192
    with no_grad():
        for _, param in ctx.named_params:
            tensor = param.detach()
            dtype_seen[str(tensor.dtype)] = dtype_seen.get(str(tensor.dtype), 0) + 1
            if target_dtype == torch.float16:
                cast = tensor.cpu().to(dtype=target_dtype)
            else:
                cast = tensor.to(dtype=target_dtype)
            inf = torch.full_like(cast, float("inf"))
            spacing = (torch.nextafter(cast, inf) - cast).abs().to(dtype=torch.float32).reshape(-1)
            finite = torch.isfinite(spacing)
            n_total = int(spacing.numel())
            n_finite = int(finite.sum().item())
            count += n_total
            finite_count += n_finite
            nonfinite_count += n_total - n_finite
            if n_finite <= 0:
                continue
            vals = spacing[finite]
            zero_spacing_count += int((vals == 0).sum().item())
            sum_sq += float(torch.sum(vals * vals).item())
            n_sample = min(sample_cap, int(vals.numel()))
            if n_sample > 0:
                if n_sample == int(vals.numel()):
                    sample = vals
                else:
                    sample = vals[:n_sample]
                samples.append(sample.detach().cpu().numpy().astype(np.float64, copy=False))
            del cast, inf, spacing
    out: Dict[str, Any] = {
        "dtype": dtype_name,
        "count": int(count),
        "finite_count": int(finite_count),
        "nonfinite_count": int(nonfinite_count),
        "zero_spacing_count": int(zero_spacing_count),
        "dtype_seen": dtype_seen,
    }
    if finite_count > 0 and sum_sq > 0.0:
        out["delta_ulp_rms"] = float(math.sqrt(sum_sq / float(finite_count)))
    if samples:
        arr = np.concatenate(samples)
        arr = arr[np.isfinite(arr)]
        if arr.size:
            out["delta_ulp_median"] = float(np.quantile(arr, 0.50))
            out["delta_ulp_p90"] = float(np.quantile(arr, 0.90))
            out["delta_ulp_p95"] = float(np.quantile(arr, 0.95))
            out["ulp_sample_count"] = int(arr.size)
    return out


def h2_grid_from_delta(delta_value: Optional[float], diagnostics: Dict[str, Any], grid_key: str) -> List[float]:
    values = list(FIXED_H2_GRID)
    if delta_value is not None and math.isfinite(float(delta_value)) and float(delta_value) > 0.0:
        h2_0 = math.sqrt(float(delta_value))
        old: List[float] = []
        for j in range(-4, 5):
            h = max(float(delta_value), (2.0 ** j) * h2_0)
            if h > 1e-2:
                diagnostics.setdefault("warnings", []).append(
                    f"{grid_key}: old h2 ladder value {h:.6g} exceeds 1e-2 and was skipped"
                )
                continue
            old.append(float(h))
        diagnostics.setdefault("h2_old_ladders", {})[grid_key] = old
        values.extend(old)
    else:
        diagnostics.setdefault("warnings", []).append(f"{grid_key}: Delta unavailable; old h2 ladder skipped")
    return sorted_unique(values)


def summarize_h2(
    *,
    precision: str,
    checkpoint: CheckpointInfo,
    l_mode: str,
    h2: float,
    m_l: int,
    ctx: EvalContext,
    base_loss: float,
    direction_outputs: List[Dict[str, Any]],
    warnings: Sequence[str],
) -> Dict[str, Any]:
    k_vals = np.asarray([row["K_sh"] for row in direction_outputs], dtype=np.float64)
    lam_vals = np.asarray([row["lambda"] for row in direction_outputs], dtype=np.float64)
    finite_k = np.isfinite(k_vals)
    finite_lam = np.isfinite(lam_vals)
    finite_rate = float(np.mean(finite_k)) if k_vals.size else 0.0
    k_fin = k_vals[finite_k]
    lam_fin = lam_vals[finite_lam]
    med_k = float(np.median(k_fin)) if k_fin.size else float("nan")
    mad_k = float(np.median(np.abs(k_fin - med_k))) if k_fin.size else float("nan")
    med_abs_k = float(np.median(np.abs(k_fin))) if k_fin.size else float("nan")
    snr2 = float(med_abs_k / (1.4826 * mad_k + EPS_NUM)) if math.isfinite(med_abs_k) and math.isfinite(mad_k) else float("nan")
    zero_k_frac = float(np.mean(np.abs(k_fin) < EPS_NUM)) if k_fin.size else float("nan")

    eff1_vals = [row.get("eff1") for row in direction_outputs if row.get("eff1")]
    eff2_vals = [row.get("eff2") for row in direction_outputs if row.get("eff2")]

    def avg_eff(items: List[Dict[str, float]], key: str) -> float:
        vals = [float(x[key]) for x in items if key in x and math.isfinite(float(x[key]))]
        return float(np.mean(vals)) if vals else float("nan")

    loss_dtype = ""
    for item in reversed(direction_outputs):
        if item.get("loss_dtype"):
            loss_dtype = str(item["loss_dtype"])
            break

    return {
        "precision": precision,
        "checkpoint_name": checkpoint.checkpoint_name,
        "checkpoint_path": str(checkpoint.checkpoint_dir),
        "target_run_h": checkpoint.target_run_h,
        "L_mode": l_mode,
        "h2": float(h2),
        "m_L": int(m_l),
        "d_trainable": int(ctx.d_trainable),
        "direction_normalization": "raw Gaussian unnormalized; torch normal per trainable parameter",
        "forward_precision": ctx.forward_precision,
        "loss_dtype": loss_dtype or ctx.loss_dtype,
        "tf32_enabled": ctx.tf32_enabled,
        "autocast_enabled": ctx.autocast_enabled,
        "autocast_dtype": ctx.autocast_dtype,
        "delta_mode": ctx.delta_mode,
        "base_loss": float(base_loss),
        "median_abs_K": med_abs_k,
        "mean_K": float(np.mean(k_fin)) if k_fin.size else float("nan"),
        "std_K": float(np.std(k_fin)) if k_fin.size else float("nan"),
        "MAD_K": mad_k,
        "SNR2": snr2,
        "finite_rate": finite_rate,
        "zero_K_frac": zero_k_frac,
        "lambda_q50": float(np.quantile(lam_fin, 0.50)) if lam_fin.size else float("nan"),
        "lambda_q90": float(np.quantile(lam_fin, 0.90)) if lam_fin.size else float("nan"),
        "lambda_q95": float(np.quantile(lam_fin, 0.95)) if lam_fin.size else float("nan"),
        "lambda_mean": float(np.mean(lam_fin)) if lam_fin.size else float("nan"),
        "lambda_std": float(np.std(lam_fin)) if lam_fin.size else float("nan"),
        "stability_q50_2x": float("nan"),
        "stability_q90_2x": float("nan"),
        "stability_q95_2x": float("nan"),
        "stability_q90_next": float("nan"),
        "stability_q90_prev": float("nan"),
        "log_slope_q90_next": float("nan"),
        "log_slope_q90_prev": float("nan"),
        "low_h2_noise_suspected": False,
        "large_h2_nonlocal_suspected": False,
        "alignment_eff_1": avg_eff(eff1_vals, "alignment"),
        "norm_ratio_eff_1": avg_eff(eff1_vals, "norm_ratio"),
        "zero_coord_frac_eff_1": avg_eff(eff1_vals, "zero_coord_frac"),
        "rms_snap_error_1": avg_eff(eff1_vals, "rms_snap_error"),
        "alignment_eff_2": avg_eff(eff2_vals, "alignment"),
        "norm_ratio_eff_2": avg_eff(eff2_vals, "norm_ratio"),
        "zero_coord_frac_eff_2": avg_eff(eff2_vals, "zero_coord_frac"),
        "rms_snap_error_2": avg_eff(eff2_vals, "rms_snap_error"),
        "warnings": ";".join(warnings),
    }


def add_stability_and_flags(rows: List[Dict[str, Any]]) -> None:
    rows.sort(key=lambda r: float(r["h2"]))
    by_h = {float(r["h2"]): r for r in rows}
    hs = [float(r["h2"]) for r in rows]
    for idx, row in enumerate(rows):
        h = float(row["h2"])
        h2x = find_close(2.0 * h, hs)
        if h2x is not None:
            other = by_h[h2x]
            for q in ["q50", "q90", "q95"]:
                a = safe_float(row.get(f"lambda_{q}"))
                b = safe_float(other.get(f"lambda_{q}"))
                row[f"stability_{q}_2x"] = (
                    abs(a - b) / (abs(a) + EPS_NUM) if a is not None and b is not None else float("nan")
                )
        if idx + 1 < len(rows):
            nxt = rows[idx + 1]
            a = safe_float(row.get("lambda_q90"))
            b = safe_float(nxt.get("lambda_q90"))
            if a is not None and b is not None:
                row["stability_q90_next"] = abs(a - b) / (abs(a) + EPS_NUM)
                row["log_slope_q90_next"] = abs(math.log(b + EPS_NUM) - math.log(a + EPS_NUM)) / abs(
                    math.log(float(nxt["h2"])) - math.log(h)
                )
        if idx > 0:
            prv = rows[idx - 1]
            a = safe_float(row.get("lambda_q90"))
            b = safe_float(prv.get("lambda_q90"))
            if a is not None and b is not None:
                row["stability_q90_prev"] = abs(a - b) / (abs(a) + EPS_NUM)
                row["log_slope_q90_prev"] = abs(math.log(a + EPS_NUM) - math.log(b + EPS_NUM)) / abs(
                    math.log(h) - math.log(float(prv["h2"]))
                )

    q90_vals = [safe_float(r.get("lambda_q90")) for r in rows]
    for idx, row in enumerate(rows):
        h = float(row["h2"])
        q90 = safe_float(row.get("lambda_q90"))
        low = False
        if q90 is not None:
            larger = [safe_float(r.get("lambda_q90")) for r in rows if float(r["h2"]) > h]
            larger = [v for v in larger if v is not None and v > 0]
            if larger and q90 / (float(np.median(larger)) + EPS_NUM) >= 5.0:
                low = True
            if idx + 1 < len(rows):
                nxt = safe_float(rows[idx + 1].get("lambda_q90"))
                if nxt is not None and q90 / (nxt + EPS_NUM) >= 5.0:
                    low = True
        align1 = safe_float(row.get("alignment_eff_1"))
        norm1 = safe_float(row.get("norm_ratio_eff_1"))
        zero1 = safe_float(row.get("zero_coord_frac_eff_1"))
        if align1 is not None and align1 < 0.90:
            low = True
        if norm1 is not None and not (0.5 <= norm1 <= 2.0):
            low = True
        if zero1 is not None and zero1 >= 0.50:
            low = True
        finite_rate = safe_float(row.get("finite_rate"))
        zero_k = safe_float(row.get("zero_K_frac"))
        if finite_rate is not None and finite_rate < 0.95:
            low = True
        if zero_k is not None and zero_k >= 0.50:
            low = True
        row["low_h2_noise_suspected"] = bool(low)

    finite_q90 = [v for v in q90_vals if v is not None and v > 0]
    median_q90 = float(np.median(finite_q90)) if finite_q90 else float("nan")
    for idx, row in enumerate(rows):
        large = False
        h = float(row["h2"])
        q90 = safe_float(row.get("lambda_q90"))
        upper = idx >= max(0, len(rows) - 3)
        prev_slope = safe_float(row.get("log_slope_q90_prev"))
        next_slope = safe_float(row.get("log_slope_q90_next"))
        slope = min([v for v in [prev_slope, next_slope] if v is not None] or [float("nan")])
        if upper and q90 is not None and math.isfinite(median_q90) and q90 > 3.0 * median_q90:
            large = True
        if upper and math.isfinite(slope) and slope > 1.5:
            large = True
        row["large_h2_nonlocal_suspected"] = bool(large)


def q_value(row: Dict[str, Any], q_name: str) -> float:
    return float(row.get(f"lambda_{q_name}", float("nan")))


def stability_score(row: Dict[str, Any], q_name: str) -> float:
    vals: List[float] = []
    if q_name == "q95":
        v = safe_float(row.get("stability_q95_2x"))
        if v is not None:
            vals.append(v)
    elif q_name == "q50":
        v = safe_float(row.get("stability_q50_2x"))
        if v is not None:
            vals.append(v)
    else:
        v = safe_float(row.get("stability_q90_2x"))
        if v is not None:
            vals.append(v)
    for key in ["stability_q90_next", "stability_q90_prev"]:
        v = safe_float(row.get(key))
        if v is not None:
            vals.append(v)
    return min(vals) if vals else float("inf")


def slope_score(row: Dict[str, Any]) -> float:
    vals = []
    for key in ["log_slope_q90_next", "log_slope_q90_prev"]:
        v = safe_float(row.get(key))
        if v is not None:
            vals.append(v)
    return min(vals) if vals else float("inf")


def selected_from_row(
    row: Dict[str, Any],
    selector_name: str,
    q_name: str,
    status: str,
    fallback_flags: Sequence[str],
    notes: str,
) -> Dict[str, Any]:
    return {
        "precision": row.get("precision"),
        "checkpoint_name": row.get("checkpoint_name"),
        "checkpoint_path": row.get("checkpoint_path"),
        "L_mode": row.get("L_mode"),
        "selector_name": selector_name,
        "selected_h2": row.get("h2"),
        "selected_L_q": q_name,
        "selected_L_value": row.get(f"lambda_{q_name}") if q_name in {"q50", "q90", "q95"} else row.get("lambda_q90"),
        "selected_L_q50": row.get("lambda_q50"),
        "selected_L_q90": row.get("lambda_q90"),
        "selected_L_q95": row.get("lambda_q95"),
        "SNR2": row.get("SNR2"),
        "stability_q90_2x": row.get("stability_q90_2x"),
        "stability_q90_next": row.get("stability_q90_next"),
        "stability_q90_prev": row.get("stability_q90_prev"),
        "low_h2_noise_suspected": row.get("low_h2_noise_suspected"),
        "large_h2_nonlocal_suspected": row.get("large_h2_nonlocal_suspected"),
        "selection_status": status,
        "fallback_flags": ";".join(fallback_flags),
        "notes": notes,
    }


def no_selection(
    template: Dict[str, Any],
    selector_name: str,
    q_name: str,
    status: str,
    fallback_flags: Sequence[str],
    notes: str,
) -> Dict[str, Any]:
    return {
        "precision": template.get("precision"),
        "checkpoint_name": template.get("checkpoint_name"),
        "checkpoint_path": template.get("checkpoint_path"),
        "L_mode": template.get("L_mode"),
        "selector_name": selector_name,
        "selected_h2": "",
        "selected_L_q": q_name,
        "selected_L_value": "",
        "selected_L_q50": "",
        "selected_L_q90": "",
        "selected_L_q95": "",
        "SNR2": "",
        "stability_q90_2x": "",
        "stability_q90_next": "",
        "stability_q90_prev": "",
        "low_h2_noise_suspected": "",
        "large_h2_nonlocal_suspected": "",
        "selection_status": status,
        "fallback_flags": ";".join(fallback_flags),
        "notes": notes,
    }


def select_old_snr_smallest(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    passing = [r for r in rows if (safe_float(r.get("SNR2")) is not None and float(r["SNR2"]) >= TAU_SNR)]
    if passing:
        return selected_from_row(passing[0], "old_snr_smallest_pass", "q90", "selected", [], "smallest h2 with SNR2 >= 2.0")
    return no_selection(rows[0], "old_snr_smallest_pass", "q90", "no_snr_pass", ["no_snr_pass"], "no h2 passed SNR2 threshold")


def select_old_snr_fallback(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    passing = [r for r in rows if (safe_float(r.get("SNR2")) is not None and float(r["SNR2"]) >= TAU_SNR)]
    if passing:
        return selected_from_row(
            passing[0],
            "old_snr_max_fallback_ablation",
            "q90",
            "selected",
            ["ablation_only"],
            "old rule selected smallest SNR-passing h2",
        )
    finite = [r for r in rows if safe_float(r.get("SNR2")) is not None]
    if not finite:
        return no_selection(
            rows[0],
            "old_snr_max_fallback_ablation",
            "q90",
            "unavailable",
            ["ablation_only", "no_finite_snr"],
            "no finite SNR2 values",
        )
    chosen = max(finite, key=lambda r: float(r["SNR2"]))
    flags = ["ablation_only", "fallback_max_snr"]
    if bool(chosen.get("low_h2_noise_suspected")):
        flags.append("selected_low_h2_noise")
    return selected_from_row(chosen, "old_snr_max_fallback_ablation", "q90", "selected", flags, "old max-SNR fallback ablation")


def plateau_candidates(rows: List[Dict[str, Any]], q_name: str, allow_large: bool) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in rows:
        finite_rate = safe_float(row.get("finite_rate"))
        qv = safe_float(row.get(f"lambda_{q_name}"))
        if finite_rate is None or finite_rate < 0.95 or qv is None:
            continue
        if bool(row.get("low_h2_noise_suspected")):
            continue
        if (not allow_large) and bool(row.get("large_h2_nonlocal_suspected")):
            continue
        stab = stability_score(row, q_name)
        slope = slope_score(row)
        if not (stab <= 0.5 or math.isinf(stab)):
            continue
        if not (slope <= 1.0 or math.isinf(slope)):
            continue
        out.append(row)
    return out


def select_plateau(rows: List[Dict[str, Any]], q_name: str, selector_name: str) -> Dict[str, Any]:
    candidates = plateau_candidates(rows, q_name, allow_large=False)
    flags: List[str] = []
    if not candidates:
        candidates = plateau_candidates(rows, q_name, allow_large=True)
        if candidates:
            flags.append("large_h2_allowed")
    if not candidates:
        diagnostic = sorted(
            rows,
            key=lambda r: (
                bool(r.get("low_h2_noise_suspected")),
                bool(r.get("large_h2_nonlocal_suspected")),
                stability_score(r, q_name),
                float(r.get("h2", float("inf"))),
            ),
        )
        if diagnostic:
            best = diagnostic[0]
            return selected_from_row(
                best,
                selector_name,
                q_name,
                "fallback_unreliable",
                ["fallback_unreliable"],
                "no strict plateau candidate; reporting least-bad diagnostic candidate",
            )
        return no_selection(rows[0], selector_name, q_name, "primary_L_unavailable", ["no_candidates"], "no rows available")
    chosen = sorted(
        candidates,
        key=lambda r: (
            bool(r.get("low_h2_noise_suspected")),
            stability_score(r, q_name),
            0 if (safe_float(r.get("SNR2")) is not None and float(r["SNR2"]) >= 1.0) else 1,
            float(r["h2"]),
        ),
    )[0]
    return selected_from_row(chosen, selector_name, q_name, "selected", flags, "stable curvature plateau selector")


def select_fixed_h2(rows: List[Dict[str, Any]], fixed_h2: float) -> Dict[str, Any]:
    match_h = find_close(fixed_h2, [float(r["h2"]) for r in rows])
    selector = f"fixed_h2_{fixed_h2:.12g}"
    if match_h is None:
        return no_selection(rows[0], selector, "q90", "missing", ["fixed_h2_missing"], f"h2={fixed_h2:g} unavailable")
    row = next(r for r in rows if h_close(float(r["h2"]), match_h))
    return selected_from_row(row, selector, "q90", "diagnostic_only", ["diagnostic_only"], "fixed h2 diagnostic")


def select_all(group_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = sorted(group_rows, key=lambda r: float(r["h2"]))
    selected = [
        select_old_snr_smallest(rows),
        select_old_snr_fallback(rows),
        select_plateau(rows, "q90", "plateau_q90_primary"),
        select_plateau(rows, "q95", "plateau_q95_conservative"),
    ]
    for h2 in FIXED_H2_DIAGNOSTIC:
        selected.append(select_fixed_h2(rows, h2))
    return selected


def run_group(
    checkpoint: CheckpointInfo,
    l_mode: str,
    *,
    args: argparse.Namespace,
    direction_seeds: Sequence[int],
    device: Any,
    diagnostics: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    import torch

    warnings: List[str] = []
    if l_mode in {"L_oracle_precision", "L_oracle_oldSNR"} and checkpoint.precision == "fp16" and not str(device).startswith("cuda"):
        raise RuntimeError("FP16 oracle mode requires CUDA in this analyzer; refusing to silently substitute CPU/BF16 behavior")
    ctx = load_eval_context(checkpoint, mode_name=l_mode, device=device, diagnostics=diagnostics)
    base_loss, base_loss_dtype = compute_loss(ctx)
    ctx.loss_dtype = base_loss_dtype

    ulp_dtype = "fp16" if (l_mode != "L_clean32" and checkpoint.precision == "fp16") else "fp32"
    ulp = estimate_ulp_stats(ctx, ulp_dtype)
    grid_key = f"{checkpoint.precision}/{checkpoint.checkpoint_name}/{l_mode}"
    diagnostics.setdefault("Delta_estimates", {})[grid_key] = ulp
    delta_for_grid = safe_float(ulp.get("delta_ulp_rms"))
    h2_grid = h2_grid_from_delta(delta_for_grid, diagnostics, grid_key)
    if args.h2_grid:
        h2_grid = sorted_unique([float(x) for x in args.h2_grid])
    diagnostics.setdefault("h2_grids", {})[grid_key] = h2_grid

    raw_by_h2: Dict[float, List[Dict[str, Any]]] = {}
    eff_count = max(0, int(args.effective_directions))
    t0 = time.time()
    print(f"[L-est] start {grid_key}: h2_count={len(h2_grid)} m_L={len(direction_seeds)} device={device}", flush=True)
    for h2 in h2_grid:
        h2_t0 = time.time()
        direction_outputs: List[Dict[str, Any]] = []
        for idx, seed in enumerate(direction_seeds):
            collect_eff = idx < eff_count
            out = second_order_probe_direction(ctx, seed=int(seed), h2=float(h2), base_loss=base_loss, collect_eff=collect_eff)
            direction_outputs.append(out)
        raw_by_h2[float(h2)] = direction_outputs
        print(f"[L-est] done {grid_key}: h2={float(h2):.6g} directions={len(direction_outputs)} elapsed_sec={time.time() - h2_t0:.1f}", flush=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    elapsed = time.time() - t0
    diagnostics.setdefault("group_runtime_sec", {})[grid_key] = elapsed

    rows = [
        summarize_h2(
            precision=checkpoint.precision,
            checkpoint=checkpoint,
            l_mode=l_mode,
            h2=h2,
            m_l=len(direction_seeds),
            ctx=ctx,
            base_loss=base_loss,
            direction_outputs=outs,
            warnings=warnings,
        )
        for h2, outs in sorted(raw_by_h2.items())
    ]
    add_stability_and_flags(rows)

    raw_json = {
        "precision": checkpoint.precision,
        "checkpoint_name": checkpoint.checkpoint_name,
        "checkpoint_path": str(checkpoint.checkpoint_dir),
        "L_mode": l_mode,
        "direction_seeds": list(map(int, direction_seeds)),
        "base_loss": base_loss,
        "base_loss_dtype": base_loss_dtype,
        "raw_by_h2": {h_key(h): outs for h, outs in raw_by_h2.items()},
    }
    restore_params(ctx)
    del ctx
    return rows, raw_json


def duplicate_old_snr_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in rows:
        dup = dict(row)
        dup["L_mode"] = "L_oracle_oldSNR"
        dup["warnings"] = str(dup.get("warnings", ""))
        if dup["warnings"]:
            dup["warnings"] += ";"
        dup["warnings"] += "raw metrics duplicated from L_oracle_precision for oldSNR ablation"
        out.append(dup)
    return out


def group_key(row: Dict[str, Any]) -> Tuple[str, str, str]:
    return (str(row["precision"]), str(row["checkpoint_name"]), str(row["L_mode"]))


def plot_group(rows: List[Dict[str, Any]], selected_rows: List[Dict[str, Any]], out_dir: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return

    if not rows:
        return
    precision, checkpoint_name, l_mode = group_key(rows[0])
    plot_dir = out_dir / "plots" / precision / checkpoint_name / l_mode
    plot_dir.mkdir(parents=True, exist_ok=True)
    hs = np.asarray([float(r["h2"]) for r in rows], dtype=np.float64)

    selected_by_name = {str(r["selector_name"]): safe_float(r.get("selected_h2")) for r in selected_rows}
    mark_names = ["plateau_q90_primary", "old_snr_smallest_pass", "old_snr_max_fallback_ablation"]
    fixed_marks = [1e-3, 2e-3, 3e-3]

    def mark(ax):
        for name in mark_names:
            h = selected_by_name.get(name)
            if h is not None:
                ax.axvline(h, linestyle="--", linewidth=1.0, label=name)
        for h in fixed_marks:
            if find_close(h, hs) is not None:
                ax.axvline(h, linestyle=":", linewidth=1.0, label=f"fixed {h:g}")

    fig, ax = plt.subplots(figsize=(7, 5))
    for q in ["q50", "q90", "q95"]:
        ax.plot(hs, [safe_float(r.get(f"lambda_{q}")) for r in rows], marker="o", label=f"L_{q}")
    mark(ax)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("h2")
    ax.set_ylabel("lambda quantile")
    ax.set_title(f"{precision} {checkpoint_name} {l_mode}: L vs h2")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(plot_dir / "L_quantiles_vs_h2.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(hs, [safe_float(r.get("SNR2")) for r in rows], marker="o")
    mark(ax)
    ax.set_xscale("log")
    ax.set_xlabel("h2")
    ax.set_ylabel("SNR2")
    ax.set_title(f"{precision} {checkpoint_name} {l_mode}: SNR2 vs h2")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(plot_dir / "SNR2_vs_h2.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(hs, [safe_float(r.get("stability_q90_2x")) for r in rows], marker="o", label="2x")
    ax.plot(hs, [safe_float(r.get("stability_q90_next")) for r in rows], marker="o", label="next")
    ax.plot(hs, [safe_float(r.get("stability_q90_prev")) for r in rows], marker="o", label="prev")
    mark(ax)
    ax.set_xscale("log")
    ax.set_xlabel("h2")
    ax.set_ylabel("q90 stability")
    ax.set_title(f"{precision} {checkpoint_name} {l_mode}: stability")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(plot_dir / "stability_q90_vs_h2.png", dpi=160)
    plt.close(fig)

    eff_keys = ["alignment_eff_1", "norm_ratio_eff_1", "zero_coord_frac_eff_1", "alignment_eff_2", "norm_ratio_eff_2", "zero_coord_frac_eff_2"]
    if any(safe_float(r.get(k)) is not None for r in rows for k in eff_keys):
        fig, ax = plt.subplots(figsize=(7, 5))
        for key in eff_keys:
            vals = [safe_float(r.get(key)) for r in rows]
            if any(v is not None for v in vals):
                ax.plot(hs, vals, marker="o", label=key)
        mark(ax)
        ax.set_xscale("log")
        ax.set_xlabel("h2")
        ax.set_ylabel("effective displacement metric")
        ax.set_title(f"{precision} {checkpoint_name} {l_mode}: effective displacement")
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(plot_dir / "effective_displacement_vs_h2.png", dpi=160)
        plt.close(fig)


def format_float(value: Any) -> str:
    f = safe_float(value)
    if f is None:
        return ""
    return f"{f:.6g}"


def make_summary(rows: List[Dict[str, Any]], selected: List[Dict[str, Any]], diagnostics: Dict[str, Any], out_dir: Path) -> str:
    lines: List[str] = []
    lines.append("# L Estimation Summary")
    lines.append("")
    lines.append(f"Analysis directory: `{out_dir}`")
    lines.append("")
    lines.append("## Table A: Selectors")
    lines.append("")
    lines.append("| precision | checkpoint | L_mode | selector | selected h2 | L_q50 | L_q90 | L_q95 | SNR2 | stability | flags |")
    lines.append("|---|---|---|---|---:|---:|---:|---:|---:|---:|---|")
    wanted = {
        "plateau_q90_primary",
        "plateau_q95_conservative",
        "old_snr_smallest_pass",
        "old_snr_max_fallback_ablation",
    }
    for item in selected:
        if item.get("selector_name") not in wanted:
            continue
        stability = item.get("stability_q90_2x")
        if safe_float(stability) is None:
            stability = item.get("stability_q90_next") if safe_float(item.get("stability_q90_next")) is not None else item.get("stability_q90_prev")
        flags = ";".join(
            [
                f
                for f in [
                    "low_h2_noise" if str(item.get("low_h2_noise_suspected")).lower() == "true" else "",
                    "large_h2_nonlocal" if str(item.get("large_h2_nonlocal_suspected")).lower() == "true" else "",
                    str(item.get("fallback_flags") or ""),
                ]
                if f
            ]
        )
        lines.append(
            "| {precision} | {checkpoint} | {mode} | {selector} | {h2} | {q50} | {q90} | {q95} | {snr} | {stab} | {flags} |".format(
                precision=item.get("precision", ""),
                checkpoint=item.get("checkpoint_name", ""),
                mode=item.get("L_mode", ""),
                selector=item.get("selector_name", ""),
                h2=format_float(item.get("selected_h2")),
                q50=format_float(item.get("selected_L_q50")),
                q90=format_float(item.get("selected_L_q90")),
                q95=format_float(item.get("selected_L_q95")),
                snr=format_float(item.get("SNR2")),
                stab=format_float(stability),
                flags=flags,
            )
        )
    lines.append("")
    lines.append("## Table B: Primary Comparison")
    lines.append("")
    lines.append("| precision | checkpoint | L_clean32 plateau q90 | L_oracle plateau q90 | L_oracle oldSNR q90 | interpretation |")
    lines.append("|---|---|---:|---:|---:|---|")
    keys = sorted({(s.get("precision"), s.get("checkpoint_name")) for s in selected})
    for precision, ckpt in keys:
        def find_sel(mode: str, selector: str) -> Optional[Dict[str, Any]]:
            for s in selected:
                if s.get("precision") == precision and s.get("checkpoint_name") == ckpt and s.get("L_mode") == mode and s.get("selector_name") == selector:
                    return s
            return None

        clean = find_sel("L_clean32", "plateau_q90_primary")
        oracle = find_sel("L_oracle_precision", "plateau_q90_primary")
        old = find_sel("L_oracle_oldSNR", "old_snr_max_fallback_ablation")
        interp: List[str] = []
        if old and str(old.get("low_h2_noise_suspected")).lower() == "true":
            interp.append("oldSNR selected flagged low-h2 noise")
        clean_v = safe_float(clean.get("selected_L_q90") if clean else None)
        oracle_v = safe_float(oracle.get("selected_L_q90") if oracle else None)
        if clean_v is not None and oracle_v is not None and max(clean_v, oracle_v) / (min(clean_v, oracle_v) + EPS_NUM) > 5.0:
            interp.append("clean/oracle differ by >5x")
        if not interp:
            interp.append("see selector flags")
        lines.append(
            f"| {precision} | {ckpt} | {format_float(clean.get('selected_L_q90') if clean else None)} | "
            f"{format_float(oracle.get('selected_L_q90') if oracle else None)} | "
            f"{format_float(old.get('selected_L_q90') if old else None)} | {'; '.join(interp)} |"
        )
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append("- The theoretical h-star formula should consume `L_clean32` with `plateau_q90_primary` when that selector is available.")
    lines.append("- `L_oracle_precision` is an oracle-consistent diagnostic; it should not replace clean FP32 curvature unless the downstream analysis explicitly wants oracle noise folded into L.")
    lines.append("- `L_oracle_oldSNR` is an ablation to expose the previous max-SNR fallback behavior.")
    if diagnostics.get("warnings"):
        lines.append("- Warnings were emitted; see `L_diagnostics.json` for details.")
    lines.append("")
    return "\n".join(lines)


def terminal_report(selected: List[Dict[str, Any]], diagnostics: Dict[str, Any], out_dir: Path) -> str:
    lines: List[str] = []
    lines.append(f"Analysis output directory: {out_dir}")

    def sel(precision: str, mode: str, selector: str) -> Optional[Dict[str, Any]]:
        candidates = [
            s
            for s in selected
            if s.get("precision") == precision and s.get("L_mode") == mode and s.get("selector_name") == selector
        ]
        order = {"step_1000": 0, "final": 1, "step_20000": 1}
        candidates.sort(key=lambda s: order.get(str(s.get("checkpoint_name")), 99))
        return candidates[0] if candidates else None

    for precision in ["fp32", "fp16"]:
        lines.append("")
        lines.append(f"{precision.upper()}:")
        any_sel = [s for s in selected if s.get("precision") == precision]
        checkpoint = any_sel[0].get("checkpoint_name") if any_sel else ""
        lines.append(f"  checkpoint: {checkpoint}")
        for label, mode, selector in [
            ("L_clean32 plateau_q90", "L_clean32", "plateau_q90_primary"),
            ("L_oracle plateau_q90", "L_oracle_precision", "plateau_q90_primary"),
            ("L_oracle oldSNR ablation", "L_oracle_oldSNR", "old_snr_max_fallback_ablation"),
        ]:
            item = sel(precision, mode, selector)
            if not item:
                lines.append(f"  {label}: unavailable")
                continue
            stability = item.get("stability_q90_2x")
            if safe_float(stability) is None:
                stability = item.get("stability_q90_next") if safe_float(item.get("stability_q90_next")) is not None else item.get("stability_q90_prev")
            flags = item.get("fallback_flags") or ""
            if str(item.get("low_h2_noise_suspected")).lower() == "true":
                flags = f"{flags};low_h2_noise" if flags else "low_h2_noise"
            lines.append(
                f"  {label}: h2={format_float(item.get('selected_h2'))}, "
                f"L={format_float(item.get('selected_L_q90'))}, "
                f"SNR2={format_float(item.get('SNR2'))}, stability={format_float(stability)}, flags={flags}"
            )
    warn = diagnostics.get("warnings") or []
    if warn:
        lines.append("")
        lines.append("Warnings:")
        for w in warn[:20]:
            lines.append(f"  - {w}")
        if len(warn) > 20:
            lines.append(f"  - ... {len(warn) - 20} more")
    return "\n".join(lines)


def parse_h2_grid(raw: str) -> Optional[List[float]]:
    raw = str(raw or "").strip()
    if not raw:
        return None
    vals: List[float] = []
    for token in raw.replace(",", " ").split():
        vals.append(float(token))
    return sorted_unique(vals)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--search-root",
        action="append",
        default=None,
        help="Root to search for run_config.json. May be repeated.",
    )
    parser.add_argument("--output-root", default=str(REPO_ROOT / "analysis"))
    parser.add_argument("--output-dir", default=None, help="Exact output directory. Refuses to overwrite existing L output files.")
    parser.add_argument("--m-L", type=int, default=32)
    parser.add_argument("--effective-directions", type=int, default=4)
    parser.add_argument("--direction-seed-base", type=int, default=16)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--checkpoint-name", action="append", default=None, help="Restrict to checkpoint name(s), e.g. step_1000.")
    parser.add_argument("--precision", action="append", choices=["fp32", "fp16"], default=None)
    parser.add_argument("--mode", action="append", choices=["L_clean32", "L_oracle_precision"], default=None)
    parser.add_argument("--h2-grid", type=parse_h2_grid, default=None, help="Override h2 grid, comma/space separated.")
    parser.add_argument("--dry-run", action="store_true", help="Only discover runs/checkpoints and write diagnostics; do not load models.")
    parser.add_argument(
        "--allow-cpu-forward",
        action="store_true",
        help="Allow full model forward probes on CPU. This is usually impractical for RoBERTa-large.",
    )
    parser.add_argument("--require-cuda", action="store_true", help="Exit nonzero instead of writing empty outputs unless CUDA is selected.")
    parser.add_argument("--require-h100", action="store_true", help="Exit nonzero unless the selected CUDA device name contains H100.")
    parser.add_argument("--fail-on-empty", action="store_true", help="Exit nonzero if no candidate or selector rows are produced.")
    parser.add_argument("--max-groups", type=int, default=0, help="Optional limit for smoke testing; 0 means no limit.")
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    if args.m_L < 1:
        raise ValueError("--m-L must be positive")
    if args.m_L < 32:
        # This is intentionally visible in diagnostics and terminal output.
        reduced_msg = f"m_L reduced from recommended 32 to {args.m_L}"
    else:
        reduced_msg = ""

    if args.output_dir:
        out_dir = Path(args.output_dir)
        existing_outputs = ["L_candidates.csv", "L_selected.csv", "L_diagnostics.json", "L_summary.md"]
        present = [name for name in existing_outputs if (out_dir / name).exists()]
        if present:
            raise FileExistsError(f"{out_dir} already contains analysis outputs: {', '.join(present)}")
        out_dir.mkdir(parents=True, exist_ok=True)
    else:
        out_dir = Path(args.output_root) / f"L_estimation_fp32_fp16_{now_stamp()}"
        out_dir.mkdir(parents=True, exist_ok=False)

    diagnostics: Dict[str, Any] = {
        "analysis_output_dir": str(out_dir),
        "git_commit": git_commit(),
        "start_time": dt.datetime.now().isoformat(),
        "warnings": [],
        "code_inspection": {
            "perturbation_application_code": "medium_models/src/trainer.py:5383 efficient_perturb_parameters; 1958 _sample_direction_and_delta; 1972 _apply_delta_list",
            "precision_handling_code": "medium_models/run.py:1421-1438 precision_mode mapping; medium_models/src/trainer.py:4236 _zo_two_point_autocast_context",
            "existing_L_estimator": "medium_models/src/trainer.py:2066 _estimate_two_point_l_raw implements shared-step L at one Delta-derived h2",
            "fp16_forward_uses": "zo_two_point_precision=fp16 maps to CUDA autocast float16; BF16 only if precision_mode=bf16",
            "parameter_restore_existing": "training probe applies additive perturbations then inverse additive perturbations; this analyzer snapshots and copies parameters back exactly after each direction",
            "eval_dropout_existing": "zo_forward and probe_window call model.eval(), disabling dropout during probes",
            "main_tex": "No main.tex/*.tex found outside generated experiment/package directories during inspection",
        },
        "compute_environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "hostname": socket.gethostname(),
        },
    }
    if reduced_msg:
        diagnostics["warnings"].append(reduced_msg)

    try:
        import torch

        diagnostics["compute_environment"].update(
            {
                "torch_version": torch.__version__,
                "cuda_available": bool(torch.cuda.is_available()),
                "cuda_device_count": int(torch.cuda.device_count()),
                "torch_cuda_version": torch.version.cuda,
            }
        )
        diagnostics["tf32_default_state"] = current_tf32_state()
        if torch.cuda.is_available():
            diagnostics["compute_environment"]["cuda_device_name_0"] = torch.cuda.get_device_name(0)
    except Exception as exc:
        diagnostics["warnings"].append(f"torch environment inspection failed: {type(exc).__name__}: {exc}")

    search_roots = [Path(p) for p in args.search_root] if args.search_root else [
        REPO_ROOT / "experiments" / "main_latest" / "mezo" / "roberta-large" / "sst5"
    ]
    diagnostics["search_roots"] = [str(p) for p in search_roots]
    runs = discover_runs(search_roots, diagnostics)
    reference_runs = choose_reference_runs(runs, diagnostics)
    checkpoints = select_checkpoints(reference_runs, diagnostics)
    if args.precision:
        allowed = set(args.precision)
        checkpoints = [c for c in checkpoints if c.precision in allowed]
    if args.checkpoint_name:
        allowed_ckpts = set(args.checkpoint_name)
        checkpoints = [c for c in checkpoints if c.checkpoint_name in allowed_ckpts]

    direction_seeds = direction_seed_list(args.direction_seed_base, args.m_L)
    diagnostics["direction_seed_base"] = int(args.direction_seed_base)
    diagnostics["direction_seeds"] = list(map(int, direction_seeds))
    diagnostics["directions_identical_across_h2"] = True
    diagnostics["direction_normalization"] = "raw Gaussian unnormalized; matches training dense direction convention"
    diagnostics["m_L"] = int(args.m_L)

    all_candidate_rows: List[Dict[str, Any]] = []
    raw_payloads: List[Dict[str, Any]] = []

    if args.dry_run:
        diagnostics["warnings"].append("dry_run requested; model forwards were not executed")
    else:
        device = get_device(args.device, diagnostics)
        if (args.require_cuda or not args.allow_cpu_forward) and not str(device).startswith("cuda"):
            return fail_analysis(
                out_dir,
                diagnostics,
                "CUDA is unavailable and --allow-cpu-forward was not set; refusing to write empty L outputs",
                2,
            )
        if args.require_h100:
            device_name = str(diagnostics.get("compute_environment", {}).get("cuda_device_name_0", ""))
            if "H100" not in device_name:
                return fail_analysis(out_dir, diagnostics, f"H100 device not detected; selected CUDA device is {device_name!r}", 2)
        if str(device) == "cpu" and not args.allow_cpu_forward:
            return fail_analysis(
                out_dir,
                diagnostics,
                "CUDA is unavailable and --allow-cpu-forward was not set; refusing to write empty L outputs",
                2,
            )
        else:
            modes = args.mode or ["L_clean32", "L_oracle_precision"]
            group_count = 0
            for checkpoint in checkpoints:
                for mode_name in modes:
                    if args.max_groups and group_count >= args.max_groups:
                        diagnostics["warnings"].append(f"max_groups={args.max_groups} reached; remaining groups skipped")
                        break
                    group_key_str = f"{checkpoint.precision}/{checkpoint.checkpoint_name}/{mode_name}"
                    try:
                        rows, raw_json = run_group(
                            checkpoint,
                            mode_name,
                            args=args,
                            direction_seeds=direction_seeds,
                            device=device,
                            diagnostics=diagnostics,
                        )
                        all_candidate_rows.extend(rows)
                        raw_payloads.append(raw_json)
                        group_count += 1
                        if mode_name == "L_oracle_precision":
                            old_rows = duplicate_old_snr_rows(rows)
                            all_candidate_rows.extend(old_rows)
                    except Exception as exc:
                        diagnostics.setdefault("group_errors", {})[group_key_str] = f"{type(exc).__name__}: {exc}"
                        diagnostics["warnings"].append(f"{group_key_str}: analysis failed: {type(exc).__name__}: {exc}")
                if args.max_groups and group_count >= args.max_groups:
                    break

    grouped: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = {}
    for row in all_candidate_rows:
        grouped.setdefault(group_key(row), []).append(row)
    selected_rows: List[Dict[str, Any]] = []
    for rows in grouped.values():
        selected_rows.extend(select_all(rows))

    if args.fail_on_empty and (not all_candidate_rows or not selected_rows):
        return fail_analysis(
            out_dir,
            diagnostics,
            "no L candidate/selector rows were produced; refusing to write empty scientific outputs",
            3,
        )

    write_csv(out_dir / "L_candidates.csv", all_candidate_rows, L_CANDIDATE_FIELDS)
    write_csv(out_dir / "L_selected.csv", selected_rows, L_SELECTED_FIELDS)
    for payload in raw_payloads:
        raw_name = "{precision}_{checkpoint_name}_{mode}.json".format(
            precision=payload["precision"],
            checkpoint_name=payload["checkpoint_name"],
            mode=payload["L_mode"],
        )
        write_json(out_dir / "raw" / raw_name, payload)
    for key, rows in grouped.items():
        plot_group(rows, [s for s in selected_rows if (s.get("precision"), s.get("checkpoint_name"), s.get("L_mode")) == key], out_dir)

    summary = make_summary(all_candidate_rows, selected_rows, diagnostics, out_dir)
    (out_dir / "L_summary.md").write_text(summary, encoding="utf-8")
    diagnostics["end_time"] = dt.datetime.now().isoformat()
    write_json(out_dir / "L_diagnostics.json", diagnostics)

    report = terminal_report(selected_rows, diagnostics, out_dir)
    print(report)
    (out_dir / "terminal_report.txt").write_text(report + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
