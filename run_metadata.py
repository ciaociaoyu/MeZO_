import json
import os
import socket
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import torch


METADATA_SCHEMA_VERSION = 1
MODEL_RUN_METADATA_ATTR = "_run_metadata"


def _normalize_for_json(value: Any):
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _normalize_for_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_normalize_for_json(v) for v in value]
    if isinstance(value, torch.device):
        return str(value)
    if hasattr(value, "item") and callable(getattr(value, "item")):
        try:
            return _normalize_for_json(value.item())
        except Exception:
            pass
    return str(value)


def update_model_run_metadata(model, **updates) -> Dict[str, Any]:
    current = {}
    if model is not None:
        existing = getattr(model, MODEL_RUN_METADATA_ATTR, None)
        if isinstance(existing, dict):
            current.update(existing)
    current.update(updates)
    if model is not None:
        setattr(model, MODEL_RUN_METADATA_ATTR, current)
    return dict(current)


def get_model_run_metadata(model) -> Dict[str, Any]:
    existing = getattr(model, MODEL_RUN_METADATA_ATTR, None) if model is not None else None
    return dict(existing) if isinstance(existing, dict) else {}


def _safe_int(value, default: Optional[int] = None) -> Optional[int]:
    if value is None:
        return default
    try:
        return int(value)
    except Exception:
        return default


def _dtype_to_label(dtype) -> str:
    mapping = {
        torch.float32: "fp32",
        torch.float16: "fp16",
        torch.bfloat16: "bf16",
        torch.float64: "fp64",
        torch.int8: "int8",
        torch.int16: "int16",
        torch.int32: "int32",
        torch.int64: "int64",
    }
    if dtype in mapping:
        return mapping[dtype]
    return "unknown"


def _collapse_dtype_labels(labels: Iterable[str]) -> str:
    unique_labels = sorted({label for label in labels if label and label != "unknown"})
    if not unique_labels:
        return "unknown"
    if len(unique_labels) == 1:
        return unique_labels[0]
    return "mixed"


def _infer_storage_dtype_from_model(model) -> str:
    if model is None:
        return "unknown"
    labels = []
    try:
        for param in model.parameters():
            labels.append(_dtype_to_label(getattr(param, "dtype", None)))
    except Exception:
        return "unknown"
    return _collapse_dtype_labels(labels)


def _infer_compute_dtype(args, storage_dtype: str, fp8_mode: str, load_int8: bool) -> str:
    if fp8_mode != "none":
        return "mixed"
    if load_int8:
        return "mixed"
    if bool(getattr(args, "load_bfloat16", False)) or bool(getattr(args, "bf16", False)):
        return "bf16"
    if bool(getattr(args, "load_float16", False)) or bool(getattr(args, "fp16", False)):
        return "fp16"
    if bool(getattr(args, "efficient_zero_order_fp16", False)):
        return "fp16" if storage_dtype == "fp16" else "mixed"
    if str(getattr(args, "zo_two_point_precision", "")).lower() == "fp16":
        return "fp16" if storage_dtype == "fp16" else "mixed"
    return storage_dtype if storage_dtype != "unknown" else "unknown"


def _infer_load_int8(args) -> bool:
    return bool(
        getattr(args, "load_int8", False)
        or getattr(args, "load_in_8bit", False)
    )


def _infer_zo_quantization(bits_value) -> str:
    bits = _safe_int(bits_value, default=32)
    if bits == 16:
        return "fp16"
    if bits == 8:
        return "int8"
    if bits == 4:
        return "int4"
    return "none"


def _infer_sparse_ratio(args) -> float:
    try:
        return float(getattr(args, "sparse_ratio", 1.0))
    except Exception:
        return 1.0


def _infer_sparse_mask_strategy(args) -> str:
    value = getattr(args, "sparse_mask_strategy", "percentile_per_layer")
    return str(value or "percentile_per_layer")


def _infer_sparse_scope(args) -> str:
    value = getattr(args, "sparse_scope", "trainable_only")
    return str(value or "trainable_only")


def _infer_sparse_log_active_fraction(args) -> bool:
    return bool(getattr(args, "sparse_log_active_fraction", True))


def _count_linear_layers(model) -> int:
    if model is None:
        return 0
    total = 0
    for _, module in model.named_modules():
        cls_name = module.__class__.__name__
        if isinstance(module, torch.nn.Linear) or "Float8Linear" in cls_name:
            total += 1
    return total


def _infer_device_type(model=None, args=None) -> str:
    model_device = None
    if model is not None:
        try:
            model_device = next(model.parameters()).device
        except Exception:
            model_device = None
    args_device = None
    if args is not None:
        maybe_device = getattr(args, "device", None)
        if maybe_device is not None:
            try:
                args_device = torch.device(str(maybe_device))
            except Exception:
                args_device = None
    device = model_device
    if args_device is not None:
        if device is None:
            device = args_device
        elif getattr(device, "type", None) == "cpu" and getattr(args_device, "type", None) != "cpu":
            device = args_device
    if device is not None:
        if device.type == "cuda" and torch.cuda.is_available():
            index = device.index if device.index is not None else 0
            return torch.cuda.get_device_name(index)
        return device.type
    if torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _infer_model_name(args, explicit_model_name=None, model=None) -> str:
    for value in (
        explicit_model_name,
        getattr(args, "model_name", None),
        getattr(args, "model_name_or_path", None),
        getattr(getattr(model, "config", None), "_name_or_path", None),
    ):
        if value:
            return str(value)
    return "unknown"


def _infer_task_name(args, explicit_task_name=None) -> str:
    value = explicit_task_name or getattr(args, "task_name", None)
    return str(value) if value else "unknown"


def _get_git_commit(repo_root: Optional[str] = None) -> Optional[str]:
    try:
        cmd = ["git", "rev-parse", "HEAD"]
        commit = subprocess.check_output(
            cmd,
            cwd=repo_root,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return commit or None
    except Exception:
        return None


def collect_run_metadata(
    *,
    zo_method: Optional[str],
    args,
    model=None,
    output_dir: Optional[str] = None,
    model_name: Optional[str] = None,
    task_name: Optional[str] = None,
    repo_root: Optional[str] = None,
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    model_metadata = get_model_run_metadata(model)
    run_output_dir = os.path.abspath(output_dir or getattr(args, "output_dir", "."))
    load_int8 = bool(model_metadata.get("load_int8", _infer_load_int8(args)))
    fp8_mode = str(model_metadata.get("fp8_mode", "none") or "none").lower()
    storage_dtype = "mixed" if load_int8 else _infer_storage_dtype_from_model(model)
    compute_dtype = _infer_compute_dtype(args, storage_dtype, fp8_mode, load_int8)
    zo_quantization = _infer_zo_quantization(getattr(args, "zo_quantization_bits", None))
    sparse_ratio = _infer_sparse_ratio(args)
    sparse_mezo_enabled = False
    trainer_name = str(getattr(args, "trainer", "") or "").lower()
    if bool(getattr(args, "zero_order_optim", False)) or trainer_name == "zo":
        sparse_mezo_enabled = sparse_ratio < 1.0
    total_linear_layers = _safe_int(model_metadata.get("total_linear_layers"), default=None)
    if total_linear_layers is None:
        total_linear_layers = _count_linear_layers(model)
    converted_linear_layers = _safe_int(model_metadata.get("converted_linear_layers"), default=0) or 0
    skipped_layer_names = list(model_metadata.get("skipped_layer_names", []) or [])

    metadata = {
        "metadata_schema_version": METADATA_SCHEMA_VERSION,
        "zo_method": str(zo_method or getattr(args, "trainer", None) or "unknown"),
        "int8_snap_enabled": zo_quantization in {"int8", "int4"},
        "zo_quantization": zo_quantization,
        "sparse_mezo_enabled": bool(sparse_mezo_enabled),
        "sparse_ratio": float(sparse_ratio),
        "sparse_mask_strategy": _infer_sparse_mask_strategy(args),
        "sparse_scope": _infer_sparse_scope(args),
        "sparse_log_active_fraction": _infer_sparse_log_active_fraction(args),
        "storage_dtype": storage_dtype,
        "compute_dtype": compute_dtype,
        "load_int8": load_int8,
        "fp8_mode": fp8_mode if fp8_mode in {"none", "native", "emulated"} else "none",
        "fp8_native_enabled": fp8_mode == "native",
        "converted_linear_layers": int(converted_linear_layers),
        "total_linear_layers": int(total_linear_layers),
        "skipped_layer_names": [str(name) for name in skipped_layer_names],
        "device_type": _infer_device_type(model=model, args=args),
        "model_name": _infer_model_name(args, explicit_model_name=model_name, model=model),
        "task_name": _infer_task_name(args, explicit_task_name=task_name),
        "seed": _safe_int(getattr(args, "seed", None), default=None),
        "run_output_dir": run_output_dir,
        "git_commit": _get_git_commit(repo_root=repo_root),
        "hostname": socket.gethostname() or None,
    }
    if extra_metadata:
        metadata.update(extra_metadata)
    return _normalize_for_json(metadata)


def write_run_metadata(
    metadata: Dict[str, Any],
    output_dir: str,
    filename: str = "run_metadata.json",
) -> str:
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_normalize_for_json(metadata), f, ensure_ascii=False, indent=2)
    return path
