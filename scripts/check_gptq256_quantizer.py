#!/usr/bin/env python3
"""Write GPTQ-256 rerun quantizer checks.

This project does not currently implement exact GPTQ/Hessian calibration for
medium_models. The script therefore reports the actual fallback explicitly:
groupwise symmetric INT8 fake quantization with block/group size 256.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
MEDIUM_ROOT = REPO_ROOT / "medium_models"
if str(MEDIUM_ROOT) not in sys.path:
    sys.path.insert(0, str(MEDIUM_ROOT))

from src.quzo import exact_gptq_available, quantize_tensor  # noqa: E402


CSV_FIELDS = [
    "name",
    "shape",
    "numel",
    "quantization_algorithm",
    "bits",
    "group_size",
    "block_size",
    "scale_shape",
    "scale_min",
    "scale_median",
    "scale_max",
    "zero_point_mode",
    "num_groups",
    "saturation_frac",
    "error_norm",
    "relative_error_norm",
    "max_abs_error",
    "roundtrip_max_abs_error",
    "roundtrip_dequant_equal",
]


def finite_float(value: Any) -> float:
    try:
        out = float(value)
    except Exception:
        return 0.0
    return out if math.isfinite(out) else 0.0


def representative_tensors(model_name: str, max_tensors: int, device: torch.device) -> Tuple[List[Tuple[str, torch.Tensor]], str]:
    try:
        from transformers import AutoModelForMaskedLM

        model = AutoModelForMaskedLM.from_pretrained(model_name)
        model.to(device)
        rows: List[Tuple[str, torch.Tensor]] = []
        preferred = ("embeddings.word_embeddings", "attention.self.query", "attention.self.value", "intermediate.dense", "output.dense")
        for name, param in model.named_parameters():
            if param.ndim < 2:
                continue
            if preferred and not any(key in name for key in preferred):
                continue
            rows.append((name, param.detach().float().clone()))
            if len(rows) >= max_tensors:
                break
        if len(rows) < max_tensors:
            seen = {name for name, _ in rows}
            for name, param in model.named_parameters():
                if name in seen or param.ndim < 2:
                    continue
                rows.append((name, param.detach().float().clone()))
                if len(rows) >= max_tensors:
                    break
        return rows, ""
    except Exception as exc:
        tensors = [
            ("synthetic_linear_1024x1024", torch.randn(1024, 1024, device=device)),
            ("synthetic_embedding_512x768", torch.randn(512, 768, device=device) * 0.02),
            ("synthetic_classifier_5x1024", torch.randn(5, 1024, device=device) * 0.01),
        ]
        return tensors[:max_tensors], f"{type(exc).__name__}: {exc}"


def summarize_tensor(name: str, tensor: torch.Tensor, *, bits: int, group_size: int) -> Dict[str, Any]:
    q, meta = quantize_tensor(
        tensor,
        bits,
        algorithm=f"groupwise_int{bits}_block{group_size}",
        group_size=group_size,
        stochastic=False,
        target_dtype=torch.float32,
        return_metadata=True,
    )
    q2 = quantize_tensor(
        q,
        bits,
        algorithm=f"groupwise_int{bits}_block{group_size}",
        group_size=group_size,
        stochastic=False,
        target_dtype=torch.float32,
    )
    err = torch.nan_to_num(tensor.float() - q.float(), nan=0.0, posinf=0.0, neginf=0.0)
    roundtrip_err = torch.nan_to_num(q.float() - q2.float(), nan=0.0, posinf=0.0, neginf=0.0)
    err_norm = float(torch.linalg.vector_norm(err.reshape(-1)).item()) if err.numel() else 0.0
    tensor_norm = float(torch.linalg.vector_norm(tensor.float().reshape(-1)).item()) if tensor.numel() else 0.0
    max_abs = float(torch.max(torch.abs(err)).item()) if err.numel() else 0.0
    rt_max_abs = float(torch.max(torch.abs(roundtrip_err)).item()) if roundtrip_err.numel() else 0.0
    return {
        "name": name,
        "shape": "x".join(str(x) for x in tensor.shape),
        "numel": int(tensor.numel()),
        "quantization_algorithm": meta.get("quantization_algorithm"),
        "bits": int(bits),
        "group_size": int(group_size),
        "block_size": int(group_size),
        "scale_shape": json.dumps(meta.get("scale_shape", [])),
        "scale_min": meta.get("scale_min"),
        "scale_median": meta.get("scale_median"),
        "scale_max": meta.get("scale_max"),
        "zero_point_mode": meta.get("zero_point_mode"),
        "num_groups": meta.get("num_groups"),
        "saturation_frac": meta.get("saturation_frac"),
        "error_norm": err_norm,
        "relative_error_norm": err_norm / tensor_norm if tensor_norm > 0.0 else 0.0,
        "max_abs_error": max_abs,
        "roundtrip_max_abs_error": rt_max_abs,
        "roundtrip_dequant_equal": bool(rt_max_abs == 0.0),
    }


def write_csv(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in CSV_FIELDS})


def write_report(path: Path, rows: List[Dict[str, Any]], *, model_name: str, load_error: str, bits: int, group_size: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    scales = [finite_float(row.get("scale_median")) for row in rows if row.get("scale_median") not in (None, "")]
    roundtrip_ok = all(bool(row.get("roundtrip_dequant_equal")) for row in rows)
    with path.open("w", encoding="utf-8") as f:
        f.write("# GPTQ-256 Quantizer Report\n\n")
        f.write("Requested experiment label: GPTQ-256 INT8 rerun.\n\n")
        f.write(f"Exact GPTQ available in medium_models: `{exact_gptq_available()}`.\n\n")
        f.write("Actual quantizer used here: `groupwise_int8_block256` fallback, not exact GPTQ.\n\n")
        f.write(f"Bits: `{bits}`; group/block size: `{group_size}`; zero point: `none_symmetric`.\n\n")
        f.write("Calibration: no Hessian/second-order calibration is run because exact GPTQ is not implemented in this code path.\n\n")
        f.write("Activation quantization: not added.\n\n")
        f.write("Storage/forward path: fake-quantized dequantized weights are written back to model parameters for the existing QuZO forward/probe path.\n\n")
        f.write("Quantized parameters: current `quantize_model_in_place(..., include_frozen=True)` quantizes all floating model parameters in the QuZO path, including LayerNorm, bias, and classifier parameters.\n\n")
        if load_error:
            f.write(f"Model load warning: `{load_error}`. Synthetic tensors were used for this check.\n\n")
        else:
            f.write(f"Representative tensors loaded from `{model_name}`.\n\n")
        f.write(f"Roundtrip dequant equality across checked tensors: `{roundtrip_ok}`.\n\n")
        if scales:
            f.write(f"Median of per-tensor median scales: `{sum(scales) / len(scales):.6g}`.\n\n")
        f.write("| name | shape | groups | scale_min | scale_median | scale_max | rel_error | roundtrip_max_abs |\n")
        f.write("| --- | --- | --- | --- | --- | --- | --- | --- |\n")
        for row in rows:
            f.write(
                "| {name} | {shape} | {num_groups} | {scale_min:.6g} | {scale_median:.6g} | {scale_max:.6g} | {relative_error_norm:.6g} | {roundtrip_max_abs_error:.6g} |\n".format(
                    **{**row, "scale_min": finite_float(row.get("scale_min")), "scale_median": finite_float(row.get("scale_median")), "scale_max": finite_float(row.get("scale_max")), "relative_error_norm": finite_float(row.get("relative_error_norm")), "roundtrip_max_abs_error": finite_float(row.get("roundtrip_max_abs_error"))}
                )
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_name_or_path", default="roberta-large")
    parser.add_argument("--bits", type=int, default=8)
    parser.add_argument("--group_size", type=int, default=256)
    parser.add_argument("--max_tensors", type=int, default=12)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    out_dir = Path(args.output_dir).expanduser().resolve()
    device = torch.device(args.device)
    tensors, load_error = representative_tensors(args.model_name_or_path, args.max_tensors, device)
    rows = [summarize_tensor(name, tensor.to(device), bits=args.bits, group_size=args.group_size) for name, tensor in tensors]
    write_csv(out_dir / "gptq256_quantizer_stats.csv", rows)
    write_report(
        out_dir / "gptq256_quantizer_report.md",
        rows,
        model_name=args.model_name_or_path,
        load_error=load_error,
        bits=args.bits,
        group_size=args.group_size,
    )
    print(json.dumps({"rows": len(rows), "load_error": load_error, "output_dir": str(out_dir)}, sort_keys=True))


if __name__ == "__main__":
    main()
