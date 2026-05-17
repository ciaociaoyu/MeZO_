#!/usr/bin/env python3
"""Write groupwise INT8 block-256 quantizer checks.

This script intentionally does not call the quantizer GPTQ. The medium_models
path currently uses group-wise symmetric INT8 fake quantization with
block/group size 256 and no Hessian calibration.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.check_gptq256_quantizer import (  # noqa: E402
    finite_float,
    representative_tensors,
    summarize_tensor,
    write_csv,
)

MEDIUM_ROOT = REPO_ROOT / "medium_models"
if str(MEDIUM_ROOT) not in sys.path:
    sys.path.insert(0, str(MEDIUM_ROOT))

from src.quzo import exact_gptq_available  # noqa: E402


def write_report(path: Path, rows: List[Dict[str, Any]], *, model_name: str, load_error: str, bits: int, group_size: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    roundtrip_ok = all(bool(row.get("roundtrip_dequant_equal")) for row in rows)
    with path.open("w", encoding="utf-8") as f:
        f.write("# groupwise_int8_block256 Quantizer Report\n\n")
        f.write("This is NOT exact GPTQ.\n\n")
        f.write("Actual quantizer used: `groupwise_int8_block256`.\n\n")
        f.write(f"Exact GPTQ available in medium_models: `{exact_gptq_available()}`.\n\n")
        f.write(f"Bits: `{bits}`.\n\n")
        f.write(f"Group size: `{group_size}`.\n\n")
        f.write(f"Block size: `{group_size}`.\n\n")
        f.write("Zero-point mode: `none_symmetric`.\n\n")
        f.write("Symmetric/asymmetric: `symmetric`.\n\n")
        f.write("Calibration samples: `0` because no GPTQ/Hessian calibration is implemented in this path.\n\n")
        f.write("exact_gptq_available = false\n\n")
        f.write("Activation quantization: not added.\n\n")
        f.write("Storage/forward path: fake-quantized dequantized weights are written back to model parameters for the existing QuZO forward/probe path.\n\n")
        f.write("Quantized parameters: current `quantize_model_in_place(..., include_frozen=True)` quantizes all floating model parameters in the QuZO path, including LayerNorm, bias, and classifier parameters.\n\n")
        f.write("Parameters left high precision: non-floating parameters only in this code path.\n\n")
        if load_error:
            f.write(f"Model load warning: `{load_error}`. Synthetic tensors were used for this check.\n\n")
        else:
            f.write(f"Representative tensors loaded from `{model_name}`.\n\n")
        f.write(f"Roundtrip dequant equality across checked tensors: `{roundtrip_ok}`.\n\n")
        f.write("| name | shape | groups | scale_shape | scale_min | scale_median | scale_max | rel_error | roundtrip_max_abs |\n")
        f.write("| --- | --- | --- | --- | --- | --- | --- | --- | --- |\n")
        for row in rows:
            f.write(
                "| {name} | {shape} | {num_groups} | {scale_shape} | {scale_min:.6g} | {scale_median:.6g} | {scale_max:.6g} | {relative_error_norm:.6g} | {roundtrip_max_abs_error:.6g} |\n".format(
                    **{
                        **row,
                        "scale_min": finite_float(row.get("scale_min")),
                        "scale_median": finite_float(row.get("scale_median")),
                        "scale_max": finite_float(row.get("scale_max")),
                        "relative_error_norm": finite_float(row.get("relative_error_norm")),
                        "roundtrip_max_abs_error": finite_float(row.get("roundtrip_max_abs_error")),
                    }
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
    write_csv(out_dir / "groupwise_int8_block256_scale_stats.csv", rows)
    write_report(
        out_dir / "groupwise_int8_block256_quantizer_report.md",
        rows,
        model_name=args.model_name_or_path,
        load_error=load_error,
        bits=args.bits,
        group_size=args.group_size,
    )
    print(json.dumps({"actual_quantizer": "groupwise_int8_block256", "exact_gptq_available": False, "rows": len(rows), "load_error": load_error, "output_dir": str(out_dir)}, sort_keys=True))


if __name__ == "__main__":
    main()

