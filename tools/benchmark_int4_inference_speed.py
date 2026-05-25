#!/usr/bin/env python
"""Benchmark FP32/FP16 and low-bit inference paths for decoder-only LMs.

The benchmark separates three cases that are often conflated:

* fp32/fp16 dense: standard dense matmul.
* fake_int4_rtn: weights are RTN-quantized then dequantized back to fp16;
  this tests quantization error and memory after materialization, but it does
  not use INT4 kernels and should not be claimed as a speedup path.
* bnb4/gptq: real low-bit loading paths when the packages/checkpoints exist.
"""

from __future__ import annotations

import argparse
import csv
import gc
import importlib.metadata as importlib_metadata
import json
import os
import platform
import socket
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn


REPO_ROOT = Path(__file__).resolve().parents[1]
LARGE_MODELS_DIR = REPO_ROOT / "large_models"
TOOLS_DIR = REPO_ROOT / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import quantizer_robustness_int8_window as qrw  # noqa: E402


MODEL_ALIASES = {
    "mistral-7b": "mistralai/Mistral-7B-v0.1",
    "mistral-7b-v0.1": "mistralai/Mistral-7B-v0.1",
}


def package_version(name: str) -> Optional[str]:
    try:
        return importlib_metadata.version(name)
    except Exception:
        return None


def load_hf_token() -> Optional[str]:
    for env_name in ("MEZO_HF_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HF_TOKEN"):
        token = os.environ.get(env_name)
        if token:
            return token.strip()
    token_file = LARGE_MODELS_DIR / ".hf_token.local"
    if token_file.exists():
        token = token_file.read_text(encoding="utf-8").strip()
        if token:
            return token
    return None


def hf_kwargs() -> Dict[str, object]:
    token = load_hf_token()
    return {"token": token} if token else {}


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def env_summary() -> Dict[str, object]:
    out = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.executable,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "transformers": package_version("transformers"),
        "bitsandbytes": package_version("bitsandbytes"),
        "auto_gptq": package_version("auto-gptq"),
        "optimum": package_version("optimum"),
        "gptqmodel": package_version("gptqmodel"),
    }
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(torch.cuda.current_device())
        out.update({"gpu_name": props.name, "gpu_total_memory_mb": int(props.total_memory / 1024 / 1024)})
    return out


def quantize_linear_weights_fake_int4(model: nn.Module, group_size: int) -> Dict[str, object]:
    rows = []
    total_numel = 0
    total_sse = 0.0
    with torch.no_grad():
        for module_name, module in model.named_modules():
            if not isinstance(module, nn.Linear):
                continue
            weight = module.weight.data
            state, stats = qrw.compute_quantizer_state(
                f"{module_name}.weight" if module_name else "weight",
                weight,
                quantizer="rtnclip",
                bitwidth=4,
                group_size=group_size,
                activation_rms=None,
            )
            q_weight = qrw.quantize_with_state(weight, state)
            diff = q_weight.float() - weight.float()
            total_sse += float(diff.double().square().sum().detach().cpu())
            total_numel += weight.numel()
            module.weight.data.copy_(q_weight.to(dtype=weight.dtype))
            rows.append(stats)
    return {
        "quantized_linear_modules": len(rows),
        "fake_int4_recon_mse": total_sse / max(total_numel, 1),
        "fake_int4_note": "dequantized fp16/fp32 weights; no packed INT4 kernel",
    }


def build_inputs(tokenizer, model, batch_size: int, seq_len: int, device: torch.device) -> Dict[str, torch.Tensor]:
    vocab = int(getattr(model.config, "vocab_size", 32000))
    gen = torch.Generator(device="cpu").manual_seed(1234)
    input_ids = torch.randint(low=10, high=max(vocab - 1, 11), size=(batch_size, seq_len), generator=gen, dtype=torch.long)
    if tokenizer.pad_token_id is not None:
        input_ids[:, 0] = int(tokenizer.pad_token_id)
    attention_mask = torch.ones_like(input_ids)
    return {"input_ids": input_ids.to(device), "attention_mask": attention_mask.to(device)}


def synchronize() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def benchmark_model(model, tokenizer, args: argparse.Namespace, mode: str) -> Dict[str, object]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    inputs = build_inputs(tokenizer, model, int(args.batch_size), int(args.seq_len), device)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        for _ in range(int(args.warmup)):
            _ = model(**inputs, use_cache=False)
        synchronize()
        start = time.perf_counter()
        for _ in range(int(args.iters)):
            _ = model(**inputs, use_cache=False)
        synchronize()
        elapsed = time.perf_counter() - start
    ms = 1000.0 * elapsed / max(int(args.iters), 1)
    tokens = int(args.batch_size) * int(args.seq_len)
    return {
        "mode": mode,
        "batch_size": int(args.batch_size),
        "seq_len": int(args.seq_len),
        "warmup": int(args.warmup),
        "iters": int(args.iters),
        "ms_per_forward": ms,
        "tokens_per_second": tokens / (ms / 1000.0),
        "peak_gpu_mem_mb": float(torch.cuda.max_memory_allocated() / 1024 / 1024) if torch.cuda.is_available() else 0.0,
        "status": "complete",
        "error": "",
    }


def cleanup() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def load_dense(model_id: str, dtype: torch.dtype, allow_download: bool):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    auth = hf_kwargs()
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False, local_files_only=not allow_download, **auth)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        local_files_only=not allow_download,
        **auth,
    )
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    return model, tokenizer


def load_bnb4(model_id: str, allow_download: bool):
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    auth = hf_kwargs()
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False, local_files_only=not allow_download, **auth)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="fp4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=False,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=quant_config,
        device_map={"": 0} if torch.cuda.is_available() else None,
        local_files_only=not allow_download,
        **auth,
    )
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id
    return model, tokenizer


def load_gptq(model_id: str, allow_download: bool):
    from auto_gptq import AutoGPTQForCausalLM
    from transformers import AutoTokenizer

    auth = hf_kwargs()
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False, local_files_only=not allow_download, **auth)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoGPTQForCausalLM.from_quantized(
        model_id,
        device="cuda:0" if torch.cuda.is_available() else "cpu",
        use_triton=True,
        local_files_only=not allow_download,
        **auth,
    )
    return model, tokenizer


def run_mode(mode: str, args: argparse.Namespace) -> Dict[str, object]:
    model_id = MODEL_ALIASES.get(args.model_id.lower(), args.model_id)
    gptq_model_id = args.gptq_model_id or model_id
    extra: Dict[str, object] = {}
    try:
        if mode == "fp32":
            model, tokenizer = load_dense(model_id, torch.float32, args.allow_download)
        elif mode == "fp16":
            model, tokenizer = load_dense(model_id, torch.float16, args.allow_download)
        elif mode == "fake_int4_rtn":
            model, tokenizer = load_dense(model_id, torch.float16, args.allow_download)
            extra = quantize_linear_weights_fake_int4(model, int(args.group_size))
        elif mode == "bnb4":
            model, tokenizer = load_bnb4(model_id, args.allow_download)
        elif mode == "gptq":
            model, tokenizer = load_gptq(gptq_model_id, args.allow_download)
        else:
            raise ValueError(f"unknown mode={mode}")
        row = benchmark_model(model, tokenizer, args, mode)
        row.update(extra)
        del model
        cleanup()
        return row
    except Exception as exc:
        cleanup()
        return {
            "mode": mode,
            "batch_size": int(args.batch_size),
            "seq_len": int(args.seq_len),
            "warmup": int(args.warmup),
            "iters": int(args.iters),
            "ms_per_forward": None,
            "tokens_per_second": None,
            "peak_gpu_mem_mb": None,
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", default="mistral-7b")
    parser.add_argument("--gptq_model_id", default="", help="pre-quantized GPTQ checkpoint id/path")
    parser.add_argument("--modes", nargs="+", default=["fp16", "fake_int4_rtn", "bnb4"])
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--group_size", type=int, default=128)
    parser.add_argument("--allow_download", action="store_true")
    parser.add_argument("--output_dir", type=Path, default=REPO_ROOT / "outputs" / "int4_inference_speed")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "env.json", env_summary())
    rows = [run_mode(mode, args) for mode in args.modes]
    write_json(args.output_dir / "benchmark_results.json", {"rows": rows})
    write_csv(args.output_dir / "benchmark_results.csv", rows)
    for row in rows:
        print(
            f"{row['mode']}: {row['status']} ms={row.get('ms_per_forward')} "
            f"tok/s={row.get('tokens_per_second')} err={row.get('error','')}",
            flush=True,
        )
    return 0 if all(row["status"] == "complete" for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
