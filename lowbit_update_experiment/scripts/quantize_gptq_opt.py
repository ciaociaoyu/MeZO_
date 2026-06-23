#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig


def calibration_texts(dataset_name: str, split: str, n: int, seed: int) -> list[str]:
    ds = load_dataset(dataset_name, split=split)
    if "text" not in ds.column_names:
        raise ValueError(f"dataset {dataset_name} split {split} does not have a text column")
    ds = ds.shuffle(seed=seed)
    texts = []
    for row in ds:
        text = (row.get("text") or "").strip()
        if text:
            texts.append(text)
        if len(texts) >= n:
            break
    if not texts:
        raise RuntimeError("no non-empty calibration texts found")
    return texts


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="facebook/opt-1.3b")
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--group_size", type=int, default=128)
    parser.add_argument("--calib_samples", type=int, default=128)
    parser.add_argument("--dataset", default="wikitext")
    parser.add_argument("--dataset_config", default="wikitext-2-raw-v1")
    parser.add_argument("--split", default="train")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_dir", default="")
    parser.add_argument("--desc_act", action="store_true")
    args = parser.parse_args()

    out = Path(args.output_dir or f"artifacts/gptq_opt_b{args.bits}_g{args.group_size}")
    out.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset_name = args.dataset if args.dataset != "wikitext" else ("wikitext", args.dataset_config)
    if isinstance(dataset_name, tuple):
        ds = load_dataset(dataset_name[0], dataset_name[1], split=args.split)
        ds = ds.shuffle(seed=args.seed)
        texts = [r["text"].strip() for r in ds if r.get("text") and r["text"].strip()][: args.calib_samples]
    else:
        texts = calibration_texts(dataset_name, args.split, args.calib_samples, args.seed)

    (out / "calibration_texts.json").write_text(json.dumps(texts, indent=2) + "\n")
    quant_config = GPTQConfig(
        bits=args.bits,
        group_size=args.group_size,
        dataset=texts,
        tokenizer=tokenizer,
        desc_act=bool(args.desc_act),
    )
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            quantization_config=quant_config,
        )
    except Exception as exc:
        msg = {
            "status": "failed",
            "reason": "GPTQ quantization failed. Install a compatible GPTQ backend such as gptqmodel/optimum support.",
            "error": repr(exc),
            "model": args.model,
            "bits": args.bits,
            "group_size": args.group_size,
        }
        (out / "quantize_error.json").write_text(json.dumps(msg, indent=2) + "\n")
        print(json.dumps(msg, indent=2), file=sys.stderr)
        return 2

    model.save_pretrained(out, safe_serialization=True)
    tokenizer.save_pretrained(out)
    meta = {
        "status": "complete",
        "model": args.model,
        "bits": args.bits,
        "group_size": args.group_size,
        "desc_act": bool(args.desc_act),
        "calib_samples": len(texts),
        "output_dir": str(out),
        "backend_note": "Hugging Face Transformers GPTQConfig path",
    }
    (out / "quantize_metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
