#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lowbit_lattice.metrics import update_geometry_metrics
from lowbit_lattice.quant import GroupwiseQuantizedWeight
from lowbit_lattice.update_rules import apply_update_rule, compute_lr_for_relative_update


DEFAULT_RELATIVE_NORMS = [1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2]
DEFAULT_RULES = [
    "fp_sgd_upper_bound",
    "nearest_requant_fixed_grid",
    "stochastic_round_fixed_grid",
    "topk_code_flip",
    "topk_code_flip_plus_stochastic_tail",
    "dense_stochastic_code_flip",
]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_texts(args) -> List[str]:
    cache = Path(args.output_dir) / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / f"texts_seed{args.seed}_n{args.calib_samples + args.eval_samples + 64}.json"
    if path.exists():
        return json.loads(path.read_text())
    try:
        ds = load_dataset(args.dataset, args.dataset_config, split="train")
        ds = ds.shuffle(seed=args.seed)
        texts = [r["text"].strip() for r in ds if r.get("text") and r["text"].strip()]
    except Exception as exc:
        if not args.allow_synthetic_data:
            raise
        texts = [
            "This is a deterministic synthetic language modeling sentence used when datasets are unavailable. "
            f"sample {i}"
            for i in range(args.calib_samples + args.eval_samples + 128)
        ]
        print(f"WARNING: using synthetic text fallback because dataset load failed: {exc}", file=sys.stderr)
    need = args.calib_samples + args.eval_samples + 64
    texts = texts[:need]
    if len(texts) < max(4, args.eval_samples):
        raise RuntimeError(f"not enough texts: found {len(texts)}")
    path.write_text(json.dumps(texts, indent=2) + "\n")
    return texts


def token_batches(tokenizer, texts: List[str], seq_len: int, n_batches: int, batch_size: int, device: torch.device):
    joined = "\n\n".join(texts)
    ids = tokenizer(joined, return_tensors="pt").input_ids[0]
    if ids.numel() < seq_len + 1:
        repeat = int(math.ceil((seq_len + 1) / max(1, ids.numel()))) + 1
        ids = ids.repeat(repeat)
    batches = []
    for i in range(n_batches):
        start = (i * seq_len) % max(1, ids.numel() - seq_len - 1)
        chunk = ids[start : start + seq_len].clone()
        batches.append({"input_ids": chunk.unsqueeze(0).repeat(batch_size, 1).to(device), "labels": chunk.unsqueeze(0).repeat(batch_size, 1).to(device)})
    return batches


def load_gptq_or_regular_model(args, tokenizer, calib_texts, device):
    artifact_dir = Path(args.gptq_artifact_dir or f"artifacts/gptq_opt_b{args.bits}_g{args.group_size}")
    source = artifact_dir if artifact_dir.exists() and any(artifact_dir.glob("*.safetensors")) else args.model
    quant_config = None
    if source == args.model:
        quant_config = GPTQConfig(
            bits=args.bits,
            group_size=args.group_size,
            dataset=calib_texts,
            tokenizer=tokenizer,
            desc_act=bool(args.desc_act),
        )
    try:
        model = AutoModelForCausalLM.from_pretrained(
            source,
            torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
            device_map=None,
            quantization_config=quant_config,
        ).to(device)
        backend = "gptq_transformers" if quant_config is not None or source != args.model else "gptq_saved"
    except Exception as exc:
        if not args.allow_surrogate_without_gptq:
            raise RuntimeError(
                "GPTQ model load/quantization failed. Install gptqmodel/compatible backend or pass "
                "--allow_surrogate_without_gptq for debugging only."
            ) from exc
        print(f"WARNING: GPTQ unavailable; using regular model as surrogate initializer: {exc}", file=sys.stderr)
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
        ).to(device)
        backend = "surrogate_regular_model_not_gptq"
    model.eval()
    return model, backend


def load_trainable_model(args, device):
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
    ).to(device)
    model.eval()
    model.config.use_cache = False
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.p = 0.0
    return model


def named_linear_modules(model) -> Dict[str, nn.Linear]:
    return {name: mod for name, mod in model.named_modules() if isinstance(mod, nn.Linear)}


def target_layer_names(model, spec: str) -> List[str]:
    linear = named_linear_modules(model)
    names = list(linear)
    if spec == "last_mlp":
        candidates = [n for n in names if n.endswith(".fc1") or n.endswith(".fc2")]
        if len(candidates) >= 2:
            return candidates[-2:]
    if spec == "last_attn":
        candidates = [n for n in names if any(n.endswith(s) for s in [".q_proj", ".k_proj", ".v_proj", ".out_proj"])]
        if len(candidates) >= 4:
            return candidates[-4:]
    if spec == "last_block_all_linear":
        layer_prefixes = sorted({n.rsplit(".", 2)[0] for n in names if ".decoder.layers." in n})
        if layer_prefixes:
            return [n for n in names if n.startswith(layer_prefixes[-1] + ".")]
    if spec.startswith("named:"):
        requested = [x.strip() for x in spec[len("named:") :].split(",") if x.strip()]
        missing = [x for x in requested if x not in linear]
        if missing:
            raise ValueError(f"missing named target layers: {missing}; available examples: {names[-20:]}")
        return requested
    raise ValueError(f"could not resolve target_layers={spec}; available linear examples: {names[-20:]}")


def get_module(model, name: str) -> nn.Module:
    cur = model
    for part in name.split("."):
        if part.isdigit() and isinstance(cur, (nn.ModuleList, list, tuple)):
            cur = cur[int(part)]
        else:
            cur = getattr(cur, part)
    return cur


def copy_weight_to_model(model, layer_name: str, weight: torch.Tensor) -> None:
    mod = get_module(model, layer_name)
    with torch.no_grad():
        mod.weight.copy_(weight.to(device=mod.weight.device, dtype=mod.weight.dtype))


def freeze_except(model, layer_name: str) -> None:
    for p in model.parameters():
        p.requires_grad_(False)
    get_module(model, layer_name).weight.requires_grad_(True)


def loss_on_batches(model, batches: List[dict]) -> float:
    vals = []
    with torch.no_grad():
        for batch in batches:
            out = model(**batch)
            vals.append(float(out.loss.detach().float().item()))
    return float(np.mean(vals))


def grad_on_batch(model, batch: dict, layer_name: str) -> Tuple[float, torch.Tensor]:
    model.zero_grad(set_to_none=True)
    out = model(**batch)
    loss = out.loss
    loss.backward()
    grad = get_module(model, layer_name).weight.grad.detach().float().clone()
    return float(loss.detach().float().item()), grad


@contextmanager
def timed_cuda():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    yield lambda: (
        time.time() - t0,
        float(torch.cuda.max_memory_allocated() / (1024 * 1024)) if torch.cuda.is_available() else None,
    )


def iter_rule_configs(args):
    for rule in args.update_rules:
        if rule == "topk_code_flip":
            for k_frac in args.k_fracs:
                yield rule, {"k_frac": k_frac}
        elif rule == "topk_code_flip_plus_stochastic_tail":
            for k_frac in args.k_fracs:
                for p_tail_max in args.p_tail_maxs:
                    yield rule, {"k_frac": k_frac, "p_tail_max": p_tail_max}
        elif rule == "dense_stochastic_code_flip":
            for p_max in args.p_maxs:
                yield rule, {"p_max": p_max}
        else:
            yield rule, {}


def stable_seed_offset(*parts: object) -> int:
    payload = "|".join(str(p) for p in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % 1000003


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="facebook/opt-1.3b")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--group_size", type=int, default=128)
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--calib_samples", type=int, default=128)
    parser.add_argument("--eval_samples", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--target_layers", default="last_mlp")
    parser.add_argument("--num_steps", type=int, default=1)
    parser.add_argument("--relative_update_norms", type=float, nargs="*", default=DEFAULT_RELATIVE_NORMS)
    parser.add_argument("--lr", type=float, nargs="*", default=[])
    parser.add_argument("--update_rules", nargs="*", default=DEFAULT_RULES)
    parser.add_argument("--k_fracs", type=float, nargs="*", default=[1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2])
    parser.add_argument("--p_tail_maxs", type=float, nargs="*", default=[0.001, 0.003, 0.01, 0.03])
    parser.add_argument("--p_maxs", type=float, nargs="*", default=[0.001, 0.003, 0.01, 0.03, 0.1])
    parser.add_argument("--scale_policy", choices=["fixed", "recompute"], default="fixed")
    parser.add_argument("--backend", choices=["surrogate_explicit_lattice", "exact_gptq_packed"], default="surrogate_explicit_lattice")
    parser.add_argument("--gptq_artifact_dir", default="")
    parser.add_argument("--dataset", default="wikitext")
    parser.add_argument("--dataset_config", default="wikitext-2-raw-v1")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--desc_act", action="store_true")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--allow_surrogate_without_gptq", action="store_true")
    parser.add_argument("--allow_synthetic_data", action="store_true")
    args = parser.parse_args()

    if args.smoke and args.model == "facebook/opt-1.3b":
        args.model = "facebook/opt-125m"
        args.calib_samples = min(args.calib_samples, 16)
        args.eval_samples = min(args.eval_samples, 8)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    texts = load_texts(args)
    calib = texts[: args.calib_samples]
    train_batches = token_batches(tokenizer, texts[args.calib_samples :], args.seq_len, max(1, args.num_steps), args.batch_size, device)
    heldout_batches = token_batches(tokenizer, texts[-args.eval_samples :], args.seq_len, max(1, min(args.eval_samples, 8)), args.batch_size, device)

    quant_model, quant_backend = load_gptq_or_regular_model(args, tokenizer, calib, device)
    grad_model = load_trainable_model(args, device)
    targets = target_layer_names(grad_model, args.target_layers)
    results_path = out_dir / "results.jsonl"
    if results_path.exists():
        results_path.unlink()

    rule_configs = list(iter_rule_configs(args))
    lr_config_count = len(args.lr) + len(args.relative_update_norms)
    if args.num_steps > 1:
        if lr_config_count != 1 or len(rule_configs) != 1:
            raise ValueError(
                "--num_steps > 1 keeps a single low-bit code state across steps, so it requires exactly "
                "one lr/relative_update_norm and exactly one concrete update-rule configuration."
            )
        if rule_configs[0][0] == "fp_sgd_upper_bound":
            raise ValueError("--num_steps > 1 is restricted to low-bit commit rules; fp_sgd_upper_bound has no code state.")

    env = {
        "model": args.model,
        "bits": args.bits,
        "group_size": args.group_size,
        "backend": args.backend,
        "quant_backend": quant_backend,
        "target_layers": targets,
        "device": str(device),
        "torch_version": torch.__version__,
        "surrogate_note": "surrogate explicit lattice initialized from GPTQ-dequantized weights; not packed GPTQ code mutation",
    }
    (out_dir / "env.json").write_text(json.dumps(env, indent=2) + "\n")

    if args.backend == "exact_gptq_packed":
        skip = {"status": "skip", "reason": "exact packed GPTQ mutation not implemented; required path uses surrogate explicit lattice"}
        (out_dir / "exact_gptq_packed_status.json").write_text(json.dumps(skip, indent=2) + "\n")
        print(json.dumps(skip))

    records = []
    for layer_name in targets:
        q_mod = get_module(quant_model, layer_name)
        if not hasattr(q_mod, "weight"):
            raise RuntimeError(f"target quantized module {layer_name} does not expose a dequantized .weight")
        initial_w = q_mod.weight.detach().float().to(device)
        base_lattice = GroupwiseQuantizedWeight.from_weight(
            layer_name,
            initial_w,
            bits=args.bits,
            group_size=args.group_size,
            scale_policy=args.scale_policy,
        ).to(device)
        persistent_lattice = base_lattice.clone().to(device)

        for step in range(args.num_steps):
            train_batch = train_batches[step % len(train_batches)]
            lattice = persistent_lattice if args.num_steps > 1 else base_lattice.clone().to(device)
            w0 = lattice.dequantize(dtype=get_module(grad_model, layer_name).weight.dtype)
            copy_weight_to_model(grad_model, layer_name, w0)
            freeze_except(grad_model, layer_name)
            train_before, grad = grad_on_batch(grad_model, train_batch, layer_name)
            heldout_before = loss_on_batches(grad_model, heldout_batches)

            lrs = list(args.lr)
            lr_labels = []
            for r in args.relative_update_norms:
                lrs.append(compute_lr_for_relative_update(lattice.dequantize(), grad, r))
                lr_labels.append(r)
            if args.lr:
                lr_labels = [None] * len(args.lr) + lr_labels

            for idx, lr in enumerate(lrs):
                rel = lr_labels[idx] if idx < len(lr_labels) else None
                for rule, rule_kwargs in rule_configs:
                    gen = torch.Generator(device=device)
                    gen.manual_seed(args.seed + step * 1000003 + idx * 9176 + stable_seed_offset(layer_name, rule, rule_kwargs))
                    with timed_cuda() as elapsed:
                        result = apply_update_rule(lattice, grad, lr=lr, rule=rule, generator=gen, **rule_kwargs)
                        copy_weight_to_model(grad_model, layer_name, result.w_new.to(dtype=get_module(grad_model, layer_name).weight.dtype))
                        train_after = float(grad_model(**train_batch).loss.detach().float().item())
                        heldout_after = loss_on_batches(grad_model, heldout_batches)
                        wall, mem = elapsed()
                    geom = update_geometry_metrics(
                        grad=grad,
                        intended_update=result.intended_update,
                        actual_update=result.w_new - result.w_old,
                        q_old=result.q_old,
                        q_new=result.q_new,
                        qmin=lattice.qmin,
                        qmax=lattice.qmax,
                        train_loss_before=train_before,
                        train_loss_after=train_after,
                        heldout_loss_before=heldout_before,
                        heldout_loss_after=heldout_after,
                    )
                    record = {
                        **env,
                        "target_layer_name": layer_name,
                        "update_rule": rule,
                        "relative_update_norm_target": rel,
                        "lr": float(lr),
                        "step": step,
                        "scale_policy": args.scale_policy,
                        "wall_clock_time_sec": wall,
                        "peak_gpu_memory_mb": mem,
                        **rule_kwargs,
                        **geom,
                    }
                    records.append(record)
                    with results_path.open("a", encoding="utf-8") as f:
                        f.write(json.dumps(record, sort_keys=True, default=str) + "\n")
                    if args.num_steps > 1:
                        if result.lattice is None:
                            raise RuntimeError(f"update rule {rule} did not return a low-bit lattice for multi-step state")
                        persistent_lattice = result.lattice.detach().clone().to(device) if hasattr(result.lattice, "detach") else result.lattice.clone().to(device)

    print(f"wrote {len(records)} records to {results_path}")
    try:
        from scripts.summarize_results import summarize

        summarize(out_dir)
    except Exception as exc:
        print(f"WARNING: summarize failed: {exc}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
