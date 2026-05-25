#!/usr/bin/env python
"""Mistral-7B / SST-5 INT8 fixed-grid quantized-forward MeZO runner.

This is intentionally isolated from ``large_models/trainer.py``.  The legacy
large-model QuZO path recomputes a tensor scale after each perturb/update.  This
runner tests the fixed-grid semantics requested for INT8 perturbation + INT8
forward:

    d_h = [L(Q_0(w_t + h u)) - L(Q_0(w_t - h u))] / (2 h)

where Q_0 uses a quantization grid computed once from the unperturbed FP16
master weights.  The plus/minus branches share that grid and fresh-round their
own integer codes.
"""

from __future__ import annotations

import argparse
import csv
import importlib.metadata as importlib_metadata
import json
import math
import os
import platform
import random
import shutil
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler


REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = REPO_ROOT / "tools"
LARGE_MODELS_DIR = REPO_ROOT / "large_models"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))
if str(LARGE_MODELS_DIR) not in sys.path:
    sys.path.insert(0, str(LARGE_MODELS_DIR))

import quantizer_robustness_int8_window as qrw  # noqa: E402
from tasks import get_task  # noqa: E402
from utils import encode_prompt  # noqa: E402


MODEL_ALIASES = {
    "mistral-7b": "mistralai/Mistral-7B-v0.1",
    "mistral-7b-v0.1": "mistralai/Mistral-7B-v0.1",
}
RTN_ALPHA_GRID = (1.0,)
RTNCLIP_ALPHA_GRID = qrw.RTN_ALPHA_GRID


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def append_jsonl(path: Path, row: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, sort_keys=True, default=str) + "\n")


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    except Exception:
        return ""


def package_version(name: str) -> Optional[str]:
    try:
        return importlib_metadata.version(name)
    except Exception:
        return None


def collect_env() -> Dict[str, object]:
    env = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "hostname": socket.gethostname(),
        "python_executable": sys.executable,
        "python_version": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "git_commit": git_commit(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "conda_default_env": os.environ.get("CONDA_DEFAULT_ENV", ""),
        "DATALOADER_SHUFFLE": os.environ.get("DATALOADER_SHUFFLE", ""),
        "transformers_version": package_version("transformers"),
        "datasets_version": package_version("datasets"),
        "accelerate_version": package_version("accelerate"),
        "auto_gptq_version": package_version("auto-gptq"),
        "gptqmodel_version": package_version("gptqmodel"),
        "optimum_version": package_version("optimum"),
        "bitsandbytes_version": package_version("bitsandbytes"),
    }
    if torch.cuda.is_available():
        idx = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(idx)
        env.update(
            {
                "gpu_index": idx,
                "gpu_name": props.name,
                "gpu_total_memory_mb": int(props.total_memory / 1024 / 1024),
                "gpu_count": torch.cuda.device_count(),
            }
        )
    return env


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


def hf_kwargs() -> Dict[str, str]:
    token = load_hf_token()
    return {"token": token} if token else {}


def h_label(h: float) -> str:
    return f"{h:g}".replace(".", "p").replace("-", "m")


def cycle(loader: Iterable[Dict[str, torch.Tensor]]) -> Iterator[Dict[str, torch.Tensor]]:
    while True:
        for batch in loader:
            yield batch


def move_batch(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}


class SST5CausalDataset(torch.utils.data.Dataset):
    def __init__(self, samples, task, tokenizer, max_length: int) -> None:
        self.samples = samples
        self.task = task
        self.template = task.get_template()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        sample = self.samples[idx]
        encoded_candidates, option_lens = encode_prompt(
            self.task,
            self.template,
            [],
            sample,
            self.tokenizer,
            max_length=self.max_length,
            generation=self.task.generation,
            generation_with_gold=True,
        )
        correct = sample.candidates.index(sample.correct_candidate)
        return {
            "input_ids": encoded_candidates,
            "option_lens": option_lens,
            "label": correct,
            "num_options": len(sample.candidates),
        }


class CausalSST5Collator:
    def __init__(self, tokenizer) -> None:
        self.tokenizer = tokenizer

    def __call__(self, features: List[Dict[str, object]]) -> Dict[str, torch.Tensor]:
        pad_id = int(self.tokenizer.pad_token_id)
        flat_ids: List[List[int]] = []
        option_lens: List[int] = []
        labels: List[int] = []
        num_options: List[int] = []
        for feat in features:
            candidates = feat["input_ids"]
            opts = feat["option_lens"]
            label = int(feat["label"])
            nopt = int(feat["num_options"])
            for ids, olen in zip(candidates, opts):
                flat_ids.append(list(ids))
                option_lens.append(int(olen))
                labels.append(label)
                num_options.append(nopt)
        max_len = max(len(ids) for ids in flat_ids)
        padded = []
        attention = []
        for ids in flat_ids:
            pad = [pad_id] * (max_len - len(ids))
            padded.append(pad + ids)
            attention.append([0] * len(pad) + [1] * len(ids))
        return {
            "input_ids": torch.tensor(padded, dtype=torch.long),
            "attention_mask": torch.tensor(attention, dtype=torch.long),
            "option_len": torch.tensor(option_lens, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "num_options": torch.tensor(num_options, dtype=torch.long),
            "example_count": torch.tensor(len(features), dtype=torch.long),
        }


def linear_weight_names(model: nn.Module) -> List[str]:
    params = dict(model.named_parameters())
    names: List[str] = []
    for module_name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            pname = f"{module_name}.weight" if module_name else "weight"
            if pname in params:
                names.append(pname)
    return names


def compute_rtn_state(name: str, weight: torch.Tensor, group_size: int, bitwidth: int) -> Tuple[qrw.QuantizerState, Dict[str, object]]:
    qmax = 127 if bitwidth == 8 else 7
    groups, lengths, valid = qrw.group_view_2d(weight, group_size)
    masked_abs = groups.float().abs().masked_fill(~valid, 0.0)
    max_abs = masked_abs.amax(dim=-1, keepdim=True)
    scales = (max_abs / float(qmax)).clamp_min(1e-12)
    scales = torch.where(max_abs <= 0, torch.ones_like(scales), scales)
    alpha_values = torch.ones(scales.shape[:-1], device=weight.device, dtype=torch.float32)
    state = qrw.QuantizerState(
        name=name,
        quantizer="rtn",
        shape=tuple(weight.shape),
        group_size=group_size,
        bitwidth=bitwidth,
        qmax=qmax,
        scales=scales.detach(),
        alpha_idx=torch.zeros_like(alpha_values, dtype=torch.long).detach(),
        alpha_values=alpha_values.detach(),
        lengths=lengths.detach(),
        valid=valid.detach(),
    )
    q_w, stats = qrw.quantize_with_state(weight, state, return_stats=True)
    diff = q_w.float() - weight.float()
    err_sq = diff.double().square().sum()
    ref_sq = weight.float().double().square().sum()
    eps = torch.tensor(1e-30, device=weight.device, dtype=torch.float64)
    stats.update(
        {
            "module_name": name,
            "quantizer": "rtn",
            "quantizer_backend": "G128_RTN_fixed_initial_grid_fake_quant",
            "bitwidth": bitwidth,
            "group_size": group_size,
            "num_groups": int(scales.numel()),
            "scale_min": float(scales.min().detach().cpu()),
            "scale_median": float(scales.median().detach().cpu()),
            "scale_max": float(scales.max().detach().cpu()),
            "alpha_mean": 1.0,
            "alpha_min": 1.0,
            "alpha_max": 1.0,
            "alpha_lt_1_frac": 0.0,
            "recon_mse": float((err_sq / max(int(weight.numel()), 1)).detach().cpu()),
            "weight_recon_mse": float((err_sq / max(int(weight.numel()), 1)).detach().cpu()),
            "weight_recon_rel_mse": float((err_sq / ref_sq.clamp_min(eps)).detach().cpu()),
            "weight_recon_sqnr_db": float((10.0 * torch.log10(ref_sq / err_sq.clamp_min(eps))).detach().cpu()),
            "weight_recon_sse": float(err_sq.detach().cpu()),
            "weight_recon_ref_sse": float(ref_sq.detach().cpu()),
            "activation_weighted_mse": None,
            "activation_rms_available": False,
            "alpha_1_count": int(scales.numel()),
        }
    )
    return state, stats


def compute_states(
    master: Dict[str, torch.Tensor],
    names: Sequence[str],
    quantizer: str,
    bitwidth: int,
    group_size: int,
) -> Tuple[Dict[str, qrw.QuantizerState], List[Dict[str, object]]]:
    states: Dict[str, qrw.QuantizerState] = {}
    rows: List[Dict[str, object]] = []
    for name in names:
        if quantizer == "rtn":
            state, stats = compute_rtn_state(name, master[name], group_size, bitwidth)
        elif quantizer == "rtnclip":
            state, stats = qrw.compute_quantizer_state(
                name,
                master[name],
                quantizer="rtnclip",
                bitwidth=bitwidth,
                group_size=group_size,
                activation_rms=None,
            )
            stats["quantizer_backend"] = "G128_RTNClip_fixed_initial_grid_fake_quant"
        else:
            raise ValueError(f"unsupported training quantizer={quantizer}")
        states[name] = state
        rows.append(stats)
    return states, rows


def aggregate_stats(rows: List[Dict[str, object]], numel_by_name: Dict[str, int]) -> Dict[str, object]:
    return qrw.aggregate_quantizer_stats(rows, numel_by_name)


@dataclass
class Harness:
    model: nn.Module
    tokenizer: object
    train_loader: DataLoader
    dev_loader: DataLoader
    train_sampler_name: str
    quantized_module_names: List[str]
    device: torch.device

    def params(self) -> Dict[str, nn.Parameter]:
        return dict(self.model.named_parameters())

    def make_master(self) -> Dict[str, torch.Tensor]:
        params = self.params()
        return {
            name: params[name].detach().clone().to(device=self.device, dtype=torch.float16)
            for name in self.quantized_module_names
        }

    def forward_loss_scores(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.inference_mode():
            outputs = self.model(input_ids=batch["input_ids"], attention_mask=batch.get("attention_mask"))
            logits = outputs.logits
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = batch["input_ids"][..., 1:].contiguous()
            pad_id = int(self.model.config.pad_token_id)
            shift_labels = shift_labels.masked_fill(shift_labels == pad_id, -100)
            option_len = batch["option_len"].to(shift_labels.device)
            for idx, olen in enumerate(option_len.tolist()):
                if olen <= 0:
                    shift_labels[idx, :] = -100
                else:
                    shift_labels[idx, :-int(olen)] = -100
            log_probs = F.log_softmax(shift_logits, dim=-1)
            gather_labels = shift_labels.masked_fill(shift_labels == -100, 0)
            selected = torch.gather(log_probs, dim=-1, index=gather_labels.unsqueeze(-1)).squeeze(-1)
            mask = shift_labels != -100
            candidate_scores = (selected * mask).sum(dim=-1) / mask.sum(dim=-1).clamp_min(1)
            nopt = int(batch["num_options"][0].item())
            scores = candidate_scores.view(-1, nopt)
            labels = batch["labels"].view(-1, nopt)[:, 0]
            loss = F.cross_entropy(scores, labels)
        return loss, scores


def load_harness(args: argparse.Namespace, device: torch.device) -> Harness:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_id = MODEL_ALIASES.get(args.model_id.lower(), args.model_id)
    auth = hf_kwargs()
    task = get_task("SST5")
    train_sets = task.sample_train_sets(
        num_train=-1,
        num_dev=0,
        num_eval=None,
        num_train_sets=1,
        seed=int(args.data_seed),
        dataset_mode=args.dataset_mode,
        num_k=int(args.num_k),
    )
    train_samples = train_sets[0]
    eval_samples = task.valid_samples
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False, **auth)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        **auth,
    )
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id
    if getattr(model, "generation_config", None) is not None and model.generation_config.pad_token_id is None:
        model.generation_config.pad_token_id = tokenizer.pad_token_id
    model.to(device)
    model.eval()
    train_dataset = SST5CausalDataset(train_samples, task, tokenizer, int(args.max_length))
    dev_dataset = SST5CausalDataset(eval_samples, task, tokenizer, int(args.max_length))
    train_gen = torch.Generator().manual_seed(int(args.data_seed))
    train_sampler = RandomSampler(train_dataset, generator=train_gen)
    collator = CausalSST5Collator(tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=int(args.batch_size), sampler=train_sampler, collate_fn=collator)
    dev_loader = DataLoader(dev_dataset, batch_size=int(args.eval_batch_size), sampler=SequentialSampler(dev_dataset), collate_fn=collator)
    q_names = linear_weight_names(model)
    if int(args.max_quantized_modules) > 0:
        q_names = q_names[: int(args.max_quantized_modules)]
    return Harness(
        model=model,
        tokenizer=tokenizer,
        train_loader=train_loader,
        dev_loader=dev_loader,
        train_sampler_name=type(train_sampler).__name__,
        quantized_module_names=q_names,
        device=device,
    )


def sample_directions(master: Dict[str, torch.Tensor], seed: int) -> Dict[str, torch.Tensor]:
    first = next(iter(master.values()))
    gen = torch.Generator(device=first.device).manual_seed(int(seed))
    return {name: torch.randn(t.shape, device=t.device, dtype=torch.float16, generator=gen) for name, t in master.items()}


def direction_seed(base_seed: int, h: float, step: int) -> int:
    return int(base_seed) + int(round(float(h) * 1_000_000_000_000)) + step * 1_000_003 + 8108


def copy_master_to_model(
    params: Dict[str, nn.Parameter],
    master: Dict[str, torch.Tensor],
    directions: Optional[Dict[str, torch.Tensor]],
    h: float,
    sign: float,
    states: Dict[str, qrw.QuantizerState],
) -> None:
    with torch.no_grad():
        for name, tensor in master.items():
            value = tensor if directions is None else tensor.float().add(directions[name].float(), alpha=sign * float(h))
            value = qrw.quantize_with_state(value, states[name])
            params[name].copy_(value.to(dtype=params[name].dtype))


def restore_master(params: Dict[str, nn.Parameter], master: Dict[str, torch.Tensor]) -> float:
    max_diff = 0.0
    with torch.no_grad():
        for name, tensor in master.items():
            params[name].copy_(tensor.to(dtype=params[name].dtype))
            diff = (params[name].detach().float() - tensor.float()).abs().max()
            max_diff = max(max_diff, float(diff.detach().cpu()))
    return max_diff


def update_master(master: Dict[str, torch.Tensor], directions: Dict[str, torch.Tensor], lr: float, d_h: float) -> float:
    sq = torch.zeros((), device=next(iter(master.values())).device, dtype=torch.float64)
    with torch.no_grad():
        for name, tensor in master.items():
            update = directions[name].float().mul(-float(lr) * float(d_h))
            sq += update.double().square().sum()
            tensor.copy_(tensor.float().add(update).to(dtype=tensor.dtype))
    return float(sq.sqrt().detach().cpu())


def perturbation_metrics(
    master: Dict[str, torch.Tensor],
    directions: Dict[str, torch.Tensor],
    states: Dict[str, qrw.QuantizerState],
    h: float,
) -> Dict[str, object]:
    return qrw.perturbation_metrics(master, directions, states, h)


def eval_quantized(harness: Harness, master: Dict[str, torch.Tensor], states: Dict[str, qrw.QuantizerState], max_batches: int) -> Tuple[Optional[float], Optional[float]]:
    if max_batches == 0:
        return None, None
    params = harness.params()
    copy_master_to_model(params, master, None, 0.0, 0.0, states)
    total_loss = 0.0
    total_correct = 0
    total_items = 0
    for idx, batch in enumerate(harness.dev_loader):
        if max_batches > 0 and idx >= max_batches:
            break
        batch = move_batch(batch, harness.device)
        loss, scores = harness.forward_loss_scores(batch)
        nopt = int(batch["num_options"][0].item())
        labels = batch["labels"].view(-1, nopt)[:, 0]
        total_loss += float(loss.detach().cpu()) * int(labels.numel())
        total_correct += int((scores.argmax(dim=-1) == labels).sum().detach().cpu())
        total_items += int(labels.numel())
    restore_master(params, master)
    if total_items == 0:
        return None, None
    return total_loss / total_items, total_correct / total_items


def save_checkpoint(run_dir: Path, step: int, master: Dict[str, torch.Tensor], best: Dict[str, object], config: Dict[str, object]) -> None:
    latest = run_dir / "checkpoints" / "latest"
    tmp = run_dir / "checkpoints" / "latest.tmp"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True, exist_ok=True)
    cpu_master = {name: value.detach().cpu().to(dtype=torch.float16) for name, value in master.items()}
    torch.save({"step": int(step), "master": cpu_master, "best": best, "config": config}, tmp / "state.pt")
    write_json(tmp / "checkpoint_manifest.json", {"step": int(step), "num_tensors": len(cpu_master), "created_at": datetime.now().isoformat()})
    if latest.exists():
        shutil.rmtree(latest)
    tmp.rename(latest)


def write_resume_command(run_dir: Path, args: argparse.Namespace, step_path: Path) -> None:
    cmd = (
        "CUDA_VISIBLE_DEVICES=0 DATALOADER_SHUFFLE=True "
        f"{sys.executable} tools/mistral_int8_fixedgrid_sst5.py "
        f"--output_root {args.output_root} --run_name {args.run_name} --model_id {args.model_id} "
        f"--quantizer {args.quantizer} --steps {args.steps} --h {args.h} --lr {args.lr} "
        f"--batch_size {args.batch_size} --eval_batch_size {args.eval_batch_size} --max_length {args.max_length} "
        f"--resume_from {step_path}"
    )
    (run_dir / "resume_command.txt").write_text(cmd + "\n", encoding="utf-8")


def check_invariants(config: Dict[str, object]) -> List[str]:
    failures = []
    if int(config["quant_bits"]) != 8:
        failures.append("quant_bits must be 8")
    if config["dataset"] != "SST-5":
        failures.append("dataset must be SST-5")
    if config["model"] != "mistralai/Mistral-7B-v0.1":
        failures.append("model must be mistralai/Mistral-7B-v0.1")
    if config["sampler_name"] != "RandomSampler":
        failures.append(f"sampler must be RandomSampler, got {config['sampler_name']}")
    if not config["pair_shared_grid"]:
        failures.append("pair_shared_grid must be true")
    if not config["fresh_round_codes"]:
        failures.append("fresh_round_codes must be true")
    if config["scale_refresh_policy"] != "initial_only_fixed_grid":
        failures.append("scale_refresh_policy must be initial_only_fixed_grid")
    if config["independent_q_plus_q_minus_scales"]:
        failures.append("plus/minus independent scale recomputation is forbidden")
    if config["q_w_plus_hu_bypass"]:
        failures.append("Q(w_t)+/-h*u bypass is forbidden")
    return failures


def write_preflight_report(output_root: Path, env: Dict[str, object], config: Dict[str, object], failures: Sequence[str]) -> None:
    audit = {
        "status": "PASS" if not failures else "FAIL",
        "failures": list(failures),
        "legacy_large_models_quzo_issue": {
            "files": ["large_models/quzo.py", "large_models/trainer.py"],
            "issue": "legacy quantize_tensor recomputes max_abs/scale from the current tensor during perturb/update",
            "decision": "do not use legacy zo_quantization_bits path for this fixed-grid Mistral experiment",
        },
        "fixed_grid_runner": "tools/mistral_int8_fixedgrid_sst5.py",
        "fixed_grid_semantics": {
            "scale_source": config.get("scale_source"),
            "grid_source": config.get("grid_source"),
            "pair_shared_grid": config.get("pair_shared_grid"),
            "fresh_round_codes": config.get("fresh_round_codes"),
            "scale_refresh_policy": config.get("scale_refresh_policy"),
        },
        "gptq": {
            "auto_gptq_version": env.get("auto_gptq_version"),
            "gptqmodel_version": env.get("gptqmodel_version"),
            "status": "auto-gptq available" if env.get("auto_gptq_version") else "missing",
            "training_note": "Exact GPTQ packed-kernel branch training is not used here; RTN fixed-grid is launched. GPTQ scale extraction for Q(w±hu) should be implemented separately before training.",
        },
        "residual_design_note": {
            "direct_int8_lattice_update": "Use fixed per-group scales and maintain residual r per quantized Linear weight. Commit k=round((r+delta)/scale), clamp to code range, keep r=(r+delta)-k*scale.",
            "blockwise_storage": "For Mistral, residual buffers should be saved per decoder block to avoid all-buffer checkpoint pressure.",
            "this_run": "FP16 master update with INT8 fixed-grid forward/probe, not residual-grid commit.",
        },
        "env": env,
        "config": config,
    }
    write_json(output_root / "preflight_audit.json", audit)
    lines = [
        "# Mistral INT8 Fixed-Grid Preflight Audit",
        "",
        f"Status: **{audit['status']}**",
        "",
        "## Key Logic",
        "",
        "- Old `large_models/quzo.py` / `large_models/trainer.py` is not used for this run because it recomputes scale from the current perturbed/updated tensor.",
        "- This runner computes the INT8 grid once from unperturbed FP16 master Linear weights.",
        "- `Q_0(w_t + h u)` and `Q_0(w_t - h u)` share the same scale/grid and separately fresh-round integer codes.",
        "- FP16 master Linear weights are updated; packed/direct INT8 lattice update is not used in this job.",
        "",
        "## GPTQ",
        "",
        f"- auto-gptq: `{env.get('auto_gptq_version')}`",
        f"- gptqmodel: `{env.get('gptqmodel_version')}`",
        "- Exact GPTQ branch training is not launched unless a separate implementation extracts fixed GPTQ group scales for `Q(w±hu)`.",
        "",
        "## Residual Block Design",
        "",
        "- Direct INT8 residual update should keep fixed per-group scales and residual buffers per Linear weight or per decoder block.",
        "- Commit rule: `k=round((r + delta)/scale)`, clamp legal code step/range, then retain `r=(r+delta)-k*scale`.",
        "- Store residual state per Mistral decoder block to keep checkpoint IO tractable.",
    ]
    if failures:
        lines.extend(["", "## Failures", ""] + [f"- {x}" for x in failures])
    (output_root / "preflight_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_training(args: argparse.Namespace) -> int:
    if args.quantizer == "gptq":
        env = collect_env()
        args.output_root.mkdir(parents=True, exist_ok=True)
        write_json(args.output_root / "env.json", env)
        write_json(
            args.output_root / "gptq_status.json",
            {
                "status": "installed_check_only",
                "auto_gptq_version": env.get("auto_gptq_version"),
                "gptqmodel_version": env.get("gptqmodel_version"),
                "note": "Exact GPTQ packed modules are available through AutoGPTQ, but Q(w_t +/- h u) shared-grid branch training is not implemented in this runner.",
            },
        )
        print("GPTQ package check complete; exact GPTQ branch training is not implemented in this runner.", flush=True)
        return 0
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Mistral-7B fixed-grid smoke/training")
    random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    torch.cuda.manual_seed_all(int(args.seed))
    device = torch.device("cuda")
    output_root = args.output_root
    run_name = args.run_name or f"mistral7b_sst5_int8_fixedgrid_{args.quantizer}_h{h_label(float(args.h))}_steps{int(args.steps)}"
    run_dir = output_root / "runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    env = collect_env()
    write_json(output_root / "env.json", env)

    print(f"[{datetime.now().isoformat(timespec='seconds')}] loading Mistral/SST-5", flush=True)
    harness = load_harness(args, device)
    params = harness.params()
    master = harness.make_master()
    numel_by_name = {name: params[name].numel() for name in harness.quantized_module_names}
    print(f"[{datetime.now().isoformat(timespec='seconds')}] computing fixed {args.quantizer} INT8 grids for {len(master)} Linear weights", flush=True)
    states, q_rows = compute_states(master, harness.quantized_module_names, args.quantizer, 8, int(args.group_size))
    quant_stats = aggregate_stats(q_rows, numel_by_name)
    append_jsonl(run_dir / "quantizer_diagnostics.jsonl", {"step": 0, "record_type": "initial_fixed_grid", **quant_stats})

    config = {
        "run_name": run_name,
        "run_dir": str(run_dir),
        "model": MODEL_ALIASES.get(args.model_id.lower(), args.model_id),
        "dataset": "SST-5",
        "dataset_mode": args.dataset_mode,
        "seed": int(args.seed),
        "data_seed": int(args.data_seed),
        "num_k": int(args.num_k),
        "batch_size": int(args.batch_size),
        "eval_batch_size": int(args.eval_batch_size),
        "sampler_name": harness.train_sampler_name,
        "direction": "dense",
        "perturb_scope": "mistral_linear_weight_only",
        "quantized_modules": harness.quantized_module_names,
        "quantized_module_count": len(harness.quantized_module_names),
        "quant_bits": 8,
        "group_size": int(args.group_size),
        "quantizer": args.quantizer,
        "quantizer_backend": "G128_RTN_fixed_initial_grid_fake_quant" if args.quantizer == "rtn" else "G128_RTNClip_fixed_initial_grid_fake_quant",
        "scale_source": "initial_unperturbed_fp16_master_linear_weight",
        "grid_source": "initial_unperturbed_fp16_master_linear_weight",
        "scale_refresh_policy": "initial_only_fixed_grid",
        "pair_shared_grid": True,
        "fresh_round_codes": True,
        "independent_q_plus_q_minus_scales": False,
        "q_w_plus_hu_bypass": False,
        "update_backend": "fp16_master",
        "direct_int_update": False,
        "residual_grid_update": False,
        "h": float(args.h),
        "lr": float(args.lr),
        "max_steps": int(args.steps),
        "eval_every": int(args.eval_every),
        "checkpoint_every": int(args.checkpoint_every),
        "env": env,
        "initial_quantizer_stats": quant_stats,
    }
    failures = check_invariants(config)
    write_json(run_dir / "run_config.json", config)
    write_json(run_dir / "run_manifest_row.json", {k: v for k, v in config.items() if k not in {"quantized_modules", "env", "initial_quantizer_stats"}})
    write_preflight_report(output_root, env, config, failures)
    write_resume_command(run_dir, args, run_dir / "checkpoints" / "latest" / "state.pt")
    if failures:
        write_json(run_dir / "run_summary.json", {**config, "status": "failed_preflight", "failures": failures})
        raise RuntimeError("preflight failed: " + "; ".join(failures))

    best = {"best_eval_acc": None, "best_eval_loss": None, "best_step": None}
    start_step = 0
    if args.resume_from:
        payload = torch.load(args.resume_from, map_location="cpu")
        start_step = int(payload.get("step", 0))
        best.update(payload.get("best", {}))
        for name, value in payload["master"].items():
            master[name].copy_(value.to(device=device, dtype=torch.float16))
        print(f"[{datetime.now().isoformat(timespec='seconds')}] resumed from step {start_step}", flush=True)

    restore_master(params, master)
    batch_iter = cycle(harness.train_loader)
    metrics_path = run_dir / "metrics.csv"
    write_header = not metrics_path.exists() or start_step == 0
    if start_step == 0 and metrics_path.exists():
        metrics_path.unlink()
    if start_step == 0:
        for stale in ("eval_metrics.jsonl", "perturbation_diagnostics.jsonl"):
            path = run_dir / stale
            if path.exists():
                path.unlink()

    finite_count = 0
    update_norm_last = None
    last_train_loss = None
    last_pert: Dict[str, object] = {}
    last_eval_loss = None
    last_eval_acc = None
    last_eval_step = None
    status = "complete"
    error_message = ""
    start_time = time.time()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    with metrics_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "step",
                "loss_plus",
                "loss_minus",
                "train_loss",
                "d_h",
                "d_h_finite",
                "update_norm",
                "seconds",
                "eval_loss",
                "eval_acc",
                "nan_flag",
            ],
        )
        if write_header:
            writer.writeheader()
        for step_idx in range(start_step, int(args.steps)):
            step_start = time.time()
            directions = sample_directions(master, direction_seed(int(args.seed), float(args.h), step_idx))
            batch = move_batch(next(batch_iter), device)
            copy_master_to_model(params, master, directions, float(args.h), +1.0, states)
            loss_plus, _ = harness.forward_loss_scores(batch)
            copy_master_to_model(params, master, directions, float(args.h), -1.0, states)
            loss_minus, _ = harness.forward_loss_scores(batch)
            restore_diff = restore_master(params, master)
            lp = float(loss_plus.detach().cpu())
            lm = float(loss_minus.detach().cpu())
            d_h = (lp - lm) / (2.0 * float(args.h))
            finite = math.isfinite(lp) and math.isfinite(lm) and math.isfinite(d_h)
            if finite:
                finite_count += 1
                update_norm_last = update_master(master, directions, float(args.lr), d_h)
                restore_master(params, master)
            last_train_loss = (lp + lm) / 2.0
            completed_step = step_idx + 1
            if step_idx == start_step or completed_step % max(1, int(args.diag_every)) == 0 or completed_step == int(args.steps):
                last_pert = perturbation_metrics(master, directions, states, float(args.h))
                last_pert.update({"grid_id_plus": 1, "grid_id_minus": 1, "scale_id_plus": 1, "scale_id_minus": 1})
                append_jsonl(run_dir / "perturbation_diagnostics.jsonl", {"step": completed_step, **last_pert})
            eval_loss = None
            eval_acc = None
            if int(args.eval_every) > 0 and (completed_step % int(args.eval_every) == 0 or completed_step == int(args.steps)):
                eval_loss, eval_acc = eval_quantized(harness, master, states, int(args.eval_batches))
                last_eval_loss = eval_loss
                last_eval_acc = eval_acc
                last_eval_step = completed_step
                append_jsonl(run_dir / "eval_metrics.jsonl", {"step": completed_step, "eval_loss": eval_loss, "eval_acc": eval_acc})
                if eval_acc is not None and (best.get("best_eval_acc") is None or eval_acc > best["best_eval_acc"]):
                    best["best_eval_acc"] = eval_acc
                    best["best_step"] = completed_step
                if eval_loss is not None and (best.get("best_eval_loss") is None or eval_loss < best["best_eval_loss"]):
                    best["best_eval_loss"] = eval_loss
            if int(args.checkpoint_every) > 0 and (completed_step % int(args.checkpoint_every) == 0 or completed_step == int(args.steps)):
                save_checkpoint(run_dir, completed_step, master, best, config)
            nan_flag = (not finite) or restore_diff > 1e-3 or (update_norm_last is not None and not math.isfinite(float(update_norm_last)))
            writer.writerow(
                {
                    "step": completed_step,
                    "loss_plus": lp,
                    "loss_minus": lm,
                    "train_loss": last_train_loss,
                    "d_h": d_h,
                    "d_h_finite": finite,
                    "update_norm": update_norm_last,
                    "seconds": time.time() - step_start,
                    "eval_loss": eval_loss,
                    "eval_acc": eval_acc,
                    "nan_flag": nan_flag,
                }
            )
            f.flush()
            if completed_step % max(1, int(args.log_every)) == 0 or completed_step == 1:
                print(
                    f"[{datetime.now().isoformat(timespec='seconds')}] step={completed_step} "
                    f"loss={(last_train_loss if last_train_loss is not None else float('nan')):.4g} "
                    f"d_h={d_h:.4g} upd={update_norm_last} eval_acc={eval_acc}",
                    flush=True,
                )
            if nan_flag:
                status = "failed"
                error_message = f"non-finite or restore invariant failure at step {completed_step}"
                break

    steps_completed = start_step
    with metrics_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if rows:
        steps_completed = int(float(rows[-1]["step"]))
    elapsed = time.time() - start_time
    peak_mem = float(torch.cuda.max_memory_allocated() / 1024 / 1024) if torch.cuda.is_available() else 0.0
    summary = {
        **config,
        "status": status,
        "error_message": error_message,
        "steps_completed": steps_completed,
        "best_eval_acc": best.get("best_eval_acc"),
        "best_eval_loss": best.get("best_eval_loss"),
        "best_step": best.get("best_step"),
        "last_eval_acc": last_eval_acc,
        "last_eval_loss": last_eval_loss,
        "last_eval_step": last_eval_step,
        "final_train_loss": last_train_loss,
        "d_h_finite_rate": finite_count / max(steps_completed - start_step, 1),
        "update_norm_last": update_norm_last,
        "active_frac": last_pert.get("active_frac"),
        "alignment": last_pert.get("alignment"),
        "norm_ratio": last_pert.get("norm_ratio"),
        "code_change_frac": last_pert.get("code_change_frac"),
        "delta_q_norm": last_pert.get("delta_q_norm"),
        "nominal_delta_norm": last_pert.get("nominal_delta_norm"),
        "delta_visibility_nmse": last_pert.get("delta_visibility_nmse"),
        "clip_frac": last_pert.get("clip_frac"),
        "saturation_frac": last_pert.get("saturation_frac"),
        "pair_shared_grid_observed": last_pert.get("pair_shared_grid_check"),
        "fresh_round_codes_observed": last_pert.get("fresh_round_codes_check"),
        "total_runtime": elapsed,
        "seconds_per_step": elapsed / max(steps_completed - start_step, 1),
        "peak_gpu_mem": peak_mem,
    }
    write_json(run_dir / "run_summary.json", summary)
    print(f"[{datetime.now().isoformat(timespec='seconds')}] done status={status} run_dir={run_dir}", flush=True)
    return 0 if status == "complete" else 1


def run_unit_test(args: argparse.Namespace) -> int:
    device = torch.device("cuda" if torch.cuda.is_available() and args.device != "cpu" else "cpu")
    torch.manual_seed(123)
    w = torch.randn(5, 257, device=device, dtype=torch.float16)
    u = torch.randn_like(w)
    state, _ = compute_rtn_state("synthetic.weight", w, int(args.group_size), 8)
    q_plus, plus_stats = qrw.quantize_with_state(w.float().add(u.float(), alpha=float(args.h)), state, True)
    q_minus, minus_stats = qrw.quantize_with_state(w.float().add(u.float(), alpha=-float(args.h)), state, True)
    same_grid = state.scales.data_ptr() == state.scales.data_ptr()
    fresh_codes_differ = bool(((q_plus.float() - q_minus.float()) != 0).any().detach().cpu())
    ok = (
        torch.isfinite(state.scales).all().item()
        and float(state.scales.min().detach().cpu()) > 0
        and int(plus_stats["code_min"]) >= -127
        and int(plus_stats["code_max"]) <= 127
        and int(minus_stats["code_min"]) >= -127
        and int(minus_stats["code_max"]) <= 127
        and same_grid
        and fresh_codes_differ
    )
    out = {
        "status": "pass" if ok else "fail",
        "group_size": int(args.group_size),
        "quant_bits": 8,
        "scale_positive_finite": bool(torch.isfinite(state.scales).all().item() and float(state.scales.min().detach().cpu()) > 0),
        "plus_code_min": plus_stats["code_min"],
        "plus_code_max": plus_stats["code_max"],
        "minus_code_min": minus_stats["code_min"],
        "minus_code_max": minus_stats["code_max"],
        "same_grid_object": same_grid,
        "fresh_rounded_codes_nonidentical": fresh_codes_differ,
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    write_json(args.output_root / "unit_test_fixed_grid.json", out)
    print(json.dumps(out, indent=2), flush=True)
    return 0 if ok else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_root", type=Path, default=REPO_ROOT / "outputs" / "mistral7b_sst5_int8_fixedgrid")
    parser.add_argument("--run_name", default="")
    parser.add_argument("--model_id", default="mistral-7b")
    parser.add_argument("--dataset_mode", default="full", choices=["full", "fewshot", "auto"])
    parser.add_argument("--num_k", type=int, default=16)
    parser.add_argument("--seed", type=int, default=16)
    parser.add_argument("--data_seed", type=int, default=16)
    parser.add_argument("--quantizer", choices=["rtn", "rtnclip", "gptq"], default="rtn")
    parser.add_argument("--group_size", type=int, default=128)
    parser.add_argument("--h", type=float, default=1e-3)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--eval_batch_size", type=int, default=1)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--eval_every", type=int, default=1000)
    parser.add_argument("--eval_batches", type=int, default=64)
    parser.add_argument("--checkpoint_every", type=int, default=5000)
    parser.add_argument("--diag_every", type=int, default=100)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--max_quantized_modules", type=int, default=0, help="debug only; 0 means all Linear weights")
    parser.add_argument("--resume_from", type=Path)
    parser.add_argument("--unit_test_only", action="store_true")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.unit_test_only:
        return run_unit_test(args)
    return run_training(args)


if __name__ == "__main__":
    raise SystemExit(main())
