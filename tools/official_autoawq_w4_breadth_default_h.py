#!/usr/bin/env python
"""Official AutoAWQ W4/G128 shared-grid MeZO smoke and breadth runner.

This runner is intentionally self-contained for the first AutoAWQ breadth
batch. It keeps a FP16 master model, periodically refreshes official AutoAWQ
parameters from the current master, and evaluates the two MeZO branches with a
cached shared AWQ grid:

    Q_r(w_t + h u), Q_r(w_t - h u)

AutoAWQ is used only at refresh steps. Plus/minus branches never rerun AWQ and
fresh-round separately with the same cached qzeros/scales.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import sys
import tempfile
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from transformers import AutoModelForCausalLM, AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = REPO_ROOT / "tools"
LARGE_MODELS_DIR = REPO_ROOT / "large_models"
for path in (TOOLS_DIR, LARGE_MODELS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from official_awq_krefresh_opt13b import (  # noqa: E402
    check_packages,
    copy_compatible_non_awq_state,
    env_info,
    extract_awq_params,
    params_json,
    unpack_awq_packed,
    write_json,
)
from tasks import get_task  # noqa: E402
from utils import DataCollatorWithPaddingAndNesting, encode_prompt, forward_wrap_with_option_len  # noqa: E402


OUT_ROOT = REPO_ROOT / "outputs" / "official_autoawq_w4_breadth_default_h_opt13b"
TASK_ALIASES = {
    "sst2": "SST2",
    "sst-2": "SST2",
    "SST-2": "SST2",
    "SST2": "SST2",
    "rte": "RTE",
    "RTE": "RTE",
    "boolq": "BoolQ",
    "BoolQ": "BoolQ",
    "sst5": "SST5",
    "sst-5": "SST5",
    "SST-5": "SST5",
    "SST5": "SST5",
}
DIR_NAMES = {
    "SST2": "sst2_h1e-3",
    "RTE": "rte_h1e-3",
    "BoolQ": "boolq_h1e-3",
    "SST5": "sst5_h1e-3",
}
EPS = 1e-12


def normalize_json(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return normalize_json(value.item())
        return normalize_json(value.detach().cpu().tolist())
    if is_dataclass(value) and not isinstance(value, type):
        return normalize_json(asdict(value))
    if isinstance(value, dict):
        return {str(k): normalize_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [normalize_json(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(normalize_json(row), sort_keys=True) + "\n")


def write_csv(path: Path, rows: Sequence[Dict[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fields), extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def append_csv(path: Path, row: Dict[str, Any], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fields), extrasaction="ignore", lineterminator="\n")
        if not exists:
            writer.writeheader()
        writer.writerow({field: row.get(field) for field in fields})


def stable_seed(*parts: object) -> int:
    text = "::".join(str(part) for part in parts)
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") % (2**31 - 1)


def normal_like_with_seed(tensor: torch.Tensor, seed: int) -> torch.Tensor:
    generator = torch.Generator(device=tensor.device)
    generator.manual_seed(int(seed))
    return torch.empty_like(tensor, dtype=tensor.dtype).normal_(0.0, 1.0, generator=generator)


def get_module_by_name(root: nn.Module, name: str) -> nn.Module:
    module = root
    for part in name.split("."):
        module = getattr(module, part)
    return module


def patch_option_loss(model: nn.Module) -> None:
    if not hasattr(model, "original_forward"):
        model.original_forward = model.forward
        model.forward = forward_wrap_with_option_len.__get__(model, type(model))


def unpatch_option_loss(model: nn.Module) -> None:
    if hasattr(model, "original_forward"):
        model.forward = model.original_forward
        delattr(model, "original_forward")


class PromptDataset(Dataset):
    def __init__(self, samples: Sequence[Any], task: Any, tokenizer: Any, max_length: int):
        self.samples = list(samples)
        self.task = task
        self.template = task.get_template()
        self.tokenizer = tokenizer
        self.max_length = int(max_length)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> List[Dict[str, Any]]:
        sample = self.samples[idx]
        encoded_candidates, option_lens = encode_prompt(
            self.task,
            self.template,
            [],
            sample,
            self.tokenizer,
            max_length=self.max_length,
            generation=getattr(self.task, "generation", False),
            generation_with_gold=True,
        )
        if getattr(self.task, "generation", False):
            correct_candidate_id = 0
        elif isinstance(sample.correct_candidate, list):
            correct_candidate_id = sample.candidates.index(sample.correct_candidate[0])
        else:
            correct_candidate_id = sample.candidates.index(sample.correct_candidate)
        return [
            {
                "input_ids": encoded_candidates[i],
                "labels": correct_candidate_id,
                "option_len": option_lens[i],
                "num_options": len(sample.candidates),
            }
            for i in range(len(encoded_candidates))
        ]


def canonical_task(name: str) -> str:
    if name not in TASK_ALIASES:
        raise ValueError(f"Unsupported dataset {name!r}; expected one of {sorted(set(TASK_ALIASES))}")
    return TASK_ALIASES[name]


def sample_id(sample: Any, fallback: int) -> str:
    sid = getattr(sample, "id", None)
    if sid is not None:
        return str(sid)
    data = getattr(sample, "data", None)
    digest = hashlib.sha256(json.dumps(normalize_json(data), sort_keys=True).encode("utf-8")).hexdigest()
    return digest[:16] if digest else str(fallback)


def sample_calib_text(task: Any, sample: Any) -> str:
    template = task.get_template()
    try:
        return template.verbalize(sample, sample.correct_candidate).strip()
    except Exception:
        try:
            return template.encode(sample).strip()
        except Exception:
            return json.dumps(normalize_json(getattr(sample, "data", {})), sort_keys=True)


def load_samples(
    task_name: str,
    data_seed: int,
    *,
    dataset_mode: str,
    num_k: int,
    num_train: int,
    num_eval: Optional[int],
) -> Tuple[Any, List[Any], List[Any]]:
    task = get_task(task_name)
    train_sets = task.sample_train_sets(
        num_train=int(num_train),
        num_dev=0,
        num_eval=num_eval,
        num_train_sets=1,
        seed=int(data_seed),
        dataset_mode=str(dataset_mode),
        num_k=int(num_k),
    )
    train_samples = list(train_sets[0])
    eval_splits = task.get_eval_splits() if hasattr(task, "get_eval_splits") else {"valid": task.valid_samples}
    eval_samples = list(eval_splits["valid" if "valid" in eval_splits else list(eval_splits.keys())[0]])
    if num_eval is not None and int(num_eval) > 0:
        eval_samples = task.sample_subset(data_split="valid", seed=0, num=int(num_eval))
    return task, train_samples, eval_samples


def infinite_loader(loader: DataLoader) -> Iterator[Dict[str, torch.Tensor]]:
    while True:
        for batch in loader:
            yield batch


def prepare_batch(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    out = {}
    for key, value in batch.items():
        out[key] = value.to(device) if isinstance(value, torch.Tensor) else value
    return out


def compute_loss(model: nn.Module, batch: Dict[str, Any]) -> torch.Tensor:
    model.eval()
    with torch.inference_mode():
        outputs = model(**batch, return_dict=True)
        return outputs.loss.detach()


def classification_accuracy(model: nn.Module, loader: DataLoader, device: torch.device, max_batches: Optional[int] = None) -> Tuple[float, float]:
    model.eval()
    total = 0
    correct = 0
    losses: List[float] = []
    with torch.inference_mode():
        for batch_id, batch in enumerate(loader):
            if max_batches is not None and batch_id >= int(max_batches):
                break
            batch = prepare_batch(batch, device)
            outputs = model(**batch, return_dict=True)
            losses.append(float(outputs.loss.detach().cpu()))
            input_ids = batch["input_ids"]
            option_len = batch["option_len"].detach().cpu().tolist()
            num_options = batch["num_options"].detach().cpu().tolist()
            labels = batch["labels"].detach().cpu().tolist()
            logits = outputs.logits[..., :-1, :].float()
            shift_labels = input_ids[..., 1:].clone()
            shift_labels[shift_labels == model.config.pad_token_id] = -100
            scores = []
            for i, opt_len in enumerate(option_len):
                labels_i = shift_labels[i].clone()
                if int(opt_len) > 0:
                    labels_i[:-int(opt_len)] = -100
                mask = labels_i != -100
                labels_i[~mask] = 0
                log_probs = torch.log_softmax(logits[i], dim=-1)
                selected = torch.gather(log_probs, -1, labels_i.unsqueeze(-1)).squeeze(-1)
                score = (selected * mask).sum() / mask.sum().clamp_min(1)
                scores.append(float(score.detach().cpu()))
            start = 0
            while start < len(scores):
                n_opt = int(num_options[start])
                pred = int(np.argmax(scores[start : start + n_opt]))
                gold = int(labels[start])
                correct += int(pred == gold)
                total += 1
                start += n_opt
    acc = float(correct) / float(max(total, 1))
    loss = float(np.mean(losses)) if losses else float("nan")
    return acc, loss


class AutoAWQSharedGridTrainer:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.task_name = canonical_task(args.dataset)
        self.output_root = Path(args.output_root)
        run_dir_name = str(args.run_dir_name or ("smoke_sst2_h1e-3" if args.smoke else DIR_NAMES[self.task_name]))
        self.run_dir = self.output_root / run_dir_name
        if bool(args.overwrite) and self.run_dir.exists():
            shutil.rmtree(self.run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "checkpoints").mkdir(exist_ok=True)
        self.device = torch.device("cuda:0")
        self.env = env_info()
        self.env["mode"] = str(args.mode_label or ("smoke" if args.smoke else "breadth"))
        self.env["dataset"] = self.task_name
        self.metrics_fields = [
            "step",
            "time",
            "epoch",
            "h",
            "lr",
            "loss_plus",
            "loss_minus",
            "finite_difference",
            "qparam_id",
            "cached_param_age",
            "code_change_frac",
            "active_frac",
            "clip_frac",
            "saturation_frac",
            "alignment",
            "norm_ratio",
            "eval_accuracy",
            "eval_loss",
        ]
        self.best_eval_acc = None
        self.best_eval_loss = None
        self.best_step = None
        self.last_eval_acc = None
        self.last_eval_loss = None
        self.current_qparam_id = -1
        self.current_refresh_step = None
        self.awq_params: Dict[str, Dict[str, torch.Tensor]] = {}
        self.awq_runtime_cache: Dict[str, Dict[str, Any]] = {}
        self.quantized_weight_param_names: set[str] = set()
        self.master: Dict[str, torch.Tensor] = {}
        self.trainable_param_items: List[Tuple[str, nn.Parameter]] = []
        self.rng = np.random.RandomState(int(args.seed))

    def static_awq_once(self) -> bool:
        return bool(getattr(self.args, "static_awq_once", False))

    def refresh_steps(self) -> List[int]:
        if self.static_awq_once():
            return [int(self.args.start_step)] if int(self.args.start_step) == 0 else []
        return list(range(int(self.args.start_step), int(self.args.steps), int(self.args.k_refresh)))

    def should_refresh(self, step: int) -> bool:
        if self.static_awq_once():
            return int(step) == 0 and int(self.args.start_step) == 0
        return step == int(self.args.start_step) or ((step - int(self.args.start_step)) % int(self.args.k_refresh) == 0)

    def write_config(self) -> None:
        config = {
            "model": self.args.model_id,
            "dataset": self.task_name,
            "dataset_mode": str(self.args.dataset_mode),
            "num_k": int(self.args.num_k),
            "num_train": int(self.args.num_train),
            "num_eval": None if self.args.num_eval is None else int(self.args.num_eval),
            "seed": int(self.args.seed),
            "data_seed": int(self.args.data_seed),
            "h": float(self.args.h),
            "max_steps": int(self.args.steps),
            "batch_size": int(self.args.batch_size),
            "eval_batch_size": int(self.args.eval_batch_size),
            "dataloader_shuffle": True,
            "sampler": "RandomSampler",
            "direction": "dense",
            "quantizer": "official_autoawq_param_shared_grid_fake_quant",
            "official_autoawq_package": True,
            "quant_bits": 4,
            "group_size": 128,
            "weight_only": True,
            "activation_dtype": "fp16",
            "master_dtype": "fp16",
            "fp16_master_update": True,
            "direct_int_update": False,
            "pair_shared_quantizer": True,
            "fresh_round_codes": True,
            "independent_plus_minus_awq": False,
            "q_w_plus_hu_bypass": False,
            "runtime_path": "optimized_cached_awq_params",
            "runtime_optimizations": [
                "cached_unpacked_zeros_scales_groups_on_gpu",
                "cached_linear_module_references",
                "single_dense_direction_materialization_reused_for_plus_minus_update",
                "no_clip_saturation_reduction_on_training_steps_without_quant_diag",
            ],
            "K_refresh": None if self.static_awq_once() else int(self.args.k_refresh),
            "refresh_policy": "static_step0_quantize_once" if self.static_awq_once() else f"K={int(self.args.k_refresh)}",
            "refresh_steps": self.refresh_steps(),
            "calibration_size_requested": int(self.args.calibration_size),
            "learning_rate": float(self.args.lr),
            "lr_scheduler": "constant",
            "autoawq_config": {"zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM"},
            "autoawq_transformed_master_rebase_at_refresh": "prequant_transformed_linear_and_non_quantized_state",
            "notes": (
                "Official AutoAWQ applies fused scaling/clipping transforms. This runner captures the transformed "
                "FP16 Linear weights immediately before AutoAWQ packs them, and uses that pre-quant transformed "
                "state as the FP16 master after each refresh. It does not use dequantized packed qweight as the "
                "trainable master."
            ),
            "environment": self.env,
        }
        write_json(self.run_dir / "run_config.json", config)
        write_json(
            self.run_dir / "run_manifest_row.json",
            {
                "dataset": self.task_name,
                "dataset_mode": str(self.args.dataset_mode),
                "num_k": int(self.args.num_k),
                "h": float(self.args.h),
                "model": self.args.model_id,
                "quantizer": "official_autoawq_param_shared_grid_fake_quant",
                "precision": "W4A16",
                "group_size": 128,
                "K_refresh": "static_step0" if self.static_awq_once() else int(self.args.k_refresh),
                "refresh_policy": "static_step0_quantize_once" if self.static_awq_once() else f"K={int(self.args.k_refresh)}",
                "max_steps": int(self.args.steps),
                "seed": int(self.args.seed),
                "data_seed": int(self.args.data_seed),
                "batch_size": int(self.args.batch_size),
                "status": "running",
                "run_dir": str(self.run_dir),
            },
        )

    def load(self) -> None:
        check_packages(self.output_root)
        self.task, self.train_samples, self.eval_samples = load_samples(
            self.task_name,
            int(self.args.data_seed),
            dataset_mode=str(self.args.dataset_mode),
            num_k=int(self.args.num_k),
            num_train=int(self.args.num_train),
            num_eval=self.args.num_eval,
        )
        self.calib_samples = self.train_samples[: int(self.args.calibration_size)]
        self.calib_texts = [sample_calib_text(self.task, sample) for sample in self.calib_samples]
        self.calib_ids = [sample_id(sample, idx) for idx, sample in enumerate(self.calib_samples)]
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.model_id, use_fast=False)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        init_path = self.args.init_model_path or self.args.model_id
        self.model = AutoModelForCausalLM.from_pretrained(init_path, torch_dtype=torch.float16).to(self.device)
        if self.model.config.pad_token_id is None:
            self.model.config.pad_token_id = self.tokenizer.pad_token_id
        if getattr(self.model, "generation_config", None) is not None and self.model.generation_config.pad_token_id is None:
            self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id
        patch_option_loss(self.model)
        self.model.eval()
        train_dataset = PromptDataset(self.train_samples, self.task, self.tokenizer, int(self.args.max_length))
        eval_dataset = PromptDataset(self.eval_samples, self.task, self.tokenizer, int(self.args.max_length))
        generator = torch.Generator()
        generator.manual_seed(int(self.args.data_seed))
        self.train_sampler = RandomSampler(train_dataset, generator=generator)
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=int(self.args.batch_size),
            sampler=self.train_sampler,
            collate_fn=DataCollatorWithPaddingAndNesting(self.tokenizer, pad_to_multiple_of=8),
        )
        self.eval_loader = DataLoader(
            eval_dataset,
            batch_size=int(self.args.eval_batch_size),
            sampler=SequentialSampler(eval_dataset),
            collate_fn=DataCollatorWithPaddingAndNesting(self.tokenizer, pad_to_multiple_of=8),
        )
        self.train_iter = infinite_loader(self.train_loader)
        self.master = {name: param.detach().clone() for name, param in self.model.named_parameters() if param.requires_grad}
        self.trainable_param_items = [(name, param) for name, param in self.model.named_parameters() if name in self.master]
        if self.static_awq_once() and int(self.args.start_step) > 0:
            self.load_static_awq_params_for_resume()
        write_json(
            self.run_dir / "calibration_subset.json",
            {
                "dataset_mode": str(self.args.dataset_mode),
                "num_k": int(self.args.num_k),
                "num_train": int(self.args.num_train),
                "train_sample_count": len(self.train_samples),
                "eval_sample_count": len(self.eval_samples),
                "calibration_size_requested": int(self.args.calibration_size),
                "calibration_size": len(self.calib_texts),
                "calibration_ids_or_hashes": self.calib_ids,
                "calibration_text_hashes": [hashlib.sha256(text.encode("utf-8")).hexdigest()[:16] for text in self.calib_texts],
            },
        )
        write_json(
            self.run_dir / "dataset_manifest.json",
            {
                "dataset": self.task_name,
                "dataset_mode": str(self.args.dataset_mode),
                "fewshot_num_k_per_class": int(self.args.num_k) if str(self.args.dataset_mode).lower() == "fewshot" else None,
                "train_sample_count": len(self.train_samples),
                "eval_sample_count": len(self.eval_samples),
                "train_label_counts": self.label_counts(self.train_samples),
                "train_sample_ids": [sample_id(sample, idx) for idx, sample in enumerate(self.train_samples)],
            },
        )

    @staticmethod
    def label_counts(samples: Sequence[Any]) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for sample in samples:
            key = str(getattr(sample, "correct_candidate", "unknown"))
            counts[key] = counts.get(key, 0) + 1
        return counts

    def load_static_awq_params_for_resume(self) -> None:
        qparams_path = Path(self.args.static_qparams_path) if self.args.static_qparams_path else self.run_dir / "awq_params_step000000.pt"
        if not qparams_path.exists():
            raise FileNotFoundError(
                f"static AWQ resume requires saved step0 qparams at {qparams_path}; "
                "rerun with --save_awq_params or pass --static_qparams_path"
            )
        self.awq_params = torch.load(qparams_path, map_location="cpu")
        if not isinstance(self.awq_params, dict) or not self.awq_params:
            raise RuntimeError(f"invalid static AWQ qparams loaded from {qparams_path}")
        self.quantized_weight_param_names = {f"{name}.weight" for name in self.awq_params}
        self.current_qparam_id = 0
        self.current_refresh_step = 0
        self.build_awq_runtime_cache()
        append_jsonl(
            self.run_dir / "awq_refresh_log.jsonl",
            {
                "refresh_step": int(self.args.start_step),
                "qparam_id": 0,
                "static_resume_loaded_qparams": str(qparams_path),
                "refresh_runtime_sec": 0.0,
                "calibration_size_requested": int(self.args.calibration_size),
                "calibration_size": len(getattr(self, "calib_texts", [])),
                "group_size": 128,
                "bit_width": 4,
                "module_count": len(self.awq_params),
                "module_list": sorted(self.awq_params.keys()),
            },
        )

    def build_awq_runtime_cache(self) -> None:
        cache: Dict[str, Dict[str, Any]] = {}
        for module_name, p in self.awq_params.items():
            module = get_module_by_name(self.model, module_name)
            if not isinstance(module, nn.Linear):
                raise TypeError(f"{module_name} is {type(module)} not nn.Linear")
            out_features = int(p["out_features"].item())
            in_features = int(p["in_features"].item())
            group_size = int(p["group_size"].item())
            cache[module_name] = {
                "module": module,
                "weight_name": f"{module_name}.weight",
                "scales": p["scales"].to(self.device).float(),
                "zeros": unpack_awq_packed(p["qzeros"].to(self.device), out_features).float(),
                "groups": torch.arange(in_features, device=self.device) // int(group_size),
            }
        self.awq_runtime_cache = cache

    def fake_quant_cached(
        self,
        x: torch.Tensor,
        cache: Dict[str, Any],
        *,
        need_codes: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Dict[str, float]]]:
        xt = x.t().float().contiguous()
        groups = cache["groups"]
        scales = cache["scales"]
        zeros = cache["zeros"]
        q = torch.round(xt / scales[groups] + zeros[groups]).clamp(0, 15)
        deq = (q - zeros[groups]) * scales[groups]
        out = deq.t().contiguous().to(torch.float16)
        if not need_codes:
            return out, None, None
        stats = {
            "clip_frac": float(((q <= 0) | (q >= 15)).float().mean().detach().cpu().item()),
            "saturation_frac": float(((q <= 0) | (q >= 15)).float().mean().detach().cpu().item()),
        }
        return out, q.to(torch.int16), stats

    def materialize_step_directions(self, step_seed: int) -> Dict[str, torch.Tensor]:
        return {
            name: normal_like_with_seed(self.master[name], stable_seed(self.args.seed, step_seed, name))
            for name, _param in self.trainable_param_items
        }

    def save_master_to_model(self) -> None:
        with torch.no_grad():
            items = self.trainable_param_items or [(name, param) for name, param in self.model.named_parameters()]
            for name, param in items:
                if name in self.master:
                    param.data.copy_(self.master[name].to(param.device, dtype=param.dtype))

    def refresh_awq(self, step: int, recon_batch: Dict[str, Any]) -> None:
        from awq import AutoAWQForCausalLM
        from awq.quantize.quantizer import AwqQuantizer
        from awq.utils.module import get_op_name
        import transformers
        import awq.quantize.quantizer as awq_quantizer

        self.save_master_to_model()
        qparam_id = self.current_qparam_id + 1
        quant_config = {"zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM"}
        t0 = time.time()
        tmp_parent = self.run_dir / "awq_tmp"
        tmp_parent.mkdir(exist_ok=True)
        captured_prequant_weights: Dict[str, torch.Tensor] = {}
        original_apply_quant = AwqQuantizer._apply_quant

        def capture_apply_quant(quantizer_self, module, named_linears):
            module_prefix = get_op_name(quantizer_self.model, module)
            for local_name, linear_layer in named_linears.items():
                full_name = f"{module_prefix}.{local_name}" if module_prefix else local_name
                captured_prequant_weights[full_name] = linear_layer.weight.detach().cpu().clone()
            return original_apply_quant(quantizer_self, module, named_linears)

        with tempfile.TemporaryDirectory(prefix=f"refresh_{step:06d}_", dir=str(tmp_parent)) as tmp:
            tmp_path = Path(tmp)
            unpatch_option_loss(self.model)
            self.model.save_pretrained(tmp_path, safe_serialization=False)
            self.tokenizer.save_pretrained(tmp_path)
            patch_option_loss(self.model)
            awq_wrapper = AutoAWQForCausalLM.from_pretrained(str(tmp_path), torch_dtype=torch.float16, safetensors=False, device_map=None)
            if getattr(awq_wrapper.model.config, "pad_token_id", None) is None:
                awq_wrapper.model.config.pad_token_id = self.tokenizer.pad_token_id
            original_transformers_version = transformers.__version__
            original_awq_quantizer_transformers_version = awq_quantizer.transformers.__version__
            if str(getattr(awq_wrapper, "model_type", "")).lower() == "opt" and original_transformers_version >= "4.48.0":
                transformers.__version__ = "4.47.1"
                awq_quantizer.transformers.__version__ = "4.47.1"
            AwqQuantizer._apply_quant = capture_apply_quant
            try:
                awq_wrapper.quantize(
                    self.tokenizer,
                    quant_config=quant_config,
                    calib_data=self.calib_texts,
                    max_calib_samples=int(self.args.calibration_size),
                    max_calib_seq_len=int(self.args.max_calib_seq_len),
                    n_parallel_calib_samples=int(self.args.n_parallel_calib_samples),
                    apply_clip=True,
                )
            finally:
                AwqQuantizer._apply_quant = original_apply_quant
                transformers.__version__ = original_transformers_version
                awq_quantizer.transformers.__version__ = original_awq_quantizer_transformers_version
            awq_model = awq_wrapper.model.to(self.device)
        awq_model.eval()
        patch_option_loss(awq_model)
        params = extract_awq_params(awq_model)
        if not params:
            raise RuntimeError("AutoAWQ produced no extractable WQLinear parameters")
        self.awq_params = params
        self.quantized_weight_param_names = {f"{name}.weight" for name in params}
        self.build_awq_runtime_cache()
        copied_non_awq = copy_compatible_non_awq_state(awq_model, self.model)
        refreshed_master: Dict[str, torch.Tensor] = {}
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if name in self.quantized_weight_param_names:
                module_name = name[: -len(".weight")]
                if module_name not in captured_prequant_weights:
                    raise RuntimeError(f"missing pre-quant transformed weight capture for {module_name}")
                refreshed_master[name] = captured_prequant_weights[module_name].to(
                    device=param.device,
                    dtype=param.dtype,
                )
            else:
                refreshed_master[name] = param.detach().clone()
        self.master = refreshed_master
        self.current_qparam_id = qparam_id
        self.current_refresh_step = int(step)
        refresh_sec = time.time() - t0

        recon = self.reconstruction_check(awq_model, recon_batch)
        recon.update(
            {
                "step": int(step),
                "qparam_id": int(qparam_id),
                "copied_non_awq_state_tensors": int(copied_non_awq),
            }
        )
        append_jsonl(self.run_dir / "reconstruction_check.jsonl", recon)
        if not recon["pass"]:
            raise RuntimeError(f"reconstruction check failed at step {step}: {recon}")
        refresh_row = {
            "refresh_step": int(step),
            "qparam_id": int(qparam_id),
            "refresh_runtime_sec": float(refresh_sec),
            "calibration_size_requested": int(self.args.calibration_size),
            "calibration_size": len(self.calib_texts),
            "calibration_ids_or_hashes": self.calib_ids,
            "group_size": 128,
            "bit_width": 4,
            "autoawq_version": self.env.get("autoawq"),
            "transformers_version": self.env.get("transformers"),
            "optimum_version": self.env.get("optimum"),
            "gpu_type": self.env.get("gpu_name"),
            "module_count": len(params),
            "module_list": sorted(params.keys()),
            "reconstruction_logits_cosine": recon["logits_cosine"],
            "reconstruction_loss_absdiff": recon["loss_absdiff"],
        }
        append_jsonl(self.run_dir / "awq_refresh_log.jsonl", refresh_row)
        write_json(
            self.run_dir / f"awq_params_step{int(step):06d}.json",
            {
                "refresh_step": int(step),
                "qparam_id": int(qparam_id),
                "quant_config": quant_config,
                "module_count": len(params),
                "modules": params_json(params),
            },
        )
        if bool(self.args.save_awq_params):
            torch.save(params, self.run_dir / f"awq_params_step{int(step):06d}.pt")
        print(
            f"[refresh] step={step} qparam_id={qparam_id} modules={len(params)} "
            f"sec={refresh_sec:.3f} recon_cos={recon['logits_cosine']:.6f}",
            flush=True,
        )
        del awq_model
        torch.cuda.empty_cache()

    def reconstruction_check(self, awq_model: nn.Module, batch: Dict[str, Any]) -> Dict[str, Any]:
        batch = prepare_batch(batch, self.device)
        self.save_master_to_model()
        self.apply_quantized_base_weights()
        with torch.inference_mode():
            awq_outputs = awq_model(**batch, return_dict=True)
            fake_outputs = self.model(**batch, return_dict=True)
            loss_awq = awq_outputs.loss.detach()
            logits_awq = awq_outputs.logits.detach().float().reshape(-1)
            loss_fake = fake_outputs.loss.detach()
            logits_fake = fake_outputs.logits.detach().float().reshape(-1)
        denom = torch.linalg.vector_norm(logits_awq) * torch.linalg.vector_norm(logits_fake)
        return {
            "logits_mse": float(torch.mean((logits_awq - logits_fake) ** 2).cpu()),
            "logits_cosine": float((torch.dot(logits_awq, logits_fake) / denom.clamp_min(EPS)).cpu()),
            "loss_awq": float(loss_awq.cpu()),
            "loss_fake": float(loss_fake.cpu()),
            "loss_absdiff": float(abs(float(loss_awq.cpu()) - float(loss_fake.cpu()))),
            "max_abs_error": float(torch.max(torch.abs(logits_awq - logits_fake)).cpu()),
            "pass": bool((torch.dot(logits_awq, logits_fake) / denom.clamp_min(EPS)).cpu() > 0.99),
        }

    def apply_quantized_base_weights(self) -> None:
        with torch.no_grad():
            for _module_name, cache in self.awq_runtime_cache.items():
                name = cache["weight_name"]
                module = cache["module"]
                q, _, _ = self.fake_quant_cached(self.master[name], cache)
                module.weight.data.copy_(q.to(module.weight.device, dtype=module.weight.dtype))

    def apply_branch(self, directions: Dict[str, torch.Tensor], sign: float) -> None:
        h = float(self.args.h)
        with torch.no_grad():
            for name, param in self.trainable_param_items:
                base = self.master[name]
                z = directions[name]
                x = base + float(sign) * h * z
                if name in self.quantized_weight_param_names:
                    module_name = name[: -len(".weight")]
                    q, _, _ = self.fake_quant_cached(x, self.awq_runtime_cache[module_name])
                    param.data.copy_(q.to(param.device, dtype=param.dtype))
                else:
                    param.data.copy_(x.to(param.device, dtype=param.dtype))

    def quant_diagnostics(self, directions: Dict[str, torch.Tensor]) -> Dict[str, float]:
        h = float(self.args.h)
        acc = {
            "dot": 0.0,
            "dq_norm_sq": 0.0,
            "ideal_norm_sq": 0.0,
            "code_changed": 0.0,
            "code_total": 0.0,
            "clip": 0.0,
            "sat": 0.0,
            "mods": 0.0,
        }
        with torch.no_grad():
            for _module_name, cache in self.awq_runtime_cache.items():
                name = cache["weight_name"]
                base = self.master[name]
                z = directions[name]
                q_plus, c_plus, st_plus = self.fake_quant_cached(base + h * z, cache, need_codes=True)
                q_minus, c_minus, st_minus = self.fake_quant_cached(base - h * z, cache, need_codes=True)
                assert c_plus is not None and c_minus is not None and st_plus is not None and st_minus is not None
                dq = (q_plus.float() - q_minus.float()).reshape(-1)
                ideal = (2.0 * h * z.float()).reshape(-1)
                acc["dot"] += float(torch.dot(dq, ideal).cpu())
                acc["dq_norm_sq"] += float(torch.dot(dq, dq).cpu())
                acc["ideal_norm_sq"] += float(torch.dot(ideal, ideal).cpu())
                acc["code_changed"] += float((c_plus != c_minus).sum().cpu())
                acc["code_total"] += float(c_plus.numel())
                acc["clip"] += 0.5 * (float(st_plus["clip_frac"]) + float(st_minus["clip_frac"]))
                acc["sat"] += 0.5 * (float(st_plus["saturation_frac"]) + float(st_minus["saturation_frac"]))
                acc["mods"] += 1.0
        alignment = acc["dot"] / max(math.sqrt(acc["dq_norm_sq"]) * math.sqrt(acc["ideal_norm_sq"]), EPS)
        norm_ratio = math.sqrt(acc["dq_norm_sq"]) / max(math.sqrt(acc["ideal_norm_sq"]), EPS)
        return {
            "code_change_frac": acc["code_changed"] / max(acc["code_total"], 1.0),
            "active_frac": 1.0,
            "clip_frac": acc["clip"] / max(acc["mods"], 1.0),
            "saturation_frac": acc["sat"] / max(acc["mods"], 1.0),
            "alignment": alignment,
            "norm_ratio": norm_ratio,
        }

    def update_master(self, directions: Dict[str, torch.Tensor], projected_grad: float) -> None:
        lr = float(self.args.lr)
        with torch.no_grad():
            for name, base in self.master.items():
                z = directions[name]
                base.add_(z, alpha=-(lr * float(projected_grad)))
        self.save_master_to_model()

    def evaluate_and_checkpoint(self, step: int) -> Dict[str, Optional[float]]:
        self.save_master_to_model()
        self.apply_quantized_base_weights()
        eval_acc, eval_loss = classification_accuracy(
            self.model,
            self.eval_loader,
            self.device,
            max_batches=(int(self.args.eval_max_batches) if int(self.args.eval_max_batches) > 0 else None),
        )
        self.last_eval_acc = eval_acc
        self.last_eval_loss = eval_loss
        improved_acc = self.best_eval_acc is None or eval_acc > float(self.best_eval_acc)
        improved_loss = self.best_eval_loss is None or eval_loss < float(self.best_eval_loss)
        if improved_acc:
            self.best_eval_acc = eval_acc
            self.best_step = int(step)
            self.save_checkpoint("best_acc", step)
        if improved_loss:
            self.best_eval_loss = eval_loss
            self.save_checkpoint("best_loss", step)
        append_jsonl(
            self.run_dir / "eval_metrics.jsonl",
            {"step": int(step), "eval_accuracy": eval_acc, "eval_loss": eval_loss, "best_eval_acc": self.best_eval_acc, "best_step": self.best_step},
        )
        return {"eval_accuracy": eval_acc, "eval_loss": eval_loss}

    def save_checkpoint(self, name: str, step: int) -> None:
        ckpt_dir = self.run_dir / "checkpoints" / name
        if ckpt_dir.exists():
            shutil.rmtree(ckpt_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.save_master_to_model()
        unpatch_option_loss(self.model)
        self.model.save_pretrained(ckpt_dir, safe_serialization=False)
        self.tokenizer.save_pretrained(ckpt_dir)
        patch_option_loss(self.model)
        write_json(
            ckpt_dir / "awq_mezo_checkpoint_metadata.json",
            {
                "step": int(step),
                "qparam_id": int(self.current_qparam_id),
                "cached_param_age": None if self.current_refresh_step is None else int(step) - int(self.current_refresh_step),
                "resume_command": self.resume_command(ckpt_dir, int(step)),
            },
        )
        write_json(
            self.run_dir / "run_state.json",
            {"latest_checkpoint": str(ckpt_dir), "latest_step": int(step), "resume_command": self.resume_command(ckpt_dir, int(step))},
        )
        (self.run_dir / "resume_command.txt").write_text(self.resume_command(ckpt_dir, int(step)) + "\n", encoding="utf-8")

    def resume_command(self, ckpt_dir: Path, step: int) -> str:
        static_flags = ""
        if self.static_awq_once():
            static_flags += " --static_awq_once --save_awq_params"
            qparams_path = self.run_dir / "awq_params_step000000.pt"
            static_flags += f" --static_qparams_path {qparams_path}"
        if self.args.run_dir_name:
            static_flags += f" --run_dir_name {self.args.run_dir_name}"
        if self.args.mode_label:
            static_flags += f" --mode_label {self.args.mode_label}"
        return (
            f"cd {REPO_ROOT} && CUDA_VISIBLE_DEVICES=0 "
            f"/home/jy03364/miniconda3/envs/mezo-env/bin/python tools/official_autoawq_w4_breadth_default_h.py "
            f"--dataset {self.task_name} --steps {int(self.args.steps)} --start_step {int(step)} "
            f"--dataset_mode {self.args.dataset_mode} --num_k {int(self.args.num_k)} --num_train {int(self.args.num_train)} "
            f"--init_model_path {ckpt_dir} --output_root {self.output_root} --batch_size {int(self.args.batch_size)} "
            f"--eval_batch_size {int(self.args.eval_batch_size)} --k_refresh {int(self.args.k_refresh)} --h {float(self.args.h):g}"
            f"{static_flags}"
        )

    def train(self) -> Dict[str, Any]:
        self.write_config()
        self.load()
        recon_batch = prepare_batch(next(self.train_iter), self.device)
        train_log_path = self.run_dir / "train.log"
        start_wall = time.time()
        failure = None
        try:
            for step in range(int(self.args.start_step), int(self.args.steps)):
                if self.should_refresh(step):
                    self.refresh_awq(step, recon_batch)
                if not self.awq_params:
                    raise RuntimeError("AWQ parameters are not loaded; static resume requires step0 qparams")
                batch = prepare_batch(next(self.train_iter), self.device)
                step_seed = int(self.rng.randint(0, 1_000_000_000))
                directions = self.materialize_step_directions(step_seed)
                diag = {}
                if bool(self.args.smoke) or (int(self.args.quant_diag_every) > 0 and step % int(self.args.quant_diag_every) == 0):
                    diag = self.quant_diagnostics(directions)
                    append_jsonl(
                        self.run_dir / "quantizer_diagnostics.jsonl",
                        {
                            "step": int(step),
                            "qparam_id": int(self.current_qparam_id),
                            "cached_param_age": int(step) - int(self.current_refresh_step),
                            **diag,
                        },
                    )
                self.apply_branch(directions, +1.0)
                loss_plus = compute_loss(self.model, batch)
                self.apply_branch(directions, -1.0)
                loss_minus = compute_loss(self.model, batch)
                fd = (float(loss_plus.cpu()) - float(loss_minus.cpu())) / (2.0 * float(self.args.h))
                self.update_master(directions, fd)
                del directions
                eval_metrics = {"eval_accuracy": None, "eval_loss": None}
                next_step = step + 1
                if next_step == int(self.args.steps) or (int(self.args.eval_steps) > 0 and next_step % int(self.args.eval_steps) == 0):
                    eval_metrics = self.evaluate_and_checkpoint(next_step)
                if next_step == int(self.args.steps) or (int(self.args.save_steps) > 0 and next_step % int(self.args.save_steps) == 0):
                    self.save_checkpoint(f"step_{next_step}", next_step)
                row = {
                    "step": next_step,
                    "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "epoch": float("nan"),
                    "h": float(self.args.h),
                    "lr": float(self.args.lr),
                    "loss_plus": float(loss_plus.cpu()),
                    "loss_minus": float(loss_minus.cpu()),
                    "finite_difference": fd,
                    "qparam_id": int(self.current_qparam_id),
                    "cached_param_age": int(step) - int(self.current_refresh_step),
                    "code_change_frac": diag.get("code_change_frac"),
                    "active_frac": diag.get("active_frac"),
                    "clip_frac": diag.get("clip_frac"),
                    "saturation_frac": diag.get("saturation_frac"),
                    "alignment": diag.get("alignment"),
                    "norm_ratio": diag.get("norm_ratio"),
                    **eval_metrics,
                }
                append_csv(self.run_dir / "metrics.csv", row, self.metrics_fields)
                append_jsonl(self.run_dir / "perturbation_diagnostics.jsonl", row)
                with train_log_path.open("a", encoding="utf-8") as f:
                    f.write(
                        f"step={next_step} qparam_id={self.current_qparam_id} age={row['cached_param_age']} "
                        f"loss_plus={row['loss_plus']:.6f} loss_minus={row['loss_minus']:.6f} fd={fd:.6f}\n"
                    )
                print(
                    f"[step] {next_step}/{self.args.steps} qparam={self.current_qparam_id} "
                    f"loss+={row['loss_plus']:.4f} loss-={row['loss_minus']:.4f} fd={fd:.4f}",
                    flush=True,
                )
            self.save_checkpoint("final", int(self.args.steps))
        except Exception as exc:
            failure = f"{type(exc).__name__}: {exc}"
            append_jsonl(self.run_dir / "failure_report.jsonl", {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "failure": failure})
            raise
        finally:
            elapsed = time.time() - start_wall
            summary = {
                "status": "failed" if failure else "completed",
                "failure": failure,
                "dataset": self.task_name,
                "model": self.args.model_id,
                "h": float(self.args.h),
                "final_step": int(self.args.steps) if failure is None else None,
                "best_eval_acc": self.best_eval_acc,
                "last_eval_acc": self.last_eval_acc,
                "best_eval_loss": self.best_eval_loss,
                "last_eval_loss": self.last_eval_loss,
                "best_step": self.best_step,
                "wallclock_sec": elapsed,
                "qparam_final": self.current_qparam_id,
                "fd_true_available": False,
                "fd_true_exception_reason": "true-gradient diagnostics not run in default-h training batch",
            }
            write_json(self.run_dir / "run_summary.json", summary)
            manifest = json.loads((self.run_dir / "run_manifest_row.json").read_text(encoding="utf-8"))
            manifest.update({"status": summary["status"], "best_eval_acc": self.best_eval_acc, "last_eval_acc": self.last_eval_acc, "best_step": self.best_step})
            write_json(self.run_dir / "run_manifest_row.json", manifest)
        return json.loads((self.run_dir / "run_summary.json").read_text(encoding="utf-8"))


def write_smoke_summary(output_root: Path, run_dir: Path, ok: bool, reason: str = "") -> None:
    summary_path = output_root / "smoke_summary.md"
    if ok:
        text = "\n".join(
            [
                "# Smoke Summary",
                "",
                "Status: pass.",
                "",
                f"Run directory: `{run_dir}`",
                "",
                "Checked invariants:",
                "- official AutoAWQ package used",
                "- W4/G128 weight-only quantization",
                "- FP16 activations and FP16 master update",
                "- h=1e-3",
                "- K=10 smoke refresh at step 0 and step 10",
                "- plus/minus branches share qparam_id within every ZO pair",
                "- qparam_id is static between refreshes and changes at scheduled refresh",
                "- fresh-round codes are used for both branches",
                "- no independent plus/minus AutoAWQ quantization",
                "- no `Q(w_t)+/-hu` bypass",
                "- RandomSampler used",
                "- reconstruction check passed after each refresh",
            ]
        )
    else:
        text = f"# Smoke Summary\n\nStatus: fail.\n\nReason: {reason}\n"
    summary_path.write_text(text + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_root", default=str(OUT_ROOT))
    parser.add_argument("--model_id", default="facebook/opt-1.3b")
    parser.add_argument("--dataset", default="SST2")
    parser.add_argument("--dataset_mode", choices=["auto", "fewshot", "full"], default="full")
    parser.add_argument("--num_k", type=int, default=16)
    parser.add_argument("--num_train", type=int, default=-1)
    parser.add_argument("--num_eval", type=int, default=None)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--start_step", type=int, default=0)
    parser.add_argument("--init_model_path", default=None)
    parser.add_argument("--k_refresh", type=int, default=500)
    parser.add_argument("--h", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--eval_max_batches", type=int, default=0)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=1e-7)
    parser.add_argument("--seed", type=int, default=16)
    parser.add_argument("--data_seed", type=int, default=16)
    parser.add_argument("--calibration_size", type=int, default=128)
    parser.add_argument("--max_calib_seq_len", type=int, default=128)
    parser.add_argument("--n_parallel_calib_samples", type=int, default=8)
    parser.add_argument("--eval_steps", type=int, default=500)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--quant_diag_every", type=int, default=50)
    parser.add_argument("--save_awq_params", action="store_true")
    parser.add_argument("--static_awq_once", action="store_true")
    parser.add_argument("--static_qparams_path", default=None)
    parser.add_argument("--run_dir_name", default=None)
    parser.add_argument("--mode_label", default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if abs(float(args.h) - 1e-3) > 0.0:
        raise ValueError("This breadth runner is restricted to h=1e-3")
    if bool(args.smoke):
        args.dataset = "SST2"
        args.steps = 20
        args.k_refresh = 10
        args.eval_steps = 10
        args.save_steps = 10
        args.quant_diag_every = 1
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA/H100 is required")
    if "H100" not in torch.cuda.get_device_name(0):
        raise RuntimeError(f"H100 required; found {torch.cuda.get_device_name(0)}")
    trainer = AutoAWQSharedGridTrainer(args)
    try:
        trainer.train()
    except Exception as exc:
        if bool(args.smoke):
            write_smoke_summary(Path(args.output_root), trainer.run_dir, False, f"{type(exc).__name__}: {exc}")
        raise
    if bool(args.smoke):
        write_smoke_summary(Path(args.output_root), trainer.run_dir, True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
