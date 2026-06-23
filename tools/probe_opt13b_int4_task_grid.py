#!/usr/bin/env python
"""Task-conditioned OPT-1.3B INT4 RTNClip probe with formula h-star.

This is probe-only.  It does not train.  It evaluates the current low-bit
finite-difference window for OPT on task prompts, using the same shared-grid
RTNClip fake-quantized forward semantics as the existing OPT probe:

    grid_t = RTNClipGrid(w_t)
    d_h = [L(Q_t(w_t + h u)) - L(Q_t(w_t - h u))] / (2 h)

The default nMSE is the scalar finite-difference error:
    default_dh_vs_gTu = MSE(d_h, g^T u) / E[(g^T u)^2]

The h-star estimate uses the repo's retained simple2pt_corrected selector.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset as TorchDataset, RandomSampler


REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = REPO_ROOT / "tools"
LARGE_MODELS_DIR = REPO_ROOT / "large_models"
for path in (TOOLS_DIR, LARGE_MODELS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import smoke_rtnclip_roberta_sst5 as rtn  # noqa: E402
from analyze_int4_sst5_calibrated_hstar import (  # noqa: E402
    EPS,
    H_GRID as FORMULA_H_GRID,
    choose_l_plateau,
    direction_norm_sq,
    simple2pt_corrected,
)
from utils import DataCollatorWithPaddingAndNesting, encode_prompt, forward_wrap_with_option_len  # noqa: E402
from opt_mezo_option_tasks import TASK_TO_LARGE, get_option_task  # noqa: E402


TASK_ALIASES = {
    "sst2": "sst-2",
    "sst-2": "sst-2",
    "sst5": "sst-5",
    "sst-5": "sst-5",
    "rte": "rte",
    "mnli": "mnli",
    "trec": "trec",
}

SUMMARY_COLUMNS = [
    "task",
    "setting",
    "h",
    "k_dirs",
    "default_fd_true_nmse",
    "default_corr_fd_true",
    "d_h_mean",
    "d_true_mean",
    "d_h_abs_mean",
    "d_true_abs_mean",
    "active_frac_mean",
    "alignment_mean",
    "norm_ratio_mean",
    "delta_q_norm_mean",
    "ideal_displacement_norm_mean",
    "sparse_p",
    "mask_strategy",
    "perturb_scope",
    "status",
]


HSTAR_COLUMNS = [
    "task",
    "setting",
    "selector_name",
    "hstar_cont",
    "hstar_nearest_grid",
    "hstar_nearest_grid_label",
    "h_empirical_min_nmse",
    "h_empirical_min_nmse_corr_positive",
    "Delta_mode",
    "Delta_value",
    "delta_int4_rtnclip_scale_rms",
    "G_mode",
    "G_value",
    "G_lowbit_abs_h1e-4",
    "G_lowbit_abs_h3e-4",
    "G_lowbit_abs_h1e-3",
    "G_lowbit_abs_median_1e-4_3e-4_1e-3",
    "G_clean_abs",
    "L_mode",
    "L_hat",
    "h2_L",
    "L_selection_status",
    "L_clean32_q90",
    "L_clean32_h2",
    "L_clean32_selection_status",
    "L_lowbit_q90",
    "L_lowbit_h2",
    "L_lowbit_selection_status",
    "L_lowbit_over_clean32",
    "d_trainable",
    "d_quantized_linear",
    "sparse_p",
    "mask_strategy",
    "active_param_count",
    "active_frac",
    "notes",
]


@dataclass
class TaskBatch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor
    task: str
    num_examples: int
    split: str
    sample_ids: List[object]

    def as_inputs(self) -> Dict[str, torch.Tensor]:
        return {
            "input_ids": self.input_ids,
            "attention_mask": self.attention_mask,
            "labels": self.labels,
        }


def json_default(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.dtype):
        return str(value)
    return str(value)


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True, default=json_default) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: List[Dict[str, object]], columns: Optional[Sequence[str]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if columns is None:
        keys: List[str] = []
        for row in rows:
            for key in row:
                if key not in keys:
                    keys.append(key)
        columns = keys
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(columns), extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def append_jsonl(path: Path, row: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, sort_keys=True, default=json_default) + "\n")


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(REPO_ROOT),
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return ""


def env_info() -> Dict[str, object]:
    out: Dict[str, object] = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "hostname": socket.gethostname(),
        "python": sys.executable,
        "python_version": platform.python_version(),
        "conda_env": os.environ.get("CONDA_DEFAULT_ENV", ""),
        "git_commit": git_commit(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "DATALOADER_SHUFFLE": os.environ.get("DATALOADER_SHUFFLE", ""),
    }
    for module_name in ("transformers", "datasets", "accelerate"):
        try:
            module = __import__(module_name)
            out[f"{module_name}_version"] = getattr(module, "__version__", "")
        except Exception:
            out[f"{module_name}_version"] = None
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(torch.cuda.current_device())
        out["gpu_name"] = props.name
        out["gpu_total_memory_mb"] = int(props.total_memory / 1024 / 1024)
    return out


def parse_h_grid(raw: Sequence[str]) -> List[float]:
    vals: List[float] = []
    for item in raw:
        for part in str(item).replace(",", " ").split():
            vals.append(float(part))
    return vals


def finite_float(value: object) -> Optional[float]:
    try:
        out = float(value)  # type: ignore[arg-type]
    except Exception:
        return None
    return out if math.isfinite(out) else None


def corr(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    pairs = [(float(x), float(y)) for x, y in zip(xs, ys) if math.isfinite(float(x)) and math.isfinite(float(y))]
    if len(pairs) < 2:
        return None
    mx = sum(x for x, _ in pairs) / len(pairs)
    my = sum(y for _, y in pairs) / len(pairs)
    vx = sum((x - mx) ** 2 for x, _ in pairs)
    vy = sum((y - my) ** 2 for _, y in pairs)
    if vx <= 1e-30 or vy <= 1e-30:
        return None
    return sum((x - mx) * (y - my) for x, y in pairs) / math.sqrt(vx * vy)


def fmt(value: object) -> str:
    val = finite_float(value)
    return "NA" if val is None else f"{val:.6g}"


def normalize_task(task: str) -> str:
    key = str(task).strip().lower()
    if key not in TASK_ALIASES:
        raise ValueError(f"Unsupported task {task!r}; expected one of {sorted(TASK_ALIASES)}")
    return TASK_ALIASES[key]


class MeZOOptionPromptDataset(TorchDataset):
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


def patch_mezo_option_loss(model: nn.Module) -> None:
    if not hasattr(model, "original_forward"):
        model.original_forward = model.forward
        model.forward = forward_wrap_with_option_len.__get__(model, type(model))


def sample_id(sample: Any, fallback: int) -> str:
    sid = getattr(sample, "id", None)
    if sid is not None:
        return str(sid)
    data = getattr(sample, "data", None)
    if data is not None:
        return str(abs(hash(json.dumps(data, sort_keys=True, default=str))))
    return str(fallback)


def build_mezo_option_batch(args: argparse.Namespace, tokenizer: Any, task_name: str, device: torch.device) -> Tuple[Dict[str, Any], Dict[str, object]]:
    large_task_name = TASK_TO_LARGE.get(str(task_name).lower(), task_name)
    task = get_option_task(task_name)
    train_sets = task.sample_train_sets(
        num_train=int(args.num_train),
        num_dev=0,
        num_eval=None,
        num_train_sets=1,
        seed=int(args.data_seed),
        dataset_mode=str(args.dataset_mode),
        num_k=int(args.num_k),
    )
    samples = list(train_sets[0])
    dataset = MeZOOptionPromptDataset(samples, task, tokenizer, int(args.max_seq_len))
    generator = torch.Generator().manual_seed(int(args.data_seed))
    loader = DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        sampler=RandomSampler(dataset, generator=generator),
        collate_fn=DataCollatorWithPaddingAndNesting(tokenizer, pad_to_multiple_of=8),
    )
    batch = next(iter(loader))
    batch = {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}
    chosen = [dataset.samples[i] for i in list(iter(loader.sampler))[: int(args.batch_size)]] if hasattr(loader, "sampler") else []
    info = {
        "task_path": "mezo_option",
        "large_task_name": large_task_name,
        "dataset_mode": str(args.dataset_mode),
        "num_train": int(args.num_train),
        "num_k": int(args.num_k),
        "train_sample_count": len(samples),
        "batch_size": int(args.batch_size),
        "effective_candidate_rows": int(batch["input_ids"].shape[0]),
        "sample_ids": [sample_id(sample, i) for i, sample in enumerate(chosen)],
    }
    return batch, info


def label_h(h: float) -> str:
    if not math.isfinite(float(h)):
        return "nan"
    known = [
        (1e-8, "1e-8"),
        (1e-7, "1e-7"),
        (1e-6, "1e-6"),
        (1e-5, "1e-5"),
        (3e-5, "3e-5"),
        (1e-4, "1e-4"),
        (3e-4, "3e-4"),
        (1e-3, "1e-3"),
        (1.5e-3, "1p5e-3"),
        (2e-3, "2e-3"),
        (3e-3, "3e-3"),
        (4e-3, "4e-3"),
        (5e-3, "5e-3"),
        (1e-2, "1e-2"),
        (3e-2, "3e-2"),
        (1e-1, "1e-1"),
        (1.0, "1e0"),
    ]
    return min(known, key=lambda x: abs(math.log(float(h)) - math.log(x[0])))[1]


def load_task_examples(task: str, data_seed: int, batch_size: int) -> List[Dict[str, object]]:
    from datasets import load_dataset
    import numpy as np

    task = normalize_task(task)
    if task == "sst-2":
        ds = load_dataset("glue", "sst2")["train"]
        candidates = [" terrible", " great"]
        rows = [
            {
                "id": ex.get("idx", i),
                "prompt": f"{str(ex['sentence']).strip()} It was",
                "answer": candidates[int(ex["label"])],
                "label": int(ex["label"]),
                "candidates": candidates,
            }
            for i, ex in enumerate(ds)
        ]
    elif task == "sst-5":
        ds = load_dataset("SetFit/sst5")["train"]
        verbalizer = {0: " terrible", 1: " bad", 2: " okay", 3: " good", 4: " great"}
        candidates = [verbalizer[i] for i in range(5)]
        rows = [
            {
                "id": i,
                "prompt": f"{str(ex['text']).strip()} It was",
                "answer": verbalizer[int(ex["label"])],
                "label": int(ex["label"]),
                "candidates": candidates,
            }
            for i, ex in enumerate(ds)
        ]
    elif task == "rte":
        ds = load_dataset("super_glue", "rte")["train"]
        verbalizer = {0: " Yes", 1: " No"}
        candidates = [verbalizer[i] for i in range(2)]
        rows = [
            {
                "id": ex.get("idx", i),
                "prompt": f"{ex['premise']}\nDoes this mean that \"{ex['hypothesis']}\" is true? Yes or No?\n",
                "answer": verbalizer[int(ex["label"])],
                "label": int(ex["label"]),
                "candidates": candidates,
            }
            for i, ex in enumerate(ds)
        ]
    elif task == "mnli":
        ds = load_dataset("glue", "mnli")["train"]
        verbalizer = {0: " Yes", 1: " Maybe", 2: " No"}
        candidates = [verbalizer[i] for i in range(3)]
        rows = [
            {
                "id": ex.get("idx", i),
                "prompt": (
                    f"Premise: {ex['premise']}\n"
                    f"Hypothesis: {ex['hypothesis']}\n"
                    "Can we infer that the hypothesis is true? Yes, No, or Maybe?\n"
                ),
                "answer": verbalizer[int(ex["label"])],
                "label": int(ex["label"]),
                "candidates": candidates,
            }
            for i, ex in enumerate(ds)
        ]
    elif task == "trec":
        try:
            ds_all = load_dataset("SetFit/TREC-QC")
        except Exception:
            try:
                ds_all = load_dataset("CogComp/trec")
            except Exception:
                ds_all = load_dataset("trec")
        split = "train" if "train" in ds_all else next(iter(ds_all.keys()))
        ds = ds_all[split]
        verbalizer = {
            0: " abbreviation",
            1: " entity",
            2: " description",
            3: " human",
            4: " location",
            5: " number",
        }
        candidates = [verbalizer[i] for i in range(6)]
        rows = []
        for i, ex in enumerate(ds):
            label = int(ex.get("coarse_label", ex.get("label_coarse", ex.get("label-coarse", ex.get("label", 0)))))
            text = str(ex.get("text", ex.get("question", ""))).strip()
            rows.append(
                {
                    "id": i,
                    "prompt": f"Question: {text}\nAnswer type:",
                    "answer": verbalizer[label],
                    "label": label,
                    "candidates": candidates,
                }
            )
    else:
        raise ValueError(task)

    rng = np.random.default_rng(int(data_seed))
    indices = rng.permutation(len(rows)).tolist()[: max(1, min(int(batch_size), len(rows)))]
    return [rows[i] for i in indices]


def build_task_batch(tokenizer, task: str, examples: List[Dict[str, object]], max_seq_len: int, device: torch.device) -> TaskBatch:
    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = getattr(tokenizer, "eos_token", None) or tokenizer.unk_token
    pad_id = int(tokenizer.pad_token_id)
    input_rows: List[List[int]] = []
    label_rows: List[List[int]] = []
    sample_ids: List[object] = []
    for ex in examples:
        prompt = str(ex["prompt"])
        answer = str(ex["answer"])
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        answer_ids = tokenizer(answer, add_special_tokens=False)["input_ids"]
        if not answer_ids:
            answer_ids = tokenizer(" " + answer.strip(), add_special_tokens=False)["input_ids"]
        if len(answer_ids) >= max_seq_len:
            answer_ids = answer_ids[-max_seq_len:]
            prompt_ids = []
        else:
            keep_prompt = max_seq_len - len(answer_ids)
            prompt_ids = prompt_ids[-keep_prompt:]
        input_ids = prompt_ids + answer_ids
        labels = [-100] * len(prompt_ids) + list(answer_ids)
        input_rows.append(input_ids)
        label_rows.append(labels)
        sample_ids.append(ex.get("id", ""))
    max_len = max(len(x) for x in input_rows)
    padded_inputs = []
    padded_labels = []
    masks = []
    for ids, labels in zip(input_rows, label_rows):
        pad = max_len - len(ids)
        padded_inputs.append(ids + [pad_id] * pad)
        padded_labels.append(labels + [-100] * pad)
        masks.append([1] * len(ids) + [0] * pad)
    return TaskBatch(
        input_ids=torch.tensor(padded_inputs, dtype=torch.long, device=device),
        attention_mask=torch.tensor(masks, dtype=torch.long, device=device),
        labels=torch.tensor(padded_labels, dtype=torch.long, device=device),
        task=task,
        num_examples=len(input_rows),
        split="train",
        sample_ids=sample_ids,
    )


def load_model_and_tokenizer(args: argparse.Namespace, device: torch.device):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, local_files_only=args.local_files_only)
    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = getattr(tokenizer, "eos_token", None) or tokenizer.unk_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        local_files_only=args.local_files_only,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
    )
    model.config.use_cache = False
    model.eval()
    model.to(device)
    return model, tokenizer


def params_map(model: nn.Module) -> Dict[str, nn.Parameter]:
    return {name: p for name, p in model.named_parameters() if p.detach().is_floating_point()}


def linear_weight_names(model: nn.Module, params: Dict[str, nn.Parameter]) -> List[str]:
    names: List[str] = []
    for module_name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            name = f"{module_name}.weight" if module_name else "weight"
            if name in params and params[name].ndim == 2:
                names.append(name)
    return names


def make_master(params: Dict[str, nn.Parameter], dtype: torch.dtype) -> Dict[str, torch.Tensor]:
    return {name: p.detach().clone().to(device=p.device, dtype=dtype) for name, p in params.items()}


def restore_master(params: Dict[str, nn.Parameter], master: Dict[str, torch.Tensor]) -> None:
    with torch.no_grad():
        for name, tensor in master.items():
            params[name].copy_(tensor.to(dtype=params[name].dtype))


def refresh_states(master: Dict[str, torch.Tensor], q_names: Iterable[str], bitwidth: int, group_size: int):
    states: Dict[str, rtn.RTNClipState] = {}
    rows: List[Dict[str, object]] = []
    for name in q_names:
        state, stats = rtn.compute_rtnclip_state(name, master[name], bitwidth, group_size)
        states[name] = state
        rows.append(stats)
    return states, rows


def apply_values(
    params: Dict[str, nn.Parameter],
    master: Dict[str, torch.Tensor],
    directions: Optional[Dict[str, torch.Tensor]],
    states: Optional[Dict[str, rtn.RTNClipState]],
    h: float,
    sign: float,
) -> None:
    with torch.no_grad():
        for name, tensor in master.items():
            value = tensor.float()
            if directions is not None and name in directions:
                value = value.add(directions[name].float(), alpha=float(sign) * float(h))
            if states is not None and name in states:
                value = rtn.quantize_with_state(value, states[name])
            params[name].copy_(value.to(dtype=params[name].dtype))


def forward_loss(model: nn.Module, batch: Any) -> torch.Tensor:
    if hasattr(batch, "as_inputs"):
        return model(**batch.as_inputs()).loss
    return model(**batch, return_dict=True).loss


def loss_value(model: nn.Module, batch: Any) -> float:
    return float(forward_loss(model, batch).detach().cpu())


def compute_true_gradient(
    model: nn.Module,
    params: Dict[str, nn.Parameter],
    master: Dict[str, torch.Tensor],
    batch: Any,
    states: Optional[Dict[str, rtn.RTNClipState]] = None,
) -> float:
    model.zero_grad(set_to_none=True)
    restore_master(params, master)
    if states is not None:
        apply_values(params, master, None, states, 0.0, 0.0)
    loss = forward_loss(model, batch)
    loss.backward()
    return float(loss.detach().cpu())


def build_highest_abs_masks(
    master: Dict[str, torch.Tensor],
    names: Sequence[str],
    sparse_p: float,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, object]]:
    masks: Dict[str, torch.Tensor] = {}
    active = 0
    total = 0
    for name in names:
        tensor = master[name]
        n = int(tensor.numel())
        total += n
        if sparse_p >= 1.0:
            mask = torch.ones_like(tensor, dtype=torch.bool)
        else:
            k = max(1, int(math.floor(float(sparse_p) * n)))
            flat = tensor.detach().abs().float().reshape(-1)
            if k >= n:
                mask = torch.ones_like(tensor, dtype=torch.bool)
            else:
                threshold = torch.kthvalue(flat, n - k + 1).values.to(device=tensor.device, dtype=flat.dtype)
                mask = tensor.detach().abs() >= threshold
        masks[name] = mask
        active += int(mask.sum().detach().cpu())
    return masks, {
        "mask_strategy": "highest_abs_per_tensor",
        "sparse_p": float(sparse_p),
        "active_param_count": active,
        "total_param_count": total,
        "active_frac": active / max(total, 1),
    }


def sample_direction(
    master: Dict[str, torch.Tensor],
    names: Sequence[str],
    seed: int,
    masks: Optional[Dict[str, torch.Tensor]] = None,
) -> Dict[str, torch.Tensor]:
    first = master[names[0]]
    gen = torch.Generator(device=first.device).manual_seed(int(seed))
    out: Dict[str, torch.Tensor] = {}
    for name in names:
        z = torch.randn(master[name].shape, device=first.device, generator=gen, dtype=master[name].dtype)
        if masks is not None:
            z = z * masks[name].to(device=first.device, dtype=z.dtype)
        out[name] = z
    return out


def grad_dot_direction(params: Dict[str, nn.Parameter], directions: Dict[str, torch.Tensor]) -> float:
    total = torch.zeros((), device=next(iter(params.values())).device, dtype=torch.float64)
    for name, direction in directions.items():
        grad = params[name].grad
        if grad is None:
            continue
        total += (grad.detach().double() * direction.double()).sum()
    return float(total.detach().cpu())


def finite_difference(
    model: nn.Module,
    params: Dict[str, nn.Parameter],
    master: Dict[str, torch.Tensor],
    batch: Any,
    states: Dict[str, rtn.RTNClipState],
    directions: Dict[str, torch.Tensor],
    h: float,
) -> Tuple[float, float, float]:
    with torch.no_grad():
        apply_values(params, master, directions, states, h, +1.0)
        loss_plus = loss_value(model, batch)
        apply_values(params, master, directions, states, h, -1.0)
        loss_minus = loss_value(model, batch)
        restore_master(params, master)
    return loss_plus, loss_minus, (loss_plus - loss_minus) / (2.0 * float(h))


def visibility_metrics(
    master: Dict[str, torch.Tensor],
    directions: Dict[str, torch.Tensor],
    states: Dict[str, rtn.RTNClipState],
    h: float,
) -> Dict[str, float]:
    active = 0
    total = 0
    dot = torch.zeros((), device=next(iter(master.values())).device, dtype=torch.float64)
    delta_sq = torch.zeros_like(dot)
    intended_sq = torch.zeros_like(dot)
    for name, state in states.items():
        if name not in directions:
            continue
        direction = directions[name].float()
        intended = 2.0 * float(h) * direction
        plus = rtn.quantize_with_state(master[name].float().add(direction, alpha=float(h)), state)
        minus = rtn.quantize_with_state(master[name].float().add(direction, alpha=-float(h)), state)
        delta = plus.float() - minus.float()
        active += int((delta != 0).sum().detach().cpu())
        total += int(delta.numel())
        dot += (delta.double() * intended.double()).sum()
        delta_sq += delta.double().square().sum()
        intended_sq += intended.double().square().sum()
    eps = 1e-30
    return {
        "active_frac": active / max(total, 1),
        "alignment": float((dot / (delta_sq.sqrt() * intended_sq.sqrt() + eps)).detach().cpu()) if float(intended_sq.detach().cpu()) > 0 else float("nan"),
        "norm_ratio": float((delta_sq.sqrt() / intended_sq.sqrt().clamp_min(eps)).detach().cpu()) if float(intended_sq.detach().cpu()) > 0 else float("nan"),
        "delta_q_norm": float(delta_sq.sqrt().detach().cpu()),
        "ideal_displacement_norm": float(intended_sq.sqrt().detach().cpu()),
    }


def grouped_mask_counts(mask: torch.Tensor, group_size: int) -> torch.Tensor:
    if mask.ndim != 2:
        raise ValueError(f"Expected 2D mask, got {tuple(mask.shape)}")
    out_features, in_features = mask.shape
    num_groups = int(math.ceil(in_features / group_size))
    padded_cols = num_groups * group_size
    pad_cols = padded_cols - in_features
    m = mask.to(dtype=torch.float32)
    if pad_cols:
        import torch.nn.functional as F

        m = F.pad(m, (0, pad_cols))
    return m.reshape(out_features, num_groups, group_size).sum(dim=-1).double().unsqueeze(-1)


def weighted_delta_with_optional_masks(
    states: Dict[str, rtn.RTNClipState],
    masks: Optional[Dict[str, torch.Tensor]],
) -> Dict[str, float]:
    if masks is None:
        from analyze_int4_sst5_calibrated_hstar import weighted_int4_delta

        return weighted_int4_delta(states)
    scale_sq_sum = 0.0
    scale_sum = 0.0
    values = 0.0
    flat_scales: List[torch.Tensor] = []
    for name, state in states.items():
        mask = masks.get(name)
        if mask is None:
            continue
        counts = grouped_mask_counts(mask, state.group_size)
        scales = state.scales.double()
        scale_sq_sum += float((scales.square() * counts).sum().detach().cpu())
        scale_sum += float((scales * counts).sum().detach().cpu())
        values += float(counts.sum().detach().cpu())
        selected = state.scales.detach().float()[counts.squeeze(-1).bool()]
        if selected.numel():
            flat_scales.append(selected.reshape(-1).cpu())
    all_scales = torch.cat(flat_scales) if flat_scales else torch.empty(0)
    denom = max(values, 1.0)
    return {
        "delta_int4_rtnclip_scale_rms": math.sqrt(scale_sq_sum / denom),
        "delta_int4_rtnclip_scale_mean": scale_sum / denom,
        "scale_median_unweighted": float(all_scales.median()) if all_scales.numel() else float("nan"),
        "scale_p90_unweighted": float(torch.quantile(all_scales, 0.90)) if all_scales.numel() else float("nan"),
        "scale_p95_unweighted": float(torch.quantile(all_scales, 0.95)) if all_scales.numel() else float("nan"),
        "num_quantized_values_for_delta": int(values),
    }


def clean_second_diff_l(
    model: nn.Module,
    params: Dict[str, nn.Parameter],
    master32: Dict[str, torch.Tensor],
    batch: TaskBatch,
    direction_names: Sequence[str],
    masks: Optional[Dict[str, torch.Tensor]],
    *,
    seed_base: int,
    h2_grid: Sequence[float],
    m_l: int,
) -> Tuple[Dict[str, object], str, List[Dict[str, object]]]:
    model.float()
    apply_values(params, master32, None, None, 0.0, 0.0)
    with torch.no_grad():
        base_loss = loss_value(model, batch)
    rows: List[Dict[str, object]] = []
    old_tf32_matmul = torch.backends.cuda.matmul.allow_tf32
    old_tf32_cudnn = torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        for h2 in h2_grid:
            lambdas: List[float] = []
            ks: List[float] = []
            for i in range(int(m_l)):
                directions = sample_direction(master32, direction_names, seed_base + i * 1009 + 88000, masks=masks)
                apply_values(params, master32, directions, None, float(h2), 1.0)
                l1 = loss_value(model, batch)
                apply_values(params, master32, directions, None, 2.0 * float(h2), 1.0)
                l2 = loss_value(model, batch)
                apply_values(params, master32, None, None, 0.0, 0.0)
                k = (l2 - 2.0 * l1 + base_loss) / (float(h2) * float(h2))
                norm_sq = direction_norm_sq(directions)
                lambdas.append(abs(k) / (norm_sq + EPS))
                ks.append(k)
            t = torch.tensor(lambdas, dtype=torch.float64)
            kt = torch.tensor(ks, dtype=torch.float64)
            med = torch.median(kt)
            mad = torch.median((kt - med).abs())
            rows.append(
                {
                    "h2": float(h2),
                    "lambda_q50": float(torch.quantile(t, 0.50)),
                    "lambda_q90": float(torch.quantile(t, 0.90)),
                    "lambda_q95": float(torch.quantile(t, 0.95)),
                    "median_abs_K": float(torch.median(kt.abs())),
                    "MAD_K": float(mad),
                    "SNR2": float(torch.median(kt.abs()) / (1.4826 * mad + EPS)),
                    "finite_rate": float(torch.isfinite(t).float().mean()),
                }
            )
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_tf32_matmul
        torch.backends.cudnn.allow_tf32 = old_tf32_cudnn
        model.half()
        restore_master(params, {k: v.to(dtype=torch.float16) for k, v in master32.items()})
    selected, status = choose_l_plateau(rows)
    return selected, status, rows


def lowbit_second_diff_l(
    model: nn.Module,
    params: Dict[str, nn.Parameter],
    master: Dict[str, torch.Tensor],
    batch: Any,
    direction_names: Sequence[str],
    masks: Optional[Dict[str, torch.Tensor]],
    states: Dict[str, rtn.RTNClipState],
    *,
    seed_base: int,
    h2_grid: Sequence[float],
    m_l: int,
) -> Tuple[Dict[str, object], str, List[Dict[str, object]]]:
    """Estimate L with the same low-bit shared-grid forward oracle as probes.

    This mirrors the RoBERTa INT4 full-data estimator: the quantization grid is
    computed once from the unperturbed FP16 master, then w, w+h2*u, and
    w+2*h2*u are evaluated through that cached low-bit grid.
    """
    apply_values(params, master, None, states, 0.0, 0.0)
    with torch.no_grad():
        base_loss = loss_value(model, batch)
    rows: List[Dict[str, object]] = []
    for h2 in h2_grid:
        lambdas: List[float] = []
        ks: List[float] = []
        for i in range(int(m_l)):
            directions = sample_direction(master, direction_names, seed_base + i * 1009 + 88000, masks=masks)
            with torch.no_grad():
                apply_values(params, master, directions, states, float(h2), 1.0)
                l1 = loss_value(model, batch)
                apply_values(params, master, directions, states, 2.0 * float(h2), 1.0)
                l2 = loss_value(model, batch)
                restore_master(params, master)
            k = (l2 - 2.0 * l1 + base_loss) / (float(h2) * float(h2))
            norm_sq = direction_norm_sq(directions)
            lambdas.append(abs(k) / (norm_sq + EPS))
            ks.append(k)
        t = torch.tensor(lambdas, dtype=torch.float64)
        kt = torch.tensor(ks, dtype=torch.float64)
        med = torch.median(kt)
        mad = torch.median((kt - med).abs())
        rows.append(
            {
                "h2": float(h2),
                "lambda_q50": float(torch.quantile(t, 0.50)),
                "lambda_q90": float(torch.quantile(t, 0.90)),
                "lambda_q95": float(torch.quantile(t, 0.95)),
                "median_abs_K": float(torch.median(kt.abs())),
                "MAD_K": float(mad),
                "SNR2": float(torch.median(kt.abs()) / (1.4826 * mad + EPS)),
                "finite_rate": float(torch.isfinite(t).float().mean()),
                "L_oracle": "rtnclip_int4_shared_grid_forward_second_diff",
            }
        )
    restore_master(params, master)
    selected, status = choose_l_plateau(rows)
    return selected, status, rows


def aggregate(records: List[Dict[str, object]], h_grid: Sequence[float]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    keys = sorted({(str(r["task"]), str(r["setting"])) for r in records})
    for task, setting in keys:
        for h in h_grid:
            group = [
                r
                for r in records
                if str(r["task"]) == task and str(r["setting"]) == setting and abs(float(r["h"]) - float(h)) <= 1e-15
            ]
            if not group:
                continue
            dh = [float(r["d_h"]) for r in group if finite_float(r.get("d_h")) is not None]
            dt = [float(r["d_true"]) for r in group if finite_float(r.get("d_true")) is not None]
            n = min(len(dh), len(dt))
            dh = dh[:n]
            dt = dt[:n]
            mse = sum((a - b) ** 2 for a, b in zip(dh, dt)) / max(n, 1)
            ref = sum(b ** 2 for b in dt) / max(n, 1)
            row = {
                "task": task,
                "setting": setting,
                "h": h,
                "k_dirs": n,
                "default_fd_true_nmse": mse / max(ref, 1e-30),
                "default_corr_fd_true": corr(dh, dt),
                "d_h_mean": sum(dh) / max(n, 1),
                "d_true_mean": sum(dt) / max(n, 1),
                "d_h_abs_mean": sum(abs(x) for x in dh) / max(n, 1),
                "d_true_abs_mean": sum(abs(x) for x in dt) / max(n, 1),
                "active_frac_mean": mean_key(group, "active_frac"),
                "alignment_mean": mean_key(group, "alignment"),
                "norm_ratio_mean": mean_key(group, "norm_ratio"),
                "delta_q_norm_mean": mean_key(group, "delta_q_norm"),
                "ideal_displacement_norm_mean": mean_key(group, "ideal_displacement_norm"),
                "sparse_p": group[0].get("sparse_p", ""),
                "mask_strategy": group[0].get("mask_strategy", ""),
                "perturb_scope": group[0].get("perturb_scope", ""),
                "status": "complete",
            }
            rows.append(row)
    return rows


def mean_key(rows: Sequence[Dict[str, object]], key: str) -> Optional[float]:
    vals = [float(r[key]) for r in rows if finite_float(r.get(key)) is not None]
    return sum(vals) / len(vals) if vals else None


def hstar_from_summary(
    task: str,
    setting: str,
    summary_rows: List[Dict[str, object]],
    states: Dict[str, rtn.RTNClipState],
    masks: Optional[Dict[str, torch.Tensor]],
    mask_stats: Dict[str, object],
    q_names: Sequence[str],
    l_clean_selected: Dict[str, object],
    l_clean_status: str,
    l_lowbit_selected: Dict[str, object],
    l_lowbit_status: str,
    l_mode: str,
    d_trainable: int,
    d_quantized_linear: int,
) -> Dict[str, object]:
    rows = [r for r in summary_rows if str(r["task"]) == task and str(r["setting"]) == setting]
    by_h = {float(r["h"]): r for r in rows}
    lowbit_vals = []
    lowbit_by_h: Dict[float, float] = {}
    for h in (1e-4, 3e-4, 1e-3):
        row = by_h.get(h)
        if row and finite_float(row.get("d_h_abs_mean")) is not None:
            g_abs = math.sqrt(math.pi / 2.0) * float(row["d_h_abs_mean"])
            lowbit_by_h[h] = g_abs
            lowbit_vals.append(g_abs)
    lowbit_g_median = sorted(lowbit_vals)[len(lowbit_vals) // 2] if lowbit_vals else float("nan")
    clean_vals = []
    for h in (1e-4, 3e-4, 1e-3):
        row = by_h.get(h)
        if row and finite_float(row.get("d_true_abs_mean")) is not None:
            clean_vals.append(math.sqrt(math.pi / 2.0) * float(row["d_true_abs_mean"]))
    clean_g = sorted(clean_vals)[len(clean_vals) // 2] if clean_vals else float("nan")

    delta_stats = weighted_delta_with_optional_masks(states, masks)
    delta_scale = float(delta_stats["delta_int4_rtnclip_scale_rms"])
    use_lowbit_l = str(l_mode).lower() == "lowbit"
    l_selected = l_lowbit_selected if use_lowbit_l else l_clean_selected
    l_status = l_lowbit_status if use_lowbit_l else l_clean_status
    l_hat = float(l_selected.get("lambda_q90", float("nan")))
    l_clean_hat = float(l_clean_selected.get("lambda_q90", float("nan")))
    l_lowbit_hat = float(l_lowbit_selected.get("lambda_q90", float("nan")))
    selected_g = lowbit_g_median if math.isfinite(lowbit_g_median) and lowbit_g_median > 0.0 else clean_g
    selected_g_mode = (
        "rtnclip_int4_shared_grid_absG_median_1e-4_3e-4_1e-3"
        if math.isfinite(lowbit_g_median) and lowbit_g_median > 0.0
        else "clean32_absG_median_1e-4_3e-4_1e-3_fallback"
    )
    corrected = simple2pt_corrected(
        "int4",
        int(d_trainable),
        l_hat,
        scale_rms=delta_scale,
        clean32_g_median=clean_g,
        clean32_g_h3e4=clean_g,
        selected_g=selected_g,
        selected_g_mode=selected_g_mode,
    )
    valid = [r for r in rows if finite_float(r.get("default_fd_true_nmse")) is not None]
    best_any = min(valid, key=lambda r: float(r["default_fd_true_nmse"])) if valid else {}
    corr_pos = [r for r in valid if finite_float(r.get("default_corr_fd_true")) is not None and float(r["default_corr_fd_true"]) > 0.0]
    best_corr = min(corr_pos, key=lambda r: float(r["default_fd_true_nmse"])) if corr_pos else {}
    return {
        "task": task,
        "setting": setting,
        "selector_name": corrected["selector_name"],
        "hstar_cont": corrected["hstar_cont"],
        "hstar_nearest_grid": corrected["hstar_nearest_grid"],
        "hstar_nearest_grid_label": label_h(float(corrected["hstar_nearest_grid"])),
        "h_empirical_min_nmse": best_any.get("h", ""),
        "h_empirical_min_nmse_corr_positive": best_corr.get("h", ""),
        "Delta_mode": corrected["Delta_mode"],
        "Delta_value": corrected["Delta_value"],
        "delta_int4_rtnclip_scale_rms": delta_scale,
        "G_mode": corrected["G_mode"],
        "G_value": corrected["G_value"],
        "G_lowbit_abs_h1e-4": lowbit_by_h.get(1e-4, ""),
        "G_lowbit_abs_h3e-4": lowbit_by_h.get(3e-4, ""),
        "G_lowbit_abs_h1e-3": lowbit_by_h.get(1e-3, ""),
        "G_lowbit_abs_median_1e-4_3e-4_1e-3": lowbit_g_median,
        "G_clean_abs": clean_g,
        "L_mode": "L_lowbit_q90" if use_lowbit_l else "L_clean32",
        "L_hat": l_hat,
        "h2_L": l_selected.get("h2", ""),
        "L_selection_status": l_status,
        "L_clean32_q90": l_clean_hat,
        "L_clean32_h2": l_clean_selected.get("h2", ""),
        "L_clean32_selection_status": l_clean_status,
        "L_lowbit_q90": l_lowbit_hat,
        "L_lowbit_h2": l_lowbit_selected.get("h2", ""),
        "L_lowbit_selection_status": l_lowbit_status,
        "L_lowbit_over_clean32": l_lowbit_hat / l_clean_hat if math.isfinite(l_lowbit_hat) and math.isfinite(l_clean_hat) and l_clean_hat > 0.0 else "",
        "d_trainable": int(d_trainable),
        "d_quantized_linear": int(d_quantized_linear),
        "sparse_p": mask_stats.get("sparse_p", ""),
        "mask_strategy": mask_stats.get("mask_strategy", ""),
        "active_param_count": mask_stats.get("active_param_count", d_trainable),
        "active_frac": mask_stats.get("active_frac", 1.0),
        "quantized_linear_param_count": sum(int(states[name].shape[0] * states[name].shape[1]) for name in q_names if name in states),
        "notes": corrected["notes"],
        **delta_stats,
    }


def write_report(output_dir: Path, summary_rows: List[Dict[str, object]], hstar_rows: List[Dict[str, object]], config: Dict[str, object]) -> None:
    lines = [
        "# OPT-1.3B INT4 Task Probe",
        "",
        "Probe-only run. Dense and sparse p=0.01 use INT4 G128 RTNClip shared-grid fake quantized Linear weights.",
        "",
        f"- model: `{config['model_id']}`",
        f"- tasks: `{config['tasks']}`",
        f"- settings: `{config['settings']}`",
        f"- h grid: `{config['h_grid']}`",
        f"- probe directions per h: `{config['k_dirs']}`",
        f"- h-star selector: `simple2pt_corrected`",
        "",
        "## h-star",
        "",
        "| task | setting | hstar_cont | nearest | empirical min nMSE | corr-positive min | G | L | Delta |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in hstar_rows:
        lines.append(
            f"| {row['task']} | {row['setting']} | {fmt(row['hstar_cont'])} | {row['hstar_nearest_grid_label']} | "
            f"{fmt(row['h_empirical_min_nmse'])} | {fmt(row['h_empirical_min_nmse_corr_positive'])} | "
            f"{fmt(row['G_value'])} | {fmt(row['L_hat'])} | {fmt(row['Delta_value'])} |"
        )
    lines.extend(
        [
            "",
            "## Probe Summary",
            "",
            "| task | setting | h | nMSE default_dh_vs_gTu | corr | active | alignment | norm_ratio |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in summary_rows:
        lines.append(
            f"| {row['task']} | {row['setting']} | {float(row['h']):.6g} | {fmt(row['default_fd_true_nmse'])} | "
            f"{fmt(row['default_corr_fd_true'])} | {fmt(row['active_frac_mean'])} | "
            f"{fmt(row['alignment_mean'])} | {fmt(row['norm_ratio_mean'])} |"
        )
    output_dir.joinpath("summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_task_setting(
    *,
    task: str,
    setting: str,
    model: nn.Module,
    tokenizer,
    params: Dict[str, nn.Parameter],
    q_names: Sequence[str],
    output_dir: Path,
    args: argparse.Namespace,
    device: torch.device,
) -> Tuple[List[Dict[str, object]], Dict[str, object], List[Dict[str, object]]]:
    if args.task_path == "mezo_option":
        batch, batch_info = build_mezo_option_batch(args, tokenizer, task, device)
        batch_sample_ids = ";".join(str(x) for x in batch_info.get("sample_ids", []))
    else:
        examples = load_task_examples(task, args.data_seed, args.batch_size)
        batch = build_task_batch(tokenizer, task, examples, args.max_seq_len, device)
        batch_info = {"task_path": "simple_lm_answer", "examples": examples, "sample_ids": batch.sample_ids}
        batch_sample_ids = ";".join(str(x) for x in batch.sample_ids)
    master = make_master(params, torch.float16)
    master32 = make_master(params, torch.float32)
    restore_master(params, master)
    states, q_rows = refresh_states(master, q_names, int(args.bitwidth), int(args.group_size))
    qstats = rtn.aggregate_quantizer_stats(q_rows, {name: int(params[name].numel()) for name in q_names})

    perturb_names = list(master.keys())
    masks: Optional[Dict[str, torch.Tensor]] = None
    mask_stats: Dict[str, object] = {
        "sparse_p": "",
        "mask_strategy": "",
        "active_param_count": sum(int(t.numel()) for t in master.values()),
        "total_param_count": sum(int(t.numel()) for t in master.values()),
        "active_frac": 1.0,
    }
    if setting == "sparse_p0p01":
        masks, mask_stats = build_highest_abs_masks(master, perturb_names, args.sparse_p)
    elif setting != "dense":
        raise ValueError(f"Unsupported setting {setting!r}")

    print(f"[{datetime.now().isoformat(timespec='seconds')}] {task} {setting}: clean true gradient", flush=True)
    clean_loss = compute_true_gradient(model, params, master, batch, states=None)
    records: List[Dict[str, object]] = []
    task_records_path = output_dir / task / setting / "probe_records.jsonl"
    if task_records_path.exists():
        task_records_path.unlink()
    for direction_id in range(args.k_dirs):
        directions = sample_direction(master, perturb_names, args.seed + direction_id * 1009 + 5000, masks=masks)
        d_true = grad_dot_direction(params, directions)
        for h in args.h_grid:
            loss_plus, loss_minus, d_h = finite_difference(model, params, master, batch, states, directions, float(h))
            vis = visibility_metrics(master, directions, states, float(h))
            record = {
                "task": task,
                "setting": setting,
                "h": float(h),
                "direction_id": int(direction_id),
                "loss_plus": loss_plus,
                "loss_minus": loss_minus,
                "d_h": d_h,
                "d_true": d_true,
                "fd_true_error": d_h - d_true,
                "clean_loss": clean_loss,
                "sparse_p": mask_stats.get("sparse_p", ""),
                "mask_strategy": mask_stats.get("mask_strategy", ""),
                "perturb_scope": "all_floating_parameters",
                "batch_sample_ids": batch_sample_ids,
                **vis,
            }
            append_jsonl(task_records_path, record)
            append_jsonl(output_dir / "probe_records.jsonl", record)
            records.append(record)
        if (direction_id + 1) % max(1, args.progress_every) == 0:
            print(f"  {task} {setting}: direction {direction_id + 1}/{args.k_dirs}", flush=True)

    summary_rows = aggregate(records, args.h_grid)
    write_csv(output_dir / task / setting / "summary.csv", summary_rows, SUMMARY_COLUMNS)

    print(f"[{datetime.now().isoformat(timespec='seconds')}] {task} {setting}: clean L for h-star", flush=True)
    l_clean_selected, l_clean_status, l_rows = clean_second_diff_l(
        model,
        params,
        master32,
        batch,
        perturb_names,
        masks,
        seed_base=args.seed,
        h2_grid=args.hstar_l_grid,
        m_l=args.hstar_m_l,
    )
    write_csv(output_dir / task / setting / "L_clean32_candidates.csv", l_rows)
    print(f"[{datetime.now().isoformat(timespec='seconds')}] {task} {setting}: low-bit L for h-star", flush=True)
    l_lowbit_selected, l_lowbit_status, lowbit_l_rows = lowbit_second_diff_l(
        model,
        params,
        master,
        batch,
        perturb_names,
        masks,
        states,
        seed_base=args.seed,
        h2_grid=args.hstar_l_grid,
        m_l=args.hstar_m_l,
    )
    write_csv(output_dir / task / setting / "L_lowbit_candidates.csv", lowbit_l_rows)
    hstar_row = hstar_from_summary(
        task,
        setting,
        summary_rows,
        states,
        masks,
        mask_stats,
        q_names,
        l_clean_selected,
        l_clean_status,
        l_lowbit_selected,
        l_lowbit_status,
        args.hstar_l_mode,
        int(mask_stats["active_param_count"]),
        sum(int(masks[name].sum().detach().cpu()) for name in q_names if masks and name in masks)
        if masks is not None
        else sum(int(master[name].numel()) for name in q_names),
    )
    hstar_row.update({"quantizer_summary": qstats})
    write_json(output_dir / task / setting / "hstar_summary.json", hstar_row)
    write_csv(output_dir / task / setting / "hstar_summary.csv", [hstar_row], HSTAR_COLUMNS)
    write_json(output_dir / task / setting / "batch_info.json", {"task": task, **batch_info})
    del master, master32, states, masks
    torch.cuda.empty_cache()
    return records, hstar_row, summary_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", default="facebook/opt-1.3b")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--tasks", nargs="+", default=["sst-2", "sst-5", "rte", "mnli", "trec"])
    parser.add_argument("--settings", nargs="+", choices=["dense", "sparse_p0p01"], default=["dense", "sparse_p0p01"])
    parser.add_argument("--h_grid", nargs="+", default=["1e-5", "3e-5", "1e-4", "3e-4", "1e-3", "3e-3", "1e-2", "3e-2", "1e-1"])
    parser.add_argument("--k_dirs", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--task_path", choices=["simple_lm_answer", "mezo_option"], default="simple_lm_answer")
    parser.add_argument("--dataset_mode", choices=["auto", "fewshot", "full"], default="full")
    parser.add_argument("--num_train", type=int, default=-1)
    parser.add_argument("--num_k", type=int, default=16)
    parser.add_argument("--max_seq_len", type=int, default=128)
    parser.add_argument("--group_size", type=int, default=128)
    parser.add_argument("--bitwidth", type=int, default=4)
    parser.add_argument("--seed", type=int, default=16)
    parser.add_argument("--data_seed", type=int, default=16)
    parser.add_argument("--sparse_p", type=float, default=0.01)
    parser.add_argument("--hstar_m_l", type=int, default=16)
    parser.add_argument("--hstar_l_grid", nargs="+", default=[str(x) for x in FORMULA_H_GRID])
    parser.add_argument("--hstar_l_mode", choices=["clean32", "lowbit"], default="clean32")
    parser.add_argument("--progress_every", type=int, default=8)
    parser.add_argument("--local_files_only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.tasks = [normalize_task(t) for t in args.tasks]
    args.h_grid = parse_h_grid(args.h_grid)
    args.hstar_l_grid = parse_h_grid(args.hstar_l_grid)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if (output_dir / "probe_records.jsonl").exists():
        (output_dir / "probe_records.jsonl").unlink()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for OPT-1.3B task probes.")
    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    write_json(output_dir / "env.json", env_info())

    config = {
        "model_id": args.model_id,
        "tasks": args.tasks,
        "settings": args.settings,
        "h_grid": args.h_grid,
        "k_dirs": args.k_dirs,
        "batch_size": args.batch_size,
        "task_path": args.task_path,
        "dataset_mode": args.dataset_mode,
        "num_train": args.num_train,
        "num_k": args.num_k,
        "max_seq_len": args.max_seq_len,
        "bitwidth": args.bitwidth,
        "group_size": args.group_size,
        "seed": args.seed,
        "data_seed": args.data_seed,
        "quantizer": "INT4_G128_RTNClip_shared_grid_fake_quant",
        "scale_source": "unperturbed_fp16_master_weight",
        "pair_shared_grid": True,
        "fresh_round_codes": True,
        "default_nmse_metric": "default_dh_vs_gTu",
        "hstar_selector": "simple2pt_corrected",
        "hstar_G_source": "probe d_h abs median over 1e-4/3e-4/1e-3",
        "hstar_L_source": f"{args.hstar_l_mode} q90 via choose_l_plateau",
        "hstar_l_mode": args.hstar_l_mode,
        "sparse_rescale": "none",
        "sparse_mask_strategy": "highest_abs_per_tensor",
    }
    write_json(output_dir / "run_config.json", config)

    start = time.time()
    model, tokenizer = load_model_and_tokenizer(args, device)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id
    if args.task_path == "mezo_option":
        patch_mezo_option_loss(model)
    params = params_map(model)
    q_names = linear_weight_names(model, params)
    config["quantized_modules"] = len(q_names)
    config["quantized_module_names"] = q_names
    write_json(output_dir / "run_config.json", config)

    all_records: List[Dict[str, object]] = []
    all_summary_rows: List[Dict[str, object]] = []
    hstar_rows: List[Dict[str, object]] = []
    failures: List[Dict[str, object]] = []
    for task in args.tasks:
        for setting in args.settings:
            try:
                records, hstar_row, summary_rows = run_task_setting(
                    task=task,
                    setting=setting,
                    model=model,
                    tokenizer=tokenizer,
                    params=params,
                    q_names=q_names,
                    output_dir=output_dir,
                    args=args,
                    device=device,
                )
                all_records.extend(records)
                all_summary_rows.extend(summary_rows)
                hstar_rows.append(hstar_row)
                print(
                    f"[DONE] {task} {setting}: hstar={fmt(hstar_row.get('hstar_cont'))} "
                    f"nearest={hstar_row.get('hstar_nearest_grid_label')} "
                    f"empirical_min={fmt(hstar_row.get('h_empirical_min_nmse'))}",
                    flush=True,
                )
            except Exception as exc:
                failure = {"task": task, "setting": setting, "error": repr(exc)}
                failures.append(failure)
                write_json(output_dir / task / setting / "failure.json", failure)
                print(f"[FAILED] {task} {setting}: {exc!r}", flush=True)
                torch.cuda.empty_cache()
    write_csv(output_dir / "summary.csv", all_summary_rows, SUMMARY_COLUMNS)
    write_csv(output_dir / "hstar_summary.csv", hstar_rows, HSTAR_COLUMNS)
    write_json(output_dir / "failures.json", failures)
    write_report(output_dir, all_summary_rows, hstar_rows, config)
    run_summary = {
        **config,
        "output_dir": str(output_dir),
        "status": "complete" if not failures else "partial_failure",
        "records": len(all_records),
        "summary_rows": len(all_summary_rows),
        "hstar_rows": len(hstar_rows),
        "failures": failures,
        "runtime_seconds": time.time() - start,
        "peak_gpu_memory_mb": float(torch.cuda.max_memory_allocated() / 1024 / 1024),
    }
    write_json(output_dir / "run_summary.json", run_summary)
    print(f"Output: {output_dir}", flush=True)
    print(f"Status: {run_summary['status']} records={len(all_records)} hstar_rows={len(hstar_rows)}", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
