#!/usr/bin/env python
"""Dense INT4 RTNClip MeZO smoke runner for OPT-1.3B task experiments.

This is intentionally small: it verifies that a task/lane can run dense
two-point ZO updates with INT4 RTNClip quantized-forward semantics and FP16
master updates.  It is not a full training sweep.
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
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset as TorchDataset, RandomSampler, SequentialSampler


REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = REPO_ROOT / "tools"
LARGE_MODELS_DIR = REPO_ROOT / "large_models"
for path in (TOOLS_DIR, LARGE_MODELS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import probe_opt13b_int4_task_grid as optprobe  # noqa: E402
from utils import DataCollatorWithPaddingAndNesting, encode_prompt, forward_wrap_with_option_len  # noqa: E402
from opt_mezo_option_tasks import get_option_task  # noqa: E402


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: List[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def read_csv_rows(path: Path) -> List[Dict[str, object]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def append_jsonl(path: Path, row: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, sort_keys=True, default=str) + "\n")


def normalize_json(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return normalize_json(value.item())
        return normalize_json(value.detach().cpu().tolist())
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, dict):
        return {str(k): normalize_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [normalize_json(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT), text=True).strip()
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
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(torch.cuda.current_device())
        out["gpu_name"] = props.name
        out["gpu_total_memory_mb"] = int(props.total_memory / 1024 / 1024)
    return out


def parse_h_values(raw: Sequence[str]) -> List[float]:
    out: List[float] = []
    for item in raw:
        for part in str(item).replace(",", " ").split():
            out.append(float(part))
    return out


def parse_h_labels(raw: Sequence[str], h_values: Sequence[float]) -> List[str]:
    labels: List[str] = []
    for item in raw:
        labels.extend(str(item).replace(",", " ").split())
    if not labels:
        labels = [optprobe.label_h(h) for h in h_values]
    if len(labels) != len(h_values):
        raise ValueError(f"Need {len(h_values)} h labels, got {len(labels)}")
    return labels


def clone_master(master: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {name: tensor.detach().clone() for name, tensor in master.items()}


def select_batch(pool: List[Dict[str, object]], step: int, batch_size: int) -> List[Dict[str, object]]:
    if not pool:
        raise ValueError("empty training pool")
    start = (step * batch_size) % len(pool)
    out = []
    for i in range(batch_size):
        out.append(pool[(start + i) % len(pool)])
    return out


class MeZOOptionPromptDataset(TorchDataset):
    """MeZO large_models option-loss dataset for OPT classification tasks."""

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


def patch_mezo_option_loss(model) -> None:
    if not hasattr(model, "original_forward"):
        model.original_forward = model.forward
        model.forward = forward_wrap_with_option_len.__get__(model, type(model))


def prepare_batch(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    return {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}


def infinite_loader(loader: DataLoader) -> Iterator[Dict[str, Any]]:
    while True:
        for batch in loader:
            yield batch


def load_mezo_option_loaders(args, tokenizer) -> Tuple[Any, DataLoader, DataLoader, int, int]:
    task = get_option_task(args.task)
    train_sets = task.sample_train_sets(
        num_train=int(args.num_train),
        num_dev=0,
        num_eval=None,
        num_train_sets=1,
        seed=int(args.data_seed),
        dataset_mode=str(args.dataset_mode),
        num_k=int(args.num_k),
    )
    train_samples = list(train_sets[0])
    eval_splits = task.get_eval_splits() if hasattr(task, "get_eval_splits") else {"valid": task.valid_samples}
    valid_key = "valid" if "valid" in eval_splits else list(eval_splits.keys())[0]
    if int(args.eval_samples) > 0:
        eval_samples = task.sample_subset(data_split=valid_key, seed=0, num=int(args.eval_samples))
    else:
        eval_samples = list(eval_splits[valid_key])
    train_dataset = MeZOOptionPromptDataset(train_samples, task, tokenizer, int(args.max_seq_len))
    eval_dataset = MeZOOptionPromptDataset(eval_samples, task, tokenizer, int(args.max_seq_len))
    generator = torch.Generator()
    generator.manual_seed(int(args.data_seed))
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(args.batch_size),
        sampler=RandomSampler(train_dataset, generator=generator),
        collate_fn=DataCollatorWithPaddingAndNesting(tokenizer, pad_to_multiple_of=8),
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=int(args.eval_batch_size),
        sampler=SequentialSampler(eval_dataset),
        collate_fn=DataCollatorWithPaddingAndNesting(tokenizer, pad_to_multiple_of=8),
    )
    return task, train_loader, eval_loader, len(train_samples), len(eval_samples)


def eval_mezo_option_quantized(
    model,
    params,
    master,
    states,
    eval_loader: DataLoader,
    device: torch.device,
    max_batches: int = 0,
) -> Dict[str, object]:
    optprobe.apply_values(params, master, None, states, 0.0, 0.0)
    model.eval()
    total = 0
    correct = 0
    losses: List[float] = []
    pred_counts: Dict[str, int] = {}
    gold_counts: Dict[str, int] = {}
    with torch.inference_mode():
        for batch_id, batch in enumerate(eval_loader):
            if max_batches and batch_id >= int(max_batches):
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
            scores: List[float] = []
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
                pred_counts[str(pred)] = pred_counts.get(str(pred), 0) + 1
                gold_counts[str(gold)] = gold_counts.get(str(gold), 0) + 1
                correct += int(pred == gold)
                total += 1
                start += n_opt
    return {
        "eval_loss": float(np.mean(losses)) if losses else None,
        "eval_acc": correct / max(total, 1) if total else None,
        "eval_examples": total,
        "pred_counts": pred_counts,
        "gold_counts": gold_counts,
    }


def forward_loss_any(model, batch: Any) -> torch.Tensor:
    if hasattr(batch, "as_inputs"):
        return optprobe.forward_loss(model, batch)
    return model(**batch, return_dict=True).loss


def finite_difference_any(
    model,
    params,
    master,
    batch: Any,
    states: Dict[str, Any],
    directions: Dict[str, torch.Tensor],
    h: float,
) -> Tuple[float, float, float]:
    with torch.no_grad():
        optprobe.apply_values(params, master, directions, states, h, +1.0)
        loss_plus = float(forward_loss_any(model, batch).detach().cpu())
        optprobe.apply_values(params, master, directions, states, h, -1.0)
        loss_minus = float(forward_loss_any(model, batch).detach().cpu())
        optprobe.restore_master(params, master)
    return loss_plus, loss_minus, (loss_plus - loss_minus) / (2.0 * float(h))


def direction_norm_sq(directions: Dict[str, torch.Tensor]) -> float:
    total = torch.zeros((), device=next(iter(directions.values())).device, dtype=torch.float64)
    for tensor in directions.values():
        total += tensor.double().square().sum()
    return float(total.detach().cpu())


def update_master(master: Dict[str, torch.Tensor], directions: Dict[str, torch.Tensor], scale: float) -> None:
    with torch.no_grad():
        for name, direction in directions.items():
            master[name].add_(direction.to(dtype=master[name].dtype), alpha=float(scale))


def checkpoint_path(run_dir: Path, tag: str) -> Path:
    return run_dir / "checkpoints" / tag / "master.pt"


def save_master_checkpoint(run_dir: Path, tag: str, step: int, master: Dict[str, torch.Tensor]) -> None:
    path = checkpoint_path(run_dir, tag)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "step": int(step),
        "master": {name: tensor.detach().cpu() for name, tensor in master.items()},
    }
    torch.save(state, path)
    write_json(path.parent / "meta.json", {"step": int(step), "path": str(path)})


def load_master_checkpoint(run_dir: Path, tag: str, master: Dict[str, torch.Tensor], device: torch.device) -> int:
    path = checkpoint_path(run_dir, tag)
    state = torch.load(path, map_location="cpu")
    saved = state["master"]
    for name, tensor in saved.items():
        if name in master:
            master[name] = tensor.to(device=device, dtype=master[name].dtype)
    return int(state.get("step", 0))


def summarize_eval(eval_path: Path) -> Dict[str, object]:
    best_eval_acc = None
    best_eval_step = None
    best_eval_loss = None
    best_eval_loss_step = None
    last_eval: Dict[str, object] = {}
    if not eval_path.exists():
        return {
            "last_eval": last_eval,
            "best_eval_acc": best_eval_acc,
            "best_eval_step": best_eval_step,
            "best_eval_loss": best_eval_loss,
            "best_eval_loss_step": best_eval_loss_step,
        }
    for line in eval_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        last_eval = row
        if row.get("eval_acc") is not None and (best_eval_acc is None or float(row["eval_acc"]) > best_eval_acc):
            best_eval_acc = float(row["eval_acc"])
            best_eval_step = int(row.get("step", 0))
        if row.get("eval_loss") is not None and (best_eval_loss is None or float(row["eval_loss"]) < best_eval_loss):
            best_eval_loss = float(row["eval_loss"])
            best_eval_loss_step = int(row.get("step", 0))
    return {
        "last_eval": last_eval,
        "best_eval_acc": best_eval_acc,
        "best_eval_step": best_eval_step,
        "best_eval_loss": best_eval_loss,
        "best_eval_loss_step": best_eval_loss_step,
    }


def eval_quantized(
    model,
    params,
    master,
    states,
    tokenizer,
    task: str,
    examples: List[Dict[str, object]],
    max_seq_len: int,
    device: torch.device,
) -> Dict[str, object]:
    optprobe.apply_values(params, master, None, states, 0.0, 0.0)
    losses: List[float] = []
    correct = 0
    total = 0
    with torch.no_grad():
        for ex in examples:
            candidates = list(ex.get("candidates") or [])
            if not candidates:
                continue
            cand_losses = []
            for cid, answer in enumerate(candidates):
                cand = dict(ex)
                cand["answer"] = answer
                batch = optprobe.build_task_batch(tokenizer, task, [cand], max_seq_len, device)
                cand_losses.append(float(optprobe.forward_loss(model, batch).detach().cpu()))
            pred = min(range(len(cand_losses)), key=lambda i: cand_losses[i])
            label = int(ex.get("label", -1))
            correct += int(pred == label)
            total += 1
            if 0 <= label < len(cand_losses):
                losses.append(cand_losses[label])
    return {
        "eval_loss": sum(losses) / max(len(losses), 1) if losses else None,
        "eval_acc": correct / max(total, 1) if total else None,
        "eval_examples": total,
    }


def run_one_h(
    *,
    model,
    tokenizer,
    params,
    q_names,
    initial_master,
    train_pool,
    eval_pool,
    train_iter,
    eval_loader,
    train_sample_count: int,
    eval_sample_count: int,
    args,
    h: float,
    h_label: str,
    output_root: Path,
    device: torch.device,
) -> Dict[str, object]:
    precision_label = f"int{int(args.bitwidth)}"
    run_name = f"{args.task}_dense_{precision_label}_{h_label}_step{args.steps}"
    run_dir = output_root / args.task / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.csv"
    eval_path = run_dir / "eval_metrics.jsonl"
    summary_path = run_dir / "run_summary.json"
    if args.skip_complete and summary_path.exists():
        old_summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if old_summary.get("status") == "complete" and int(old_summary.get("steps_completed", 0)) >= int(args.steps):
            print(f"{run_name} already complete; skipping", flush=True)
            return old_summary
    if eval_path.exists() and not args.resume:
        eval_path.unlink()
    master = clone_master(initial_master)
    optprobe.restore_master(params, master)
    start_step = 1
    if args.resume and checkpoint_path(run_dir, "latest").exists():
        loaded_step = load_master_checkpoint(run_dir, "latest", master, device)
        start_step = loaded_step + 1
        optprobe.restore_master(params, master)
        print(f"{run_name} resuming from step {loaded_step}", flush=True)
    config = {
        "run_name": run_name,
        "model_id": args.model_id,
        "task": args.task,
        "precision": precision_label,
        "quantizer": f"INT{int(args.bitwidth)}_G128_RTNClip_shared_grid_fake_quant",
        "group_size": args.group_size,
        "bitwidth": args.bitwidth,
        "h": h,
        "h_label": h_label,
        "lr": args.lr,
        "steps": args.steps,
        "seed": args.seed,
        "data_seed": args.data_seed,
        "batch_size": args.batch_size,
        "eval_batch_size": args.eval_batch_size,
        "max_seq_len": args.max_seq_len,
        "task_path": args.task_path,
        "dataset_mode": args.dataset_mode,
        "num_train": args.num_train,
        "num_k": args.num_k,
        "direction": "dense",
        "update_backend": "fp16_master",
        "scale_refresh_k": 1,
        "pair_shared_grid": True,
        "fresh_round_codes": True,
        "scale_source": "unperturbed_fp16_master_weight",
        "perturb_scope": "all_floating_parameters",
        "eval_samples": args.eval_samples,
        "train_sample_count": train_sample_count,
        "eval_sample_count": eval_sample_count,
        "mezo_option_loss": args.task_path == "mezo_option",
        "eval_at_start": bool(args.eval_at_start),
    }
    write_json(run_dir / "run_config.json", config)
    write_json(run_dir / "env.json", env_info())
    (run_dir / "resume_command.txt").write_text(
        " ".join([sys.executable, *sys.argv, "--resume"]) + "\n",
        encoding="utf-8",
    )
    metrics: List[Dict[str, object]] = read_csv_rows(metrics_path) if args.resume else []
    eval_summary = summarize_eval(eval_path)
    best_eval_acc = eval_summary["best_eval_acc"]
    best_eval_step = eval_summary["best_eval_step"]
    best_eval_loss = eval_summary["best_eval_loss"]
    best_eval_loss_step = eval_summary["best_eval_loss_step"]
    if args.eval_at_start and start_step == 1:
        states_eval, _ = optprobe.refresh_states(master, q_names, int(args.bitwidth), int(args.group_size))
        if args.task_path == "mezo_option":
            ev = eval_mezo_option_quantized(
                model,
                params,
                master,
                states_eval,
                eval_loader,
                device,
                max_batches=int(args.eval_max_batches),
            )
        else:
            ev = eval_quantized(model, params, master, states_eval, tokenizer, args.task, eval_pool, args.max_seq_len, device)
        ev_row = {"step": 0, **ev}
        append_jsonl(eval_path, ev_row)
        if ev.get("eval_acc") is not None and (best_eval_acc is None or float(ev["eval_acc"]) > best_eval_acc):
            best_eval_acc = float(ev["eval_acc"])
            best_eval_step = 0
        if ev.get("eval_loss") is not None and (best_eval_loss is None or float(ev["eval_loss"]) < best_eval_loss):
            best_eval_loss = float(ev["eval_loss"])
            best_eval_loss_step = 0
    nan_occurred = False
    start = time.time()
    for step in range(start_step, args.steps + 1):
        if args.task_path == "mezo_option":
            batch = prepare_batch(next(train_iter), device)
        else:
            batch_examples = select_batch(train_pool, step - 1, args.batch_size)
            batch = optprobe.build_task_batch(tokenizer, args.task, batch_examples, args.max_seq_len, device)
        states, _q_rows = optprobe.refresh_states(master, q_names, int(args.bitwidth), int(args.group_size))
        directions = optprobe.sample_direction(master, list(master.keys()), args.seed * 1_000_003 + step * 1009, masks=None)
        loss_plus, loss_minus, d_h = finite_difference_any(model, params, master, batch, states, directions, float(h))
        finite = all(math.isfinite(x) for x in (loss_plus, loss_minus, d_h))
        dir_norm_sq = direction_norm_sq(directions)
        update_scale = -float(args.lr) * float(d_h)
        update_norm = abs(update_scale) * math.sqrt(max(dir_norm_sq, 0.0))
        if finite:
            update_master(master, directions, update_scale)
            optprobe.restore_master(params, master)
        else:
            nan_occurred = True
        row: Dict[str, object] = {
            "step": step,
            "h": h,
            "loss_plus": loss_plus,
            "loss_minus": loss_minus,
            "d_h": d_h,
            "train_loss": (loss_plus + loss_minus) / 2.0 if finite else float("nan"),
            "update_norm": update_norm,
            "direction_norm": math.sqrt(max(dir_norm_sq, 0.0)),
            "finite": finite,
            "nan_occurred": nan_occurred,
        }
        if step % max(1, args.diag_every) == 0 or step == 1:
            vis = optprobe.visibility_metrics(master, directions, states, float(h))
            row.update({f"last_{k}": v for k, v in vis.items()})
        metrics.append(row)
        if step % max(1, args.log_every) == 0 or step == 1:
            print(
                f"{run_name} step={step} loss={(loss_plus + loss_minus) / 2.0:.6g} "
                f"d_h={d_h:.6g} update_norm={update_norm:.6g}",
                flush=True,
            )
        if step % max(1, args.eval_every) == 0 or step == args.steps:
            states_eval, _ = optprobe.refresh_states(master, q_names, int(args.bitwidth), int(args.group_size))
            if args.task_path == "mezo_option":
                ev = eval_mezo_option_quantized(
                    model,
                    params,
                    master,
                    states_eval,
                    eval_loader,
                    device,
                    max_batches=int(args.eval_max_batches),
                )
            else:
                ev = eval_quantized(model, params, master, states_eval, tokenizer, args.task, eval_pool, args.max_seq_len, device)
            ev_row = {"step": step, **ev}
            append_jsonl(eval_path, ev_row)
            if ev.get("eval_acc") is not None and (best_eval_acc is None or float(ev["eval_acc"]) > best_eval_acc):
                best_eval_acc = float(ev["eval_acc"])
                best_eval_step = step
                if args.save_best_checkpoints:
                    save_master_checkpoint(run_dir, "best_acc", step, master)
            if ev.get("eval_loss") is not None and (best_eval_loss is None or float(ev["eval_loss"]) < best_eval_loss):
                best_eval_loss = float(ev["eval_loss"])
                best_eval_loss_step = step
                if args.save_best_checkpoints:
                    save_master_checkpoint(run_dir, "best_loss", step, master)
            metrics[-1].update(ev_row)
        if nan_occurred:
            break
        if args.checkpoint_every > 0 and step % int(args.checkpoint_every) == 0:
            write_csv(metrics_path, metrics)
            save_master_checkpoint(run_dir, "latest", step, master)
        del directions
        torch.cuda.empty_cache()
    write_csv(metrics_path, metrics)
    last_eval = {}
    if eval_path.exists():
        lines = [line for line in eval_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if lines:
            last_eval = json.loads(lines[-1])
    summary = {
        **config,
        "status": "failed_nan" if nan_occurred else "complete",
        "steps_completed": metrics[-1]["step"] if metrics else 0,
        "final_train_loss": metrics[-1].get("train_loss") if metrics else None,
        "last_eval_acc": last_eval.get("eval_acc"),
        "last_eval_loss": last_eval.get("eval_loss"),
        "last_eval_step": last_eval.get("step"),
        "best_eval_acc": best_eval_acc,
        "best_eval_step": best_eval_step,
        "best_eval_loss": best_eval_loss,
        "best_eval_loss_step": best_eval_loss_step,
        "nan_occurred": nan_occurred,
        "runtime_sec": time.time() - start,
        "peak_gpu_memory_mb": float(torch.cuda.max_memory_allocated() / 1024 / 1024) if torch.cuda.is_available() else 0.0,
    }
    if metrics:
        save_master_checkpoint(run_dir, "final" if not nan_occurred else "latest", int(metrics[-1]["step"]), master)
    write_json(run_dir / "run_summary.json", summary)
    optprobe.restore_master(params, initial_master)
    torch.cuda.empty_cache()
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--task", required=True, choices=["sst-2", "sst-5", "rte", "mnli", "trec"])
    parser.add_argument("--model_id", default="facebook/opt-1.3b")
    parser.add_argument("--h_values", nargs="+", required=True)
    parser.add_argument("--h_labels", nargs="+", default=[])
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--task_path", choices=["simple_lm_answer", "mezo_option"], default="simple_lm_answer")
    parser.add_argument("--dataset_mode", choices=["auto", "fewshot", "full"], default="full")
    parser.add_argument("--num_train", type=int, default=-1)
    parser.add_argument("--num_k", type=int, default=16)
    parser.add_argument("--train_pool_size", type=int, default=128)
    parser.add_argument("--eval_samples", type=int, default=32)
    parser.add_argument("--eval_max_batches", type=int, default=0)
    parser.add_argument("--eval_every", type=int, default=20)
    parser.add_argument("--log_every", type=int, default=5)
    parser.add_argument("--diag_every", type=int, default=10)
    parser.add_argument("--max_seq_len", type=int, default=128)
    parser.add_argument("--bitwidth", type=int, default=4)
    parser.add_argument("--group_size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=16)
    parser.add_argument("--data_seed", type=int, default=16)
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--checkpoint_every", type=int, default=0)
    parser.add_argument("--eval_at_start", action="store_true")
    parser.add_argument("--save_best_checkpoints", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip_complete", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.task = optprobe.normalize_task(args.task)
    h_values = parse_h_values(args.h_values)
    h_labels = parse_h_labels(args.h_labels, h_values)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for OPT-1.3B smoke training.")
    device = torch.device("cuda")
    write_json(output_root / f"env_{args.task}.json", env_info())
    model, tokenizer = optprobe.load_model_and_tokenizer(args, device)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id
    if args.task_path == "mezo_option":
        patch_mezo_option_loss(model)
    params = optprobe.params_map(model)
    q_names = optprobe.linear_weight_names(model, params)
    initial_master = optprobe.make_master(params, torch.float16)
    train_pool: List[Dict[str, object]] = []
    eval_pool: List[Dict[str, object]] = []
    train_iter = None
    eval_loader = None
    train_sample_count = 0
    eval_sample_count = 0
    if args.task_path == "mezo_option":
        _task, train_loader, eval_loader_obj, train_sample_count, eval_sample_count = load_mezo_option_loaders(args, tokenizer)
        train_iter = infinite_loader(train_loader)
        eval_loader = eval_loader_obj
    else:
        train_pool = optprobe.load_task_examples(args.task, args.data_seed, args.train_pool_size)
        eval_pool = optprobe.load_task_examples(args.task, args.data_seed + 9999, args.eval_samples)
        train_sample_count = len(train_pool)
        eval_sample_count = len(eval_pool)
    summaries: List[Dict[str, object]] = []
    for h, h_label in zip(h_values, h_labels):
        summaries.append(
            run_one_h(
                model=model,
                tokenizer=tokenizer,
                params=params,
                q_names=q_names,
                initial_master=initial_master,
                train_pool=train_pool,
                eval_pool=eval_pool,
                train_iter=train_iter,
                eval_loader=eval_loader,
                train_sample_count=train_sample_count,
                eval_sample_count=eval_sample_count,
                args=args,
                h=float(h),
                h_label=h_label,
                output_root=output_root,
                device=device,
            )
        )
    write_csv(output_root / f"summary_{args.task}.csv", summaries)
    write_json(output_root / f"summary_{args.task}.json", summaries)
    for row in summaries:
        print(
            f"{row['task']} {row['h_label']} status={row['status']} "
            f"steps={row['steps_completed']} final_loss={row['final_train_loss']} "
            f"eval_acc={row.get('last_eval_acc')}",
            flush=True,
        )
    return 0 if all(row.get("status") == "complete" for row in summaries) else 1


if __name__ == "__main__":
    raise SystemExit(main())
