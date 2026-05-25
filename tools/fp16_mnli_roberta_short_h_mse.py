#!/usr/bin/env python
"""Short FP16 MeZO h probe on RoBERTa-large / MNLI.

This is a cheap diagnostic runner: train a few fixed h values for a small
number of dense two-point MeZO steps, then measure FP16 finite-difference MSE
against FP32 true directional derivatives at the final checkpoint.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import random
import socket
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler


REPO_ROOT = Path(__file__).resolve().parents[1]
MEDIUM_ROOT = REPO_ROOT / "medium_models"
if str(MEDIUM_ROOT) not in sys.path:
    sys.path.insert(0, str(MEDIUM_ROOT))


H_VALUES: List[Tuple[str, float]] = [
    ("1e-5", 1e-5),
    ("1e-4", 1e-4),
    ("1e-3", 1e-3),
    ("1e-2", 1e-2),
]
EPS = 1e-30


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def append_jsonl(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(data, sort_keys=True, default=str) + "\n")


def write_csv(path: Path, rows: Sequence[Dict[str, Any]], columns: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(columns), extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT), text=True).strip()
    except Exception:
        return ""


def collect_env() -> Dict[str, Any]:
    env = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.executable,
        "git_commit": git_commit(),
        "torch_version": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
    }
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(torch.cuda.current_device())
        env.update({"gpu_name": props.name, "gpu_total_memory_mb": int(props.total_memory / 1024 / 1024)})
    return env


def set_seed(seed: int) -> None:
    random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def reset_direction_rng(seed: int) -> None:
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def direction_seed(base_seed: int, h: float, step_or_batch: int, direction_id: int = 0) -> int:
    return int(base_seed) + int(round(float(h) * 1_000_000_000_000)) + 1_000_003 * int(step_or_batch) + 97_531 * int(direction_id) + 17_071


def mnli_data_args() -> SimpleNamespace:
    return SimpleNamespace(
        task_name="mnli",
        max_seq_length=256,
        overwrite_cache=False,
        num_k=16,
        num_sample=16,
        num_demo=1,
        auto_demo=True,
        sfc_prompt=None,
        template_path=None,
        mapping_path=None,
        prompt_path=None,
        template_id=None,
        mapping_id=None,
        prompt_id=None,
        top_n_template=None,
        tag="fp16_mnli_roberta_short_h_mse",
        demo_filter=False,
        demo_filter_rate=0.5,
        demo_filter_model=None,
        debug_mode=False,
        double_demo=False,
        first_sent_limit=240,
        other_sent_limit=None,
        use_full_length=None,
        dataset_mode="full",
        data_root="data/k-shot-1k-test",
        full_dev_ratio=0.1,
        gpt3_in_context_head=False,
        gpt3_in_context_tail=False,
        gpt3_in_context_num=32,
        gpt3_demo_separator="\n\n\n",
        truncate_head=False,
        prompt=True,
        template_list=None,
        template="*cls**sent-_0*?*mask*,*+sentl_1**sep+*",
        mapping="{'contradiction':'No','entailment':'Yes','neutral':'Maybe'}",
        data_dir=str(MEDIUM_ROOT / "data/k-shot-1k-test" / "MNLI" / "full-16"),
    )


def collate_with_padding(tokenizer: Any, features: Sequence[Any]) -> Dict[str, torch.Tensor]:
    items: List[Dict[str, Any]] = []
    mask_pos: List[Any] = []
    for item in features:
        row: Dict[str, Any] = {}
        for field in ("input_ids", "label", "attention_mask", "token_type_ids"):
            value = getattr(item, field, None)
            if value is not None:
                row[field] = value
        items.append(row)
        mask_pos.append(getattr(item, "mask_pos", None))
    batch = tokenizer.pad(items, padding=True, return_tensors="pt")
    if any(x is not None for x in mask_pos):
        batch["mask_pos"] = torch.tensor(mask_pos)
    if "label" in batch:
        batch["labels"] = batch.pop("label")
    return batch


def move_batch(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}


class MnliRobertaContext:
    def __init__(self, args: argparse.Namespace, device: torch.device) -> None:
        from transformers import AutoConfig, AutoTokenizer
        from src.data_utils import resolve_and_prepare_data
        from src.dataset import FewShotDataset
        from src.models import MODEL_TYPES, RobertaModelForPromptFinetuning
        from src.modeling_roberta import RobertaModel
        from src.processors import num_labels_mapping

        if not hasattr(RobertaModelForPromptFinetuning, "all_tied_weights_keys"):
            RobertaModelForPromptFinetuning.all_tied_weights_keys = {}
        if not hasattr(RobertaModel, "get_head_mask"):
            def _compat_get_head_mask(self, head_mask, num_hidden_layers, is_attention_chunked=False):
                if head_mask is None:
                    return [None] * num_hidden_layers
                if head_mask.dim() == 1:
                    head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                    head_mask = head_mask.expand(num_hidden_layers, -1, -1, -1, -1)
                elif head_mask.dim() == 2:
                    head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
                return head_mask.to(dtype=self.dtype)

            RobertaModel.get_head_mask = _compat_get_head_mask

        self.args = args
        self.device = device
        self.data_args = mnli_data_args()
        train_args = SimpleNamespace(seed=int(args.seed), data_seed=int(args.data_seed))
        resolution = resolve_and_prepare_data(self.data_args, train_args)
        self.data_args.data_dir = resolution.resolved_data_dir
        self.data_args.dataset_mode = resolution.resolved_dataset_mode
        self.tokenizer = AutoTokenizer.from_pretrained("roberta-large")
        config = AutoConfig.from_pretrained("roberta-large", num_labels=int(num_labels_mapping["mnli"]), finetuning_task="mnli")
        self.tokenizer.model_type = config.model_type
        model_fn = MODEL_TYPES[config.model_type]
        self.model = model_fn.from_pretrained("roberta-large", config=config)
        self.model.model_args = SimpleNamespace(
            model_name_or_path="roberta-large",
            few_shot_type="prompt",
            random_segment=False,
            l2_loss=False,
            use_task_word=False,
            apply_lora=False,
            sfc=False,
            icl_sfc=False,
        )
        self.model.data_args = self.data_args
        self.model.tokenizer = self.tokenizer
        self.model.return_full_softmax = False
        self.train_dataset = FewShotDataset(self.data_args, tokenizer=self.tokenizer, mode="train", use_demo=False)
        self.dev_dataset = FewShotDataset(self.data_args, tokenizer=self.tokenizer, mode="dev", use_demo=False)
        if getattr(self.train_dataset, "label_word_list", None) is not None:
            self.model.label_word_list = torch.tensor(self.train_dataset.label_word_list, dtype=torch.long, device=device)
        self.model.to(device)
        self.model.eval()
        self.params = [(name, p) for name, p in self.model.named_parameters() if p.requires_grad and p.detach().is_floating_point()]
        self.train_sampler_name = "RandomSampler"

    def make_train_loader(self, seed_offset: int = 0) -> DataLoader:
        gen = torch.Generator().manual_seed(int(self.args.data_seed) + int(seed_offset))
        sampler = RandomSampler(self.train_dataset, generator=gen)
        return DataLoader(
            self.train_dataset,
            batch_size=int(self.args.batch_size),
            sampler=sampler,
            collate_fn=lambda xs: collate_with_padding(self.tokenizer, xs),
            num_workers=0,
        )

    def make_eval_loader(self) -> DataLoader:
        return DataLoader(
            self.dev_dataset,
            batch_size=int(self.args.eval_batch_size),
            sampler=SequentialSampler(self.dev_dataset),
            collate_fn=lambda xs: collate_with_padding(self.tokenizer, xs),
            num_workers=0,
        )


def master_from_model(ctx: MnliRobertaContext, dtype: torch.dtype = torch.float16) -> Dict[str, torch.Tensor]:
    return {name: p.detach().clone().to(device=ctx.device, dtype=dtype) for name, p in ctx.params}


def copy_master(ctx: MnliRobertaContext, master: Dict[str, torch.Tensor], dtype: Optional[torch.dtype] = None) -> None:
    with torch.no_grad():
        for name, p in ctx.params:
            value = master[name]
            p.copy_(value.to(device=p.device, dtype=dtype or p.dtype))


def sample_z_like(tensor: torch.Tensor, dtype: torch.dtype = torch.float16) -> torch.Tensor:
    return torch.empty(tensor.shape, device=tensor.device, dtype=dtype).normal_(0.0, 1.0)


def apply_signed_from_master(ctx: MnliRobertaContext, master: Dict[str, torch.Tensor], seed: int, h: float, sign: float) -> None:
    reset_direction_rng(seed)
    with torch.no_grad():
        for name, p in ctx.params:
            z = sample_z_like(master[name], torch.float16)
            value = master[name].float().add(z.float(), alpha=float(sign) * float(h))
            p.copy_(value.to(dtype=p.dtype))
            del z, value


def update_master(ctx: MnliRobertaContext, master: Dict[str, torch.Tensor], seed: int, lr: float, d_h: float) -> float:
    reset_direction_rng(seed)
    sq = torch.zeros((), device=ctx.device, dtype=torch.float64)
    with torch.no_grad():
        for name, _p in ctx.params:
            z = sample_z_like(master[name], torch.float16)
            update = z.float().mul(-float(lr) * float(d_h))
            sq += update.double().square().sum()
            master[name].copy_(master[name].float().add(update).to(dtype=master[name].dtype))
            del z, update
    copy_master(ctx, master)
    return float(sq.sqrt().detach().cpu())


def forward_loss_logits(ctx: MnliRobertaContext, batch: Dict[str, torch.Tensor], grad: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
    if grad:
        outputs = ctx.model(**batch)
        return outputs[0], outputs[1]
    with torch.no_grad():
        outputs = ctx.model(**batch)
        return outputs[0], outputs[1]


def finite_difference(ctx: MnliRobertaContext, master: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor], seed: int, h: float) -> Tuple[float, float, float]:
    ctx.model.half()
    copy_master(ctx, master, torch.float16)
    apply_signed_from_master(ctx, master, seed, h, +1.0)
    loss_plus, _ = forward_loss_logits(ctx, batch, grad=False)
    apply_signed_from_master(ctx, master, seed, h, -1.0)
    loss_minus, _ = forward_loss_logits(ctx, batch, grad=False)
    copy_master(ctx, master, torch.float16)
    lp = float(loss_plus.detach().float().cpu())
    lm = float(loss_minus.detach().float().cpu())
    return lp, lm, (lp - lm) / (2.0 * float(h))


def evaluate_subset(ctx: MnliRobertaContext, master: Dict[str, torch.Tensor], max_batches: int) -> Tuple[Optional[float], Optional[float], int]:
    if max_batches <= 0:
        return None, None, 0
    ctx.model.half()
    copy_master(ctx, master, torch.float16)
    total_loss = 0.0
    total_correct = 0
    total_items = 0
    loader = ctx.make_eval_loader()
    for idx, batch in enumerate(loader):
        if idx >= max_batches:
            break
        batch = move_batch(batch, ctx.device)
        loss, logits = forward_loss_logits(ctx, batch, grad=False)
        labels = batch["labels"]
        total_loss += float(loss.detach().float().cpu()) * int(labels.numel())
        total_correct += int((logits.argmax(dim=-1) == labels).sum().detach().cpu())
        total_items += int(labels.numel())
    if total_items == 0:
        return None, None, 0
    return total_loss / total_items, total_correct / total_items, total_items


def compute_true_directionals(ctx: MnliRobertaContext, master: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor], h: float, batch_id: int, dirs: int) -> List[Optional[float]]:
    ctx.model.float()
    copy_master(ctx, master, torch.float32)
    ctx.model.zero_grad(set_to_none=True)
    loss, _ = forward_loss_logits(ctx, batch, grad=True)
    loss.backward()
    values: List[Optional[float]] = []
    for direction_id in range(int(dirs)):
        seed = direction_seed(int(ctx.args.seed), h, batch_id, direction_id)
        reset_direction_rng(seed)
        acc = torch.zeros((), device=ctx.device, dtype=torch.float64)
        seen = False
        with torch.no_grad():
            for _name, p in ctx.params:
                z = sample_z_like(p.data, torch.float16)
                if p.grad is not None:
                    acc += (p.grad.detach().float() * z.float()).double().sum()
                    seen = True
                del z
        values.append(float(acc.detach().cpu()) if seen else None)
    ctx.model.zero_grad(set_to_none=True)
    return values


def fp16_effective_stats(ctx: MnliRobertaContext, master: Dict[str, torch.Tensor], seed: int, h: float) -> Dict[str, float]:
    reset_direction_rng(seed)
    dot = torch.zeros((), device=ctx.device, dtype=torch.float64)
    eff_sq = torch.zeros((), device=ctx.device, dtype=torch.float64)
    ideal_sq = torch.zeros((), device=ctx.device, dtype=torch.float64)
    err_sq = torch.zeros((), device=ctx.device, dtype=torch.float64)
    active = 0
    total = 0
    with torch.no_grad():
        for name, _p in ctx.params:
            base = master[name]
            z = sample_z_like(base, torch.float16)
            plus = base.float().add(z.float(), alpha=float(h)).to(dtype=torch.float16)
            minus = base.float().add(z.float(), alpha=-float(h)).to(dtype=torch.float16)
            eff = plus.float() - minus.float()
            ideal = 2.0 * float(h) * z.float()
            err = eff - ideal
            active += int((eff != 0).sum().detach().cpu())
            total += int(eff.numel())
            dot += (eff.double() * ideal.double()).sum()
            eff_sq += eff.double().square().sum()
            ideal_sq += ideal.double().square().sum()
            err_sq += err.double().square().sum()
            del z, plus, minus, eff, ideal, err
    eff_norm = math.sqrt(float(eff_sq.detach().cpu()))
    ideal_norm = math.sqrt(float(ideal_sq.detach().cpu()))
    return {
        "delta_visibility_mse": float((err_sq / max(total, 1)).detach().cpu()),
        "delta_visibility_nmse": float((err_sq / ideal_sq.clamp_min(EPS)).detach().cpu()),
        "delta_visibility_rel_l2": math.sqrt(float(err_sq.detach().cpu()) / max(float(ideal_sq.detach().cpu()), EPS)),
        "alignment": float(dot.detach().cpu()) / max(eff_norm * ideal_norm, EPS),
        "norm_ratio": eff_norm / max(ideal_norm, EPS),
        "active_frac": active / max(total, 1),
        "code_change_frac": active / max(total, 1),
    }


def corr(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    pairs = [(float(x), float(y)) for x, y in zip(xs, ys) if x is not None and y is not None and math.isfinite(float(x)) and math.isfinite(float(y))]
    if len(pairs) < 2:
        return None
    x_vals = [p[0] for p in pairs]
    y_vals = [p[1] for p in pairs]
    mx = sum(x_vals) / len(x_vals)
    my = sum(y_vals) / len(y_vals)
    vx = sum((x - mx) ** 2 for x in x_vals)
    vy = sum((y - my) ** 2 for y in y_vals)
    if vx <= EPS or vy <= EPS:
        return None
    return sum((x - mx) * (y - my) for x, y in pairs) / math.sqrt(vx * vy)


def summarize_probe(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    fd = [float(r["d_h"]) for r in records]
    half = [float(r["d_half"]) for r in records]
    true = [r.get("d_true") for r in records]
    true_pairs = [(f, float(t)) for f, t in zip(fd, true) if t is not None and math.isfinite(float(t))]
    rich_diff_sq = sum((x - y) ** 2 for x, y in zip(fd, half))
    rich_half_sq = sum(y ** 2 for y in half)
    out = {
        "n_records": len(records),
        "delta_visibility_nmse_mean": mean(r["delta_visibility_nmse"] for r in records),
        "alignment_mean": mean(r["alignment"] for r in records),
        "norm_ratio_mean": mean(r["norm_ratio"] for r in records),
        "active_frac_mean": mean(r["active_frac"] for r in records),
        "richardson_rmse_rel": math.sqrt(rich_diff_sq / max(rich_half_sq, EPS)),
        "richardson_absdiff_mean": mean(abs(x - y) for x, y in zip(fd, half)),
    }
    if true_pairs:
        err_sq = sum((x - y) ** 2 for x, y in true_pairs)
        true_sq = sum(y ** 2 for _, y in true_pairs)
        mse = err_sq / len(true_pairs)
        out.update(
            {
                "fd_true_available": True,
                "fd_true_mse": mse,
                "fd_true_nmse": err_sq / max(true_sq, EPS),
                "fd_true_rmse": math.sqrt(mse),
                "corr_fd_true": corr([x for x, _ in true_pairs], [y for _, y in true_pairs]),
                "fd_true_bias": sum(x - y for x, y in true_pairs) / len(true_pairs),
            }
        )
    else:
        out.update({"fd_true_available": False, "fd_true_mse": None, "fd_true_nmse": None, "fd_true_rmse": None, "corr_fd_true": None, "fd_true_bias": None})
    return out


def mean(values: Iterable[float]) -> Optional[float]:
    xs = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return sum(xs) / len(xs) if xs else None


def run_one_h(ctx: MnliRobertaContext, h_label: str, h: float, output_root: Path) -> Dict[str, Any]:
    run_dir = output_root / f"h_{h_label}"
    run_dir.mkdir(parents=True, exist_ok=True)
    train_loader = ctx.make_train_loader(seed_offset=0)
    train_iter = iter(train_loader)
    ctx.model.half()
    master = master_from_model(ctx, torch.float16)
    metrics_path = run_dir / "metrics.csv"
    metric_cols = ["step", "h", "loss_plus", "loss_minus", "d_h", "train_loss", "update_norm", "eval_loss", "eval_acc", "eval_n"]
    rows: List[Dict[str, Any]] = []
    best_acc: Optional[float] = None
    best_step: Optional[int] = None
    last_eval_loss: Optional[float] = None
    last_eval_acc: Optional[float] = None
    t0 = time.time()
    for step in range(1, int(ctx.args.steps) + 1):
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(ctx.make_train_loader(seed_offset=step))
            batch = next(train_iter)
        batch = move_batch(batch, ctx.device)
        seed = direction_seed(int(ctx.args.seed), h, step)
        lp, lm, d_h = finite_difference(ctx, master, batch, seed, h)
        update_norm = update_master(ctx, master, seed, float(ctx.args.lr), d_h)
        train_loss = 0.5 * (lp + lm)
        eval_loss = eval_acc = None
        eval_n = 0
        if step == 1 or step % int(ctx.args.eval_every) == 0 or step == int(ctx.args.steps):
            eval_loss, eval_acc, eval_n = evaluate_subset(ctx, master, int(ctx.args.eval_batches))
            last_eval_loss = eval_loss
            last_eval_acc = eval_acc
            if eval_acc is not None and (best_acc is None or eval_acc > best_acc):
                best_acc = float(eval_acc)
                best_step = step
        row = {
            "step": step,
            "h": h,
            "loss_plus": lp,
            "loss_minus": lm,
            "d_h": d_h,
            "train_loss": train_loss,
            "update_norm": update_norm,
            "eval_loss": eval_loss,
            "eval_acc": eval_acc,
            "eval_n": eval_n,
        }
        rows.append(row)
        if step % max(1, int(ctx.args.log_every)) == 0 or step == 1:
            print(f"[train] h={h_label} step={step}/{ctx.args.steps} loss={train_loss:.4g} d_h={d_h:.4g} eval_acc={eval_acc}", flush=True)
    write_csv(metrics_path, rows, metric_cols)
    ckpt_path = run_dir / "final_state.pt"
    torch.save({"h": h, "h_label": h_label, "step": int(ctx.args.steps), "master": {k: v.detach().cpu() for k, v in master.items()}}, ckpt_path)
    summary = {
        "h": h,
        "h_label": h_label,
        "steps": int(ctx.args.steps),
        "batch_size": int(ctx.args.batch_size),
        "lr": float(ctx.args.lr),
        "best_eval_acc": best_acc,
        "best_step": best_step,
        "last_eval_loss": last_eval_loss,
        "last_eval_acc": last_eval_acc,
        "eval_subset_n": rows[-1].get("eval_n", 0),
        "final_train_loss": rows[-1]["train_loss"],
        "mean_abs_d_h": mean(abs(float(r["d_h"])) for r in rows),
        "mean_update_norm": mean(float(r["update_norm"]) for r in rows),
        "runtime_seconds": time.time() - t0,
        "run_dir": str(run_dir),
        "checkpoint": str(ckpt_path),
    }
    write_json(run_dir / "run_summary.json", summary)
    return {**summary, "_master": master}


def probe_final(ctx: MnliRobertaContext, run_summary: Dict[str, Any], output_root: Path) -> Dict[str, Any]:
    h = float(run_summary["h"])
    h_label = str(run_summary["h_label"])
    master = run_summary["_master"]
    probe_loader = ctx.make_train_loader(seed_offset=10_000)
    batches: List[Dict[str, torch.Tensor]] = []
    it = iter(probe_loader)
    for _ in range(int(ctx.args.probe_batches)):
        batches.append(move_batch(next(it), ctx.device))
    records: List[Dict[str, Any]] = []
    records_path = output_root / "mse_probe_records.jsonl"
    for batch_id, batch in enumerate(batches):
        d_true_values = compute_true_directionals(ctx, master, batch, h, batch_id, int(ctx.args.probe_dirs))
        for direction_id in range(int(ctx.args.probe_dirs)):
            seed = direction_seed(int(ctx.args.seed), h, batch_id, direction_id)
            lp, lm, d_h = finite_difference(ctx, master, batch, seed, h)
            _, _, d_half = finite_difference(ctx, master, batch, seed, h / 2.0)
            eff = fp16_effective_stats(ctx, master, seed, h)
            d_true = d_true_values[direction_id]
            record = {
                "h": h,
                "h_label": h_label,
                "batch_id": batch_id,
                "direction_id": direction_id,
                "direction_seed": seed,
                "checkpoint_step": int(run_summary["steps"]),
                "loss_plus": lp,
                "loss_minus": lm,
                "d_h": d_h,
                "d_half": d_half,
                "d_true": d_true,
                "fd_true_error": None if d_true is None else d_h - float(d_true),
                "richardson_absdiff": abs(d_h - d_half),
                "richardson_relerr": abs(d_h - d_half) / max(abs(d_half), EPS),
                **eff,
            }
            append_jsonl(records_path, record)
            records.append(record)
    summary = summarize_probe(records)
    return {k: v for k, v in {**run_summary, **summary}.items() if k != "_master"}


def write_markdown(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    def fmt(v: Any) -> str:
        if v is None:
            return "n/a"
        try:
            return f"{float(v):.4g}"
        except Exception:
            return str(v)

    lines = [
        "# RoBERTa-large / MNLI FP16 Short h Probe",
        "",
        "Dense two-point MeZO, FP16 parameters and perturbations, 200 training steps per h.",
        "",
        "| h | best_eval_acc | last_eval_acc | fd_true_nmse | corr_fd_true | richardson_rmse_rel | delta_visibility_nmse | alignment | norm_ratio | active_frac |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['h_label']} | {fmt(row.get('best_eval_acc'))} | {fmt(row.get('last_eval_acc'))} | "
            f"{fmt(row.get('fd_true_nmse'))} | {fmt(row.get('corr_fd_true'))} | {fmt(row.get('richardson_rmse_rel'))} | "
            f"{fmt(row.get('delta_visibility_nmse_mean'))} | {fmt(row.get('alignment_mean'))} | {fmt(row.get('norm_ratio_mean'))} | {fmt(row.get('active_frac_mean'))} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_root", default=str(REPO_ROOT / "outputs" / "fp16_mnli_roberta_short_h_mse"))
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--eval_batches", type=int, default=16)
    parser.add_argument("--eval_every", type=int, default=50)
    parser.add_argument("--probe_batches", type=int, default=3)
    parser.add_argument("--probe_dirs", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=16)
    parser.add_argument("--data_seed", type=int, default=16)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--h_values", default="1e-5,1e-4,1e-3,1e-2")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    records_path = output_root / "mse_probe_records.jsonl"
    if records_path.exists():
        records_path.unlink()
    write_json(output_root / "env.json", collect_env())
    write_json(output_root / "run_config.json", vars(args))
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this short MNLI probe")
    set_seed(int(args.seed))
    ctx = MnliRobertaContext(args, torch.device("cuda"))
    selected = []
    wanted = {x.strip() for x in str(args.h_values).split(",") if x.strip()}
    for label, value in H_VALUES:
        if label in wanted or f"{value:g}" in wanted:
            selected.append((label, value))
    train_rows = []
    final_rows = []
    for label, h in selected:
        summary = run_one_h(ctx, label, h, output_root)
        train_rows.append({k: v for k, v in summary.items() if k != "_master"})
        final_rows.append(probe_final(ctx, summary, output_root))
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    train_cols = [
        "h",
        "h_label",
        "steps",
        "batch_size",
        "lr",
        "best_eval_acc",
        "best_step",
        "last_eval_loss",
        "last_eval_acc",
        "eval_subset_n",
        "final_train_loss",
        "mean_abs_d_h",
        "mean_update_norm",
        "runtime_seconds",
        "run_dir",
        "checkpoint",
    ]
    probe_cols = [
        *train_cols,
        "n_records",
        "delta_visibility_nmse_mean",
        "alignment_mean",
        "norm_ratio_mean",
        "active_frac_mean",
        "richardson_rmse_rel",
        "richardson_absdiff_mean",
        "fd_true_available",
        "fd_true_mse",
        "fd_true_nmse",
        "fd_true_rmse",
        "corr_fd_true",
        "fd_true_bias",
    ]
    write_csv(output_root / "short_train_summary.csv", train_rows, train_cols)
    write_csv(output_root / "mse_probe_summary.csv", final_rows, probe_cols)
    write_markdown(output_root / "summary.md", final_rows)
    print(f"wrote {output_root / 'summary.md'}", flush=True)


if __name__ == "__main__":
    main()
