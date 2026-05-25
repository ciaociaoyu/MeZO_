#!/usr/bin/env python
"""SST-5 extreme-sparse ZO runner.

This runner is intentionally separate from the RTNClip low-bit h-sweep runner.
It approximates the "extreme sparsity" setup: select a tiny static set of
sensitive parameters, keep/update those coordinates in FP32, quantize the
inactive Linear weights for forward, and freeze everything else.
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
from typing import Dict, Iterable, List, Optional, Tuple

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = REPO_ROOT / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import smoke_rtnclip_roberta_sst5 as smoke  # noqa: E402
import rtnclip_roberta_sst5_batch as rtn_batch  # noqa: E402


def run_cmd(cmd: List[str]) -> str:
    try:
        return subprocess.check_output(cmd, cwd=REPO_ROOT, text=True, stderr=subprocess.STDOUT).strip()
    except Exception as exc:
        return f"unavailable: {exc}"


def write_json(path: Path, data: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def append_jsonl(path: Path, data: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(data, sort_keys=True) + "\n")


def env_record() -> Dict[str, object]:
    gpu_name = ""
    gpu_mem_mb = None
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        gpu_name = props.name
        gpu_mem_mb = int(props.total_memory // (1024 * 1024))
    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "hostname": socket.gethostname(),
        "python": sys.executable,
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "gpu_name": gpu_name,
        "gpu_mem_mb": gpu_mem_mb,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "git_commit": run_cmd(["git", "rev-parse", "HEAD"]),
        "conda_default_env": os.environ.get("CONDA_DEFAULT_ENV", ""),
    }


def mixed_copy_master_to_model(
    params: Dict[str, torch.nn.Parameter],
    master: Dict[str, torch.Tensor],
    directions: Optional[Dict[str, torch.Tensor]],
    h: float,
    sign: float,
    quantizer_states: Dict[str, smoke.RTNClipState],
    active_masks: Dict[str, torch.Tensor],
) -> None:
    """Copy a mixed quantized/full-precision view into model parameters.

    For quantized Linear weights, inactive coordinates use the frozen 4-bit
    dequantized value, while active coordinates use the current FP32 master
    value plus the current probe perturbation. Non-quantized tensors are copied
    directly from master; their directions are zero in this runner.
    """
    with torch.no_grad():
        for name, param in params.items():
            base_value = master[name].float()
            if directions is not None and name in directions:
                base_value = base_value.add(directions[name].float(), alpha=sign * h)
            if name in quantizer_states:
                quantized = smoke.quantize_with_state(base_value, quantizer_states[name])
                mask = active_masks.get(name)
                if mask is not None:
                    value = torch.where(mask.to(device=base_value.device, dtype=torch.bool), base_value, quantized.float())
                else:
                    value = quantized.float()
            else:
                value = base_value
            param.copy_(value.to(device=param.device, dtype=param.dtype))


def restore_master(params: Dict[str, torch.nn.Parameter], master: Dict[str, torch.Tensor]) -> float:
    max_diff = 0.0
    with torch.no_grad():
        for name, param in params.items():
            param.copy_(master[name].to(device=param.device, dtype=param.dtype))
            diff = (param.detach().float() - master[name].float()).abs().max()
            max_diff = max(max_diff, float(diff.detach().cpu()))
    return max_diff


def update_active_master(
    master: Dict[str, torch.Tensor],
    directions: Dict[str, torch.Tensor],
    lr: float,
    projected_grad: float,
) -> float:
    sq = torch.zeros((), device=next(iter(master.values())).device, dtype=torch.float64)
    with torch.no_grad():
        for name, tensor in master.items():
            direction = directions.get(name)
            if direction is None:
                continue
            update = direction.float().mul(-float(lr) * float(projected_grad))
            sq += update.double().square().sum()
            tensor.copy_(tensor.float().add(update).to(dtype=tensor.dtype))
    return float(sq.sqrt().detach().cpu())


def mixed_evaluate(
    model,
    params: Dict[str, torch.nn.Parameter],
    master: Dict[str, torch.Tensor],
    states: Dict[str, smoke.RTNClipState],
    masks: Dict[str, torch.Tensor],
    dev_loader,
    device: torch.device,
    max_batches: int,
) -> Tuple[Optional[float], Optional[float]]:
    if max_batches == 0:
        return None, None
    mixed_copy_master_to_model(params, master, None, 0.0, 0.0, states, masks)
    total_loss = 0.0
    total_correct = 0
    total_items = 0
    for idx, batch in enumerate(dev_loader):
        if max_batches > 0 and idx >= max_batches:
            break
        batch = smoke.move_batch(batch, device)
        loss, logits = smoke.forward_loss_and_logits(model, batch)
        labels = batch["labels"]
        total_loss += float(loss.detach().cpu()) * int(labels.numel())
        total_correct += int((logits.argmax(dim=-1) == labels).sum().detach().cpu())
        total_items += int(labels.numel())
    restore_master(params, master)
    if total_items == 0:
        return None, None
    return total_loss / total_items, total_correct / total_items


def mixed_perturbation_metrics(
    master: Dict[str, torch.Tensor],
    directions: Dict[str, torch.Tensor],
    states: Dict[str, smoke.RTNClipState],
    masks: Dict[str, torch.Tensor],
    h: float,
) -> Dict[str, object]:
    active = 0
    total = 0
    dot = torch.zeros((), device=next(iter(master.values())).device, dtype=torch.float64)
    delta_sq = torch.zeros_like(dot)
    intended_sq = torch.zeros_like(dot)
    for name, state in states.items():
        direction = directions[name].float()
        plus_base = master[name].float().add(direction, alpha=float(h))
        minus_base = master[name].float().add(direction, alpha=-float(h))
        plus_q = smoke.quantize_with_state(plus_base, state).float()
        minus_q = smoke.quantize_with_state(minus_base, state).float()
        mask = masks.get(name)
        if mask is not None:
            mask_b = mask.to(device=plus_base.device, dtype=torch.bool)
            plus = torch.where(mask_b, plus_base, plus_q)
            minus = torch.where(mask_b, minus_base, minus_q)
        else:
            plus = plus_q
            minus = minus_q
        delta = plus - minus
        intended = 2.0 * float(h) * direction
        active += int((delta != 0).sum().detach().cpu())
        total += int(delta.numel())
        dot += (delta.double() * intended.double()).sum()
        delta_sq += delta.double().square().sum()
        intended_sq += intended.double().square().sum()
    eps = 1e-12
    return {
        "active_frac": active / max(total, 1),
        "active_count": active,
        "total_count": total,
        "delta_q_norm": float(delta_sq.sqrt().detach().cpu()),
        "ideal_displacement_norm": float(intended_sq.sqrt().detach().cpu()),
        "alignment": float((dot / (delta_sq.sqrt() * intended_sq.sqrt() + eps)).detach().cpu()),
        "norm_ratio": float((delta_sq.sqrt() / (intended_sq.sqrt() + eps)).detach().cpu()),
        "zero_effective_displacement_frac": 1.0 - active / max(total, 1),
        "pair_shared_grid": True,
        "fresh_round_codes": True,
        "active_coordinates_full_precision_overlay": True,
    }


def zero_inactive_directions(
    master: Dict[str, torch.Tensor],
    masks: Dict[str, torch.Tensor],
    seed: int,
    h: float,
    step: int,
) -> Dict[str, torch.Tensor]:
    directions = rtn_batch.sample_directions_for_step(master, seed, 4, 1, h, step)
    for name, direction in list(directions.items()):
        mask = masks.get(name)
        if mask is None:
            directions[name] = torch.zeros_like(direction)
        else:
            directions[name] = direction * mask.to(device=direction.device, dtype=direction.dtype)
    return directions


def latest_state(run_dir: Path) -> Optional[Path]:
    return rtn_batch.latest_step_checkpoint(run_dir)


def train(args: argparse.Namespace) -> Dict[str, object]:
    output_dir = Path(args.output_dir)
    run_dir = output_dir / "run_extreme_sparse_0p1pct_sst5"
    run_dir.mkdir(parents=True, exist_ok=True)
    env = env_record()
    write_json(output_dir / "env.json", env)
    (output_dir / "commands.txt").write_text(" ".join(sys.argv) + "\n", encoding="utf-8")

    config: Dict[str, object] = {
        "experiment": "extreme_sparse_original_like_sst5",
        "paper_target": "Zeroth-Order Fine-Tuning of LLMs with Extreme Sparsity",
        "paper_sparse_fraction": "0.1%",
        "sparse_ratio": float(args.sparse_ratio),
        "sparse_ratio_interpreted_as_fraction": float(args.sparse_ratio),
        "sparse_selection": "global_topk_task_gradient_square_linear_weight",
        "active_parameter_dtype": "fp32",
        "inactive_linear_weight_quantization": "4bit_group_rtn_clip_fake_quant_once_at_start",
        "exact_paper_quantization_available": False,
        "exact_paper_repo_code_available": False,
        "fallback_note": "Official SensZOQ repository is a placeholder in this environment; frozen weights use local group RTNClip fake quantization.",
        "model": "roberta-large",
        "dataset": "SST-5",
        "dataset_mode": "full",
        "seed": int(args.seed),
        "data_seed": int(args.data_seed),
        "batch_size": int(args.batch_size),
        "eval_batch_size": int(args.eval_batch_size),
        "dataloader_shuffle": True,
        "sampler_required": "RandomSampler",
        "h": float(args.h),
        "lr": float(args.lr),
        "steps": int(args.steps),
        "eval_every": int(args.eval_every),
        "checkpoint_steps": int(args.checkpoint_steps),
        "mask_batches": int(args.mask_batches),
        "mask_scope": "linear_weight",
        "update_backend": "active_fp32_master_zo",
        "direct_int_update": False,
        "residual_grid": False,
        "gptq": False,
        "direction": "dense_on_active_sparse_mask_unscaled",
        "inactive_weights_frozen": True,
        "active_weights_trainable": True,
        "quant_grid_refresh": "once_at_start_from_initial_master",
        "repo_commit": env.get("git_commit"),
        "gpu_name": env.get("gpu_name"),
        "run_dir": str(run_dir),
    }
    write_json(run_dir / "run_config.json", config)
    write_json(run_dir / "run_manifest_row.json", config)

    rtn_batch.reset_run_seed(int(args.seed))
    device = torch.device("cuda" if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    if device.type != "cuda":
        raise RuntimeError("This RoBERTa-large smoke/training runner expects CUDA.")

    load_args = argparse.Namespace(
        repo_root=REPO_ROOT,
        model_id=args.model_id,
        task_name="sst-5",
        dataset_mode="full",
        data_seed=int(args.data_seed),
        num_k=16,
        data_dir=None,
        batch_size=int(args.batch_size),
        eval_batch_size=int(args.eval_batch_size),
    )
    model, train_loader, dev_loader, data_args, train_sampler = smoke.load_prompt_model_and_data(load_args, device)
    model.float()
    model.eval()
    params = smoke.named_parameter_map(model)
    q_names = [name for name in smoke.linear_weight_names(model) if name in params]
    if type(train_sampler).__name__ != "RandomSampler":
        raise RuntimeError(f"Expected RandomSampler, got {type(train_sampler).__name__}")

    master = {name: p.detach().clone().to(device=device, dtype=torch.float32) for name, p in params.items() if p.detach().is_floating_point()}
    restore_master(params, master)

    states, quant_rows = smoke.refresh_quantizer_states(master, q_names, 4, int(args.group_size))
    numel_by_name = {name: int(params[name].numel()) for name in q_names}
    quant_stats = smoke.aggregate_quantizer_stats(quant_rows, numel_by_name)
    append_jsonl(run_dir / "quantizer_diagnostics.jsonl", {"step": 0, "record_type": "initial_refresh_summary", **quant_stats})
    for row in quant_rows:
        append_jsonl(run_dir / "quantizer_diagnostics.jsonl", {"step": 0, "record_type": "per_module_initial_refresh", **row})

    sparse_masks, sparse_stats = rtn_batch.build_task_grad_sparse_masks(
        model,
        params,
        master,
        train_loader,
        device,
        sparse_ratio=float(args.sparse_ratio),
        quantized_names=q_names,
        mask_batches=int(args.mask_batches),
        mask_scope="linear_weight",
    )
    mask_hash = rtn_batch.sparse_mask_hash(sparse_masks)
    config.update(sparse_stats)
    config.update(
        {
            "data_dir_resolved": getattr(data_args, "data_dir", ""),
            "sampler_name": type(train_sampler).__name__,
            "quantized_modules": q_names,
            "sparse_mask_hash": mask_hash,
            "sparse_mask_saved_in_checkpoint": True,
            "active_param_count": int(sparse_stats.get("active_params_all", 0) or 0),
            "total_param_count": int(sparse_stats.get("total_params_all", 0) or 0),
            "active_param_frac": float(sparse_stats.get("mask_active_frac_all", 0.0) or 0.0),
        }
    )
    write_json(run_dir / "run_config.json", config)
    write_json(run_dir / "run_manifest_row.json", config)
    torch.save(rtn_batch.sparse_masks_to_cpu(sparse_masks), run_dir / "sparse_masks.pt")

    batch_iter = smoke.cycle(train_loader)
    start_step = 0
    best = {"best_eval_acc": None, "best_eval_step": None, "best_eval_loss": None, "best_eval_loss_step": None}
    ckpt = latest_state(run_dir)
    if ckpt is not None and args.resume:
        start_step, master, best_loaded, loaded_masks = rtn_batch.load_checkpoint(ckpt, device, master_dtype=torch.float32)
        best.update(best_loaded or {})
        if loaded_masks is not None:
            sparse_masks = rtn_batch.sparse_masks_to_device(loaded_masks, device)
        restore_master(params, master)
        for _ in range(start_step):
            next(batch_iter)

    metrics_path = run_dir / "metrics.csv"
    write_header = not metrics_path.exists() or metrics_path.stat().st_size == 0 or start_step == 0
    last_train_loss = None
    last_update_norm = None
    last_dh = None
    last_pert: Dict[str, object] = {}
    finite_steps = 0
    status = "running"
    error_message = ""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    with metrics_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "step",
                "loss_plus",
                "loss_minus",
                "train_loss",
                "d_h",
                "update_norm",
                "eval_loss",
                "eval_acc",
                "seconds",
                "nan_flag",
            ],
        )
        if write_header:
            writer.writeheader()
        for step_idx in range(start_step, int(args.steps)):
            step_t0 = time.time()
            directions = zero_inactive_directions(master, sparse_masks, int(args.seed), float(args.h), step_idx)
            batch = smoke.move_batch(next(batch_iter), device)
            mixed_copy_master_to_model(params, master, directions, float(args.h), +1.0, states, sparse_masks)
            loss_plus, _ = smoke.forward_loss_and_logits(model, batch)
            mixed_copy_master_to_model(params, master, directions, float(args.h), -1.0, states, sparse_masks)
            loss_minus, _ = smoke.forward_loss_and_logits(model, batch)
            restore_diff = restore_master(params, master)
            loss_plus_f = float(loss_plus.detach().cpu())
            loss_minus_f = float(loss_minus.detach().cpu())
            d_h = (loss_plus_f - loss_minus_f) / (2.0 * float(args.h))
            finite = math.isfinite(loss_plus_f) and math.isfinite(loss_minus_f) and math.isfinite(d_h)
            eval_loss = None
            eval_acc = None
            if finite:
                finite_steps += 1
                last_update_norm = update_active_master(master, directions, float(args.lr), d_h)
                restore_master(params, master)
            last_train_loss = (loss_plus_f + loss_minus_f) / 2.0
            last_dh = d_h
            completed_step = step_idx + 1

            if completed_step % int(args.diag_every) == 0 or completed_step == 1 or completed_step == int(args.steps):
                last_pert = mixed_perturbation_metrics(master, directions, states, sparse_masks, float(args.h))
                append_jsonl(run_dir / "perturbation_diagnostics.jsonl", {"step": completed_step, **last_pert})

            if completed_step % int(args.eval_every) == 0 or completed_step == int(args.steps):
                eval_loss, eval_acc = mixed_evaluate(
                    model,
                    params,
                    master,
                    states,
                    sparse_masks,
                    dev_loader,
                    device,
                    int(args.eval_batches),
                )
                append_jsonl(run_dir / "eval_metrics.jsonl", {"step": completed_step, "eval_loss": eval_loss, "eval_acc": eval_acc})
                if eval_acc is not None and (best["best_eval_acc"] is None or float(eval_acc) > float(best["best_eval_acc"])):
                    best["best_eval_acc"] = float(eval_acc)
                    best["best_eval_step"] = completed_step
                    rtn_batch.save_checkpoint(run_dir / "checkpoints" / f"step_{completed_step}", completed_step, master, best, config, sparse_masks=sparse_masks)
                    rtn_batch.copy_checkpoint(run_dir / "checkpoints" / f"step_{completed_step}", run_dir / "checkpoints" / "best_acc")
                if eval_loss is not None and (best["best_eval_loss"] is None or float(eval_loss) < float(best["best_eval_loss"])):
                    best["best_eval_loss"] = float(eval_loss)
                    best["best_eval_loss_step"] = completed_step
                    rtn_batch.save_checkpoint(run_dir / "checkpoints" / f"step_{completed_step}", completed_step, master, best, config, sparse_masks=sparse_masks)
                    rtn_batch.copy_checkpoint(run_dir / "checkpoints" / f"step_{completed_step}", run_dir / "checkpoints" / "best_loss")

            if completed_step % int(args.checkpoint_steps) == 0 or completed_step == int(args.steps):
                rtn_batch.save_checkpoint(run_dir / "checkpoints" / f"step_{completed_step}", completed_step, master, best, config, sparse_masks=sparse_masks)

            nan_flag = (not finite) or (restore_diff > 1e-5) or (last_update_norm is not None and not math.isfinite(float(last_update_norm)))
            writer.writerow(
                {
                    "step": completed_step,
                    "loss_plus": loss_plus_f,
                    "loss_minus": loss_minus_f,
                    "train_loss": last_train_loss,
                    "d_h": d_h,
                    "update_norm": last_update_norm,
                    "eval_loss": eval_loss,
                    "eval_acc": eval_acc,
                    "seconds": time.time() - step_t0,
                    "nan_flag": nan_flag,
                }
            )
            f.flush()
            if completed_step == 1 or completed_step % int(args.log_every) == 0:
                with (run_dir / "train.log").open("a", encoding="utf-8") as log:
                    log.write(
                        f"step={completed_step}/{args.steps} loss={last_train_loss:.6g} "
                        f"d_h={d_h:.6g} update_norm={last_update_norm} eval_acc={eval_acc} "
                        f"active={last_pert.get('active_frac')}\n"
                    )
            if nan_flag:
                status = "failed"
                error_message = f"non-finite or restore violation at step {completed_step}"
                break

    steps_completed = start_step
    eval_rows: List[Dict[str, str]] = []
    with metrics_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
        if rows:
            steps_completed = int(float(rows[-1]["step"]))
            eval_rows = [r for r in rows if r.get("eval_acc") not in (None, "")]
    rtn_batch.save_checkpoint(run_dir / "checkpoints" / "final", steps_completed, master, best, config, sparse_masks=sparse_masks)
    if not (run_dir / "checkpoints" / "best_acc").exists():
        rtn_batch.copy_checkpoint(run_dir / "checkpoints" / "final", run_dir / "checkpoints" / "best_acc")
    if not (run_dir / "checkpoints" / "best_loss").exists():
        rtn_batch.copy_checkpoint(run_dir / "checkpoints" / "final", run_dir / "checkpoints" / "best_loss")

    last_eval_acc = None
    last_eval_loss = None
    last_eval_step = None
    if eval_rows:
        row = eval_rows[-1]
        last_eval_acc = float(row["eval_acc"]) if row.get("eval_acc") else None
        last_eval_loss = float(row["eval_loss"]) if row.get("eval_loss") else None
        last_eval_step = int(float(row["step"]))
    if status != "failed":
        status = "complete" if steps_completed >= int(args.steps) else "partial"
    summary = {
        **config,
        "status": status,
        "error_message": error_message,
        "steps_completed": steps_completed,
        "best_eval_acc": best.get("best_eval_acc"),
        "best_eval_step": best.get("best_eval_step"),
        "best_eval_loss": best.get("best_eval_loss"),
        "best_eval_loss_step": best.get("best_eval_loss_step"),
        "last_eval_acc": last_eval_acc,
        "last_eval_loss": last_eval_loss,
        "last_eval_step": last_eval_step,
        "final_train_loss": last_train_loss,
        "d_h_last": last_dh,
        "d_h_finite_rate": finite_steps / max(steps_completed - start_step, 1),
        "update_norm_last": last_update_norm,
        "seconds_per_step": (time.time() - t0) / max(steps_completed - start_step, 1),
        "peak_gpu_mem_mb": float(torch.cuda.max_memory_allocated() / 1024 / 1024) if torch.cuda.is_available() else 0.0,
        **{f"perturb_{k}": v for k, v in last_pert.items()},
        **quant_stats,
    }
    write_json(run_dir / "run_summary.json", summary)
    write_json(output_dir / "summary.json", summary)
    with (output_dir / "summary.md").open("w", encoding="utf-8") as f:
        f.write("# Extreme Sparse SST-5 Summary\n\n")
        f.write(f"- status: `{status}`\n")
        f.write(f"- output: `{run_dir}`\n")
        f.write(f"- active fraction: `{summary.get('active_param_frac')}`\n")
        f.write(f"- best_eval_acc: `{summary.get('best_eval_acc')}` at step `{summary.get('best_eval_step')}`\n")
        f.write(f"- last_eval_acc: `{summary.get('last_eval_acc')}` at step `{summary.get('last_eval_step')}`\n")
        f.write(f"- exact paper repo code available: `{summary.get('exact_paper_repo_code_available')}`\n")
        f.write(f"- fallback note: {summary.get('fallback_note')}\n")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="outputs/extreme_sparse_original_sst5_seed16_20260523")
    parser.add_argument("--model_id", default="roberta-large")
    parser.add_argument("--device", choices=["cuda"], default="cuda")
    parser.add_argument("--seed", type=int, default=16)
    parser.add_argument("--data_seed", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--h", type=float, default=1e-3)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--eval_every", type=int, default=100)
    parser.add_argument("--eval_batches", type=int, default=-1)
    parser.add_argument("--checkpoint_steps", type=int, default=500)
    parser.add_argument("--diag_every", type=int, default=50)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--sparse_ratio", type=float, default=0.001)
    parser.add_argument("--mask_batches", type=int, default=1)
    parser.add_argument("--group_size", type=int, default=128)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = train(args)
    print(json.dumps({
        "output_dir": args.output_dir,
        "status": summary.get("status"),
        "steps_completed": summary.get("steps_completed"),
        "best_eval_acc": summary.get("best_eval_acc"),
        "last_eval_acc": summary.get("last_eval_acc"),
        "active_param_frac": summary.get("active_param_frac"),
    }, indent=2, sort_keys=True))
    return 0 if summary.get("status") in {"complete", "partial"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
