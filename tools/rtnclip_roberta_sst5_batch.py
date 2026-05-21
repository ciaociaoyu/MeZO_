#!/usr/bin/env python
"""Batch runner for RoBERTa-large / SST-5 G128 RTNClip low-bit MeZO.

The runner intentionally uses the shared-grid fake-quantized forward oracle:
scales/alphas come from the unperturbed FP16 master weight, while plus/minus
perturbed weights are freshly rounded on the same cached grid.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shutil
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


H_GRID: List[Tuple[str, float]] = [
    ("1e-5", 1e-5),
    ("3e-5", 3e-5),
    ("1e-4", 1e-4),
    ("3e-4", 3e-4),
    ("1e-3", 1e-3),
    ("1p5e-3", 1.5e-3),
    ("2e-3", 2e-3),
    ("3e-3", 3e-3),
    ("4e-3", 4e-3),
    ("5e-3", 5e-3),
    ("1e-2", 1e-2),
]

SMOKE_SUMMARY_COLUMNS = [
    "bitwidth",
    "h",
    "group_size",
    "quantizer_backend",
    "scale_refresh_k",
    "pair_shared_grid",
    "grid_source",
    "steps_completed",
    "seed",
    "data_seed",
    "batch_size",
    "shuffle",
    "sampler_name",
    "update_backend",
    "quantized_modules",
    "seconds_per_step",
    "peak_gpu_mem",
    "final_train_loss",
    "final_eval_acc_if_available",
    "d_h_finite_rate",
    "update_norm_last",
    "active_frac",
    "alignment",
    "norm_ratio",
    "zero_effective_displacement_frac",
    "saturation_frac_w",
    "saturation_frac_w_plus",
    "saturation_frac_w_minus",
    "recon_mse_global",
    "weight_recon_mse",
    "weight_recon_rel_mse",
    "weight_recon_sqnr_db",
    "delta_visibility_mse",
    "delta_visibility_nmse",
    "delta_visibility_rel_l2",
    "alpha_lt_1_frac",
    "num_scale_refreshes",
    "status",
    "warnings",
]

HSEARCH_SUMMARY_COLUMNS = [
    "run_name",
    "bitwidth",
    "h",
    "h_label",
    "scale_refresh_k",
    "status",
    "steps_completed",
    "seed",
    "data_seed",
    "batch_size",
    "gpu_name",
    "gpu_type_requested",
    "fallback_used",
    "best_eval_acc",
    "best_eval_step",
    "last_eval_acc",
    "last_eval_step",
    "best_eval_loss",
    "best_eval_loss_step",
    "last_eval_loss",
    "last_eval_loss_step",
    "final_train_loss",
    "update_variant",
    "perturbed_parameter_scope",
    "quantized_forward_scope",
    "active_frac",
    "alignment",
    "norm_ratio",
    "delta_q_norm",
    "ideal_displacement_norm",
    "code_change_frac",
    "delta_visibility_mse",
    "delta_visibility_nmse",
    "delta_visibility_rel_l2",
    "saturation_frac_w",
    "saturation_frac_w_plus",
    "saturation_frac_w_minus",
    "weight_recon_mse",
    "weight_recon_rel_mse",
    "weight_recon_sqnr_db",
    "corr_fd_true",
    "nMSE_fd_true",
    "fd_true_available",
    "fd_true_mse",
    "fd_true_nmse",
    "fd_true_rmse",
    "fd_true_bias",
    "run_dir",
    "resume_command",
    "warnings",
]


def write_csv(path: Path, rows: List[Dict[str, object]], columns: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def append_line(path: Path, line: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(line.rstrip() + "\n")


def stable_h_key(h: float) -> int:
    return int(round(float(h) * 1_000_000_000_000))


def sample_directions_for_step(master: Dict[str, torch.Tensor], seed: int, bitwidth: int, scale_refresh_k: int, h: float, step: int):
    first = next(iter(master.values()))
    gen = torch.Generator(device=first.device)
    gen.manual_seed(int(seed) + bitwidth * 1009 + scale_refresh_k * 9176 + stable_h_key(h) + step * 1_000_003)
    return smoke.sample_directions(master, gen)


def latest_step_checkpoint(run_dir: Path) -> Optional[Path]:
    ckpt_root = run_dir / "checkpoints"
    if not ckpt_root.exists():
        return None
    best: Optional[Tuple[int, Path]] = None
    for path in ckpt_root.glob("step_*"):
        m = re.fullmatch(r"step_(\d+)", path.name)
        if m and path.is_dir():
            item = (int(m.group(1)), path)
            if best is None or item[0] > best[0]:
                best = item
    return None if best is None else best[1]


def load_checkpoint(path: Path, device: torch.device):
    payload = torch.load(path / "state.pt", map_location=device)
    master = {k: v.to(device=device, dtype=torch.float16) for k, v in payload["master"].items()}
    return int(payload["step"]), master, payload.get("best", {})


def save_checkpoint(path: Path, step: int, master: Dict[str, torch.Tensor], best: Dict[str, object], config: Dict[str, object]) -> None:
    path.mkdir(parents=True, exist_ok=True)
    cpu_master = {k: v.detach().cpu().to(dtype=torch.float16) for k, v in master.items()}
    torch.save({"step": step, "master": cpu_master, "best": best, "config": config}, path / "state.pt")
    smoke.write_json(path / "checkpoint_manifest.json", {"step": step, "created_at": datetime.now().isoformat(), "keys": len(cpu_master)})


def copy_checkpoint(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def log_to(run_dir: Path, message: str) -> None:
    line = f"[{datetime.now().isoformat(timespec='seconds')}] {message}"
    print(line, flush=True)
    append_line(run_dir / "train.log", line)


def write_resume_command(run_dir: Path, config: Dict[str, object], manifest_path: Optional[str] = None) -> None:
    if manifest_path:
        cmd = (
            f"CUDA_VISIBLE_DEVICES=0 DATALOADER_SHUFFLE=True python tools/rtnclip_roberta_sst5_batch.py "
            f"run-manifest --manifest {manifest_path} --only-run-name {config['run_name']}"
        )
    else:
        cmd = (
            f"CUDA_VISIBLE_DEVICES=0 DATALOADER_SHUFFLE=True python tools/rtnclip_roberta_sst5_batch.py train-one "
            f"--run_dir {run_dir} --run_name {config['run_name']} --bitwidth {config['bitwidth']} "
            f"--h {config['h']} --h_label {config.get('h_label', '')} --steps {config['max_steps']} "
            f"--scale_refresh_k {config['scale_refresh_k']} --lr {config['lr']} "
            f"--eval_every {config['eval_every']} --checkpoint_steps {config['checkpoint_steps']} "
            f"--eval_batch_size {config['eval_batch_size']} --eval_batches {config['eval_batches']}"
        )
    (run_dir / "resume_command.txt").write_text(cmd + "\n", encoding="utf-8")


def make_train_config(args, run_dir: Path, run_name: str, bitwidth: int, h: float, h_label: str, steps: int, scale_refresh_k: int, phase: str) -> Dict[str, object]:
    env = smoke.collect_env(REPO_ROOT)
    return {
        "run_name": run_name,
        "phase": phase,
        "model": "roberta-large",
        "dataset": "SST-5",
        "dataset_mode": "full",
        "seed": 16,
        "data_seed": 16,
        "batch_size": 64,
        "shuffle": True,
        "DATALOADER_SHUFFLE": os.environ.get("DATALOADER_SHUFFLE", ""),
        "direction": "dense",
        "update_variant": "standard",
        "estimator": "two_point_symmetric_mezo",
        "h": float(h),
        "h_label": h_label,
        "max_steps": int(steps),
        "eval_every": int(args.eval_every),
        "checkpoint_steps": int(args.checkpoint_steps),
        "eval_batch_size": int(args.eval_batch_size),
        "eval_batches": int(args.eval_batches),
        "lr": float(args.lr),
        "update_backend": "fp16_master",
        "master_dtype": "fp16",
        "perturbed_parameter_scope": "full_dense_all_trainable",
        "quantized_forward_scope": "Linear.weight_only",
        "perturbation_diagnostics_scope": "quantized_linear_weights_only",
        "direct_int_update": False,
        "quantizer_backend": "G128_groupwise_RTNClip_fake_quant",
        "quantizer": "group_rtn_clip",
        "bitwidth": int(bitwidth),
        "quant_bits": int(bitwidth),
        "group_size": 128,
        "scale_refresh_k": int(scale_refresh_k),
        "grid_refresh_k": int(scale_refresh_k),
        "pair_shared_grid": True,
        "fresh_round_codes": True,
        "grid_source": "unperturbed_fp16_master_weight",
        "scale_source": "unperturbed_master_w_t",
        "activation_quantization": False,
        "real_int_packing": False,
        "zero_point": "none",
        "rounding": "deterministic_round_to_nearest",
        "excluded_methods": ["GPTQ", "residual_grid", "direct_int_update", "sparse", "LoRA", "RTE", "MNLI", "OPT", "Mistral"],
        "run_dir": str(run_dir),
        "gpu_name": env.get("gpu_name", ""),
        "gpu_type_requested": os.environ.get("REQUESTED_GPU_TYPE", "local"),
        "fallback_used": os.environ.get("FALLBACK_USED", "0") in {"1", "true", "True"},
        "env": env,
    }


def should_skip_complete(run_dir: Path, target_steps: Optional[int] = None) -> bool:
    summary = run_dir / "run_summary.json"
    final = run_dir / "checkpoints" / "final" / "state.pt"
    if not summary.exists() or not final.exists():
        return False
    try:
        data = read_json(summary)
        if data.get("status") != "complete":
            return False
        if target_steps is None:
            return True
        return int(data.get("steps_completed", 0) or 0) >= int(target_steps)
    except Exception:
        return False


def evaluate_full(model, params, master, states, dev_loader, device, max_batches: int):
    if max_batches == 0:
        return None, None
    smoke.copy_master_to_model(params, master, None, 0.0, 0.0, states)
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
    smoke.restore_master(params, master)
    if total_items == 0:
        return None, None
    return total_loss / total_items, total_correct / total_items


def advance_batch_iter(batch_iter, n: int) -> None:
    for _ in range(max(0, n)):
        next(batch_iter)


def train_one(args, run_dir: Path, run_name: str, bitwidth: int, h: float, h_label: str, steps: int, scale_refresh_k: int, phase: str, manifest_path: Optional[str] = None) -> Dict[str, object]:
    run_dir.mkdir(parents=True, exist_ok=True)
    if should_skip_complete(run_dir, steps):
        log_to(run_dir, f"skip complete run {run_name}")
        return read_json(run_dir / "run_summary.json")

    for file_name in ("metrics.csv", "eval_metrics.jsonl", "quantizer_diagnostics.jsonl", "perturbation_diagnostics.jsonl"):
        if not (run_dir / file_name).exists():
            (run_dir / file_name).touch()

    config = make_train_config(args, run_dir, run_name, bitwidth, h, h_label, steps, scale_refresh_k, phase)
    smoke.write_json(run_dir / "run_config.json", config)
    smoke.write_json(run_dir / "run_manifest_row.json", config)
    write_resume_command(run_dir, config, manifest_path)

    if os.environ.get("DATALOADER_SHUFFLE") != "True":
        summary = {**config, "status": "failed", "error_message": "DATALOADER_SHUFFLE is not True", "steps_completed": 0}
        smoke.write_json(run_dir / "run_summary.json", summary)
        return summary

    device = torch.device("cuda")
    model, train_loader, dev_loader, _, train_sampler = smoke.load_prompt_model_and_data(argparse.Namespace(repo_root=REPO_ROOT, model_id="roberta-large", seed=16, data_seed=16, batch_size=64, eval_batch_size=args.eval_batch_size), device)
    params = smoke.named_parameter_map(model)
    q_names = smoke.linear_weight_names(model)
    numel_by_name = {name: params[name].numel() for name in q_names}
    config["sampler_name"] = type(train_sampler).__name__
    config["quantized_modules"] = q_names
    smoke.write_json(run_dir / "run_config.json", config)
    smoke.write_json(run_dir / "run_manifest_row.json", config)

    if type(train_sampler).__name__ != "RandomSampler":
        summary = {**config, "status": "failed", "error_message": "sampler is not RandomSampler", "steps_completed": 0}
        smoke.write_json(run_dir / "run_summary.json", summary)
        return summary

    start_step = 0
    best = {"best_eval_acc": None, "best_eval_step": None, "best_eval_loss": None, "best_eval_loss_step": None}
    ckpt = latest_step_checkpoint(run_dir)
    previous_steps = 0
    summary_path = run_dir / "run_summary.json"
    if summary_path.exists():
        try:
            previous_steps = int(read_json(summary_path).get("steps_completed", 0) or 0)
        except Exception:
            previous_steps = 0
    should_resume = ckpt is not None and (previous_steps < int(steps) or not (run_dir / "checkpoints" / "final" / "state.pt").exists())
    if should_resume:
        start_step, master, best_loaded = load_checkpoint(ckpt, device)
        best.update(best_loaded or {})
        smoke.restore_master(params, master)
        log_to(run_dir, f"resuming {run_name} from {ckpt} at step {start_step}")
    else:
        master = {name: p.detach().clone().to(device=device, dtype=torch.float16) for name, p in params.items() if p.detach().is_floating_point()}
        smoke.restore_master(params, master)
        log_to(run_dir, f"starting {run_name}")

    batch_iter = smoke.cycle(train_loader)
    if start_step:
        advance_batch_iter(batch_iter, start_step)

    states: Dict[str, smoke.RTNClipState] = {}
    last_refresh_stats: List[Dict[str, object]] = []
    last_quant: Dict[str, object] = {}
    last_pert: Dict[str, object] = {}
    last_train_loss = None
    update_norm_last = None
    finite_count = 0
    num_refreshes = 0
    status = "running"
    error_message = ""
    warnings: List[str] = []
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    metrics_path = run_dir / "metrics.csv"
    write_header = not metrics_path.exists() or metrics_path.stat().st_size == 0
    total_start = time.time()
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
                "scale_refreshed",
                "eval_loss",
                "eval_acc",
                "nan_flag",
            ],
        )
        if write_header:
            writer.writeheader()
        for step_idx in range(start_step, steps):
            step_start = time.time()
            scale_refreshed = step_idx % scale_refresh_k == 0 or not states
            if scale_refreshed:
                states, last_refresh_stats = smoke.refresh_quantizer_states(master, q_names, bitwidth, 128)
                num_refreshes += 1
                last_quant = smoke.aggregate_quantizer_stats(last_refresh_stats, numel_by_name)
                if step_idx % args.quant_log_every == 0:
                    smoke.append_jsonl(run_dir / "quantizer_diagnostics.jsonl", {"step": step_idx, "record_type": "refresh_summary", "grid_id": num_refreshes, "scale_id": num_refreshes, "scale_refresh_k": scale_refresh_k, **last_quant})
                    for row in last_refresh_stats:
                        smoke.append_jsonl(run_dir / "quantizer_diagnostics.jsonl", {"step": step_idx, "record_type": "per_module_refresh", "grid_id": num_refreshes, "scale_id": num_refreshes, **row})

            directions = sample_directions_for_step(master, 16, bitwidth, scale_refresh_k, h, step_idx)
            batch = smoke.move_batch(next(batch_iter), device)
            smoke.copy_master_to_model(params, master, directions, h, +1.0, states)
            loss_plus, _ = smoke.forward_loss_and_logits(model, batch)
            smoke.copy_master_to_model(params, master, directions, h, -1.0, states)
            loss_minus, _ = smoke.forward_loss_and_logits(model, batch)
            restore_diff = smoke.restore_master(params, master)
            loss_plus_f = float(loss_plus.detach().cpu())
            loss_minus_f = float(loss_minus.detach().cpu())
            d_h = (loss_plus_f - loss_minus_f) / (2.0 * h)
            finite = math.isfinite(loss_plus_f) and math.isfinite(loss_minus_f) and math.isfinite(d_h)
            if finite:
                finite_count += 1
                update_norm_last = smoke.update_master(master, directions, float(args.lr), d_h)
                smoke.restore_master(params, master)
            last_train_loss = (loss_plus_f + loss_minus_f) / 2.0

            if step_idx % args.diag_every == 0 or step_idx == steps - 1:
                last_pert = smoke.perturbation_metrics(master, directions, states, h)
                last_pert["code_change_frac"] = last_pert["active_frac"]
                last_pert["grid_id_plus"] = num_refreshes
                last_pert["grid_id_minus"] = num_refreshes
                last_pert["scale_id_plus"] = num_refreshes
                last_pert["scale_id_minus"] = num_refreshes
                smoke.append_jsonl(run_dir / "perturbation_diagnostics.jsonl", {"step": step_idx + 1, **last_pert})

            eval_loss = None
            eval_acc = None
            completed_step = step_idx + 1
            if completed_step % int(args.eval_every) == 0 or completed_step == steps:
                eval_loss, eval_acc = evaluate_full(model, params, master, states, dev_loader, device, int(args.eval_batches))
                eval_row = {"step": completed_step, "eval_loss": eval_loss, "eval_acc": eval_acc}
                smoke.append_jsonl(run_dir / "eval_metrics.jsonl", eval_row)
                if eval_acc is not None and (best["best_eval_acc"] is None or eval_acc > best["best_eval_acc"]):
                    best["best_eval_acc"] = eval_acc
                    best["best_eval_step"] = completed_step
                    save_checkpoint(run_dir / "checkpoints" / f"step_{completed_step}", completed_step, master, best, config)
                    copy_checkpoint(run_dir / "checkpoints" / f"step_{completed_step}", run_dir / "checkpoints" / "best_acc")
                if eval_loss is not None and (best["best_eval_loss"] is None or eval_loss < best["best_eval_loss"]):
                    best["best_eval_loss"] = eval_loss
                    best["best_eval_loss_step"] = completed_step
                    save_checkpoint(run_dir / "checkpoints" / f"step_{completed_step}", completed_step, master, best, config)
                    copy_checkpoint(run_dir / "checkpoints" / f"step_{completed_step}", run_dir / "checkpoints" / "best_loss")

            if completed_step % int(args.checkpoint_steps) == 0 or completed_step == steps:
                save_checkpoint(run_dir / "checkpoints" / f"step_{completed_step}", completed_step, master, best, config)

            nan_flag = not finite or restore_diff > 1e-3 or (update_norm_last is not None and not math.isfinite(float(update_norm_last)))
            writer.writerow(
                {
                    "step": completed_step,
                    "loss_plus": loss_plus_f,
                    "loss_minus": loss_minus_f,
                    "train_loss": last_train_loss,
                    "d_h": d_h,
                    "d_h_finite": finite,
                    "update_norm": update_norm_last,
                    "seconds": time.time() - step_start,
                    "scale_refreshed": scale_refreshed,
                    "eval_loss": eval_loss,
                    "eval_acc": eval_acc,
                    "nan_flag": nan_flag,
                }
            )
            f.flush()
            if nan_flag:
                status = "failed"
                error_message = f"non-finite or restore violation at step {completed_step}"
                break
            if completed_step % int(args.log_every) == 0 or completed_step == 1:
                log_to(run_dir, f"step={completed_step}/{steps} loss={last_train_loss:.6g} d_h={d_h:.6g} eval_acc={eval_acc} active={last_pert.get('active_frac')}")

    steps_completed = 0
    try:
        with metrics_path.open(newline="", encoding="utf-8") as mf:
            rows = list(csv.DictReader(mf))
            steps_completed = int(float(rows[-1]["step"])) if rows else start_step
            last_eval_rows = [r for r in rows if r.get("eval_acc") not in (None, "")]
    except Exception:
        last_eval_rows = []
        steps_completed = start_step

    save_checkpoint(run_dir / "checkpoints" / "final", steps_completed, master, best, config)
    if not (run_dir / "checkpoints" / "best_acc").exists():
        copy_checkpoint(run_dir / "checkpoints" / "final", run_dir / "checkpoints" / "best_acc")
    if not (run_dir / "checkpoints" / "best_loss").exists():
        copy_checkpoint(run_dir / "checkpoints" / "final", run_dir / "checkpoints" / "best_loss")

    last_eval_acc = None
    last_eval_loss = None
    last_eval_step = None
    if last_eval_rows:
        row = last_eval_rows[-1]
        last_eval_acc = float(row["eval_acc"]) if row["eval_acc"] else None
        last_eval_loss = float(row["eval_loss"]) if row["eval_loss"] else None
        last_eval_step = int(float(row["step"]))

    if status != "failed":
        status = "complete" if steps_completed >= steps else "partial"
    peak_mem = float(torch.cuda.max_memory_allocated() / 1024 / 1024) if torch.cuda.is_available() else 0.0
    total_runtime = time.time() - total_start
    new_steps = max(steps_completed - start_step, 1)
    summary = {
        **config,
        "status": status,
        "error_message": error_message,
        "steps_completed": steps_completed,
        "best_eval_acc": best.get("best_eval_acc"),
        "best_eval_step": best.get("best_eval_step"),
        "last_eval_acc": last_eval_acc,
        "last_eval_step": last_eval_step,
        "best_eval_loss": best.get("best_eval_loss"),
        "best_eval_loss_step": best.get("best_eval_loss_step"),
        "last_eval_loss": last_eval_loss,
        "last_eval_loss_step": last_eval_step,
        "final_train_loss": last_train_loss,
        "d_h_finite_rate": finite_count / max(steps_completed - start_step, 1),
        "update_norm_last": update_norm_last,
        "update_variant": config.get("update_variant"),
        "perturbed_parameter_scope": config.get("perturbed_parameter_scope"),
        "quantized_forward_scope": config.get("quantized_forward_scope"),
        "active_frac": last_pert.get("active_frac"),
        "alignment": last_pert.get("alignment"),
        "norm_ratio": last_pert.get("norm_ratio"),
        "delta_q_norm": last_pert.get("delta_q_norm"),
        "ideal_displacement_norm": last_pert.get("ideal_displacement_norm"),
        "code_change_frac": last_pert.get("code_change_frac"),
        "delta_visibility_mse": last_pert.get("delta_visibility_mse"),
        "delta_visibility_nmse": last_pert.get("delta_visibility_nmse"),
        "delta_visibility_rel_l2": last_pert.get("delta_visibility_rel_l2"),
        "saturation_frac_w": last_quant.get("saturation_frac_w"),
        "saturation_frac_w_plus": last_pert.get("saturation_frac_w_plus"),
        "saturation_frac_w_minus": last_pert.get("saturation_frac_w_minus"),
        "clip_frac": last_quant.get("clip_frac"),
        "recon_mse_global": last_quant.get("recon_mse_global"),
        "weight_recon_mse": last_quant.get("weight_recon_mse", last_quant.get("recon_mse_global")),
        "weight_recon_rel_mse": last_quant.get("weight_recon_rel_mse"),
        "weight_recon_sqnr_db": last_quant.get("weight_recon_sqnr_db"),
        "alpha_lt_1_frac": last_quant.get("alpha_lt_1_frac"),
        "num_scale_refreshes": num_refreshes,
        "peak_gpu_mem": peak_mem,
        "total_runtime": total_runtime,
        "seconds_per_step": total_runtime / new_steps,
        "corr_fd_true": None,
        "nMSE_fd_true": None,
        "fd_true_available": False,
        "fd_true_mse": None,
        "fd_true_nmse": None,
        "fd_true_rmse": None,
        "fd_true_bias": None,
        "true_grad_diagnostics": "unavailable_not_computed",
        "warnings": "; ".join(warnings),
    }
    smoke.write_json(run_dir / "run_summary.json", summary)
    log_to(run_dir, f"finished {run_name} status={status} steps={steps_completed}")
    return summary


def run_probe_grid(args) -> List[Dict[str, object]]:
    root = Path(args.output_root)
    bitwidth = int(args.bitwidth or 4)
    if bitwidth not in {4, 8}:
        raise ValueError(f"probe bitwidth must be 4 or 8, got {bitwidth}")
    probe_name = f"int{bitwidth}_probe"
    probe_dir = root / probe_name
    probe_dir.mkdir(parents=True, exist_ok=True)
    env = smoke.collect_env(REPO_ROOT)
    smoke.write_json(probe_dir / "env.json", env)
    smoke.write_json(probe_dir / "run_config.json", {
        "phase": probe_name,
        "model": "roberta-large",
        "dataset": "SST-5",
        "dataset_mode": "full",
        "seed": 16,
        "data_seed": 16,
        "batch_size": 64,
        "DATALOADER_SHUFFLE": os.environ.get("DATALOADER_SHUFFLE", ""),
        "bitwidth": bitwidth,
        "group_size": 128,
        "scale_refresh_k": 1,
        "pair_shared_grid": True,
        "fresh_round_codes": True,
        "grid_source": "unperturbed_fp16_master_weight",
        "h_grid": [{"label": label, "h": h} for label, h in H_GRID],
        "k_dirs": int(args.probe_dirs),
    })

    device = torch.device("cuda")
    model, train_loader, _, _, train_sampler = smoke.load_prompt_model_and_data(argparse.Namespace(repo_root=REPO_ROOT, model_id="roberta-large", seed=16, data_seed=16, batch_size=64, eval_batch_size=args.eval_batch_size), device)
    params = smoke.named_parameter_map(model)
    q_names = smoke.linear_weight_names(model)
    numel_by_name = {name: params[name].numel() for name in q_names}
    master = {name: p.detach().clone().to(device=device, dtype=torch.float16) for name, p in params.items() if p.detach().is_floating_point()}
    states, refresh_stats = smoke.refresh_quantizer_states(master, q_names, bitwidth, 128)
    quant = smoke.aggregate_quantizer_stats(refresh_stats, numel_by_name)
    batch = smoke.move_batch(next(iter(train_loader)), device)
    rows: List[Dict[str, object]] = []
    stats_path = probe_dir / "probe_stats.jsonl"
    if stats_path.exists():
        stats_path.unlink()
    for label, h in H_GRID:
        acc = {
            "active_frac": [],
            "alignment": [],
            "norm_ratio": [],
            "delta_visibility_mse": [],
            "delta_visibility_nmse": [],
            "delta_visibility_rel_l2": [],
            "code_change_frac": [],
            "clip_frac": [],
            "saturation_frac": [],
            "fd": [],
            "finite": [],
            "loss_plus": [],
            "loss_minus": [],
        }
        for k in range(int(args.probe_dirs)):
            directions = sample_directions_for_step(master, 16 + k, bitwidth, 1, h, k)
            smoke.copy_master_to_model(params, master, directions, h, +1.0, states)
            loss_plus, _ = smoke.forward_loss_and_logits(model, batch)
            smoke.copy_master_to_model(params, master, directions, h, -1.0, states)
            loss_minus, _ = smoke.forward_loss_and_logits(model, batch)
            smoke.restore_master(params, master)
            lp = float(loss_plus.detach().cpu())
            lm = float(loss_minus.detach().cpu())
            fd = (lp - lm) / (2.0 * h)
            pert = smoke.perturbation_metrics(master, directions, states, h)
            pert["code_change_frac"] = pert["active_frac"]
            finite = math.isfinite(lp) and math.isfinite(lm) and math.isfinite(fd)
            item = {"h": h, "h_label": label, "k_dir": k, "loss_plus": lp, "loss_minus": lm, "fd": fd, "finite": finite, **pert}
            smoke.append_jsonl(stats_path, item)
            for key in (
                "active_frac",
                "alignment",
                "norm_ratio",
                "delta_visibility_mse",
                "delta_visibility_nmse",
                "delta_visibility_rel_l2",
                "code_change_frac",
                "clip_frac",
                "saturation_frac",
            ):
                acc[key].append(float(pert[key]))
            acc["fd"].append(fd)
            acc["finite"].append(1.0 if finite else 0.0)
            acc["loss_plus"].append(lp)
            acc["loss_minus"].append(lm)
        row = {
            "h_label": label,
            "h": h,
            "bitwidth": bitwidth,
            "scale_refresh_k": 1,
            "probe_dirs": int(args.probe_dirs),
            "finite_rate": sum(acc["finite"]) / len(acc["finite"]),
            "active_frac": sum(acc["active_frac"]) / len(acc["active_frac"]),
            "alignment": sum(acc["alignment"]) / len(acc["alignment"]),
            "norm_ratio": sum(acc["norm_ratio"]) / len(acc["norm_ratio"]),
            "delta_visibility_mse": sum(acc["delta_visibility_mse"]) / len(acc["delta_visibility_mse"]),
            "delta_visibility_nmse": sum(acc["delta_visibility_nmse"]) / len(acc["delta_visibility_nmse"]),
            "delta_visibility_rel_l2": sum(acc["delta_visibility_rel_l2"]) / len(acc["delta_visibility_rel_l2"]),
            "code_change_frac": sum(acc["code_change_frac"]) / len(acc["code_change_frac"]),
            "clip_frac": sum(acc["clip_frac"]) / len(acc["clip_frac"]),
            "saturation_frac": sum(acc["saturation_frac"]) / len(acc["saturation_frac"]),
            "loss_plus_mean": sum(acc["loss_plus"]) / len(acc["loss_plus"]),
            "loss_minus_mean": sum(acc["loss_minus"]) / len(acc["loss_minus"]),
            "fd_mean": sum(acc["fd"]) / len(acc["fd"]),
            "corr_fd_true": None,
            "nMSE_fd_true": None,
            "fd_true_available": False,
            "fd_true_mse": None,
            "fd_true_nmse": None,
            "fd_true_rmse": None,
            "fd_true_bias": None,
            "true_grad_diagnostics": "unavailable_not_computed",
            **{
                k: quant.get(k)
                for k in (
                    "saturation_frac_w",
                    "recon_mse_global",
                    "weight_recon_mse",
                    "weight_recon_rel_mse",
                    "weight_recon_sqnr_db",
                    "alpha_lt_1_frac",
                )
            },
        }
        rows.append(row)

    viable_rows = [r for r in rows if r["finite_rate"] == 1.0 and r["active_frac"] >= 0.02 and r["alignment"] > 0.05]
    selectable_rows = [r for r in viable_rows if r["h_label"] != "1e-2"]
    if not viable_rows:
        verdict = "collapsed_window"
        selected = None
    else:
        selected_pool = selectable_rows or viable_rows
        selected = max(selected_pool, key=lambda r: (r["alignment"], -abs(math.log(max(r["norm_ratio"], 1e-12)))))
        verdict = "viable_window" if len(viable_rows) >= 3 else "narrow_window"
    small = [r for r in rows if r["h"] <= 1e-4]
    bad_small = min(small, key=lambda r: (r["active_frac"], r["alignment"])) if small else rows[0]
    bad_large = next(r for r in rows if r["h_label"] == "1e-2")
    write_csv(root / f"{probe_name}_summary.csv", rows, list(rows[0].keys()))
    write_csv(root / probe_name / f"{probe_name}_summary.csv", rows, list(rows[0].keys()))
    smoke.write_json(root / f"{probe_name}_verdict.json", {
        "verdict": verdict,
        "selected_h": None if selected is None else selected["h"],
        "selected_h_label": None if selected is None else selected["h_label"],
        "bad_small_h": bad_small["h"],
        "bad_small_h_label": bad_small["h_label"],
        "bad_large_h": bad_large["h"],
        "bad_large_h_label": bad_large["h_label"],
    })
    md = [
        f"# INT{bitwidth} RTNClip Probe Summary",
        "",
        f"Verdict: `{verdict}`",
        f"Selected h: `{None if selected is None else selected['h_label']}`",
        f"Bad small h: `{bad_small['h_label']}`",
        "Bad large h: `1e-2`",
        "",
        "| h | active_frac | alignment | norm_ratio | finite_rate |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for r in rows:
        md.append(f"| {r['h_label']} | {r['active_frac']:.4g} | {r['alignment']:.4g} | {r['norm_ratio']:.4g} | {r['finite_rate']:.3g} |")
    (root / f"{probe_name}_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return rows


def run_smoke_phase(args) -> List[Dict[str, object]]:
    root = Path(args.output_root)
    smoke_root = root / "smoke"
    configs = [
        ("int8_wo_g128_rtnclip_sharedgrid_k1_h1e-3_step50", 8, 1),
        ("int8_wo_g128_rtnclip_sharedgrid_k10_h1e-3_step50", 8, 10),
        ("int4_wo_g128_rtnclip_sharedgrid_k1_h1e-3_step50", 4, 1),
        ("int4_wo_g128_rtnclip_sharedgrid_k10_h1e-3_step50", 4, 10),
    ]
    rows = []
    for run_name, bitwidth, k in configs:
        run_dir = smoke_root / run_name
        rows.append(train_one(args, run_dir, run_name, bitwidth, 1e-3, "1e-3", 50, k, "smoke"))
    smoke_json = {r["run_name"]: r for r in rows}
    smoke.write_json(root / "smoke_summary.json", smoke_json)
    write_csv(root / "smoke_summary.csv", rows, SMOKE_SUMMARY_COLUMNS)
    all_pass = all(r.get("status") == "complete" and int(r.get("steps_completed", 0)) == 50 for r in rows)
    missing = []
    for r in rows:
        rd = Path(r["run_dir"])
        for name in ("run_config.json", "run_manifest_row.json", "metrics.csv", "eval_metrics.jsonl", "run_summary.json", "resume_command.txt"):
            if not (rd / name).exists():
                missing.append(f"{r['run_name']}:{name}")
    md = [
        "# RTNClip Low-Bit Smoke Summary",
        "",
        f"All smoke checks passed: {'yes' if all_pass and not missing else 'no'}",
        f"Missing required files: {', '.join(missing) if missing else 'none'}",
        "",
        "| run | status | sec/step | loss | active_frac | alignment | norm_ratio | refreshes |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in rows:
        md.append(
            f"| {r['run_name']} | {r['status']} | {r.get('seconds_per_step')} | {r.get('final_train_loss')} | "
            f"{r.get('active_frac')} | {r.get('alignment')} | {r.get('norm_ratio')} | {r.get('num_scale_refreshes')} |"
        )
    (root / "smoke_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    if not all_pass or missing:
        smoke.write_json(root / "smoke_failure_report.json", {"all_pass": all_pass, "missing": missing, "rows": rows})
        raise SystemExit("smoke failed; refusing to proceed")
    return rows


def run_manifest(args) -> List[Dict[str, object]]:
    manifest = Path(args.manifest)
    rows = list(csv.DictReader(manifest.open(newline="", encoding="utf-8")))
    out = []
    for row in rows:
        if args.only_run_name and row["run_name"] != args.only_run_name:
            continue
        run_dir = Path(row["run_dir"])
        out.append(
            train_one(
                args,
                run_dir,
                row["run_name"],
                int(row["bitwidth"]),
                float(row["h"]),
                row["h_label"],
                int(row["max_steps"]),
                int(row["scale_refresh_k"]),
                row.get("phase", "int8_hsearch"),
                str(manifest),
            )
        )
    summarize(args.output_root)
    return out


def summarize(output_root: str) -> None:
    root = Path(output_root)
    rows = []
    manifest_path = root / "int8_hsearch_manifest.csv"
    manifest_rows = []
    if manifest_path.exists():
        manifest_rows = list(csv.DictReader(manifest_path.open(newline="", encoding="utf-8")))
    summary_by_dir = {str(path.parent): read_json(path) for path in root.glob("int8_hsearch/**/run_summary.json")}
    if manifest_rows:
        for mrow in manifest_rows:
            run_dir = Path(mrow["run_dir"])
            item = dict(summary_by_dir.get(str(run_dir), {}))
            item.setdefault("run_name", mrow["run_name"])
            item.setdefault("bitwidth", int(mrow["bitwidth"]))
            item.setdefault("h", float(mrow["h"]))
            item.setdefault("h_label", mrow["h_label"])
            item.setdefault("scale_refresh_k", int(mrow["scale_refresh_k"]))
            item.setdefault("seed", int(mrow["seed"]))
            item.setdefault("data_seed", int(mrow["data_seed"]))
            item.setdefault("batch_size", int(mrow["batch_size"]))
            item.setdefault("status", "pending")
            item.setdefault("steps_completed", 0)
            metrics_path = run_dir / "metrics.csv"
            if metrics_path.exists() and metrics_path.stat().st_size > 0 and "run_summary.json" not in {p.name for p in run_dir.glob("run_summary.json")}:
                try:
                    metrics = list(csv.DictReader(metrics_path.open(newline="", encoding="utf-8")))
                    if metrics:
                        item["steps_completed"] = int(float(metrics[-1]["step"]))
                        item["final_train_loss"] = float(metrics[-1]["train_loss"]) if metrics[-1].get("train_loss") else None
                        eval_rows = [r for r in metrics if r.get("eval_acc") not in (None, "")]
                        if eval_rows:
                            item["last_eval_acc"] = float(eval_rows[-1]["eval_acc"])
                            item["last_eval_loss"] = float(eval_rows[-1]["eval_loss"])
                            item["last_eval_step"] = int(float(eval_rows[-1]["step"]))
                        item["status"] = "running"
                except Exception as exc:
                    item["warnings"] = f"summary metrics parse failed: {exc}"
            if (run_dir / "run_summary.json").exists():
                item["status"] = item.get("status", "complete")
            item["run_dir"] = str(run_dir)
            resume = run_dir / "resume_command.txt"
            item["resume_command"] = resume.read_text(encoding="utf-8").strip() if resume.exists() else ""
            rows.append(item)
    else:
        for path in sorted(root.glob("int8_hsearch/**/run_summary.json")):
            item = read_json(path)
            item["run_dir"] = str(path.parent)
            resume = path.parent / "resume_command.txt"
            item["resume_command"] = resume.read_text(encoding="utf-8").strip() if resume.exists() else ""
            rows.append(item)
    if rows:
        write_csv(root / "int8_hsearch_summary.csv", rows, HSEARCH_SUMMARY_COLUMNS)
        md = [
            "# INT8 RTNClip H-Search Summary",
            "",
            "| h | status | steps | best_acc | last_acc | best_loss | last_loss | run_dir |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
        for r in rows:
            md.append(f"| {r.get('h_label')} | {r.get('status')} | {r.get('steps_completed')} | {r.get('best_eval_acc')} | {r.get('last_eval_acc')} | {r.get('best_eval_loss')} | {r.get('last_eval_loss')} | `{r.get('run_dir')}` |")
        (root / "int8_hsearch_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    verdict_path = root / "int4_probe_verdict.json"
    if verdict_path.exists():
        verdict = read_json(verdict_path)
        if verdict["verdict"] == "collapsed_window":
            text = "# INT4 Training Plan\n\nINT4 probe verdict is `collapsed_window`; full INT4 training was not launched.\n"
        else:
            text = (
                "# INT4 Training Plan\n\n"
                f"INT4 probe verdict is `{verdict['verdict']}`. Candidate h values for a later approved training stage: "
                f"`1e-3`, `{verdict['bad_small_h_label']}`, `{verdict['selected_h_label']}`, `1e-2`.\n"
            )
        (root / "int4_training_plan_or_results.md").write_text(text, encoding="utf-8")

    jobs_md = root / "scheduler_jobs.md"
    job_logs = sorted((root / "jobs").glob("*")) if (root / "jobs").exists() else []
    lines = ["# Scheduler Jobs", ""]
    for p in job_logs:
        if p.is_file() and p.name in {"job_ids.txt", "dry_run_table.txt"}:
            lines.extend([f"## {p.name}", "", "```", p.read_text(encoding="utf-8", errors="replace").strip(), "```", ""])
    jobs_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_root", default=str(REPO_ROOT / "outputs" / "rtnclip_lowbit_roberta_sst5_seed16"))
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--eval_every", type=int, default=1000)
    parser.add_argument("--checkpoint_steps", type=int, default=1000)
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--eval_batches", type=int, default=-1, help="-1 means full dev set, 0 disables eval, positive caps batches.")
    parser.add_argument("--diag_every", type=int, default=100)
    parser.add_argument("--quant_log_every", type=int, default=1000)
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--probe_dirs", type=int, default=8)
    parser.add_argument("--manifest")
    parser.add_argument("--only-run-name", default="")
    parser.add_argument("--run_dir")
    parser.add_argument("--run_name")
    parser.add_argument("--bitwidth", type=int)
    parser.add_argument("--h", type=float)
    parser.add_argument("--h_label", default="")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--scale_refresh_k", type=int)
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("smoke")
    sub.add_parser("probe-int4")
    sub.add_parser("run-manifest")
    sub.add_parser("summarize")
    sub.add_parser("train-one")
    args = parser.parse_args()
    return args


def main() -> int:
    args = parse_args()
    if args.cmd in {"smoke", "probe-int4", "run-manifest", "train-one"}:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for RoBERTa-large RTNClip batch experiments.")
        if os.environ.get("DATALOADER_SHUFFLE") != "True":
            raise RuntimeError("DATALOADER_SHUFFLE=True must be exported.")
    if args.cmd == "smoke":
        run_smoke_phase(args)
    elif args.cmd == "probe-int4":
        run_probe_grid(args)
    elif args.cmd == "run-manifest":
        run_manifest(args)
    elif args.cmd == "train-one":
        train_one(args, Path(args.run_dir), args.run_name, args.bitwidth, args.h, args.h_label, args.steps, args.scale_refresh_k, "manual")
    elif args.cmd == "summarize":
        summarize(args.output_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
