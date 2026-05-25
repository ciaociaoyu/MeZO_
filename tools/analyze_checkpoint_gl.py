#!/usr/bin/env python
"""Backprop-based G/L checkpoint diagnostics for RTNClip RoBERTa runs.

This script is intentionally analysis-only: it loads saved FP16-master
checkpoints, rebuilds the SST-5 prompt model/data path, and estimates:

  G = ||grad loss(theta)||_2
  L_dir = ||grad(theta + eps v) - grad(theta - eps v)|| / (2 eps ||v||)

The L estimate is a directional lower-bound/proxy for local smoothness, not an
exact Hessian spectral norm.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import re
import socket
import sys
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, List, Optional, Tuple

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import smoke_rtnclip_roberta_sst5 as smoke  # noqa: E402


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def checkpoint_sort_key(path: Path):
    name = path.parent.name
    match = re.match(r"step_(\d+)$", name)
    if match:
        return (0, int(match.group(1)), name)
    order = {"best_acc": 1, "best_loss": 2, "final": 3}
    return (order.get(name, 9), 10**12, name)


def discover_checkpoints(run_dir: Path, include_aliases: bool) -> List[Path]:
    paths = sorted((run_dir / "checkpoints").glob("*/state.pt"), key=checkpoint_sort_key)
    if include_aliases:
        return paths
    return [p for p in paths if re.match(r"step_\d+$", p.parent.name)]


def move_batch(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device) for k, v in batch.items()}


def forward_loss(model, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
    batch = dict(batch)
    batch["token_type_ids"] = torch.zeros_like(batch["input_ids"])
    outputs = model(**batch)
    return outputs[0].float()


def load_checkpoint_master(path: Path, device: torch.device, dtype: torch.dtype) -> Tuple[int, Dict[str, torch.Tensor], Dict[str, object]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    step = int(payload.get("step", -1))
    master = {
        name: tensor.to(device=device, dtype=dtype, non_blocking=True)
        for name, tensor in payload["master"].items()
        if torch.is_tensor(tensor) and tensor.is_floating_point()
    }
    return step, master, payload.get("best", {})


def set_weights(
    params: Dict[str, torch.nn.Parameter],
    master: Dict[str, torch.Tensor],
    states: Dict[str, smoke.RTNClipState],
    surface: str,
    directions: Optional[Dict[str, torch.Tensor]] = None,
    eps: float = 0.0,
    sign: float = 0.0,
) -> None:
    with torch.no_grad():
        for name, param in params.items():
            base = master[name]
            if surface == "quantized_forward_view" and name in states:
                base = smoke.quantize_with_state(master[name], states[name])
            if directions is not None and eps:
                value = base.float().add(directions[name].float(), alpha=sign * eps)
            else:
                value = base
            param.copy_(value.to(dtype=param.dtype))


def zero_model_grads(model) -> None:
    model.zero_grad(set_to_none=True)


def grad_stats(
    model,
    params: Dict[str, torch.nn.Parameter],
    batch: Dict[str, torch.Tensor],
    store: bool = False,
) -> Tuple[float, float, float, int, Optional[Dict[str, torch.Tensor]]]:
    zero_model_grads(model)
    loss = forward_loss(model, batch)
    loss.backward()
    grad_sq = torch.zeros((), device=loss.device, dtype=torch.float64)
    grad_abs_max = 0.0
    numel = 0
    stored = {} if store else None
    with torch.no_grad():
        for name, param in params.items():
            grad = param.grad
            if grad is None:
                continue
            gf = grad.detach().float()
            grad_sq += gf.double().square().sum()
            grad_abs_max = max(grad_abs_max, float(gf.abs().max().detach().cpu()))
            numel += gf.numel()
            if store:
                stored[name] = gf.detach().clone()
    return float(loss.detach().cpu()), float(grad_sq.sqrt().detach().cpu()), grad_abs_max, numel, stored


def direction_stats(directions: Dict[str, torch.Tensor]) -> Tuple[float, int]:
    sq = torch.zeros((), device=next(iter(directions.values())).device, dtype=torch.float64)
    numel = 0
    with torch.no_grad():
        for tensor in directions.values():
            tf = tensor.float()
            sq += tf.double().square().sum()
            numel += tf.numel()
    return float(sq.sqrt().detach().cpu()), numel


def sample_direction(master: Dict[str, torch.Tensor], seed: int) -> Dict[str, torch.Tensor]:
    generator = torch.Generator(device=next(iter(master.values())).device)
    generator.manual_seed(int(seed))
    return smoke.sample_directions(master, generator)


def gradient_difference_l(
    model,
    params: Dict[str, torch.nn.Parameter],
    master: Dict[str, torch.Tensor],
    states: Dict[str, smoke.RTNClipState],
    surface: str,
    batch: Dict[str, torch.Tensor],
    eps: float,
    seed: int,
) -> Dict[str, float]:
    directions = sample_direction(master, seed)
    direction_norm, direction_numel = direction_stats(directions)

    set_weights(params, master, states, surface, directions=directions, eps=eps, sign=-1.0)
    loss_minus, grad_minus_norm, _, grad_numel, grad_minus = grad_stats(model, params, batch, store=True)

    set_weights(params, master, states, surface, directions=directions, eps=eps, sign=+1.0)
    loss_plus, grad_plus_norm, _, _, _ = grad_stats(model, params, batch, store=False)

    diff_sq = torch.zeros((), device=next(iter(master.values())).device, dtype=torch.float64)
    rayleigh_num = torch.zeros_like(diff_sq)
    with torch.no_grad():
        for name, param in params.items():
            if param.grad is None or name not in grad_minus:
                continue
            diff = param.grad.detach().float() - grad_minus[name]
            diff_sq += diff.double().square().sum()
            rayleigh_num += (diff.double() * directions[name].double()).sum()
    diff_norm = float(diff_sq.sqrt().detach().cpu())
    denom = max(2.0 * eps * direction_norm, 1e-30)
    direction_sq = max(direction_norm * direction_norm, 1e-30)
    l_dir = diff_norm / denom
    rayleigh_abs = abs(float(rayleigh_num.detach().cpu())) / max(2.0 * eps * direction_sq, 1e-30)

    del grad_minus, directions
    zero_model_grads(model)
    return {
        "l_eps": eps,
        "l_direction_seed": seed,
        "direction_norm": direction_norm,
        "direction_numel": direction_numel,
        "loss_plus": loss_plus,
        "loss_minus": loss_minus,
        "grad_plus_norm": grad_plus_norm,
        "grad_minus_norm": grad_minus_norm,
        "grad_diff_norm": diff_norm,
        "L_dir_grad_diff_norm_over_step_norm": l_dir,
        "L_rayleigh_abs": rayleigh_abs,
        "grad_numel": grad_numel,
    }


def exact_hvp_l(
    model,
    params: Dict[str, torch.nn.Parameter],
    master: Dict[str, torch.Tensor],
    states: Dict[str, smoke.RTNClipState],
    surface: str,
    batch: Dict[str, torch.Tensor],
    seed: int,
) -> Dict[str, float]:
    directions = sample_direction(master, seed)
    direction_norm, direction_numel = direction_stats(directions)
    set_weights(params, master, states, surface)
    zero_model_grads(model)
    loss = forward_loss(model, batch)
    param_names = list(params.keys())
    param_list = [params[name] for name in param_names]
    grads = torch.autograd.grad(loss, param_list, create_graph=True, allow_unused=True)
    dot = torch.zeros((), device=loss.device, dtype=torch.float32)
    for name, grad in zip(param_names, grads):
        if grad is not None:
            dot = dot + (grad.float() * directions[name].float()).sum()
    hvps = torch.autograd.grad(dot, param_list, allow_unused=True)
    hvp_sq = torch.zeros((), device=loss.device, dtype=torch.float64)
    rayleigh_num = torch.zeros_like(hvp_sq)
    hvp_numel = 0
    with torch.no_grad():
        for name, hvp in zip(param_names, hvps):
            if hvp is None:
                continue
            hf = hvp.detach().float()
            hvp_sq += hf.double().square().sum()
            rayleigh_num += (hf.double() * directions[name].double()).sum()
            hvp_numel += hf.numel()
    hvp_norm = float(hvp_sq.sqrt().detach().cpu())
    direction_sq = max(direction_norm * direction_norm, 1e-30)
    del grads, hvps, directions
    zero_model_grads(model)
    return {
        "l_direction_seed": seed,
        "direction_norm": direction_norm,
        "direction_numel": direction_numel,
        "loss_center": float(loss.detach().cpu()),
        "hvp_norm": hvp_norm,
        "L_hvp_norm_over_direction_norm": hvp_norm / max(direction_norm, 1e-30),
        "L_hvp_rayleigh_abs": abs(float(rayleigh_num.detach().cpu())) / direction_sq,
        "hvp_numel": hvp_numel,
    }


def load_eval_metrics(run_dir: Path) -> Dict[int, Dict[str, float]]:
    out = {}
    path = run_dir / "eval_metrics.jsonl"
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "step" in row:
            out[int(row["step"])] = row
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--surface", default="quantized_forward_view", choices=["quantized_forward_view", "fp16_master"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--model_id", default="roberta-large")
    parser.add_argument("--task_name", default="sst-5")
    parser.add_argument("--dataset_mode", default="full")
    parser.add_argument("--seed", type=int, default=16)
    parser.add_argument("--data_seed", type=int, default=16)
    parser.add_argument("--num_k", type=int, default=16)
    parser.add_argument("--data_dir", default=None)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--bitwidth", type=int, default=4)
    parser.add_argument("--group_size", type=int, default=128)
    parser.add_argument("--l_eps", type=float, default=1e-4)
    parser.add_argument("--l_dirs", type=int, default=1)
    parser.add_argument("--l_method", default="grad_diff", choices=["grad_diff", "hvp"])
    parser.add_argument("--autograd_dtype", default="fp16", choices=["fp16", "fp32"])
    parser.add_argument("--checkpoint_names", nargs="*", default=None)
    parser.add_argument("--max_checkpoints", type=int, default=0)
    parser.add_argument("--include_aliases", action="store_true")
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(args.device)

    env = {
        "timestamp": datetime.now().isoformat(),
        "hostname": socket.gethostname(),
        "python": sys.version,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "surface": args.surface,
        "run_dir": str(run_dir),
    }
    write_json(out_dir / "env.json", env)
    write_json(out_dir / "config.json", vars(args))

    model, train_loader, _dev_loader, data_args, train_sampler = smoke.load_prompt_model_and_data(
        SimpleNamespace(
            repo_root=REPO_ROOT,
            model_id=args.model_id,
            task_name=args.task_name,
            seed=args.seed,
            data_seed=args.data_seed,
            batch_size=args.batch_size,
            eval_batch_size=args.eval_batch_size,
            dataset_mode=args.dataset_mode,
            data_dir=args.data_dir,
            num_k=args.num_k,
        ),
        device,
    )
    master_dtype = torch.float16
    if args.autograd_dtype == "fp32":
        model.float()
        master_dtype = torch.float32
    # Use eval mode so dropout does not inject artificial variation into the
    # checkpoint-to-checkpoint G/L trend. Gradients are still enabled.
    model.eval()
    params = smoke.named_parameter_map(model)
    q_names = smoke.linear_weight_names(model)
    batch = move_batch(next(iter(train_loader)), device)
    sampler_name = type(train_sampler).__name__
    eval_by_step = load_eval_metrics(run_dir)

    checkpoints = discover_checkpoints(run_dir, args.include_aliases)
    if args.max_checkpoints > 0:
        checkpoints = checkpoints[: args.max_checkpoints]
    if args.checkpoint_names:
        wanted = set(args.checkpoint_names)
        checkpoints = [p for p in checkpoints if p.parent.name in wanted]
    if not checkpoints:
        raise RuntimeError(f"No checkpoints found under {run_dir / 'checkpoints'}")

    rows: List[Dict[str, object]] = []
    jsonl_path = out_dir / "gl_checkpoint_records.jsonl"
    csv_path = out_dir / "gl_checkpoint_summary.csv"
    with jsonl_path.open("w") as jf:
        for index, ckpt in enumerate(checkpoints, start=1):
            start = time.time()
            step, master, best = load_checkpoint_master(ckpt, device, master_dtype)
            # Some prompt-model versions expose the MLM bias under
            # lm_head.decoder.bias, while the saved master uses lm_head.bias.
            if "lm_head.decoder.bias" not in master and "lm_head.bias" in master:
                master["lm_head.decoder.bias"] = master["lm_head.bias"]
            active_params = {name: param for name, param in params.items() if name in master}
            active_master = {name: master[name] for name in active_params}
            active_q_names = [name for name in q_names if name in active_master]
            numel_by_name = {name: active_params[name].numel() for name in active_q_names}
            states = {}
            refresh_summary = {}
            if args.surface == "quantized_forward_view":
                states, refresh_rows = smoke.refresh_quantizer_states(active_master, active_q_names, args.bitwidth, args.group_size)
                refresh_summary = smoke.aggregate_quantizer_stats(refresh_rows, numel_by_name)

            set_weights(active_params, active_master, states, args.surface)
            base_loss, g_norm, g_abs_max, grad_numel, _ = grad_stats(model, active_params, batch, store=False)

            l_records = []
            for d in range(args.l_dirs):
                # Use the same probe direction(s) for every checkpoint so the
                # checkpoint trend is not dominated by direction resampling.
                seed = args.seed + d
                if args.l_method == "hvp":
                    l_records.append(
                        exact_hvp_l(
                            model,
                            active_params,
                            active_master,
                            states,
                            args.surface,
                            batch,
                            seed,
                        )
                    )
                else:
                    l_records.append(
                        gradient_difference_l(
                            model,
                            active_params,
                            active_master,
                            states,
                            args.surface,
                            batch,
                            args.l_eps,
                            seed,
                        )
                    )

            set_weights(active_params, active_master, states, args.surface)
            if args.l_method == "hvp":
                l_dir_vals = [r["L_hvp_norm_over_direction_norm"] for r in l_records]
                l_ray_vals = [r["L_hvp_rayleigh_abs"] for r in l_records]
            else:
                l_dir_vals = [r["L_dir_grad_diff_norm_over_step_norm"] for r in l_records]
                l_ray_vals = [r["L_rayleigh_abs"] for r in l_records]
            eval_row = eval_by_step.get(step, {})
            row = {
                "checkpoint_name": ckpt.parent.name,
                "checkpoint_path": str(ckpt),
                "step": step,
                "surface": args.surface,
                "batch_size": args.batch_size,
                "sampler_name": sampler_name,
                "loss": base_loss,
                "G_grad_norm": g_norm,
                "G_grad_abs_max": g_abs_max,
                "grad_numel": grad_numel,
                "L_dir_mean": sum(l_dir_vals) / len(l_dir_vals),
                "L_dir_max": max(l_dir_vals),
                "L_rayleigh_abs_mean": sum(l_ray_vals) / len(l_ray_vals),
                "L_rayleigh_abs_max": max(l_ray_vals),
                "L_method": args.l_method,
                "l_eps": args.l_eps,
                "l_dirs": args.l_dirs,
                "eval_acc": eval_row.get("eval_acc"),
                "eval_loss": eval_row.get("eval_loss"),
                "best_eval_acc": best.get("best_eval_acc"),
                "best_eval_step": best.get("best_eval_step"),
                "runtime_sec": time.time() - start,
                "quant_recon_mse_global": refresh_summary.get("recon_mse_global"),
                "quant_clip_frac_w": refresh_summary.get("clip_frac_w") or refresh_summary.get("clip_frac"),
                "quant_alpha_lt_1_frac": refresh_summary.get("alpha_lt_1_frac"),
            }
            rows.append(row)
            jf.write(json.dumps(row, sort_keys=True) + "\n")
            jf.flush()
            print(
                f"[{index}/{len(checkpoints)}] {ckpt.parent.name} step={step} "
                f"loss={base_loss:.6g} G={g_norm:.6g} L_dir={row['L_dir_mean']:.6g} "
                f"L_ray={row['L_rayleigh_abs_mean']:.6g} runtime={row['runtime_sec']:.1f}s",
                flush=True,
            )

            del master, active_master, active_params, states
            zero_model_grads(model)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    fieldnames = list(rows[0].keys())
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # Write a small markdown report.
    lines = [
        "# Backprop G/L Checkpoint Trend",
        "",
        f"- Run: `{run_dir}`",
        f"- Surface: `{args.surface}`",
        f"- Checkpoints: {len(rows)}",
        f"- Batch: first deterministic RandomSampler train batch, batch_size={args.batch_size}, data_seed={args.data_seed}",
        f"- G: `||grad loss(theta)||_2` over all trainable parameters.",
        f"- L method: `{args.l_method}`.",
        f"- For grad_diff: `||g(theta+eps v)-g(theta-eps v)||/(2 eps ||v||)` with eps={args.l_eps}.",
        f"- For hvp: exact autograd Hessian-vector product `||Hv||/||v||`.",
        f"- L_rayleigh_abs: directional absolute Rayleigh curvature.",
        "",
        "| checkpoint | step | loss | G_grad_norm | L_dir_mean | L_rayleigh_abs_mean | eval_acc | runtime_sec |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {checkpoint_name} | {step} | {loss:.6g} | {G_grad_norm:.6g} | "
            "{L_dir_mean:.6g} | {L_rayleigh_abs_mean:.6g} | {eval_acc} | {runtime_sec:.1f} |".format(
                **{**row, "eval_acc": "NA" if row.get("eval_acc") is None else f"{row['eval_acc']:.6g}"}
            )
        )
    lines.extend(
        [
            "",
            "## Files",
            "",
            f"- `{csv_path}`",
            f"- `{jsonl_path}`",
            f"- `{out_dir / 'env.json'}`",
            f"- `{out_dir / 'config.json'}`",
        ]
    )
    (out_dir / "gl_checkpoint_report.md").write_text("\n".join(lines) + "\n")
    print(f"Wrote {csv_path}")
    print(f"Wrote {out_dir / 'gl_checkpoint_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
