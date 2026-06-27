#!/usr/bin/env python
"""Compare OPT checkpoint finite-difference L estimates with clean gradient HVP.

This is a diagnostic-only tool.  It reuses the same OPT/SST option batch and
checkpoint loading path as ``probe_opt13b_int4_checkpoint_hstar.py``, then
estimates clean smooth-loss curvature by finite-differencing true gradients:

    H v ~= (grad(w + eps v) - grad(w)) / eps
    lambda_v = |v^T H v| / ||v||^2

The low-bit RTNClip forward is intentionally not used for HVP because the
rounding quantizer is not classically differentiable.
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
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Sequence

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = REPO_ROOT / "tools"
LARGE_MODELS_DIR = REPO_ROOT / "large_models"
for path in (TOOLS_DIR, LARGE_MODELS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import probe_opt13b_int4_checkpoint_hstar as ckprobe  # noqa: E402
import probe_opt13b_int4_task_grid as optprobe  # noqa: E402
import train_opt13b_int4_dense_smoke as opttrain  # noqa: E402
from analyze_int4_sst5_calibrated_hstar import EPS, direction_norm_sq  # noqa: E402


def write_json(path: Path, data: Any) -> None:
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
    }
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(torch.cuda.current_device())
        out["gpu_name"] = props.name
        out["gpu_total_memory_mb"] = int(props.total_memory / 1024 / 1024)
    return out


def parse_float_list(raw: Sequence[str]) -> List[float]:
    vals: List[float] = []
    for item in raw:
        for part in str(item).replace(",", " ").split():
            vals.append(float(part))
    return vals


def grad_snapshot(params: Dict[str, torch.nn.Parameter], names: Sequence[str]) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for name in names:
        grad = params[name].grad
        if grad is None:
            out[name] = torch.zeros_like(params[name], dtype=torch.float32)
        else:
            out[name] = grad.detach().clone().float()
    return out


def grad_dot_direction_from_snapshot(grads: Dict[str, torch.Tensor], directions: Dict[str, torch.Tensor]) -> float:
    total = torch.zeros((), device=next(iter(directions.values())).device, dtype=torch.float64)
    for name, direction in directions.items():
        total += (grads[name].double() * direction.double()).sum()
    return float(total.detach().cpu())


def hvp_rayleigh(
    params: Dict[str, torch.nn.Parameter],
    grad0: Dict[str, torch.Tensor],
    directions: Dict[str, torch.Tensor],
    eps: float,
) -> Dict[str, float]:
    hv_dot_v = torch.zeros((), device=next(iter(directions.values())).device, dtype=torch.float64)
    hv_norm_sq = torch.zeros_like(hv_dot_v)
    grad_delta_norm_sq = torch.zeros_like(hv_dot_v)
    for name, direction in directions.items():
        grad = params[name].grad
        if grad is None:
            hv = -grad0[name].double() / float(eps)
        else:
            hv = (grad.detach().double() - grad0[name].double()) / float(eps)
        d = direction.double()
        hv_dot_v += (hv * d).sum()
        hv_norm_sq += hv.square().sum()
        grad_delta_norm_sq += (hv * float(eps)).square().sum()
    norm_sq = direction_norm_sq(directions)
    return {
        "hvp_rayleigh_signed": float((hv_dot_v / (norm_sq + EPS)).detach().cpu()),
        "hvp_rayleigh_abs": float((hv_dot_v.abs() / (norm_sq + EPS)).detach().cpu()),
        "hvp_norm_over_v_norm": float((hv_norm_sq.sqrt() / math.sqrt(norm_sq + EPS)).detach().cpu()),
        "grad_delta_norm": float(grad_delta_norm_sq.sqrt().detach().cpu()),
    }


def apply_clean(
    params: Dict[str, torch.nn.Parameter],
    master32: Dict[str, torch.Tensor],
    directions: Dict[str, torch.Tensor] | None,
    eps: float,
) -> None:
    with torch.no_grad():
        for name, tensor in master32.items():
            value = tensor
            if directions is not None and name in directions:
                value = value + float(eps) * directions[name].float()
            params[name].copy_(value.to(dtype=params[name].dtype))


def backward_clean(model: torch.nn.Module, params: Dict[str, torch.nn.Parameter], batch: Any) -> float:
    model.zero_grad(set_to_none=True)
    loss = optprobe.forward_loss(model, batch)
    loss.backward()
    return float(loss.detach().cpu())


def summarize(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"q50": float("nan"), "q90": float("nan"), "mean": float("nan")}
    t = torch.tensor(values, dtype=torch.float64)
    return {
        "q50": float(torch.quantile(t, 0.50)),
        "q90": float(torch.quantile(t, 0.90)),
        "mean": float(t.mean()),
    }


def formula_h(delta: float, g: float, lval: float, d: int) -> float:
    if not (delta > 0 and g > 0 and lval > 0 and d > 0):
        return float("nan")
    return math.sqrt(delta * g / (4.0 * lval * math.sqrt(float(d) * float(d + 2))))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--checkpoint_tags", nargs="+", default=["best_acc", "final"])
    parser.add_argument("--task", default="sst-2", choices=["sst-2", "sst-5", "rte", "mnli", "trec"])
    parser.add_argument("--model_id", default="facebook/opt-1.3b")
    parser.add_argument("--task_path", choices=["mezo_option"], default="mezo_option")
    parser.add_argument("--dataset_mode", choices=["full", "fewshot", "auto"], default="full")
    parser.add_argument("--num_train", type=int, default=-1)
    parser.add_argument("--num_k", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--eval_samples", type=int, default=0)
    parser.add_argument("--max_seq_len", type=int, default=128)
    parser.add_argument("--bitwidth", type=int, default=4)
    parser.add_argument("--group_size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=16)
    parser.add_argument("--data_seed", type=int, default=16)
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--eps_values", nargs="+", default=["1e-5", "3e-5", "1e-4", "3e-4", "1e-3"])
    parser.add_argument("--num_dirs", type=int, default=2)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for OPT-1.3B HVP diagnostic.")
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    eps_values = parse_float_list(args.eps_values)
    write_json(output_root / "run_config.json", {**vars(args), "eps_values": eps_values})
    write_json(output_root / "env.json", env_info())

    device = torch.device("cuda")
    model, tokenizer = optprobe.load_model_and_tokenizer(args, device)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id
    opttrain.patch_mezo_option_loss(model)
    params = optprobe.params_map(model)
    q_names = optprobe.linear_weight_names(model, params)
    base_master = optprobe.make_master(params, torch.float16)
    _task, train_loader, _eval_loader, train_count, _eval_count = opttrain.load_mezo_option_loaders(args, tokenizer)
    batch = opttrain.prepare_batch(next(iter(train_loader)), device)
    perturb_names = list(base_master.keys())
    run_dir = Path(args.run_dir)

    raw_rows: List[Dict[str, object]] = []
    summary_rows: List[Dict[str, object]] = []
    old_tf32_matmul = torch.backends.cuda.matmul.allow_tf32
    old_tf32_cudnn = torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        for tag in args.checkpoint_tags:
            master = {name: tensor.detach().clone() for name, tensor in base_master.items()}
            step = ckprobe.load_checkpoint_master(run_dir / "checkpoints" / tag / "master.pt", master, device)
            states, _ = optprobe.refresh_states(master, q_names, int(args.bitwidth), int(args.group_size))
            delta_stats = optprobe.weighted_delta_with_optional_masks(states, None)
            delta = float(delta_stats["delta_int4_rtnclip_scale_rms"]) / math.sqrt(6.0)
            d_trainable = sum(int(t.numel()) for t in master.values())

            master32 = ckprobe.refresh_master32(master)
            model.float()
            apply_clean(params, master32, None, 0.0)
            base_loss = backward_clean(model, params, batch)
            grad0 = grad_snapshot(params, perturb_names)
            for direction_id in range(int(args.num_dirs)):
                directions = optprobe.sample_direction(
                    master32,
                    perturb_names,
                    int(args.seed) + direction_id * 1009 + 88000,
                    masks=None,
                )
                norm_sq = direction_norm_sq(directions)
                g_dot_v = grad_dot_direction_from_snapshot(grad0, directions)
                g_abs_from_dir = math.sqrt(math.pi / 2.0) * abs(g_dot_v)
                for eps in eps_values:
                    apply_clean(params, master32, directions, float(eps))
                    loss_plus = backward_clean(model, params, batch)
                    hvp = hvp_rayleigh(params, grad0, directions, float(eps))
                    row = {
                        "checkpoint_tag": tag,
                        "checkpoint_step": step,
                        "direction_id": direction_id,
                        "eps": float(eps),
                        "base_loss": base_loss,
                        "loss_plus": loss_plus,
                        "g_dot_v": g_dot_v,
                        "G_abs_from_direction": g_abs_from_dir,
                        "direction_norm_sq": norm_sq,
                        **hvp,
                    }
                    raw_rows.append(row)
                    print(
                        f"{tag}@{step} dir={direction_id} eps={eps:g} "
                        f"lambda_abs={row['hvp_rayleigh_abs']:.6g}",
                        flush=True,
                    )
                    write_csv(output_root / "hvp_raw.csv", raw_rows)
                    model.zero_grad(set_to_none=True)
                    apply_clean(params, master32, None, 0.0)

                del directions
                torch.cuda.empty_cache()

            rows_this = [r for r in raw_rows if r["checkpoint_tag"] == tag]
            for eps in eps_values:
                vals = [float(r["hvp_rayleigh_abs"]) for r in rows_this if abs(float(r["eps"]) - float(eps)) <= 1e-15]
                hs = summarize(vals)
                g_vals = [
                    float(r["G_abs_from_direction"])
                    for r in rows_this
                    if abs(float(r["eps"]) - float(eps)) <= 1e-15
                ]
                gs = summarize(g_vals)
                summary_rows.append(
                    {
                        "checkpoint_tag": tag,
                        "checkpoint_step": step,
                        "eps": float(eps),
                        "hvp_L_q50": hs["q50"],
                        "hvp_L_q90": hs["q90"],
                        "hvp_L_mean": hs["mean"],
                        "G_clean_dir_q50": gs["q50"],
                        "G_clean_dir_q90": gs["q90"],
                        "G_clean_dir_mean": gs["mean"],
                        "Delta_scale_rms_over_sqrt6": delta,
                        "d_trainable": d_trainable,
                        "hstar_cleanG_hvpL_q90": formula_h(delta, gs["q50"], hs["q90"], d_trainable),
                    }
                )
            write_csv(output_root / "hvp_summary.csv", summary_rows)
            model.half()
            optprobe.restore_master(params, master)
            del master, master32, grad0, states
            torch.cuda.empty_cache()
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_tf32_matmul
        torch.backends.cudnn.allow_tf32 = old_tf32_cudnn

    write_csv(output_root / "hvp_raw.csv", raw_rows)
    write_csv(output_root / "hvp_summary.csv", summary_rows)
    write_json(
        output_root / "run_summary.json",
        {
            "status": "complete",
            "rows": len(summary_rows),
            "train_sample_count": train_count,
            "summary_csv": str(output_root / "hvp_summary.csv"),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
