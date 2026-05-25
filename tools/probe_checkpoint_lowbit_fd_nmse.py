#!/usr/bin/env python
"""Post-hoc low-bit finite-difference probe for a saved RTNClip checkpoint."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import socket
import sys
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import smoke_rtnclip_roberta_sst5 as smoke  # noqa: E402

EPS = 1e-30


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(data, f, indent=2, sort_keys=True)


def append_jsonl(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(data, sort_keys=True) + "\n")


def write_csv(path: Path, rows: List[Dict[str, object]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def corr(xs: List[float], ys: List[float]) -> Optional[float]:
    pairs = [(float(x), float(y)) for x, y in zip(xs, ys) if math.isfinite(float(x)) and math.isfinite(float(y))]
    if len(pairs) < 2:
        return None
    xv = [p[0] for p in pairs]
    yv = [p[1] for p in pairs]
    mx = sum(xv) / len(xv)
    my = sum(yv) / len(yv)
    vx = sum((x - mx) ** 2 for x in xv)
    vy = sum((y - my) ** 2 for y in yv)
    if vx <= EPS or vy <= EPS:
        return None
    return sum((x - mx) * (y - my) for x, y in pairs) / math.sqrt(vx * vy)


def load_master(checkpoint: Path, device: torch.device, dtype: torch.dtype) -> tuple[int, Dict[str, torch.Tensor], Dict[str, object]]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    master = {
        name: tensor.to(device=device, dtype=dtype)
        for name, tensor in payload["master"].items()
        if torch.is_tensor(tensor) and tensor.is_floating_point()
    }
    if "lm_head.decoder.bias" not in master and "lm_head.bias" in master:
        master["lm_head.decoder.bias"] = master["lm_head.bias"]
    return int(payload.get("step", -1)), master, payload.get("best", {})


def move_batch(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device) for k, v in batch.items()}


def forward_loss(model, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
    payload = dict(batch)
    payload["token_type_ids"] = torch.zeros_like(payload["input_ids"])
    return model(**payload)[0].float()


def set_center(
    params: Dict[str, torch.nn.Parameter],
    master: Dict[str, torch.Tensor],
    states: Dict[str, smoke.RTNClipState],
) -> None:
    with torch.no_grad():
        for name, param in params.items():
            if name not in master:
                continue
            value = master[name]
            if name in states:
                value = smoke.quantize_with_state(value, states[name])
            param.copy_(value.to(dtype=param.dtype))


def compute_center_grads(model, params, master, states, batch) -> tuple[float, Dict[str, torch.Tensor]]:
    set_center(params, master, states)
    model.zero_grad(set_to_none=True)
    loss = forward_loss(model, batch)
    loss.backward()
    grads = {}
    for name, param in params.items():
        if name in master and param.grad is not None:
            grads[name] = param.grad.detach().float().clone()
    return float(loss.detach().cpu()), grads


def finite_difference(model, params, master, states, directions, batch, h: float) -> tuple[float, float, float]:
    with torch.no_grad():
        smoke.copy_master_to_model(params, master, directions, h, +1.0, states)
        loss_plus = forward_loss(model, batch)
        smoke.copy_master_to_model(params, master, directions, h, -1.0, states)
        loss_minus = forward_loss(model, batch)
        smoke.restore_master(params, master)
    lp = float(loss_plus.detach().cpu())
    lm = float(loss_minus.detach().cpu())
    return lp, lm, (lp - lm) / (2.0 * h)


def true_directionals(
    master: Dict[str, torch.Tensor],
    directions: Dict[str, torch.Tensor],
    states: Dict[str, smoke.RTNClipState],
    grads: Dict[str, torch.Tensor],
    h: float,
) -> Dict[str, float]:
    device = next(iter(master.values())).device
    lowbit = torch.zeros((), device=device, dtype=torch.float64)
    u_target = torch.zeros_like(lowbit)
    lowbit_dir_sq = torch.zeros_like(lowbit)
    u_dir_sq = torch.zeros_like(lowbit)
    for name, grad in grads.items():
        if name not in directions or name not in master:
            continue
        direction = directions[name].float()
        if name in states:
            state = states[name]
            q_plus = smoke.quantize_with_state(master[name].float().add(direction, alpha=h), state)
            q_minus = smoke.quantize_with_state(master[name].float().add(direction, alpha=-h), state)
            effective_dir = (q_plus.float() - q_minus.float()) / (2.0 * h)
        else:
            effective_dir = direction
        gf = grad.float()
        lowbit += (gf.double() * effective_dir.double()).sum()
        u_target += (gf.double() * direction.double()).sum()
        lowbit_dir_sq += effective_dir.double().square().sum()
        u_dir_sq += direction.double().square().sum()
    return {
        "d_true_lowbit": float(lowbit.detach().cpu()),
        "d_true_u": float(u_target.detach().cpu()),
        "effective_direction_norm": float(lowbit_dir_sq.sqrt().detach().cpu()),
        "u_direction_norm": float(u_dir_sq.sqrt().detach().cpu()),
    }


def summarize(records: List[Dict[str, object]], h_grid: List[float]) -> List[Dict[str, object]]:
    rows = []
    for h in h_grid:
        group = [r for r in records if abs(float(r["h"]) - h) <= max(1e-15, h * 1e-12)]
        if not group:
            continue
        fd = [float(r["d_h_Q"]) for r in group]
        low = [float(r["d_true_lowbit"]) for r in group]
        u = [float(r["d_true_u"]) for r in group]
        half = [float(r["d_half_Q"]) for r in group]
        def nmse(target):
            err = sum((a - b) ** 2 for a, b in zip(fd, target))
            den = sum(b ** 2 for b in target)
            return err / max(den, EPS)
        rich_err = sum((a - b) ** 2 for a, b in zip(fd, half))
        rich_den = sum(b ** 2 for b in half)
        keys = [
            "active_frac", "alignment", "norm_ratio", "delta_visibility_nmse",
            "delta_visibility_rel_l2", "zero_effective_displacement_frac",
        ]
        row = {
            "h": h,
            "n_dirs": len(group),
            "fd_true_lowbit_nmse": nmse(low),
            "fd_true_lowbit_corr": corr(fd, low),
            "fd_true_u_nmse": nmse(u),
            "fd_true_u_corr": corr(fd, u),
            "richardson_rmse_rel": math.sqrt(rich_err / max(rich_den, EPS)),
            "d_h_mean": sum(fd) / len(fd),
            "d_true_lowbit_mean": sum(low) / len(low),
            "d_true_u_mean": sum(u) / len(u),
            "loss_plus_mean": sum(float(r["loss_plus"]) for r in group) / len(group),
            "loss_minus_mean": sum(float(r["loss_minus"]) for r in group) / len(group),
        }
        for key in keys:
            row[key] = sum(float(r[key]) for r in group) / len(group)
        rows.append(row)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--h_grid", nargs="+", type=float, default=[3e-4, 5e-4, 7e-4, 1e-3, 1.5e-3, 2e-3, 3e-3])
    parser.add_argument("--directions", type=int, default=16)
    parser.add_argument("--bitwidth", type=int, default=4)
    parser.add_argument("--group_size", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=16)
    parser.add_argument("--data_seed", type=int, default=16)
    parser.add_argument("--task_name", default="sst-5")
    parser.add_argument("--dataset_mode", default="full")
    parser.add_argument("--model_id", default="roberta-large")
    parser.add_argument("--data_dir", default=None)
    parser.add_argument("--num_k", type=int, default=16)
    parser.add_argument("--autograd_dtype", default="fp16", choices=["fp16", "fp32"])
    args = parser.parse_args()

    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    stats_path = out / "probe_records.jsonl"
    if stats_path.exists():
        stats_path.unlink()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if args.autograd_dtype == "fp16" else torch.float32
    env = {
        "timestamp": datetime.now().isoformat(),
        "hostname": socket.gethostname(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    write_json(out / "env.json", env)
    write_json(out / "run_config.json", vars(args))

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
    if args.autograd_dtype == "fp32":
        model.float()
    model.eval()
    params = smoke.named_parameter_map(model)
    q_names = smoke.linear_weight_names(model)
    step, master, best = load_master(Path(args.checkpoint), device, dtype)
    active_params = {name: param for name, param in params.items() if name in master}
    active_q_names = [name for name in q_names if name in master]
    numel_by_name = {name: active_params[name].numel() for name in active_q_names}
    states, refresh_rows = smoke.refresh_quantizer_states(master, active_q_names, args.bitwidth, args.group_size)
    quant = smoke.aggregate_quantizer_stats(refresh_rows, numel_by_name)
    write_json(out / "quantizer_summary.json", quant)

    batch = move_batch(next(iter(train_loader)), device)
    center_loss, grads = compute_center_grads(model, active_params, master, states, batch)

    records = []
    start = time.time()
    for k in range(args.directions):
        gen = torch.Generator(device=device)
        gen.manual_seed(args.seed + k)
        directions = smoke.sample_directions(master, gen)
        for h in args.h_grid:
            loss_plus, loss_minus, d_h = finite_difference(model, active_params, master, states, directions, batch, h)
            _, _, d_half = finite_difference(model, active_params, master, states, directions, batch, h / 2.0)
            true = true_directionals(master, directions, states, grads, h)
            pert = smoke.perturbation_metrics(master, directions, states, h)
            record = {
                "checkpoint_step": step,
                "direction_id": k,
                "direction_seed": args.seed + k,
                "h": h,
                "loss_center": center_loss,
                "loss_plus": loss_plus,
                "loss_minus": loss_minus,
                "d_h_Q": d_h,
                "d_half_Q": d_half,
                **true,
                **pert,
            }
            append_jsonl(stats_path, record)
            records.append(record)
        print(f"direction {k+1}/{args.directions} done", flush=True)

    rows = summarize(records, args.h_grid)
    fieldnames = [
        "h", "n_dirs", "fd_true_lowbit_nmse", "fd_true_lowbit_corr", "fd_true_u_nmse", "fd_true_u_corr",
        "richardson_rmse_rel", "active_frac", "alignment", "norm_ratio", "delta_visibility_nmse",
        "delta_visibility_rel_l2", "zero_effective_displacement_frac", "d_h_mean", "d_true_lowbit_mean",
        "d_true_u_mean", "loss_plus_mean", "loss_minus_mean",
    ]
    write_csv(out / "probe_summary.csv", rows, fieldnames)
    summary = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_step": step,
        "center_loss": center_loss,
        "directions": args.directions,
        "h_grid": args.h_grid,
        "runtime_sec": time.time() - start,
        "main_metric": "fd_true_lowbit_nmse",
        "main_true_target": "grad_Q_center^T[(Q_t(w+h*u)-Q_t(w-h*u))/(2h)]",
        "reference_metric": "fd_true_u_nmse",
        "sampler_name": type(train_sampler).__name__,
        "data_dir_resolved": getattr(data_args, "data_dir", ""),
        "best": best,
    }
    write_json(out / "run_summary.json", summary)

    lines = [
        "# INT4 Checkpoint FD True-nMSE Probe",
        "",
        f"- checkpoint: `{Path(args.checkpoint).resolve()}`",
        f"- checkpoint_step: {step}",
        f"- directions: {args.directions}",
        "- main target: `grad(Q_t(w))^T Delta_Q/(2h)`",
        "- reference target: `grad(Q_t(w))^T u`",
        "",
        "| h | fd_true_lowbit_nmse | fd_true_lowbit_corr | fd_true_u_nmse | richardson_rmse_rel | active_frac | alignment | norm_ratio | delta_visibility_nmse |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['h']:.6g} | {row['fd_true_lowbit_nmse']:.6g} | "
            f"{'NA' if row['fd_true_lowbit_corr'] is None else f'{row['fd_true_lowbit_corr']:.6g}'} | "
            f"{row['fd_true_u_nmse']:.6g} | {row['richardson_rmse_rel']:.6g} | "
            f"{row['active_frac']:.6g} | {row['alignment']:.6g} | {row['norm_ratio']:.6g} | {row['delta_visibility_nmse']:.6g} |"
        )
    (out / "probe_summary.md").write_text("\n".join(lines) + "\n")
    print(f"Wrote {out / 'probe_summary.csv'}")
    print(f"Wrote {out / 'probe_summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
