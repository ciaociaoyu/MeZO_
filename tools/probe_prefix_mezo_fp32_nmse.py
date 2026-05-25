#!/usr/bin/env python
"""Probe original all-FP32 Prefix-MeZO finite-difference nMSE."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import torch
from torch import nn


REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = REPO_ROOT / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import smoke_rtnclip_roberta_sst5 as smoke  # noqa: E402
from rtnclip_int8_mse_reprobe import (  # noqa: E402
    finite_float,
    pooled_fd_true_stats,
    pooled_richardson_stats,
)


DEFAULT_H_GRID = [
    1.0,
    3e-1,
    1e-1,
    3e-2,
    1e-2,
    3e-3,
    1e-3,
    3e-4,
    1e-4,
    3e-5,
    1e-5,
    3e-6,
    1e-6,
    3e-7,
    1e-7,
    3e-8,
    1e-8,
]


SUMMARY_COLUMNS = [
    "task_name",
    "h",
    "n_directions",
    "fd_true_nmse",
    "corr_fd_true",
    "fd_true_mse",
    "fd_true_rmse",
    "fd_true_bias",
    "richardson_rmse_rel",
    "d_h_mean",
    "d_true_mean",
    "loss_plus_mean",
    "loss_minus_mean",
    "finite_rate",
]


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: List[Dict[str, object]], columns: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def parse_h_grid(raw: str) -> List[float]:
    return [float(x) for x in str(raw).replace(",", " ").split() if x.strip()]


def add_medium_path() -> None:
    path = REPO_ROOT / "medium_models"
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def load_model_and_batch(args: argparse.Namespace, device: torch.device):
    add_medium_path()
    load_args = argparse.Namespace(
        repo_root=REPO_ROOT,
        model_id="roberta-large",
        task_name=args.task_name,
        dataset_mode="full",
        data_seed=int(args.data_seed),
        num_k=16,
        data_dir=None,
        seed=int(args.seed),
        batch_size=int(args.batch_size),
        eval_batch_size=int(args.eval_batch_size),
    )
    orig_torch_load = torch.load

    def _compat_torch_load(*load_args_, **load_kwargs_):
        load_kwargs_.setdefault("weights_only", False)
        return orig_torch_load(*load_args_, **load_kwargs_)

    torch.load = _compat_torch_load
    try:
        model, train_loader, _dev_loader, data_args, sampler = smoke.load_prompt_model_and_data(load_args, device)
    finally:
        torch.load = orig_torch_load
    model.float()
    from src.prefix import PrefixTuning  # noqa: E402

    PrefixTuning(
        model,
        num_prefix=int(args.num_prefix),
        reparam=False,
        float16=False,
        init_by_real_act=bool(args.prefix_init_by_real_act),
    )
    model.float()
    model.eval()
    prefix_names = [name for name, _ in model.named_parameters() if "prefix" in name]
    if not prefix_names:
        raise RuntimeError("No prefix parameters were created.")
    batch = smoke.move_batch(next(iter(train_loader)), device)
    return model, batch, data_args, sampler, prefix_names


def forward_loss(model: nn.Module, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
    payload = dict(batch)
    payload["token_type_ids"] = torch.zeros_like(payload["input_ids"])
    outputs = model(**payload)
    return outputs[0]


def restore_master(params: Dict[str, nn.Parameter], master: Dict[str, torch.Tensor]) -> None:
    with torch.no_grad():
        for name, param in params.items():
            if name in master:
                param.copy_(master[name].to(dtype=param.dtype))


def copy_perturbed(
    params: Dict[str, nn.Parameter],
    master: Dict[str, torch.Tensor],
    directions: Optional[Dict[str, torch.Tensor]],
    h: float,
    sign: float,
) -> None:
    with torch.no_grad():
        for name, param in params.items():
            if name not in master:
                continue
            value = master[name]
            if directions is not None and name in directions:
                value = value.float().add(directions[name].float(), alpha=sign * float(h))
            param.copy_(value.to(dtype=param.dtype))


def sample_prefix_direction(
    master: Dict[str, torch.Tensor],
    prefix_names: Iterable[str],
    seed: int,
) -> Dict[str, torch.Tensor]:
    names = set(prefix_names)
    first = next(iter(master.values()))
    gen = torch.Generator(device=first.device).manual_seed(int(seed))
    return {
        name: torch.randn(t.shape, generator=gen, device=t.device, dtype=torch.float32)
        for name, t in master.items()
        if name in names
    }


def direction_norm(directions: Dict[str, torch.Tensor]) -> float:
    if not directions:
        return 0.0
    total = torch.zeros((), device=next(iter(directions.values())).device, dtype=torch.float64)
    for direction in directions.values():
        total += direction.double().square().sum()
    return float(torch.sqrt(total).detach().cpu())


def compute_true_grad(
    model: nn.Module,
    params: Dict[str, nn.Parameter],
    master: Dict[str, torch.Tensor],
    directions: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
) -> float:
    restore_master(params, master)
    model.zero_grad(set_to_none=True)
    loss = forward_loss(model, batch)
    loss.backward()
    total = torch.zeros((), device=next(iter(master.values())).device, dtype=torch.float64)
    for name, direction in directions.items():
        grad = params[name].grad
        if grad is not None:
            total += (grad.detach().double() * direction.detach().double()).sum()
    for param in params.values():
        param.grad = None
    restore_master(params, master)
    return float(total.detach().cpu())


def finite_difference(
    model: nn.Module,
    params: Dict[str, nn.Parameter],
    master: Dict[str, torch.Tensor],
    directions: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    h: float,
) -> tuple[float, float, float]:
    with torch.no_grad():
        copy_perturbed(params, master, directions, h, +1.0)
        loss_plus = forward_loss(model, batch)
        copy_perturbed(params, master, directions, h, -1.0)
        loss_minus = forward_loss(model, batch)
        restore_master(params, master)
    lp = float(loss_plus.detach().cpu())
    lm = float(loss_minus.detach().cpu())
    return lp, lm, (lp - lm) / (2.0 * float(h))


def summarize(records: List[Dict[str, object]], task_name: str) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    h_values = sorted({float(r["h"]) for r in records}, reverse=True)
    for h in h_values:
        group = [r for r in records if abs(float(r["h"]) - h) <= max(1e-30, abs(h) * 1e-12)]
        fd_stats = pooled_fd_true_stats([r.get("d_h") for r in group], [r.get("d_true") for r in group])
        rich = pooled_richardson_stats([r.get("d_h") for r in group], [r.get("d_half") for r in group])
        finite = [
            r for r in group
            if finite_float(r.get("d_h")) is not None and finite_float(r.get("d_true")) is not None
        ]
        loss_plus_vals = [float(r["loss_plus"]) for r in group if finite_float(r.get("loss_plus")) is not None]
        loss_minus_vals = [float(r["loss_minus"]) for r in group if finite_float(r.get("loss_minus")) is not None]
        rows.append(
            {
                "task_name": task_name,
                "h": h,
                "n_directions": len(group),
                **fd_stats,
                **rich,
                "d_h_mean": sum(float(r["d_h"]) for r in finite) / len(finite) if finite else None,
                "d_true_mean": sum(float(r["d_true"]) for r in finite) / len(finite) if finite else None,
                "loss_plus_mean": sum(loss_plus_vals) / len(loss_plus_vals) if loss_plus_vals else None,
                "loss_minus_mean": sum(loss_minus_vals) / len(loss_minus_vals) if loss_minus_vals else None,
                "finite_rate": len(finite) / len(group) if group else 0.0,
            }
        )
    return rows


def write_report(path: Path, rows: List[Dict[str, object]], config: Dict[str, object]) -> None:
    def fmt(value) -> str:
        v = finite_float(value)
        return "NA" if v is None else f"{v:.6g}"

    lines = [
        "# Prefix-MeZO FP32 h-nMSE Probe",
        "",
        "- path: original medium_models Prefix-MeZO, all FP32",
        f"- task: `{config['task_name']}`",
        f"- h grid: `{config['h_grid']}`",
        f"- directions: `{config['directions']}`",
        "",
        "| h | fd_true_nmse | corr_fd_true | richardson_rmse_rel | finite_rate | d_h_mean | d_true_mean |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {float(row['h']):.6g} | {fmt(row.get('fd_true_nmse'))} | {fmt(row.get('corr_fd_true'))} | "
            f"{fmt(row.get('richardson_rmse_rel'))} | {fmt(row.get('finite_rate'))} | "
            f"{fmt(row.get('d_h_mean'))} | {fmt(row.get('d_true_mean'))} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--task_name", default="sst-5")
    parser.add_argument("--h_grid", default=" ".join(str(h) for h in DEFAULT_H_GRID))
    parser.add_argument("--directions", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=16)
    parser.add_argument("--data_seed", type=int, default=16)
    parser.add_argument("--num_prefix", type=int, default=5)
    parser.add_argument("--prefix_init_by_real_act", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for RoBERTa-large prefix probe.")
    os.environ.setdefault("DATALOADER_SHUFFLE", "True")
    device = torch.device("cuda")
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    h_grid = parse_h_grid(args.h_grid)
    config = {
        "task_name": args.task_name,
        "dataset_mode": "full",
        "model": "roberta-large",
        "seed": args.seed,
        "data_seed": args.data_seed,
        "batch_size": args.batch_size,
        "directions": args.directions,
        "h_grid": h_grid,
        "path": "original_medium_models_prefix_mezo_all_fp32",
        "precision_mode": "fp32",
        "zo_quantization_bits": 32,
        "zo_two_point_precision": "fp32",
        "num_prefix": args.num_prefix,
        "prefix_init_by_real_act": bool(args.prefix_init_by_real_act),
    }
    write_json(output_dir / "run_config.json", config)

    start = time.time()
    torch.manual_seed(int(args.seed))
    torch.cuda.manual_seed_all(int(args.seed))
    model, batch, data_args, sampler, prefix_names = load_model_and_batch(args, device)
    params = dict(model.named_parameters())
    master = {
        name: p.detach().clone().float()
        for name, p in params.items()
        if p.detach().is_floating_point()
    }
    config.update(
        {
            "sampler_name": type(sampler).__name__,
            "resolved_data_dir": getattr(data_args, "data_dir", ""),
            "prefix_param_count": sum(int(master[name].numel()) for name in prefix_names if name in master),
            "prefix_tensor_count": len(prefix_names),
            "prefix_param_names": prefix_names,
        }
    )
    write_json(output_dir / "run_config.json", config)

    records: List[Dict[str, object]] = []
    for direction_id in range(int(args.directions)):
        seed = int(args.seed) * 1_000_003 + 77_777 + direction_id
        directions = sample_prefix_direction(master, prefix_names, seed)
        d_true = compute_true_grad(model, params, master, directions, batch)
        dnorm = direction_norm(directions)
        for h in h_grid:
            lp, lm, d_h = finite_difference(model, params, master, directions, batch, float(h))
            _, _, d_half = finite_difference(model, params, master, directions, batch, float(h) / 2.0)
            records.append(
                {
                    "task_name": args.task_name,
                    "h": float(h),
                    "direction_id": direction_id,
                    "direction_seed": seed,
                    "loss_plus": lp,
                    "loss_minus": lm,
                    "d_h": d_h,
                    "d_half": d_half,
                    "d_true": d_true,
                    "direction_norm": dnorm,
                }
            )
        print(f"direction {direction_id + 1}/{args.directions} done", flush=True)

    rows = summarize(records, args.task_name)
    write_csv(output_dir / "probe_records.csv", records, sorted({k for r in records for k in r.keys()}))
    with (output_dir / "probe_records.jsonl").open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, sort_keys=True, default=str) + "\n")
    write_csv(output_dir / "summary.csv", rows, SUMMARY_COLUMNS)
    write_report(output_dir / "summary.md", rows, config)
    write_json(output_dir / "run_summary.json", {"elapsed_sec": time.time() - start, "summary": rows, "config": config})
    print(f"Output: {output_dir}")
    for row in rows:
        print(
            f"h={float(row['h']):.6g} nmse={finite_float(row.get('fd_true_nmse'))} "
            f"corr={finite_float(row.get('corr_fd_true'))}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
