#!/usr/bin/env python
"""Task-gradient sparse INT4 RTNClip fd-vs-true probe for RoBERTa tasks."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = REPO_ROOT / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import smoke_rtnclip_roberta_sst5 as smoke  # noqa: E402
import rtnclip_roberta_sst5_batch as batch_runner  # noqa: E402


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def append_jsonl(path: Path, row: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, sort_keys=True, default=str) + "\n")


def write_csv(path: Path, rows: List[Dict[str, object]], columns: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(columns), extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def parse_h_grid(raw: str) -> List[float]:
    if not raw.strip():
        return [2e-4, 3e-4, 5e-4, 7e-4, 1e-3, 1.5e-3, 2e-3, 3e-3, 5e-3, 1e-2]
    return [float(x) for x in raw.replace(",", " ").split()]


def finite_float(value) -> Optional[float]:
    try:
        x = float(value)
    except Exception:
        return None
    return x if math.isfinite(x) else None


def pooled_stats(dh: Iterable[object], dt: Iterable[object]) -> Dict[str, Optional[float]]:
    pairs = []
    for a, b in zip(dh, dt):
        af = finite_float(a)
        bf = finite_float(b)
        if af is not None and bf is not None:
            pairs.append((af, bf))
    if not pairs:
        return {"fd_true_mse": None, "fd_true_nmse": None, "fd_true_rmse": None, "corr_fd_true": None}
    err2 = sum((a - b) ** 2 for a, b in pairs)
    ref2 = sum(b * b for _, b in pairs)
    mse = err2 / len(pairs)
    corr = None
    if len(pairs) >= 2:
        xs = [a for a, _ in pairs]
        ys = [b for _, b in pairs]
        mx = sum(xs) / len(xs)
        my = sum(ys) / len(ys)
        vx = sum((x - mx) ** 2 for x in xs)
        vy = sum((y - my) ** 2 for y in ys)
        if vx > 1e-30 and vy > 1e-30:
            corr = sum((x - mx) * (y - my) for x, y in pairs) / math.sqrt(vx * vy)
    return {
        "fd_true_mse": mse,
        "fd_true_nmse": err2 / max(ref2, 1e-30),
        "fd_true_rmse": math.sqrt(mse),
        "corr_fd_true": corr,
    }


def mean(rows: Iterable[Dict[str, object]], key: str) -> Optional[float]:
    vals = [finite_float(r.get(key)) for r in rows]
    xs = [x for x in vals if x is not None]
    return sum(xs) / len(xs) if xs else None


def compute_true_gradient(model, params, master, batch) -> None:
    smoke.restore_master(params, master)
    model.zero_grad(set_to_none=True)
    for p in params.values():
        p.requires_grad_(p.is_floating_point())
    loss = batch_runner.forward_loss_for_grad(model, batch)
    loss.backward()


def load_checkpoint_master(checkpoint: Path, device: torch.device) -> tuple[int, Dict[str, torch.Tensor], Dict[str, object]]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    master = {
        name: tensor.to(device=device, dtype=torch.float16)
        for name, tensor in payload["master"].items()
        if torch.is_tensor(tensor) and tensor.is_floating_point()
    }
    return int(payload.get("step", -1)), master, payload.get("config", {})


def directional_true(params, directions: Dict[str, torch.Tensor]) -> float:
    total = torch.zeros((), device=next(iter(directions.values())).device, dtype=torch.float64)
    for name, direction in directions.items():
        grad = params[name].grad
        if grad is not None:
            total += (grad.detach().double() * direction.detach().double()).sum()
    return float(total.detach().cpu())


def finite_difference(model, params, master, states, directions, batch, h: float) -> Dict[str, float]:
    smoke.copy_master_to_model(params, master, directions, h, +1.0, states)
    loss_plus, _ = smoke.forward_loss_and_logits(model, batch)
    smoke.copy_master_to_model(params, master, directions, h, -1.0, states)
    loss_minus, _ = smoke.forward_loss_and_logits(model, batch)
    smoke.restore_master(params, master)
    lp = float(loss_plus.detach().cpu())
    lm = float(loss_minus.detach().cpu())
    return {"loss_plus": lp, "loss_minus": lm, "d_h": (lp - lm) / (2.0 * h)}


def sample_fixed_masked_directions(master: Dict[str, torch.Tensor], masks: Dict[str, torch.Tensor], seed: int) -> Dict[str, torch.Tensor]:
    first = next(iter(master.values()))
    gen = torch.Generator(device=first.device)
    gen.manual_seed(int(seed))
    directions: Dict[str, torch.Tensor] = {}
    for name, tensor in master.items():
        if not tensor.is_floating_point():
            continue
        direction = torch.randn(tensor.shape, device=tensor.device, generator=gen, dtype=torch.float16)
        mask = masks.get(name)
        if mask is not None:
            direction = direction * mask.to(device=direction.device, dtype=direction.dtype)
        directions[name] = direction
    return directions


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--task_name", default="sst-2")
    parser.add_argument("--dataset_mode", default="full")
    parser.add_argument("--num_k", type=int, default=16)
    parser.add_argument("--seed", type=int, default=16)
    parser.add_argument("--data_seed", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--sparse_ratio", type=float, default=0.1)
    parser.add_argument("--sparse_mask_batches", type=int, default=1)
    parser.add_argument("--h_grid", default="")
    parser.add_argument("--directions", type=int, default=8)
    parser.add_argument("--checkpoint", default="", help="Optional checkpoint state.pt to probe after training.")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if os.environ.get("DATALOADER_SHUFFLE") != "True":
        raise RuntimeError("DATALOADER_SHUFFLE=True must be exported")

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    records_path = out / "probe_records.jsonl"
    if records_path.exists():
        records_path.unlink()

    device = torch.device("cuda")
    orig_torch_load = torch.load

    def _compat_torch_load(*load_args, **load_kwargs):
        load_kwargs.setdefault("weights_only", False)
        return orig_torch_load(*load_args, **load_kwargs)

    torch.load = _compat_torch_load
    try:
        model, train_loader, _, data_args, train_sampler = smoke.load_prompt_model_and_data(
            argparse.Namespace(
                repo_root=REPO_ROOT,
                model_id="roberta-large",
                task_name=args.task_name,
                seed=args.seed,
                data_seed=args.data_seed,
                batch_size=args.batch_size,
                eval_batch_size=args.eval_batch_size,
                dataset_mode=args.dataset_mode,
                data_dir=None,
                num_k=args.num_k,
            ),
            device,
        )
    finally:
        torch.load = orig_torch_load
    if type(train_sampler).__name__ != "RandomSampler":
        raise RuntimeError(f"Expected RandomSampler, got {type(train_sampler).__name__}")

    params = smoke.named_parameter_map(model)
    q_names = [name for name in smoke.linear_weight_names(model) if name in params]
    initial_master = {name: p.detach().clone().to(device=device, dtype=torch.float16) for name, p in params.items() if p.detach().is_floating_point()}
    masks, mask_stats = batch_runner.build_task_grad_sparse_masks(
        model,
        params,
        initial_master,
        train_loader,
        device,
        sparse_ratio=args.sparse_ratio,
        quantized_names=q_names,
        mask_batches=args.sparse_mask_batches,
        mask_scope="linear_weight",
    )
    checkpoint_step = -1
    checkpoint_config: Dict[str, object] = {}
    if args.checkpoint:
        checkpoint_step, master, checkpoint_config = load_checkpoint_master(Path(args.checkpoint), device)
    else:
        master = initial_master
    q_names = [name for name in q_names if name in master]
    states, refresh_rows = smoke.refresh_quantizer_states(master, q_names, 4, 128)
    batch = smoke.move_batch(next(iter(train_loader)), device)
    compute_true_gradient(model, params, master, batch)

    h_grid = parse_h_grid(args.h_grid)
    config = {
        "model": "roberta-large",
        "dataset": smoke.normalize_task_name(args.task_name),
        "dataset_mode": args.dataset_mode,
        "data_dir_resolved": getattr(data_args, "data_dir", ""),
        "seed": args.seed,
        "data_seed": args.data_seed,
        "batch_size": args.batch_size,
        "sampler": type(train_sampler).__name__,
        "quant_bits": 4,
        "quantizer": "G128_groupwise_RTNClip_fake_quant",
        "direction_mode": "sparse",
        "sparse_ratio": args.sparse_ratio,
        "sparse_mask_strategy": "task_grad_static",
        "sparse_selection": mask_stats.get("sparse_selection"),
        "sparse_rescale": "none",
        "default_mse_metric": "d_h_minus_gTu",
        "direction_seed_policy": "h_independent_fixed_per_direction_id",
        "h_grid": h_grid,
        "directions": args.directions,
        "checkpoint": str(Path(args.checkpoint).resolve()) if args.checkpoint else "",
        "checkpoint_step": checkpoint_step,
        "checkpoint_run_name": checkpoint_config.get("run_name", ""),
        **mask_stats,
    }
    write_json(out / "run_config.json", config)
    write_json(out / "env.json", smoke.collect_env(REPO_ROOT))
    write_json(out / "quantizer_refresh_summary.json", smoke.aggregate_quantizer_stats(refresh_rows, {name: params[name].numel() for name in q_names}))
    write_json(out / "mask_stats.json", mask_stats)

    records: List[Dict[str, object]] = []
    for h in h_grid:
        for k in range(args.directions):
            directions = sample_fixed_masked_directions(master, masks, args.seed * 1_000_003 + k)
            d_true = directional_true(params, directions)
            fd = finite_difference(model, params, master, states, directions, batch, float(h))
            diag = smoke.perturbation_metrics(master, directions, states, float(h))
            row = {
                "h": float(h),
                "direction_id": k,
                "d_true": d_true,
                "d_h": fd["d_h"],
                "loss_plus": fd["loss_plus"],
                "loss_minus": fd["loss_minus"],
                "fd_true_error": fd["d_h"] - d_true,
                **diag,
            }
            append_jsonl(records_path, row)
            records.append(row)

    summary_rows: List[Dict[str, object]] = []
    for h in h_grid:
        group = [r for r in records if abs(float(r["h"]) - float(h)) <= 1e-15]
        stats = pooled_stats([r["d_h"] for r in group], [r["d_true"] for r in group])
        summary_rows.append(
            {
                "h": float(h),
                "directions": len(group),
                "default_mse_metric": "d_h_minus_gTu",
                "default_fd_true_nmse": stats["fd_true_nmse"],
                "default_corr_fd_true": stats["corr_fd_true"],
                "fd_true_mse": stats["fd_true_mse"],
                "fd_true_rmse": stats["fd_true_rmse"],
                "alignment_mean": mean(group, "alignment"),
                "norm_ratio_mean": mean(group, "norm_ratio"),
                "active_frac_mean": mean(group, "active_frac"),
                "delta_visibility_nmse_mean": mean(group, "delta_visibility_nmse"),
                "d_h_mean": mean(group, "d_h"),
                "d_true_mean": mean(group, "d_true"),
            }
        )
    columns = list(summary_rows[0].keys()) if summary_rows else []
    write_csv(out / "summary.csv", summary_rows, columns)
    lines = [
        "# INT4 Sparse Task-Gradient Probe",
        "",
        "Default MSE/NMSE compares quantized finite difference `d_h` with true directional derivative `g^T u`.",
        "",
        "| h | default_fd_true_nmse | corr | alignment | norm_ratio | active_frac |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in summary_rows:
        def fmt(x):
            xf = finite_float(x)
            return "NA" if xf is None else f"{xf:.6g}"
        lines.append(
            f"| {fmt(r['h'])} | {fmt(r['default_fd_true_nmse'])} | {fmt(r['default_corr_fd_true'])} | "
            f"{fmt(r['alignment_mean'])} | {fmt(r['norm_ratio_mean'])} | {fmt(r['active_frac_mean'])} |"
        )
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(out / "run_summary.json", {"status": "complete", "summary_rows": len(summary_rows), **config})
    print(f"Output: {out}")
    for r in summary_rows:
        print(
            f"h={r['h']:.6g} default_fd_true_nmse={r['default_fd_true_nmse']:.6g} "
            f"corr={r['default_corr_fd_true'] if r['default_corr_fd_true'] is not None else 'NA'} "
            f"align={r['alignment_mean']:.6g} norm_ratio={r['norm_ratio_mean']:.6g} active={r['active_frac_mean']:.6g}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
