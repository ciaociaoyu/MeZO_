#!/usr/bin/env python
"""Probe INT4 RTNClip true-nMSE under Sparse-MeZO-style magnitude masks.

This is a probe-only diagnostic.  It uses Sparse-MeZO's parameter-selection
semantics (magnitude-ranked fraction per trainable tensor), not random sparse
directions and not 1/sqrt(p) rescaling.  The mask can target either the
lowest-|w| or highest-|w| coordinates for ablation.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = REPO_ROOT / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import quantizer_robustness_int8_window as qrw  # noqa: E402
from rtnclip_int8_mse_reprobe import (  # noqa: E402
    EPS,
    build_args,
    compute_true_gradient,
    directional_true_derivative,
    finite_difference_pair,
    finite_float,
    forward_loss_roberta,
    pooled_fd_true_stats,
    pooled_richardson_stats,
    quantized_pair_diagnostics,
)


DEFAULT_H_GRID: List[float] = [1e-2, 5e-3, 3e-3, 1e-3, 5e-4, 3e-4, 1e-4]
SUMMARY_COLUMNS = [
    "mask_strategy",
    "sparse_ratio",
    "mask_active_frac_all",
    "mask_active_frac_quantized_linear",
    "h",
    "n_directions",
    "nMSE_default_metric",
    "default_fd_true_nmse",
    "default_corr_fd_true",
    "default_true_direction",
    "fd_true_nmse",
    "corr_fd_true",
    "fd_true_mse",
    "fd_true_rmse",
    "fd_true_bias",
    "richardson_rmse_rel",
    "delta_visibility_nmse_mean",
    "alignment_mean",
    "norm_ratio_mean",
    "active_frac_mean",
    "code_change_frac_mean",
    "clip_frac_mean",
    "saturation_frac_mean",
    "d_h_mean",
    "d_true_mean",
]


def json_default(value):
    if isinstance(value, Path):
        return str(value)
    return str(value)


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True, default=json_default) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: List[Dict[str, object]], columns: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(columns), extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def append_jsonl(path: Path, row: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, sort_keys=True, default=json_default) + "\n")


def mean(values: Iterable[object]) -> Optional[float]:
    xs = [float(v) for v in values if finite_float(v) is not None]
    return sum(xs) / len(xs) if xs else None


def corr(xs: Sequence[object], ys: Sequence[object]) -> Optional[float]:
    pairs = []
    for x, y in zip(xs, ys):
        xf = finite_float(x)
        yf = finite_float(y)
        if xf is not None and yf is not None:
            pairs.append((xf, yf))
    if len(pairs) < 2:
        return None
    xv = [x for x, _ in pairs]
    yv = [y for _, y in pairs]
    mx = sum(xv) / len(xv)
    my = sum(yv) / len(yv)
    vx = sum((x - mx) ** 2 for x in xv)
    vy = sum((y - my) ** 2 for y in yv)
    if vx <= EPS or vy <= EPS:
        return None
    cov = sum((x - mx) * (y - my) for x, y in pairs)
    return cov / math.sqrt(vx * vy)


def parse_h_grid(raw: str) -> List[float]:
    if not raw.strip():
        return list(DEFAULT_H_GRID)
    values = []
    for part in raw.replace(",", " ").split():
        values.append(float(part))
    return values


def build_sparse_mezo_masks(
    master: Dict[str, torch.Tensor],
    *,
    sparse_ratio: float,
    quantized_names: Iterable[str],
    mask_strategy: str,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, object]]:
    ratio = float(sparse_ratio)
    if ratio <= 0.0 or ratio > 1.0 or not math.isfinite(ratio):
        raise ValueError(f"sparse_ratio must be in (0,1], got {sparse_ratio}")
    strategy = str(mask_strategy).strip().lower()
    if strategy not in {"lowest_abs", "highest_abs"}:
        raise ValueError(f"mask_strategy must be lowest_abs or highest_abs, got {mask_strategy!r}")
    quantized_set = set(quantized_names)
    masks: Dict[str, torch.Tensor] = {}
    active_all = 0
    total_all = 0
    active_quant = 0
    total_quant = 0
    for name, tensor in master.items():
        if not tensor.is_floating_point():
            continue
        total = int(tensor.numel())
        total_all += total
        if name in quantized_set:
            total_quant += total
        if ratio >= 1.0:
            mask = torch.ones_like(tensor, dtype=torch.bool)
        else:
            k = max(int(math.floor(ratio * total)), 1)
            flat_abs = tensor.detach().abs().reshape(-1)
            if k >= total:
                threshold = flat_abs.max() if strategy == "lowest_abs" else flat_abs.min()
            else:
                threshold = torch.kthvalue(flat_abs.float(), k).values.to(device=tensor.device, dtype=flat_abs.dtype)
                if strategy == "highest_abs":
                    kth_largest = total - k + 1
                    threshold = torch.kthvalue(flat_abs.float(), kth_largest).values.to(device=tensor.device, dtype=flat_abs.dtype)
            if strategy == "highest_abs":
                mask = tensor.detach().abs() >= threshold
            else:
                mask = tensor.detach().abs() <= threshold
        active = int(mask.sum().item())
        active_all += active
        if name in quantized_set:
            active_quant += active
        masks[name] = mask
    stats = {
        "mask_strategy": strategy,
        "sparse_selection": f"percentile_per_layer {strategy}_weight",
        "sparse_rescale": "none",
        "sparse_ratio": ratio,
        "active_params_all": active_all,
        "total_params_all": total_all,
        "active_frac_all": active_all / max(total_all, 1),
        "active_params_quantized_linear": active_quant,
        "total_params_quantized_linear": total_quant,
        "active_frac_quantized_linear": active_quant / max(total_quant, 1),
    }
    return masks, stats


def sample_masked_directions(
    master: Dict[str, torch.Tensor],
    masks: Dict[str, torch.Tensor],
    seed: int,
) -> Dict[str, torch.Tensor]:
    first = next(iter(master.values()))
    gen = torch.Generator(device=first.device).manual_seed(int(seed))
    directions: Dict[str, torch.Tensor] = {}
    for name, tensor in master.items():
        if not tensor.is_floating_point():
            continue
        z = torch.randn(tensor.shape, device=tensor.device, generator=gen, dtype=torch.float16)
        mask = masks.get(name)
        if mask is not None:
            z = z * mask.to(device=tensor.device, dtype=z.dtype)
        directions[name] = z
    return directions


def summarize(records: List[Dict[str, object]], h_grid: Sequence[float], mask_stats: Dict[Tuple[str, float], Dict[str, object]]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    keys = sorted({(str(r["mask_strategy"]), float(r["sparse_ratio"])) for r in records})
    for strategy, ratio in keys:
        for h in h_grid:
            group = [
                r for r in records
                if str(r["mask_strategy"]) == strategy
                and float(r["sparse_ratio"]) == ratio
                and abs(float(r["h"]) - float(h)) <= 1e-15
            ]
            if not group:
                continue
            fd = pooled_fd_true_stats([r.get("d_h_Q") for r in group], [r.get("d_true") for r in group])
            rich = pooled_richardson_stats([r.get("d_h_Q") for r in group], [r.get("d_half_Q") for r in group])
            stats = mask_stats[(strategy, ratio)]
            # Default nMSE for sparse/low-bit window selection is the original
            # ZO directional target: d_h versus g^T u. Effective-displacement
            # metrics are diagnostics and should be named separately.
            rows.append(
                {
                    "mask_strategy": strategy,
                    "sparse_ratio": ratio,
                    "mask_active_frac_all": stats.get("active_frac_all"),
                    "mask_active_frac_quantized_linear": stats.get("active_frac_quantized_linear"),
                    "h": h,
                    "n_directions": len(group),
                    "nMSE_default_metric": "default_dh_vs_gTu",
                    "default_fd_true_nmse": fd["fd_true_nmse"],
                    "default_corr_fd_true": fd["corr_fd_true"],
                    "default_true_direction": "gTu",
                    "fd_true_nmse": fd["fd_true_nmse"],
                    "corr_fd_true": fd["corr_fd_true"],
                    "fd_true_mse": fd["fd_true_mse"],
                    "fd_true_rmse": fd["fd_true_rmse"],
                    "fd_true_bias": fd["fd_true_bias"],
                    "richardson_rmse_rel": rich["richardson_rmse_rel"],
                    "delta_visibility_nmse_mean": mean(r.get("delta_visibility_nmse") for r in group),
                    "alignment_mean": mean(r.get("alignment") for r in group),
                    "norm_ratio_mean": mean(r.get("norm_ratio") for r in group),
                    "active_frac_mean": mean(r.get("active_frac") for r in group),
                    "code_change_frac_mean": mean(r.get("code_change_frac") for r in group),
                    "clip_frac_mean": mean(r.get("clip_frac") for r in group),
                    "saturation_frac_mean": mean(r.get("saturation_frac") for r in group),
                    "d_h_mean": mean(r.get("d_h_Q") for r in group),
                    "d_true_mean": mean(r.get("d_true") for r in group),
                }
            )
    return rows


def write_report(output_dir: Path, summary_rows: List[Dict[str, object]], config: Dict[str, object]) -> None:
    lines = [
        "# INT4 Sparse-MeZO Mask True-nMSE Probe",
        "",
        "This is a probe-only test. Sparse selection uses per-trainable-tensor percentile masks. No `1/sqrt(p)` direction rescaling is applied.",
        "",
        "Default nMSE is `default_dh_vs_gTu`: finite-difference `d_h` compared with the original directional derivative `g^T u`.",
        "",
        f"- quantizer: `{config['quantizer']}`",
        f"- h grid: `{config['h_grid']}`",
        f"- directions per h: `{config['directions']}`",
        "",
        "| mask_strategy | sparse_ratio | h | default_fd_true_nmse | default_corr_fd_true | alignment | norm_ratio | active_frac | richardson_rmse_rel |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in summary_rows:
        def fmt(x):
            xf = finite_float(x)
            return "NA" if xf is None else f"{xf:.6g}"

        lines.append(
            f"| {row.get('mask_strategy')} | {fmt(row['sparse_ratio'])} | {fmt(row['h'])} | {fmt(row['default_fd_true_nmse'])} | "
            f"{fmt(row['default_corr_fd_true'])} | {fmt(row['alignment_mean'])} | {fmt(row['norm_ratio_mean'])} | "
            f"{fmt(row['active_frac_mean'])} | {fmt(row['richardson_rmse_rel'])} |"
        )
    output_dir.joinpath("summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default=str(REPO_ROOT / "outputs" / "rtnclip_int4_sparse_mezo_nmse_probe"))
    parser.add_argument("--sparse_ratios", nargs="+", type=float, default=[0.1, 1.0])
    parser.add_argument("--mask_strategies", nargs="+", choices=["lowest_abs", "highest_abs"], default=["lowest_abs"])
    parser.add_argument("--h_grid", default="")
    parser.add_argument("--directions", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--probe_batches", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ["DATALOADER_SHUFFLE"] = "true"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records_path = output_dir / "probe_records.jsonl"
    if records_path.exists():
        records_path.unlink()

    h_grid = parse_h_grid(args.h_grid)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    harness = qrw.RobertaHarness(build_args(int(args.batch_size), int(args.eval_batch_size), int(args.directions)), device)
    if harness.train_sampler_name != "RandomSampler":
        raise RuntimeError(f"Expected RandomSampler, got {harness.train_sampler_name}")
    params = harness.params()
    master = harness.make_master()
    qrw.restore_master(params, master)
    states, refresh_rows = qrw.refresh_quantizer_states(
        master,
        harness.quantized_module_names,
        quantizer="rtnclip",
        activation_rms={},
        bitwidth=4,
        group_size=128,
    )

    config = {
        "model": "roberta-large",
        "dataset": "SST-5",
        "dataset_mode": "full",
        "seed": 16,
        "data_seed": 16,
        "batch_size": int(args.batch_size),
        "sampler": harness.train_sampler_name,
        "quant_bits": 4,
        "quantizer": "INT4_G128_RTNClip_shared_grid_fake_quant",
        "sparse_selection": "percentile_per_layer weight magnitude",
        "mask_strategies": list(args.mask_strategies),
        "sparse_rescale": "none",
        "nMSE_default_metric": "default_dh_vs_gTu",
        "default_true_direction": "gTu",
        "sparse_ratios": list(args.sparse_ratios),
        "h_grid": h_grid,
        "directions": int(args.directions),
        "probe_batches": int(args.probe_batches),
    }
    write_json(output_dir / "run_config.json", config)
    write_json(output_dir / "env.json", qrw.collect_env())
    write_json(output_dir / "quantizer_refresh_summary.json", qrw.aggregate_quantizer_stats(refresh_rows, harness.numel_by_quantized_name()))

    records: List[Dict[str, object]] = []
    mask_stats_by_ratio: Dict[Tuple[str, float], Dict[str, object]] = {}
    data_iter = iter(harness.train_loader)
    start = time.time()
    for batch_id in range(int(args.probe_batches)):
        try:
            batch = qrw.move_batch(next(data_iter), device)
        except StopIteration:
            break
        compute_true_gradient(harness, master, batch)
        for strategy in args.mask_strategies:
            for ratio in args.sparse_ratios:
                masks, mask_stats = build_sparse_mezo_masks(
                    master,
                    sparse_ratio=float(ratio),
                    quantized_names=harness.quantized_module_names,
                    mask_strategy=strategy,
                )
                mask_stats_by_ratio[(str(strategy), float(ratio))] = mask_stats
                for h in h_grid:
                    for direction_id in range(int(args.directions)):
                        # Keep the underlying Gaussian direction fixed across h and
                        # mask strategies; only the selected coordinates differ.
                        seed = qrw.direction_seed(16, f"rtnclip_int4_sparse_mezo_p{ratio:g}", 0.0, batch_id, extra=direction_id)
                        directions = sample_masked_directions(master, masks, seed)
                        d_true = directional_true_derivative(params, directions)
                        loss_plus, loss_minus, d_h = finite_difference_pair(harness, master, states, directions, batch, float(h))
                        _, _, d_half = finite_difference_pair(harness, master, states, directions, batch, float(h) / 2.0)
                        diag = quantized_pair_diagnostics(master, directions, states, float(h))
                        record = {
                            "mask_strategy": str(strategy),
                            "sparse_ratio": float(ratio),
                            "mask_active_frac_all": mask_stats["active_frac_all"],
                            "mask_active_frac_quantized_linear": mask_stats["active_frac_quantized_linear"],
                            "h": float(h),
                            "batch_id": int(batch_id),
                            "direction_id": int(direction_id),
                            "direction_seed": int(seed),
                            "loss_plus": loss_plus,
                            "loss_minus": loss_minus,
                            "d_h_Q": d_h,
                            "d_half_Q": d_half,
                            "d_true": d_true,
                            "fd_true_error": None if d_true is None else d_h - d_true,
                            **diag,
                        }
                        append_jsonl(records_path, record)
                        records.append(record)
        harness.model.zero_grad(set_to_none=True)
        qrw.restore_master(params, master)

    summary_rows = summarize(records, h_grid, mask_stats_by_ratio)
    write_csv(output_dir / "summary.csv", summary_rows, SUMMARY_COLUMNS)
    write_json(output_dir / "mask_stats.json", {f"{k[0]}:p={k[1]:g}": v for k, v in mask_stats_by_ratio.items()})
    write_report(output_dir, summary_rows, config)
    write_json(
        output_dir / "run_summary.json",
        {
            **config,
            "status": "complete",
            "records": len(records),
            "summary_rows": len(summary_rows),
            "runtime_seconds": time.time() - start,
            "peak_gpu_mem_mb": float(torch.cuda.max_memory_allocated() / 1024 / 1024) if torch.cuda.is_available() else 0.0,
        },
    )
    print(f"Output: {output_dir}")
    for row in summary_rows:
        print(
            f"p={row['sparse_ratio']:.3g} h={row['h']:.3g} "
            f"fd_true_nmse={row['fd_true_nmse']:.6g} corr={row['corr_fd_true'] if row['corr_fd_true'] is not None else 'NA'} "
            f"align={row['alignment_mean']:.6g} norm_ratio={row['norm_ratio_mean']:.6g} active={row['active_frac_mean']:.6g}"
        )


if __name__ == "__main__":
    main()
