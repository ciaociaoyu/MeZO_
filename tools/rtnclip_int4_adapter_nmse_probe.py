#!/usr/bin/env python
"""Probe INT4 RTNClip finite-difference nMSE for adapter-only directions.

This is a probe-only diagnostic for RoBERTa-large / SST-5.  The base model
Linear weights are evaluated through the existing INT4 G128 RTNClip shared-grid
fake-quantized forward path.  The perturbed coordinates are adapter parameters:
LoRA query/value parameters or prefix-tuning parameters.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
import types
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import torch
from torch import nn
from torch.nn import functional as F


REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = REPO_ROOT / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import quantizer_robustness_int8_window as qrw  # noqa: E402
from rtnclip_int8_mse_reprobe import (  # noqa: E402
    directional_true_derivative,
    finite_difference_pair,
    finite_float,
    forward_loss_roberta,
    pooled_fd_true_stats,
    pooled_richardson_stats,
)


DEFAULT_H_GRID = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0]
SUMMARY_COLUMNS = [
    "adapter",
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
        writer = csv.DictWriter(f, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def parse_h_grid(raw: str) -> List[float]:
    return [float(x) for x in str(raw).replace(",", " ").split() if x.strip()]


def make_harness(args: argparse.Namespace, device: torch.device) -> qrw.RobertaHarness:
    hargs = argparse.Namespace(
        roberta_batch_size=int(args.batch_size),
        roberta_eval_batch_size=int(args.eval_batch_size),
        roberta_lr=1e-5,
        roberta_task_name=str(args.task_name),
        task_name=str(args.task_name),
        dataset_mode="full",
        num_k=16,
        data_seed=int(args.data_seed),
    )
    return qrw.RobertaHarness(hargs, device)


def _lora_forward(self: nn.Linear, x: torch.Tensor) -> torch.Tensor:
    result = F.linear(x, self.weight, self.bias)
    lora_x = self.lora_dropout(x).float()
    lora_a = self.lora_A.float()
    lora_b = self.lora_B.float()
    delta = (lora_x @ lora_a.transpose(0, 1) @ lora_b.transpose(0, 1)) * float(self.lora_scaling)
    return result + torch.nan_to_num(delta).to(dtype=result.dtype)


def inject_inline_lora(model: nn.Module, rank: int, alpha: int) -> List[str]:
    """Attach a minimal LoRA path to RoBERTa attention query/value Linear layers."""
    targets: List[str] = []
    dtype = next(model.parameters()).dtype
    device = next(model.parameters()).device
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if not (name.endswith(".attention.self.query") or name.endswith(".attention.self.value")):
            continue
        module.register_parameter(
            "lora_A",
            nn.Parameter(torch.empty(int(rank), module.in_features, device=device, dtype=dtype)),
        )
        module.register_parameter(
            "lora_B",
            nn.Parameter(torch.zeros(module.out_features, int(rank), device=device, dtype=dtype)),
        )
        nn.init.kaiming_uniform_(module.lora_A, a=math.sqrt(5))
        module.lora_scaling = float(alpha) / float(rank)
        module.lora_dropout = nn.Dropout(p=0.0).to(device)
        module.forward = types.MethodType(_lora_forward, module)
        targets.append(name)

    for pname, param in model.named_parameters():
        param.requires_grad_(("lora_A" in pname) or ("lora_B" in pname))
    return targets


def inject_prefix(
    model: nn.Module,
    repo_root: Path,
    num_prefix: int,
    init_by_real_act: bool,
    prefix_precision: str,
) -> List[str]:
    qrw.roberta_smoke.add_medium_models_to_path(repo_root)
    from src.prefix import PrefixTuning  # noqa: E402

    precision = str(prefix_precision).strip().lower()
    if precision not in {"fp16", "fp32"}:
        raise ValueError(f"prefix_precision must be fp16 or fp32, got {prefix_precision!r}")
    if precision == "fp32":
        model.float()
    PrefixTuning(
        model,
        num_prefix=int(num_prefix),
        reparam=False,
        float16=(precision == "fp16"),
        init_by_real_act=bool(init_by_real_act),
    )
    if precision == "fp32":
        for name, param in model.named_parameters():
            if "prefix" in name and param.is_floating_point():
                param.data = param.data.float()
    return [name for name, _ in model.named_parameters() if "prefix" in name]


def adapter_names(model: nn.Module, adapter: str) -> List[str]:
    if adapter == "lora":
        return [name for name, p in model.named_parameters() if p.requires_grad and "lora_" in name]
    if adapter == "prefix":
        return [name for name, p in model.named_parameters() if p.requires_grad and "prefix" in name]
    raise ValueError(adapter)


def sample_adapter_directions(
    master: Dict[str, torch.Tensor],
    trainable_names: Iterable[str],
    seed: int,
) -> Dict[str, torch.Tensor]:
    names = set(trainable_names)
    first = next(iter(master.values()))
    gen = torch.Generator(device=first.device).manual_seed(int(seed))
    directions: Dict[str, torch.Tensor] = {}
    for name, value in master.items():
        if name in names:
            directions[name] = torch.randn(value.shape, generator=gen, device=value.device, dtype=torch.float32)
        else:
            directions[name] = torch.zeros_like(value, dtype=torch.float32)
    return directions


def direction_norm(directions: Dict[str, torch.Tensor], trainable_names: Iterable[str]) -> float:
    total = torch.zeros((), device=next(iter(directions.values())).device, dtype=torch.float64)
    for name in trainable_names:
        d = directions[name].float()
        total += (d * d).double().sum()
    return float(torch.sqrt(total).detach().cpu())


def make_master_with_dtype(model: nn.Module, device: torch.device, dtype_name: str) -> Dict[str, torch.Tensor]:
    dtype = torch.float32 if str(dtype_name).strip().lower() == "fp32" else torch.float16
    return {
        name: p.detach().clone().to(device=device, dtype=dtype)
        for name, p in model.named_parameters()
        if p.detach().is_floating_point()
    }


def compute_true_gradient_quantized(
    harness: qrw.RobertaHarness,
    master: Dict[str, torch.Tensor],
    states: Dict[str, qrw.QuantizerState],
    batch: Dict[str, torch.Tensor],
) -> float:
    params = harness.params()
    qrw.copy_master_to_model(params, master, None, 0.0, 0.0, states)
    harness.model.zero_grad(set_to_none=True)
    loss, _ = forward_loss_roberta(harness, batch)
    loss.backward()
    qrw.restore_master(params, master)
    return float(loss.detach().cpu())


def summarize(records: List[Dict[str, object]]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    keys = sorted({(str(r["adapter"]), float(r["h"])) for r in records})
    for adapter, h in keys:
        group = [r for r in records if str(r["adapter"]) == adapter and abs(float(r["h"]) - h) < 1e-18]
        fd_stats = pooled_fd_true_stats(
            [r.get("d_h_Q") for r in group],
            [r.get("d_true") for r in group],
        )
        rich = pooled_richardson_stats(
            [r.get("d_h_Q") for r in group],
            [r.get("d_half_Q") for r in group],
        )
        finite = [
            r for r in group
            if finite_float(r.get("d_h_Q")) and finite_float(r.get("d_true"))
        ]
        rows.append(
            {
                "adapter": adapter,
                "h": h,
                "n_directions": len(group),
                **fd_stats,
                **rich,
                "d_h_mean": sum(float(r["d_h_Q"]) for r in finite) / len(finite) if finite else None,
                "d_true_mean": sum(float(r["d_true"]) for r in finite) / len(finite) if finite else None,
                "loss_plus_mean": sum(float(r["loss_plus"]) for r in group if finite_float(r.get("loss_plus"))) / max(1, sum(1 for r in group if finite_float(r.get("loss_plus")))),
                "loss_minus_mean": sum(float(r["loss_minus"]) for r in group if finite_float(r.get("loss_minus"))) / max(1, sum(1 for r in group if finite_float(r.get("loss_minus")))),
                "finite_rate": len(finite) / len(group) if group else 0.0,
            }
        )
    return rows


def write_report(path: Path, rows: List[Dict[str, object]], config: Dict[str, object]) -> None:
    def fmt(value) -> str:
        v = finite_float(value)
        return "NA" if v is None else f"{v:.6g}"

    lines = [
        "# INT4 Adapter h-nMSE Probe",
        "",
        "- base quantizer: `INT4_G128_RTNClip_shared_grid_fake_quant`",
        f"- h grid: `{config['h_grid']}`",
        f"- directions per h: `{config['directions']}`",
        "",
        "| adapter | h | fd_true_nmse | corr_fd_true | richardson_rmse_rel | finite_rate |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['adapter']} | {row['h']:.6g} | {fmt(row['fd_true_nmse'])} | "
            f"{fmt(row['corr_fd_true'])} | {fmt(row['richardson_rmse_rel'])} | "
            f"{fmt(row['finite_rate'])} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_adapter(args: argparse.Namespace, adapter: str, device: torch.device, output_dir: Path) -> List[Dict[str, object]]:
    torch.manual_seed(int(args.seed))
    harness = make_harness(args, device)
    model = harness.model

    if adapter == "lora":
        targets = inject_inline_lora(model, int(args.lora_rank), int(args.lora_alpha))
        init_detail = {"lora_targets": targets, "lora_rank": int(args.lora_rank), "lora_alpha": int(args.lora_alpha)}
    elif adapter == "prefix":
        targets = inject_prefix(
            model,
            REPO_ROOT,
            int(args.num_prefix),
            bool(args.prefix_init_by_real_act),
            str(args.prefix_precision),
        )
        init_detail = {
            "prefix_param_names": targets,
            "num_prefix": int(args.num_prefix),
            "prefix_reparam": False,
            "prefix_init_by_real_act": bool(args.prefix_init_by_real_act),
            "prefix_precision": str(args.prefix_precision),
        }
    else:
        raise ValueError(adapter)

    train_names = adapter_names(model, adapter)
    if not train_names:
        raise RuntimeError(f"No trainable {adapter} parameters found")

    master = make_master_with_dtype(model, device, str(args.master_dtype))
    params = harness.params()
    q_names = [
        name for name in qrw.linear_weight_names(model)
        if name in master and "prefix" not in name and "lora_" not in name
    ]
    states, refresh_rows = qrw.refresh_quantizer_states(
        master,
        q_names,
        quantizer="rtnclip",
        activation_rms={},
        bitwidth=4,
        group_size=128,
    )
    data_iter = iter(harness.train_loader)
    batch = qrw.move_batch(next(data_iter), device)
    loss_true = compute_true_gradient_quantized(harness, master, states, batch)

    records: List[Dict[str, object]] = []
    for h in parse_h_grid(args.h_grid):
        for direction_id in range(int(args.directions)):
            seed = qrw.direction_seed(16, f"int4_{adapter}_adapter", 0.0, 0, extra=direction_id)
            directions = sample_adapter_directions(master, train_names, seed)
            d_true = directional_true_derivative(params, directions)
            loss_plus, loss_minus, d_h = finite_difference_pair(harness, master, states, directions, batch, float(h))
            _, _, d_half = finite_difference_pair(harness, master, states, directions, batch, float(h) / 2.0)
            records.append(
                {
                    "adapter": adapter,
                    "h": float(h),
                    "direction_id": int(direction_id),
                    "direction_seed": int(seed),
                    "loss_true_quantized": loss_true,
                    "loss_plus": loss_plus,
                    "loss_minus": loss_minus,
                    "d_h_Q": d_h,
                    "d_half_Q": d_half,
                    "d_true": d_true,
                    "adapter_direction_norm": direction_norm(directions, train_names),
                    "trainable_param_count": sum(master[name].numel() for name in train_names),
                    "trainable_tensor_count": len(train_names),
                    **init_detail,
                }
            )
            qrw.restore_master(params, master)

    write_json(output_dir / f"{adapter}_quantizer_refresh_summary.json", qrw.aggregate_quantizer_stats(refresh_rows, harness.numel_by_quantized_name()))
    write_json(output_dir / f"{adapter}_adapter_config.json", {"adapter": adapter, "trainable_names": train_names, **init_detail})
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="INT4 RTNClip adapter-only h-nMSE probe")
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--task_name", default="sst-5")
    parser.add_argument("--adapters", nargs="+", choices=["lora", "prefix"], default=["lora", "prefix"])
    parser.add_argument("--h_grid", default=" ".join(str(x) for x in DEFAULT_H_GRID))
    parser.add_argument("--directions", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=16)
    parser.add_argument("--data_seed", type=int, default=16)
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--num_prefix", type=int, default=5)
    parser.add_argument("--prefix_init_by_real_act", action="store_true")
    parser.add_argument("--prefix_precision", choices=["fp16", "fp32"], default="fp16")
    parser.add_argument("--master_dtype", choices=["fp16", "fp32"], default="fp16")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("RoBERTa-large INT4 adapter probe requires CUDA")
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "model": "roberta-large",
        "dataset": str(args.task_name),
        "dataset_mode": "full",
        "seed": int(args.seed),
        "data_seed": int(args.data_seed),
        "batch_size": int(args.batch_size),
        "quantizer": "INT4_G128_RTNClip_shared_grid_fake_quant",
        "base_quantized_modules": "Linear.weight excluding adapter params",
        "adapter_params": list(args.adapters),
        "prefix_precision": str(args.prefix_precision),
        "master_dtype": str(args.master_dtype),
        "h_grid": parse_h_grid(args.h_grid),
        "directions": int(args.directions),
    }
    write_json(output_dir / "run_config.json", config)
    write_json(output_dir / "env.json", qrw.collect_env())

    start = time.time()
    all_records: List[Dict[str, object]] = []
    for adapter in args.adapters:
        all_records.extend(run_adapter(args, adapter, device, output_dir))

    rows = summarize(all_records)
    write_csv(output_dir / "probe_records.csv", all_records, sorted({k for r in all_records for k in r.keys()}))
    with (output_dir / "probe_records.jsonl").open("w", encoding="utf-8") as f:
        for record in all_records:
            f.write(json.dumps(record, sort_keys=True, default=json_default) + "\n")
    write_csv(output_dir / "summary.csv", rows, SUMMARY_COLUMNS)
    write_report(output_dir / "summary.md", rows, config)
    write_json(output_dir / "run_summary.json", {"elapsed_sec": time.time() - start, "rows": rows, "config": config})

    print(f"Output: {output_dir}")
    for row in rows:
        nmse = finite_float(row.get("fd_true_nmse"))
        corr_val = finite_float(row.get("corr_fd_true"))
        print(
            f"{row['adapter']} h={row['h']:.6g} "
            f"fd_true_nmse={'NA' if nmse is None else f'{nmse:.6g}'} "
            f"corr={'NA' if corr_val is None else f'{corr_val:.6g}'}"
        )


if __name__ == "__main__":
    main()
