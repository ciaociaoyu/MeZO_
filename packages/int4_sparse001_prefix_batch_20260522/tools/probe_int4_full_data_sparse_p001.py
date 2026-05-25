#!/usr/bin/env python
"""Full-data INT4 RTNClip sparse p=0.01 probe.

This is probe-only. It reuses the shared-grid RTNClip oracle and computes
effective-displacement true nMSE/corr for sparse unscaled highest-|w| masks.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import socket
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Sequence

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = REPO_ROOT / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import int4_window_preflight_probe as preflight  # noqa: E402
import smoke_rtnclip_roberta_sst5 as smoke  # noqa: E402
from rtnclip_roberta_sst5_batch import build_sparse_masks  # noqa: E402


DEFAULT_H_GRID = [
    1e-5, 2e-5, 5e-5,
    1e-4, 2e-4, 5e-4,
    1e-3, 2e-3, 5e-3,
    1e-2, 2e-2, 5e-2,
    1e-1, 2e-1, 5e-1,
    1.0,
]


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: List[Dict[str, object]], fieldnames: Optional[Sequence[str]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys: List[str] = []
        for row in rows:
            for key in row:
                if key not in keys:
                    keys.append(key)
        fieldnames = keys
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def append_jsonl(path: Path, row: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, sort_keys=True, default=str) + "\n")


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT), text=True).strip()
    except Exception:
        return ""


def env_report() -> Dict[str, object]:
    out: Dict[str, object] = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "hostname": socket.gethostname(),
        "python": sys.executable,
        "python_version": sys.version.replace("\n", " "),
        "conda_default_env": os.environ.get("CONDA_DEFAULT_ENV", ""),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "torch_version": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "git_commit": git_commit(),
    }
    if torch.cuda.is_available():
        out["gpu_name"] = torch.cuda.get_device_name(0)
        out["gpu_count"] = torch.cuda.device_count()
    return out


def parse_h_grid(raw: str) -> List[float]:
    if not raw.strip():
        return list(DEFAULT_H_GRID)
    return [float(x) for x in raw.replace(",", " ").split() if x.strip()]


def load_task(task_name: str, args: argparse.Namespace, device: torch.device):
    load_args = SimpleNamespace(
        repo_root=REPO_ROOT,
        model_id=args.model_id,
        task_name=task_name,
        seed=args.seed,
        data_seed=args.data_seed,
        batch_size=args.batch_size,
        eval_batch_size=args.batch_size,
        dataset_mode="full",
        data_dir=None,
        num_k=16,
    )
    orig_torch_load = torch.load

    def _compat_torch_load(*load_args_, **load_kwargs_):
        load_kwargs_.setdefault("weights_only", False)
        return orig_torch_load(*load_args_, **load_kwargs_)

    torch.load = _compat_torch_load
    try:
        return smoke.load_prompt_model_and_data(load_args, device)
    finally:
        torch.load = orig_torch_load


def sample_sparse(master: Dict[str, torch.Tensor], masks: Dict[str, torch.Tensor], seed: int) -> Dict[str, torch.Tensor]:
    gen = torch.Generator(device=next(iter(master.values())).device).manual_seed(int(seed))
    directions = smoke.sample_directions(master, gen)
    for name, mask in masks.items():
        if name in directions:
            directions[name] = directions[name] * mask.to(device=directions[name].device, dtype=directions[name].dtype)
    return directions


def mean_metric(rows: List[Dict[str, object]], key: str):
    vals = [float(r[key]) for r in rows if preflight.finite_float(r.get(key)) is not None]
    return sum(vals) / len(vals) if vals else None


def summarize_task(
    task: str,
    records: List[Dict[str, object]],
    h_grid: Sequence[float],
    mask_stats: Dict[str, object],
) -> tuple[List[Dict[str, object]], Dict[str, object]]:
    summary_rows: List[Dict[str, object]] = []
    for h in h_grid:
        group = [r for r in records if abs(float(r["h"]) - float(h)) < 1e-18]
        row = {
            "dataset": task,
            "setting": "sparse_p0p01_full",
            "h": float(h),
            "directions": len(group),
            **preflight.summarize_h(group),
            "active_frac": mean_metric(group, "active_frac"),
            "alignment": mean_metric(group, "alignment"),
            "norm_ratio": mean_metric(group, "norm_ratio"),
            "delta_visibility_nmse": mean_metric(group, "delta_visibility_nmse"),
            "code_change_frac": mean_metric(group, "code_change_frac"),
            "nMSE_default_metric": "default_dh_vs_gTu_not_computed",
            "default_fd_true_nmse": None,
            "default_corr_fd_true": None,
            "lowbit_effective_true_nmse": None,
            "lowbit_effective_corr": None,
            **mask_stats,
        }
        row["lowbit_effective_true_nmse"] = row.get("fd_true_nmse")
        row["lowbit_effective_corr"] = row.get("corr")
        summary_rows.append(row)

    low, high, best = preflight.window_from_rows(summary_rows)
    h_emp = float(best["h"]) if best else float("nan")
    hstar = {
        "dataset": task,
        "setting": "sparse_p0p01_full",
        "sparse_ratio": mask_stats.get("sparse_ratio"),
        "sparse_rescale": mask_stats.get("sparse_rescale"),
        "mask_strategy": mask_stats.get("sparse_mask_strategy"),
        "h_empirical_min_nmse": h_emp,
        "window_low": low,
        "window_high": high,
        "membership_1e-5": preflight.membership(1e-5, low, high),
        "membership_1e-3": preflight.membership(1e-3, low, high),
        "membership_h_empirical": preflight.membership(h_emp, low, high) if math.isfinite(h_emp) else "NA",
        "fd_true_nmse_metric": "effective_quantized_displacement_grad_dot",
        "nMSE_default_metric": "default_dh_vs_gTu_not_computed",
        "default_fd_true_nmse": None,
        "default_corr_fd_true": None,
        **mask_stats,
    }
    if best:
        hstar.update({
            "best_lowbit_effective_true_nmse": best.get("lowbit_effective_true_nmse"),
            "best_lowbit_effective_corr": best.get("lowbit_effective_corr"),
            "best_fd_true_nmse": best.get("fd_true_nmse"),
            "best_corr": best.get("corr"),
            "best_alignment": best.get("alignment"),
            "best_norm_ratio": best.get("norm_ratio"),
        })
    return summary_rows, hstar


def run_task(task_name: str, args: argparse.Namespace, output_dir: Path) -> Dict[str, object]:
    task = smoke.normalize_task_name(task_name)
    t0 = time.time()
    device = torch.device("cuda:0")
    torch.manual_seed(int(args.seed))
    torch.cuda.manual_seed_all(int(args.seed))

    model, train_loader, _dev_loader, data_args, sampler = load_task(task, args, device)
    batch = smoke.move_batch(next(iter(train_loader)), device)
    params = smoke.named_parameter_map(model)
    master = {
        name: p.detach().clone().to(device=device, dtype=torch.float16)
        for name, p in params.items()
        if p.detach().is_floating_point()
    }
    q_names = smoke.linear_weight_names(model)
    states, qrows = smoke.refresh_quantizer_states(master, q_names, 4, 128)
    qstats = smoke.aggregate_quantizer_stats(qrows, {name: params[name].numel() for name in q_names})
    masks, mask_stats = build_sparse_masks(
        master,
        sparse_ratio=float(args.sparse_ratio),
        quantized_names=q_names,
        mask_strategy=args.sparse_mask_strategy,
    )
    preflight.set_probe_grad_flags(params, set(master.keys()))
    base_loss = preflight.compute_quantized_grad(model, params, master, states, batch)

    h_grid = parse_h_grid(args.h_grid)
    task_dir = output_dir / task
    records_path = task_dir / "probe_records.jsonl"
    if records_path.exists():
        records_path.unlink()
    records: List[Dict[str, object]] = []

    for h in h_grid:
        for k in range(int(args.directions)):
            seed = int(args.seed) * 1_000_003 + sum(ord(c) for c in task) * 997 + k
            directions = sample_sparse(master, masks, seed)
            d_true = preflight.effective_true_derivative(params, master, states, directions, float(h))
            lp, lm, d_h = preflight.quantized_fd(model, params, master, states, directions, batch, float(h))
            _, _, d_half = preflight.quantized_fd(model, params, master, states, directions, batch, float(h) / 2.0)
            pert = smoke.perturbation_metrics(master, directions, states, float(h))
            record = {
                "dataset": task,
                "setting": "sparse_p0p01_full",
                "dataset_mode": "full",
                "h": float(h),
                "k_dir": k,
                "direction_seed": seed,
                "loss_base_quantized": base_loss,
                "loss_plus": lp,
                "loss_minus": lm,
                "d_h": d_h,
                "d_half": d_half,
                "d_true": d_true,
                "fd_true_nmse_metric": "effective_quantized_displacement_grad_dot",
                **pert,
                **mask_stats,
            }
            records.append(record)
            append_jsonl(records_path, record)
            smoke.restore_master(params, master)

    summary_rows, hstar = summarize_task(task, records, h_grid, mask_stats)
    run_config = {
        "model": args.model_id,
        "dataset": task,
        "dataset_mode": "full",
        "seed": args.seed,
        "data_seed": args.data_seed,
        "batch_size": args.batch_size,
        "sampler_name": type(sampler).__name__,
        "train_size": len(train_loader.dataset),
        "h_grid": h_grid,
        "directions": args.directions,
        "quantizer": "INT4_G128_RTNClip_shared_grid_fake_quant",
        "quant_bits": 4,
        "group_size": 128,
        "direction_mode": "sparse",
        "sparse_ratio": args.sparse_ratio,
        "sparse_mask_strategy": args.sparse_mask_strategy,
        "sparse_rescale": "none",
        "qstats": qstats,
        "resolved_data_dir": getattr(data_args, "data_dir", ""),
    }
    task_dir.mkdir(parents=True, exist_ok=True)
    write_json(task_dir / "run_config.json", run_config)
    write_csv(task_dir / "probe_records.csv", records)
    write_csv(task_dir / "probe_results.csv", summary_rows)
    write_json(task_dir / "hstar_summary.json", hstar)
    write_csv(task_dir / "hstar_summary.csv", [hstar])

    result = {
        **hstar,
        "runtime_sec": time.time() - t0,
        "sampler_name": type(sampler).__name__,
        "train_size": len(train_loader.dataset),
        "base_loss": base_loss,
        "task_dir": str(task_dir),
    }
    del model, train_loader, batch, params, master, states, records
    torch.cuda.empty_cache()
    return result


def write_report(output_dir: Path, task_summaries: List[Dict[str, object]], all_probe_rows: List[Dict[str, object]]) -> None:
    lines = [
        "# Full-Data INT4 Sparse p=0.01 Probe",
        "",
        "Setting: INT4 G128 RTNClip shared-grid fake quantization, sparse highest-|w| mask, p=0.01, unscaled directions.",
        "",
        "This report is a low-bit effective-displacement diagnostic. It does not compute the default nMSE. The default nMSE is `default_dh_vs_gTu` and should be produced by `rtnclip_int4_sparse_mezo_nmse_probe.py`.",
        "",
        "| dataset | window | empirical best h | best effective nMSE | best effective corr | effective nMSE@1e-3 | effective corr@1e-3 | membership@1e-3 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    by_task_h = {}
    for row in all_probe_rows:
        by_task_h[(row["dataset"], float(row["h"]))] = row

    def fmt(value) -> str:
        v = preflight.finite_float(value)
        return "NA" if v is None else f"{v:.6g}"

    for summary in task_summaries:
        task = str(summary["dataset"])
        h1e3 = by_task_h.get((task, 1e-3), {})
        lines.append(
            f"| {task} | [{fmt(summary.get('window_low'))}, {fmt(summary.get('window_high'))}] | "
            f"{fmt(summary.get('h_empirical_min_nmse'))} | {fmt(summary.get('best_fd_true_nmse'))} | "
            f"{fmt(summary.get('best_corr'))} | {fmt(h1e3.get('fd_true_nmse'))} | {fmt(h1e3.get('corr'))} | "
            f"{summary.get('membership_1e-3')} |"
        )
    lines.extend(["", "## Full h table", ""])
    for task in [str(s["dataset"]) for s in task_summaries]:
        lines.extend([
            f"### {task}",
            "",
            "| h | lowbit_effective_true_nmse | lowbit_effective_corr | alignment | norm_ratio | active_frac |",
            "| ---: | ---: | ---: | ---: | ---: | ---: |",
        ])
        for row in [r for r in all_probe_rows if r["dataset"] == task]:
            lines.append(
                f"| {fmt(row.get('h'))} | {fmt(row.get('lowbit_effective_true_nmse', row.get('fd_true_nmse')))} | {fmt(row.get('lowbit_effective_corr', row.get('corr')))} | "
                f"{fmt(row.get('alignment'))} | {fmt(row.get('norm_ratio'))} | {fmt(row.get('active_frac'))} |"
            )
        lines.append("")
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_id", default="roberta-large")
    parser.add_argument("--tasks", nargs="+", default=["sst-2", "sst-5", "rte", "mnli", "trec"])
    parser.add_argument("--h_grid", default=" ".join(str(h) for h in DEFAULT_H_GRID))
    parser.add_argument("--directions", type=int, default=8)
    parser.add_argument("--seed", type=int, default=16)
    parser.add_argument("--data_seed", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--sparse_ratio", type=float, default=0.01)
    parser.add_argument("--sparse_mask_strategy", choices=["highest_abs", "lowest_abs"], default="highest_abs")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for full-data RoBERTa probe.")
    os.environ["DATALOADER_SHUFFLE"] = "True"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "env.json", env_report())
    write_json(output_dir / "run_config.json", vars(args))

    task_summaries: List[Dict[str, object]] = []
    failures: List[Dict[str, object]] = []
    all_probe_rows: List[Dict[str, object]] = []
    start = time.time()
    for task in args.tasks:
        try:
            summary = run_task(task, args, output_dir)
            task_summaries.append(summary)
            probe_path = Path(summary["task_dir"]) / "probe_results.csv"
            all_probe_rows.extend(csv.DictReader(probe_path.open(newline="", encoding="utf-8")))
            print(
                f"{summary['dataset']}: best_h={summary.get('h_empirical_min_nmse')} "
                f"window=[{summary.get('window_low')}, {summary.get('window_high')}] "
                f"nMSE_best={summary.get('best_fd_true_nmse')} corr_best={summary.get('best_corr')}",
                flush=True,
            )
        except Exception as exc:
            failure = {"task": task, "error": repr(exc)}
            failures.append(failure)
            write_json(output_dir / f"failure_{task}.json", failure)
            print(f"FAILED {task}: {exc}", flush=True)
            torch.cuda.empty_cache()

    write_csv(output_dir / "summary_hstar.csv", task_summaries)
    write_csv(output_dir / "summary_all_probe_results.csv", all_probe_rows)
    write_json(output_dir / "failures.json", failures)
    write_report(output_dir, task_summaries, all_probe_rows)
    write_json(
        output_dir / "run_summary.json",
        {
            "status": "complete" if not failures else "partial_failure",
            "elapsed_sec": time.time() - start,
            "tasks": [s["dataset"] for s in task_summaries],
            "failures": failures,
        },
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
