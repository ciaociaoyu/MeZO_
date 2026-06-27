#!/usr/bin/env python
"""Track OPT-1.3B INT4 curvature estimates during short MeZO training.

This is a diagnostic script.  It reuses the existing OPT INT4 RTNClip shared-grid
forward path and FP16-master dense MeZO update, then periodically re-estimates
the clean FP32 and low-bit second-difference L candidates on the current master.
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
from typing import Any, Dict, Iterable, Iterator, List, Sequence

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = REPO_ROOT / "tools"
LARGE_MODELS_DIR = REPO_ROOT / "large_models"
for path in (TOOLS_DIR, LARGE_MODELS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import probe_opt13b_int4_task_grid as optprobe  # noqa: E402
import train_opt13b_int4_dense_smoke as opttrain  # noqa: E402


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
        "DATALOADER_SHUFFLE": os.environ.get("DATALOADER_SHUFFLE", ""),
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


def parse_int_list(raw: Sequence[str]) -> List[int]:
    vals: List[int] = []
    for item in raw:
        for part in str(item).replace(",", " ").split():
            vals.append(int(part))
    return sorted(set(vals))


def infinite_loader(loader) -> Iterator[Dict[str, Any]]:
    while True:
        for batch in loader:
            yield batch


def clone_master(master: Dict[str, torch.Tensor], dtype: torch.dtype | None = None) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for name, tensor in master.items():
        out[name] = tensor.detach().clone().to(dtype=dtype or tensor.dtype)
    return out


def refresh_master32(master: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {name: tensor.detach().clone().to(dtype=torch.float32) for name, tensor in master.items()}


def direction_norm_sq(directions: Dict[str, torch.Tensor]) -> float:
    total = torch.zeros((), device=next(iter(directions.values())).device, dtype=torch.float64)
    for tensor in directions.values():
        total += tensor.double().square().sum()
    return float(total.detach().cpu())


def add_context(rows: Iterable[Dict[str, object]], *, step: int, kind: str, selected: Dict[str, object], status: str) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    selected_h2 = float(selected.get("h2", float("nan"))) if selected else float("nan")
    selected_q90 = float(selected.get("lambda_q90", float("nan"))) if selected else float("nan")
    for row in rows:
        new = dict(row)
        new.update(
            {
                "step": int(step),
                "kind": kind,
                "selected_h2": selected_h2,
                "selected_lambda_q90": selected_q90,
                "selection_status": status,
            }
        )
        out.append(new)
    return out


def estimate_l_at_step(
    *,
    step: int,
    model,
    params,
    master,
    q_names,
    batch,
    h2_grid: Sequence[float],
    m_l: int,
    seed_base: int,
    bitwidth: int,
    group_size: int,
) -> Dict[str, object]:
    states, _ = optprobe.refresh_states(master, q_names, int(bitwidth), int(group_size))
    clean_master32 = refresh_master32(master)
    clean_selected, clean_status, clean_rows = optprobe.clean_second_diff_l(
        model,
        params,
        clean_master32,
        batch,
        list(master.keys()),
        None,
        seed_base=int(seed_base) + int(step) * 100_003,
        h2_grid=h2_grid,
        m_l=int(m_l),
    )
    optprobe.restore_master(params, master)
    states, _ = optprobe.refresh_states(master, q_names, int(bitwidth), int(group_size))
    low_selected, low_status, low_rows = optprobe.lowbit_second_diff_l(
        model,
        params,
        master,
        batch,
        list(master.keys()),
        None,
        states,
        seed_base=int(seed_base) + int(step) * 100_003,
        h2_grid=h2_grid,
        m_l=int(m_l),
    )
    optprobe.restore_master(params, master)
    delta_stats = optprobe.weighted_delta_with_optional_masks(states, None)
    return {
        "step": int(step),
        "clean_selected": clean_selected,
        "clean_status": clean_status,
        "clean_rows": clean_rows,
        "lowbit_selected": low_selected,
        "lowbit_status": low_status,
        "lowbit_rows": low_rows,
        "delta_stats": delta_stats,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--task", default="sst-2", choices=["sst-2", "sst-5", "rte", "mnli", "trec"])
    parser.add_argument("--model_id", default="facebook/opt-1.3b")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--probe_steps", nargs="+", default=["0", "100", "500", "1000", "2000"])
    parser.add_argument("--h", type=float, default=3e-3)
    parser.add_argument("--lr", type=float, default=3e-7)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--eval_samples", type=int, default=0)
    parser.add_argument("--task_path", choices=["mezo_option"], default="mezo_option")
    parser.add_argument("--dataset_mode", choices=["full", "fewshot", "auto"], default="full")
    parser.add_argument("--num_train", type=int, default=-1)
    parser.add_argument("--num_k", type=int, default=16)
    parser.add_argument("--max_seq_len", type=int, default=128)
    parser.add_argument("--bitwidth", type=int, default=4)
    parser.add_argument("--group_size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=16)
    parser.add_argument("--data_seed", type=int, default=16)
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--h2_grid", nargs="+", default=["1e-5", "3e-5", "1e-4", "3e-4", "1e-3", "3e-3", "1e-2"])
    parser.add_argument("--m_l", type=int, default=4)
    parser.add_argument("--log_every", type=int, default=50)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for OPT-1.3B L-evolution diagnostics.")
    args.task = optprobe.normalize_task(args.task)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    probe_steps = parse_int_list(args.probe_steps)
    h2_grid = parse_float_list(args.h2_grid)
    if 0 not in probe_steps:
        probe_steps.insert(0, 0)
    if int(args.steps) not in probe_steps:
        probe_steps.append(int(args.steps))
    probe_steps = sorted(set(s for s in probe_steps if 0 <= s <= int(args.steps)))
    device = torch.device("cuda")
    config = {
        **vars(args),
        "probe_steps": probe_steps,
        "h2_grid": h2_grid,
        "purpose": "OPT-1.3B SST-2 INT4 dense L evolution under short h=3e-3 MeZO training",
        "quantizer": f"INT{int(args.bitwidth)}_G{int(args.group_size)}_RTNClip_shared_grid_fake_quant",
        "update_backend": "fp16_master",
        "direction": "dense",
        "pair_shared_grid": True,
        "fresh_round_codes": True,
    }
    write_json(output_root / "run_config.json", config)
    write_json(output_root / "env.json", env_info())

    model, tokenizer = optprobe.load_model_and_tokenizer(args, device)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id
    opttrain.patch_mezo_option_loss(model)
    params = optprobe.params_map(model)
    q_names = optprobe.linear_weight_names(model, params)
    master = optprobe.make_master(params, torch.float16)
    _task, train_loader, _eval_loader, train_sample_count, _eval_sample_count = opttrain.load_mezo_option_loaders(args, tokenizer)
    train_iter = infinite_loader(train_loader)

    metrics_rows: List[Dict[str, object]] = []
    l_rows: List[Dict[str, object]] = []
    summary_rows: List[Dict[str, object]] = []
    start_time = time.time()

    def current_batch_for_probe() -> Dict[str, Any]:
        return opttrain.prepare_batch(next(train_iter), device)

    for step in range(0, int(args.steps) + 1):
        if step in probe_steps:
            batch_probe = current_batch_for_probe()
            print(f"Estimating L at step {step} ...", flush=True)
            probe = estimate_l_at_step(
                step=step,
                model=model,
                params=params,
                master=master,
                q_names=q_names,
                batch=batch_probe,
                h2_grid=h2_grid,
                m_l=int(args.m_l),
                seed_base=int(args.seed) * 17_171 + 55_000,
                bitwidth=int(args.bitwidth),
                group_size=int(args.group_size),
            )
            l_rows.extend(
                add_context(
                    probe["clean_rows"],
                    step=step,
                    kind="clean32",
                    selected=probe["clean_selected"],
                    status=str(probe["clean_status"]),
                )
            )
            l_rows.extend(
                add_context(
                    probe["lowbit_rows"],
                    step=step,
                    kind="lowbit_rtnclip",
                    selected=probe["lowbit_selected"],
                    status=str(probe["lowbit_status"]),
                )
            )
            summary = {
                "step": step,
                "clean_L_q90": probe["clean_selected"].get("lambda_q90"),
                "clean_h2": probe["clean_selected"].get("h2"),
                "clean_status": probe["clean_status"],
                "lowbit_L_q90": probe["lowbit_selected"].get("lambda_q90"),
                "lowbit_h2": probe["lowbit_selected"].get("h2"),
                "lowbit_status": probe["lowbit_status"],
                **{f"delta_{k}": v for k, v in probe["delta_stats"].items()},
            }
            summary_rows.append(summary)
            write_csv(output_root / "L_candidates_by_step.csv", l_rows)
            write_csv(output_root / "L_evolution_summary.csv", summary_rows)
            print(
                f"step={step} clean_L={summary['clean_L_q90']:.6g} @h2={summary['clean_h2']} "
                f"lowbit_L={summary['lowbit_L_q90']:.6g} @h2={summary['lowbit_h2']}",
                flush=True,
            )
        if step == int(args.steps):
            break

        batch_train = opttrain.prepare_batch(next(train_iter), device)
        states, _ = optprobe.refresh_states(master, q_names, int(args.bitwidth), int(args.group_size))
        directions = optprobe.sample_direction(master, list(master.keys()), int(args.seed) * 1_000_003 + (step + 1) * 1009, masks=None)
        loss_plus, loss_minus, d_h = opttrain.finite_difference_any(model, params, master, batch_train, states, directions, float(args.h))
        finite = all(math.isfinite(x) for x in (loss_plus, loss_minus, d_h))
        dir_norm_sq = direction_norm_sq(directions)
        update_scale = -float(args.lr) * float(d_h)
        update_norm = abs(update_scale) * math.sqrt(max(dir_norm_sq, 0.0))
        if finite:
            opttrain.update_master(master, directions, update_scale)
            optprobe.restore_master(params, master)
        else:
            print(f"Non-finite update at step {step + 1}; stopping.", flush=True)
            break
        row = {
            "step": step + 1,
            "h": float(args.h),
            "loss_plus": loss_plus,
            "loss_minus": loss_minus,
            "d_h": d_h,
            "train_loss": (loss_plus + loss_minus) / 2.0,
            "direction_norm": math.sqrt(max(dir_norm_sq, 0.0)),
            "update_norm": update_norm,
            "finite": finite,
        }
        metrics_rows.append(row)
        if (step + 1) % max(1, int(args.log_every)) == 0 or step == 0:
            print(
                f"train step={step + 1} loss={row['train_loss']:.6g} d_h={d_h:.6g} update_norm={update_norm:.6g}",
                flush=True,
            )
        if (step + 1) % max(1, int(args.log_every)) == 0:
            write_csv(output_root / "training_metrics.csv", metrics_rows)
        del directions, states
        torch.cuda.empty_cache()

    write_csv(output_root / "training_metrics.csv", metrics_rows)
    write_csv(output_root / "L_candidates_by_step.csv", l_rows)
    write_csv(output_root / "L_evolution_summary.csv", summary_rows)
    write_json(
        output_root / "run_summary.json",
        {
            "status": "complete",
            "steps_completed": metrics_rows[-1]["step"] if metrics_rows else 0,
            "train_sample_count": train_sample_count,
            "runtime_sec": time.time() - start_time,
            "peak_gpu_memory_mb": float(torch.cuda.max_memory_allocated() / 1024 / 1024) if torch.cuda.is_available() else 0.0,
            "summary_csv": str(output_root / "L_evolution_summary.csv"),
            "candidate_csv": str(output_root / "L_candidates_by_step.csv"),
        },
    )
    print(f"Wrote {output_root / 'L_evolution_summary.csv'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
