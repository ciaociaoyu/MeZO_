#!/usr/bin/env python
"""Measure actual low-bit effective displacements for OPT INT4 RTNClip.

This reports coordinate-level quantities such as

    Delta_Q_i(h,u) = Q_t(w_i + h u_i) - Q_t(w_i - h u_i)

instead of using only a scale-based Delta surrogate.  It streams over quantized
Linear weights and accumulates exact full-tensor sums for the current model
state.
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
from typing import Any, Dict, Iterable, List, Sequence

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = REPO_ROOT / "tools"
LARGE_MODELS_DIR = REPO_ROOT / "large_models"
for path in (TOOLS_DIR, LARGE_MODELS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import probe_opt13b_int4_task_grid as optprobe  # noqa: E402
import smoke_rtnclip_roberta_sst5 as rtn  # noqa: E402


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


def init_acc() -> Dict[str, float]:
    return {
        "count": 0.0,
        "active": 0.0,
        "delta_abs_sum": 0.0,
        "delta_sq_sum": 0.0,
        "delta_nonzero_abs_sum": 0.0,
        "delta_nonzero_sq_sum": 0.0,
        "intended_abs_sum": 0.0,
        "intended_sq_sum": 0.0,
        "delta_error_abs_sum": 0.0,
        "delta_error_sq_sum": 0.0,
        "dot_delta_intended": 0.0,
    }


def add_acc(acc: Dict[str, float], delta: torch.Tensor, intended: torch.Tensor) -> None:
    delta64 = delta.double()
    intended64 = intended.double()
    err64 = delta64 - intended64
    nonzero = delta != 0
    acc["count"] += float(delta.numel())
    acc["active"] += float(nonzero.sum().detach().cpu())
    acc["delta_abs_sum"] += float(delta64.abs().sum().detach().cpu())
    acc["delta_sq_sum"] += float(delta64.square().sum().detach().cpu())
    if bool(nonzero.any()):
        selected = delta64[nonzero]
        acc["delta_nonzero_abs_sum"] += float(selected.abs().sum().detach().cpu())
        acc["delta_nonzero_sq_sum"] += float(selected.square().sum().detach().cpu())
    acc["intended_abs_sum"] += float(intended64.abs().sum().detach().cpu())
    acc["intended_sq_sum"] += float(intended64.square().sum().detach().cpu())
    acc["delta_error_abs_sum"] += float(err64.abs().sum().detach().cpu())
    acc["delta_error_sq_sum"] += float(err64.square().sum().detach().cpu())
    acc["dot_delta_intended"] += float((delta64 * intended64).sum().detach().cpu())


def finalize_acc(acc: Dict[str, float], h: float, n_dirs: int, module_name: str = "ALL") -> Dict[str, object]:
    count = max(acc["count"], 1.0)
    active = max(acc["active"], 0.0)
    delta_norm = math.sqrt(max(acc["delta_sq_sum"], 0.0))
    intended_norm = math.sqrt(max(acc["intended_sq_sum"], 0.0))
    return {
        "module_name": module_name,
        "h": float(h),
        "n_dirs": int(n_dirs),
        "count": int(acc["count"]),
        "active_frac": acc["active"] / count,
        "actual_delta_abs_mean": acc["delta_abs_sum"] / count,
        "actual_delta_rms": math.sqrt(max(acc["delta_sq_sum"], 0.0) / count),
        "actual_delta_nonzero_abs_mean": acc["delta_nonzero_abs_sum"] / max(active, 1.0),
        "actual_delta_nonzero_rms": math.sqrt(max(acc["delta_nonzero_sq_sum"], 0.0) / max(active, 1.0)),
        "intended_abs_mean": acc["intended_abs_sum"] / count,
        "intended_rms": math.sqrt(max(acc["intended_sq_sum"], 0.0) / count),
        "effective_error_abs_mean": acc["delta_error_abs_sum"] / count,
        "effective_error_rms": math.sqrt(max(acc["delta_error_sq_sum"], 0.0) / count),
        "b_error_rms": math.sqrt(max(acc["delta_error_sq_sum"], 0.0) / count) / max(2.0 * float(h), 1e-30),
        "norm_ratio": delta_norm / max(intended_norm, 1e-30),
        "alignment": acc["dot_delta_intended"] / max(delta_norm * intended_norm, 1e-30),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--model_id", default="facebook/opt-1.3b")
    parser.add_argument("--bitwidth", type=int, default=4)
    parser.add_argument("--group_size", type=int, default=128)
    parser.add_argument("--h_values", nargs="+", default=["1e-4", "3e-4", "1e-3", "3e-3", "1e-2"])
    parser.add_argument("--n_dirs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=16)
    parser.add_argument("--local_files_only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for OPT-1.3B actual-delta probe.")
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    h_values = parse_float_list(args.h_values)
    write_json(output_root / "run_config.json", {**vars(args), "h_values": h_values})
    write_json(output_root / "env.json", env_info())
    device = torch.device("cuda")
    model, _tokenizer = optprobe.load_model_and_tokenizer(args, device)
    params = optprobe.params_map(model)
    q_names = optprobe.linear_weight_names(model, params)
    master = optprobe.make_master(params, torch.float16)
    states, q_rows = optprobe.refresh_states(master, q_names, int(args.bitwidth), int(args.group_size))

    quant_rows: List[Dict[str, object]] = []
    scale_sq_sum = 0.0
    scale_sum = 0.0
    qerr_sq_sum = 0.0
    qerr_abs_sum = 0.0
    count_total = 0.0
    for name in q_names:
        w = master[name].float()
        state = states[name]
        q0 = rtn.quantize_with_state(w, state).float()
        err = q0 - w
        n = float(w.numel())
        scale_expanded = state.scales.expand(-1, -1, state.group_size).reshape(w.shape[0], -1)[:, : w.shape[1]]
        scale_sq_sum += float(scale_expanded.double().square().sum().detach().cpu())
        scale_sum += float(scale_expanded.double().sum().detach().cpu())
        qerr_sq_sum += float(err.double().square().sum().detach().cpu())
        qerr_abs_sum += float(err.double().abs().sum().detach().cpu())
        count_total += n
        quant_rows.append(
            {
                "module_name": name,
                "count": int(n),
                "scale_mean": float(scale_expanded.mean().detach().cpu()),
                "scale_rms": math.sqrt(float(scale_expanded.double().square().mean().detach().cpu())),
                "quant_error_abs_mean": float(err.double().abs().mean().detach().cpu()),
                "quant_error_rms": math.sqrt(float(err.double().square().mean().detach().cpu())),
            }
        )
        del w, q0, err, scale_expanded
    quant_summary = {
        "module_name": "ALL",
        "count": int(count_total),
        "scale_mean": scale_sum / max(count_total, 1.0),
        "scale_rms": math.sqrt(scale_sq_sum / max(count_total, 1.0)),
        "quant_error_abs_mean": qerr_abs_sum / max(count_total, 1.0),
        "quant_error_rms": math.sqrt(qerr_sq_sum / max(count_total, 1.0)),
        "sqrt2_quant_error_rms": math.sqrt(2.0 * qerr_sq_sum / max(count_total, 1.0)),
        "scale_rms_over_sqrt6": math.sqrt(scale_sq_sum / max(count_total, 1.0)) / math.sqrt(6.0),
        "scale_rms_over_sqrt12": math.sqrt(scale_sq_sum / max(count_total, 1.0)) / math.sqrt(12.0),
    }
    quant_rows.insert(0, quant_summary)
    write_csv(output_root / "actual_quant_error_by_module.csv", quant_rows)
    write_json(output_root / "actual_quant_error_summary.json", quant_summary)
    write_csv(output_root / "rtnclip_state_stats.csv", q_rows)

    summary_rows: List[Dict[str, object]] = []
    module_rows: List[Dict[str, object]] = []
    for h in h_values:
        global_acc = init_acc()
        module_accs = {name: init_acc() for name in q_names}
        for dir_id in range(int(args.n_dirs)):
            directions = optprobe.sample_direction(master, q_names, int(args.seed) * 1_000_003 + dir_id * 1009, masks=None)
            for name in q_names:
                w = master[name].float()
                z = directions[name].float()
                state = states[name]
                plus = rtn.quantize_with_state(w.add(z, alpha=float(h)), state).float()
                minus = rtn.quantize_with_state(w.add(z, alpha=-float(h)), state).float()
                delta = plus - minus
                intended = 2.0 * float(h) * z
                add_acc(global_acc, delta, intended)
                add_acc(module_accs[name], delta, intended)
                del w, z, plus, minus, delta, intended
            del directions
            torch.cuda.empty_cache()
        summary_rows.append(finalize_acc(global_acc, h, int(args.n_dirs), module_name="ALL"))
        for name, acc in module_accs.items():
            module_rows.append(finalize_acc(acc, h, int(args.n_dirs), module_name=name))
        write_csv(output_root / "actual_delta_summary.csv", summary_rows)
        write_csv(output_root / "actual_delta_by_module.csv", module_rows)
        print(
            f"h={h:.6g} active={summary_rows[-1]['active_frac']:.6g} "
            f"delta_rms={summary_rows[-1]['actual_delta_rms']:.6g} "
            f"err_rms={summary_rows[-1]['effective_error_rms']:.6g} "
            f"align={summary_rows[-1]['alignment']:.6g}",
            flush=True,
        )
    write_json(
        output_root / "run_summary.json",
        {
            "status": "complete",
            "summary_csv": str(output_root / "actual_delta_summary.csv"),
            "quant_error_summary": quant_summary,
            "num_quantized_modules": len(q_names),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
