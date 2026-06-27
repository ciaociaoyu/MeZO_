#!/usr/bin/env python
"""Diagnose OPT INT4 RTNClip finite-difference visibility vs theory.

For a fixed unnormalized random direction u, this script compares three active
fractions over Linear weights:

1. uniform-theory: assumes uniform phase within quantization cells;
2. phase-theory: uses the real weight phase within each RTNClip grid cell;
3. actual: applies the project RTNClip quantizer to w +/- h*u.

It also reports the gain distortion

    chi(h) = mean(((Q(w+h*u)-Q(w-h*u))/(2*h*u) - 1)^2).

The quantizer implementation is imported from ``smoke_rtnclip_roberta_sst5``;
no separate ad-hoc RTN quantizer is used for the actual curve.
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
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = REPO_ROOT / "tools"
LARGE_MODELS_DIR = REPO_ROOT / "large_models"
for path in (TOOLS_DIR, LARGE_MODELS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import probe_opt13b_int4_checkpoint_hstar as ckprobe  # noqa: E402
import probe_opt13b_int4_task_grid as optprobe  # noqa: E402
import smoke_rtnclip_roberta_sst5 as rtn  # noqa: E402


DEFAULT_HS = [
    1e-5,
    1.5e-5,
    2e-5,
    3e-5,
    5e-5,
    7e-5,
    1e-4,
    1.5e-4,
    2e-4,
    3e-4,
    5e-4,
    7e-4,
    1e-3,
    1.5e-3,
    2e-3,
    3e-3,
    5e-3,
    7e-3,
    1e-2,
]


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


def finite_float(value: object) -> Optional[float]:
    try:
        out = float(value)  # type: ignore[arg-type]
    except Exception:
        return None
    return out if math.isfinite(out) else None


def linear_weight_names(model: torch.nn.Module, params: Dict[str, torch.nn.Parameter], include_lm_head: bool) -> List[str]:
    names = optprobe.linear_weight_names(model, params)
    if include_lm_head:
        return names
    return [name for name in names if not name.endswith("lm_head.weight") and name != "lm_head.weight"]


def load_checkpoint_master(checkpoint_path: Path, master: Dict[str, torch.Tensor], device: torch.device) -> int:
    state = torch.load(checkpoint_path, map_location="cpu")
    saved = state["master"]
    for name, tensor in saved.items():
        if name in master:
            master[name] = tensor.to(device=device, dtype=master[name].dtype)
    return int(state.get("step", 0))


def scale_like_weight(state: rtn.RTNClipState, weight: torch.Tensor) -> torch.Tensor:
    groups, _, valid = rtn._group_view_2d(weight, state.group_size)
    scale_groups = state.scales.expand_as(groups)
    scale_groups = scale_groups.masked_fill(~valid, 1.0)
    return scale_groups.reshape(weight.shape[0], -1)[:, : weight.shape[1]].reshape_as(weight)


def sample_indices(numel: int, k: int, device: torch.device, gen: torch.Generator) -> Optional[torch.Tensor]:
    if k <= 0 or k >= numel:
        return None
    # Sampling without replacement via randperm is expensive for very large
    # tensors. Duplicates have negligible effect for the diagnostic budgets used
    # here and keep memory bounded.
    return torch.randint(numel, (k,), device=device, generator=gen)


def tensor_values(tensor: torch.Tensor, idx: Optional[torch.Tensor]) -> torch.Tensor:
    flat = tensor.reshape(-1)
    return flat if idx is None else flat.index_select(0, idx)


def allocate_sample_counts(names: Sequence[str], master: Dict[str, torch.Tensor], max_total: int) -> Dict[str, int]:
    total = sum(int(master[name].numel()) for name in names)
    if max_total <= 0 or max_total >= total:
        return {name: int(master[name].numel()) for name in names}
    counts: Dict[str, int] = {}
    remaining = int(max_total)
    for i, name in enumerate(names):
        n = int(master[name].numel())
        if i == len(names) - 1:
            k = min(n, max(1, remaining))
        else:
            k = min(n, max(1, int(round(max_total * n / max(total, 1)))))
            remaining -= k
        counts[name] = k
    return counts


def first_crossing(rows: Sequence[Dict[str, object]], threshold: float) -> Optional[float]:
    for row in rows:
        val = finite_float(row.get("a_actual"))
        if val is not None and val >= threshold:
            return float(row["h"])
    return None


def nearest_row(rows: Sequence[Dict[str, object]], h: float) -> Optional[Dict[str, object]]:
    if not rows or not math.isfinite(h):
        return None
    return min(rows, key=lambda r: abs(math.log(float(r["h"])) - math.log(float(h))))


def safe_mean(total: float, count: int) -> float:
    return float(total / max(count, 1))


def make_plots(rows: List[Dict[str, object]], output_dir: Path, h_smooth: Optional[float]) -> Dict[str, str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        return {"plot_status": f"skipped: {exc}"}

    hs = [float(r["h"]) for r in rows]
    out: Dict[str, str] = {}

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ax.plot(hs, [float(r["a_uniform"]) for r in rows], marker="o", label="uniform theory")
    ax.plot(hs, [float(r["a_phase"]) for r in rows], marker="o", label="phase theory")
    ax.plot(hs, [float(r["a_actual"]) for r in rows], marker="o", label="actual Q")
    ax.axvline(1e-3, color="black", linestyle="--", linewidth=1.0, label="h=1e-3")
    if h_smooth is not None and math.isfinite(h_smooth) and h_smooth > 0:
        ax.axvline(h_smooth, color="tab:red", linestyle=":", linewidth=1.5, label="h_smooth")
    ax.set_xscale("log")
    ax.set_xlabel("h")
    ax.set_ylabel("active fraction")
    ax.set_ylim(bottom=0.0)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        path = output_dir / f"activation_fraction_vs_h.{ext}"
        fig.savefig(path, dpi=200)
        out[f"activation_plot_{ext}"] = str(path)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ax.plot(hs, [float(r["chi"]) for r in rows], marker="o", label="chi")
    ax.axvline(1e-3, color="black", linestyle="--", linewidth=1.0, label="h=1e-3")
    if h_smooth is not None and math.isfinite(h_smooth) and h_smooth > 0:
        ax.axvline(h_smooth, color="tab:red", linestyle=":", linewidth=1.5, label="h_smooth")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("h")
    ax.set_ylabel("chi = mean((gain - 1)^2)")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        path = output_dir / f"gain_distortion_chi_vs_h.{ext}"
        fig.savefig(path, dpi=200)
        out[f"chi_plot_{ext}"] = str(path)
    plt.close(fig)
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", default="facebook/opt-1.3b")
    parser.add_argument("--run_dir", default="", help="Optional run directory containing checkpoints/<tag>/master.pt")
    parser.add_argument("--checkpoint_tag", default="", help="Checkpoint tag to load from --run_dir, e.g. best_acc or final")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--bitwidth", type=int, default=4)
    parser.add_argument("--group_size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=16)
    parser.add_argument("--hs", nargs="+", default=[str(x) for x in DEFAULT_HS])
    parser.add_argument("--h_smooth", type=float, default=float("nan"))
    parser.add_argument("--direction_distribution", choices=["gaussian", "rademacher"], default="gaussian")
    parser.add_argument("--max_total_elements", type=int, default=10_000_000, help="0 means use all coordinates")
    parser.add_argument("--include_lm_head", action="store_true")
    parser.add_argument("--actual_mode", choices=["full_q", "sampled_codes"], default="full_q")
    parser.add_argument("--local_files_only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for OPT activation-radius diagnostic.")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    hs = parse_float_list(args.hs)
    h_smooth = float(args.h_smooth) if finite_float(args.h_smooth) is not None else float("nan")
    write_json(output_dir / "run_config.json", {**vars(args), "hs": hs})
    write_json(output_dir / "env.json", env_info())

    device = torch.device("cuda")
    model_args = argparse.Namespace(model_id=args.model_id, local_files_only=args.local_files_only)
    model, _tokenizer = optprobe.load_model_and_tokenizer(model_args, device)
    params = optprobe.params_map(model)
    master = optprobe.make_master(params, torch.float16)
    checkpoint_step = None
    if args.run_dir and args.checkpoint_tag:
        checkpoint_path = Path(args.run_dir) / "checkpoints" / args.checkpoint_tag / "master.pt"
        checkpoint_step = load_checkpoint_master(checkpoint_path, master, device)
        optprobe.restore_master(params, master)

    q_names = linear_weight_names(model, params, bool(args.include_lm_head))
    sample_counts = allocate_sample_counts(q_names, master, int(args.max_total_elements))
    total_quantized = sum(int(master[name].numel()) for name in q_names)
    total_sampled = sum(sample_counts[name] for name in q_names)

    accum: Dict[float, Dict[str, float]] = {
        h: {
            "uniform_sum": 0.0,
            "phase_sum": 0.0,
            "actual_sum": 0.0,
            "chi_sum": 0.0,
            "n": 0.0,
            "chi_n": 0.0,
        }
        for h in hs
    }
    module_rows: List[Dict[str, object]] = []
    gen = torch.Generator(device=device).manual_seed(int(args.seed))
    index_gen = torch.Generator(device=device).manual_seed(int(args.seed) + 99173)

    with torch.no_grad():
        for module_idx, name in enumerate(q_names):
            weight = master[name].detach()
            state, stats = rtn.compute_rtnclip_state(name, weight, int(args.bitwidth), int(args.group_size))
            scale = scale_like_weight(state, weight).float()
            if args.direction_distribution == "gaussian":
                direction = torch.randn(weight.shape, device=device, generator=gen, dtype=torch.float32)
            else:
                direction = torch.empty(weight.shape, device=device, dtype=torch.float32)
                direction.bernoulli_(0.5, generator=gen)
                direction.mul_(2.0).sub_(1.0)
            k = sample_counts[name]
            idx = sample_indices(int(weight.numel()), k, device, index_gen)
            w_s = tensor_values(weight.float(), idx)
            scale_s = tensor_values(scale, idx).clamp_min(1e-12)
            u_s = tensor_values(direction, idx)
            r = w_s / scale_s - torch.round(w_s / scale_s)
            boundary_dist = scale_s * (0.5 - r.abs()).clamp_min(0.0)

            for h in hs:
                hu_abs = float(h) * u_s.abs()
                if args.direction_distribution == "rademacher":
                    uniform_active = torch.minimum(
                        torch.ones_like(scale_s),
                        torch.full_like(scale_s, 2.0 * float(h)) / scale_s,
                    )
                else:
                    uniform_active = torch.minimum(torch.ones_like(scale_s), 2.0 * hu_abs / scale_s)
                phase_active = (hu_abs >= boundary_dist).float()

                if args.actual_mode == "full_q":
                    q_plus = rtn.quantize_with_state(weight.float().add(direction, alpha=float(h)), state)
                    q_minus = rtn.quantize_with_state(weight.float().add(direction, alpha=-float(h)), state)
                    delta_s = tensor_values((q_plus.float() - q_minus.float()), idx)
                    del q_plus, q_minus
                else:
                    q_plus_code = torch.round((w_s + float(h) * u_s) / scale_s).clamp(-state.qmax, state.qmax)
                    q_minus_code = torch.round((w_s - float(h) * u_s) / scale_s).clamp(-state.qmax, state.qmax)
                    delta_s = (q_plus_code - q_minus_code) * scale_s

                actual_active = (delta_s != 0).float()
                denom = 2.0 * float(h) * u_s
                chi_mask = denom.abs() > 1e-12
                gain = delta_s[chi_mask] / denom[chi_mask]
                chi = (gain - 1.0).square()

                n = int(u_s.numel())
                chi_n = int(chi.numel())
                acc = accum[h]
                acc["uniform_sum"] += float(uniform_active.double().sum().detach().cpu())
                acc["phase_sum"] += float(phase_active.double().sum().detach().cpu())
                acc["actual_sum"] += float(actual_active.double().sum().detach().cpu())
                acc["chi_sum"] += float(chi.double().sum().detach().cpu())
                acc["n"] += n
                acc["chi_n"] += chi_n

                module_rows.append(
                    {
                        "module_name": name,
                        "module_idx": module_idx,
                        "h": float(h),
                        "numel": int(weight.numel()),
                        "sampled": n,
                        "a_uniform": float(uniform_active.float().mean().detach().cpu()),
                        "a_phase": float(phase_active.float().mean().detach().cpu()),
                        "a_actual": float(actual_active.float().mean().detach().cpu()),
                        "chi": float(chi.float().mean().detach().cpu()) if chi_n else float("nan"),
                        "scale_min": stats.get("scale_min"),
                        "scale_median": stats.get("scale_median"),
                        "scale_max": stats.get("scale_max"),
                    }
                )
            del weight, state, scale, direction, w_s, scale_s, u_s, r, boundary_dist
            torch.cuda.empty_cache()
            if (module_idx + 1) % 20 == 0:
                print(f"processed {module_idx + 1}/{len(q_names)} linear weights", flush=True)

    rows: List[Dict[str, object]] = []
    for h in hs:
        acc = accum[h]
        rows.append(
            {
                "h": float(h),
                "a_uniform": safe_mean(acc["uniform_sum"], int(acc["n"])),
                "a_phase": safe_mean(acc["phase_sum"], int(acc["n"])),
                "a_actual": safe_mean(acc["actual_sum"], int(acc["n"])),
                "chi": safe_mean(acc["chi_sum"], int(acc["chi_n"])),
                "sampled_coordinates": int(acc["n"]),
            }
        )
    write_csv(output_dir / "activation_radius_diagnostic.csv", rows)
    write_csv(output_dir / "activation_radius_by_module.csv", module_rows)

    row_h_smooth = nearest_row(rows, h_smooth) if math.isfinite(h_smooth) and h_smooth > 0 else None
    row_default = nearest_row(rows, 1e-3)
    max_phase_actual = max(abs(float(r["a_phase"]) - float(r["a_actual"])) for r in rows)
    max_uniform_actual = max(abs(float(r["a_uniform"]) - float(r["a_actual"])) for r in rows)
    summary = {
        "status": "complete",
        "model_id": args.model_id,
        "bitwidth": int(args.bitwidth),
        "group_size": int(args.group_size),
        "direction_distribution": args.direction_distribution,
        "actual_mode": args.actual_mode,
        "run_dir": args.run_dir,
        "checkpoint_tag": args.checkpoint_tag,
        "checkpoint_step": checkpoint_step,
        "h_smooth": h_smooth,
        "h_smooth_nearest_grid": row_h_smooth["h"] if row_h_smooth else None,
        "h_smooth_a_actual": row_h_smooth["a_actual"] if row_h_smooth else None,
        "h_default_a_actual": row_default["a_actual"] if row_default else None,
        "first_h_a_actual_ge_0p1": first_crossing(rows, 0.1),
        "first_h_a_actual_ge_0p2": first_crossing(rows, 0.2),
        "first_h_a_actual_ge_0p3": first_crossing(rows, 0.3),
        "max_abs_diff_phase_actual": max_phase_actual,
        "max_abs_diff_uniform_actual": max_uniform_actual,
        "total_quantized_linear_coordinates": total_quantized,
        "sampled_coordinates_per_h": total_sampled,
        "q_modules": len(q_names),
        "csv": str(output_dir / "activation_radius_diagnostic.csv"),
    }
    summary.update(make_plots(rows, output_dir, h_smooth if math.isfinite(h_smooth) else None))
    write_json(output_dir / "activation_radius_summary.json", summary)

    print(f"Output CSV: {summary['csv']}")
    print(f"h_smooth nearest-grid a_actual: {summary['h_smooth_nearest_grid']} -> {summary['h_smooth_a_actual']}")
    print(f"h=1e-3 a_actual: {summary['h_default_a_actual']}")
    print(f"first a_actual >= 0.1: {summary['first_h_a_actual_ge_0p1']}")
    print(f"first a_actual >= 0.2: {summary['first_h_a_actual_ge_0p2']}")
    print(f"first a_actual >= 0.3: {summary['first_h_a_actual_ge_0p3']}")
    print(f"max |a_phase-a_actual|: {max_phase_actual}")
    print(f"max |a_uniform-a_actual|: {max_uniform_actual}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
