#!/usr/bin/env python
"""Diagnose whether OPT INT4 h-star double-counts direction dimension in L.

This script recomputes clean directional curvature at a saved OPT FP16-master
checkpoint and compares three radius formulas:

  h_old_style        : raw directional curvature plus sqrt(K_u)
  h_unitL_corrected : curvature divided by ||u||^2 plus sqrt(K_u)
  h_moment_corrected: direct sqrt(E[(u^T H u)^2]) moment, no sqrt(K_u)

It also evaluates INT4 RTNClip actual active fraction and gain distortion near
those radii.  This is diagnostic-only and does not train.
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
from typing import Any, Dict, Iterable, List, Optional, Sequence

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = REPO_ROOT / "tools"
LARGE_MODELS_DIR = REPO_ROOT / "large_models"
for path in (TOOLS_DIR, LARGE_MODELS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import probe_opt13b_int4_task_grid as optprobe  # noqa: E402
import smoke_rtnclip_roberta_sst5 as rtn  # noqa: E402
import train_opt13b_int4_dense_smoke as opttrain  # noqa: E402


DEFAULT_HS = [
    1e-4,
    3e-4,
    5e-4,
    7e-4,
    1e-3,
    1.5e-3,
    2e-3,
    3e-3,
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


def load_checkpoint_master(checkpoint_path: Path, master: Dict[str, torch.Tensor], device: torch.device) -> int:
    state = torch.load(checkpoint_path, map_location="cpu")
    saved = state["master"]
    for name, tensor in saved.items():
        if name in master:
            master[name] = tensor.to(device=device, dtype=master[name].dtype)
    return int(state.get("step", 0))


def direction_norm_sq(directions: Dict[str, torch.Tensor]) -> float:
    total = torch.zeros((), device=next(iter(directions.values())).device, dtype=torch.float64)
    for tensor in directions.values():
        total += tensor.double().square().sum()
    return float(total.detach().cpu())


def apply_clean(
    params: Dict[str, torch.nn.Parameter],
    master32: Dict[str, torch.Tensor],
    directions: Optional[Dict[str, torch.Tensor]],
    rho: float,
    sign: float,
) -> None:
    with torch.no_grad():
        for name, tensor in master32.items():
            value = tensor
            if directions is not None and name in directions:
                value = value.add(directions[name].float(), alpha=float(sign) * float(rho))
            params[name].copy_(value.to(dtype=params[name].dtype))


def clean_loss(model: torch.nn.Module, batch: Any) -> float:
    return float(optprobe.forward_loss(model, batch).detach().cpu())


def clean_backward(model: torch.nn.Module, params: Dict[str, torch.nn.Parameter], master32: Dict[str, torch.Tensor], batch: Any) -> float:
    model.zero_grad(set_to_none=True)
    apply_clean(params, master32, None, 0.0, 0.0)
    loss = optprobe.forward_loss(model, batch)
    loss.backward()
    return float(loss.detach().cpu())


def grad_dot_direction(params: Dict[str, torch.nn.Parameter], directions: Dict[str, torch.Tensor]) -> float:
    total = torch.zeros((), device=next(iter(params.values())).device, dtype=torch.float64)
    for name, direction in directions.items():
        grad = params[name].grad
        if grad is not None:
            total += (grad.detach().double() * direction.double()).sum()
    return float(total.detach().cpu())


def scale_like_weight(state: rtn.RTNClipState, weight: torch.Tensor) -> torch.Tensor:
    groups, _, valid = rtn._group_view_2d(weight, state.group_size)
    scale_groups = state.scales.expand_as(groups)
    scale_groups = scale_groups.masked_fill(~valid, 1.0)
    return scale_groups.reshape(weight.shape[0], -1)[:, : weight.shape[1]].reshape_as(weight)


def sample_indices(numel: int, k: int, device: torch.device, gen: torch.Generator) -> Optional[torch.Tensor]:
    if k <= 0 or k >= numel:
        return None
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


def linear_weight_names(model: torch.nn.Module, params: Dict[str, torch.nn.Parameter], include_lm_head: bool) -> List[str]:
    names = optprobe.linear_weight_names(model, params)
    if include_lm_head:
        return names
    return [name for name in names if not name.endswith("lm_head.weight") and name != "lm_head.weight"]


def active_and_chi(
    master: Dict[str, torch.Tensor],
    q_names: Sequence[str],
    hs: Sequence[float],
    bitwidth: int,
    group_size: int,
    seed: int,
    max_total_elements: int,
) -> List[Dict[str, object]]:
    device = next(iter(master.values())).device
    sample_counts = allocate_sample_counts(q_names, master, int(max_total_elements))
    accum: Dict[float, Dict[str, float]] = {
        float(h): {"actual_sum": 0.0, "chi_sum": 0.0, "n": 0.0, "chi_n": 0.0}
        for h in hs
        if math.isfinite(float(h)) and float(h) > 0
    }
    gen = torch.Generator(device=device).manual_seed(int(seed))
    index_gen = torch.Generator(device=device).manual_seed(int(seed) + 99173)
    with torch.no_grad():
        for module_idx, name in enumerate(q_names):
            weight = master[name].detach()
            state, _stats = rtn.compute_rtnclip_state(name, weight, int(bitwidth), int(group_size))
            direction = torch.randn(weight.shape, device=device, generator=gen, dtype=torch.float32)
            k = sample_counts[name]
            idx = sample_indices(int(weight.numel()), k, device, index_gen)
            u_s = tensor_values(direction, idx)
            for h in accum:
                q_plus = rtn.quantize_with_state(weight.float().add(direction, alpha=float(h)), state)
                q_minus = rtn.quantize_with_state(weight.float().add(direction, alpha=-float(h)), state)
                delta_s = tensor_values((q_plus.float() - q_minus.float()), idx)
                actual = (delta_s != 0).float()
                denom = 2.0 * float(h) * u_s
                mask = denom.abs() > 1e-12
                gain = delta_s[mask] / denom[mask]
                chi = (gain - 1.0).square()
                accum[h]["actual_sum"] += float(actual.double().sum().detach().cpu())
                accum[h]["chi_sum"] += float(chi.double().sum().detach().cpu())
                accum[h]["n"] += int(actual.numel())
                accum[h]["chi_n"] += int(chi.numel())
                del q_plus, q_minus, delta_s, actual, gain, chi
            del weight, state, direction, u_s
            torch.cuda.empty_cache()
            if (module_idx + 1) % 20 == 0:
                print(f"active diagnostic processed {module_idx + 1}/{len(q_names)} linear weights", flush=True)
    rows: List[Dict[str, object]] = []
    for h in sorted(accum):
        row = accum[h]
        rows.append(
            {
                "h": h,
                "a_actual": row["actual_sum"] / max(row["n"], 1.0),
                "chi": row["chi_sum"] / max(row["chi_n"], 1.0),
                "sampled_coordinates": int(row["n"]),
            }
        )
    return rows


def nearest_metric(rows: Sequence[Dict[str, object]], key: str, h: float) -> float:
    if not rows or not math.isfinite(float(h)) or h <= 0:
        return float("nan")
    row = min(rows, key=lambda r: abs(math.log(float(r["h"])) - math.log(float(h))))
    return float(row[key])


def median_abs(values: Sequence[float]) -> float:
    if not values:
        return float("nan")
    t = torch.tensor([abs(float(x)) for x in values], dtype=torch.float64)
    return float(torch.median(t))


def rms(values: Sequence[float]) -> float:
    if not values:
        return float("nan")
    t = torch.tensor([float(x) for x in values], dtype=torch.float64)
    return float(torch.sqrt(torch.mean(t.square())))


def h_formula(delta_eff: float, g_hat: float, curvature: float) -> float:
    if not (delta_eff > 0 and g_hat > 0 and curvature > 0):
        return float("nan")
    return 0.5 * math.sqrt(delta_eff * g_hat / curvature)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--checkpoint_tag", default="final")
    parser.add_argument("--output_dir", required=True)
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
    parser.add_argument("--num_dirs", type=int, default=32)
    parser.add_argument("--rho_values", nargs="+", default=["1e-3", "3e-3"])
    parser.add_argument("--hs", nargs="+", default=[str(x) for x in DEFAULT_HS])
    parser.add_argument("--max_total_elements", type=int, default=5_000_000)
    parser.add_argument("--include_lm_head", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for OPT-1.3B dimension-correction diagnostic.")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rho_values = parse_float_list(args.rho_values)
    base_hs = parse_float_list(args.hs)
    write_json(output_dir / "run_config.json", {**vars(args), "rho_values": rho_values, "hs": base_hs})
    write_json(output_dir / "env.json", env_info())

    device = torch.device("cuda")
    model, tokenizer = optprobe.load_model_and_tokenizer(args, device)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id
    opttrain.patch_mezo_option_loss(model)
    params = optprobe.params_map(model)
    base_master = optprobe.make_master(params, torch.float16)
    checkpoint_path = Path(args.run_dir) / "checkpoints" / args.checkpoint_tag / "master.pt"
    checkpoint_step = load_checkpoint_master(checkpoint_path, base_master, device)
    optprobe.restore_master(params, base_master)

    q_names = linear_weight_names(model, params, bool(args.include_lm_head))
    states, _ = optprobe.refresh_states(base_master, q_names, int(args.bitwidth), int(args.group_size))
    delta_stats = optprobe.weighted_delta_with_optional_masks(states, None)
    delta_eff = float(delta_stats["delta_int4_rtnclip_scale_rms"]) / math.sqrt(6.0)
    d_eff = int(sum(int(base_master[name].numel()) for name in base_master))
    q_dim = int(sum(int(base_master[name].numel()) for name in q_names))

    _task, train_loader, _eval_loader, train_count, _eval_count = opttrain.load_mezo_option_loaders(args, tokenizer)
    batch = opttrain.prepare_batch(next(iter(train_loader)), device)
    master32 = {name: tensor.detach().clone().float() for name, tensor in base_master.items()}
    perturb_names = list(master32.keys())

    old_tf32_matmul = torch.backends.cuda.matmul.allow_tf32
    old_tf32_cudnn = torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model.float()
    direction_rows: List[Dict[str, object]] = []
    summary_rows: List[Dict[str, object]] = []
    try:
        base_loss = clean_backward(model, params, master32, batch)
        d_values: List[float] = []
        norm2_values: List[float] = []
        c_raw_by_rho: Dict[float, List[float]] = {float(rho): [] for rho in rho_values}
        unit_by_rho: Dict[float, List[float]] = {float(rho): [] for rho in rho_values}
        for i in range(int(args.num_dirs)):
            directions = optprobe.sample_direction(master32, perturb_names, int(args.seed) + i * 1009 + 88000, masks=None)
            norm2 = direction_norm_sq(directions)
            d_val = grad_dot_direction(params, directions)
            d_values.append(d_val)
            norm2_values.append(norm2)
            for rho in rho_values:
                rho_f = float(rho)
                apply_clean(params, master32, directions, rho_f, +1.0)
                loss_plus = clean_loss(model, batch)
                apply_clean(params, master32, directions, rho_f, -1.0)
                loss_minus = clean_loss(model, batch)
                apply_clean(params, master32, None, 0.0, 0.0)
                c_raw = (loss_plus - 2.0 * base_loss + loss_minus) / (rho_f * rho_f)
                c_unit = c_raw / max(norm2, 1e-30)
                c_raw_by_rho[rho_f].append(c_raw)
                unit_by_rho[rho_f].append(c_unit)
                direction_rows.append(
                    {
                        "task": args.task,
                        "checkpoint": str(checkpoint_path),
                        "checkpoint_step": checkpoint_step,
                        "rho_for_L": rho_f,
                        "direction_id": i,
                        "norm2": norm2,
                        "d_clean": d_val,
                        "c_raw": c_raw,
                        "c_unit": c_unit,
                        "loss_base": base_loss,
                        "loss_plus": loss_plus,
                        "loss_minus": loss_minus,
                    }
                )
            del directions
            torch.cuda.empty_cache()

        g_hat = rms(d_values)
        sqrt_ku_gaussian = math.sqrt(float(d_eff) * float(d_eff + 2))
        sqrt_ku_sampled = math.sqrt(sum(v * v for v in norm2_values) / max(len(norm2_values), 1))
        sqrt_ku = sqrt_ku_sampled

        for rho in rho_values:
            c_raw_values = c_raw_by_rho[float(rho)]
            unit_values = unit_by_rho[float(rho)]
            l_raw = median_abs(c_raw_values)
            l_unit = median_abs(unit_values)
            kappa2 = rms(c_raw_values)
            h_old_style = h_formula(delta_eff, g_hat, l_raw * sqrt_ku)
            h_unit = h_formula(delta_eff, g_hat, l_unit * sqrt_ku)
            h_moment = h_formula(delta_eff, g_hat, kappa2)
            summary_rows.append(
                {
                    "task": args.task,
                    "checkpoint": str(checkpoint_path),
                    "checkpoint_step": checkpoint_step,
                    "rho_for_L": float(rho),
                    "d_eff": d_eff,
                    "d_quantized_linear": q_dim,
                    "Delta_eff": delta_eff,
                    "G_hat": g_hat,
                    "L_raw": l_raw,
                    "L_unit": l_unit,
                    "kappa2": kappa2,
                    "sqrt_Ku": sqrt_ku,
                    "sqrt_Ku_gaussian": sqrt_ku_gaussian,
                    "sqrt_Ku_sampled": sqrt_ku_sampled,
                    "mean_norm2": sum(norm2_values) / max(len(norm2_values), 1),
                    "L_raw_over_L_unit": l_raw / max(l_unit, 1e-300),
                    "h_old_style": h_old_style,
                    "h_unitL_corrected": h_unit,
                    "h_moment_corrected": h_moment,
                    "num_dirs": int(args.num_dirs),
                    "base_loss": base_loss,
                    "train_count": train_count,
                    "eval_count": _eval_count,
                }
            )
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_tf32_matmul
        torch.backends.cudnn.allow_tf32 = old_tf32_cudnn
        model.half()
        optprobe.restore_master(params, base_master)

    candidate_hs = set(float(h) for h in base_hs)
    for row in summary_rows:
        for key in ("h_old_style", "h_unitL_corrected", "h_moment_corrected"):
            val = float(row[key])
            if math.isfinite(val) and val > 0:
                candidate_hs.add(val)
    active_rows = active_and_chi(
        base_master,
        q_names,
        sorted(candidate_hs),
        int(args.bitwidth),
        int(args.group_size),
        int(args.seed) + 1234567,
        int(args.max_total_elements),
    )
    active_by_h = {float(r["h"]): r for r in active_rows}
    for row in summary_rows:
        for label, h_key in (
            ("old", "h_old_style"),
            ("unitL", "h_unitL_corrected"),
            ("moment", "h_moment_corrected"),
        ):
            h_val = float(row[h_key])
            row[f"active_at_h_{label}"] = nearest_metric(active_rows, "a_actual", h_val)
            row[f"chi_at_h_{label}"] = nearest_metric(active_rows, "chi", h_val)
        row["active_at_1e-3"] = nearest_metric(active_rows, "a_actual", 1e-3)
        row["chi_at_1e-3"] = nearest_metric(active_rows, "chi", 1e-3)
        row["active_at_3e-3"] = nearest_metric(active_rows, "a_actual", 3e-3)
        row["chi_at_3e-3"] = nearest_metric(active_rows, "chi", 3e-3)

    write_csv(output_dir / "direction_curvature_samples.csv", direction_rows)
    write_csv(output_dir / "activation_near_hstar.csv", active_rows)
    write_csv(output_dir / "opt_l_dimension_correction_summary.csv", summary_rows)

    lines = [
        "# OPT INT4 L Dimension-Correction Diagnostic",
        "",
        f"- run_dir: `{args.run_dir}`",
        f"- checkpoint: `{checkpoint_path}`",
        f"- checkpoint_step: `{checkpoint_step}`",
        f"- task: `{args.task}`",
        f"- num_dirs: `{args.num_dirs}`",
        f"- Delta_eff: `{delta_eff:.6g}`",
        f"- d_eff: `{d_eff}`",
        f"- d_quantized_linear: `{q_dim}`",
        "",
        "| rho | G_hat | L_raw | L_unit | kappa2 | mean_norm2 | L_raw/L_unit | h_old | h_unitL | h_moment | active_old | active_unitL | active_moment | active_1e-3 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    f"{float(row['rho_for_L']):.3g}",
                    f"{float(row['G_hat']):.6g}",
                    f"{float(row['L_raw']):.6g}",
                    f"{float(row['L_unit']):.6g}",
                    f"{float(row['kappa2']):.6g}",
                    f"{float(row['mean_norm2']):.6g}",
                    f"{float(row['L_raw_over_L_unit']):.6g}",
                    f"{float(row['h_old_style']):.6g}",
                    f"{float(row['h_unitL_corrected']):.6g}",
                    f"{float(row['h_moment_corrected']):.6g}",
                    f"{float(row['active_at_h_old']):.4f}",
                    f"{float(row['active_at_h_unitL']):.4f}",
                    f"{float(row['active_at_h_moment']):.4f}",
                    f"{float(row['active_at_1e-3']):.4f}",
                ]
            )
            + " |"
        )
    lines.extend(["", "## Interpretation", ""])
    for row in summary_rows:
        old_h = float(row["h_old_style"])
        old_active = float(row["active_at_h_old"])
        unit_h = float(row["h_unitL_corrected"])
        moment_h = float(row["h_moment_corrected"])
        if 1e-4 <= old_h <= 7e-4 and old_active < 0.15:
            lines.append(f"- rho={float(row['rho_for_L']):.3g}: old-style h falls in the low-visibility region.")
        if unit_h > 1.5 * old_h or moment_h > 1.5 * old_h:
            lines.append(f"- rho={float(row['rho_for_L']):.3g}: corrected h is materially larger than old-style h.")
        if unit_h < 7e-4 and moment_h < 7e-4:
            lines.append(
                f"- rho={float(row['rho_for_L']):.3g}: both corrected radii remain below 1e-3; "
                "visibility constraints are still needed."
            )
        if abs(float(row["L_raw_over_L_unit"]) / max(float(row["mean_norm2"]), 1e-30) - 1.0) < 0.05:
            lines.append(
                f"- rho={float(row['rho_for_L']):.3g}: L_raw/L_unit matches mean ||u||^2, "
                "so raw curvature is unit curvature times direction length."
            )
    (output_dir / "OPT_L_DIMENSION_CORRECTION.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(
        output_dir / "summary.json",
        {
            "summary_csv": str(output_dir / "opt_l_dimension_correction_summary.csv"),
            "direction_csv": str(output_dir / "direction_curvature_samples.csv"),
            "activation_csv": str(output_dir / "activation_near_hstar.csv"),
            "report": str(output_dir / "OPT_L_DIMENSION_CORRECTION.md"),
            "rows": summary_rows,
        },
    )

    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
