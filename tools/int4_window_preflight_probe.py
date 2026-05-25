#!/usr/bin/env python
"""INT4 RTNClip window preflight probes for RoBERTa-large / SST-5 K=16.

Probe-only.  No training is launched here.
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
from pathlib import Path
from statistics import median
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = REPO_ROOT / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import smoke_rtnclip_roberta_sst5 as smoke  # noqa: E402


H_GRID = [
    1e-5, 2e-5, 5e-5,
    1e-4, 2e-4, 5e-4,
    1e-3, 2e-3, 5e-3,
    1e-2, 2e-2, 5e-2,
    1e-1, 2e-1, 5e-1,
    1e0,
]
EPS = 1e-12


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


def collect_env() -> Dict[str, object]:
    out = {
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
    return out


def finite_float(value) -> Optional[float]:
    try:
        out = float(value)
    except Exception:
        return None
    return out if math.isfinite(out) else None


def corr(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    pairs = [(float(x), float(y)) for x, y in zip(xs, ys) if math.isfinite(float(x)) and math.isfinite(float(y))]
    if len(pairs) < 2:
        return None
    mx = sum(x for x, _ in pairs) / len(pairs)
    my = sum(y for _, y in pairs) / len(pairs)
    vx = sum((x - mx) ** 2 for x, _ in pairs)
    vy = sum((y - my) ** 2 for _, y in pairs)
    if vx <= 0.0 or vy <= 0.0:
        return None
    return sum((x - mx) * (y - my) for x, y in pairs) / math.sqrt(vx * vy)


def parse_h_grid(raw: str) -> List[float]:
    if not raw:
        return list(H_GRID)
    return [float(x) for x in raw.replace(",", " ").split() if x.strip()]


def load_base(args: argparse.Namespace, device: torch.device):
    load_args = argparse.Namespace(
        repo_root=REPO_ROOT,
        model_id="roberta-large",
        task_name="sst-5",
        seed=int(args.seed),
        data_seed=int(args.data_seed),
        batch_size=int(args.batch_size),
        eval_batch_size=int(args.batch_size),
        dataset_mode="fewshot",
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


def inject_prefix(model, repo_root: Path, num_prefix: int = 5, init_by_real_act: bool = True) -> List[str]:
    smoke.add_medium_models_to_path(repo_root)
    from src.prefix import PrefixTuning  # noqa: E402

    PrefixTuning(model, num_prefix=num_prefix, reparam=False, float16=True, init_by_real_act=init_by_real_act)
    return [name for name, _ in model.named_parameters() if "prefix" in name]


def make_master(params: Dict[str, torch.nn.Parameter], device: torch.device) -> Dict[str, torch.Tensor]:
    return {
        name: p.detach().clone().to(device=device, dtype=torch.float16)
        for name, p in params.items()
        if p.detach().is_floating_point()
    }


def set_probe_grad_flags(params: Dict[str, torch.nn.Parameter], trainable_names: Optional[set] = None) -> None:
    for name, param in params.items():
        if not param.detach().is_floating_point():
            param.requires_grad_(False)
        elif trainable_names is None:
            param.requires_grad_(True)
        else:
            param.requires_grad_(name in trainable_names)


def forward_loss_and_logits_grad(model, batch: Dict[str, torch.Tensor]):
    batch = dict(batch)
    batch["token_type_ids"] = torch.zeros_like(batch["input_ids"])
    outputs = model(**batch)
    return outputs[0], outputs[1]


def compute_quantized_grad(model, params, master, states, batch) -> float:
    smoke.copy_master_to_model(params, master, None, 0.0, 0.0, states)
    model.zero_grad(set_to_none=True)
    loss, _ = forward_loss_and_logits_grad(model, batch)
    loss.backward()
    smoke.restore_master(params, master)
    return float(loss.detach().cpu())


def quantized_fd(model, params, master, states, directions, batch, h: float) -> Tuple[float, float, float]:
    with torch.no_grad():
        smoke.copy_master_to_model(params, master, directions, h, +1.0, states)
        lp, _ = smoke.forward_loss_and_logits(model, batch)
        smoke.copy_master_to_model(params, master, directions, h, -1.0, states)
        lm, _ = smoke.forward_loss_and_logits(model, batch)
        smoke.restore_master(params, master)
    lp_f = float(lp.detach().cpu())
    lm_f = float(lm.detach().cpu())
    return lp_f, lm_f, (lp_f - lm_f) / (2.0 * h)


def effective_true_derivative(params, master, states, directions, h: float) -> Optional[float]:
    device = next(iter(master.values())).device
    acc = torch.zeros((), device=device, dtype=torch.float64)
    seen = False
    for name, param in params.items():
        if name not in master or name not in directions:
            continue
        grad = param.grad
        if grad is None:
            continue
        if name in states:
            state = states[name]
            plus = smoke.quantize_with_state(master[name].float().add(directions[name].float(), alpha=h), state)
            minus = smoke.quantize_with_state(master[name].float().add(directions[name].float(), alpha=-h), state)
            eff = (plus.float() - minus.float()) / (2.0 * h)
        else:
            eff = directions[name].float()
        acc += (grad.detach().float() * eff.float()).double().sum()
        seen = True
    return float(acc.detach().cpu()) if seen else None


def sample_dense(master: Dict[str, torch.Tensor], seed: int) -> Dict[str, torch.Tensor]:
    gen = torch.Generator(device=next(iter(master.values())).device).manual_seed(int(seed))
    return {name: torch.randn(t.shape, device=t.device, generator=gen, dtype=torch.float16) for name, t in master.items()}


def task_gradient_masks(
    master: Dict[str, torch.Tensor],
    params: Dict[str, torch.nn.Parameter],
    q_names: List[str],
    ratio: float,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, object]]:
    q_set = set(q_names)
    chunks: List[torch.Tensor] = []
    names: List[str] = []
    total = 0
    for name, tensor in master.items():
        if name in q_set:
            grad = params[name].grad
            score = torch.zeros(tensor.numel(), device=tensor.device, dtype=torch.float32) if grad is None else grad.detach().float().square().reshape(-1)
            chunks.append(score)
            names.append(name)
            total += int(tensor.numel())
    if not chunks:
        raise RuntimeError("task_grad_static sparse mask has no Linear.weight scores")
    scores = torch.cat(chunks)
    k = min(max(int(math.ceil(float(ratio) * scores.numel())), 1), int(scores.numel()))
    threshold = scores.min() if k >= scores.numel() else torch.kthvalue(scores, scores.numel() - k + 1).values
    masks = {name: torch.zeros_like(tensor, dtype=torch.bool) for name, tensor in master.items()}
    active = 0
    offset = 0
    for name, score in zip(names, chunks):
        numel = score.numel()
        mask = (scores[offset : offset + numel] >= threshold).reshape(master[name].shape).to(device=master[name].device)
        offset += numel
        masks[name] = mask
        active += int(mask.sum().detach().cpu())
    return masks, {
        "mask_strategy": "task_grad_static",
        "mask_source": "task_gradient_square_before_probe",
        "sparse_selection": "global_topk_grad_square_linear_weight",
        "active_param_count": active,
        "total_param_count": sum(int(t.numel()) for t in master.values()),
        "active_param_frac": active / max(sum(int(t.numel()) for t in master.values()), 1),
    }


def magnitude_masks(master: Dict[str, torch.Tensor], ratio: float, strategy: str = "highest_abs") -> Tuple[Dict[str, torch.Tensor], Dict[str, object]]:
    masks: Dict[str, torch.Tensor] = {}
    total = 0
    active = 0
    for name, tensor in master.items():
        flat = tensor.detach().abs().flatten()
        n = flat.numel()
        k = int(math.ceil(float(ratio) * n))
        k = min(max(k, 0), n)
        if k <= 0:
            mask = torch.zeros_like(tensor, dtype=torch.bool)
        elif k >= n:
            mask = torch.ones_like(tensor, dtype=torch.bool)
        elif strategy == "lowest_abs":
            threshold = torch.kthvalue(flat.float(), k).values
            mask = tensor.detach().abs() <= threshold
        else:
            kth_largest = n - k + 1
            threshold = torch.kthvalue(flat.float(), kth_largest).values
            mask = tensor.detach().abs() >= threshold
        masks[name] = mask
        total += n
        active += int(mask.sum().detach().cpu())
    return masks, {
        "mask_strategy": strategy,
        "active_param_count": active,
        "total_param_count": total,
        "active_param_frac": active / max(total, 1),
    }


def sample_masked(master: Dict[str, torch.Tensor], masks: Dict[str, torch.Tensor], seed: int) -> Dict[str, torch.Tensor]:
    dense = sample_dense(master, seed)
    return {name: dense[name] * masks[name].to(device=dense[name].device, dtype=dense[name].dtype) for name in dense}


def sample_prefix_only(master: Dict[str, torch.Tensor], prefix_names: Iterable[str], seed: int) -> Dict[str, torch.Tensor]:
    prefix_set = set(prefix_names)
    dense = sample_dense(master, seed)
    return {name: (dense[name] if name in prefix_set else torch.zeros_like(t, dtype=torch.float16)) for name, t in master.items()}


def scale_rms(states: Dict[str, smoke.RTNClipState]) -> float:
    num = 0.0
    den = 0
    for state in states.values():
        lengths = state.lengths.view(1, -1, 1).double()
        scales = state.scales.double()
        num += float((scales.square() * lengths).sum().detach().cpu()) * int(state.shape[0])
        den += int(lengths.sum().detach().cpu()) * int(state.shape[0])
    return math.sqrt(num / max(den, 1))


def summarize_h(rows: List[Dict[str, object]]) -> Dict[str, object]:
    pairs = [
        (float(r["d_h"]), float(r["d_true"]))
        for r in rows
        if finite_float(r.get("d_h")) is not None and finite_float(r.get("d_true")) is not None
    ]
    if pairs:
        err_sq = sum((a - b) ** 2 for a, b in pairs)
        true_sq = sum(b * b for _, b in pairs)
        nmse = err_sq / max(true_sq, EPS)
        cr = corr([a for a, _ in pairs], [b for _, b in pairs])
        bias = sum(a - b for a, b in pairs) / len(pairs)
    else:
        nmse = None
        cr = None
        bias = None
    rich_pairs = [
        (float(r["d_h"]), float(r["d_half"]))
        for r in rows
        if finite_float(r.get("d_h")) is not None and finite_float(r.get("d_half")) is not None
    ]
    if rich_pairs:
        diff_sq = sum((a - b) ** 2 for a, b in rich_pairs)
        half_sq = sum(b * b for _, b in rich_pairs)
        rich = math.sqrt(diff_sq / max(half_sq, EPS))
    else:
        rich = None
    return {
        "fd_true_nmse": nmse,
        "corr": cr,
        "fd_true_bias": bias,
        "richardson_rmse_rel": rich,
        "finite_rate": len(pairs) / max(len(rows), 1),
        "d_h_mean": sum(float(r["d_h"]) for r in rows if finite_float(r.get("d_h")) is not None) / max(1, sum(1 for r in rows if finite_float(r.get("d_h")) is not None)),
        "d_true_mean": sum(float(r["d_true"]) for r in rows if finite_float(r.get("d_true")) is not None) / max(1, sum(1 for r in rows if finite_float(r.get("d_true")) is not None)),
    }


def membership(h: float, low: Optional[float], high: Optional[float]) -> str:
    if low is None or high is None or not math.isfinite(low) or not math.isfinite(high):
        return "NA"
    if h < low:
        return "L"
    if h > high:
        return "R"
    return "✓"


def window_from_rows(summary_rows: List[Dict[str, object]]) -> Tuple[Optional[float], Optional[float], Optional[Dict[str, object]]]:
    valid = [r for r in summary_rows if finite_float(r.get("fd_true_nmse")) is not None]
    if not valid:
        return None, None, None
    best = min(valid, key=lambda r: float(r["fd_true_nmse"]))
    threshold = max(1.0, 2.0 * float(best["fd_true_nmse"]))
    inside = [r for r in valid if float(r["fd_true_nmse"]) <= threshold and (finite_float(r.get("corr")) is None or float(r["corr"]) >= 0.0)]
    if not inside:
        inside = [best]
    return min(float(r["h"]) for r in inside), max(float(r["h"]) for r in inside), best


def hstar_proxy(setting: str, summary_rows: List[Dict[str, object]], records: List[Dict[str, object]], scale: float, trainable_count: int) -> Dict[str, object]:
    low, high, best = window_from_rows(summary_rows)
    h_emp = float(best["h"]) if best else float("nan")
    near = [r for r in records if abs(float(r["h"]) - 1e-3) < 1e-18 and finite_float(r.get("d_true")) is not None]
    g_value = median([abs(float(r["d_true"])) for r in near]) if near else float("nan")
    rich_vals = [float(r["richardson_rmse_rel"]) for r in summary_rows if finite_float(r.get("richardson_rmse_rel")) is not None]
    l_proxy = median(rich_vals) if rich_vals else float("nan")
    delta = scale / math.sqrt(6.0) if math.isfinite(scale) and scale > 0 else float("nan")
    h_raw = float("nan")
    clip_status = "empirical_fallback"
    if all(math.isfinite(x) and x > 0 for x in (delta, g_value, l_proxy)) and trainable_count > 0:
        h_raw = (delta * delta * g_value * g_value / (16.0 * l_proxy * l_proxy * trainable_count * (trainable_count + 2))) ** 0.25
        clip_status = "simple2pt_proxy_computed"
    h_final = h_raw if math.isfinite(h_raw) and h_raw > 0 else h_emp
    if low is not None and high is not None and math.isfinite(h_final):
        h_final = min(max(h_final, low), high)
        if clip_status == "simple2pt_proxy_computed":
            clip_status = "clipped_to_mse_window" if h_final != h_raw else "inside_mse_window"
    return {
        "dataset": "SST-5",
        "setting": setting,
        "h_star": h_final,
        "h_raw": h_raw,
        "h_final": h_final,
        "clip_status": clip_status,
        "h_empirical_min_nmse": h_emp,
        "window_low": low,
        "window_high": high,
        "membership_1e-5": membership(1e-5, low, high),
        "membership_1e-3": membership(1e-3, low, high),
        "membership_hstar": membership(h_final, low, high) if math.isfinite(h_final) else "NA",
        "selector_name": "simple2pt_corrected",
        "Delta": delta,
        "Delta_mode": "rtnclip_scale_rms_over_sqrt6",
        "G": g_value,
        "G_mode": "median_abs_true_dir_at_h_1e-3",
        "L": l_proxy,
        "L_mode": "richardson_rmse_rel_median_proxy",
        "K": trainable_count,
    }


def run_setting(args: argparse.Namespace, setting: str, out_dir: Path) -> Dict[str, object]:
    device = torch.device("cuda:0")
    torch.manual_seed(int(args.seed))
    model, train_loader, _dev_loader, data_args, sampler = load_base(args, device)
    prefix_names: List[str] = []
    mask_stats: Dict[str, object] = {}
    if setting == "prefix":
        try:
            prefix_names = inject_prefix(model, REPO_ROOT, num_prefix=5, init_by_real_act=True)
            prefix_status = "prefix_init_by_real_act"
        except Exception as exc:
            # Keep going with random prefix if real-act init is not compatible.
            prefix_names = inject_prefix(model, REPO_ROOT, num_prefix=5, init_by_real_act=False)
            prefix_status = f"prefix_random_fallback_after_real_act_error: {type(exc).__name__}: {exc}"
    else:
        prefix_status = ""
    params = smoke.named_parameter_map(model)
    master = make_master(params, device)
    q_names = [name for name in smoke.linear_weight_names(model) if name in master and "prefix" not in name]
    states, qrows = smoke.refresh_quantizer_states(master, q_names, 4, 128)
    qstats = smoke.aggregate_quantizer_stats(qrows, {name: params[name].numel() for name in q_names})
    batch = smoke.move_batch(next(iter(train_loader)), device)
    if setting == "prefix":
        set_probe_grad_flags(params, set(prefix_names))
    else:
        set_probe_grad_flags(params, set(master.keys()))
    base_loss = compute_quantized_grad(model, params, master, states, batch)
    scale = scale_rms(states)
    h_grid = parse_h_grid(args.h_grid)

    if setting.startswith("sparse"):
        ratio = 0.1 if "p0p1" in setting else 0.01
        sparse_strategy = str(args.sparse_mask_strategy).strip().lower()
        if sparse_strategy in {"highest_abs", "lowest_abs"}:
            sparse_strategy = "task_grad_static"
        if sparse_strategy == "task_grad_static":
            masks, mask_stats = task_gradient_masks(master, params, q_names, ratio)
        else:
            masks, mask_stats = magnitude_masks(master, ratio, strategy=sparse_strategy)
        trainable_count = int(mask_stats["active_param_count"])
    elif setting == "prefix":
        trainable_count = sum(int(master[name].numel()) for name in prefix_names)
    else:
        trainable_count = sum(int(t.numel()) for t in master.values())

    records: List[Dict[str, object]] = []
    setting_dir = out_dir / setting
    setting_dir.mkdir(parents=True, exist_ok=True)
    stats_jsonl = setting_dir / "probe_records.jsonl"
    if stats_jsonl.exists():
        stats_jsonl.unlink()

    for h in h_grid:
        for k in range(int(args.directions)):
            seed = int(args.seed) * 1000003 + k
            if setting == "dense":
                directions = sample_dense(master, seed)
            elif setting.startswith("sparse"):
                directions = sample_masked(master, masks, seed)
            elif setting == "prefix":
                directions = sample_prefix_only(master, prefix_names, seed)
            else:
                raise ValueError(setting)
            d_true = effective_true_derivative(params, master, states, directions, float(h))
            lp, lm, d_h = quantized_fd(model, params, master, states, directions, batch, float(h))
            _, _, d_half = quantized_fd(model, params, master, states, directions, batch, float(h) / 2.0)
            pert = smoke.perturbation_metrics(master, directions, states, float(h))
            record = {
                "dataset": "SST-5",
                "setting": setting,
                "h": float(h),
                "k_dir": k,
                "loss_base_quantized": base_loss,
                "loss_plus": lp,
                "loss_minus": lm,
                "d_h": d_h,
                "d_half": d_half,
                "d_true": d_true,
                "fd_true_nmse_metric": "effective_quantized_displacement_grad_dot",
                "mask_strategy": mask_stats.get("mask_strategy", ""),
                "p": mask_stats.get("active_param_frac", ""),
                "active_param_count": mask_stats.get("active_param_count", trainable_count),
                "prefix_status": prefix_status,
                **pert,
            }
            records.append(record)
            append_jsonl(stats_jsonl, record)
            smoke.restore_master(params, master)

    summary_rows: List[Dict[str, object]] = []
    for h in h_grid:
        group = [r for r in records if abs(float(r["h"]) - h) < 1e-18]
        row = {
            "dataset": "SST-5",
            "setting": setting,
            "h": h,
            **summarize_h(group),
        }
        for key in ("active_frac", "alignment", "norm_ratio", "delta_visibility_nmse"):
            vals = [float(r[key]) for r in group if finite_float(r.get(key)) is not None]
            row[key] = sum(vals) / len(vals) if vals else None
        summary_rows.append(row)
    hsum = hstar_proxy(setting, summary_rows, records, scale, trainable_count)
    hsum.update({
        "fisher_default_path_status": "not_found_in_repo; used default dense/fewshot path" if setting == "dense" else "not_applicable",
        "nan_or_inf": any(finite_float(r.get("d_h")) is None or finite_float(r.get("d_true")) is None for r in records),
        "quantizer_scale_rms": scale,
        "quantizer": "INT4_G128_RTNClip_shared_grid_fake_quant",
        "data_dir": getattr(data_args, "data_dir", ""),
        "sampler_name": type(sampler).__name__,
        "prefix_status": prefix_status,
        **mask_stats,
    })
    write_csv(setting_dir / "probe_results.csv", summary_rows, ["dataset", "setting", "h", "fd_true_nmse", "corr", "active_frac", "alignment", "norm_ratio", "finite_rate"])
    write_csv(setting_dir / "probe_records.csv", records)
    write_json(setting_dir / "hstar_summary.json", hsum)
    write_csv(setting_dir / "hstar_summary.csv", [hsum])
    write_json(setting_dir / "run_config.json", {
        "dataset": "SST-5",
        "dataset_mode": "fewshot",
        "num_k": 16,
        "seed": args.seed,
        "data_seed": args.data_seed,
        "batch_size": args.batch_size,
        "h_grid": h_grid,
        "directions": args.directions,
        "setting": setting,
        "quantizer": "INT4_G128_RTNClip_shared_grid_fake_quant",
        "sparse_rescale": "none",
        "prefix_status": prefix_status,
        "qstats": qstats,
    })
    return {"summary_rows": summary_rows, "hstar": hsum, "setting_dir": str(setting_dir)}


def write_summary_md(out_dir: Path, results: Dict[str, Dict[str, object]]) -> None:
    lines = [
        "# INT4 Window Preflight SST-5 Probe",
        "",
        "| Setting | MSE window | h* | nMSE@1e-5 | nMSE@1e-3 | nMSE@h* | corr@1e-5 | corr@1e-3 | corr@h* |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    all_probe_rows: List[Dict[str, object]] = []
    all_hstar_rows: List[Dict[str, object]] = []
    for setting, result in results.items():
        rows = list(result["summary_rows"])
        hsum = dict(result["hstar"])
        all_probe_rows.extend(rows)
        all_hstar_rows.append(hsum)
        by_h = {float(r["h"]): r for r in rows}
        hstar = finite_float(hsum.get("h_final"))
        hstar_row = min(rows, key=lambda r: abs(math.log(float(r["h"])) - math.log(hstar))) if hstar and hstar > 0 else {}
        def v(h, key):
            return by_h.get(h, {}).get(key)
        def fmt(x):
            y = finite_float(x)
            return "NA" if y is None else f"{y:.4g}"
        lines.append(
            f"| {setting} | [{fmt(hsum.get('window_low'))}, {fmt(hsum.get('window_high'))}] | {fmt(hsum.get('h_final'))} | "
            f"{fmt(v(1e-5, 'fd_true_nmse'))} | {fmt(v(1e-3, 'fd_true_nmse'))} | {fmt(hstar_row.get('fd_true_nmse'))} | "
            f"{fmt(v(1e-5, 'corr'))} | {fmt(v(1e-3, 'corr'))} | {fmt(hstar_row.get('corr'))} |"
        )
    write_csv(out_dir / "summary_all_probe_results.csv", all_probe_rows)
    write_csv(out_dir / "summary_all_hstar.csv", all_hstar_rows)
    (out_dir / "summary_sst5_probe.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=Path, default=REPO_ROOT / "outputs/int4_window_preflight/probes_sst5_all_settings")
    parser.add_argument("--settings", nargs="+", default=["dense", "sparse_p0p1", "sparse_p0p01", "prefix"])
    parser.add_argument("--h_grid", default=" ".join(str(h) for h in H_GRID))
    parser.add_argument("--directions", type=int, default=16)
    parser.add_argument("--seed", type=int, default=16)
    parser.add_argument("--data_seed", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--sparse_mask_strategy", choices=["highest_abs", "lowest_abs", "task_grad_static"], default="task_grad_static")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "env.json", collect_env())
    write_json(args.output_dir / "run_config.json", vars(args))
    results: Dict[str, Dict[str, object]] = {}
    start = time.time()
    for setting in args.settings:
        results[setting] = run_setting(args, setting, args.output_dir)
    write_summary_md(args.output_dir, results)
    write_json(args.output_dir / "run_summary.json", {"elapsed_sec": time.time() - start, "settings": {k: v["hstar"] for k, v in results.items()}})
    for setting, result in results.items():
        hsum = result["hstar"]
        print(f"{setting}: h*={hsum.get('h_final')} window=[{hsum.get('window_low')}, {hsum.get('window_high')}] membership_1e-3={hsum.get('membership_1e-3')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
