#!/usr/bin/env python
"""Official AutoAWQ parameter extraction and shared-grid fake-quant probe.

This is a feasibility runner for OPT-1.3B only. It first runs AutoAWQ W4/G128
calibration on SST-5 text, extracts packed AWQ parameters, verifies that a
dequantized fake model reproduces the packed AWQ forward, then runs a small
static-K h-window probe using cached scales/zeros.

The probe is intentionally small and does not launch long training.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import math
import os
import platform
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[1]
LARGE_MODELS_DIR = REPO_ROOT / "large_models"
if str(LARGE_MODELS_DIR) not in sys.path:
    sys.path.insert(0, str(LARGE_MODELS_DIR))

OUT_DIR = REPO_ROOT / "outputs" / "official_awq_krefresh_opt13b"
H_GRID = [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 2e-3, 3e-3, 5e-3, 1e-2]
ORDER_MAP = [0, 2, 4, 6, 1, 3, 5, 7]
EPS = 1e-12


def run_cmd(args: Sequence[str]) -> str:
    try:
        return subprocess.check_output(list(args), cwd=str(REPO_ROOT), text=True, stderr=subprocess.STDOUT).strip()
    except Exception as exc:
        return f"ERROR: {exc}"


def package_version(name: str) -> Optional[str]:
    try:
        import importlib.metadata

        return importlib.metadata.version(name)
    except Exception:
        return None


def env_info() -> Dict[str, Any]:
    return {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "hostname": socket.gethostname(),
        "python": sys.executable,
        "python_version": platform.python_version(),
        "conda_env": os.environ.get("CONDA_DEFAULT_ENV"),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
        "transformers": package_version("transformers"),
        "autoawq": package_version("autoawq"),
        "optimum": package_version("optimum"),
        "datasets": package_version("datasets"),
        "accelerate": package_version("accelerate"),
        "git_commit": run_cmd(["git", "rev-parse", "--short", "HEAD"]),
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")


def write_csv(path: Path, rows: Iterable[Dict[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fields), extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def write_package_check(out_dir: Path, info: Dict[str, Any], ok: bool, error: str = "") -> None:
    lines = [
        "# Official AWQ K-Refresh Package Check",
        "",
        f"Date: {time.strftime('%Y-%m-%d')}",
        f"Python: `{sys.executable}`",
        "",
        f"Status: {'pass' if ok else 'fail'}",
        "",
        "| package / feature | status | observed version / note |",
        "|---|---|---|",
    ]
    for name in ["torch", "transformers", "autoawq", "optimum", "datasets", "accelerate"]:
        lines.append(f"| `{name}` | {'installed' if info.get(name) else 'missing'} | `{info.get(name)}` |")
    lines += [
        f"| CUDA | {'available' if info.get('cuda_available') else 'unavailable'} | `{info.get('gpu_name')}` |",
        "",
        "Supported local AWQ bit width: W4 only, via AutoAWQ.",
        "Quantization target: `facebook/opt-1.3b`, `w_bit=4`, `q_group_size=128`, `zero_point=True`, `version=GEMM`.",
    ]
    if error:
        lines += ["", "## Error", "", f"`{error}`"]
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "package_check.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def check_packages(out_dir: Path) -> Dict[str, Any]:
    info = env_info()
    ok = True
    errors = []
    for mod in ["torch", "transformers", "awq", "optimum"]:
        try:
            importlib.import_module(mod)
        except Exception as exc:
            ok = False
            errors.append(f"{mod}: {type(exc).__name__}: {exc}")
    try:
        from awq import AutoAWQForCausalLM  # noqa: F401
    except Exception as exc:
        ok = False
        errors.append(f"AutoAWQForCausalLM: {type(exc).__name__}: {exc}")
    if not torch.cuda.is_available():
        ok = False
        errors.append("CUDA unavailable")
    write_package_check(out_dir, info, ok, "; ".join(errors))
    if not ok:
        raise RuntimeError("; ".join(errors))
    return info


def stable_seed(base: int, name: str, direction_id: int) -> int:
    digest = hashlib.sha256(f"{base}:{direction_id}:{name}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") % (2**31 - 1)


def load_sst5_texts(n: int) -> Tuple[List[str], str]:
    from tasks import get_task

    task = get_task("SST5")
    train_sets = task.sample_train_sets(
        num_train=-1,
        num_dev=0,
        num_eval=None,
        num_train_sets=1,
        seed=16,
        dataset_mode="full",
        num_k=16,
    )
    samples = train_sets[0]
    texts = [str(s.data.get("text", "")) for s in samples if str(s.data.get("text", "")).strip()]
    return texts[: int(n)], "SST-5"


def tokenized_batches(tokenizer: Any, texts: Sequence[str], batch_size: int, max_length: int, device: torch.device) -> List[Dict[str, torch.Tensor]]:
    batches: List[Dict[str, torch.Tensor]] = []
    for start in range(0, len(texts), batch_size):
        batch_texts = list(texts[start : start + batch_size])
        enc = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        batches.append({k: v.to(device) for k, v in enc.items()})
    return batches


def lm_loss_and_logits(model: nn.Module, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    out = model(input_ids=batch["input_ids"], attention_mask=batch.get("attention_mask"))
    logits = out.logits
    labels = batch["input_ids"].clone()
    if "attention_mask" in batch:
        labels = labels.masked_fill(batch["attention_mask"] == 0, -100)
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)).float(), shift_labels.reshape(-1), ignore_index=-100)
    return loss, logits


def unpack_awq_packed(qpacked: torch.Tensor, out_features: int) -> torch.Tensor:
    q = qpacked.to(torch.int32)
    out = torch.empty((q.shape[0], out_features), dtype=torch.int16, device=q.device)
    pack_cols = q.shape[1]
    for col in range(pack_cols):
        packed = q[:, col]
        for i, orig in enumerate(ORDER_MAP):
            target = col * 8 + orig
            if target < out_features:
                out[:, target] = ((packed >> (4 * i)) & 0xF).to(torch.int16)
    return out


def dequant_awq_weight(qweight: torch.Tensor, qzeros: torch.Tensor, scales: torch.Tensor, group_size: int, out_features: int) -> torch.Tensor:
    intw = unpack_awq_packed(qweight, out_features).to(torch.float32)
    zeros = unpack_awq_packed(qzeros, out_features).to(torch.float32)
    scales_f = scales.to(torch.float32)
    groups = torch.arange(intw.shape[0], device=intw.device) // int(group_size)
    deq_t = (intw - zeros[groups]) * scales_f[groups]
    return deq_t.t().contiguous().to(torch.float16)


def extract_awq_params(awq_model: nn.Module) -> Dict[str, Dict[str, torch.Tensor]]:
    params: Dict[str, Dict[str, torch.Tensor]] = {}
    for name, module in awq_model.named_modules():
        if all(hasattr(module, attr) for attr in ("qweight", "qzeros", "scales", "group_size", "w_bit", "out_features")):
            params[name] = {
                "qweight": module.qweight.detach().cpu().clone(),
                "qzeros": module.qzeros.detach().cpu().clone(),
                "scales": module.scales.detach().cpu().clone(),
                "bias": None if getattr(module, "bias", None) is None else module.bias.detach().cpu().clone(),
                "group_size": torch.tensor(int(module.group_size), dtype=torch.int32),
                "w_bit": torch.tensor(int(module.w_bit), dtype=torch.int32),
                "in_features": torch.tensor(int(module.in_features), dtype=torch.int32),
                "out_features": torch.tensor(int(module.out_features), dtype=torch.int32),
            }
    return params


def params_json(params: Dict[str, Dict[str, torch.Tensor]]) -> Dict[str, Any]:
    rows = {}
    for name, p in params.items():
        rows[name] = {
            "qweight_shape": list(p["qweight"].shape),
            "qzeros_shape": list(p["qzeros"].shape),
            "scales_shape": list(p["scales"].shape),
            "bias_shape": None if p["bias"] is None else list(p["bias"].shape),
            "group_size": int(p["group_size"].item()),
            "w_bit": int(p["w_bit"].item()),
            "in_features": int(p["in_features"].item()),
            "out_features": int(p["out_features"].item()),
            "scale_min": float(p["scales"].float().min().item()),
            "scale_median": float(p["scales"].float().median().item()),
            "scale_max": float(p["scales"].float().max().item()),
        }
    return rows


def set_module_by_name(root: nn.Module, name: str, module: nn.Module) -> None:
    parent = root
    parts = name.split(".")
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], module)


def get_module_by_name(root: nn.Module, name: str) -> nn.Module:
    module = root
    for part in name.split("."):
        module = getattr(module, part)
    return module


def apply_dequant_params_to_fp_model(model: nn.Module, params: Dict[str, Dict[str, torch.Tensor]], device: torch.device) -> Dict[str, torch.Tensor]:
    master: Dict[str, torch.Tensor] = {}
    with torch.no_grad():
        for name, p in params.items():
            module = get_module_by_name(model, name)
            if not isinstance(module, nn.Linear):
                raise TypeError(f"{name} is {type(module)} not nn.Linear")
            deq = dequant_awq_weight(
                p["qweight"].to(device),
                p["qzeros"].to(device),
                p["scales"].to(device),
                int(p["group_size"].item()),
                int(p["out_features"].item()),
            )
            module.weight.data.copy_(deq.to(module.weight.device, dtype=module.weight.dtype))
            if module.bias is not None and p["bias"] is not None:
                module.bias.data.copy_(p["bias"].to(module.bias.device, dtype=module.bias.dtype))
            master[name] = module.weight.detach().clone()
    return master


def copy_compatible_non_awq_state(src: nn.Module, dst: nn.Module) -> int:
    src_state = src.state_dict()
    dst_state = dst.state_dict()
    copied = 0
    packed_suffixes = (".qweight", ".qzeros", ".scales")
    with torch.no_grad():
        for key, value in src_state.items():
            if key.endswith(packed_suffixes):
                continue
            if key not in dst_state:
                continue
            if tuple(value.shape) != tuple(dst_state[key].shape):
                continue
            dst_state[key].copy_(value.to(dst_state[key].device, dtype=dst_state[key].dtype))
            copied += 1
    return copied


def fake_quant_with_params(x: torch.Tensor, p: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
    group_size = int(p["group_size"].item())
    out_features = int(p["out_features"].item())
    scales = p["scales"].to(x.device).float()
    zeros = unpack_awq_packed(p["qzeros"].to(x.device), out_features).float()
    xt = x.t().float().contiguous()
    groups = torch.arange(xt.shape[0], device=xt.device) // group_size
    q = torch.round(xt / scales[groups] + zeros[groups]).clamp(0, 15)
    deq = (q - zeros[groups]) * scales[groups]
    stats = {
        "clip_frac": float(((q <= 0) | (q >= 15)).float().mean().detach().cpu().item()),
        "saturation_frac": float(((q <= 0) | (q >= 15)).float().mean().detach().cpu().item()),
    }
    return deq.t().contiguous().to(torch.float16), q.to(torch.int16), stats


def reconstruction_check(awq_model: nn.Module, fake_model: nn.Module, batches: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, Any]:
    rows = []
    for idx, batch in enumerate(batches):
        with torch.no_grad():
            loss_awq, logits_awq = lm_loss_and_logits(awq_model, batch)
            loss_fake, logits_fake = lm_loss_and_logits(fake_model, batch)
        a = logits_awq.detach().float().reshape(-1)
        b = logits_fake.detach().float().reshape(-1)
        denom = torch.linalg.vector_norm(a) * torch.linalg.vector_norm(b)
        rows.append(
            {
                "batch_id": idx,
                "logits_mse": float(torch.mean((a - b) ** 2).cpu().item()),
                "logits_cosine": float((torch.dot(a, b) / denom.clamp_min(EPS)).cpu().item()),
                "loss_awq": float(loss_awq.detach().cpu().item()),
                "loss_fake": float(loss_fake.detach().cpu().item()),
                "loss_absdiff": float(abs(loss_awq.detach().cpu().item() - loss_fake.detach().cpu().item())),
                "max_abs_error": float(torch.max(torch.abs(a - b)).cpu().item()),
            }
        )
    return {
        "rows": rows,
        "mean_logits_mse": sum(r["logits_mse"] for r in rows) / max(len(rows), 1),
        "mean_logits_cosine": sum(r["logits_cosine"] for r in rows) / max(len(rows), 1),
        "mean_loss_absdiff": sum(r["loss_absdiff"] for r in rows) / max(len(rows), 1),
        "max_abs_error": max((r["max_abs_error"] for r in rows), default=None),
        "pass": bool(rows and min(r["logits_cosine"] for r in rows) > 0.99),
    }


def write_reconstruction_md(path: Path, result: Dict[str, Any]) -> None:
    lines = [
        "# Reconstruction Check",
        "",
        f"Status: {'pass' if result.get('pass') else 'fail'}",
        "",
        f"Mean logits MSE: `{result.get('mean_logits_mse')}`",
        f"Mean logits cosine: `{result.get('mean_logits_cosine')}`",
        f"Mean loss absdiff: `{result.get('mean_loss_absdiff')}`",
        f"Max abs error: `{result.get('max_abs_error')}`",
        "",
        "The fake model uses extracted packed AWQ qweight/qzeros/scales dequantized into FP16 Linear weights.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def apply_branch_weights(model: nn.Module, weights: Dict[str, torch.Tensor]) -> None:
    with torch.no_grad():
        for name, weight in weights.items():
            module = get_module_by_name(model, name)
            module.weight.data.copy_(weight.to(module.weight.device, dtype=module.weight.dtype))


def probe_static(
    model: nn.Module,
    params: Dict[str, Dict[str, torch.Tensor]],
    master: Dict[str, torch.Tensor],
    batches: Sequence[Dict[str, torch.Tensor]],
    h_grid: Sequence[float],
    probe_dirs: int,
    seed: int,
) -> List[Dict[str, Any]]:
    device = next(model.parameters()).device
    records: List[Dict[str, Any]] = []
    model.eval()
    for h in h_grid:
        for direction_id in range(int(probe_dirs)):
            batch = batches[direction_id % len(batches)]
            branch_weights: Dict[str, torch.Tensor] = {}
            branch_weights_half: Dict[str, torch.Tensor] = {}
            acc = {
                "dot": 0.0,
                "dq_norm_sq": 0.0,
                "ideal_norm_sq": 0.0,
                "err_sq": 0.0,
                "code_changed": 0.0,
                "code_total": 0.0,
                "clip_sum": 0.0,
                "sat_sum": 0.0,
                "mod_count": 0,
            }
            q_minus_cache: Dict[str, torch.Tensor] = {}
            q_half_minus_cache: Dict[str, torch.Tensor] = {}
            for name, base in master.items():
                p = params[name]
                gen = torch.Generator(device=device)
                gen.manual_seed(stable_seed(seed, name, direction_id))
                direction = torch.randn(base.shape, generator=gen, device=device, dtype=torch.float16)
                base_d = base.to(device)
                q_plus, c_plus, st_plus = fake_quant_with_params(base_d + float(h) * direction, p)
                q_minus, c_minus, st_minus = fake_quant_with_params(base_d - float(h) * direction, p)
                q_half_plus, _, _ = fake_quant_with_params(base_d + float(h) * 0.5 * direction, p)
                q_half_minus, _, _ = fake_quant_with_params(base_d - float(h) * 0.5 * direction, p)
                branch_weights[name] = q_plus
                q_minus_cache[name] = q_minus
                branch_weights_half[name] = q_half_plus
                q_half_minus_cache[name] = q_half_minus

                dq = (q_plus.float() - q_minus.float()).reshape(-1)
                ideal = (2.0 * float(h) * direction.float()).reshape(-1)
                acc["dot"] += float(torch.dot(dq, ideal).detach().cpu().item())
                acc["dq_norm_sq"] += float(torch.dot(dq, dq).detach().cpu().item())
                acc["ideal_norm_sq"] += float(torch.dot(ideal, ideal).detach().cpu().item())
                diff = dq - ideal
                acc["err_sq"] += float(torch.dot(diff, diff).detach().cpu().item())
                acc["code_changed"] += float((c_plus != c_minus).sum().detach().cpu().item())
                acc["code_total"] += float(c_plus.numel())
                acc["clip_sum"] += 0.5 * (st_plus["clip_frac"] + st_minus["clip_frac"])
                acc["sat_sum"] += 0.5 * (st_plus["saturation_frac"] + st_minus["saturation_frac"])
                acc["mod_count"] += 1
                del direction, q_plus, q_minus, q_half_plus, q_half_minus

            apply_branch_weights(model, branch_weights)
            with torch.no_grad():
                loss_plus, _ = lm_loss_and_logits(model, batch)
            apply_branch_weights(model, q_minus_cache)
            with torch.no_grad():
                loss_minus, _ = lm_loss_and_logits(model, batch)
            apply_branch_weights(model, branch_weights_half)
            with torch.no_grad():
                loss_half_plus, _ = lm_loss_and_logits(model, batch)
            apply_branch_weights(model, q_half_minus_cache)
            with torch.no_grad():
                loss_half_minus, _ = lm_loss_and_logits(model, batch)
            d_h = (float(loss_plus.detach().cpu()) - float(loss_minus.detach().cpu())) / (2.0 * float(h))
            d_half = (float(loss_half_plus.detach().cpu()) - float(loss_half_minus.detach().cpu())) / float(h)
            alignment = acc["dot"] / max(math.sqrt(acc["dq_norm_sq"]) * math.sqrt(acc["ideal_norm_sq"]), EPS)
            norm_ratio = math.sqrt(acc["dq_norm_sq"]) / max(math.sqrt(acc["ideal_norm_sq"]), EPS)
            rel_l2 = math.sqrt(acc["err_sq"]) / max(math.sqrt(acc["ideal_norm_sq"]), EPS)
            records.append(
                {
                    "h": float(h),
                    "direction_id": int(direction_id),
                    "K_refresh": "infinity",
                    "awq_params_step_id": 0,
                    "cached_param_age": 0,
                    "quantizer": "official_awq_param_shared_grid_fake_quant",
                    "quant_bits": 4,
                    "group_size": 128,
                    "pair_shared_grid": True,
                    "fresh_round_codes": True,
                    "loss_plus": float(loss_plus.detach().cpu()),
                    "loss_minus": float(loss_minus.detach().cpu()),
                    "loss_half_plus": float(loss_half_plus.detach().cpu()),
                    "loss_half_minus": float(loss_half_minus.detach().cpu()),
                    "fd": d_h,
                    "d_half": d_half,
                    "richardson_absdiff": abs(d_h - d_half),
                    "richardson_relerr_per_direction": abs(d_h - d_half) / max(abs(d_half), EPS),
                    "alignment": alignment,
                    "norm_ratio": norm_ratio,
                    "delta_visibility_rel_l2": rel_l2,
                    "code_change_frac": acc["code_changed"] / max(acc["code_total"], 1.0),
                    "clip_frac": acc["clip_sum"] / max(acc["mod_count"], 1),
                    "saturation_frac": acc["sat_sum"] / max(acc["mod_count"], 1),
                    "fd_true_available": False,
                    "corr_fd_true": None,
                    "nMSE_fd_true": None,
                    "fd_true_nmse": None,
                }
            )
            apply_branch_weights(model, master)
            torch.cuda.empty_cache()
            print(f"[probe] h={h:g} dir={direction_id} fd={d_h:.6g} d_half={d_half:.6g}", flush=True)
    return records


def summarize_probe(records: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for h in sorted({float(r["h"]) for r in records}):
        group = [r for r in records if float(r["h"]) == h]
        rich_sq = sum((float(r["fd"]) - float(r["d_half"])) ** 2 for r in group)
        half_sq = sum(float(r["d_half"]) ** 2 for r in group)
        rows.append(
            {
                "h": h,
                "n_directions": len(group),
                "alignment_mean": sum(float(r["alignment"]) for r in group) / len(group),
                "norm_ratio_mean": sum(float(r["norm_ratio"]) for r in group) / len(group),
                "code_change_frac_mean": sum(float(r["code_change_frac"]) for r in group) / len(group),
                "clip_frac_mean": sum(float(r["clip_frac"]) for r in group) / len(group),
                "saturation_frac_mean": sum(float(r["saturation_frac"]) for r in group) / len(group),
                "richardson_rmse_rel": math.sqrt(rich_sq / max(half_sq, EPS)),
                "richardson_relerr_median": sorted(float(r["richardson_relerr_per_direction"]) for r in group)[len(group) // 2],
            }
        )
    return rows


def write_probe_summary(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    if not rows:
        path.write_text("# Probe Summary\n\nStatus: not run.\n", encoding="utf-8")
        return
    selected = min(rows, key=lambda r: (float(r["richardson_rmse_rel"]), -float(r["code_change_frac_mean"])))
    lines = [
        "# Probe Summary",
        "",
        "Quantizer: `official_awq_param_shared_grid_fake_quant`",
        "K refresh tested: static step0 / infinity only.",
        "True-gradient diagnostics were not computed in this quick feasibility run.",
        "",
        f"Selected h by Richardson/code-change heuristic: `{selected['h']}`",
        "",
        "| h | alignment | norm_ratio | code_change_frac | richardson_rmse_rel |",
        "|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {float(r['h']):.6g} | {float(r['alignment_mean']):.6g} | {float(r['norm_ratio_mean']):.6g} | "
            f"{float(r['code_change_frac_mean']):.6g} | {float(r['richardson_rmse_rel']):.6g} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default=str(OUT_DIR))
    parser.add_argument("--model_id", default="facebook/opt-1.3b")
    parser.add_argument("--calibration_size", type=int, default=128)
    parser.add_argument("--max_calib_seq_len", type=int, default=128)
    parser.add_argument("--n_parallel_calib_samples", type=int, default=8)
    parser.add_argument("--eval_batches", type=int, default=2)
    parser.add_argument("--eval_batch_size", type=int, default=2)
    parser.add_argument("--probe_dirs", type=int, default=2)
    parser.add_argument("--skip_probe", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    info = check_packages(out_dir)
    write_json(out_dir / "run_environment.json", info)

    from awq import AutoAWQForCausalLM
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device("cuda:0")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    texts, dataset_name = load_sst5_texts(max(int(args.calibration_size), int(args.eval_batches) * int(args.eval_batch_size)))
    calib_texts = texts[: int(args.calibration_size)]
    eval_texts = texts[int(args.calibration_size) : int(args.calibration_size) + int(args.eval_batches) * int(args.eval_batch_size)]
    if len(eval_texts) < int(args.eval_batches) * int(args.eval_batch_size):
        eval_texts = texts[: int(args.eval_batches) * int(args.eval_batch_size)]

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, use_fast=False)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant_config = {"zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM"}
    t0 = time.time()
    awq_wrapper = AutoAWQForCausalLM.from_pretrained(args.model_id, torch_dtype=torch.float16, safetensors=False, device_map=None)
    if getattr(awq_wrapper.model.config, "pad_token_id", None) is None:
        awq_wrapper.model.config.pad_token_id = tokenizer.pad_token_id
    # AutoAWQ 0.2.9 has a Transformers >=4.48 rotary-embedding compatibility
    # branch that is correct for rotary models but not OPT. Keep the installed
    # Transformers version, but bypass that branch for OPT calibration.
    import transformers

    original_transformers_version = transformers.__version__
    if str(getattr(awq_wrapper, "model_type", "")).lower() == "opt" and original_transformers_version >= "4.48.0":
        transformers.__version__ = "4.47.1"
    try:
        awq_wrapper.quantize(
            tokenizer,
            quant_config=quant_config,
            calib_data=calib_texts,
            max_calib_samples=int(args.calibration_size),
            max_calib_seq_len=int(args.max_calib_seq_len),
            n_parallel_calib_samples=int(args.n_parallel_calib_samples),
            apply_clip=True,
        )
    finally:
        transformers.__version__ = original_transformers_version
    awq_model = awq_wrapper.model.to(device)
    awq_model.eval()
    refresh_seconds = time.time() - t0

    params = extract_awq_params(awq_model)
    torch.save(params, out_dir / "awq_params_step000000.pt")
    write_json(
        out_dir / "awq_params_step000000.json",
        {
            "model": args.model_id,
            "dataset": dataset_name,
            "seed": 16,
            "data_seed": 16,
            "quant_config": quant_config,
            "calibration_size": int(args.calibration_size),
            "refresh_seconds": refresh_seconds,
            "module_count": len(params),
            "modules": params_json(params),
        },
    )
    (out_dir / "awq_param_extraction_summary.md").write_text(
        "\n".join(
            [
                "# AWQ Parameter Extraction Summary",
                "",
                "Status: pass",
                f"Model: `{args.model_id}`",
                f"Dataset: `{dataset_name}`",
                f"Quantizer: `official_awq_param_shared_grid_fake_quant`",
                f"Quant config: `{quant_config}`",
                f"Calibration examples: `{int(args.calibration_size)}`",
                f"Quantized module count: `{len(params)}`",
                f"AWQ refresh wall time seconds: `{refresh_seconds:.3f}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    fake_model = AutoModelForCausalLM.from_pretrained(args.model_id, torch_dtype=torch.float16).to(device)
    if fake_model.config.pad_token_id is None:
        fake_model.config.pad_token_id = tokenizer.pad_token_id
    fake_model.eval()
    copied_non_awq = copy_compatible_non_awq_state(awq_model, fake_model)
    master = apply_dequant_params_to_fp_model(fake_model, params, device)
    eval_batches = tokenized_batches(tokenizer, eval_texts, int(args.eval_batch_size), int(args.max_calib_seq_len), device)[: int(args.eval_batches)]

    recon = reconstruction_check(awq_model, fake_model, eval_batches)
    recon["copied_non_awq_state_tensors"] = copied_non_awq
    write_json(out_dir / "reconstruction_check.json", recon)
    write_reconstruction_md(out_dir / "reconstruction_check.md", recon)
    if not recon.get("pass"):
        (out_dir / "failure_report.md").write_text(
            "Reconstruction check failed; probe was not run.\n",
            encoding="utf-8",
        )
        return 2

    smoke = {
        "status": "pass",
        "smoke_type": "static_K_infinity_h_1e-3_probe_iteration",
        "pair_shared_grid": True,
        "fresh_round_codes": True,
        "independent_plus_minus_awq": False,
        "q_w_plus_hu_bypass": False,
        "reconstruction_pass": True,
    }
    write_json(out_dir / "smoke_summary.json", smoke)
    (out_dir / "smoke_summary.md").write_text("# Smoke Summary\n\nStatus: pass after package, extraction, and reconstruction gates.\n", encoding="utf-8")

    if args.skip_probe:
        write_probe_summary(out_dir / "probe_summary.md", [])
        return 0

    records = probe_static(fake_model, params, master, eval_batches, H_GRID, int(args.probe_dirs), 16)
    record_fields = list(records[0].keys()) if records else ["h"]
    write_csv(out_dir / "probe_records.csv", records, record_fields)
    summary = summarize_probe(records)
    summary_fields = list(summary[0].keys()) if summary else ["h"]
    write_csv(out_dir / "probe_results.csv", summary, summary_fields)
    write_probe_summary(out_dir / "probe_summary.md", summary)
    (out_dir / "h_acc_results.csv").write_text("status\nnot_run\n", encoding="utf-8")
    (out_dir / "h_acc_summary.md").write_text("# h-Acc Summary\n\nStatus: not run in this feasibility pass.\n", encoding="utf-8")
    (out_dir / "scheduler_jobs.md").write_text("# Scheduler Jobs\n\nNo scheduler jobs submitted; ran locally on the visible H100.\n", encoding="utf-8")
    (out_dir / "final_report.md").write_text(
        "\n".join(
            [
                "# Final Report",
                "",
                "Status: package/extraction/reconstruction/probe completed for static K=infinity.",
                f"AWQ refresh seconds: `{refresh_seconds:.3f}`",
                "K=100 refresh and h-acc were not run in this feasibility pass.",
                "",
                "The fake quantizer uses extracted AutoAWQ packed qweight/qzeros/scales as a shared grid and fresh-rounds plus/minus branches separately.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
