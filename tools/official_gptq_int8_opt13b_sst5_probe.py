#!/usr/bin/env python
"""Official AutoGPTQ INT8/G128 shared-grid probe for OPT-1.3B on SST-5 text.

This is a small diagnostics script: it runs official GPTQ quantization once on
the unperturbed FP16 OPT-1.3B weights, extracts the cached groupwise scales and
zero points, then applies those parameters as a shared-grid fake quantizer to
the +h and -h MeZO branches. It does not train.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import socket
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[1]
LARGE_MODELS_DIR = REPO_ROOT / "large_models"
if str(LARGE_MODELS_DIR) not in sys.path:
    sys.path.insert(0, str(LARGE_MODELS_DIR))

EPS = 1e-12


def parse_h_grid(text: str) -> List[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: Sequence[Dict[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fields), extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, sort_keys=True, default=str) + "\n")


def package_version(name: str) -> str | None:
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
        "optimum": package_version("optimum"),
        "auto-gptq": package_version("auto-gptq"),
        "peft": package_version("peft"),
    }


def mean(xs: Iterable[Any]) -> float | None:
    vals = [float(x) for x in xs if x is not None and math.isfinite(float(x))]
    return sum(vals) / len(vals) if vals else None


def median(xs: Iterable[Any]) -> float | None:
    vals = [float(x) for x in xs if x is not None and math.isfinite(float(x))]
    return float(np.median(vals)) if vals else None


def stable_seed(base: int, name: str, direction_id: int) -> int:
    import hashlib

    digest = hashlib.sha256(f"{base}:{direction_id}:{name}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") % (2**31 - 1)


def load_sst5_texts(mode: str, data_seed: int, num_k: int, n: int) -> List[str]:
    from tasks import get_task

    task = get_task("SST5")
    train_sets = task.sample_train_sets(
        num_train=-1,
        num_dev=0,
        num_eval=None,
        num_train_sets=1,
        seed=int(data_seed),
        dataset_mode=str(mode),
        num_k=int(num_k),
    )
    template = task.get_template()
    texts: List[str] = []
    for sample in train_sets[0]:
        try:
            text = template.verbalize(sample, sample.correct_candidate).strip()
        except Exception:
            text = str(getattr(sample, "data", {}).get("text", "")).strip()
        if text:
            texts.append(text)
        if len(texts) >= int(n):
            break
    return texts


def tokenized_batches(tokenizer: Any, texts: Sequence[str], batch_size: int, max_length: int, device: torch.device):
    batches = []
    for start in range(0, len(texts), batch_size):
        enc = tokenizer(
            list(texts[start : start + batch_size]),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        batches.append({k: v.to(device) for k, v in enc.items()})
    return batches


def lm_loss(model: nn.Module, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
    out = model(input_ids=batch["input_ids"], attention_mask=batch.get("attention_mask"))
    logits = out.logits
    labels = batch["input_ids"].clone()
    if "attention_mask" in batch:
        labels = labels.masked_fill(batch["attention_mask"] == 0, -100)
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)).float(),
        shift_labels.reshape(-1),
        ignore_index=-100,
    )


def get_module_by_name(root: nn.Module, name: str) -> nn.Module:
    module = root
    for part in name.split("."):
        module = getattr(module, part)
    return module


def unpack_gptq_zeros(qzeros: torch.Tensor, bits: int, scales_shape: Sequence[int], device: torch.device) -> torch.Tensor:
    wf = torch.tensor(list(range(0, 32, bits)), dtype=torch.int32, device=device).view(1, 1, -1)
    z = qzeros.to(device).to(torch.int32).unsqueeze(2)
    zeros = torch.bitwise_and(torch.bitwise_right_shift(z, wf), (2**bits) - 1).to(torch.float32)
    return (zeros + 1.0).reshape(tuple(scales_shape)).contiguous()


def dequant_gptq_weight(module: nn.Module, zeros: torch.Tensor) -> torch.Tensor:
    bits = int(module.bits)
    device = module.qweight.device
    wf = torch.tensor(list(range(0, 32, bits)), dtype=torch.int32, device=device).view(1, -1, 1)
    w = module.qweight.to(torch.int32).unsqueeze(1)
    codes = torch.bitwise_and(torch.bitwise_right_shift(w, wf), (2**bits) - 1).to(torch.float32)
    codes = codes.reshape(module.qweight.shape[0] * (32 // bits), module.qweight.shape[1])
    scales = module.scales.float()
    g_idx = module.g_idx.long()
    deq_t = scales[g_idx] * (codes - zeros[g_idx])
    return deq_t.t().contiguous().to(torch.float16)


def extract_gptq_params(qmodel: nn.Module, device: torch.device) -> Dict[str, Dict[str, torch.Tensor | int]]:
    params: Dict[str, Dict[str, torch.Tensor | int]] = {}
    for name, module in qmodel.named_modules():
        if all(hasattr(module, attr) for attr in ("qweight", "qzeros", "scales", "g_idx", "bits", "group_size")):
            bits = int(module.bits)
            zeros = unpack_gptq_zeros(module.qzeros.detach(), bits, module.scales.shape, device)
            params[name] = {
                "bits": bits,
                "group_size": int(module.group_size),
                "infeatures": int(module.infeatures),
                "outfeatures": int(module.outfeatures),
                "scales": module.scales.detach().to(device).float().contiguous(),
                "zeros": zeros,
                "g_idx": module.g_idx.detach().to(device).long().contiguous(),
                "bias": None if getattr(module, "bias", None) is None else module.bias.detach().cpu().clone(),
                "qweight_shape": list(module.qweight.shape),
                "qzeros_shape": list(module.qzeros.shape),
                "scales_shape": list(module.scales.shape),
            }
    return params


def fake_quant_gptq(x: torch.Tensor, p: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
    scales = p["scales"]
    zeros = p["zeros"]
    g_idx = p["g_idx"]
    bits = int(p["bits"])
    maxq = float((2**bits) - 1)
    xt = x.t().float().contiguous()
    q = torch.round(xt / scales[g_idx] + zeros[g_idx]).clamp(0.0, maxq)
    deq = (q - zeros[g_idx]) * scales[g_idx]
    clip = ((q <= 0.0) | (q >= maxq)).float().mean()
    return deq.t().contiguous().to(torch.float16), q.to(torch.int16), {
        "clip_frac": float(clip.detach().cpu()),
        "saturation_frac": float(clip.detach().cpu()),
    }


def make_direction(shape: torch.Size, device: torch.device, seed: int, name: str, direction_id: int) -> torch.Tensor:
    gen = torch.Generator(device=device)
    gen.manual_seed(stable_seed(seed, name, direction_id))
    return torch.randn(shape, generator=gen, device=device, dtype=torch.float16)


def apply_branch_weights(model: nn.Module, weights: Dict[str, torch.Tensor]) -> None:
    with torch.no_grad():
        for name, weight in weights.items():
            module = get_module_by_name(model, name)
            module.weight.data.copy_(weight.to(module.weight.device, dtype=module.weight.dtype))


def restore_master(model: nn.Module, master: Dict[str, torch.Tensor]) -> None:
    apply_branch_weights(model, master)


def quantize_official_gptq(args: argparse.Namespace, texts: Sequence[str], tokenizer: Any, device: torch.device):
    from optimum.gptq import GPTQQuantizer
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.float16,
        device_map={"": 0},
    ).eval()
    if getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = tokenizer.pad_token_id

    t0 = time.time()
    quantizer = GPTQQuantizer(
        bits=int(args.bits),
        dataset=list(texts),
        group_size=int(args.group_size),
        batch_size=1,
        pad_token_id=tokenizer.pad_token_id,
        model_seqlen=int(args.max_calib_seq_len),
        use_cuda_fp16=True,
        disable_exllama=True,
        true_sequential=True,
        sym=True,
    )
    qmodel = quantizer.quantize_model(model, tokenizer=tokenizer)
    elapsed = time.time() - t0
    params = extract_gptq_params(qmodel, device)
    module_summary = {}
    for name, p in params.items():
        module_summary[name] = {
            "bits": p["bits"],
            "group_size": p["group_size"],
            "infeatures": p["infeatures"],
            "outfeatures": p["outfeatures"],
            "qweight_shape": p["qweight_shape"],
            "qzeros_shape": p["qzeros_shape"],
            "scales_shape": p["scales_shape"],
            "scale_min": float(p["scales"].min().detach().cpu()),
            "scale_median": float(p["scales"].median().detach().cpu()),
            "scale_max": float(p["scales"].max().detach().cpu()),
        }
    return qmodel, params, module_summary, elapsed


def reconstruction_check(qmodel: nn.Module, fp_model: nn.Module, params: Dict[str, Dict[str, Any]], batch: Dict[str, torch.Tensor]):
    deq_weights: Dict[str, torch.Tensor] = {}
    with torch.no_grad():
        for name, p in params.items():
            # Recreate the official QuantLinear dequantization exactly from packed codes.
            module = get_module_by_name(qmodel, name)
            deq_weights[name] = dequant_gptq_weight(module, p["zeros"])
        apply_branch_weights(fp_model, deq_weights)
        loss_q = lm_loss(qmodel, batch)
        logits_q = qmodel(input_ids=batch["input_ids"], attention_mask=batch.get("attention_mask")).logits
        loss_f = lm_loss(fp_model, batch)
        logits_f = fp_model(input_ids=batch["input_ids"], attention_mask=batch.get("attention_mask")).logits
    a = logits_q.float().reshape(-1)
    b = logits_f.float().reshape(-1)
    denom = torch.linalg.vector_norm(a) * torch.linalg.vector_norm(b)
    return {
        "logits_mse": float(torch.mean((a - b) ** 2).detach().cpu()),
        "logits_cosine": float((torch.dot(a, b) / denom.clamp_min(EPS)).detach().cpu()),
        "loss_official_gptq": float(loss_q.detach().cpu()),
        "loss_fake_dequant": float(loss_f.detach().cpu()),
        "loss_absdiff": float(abs(float(loss_q.detach().cpu()) - float(loss_f.detach().cpu()))),
        "max_abs_error": float(torch.max(torch.abs(a - b)).detach().cpu()),
    }


def true_directional_derivative(
    model: nn.Module,
    master: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    seed: int,
    direction_id: int,
) -> float:
    restore_master(model, master)
    model.zero_grad(set_to_none=True)
    with torch.enable_grad():
        loss = lm_loss(model, batch)
        loss.backward()
    total = 0.0
    device = next(model.parameters()).device
    with torch.no_grad():
        for name, base in master.items():
            module = get_module_by_name(model, name)
            if module.weight.grad is None:
                continue
            direction = make_direction(base.shape, device, seed, name, direction_id)
            total += float(torch.sum(module.weight.grad.detach().float() * direction.float()).detach().cpu())
    model.zero_grad(set_to_none=True)
    return float(total)


def probe(
    model: nn.Module,
    params: Dict[str, Dict[str, Any]],
    master: Dict[str, torch.Tensor],
    batches: Sequence[Dict[str, torch.Tensor]],
    h_grid: Sequence[float],
    probe_dirs: int,
    seed: int,
    output_records: Path,
    true_grad: bool,
) -> List[Dict[str, Any]]:
    if output_records.exists():
        output_records.unlink()
    device = next(model.parameters()).device
    records: List[Dict[str, Any]] = []
    for h in h_grid:
        for direction_id in range(int(probe_dirs)):
            batch = batches[direction_id % len(batches)]
            plus: Dict[str, torch.Tensor] = {}
            minus: Dict[str, torch.Tensor] = {}
            half_plus: Dict[str, torch.Tensor] = {}
            half_minus: Dict[str, torch.Tensor] = {}
            acc = {
                "dot": 0.0,
                "dq_norm_sq": 0.0,
                "ideal_norm_sq": 0.0,
                "err_sq": 0.0,
                "err_count": 0.0,
                "code_changed": 0.0,
                "code_total": 0.0,
                "clip": 0.0,
                "sat": 0.0,
                "mods": 0.0,
            }

            with torch.no_grad():
                for name, base in master.items():
                    p = params[name]
                    direction = make_direction(base.shape, device, seed, name, direction_id)
                    base_d = base.to(device)
                    q_plus, c_plus, st_plus = fake_quant_gptq(base_d + float(h) * direction, p)
                    q_minus, c_minus, st_minus = fake_quant_gptq(base_d - float(h) * direction, p)
                    q_half_plus, _, _ = fake_quant_gptq(base_d + 0.5 * float(h) * direction, p)
                    q_half_minus, _, _ = fake_quant_gptq(base_d - 0.5 * float(h) * direction, p)

                    plus[name] = q_plus
                    minus[name] = q_minus
                    half_plus[name] = q_half_plus
                    half_minus[name] = q_half_minus

                    dq = (q_plus.float() - q_minus.float()).reshape(-1)
                    ideal = (2.0 * float(h) * direction.float()).reshape(-1)
                    err = dq - ideal
                    acc["dot"] += float(torch.dot(dq, ideal).detach().cpu())
                    acc["dq_norm_sq"] += float(torch.dot(dq, dq).detach().cpu())
                    acc["ideal_norm_sq"] += float(torch.dot(ideal, ideal).detach().cpu())
                    acc["err_sq"] += float(torch.dot(err, err).detach().cpu())
                    acc["err_count"] += float(err.numel())
                    acc["code_changed"] += float((c_plus != c_minus).sum().detach().cpu())
                    acc["code_total"] += float(c_plus.numel())
                    acc["clip"] += 0.5 * (st_plus["clip_frac"] + st_minus["clip_frac"])
                    acc["sat"] += 0.5 * (st_plus["saturation_frac"] + st_minus["saturation_frac"])
                    acc["mods"] += 1.0

            apply_branch_weights(model, plus)
            with torch.no_grad():
                loss_plus = lm_loss(model, batch)
            apply_branch_weights(model, minus)
            with torch.no_grad():
                loss_minus = lm_loss(model, batch)
            apply_branch_weights(model, half_plus)
            with torch.no_grad():
                loss_half_plus = lm_loss(model, batch)
            apply_branch_weights(model, half_minus)
            with torch.no_grad():
                loss_half_minus = lm_loss(model, batch)

            d_h = (float(loss_plus.detach().cpu()) - float(loss_minus.detach().cpu())) / (2.0 * float(h))
            d_half = (float(loss_half_plus.detach().cpu()) - float(loss_half_minus.detach().cpu())) / float(h)
            d_true = None
            fd_true_error = None
            if true_grad:
                d_true = true_directional_derivative(model, master, batch, seed, direction_id)
                fd_true_error = d_h - d_true
            alignment = acc["dot"] / max(math.sqrt(acc["dq_norm_sq"]) * math.sqrt(acc["ideal_norm_sq"]), EPS)
            norm_ratio = math.sqrt(acc["dq_norm_sq"]) / max(math.sqrt(acc["ideal_norm_sq"]), EPS)
            delta_nmse = acc["err_sq"] / max(acc["ideal_norm_sq"], EPS)
            row = {
                "h": float(h),
                "direction_id": int(direction_id),
                "quantizer": "official_gptq_int8_param_shared_grid_fake_quant",
                "quant_bits": 8,
                "group_size": 128,
                "pair_shared_grid": True,
                "fresh_round_codes": True,
                "loss_plus": float(loss_plus.detach().cpu()),
                "loss_minus": float(loss_minus.detach().cpu()),
                "loss_half_plus": float(loss_half_plus.detach().cpu()),
                "loss_half_minus": float(loss_half_minus.detach().cpu()),
                "d_h_Q": d_h,
                "d_half_Q": d_half,
                "richardson_absdiff": abs(d_h - d_half),
                "richardson_relerr_per_direction": abs(d_h - d_half) / max(abs(d_half), EPS),
                "delta_visibility_mse": acc["err_sq"] / max(acc["err_count"], 1.0),
                "delta_visibility_nmse": delta_nmse,
                "delta_visibility_rel_l2": math.sqrt(delta_nmse),
                "alignment": alignment,
                "norm_ratio": norm_ratio,
                "code_change_frac": acc["code_changed"] / max(acc["code_total"], 1.0),
                "active_frac": 1.0,
                "clip_frac": acc["clip"] / max(acc["mods"], 1.0),
                "saturation_frac": acc["sat"] / max(acc["mods"], 1.0),
                "fd_true_available": bool(true_grad),
                "d_true": d_true,
                "fd_true_error": fd_true_error,
                "fd_true_nmse": None,
                "corr_fd_true": None,
            }
            records.append(row)
            append_jsonl(output_records, row)
            restore_master(model, master)
            print(
                f"[probe] h={h:g} dir={direction_id} "
                f"align={alignment:.4g} norm={norm_ratio:.4g} rich={row['richardson_relerr_per_direction']:.4g}",
                flush=True,
            )
            del plus, minus, half_plus, half_minus
            torch.cuda.empty_cache()
    return records


def summarize(records: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for h in sorted({float(r["h"]) for r in records}):
        group = [r for r in records if float(r["h"]) == h]
        diff_sq = sum((float(r["d_h_Q"]) - float(r["d_half_Q"])) ** 2 for r in group)
        half_sq = sum(float(r["d_half_Q"]) ** 2 for r in group)
        fd_true_pairs = [
            (float(r["d_h_Q"]), float(r["d_true"]))
            for r in group
            if r.get("fd_true_available") and r.get("d_true") is not None
        ]
        if fd_true_pairs:
            err_sq = sum((fd - true) ** 2 for fd, true in fd_true_pairs)
            true_sq = sum(true**2 for _fd, true in fd_true_pairs)
            fd_true_mse = err_sq / len(fd_true_pairs)
            fd_true_nmse = err_sq / max(true_sq, EPS)
            fd_true_rmse = math.sqrt(fd_true_mse)
            fd_true_bias = sum(fd - true for fd, true in fd_true_pairs) / len(fd_true_pairs)
            fd_arr = np.array([fd for fd, _true in fd_true_pairs], dtype=np.float64)
            true_arr = np.array([true for _fd, true in fd_true_pairs], dtype=np.float64)
            if len(fd_true_pairs) >= 2 and float(np.std(fd_arr)) > 0.0 and float(np.std(true_arr)) > 0.0:
                corr_fd_true = float(np.corrcoef(fd_arr, true_arr)[0, 1])
            else:
                corr_fd_true = None
        else:
            fd_true_mse = None
            fd_true_nmse = None
            fd_true_rmse = None
            fd_true_bias = None
            corr_fd_true = None
        rows.append(
            {
                "h": h,
                "n_directions": len(group),
                "alignment_mean": mean(r["alignment"] for r in group),
                "norm_ratio_mean": mean(r["norm_ratio"] for r in group),
                "delta_visibility_nmse_mean": mean(r["delta_visibility_nmse"] for r in group),
                "delta_visibility_nmse_median": median(r["delta_visibility_nmse"] for r in group),
                "delta_visibility_rel_l2_mean": mean(r["delta_visibility_rel_l2"] for r in group),
                "code_change_frac_mean": mean(r["code_change_frac"] for r in group),
                "clip_frac_mean": mean(r["clip_frac"] for r in group),
                "saturation_frac_mean": mean(r["saturation_frac"] for r in group),
                "richardson_rmse_rel": math.sqrt(diff_sq / max(half_sq, EPS)),
                "richardson_relerr_median": median(r["richardson_relerr_per_direction"] for r in group),
                "d_h_Q_mean": mean(r["d_h_Q"] for r in group),
                "d_half_Q_mean": mean(r["d_half_Q"] for r in group),
                "fd_true_available": bool(fd_true_pairs),
                "fd_true_mse": fd_true_mse,
                "fd_true_nmse": fd_true_nmse,
                "fd_true_rmse": fd_true_rmse,
                "fd_true_bias": fd_true_bias,
                "corr_fd_true": corr_fd_true,
            }
        )
    return rows


def write_summary_md(path: Path, rows: Sequence[Dict[str, Any]], quantize_seconds: float, recon: Dict[str, Any]) -> None:
    lines = [
        "# Official GPTQ INT8 OPT-1.3B SST-5 Probe",
        "",
        "This is a probe-only run using official `auto-gptq` through `optimum.gptq.GPTQQuantizer`.",
        "The ZO branches use cached GPTQ scales/zero points from unperturbed FP16 weights, shared by +h and -h, with fresh rounding.",
        "",
        f"- GPTQ quantization runtime: `{quantize_seconds:.1f}` sec",
        f"- reconstruction logits cosine: `{recon.get('logits_cosine')}`",
        f"- reconstruction loss absdiff: `{recon.get('loss_absdiff')}`",
        "",
        "| h | alignment | norm_ratio | delta_nmse | fd_true_nmse | corr_fd_true | code_change | Richardson rmse rel | d_h mean |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        fd_nmse = "NA" if row.get("fd_true_nmse") is None else f"{row['fd_true_nmse']:.4g}"
        corr = "NA" if row.get("corr_fd_true") is None else f"{row['corr_fd_true']:.4g}"
        lines.append(
            f"| {row['h']:.1e} | {row['alignment_mean']:.4g} | {row['norm_ratio_mean']:.4g} | "
            f"{row['delta_visibility_nmse_mean']:.4g} | {fd_nmse} | {corr} | {row['code_change_frac_mean']:.4g} | "
            f"{row['richardson_rmse_rel']:.4g} | {row['d_h_Q_mean']:.4g} |"
        )
    h1 = next((r for r in rows if abs(float(r["h"]) - 1e-3) < 1e-12), None)
    if h1:
        lines.extend(
            [
                "",
                "## 1e-3 Check",
                "",
                (
                    "At h=1e-3, GPTQ INT8 does not show the AWQ-W4-style severe geometry collapse "
                    f"in this probe: alignment={h1['alignment_mean']:.4g}, "
                    f"norm_ratio={h1['norm_ratio_mean']:.4g}, "
                    f"delta_visibility_nmse={h1['delta_visibility_nmse_mean']:.4g}."
                ),
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_root", type=Path, default=Path("outputs/official_gptq_int8_opt13b_sst5_probe"))
    parser.add_argument("--model_id", default="facebook/opt-1.3b")
    parser.add_argument("--dataset_mode", choices=["fewshot", "full"], default="fewshot")
    parser.add_argument("--num_k", type=int, default=16)
    parser.add_argument("--seed", type=int, default=16)
    parser.add_argument("--data_seed", type=int, default=16)
    parser.add_argument("--bits", type=int, default=8)
    parser.add_argument("--group_size", type=int, default=128)
    parser.add_argument("--calibration_size", type=int, default=8)
    parser.add_argument("--eval_batch_size", type=int, default=2)
    parser.add_argument("--eval_batches", type=int, default=1)
    parser.add_argument("--max_length", type=int, default=64)
    parser.add_argument("--max_calib_seq_len", type=int, default=64)
    parser.add_argument("--probe_dirs", type=int, default=2)
    parser.add_argument("--h_grid", default="1e-4,3e-4,1e-3,3e-3,1e-2")
    parser.add_argument("--true_grad", action="store_true")
    args = parser.parse_args()

    if int(args.bits) != 8:
        raise ValueError("This script is scoped to official GPTQ INT8 only.")
    if int(args.group_size) != 128:
        raise ValueError("This script is scoped to group_size=128 only.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this probe.")

    out = args.output_root
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "run_environment.json", env_info())
    torch.manual_seed(int(args.seed))
    device = torch.device("cuda:0")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    needed_texts = int(args.calibration_size) + int(args.eval_batches) * int(args.eval_batch_size)
    texts = load_sst5_texts(args.dataset_mode, args.data_seed, args.num_k, needed_texts)
    calib_texts = texts[: int(args.calibration_size)]
    eval_texts = texts[int(args.calibration_size) : needed_texts]
    if len(eval_texts) < int(args.eval_batches) * int(args.eval_batch_size):
        eval_texts = texts[: int(args.eval_batches) * int(args.eval_batch_size)]

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, use_fast=False)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("[gptq] running official quantization", flush=True)
    qmodel, params, module_summary, quantize_seconds = quantize_official_gptq(args, calib_texts, tokenizer, device)
    write_json(
        out / "gptq_param_extraction_summary.json",
        {
            "model_id": args.model_id,
            "dataset": "SST-5",
            "dataset_mode": args.dataset_mode,
            "seed": args.seed,
            "data_seed": args.data_seed,
            "bits": args.bits,
            "group_size": args.group_size,
            "calibration_size": args.calibration_size,
            "quantize_seconds": quantize_seconds,
            "module_count": len(params),
            "modules": module_summary,
        },
    )

    print("[gptq] loading FP16 model for shared-grid fake quant", flush=True)
    fp_model = AutoModelForCausalLM.from_pretrained(args.model_id, torch_dtype=torch.float16).to(device).eval()
    if getattr(fp_model.config, "pad_token_id", None) is None:
        fp_model.config.pad_token_id = tokenizer.pad_token_id
    batches = tokenized_batches(tokenizer, eval_texts, args.eval_batch_size, args.max_length, device)
    recon = reconstruction_check(qmodel, fp_model, params, batches[0])
    write_json(out / "reconstruction_check.json", recon)

    master: Dict[str, torch.Tensor] = {}
    with torch.no_grad():
        for name in params:
            module = get_module_by_name(fp_model, name)
            if not isinstance(module, nn.Linear):
                raise TypeError(f"{name} is {type(module)} not nn.Linear")
            master[name] = module.weight.detach().clone()

    del qmodel
    torch.cuda.empty_cache()

    records = probe(
        fp_model,
        params,
        master,
        batches,
        parse_h_grid(args.h_grid),
        int(args.probe_dirs),
        int(args.seed),
        out / "probe_records.jsonl",
        bool(args.true_grad),
    )
    rows = summarize(records)
    fields = [
        "h",
        "n_directions",
        "alignment_mean",
        "norm_ratio_mean",
        "delta_visibility_nmse_mean",
        "delta_visibility_nmse_median",
        "delta_visibility_rel_l2_mean",
        "code_change_frac_mean",
        "clip_frac_mean",
        "saturation_frac_mean",
        "richardson_rmse_rel",
        "richardson_relerr_median",
        "d_h_Q_mean",
        "d_half_Q_mean",
        "fd_true_available",
        "fd_true_mse",
        "fd_true_nmse",
        "fd_true_rmse",
        "fd_true_bias",
        "corr_fd_true",
    ]
    write_csv(out / "probe_summary.csv", rows, fields)
    write_summary_md(out / "probe_summary.md", rows, quantize_seconds, recon)
    print(f"[done] wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
