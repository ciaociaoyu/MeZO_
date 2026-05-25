#!/usr/bin/env python
"""Probe AutoAWQ W4/G128 shared-grid geometry on SST-5 full vs few-shot.

This is probe-only: no master update and no checkpoint training.  For each
dataset mode it runs one official AutoAWQ calibration at step 0, caches the
extracted W4/G128 parameters, and evaluates shared-grid two-point losses and
Richardson self-consistency over a small h grid.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

from official_autoawq_w4_breadth_default_h import (
    EPS,
    AutoAWQSharedGridTrainer,
    append_jsonl,
    compute_loss,
    normalize_json,
    prepare_batch,
    stable_seed,
    write_json,
)


DEFAULT_H_GRID = [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 2e-3, 3e-3, 5e-3, 1e-2]


def parse_h_grid(text: str) -> List[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def write_csv(path: Path, rows: Sequence[Dict[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fields), extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def mean(values: Iterable[Any]) -> Optional[float]:
    xs = [float(x) for x in values if x is not None and math.isfinite(float(x))]
    return float(sum(xs) / len(xs)) if xs else None


def median(values: Iterable[Any]) -> Optional[float]:
    xs = [float(x) for x in values if x is not None and math.isfinite(float(x))]
    return float(np.median(xs)) if xs else None


def make_trainer_args(args: argparse.Namespace, mode: str, calibration_size: int) -> SimpleNamespace:
    return SimpleNamespace(
        output_root=str(args.output_root),
        model_id=str(args.model_id),
        dataset="SST5",
        dataset_mode=str(mode),
        num_k=int(args.num_k),
        num_train=-1,
        num_eval=args.num_eval,
        steps=0,
        start_step=0,
        init_model_path=None,
        k_refresh=500,
        h=1e-3,
        batch_size=int(args.batch_size),
        eval_batch_size=int(args.eval_batch_size),
        eval_max_batches=0,
        max_length=int(args.max_length),
        lr=1e-7,
        seed=int(args.seed),
        data_seed=int(args.data_seed),
        calibration_size=int(calibration_size),
        max_calib_seq_len=int(args.max_calib_seq_len),
        n_parallel_calib_samples=int(args.n_parallel_calib_samples),
        eval_steps=0,
        save_steps=0,
        quant_diag_every=0,
        save_awq_params=True,
        static_awq_once=True,
        static_qparams_path=None,
        run_dir_name=f"{mode}_probe",
        mode_label=f"probe_{mode}",
        smoke=False,
        overwrite=bool(args.overwrite),
    )


def apply_branch_for_h(
    trainer: AutoAWQSharedGridTrainer,
    directions: Dict[str, torch.Tensor],
    *,
    sign: float,
    h: float,
) -> None:
    with torch.no_grad():
        for name, param in trainer.trainable_param_items:
            base = trainer.master[name]
            z = directions[name]
            x = base + float(sign) * float(h) * z
            if name in trainer.quantized_weight_param_names:
                module_name = name[: -len(".weight")]
                q, _, _ = trainer.fake_quant_cached(x, trainer.awq_runtime_cache[module_name])
                param.data.copy_(q.to(param.device, dtype=param.dtype))
            else:
                param.data.copy_(x.to(param.device, dtype=param.dtype))


def quant_geometry_for_h(
    trainer: AutoAWQSharedGridTrainer,
    directions: Dict[str, torch.Tensor],
    h: float,
) -> Dict[str, float]:
    acc = {
        "dot": 0.0,
        "dq_norm_sq": 0.0,
        "ideal_norm_sq": 0.0,
        "err_sq": 0.0,
        "count": 0.0,
        "code_changed": 0.0,
        "code_total": 0.0,
        "clip": 0.0,
        "sat": 0.0,
        "mods": 0.0,
    }
    with torch.no_grad():
        for _module_name, cache in trainer.awq_runtime_cache.items():
            name = cache["weight_name"]
            base = trainer.master[name]
            z = directions[name]
            q_plus, c_plus, st_plus = trainer.fake_quant_cached(base + float(h) * z, cache, need_codes=True)
            q_minus, c_minus, st_minus = trainer.fake_quant_cached(base - float(h) * z, cache, need_codes=True)
            assert c_plus is not None and c_minus is not None and st_plus is not None and st_minus is not None

            dq = (q_plus.float() - q_minus.float()).reshape(-1)
            ideal = (2.0 * float(h) * z.float()).reshape(-1)
            err = dq - ideal
            acc["dot"] += float(torch.dot(dq, ideal).cpu())
            acc["dq_norm_sq"] += float(torch.dot(dq, dq).cpu())
            acc["ideal_norm_sq"] += float(torch.dot(ideal, ideal).cpu())
            acc["err_sq"] += float(torch.dot(err, err).cpu())
            acc["count"] += float(err.numel())
            acc["code_changed"] += float((c_plus != c_minus).sum().cpu())
            acc["code_total"] += float(c_plus.numel())
            acc["clip"] += 0.5 * (float(st_plus["clip_frac"]) + float(st_minus["clip_frac"]))
            acc["sat"] += 0.5 * (float(st_plus["saturation_frac"]) + float(st_minus["saturation_frac"]))
            acc["mods"] += 1.0

    alignment = acc["dot"] / max(math.sqrt(acc["dq_norm_sq"]) * math.sqrt(acc["ideal_norm_sq"]), EPS)
    norm_ratio = math.sqrt(acc["dq_norm_sq"]) / max(math.sqrt(acc["ideal_norm_sq"]), EPS)
    delta_visibility_nmse = acc["err_sq"] / max(acc["ideal_norm_sq"], EPS)
    return {
        "delta_visibility_mse": acc["err_sq"] / max(acc["count"], 1.0),
        "delta_visibility_nmse": delta_visibility_nmse,
        "delta_visibility_rel_l2": math.sqrt(delta_visibility_nmse),
        "alignment": alignment,
        "norm_ratio": norm_ratio,
        "code_change_frac": acc["code_changed"] / max(acc["code_total"], 1.0),
        "active_frac": 1.0,
        "clip_frac": acc["clip"] / max(acc["mods"], 1.0),
        "saturation_frac": acc["sat"] / max(acc["mods"], 1.0),
    }


def fd_for_h(
    trainer: AutoAWQSharedGridTrainer,
    batch: Dict[str, Any],
    directions: Dict[str, torch.Tensor],
    h: float,
) -> Tuple[float, float, float]:
    apply_branch_for_h(trainer, directions, sign=+1.0, h=h)
    loss_plus = compute_loss(trainer.model, batch)
    apply_branch_for_h(trainer, directions, sign=-1.0, h=h)
    loss_minus = compute_loss(trainer.model, batch)
    fd = (float(loss_plus.cpu()) - float(loss_minus.cpu())) / (2.0 * float(h))
    return float(loss_plus.cpu()), float(loss_minus.cpu()), float(fd)


def true_directional_derivative(
    trainer: AutoAWQSharedGridTrainer,
    batch: Dict[str, Any],
    directions: Dict[str, torch.Tensor],
) -> float:
    trainer.save_master_to_model()
    trainer.model.zero_grad(set_to_none=True)
    with torch.enable_grad():
        outputs = trainer.model(**batch, return_dict=True)
        loss = outputs.loss
        loss.backward()
    total = 0.0
    with torch.no_grad():
        for name, param in trainer.trainable_param_items:
            if param.grad is None:
                continue
            contrib = torch.sum(param.grad.detach().float() * directions[name].float())
            total += float(contrib.detach().cpu())
    trainer.model.zero_grad(set_to_none=True)
    return float(total)


def summarize(records: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, float, float], List[Dict[str, Any]]] = defaultdict(list)
    for row in records:
        grouped[(str(row["dataset_mode"]), float(row.get("direction_density", 1.0)), float(row["h"]))].append(row)

    rows: List[Dict[str, Any]] = []
    for (mode, density, h), group in sorted(grouped.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2])):
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
                "dataset_mode": mode,
                "direction_density": density,
                "direction_sparsity": 1.0 - density,
                "h": h,
                "n_records": len(group),
                "n_batches": len({int(r["batch_id"]) for r in group}),
                "n_directions": len({int(r["direction_id"]) for r in group}),
                "train_sample_count": group[0].get("train_sample_count"),
                "calibration_size": group[0].get("calibration_size"),
                "loss_plus_mean": mean(r["loss_plus"] for r in group),
                "loss_minus_mean": mean(r["loss_minus"] for r in group),
                "d_h_Q_mean": mean(r["d_h_Q"] for r in group),
                "d_half_Q_mean": mean(r["d_half_Q"] for r in group),
                "delta_visibility_nmse_mean": mean(r["delta_visibility_nmse"] for r in group),
                "delta_visibility_nmse_median": median(r["delta_visibility_nmse"] for r in group),
                "delta_visibility_rel_l2_mean": mean(r["delta_visibility_rel_l2"] for r in group),
                "alignment_mean": mean(r["alignment"] for r in group),
                "norm_ratio_mean": mean(r["norm_ratio"] for r in group),
                "code_change_frac_mean": mean(r["code_change_frac"] for r in group),
                "clip_frac_mean": mean(r["clip_frac"] for r in group),
                "saturation_frac_mean": mean(r["saturation_frac"] for r in group),
                "richardson_absdiff_mean": mean(abs(float(r["d_h_Q"]) - float(r["d_half_Q"])) for r in group),
                "richardson_rmse_rel": math.sqrt(diff_sq / max(half_sq, EPS)),
                "richardson_relerr_median": median(r["richardson_relerr_per_direction"] for r in group),
                "fd_true_available": bool(fd_true_pairs),
                "fd_true_mse": fd_true_mse,
                "fd_true_nmse": fd_true_nmse,
                "fd_true_rmse": fd_true_rmse,
                "corr_fd_true": corr_fd_true,
                "fd_true_bias": fd_true_bias,
            }
        )
    return rows


def write_markdown(output_root: Path, summary_rows: Sequence[Dict[str, Any]], args: argparse.Namespace) -> None:
    lines = [
        "# AutoAWQ W4 SST-5 Probe: Full vs Few-shot",
        "",
        "Probe-only run. No training updates were applied.",
        "",
        f"- h grid: `{', '.join(f'{h:g}' for h in parse_h_grid(args.h_grid))}`",
        f"- batches per mode: `{int(args.probe_batches)}`",
        f"- directions per batch: `{int(args.probe_directions)}`",
        f"- true-gradient diagnostics: `{'enabled' if args.true_grad else 'disabled'}`",
        "- locality proxy: Richardson `d_h` vs `d_{h/2}` on the same batch and direction",
        "",
        "| mode | density | h | align | norm_ratio | delta_nmse | richardson_rmse_rel | fd_true_nmse | corr_fd_true | code_change |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            "| {dataset_mode} | {direction_density:.3g} | {h:.1e} | {alignment_mean:.4g} | {norm_ratio_mean:.4g} | "
            "{delta_visibility_nmse_mean:.4g} | {richardson_rmse_rel:.4g} | {fd_true_nmse} | {corr_fd_true} | "
            "{code_change_frac_mean:.4g} |".format(
                **{
                    **row,
                    "fd_true_nmse": "NA" if row.get("fd_true_nmse") is None else f"{row['fd_true_nmse']:.4g}",
                    "corr_fd_true": "NA" if row.get("corr_fd_true") is None else f"{row['corr_fd_true']:.4g}",
                }
            )
        )

    by_mode = defaultdict(dict)
    for row in summary_rows:
        by_mode[(row["dataset_mode"], float(row.get("direction_density", 1.0)))][float(row["h"])] = row
    lines.extend(["", "## 1e-3 Check", ""])
    for mode, density in sorted(by_mode):
        row = by_mode[(mode, density)].get(1e-3)
        if not row:
            continue
        lines.append(
            f"- `{mode}` density={density:g} at h=1e-3: alignment={row['alignment_mean']:.4g}, "
            f"norm_ratio={row['norm_ratio_mean']:.4g}, "
            f"delta_visibility_nmse={row['delta_visibility_nmse_mean']:.4g}, "
            f"richardson_rmse_rel={row['richardson_rmse_rel']:.4g}, "
            f"fd_true_nmse={row.get('fd_true_nmse')}."
        )
    (output_root / "probe_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_direction_densities(text: str) -> List[float]:
    densities = [float(x.strip()) for x in text.split(",") if x.strip()]
    if not densities:
        raise ValueError("at least one direction density is required")
    for density in densities:
        if density <= 0.0 or density > 1.0:
            raise ValueError(f"direction density must be in (0, 1], got {density}")
    return densities


def materialize_sparse_directions(
    trainer: AutoAWQSharedGridTrainer,
    step_seed: int,
    density: float,
) -> Tuple[Dict[str, torch.Tensor], float]:
    nonzero = 0
    total = 0
    directions: Dict[str, torch.Tensor] = {}
    for name, _param in trainer.trainable_param_items:
        z = trainer.master[name].new_empty(trainer.master[name].shape)
        generator = torch.Generator(device=z.device)
        generator.manual_seed(stable_seed(trainer.args.seed, step_seed, name))
        z.normal_(0.0, 1.0, generator=generator)
        if density < 1.0:
            mask_gen = torch.Generator(device=z.device)
            mask_gen.manual_seed(stable_seed(trainer.args.seed, "mask", step_seed, name, density))
            mask = torch.rand(z.shape, device=z.device, generator=mask_gen) < float(density)
            z.mul_(mask.to(dtype=z.dtype))
            nonzero += int(mask.sum().detach().cpu())
            total += int(mask.numel())
            del mask
        else:
            nonzero += int(z.numel())
            total += int(z.numel())
        directions[name] = z
    actual_density = float(nonzero) / float(max(total, 1))
    return directions, actual_density


def run_mode(args: argparse.Namespace, mode: str, calibration_size: int) -> List[Dict[str, Any]]:
    trainer_args = make_trainer_args(args, mode, calibration_size)
    trainer = AutoAWQSharedGridTrainer(trainer_args)
    trainer.write_config()
    trainer.load()
    recon_batch = prepare_batch(next(trainer.train_iter), trainer.device)
    trainer.refresh_awq(0, recon_batch)

    probe_batches = [prepare_batch(next(trainer.train_iter), trainer.device) for _ in range(int(args.probe_batches))]
    h_grid = parse_h_grid(args.h_grid)
    records: List[Dict[str, Any]] = []
    for density in parse_direction_densities(args.direction_densities):
        for direction_id in range(int(args.probe_directions)):
            direction_seed = stable_seed(args.seed, "autoawq_probe", mode, density, direction_id)
            directions, actual_density = materialize_sparse_directions(trainer, direction_seed, density)
            d_true_by_batch: Dict[int, Optional[float]] = {}
            true_grad_error: Optional[str] = None
            if bool(args.true_grad):
                for batch_id, batch in enumerate(probe_batches):
                    try:
                        d_true_by_batch[batch_id] = true_directional_derivative(trainer, batch, directions)
                    except torch.cuda.OutOfMemoryError as exc:
                        true_grad_error = f"OutOfMemoryError: {exc}"
                        d_true_by_batch[batch_id] = None
                        torch.cuda.empty_cache()
                    except Exception as exc:  # Preserve probe records rather than fabricating true-gradient values.
                        true_grad_error = f"{type(exc).__name__}: {exc}"
                        d_true_by_batch[batch_id] = None
            for h in h_grid:
                geom = quant_geometry_for_h(trainer, directions, h)
                for batch_id, batch in enumerate(probe_batches):
                    loss_plus, loss_minus, d_h = fd_for_h(trainer, batch, directions, h)
                    half_plus, half_minus, d_half = fd_for_h(trainer, batch, directions, h / 2.0)
                    absdiff = abs(d_h - d_half)
                    relerr = absdiff / max(abs(d_half), EPS)
                    d_true = d_true_by_batch.get(batch_id)
                    row = {
                        "dataset": "SST5",
                        "dataset_mode": mode,
                        "train_sample_count": len(trainer.train_samples),
                        "eval_sample_count": len(trainer.eval_samples),
                        "calibration_size": len(trainer.calib_texts),
                        "h": h,
                        "batch_id": batch_id,
                        "direction_id": direction_id,
                        "direction_seed": direction_seed,
                        "direction_density": float(density),
                        "actual_direction_density": float(actual_density),
                        "direction_sparsity": 1.0 - float(density),
                        "direction_rescaled": False,
                        "quantizer": "official_autoawq_param_shared_grid_fake_quant",
                        "quant_bits": 4,
                        "group_size": 128,
                        "pair_shared_grid": True,
                        "fresh_round_codes": True,
                        "qparam_id": int(trainer.current_qparam_id),
                        "loss_plus": loss_plus,
                        "loss_minus": loss_minus,
                        "d_h_Q": d_h,
                        "half_loss_plus": half_plus,
                        "half_loss_minus": half_minus,
                        "d_half_Q": d_half,
                        "richardson_absdiff": absdiff,
                        "richardson_relerr_per_direction": relerr,
                        "fd_true_available": d_true is not None,
                        "d_true": d_true,
                        "fd_true_error": None if d_true is None else d_h - float(d_true),
                        "fd_true_exception_reason": true_grad_error if d_true is None and bool(args.true_grad) else None,
                        "fd_true_mse": None,
                        "fd_true_nmse": None,
                        "corr_fd_true": None,
                        **geom,
                    }
                    append_jsonl(Path(args.output_root) / "probe_records.jsonl", row)
                    records.append(row)
                print(
                    f"[probe] mode={mode} density={density:g} actual={actual_density:.4g} "
                    f"dir={direction_id} h={h:g} align={geom['alignment']:.4g} "
                    f"norm={geom['norm_ratio']:.4g} rich_last={records[-1]['richardson_relerr_per_direction']:.4g}",
                    flush=True,
                )
            del directions
            torch.cuda.empty_cache()
    return records


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_root", type=Path, default=Path("outputs/official_autoawq_w4_sst5_probe_full_vs_fewshot"))
    parser.add_argument("--model_id", default="facebook/opt-1.3b")
    parser.add_argument("--seed", type=int, default=16)
    parser.add_argument("--data_seed", type=int, default=16)
    parser.add_argument("--num_k", type=int, default=16)
    parser.add_argument("--num_eval", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--max_calib_seq_len", type=int, default=128)
    parser.add_argument("--n_parallel_calib_samples", type=int, default=8)
    parser.add_argument("--full_calibration_size", type=int, default=128)
    parser.add_argument("--fewshot_calibration_size", type=int, default=80)
    parser.add_argument("--probe_batches", type=int, default=2)
    parser.add_argument("--probe_directions", type=int, default=4)
    parser.add_argument("--h_grid", default=",".join(f"{h:g}" for h in DEFAULT_H_GRID))
    parser.add_argument("--modes", default="fewshot,full")
    parser.add_argument("--direction_densities", default="1.0")
    parser.add_argument("--true_grad", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)
    if bool(args.overwrite):
        for name in ("probe_records.jsonl", "probe_summary.csv", "probe_summary.md", "probe_run_config.json"):
            path = args.output_root / name
            if path.exists():
                path.unlink()
    write_json(
        args.output_root / "probe_run_config.json",
        {
            "model": args.model_id,
            "dataset": "SST5",
            "modes": [x.strip() for x in args.modes.split(",") if x.strip()],
            "seed": args.seed,
            "data_seed": args.data_seed,
            "num_k": args.num_k,
            "h_grid": parse_h_grid(args.h_grid),
            "probe_batches": args.probe_batches,
            "probe_directions": args.probe_directions,
            "direction_densities": parse_direction_densities(args.direction_densities),
            "direction_rescaled": False,
            "true_grad": bool(args.true_grad),
            "quantizer": "official_autoawq_param_shared_grid_fake_quant",
            "quant_bits": 4,
            "group_size": 128,
            "no_training_updates": True,
        },
    )

    start = time.time()
    all_records: List[Dict[str, Any]] = []
    for mode in [x.strip() for x in args.modes.split(",") if x.strip()]:
        if mode not in {"fewshot", "full"}:
            raise ValueError(f"unsupported mode {mode}; expected fewshot or full")
        calibration_size = int(args.fewshot_calibration_size if mode == "fewshot" else args.full_calibration_size)
        all_records.extend(run_mode(args, mode, calibration_size))
    summary_rows = summarize(all_records)
    fields = list(summary_rows[0].keys()) if summary_rows else []
    write_csv(args.output_root / "probe_summary.csv", summary_rows, fields)
    write_markdown(args.output_root, summary_rows, args)
    write_json(
        args.output_root / "probe_completion_summary.json",
        {
            "status": "completed",
            "wallclock_sec": time.time() - start,
            "n_records": len(all_records),
            "summary_csv": str(args.output_root / "probe_summary.csv"),
        },
    )
    print(json.dumps(normalize_json({"status": "completed", "records": len(all_records), "output_root": str(args.output_root)}), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
