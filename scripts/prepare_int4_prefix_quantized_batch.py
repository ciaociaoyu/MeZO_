#!/usr/bin/env python
"""Prepare INT4-quantized Prefix-MeZO full-data training batch."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, List, Optional, Sequence

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = REPO_ROOT / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import smoke_rtnclip_roberta_sst5 as smoke  # noqa: E402
from analyze_int4_sst5_calibrated_hstar import choose_l_plateau, simple2pt_corrected, weighted_int4_delta  # noqa: E402
from rtnclip_roberta_sst5_batch import inject_prefix_for_training, reset_run_seed  # noqa: E402


TASKS = ["sst-2", "sst-5", "rte", "mnli", "trec"]
POLICIES = [
    ("fixed_small", 1e-5, "1e-5"),
    ("mezo_default", 1e-3, "1e-3"),
    ("prefix_h1e-1", 1e-1, "1e-1"),
    ("hstar_cleanGL", None, None),
]
DEFAULT_CLEAN_SOURCE = (
    REPO_ROOT
    / "outputs/int4_prefix_mezo32_full_data_20k_20260523_062851/hstar_prefix_mezo32"
)


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: List[Dict[str, object]], columns: Optional[Sequence[str]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if columns is None:
        keys: List[str] = []
        for row in rows:
            for key in row:
                if key not in keys:
                    keys.append(key)
        columns = keys
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT), text=True).strip()
    except Exception:
        return ""


def safe_float(value: object) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def median(values: Iterable[float]) -> float:
    vals = sorted(float(v) for v in values if math.isfinite(float(v)))
    if not vals:
        return float("nan")
    return vals[len(vals) // 2]


def h_label_from_value(h: float) -> str:
    known = {1e-5: "1e-5", 3e-5: "3e-5", 1e-4: "1e-4", 3e-4: "3e-4", 1e-3: "1e-3", 1e-2: "1e-2", 1e-1: "1e-1"}
    for value, label in known.items():
        if abs(float(h) - value) <= max(abs(value) * 1e-9, 1e-15):
            return label
    return f"{h:.10g}".replace(".", "p").replace("-", "m")


def load_prefix_int4_delta(task: str, *, prefix_precision: str, seed: int, data_seed: int) -> Dict[str, object]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to compute prefix RTNClip scale statistics.")
    device = torch.device("cuda")
    reset_run_seed(int(seed))
    model, _, _, data_args, sampler = smoke.load_prompt_model_and_data(
        SimpleNamespace(
            repo_root=REPO_ROOT,
            model_id="roberta-large",
            task_name=task,
            seed=int(seed),
            data_seed=int(data_seed),
            batch_size=64,
            eval_batch_size=64,
            dataset_mode="full",
            data_dir=None,
            num_k=16,
        ),
        device,
    )
    reset_run_seed(int(seed))
    prefix_names, prefix_status = inject_prefix_for_training(
        model,
        5,
        prefix_precision=prefix_precision,
        init_strategy="real_act_with_random_fallback",
    )
    params = smoke.named_parameter_map(model)
    master = {name: params[name].detach().clone().to(device=device, dtype=torch.float16) for name in prefix_names}
    states: Dict[str, smoke.RTNClipState] = {}
    refresh_rows: List[Dict[str, object]] = []
    for name in prefix_names:
        state, stats = smoke.compute_rtnclip_state(name, master[name], 4, 128)
        states[name] = state
        refresh_rows.append(stats)
    delta_stats = weighted_int4_delta(states)
    numel_by_name = {name: int(master[name].numel()) for name in prefix_names}
    quant_stats = smoke.aggregate_quantizer_stats(refresh_rows, numel_by_name)
    return {
        **delta_stats,
        **{f"prefix_{k}": v for k, v in quant_stats.items()},
        "prefix_status": prefix_status,
        "prefix_param_names": ";".join(prefix_names),
        "prefix_param_count": sum(int(t.numel()) for t in master.values()),
        "prefix_tensor_count": len(prefix_names),
        "sampler_name_for_delta": type(sampler).__name__,
        "data_dir_resolved_for_delta": getattr(data_args, "data_dir", ""),
    }


def recompute_hstar(clean_source: Path, output_root: Path, *, prefix_precision: str, seed: int, data_seed: int) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for task in TASKS:
        task_dir = clean_source / f"{task}_prefix"
        if not task_dir.exists():
            raise FileNotFoundError(f"Missing clean prefix source directory: {task_dir}")
        g_rows = read_csv(task_dir / "clean32_G_candidates.csv")
        l_rows = read_csv(task_dir / "L_candidates.csv")
        clean_g_vals = [
            safe_float(r.get("G_clean32_abs"))
            for r in g_rows
            if safe_float(r.get("h")) in {1e-4, 3e-4, 1e-3}
        ]
        clean_g = median(clean_g_vals)
        clean_g_h3e4 = next((safe_float(r.get("G_clean32_abs")) for r in g_rows if abs(safe_float(r.get("h")) - 3e-4) < 1e-18), clean_g)
        l_selected, l_status = choose_l_plateau(l_rows)
        l_hat = safe_float(l_selected.get("lambda_q90"))
        delta_stats = load_prefix_int4_delta(task, prefix_precision=prefix_precision, seed=seed, data_seed=data_seed)
        d_trainable = int(delta_stats["prefix_param_count"])
        corrected = simple2pt_corrected(
            "int4_prefix_quantized",
            d_trainable,
            l_hat,
            scale_rms=float(delta_stats["delta_int4_rtnclip_scale_rms"]),
            clean32_g_median=clean_g,
            clean32_g_h3e4=clean_g_h3e4,
            selected_g=clean_g,
            selected_g_mode="clean32_absG_median_1e-4_3e-4_1e-3",
        )
        row: Dict[str, object] = {
            "dataset": task,
            "task_name": task,
            "model": "roberta-large",
            "dataset_mode": "full",
            "seed": seed,
            "data_seed": data_seed,
            "batch_size": 64,
            "direction_mode": "prefix",
            "precision": "int4",
            "bitwidth": 4,
            "group_size": 128,
            "quantizer": "G128_groupwise_RTNClip_fake_quant",
            "master_dtype": "fp16",
            "prefix_precision": prefix_precision,
            "prefix_quantize": True,
            "prefix_num": 5,
            "prefix_reparam": False,
            "prefix_init_by_real_act": True,
            "selector_name": "simple2pt_corrected_cleanGL_prefix_int4_scale",
            "Delta_mode": "prefix_int4_rtnclip_scale_rms_over_sqrt6",
            "Delta_value": corrected["Delta_value"],
            "delta_int4_rtnclip_scale_rms": delta_stats["delta_int4_rtnclip_scale_rms"],
            "delta_int4_rtnclip_scale_mean": delta_stats["delta_int4_rtnclip_scale_mean"],
            "G_mode": "clean32_absG_median_1e-4_3e-4_1e-3",
            "G_value": clean_g,
            "G_clean32_abs_h3e-4": clean_g_h3e4,
            "L_mode": "L_clean32",
            "L_q": "q90",
            "L_hat": l_hat,
            "h2_L": l_selected.get("h2", ""),
            "L_selection_status": l_status,
            "d_trainable": d_trainable,
            "hstar_cont": corrected["hstar_cont"],
            "hstar_nearest_grid": corrected["hstar_nearest_grid"],
            "hstar_nearest_grid_label": h_label_from_value(float(corrected["hstar_nearest_grid"])),
            "clean_source": str(clean_source),
            "notes": (
                "INT4 prefix-quantized selector: frozen base Linear.weight and prefix tensors use RTNClip forward; "
                "h-star uses prefix INT4 scale RMS Delta with clean FP32 prefix G/L."
            ),
            **delta_stats,
        }
        rows.append(row)

    write_csv(output_root / "hstar_prefix_int4_quantized_cleanGL.csv", rows)
    md = [
        "# INT4-Quantized Prefix Clean-GL h-star",
        "",
        "Prefix tensors are included in G128 RTNClip scale statistics. G and L are clean FP32 prefix-only estimates from the existing source.",
        "",
        "| dataset | hstar_cont | nearest | Delta | scale_rms | G clean | L clean q90 | d |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in rows:
        md.append(
            f"| {r['task_name']} | {float(r['hstar_cont']):.6g} | {r['hstar_nearest_grid_label']} | "
            f"{float(r['Delta_value']):.6g} | {float(r['delta_int4_rtnclip_scale_rms']):.6g} | "
            f"{float(r['G_value']):.6g} | {float(r['L_hat']):.6g} | {r['d_trainable']} |"
        )
    (output_root / "hstar_prefix_int4_quantized_cleanGL.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return rows


def split_lanes(rows: List[Dict[str, object]], n_lanes: int) -> List[List[Dict[str, object]]]:
    lanes = [[] for _ in range(n_lanes)]
    for idx, row in enumerate(rows):
        lanes[idx % n_lanes].append(row)
    return lanes


def build_manifest(hstar_rows: List[Dict[str, object]], output_root: Path, lanes: int, *, lr: float) -> List[Dict[str, object]]:
    by_task = {str(r["task_name"]): r for r in hstar_rows}
    rows: List[Dict[str, object]] = []
    for task in TASKS:
        src = by_task[task]
        for policy, fixed_h, fixed_label in POLICIES:
            if policy == "hstar_cleanGL":
                h = float(src["hstar_cont"])
                h_label = h_label_from_value(h)
            else:
                h = float(fixed_h)
                h_label = str(fixed_label)
            run_name = f"int4_prefix_quantized_{task.replace('-', '')}_{policy}_h{h_label}_seed16_full_bs64_step20k"
            rows.append(
                {
                    "run_name": run_name,
                    "run_dir": str(output_root / "int4_hsearch" / "prefix_quantized" / run_name),
                    "phase": "int4_prefix_quantized_full_data_cleanGL_20k",
                    "task_name": task,
                    "dataset": task,
                    "dataset_mode": "full",
                    "data_dir": "",
                    "num_k": 16,
                    "seed": 16,
                    "data_seed": 16,
                    "batch_size": 64,
                    "bitwidth": 4,
                    "h": h,
                    "h_label": h_label,
                    "h_policy": policy,
                    "max_steps": 20000,
                    "scale_refresh_k": 1,
                    "lr": lr,
                    "eval_every": 1000,
                    "checkpoint_steps": 1000,
                    "eval_batch_size": 64,
                    "eval_batches": -1,
                    "diag_every": 100,
                    "quant_log_every": 1000,
                    "log_every": 100,
                    "direction_mode": "prefix",
                    "prefix_num": 5,
                    "prefix_precision": "fp16",
                    "prefix_init_strategy": "real_act_with_random_fallback",
                    "prefix_quantize": True,
                    "master_dtype": "fp16",
                    "update_backend": "fp16_master",
                    "hstar_source": str(output_root / "hstar_prefix_int4_quantized_cleanGL.csv"),
                    "selector_name": src["selector_name"] if policy == "hstar_cleanGL" else "",
                    "hstar_cont": src["hstar_cont"],
                    "hstar_nearest_grid": src["hstar_nearest_grid"],
                    "hstar_used": h if policy == "hstar_cleanGL" else "",
                    "hstar_used_policy": "continuous_unsnapped_cleanGL_prefix_int4_scale" if policy == "hstar_cleanGL" else "",
                    "Delta_mode": src["Delta_mode"],
                    "Delta_value": src["Delta_value"],
                    "delta_int4_rtnclip_scale_rms": src["delta_int4_rtnclip_scale_rms"],
                    "G_mode": src["G_mode"],
                    "G_value": src["G_value"],
                    "L_mode": src["L_mode"],
                    "L_hat": src["L_hat"],
                    "h2_L": src["h2_L"],
                    "d_trainable": src["d_trainable"],
                    "perturbed_parameter_scope": "prefix_parameters_only",
                    "quantized_forward_scope": "base_Linear.weight_plus_prefix_params_int4",
                    "notes": "Frozen base; prefix tensors are quantized in forward/probe with shared-grid fresh-round INT4; hstar_cleanGL is continuous.",
                }
            )
    cols = columns(rows)
    write_csv(output_root / "manifests" / "prefix_quantized_manifest.csv", rows, cols)
    write_csv(output_root / "int4_hsearch_manifest.csv", rows, cols)
    for idx, lane in enumerate(split_lanes(rows, lanes)):
        write_csv(output_root / "manifests" / f"prefix_quantized_lane{idx}.csv", lane, cols)
    return rows


def columns(rows: List[Dict[str, object]]) -> List[str]:
    keys: List[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    return keys


def write_lane_sbatch(path: Path, output_root: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"""#!/bin/bash
set -euo pipefail

REPO_ROOT="${{REPO_ROOT:-$SLURM_SUBMIT_DIR}}"
OUTPUT_ROOT="${{OUTPUT_ROOT:-{output_root}}}"
CONDA_ENV="${{CONDA_ENV:-ciao}}"
LANE_ID="${{SLURM_ARRAY_TASK_ID:-0}}"

if [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [[ -f "$HOME/.conda/etc/profile.d/conda.sh" ]]; then
  source "$HOME/.conda/etc/profile.d/conda.sh"
fi
conda activate "$CONDA_ENV"
cd "$REPO_ROOT"

export DATALOADER_SHUFFLE=True
export REQUESTED_GPU_TYPE="${{GPU_TYPE:-H100}}"
export FALLBACK_USED="${{FALLBACK_USED:-0}}"

MANIFEST="$OUTPUT_ROOT/manifests/prefix_quantized_lane${{LANE_ID}}.csv"
echo "Running INT4-quantized prefix lane $LANE_ID with manifest $MANIFEST"
nvidia-smi || true
python tools/rtnclip_roberta_sst5_batch.py --output_root "$OUTPUT_ROOT" --manifest "$MANIFEST" run-manifest
""",
        encoding="utf-8",
    )
    path.chmod(0o755)


def write_submit_script(path: Path, output_root: Path, lanes: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"""#!/bin/bash
set -euo pipefail

ACCOUNT="${{ACCOUNT:-}}"
PARTITION="${{PARTITION:-}}"
TIME="${{TIME:-48:00:00}}"
GPUS="${{GPUS:-1}}"
GPU_TYPE="${{GPU_TYPE:-H100}}"
CONDA_ENV="${{CONDA_ENV:-ciao}}"
OUTPUT_ROOT="${{OUTPUT_ROOT:-{output_root}}}"
REPO_ROOT="${{REPO_ROOT:-$(pwd)}}"

mkdir -p "$OUTPUT_ROOT/slurm_logs" "$OUTPUT_ROOT/jobs"
args=(
  --job-name=int4_prefix_q
  --nodes=1
  --ntasks=1
  --cpus-per-task=8
  --mem=96G
  --time="$TIME"
  --gres="gpu:${{GPU_TYPE}}:${{GPUS}}"
  --array=0-{lanes - 1}
  --output="$OUTPUT_ROOT/slurm_logs/%x_%A_%a.out"
  --error="$OUTPUT_ROOT/slurm_logs/%x_%A_%a.err"
  --export=ALL,OUTPUT_ROOT="$OUTPUT_ROOT",REPO_ROOT="$REPO_ROOT",CONDA_ENV="$CONDA_ENV",GPU_TYPE="$GPU_TYPE"
)
if [[ -n "$ACCOUNT" ]]; then args+=(--account="$ACCOUNT"); fi
if [[ -n "$PARTITION" ]]; then args+=(--partition="$PARTITION"); fi
sbatch "${{args[@]}}" slurm/int4_prefix_quantized_lane.sbatch | tee -a "$OUTPUT_ROOT/jobs/job_ids.txt"
""",
        encoding="utf-8",
    )
    path.chmod(0o755)


def write_readme(path: Path, output_root: Path, clean_source: Path, lr: float) -> None:
    path.write_text(
        f"""# INT4-Quantized Prefix Full-Data Clean-GL Batch

Output root: `{output_root}`

This replaces the invalid hybrid prefix setting for the new run:

- frozen RoBERTa-large base
- base Linear.weight forward uses G128 RTNClip INT4 fake quantization
- prefix_keys/prefix_values also use G128 RTNClip INT4 fake quantization
- finite-difference probes evaluate `Q_t(prefix +/- h u)` with shared grid and fresh rounded codes
- ZO update direction is prefix only
- prefix master/update dtype is FP16, not FP32
- no GPTQ, no sparse, no residual-grid, no direct INT update

Datasets: SST-2, SST-5, RTE, MNLI, TREC.

H policies:

- `fixed_small`: h=1e-5
- `mezo_default`: h=1e-3
- `prefix_h1e-1`: h=0.1
- `hstar_cleanGL`: continuous h-star from prefix INT4 scale Delta with clean FP32 prefix G/L

LR: `{lr}`.
Clean G/L source: `{clean_source}`.

Submit:

```bash
PARTITION=gpu_p GPU_TYPE=H100 bash scripts/submit_int4_prefix_quantized_batch.sh
```
""",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--clean_probe_source", default=str(DEFAULT_CLEAN_SOURCE))
    parser.add_argument("--lanes", type=int, default=4)
    parser.add_argument("--prefix_precision", choices=["fp16"], default="fp16")
    parser.add_argument("--lr", type=float, default=1e-2)
    args = parser.parse_args()

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    clean_source = Path(args.clean_probe_source)
    lanes = int(args.lanes)
    hstar_rows = recompute_hstar(clean_source, output_root, prefix_precision=args.prefix_precision, seed=16, data_seed=16)
    manifest_rows = build_manifest(hstar_rows, output_root, lanes, lr=float(args.lr))
    write_lane_sbatch(Path("slurm/int4_prefix_quantized_lane.sbatch"), output_root)
    write_submit_script(Path("scripts/submit_int4_prefix_quantized_batch.sh"), output_root, lanes)
    write_readme(output_root / "README.md", output_root, clean_source, float(args.lr))

    config = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "output_root": str(output_root),
        "clean_probe_source": str(clean_source),
        "lanes": lanes,
        "num_runs": len(manifest_rows),
        "training_path": "tools/rtnclip_roberta_sst5_batch.py prefix_quantize=True",
        "prefix_precision": args.prefix_precision,
        "lr": float(args.lr),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
    }
    write_json(output_root / "config_manifest.json", config)
    commands = [
        f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '')} python scripts/prepare_int4_prefix_quantized_batch.py --output_root {output_root} --clean_probe_source {clean_source} --lanes {lanes} --lr {float(args.lr):.12g}",
        "PARTITION=gpu_p GPU_TYPE=H100 bash scripts/submit_int4_prefix_quantized_batch.sh",
    ]
    (output_root / "commands.txt").write_text("\n".join(commands) + "\n", encoding="utf-8")
    print(f"Wrote {len(manifest_rows)} INT4-quantized prefix runs under {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
