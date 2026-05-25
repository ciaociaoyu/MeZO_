#!/usr/bin/env python
"""Prepare seed-fixed INT4 sparse/prefix RTNClip training lanes."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
TASKS = ["sst-2", "sst-5", "rte", "mnli", "trec"]


def read_rows(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: List[Dict[str, object]], columns: Sequence[str] | None = None) -> None:
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


def h_label_from_value(h: float) -> str:
    known = {
        1e-5: "1e-5",
        1e-3: "1e-3",
        1e-1: "1e-1",
    }
    for value, label in known.items():
        if abs(float(h) - value) <= max(abs(value) * 1e-9, 1e-15):
            return label
    return f"{float(h):.12g}".replace(".", "p").replace("-", "m")


def by_task(rows: Iterable[Dict[str, str]], direction: str | None = None) -> Dict[str, Dict[str, str]]:
    out: Dict[str, Dict[str, str]] = {}
    for row in rows:
        if direction is not None and str(row.get("direction_mode", "")).strip().lower() != direction:
            continue
        out[row.get("task_name") or row.get("dataset")] = row
    return out


def base_row(output_root: Path, task: str, policy: str, h: float, h_label: str, group: str) -> Dict[str, object]:
    return {
        "phase": "int4_hsearch",
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
        "eval_every": 1000,
        "checkpoint_steps": 1000,
        "eval_batch_size": 64,
        "eval_batches": -1,
        "diag_every": 100,
        "quant_log_every": 1000,
        "log_every": 100,
        "update_scalar_source": "finite_difference",
        "update_backend": "fp16_master",
        "quantizer": "G128_groupwise_RTNClip_fake_quant",
        "pair_shared_grid": True,
        "fresh_round_codes": True,
        "grid_source": "unperturbed_fp16_master_weight",
        "seed_mask_fix_applied": True,
        "seed_reset_before_model_load_required": True,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "result_group": group,
        "invalidates_previous_reason": "previous sparse/prefix runs were launched before seed reset and sparse-mask checkpoint restore fixes",
    }


def sparse_row(output_root: Path, task: str, policy: str, h: float, h_label: str, src: Dict[str, str], source_path: Path) -> Dict[str, object]:
    run_name = f"int4_sparsep0p1_taskgrad_seedfixed_{task.replace('-', '')}_{policy}_h{h_label}_seed16_full_bs64_step20k"
    row = base_row(output_root, task, policy, h, h_label, "sparse_p0p1_seedfixed")
    row.update(
        {
            "run_name": run_name,
            "run_dir": str(output_root / "int4_hsearch" / "sparse_p0p1_taskgrad" / policy / run_name),
            "lr": 1e-6,
            "direction_mode": "sparse",
            "sparse_ratio": 0.1,
            "sparse_p": 0.1,
            "sparse_mask_strategy": "task_grad_static",
            "mask_strategy": "task_grad_static",
            "sparse_mask_batches": 1,
            "sparse_mask_scope": "linear_weight",
            "sparse_rescale": "none",
            "sparse_selection": src.get("sparse_selection", "global_topk_grad_square_linear_weight"),
            "sparse_mask_saved_in_checkpoint_required": True,
            "hstar_source": str(source_path),
            "selector_name": src.get("selector_name", ""),
            "hstar_cont": src.get("hstar_cont", ""),
            "hstar_nearest_grid": src.get("hstar_nearest_grid", ""),
            "hstar_used": h if policy.startswith("hstar") else "",
            "hstar_used_policy": "continuous_unsnapped" if policy.startswith("hstar") else "",
            "Delta_mode": src.get("Delta_mode", ""),
            "Delta_value": src.get("Delta_value", ""),
            "G_mode": src.get("G_mode", ""),
            "G_value": src.get("G_value", ""),
            "L_mode": src.get("L_mode", ""),
            "L_hat": src.get("L_hat", ""),
            "h2_L": src.get("h2_L", ""),
            "d_trainable": src.get("d_trainable", ""),
            "mask_active_frac_all": src.get("mask_active_frac_all", ""),
            "active_params_all": src.get("active_params_all", ""),
            "mask_active_frac_quantized_linear": src.get("mask_active_frac_quantized_linear", ""),
            "active_params_quantized_linear": src.get("active_params_quantized_linear", ""),
            "notes": "INT4 sparse p=0.1 task_grad_static; unscaled mask; finite-difference scalar from INT4 quantized forward; sparse masks are checkpointed and restored.",
        }
    )
    return row


def prefix_row(output_root: Path, task: str, policy: str, h: float, h_label: str, src: Dict[str, str], source_path: Path) -> Dict[str, object]:
    run_name = f"int4_prefix_quantized_seedfixed_{task.replace('-', '')}_{policy}_h{h_label}_seed16_full_bs64_step20k"
    row = base_row(output_root, task, policy, h, h_label, "prefix_quantized_seedfixed")
    row.update(
        {
            "run_name": run_name,
            "run_dir": str(output_root / "int4_hsearch" / "prefix_quantized" / policy / run_name),
            "lr": 0.01,
            "direction_mode": "prefix",
            "prefix_num": 5,
            "prefix_precision": "fp16",
            "prefix_init_strategy": "real_act_with_random_fallback",
            "prefix_quantize": True,
            "master_dtype": "fp16",
            "hstar_source": str(source_path),
            "selector_name": src.get("selector_name", ""),
            "hstar_cont": src.get("hstar_cont", ""),
            "hstar_nearest_grid": src.get("hstar_nearest_grid", ""),
            "hstar_used": h if policy.startswith("hstar") else "",
            "hstar_used_policy": "continuous_unsnapped" if policy.startswith("hstar") else "",
            "Delta_mode": src.get("Delta_mode", ""),
            "Delta_value": src.get("Delta_value", ""),
            "delta_int4_rtnclip_scale_rms": src.get("delta_int4_rtnclip_scale_rms", ""),
            "G_mode": src.get("G_mode", ""),
            "G_value": src.get("G_value", ""),
            "L_mode": src.get("L_mode", ""),
            "L_hat": src.get("L_hat", ""),
            "h2_L": src.get("h2_L", ""),
            "d_trainable": src.get("d_trainable", ""),
            "perturbed_parameter_scope": "prefix_parameters_only",
            "quantized_forward_scope": "base_Linear.weight_plus_prefix_params_int4",
            "notes": "INT4 prefix-quantized forward/probe; finite-difference scalar from INT4 quantized forward; prefix init is seed-reset before injection.",
        }
    )
    return row


def write_lane_sbatch(path: Path, output_root: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"""#!/bin/bash
set -euo pipefail

REPO_ROOT="${{REPO_ROOT:-$SLURM_SUBMIT_DIR}}"
OUTPUT_ROOT="${{OUTPUT_ROOT:-{output_root}}}"
CONDA_ENV="${{CONDA_ENV:-mezo-env}}"
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

MANIFEST="$OUTPUT_ROOT/manifests/lane${{LANE_ID}}.csv"
echo "Running seed-fixed INT4 sparse/prefix lane $LANE_ID with manifest $MANIFEST"
nvidia-smi || true
python tools/rtnclip_roberta_sst5_batch.py --output_root "$OUTPUT_ROOT" --manifest "$MANIFEST" run-manifest
""",
        encoding="utf-8",
    )
    path.chmod(0o755)


def write_submit_script(path: Path, output_root: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"""#!/bin/bash
set -euo pipefail

ACCOUNT="${{ACCOUNT:-}}"
PARTITION="${{PARTITION:-}}"
TIME="${{TIME:-48:00:00}}"
GPUS="${{GPUS:-1}}"
GPU_TYPE="${{GPU_TYPE:-H100}}"
CONDA_ENV="${{CONDA_ENV:-mezo-env}}"
OUTPUT_ROOT="${{OUTPUT_ROOT:-{output_root}}}"
REPO_ROOT="${{REPO_ROOT:-$(pwd)}}"

mkdir -p "$OUTPUT_ROOT/slurm_logs" "$OUTPUT_ROOT/jobs"
args=(
  --job-name="int4_sp_pref_fix"
  --nodes=1
  --ntasks=1
  --cpus-per-task=8
  --mem=96G
  --time="$TIME"
  --gres="gpu:${{GPU_TYPE}}:${{GPUS}}"
  --array=0-6
  --output="$OUTPUT_ROOT/slurm_logs/%x_%A_%a.out"
  --export=ALL,OUTPUT_ROOT="$OUTPUT_ROOT",REPO_ROOT="$REPO_ROOT",CONDA_ENV="$CONDA_ENV",GPU_TYPE="$GPU_TYPE"
)
if [[ -n "$ACCOUNT" ]]; then args+=(--account="$ACCOUNT"); fi
if [[ -n "$PARTITION" ]]; then args+=(--partition="$PARTITION"); fi
sbatch "${{args[@]}}" slurm/int4_sparse_prefix_seedfixed_lane.sbatch | tee -a "$OUTPUT_ROOT/jobs/job_ids.txt"
""",
        encoding="utf-8",
    )
    path.chmod(0o755)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default=f"outputs/int4_sparse_prefix_seedfixed_int4fd_20k_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    parser.add_argument("--clean-hstar", default="outputs/int4_cleanGL_hstar_dense_sparsep0p1_20k_20260523_142501/int4_hsearch_manifest.csv")
    parser.add_argument("--lowbit-hstar", default="outputs/int4_lowbitL_hstar_dense_sparse_20260522_20260522_223513/hstar/sparse_p0p1_lowbitL_hstar_summary.csv")
    parser.add_argument("--prefix-hstar", default="outputs/int4_prefix_quantized_cleanGL_20k_20260523_154026/hstar_prefix_int4_quantized_cleanGL.csv")
    args = parser.parse_args()

    output_root = Path(args.output_root)
    clean_path = Path(args.clean_hstar)
    lowbit_path = Path(args.lowbit_hstar)
    prefix_path = Path(args.prefix_hstar)
    clean_sparse = by_task(read_rows(clean_path), "sparse")
    lowbit_sparse = by_task(read_rows(lowbit_path), "sparse")
    prefix_hstar = by_task(read_rows(prefix_path), "prefix")

    lane_specs = [
        ("sparse_fixed_small", "sparse", "fixed_small", 1e-5, "1e-5", clean_sparse, clean_path),
        ("sparse_mezo_default", "sparse", "mezo_default", 1e-3, "1e-3", clean_sparse, clean_path),
        ("sparse_hstar_cleanGL", "sparse", "hstar_cleanGL", None, None, clean_sparse, clean_path),
        ("sparse_hstar_lowbitL", "sparse", "hstar_lowbitL", None, None, lowbit_sparse, lowbit_path),
        ("prefix_fixed_small", "prefix", "fixed_small", 1e-5, "1e-5", prefix_hstar, prefix_path),
        ("prefix_mezo_default", "prefix", "mezo_default", 1e-3, "1e-3", prefix_hstar, prefix_path),
        ("prefix_hstar_cleanGL", "prefix", "hstar_cleanGL", None, None, prefix_hstar, prefix_path),
    ]

    all_rows: List[Dict[str, object]] = []
    lane_table: List[Dict[str, object]] = []
    for lane_id, (lane_name, mode, policy, fixed_h, fixed_label, source, source_path) in enumerate(lane_specs):
        lane_rows: List[Dict[str, object]] = []
        for task in TASKS:
            src = source[task]
            h = float(src["hstar_cont"]) if fixed_h is None else float(fixed_h)
            h_label = h_label_from_value(h) if fixed_label is None else fixed_label
            row = sparse_row(output_root, task, policy, h, h_label, src, source_path) if mode == "sparse" else prefix_row(output_root, task, policy, h, h_label, src, source_path)
            row["lane_id"] = lane_id
            row["lane_name"] = lane_name
            lane_rows.append(row)
        write_csv(output_root / "manifests" / f"lane{lane_id}.csv", lane_rows)
        all_rows.extend(lane_rows)
        lane_table.append({"lane_id": lane_id, "lane_name": lane_name, "mode": mode, "policy": policy, "runs": len(lane_rows)})

    write_csv(output_root / "int4_hsearch_manifest.csv", all_rows)
    write_csv(output_root / "lane_table.csv", lane_table)
    write_json(
        output_root / "batch_manifest.json",
        {
            "output_root": str(output_root),
            "lanes": lane_table,
            "total_runs": len(all_rows),
            "sparse_p": 0.1,
            "bitwidth": 4,
            "update_scalar_source": "finite_difference",
            "int4_quantized_forward": True,
            "seed_fix_required": True,
            "sparse_masks_saved_in_checkpoint": True,
            "previous_invalidated_roots": [
                "outputs/int4_cleanGL_hstar_dense_sparsep0p1_20k_20260523_142501",
                "outputs/int4_prefix_quantized_cleanGL_20k_20260523_154026",
            ],
        },
    )
    (output_root / "README.md").write_text(
        "# INT4 Sparse/Prefix Seed-Fixed Rerun\n\n"
        "Seven Slurm array tasks are generated: four sparse p=0.1 policy lanes and three INT4-quantized prefix policy lanes. "
        "All runs use INT4 G128 RTNClip quantized forward/probe, pair-shared grid, fresh rounding, FP16 master update, and finite-difference scalar from INT4 losses.\n\n"
        "Sparse masks are saved in checkpoint and restored on resume. Previous sparse/prefix outputs were invalidated because they were launched before seed/mask fixes.\n",
        encoding="utf-8",
    )

    write_lane_sbatch(REPO_ROOT / "slurm" / "int4_sparse_prefix_seedfixed_lane.sbatch", output_root)
    write_submit_script(REPO_ROOT / "scripts" / "submit_int4_sparse_prefix_seedfixed_batch.sh", output_root)
    print(output_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
