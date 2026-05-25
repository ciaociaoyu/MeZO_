#!/usr/bin/env python
"""Prepare dense and sparse INT4 full-data h-star training manifests."""

from __future__ import annotations

import argparse
import csv
import math
import os
from pathlib import Path
from typing import Dict, List, Sequence


TASKS = ["sst-2", "sst-5", "rte", "mnli", "trec"]
POLICIES = [
    ("fixed_small", 1e-5, "1e-5"),
    ("mezo_default", 1e-3, "1e-3"),
    ("hstar_ours", None, None),
]


def read_rows(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: List[Dict[str, object]], columns: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def h_label_from_value(h: float) -> str:
    known = {
        1e-5: "1e-5",
        3e-5: "3e-5",
        1e-4: "1e-4",
        3e-4: "3e-4",
        1e-3: "1e-3",
        1.5e-3: "1p5e-3",
        2e-3: "2e-3",
        3e-3: "3e-3",
        4e-3: "4e-3",
        5e-3: "5e-3",
        1e-2: "1e-2",
    }
    for value, label in known.items():
        if abs(float(h) - value) <= max(abs(value) * 1e-9, 1e-15):
            return label
    return f"{h:g}".replace(".", "p")


def hstar_cont_from_row(src: Dict[str, str]) -> float:
    value = src.get("hstar_cont", "")
    h = float(value)
    if not math.isfinite(h) or h <= 0.0:
        raise ValueError(f"Invalid hstar_cont={value!r} for {src.get('task_name')} {src.get('direction_mode')}")
    return h


def policy_rows(hstar_rows: List[Dict[str, str]], output_root: Path, direction_mode: str) -> List[Dict[str, object]]:
    by_key = {(r["task_name"], r["direction_mode"]): r for r in hstar_rows}
    out: List[Dict[str, object]] = []
    for task in TASKS:
        src = by_key[(task, direction_mode)]
        for policy, fixed_h, fixed_label in POLICIES:
            if policy == "hstar_ours":
                h = hstar_cont_from_row(src)
                h_label = h_label_from_value(h)
            else:
                h = float(fixed_h)
                h_label = str(fixed_label)
            prefix = "dense" if direction_mode == "dense" else "sparsep0p1_taskgrad"
            run_name = f"int4_{prefix}_{task.replace('-', '')}_{policy}_h{h_label}_seed16_full_bs64_step20k"
            subdir = "dense" if direction_mode == "dense" else "sparse_p0p1_taskgrad"
            row: Dict[str, object] = {
                "run_name": run_name,
                "run_dir": str(output_root / "int4_hsearch" / subdir / run_name),
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
                "lr": 1e-6,
                "eval_every": 1000,
                "checkpoint_steps": 1000,
                "eval_batch_size": 64,
                "eval_batches": -1,
                "diag_every": 100,
                "quant_log_every": 1000,
                "log_every": 100,
                "direction_mode": direction_mode,
                "sparse_ratio": 0.1 if direction_mode == "sparse" else "",
                "sparse_p": 0.1 if direction_mode == "sparse" else "",
                "sparse_mask_strategy": "task_grad_static" if direction_mode == "sparse" else "",
                "mask_strategy": "task_grad_static" if direction_mode == "sparse" else "",
                "sparse_mask_batches": 1 if direction_mode == "sparse" else "",
                "sparse_mask_scope": "linear_weight" if direction_mode == "sparse" else "",
                "sparse_rescale": "none" if direction_mode == "sparse" else "",
                "hstar_source": str(output_root / "hstar" / "hstar_full_data_summary.csv"),
                "selector_name": src.get("selector_name", "simple2pt_corrected"),
                "hstar_cont": src.get("hstar_cont", ""),
                "hstar_nearest_grid": src.get("hstar_nearest_grid", ""),
                "hstar_used": h if policy == "hstar_ours" else "",
                "hstar_used_policy": "continuous_unsnapped" if policy == "hstar_ours" else "",
                "Delta_mode": src.get("Delta_mode", ""),
                "Delta_value": src.get("Delta_value", ""),
                "G_mode": src.get("G_mode", ""),
                "G_value": src.get("G_value", ""),
                "L_mode": src.get("L_mode", ""),
                "L_hat": src.get("L_hat", ""),
                "h2_L": src.get("h2_L", ""),
                "d_trainable": src.get("d_trainable", ""),
                "mask_active_frac_all": src.get("mask_active_frac_all", ""),
                "mask_active_frac_quantized_linear": src.get("mask_active_frac_quantized_linear", ""),
                "sparse_selection": src.get("sparse_selection", ""),
                "notes": (
                    f"{src.get('notes', '')}; hstar_ours uses continuous hstar_cont without nearest-grid snapping"
                    if policy == "hstar_ours"
                    else src.get("notes", "")
                ),
            }
            out.append(row)
    return out


def columns(rows: List[Dict[str, object]]) -> List[str]:
    keys: List[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    return keys


def split_lanes(rows: List[Dict[str, object]], n_lanes: int) -> List[List[Dict[str, object]]]:
    lanes = [[] for _ in range(n_lanes)]
    for idx, row in enumerate(rows):
        lanes[idx % n_lanes].append(row)
    return lanes


def write_lane_sbatch(path: Path, manifest_prefix: str, output_root: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"""#!/bin/bash
set -euo pipefail

REPO_ROOT="${{REPO_ROOT:-$SLURM_SUBMIT_DIR}}"
OUTPUT_ROOT="${{OUTPUT_ROOT:-{output_root}}}"
CONDA_ENV="${{CONDA_ENV:-ciao}}"
LANE_ID="${{SLURM_ARRAY_TASK_ID:-0}}"

source /home/jy03364/miniconda3/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"
cd "$REPO_ROOT"

export DATALOADER_SHUFFLE=True
export REQUESTED_GPU_TYPE="${{GPU_TYPE:-H100}}"
export FALLBACK_USED="${{FALLBACK_USED:-0}}"

MANIFEST="$OUTPUT_ROOT/manifests/{manifest_prefix}_lane${{LANE_ID}}.csv"
echo "Running lane $LANE_ID with manifest $MANIFEST"
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
CONDA_ENV="${{CONDA_ENV:-ciao}}"
OUTPUT_ROOT="${{OUTPUT_ROOT:-{output_root}}}"
REPO_ROOT="${{REPO_ROOT:-$(pwd)}}"

mkdir -p "$OUTPUT_ROOT/slurm_logs" "$OUTPUT_ROOT/jobs"

submit_group() {{
  local name="$1"
  local script="$2"
  local args=(
    --job-name="$name"
    --nodes=1
    --ntasks=1
    --cpus-per-task=8
    --mem=96G
    --time="$TIME"
    --gres="gpu:${{GPU_TYPE}}:${{GPUS}}"
    --array=0-2
    --output="$OUTPUT_ROOT/slurm_logs/%x_%A_%a.out"
    --export=ALL,OUTPUT_ROOT="$OUTPUT_ROOT",REPO_ROOT="$REPO_ROOT",CONDA_ENV="$CONDA_ENV",GPU_TYPE="$GPU_TYPE"
  )
  if [[ -n "$ACCOUNT" ]]; then args+=(--account="$ACCOUNT"); fi
  if [[ -n "$PARTITION" ]]; then args+=(--partition="$PARTITION"); fi
  sbatch "${{args[@]}}" "$script"
}}

submit_group int4_dense_hstar slurm/int4_full_data_dense_hstar_lane.sbatch | tee -a "$OUTPUT_ROOT/jobs/job_ids.txt"
submit_group int4_sparse_p0p1_hstar slurm/int4_full_data_sparse_p0p1_hstar_lane.sbatch | tee -a "$OUTPUT_ROOT/jobs/job_ids.txt"
""",
        encoding="utf-8",
    )
    path.chmod(0o755)


def write_readme(path: Path, output_root: Path) -> None:
    path.write_text(
        f"""# INT4 full-data h-star dense/sparse batch

Output root: `{output_root}`

This batch runs RoBERTa-large full-data INT4 G128 RTNClip fake-quantized forward with FP16 master update.

Datasets: SST-2, SST-5, RTE, MNLI, TREC.

Policies per dataset:
- `fixed_small`: h=1e-5
- `mezo_default`: h=1e-3
- `hstar_ours`: continuous unsnapped `simple2pt_corrected` `hstar_cont` from `hstar/hstar_full_data_summary.csv`

Sparse setting:
- p=0.1
- mask_strategy=task_grad_static
- sparse_selection=global_topk_grad_square_linear_weight
- sparse_rescale=none
- mask source is the initial unperturbed FP16 master weight

Submit after smoke:

```bash
PARTITION=gpu_p GPU_TYPE=H100 bash scripts/submit_int4_full_data_hstar_batch.sh
```
""",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--hstar_summary", required=True)
    args = parser.parse_args()

    output_root = Path(args.output_root)
    hstar_rows = read_rows(Path(args.hstar_summary))
    dense_rows = policy_rows(hstar_rows, output_root, "dense")
    sparse_rows = policy_rows(hstar_rows, output_root, "sparse")
    all_rows = dense_rows + sparse_rows
    cols = columns(all_rows)

    write_csv(output_root / "manifests" / "dense_manifest.csv", dense_rows, cols)
    write_csv(output_root / "manifests" / "sparse_p0p1_manifest.csv", sparse_rows, cols)
    write_csv(output_root / "int4_hsearch_manifest.csv", all_rows, cols)
    for prefix, rows in (("dense", dense_rows), ("sparse_p0p1", sparse_rows)):
        for idx, lane in enumerate(split_lanes(rows, 3)):
            write_csv(output_root / "manifests" / f"{prefix}_lane{idx}.csv", lane, cols)

    write_lane_sbatch(Path("slurm/int4_full_data_dense_hstar_lane.sbatch"), "dense", output_root)
    write_lane_sbatch(Path("slurm/int4_full_data_sparse_p0p1_hstar_lane.sbatch"), "sparse_p0p1", output_root)
    write_submit_script(Path("scripts/submit_int4_full_data_hstar_batch.sh"), output_root)
    write_readme(output_root / "README.md", output_root)

    commands = [
        "source /home/jy03364/miniconda3/etc/profile.d/conda.sh && conda activate ciao",
        f"CUDA_VISIBLE_DEVICES=0 DATALOADER_SHUFFLE=True python tools/estimate_int4_full_data_hstar.py --output_dir {output_root / 'hstar'} --tasks sst-2 sst-5 rte mnli trec --directions dense sparse --m_g 8 --m_l 4 --sparse_ratio 0.1 --sparse_mask_strategy task_grad_static --sparse_mask_batches 1 --sparse_mask_scope linear_weight",
        f"python scripts/prepare_int4_full_data_hstar_batch.py --output_root {output_root} --hstar_summary {output_root / 'hstar' / 'hstar_full_data_summary.csv'}",
        f"PARTITION=gpu_p GPU_TYPE=H100 bash scripts/submit_int4_full_data_hstar_batch.sh",
    ]
    (output_root / "commands.txt").write_text("\n".join(commands) + "\n", encoding="utf-8")
    print(f"Wrote {len(dense_rows)} dense rows and {len(sparse_rows)} sparse rows under {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
