#!/usr/bin/env python
"""Prepare INT4 sparse p=0.01 and prefix full-data training batch."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, List, Sequence


TASKS = ["sst-2", "sst-5", "rte", "mnli", "trec"]
POLICIES = [
    ("fixed_small", 1e-5, "1e-5"),
    ("mezo_default", 1e-3, "1e-3"),
    ("hstar_ours", None, None),
]
PREFIX_HSTAR = 1e-1
PREFIX_HSTAR_LABEL = "1e-1"


def read_rows(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: List[Dict[str, object]], columns: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


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


def h_label_from_value(h: float) -> str:
    known = {
        1e-5: "1e-5",
        1e-3: "1e-3",
        1e-1: "1e-1",
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


def sparse001_rows(hstar_rows: List[Dict[str, str]], output_root: Path, hstar_source: Path) -> List[Dict[str, object]]:
    by_task = {r["task_name"]: r for r in hstar_rows if r.get("direction_mode") == "sparse"}
    out: List[Dict[str, object]] = []
    for task in TASKS:
        src = by_task[task]
        for policy, fixed_h, fixed_label in POLICIES:
            if policy == "hstar_ours":
                h = hstar_cont_from_row(src)
                h_label = h_label_from_value(h)
            else:
                h = float(fixed_h)
                h_label = str(fixed_label)
            run_name = f"int4_sparsep0p01_taskgrad_{task.replace('-', '')}_{policy}_h{h_label}_seed16_full_bs64_step20k"
            out.append({
                "run_name": run_name,
                "run_dir": str(output_root / "int4_hsearch" / "sparse_p0p01_taskgrad" / run_name),
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
                "direction_mode": "sparse",
                "sparse_ratio": 0.01,
                "sparse_p": 0.01,
                "sparse_mask_strategy": "task_grad_static",
                "mask_strategy": "task_grad_static",
                "sparse_mask_batches": 1,
                "sparse_mask_scope": "linear_weight",
                "sparse_rescale": "none",
                "hstar_source": str(hstar_source),
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
                    "sparse p=0.01 task_grad_static unscaled; default nMSE target is dh_vs_gTu; "
                    "hstar_ours uses continuous hstar_cont without nearest-grid snapping"
                    if policy == "hstar_ours"
                    else "sparse p=0.01 task_grad_static unscaled; default nMSE target is dh_vs_gTu"
                ),
            })
    return out


def prefix_rows(output_root: Path, prefix_probe_source: str) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    for task in TASKS:
        for policy, fixed_h, fixed_label in POLICIES:
            if policy == "hstar_ours":
                h = PREFIX_HSTAR
                h_label = PREFIX_HSTAR_LABEL
            else:
                h = float(fixed_h)
                h_label = str(fixed_label)
            run_name = f"int4_prefix_{task.replace('-', '')}_{policy}_h{h_label}_seed16_full_bs64_step20k"
            out.append({
                "run_name": run_name,
                "run_dir": str(output_root / "int4_hsearch" / "prefix" / run_name),
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
                "direction_mode": "prefix",
                "sparse_ratio": "",
                "sparse_p": "",
                "sparse_mask_strategy": "",
                "mask_strategy": "",
                "sparse_rescale": "",
                "prefix_num": 5,
                "hstar_source": prefix_probe_source,
                "selector_name": "prefix_probe_selected",
                "hstar_cont": PREFIX_HSTAR,
                "hstar_nearest_grid": PREFIX_HSTAR,
                "Delta_mode": "not_applicable_prefix_fp16_params",
                "Delta_value": "",
                "G_mode": "prefix_probe_default_dh_vs_gTu",
                "G_value": "",
                "L_mode": "prefix_probe_default_dh_vs_gTu",
                "L_hat": "",
                "h2_L": "",
                "d_trainable": 245760,
                "mask_active_frac_all": "",
                "mask_active_frac_quantized_linear": "",
                "sparse_selection": "",
                "notes": "prefix parameters only; base Linear weights use INT4 RTNClip forward; hstar_ours uses SST-5 full-data prefix probe best h=0.1",
            })
    return out


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
export FALLBACK_USED=0

MANIFEST="$OUTPUT_ROOT/manifests/lane${{LANE_ID}}.csv"
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
args=(
  --job-name=int4_sparse001_prefix
  --nodes=1
  --ntasks=1
  --cpus-per-task=8
  --mem=96G
  --time="$TIME"
  --gres="gpu:${{GPU_TYPE}}:${{GPUS}}"
  --array=0-5
  --output="$OUTPUT_ROOT/slurm_logs/%x_%A_%a.out"
  --export=ALL,OUTPUT_ROOT="$OUTPUT_ROOT",REPO_ROOT="$REPO_ROOT",CONDA_ENV="$CONDA_ENV",GPU_TYPE="$GPU_TYPE"
)
if [[ -n "$ACCOUNT" ]]; then args+=(--account="$ACCOUNT"); fi
if [[ -n "$PARTITION" ]]; then args+=(--partition="$PARTITION"); fi

sbatch "${{args[@]}}" slurm/int4_sparse001_prefix_lane.sbatch | tee -a "$OUTPUT_ROOT/jobs/job_ids.txt"
""",
        encoding="utf-8",
    )
    path.chmod(0o755)


def write_bootstrap(path: Path, output_root: Path) -> None:
    path.write_text(
        f"""#!/bin/bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
cd "$REPO_ROOT"

ENV_NAME="${{ENV_NAME:-mezo-rtnclip-int4}}"
SOURCE_ENV="${{SOURCE_ENV:-ciao}}"
OUTPUT_ROOT="${{OUTPUT_ROOT:-{output_root}}}"

if [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [[ -f "$HOME/.conda/etc/profile.d/conda.sh" ]]; then
  source "$HOME/.conda/etc/profile.d/conda.sh"
else
  echo "Could not find conda.sh" >&2
  exit 1
fi

if conda env list | awk '{{print $1}}' | grep -Fxq "$ENV_NAME"; then
  echo "Conda env $ENV_NAME already exists; skipping creation."
else
  echo "Creating conda env $ENV_NAME by cloning $SOURCE_ENV"
  conda create -y -n "$ENV_NAME" --clone "$SOURCE_ENV"
fi

conda activate "$ENV_NAME"
python -m py_compile tools/rtnclip_roberta_sst5_batch.py

export CONDA_ENV="$ENV_NAME"
export OUTPUT_ROOT="$OUTPUT_ROOT"
export REPO_ROOT="$REPO_ROOT"
export DATALOADER_SHUFFLE=True

bash scripts/submit_int4_sparse001_prefix_batch.sh
""",
        encoding="utf-8",
    )
    path.chmod(0o755)


def write_readme(path: Path, output_root: Path, prefix_probe_source: str) -> None:
    path.write_text(
        f"""# INT4 sparse p=0.01 + prefix full-data batch

Run from repo root after unpacking:

```bash
PARTITION=gpu_p GPU_TYPE=H100 ./run_int4_sparse001_prefix_batch.sh
```

The script creates conda env `mezo-rtnclip-int4` by cloning `SOURCE_ENV` (default `ciao`) if it does not already exist. If the env already exists, creation is skipped.

Output root: `{output_root}`

Runs:
- Sparse INT4 p=0.01 task_grad_static unscaled, h policies: `1e-5`, `1e-3`, `hstar_ours` from `hstar_sparse_p0p01/hstar_full_data_summary.csv`.
- Prefix INT4-base Prefix-MeZO, h policies: `1e-5`, `1e-3`, `0.1`.

Common settings:
- RoBERTa-large
- datasets: SST-2, SST-5, RTE, MNLI, TREC
- full data, seed/data_seed 16, bs64, RandomSampler
- INT4 G128 RTNClip shared-grid fake quantized base forward
- FP16 master update
- max_steps 20000, eval/checkpoint every 1000
- 6 H100 lanes

Prefix note:
- Prefix uses random prefix init. The real-activation initializer is disabled because it triggers CUDA index asserts with the current tokenizer/model stack.
- Prefix h=0.1 is selected from prior SST-5 full-data prefix probe: `{prefix_probe_source}`.
""",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--sparse001_hstar_summary", required=True)
    parser.add_argument("--prefix_probe_source", default="outputs/rtnclip_int4_adapter_nmse_probe_20260522_dirs32/summary.csv")
    args = parser.parse_args()

    output_root = Path(args.output_root)
    sparse_rows = sparse001_rows(read_rows(Path(args.sparse001_hstar_summary)), output_root, Path(args.sparse001_hstar_summary))
    pref_rows = prefix_rows(output_root, args.prefix_probe_source)
    rows = sparse_rows + pref_rows
    cols = columns(rows)

    write_csv(output_root / "manifests" / "sparse_p0p01_manifest.csv", sparse_rows, cols)
    write_csv(output_root / "manifests" / "prefix_manifest.csv", pref_rows, cols)
    write_csv(output_root / "int4_sparse001_prefix_manifest.csv", rows, cols)
    for idx, lane in enumerate(split_lanes(rows, 6)):
        write_csv(output_root / "manifests" / f"lane{idx}.csv", lane, cols)

    write_lane_sbatch(Path("slurm/int4_sparse001_prefix_lane.sbatch"), output_root)
    write_submit_script(Path("scripts/submit_int4_sparse001_prefix_batch.sh"), output_root)
    write_bootstrap(Path("run_int4_sparse001_prefix_batch.sh"), output_root)
    write_readme(output_root / "README.md", output_root, args.prefix_probe_source)
    (output_root / "commands.txt").write_text(
        "\n".join([
            f"python scripts/prepare_int4_sparse001_prefix_batch.py --output_root {output_root} --sparse001_hstar_summary {args.sparse001_hstar_summary}",
            "PARTITION=gpu_p GPU_TYPE=H100 ./run_int4_sparse001_prefix_batch.sh",
        ]) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(rows)} rows to {output_root}; lanes=6")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
