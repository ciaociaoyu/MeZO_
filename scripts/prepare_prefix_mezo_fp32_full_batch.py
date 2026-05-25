#!/usr/bin/env python
"""Prepare original all-FP32 Prefix-MeZO full-data clean-GL batch."""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = REPO_ROOT / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from analyze_int4_sst5_calibrated_hstar import choose_l_plateau, simple2pt_corrected  # noqa: E402


TASKS = ["sst-2", "sst-5", "rte", "mnli", "trec"]
POLICIES = [
    ("fixed_small", 1e-5, "1e-5"),
    ("mezo_default", 1e-3, "1e-3"),
    ("hstar_cleanGL", None, None),
    ("prefix_h1e-2", 1e-2, "1e-2"),
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


def h_label_from_value(h: float) -> str:
    known = {
        1e-5: "1e-5",
        3e-5: "3e-5",
        1e-4: "1e-4",
        3e-4: "3e-4",
        1e-3: "1e-3",
        1e-2: "1e-2",
        1e-1: "1e-1",
    }
    for value, label in known.items():
        if abs(float(h) - value) <= max(abs(value) * 1e-9, 1e-15):
            return label
    return f"{h:.10g}".replace(".", "p").replace("-", "m")


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


def recompute_clean_hstar(clean_source: Path, output_root: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for task in TASKS:
        task_dir = clean_source / f"{task}_prefix"
        if not task_dir.exists():
            raise FileNotFoundError(f"Missing clean prefix source directory: {task_dir}")
        base = json.loads((task_dir / "hstar_summary.json").read_text(encoding="utf-8"))
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
        delta = safe_float(base.get("prefix_lattice_delta_ulp_rms") or base.get("Delta_value"))
        d_trainable = int(float(base.get("d_trainable", 245760)))
        corrected = simple2pt_corrected(
            "fp32",
            d_trainable,
            l_hat,
            scale_rms=delta * math.sqrt(6.0),
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
            "seed": 16,
            "data_seed": 16,
            "batch_size": 64,
            "direction_mode": "prefix",
            "forward_path": "original_medium_models_mezo_fp32",
            "precision": "fp32",
            "quantizer": "none",
            "master_dtype": "fp32",
            "prefix_precision": "fp32",
            "prefix_num": 5,
            "prefix_reparam": False,
            "prefix_init_by_real_act": True,
            "selector_name": "simple2pt_corrected_cleanGL",
            "Delta_mode": "prefix_fp32_delta_ulp_rms",
            "Delta_value": delta,
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
                "Original medium_models Prefix-MeZO all-FP32 path. "
                "h-star recomputed from clean FP32 prefix-only G/L; no INT4 forward is used for these training jobs."
            ),
        }
        rows.append(row)

    write_csv(output_root / "hstar_prefix_fp32_cleanGL.csv", rows)
    md = [
        "# Prefix-MeZO FP32 Clean-GL h-star",
        "",
        "This recomputes h-star from clean FP32 prefix-only G/L and uses the original `medium_models/mezo.sh` prefix path for training.",
        "",
        "| dataset | hstar_cont | nearest | G clean | L clean q90 | Delta | d |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in rows:
        md.append(
            f"| {r['task_name']} | {float(r['hstar_cont']):.6g} | {r['hstar_nearest_grid_label']} | "
            f"{float(r['G_value']):.6g} | {float(r['L_hat']):.6g} | {float(r['Delta_value']):.6g} | {r['d_trainable']} |"
        )
    (output_root / "hstar_prefix_fp32_cleanGL.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return rows


def split_lanes(rows: List[Dict[str, object]], n_lanes: int) -> List[List[Dict[str, object]]]:
    lanes = [[] for _ in range(n_lanes)]
    for idx, row in enumerate(rows):
        lanes[idx % n_lanes].append(row)
    return lanes


def build_manifest(hstar_rows: List[Dict[str, object]], output_root: Path, lanes: int) -> List[Dict[str, object]]:
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
            run_name = f"prefix_mezo_fp32_{task.replace('-', '')}_{policy}_h{h_label}_seed16_full_bs64_step20k"
            rows.append(
                {
                    "run_name": run_name,
                    "run_dir": str(output_root / "results" / run_name / "seed16"),
                    "phase": "prefix_mezo_fp32_full_data_cleanGL_20k",
                    "task_name": task,
                    "dataset": task,
                    "dataset_mode": "full",
                    "num_k": 16,
                    "seed": 16,
                    "data_seed": 16,
                    "batch_size": 64,
                    "h": h,
                    "h_label": h_label,
                    "h_policy": policy,
                    "max_steps": 20000,
                    "lr": 1e-2,
                    "eval_every": 1000,
                    "checkpoint_steps": 1000,
                    "direction_mode": "prefix",
                    "prefix_num": 5,
                    "prefix_reparam": False,
                    "prefix_init_by_real_act": True,
                    "precision_mode": "fp32",
                    "zo_quantization": "fp32",
                    "zo_quantization_bits": 32,
                    "zo_two_point_precision": "fp32",
                    "update_backend": "original_mezo_fp32",
                    "quantizer": "none",
                    "master_dtype": "fp32",
                    "hstar_source": str(output_root / "hstar_prefix_fp32_cleanGL.csv"),
                    "selector_name": src["selector_name"] if policy == "hstar_cleanGL" else "",
                    "hstar_cont": src["hstar_cont"],
                    "hstar_nearest_grid": src["hstar_nearest_grid"],
                    "hstar_used": h if policy == "hstar_cleanGL" else "",
                    "hstar_used_policy": "continuous_unsnapped_cleanGL" if policy == "hstar_cleanGL" else "",
                    "Delta_mode": src["Delta_mode"],
                    "Delta_value": src["Delta_value"],
                    "G_mode": src["G_mode"],
                    "G_value": src["G_value"],
                    "L_mode": src["L_mode"],
                    "L_hat": src["L_hat"],
                    "h2_L": src["h2_L"],
                    "d_trainable": src["d_trainable"],
                    "notes": "Original all-FP32 Prefix-MeZO; full dataset; no low-bit forward.",
                }
            )
    write_csv(output_root / "prefix_mezo_fp32_manifest.csv", rows)
    for idx, lane in enumerate(split_lanes(rows, lanes)):
        write_csv(output_root / "manifests" / f"prefix_fp32_lane{idx}.csv", lane)
    return rows


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
echo "Running original FP32 Prefix-MeZO lane $LANE_ID"
nvidia-smi || true
python scripts/run_prefix_mezo_fp32_lane.py \\
  --manifest "$OUTPUT_ROOT/manifests/prefix_fp32_lane${{LANE_ID}}.csv" \\
  --output-root "$OUTPUT_ROOT"
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
  --job-name=prefix_mezo_fp32
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
sbatch "${{args[@]}}" slurm/prefix_mezo_fp32_full_lane.sbatch | tee -a "$OUTPUT_ROOT/jobs/job_ids.txt"
""",
        encoding="utf-8",
    )
    path.chmod(0o755)


def write_readme(path: Path, output_root: Path, clean_source: Path) -> None:
    path.write_text(
        f"""# Prefix-MeZO FP32 Full-Data Clean-GL Batch

Output root: `{output_root}`

This batch replaces the old INT4-base prefix jobs. It uses the original `medium_models/mezo.sh` Prefix-MeZO path:

- RoBERTa-large
- full datasets: SST-2, SST-5, RTE, MNLI, TREC
- seed/data_seed = 16
- batch size = 64
- prefix tuning only, `num_prefix=5`, `--no_reparam`, `--prefix_init_by_real_act`
- all FP32: `precision_mode=fp32`, `zo_quantization_bits=32`, `zo_two_point_precision=fp32`
- no INT4 fake quantization, no RTNClip, no sparse, no residual-grid

H policies per dataset:

- `fixed_small`: h=1e-5
- `mezo_default`: h=1e-3
- `hstar_cleanGL`: continuous h-star recomputed from clean FP32 prefix-only G/L
- `prefix_h1e-2`: h=1e-2

Clean G/L source used for h-star recomputation: `{clean_source}`.

Submit:

```bash
PARTITION=gpu_p GPU_TYPE=H100 bash scripts/submit_prefix_mezo_fp32_full_batch.sh
```
""",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--clean_probe_source", default=str(DEFAULT_CLEAN_SOURCE))
    parser.add_argument("--lanes", type=int, default=4)
    args = parser.parse_args()

    output_root = Path(args.output_root)
    clean_source = Path(args.clean_probe_source)
    output_root.mkdir(parents=True, exist_ok=True)
    lanes = int(args.lanes)
    hstar_rows = recompute_clean_hstar(clean_source, output_root)
    manifest_rows = build_manifest(hstar_rows, output_root, lanes)
    write_lane_sbatch(Path("slurm/prefix_mezo_fp32_full_lane.sbatch"), output_root)
    write_submit_script(Path("scripts/submit_prefix_mezo_fp32_full_batch.sh"), output_root, lanes)
    write_readme(output_root / "README.md", output_root, clean_source)

    config = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "output_root": str(output_root),
        "clean_probe_source": str(clean_source),
        "lanes": lanes,
        "num_runs": len(manifest_rows),
        "training_path": "medium_models/mezo.sh original Prefix-MeZO all-FP32",
    }
    write_json(output_root / "config_manifest.json", config)
    commands = [
        f"python scripts/prepare_prefix_mezo_fp32_full_batch.py --output_root {output_root} --clean_probe_source {clean_source} --lanes {lanes}",
        "PARTITION=gpu_p GPU_TYPE=H100 bash scripts/submit_prefix_mezo_fp32_full_batch.sh",
    ]
    (output_root / "commands.txt").write_text("\n".join(commands) + "\n", encoding="utf-8")
    print(f"Wrote {len(manifest_rows)} original FP32 Prefix-MeZO runs under {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
