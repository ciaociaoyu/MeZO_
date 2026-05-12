#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TS="${TS:-$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-${REPO_ROOT}/runs/int8_residual_round2_${TS}}"
CONDA_ENV="${CONDA_ENV:-ciao}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

mkdir -p "${RUN_ROOT}/logs"

if [[ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]]; then
  source "${HOME}/miniconda3/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV}"
fi

echo "RUN_ROOT=${RUN_ROOT}" | tee "${RUN_ROOT}/run_manifest.txt"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader | tee "${RUN_ROOT}/nvidia_smi_start.txt"

cd "${REPO_ROOT}/medium_models"

run_case() {
  local run_name="$1"
  local backend="$2"
  local lr="$3"
  local steps="$4"
  local commit_mode="$5"
  local max_code_step="$6"
  local update_clip="$7"
  local extra_args=()
  if [[ "${backend}" == "residual_grid" ]]; then
    extra_args+=(
      --residual_dtype fp32
      --residual_commit_mode "${commit_mode}"
      --residual_max_code_step "${max_code_step}"
      --int8_freeze_scale True
    )
  fi
  if [[ "${update_clip}" != "0" ]]; then
    extra_args+=(--zo_update_norm_clip "${update_clip}")
  fi

  echo "${run_name} backend=${backend} lr=${lr} steps=${steps} commit=${commit_mode} max_code_step=${max_code_step} clip=${update_clip}" | tee -a "${RUN_ROOT}/run_manifest.txt"
  (
    set -x
    TASK=SST-5 \
    K=16 \
    SEED=16 \
    DATA_SEED=16 \
    DATASET_MODE=full \
    FULL_DEV_RATIO=0.1 \
    BS=64 \
    LR="${lr}" \
    EPS=3e-3 \
    WD=0 \
    STEP="${steps}" \
    EVAL_STEP=100000 \
    MODEL=roberta-large \
    USE_H=False \
    USE_C=False \
    DATALOADER_SHUFFLE=False \
    EFFICIENT_ZERO_ORDER=True \
    EXTRA_TAG=int8-residual-round2 \
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
    bash ./mezo.sh \
      --result_root "${RUN_ROOT}" \
      --job_name "${run_name}" \
      --dataset_mode full \
      --zo_quantization int8 \
      --zo_two_point_precision fp16 \
      --zo_h 3e-3 \
      --zo_update_backend "${backend}" \
      --log_update_stats_every 1 \
      --save_update_stats_jsonl update_stats.jsonl \
      --zo_probe_every 0 \
      --random_prediction_guard_enabled False \
      --save_strategy no \
      --no_predict \
      "${extra_args[@]}"
  ) 2>&1 | tee "${RUN_ROOT}/logs/${run_name}.log"
}

python "${REPO_ROOT}/medium_models/tests/test_residual_grid_update.py" 2>&1 | tee "${RUN_ROOT}/logs/synthetic_residual_test.log"

run_case noop_residual_grid_lr0 residual_grid 0 1 round 0 0
run_case direct_int8_lr1e-5 direct_int8 1e-5 50 round 0 0
run_case residual_grid_lr1e-5 residual_grid 1e-5 50 round 0 0
run_case residual_grid_lr3e-5 residual_grid 3e-5 50 round 0 0
run_case residual_grid_lr1e-4 residual_grid 1e-4 50 round 0 0
run_case residual_grid_round_step1_lr1e-4_clip5 residual_grid 1e-4 50 round 1 5
run_case residual_grid_stoch_step1_lr1e-4_clip5 residual_grid 1e-4 50 stochastic 1 5
run_case residual_grid_stoch_step1_lr3e-4_clip5 residual_grid 3e-4 50 stochastic 1 5
run_case residual_grid_stoch_step1_lr3e-4_clip10 residual_grid 3e-4 50 stochastic 1 10

python - "${RUN_ROOT}" <<'PY'
import csv
import json
import math
import pathlib
import re
import sys

run_root = pathlib.Path(sys.argv[1])
rows = []
for path in sorted(run_root.glob("*/seed*/run_summary.json")):
    run_dir = path.parent
    summary = json.loads(path.read_text())
    cfg = summary.get("config", {}).get("training_args", {})
    update_path = run_dir / "update_stats.jsonl"
    updates = [json.loads(x) for x in update_path.read_text().splitlines() if x.strip()] if update_path.exists() else []
    last = updates[-1] if updates else {}
    eval_loss = None
    eval_acc = None
    for metrics in (summary.get("eval", {}) or {}).values():
        if isinstance(metrics, dict):
            eval_loss = metrics.get("eval_loss", eval_loss)
            for key, val in metrics.items():
                if "acc" in key:
                    eval_acc = val
    metrics_csv = run_dir / "metrics_logs" / "metrics_adaptiveH-0_cscale-0.csv"
    final_train_loss = None
    if metrics_csv.exists():
        with metrics_csv.open(newline="") as f:
            records = list(csv.DictReader(f))
        if records:
            final_train_loss = records[-1].get("train_loss")
    values = list(updates)
    nan_occurred = any(
        any(isinstance(v, float) and (not math.isfinite(v)) for v in row.values())
        for row in values
    )
    rows.append({
        "run_name": run_dir.parent.name,
        "backend": cfg.get("zo_update_backend", last.get("update_backend")),
        "h": cfg.get("zero_order_eps", last.get("h")),
        "lr": cfg.get("learning_rate", last.get("lr")),
        "residual_dtype": cfg.get("residual_dtype", ""),
        "commit_mode": cfg.get("residual_commit_mode", ""),
        "max_code_step": cfg.get("residual_max_code_step", ""),
        "update_norm_clip": cfg.get("zo_update_norm_clip", last.get("zo_update_norm_clip")),
        "steps_completed": len(updates),
        "final_train_loss": final_train_loss,
        "final_eval_loss": eval_loss,
        "best_acc": eval_acc,
        "final_acc": eval_acc,
        "global_active_frac_last": last.get("global_active_frac", last.get("active_frac")),
        "global_cos_intended_actual_last": last.get("global_cos_intended_actual", last.get("cos_intended_actual")),
        "global_actual_over_intended_norm_ratio_last": last.get("global_actual_over_intended_norm_ratio", last.get("actual_over_intended_norm_ratio")),
        "saturation_frac_last": last.get("global_saturation_frac", last.get("saturation_frac")),
        "residual_over_scale_p99_last": last.get("residual_over_scale_p99"),
        "residual_over_scale_max_last": last.get("residual_over_scale_max"),
        "grid_error_norm_last": last.get("grid_error_norm"),
        "nan_occurred": nan_occurred,
    })
fields = [
    "run_name","backend","h","lr","residual_dtype","commit_mode","max_code_step","update_norm_clip",
    "steps_completed","final_train_loss","final_eval_loss","best_acc","final_acc",
    "global_active_frac_last","global_cos_intended_actual_last","global_actual_over_intended_norm_ratio_last",
    "saturation_frac_last","residual_over_scale_p99_last","residual_over_scale_max_last","grid_error_norm_last","nan_occurred"
]
csv_path = run_root / "summary.csv"
md_path = run_root / "summary.md"
with csv_path.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
with md_path.open("w") as f:
    f.write("| " + " | ".join(fields) + " |\n")
    f.write("| " + " | ".join(["---"] * len(fields)) + " |\n")
    for row in rows:
        vals = []
        for field in fields:
            val = row.get(field)
            vals.append(f"{val:.6g}" if isinstance(val, float) else "" if val is None else str(val))
        f.write("| " + " | ".join(vals) + " |\n")
print(f"summary_csv={csv_path}")
print(f"summary_md={md_path}")
PY

echo "Done. RUN_ROOT=${RUN_ROOT}"
