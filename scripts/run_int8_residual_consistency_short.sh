#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TS="${TS:-$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-${REPO_ROOT}/runs/int8_residual_consistency_${TS}}"
CONDA_ENV="${CONDA_ENV:-ciao}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
INT8_SCALE_FLOOR="${INT8_SCALE_FLOOR:-0}"

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
  local commit_mode="$4"
  local max_code_step="$5"
  local update_clip="$6"
  local extra_args=()

  if [[ "${backend}" == "residual_grid" ]]; then
    extra_args+=(
      --residual_dtype fp32
      --residual_commit_mode "${commit_mode}"
      --residual_max_code_step "${max_code_step}"
      --int8_freeze_scale True
      --int8_scale_floor "${INT8_SCALE_FLOOR}"
    )
  fi
  if [[ "${update_clip}" != "0" ]]; then
    extra_args+=(--zo_update_norm_clip "${update_clip}")
  fi

  echo "${run_name} backend=${backend} lr=${lr} commit=${commit_mode} max_code_step=${max_code_step} clip=${update_clip}" | tee -a "${RUN_ROOT}/run_manifest.txt"
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
    STEP=50 \
    EVAL_STEP=100000 \
    MODEL=roberta-large \
    USE_H=False \
    USE_C=False \
    DATALOADER_SHUFFLE=False \
    EFFICIENT_ZERO_ORDER=True \
    EXTRA_TAG=int8-residual-consistency \
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

run_case direct_int8_lr1e-5 direct_int8 1e-5 round 0 0
run_case residual_grid_round_lr3e-5 residual_grid 3e-5 round 0 0
run_case residual_grid_round_step1_lr1e-4_clip5 residual_grid 1e-4 round 1 5
run_case residual_grid_stoch_step1_lr3e-4_clip10 residual_grid 3e-4 stochastic 1 10

python "${REPO_ROOT}/scripts/summarize_int8_residual_runs.py" "${RUN_ROOT}" 2>&1 | tee "${RUN_ROOT}/logs/summary.log"

echo "Done. RUN_ROOT=${RUN_ROOT}"
