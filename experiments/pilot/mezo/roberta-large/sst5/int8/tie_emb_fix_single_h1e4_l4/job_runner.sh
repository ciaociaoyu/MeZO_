#!/bin/bash

set -euo pipefail

SCRATCH_ROOT="/scratch/jy03364/MeZO_"
MEDIUM_ROOT="${SCRATCH_ROOT}/medium_models"
NAN_GUARD="${SCRATCH_ROOT}/experiments/main/_shared/h_sweep_14h/nan_guard.py"
EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-${SCRATCH_ROOT}/experiments/pilot/mezo/roberta-large/sst5/int8/tie_emb_fix_single_h1e4_l4/runs/manual}"

ml jq
set +u
source /home/jy03364/miniconda3/etc/profile.d/conda.sh
conda activate ciao
set -u

VARIANT="mezo_int8_tieemb_fix_single_h1e4"
JOB_NAME="run_${VARIANT}"
H_VALUE="1e-4"
RUN_ROOT="${EXPERIMENT_ROOT}/results/${VARIANT}/roberta-large/sst5/h_${H_VALUE}"
LOG_DIR="${EXPERIMENT_ROOT}/logs/${VARIANT}/roberta-large/sst5/h_${H_VALUE}/seed_16"
RUN_LOG="${LOG_DIR}/train.log"
RUN_ERR="${LOG_DIR}/train.err"

mkdir -p "${RUN_ROOT}" "${LOG_DIR}"

export TASK="sst-5"
export K=16
export SEED=16
export DATA_SEED=16
export DATASET_MODE="fewshot"
export BS=64
export LR=1e-6
export WD=0
export STEP=10000
export EVAL_STEP=1000
export MODEL="roberta-large"
export USE_H=False
export USE_C=False
export DATALOADER_SHUFFLE=True
export EPS="${H_VALUE}"
export EXTRA_TAG="tie-emb-fix-h1e4"

python "${NAN_GUARD}" \
  --cwd "${MEDIUM_ROOT}" \
  --stdout-log "${RUN_LOG}" \
  --stderr-log "${RUN_ERR}" \
  --max-consecutive-nan 100 \
  -- \
  bash "${MEDIUM_ROOT}/mezo.sh" \
    --result_root "${RUN_ROOT}" \
    --job_name "${JOB_NAME}" \
    --dataset_mode fewshot \
    --zo_two_point_precision fp16 \
    --zo_quantization int8 \
    --zo_probe_every 200 \
    --zo_probe_num_seeds 16 \
    --zo_probe_log_csv True \
    --random_prediction_guard_enabled False \
    --zo_probe_health_guard_enabled False \
    --tie_emb True
