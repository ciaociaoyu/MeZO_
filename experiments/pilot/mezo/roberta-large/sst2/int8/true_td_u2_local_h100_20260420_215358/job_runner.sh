#!/usr/bin/env bash
set -euo pipefail

ROOT="/scratch/jy03364/MeZO_"
EXP_ROOT="$ROOT/experiments/pilot/mezo/roberta-large/sst2/int8/true_td_u2_local_h100_20260420_215358"
LOG_DIR="$EXP_ROOT/logs"
RESULT_ROOT="$EXP_ROOT/results"
JOB_NAME="run_local_h100_mezo_int8_sst2_true_td_u2"

mkdir -p "$LOG_DIR" "$RESULT_ROOT"

cd "$ROOT/medium_models"
source /home/jy03364/miniconda3/etc/profile.d/conda.sh
conda activate ciao

export TASK="SST-2"
export K="16"
export SEED="16"
export DATA_SEED="16"
export DATASET_MODE="full"
export FULL_DEV_RATIO="0.1"
export BS="64"
export LR="1e-6"
export EPS="1e-4"
export WD="0"
export OPT="sgd"
export ZERO_ORDER_USE_TRAINER_OPTIM="False"
export EFFICIENT_ZERO_ORDER="True"
export USE_H="False"
export USE_C="False"
export DATALOADER_SHUFFLE="True"
export STEP="10000"
export EVAL_STEP="1000"
export MODEL="roberta-large"
export EXTRA_TAG="pilot-local-h100-mezo-int8-sst2-true-td-u2-h1e-4-10k"

echo "[launch] $(date --iso-8601=seconds)"
echo "[launch] exp_root=$EXP_ROOT"
echo "[launch] result_root=$RESULT_ROOT"
echo "[launch] job_name=$JOB_NAME"

bash "$ROOT/medium_models/mezo.sh" \
  --result_root "$RESULT_ROOT" \
  --job_name "$JOB_NAME" \
  --zo_two_point_precision fp16 \
  --zo_quantization int8 \
  --zo_use_true_directional_derivative True \
  --zo_probe_every 200 \
  --zo_probe_num_seeds 16 \
  --zo_probe_log_csv True \
  --random_prediction_guard_enabled False \
  --zo_probe_health_guard_enabled False
