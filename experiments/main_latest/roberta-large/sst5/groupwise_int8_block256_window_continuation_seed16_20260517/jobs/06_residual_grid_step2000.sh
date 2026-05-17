#!/usr/bin/env bash
set -euo pipefail

ROOT="/scratch/jy03364/MeZO_/experiments/main_latest/roberta-large/sst5/groupwise_int8_block256_window_continuation_seed16_20260517"
export CUDA_VISIBLE_DEVICES=0

if [[ "${CONDA_DEFAULT_ENV:-}" != "ciao" ]]; then
  exec conda run --no-capture-output -n ciao bash "$0" "$@"
fi

cd /scratch/jy03364/MeZO_/medium_models

run_name="residual_grid_groupwise256_h3e-3_lr7e-5_clip3_step2000"
TASK=SST-5 K=16 SEED=16 DATA_SEED=16 DATASET_MODE=full FULL_DEV_RATIO=0.1 \
BS=64 LR=7e-5 WD=0 EPS=0.003 STEP=2000 EVAL_STEP=500 MODEL=roberta-large \
USE_H=False USE_C=False DATALOADER_SHUFFLE=True EFFICIENT_ZERO_ORDER=True ZERO_ORDER_USE_TRAINER_OPTIM=False \
EXTRA_TAG="${run_name}" \
bash ./mezo.sh \
  --result_root "${ROOT}/06_summaries" \
  --job_name "${run_name}" \
  --dataset_mode full \
  --precision_mode int8 \
  --zo_quantization int8 \
  --quantization_algorithm groupwise_int8_block256 \
  --quantization_group_size 256 \
  --quantization_block_size 256 \
  --zo_update_backend residual_grid \
  --zo_h 0.003 \
  --direction_type dense \
  --residual_dtype fp32 \
  --residual_commit_mode round \
  --residual_max_code_step 1 \
  --residual_scale_mode block \
  --residual_block_size 256 \
  --int8_freeze_scale True \
  --zo_update_norm_clip 3 \
  --gradient_accumulation_steps 1 \
  --save_strategy no \
  --no_predict \
  --random_prediction_guard_enabled False \
  --log_update_stats_every 100 \
  --save_update_stats_jsonl update_stats.jsonl \
  2>&1 | tee -a "${ROOT}/logs/${run_name}.log"
