#!/usr/bin/env bash
set -euo pipefail

ROOT="/scratch/jy03364/MeZO_/experiments/main_latest/roberta-large/sst5/groupwise_int8_block256_window_continuation_seed16_20260517"
export CUDA_VISIBLE_DEVICES=0

if [[ "${CONDA_DEFAULT_ENV:-}" != "ciao" ]]; then
  exec conda run --no-capture-output -n ciao bash "$0" "$@"
fi

cd /scratch/jy03364/MeZO_/medium_models

run_name="dense_groupwise256_fp16master_h3e-3_step5000"
TASK=SST-5 K=16 SEED=16 DATA_SEED=16 DATASET_MODE=full FULL_DEV_RATIO=0.1 \
BS=64 LR=1e-5 WD=0 EPS=0.003 STEP=5000 EVAL_STEP=500 MODEL=roberta-large \
USE_H=False USE_C=False DATALOADER_SHUFFLE=True EFFICIENT_ZERO_ORDER=True ZERO_ORDER_USE_TRAINER_OPTIM=False \
EXTRA_TAG="${run_name}" \
bash ./mezo.sh \
  --result_root "${ROOT}/03_dense_fp16master_training" \
  --job_name "${run_name}" \
  --dataset_mode full \
  --precision_mode int8 \
  --zo_quantization int8 \
  --quantization_algorithm groupwise_int8_block256 \
  --quantization_group_size 256 \
  --quantization_block_size 256 \
  --zo_update_backend fp16_master \
  --zo_h 0.003 \
  --direction_type dense \
  --gradient_accumulation_steps 1 \
  --save_strategy no \
  --main_save_checkpoints True \
  --main_checkpoint_steps 1000 \
  --main_save_final_checkpoint True \
  --main_save_best_acc_checkpoint True \
  --main_save_best_loss_checkpoint False \
  --no_predict \
  --random_prediction_guard_enabled False \
  --log_update_stats_every 500 \
  --save_update_stats_jsonl update_stats.jsonl \
  2>&1 | tee -a "${ROOT}/logs/${run_name}.log"
