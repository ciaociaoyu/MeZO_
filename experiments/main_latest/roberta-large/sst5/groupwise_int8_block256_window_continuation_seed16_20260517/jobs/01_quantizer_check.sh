#!/usr/bin/env bash
set -euo pipefail

ROOT="/scratch/jy03364/MeZO_/experiments/main_latest/roberta-large/sst5/groupwise_int8_block256_window_continuation_seed16_20260517"
export CUDA_VISIBLE_DEVICES=0

cd /scratch/jy03364/MeZO_
conda run --no-capture-output -n ciao python scripts/check_groupwise256_quantizer.py \
  --output_dir "${ROOT}/01_quantizer_checks" \
  --model_name_or_path roberta-large \
  --bits 8 \
  --group_size 256 \
  --device cuda \
  2>&1 | tee "${ROOT}/logs/quantizer_check.log"

