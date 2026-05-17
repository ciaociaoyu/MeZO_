#!/usr/bin/env bash
set -euo pipefail

ROOT="/scratch/jy03364/MeZO_/experiments/main_latest/roberta-large/sst5/groupwise_int8_block256_window_continuation_seed16_20260517"
export CUDA_VISIBLE_DEVICES=0

cd /scratch/jy03364/MeZO_
conda run --no-capture-output -n ciao python scripts/probe_h_window_diagnostic.py \
  --model_name_or_path roberta-large \
  --task_name SST-5 \
  --precision_mode int8 \
  --quant_bits 8 \
  --quantization_algorithm groupwise_int8_block256 \
  --quantization_group_size 256 \
  --quantization_block_size 256 \
  --zo_h 0.001 \
  --h_list 1e-4,3e-4,7e-4,1e-3,1.5e-3,2e-3,2.5e-3,3e-3,4e-3,5e-3,1e-2 \
  --num_probe_directions 50 \
  --num_probe_batches 1 \
  --direction_type dense \
  --compute_true_grad_directional True \
  --output_dir "${ROOT}/02_dense_probe_window" \
  --seed 16 \
  --data_seed 16 \
  --batch_size 64 \
  --cuda_visible_devices 0 \
  2>&1 | tee "${ROOT}/logs/dense_probe_window_k50.log"

