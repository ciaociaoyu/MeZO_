#!/usr/bin/env bash
set -euo pipefail

ROOT="/scratch/jy03364/MeZO_/experiments/main_latest/roberta-large/sst5/groupwise_int8_block256_window_continuation_seed16_20260517"
export CUDA_VISIBLE_DEVICES=0

cd /scratch/jy03364/MeZO_

run_sparse_probe() {
  local p="$1"
  local h_list="$2"
  local tag="${p/./p}"
  local out_dir="${ROOT}/04_sparse_probe_by_rate/bernoulli_p${tag}"
  mkdir -p "${out_dir}"
  conda run --no-capture-output -n ciao python scripts/probe_h_window_diagnostic.py \
    --model_name_or_path roberta-large \
    --task_name SST-5 \
    --precision_mode int8 \
    --quant_bits 8 \
    --quantization_algorithm groupwise_int8_block256 \
    --quantization_group_size 256 \
    --quantization_block_size 256 \
    --zo_h 0.0006 \
    --h_list "${h_list}" \
    --num_probe_directions 50 \
    --num_probe_batches 1 \
    --direction_type sparse \
    --sparse_rate "${p}" \
    --sparse_mode bernoulli \
    --sparse_rescale inv_sqrt_p \
    --compute_true_grad_directional True \
    --output_dir "${out_dir}" \
    --seed 16 \
    --data_seed 16 \
    --batch_size 64 \
    --cuda_visible_devices 0 \
    2>&1 | tee "${ROOT}/logs/sparse_probe_bernoulli_p${tag}_k50.log"
}

run_sparse_probe "0.003" "4.10792e-5,8.21584e-5,1.64317e-4,3.28634e-4,6.57267e-4,1.31453e-3"
run_sparse_probe "0.01" "7.5e-5,1.5e-4,3e-4,6e-4,1.2e-3,2.4e-3"
run_sparse_probe "0.03" "1.29904e-4,2.59808e-4,5.19615e-4,1.03923e-3,2.07846e-3,4.15692e-3"
run_sparse_probe "0.1" "2.37171e-4,4.74342e-4,9.48683e-4,1.89737e-3,3.79473e-3,7.58947e-3"

