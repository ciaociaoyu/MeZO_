#!/usr/bin/env bash
set -euo pipefail

ROOT="/scratch/jy03364/MeZO_/experiments/main_latest/roberta-large/sst5/groupwise_int8_block256_window_continuation_seed16_20260517"
export CUDA_VISIBLE_DEVICES=0

{
  echo "hostname=$(hostname)"
  echo "date=$(date)"
  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
  echo "git_commit=$(cd /scratch/jy03364/MeZO_ && git rev-parse HEAD 2>/dev/null || true)"
} > "${ROOT}/manifests/environment.txt"

nvidia-smi > "${ROOT}/manifests/nvidia_smi.txt"

