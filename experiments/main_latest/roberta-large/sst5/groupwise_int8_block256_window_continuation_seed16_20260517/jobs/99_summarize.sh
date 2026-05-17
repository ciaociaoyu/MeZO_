#!/usr/bin/env bash
set -euo pipefail

ROOT="/scratch/jy03364/MeZO_/experiments/main_latest/roberta-large/sst5/groupwise_int8_block256_window_continuation_seed16_20260517"

cd /scratch/jy03364/MeZO_
conda run --no-capture-output -n ciao python scripts/summarize_groupwise256_continuation.py \
  --root "${ROOT}" \
  2>&1 | tee "${ROOT}/logs/summarize_groupwise256.log"

