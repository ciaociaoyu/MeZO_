#!/bin/bash

set -euo pipefail

SCRATCH_ROOT="/scratch/jy03364/MeZO_"
EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-${SCRATCH_ROOT}/experiments/pilot/mezo/roberta-large/sst5/int8/noguard_50k_20260420}"

ml jq
set +u
source /home/jy03364/miniconda3/etc/profile.d/conda.sh
conda activate ciao
set -u

export EXPERIMENT_ROOT
export VARIANT="${VARIANT:-mezo_int8_noguard_50k}"
export ZO_QUANTIZATION_ALIAS="int8"
export PRECISION_LABEL="int8"
export SEED=16
export TASK_NAME="sst-5"
export TASK_KEY="sst5"
export MODEL_KEY="roberta-large"
export MODEL_NAME="roberta-large"
export K=16
export DATASET_MODE="full"
export DATA_SEED=16
export BS=64
export LR=1e-6
export MAX_STEPS=50000
export EVAL_STEPS=5000
export LOGGING_STEPS=10
export ZO_PROBE_EVERY=200
export ZO_PROBE_NUM_SEEDS=16
export RANDOM_PREDICTION_GUARD_ENABLED=False
export ZO_PROBE_HEALTH_GUARD_ENABLED=False
export NAN_GUARD_LIMIT=100

source "${SCRATCH_ROOT}/experiments/pilot/_shared/h_sweep_8h/run_medium_sweep.sh"
