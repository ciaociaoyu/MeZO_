#!/bin/bash
#SBATCH --job-name=hsweep8h_sparse_fp16_roberta_sst5
#SBATCH --partition=gpu_p
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=120G
#SBATCH --gres=gpu:H100:1
#SBATCH --time=12:00:00
#SBATCH --chdir=/scratch/jy03364/MeZO_/experiments/pilot/sparse_mezo/roberta-large/sst5/fp16/h_sweep_8h
#SBATCH --output=/scratch/jy03364/MeZO_/experiments/pilot/sparse_mezo/roberta-large/sst5/fp16/h_sweep_8h/logs/slurm_%x_%j.out

set -uo pipefail

ml jq
set +u
source /home/jy03364/miniconda3/etc/profile.d/conda.sh
conda activate ciao
set -u

export VARIANT="sparse_mezo_fp16"
export ZO_QUANTIZATION_ALIAS="fp16"
export PRECISION_LABEL="fp16"
export SEED=16
export TASK_NAME="sst-5"
export TASK_KEY="sst5"
export MODEL_KEY="roberta-large"
export MODEL_NAME="roberta-large"
export BS=64
export LR=1e-6
export WD=0
export MAX_STEPS=25000
export EVAL_STEPS=1000
export LOGGING_STEPS=10
export DATASET_MODE="full"
export DATALOADER_SHUFFLE="True"
export RANDOM_PREDICTION_GUARD_ENABLED="False"
export ZO_PROBE_HEALTH_GUARD_ENABLED="False"
export ZO_PROBE_EVERY=200
export ZO_PROBE_NUM_SEEDS=16
export SPARSE_RATIO=0.25
export SPARSE_MASK_STRATEGY="percentile_per_layer"
export SPARSE_SCOPE="trainable_only"
export SPARSE_LOG_ACTIVE_FRACTION="True"

source /scratch/jy03364/MeZO_/experiments/pilot/_shared/h_sweep_8h/run_medium_sweep.sh
