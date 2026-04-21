#!/bin/bash
#SBATCH --job-name=roberta_mnli_sparse_mezo_int8_l40s
#SBATCH --partition=iai_L40_p
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --gres=gpu:L40S:1
#SBATCH --time=48:00:00
#SBATCH --chdir=/scratch/jy03364/MeZO_/experiments/pilot/sparse_mezo/roberta-large/mnli/int8/h_sweep_8h
#SBATCH --output=/scratch/jy03364/MeZO_/experiments/pilot/sparse_mezo/roberta-large/mnli/int8/h_sweep_8h/logs/slurm_%x_%j.out

set -uo pipefail

ml jq
set +u
source /home/jy03364/miniconda3/etc/profile.d/conda.sh
conda activate ciao
set -u

export VARIANT="sparse_mezo_int8"
export SEED=16
export TASK_NAME="MNLI"
export TASK_KEY="mnli"
export MODEL_KEY="roberta-large"
export MODEL_NAME="roberta-large"
export BS=32
export LR=1e-6
export MAX_STEPS=10000
export EVAL_STEPS=1000
export LOGGING_STEPS=10
export ZO_PROBE_EVERY=200
export ZO_PROBE_NUM_SEEDS=16
export SPARSE_RATIO=0.25
export SPARSE_MASK_STRATEGY="percentile_per_layer"
export SPARSE_SCOPE="trainable_only"
export SPARSE_LOG_ACTIVE_FRACTION="True"

source /scratch/jy03364/MeZO_/experiments/pilot/_shared/h_sweep_8h/run_medium_sweep.sh
