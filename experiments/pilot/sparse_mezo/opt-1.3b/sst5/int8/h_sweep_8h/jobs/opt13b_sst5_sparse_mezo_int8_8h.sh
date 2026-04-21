#!/bin/bash
#SBATCH --job-name=hsweep8h_sparse_int8_opt13b_sst5
#SBATCH --partition=gpu_p
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=160G
#SBATCH --gres=gpu:H100:1
#SBATCH --time=72:00:00
#SBATCH --chdir=/scratch/jy03364/MeZO_/experiments/pilot/sparse_mezo/opt-1.3b/sst5/int8/h_sweep_8h
#SBATCH --output=/scratch/jy03364/MeZO_/experiments/pilot/sparse_mezo/opt-1.3b/sst5/int8/h_sweep_8h/logs/slurm_%x_%j.out

set -uo pipefail

set +u
source /home/jy03364/miniconda3/etc/profile.d/conda.sh
conda activate mezo-env
set -u

export VARIANT="sparse_mezo_int8"
export SEED=42
export TASK_NAME="SST5"
export TASK_KEY="sst5"
export MODEL_NAME="opt-1.3b"
export BS=16
export LR=1e-6
export NUM_EPOCHS=5
export MAX_STEPS=10000
export EVAL_STEPS=1000
export LOGGING_STEPS=10
export ZO_PROBE_EVERY=200
export ZO_PROBE_NUM_SEEDS=16
export SPARSE_RATIO=0.25
export SPARSE_MASK_STRATEGY="percentile_per_layer"
export SPARSE_SCOPE="trainable_only"
export SPARSE_LOG_ACTIVE_FRACTION="True"

source /scratch/jy03364/MeZO_/experiments/pilot/_shared/h_sweep_8h/run_large_sweep.sh
