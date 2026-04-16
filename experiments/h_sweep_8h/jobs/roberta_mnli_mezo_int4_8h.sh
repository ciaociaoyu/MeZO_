#!/bin/bash
#SBATCH --job-name=hsweep8h_mezo_int4_roberta_mnli
#SBATCH --partition=gpu_p
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=120G
#SBATCH --gres=gpu:H100:1
#SBATCH --time=72:00:00
#SBATCH --chdir=/scratch/jy03364/MeZO_/experiments/h_sweep_8h
#SBATCH --output=logs/slurm_%x_%j.out

set -uo pipefail

ml jq
set +u
source /home/jy03364/miniconda3/etc/profile.d/conda.sh
conda activate ciao
set -u

export VARIANT="mezo_int4"
export ZO_QUANTIZATION_ALIAS="int4"
export PRECISION_LABEL="int4"
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

source /scratch/jy03364/MeZO_/experiments/h_sweep_8h/run_medium_sweep.sh
