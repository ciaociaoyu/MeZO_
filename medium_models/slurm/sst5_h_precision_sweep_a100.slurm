#!/bin/bash
#SBATCH --job-name=sst5-h-precision-sweep
#SBATCH --output=jobs/%x_%j.out
#SBATCH --error=jobs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:1
#SBATCH --account=YOUR_ACCOUNT

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"
mkdir -p jobs

# Optional: load your cluster modules / conda env here.
# module load cuda/12.1
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate YOUR_ENV

export TASK=sst-5
export MODEL=roberta-large
export K=16
export BS=32
export LR=1e-6
export STEP=20000
export EVAL_STEP=200
export DATALOADER_SHUFFLE=False
export USE_H=False
export USE_C=False
export ZO_PROBE_EVERY=200
export ZO_PROBE_NUM_SEEDS=16

# Example: override default single-seed run:
# export SEEDS="13 21 42"

bash scripts/exp_sst5_h_precision_sweep.sh
