#!/usr/bin/env bash
#SBATCH --job-name=next-all
#SBATCH --output=/scratch/jy03364/MeZO_/runs/next_window_sparse_20260515_085924_chunk8_a100/slurm/%x_%A_%a.out
#SBATCH --error=/scratch/jy03364/MeZO_/runs/next_window_sparse_20260515_085924_chunk8_a100/slurm/%x_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --partition=gpu_p
#SBATCH --gres=gpu:A100:1

set -euo pipefail
cd "/scratch/jy03364/MeZO_"
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate "ciao"
export RESULT_ROOT="/scratch/jy03364/MeZO_/runs/next_window_sparse_20260515_085924_chunk8_a100"
export MODE="all"
export DATA_SEED="16"
export BS="64"
bash scripts/submit_next_window_sparse.sh --run-case "${SLURM_ARRAY_TASK_ID}"
