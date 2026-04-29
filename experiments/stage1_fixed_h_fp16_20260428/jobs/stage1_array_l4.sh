#!/usr/bin/env bash
#SBATCH --job-name=stage1-fp16-l4
#SBATCH --partition=gpu_p
#SBATCH --ntasks=1
#SBATCH --gres=gpu:L4:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/jy03364/MeZO_/experiments/stage1_fixed_h_fp16_20260428/logs/slurm_%A_%a.out
#SBATCH --error=/scratch/jy03364/MeZO_/experiments/stage1_fixed_h_fp16_20260428/logs/slurm_%A_%a.err

set -euo pipefail

REPO_ROOT="/scratch/jy03364/MeZO_"
CASE_INDEX="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"

cd "$REPO_ROOT"
bash "$REPO_ROOT/experiments/stage1_fixed_h_fp16_20260428/jobs/run_stage1_case.sh" "$CASE_INDEX"
