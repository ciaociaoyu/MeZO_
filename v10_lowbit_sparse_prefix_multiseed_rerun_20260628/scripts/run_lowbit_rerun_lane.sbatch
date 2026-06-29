#!/bin/bash
#SBATCH --job-name=v10-lb-rerun
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=72:00:00
#SBATCH --gres=gpu:H100:1
#SBATCH --array=0-5%6
#SBATCH --output=v10_lowbit_sparse_prefix_multiseed_rerun_20260628/jobs/%x_%A_%a.out
#SBATCH --error=v10_lowbit_sparse_prefix_multiseed_rerun_20260628/jobs/%x_%A_%a.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/scratch/jy03364/MeZO_}"
OUT_DIR="${OUT_DIR:-v10_lowbit_sparse_prefix_multiseed_rerun_20260628}"
LANE_ID="${SLURM_ARRAY_TASK_ID:-0}"
CONDA_ENV_LB="${CONDA_ENV_LB:-mezo-env}"

cd "$REPO_ROOT"
mkdir -p "$OUT_DIR/jobs" "$OUT_DIR/raw_runs/slurm_environment"

echo "job_id=${SLURM_JOB_ID:-local} lane=${LANE_ID} host=$(hostname) start=$(date -Is)" | tee "$OUT_DIR/jobs/lane${LANE_ID}_start.txt"
nvidia-smi | tee "$OUT_DIR/raw_runs/slurm_environment/lane${LANE_ID}_nvidia_smi.txt" || true
git rev-parse HEAD | tee "$OUT_DIR/raw_runs/slurm_environment/lane${LANE_ID}_git_commit.txt" || true
env | sort > "$OUT_DIR/raw_runs/slurm_environment/lane${LANE_ID}_env.txt"

source_conda() {
  if [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
  elif [[ -f "$HOME/.conda/etc/profile.d/conda.sh" ]]; then
    source "$HOME/.conda/etc/profile.d/conda.sh"
  elif [[ -f "/apps/conda/etc/profile.d/conda.sh" ]]; then
    source "/apps/conda/etc/profile.d/conda.sh"
  fi
}

LB_MANIFEST="$OUT_DIR/manifests/lowbit_lane${LANE_ID}.csv"
if [[ -s "$LB_MANIFEST" ]] && [[ "$(wc -l < "$LB_MANIFEST")" -gt 1 ]]; then
  source_conda
  conda activate "$CONDA_ENV_LB"
  export DATALOADER_SHUFFLE=True
  export REQUESTED_GPU_TYPE=H100
  export FALLBACK_USED=0
  echo "[lane $LANE_ID] running low-bit rerun manifest $LB_MANIFEST"
  python tools/rtnclip_roberta_sst5_batch.py \
    --output_root "$OUT_DIR/raw_runs/roberta_int4_sparse_prefix_multiseed" \
    --manifest "$LB_MANIFEST" \
    run-manifest
else
  echo "[lane $LANE_ID] no low-bit rows"
fi

echo "job_id=${SLURM_JOB_ID:-local} lane=${LANE_ID} end=$(date -Is)" | tee "$OUT_DIR/jobs/lane${LANE_ID}_end.txt"

