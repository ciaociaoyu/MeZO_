#!/bin/bash
#SBATCH --job-name=v10-med
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=72:00:00
#SBATCH --gres=gpu:H100:1
#SBATCH --array=0-5%6
#SBATCH --output=v10_medium_supplement_results_20260627/jobs/%x_%A_%a.out
#SBATCH --error=v10_medium_supplement_results_20260627/jobs/%x_%A_%a.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/scratch/jy03364/MeZO_}"
OUT_DIR="${OUT_DIR:-v10_medium_supplement_results_20260627}"
LANE_ID="${SLURM_ARRAY_TASK_ID:-0}"
CONDA_ENV_HP="${CONDA_ENV_HP:-ciao}"
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

HP_MANIFEST="$OUT_DIR/manifests/high_precision_lane${LANE_ID}.csv"
if [[ -s "$HP_MANIFEST" ]] && [[ "$(wc -l < "$HP_MANIFEST")" -gt 1 ]]; then
  source_conda
  conda activate "$CONDA_ENV_HP"
  echo "[lane $LANE_ID] running high-precision manifest $HP_MANIFEST"
  EXP_ROOT="$OUT_DIR/raw_runs/high_precision_sst5_fp32_multiseed" \
  LANE_ID="$LANE_ID" \
  LANE_MANIFEST="$HP_MANIFEST" \
  CONDA_ENV="$CONDA_ENV_HP" \
  PROBE_DIRECTIONS="${PROBE_DIRECTIONS:-4}" \
  bash scripts/run_latest_main_lane.sh
else
  echo "[lane $LANE_ID] no high-precision rows"
fi

LB_MANIFEST="$OUT_DIR/manifests/lowbit_lane${LANE_ID}.csv"
if [[ -s "$LB_MANIFEST" ]] && [[ "$(wc -l < "$LB_MANIFEST")" -gt 1 ]]; then
  source_conda
  conda activate "$CONDA_ENV_LB"
  export DATALOADER_SHUFFLE=True
  export REQUESTED_GPU_TYPE=H100
  export FALLBACK_USED=0
  echo "[lane $LANE_ID] running low-bit manifest $LB_MANIFEST"
  python tools/rtnclip_roberta_sst5_batch.py \
    --output_root "$OUT_DIR/raw_runs/roberta_int4_sparse_prefix_multiseed" \
    --manifest "$LB_MANIFEST" \
    run-manifest
else
  echo "[lane $LANE_ID] no low-bit rows"
fi

echo "job_id=${SLURM_JOB_ID:-local} lane=${LANE_ID} end=$(date -Is)" | tee "$OUT_DIR/jobs/lane${LANE_ID}_end.txt"
