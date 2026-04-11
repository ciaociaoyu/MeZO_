#!/bin/bash
#SBATCH --job-name=hsweep14h_roberta_mnli
#SBATCH --partition=gpu_p
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=120G
#SBATCH --gres=gpu:H100:1
#SBATCH --time=72:00:00
#SBATCH --output=/Users/jichaoyu/Documents/GitHub/MeZO/experiments/h_sweep_14h/logs/slurm_%x_%j.out

set -euo pipefail

ml jq
source ~/.bashrc
conda activate ciao

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${EXPERIMENT_ROOT}/../.." && pwd)"
source "${EXPERIMENT_ROOT}/h_values.sh"

SEED=16
TASK_NAME="MNLI"
TASK_KEY="mnli"
MODEL_KEY="roberta-large"
SUMMARY_FILE="${EXPERIMENT_ROOT}/results/${MODEL_KEY}/${TASK_KEY}/summary.jsonl"

mkdir -p "${EXPERIMENT_ROOT}/logs" "${EXPERIMENT_ROOT}/results/${MODEL_KEY}/${TASK_KEY}"

append_run_summary() {
  local run_summary_path="$1"
  local h_value="$2"
  if [[ ! -f "${run_summary_path}" ]]; then
    echo "Missing run summary: ${run_summary_path}"
    return 1
  fi
  python - "${run_summary_path}" "${SUMMARY_FILE}" "${TASK_NAME}" "${MODEL_KEY}" "fp16" "${h_value}" "${SEED}" "full" <<'PY'
import json
import os
import sys

try:
    import fcntl
except ImportError:
    fcntl = None

run_summary_path, summary_file, task, model, precision, h_value, seed, dataset_mode = sys.argv[1:]
with open(run_summary_path, "r", encoding="utf-8") as f:
    record = json.load(f)

record.update({
    "task": task,
    "model": model,
    "precision": precision,
    "h": h_value,
    "seed": int(seed),
    "dataset_mode": dataset_mode,
})

summary_dir = os.path.dirname(summary_file)
if summary_dir:
    os.makedirs(summary_dir, exist_ok=True)

lock_path = summary_file + ".lock"
with open(lock_path, "w", encoding="utf-8") as lock_file:
    if fcntl is not None:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
    with open(summary_file, "a", encoding="utf-8") as out_file:
        out_file.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
PY
}

cd "${REPO_ROOT}/medium_models"
mkdir -p jobs

for H in "${H_VALUES[@]}"; do
  RUN_ROOT="${EXPERIMENT_ROOT}/results/${MODEL_KEY}/${TASK_KEY}/h_${H}"
  JOB_NAME="run"
  RUN_SUMMARY_PATH="${RUN_ROOT}/${JOB_NAME}/seed${SEED}/run_summary.json"
  LOG_DIR="${EXPERIMENT_ROOT}/logs/${MODEL_KEY}/${TASK_KEY}/h_${H}/seed_${SEED}"
  RUN_LOG="${LOG_DIR}/train.log"
  RUN_ERR="${LOG_DIR}/train.err"

  mkdir -p "${RUN_ROOT}" "${LOG_DIR}"

  TASK="${TASK_NAME}" \
  K=16 \
  SEED="${SEED}" \
  DATA_SEED="${SEED}" \
  DATASET_MODE=full \
  BS=32 \
  LR=1e-6 \
  WD=0 \
  STEP=50000 \
  EVAL_STEP=5000 \
  MODEL=roberta-large \
  USE_H=False \
  USE_C=False \
  DATALOADER_SHUFFLE=False \
  EPS="${H}" \
  EXTRA_TAG="hsweep14h-${TASK_KEY}-fp16-h${H}" \
  bash mezo.sh \
    --result_root "${RUN_ROOT}" \
    --job_name "${JOB_NAME}" \
    --dataset_mode full \
    --zo_two_point_precision fp16 \
    --zo_probe_every 200 \
    --zo_probe_num_seeds 16 \
    --zo_probe_log_csv True \
    >"${RUN_LOG}" 2>"${RUN_ERR}"

  append_run_summary "${RUN_SUMMARY_PATH}" "${H}"
done
