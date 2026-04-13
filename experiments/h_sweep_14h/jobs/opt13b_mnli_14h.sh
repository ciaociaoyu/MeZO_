#!/bin/bash
#SBATCH --job-name=hsweep14h_opt13b_mnli
#SBATCH --partition=gpu_p
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=160G
#SBATCH --gres=gpu:H100:1
#SBATCH --time=168:00:00
#SBATCH --chdir=/scratch/jy03364/MeZO_/experiments/h_sweep_14h
#SBATCH --output=logs/slurm_%x_%j.out

set -eo pipefail

set +u
source /home/jy03364/miniconda3/etc/profile.d/conda.sh
conda activate mezo-env
set -u

SCRATCH_ROOT="/scratch/jy03364/MeZO_"
EXPERIMENT_ROOT="${SCRATCH_ROOT}/experiments/h_sweep_14h"
LARGE_ROOT="${SCRATCH_ROOT}/large_models"

cd "${EXPERIMENT_ROOT}"
source "${EXPERIMENT_ROOT}/h_values.sh"

SEED=42
TASK_NAME="MNLI"
TASK_KEY="mnli"
MODEL_NAME="opt-1.3b"
SUMMARY_FILE="${EXPERIMENT_ROOT}/results/${MODEL_NAME}/${TASK_KEY}/summary.jsonl"

mkdir -p "${EXPERIMENT_ROOT}/logs" "${EXPERIMENT_ROOT}/results/${MODEL_NAME}/${TASK_KEY}"

append_run_summary() {
  local run_summary_path="$1"
  local h_value="$2"
  if [[ ! -f "${run_summary_path}" ]]; then
    echo "Missing run summary: ${run_summary_path}"
    return 1
  fi
  python - "${run_summary_path}" "${SUMMARY_FILE}" "${TASK_NAME}" "${MODEL_NAME}" "fp16" "${h_value}" "${SEED}" "full" <<'PY'
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

for H in "${H_VALUES[@]}"; do
  RUN_DIR="${EXPERIMENT_ROOT}/results/${MODEL_NAME}/${TASK_KEY}/h_${H}/seed_${SEED}"
  LOG_DIR="${EXPERIMENT_ROOT}/logs/${MODEL_NAME}/${TASK_KEY}/h_${H}/seed_${SEED}"
  RUN_LOG="${LOG_DIR}/train.log"
  RUN_ERR="${LOG_DIR}/train.err"
  RUN_SUMMARY_PATH="${RUN_DIR}/run_summary.json"

  mkdir -p "${RUN_DIR}" "${LOG_DIR}"

  python "${LARGE_ROOT}/run.py" \
    --model_name "${MODEL_NAME}" \
    --task_name "${TASK_NAME}" \
    --output_dir "${RUN_DIR}" \
    --overwrite_output_dir \
    --tag "hsweep14h-${TASK_KEY}-h${H}-seed${SEED}" \
    --dataset_mode full \
    --num_k 16 \
    --data_seed "${SEED}" \
    --train_set_seed "${SEED}" \
    --num_dev 0 \
    --trainer zo \
    --load_float16 \
    --train_as_classification \
    --learning_rate 1e-6 \
    --zo_eps "${H}" \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 1 \
    --lr_scheduler_type constant \
    --evaluation_strategy steps \
    --save_strategy steps \
    --eval_steps 5000 \
    --save_steps 5000 \
    --save_total_limit 1 \
    --load_best_model_at_end \
    --logging_steps 10 \
    --zo_probe_every 200 \
    --zo_probe_num_seeds 16 \
    --zo_probe_log_csv \
    >"${RUN_LOG}" 2>"${RUN_ERR}"

  append_run_summary "${RUN_SUMMARY_PATH}" "${H}"
done
