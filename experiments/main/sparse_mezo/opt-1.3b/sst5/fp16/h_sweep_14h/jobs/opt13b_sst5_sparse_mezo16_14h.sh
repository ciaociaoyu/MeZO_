#!/bin/bash
#SBATCH --job-name=hsweep14h_sparsemezo16_opt13b_sst5
#SBATCH --partition=gpu_p
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=160G
#SBATCH --gres=gpu:H100:1
#SBATCH --time=168:00:00
#SBATCH --chdir=/scratch/jy03364/MeZO_/experiments/main/sparse_mezo/opt-1.3b/sst5/fp16/h_sweep_14h
#SBATCH --output=/scratch/jy03364/MeZO_/experiments/main/sparse_mezo/opt-1.3b/sst5/fp16/h_sweep_14h/logs/slurm_%x_%j.out

set -euo pipefail

set +u
source /home/jy03364/miniconda3/etc/profile.d/conda.sh
conda activate mezo-env
set -u

SCRATCH_ROOT="/scratch/jy03364/MeZO_"
EXPERIMENT_ROOT="/scratch/jy03364/MeZO_/experiments/main/sparse_mezo/opt-1.3b/sst5/fp16/h_sweep_14h"
LARGE_ROOT="${SCRATCH_ROOT}/large_models"
HSWEEP_HELPERS="${SCRATCH_ROOT}/experiments/h_sweep_helpers.sh"
VARIANT="sparse_mezo16"
QUZO_BITS=16
SPARSE_RATIO=0.25
SPARSE_MASK_STRATEGY="percentile_per_layer"
SPARSE_SCOPE="trainable_only"
SPARSE_LOG_ACTIVE_FRACTION="True"
NAN_GUARD="${EXPERIMENT_ROOT}/nan_guard.py"
NAN_GUARD_LIMIT=1
NAN_GUARD_EXIT_CODE=86

cd "${EXPERIMENT_ROOT}"
source "${HSWEEP_HELPERS}"
source "/scratch/jy03364/MeZO_/experiments/main/_shared/h_sweep_14h/h_values.sh"

SEED=42
TASK_NAME="SST5"
TASK_KEY="sst5"
MODEL_NAME="opt-1.3b"
RESULT_ROOT_BASE="${EXPERIMENT_ROOT}/results/${VARIANT}/${MODEL_NAME}/${TASK_KEY}"
SUMMARY_FILE="${RESULT_ROOT_BASE}/summary.jsonl"

mkdir -p "${EXPERIMENT_ROOT}/logs" "${RESULT_ROOT_BASE}"

append_run_summary() {
  local run_summary_path="$1"
  local h_value="$2"
  if [[ ! -f "${run_summary_path}" ]]; then
    echo "Missing run summary: ${run_summary_path}"
    return 1
  fi
  python - "${run_summary_path}" "${SUMMARY_FILE}" "${TASK_NAME}" "${MODEL_NAME}" "fp16" "${h_value}" "${SEED}" "full" "${QUZO_BITS}" "${VARIANT}" <<'PY'
import json
import os
import sys

try:
    import fcntl
except ImportError:
    fcntl = None

run_summary_path, summary_file, task, model, precision, h_value, seed, dataset_mode, qbits, variant = sys.argv[1:]
with open(run_summary_path, "r", encoding="utf-8") as f:
    record = json.load(f)

record.update({
    "task": task,
    "model": model,
    "precision": precision,
    "h": h_value,
    "seed": int(seed),
    "dataset_mode": dataset_mode,
    "zo_quantization_bits": int(qbits),
    "variant": variant,
    "status": "completed",
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

append_failure_summary() {
  local h_value="$1"
  local status="$2"
  local exit_code="$3"
  local run_root="$4"
  local run_log="$5"
  local run_err="$6"
  python - "${SUMMARY_FILE}" "${TASK_NAME}" "${MODEL_NAME}" "fp16" "${h_value}" "${SEED}" "full" "${QUZO_BITS}" "${VARIANT}" "${status}" "${exit_code}" "${run_root}" "${run_log}" "${run_err}" <<'PY'
import json
import os
import sys

try:
    import fcntl
except ImportError:
    fcntl = None

summary_file, task, model, precision, h_value, seed, dataset_mode, qbits, variant, status, exit_code, run_root, run_log, run_err = sys.argv[1:]
record = {
    "task": task,
    "model": model,
    "precision": precision,
    "h": h_value,
    "seed": int(seed),
    "dataset_mode": dataset_mode,
    "zo_quantization_bits": int(qbits),
    "variant": variant,
    "status": status,
    "exit_code": int(exit_code),
    "paths": {
        "run_root": run_root,
        "stdout_log": run_log,
        "stderr_log": run_err,
    },
}

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
  RUN_DIR="${RESULT_ROOT_BASE}/h_${H}/seed_${SEED}"
  LOG_DIR="${EXPERIMENT_ROOT}/logs/${VARIANT}/${MODEL_NAME}/${TASK_KEY}/h_${H}/seed_${SEED}"
  RUN_LOG="${LOG_DIR}/train.log"
  RUN_ERR="${LOG_DIR}/train.err"
  RUN_SUMMARY_PATH="${RUN_DIR}/run_summary.json"

  if hsweep_run_completed "${RUN_SUMMARY_PATH}" "${SUMMARY_FILE}" "${H}"; then
    echo "[skip] ${VARIANT} ${MODEL_NAME} ${TASK_KEY} h=${H} already completed"
    continue
  fi

  hsweep_drop_h_rows "${SUMMARY_FILE}" "${H}"
  hsweep_cleanup_paths "${RUN_DIR}" "${LOG_DIR}"
  mkdir -p "${RUN_DIR}" "${LOG_DIR}"

  python "${NAN_GUARD}" \
    --cwd "${EXPERIMENT_ROOT}" \
    --stdout-log "${RUN_LOG}" \
    --stderr-log "${RUN_ERR}" \
    --max-consecutive-nan "${NAN_GUARD_LIMIT}" \
    -- \
    python "${LARGE_ROOT}/run.py" \
      --model_name "${MODEL_NAME}" \
      --task_name "${TASK_NAME}" \
      --output_dir "${RUN_DIR}" \
      --overwrite_output_dir \
      --tag "hsweep14h-${VARIANT}-${TASK_KEY}-h${H}-seed${SEED}" \
      --dataset_mode full \
      --num_k 16 \
      --data_seed "${SEED}" \
      --train_set_seed "${SEED}" \
      --trainer zo \
      --load_float16 \
      --zo_quantization_bits "${QUZO_BITS}" \
      --sparse_ratio "${SPARSE_RATIO}" \
      --sparse_mask_strategy "${SPARSE_MASK_STRATEGY}" \
      --sparse_scope "${SPARSE_SCOPE}" \
      --sparse_log_active_fraction "${SPARSE_LOG_ACTIVE_FRACTION}" \
      --train_as_classification \
      --learning_rate 1e-6 \
      --zo_eps "${H}" \
      --num_train_epochs 5 \
      --per_device_train_batch_size 16 \
      --gradient_accumulation_steps 1 \
      --lr_scheduler_type constant \
      --evaluation_strategy steps \
      --save_strategy no \
      --eval_steps 5000 \
      --logging_steps 10 \
      --zo_probe_every 100 \
      --zo_probe_num_seeds 16 \
      --zo_probe_log_csv
  run_status=$?

  if [[ ${run_status} -eq 0 ]]; then
    if [[ -f "${RUN_SUMMARY_PATH}" ]]; then
      append_run_summary "${RUN_SUMMARY_PATH}" "${H}"
    else
      append_failure_summary "${H}" "missing_run_summary" 1 "${RUN_DIR}" "${RUN_LOG}" "${RUN_ERR}"
    fi
  elif [[ ${run_status} -eq ${NAN_GUARD_EXIT_CODE} ]]; then
    append_failure_summary "${H}" "skipped_nan_guard" "${run_status}" "${RUN_DIR}" "${RUN_LOG}" "${RUN_ERR}"
  else
    append_failure_summary "${H}" "failed" "${run_status}" "${RUN_DIR}" "${RUN_LOG}" "${RUN_ERR}"
  fi
done
