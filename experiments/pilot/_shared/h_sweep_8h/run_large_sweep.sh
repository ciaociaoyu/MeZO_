#!/bin/bash

set -uo pipefail

SCRATCH_ROOT="/scratch/jy03364/MeZO_"
EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-$PWD}"
LARGE_ROOT="${SCRATCH_ROOT}/large_models"
HSWEEP_HELPERS="${SCRATCH_ROOT}/experiments/h_sweep_helpers.sh"
SHARED_ROOT="${SCRATCH_ROOT}/experiments/pilot/_shared/h_sweep_8h"
NAN_GUARD="${SCRATCH_ROOT}/experiments/main/_shared/h_sweep_14h/nan_guard.py"
NAN_GUARD_LIMIT="${NAN_GUARD_LIMIT:-100}"
NAN_GUARD_EXIT_CODE=86
RANDOM_GUARD_EXIT_CODE=87
PROBE_GUARD_EXIT_CODE=88
ZO_QUANTIZATION_ALIAS="${ZO_QUANTIZATION_ALIAS:-int8}"
PRECISION_LABEL="${PRECISION_LABEL:-${ZO_QUANTIZATION_ALIAS}}"
case "${ZO_QUANTIZATION_ALIAS}" in
  int8)
    QUZO_BITS=8
    ;;
  int4)
    QUZO_BITS=4
    ;;
  *)
    echo "Unsupported ZO_QUANTIZATION_ALIAS=${ZO_QUANTIZATION_ALIAS}; expected int8 or int4" >&2
    exit 1
    ;;
esac

: "${VARIANT:?}"
: "${SEED:?}"
: "${TASK_NAME:?}"
: "${TASK_KEY:?}"
: "${MODEL_NAME:?}"
: "${NUM_EPOCHS:?}"

MAX_STEPS="${MAX_STEPS:-10000}"
EVAL_STEPS="${EVAL_STEPS:-1000}"
LOGGING_STEPS="${LOGGING_STEPS:-10}"
ZO_PROBE_EVERY="${ZO_PROBE_EVERY:-200}"
ZO_PROBE_NUM_SEEDS="${ZO_PROBE_NUM_SEEDS:-16}"
ZO_PROBE_HEALTH_GUARD_ENABLED="${ZO_PROBE_HEALTH_GUARD_ENABLED:-True}"
ZO_PROBE_HEALTH_GUARD_STEP="${ZO_PROBE_HEALTH_GUARD_STEP:-2000}"
ZO_PROBE_HEALTH_GUARD_MAX_BAD_PROBES="${ZO_PROBE_HEALTH_GUARD_MAX_BAD_PROBES:-3}"
RANDOM_PREDICTION_GUARD_ENABLED="${RANDOM_PREDICTION_GUARD_ENABLED:-True}"
RANDOM_PREDICTION_GUARD_STEP="${RANDOM_PREDICTION_GUARD_STEP:-2000}"
RANDOM_PREDICTION_GUARD_ACC_TOLERANCE="${RANDOM_PREDICTION_GUARD_ACC_TOLERANCE:-0.05}"
RANDOM_PREDICTION_GUARD_LOSS_TOLERANCE="${RANDOM_PREDICTION_GUARD_LOSS_TOLERANCE:-0.03}"
RANDOM_PREDICTION_GUARD_BAD_LOSS_EXCESS="${RANDOM_PREDICTION_GUARD_BAD_LOSS_EXCESS:-0.5}"
RANDOM_PREDICTION_GUARD_RECENT_EVALS="${RANDOM_PREDICTION_GUARD_RECENT_EVALS:-2}"
RANDOM_PREDICTION_GUARD_MIN_LOSS_DROP="${RANDOM_PREDICTION_GUARD_MIN_LOSS_DROP:-0.05}"
RANDOM_PREDICTION_GUARD_MIN_ACC_GAIN="${RANDOM_PREDICTION_GUARD_MIN_ACC_GAIN:-0.02}"
BS="${BS:-16}"
LR="${LR:-1e-6}"
DATASET_MODE="${DATASET_MODE:-full}"
K="${K:-16}"
DATA_SEED="${DATA_SEED:-${SEED}}"
TRAIN_SET_SEED="${TRAIN_SET_SEED:-${SEED}}"
SUMMARY_FILE="${EXPERIMENT_ROOT}/results/${VARIANT}/${MODEL_NAME}/${TASK_KEY}/summary.jsonl"
MANIFEST_FILE="${EXPERIMENT_ROOT}/results/${VARIANT}/${MODEL_NAME}/${TASK_KEY}/manifest.jsonl"

mkdir -p "${EXPERIMENT_ROOT}"
cd "${EXPERIMENT_ROOT}"
[[ -f "${HSWEEP_HELPERS}" ]] || { echo "[path-check] Missing shared helper: ${HSWEEP_HELPERS}" >&2; exit 2; }
source "${HSWEEP_HELPERS}"
hsweep_require_file "${SHARED_ROOT}/h_values.sh" "pilot 8h h_values"
hsweep_require_file "${NAN_GUARD}" "shared nan_guard"
hsweep_require_file "${LARGE_ROOT}/run.py" "large_models runner"
source "${SHARED_ROOT}/h_values.sh"
if [[ -n "${H_VALUES_OVERRIDE:-}" ]]; then
  read -r -a H_VALUES <<< "${H_VALUES_OVERRIDE}"
fi

mkdir -p "${EXPERIMENT_ROOT}/logs" "${EXPERIMENT_ROOT}/results/${VARIANT}/${MODEL_NAME}/${TASK_KEY}"

append_run_summary() {
  local run_summary_path="$1"
  local h_value="$2"
  python - "${run_summary_path}" "${SUMMARY_FILE}" "${TASK_NAME}" "${MODEL_NAME}" "${PRECISION_LABEL}" "${h_value}" "${SEED}" "${DATASET_MODE}" "${QUZO_BITS}" "${VARIANT}" <<'PY'
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
os.makedirs(os.path.dirname(summary_file), exist_ok=True)
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
  python - "${SUMMARY_FILE}" "${TASK_NAME}" "${MODEL_NAME}" "${PRECISION_LABEL}" "${h_value}" "${SEED}" "${DATASET_MODE}" "${QUZO_BITS}" "${VARIANT}" "${status}" "${exit_code}" "${run_root}" "${run_log}" "${run_err}" <<'PY'
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
os.makedirs(os.path.dirname(summary_file), exist_ok=True)
lock_path = summary_file + ".lock"
with open(lock_path, "w", encoding="utf-8") as lock_file:
    if fcntl is not None:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
    with open(summary_file, "a", encoding="utf-8") as out_file:
        out_file.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
PY
}

append_manifest_row() {
  local h_value="$1"
  local status="$2"
  local exit_code="$3"
  local run_root="$4"
  local run_log="$5"
  local run_err="$6"
  local command_desc="$7"
  python - "${MANIFEST_FILE}" "${TASK_NAME}" "${h_value}" "${command_desc}" "${run_root}" "${status}" "${exit_code}" "${run_log}" "${run_err}" <<'PY'
import json
import os
import sys

try:
    import fcntl
except ImportError:
    fcntl = None

manifest_file, task, h_value, command_desc, output_dir, status, exit_code, stdout_log, stderr_log = sys.argv[1:]
record = {
    "task": task,
    "h": h_value,
    "command": command_desc,
    "output_dir": output_dir,
    "status": status,
    "exit_code": int(exit_code),
    "stdout_log": stdout_log,
    "stderr_log": stderr_log,
}
os.makedirs(os.path.dirname(manifest_file), exist_ok=True)
lock_path = manifest_file + ".lock"
with open(lock_path, "w", encoding="utf-8") as lock_file:
    if fcntl is not None:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
    with open(manifest_file, "a", encoding="utf-8") as out_file:
        out_file.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
PY
}

for H in "${H_VALUES[@]}"; do
  RUN_DIR="${EXPERIMENT_ROOT}/results/${VARIANT}/${MODEL_NAME}/${TASK_KEY}/h_${H}/seed_${SEED}"
  LOG_DIR="${EXPERIMENT_ROOT}/logs/${VARIANT}/${MODEL_NAME}/${TASK_KEY}/h_${H}/seed_${SEED}"
  RUN_LOG="${LOG_DIR}/train.log"
  RUN_ERR="${LOG_DIR}/train.err"
  RUN_SUMMARY_PATH="${RUN_DIR}/run_summary.json"

  if hsweep_run_completed "${RUN_SUMMARY_PATH}" "${SUMMARY_FILE}" "${H}"; then
    echo "[skip] ${VARIANT} ${MODEL_NAME} ${TASK_KEY} h=${H} already completed"
    continue
  fi

  hsweep_drop_h_rows "${SUMMARY_FILE}" "${H}"
  hsweep_drop_h_rows "${MANIFEST_FILE}" "${H}"
  hsweep_cleanup_paths "${RUN_DIR}" "${LOG_DIR}"
  mkdir -p "${RUN_DIR}" "${LOG_DIR}"

  CMD=(
    python "${LARGE_ROOT}/run.py"
    --model_name "${MODEL_NAME}"
    --task_name "${TASK_NAME}"
    --output_dir "${RUN_DIR}"
    --overwrite_output_dir
    --tag "hsweep8h-${VARIANT}-${TASK_KEY}-h${H}-seed${SEED}"
    --dataset_mode "${DATASET_MODE}"
    --num_k "${K}"
    --data_seed "${DATA_SEED}"
    --train_set_seed "${TRAIN_SET_SEED}"
    --trainer zo
    --load_float16
    --zo_quantization "${ZO_QUANTIZATION_ALIAS}"
    --train_as_classification
    --learning_rate "${LR}"
    --zo_eps "${H}"
    --num_train_epochs "${NUM_EPOCHS}"
    --per_device_train_batch_size "${BS}"
    --gradient_accumulation_steps 1
    --lr_scheduler_type constant
    --evaluation_strategy steps
    --save_strategy no
    --eval_steps "${EVAL_STEPS}"
    --logging_steps "${LOGGING_STEPS}"
    --zo_probe_every "${ZO_PROBE_EVERY}"
    --zo_probe_num_seeds "${ZO_PROBE_NUM_SEEDS}"
    --zo_probe_log_csv
    --zo_probe_health_guard_enabled "${ZO_PROBE_HEALTH_GUARD_ENABLED}"
    --zo_probe_health_guard_step "${ZO_PROBE_HEALTH_GUARD_STEP}"
    --zo_probe_health_guard_max_bad_probes "${ZO_PROBE_HEALTH_GUARD_MAX_BAD_PROBES}"
    --random_prediction_guard_enabled "${RANDOM_PREDICTION_GUARD_ENABLED}"
    --random_prediction_guard_step "${RANDOM_PREDICTION_GUARD_STEP}"
    --random_prediction_guard_acc_tolerance "${RANDOM_PREDICTION_GUARD_ACC_TOLERANCE}"
    --random_prediction_guard_loss_tolerance "${RANDOM_PREDICTION_GUARD_LOSS_TOLERANCE}"
    --random_prediction_guard_bad_loss_excess "${RANDOM_PREDICTION_GUARD_BAD_LOSS_EXCESS}"
    --random_prediction_guard_recent_evals "${RANDOM_PREDICTION_GUARD_RECENT_EVALS}"
    --random_prediction_guard_min_loss_drop "${RANDOM_PREDICTION_GUARD_MIN_LOSS_DROP}"
    --random_prediction_guard_min_acc_gain "${RANDOM_PREDICTION_GUARD_MIN_ACC_GAIN}"
    --max_steps "${MAX_STEPS}"
  )
  if [[ -n "${NUM_DEV:-}" ]]; then
    CMD+=(--num_dev "${NUM_DEV}")
  fi
  if [[ -n "${NUM_EVAL:-}" ]]; then
    CMD+=(--num_eval "${NUM_EVAL}")
  fi
  if [[ -n "${SPARSE_RATIO:-}" ]]; then
    CMD+=(
      --sparse_ratio "${SPARSE_RATIO}"
      --sparse_mask_strategy "${SPARSE_MASK_STRATEGY:-percentile_per_layer}"
      --sparse_scope "${SPARSE_SCOPE:-trainable_only}"
      --sparse_log_active_fraction "${SPARSE_LOG_ACTIVE_FRACTION:-True}"
    )
  fi
  COMMAND_DESC="$(printf '%q ' "${CMD[@]}")"

  python "${NAN_GUARD}" \
    --cwd "${EXPERIMENT_ROOT}" \
    --stdout-log "${RUN_LOG}" \
    --stderr-log "${RUN_ERR}" \
    --max-consecutive-nan "${NAN_GUARD_LIMIT}" \
    -- \
    "${CMD[@]}"
  run_status=$?

  if [[ ${run_status} -eq 0 ]]; then
    if [[ -f "${RUN_SUMMARY_PATH}" ]]; then
      append_run_summary "${RUN_SUMMARY_PATH}" "${H}"
      status_label="completed"
    else
      append_failure_summary "${H}" "missing_run_summary" 1 "${RUN_DIR}" "${RUN_LOG}" "${RUN_ERR}"
      status_label="missing_run_summary"
      run_status=1
    fi
  elif [[ ${run_status} -eq ${NAN_GUARD_EXIT_CODE} ]]; then
    append_failure_summary "${H}" "skipped_nan_guard" "${run_status}" "${RUN_DIR}" "${RUN_LOG}" "${RUN_ERR}"
    status_label="skipped_nan_guard"
  elif [[ ${run_status} -eq ${RANDOM_GUARD_EXIT_CODE} ]]; then
    append_failure_summary "${H}" "skipped_random_prediction_guard" "${run_status}" "${RUN_DIR}" "${RUN_LOG}" "${RUN_ERR}"
    status_label="skipped_random_prediction_guard"
  elif [[ ${run_status} -eq ${PROBE_GUARD_EXIT_CODE} ]]; then
    append_failure_summary "${H}" "skipped_probe_health_guard" "${run_status}" "${RUN_DIR}" "${RUN_LOG}" "${RUN_ERR}"
    status_label="skipped_probe_health_guard"
  else
    append_failure_summary "${H}" "failed" "${run_status}" "${RUN_DIR}" "${RUN_LOG}" "${RUN_ERR}"
    status_label="failed"
  fi

  append_manifest_row "${H}" "${status_label}" "${run_status}" "${RUN_DIR}" "${RUN_LOG}" "${RUN_ERR}" "${COMMAND_DESC}"
done
