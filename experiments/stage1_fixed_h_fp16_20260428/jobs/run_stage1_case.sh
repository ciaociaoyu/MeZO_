#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <manifest-index>" >&2
  exit 2
fi

CASE_INDEX="$1"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
EXPERIMENT_ROOT="$REPO_ROOT/experiments/stage1_fixed_h_fp16_20260428"
MANIFEST="$EXPERIMENT_ROOT/jobs/stage1_manifest.tsv"
NAN_GUARD="$REPO_ROOT/experiments/main/_shared/h_sweep_14h/nan_guard.py"

if [[ ! -f "$NAN_GUARD" ]]; then
  echo "Missing nan guard: $NAN_GUARD" >&2
  exit 2
fi

ROW="$(awk -F '\t' -v idx="$CASE_INDEX" 'NR > 1 && $1 == idx {print; found=1} END {if (!found) exit 1}' "$MANIFEST")" || {
  echo "No manifest row for index $CASE_INDEX" >&2
  exit 2
}

IFS=$'\t' read -r index model_family model_key model_name task_key task_name method h env_name batch_size <<< "$ROW"

SEED="${SEED:-16}"
DATA_SEED="${DATA_SEED:-16}"
TRAIN_SET_SEED="${TRAIN_SET_SEED:-16}"
MAX_STEPS="${MAX_STEPS:-5000}"
LR="${LR:-1e-6}"
WD="${WD:-0}"
EVAL_STEPS="${EVAL_STEPS:-1000}"
LOGGING_STEPS="${LOGGING_STEPS:-10}"
PROBE_EVERY="${PROBE_EVERY:-200}"
PROBE_NUM_SEEDS="${PROBE_NUM_SEEDS:-16}"
SPARSE_RATIO="${SPARSE_RATIO:-0.25}"
LORA_R="${LORA_R:-8}"
LORA_ALPHA="${LORA_ALPHA:-16}"

RUN_NAME="${index}_${model_key}_${task_key}_${method}_h${h}"
LOG_DIR="$EXPERIMENT_ROOT/logs/$RUN_NAME"
STATUS_DIR="$EXPERIMENT_ROOT/results/status"
mkdir -p "$LOG_DIR" "$STATUS_DIR"

STDOUT_LOG="$LOG_DIR/train.log"
STDERR_LOG="$LOG_DIR/train.err"
STATUS_JSON="$STATUS_DIR/${index}.json"

write_status() {
  local state="$1"
  local exit_code="${2:-0}"
  local output_dir="${3:-}"
  python - "$STATUS_JSON" <<PY
import json
import os
import sys
from datetime import datetime

path = sys.argv[1]
record = {
    "state": "$state",
    "exit_code": int("$exit_code"),
    "index": int("$index"),
    "model_family": "$model_family",
    "model_key": "$model_key",
    "model_name": "$model_name",
    "task_key": "$task_key",
    "task_name": "$task_name",
    "method": "$method",
    "h": "$h",
    "env_name": "$env_name",
    "batch_size": int("$batch_size"),
    "max_steps": int("$MAX_STEPS"),
    "output_dir": "$output_dir",
    "stdout_log": "$STDOUT_LOG",
    "stderr_log": "$STDERR_LOG",
    "updated_at": datetime.now().isoformat(timespec="seconds"),
    "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
}
with open(path, "w", encoding="utf-8") as f:
    json.dump(record, f, indent=2, sort_keys=True)
    f.write("\n")
PY
}

activate_env() {
  if [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
    # shellcheck source=/dev/null
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
  elif [[ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]]; then
    # shellcheck source=/dev/null
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
  else
    echo "Cannot find conda.sh" >&2
    exit 2
  fi
  conda activate "$env_name"
}

build_medium_command() {
  local result_parent="$EXPERIMENT_ROOT/results/$model_key/$task_key/$method/h_${h}"
  local -a method_args=()

  case "$method" in
    mezo)
      method_args=(--zo_method mezo)
      ;;
    sparse_mezo)
      method_args=(
        --zo_method sparse_mezo
        --sparse_ratio "$SPARSE_RATIO"
        --sparse_mask_strategy percentile_per_layer
        --sparse_scope trainable_only
        --sparse_log_active_fraction True
      )
      ;;
    mezo_lora)
      method_args=(
        --zo_method mezo
        --apply_lora
        --lora_r "$LORA_R"
        --lora_alpha "$LORA_ALPHA"
      )
      ;;
    *)
      echo "Unsupported method for medium path: $method" >&2
      exit 2
      ;;
  esac

  export TASK="$task_name"
  export K=16
  export SEED
  export DATA_SEED
  export DATASET_MODE=full
  export FULL_DEV_RATIO=0.1
  export BS="$batch_size"
  export LR
  export WD
  export STEP="$MAX_STEPS"
  export EVAL_STEP="$EVAL_STEPS"
  export MODEL="$model_name"
  export USE_H=False
  export USE_C=False
  export DATALOADER_SHUFFLE=True
  export EPS="$h"
  export EXTRA_TAG="stage1-fixedh-${method}-${task_key}-h${h}"

  MEDIUM_OUT_DIR="$result_parent/run/seed${SEED}"
  CMD=(
    bash "$REPO_ROOT/medium_models/mezo.sh"
    --result_root "$result_parent"
    --job_name run
    --dataset_mode full
    "${method_args[@]}"
    --zo_two_point_precision fp16
    --zo_quantization_bits 16
    --zo_probe_every "$PROBE_EVERY"
    --zo_probe_num_seeds "$PROBE_NUM_SEEDS"
    --zo_probe_log_csv True
    --enable_two_point_h_estimation True
    --h_estimation_active_source fixed
    --two_point_h_refresh_every 50
    --two_point_h_log_csv True
    --save_at_last True
    --save_strategy no
  )
  RUN_CWD="$REPO_ROOT/medium_models"
  OUTPUT_DIR="$MEDIUM_OUT_DIR"
}

build_large_command() {
  local output_dir="$EXPERIMENT_ROOT/results/$model_key/$task_key/$method/h_${h}/seed_${SEED}"
  local -a method_args=()

  case "$method" in
    mezo)
      method_args=(--zo_method mezo)
      ;;
    sparse_mezo)
      method_args=(
        --zo_method sparse_mezo
        --sparse_ratio "$SPARSE_RATIO"
      )
      ;;
    mezo_lora)
      method_args=(
        --zo_method mezo
        --lora
        --lora_r "$LORA_R"
        --lora_alpha "$LORA_ALPHA"
      )
      ;;
    *)
      echo "Unsupported method for large path: $method" >&2
      exit 2
      ;;
  esac

  CMD=(
    python "$REPO_ROOT/large_models/run.py"
    --model_name "$model_name"
    --task_name "$task_name"
    --output_dir "$output_dir"
    --overwrite_output_dir
    --tag "stage1-fixedh-${method}-${task_key}-h${h}"
    --dataset_mode full
    --num_k 16
    --seed "$SEED"
    --data_seed "$DATA_SEED"
    --train_set_seed "$TRAIN_SET_SEED"
    --trainer zo
    --load_float16
    --zo_quantization_bits 16
    "${method_args[@]}"
    --train_as_classification
    --learning_rate "$LR"
    --weight_decay "$WD"
    --zo_eps "$h"
    --max_steps "$MAX_STEPS"
    --per_device_train_batch_size "$batch_size"
    --per_device_eval_batch_size "$batch_size"
    --gradient_accumulation_steps 1
    --lr_scheduler_type constant
    --evaluation_strategy steps
    --eval_steps "$EVAL_STEPS"
    --save_strategy no
    --logging_steps "$LOGGING_STEPS"
    --zo_probe_every "$PROBE_EVERY"
    --zo_probe_num_seeds "$PROBE_NUM_SEEDS"
    --zo_probe_log_csv
    --save_model
  )
  RUN_CWD="$REPO_ROOT/large_models"
  OUTPUT_DIR="$output_dir"
}

activate_env

case "$model_family" in
  medium)
    build_medium_command
    ;;
  large)
    build_large_command
    ;;
  *)
    echo "Unsupported model_family: $model_family" >&2
    exit 2
    ;;
esac

write_status running 0 "$OUTPUT_DIR"

echo "Stage1 case $index"
echo "model=$model_key task=$task_key method=$method h=$h env=$env_name"
echo "output_dir=$OUTPUT_DIR"
printf 'command:'
printf ' %q' "${CMD[@]}"
printf '\n'

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  write_status dry_run 0 "$OUTPUT_DIR"
  exit 0
fi

set +e
python "$NAN_GUARD" \
  --cwd "$RUN_CWD" \
  --stdout-log "$STDOUT_LOG" \
  --stderr-log "$STDERR_LOG" \
  --max-consecutive-nan 100 \
  -- "${CMD[@]}"
EXIT_CODE=$?
set -e

if [[ "$EXIT_CODE" -eq 0 ]]; then
  write_status completed "$EXIT_CODE" "$OUTPUT_DIR"
else
  write_status failed "$EXIT_CODE" "$OUTPUT_DIR"
fi

exit "$EXIT_CODE"
