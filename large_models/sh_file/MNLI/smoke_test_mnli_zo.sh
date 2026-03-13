#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT_DIR}"

MODEL=${MODEL:-facebook/opt-125m}
TASK=${TASK:-MNLI}
MODEL_NAME="${MODEL##*/}"

SEED=${SEED:-0}
BS=${BS:-1}
GA=${GA:-1}
LR=${LR:-1e-6}
EPS=${EPS:-1e-3}
STEPS=${STEPS:-2}
TRAIN=${TRAIN:-64}
EVAL=${EVAL:-64}

OUT_DIR="result/${TASK}-${MODEL_NAME}-smoke-zo"
RESULT_FILE="result/${TASK}-${MODEL_NAME}-smoke-zo-metrics.json"

echo "[Smoke] model=${MODEL} task=${TASK} seed=${SEED}"
echo "[Smoke] steps=${STEPS} train=${TRAIN} eval=${EVAL} bs=${BS} ga=${GA} lr=${LR} eps=${EPS}"
echo "[Smoke] output_dir=${OUT_DIR}"
echo "[Smoke] result_file=${RESULT_FILE}"

python run.py \
  --model_name "${MODEL}" \
  --task_name "${TASK}" \
  --trainer zo \
  --train_as_classification \
  --load_float16 \
  --train_set_seed "${SEED}" \
  --num_train "${TRAIN}" \
  --num_dev 0 \
  --num_eval "${EVAL}" \
  --max_steps "${STEPS}" \
  --per_device_train_batch_size "${BS}" \
  --gradient_accumulation_steps "${GA}" \
  --learning_rate "${LR}" \
  --zo_eps "${EPS}" \
  --evaluation_strategy steps \
  --eval_steps 1 \
  --save_strategy no \
  --logging_steps 1 \
  --output_dir "${OUT_DIR}" \
  --result_file "${RESULT_FILE}" \
  "$@"

python - "${RESULT_FILE}" <<'PY'
import json
import os
import sys

result_file = sys.argv[1]
if not os.path.exists(result_file):
    print(f"[FAIL] result file not found: {result_file}")
    sys.exit(1)

with open(result_file, "r", encoding="utf-8") as f:
    metrics = json.load(f)

required_keys = ["accuracy", "valid_mismatched_accuracy"]
missing = [k for k in required_keys if k not in metrics]
if missing:
    print("[FAIL] missing metric keys:", ", ".join(missing))
    print("metrics keys:", ", ".join(metrics.keys()))
    sys.exit(1)

print("[PASS] MNLI smoke test finished.")
print("accuracy =", metrics["accuracy"])
print("valid_mismatched_accuracy =", metrics["valid_mismatched_accuracy"])
PY

echo "[Smoke] done."
