#!/bin/bash
set -eu
set -o pipefail 2>/dev/null || true

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

TASK=${TASK:-sst-5}
MODEL=${MODEL:-roberta-large}
K=${K:-16}
SEEDS=${SEEDS:-13}
DATA_SEED=${DATA_SEED:-16}
DATASET_MODE=${DATASET_MODE:-full}
BS=${BS:-32}
LR=${LR:-1e-6}
WD=${WD:-0}
STEP=${STEP:-50000}
EVAL_STEP=${EVAL_STEP:-5000}
DATALOADER_SHUFFLE=${DATALOADER_SHUFFLE:-False}
USE_H=${USE_H:-False}
USE_C=${USE_C:-False}
ZO_PROBE_EVERY=${ZO_PROBE_EVERY:-200}
ZO_PROBE_NUM_SEEDS=${ZO_PROBE_NUM_SEEDS:-16}
ZO_PROBE_LOG_CSV=${ZO_PROBE_LOG_CSV:-True}
ZERO_ORDER_SAMPLE=${ZERO_ORDER_SAMPLE:-1}
PRECISIONS=${PRECISIONS:-"fp16"}
RESULT_ROOT=${RESULT_ROOT:-result}
SWEEP_NAME=${SWEEP_NAME:-${TASK}-roberta-large-full-fp16-h-sweep}
EXTRA_TAG=${EXTRA_TAG:-full-fp16-h-sweep}
SUMMARY_FILE=${SUMMARY_FILE:-${RESULT_ROOT}/${SWEEP_NAME}/summary.jsonl}
TASK_INDEX=${TASK_INDEX:-}
FORCE_ALL_CASES=${FORCE_ALL_CASES:-0}
EPS_LIST_OVERRIDE=${EPS_LIST_OVERRIDE:-}
EXTRA_PY_ARGS=("$@")

if [[ -n "${EPS_LIST_OVERRIDE}" ]]; then
    read -r -a EPS_LIST <<< "${EPS_LIST_OVERRIDE}"
else
    EPS_LIST=(
        1e-9
        3e-9
        1e-8
        3e-8
        1e-7
        3e-7
        1e-6
        3e-6
        1e-5
        3e-5
        1e-4
        3e-4
        1e-3
        3e-3
        1e-2
    )
fi

echo "TASK=${TASK} MODEL=${MODEL} K=${K}"
echo "DATASET_MODE=${DATASET_MODE} DATA_SEED=${DATA_SEED}"
echo "BS=${BS} LR=${LR} STEP=${STEP} EVAL_STEP=${EVAL_STEP}"
echo "USE_H=${USE_H} USE_C=${USE_C} DATALOADER_SHUFFLE=${DATALOADER_SHUFFLE}"
echo "ZO_PROBE_EVERY=${ZO_PROBE_EVERY} ZO_PROBE_NUM_SEEDS=${ZO_PROBE_NUM_SEEDS}"
echo "PRECISIONS=${PRECISIONS}"
echo "SEEDS=${SEEDS}"
echo "SUMMARY_FILE=${SUMMARY_FILE}"
echo "FORCE_ALL_CASES=${FORCE_ALL_CASES}"
echo "EPS_LIST=${EPS_LIST[*]}"

read -r -a PREC_ARRAY <<< "${PRECISIONS}"
read -r -a SEED_ARRAY <<< "${SEEDS}"

num_prec=${#PREC_ARRAY[@]}
num_eps=${#EPS_LIST[@]}
num_seed=${#SEED_ARRAY[@]}
total_tasks=$((num_prec * num_eps * num_seed))

if (( total_tasks <= 0 )); then
    echo "No tasks to run (check PRECISIONS/SEEDS/EPS_LIST)."
    exit 1
fi

append_case_summary() {
    local run_summary_path="$1"
    local prec="$2"
    local eps="$3"
    local seed="$4"

    if [[ ! -f "${run_summary_path}" ]]; then
        echo "Missing run summary: ${run_summary_path}"
        exit 1
    fi

    python - "${run_summary_path}" "${SUMMARY_FILE}" "${TASK}" "${MODEL}" "${prec}" "${eps}" "${seed}" "${DATASET_MODE}" <<'PY'
import json
import os
import sys

try:
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None

run_summary_path, summary_file, task, model, precision, eps_value, seed, dataset_mode = sys.argv[1:]

with open(run_summary_path, "r", encoding="utf-8") as f:
    record = json.load(f)

record.update({
    "task": task,
    "model": model,
    "precision": precision,
    "h": eps_value,
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

run_one_case() {
    local prec="$1"
    local eps="$2"
    local seed="$3"
    local h_tag="h_${eps}"
    local case_result_root="${RESULT_ROOT}/${SWEEP_NAME}/${prec}"
    local run_summary_path="${case_result_root}/${h_tag}/seed${seed}/run_summary.json"
    echo ">>> Running precision=${prec}, eps=${eps}, seed=${seed}"
    TASK=${TASK} \
    MODEL=${MODEL} \
    K=${K} \
    SEED=${seed} \
    DATA_SEED=${DATA_SEED} \
    DATASET_MODE=${DATASET_MODE} \
    BS=${BS} \
    LR=${LR} \
    EPS=${eps} \
    WD=${WD} \
    STEP=${STEP} \
    EVAL_STEP=${EVAL_STEP} \
    USE_H=${USE_H} \
    USE_C=${USE_C} \
    DATALOADER_SHUFFLE=${DATALOADER_SHUFFLE} \
    EXTRA_TAG=${EXTRA_TAG} \
    NUM_GPU=1 \
    bash mezo.sh \
        --result_root "${case_result_root}" \
        --job_name "${h_tag}" \
        --dataset_mode "${DATASET_MODE}" \
        --zero_order_sample "${ZERO_ORDER_SAMPLE}" \
        --zo_two_point_precision "${prec}" \
        --zo_probe_every "${ZO_PROBE_EVERY}" \
        --zo_probe_num_seeds "${ZO_PROBE_NUM_SEEDS}" \
        --zo_probe_log_csv "${ZO_PROBE_LOG_CSV}" \
        "${EXTRA_PY_ARGS[@]}"

    append_case_summary "${run_summary_path}" "${prec}" "${eps}" "${seed}"
}

array_idx=""
if [[ "${FORCE_ALL_CASES}" != "1" ]]; then
    array_idx="${TASK_INDEX}"
    if [[ -z "${array_idx}" && -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
        array_idx="${SLURM_ARRAY_TASK_ID}"
    fi
fi

if [[ -n "${array_idx}" ]]; then
    if ! [[ "${array_idx}" =~ ^[0-9]+$ ]]; then
        echo "Invalid TASK_INDEX/SLURM_ARRAY_TASK_ID: ${array_idx}"
        exit 1
    fi
    if (( array_idx < 0 || array_idx >= total_tasks )); then
        echo "TASK_INDEX=${array_idx} out of range [0, $((total_tasks - 1))], skip."
        exit 0
    fi

    per_prec=$((num_eps * num_seed))
    prec_idx=$((array_idx / per_prec))
    rem_idx=$((array_idx % per_prec))
    eps_idx=$((rem_idx / num_seed))
    seed_idx=$((rem_idx % num_seed))

    prec="${PREC_ARRAY[$prec_idx]}"
    eps="${EPS_LIST[$eps_idx]}"
    seed="${SEED_ARRAY[$seed_idx]}"

    echo "ARRAY MODE: index=${array_idx}/${total_tasks} -> precision=${prec}, eps=${eps}, seed=${seed}"
    run_one_case "${prec}" "${eps}" "${seed}"
    exit 0
fi

echo "NON-ARRAY MODE: run all ${total_tasks} tasks sequentially."
for prec in "${PREC_ARRAY[@]}"; do
    for eps in "${EPS_LIST[@]}"; do
        for seed in "${SEED_ARRAY[@]}"; do
            run_one_case "${prec}" "${eps}" "${seed}"
        done
    done
done
