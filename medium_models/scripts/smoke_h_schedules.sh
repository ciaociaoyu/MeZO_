#!/usr/bin/env bash
set -euo pipefail

trap 'echo "ERROR: smoke_h_schedules.sh failed at line ${LINENO}. Check the command above and the run output under ${SMOKE_RESULT_ROOT:-outputs/smoke_h_schedules}." >&2' ERR

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MEDIUM_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${MEDIUM_DIR}/.." && pwd)"

H_WINDOW_MIN="${H_WINDOW_MIN:-1e-5}"
H_WINDOW_MAX="${H_WINDOW_MAX:-1e-2}"
H_GRID="${H_GRID:-1e-5,3e-5,1e-4,3e-4,1e-3,3e-3,1e-2}"
SMOKE_STEP="${STEP:-2}"
SMOKE_EVAL_STEP="${EVAL_STEP:-1}"
SMOKE_RESULT_ROOT="${SMOKE_RESULT_ROOT:-${REPO_ROOT}/outputs/smoke_h_schedules}"

echo "[1/3] pytest: medium_models/tests/test_h_schedules.py"
cd "${REPO_ROOT}"
if ! python -c "import pytest" >/dev/null 2>&1; then
    echo "ERROR: pytest is not installed in $(command -v python). Install pytest or run the unit tests with: python -m unittest medium_models.tests.test_h_schedules" >&2
    exit 1
fi
python -m pytest medium_models/tests/test_h_schedules.py

echo "[2/3] print_h_schedule diagnostics"
for schedule in fixed spall_clip shamir_clip ji_sqrtk_clip ji_theory_clip pf_vrzo_clip; do
    extra_args=()
    if [[ "${schedule}" == "ji_theory_clip" ]]; then
        extra_args+=(--h_schedule_lipschitz_l "${H_LIPSCHITZ_L:-10.0}")
    fi
    python medium_models/tools/print_h_schedule.py \
        --format csv \
        --steps 5 \
        --zero_order_eps "${EPS:-1e-3}" \
        --max_steps "${SMOKE_STEP}" \
        --h_schedule "${schedule}" \
        --h_schedule_window_min "${H_WINDOW_MIN}" \
        --h_schedule_window_max "${H_WINDOW_MAX}" \
        --h_schedule_grid "${H_GRID}" \
        --h_schedule_total_steps "${SMOKE_STEP}" \
        --h_schedule_h0 "${H_WINDOW_MAX}" \
        --h_schedule_d_eff "${H_D_EFF:-1.0}" \
        "${extra_args[@]}" >/dev/null
done

echo "[3/3] very short MeZO schedule smoke matrix"
cd "${MEDIUM_DIR}"
for precision in fp32 fp16 int8; do
    for schedule in spall_clip shamir_clip pf_vrzo_clip; do
        echo ">>> precision=${precision} schedule=${schedule}"
        JOB_NAME="smoke-iterh-${schedule}-${precision}" \
        RESULT_ROOT="${SMOKE_RESULT_ROOT}" \
        TASK=sst-5 \
        K=16 \
        SEED=42 \
        BS=2 \
        LR=1e-6 \
        STEP="${SMOKE_STEP}" \
        EVAL_STEP="${SMOKE_EVAL_STEP}" \
        USE_H=False \
        MODEL=roberta-large \
        EXTRA_TAG="iterh-${schedule}-${precision}" \
        TAG="iterh-${schedule}-${precision}" \
        bash mezo.sh \
            --precision_mode "${precision}" \
            --h_schedule "${schedule}" \
            --h_schedule_window_min "${H_WINDOW_MIN}" \
            --h_schedule_window_max "${H_WINDOW_MAX}" \
            --h_schedule_grid "${H_GRID}" \
            --h_schedule_total_steps "${SMOKE_STEP}" \
            --h_schedule_h0 "${H_WINDOW_MAX}" \
            --h_schedule_d_eff "${H_D_EFF:-1.0}"
    done
done

echo "smoke_h_schedules.sh completed"
