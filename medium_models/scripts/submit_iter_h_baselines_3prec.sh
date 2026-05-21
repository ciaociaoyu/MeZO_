#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MEDIUM_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${MEDIUM_DIR}/.." && pwd)"

TASK="${TASK:-sst-5}"
K="${K:-16}"
SEED="${SEED:-42}"
BS="${BS:-64}"
LR="${LR:-1e-6}"
EPS="${EPS:-1e-3}"
WD="${WD:-0}"
STEP="${STEP:-50000}"
EVAL_STEP="${EVAL_STEP:-5000}"
MODEL="${MODEL:-roberta-large}"
USE_C="${USE_C:-False}"

H_WINDOW_MIN="${H_WINDOW_MIN:-1e-5}"
H_WINDOW_MAX="${H_WINDOW_MAX:-1e-2}"
H_GRID="${H_GRID:-1e-5,3e-5,1e-4,3e-4,1e-3,3e-3,1e-2}"
H_SCHEDULE_H0="${H_SCHEDULE_H0:-${H_WINDOW_MAX}}"
H_SCHEDULE_D_EFF="${H_SCHEDULE_D_EFF:-1.0}"
H_SCHEDULE_TOTAL_STEPS="${H_SCHEDULE_TOTAL_STEPS:-${STEP}}"

RESULT_ROOT="${RESULT_ROOT:-${REPO_ROOT}/outputs/iter_h_baselines_3prec}"
COMMANDS_FILE="${COMMANDS_FILE:-${SCRIPT_DIR}/iter_h_baselines_3prec_commands.sh}"
LOG_DIR="${LOG_DIR:-${RESULT_ROOT}/slurm_logs}"

SUBMIT="${SUBMIT:-0}"
RUN_LOCAL="${RUN_LOCAL:-0}"
PARTITION="${PARTITION:-}"
ACCOUNT="${ACCOUNT:-}"
TIME="${TIME:-24:00:00}"
CPUS="${CPUS:-8}"
MEM="${MEM:-32G}"
GPUS="${GPUS:-1}"
SBATCH_EXTRA="${SBATCH_EXTRA:-}"

schedules=(spall_clip shamir_clip pf_vrzo_clip)
if [[ "${RUN_JI:-0}" == "1" ]]; then
    schedules+=(ji_sqrtk_clip)
fi
precisions=(fp32 fp16 int8)

quote_cmd() {
    local out="" q
    for arg in "$@"; do
        printf -v q '%q' "$arg"
        out+="${q} "
    done
    printf '%s' "${out% }"
}

mkdir -p "$(dirname "${COMMANDS_FILE}")" "${LOG_DIR}"
{
    echo "#!/usr/bin/env bash"
    echo "set -euo pipefail"
} >"${COMMANDS_FILE}"

for precision in "${precisions[@]}"; do
    for schedule in "${schedules[@]}"; do
        tag="iterh-${schedule}-${precision}"
        cmd=(
            env
            JOB_NAME="${tag}"
            RESULT_ROOT="${RESULT_ROOT}"
            TASK="${TASK}"
            K="${K}"
            SEED="${SEED}"
            BS="${BS}"
            LR="${LR}"
            EPS="${EPS}"
            WD="${WD}"
            STEP="${STEP}"
            EVAL_STEP="${EVAL_STEP}"
            MODEL="${MODEL}"
            USE_H=False
            USE_C="${USE_C}"
            EXTRA_TAG="${tag}"
            TAG="${tag}"
            bash mezo.sh
            --precision_mode "${precision}"
            --h_schedule "${schedule}"
            --h_schedule_window_min "${H_WINDOW_MIN}"
            --h_schedule_window_max "${H_WINDOW_MAX}"
            --h_schedule_grid "${H_GRID}"
            --h_schedule_h0 "${H_SCHEDULE_H0}"
            --h_schedule_total_steps "${H_SCHEDULE_TOTAL_STEPS}"
            --h_schedule_d_eff "${H_SCHEDULE_D_EFF}"
        )
        printf -v cd_prefix 'cd %q && ' "${MEDIUM_DIR}"
        cmd_line="${cd_prefix}$(quote_cmd "${cmd[@]}")"
        echo "${cmd_line}" >>"${COMMANDS_FILE}"

        if [[ "${RUN_LOCAL}" == "1" ]]; then
            echo "RUN_LOCAL=1 executing ${tag}"
            eval "${cmd_line}"
        elif [[ "${SUBMIT}" == "1" ]]; then
            sbatch_args=(
                --job-name "${tag}"
                --output "${LOG_DIR}/%x-%j.out"
                --time "${TIME}"
                --cpus-per-task "${CPUS}"
                --mem "${MEM}"
            )
            if [[ -n "${PARTITION}" ]]; then
                sbatch_args+=(--partition "${PARTITION}")
            fi
            if [[ -n "${ACCOUNT}" ]]; then
                sbatch_args+=(--account "${ACCOUNT}")
            fi
            if [[ "${GPUS}" != "0" ]]; then
                sbatch_args+=(--gres "gpu:${GPUS}")
            fi
            if [[ -n "${SBATCH_EXTRA}" ]]; then
                read -r -a extra_args <<<"${SBATCH_EXTRA}"
                sbatch_args+=("${extra_args[@]}")
            fi
            echo "Submitting ${tag}"
            sbatch "${sbatch_args[@]}" --wrap "${cmd_line}"
        else
            echo "Prepared ${tag}"
        fi
    done
done

chmod +x "${COMMANDS_FILE}"
echo "Commands written to ${COMMANDS_FILE}"
echo "Set SUBMIT=1 to sbatch them, or RUN_LOCAL=1 to execute sequentially."
