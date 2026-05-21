#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MEDIUM_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${MEDIUM_DIR}/.." && pwd)"

TASK="${TASK:-sst-5}"
DATASET_MODE="${DATASET_MODE:-full}"
K="${K:-16}"
SEED="${SEED:-16}"
DATA_SEED="${DATA_SEED:-16}"
BS="${BS:-64}"
LR="${LR:-1e-6}"
EPS="${EPS:-1e-3}"
STEP="${STEP:-20000}"
EVAL_STEP="${EVAL_STEP:-1000}"
MODEL="${MODEL:-roberta-large}"
DATALOADER_SHUFFLE="${DATALOADER_SHUFFLE:-True}"
H_GRID_POLICY="${H_GRID_POLICY:-continuous}"
H_WINDOW_MIN="${H_WINDOW_MIN:-1e-5}"
H_WINDOW_MAX="${H_WINDOW_MAX:-1e-2}"
SPALL_H0="${SPALL_H0:-1e-3}"
SPALL_GAMMA="${SPALL_GAMMA:-0.101}"

RESULT_ROOT="${RESULT_ROOT:-${REPO_ROOT}/outputs/main_mechanism_classical_h_baselines}"
COMMANDS_FILE="${COMMANDS_FILE:-${SCRIPT_DIR}/main_mechanism_classical_h_commands.sh}"
ARRAY_SCRIPT="${ARRAY_SCRIPT:-${SCRIPT_DIR}/main_mechanism_classical_h_array_runner.sh}"
SUMMARY_DIR="${SUMMARY_DIR:-${REPO_ROOT}/medium_models/pilot_results}"
MANIFEST_CSV="${MANIFEST_CSV:-${SUMMARY_DIR}/main_mechanism_classical_h_long_manifest.csv}"
MANIFEST_MD="${MANIFEST_MD:-${SUMMARY_DIR}/main_mechanism_classical_h_long_manifest.md}"
LOG_DIR="${LOG_DIR:-${RESULT_ROOT}/slurm_logs}"

SUBMIT="${SUBMIT:-0}"
GPU_GRES="${GPU_GRES:-gpu:L4:1}"
ARRAY_CONCURRENCY="${ARRAY_CONCURRENCY:-3}"
SBATCH_EXTRA="${SBATCH_EXTRA:-}"
PARTITION="${PARTITION:-}"
ACCOUNT="${ACCOUNT:-}"
TIME="${TIME:-48:00:00}"
CPUS="${CPUS:-8}"
SBATCH_NTASKS="${SBATCH_NTASKS:-1}"
MEM="${MEM:-32G}"

run_py_has() {
    grep -q "$1" "${MEDIUM_DIR}/run.py"
}

quote_cmd() {
    local out="" q
    for arg in "$@"; do
        printf -v q '%q' "$arg"
        out+="${q} "
    done
    printf '%s' "${out% }"
}

mkdir -p "$(dirname "${COMMANDS_FILE}")" "${LOG_DIR}" "${SUMMARY_DIR}"
{
    echo "#!/usr/bin/env bash"
    echo "set -euo pipefail"
} >"${COMMANDS_FILE}"

policies=(fd_eps13 spall_ck)
precisions=(fp32 fp16 int8)
for policy in "${policies[@]}"; do
    for precision in "${precisions[@]}"; do
        tag="mainmech-classicalh-${policy}-${precision}"
        extra_args=()
        if run_py_has "random_prediction_guard_enabled"; then
            extra_args+=(--random_prediction_guard_enabled True)
        fi
        if run_py_has "zo_probe_health_guard_enabled"; then
            extra_args+=(--zo_probe_health_guard_enabled True)
        fi
        if [[ "${precision}" == "int8" ]]; then
            extra_args+=(
                --quantization_algorithm groupwise_symmetric
                --quantization_group_size 128
                --zo_update_backend fp16_master
            )
        fi
        cmd=(
            env
            JOB_NAME="${tag}"
            RESULT_ROOT="${RESULT_ROOT}"
            TASK="${TASK}"
            DATASET_MODE="${DATASET_MODE}"
            K="${K}"
            SEED="${SEED}"
            DATA_SEED="${DATA_SEED}"
            BS="${BS}"
            LR="${LR}"
            EPS="${EPS}"
            STEP="${STEP}"
            EVAL_STEP="${EVAL_STEP}"
            MODEL="${MODEL}"
            USE_H=False
            USE_C=False
            DATALOADER_SHUFFLE="${DATALOADER_SHUFFLE}"
            EXTRA_TAG="${tag}"
            TAG="${tag}"
            bash mezo.sh
            --precision_mode "${precision}"
            --h_schedule "${policy}"
            --h_schedule_grid_policy "${H_GRID_POLICY}"
            --h_schedule_window_min "${H_WINDOW_MIN}"
            --h_schedule_window_max "${H_WINDOW_MAX}"
            --h_schedule_fd_clip_min "${H_WINDOW_MIN}"
            --h_schedule_fd_clip_max "${H_WINDOW_MAX}"
            --h_schedule_fd_int8_policy capped_stress
            --h_schedule_h0 "${SPALL_H0}"
            --h_schedule_gamma "${SPALL_GAMMA}"
            "${extra_args[@]}"
        )
        printf -v cd_prefix 'cd %q && ' "${MEDIUM_DIR}"
        echo "${cd_prefix}$(quote_cmd "${cmd[@]}")" >>"${COMMANDS_FILE}"
    done
done
chmod +x "${COMMANDS_FILE}"

cat >"${ARRAY_SCRIPT}" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
if [[ -z "${COMMANDS_FILE:-}" ]]; then
    echo "COMMANDS_FILE is required" >&2
    exit 2
fi
task_id="${SLURM_ARRAY_TASK_ID:-0}"
cmd="$(sed -n "$((task_id + 1))p" "${COMMANDS_FILE}")"
if [[ -z "${cmd}" ]]; then
    echo "No command for SLURM_ARRAY_TASK_ID=${task_id}" >&2
    exit 2
fi
echo "[array] task_id=${task_id}"
echo "[array] command=${cmd}"
eval "${cmd}"
EOF
chmod +x "${ARRAY_SCRIPT}"

submitted_or_launched="prepared"
job_id="NA"
if [[ "${SUBMIT}" == "1" ]]; then
    sbatch_args=(
        --array "0-5%${ARRAY_CONCURRENCY}"
        --gres "${GPU_GRES}"
        --job-name "mainmech-classicalh"
        --output "${LOG_DIR}/%x-%A_%a.out"
        --time "${TIME}"
        --ntasks "${SBATCH_NTASKS}"
        --cpus-per-task "${CPUS}"
        --mem "${MEM}"
        --export "ALL,COMMANDS_FILE=${COMMANDS_FILE}"
    )
    if [[ -n "${PARTITION}" ]]; then
        sbatch_args+=(--partition "${PARTITION}")
    fi
    if [[ -n "${ACCOUNT}" ]]; then
        sbatch_args+=(--account "${ACCOUNT}")
    fi
    if [[ -n "${SBATCH_EXTRA}" ]]; then
        read -r -a extra_args <<<"${SBATCH_EXTRA}"
        sbatch_args+=("${extra_args[@]}")
    fi
    submit_out="$(sbatch "${sbatch_args[@]}" "${ARRAY_SCRIPT}")"
    echo "${submit_out}"
    job_id="$(awk '{print $NF}' <<<"${submit_out}")"
    submitted_or_launched="submitted"
else
    echo "Commands written to ${COMMANDS_FILE}"
    echo "Set SUBMIT=1 to submit a 6-task L4 job array with concurrency ${ARRAY_CONCURRENCY}."
fi

python - "${COMMANDS_FILE}" "${MANIFEST_CSV}" "${MANIFEST_MD}" "${RESULT_ROOT}" "${submitted_or_launched}" "${job_id}" "${GPU_GRES}" "${ARRAY_CONCURRENCY}" "${STEP}" "${EVAL_STEP}" "${SEED}" "${DATA_SEED}" "${BS}" "${LR}" "${MODEL}" "${TASK}" "${DATASET_MODE}" <<'PY'
import csv
import sys
from pathlib import Path

commands_file = Path(sys.argv[1])
manifest_csv = Path(sys.argv[2])
manifest_md = Path(sys.argv[3])
result_root = Path(sys.argv[4])
submitted_or_launched = sys.argv[5]
job_id = sys.argv[6]
gpu_gres = sys.argv[7]
array_concurrency = sys.argv[8]
step = sys.argv[9]
eval_step = sys.argv[10]
seed = sys.argv[11]
data_seed = sys.argv[12]
batch_size = sys.argv[13]
lr = sys.argv[14]
model = sys.argv[15]
dataset = sys.argv[16]
dataset_mode = sys.argv[17]

commands = [line.strip() for line in commands_file.read_text().splitlines() if line.strip() and not line.startswith("#") and not line.startswith("set ")]
rows = []
for command in commands:
    policy = "fd_eps13" if "--h_schedule fd_eps13" in command else "spall_ck"
    precision = "int8" if "--precision_mode int8" in command else ("fp16" if "--precision_mode fp16" in command else "fp32")
    tag = f"mainmech-classicalh-{policy}-{precision}"
    notes = []
    if precision == "int8":
        notes.append("uses groupwise_symmetric G128 with fp16_master backend; no GPTQ/residual_grid/direct_int8")
    rows.append({
        "policy": policy,
        "precision": precision,
        "command": command,
        "result_dir": str(result_root / tag / f"seed{seed}"),
        "submitted_or_launched": submitted_or_launched,
        "job_id": job_id,
        "gpu_gres": gpu_gres,
        "array_concurrency": array_concurrency,
        "step": step,
        "eval_step": eval_step,
        "seed": seed,
        "data_seed": data_seed,
        "batch_size": batch_size,
        "lr": lr,
        "model": model,
        "dataset": dataset,
        "dataset_mode": dataset_mode,
        "notes": "; ".join(notes),
    })

manifest_csv.parent.mkdir(parents=True, exist_ok=True)
with manifest_csv.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)

with manifest_md.open("w") as f:
    f.write("# Main Mechanism Classical h Baselines Long Manifest\n\n")
    f.write(f"- Submission state: `{submitted_or_launched}`\n")
    f.write(f"- Slurm job id: `{job_id}`\n")
    f.write(f"- GPU request: `{gpu_gres}`\n")
    f.write(f"- Array concurrency: `{array_concurrency}`\n")
    f.write(f"- Commands file: `{commands_file}`\n")
    f.write("- Matrix: fd_eps13/spall_ck x fp32/fp16/int8\n")
    f.write("- Default h=1e-3 is not included in this batch.\n")
    f.write("- Raw outputs are under the ignored `outputs/` tree.\n")

print(f"Manifest written to {manifest_csv}")
print(f"Manifest written to {manifest_md}")
PY
