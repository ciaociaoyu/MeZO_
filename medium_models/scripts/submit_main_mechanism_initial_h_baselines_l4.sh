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
H_SCHEDULE_FD_CLIP_POLICY="${H_SCHEDULE_FD_CLIP_POLICY:-none}"
H_SCHEDULE_FD_FLOOR_MIN="${H_SCHEDULE_FD_FLOOR_MIN:-1e-5}"
H_SCHEDULE_FD_CLIP_MAX="${H_SCHEDULE_FD_CLIP_MAX:-0.0}"
H_SCHEDULE_FD_INT8_POLICY="${H_SCHEDULE_FD_INT8_POLICY:-fp16_proxy_raw}"

RESULT_ROOT="${RESULT_ROOT:-${REPO_ROOT}/outputs/main_mechanism_initial_h_baselines}"
COMMANDS_FILE="${COMMANDS_FILE:-${SCRIPT_DIR}/main_mechanism_initial_h_commands.sh}"
ARRAY_SCRIPT="${ARRAY_SCRIPT:-${SCRIPT_DIR}/main_mechanism_initial_h_array_runner.sh}"
SUMMARY_DIR="${SUMMARY_DIR:-${REPO_ROOT}/medium_models/pilot_results}"
MANIFEST_CSV="${MANIFEST_CSV:-${SUMMARY_DIR}/main_mechanism_initial_h_long_manifest.csv}"
MANIFEST_MD="${MANIFEST_MD:-${SUMMARY_DIR}/main_mechanism_initial_h_long_manifest.md}"
DEFAULT_PREFLIGHT="${DEFAULT_PREFLIGHT:-${SUMMARY_DIR}/main_mechanism_initial_h_default_preflight.txt}"
LOG_DIR="${LOG_DIR:-${RESULT_ROOT}/slurm_logs}"

SUBMIT="${SUBMIT:-0}"
GPU_GRES="${GPU_GRES:-gpu:L4:1}"
ARRAY_CONCURRENCY="${ARRAY_CONCURRENCY:-3}"
SBATCH_EXTRA="${SBATCH_EXTRA:-}"
PARTITION="${PARTITION:-gpu_p}"
ACCOUNT="${ACCOUNT:-}"
TIME="${TIME:-48:00:00}"
CPUS="${CPUS:-8}"
SBATCH_NTASKS="${SBATCH_NTASKS:-1}"
MEM="${MEM:-32G}"
CONDA_ENV="${CONDA_ENV:-mezo-mistral}"
RUN_DEFAULT_IF_MISSING="${RUN_DEFAULT_IF_MISSING:-0}"

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
    find "${REPO_ROOT}/outputs" "${REPO_ROOT}/experiments" -maxdepth 8 -type d \
        \( -iname '*h1e-3*' -o -iname '*eps1e-3*' -o -iname '*default*' \) 2>/dev/null | sort | head -200 || true
} >"${DEFAULT_PREFLIGHT}"
default_found_count="$(grep -c . "${DEFAULT_PREFLIGHT}" || true)"
if [[ "${default_found_count}" == "0" && "${RUN_DEFAULT_IF_MISSING}" == "1" ]]; then
    echo "RUN_DEFAULT_IF_MISSING=1 was set, but this launcher is scoped to the six add-on initial-h runs." >&2
    echo "Prepare default h=1e-3 reruns separately to avoid mixing baselines." >&2
fi

: >"${COMMANDS_FILE}"
run_prefix=()
if [[ -n "${CONDA_ENV}" && "${CONDA_ENV}" != "none" ]]; then
    run_prefix=(conda run -n "${CONDA_ENV}")
fi

guard_args=()
if run_py_has "random_prediction_guard_enabled"; then
    guard_args+=(
        --random_prediction_guard_enabled True
        --random_prediction_guard_step 1000
        --random_prediction_guard_recent_evals 2
        --random_prediction_guard_acc_tolerance 0.05
        --random_prediction_guard_loss_tolerance 0.03
        --random_prediction_guard_bad_loss_excess 0.5
    )
fi
if run_py_has "zo_probe_health_guard_enabled"; then
    guard_args+=(
        --zo_probe_health_guard_enabled True
        --zo_probe_health_guard_step 1000
        --zo_probe_health_guard_max_bad_probes 3
    )
fi

policies=(fixed_small_1e-5 fd_eps13_raw)
precisions=(fp32 fp16 int8)
for policy_tag in "${policies[@]}"; do
    for precision in "${precisions[@]}"; do
        if [[ "${policy_tag}" == "fixed_small_1e-5" ]]; then
            schedule="fixed_small"
            schedule_args=(--h_schedule_h0 1e-5)
        else
            schedule="fd_eps13_raw"
            schedule_args=()
        fi
        tag="mainmech-initialh-${policy_tag}-${precision}"
        extra_args=()
        if [[ "${precision}" == "int8" ]]; then
            extra_args+=(
                --quantization_algorithm groupwise_symmetric
                --quantization_group_size 128
                --zo_update_backend fp16_master
            )
        fi
        cmd=(
            "${run_prefix[@]}"
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
            --h_schedule "${schedule}"
            --h_schedule_grid_policy "${H_GRID_POLICY}"
            --h_schedule_window_min "${H_WINDOW_MIN}"
            --h_schedule_window_max "${H_WINDOW_MAX}"
            --h_schedule_fd_clip_policy "${H_SCHEDULE_FD_CLIP_POLICY}"
            --h_schedule_fd_floor_min "${H_SCHEDULE_FD_FLOOR_MIN}"
            --h_schedule_fd_clip_max "${H_SCHEDULE_FD_CLIP_MAX}"
            --h_schedule_fd_int8_policy "${H_SCHEDULE_FD_INT8_POLICY}"
            "${schedule_args[@]}"
            "${guard_args[@]}"
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
mapfile -t commands < <(grep -vE '^[[:space:]]*(#|set[[:space:]]|$)' "${COMMANDS_FILE}")
cmd="${commands[${task_id}]:-}"
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
        --job-name "mainmech-initialh"
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

python - "${COMMANDS_FILE}" "${MANIFEST_CSV}" "${MANIFEST_MD}" "${RESULT_ROOT}" "${submitted_or_launched}" "${job_id}" "${GPU_GRES}" "${ARRAY_CONCURRENCY}" "${STEP}" "${EVAL_STEP}" "${SEED}" "${DATA_SEED}" "${BS}" "${LR}" "${MODEL}" "${TASK}" "${DATASET_MODE}" "${DEFAULT_PREFLIGHT}" "${PARTITION}" "${SBATCH_EXTRA}" "${#guard_args[@]}" <<'PY'
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
default_preflight = Path(sys.argv[18])
partition = sys.argv[19]
sbatch_extra = sys.argv[20]
guard_arg_count = int(sys.argv[21])

commands = [
    line.strip()
    for line in commands_file.read_text().splitlines()
    if line.strip() and not line.lstrip().startswith("#") and not line.lstrip().startswith("set ")
]
default_paths = [line.strip() for line in default_preflight.read_text().splitlines() if line.strip()]
rows = []
for command in commands:
    policy = "fixed_small_1e-5" if "--h_schedule fixed_small" in command else "fd_eps13_raw"
    precision = "int8" if "--precision_mode int8" in command else ("fp16" if "--precision_mode fp16" in command else "fp32")
    tag = f"mainmech-initialh-{policy}-{precision}"
    notes = []
    if precision == "int8":
        notes.append("uses groupwise_symmetric G128 with fp16_master backend; no GPTQ/residual_grid/direct_int8")
    if policy == "fd_eps13_raw" and precision in {"fp16", "int8"}:
        notes.append("raw FD is intentionally out-of-window and uncapped")
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
        "default_result_paths_found": len(default_paths),
        "notes": "; ".join(notes),
    })

manifest_csv.parent.mkdir(parents=True, exist_ok=True)
with manifest_csv.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)

with manifest_md.open("w") as f:
    f.write("# Main Mechanism Initial-h Baselines Long Manifest\n\n")
    f.write(f"- Submission state: `{submitted_or_launched}`\n")
    f.write(f"- Slurm job id: `{job_id}`\n")
    f.write(f"- GPU request: `{gpu_gres}`\n")
    f.write(f"- Partition: `{partition or 'default'}`\n")
    f.write(f"- SBATCH extra: `{sbatch_extra or 'none'}`\n")
    f.write(f"- Array concurrency: `{array_concurrency}`\n")
    f.write(f"- Commands file: `{commands_file}`\n")
    f.write(f"- Matrix: `fixed_small_1e-5/fd_eps13_raw x fp32/fp16/int8`\n")
    f.write(f"- Default h=1e-3 paths found in preflight: `{len(default_paths)}`\n")
    f.write("- Default h=1e-3 is not rerun by this launcher.\n")
    f.write(f"- Early guard args enabled: `{guard_arg_count > 0}`\n")
    f.write("- Raw FD FP16/INT8 use the uncapped FP16 eps^(1/3) proxy where applicable.\n")
    f.write("- Raw outputs are under the ignored `outputs/` tree.\n")

print(f"Manifest written to {manifest_csv}")
print(f"Manifest written to {manifest_md}")
PY
