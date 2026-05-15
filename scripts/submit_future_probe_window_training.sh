#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

timestamp() {
  date +"%Y%m%d_%H%M%S"
}

MODE="${MODE:-all_screen}"
RESULT_ROOT="${RESULT_ROOT:-${ROOT_DIR}/runs/future_probe_window_training_$(timestamp)}"
CONCURRENCY="${CONCURRENCY:-4}"
TIME_LIMIT="${TIME_LIMIT:-24:00:00}"
CPUS_PER_TASK="${CPUS_PER_TASK:-8}"
MEMORY="${MEMORY:-96G}"
CONDA_ENV="${CONDA_ENV:-ciao}"

mkdir -p "${RESULT_ROOT}/slurm" "${RESULT_ROOT}/logs"

detect_gpu_type() {
  if ! command -v sinfo >/dev/null 2>&1; then
    echo "local"
    return
  fi
  local info
  info="$(sinfo -h -o '%P|%G|%t|%D|%C' 2>/dev/null || true)"
  if echo "${info}" | awk -F'|' 'tolower($2) ~ /h100/ && $3 ~ /idle|mix/ {found=1} END{exit found?0:1}'; then
    echo "h100"
  elif echo "${info}" | awk -F'|' 'tolower($2) ~ /a100/ && $3 ~ /idle|mix/ {found=1} END{exit found?0:1}'; then
    echo "a100"
  else
    echo "l4"
  fi
}

gpu_type="${GPU_TYPE:-$(detect_gpu_type)}"
partition="${SLURM_PARTITION:-gpu_p}"
gres=""
case "${gpu_type}" in
  h100) gres="gpu:H100:1" ;;
  a100) gres="gpu:A100:1" ;;
  l4) gres="gpu:L4:1" ;;
  local) gres="" ;;
  *) gres="gpu:${gpu_type}:1" ;;
esac

num_cases="$(MODE="${MODE}" RESULT_ROOT="${RESULT_ROOT}" bash scripts/run_future_probe_window_training.sh --count | tr -d '[:space:]')"
if [[ -z "${num_cases}" || "${num_cases}" == "0" ]]; then
  echo "No cases generated for MODE=${MODE}" >&2
  exit 1
fi
array_max=$((num_cases - 1))

submit_script="${RESULT_ROOT}/slurm/submit_${MODE}.sbatch"
cat > "${submit_script}" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=future-${MODE}
#SBATCH --output=${RESULT_ROOT}/slurm/%x_%A_%a.out
#SBATCH --error=${RESULT_ROOT}/slurm/%x_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=${CPUS_PER_TASK}
#SBATCH --mem=${MEMORY}
#SBATCH --time=${TIME_LIMIT}
#SBATCH --partition=${partition}
EOF

if [[ -n "${gres}" ]]; then
  echo "#SBATCH --gres=${gres}" >> "${submit_script}"
fi
if [[ -n "${SBATCH_ACCOUNT:-}" ]]; then
  echo "#SBATCH --account=${SBATCH_ACCOUNT}" >> "${submit_script}"
fi
cat >> "${submit_script}" <<'EOF'

set -euo pipefail
cd "__ROOT_DIR__"
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate "__CONDA_ENV__"
export RESULT_ROOT="__RESULT_ROOT__"
export MODE="__MODE__"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
bash scripts/run_future_probe_window_training.sh --case "${SLURM_ARRAY_TASK_ID}"
EOF

python - "${submit_script}" "${ROOT_DIR}" "${CONDA_ENV}" "${RESULT_ROOT}" "${MODE}" <<'PY'
from pathlib import Path
import sys
path = Path(sys.argv[1])
text = path.read_text()
text = text.replace("__ROOT_DIR__", sys.argv[2])
text = text.replace("__CONDA_ENV__", sys.argv[3])
text = text.replace("__RESULT_ROOT__", sys.argv[4])
text = text.replace("__MODE__", sys.argv[5])
path.write_text(text)
PY

echo "RESULT_ROOT=${RESULT_ROOT}" | tee "${RESULT_ROOT}/submission_${MODE}.txt"
echo "MODE=${MODE}" | tee -a "${RESULT_ROOT}/submission_${MODE}.txt"
echo "num_cases=${num_cases}" | tee -a "${RESULT_ROOT}/submission_${MODE}.txt"
echo "gpu_type=${gpu_type}" | tee -a "${RESULT_ROOT}/submission_${MODE}.txt"
echo "partition=${partition}" | tee -a "${RESULT_ROOT}/submission_${MODE}.txt"
echo "gres=${gres}" | tee -a "${RESULT_ROOT}/submission_${MODE}.txt"
echo "concurrency=${CONCURRENCY}" | tee -a "${RESULT_ROOT}/submission_${MODE}.txt"
echo "submit_script=${submit_script}" | tee -a "${RESULT_ROOT}/submission_${MODE}.txt"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "DRY_RUN=1; not submitting"
  exit 0
fi

if command -v sbatch >/dev/null 2>&1; then
  sbatch --array="0-${array_max}%${CONCURRENCY}" "${submit_script}" | tee -a "${RESULT_ROOT}/submission_${MODE}.txt"
else
  echo "sbatch not found; run locally with:" | tee -a "${RESULT_ROOT}/submission_${MODE}.txt"
  echo "RESULT_ROOT='${RESULT_ROOT}' MODE='${MODE}' bash scripts/run_future_probe_window_training.sh" | tee -a "${RESULT_ROOT}/submission_${MODE}.txt"
fi
