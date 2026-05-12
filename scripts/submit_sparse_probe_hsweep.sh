#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TS="${TS:-$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-${REPO_ROOT}/runs/sparse_probe_hsweep_${TS}}"
CONDA_ENV="${CONDA_ENV:-ciao}"
MAX_ARRAY_CONCURRENCY="${MAX_ARRAY_CONCURRENCY:-4}"
NUM_PROBE_DIRECTIONS="${NUM_PROBE_DIRECTIONS:-50}"
SEED="${SEED:-16}"
DATA_SEED="${DATA_SEED:-16}"
DENSE_H="${DENSE_H:-3e-3}"

mkdir -p "${RUN_ROOT}/logs" "${RUN_ROOT}/jobs"

detect_gpu_type() {
  if command -v sinfo >/dev/null 2>&1; then
    local info
    info="$(sinfo -h -o '%P|%G|%t|%D|%C' 2>/dev/null || true)"
    if printf '%s\n' "${info}" | awk -F'|' 'tolower($2) ~ /h100/ && tolower($3) ~ /(idle|mix)/ {found=1} END{exit found ? 0 : 1}'; then
      echo "h100"
      return 0
    fi
    if printf '%s\n' "${info}" | awk -F'|' 'tolower($2) ~ /a100/ && tolower($3) ~ /(idle|mix)/ {found=1} END{exit found ? 0 : 1}'; then
      echo "a100"
      return 0
    fi
  fi
  echo "l4"
}

GPU_TYPE="${GPU_TYPE:-$(detect_gpu_type)}"
case "${GPU_TYPE}" in
  h100|H100)
    PARTITION="${PARTITION:-gpu_p}"
    GPU_GRES="${GPU_GRES:-gpu:H100:1}"
    ;;
  a100|A100)
    PARTITION="${PARTITION:-a100}"
    GPU_GRES="${GPU_GRES:-gpu:a100:1}"
    ;;
  l4|L4|L4GPU|l4gpu)
    PARTITION="${PARTITION:-gpu_p}"
    GPU_GRES="${GPU_GRES:-gpu:L4:1}"
    ;;
  *)
    echo "Unsupported GPU_TYPE=${GPU_TYPE}" >&2
    exit 2
    ;;
esac

MANIFEST="${RUN_ROOT}/manifest.tsv"
python - "${MANIFEST}" "${DENSE_H}" <<'PY'
import math
import sys

manifest, dense_h = sys.argv[1], float(sys.argv[2])
dense = [1e-4, 3e-4, 1e-3, 2e-3, 3e-3, 5e-3, 1e-2]
p_list = [0.03, 0.01, 0.003]
factors = [0.25, 0.5, 1, 2, 4, 8]
rows = []
for h in dense:
    rows.append((1.0, h, "none", "none", f"dense_h{h:.3g}".replace(".", "p")))
for p in p_list:
    base = dense_h * math.sqrt(p)
    for factor in factors:
        h = base * factor
        rows.append((p, h, "exact_random", "inv_sqrt_p", f"p{p:.3g}_f{factor:g}".replace(".", "p")))
with open(manifest, "w", encoding="utf-8") as f:
    f.write("p\th\tsparse_mode\tsparse_rescale\trun_name\n")
    for row in rows:
        f.write("\t".join(str(x) for x in row) + "\n")
print(len(rows))
PY

NUM_CASES=$(( $(wc -l < "${MANIFEST}") - 1 ))
if (( NUM_CASES <= 0 )); then
  echo "No sparse probe cases generated." >&2
  exit 1
fi

ARRAY_MAX=$((NUM_CASES - 1))
SBATCH_SCRIPT="${RUN_ROOT}/jobs/sparse_probe_hsweep_array.sbatch"
cat > "${SBATCH_SCRIPT}" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=sparse_probe_hsweep
#SBATCH --partition=${PARTITION}
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --gres=${GPU_GRES}
#SBATCH --time=08:00:00
#SBATCH --array=0-${ARRAY_MAX}%${MAX_ARRAY_CONCURRENCY}
#SBATCH --output=${RUN_ROOT}/logs/%x_%A_%a.out
#SBATCH --error=${RUN_ROOT}/logs/%x_%A_%a.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT}"
RUN_ROOT="${RUN_ROOT}"
MANIFEST="${MANIFEST}"
CONDA_ENV="${CONDA_ENV}"
NUM_PROBE_DIRECTIONS="${NUM_PROBE_DIRECTIONS}"
SEED="${SEED}"
DATA_SEED="${DATA_SEED}"

if [[ -f "\${HOME}/miniconda3/etc/profile.d/conda.sh" ]]; then
  source "\${HOME}/miniconda3/etc/profile.d/conda.sh"
  conda activate "\${CONDA_ENV}"
fi

idx="\${SLURM_ARRAY_TASK_ID}"
line="\$(awk -v n=\$((idx + 2)) 'NR==n {print}' "\${MANIFEST}")"
IFS=\$'\t' read -r P H SPARSE_MODE SPARSE_RESCALE RUN_NAME <<< "\${line}"

case_dir="\${RUN_ROOT}/\${RUN_NAME}"
mkdir -p "\${case_dir}"

export RESULT_ROOT="\${RUN_ROOT}"
export JOB_NAME="\${RUN_NAME}"
export SEED="\${SEED}"
export DATA_SEED="\${DATA_SEED}"
export LR=0
export STEP=1
export EVAL_STEP=100000
export NUM_SEEDS="\${NUM_PROBE_DIRECTIONS}"
export H_LIST="\${H}"
export ZO_PROBE_INCLUDE_STEP0=1
export ZO_PROBE_UPDATE_STATS=1
export CUDA_VISIBLE_DEVICES="\${CUDA_VISIBLE_DEVICES:-0}"

cd "\${REPO_ROOT}"
bash experiments/int8_error_origin_probe/run_roberta_sst5_int8_local.sh \\
  --probe_diagnostics_only True \\
  --num_probe_directions "\${NUM_PROBE_DIRECTIONS}" \\
  --save_probe_stats_jsonl probe_stats.jsonl \\
  --zo_h "\${H}" \\
  --zo_update_backend direct_int8 \\
  --zo_direction_sparse_rate "\${P}" \\
  --zo_direction_sparse_mode "\${SPARSE_MODE}" \\
  --zo_sparse_rescale "\${SPARSE_RESCALE}" \\
  --zo_sparse_per_layer_exact True \\
  --random_prediction_guard_enabled False \\
  --save_strategy no \\
  --no_predict
EOF

echo "RUN_ROOT=${RUN_ROOT}"
echo "MANIFEST=${MANIFEST}"
echo "GPU_TYPE=${GPU_TYPE} PARTITION=${PARTITION} GPU_GRES=${GPU_GRES}"
echo "SBATCH_SCRIPT=${SBATCH_SCRIPT}"
JOB_OUTPUT="$(sbatch "${SBATCH_SCRIPT}")"
echo "${JOB_OUTPUT}" | tee "${RUN_ROOT}/submitted_job.txt"
echo "Submitted ${NUM_CASES} sparse probe cases."
