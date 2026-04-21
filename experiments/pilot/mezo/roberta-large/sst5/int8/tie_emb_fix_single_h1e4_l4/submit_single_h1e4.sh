#!/bin/bash

set -euo pipefail

SCRATCH_ROOT="/scratch/jy03364/MeZO_"
BASE_DIR="${SCRATCH_ROOT}/experiments/pilot/mezo/roberta-large/sst5/int8/tie_emb_fix_single_h1e4_l4"
RUNNER="${BASE_DIR}/job_runner.sh"
STAMP="$(date +%Y%m%d_%H%M%S)"
EXPERIMENT_ROOT="${BASE_DIR}/runs/${STAMP}"
LOG_DIR="${EXPERIMENT_ROOT}/logs"
DATA_PREP_LOG="${EXPERIMENT_ROOT}/data_prep.log"
SUBMIT_LOG="${EXPERIMENT_ROOT}/submission_manifest.json"
CONFIG_SNAPSHOT="${EXPERIMENT_ROOT}/launch_config.json"

mkdir -p "${LOG_DIR}"

set +u
source /home/jy03364/miniconda3/etc/profile.d/conda.sh
conda activate ciao
set -u

python - "${CONFIG_SNAPSHOT}" "${EXPERIMENT_ROOT}" <<'PY'
import json
import sys
from datetime import datetime

path, experiment_root = sys.argv[1:]
record = {
    "created_at": datetime.now().isoformat(),
    "experiment_root": experiment_root,
    "model": "roberta-large",
    "task": "sst-5",
    "precision": "int8",
    "method": "mezo",
    "dataset_mode": "fewshot",
    "k": 16,
    "seed": 16,
    "data_seed": 16,
    "dataloader_shuffle": True,
    "per_device_train_batch_size": 64,
    "learning_rate": 1e-6,
    "weight_decay": 0.0,
    "max_steps": 10000,
    "eval_steps": 1000,
    "h": "1e-4",
    "zo_probe_every": 200,
    "zo_probe_num_seeds": 16,
    "random_prediction_guard_enabled": False,
    "zo_probe_health_guard_enabled": False,
    "nan_guard_limit": 100,
    "tie_emb": True,
    "notes": [
        "Single-job validation of the RoBERTa prompt lm_head decoder tie fix.",
        "Matches the current paper-style k=16/int8 baseline except for tie_emb=True.",
        "Results are isolated under a fresh timestamped experiment root."
    ]
}
with open(path, "w", encoding="utf-8") as f:
    json.dump(record, f, ensure_ascii=False, indent=2, sort_keys=True)
PY

(
  cd "${SCRATCH_ROOT}/medium_models"
  python tools/generate_k_shot_data.py \
    --mode k-shot-1k-test \
    --dataset_mode fewshot \
    --k 16 \
    --seed 16 \
    --task SST-5
) | tee "${DATA_PREP_LOG}"

job_id="$(
  sbatch --parsable \
    --job-name="rb_sst5_i8_tie_h1e4_L4" \
    --partition=gpu_p \
    --ntasks=1 \
    --cpus-per-task=4 \
    --mem=32G \
    --gres=gpu:L4:1 \
    --time=12:00:00 \
    --chdir="${BASE_DIR}" \
    --output="${LOG_DIR}/slurm_rb_sst5_i8_tie_h1e4_L4_%j.out" \
    --export=ALL,EXPERIMENT_ROOT="${EXPERIMENT_ROOT}" \
    "${RUNNER}"
)"

python - "${SUBMIT_LOG}" "${job_id}" "${EXPERIMENT_ROOT}" <<'PY'
import json
import sys
from datetime import datetime

path, job_id, experiment_root = sys.argv[1:]
record = {
    "submitted_at": datetime.now().isoformat(),
    "job_id": int(job_id),
    "gpu_type": "L4",
    "time_limit": "12:00:00",
    "experiment_root": experiment_root,
}
with open(path, "w", encoding="utf-8") as f:
    json.dump(record, f, ensure_ascii=False, indent=2, sort_keys=True)
PY

echo "[submitted] job_id=${job_id}"
echo "[summary] experiment_root=${EXPERIMENT_ROOT}"
echo "[summary] submission_manifest=${SUBMIT_LOG}"
