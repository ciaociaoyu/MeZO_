#!/bin/bash

set -euo pipefail

SCRATCH_ROOT="/scratch/jy03364/MeZO_"
BASE_DIR="${SCRATCH_ROOT}/experiments/pilot/mezo/roberta-large/sst5/int8/paper_k16_10k_bs64_lr1e-6_wd0_shuffle_nanguardonly_l4"
RUNNER="${BASE_DIR}/job_runner.sh"
STAMP="$(date +%Y%m%d_%H%M%S)"
EXPERIMENT_ROOT="${BASE_DIR}/runs/${STAMP}"
LOG_DIR="${EXPERIMENT_ROOT}/logs"
SUBMIT_LOG="${EXPERIMENT_ROOT}/submission_manifest.jsonl"
CONFIG_SNAPSHOT="${EXPERIMENT_ROOT}/launch_config.json"
DATA_PREP_LOG="${EXPERIMENT_ROOT}/data_prep.log"

H_VALUES=(1e-6 3e-6 1e-5 3e-5 1e-4 3e-4 1e-3 3e-3)
if [[ -n "${SUBMIT_ONLY_HS:-}" ]]; then
  read -r -a H_VALUES <<< "${SUBMIT_ONLY_HS}"
fi

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
    "paper_regime_label": "paper_kshot",
    "k": 16,
    "seed": 16,
    "data_seed": 16,
    "dataloader_shuffle": True,
    "max_steps": 10000,
    "eval_steps": 1000,
    "per_device_train_batch_size": 64,
    "learning_rate": 1e-6,
    "weight_decay": 0.0,
    "zo_probe_every": 200,
    "zo_probe_num_seeds": 16,
    "random_prediction_guard_enabled": False,
    "zo_probe_health_guard_enabled": False,
    "nan_guard_limit": 100,
    "h_values": ["1e-6", "3e-6", "1e-5", "3e-5", "1e-4", "3e-4", "1e-3", "3e-3"],
    "doc_reference": "docs/pilot_experiments_20260419.md",
    "notes": [
        "Results are isolated under a fresh timestamped experiment root.",
        "Only NaN guard remains enabled; random-prediction and probe-health guards are disabled.",
        "This batch uses the doc's primary LR anchor 1e-6 and wd=0 for a single 8-job h-sweep.",
    ],
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

touch "${SUBMIT_LOG}"

submit_one() {
  local h="$1"
  local job_name="rb_sst5_i8_p16_${h}_L4"
  local slurm_out="${LOG_DIR}/slurm_${job_name}_%j.out"
  local job_id

  job_id="$(
    sbatch --parsable \
      --job-name="${job_name}" \
      --partition=gpu_p \
      --ntasks=1 \
      --cpus-per-task=4 \
      --mem=32G \
      --gres=gpu:L4:1 \
      --time=12:00:00 \
      --chdir="${BASE_DIR}" \
      --output="${slurm_out}" \
      --export=ALL,EXPERIMENT_ROOT="${EXPERIMENT_ROOT}",VARIANT=mezo_int8_paperk16_10k_bs64_lr1e-6_wd0_shuffle_nanguardonly,H_VALUES_OVERRIDE="${h}" \
      "${RUNNER}"
  )"

  python - "${SUBMIT_LOG}" "${job_id}" "${h}" "${job_name}" "${slurm_out}" "${EXPERIMENT_ROOT}" <<'PY'
import json
import sys
from datetime import datetime

path, job_id, h, job_name, slurm_out, experiment_root = sys.argv[1:]
record = {
    "submitted_at": datetime.now().isoformat(),
    "job_id": int(job_id),
    "h": h,
    "gpu_type": "L4",
    "job_name": job_name,
    "time_limit": "12:00:00",
    "mem": "32G",
    "slurm_out": slurm_out,
    "experiment_root": experiment_root,
}
with open(path, "a", encoding="utf-8") as f:
    f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
PY

  echo "[submitted] h=${h} gpu=L4 job_id=${job_id}"
}

for h in "${H_VALUES[@]}"; do
  submit_one "${h}"
done

echo "[summary] experiment_root=${EXPERIMENT_ROOT}"
echo "[summary] submission_manifest=${SUBMIT_LOG}"
