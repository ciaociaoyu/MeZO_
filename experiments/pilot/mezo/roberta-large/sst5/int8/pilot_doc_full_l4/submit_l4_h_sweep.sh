#!/bin/bash

set -euo pipefail

SCRATCH_ROOT="/scratch/jy03364/MeZO_"
BASE_DIR="${SCRATCH_ROOT}/experiments/pilot/mezo/roberta-large/sst5/int8/pilot_doc_full_l4"
RUNNER="${BASE_DIR}/job_runner.sh"
STAMP="$(date +%Y%m%d_%H%M%S)"
EXPERIMENT_ROOT="${BASE_DIR}/runs/${STAMP}"
LOG_DIR="${EXPERIMENT_ROOT}/logs"
SUBMIT_LOG="${EXPERIMENT_ROOT}/submission_manifest.jsonl"
CONFIG_SNAPSHOT="${EXPERIMENT_ROOT}/launch_config.json"
VARIANT_NAME="mezo_int8_pilotdoc_full_10k_bs64_lr1e-6_wd0_nanonly"

H_VALUES=(1e-2 1e-3 1e-4 1e-5 1e-6 1e-7 1e-8 1e-9)
if [[ -n "${SUBMIT_ONLY_HS:-}" ]]; then
  read -r -a H_VALUES <<< "${SUBMIT_ONLY_HS}"
fi

mkdir -p "${LOG_DIR}"

python - "${CONFIG_SNAPSHOT}" "${EXPERIMENT_ROOT}" <<'PY'
import json
import sys
from datetime import datetime

path, experiment_root = sys.argv[1:]
record = {
    "created_at": datetime.now().isoformat(),
    "doc_reference": "docs/pilot_experiments_20260419.md",
    "experiment_root": experiment_root,
    "model": "roberta-large",
    "task": "sst-5",
    "method": "mezo",
    "precision": "int8",
    "variant": "mezo_int8_pilotdoc_full_10k_bs64_lr1e-6_wd0_nanonly",
    "dataset_mode": "full",
    "seed": 16,
    "data_seed": 16,
    "dataloader_shuffle": True,
    "k": 16,
    "max_steps": 10000,
    "eval_steps": 1000,
    "per_device_train_batch_size": 64,
    "learning_rate": 1e-6,
    "weight_decay": 0.0,
    "zo_probe_every": 200,
    "zo_probe_num_seeds": 16,
    "nan_guard_limit": 1,
    "random_prediction_guard_enabled": False,
    "zo_probe_health_guard_enabled": False,
    "h_values": ["1e-2", "1e-3", "1e-4", "1e-5", "1e-6", "1e-7", "1e-8", "1e-9"],
}
with open(path, "w", encoding="utf-8") as f:
    json.dump(record, f, ensure_ascii=False, indent=2, sort_keys=True)
PY

touch "${SUBMIT_LOG}"

submit_one() {
  local h="$1"
  local sanitized_h="${h/-/m}"
  local job_name="rb_sst5_i8_${sanitized_h}_L4"
  local slurm_out="${LOG_DIR}/slurm_${job_name}_%j.out"
  local job_id

  job_id="$(
    sbatch --parsable \
      --job-name="${job_name}" \
      --partition=gpu_p \
      --ntasks=1 \
      --cpus-per-task=4 \
      --mem=48G \
      --gres=gpu:L4:1 \
      --time=24:00:00 \
      --chdir="${BASE_DIR}" \
      --output="${slurm_out}" \
      --export=ALL,EXPERIMENT_ROOT="${EXPERIMENT_ROOT}",VARIANT="${VARIANT_NAME}",H_VALUES_OVERRIDE="${h}" \
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
    "time_limit": "24:00:00",
    "mem": "48G",
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
