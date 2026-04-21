#!/bin/bash

set -euo pipefail

SCRATCH_ROOT="/scratch/jy03364/MeZO_"
EXPERIMENT_ROOT="${SCRATCH_ROOT}/experiments/pilot/mezo/roberta-large/sst5/int8/noguard_50k_20260420"
RUNNER="${EXPERIMENT_ROOT}/job_runner.sh"
LOG_DIR="${EXPERIMENT_ROOT}/logs"
SUBMIT_LOG="${EXPERIMENT_ROOT}/submission_manifest.jsonl"

mkdir -p "${LOG_DIR}"

H_VALUES=(1e-6 3e-6 1e-5 3e-5 1e-4 3e-4 1e-3 3e-3)
L40S_PRIORITY=(1e-4 3e-4 3e-5 1e-5 1e-3 1e-6 3e-3 3e-6)

if [[ -n "${SUBMIT_ONLY_HS:-}" ]]; then
  read -r -a H_VALUES <<< "${SUBMIT_ONLY_HS}"
fi

free_l40s() {
  python - <<'PY'
import re
import subprocess

text = subprocess.check_output(["scontrol", "show", "node", "rb7-4"], text=True)
cfg = re.search(r"Gres=gpu:L40S:(\d+)", text)
alloc = re.search(r"AllocTRES=.*?gres/gpu:l40s=(\d+)", text)
cfg_count = int(cfg.group(1)) if cfg else 0
alloc_count = int(alloc.group(1)) if alloc else 0
print(max(0, cfg_count - alloc_count))
PY
}

FREE_L40S="$(free_l40s)"
if [[ "${FREE_L40S}" -gt 0 ]]; then
  mapfile -t L40S_HS < <(printf '%s\n' "${L40S_PRIORITY[@]}" | head -n "${FREE_L40S}")
else
  L40S_HS=()
fi

declare -A USE_L40S=()
for h in "${L40S_HS[@]}"; do
  USE_L40S["${h}"]=1
done

touch "${SUBMIT_LOG}"

submit_one() {
  local h="$1"
  local gpu_type="$2"
  local time_limit="$3"
  local mem_gb="$4"
  local job_name="rb_sst5_i8_${h}_${gpu_type}"
  local slurm_out="${LOG_DIR}/slurm_${job_name}_%j.out"
  local job_id

  set +e
  job_id="$(
    sbatch --parsable \
      --job-name="${job_name}" \
      --partition=gpu_p \
      --ntasks=1 \
      --cpus-per-task=4 \
      --mem="${mem_gb}" \
      --gres="gpu:${gpu_type}:1" \
      --time="${time_limit}" \
      --chdir="${EXPERIMENT_ROOT}" \
      --output="${slurm_out}" \
      --export=ALL,EXPERIMENT_ROOT="${EXPERIMENT_ROOT}",VARIANT=mezo_int8_noguard_50k,H_VALUES_OVERRIDE="${h}" \
      "${RUNNER}" 2>&1
  )"
  submit_status=$?
  set -e

  if [[ ${submit_status} -ne 0 ]]; then
    if [[ "${gpu_type}" == "L40S" ]]; then
      echo "[fallback] h=${h} L40S submit failed: ${job_id}"
      submit_one "${h}" "L4" "36:00:00" "32G"
      return
    fi
    echo "[submit-failed] h=${h} gpu=${gpu_type} error=${job_id}" >&2
    return 1
  fi

  python - "${SUBMIT_LOG}" "${job_id}" "${h}" "${gpu_type}" "${time_limit}" "${mem_gb}" "${job_name}" "${slurm_out}" <<'PY'
import json
import sys
from datetime import datetime

path, job_id, h, gpu_type, time_limit, mem_gb, job_name, slurm_out = sys.argv[1:]
record = {
    "submitted_at": datetime.now().isoformat(),
    "job_id": int(job_id),
    "h": h,
    "gpu_type": gpu_type,
    "time_limit": time_limit,
    "mem": mem_gb,
    "job_name": job_name,
    "slurm_out": slurm_out,
}
with open(path, "a", encoding="utf-8") as f:
    f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
PY

  echo "[submitted] h=${h} gpu=${gpu_type} job_id=${job_id}"
}

for h in "${H_VALUES[@]}"; do
  if [[ -n "${USE_L40S[$h]:-}" ]]; then
    submit_one "${h}" "L40S" "24:00:00" "48G"
  else
    submit_one "${h}" "L4" "36:00:00" "32G"
  fi
done

echo "[summary] submission_manifest=${SUBMIT_LOG}"
