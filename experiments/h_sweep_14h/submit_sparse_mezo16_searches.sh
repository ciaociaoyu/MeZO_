#!/bin/bash

set -euo pipefail

SCRATCH_ROOT="/scratch/jy03364/MeZO_"
EXPERIMENT_ROOT="${SCRATCH_ROOT}/experiments/h_sweep_14h"
VARIANT="sparse_mezo16"
LAUNCHER_ROOT="${EXPERIMENT_ROOT}/results/${VARIANT}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LAUNCHER_LOG="${EXPERIMENT_ROOT}/logs/submit_${VARIANT}_${TIMESTAMP}.log"
LAUNCHER_MANIFEST="${LAUNCHER_ROOT}/launcher_manifest.jsonl"
BASE_DEPENDENCY_JOB_ID="${BASE_DEPENDENCY_JOB_ID:-}"

mkdir -p "${EXPERIMENT_ROOT}/logs" "${LAUNCHER_ROOT}"
exec > >(tee -a "${LAUNCHER_LOG}") 2>&1

append_launcher_manifest() {
  local task_key="$1"
  local script_path="$2"
  local job_id="$3"
  local dependency="$4"
  python - "${LAUNCHER_MANIFEST}" "${task_key}" "${script_path}" "${job_id}" "${dependency}" "${LAUNCHER_LOG}" <<'PY'
import json
import os
import sys
from datetime import datetime

try:
    import fcntl
except ImportError:
    fcntl = None

manifest_file, task_key, script_path, job_id, dependency, launcher_log = sys.argv[1:]
record = {
    "submitted_at": datetime.now().isoformat(),
    "task": task_key,
    "script": script_path,
    "job_id": int(job_id),
    "dependency": dependency or None,
    "status": "submitted",
    "launcher_log": launcher_log,
}

manifest_dir = os.path.dirname(manifest_file)
if manifest_dir:
    os.makedirs(manifest_dir, exist_ok=True)

lock_path = manifest_file + ".lock"
with open(lock_path, "w", encoding="utf-8") as lock_file:
    if fcntl is not None:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
    with open(manifest_file, "a", encoding="utf-8") as out_file:
        out_file.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
PY
}

submit_job() {
  local task_key="$1"
  local script_path="$2"
  local dependency="$3"
  local sbatch_output
  if [[ -n "${dependency}" ]]; then
    sbatch_output="$(sbatch --dependency=afterany:${dependency} "${script_path}")"
  else
    sbatch_output="$(sbatch "${script_path}")"
  fi
  echo "${sbatch_output}" >&2
  local job_id
  job_id="$(echo "${sbatch_output}" | grep -oE '[0-9]+$')"
  if [[ -z "${job_id}" ]]; then
    echo "Failed to parse job id from: ${sbatch_output}" >&2
    return 1
  fi
  append_launcher_manifest "${task_key}" "${script_path}" "${job_id}" "${dependency}"
  echo "${job_id}"
}

cd "${EXPERIMENT_ROOT}"

echo "Submitting Sparse MeZO 16-bit MNLI 14-value search"
mnli_job_id="$(submit_job "mnli" "jobs/roberta_mnli_sparse_mezo16_14h.sh" "${BASE_DEPENDENCY_JOB_ID}")"
echo "Submitting Sparse MeZO 16-bit SST-5 14-value search with dependency afterany:${mnli_job_id}"
sst5_job_id="$(submit_job "sst5" "jobs/roberta_sst5_sparse_mezo16_14h.sh" "${mnli_job_id}")"

if [[ -n "${BASE_DEPENDENCY_JOB_ID}" ]]; then
  echo "Base dependency job id: ${BASE_DEPENDENCY_JOB_ID}"
fi
echo "MNLI job id: ${mnli_job_id}"
echo "SST-5 job id: ${sst5_job_id}"
echo "Launcher log: ${LAUNCHER_LOG}"
echo "Launcher manifest: ${LAUNCHER_MANIFEST}"
echo "Monitor: squeue -j ${mnli_job_id},${sst5_job_id}"
