#!/bin/bash

set -euo pipefail

SCRATCH_ROOT="/scratch/jy03364/MeZO_"
EXPERIMENT_ROOT="${SCRATCH_ROOT}/experiments/h_sweep_8h"
LAUNCHER_ROOT="${EXPERIMENT_ROOT}/results"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LAUNCHER_LOG="${EXPERIMENT_ROOT}/logs/submit_int4_pilot_${TIMESTAMP}.log"
LAUNCHER_MANIFEST="${LAUNCHER_ROOT}/launcher_manifest_int4.jsonl"
BASE_DEPENDENCY_JOB_ID="${BASE_DEPENDENCY_JOB_ID:-}"

mkdir -p "${EXPERIMENT_ROOT}/logs" "${LAUNCHER_ROOT}"
exec > >(tee -a "${LAUNCHER_LOG}") 2>&1

append_launcher_manifest() {
  local label="$1"
  local script_path="$2"
  local job_id="$3"
  local dependency="$4"
  python - "${LAUNCHER_MANIFEST}" "${label}" "${script_path}" "${job_id}" "${dependency}" "${LAUNCHER_LOG}" <<'PY'
import json
import os
import sys
from datetime import datetime

try:
    import fcntl
except ImportError:
    fcntl = None

manifest_file, label, script_path, job_id, dependency, launcher_log = sys.argv[1:]
record = {
    "submitted_at": datetime.now().isoformat(),
    "label": label,
    "script": script_path,
    "job_id": int(job_id),
    "dependency": dependency or None,
    "status": "submitted",
    "launcher_log": launcher_log,
}
os.makedirs(os.path.dirname(manifest_file), exist_ok=True)
lock_path = manifest_file + ".lock"
with open(lock_path, "w", encoding="utf-8") as lock_file:
    if fcntl is not None:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
    with open(manifest_file, "a", encoding="utf-8") as out_file:
        out_file.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
PY
}

submit_job() {
  local label="$1"
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
  append_launcher_manifest "${label}" "${script_path}" "${job_id}" "${dependency}"
  echo "${job_id}"
}

cd "${EXPERIMENT_ROOT}"

dependency="${BASE_DEPENDENCY_JOB_ID}"
for item in \
  "roberta_mnli_mezo_int4 jobs/roberta_mnli_mezo_int4_8h.sh" \
  "roberta_sst5_mezo_int4 jobs/roberta_sst5_mezo_int4_8h.sh" \
  "opt13b_mnli_mezo_int4 jobs/opt13b_mnli_mezo_int4_8h.sh" \
  "opt13b_sst5_mezo_int4 jobs/opt13b_sst5_mezo_int4_8h.sh" \
  "roberta_mnli_sparse_mezo_int4 jobs/roberta_mnli_sparse_mezo_int4_8h.sh" \
  "roberta_sst5_sparse_mezo_int4 jobs/roberta_sst5_sparse_mezo_int4_8h.sh" \
  "opt13b_mnli_sparse_mezo_int4 jobs/opt13b_mnli_sparse_mezo_int4_8h.sh" \
  "opt13b_sst5_sparse_mezo_int4 jobs/opt13b_sst5_sparse_mezo_int4_8h.sh"
do
  label="${item%% *}"
  script="${item#* }"
  echo "Submitting ${label} (${script})"
  dependency="$(submit_job "${label}" "${script}" "${dependency}")"
done

if [[ -n "${BASE_DEPENDENCY_JOB_ID}" ]]; then
  echo "Base dependency job id: ${BASE_DEPENDENCY_JOB_ID}"
fi
echo "Last job id in chain: ${dependency}"
echo "Launcher log: ${LAUNCHER_LOG}"
echo "Launcher manifest: ${LAUNCHER_MANIFEST}"
echo "Monitor: squeue -u jy03364 -o '%.18i %.9P %.40j %.8T %.10M %.9l %.20R'"
