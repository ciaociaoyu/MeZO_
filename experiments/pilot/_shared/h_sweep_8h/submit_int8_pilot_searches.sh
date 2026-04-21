#!/bin/bash

set -euo pipefail

SCRATCH_ROOT="/scratch/jy03364/MeZO_"
EXPERIMENT_ROOT="${SCRATCH_ROOT}/experiments/pilot/_shared/h_sweep_8h"
LAUNCHER_ROOT="${EXPERIMENT_ROOT}"
SHARED_ROOT="${SCRATCH_ROOT}/experiments/pilot/_shared/h_sweep_8h"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LAUNCHER_LOG="${EXPERIMENT_ROOT}/logs/submit_int8_pilot_${TIMESTAMP}.log"
LAUNCHER_MANIFEST="${LAUNCHER_ROOT}/launcher_manifest.jsonl"
BASE_DEPENDENCY_JOB_IDS="${BASE_DEPENDENCY_JOB_IDS:-}"
BASE_DEPENDENCY_JOB_IDS_BY_LANE="${BASE_DEPENDENCY_JOB_IDS_BY_LANE:-}"
H_SWEEP_SHARDS="${H_SWEEP_SHARDS:-4}"
LANE_COUNT="${LANE_COUNT:-2}"
EXPERIMENT_START_INDEX="${EXPERIMENT_START_INDEX:-0}"
EXPERIMENT_LIMIT="${EXPERIMENT_LIMIT:-0}"
ROBERTA_INT8_MEM="${ROBERTA_INT8_MEM:-32G}"
ROBERTA_INT8_TIME="${ROBERTA_INT8_TIME:-16:00:00}"
OPT13B_MNLI_INT8_MEM="${OPT13B_MNLI_INT8_MEM:-48G}"
OPT13B_MNLI_INT8_TIME="${OPT13B_MNLI_INT8_TIME:-20:00:00}"

mkdir -p "${EXPERIMENT_ROOT}/logs" "${LAUNCHER_ROOT}"
exec > >(tee -a "${LAUNCHER_LOG}") 2>&1

source "${SHARED_ROOT}/h_values.sh"

append_launcher_manifest() {
  local label="$1"
  local script_path="$2"
  local lane="$3"
  local shard_index="$4"
  local shard_values="$5"
  local job_id="$6"
  local dependency="$7"
  python - "${LAUNCHER_MANIFEST}" "${label}" "${script_path}" "${lane}" "${shard_index}" "${shard_values}" "${job_id}" "${dependency}" "${LAUNCHER_LOG}" <<'PY'
import json
import os
import sys
from datetime import datetime

try:
    import fcntl
except ImportError:
    fcntl = None

manifest_file, label, script_path, lane, shard_index, shard_values, job_id, dependency, launcher_log = sys.argv[1:]
record = {
    "submitted_at": datetime.now().isoformat(),
    "label": label,
    "script": script_path,
    "lane": int(lane),
    "shard_index": int(shard_index),
    "shard_values": shard_values.split(),
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

mapfile -t H_SHARDS < <(
  python - "${H_SWEEP_SHARDS}" "${H_VALUES[@]}" <<'PY'
import math
import sys

shard_count = max(1, int(sys.argv[1]))
values = sys.argv[2:]
if not values:
    raise SystemExit("no h values found")
chunk = int(math.ceil(len(values) / float(shard_count)))
for start in range(0, len(values), chunk):
    subset = values[start:start + chunk]
    if subset:
        print(" ".join(subset))
PY
)

if [[ "${#H_SHARDS[@]}" -eq 0 ]]; then
  echo "No h-value shards were generated." >&2
  exit 1
fi

submit_shard_job() {
  local label="$1"
  local script_path="$2"
  local lane="$3"
  local shard_index="$4"
  local shard_values="$5"
  local dependency="$6"
  local job_name="${label}_s${shard_index}"
  local sbatch_output
  local export_spec="ALL,H_VALUES_OVERRIDE=${shard_values}"
  local -a sbatch_args=()

  # Int8 launcher shards the 8-value sweep into 4 jobs, so each RoBERTa job
  # only covers 2 h-values. Historical full-sweep jobs completed in <18h and
  # used <7G RSS, so lower resource requests help backfill without cutting it close.
  case "${label}" in
    roberta_*_int8)
      sbatch_args+=("--mem=${ROBERTA_INT8_MEM}" "--time=${ROBERTA_INT8_TIME}")
      ;;
    opt13b_mnli_mezo_int8)
      sbatch_args+=("--mem=${OPT13B_MNLI_INT8_MEM}" "--time=${OPT13B_MNLI_INT8_TIME}")
      ;;
  esac

  if [[ -n "${dependency}" ]]; then
    sbatch_output="$(sbatch "${sbatch_args[@]}" --dependency="afterok:${dependency}" --job-name "${job_name}" --export="${export_spec}" "${script_path}")"
  else
    sbatch_output="$(sbatch "${sbatch_args[@]}" --job-name "${job_name}" --export="${export_spec}" "${script_path}")"
  fi
  echo "${sbatch_output}" >&2
  local job_id
  job_id="$(echo "${sbatch_output}" | grep -oE '[0-9]+$')"
  if [[ -z "${job_id}" ]]; then
    echo "Failed to parse job id from: ${sbatch_output}" >&2
    return 1
  fi
  append_launcher_manifest "${label}" "${script_path}" "${lane}" "${shard_index}" "${shard_values}" "${job_id}" "${dependency}"
  echo "${job_id}"
}

submit_experiment_group() {
  local label="$1"
  local script_path="$2"
  local lane="$3"
  local dependency="$4"
  local -a job_ids=()
  local shard_values
  local shard_index=1
  local job_id
  for shard_values in "${H_SHARDS[@]}"; do
    echo "Submitting ${label} lane=${lane} shard=${shard_index} h=[${shard_values}]" >&2
    job_id="$(submit_shard_job "${label}" "${script_path}" "${lane}" "${shard_index}" "${shard_values}" "${dependency}")" || return 1
    job_ids+=("${job_id}")
    shard_index=$((shard_index + 1))
  done
  local joined=""
  for job_id in "${job_ids[@]}"; do
    if [[ -z "${joined}" ]]; then
      joined="${job_id}"
    else
      joined="${joined}:${job_id}"
    fi
  done
  echo "${joined}"
}

cd "${SCRATCH_ROOT}/experiments/pilot"

declare -a EXPERIMENTS=(
  "roberta_sst5_mezo_int8|mezo/roberta-large/sst5/int8/h_sweep_8h/jobs/roberta_sst5_mezo_int8_8h.sh"
  "roberta_mnli_mezo_int8|mezo/roberta-large/mnli/int8/h_sweep_8h/jobs/roberta_mnli_mezo_int8_8h.sh"
  "opt13b_mnli_mezo_int8|mezo/opt-1.3b/mnli/int8/h_sweep_8h/jobs/opt13b_mnli_mezo_int8_8h.sh"
  "roberta_sst5_sparse_mezo_int8|sparse_mezo/roberta-large/sst5/int8/h_sweep_8h/jobs/roberta_sst5_sparse_mezo_int8_8h.sh"
  "roberta_mnli_sparse_mezo_int8|sparse_mezo/roberta-large/mnli/int8/h_sweep_8h/jobs/roberta_mnli_sparse_mezo_int8_8h.sh"
  "opt13b_sst5_sparse_mezo_int8|sparse_mezo/opt-1.3b/sst5/int8/h_sweep_8h/jobs/opt13b_sst5_sparse_mezo_int8_8h.sh"
  "opt13b_mnli_sparse_mezo_int8|sparse_mezo/opt-1.3b/mnli/int8/h_sweep_8h/jobs/opt13b_mnli_sparse_mezo_int8_8h.sh"
)

declare -a lane_dependencies=()
IFS=',' read -r -a lane_base_dependencies <<< "${BASE_DEPENDENCY_JOB_IDS_BY_LANE}"
for ((lane=0; lane<LANE_COUNT; lane++)); do
  if [[ ${lane} -lt ${#lane_base_dependencies[@]} && -n "${lane_base_dependencies[$lane]}" ]]; then
    lane_dependencies[$lane]="${lane_base_dependencies[$lane]}"
  else
    lane_dependencies[$lane]="${BASE_DEPENDENCY_JOB_IDS}"
  fi
done

index=0
for item in "${EXPERIMENTS[@]}"; do
  if (( index < EXPERIMENT_START_INDEX )); then
    index=$((index + 1))
    continue
  fi
  if (( EXPERIMENT_LIMIT > 0 && index >= EXPERIMENT_START_INDEX + EXPERIMENT_LIMIT )); then
    break
  fi
  label="${item%%|*}"
  script="${item#*|}"
  lane=$((index % LANE_COUNT))
  dependency="${lane_dependencies[$lane]}"
  lane_dependencies[$lane]="$(submit_experiment_group "${label}" "${script}" "${lane}" "${dependency}")"
  index=$((index + 1))
done

echo "Shard layout:"
printf '  %s\n' "${H_SHARDS[@]}"
echo "Lane tails:"
for ((lane=0; lane<LANE_COUNT; lane++)); do
  echo "  lane ${lane}: ${lane_dependencies[$lane]}"
done
echo "Launcher log: ${LAUNCHER_LOG}"
echo "Launcher manifest: ${LAUNCHER_MANIFEST}"
echo "Monitor: squeue -u jy03364 -o '%.18i %.9P %.40j %.8T %.10M %.9l %.20R'"
