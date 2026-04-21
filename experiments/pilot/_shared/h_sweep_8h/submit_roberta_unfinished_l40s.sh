#!/bin/bash

set -euo pipefail

SCRATCH_ROOT="/scratch/jy03364/MeZO_"
EXPERIMENT_ROOT="${SCRATCH_ROOT}/experiments/pilot/_shared/h_sweep_8h"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LAUNCHER_LOG="${EXPERIMENT_ROOT}/logs/submit_roberta_unfinished_l40s_${TIMESTAMP}.log"
LAUNCHER_MANIFEST="${EXPERIMENT_ROOT}/launcher_manifest_roberta_unfinished_l40s.jsonl"

# The dedicated iai_L40_p partition is visible but not actually usable from this
# account/QOS combination. Submit to gpu_p with an 8h walltime and let the job
# requeue itself to continue the sweep until all h-values are finished.
PARTITION="${PARTITION:-gpu_p}"
GPU_SPEC="${GPU_SPEC:-gpu:L40S:1}"
CPUS_PER_TASK="${CPUS_PER_TASK:-4}"
MEM_PER_JOB="${MEM_PER_JOB:-48G}"
TIME_LIMIT="${TIME_LIMIT:-08:00:00}"
REQUEUE_SIGNAL_LEAD="${REQUEUE_SIGNAL_LEAD:-300}"

mkdir -p "${EXPERIMENT_ROOT}/logs"
GENERATED_JOB_ROOT="${EXPERIMENT_ROOT}/generated_l40s_jobs/${TIMESTAMP}"
mkdir -p "${GENERATED_JOB_ROOT}"
exec > >(tee -a "${LAUNCHER_LOG}") 2>&1

append_launcher_manifest() {
  local label="$1"
  local script_path="$2"
  local job_id="$3"
  python - "${LAUNCHER_MANIFEST}" "${label}" "${script_path}" "${job_id}" "${PARTITION}" "${GPU_SPEC}" "${CPUS_PER_TASK}" "${MEM_PER_JOB}" "${TIME_LIMIT}" "${LAUNCHER_LOG}" <<'PY'
import json
import os
import sys
from datetime import datetime

try:
    import fcntl
except ImportError:
    fcntl = None

manifest_file, label, script_path, job_id, partition, gpu_spec, cpus_per_task, mem_per_job, time_limit, launcher_log = sys.argv[1:]
record = {
    "submitted_at": datetime.now().isoformat(),
    "label": label,
    "script": script_path,
    "job_id": int(job_id),
    "partition": partition,
    "gpu_spec": gpu_spec,
    "cpus_per_task": int(cpus_per_task),
    "mem_per_job": mem_per_job,
    "time_limit": time_limit,
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
  local job_name="${label}_l40s"
  local rendered_script="${GENERATED_JOB_ROOT}/${job_name}.sh"
  cat > "${rendered_script}" <<EOF
#!/bin/bash
#SBATCH --job-name=${job_name}
#SBATCH --partition=${PARTITION}
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=${CPUS_PER_TASK}
#SBATCH --mem=${MEM_PER_JOB}
#SBATCH --gres=${GPU_SPEC}
#SBATCH --time=${TIME_LIMIT}
#SBATCH --signal=B:USR1@${REQUEUE_SIGNAL_LEAD}
#SBATCH --requeue
#SBATCH --output=${EXPERIMENT_ROOT}/logs/${job_name}_%j.out

set -euo pipefail

ORIG_SCRIPT="${SCRATCH_ROOT}/experiments/pilot/${script_path}"
RUN_DIR="\$(dirname "\$(dirname "\${ORIG_SCRIPT}")")"
REQUEUE_REQUESTED=0
child_pid=""

handle_requeue_signal() {
  echo "[requeue] \$(date --iso-8601=seconds) caught USR1 for \${SLURM_JOB_ID}; terminating child and requeueing"
  REQUEUE_REQUESTED=1
  if [[ -n "\${child_pid}" ]]; then
    kill -TERM "\${child_pid}" 2>/dev/null || true
  fi
}

trap handle_requeue_signal USR1

cd "\${RUN_DIR}"
bash "\${ORIG_SCRIPT}" &
child_pid=\$!
set +e
wait \${child_pid}
status=\$?
set -e
trap - USR1

if [[ \${REQUEUE_REQUESTED} -eq 1 ]]; then
  scontrol requeue "\${SLURM_JOB_ID}"
  exit 0
fi

exit \${status}
EOF
  chmod +x "${rendered_script}"

  local sbatch_output
  sbatch_output="$(sbatch "${rendered_script}")"
  echo "${sbatch_output}" >&2
  local job_id
  job_id="$(echo "${sbatch_output}" | grep -oE '[0-9]+$')"
  if [[ -z "${job_id}" ]]; then
    echo "Failed to parse job id from: ${sbatch_output}" >&2
    return 1
  fi
  append_launcher_manifest "${label}" "${rendered_script}" "${job_id}"
  echo "${job_id}"
}

cd "${SCRATCH_ROOT}/experiments/pilot"

# Dense int8 RoBERTa pilot runs already completed on H100. The unfinished set is:
# - sparse int8: started but never finished successfully
# - dense/sparse int4: no completed pilot runs submitted yet
declare -a EXPERIMENTS=(
  "roberta_sst5_sparse_mezo_int8|sparse_mezo/roberta-large/sst5/int8/h_sweep_8h/jobs/roberta_sst5_sparse_mezo_int8_8h.sh"
  "roberta_mnli_sparse_mezo_int8|sparse_mezo/roberta-large/mnli/int8/h_sweep_8h/jobs/roberta_mnli_sparse_mezo_int8_8h.sh"
  "roberta_sst5_mezo_int4|mezo/roberta-large/sst5/int4/h_sweep_8h/jobs/roberta_sst5_mezo_int4_8h.sh"
  "roberta_mnli_mezo_int4|mezo/roberta-large/mnli/int4/h_sweep_8h/jobs/roberta_mnli_mezo_int4_8h.sh"
  "roberta_sst5_sparse_mezo_int4|sparse_mezo/roberta-large/sst5/int4/h_sweep_8h/jobs/roberta_sst5_sparse_mezo_int4_8h.sh"
  "roberta_mnli_sparse_mezo_int4|sparse_mezo/roberta-large/mnli/int4/h_sweep_8h/jobs/roberta_mnli_sparse_mezo_int4_8h.sh"
)

for item in "${EXPERIMENTS[@]}"; do
  label="${item%%|*}"
  script="${item#*|}"
  echo "Submitting ${label} on ${GPU_SPEC} (${script})"
  submit_job "${label}" "${script}" >/dev/null
done

echo "Launcher log: ${LAUNCHER_LOG}"
echo "Launcher manifest: ${LAUNCHER_MANIFEST}"
echo "Monitor: squeue -u jy03364 -o '%.18i %.9P %.40j %.8T %.10M %.9l %.20R'"
