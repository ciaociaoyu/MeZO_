#!/bin/bash
#SBATCH --job-name=roberta_mnli_sparse_mezo_int4_l40s
#SBATCH --partition=gpu_p
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --gres=gpu:L40S:1
#SBATCH --time=08:00:00
#SBATCH --signal=B:USR1@300
#SBATCH --requeue
#SBATCH --output=/scratch/jy03364/MeZO_/experiments/pilot/_shared/h_sweep_8h/logs/roberta_mnli_sparse_mezo_int4_l40s_%j.out

set -euo pipefail

ORIG_SCRIPT="/scratch/jy03364/MeZO_/experiments/pilot/sparse_mezo/roberta-large/mnli/int4/h_sweep_8h/jobs/roberta_mnli_sparse_mezo_int4_8h.sh"
RUN_DIR="$(dirname "$(dirname "${ORIG_SCRIPT}")")"
REQUEUE_REQUESTED=0
child_pid=""

handle_requeue_signal() {
  echo "[requeue] $(date --iso-8601=seconds) caught USR1 for ${SLURM_JOB_ID}; terminating child and requeueing"
  REQUEUE_REQUESTED=1
  if [[ -n "${child_pid}" ]]; then
    kill -TERM "${child_pid}" 2>/dev/null || true
  fi
}

trap handle_requeue_signal USR1

cd "${RUN_DIR}"
bash "${ORIG_SCRIPT}" &
child_pid=$!
set +e
wait ${child_pid}
status=$?
set -e
trap - USR1

if [[ ${REQUEUE_REQUESTED} -eq 1 ]]; then
  scontrol requeue "${SLURM_JOB_ID}"
  exit 0
fi

exit ${status}
