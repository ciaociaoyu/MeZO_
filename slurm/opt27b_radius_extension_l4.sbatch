#!/usr/bin/env bash
#SBATCH --job-name=opt27b-hwindow
#SBATCH --partition=gpu_p
#SBATCH --account=nl2lab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:L4:1
#SBATCH --time=72:00:00
#SBATCH --signal=B:USR1@600
#SBATCH --requeue

set -euo pipefail
cd /scratch/jy03364/MeZO_
RUN_ROOT="${1:?result directory required}"
PYTHON=/home/jy03364/miniconda3/envs/mezo-mistral/bin/python
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
test -s "${RUN_ROOT}/SMOKE_PASSED.json"
"${PYTHON}" -c 'import torch, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot'
nvidia-smi
"${PYTHON}" -u tools/opt27b_radius_extension.py worker --root "${RUN_ROOT}" --kind train &
worker_pid=$!
requeue_requested=0
trap 'requeue_requested=1; kill -USR1 "${worker_pid}" 2>/dev/null || true' USR1
trap 'kill -TERM "${worker_pid}" 2>/dev/null || true' TERM
set +e
wait "${worker_pid}"
rc=$?
if kill -0 "${worker_pid}" 2>/dev/null; then
  wait "${worker_pid}"
  rc=$?
fi
set -e
"${PYTHON}" tools/opt27b_radius_extension.py aggregate --root "${RUN_ROOT}" || echo "Aggregation failed; retained raw records and worker exit status." >&2
if [[ "${rc}" == 75 && "${requeue_requested}" == 1 ]]; then
  scontrol requeue "${SLURM_JOB_ID}"
  exit 0
fi
exit "${rc}"
