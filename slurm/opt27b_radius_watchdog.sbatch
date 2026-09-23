#!/usr/bin/env bash
#SBATCH --job-name=opt27b-halfhour-audit
#SBATCH --partition=batch
#SBATCH --account=nl2lab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:25:00
set -euo pipefail
cd /scratch/jy03364/MeZO_
export PATH=/home/jy03364/.nvm/versions/node/v24.14.1/bin:$PATH
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
ARGS=(--root "${1:?experiment root required}" --once --reschedule --backup --interval 1800)
if [[ "${2:-0}" == 1 ]]; then ARGS+=(--agent); fi
/home/jy03364/miniconda3/envs/mezo-mistral/bin/python tools/opt27b_radius_watchdog.py "${ARGS[@]}"
