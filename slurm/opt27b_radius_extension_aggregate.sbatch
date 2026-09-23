#!/usr/bin/env bash
#SBATCH --job-name=opt27b-h-summary
#SBATCH --partition=batch
#SBATCH --account=nl2lab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:30:00
set -euo pipefail
cd /scratch/jy03364/MeZO_
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
/home/jy03364/miniconda3/envs/mezo-mistral/bin/python tools/opt27b_radius_extension.py aggregate --root "${1:?root required}"
