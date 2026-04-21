#!/bin/bash
#SBATCH --job-name=roberta_sst5_mezo_int8_l40s_speed
#SBATCH --partition=gpu_p
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:L40S:1
#SBATCH --time=00:20:00
#SBATCH --chdir=/scratch/jy03364/MeZO_/medium_models
#SBATCH --output=/scratch/jy03364/MeZO_/experiments/pilot/mezo/roberta-large/sst5/int8/speed_bench_l40s/logs/slurm_%x_%j.out

set -euo pipefail

SCRATCH_ROOT="/scratch/jy03364/MeZO_"
RESULT_ROOT="${SCRATCH_ROOT}/experiments/pilot/mezo/roberta-large/sst5/int8/speed_bench_l40s/zo_method_matrix_20260420"
COMMAND_LOG="${RESULT_ROOT}/command.sh"

mkdir -p "${RESULT_ROOT}" "${SCRATCH_ROOT}/experiments/pilot/mezo/roberta-large/sst5/int8/speed_bench_l40s/logs"

set +u
source /home/jy03364/miniconda3/etc/profile.d/conda.sh
conda activate ciao
set -u

CMD=(
  env
  TASK=SST-5
  K=16
  SEED=16
  DATA_SEED=16
  DATASET_MODE=full
  FULL_DEV_RATIO=0.1
  BS=32
  LR=1e-6
  WD=0
  STEP=5
  EVAL_STEP=5000
  MODEL=roberta-large
  USE_H=False
  USE_C=False
  DATALOADER_SHUFFLE=False
  EPS=1e-4
  EFFICIENT_ZERO_ORDER=True
  EXTRA_TAG=zo-matrix-mezo-int8-sst5-l40s
  bash
  ./mezo.sh
  --result_root
  "${RESULT_ROOT}"
  --job_name
  run
  --measure_perf_tail_window_steps
  3
  --zo_probe_every
  0
  --zo_method
  mezo
  --zo_two_point_precision
  fp16
  --zo_quantization
  int8
)

printf '%q ' "${CMD[@]}" > "${COMMAND_LOG}"
printf '\n' >> "${COMMAND_LOG}"

"${CMD[@]}"
