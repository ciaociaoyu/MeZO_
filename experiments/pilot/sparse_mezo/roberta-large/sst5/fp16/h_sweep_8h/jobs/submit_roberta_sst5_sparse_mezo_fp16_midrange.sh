#!/bin/bash

set -euo pipefail

SCRATCH_ROOT="/scratch/jy03364/MeZO_"
JOB_SCRIPT="${SCRATCH_ROOT}/experiments/pilot/sparse_mezo/roberta-large/sst5/fp16/h_sweep_8h/jobs/roberta_sst5_sparse_mezo_fp16_8h.sh"

[[ -f "${JOB_SCRIPT}" ]] || { echo "Missing job script: ${JOB_SCRIPT}" >&2; exit 1; }

declare -a H_VALUES=("3e-5" "3e-4" "3e-3" "3e-2")

for H in "${H_VALUES[@]}"; do
  job_name="pilot_roberta_sst5_sparse_fp16_h${H}"
  sbatch_output="$(sbatch --job-name="${job_name}" --export=ALL,H_VALUES_OVERRIDE="${H}" "${JOB_SCRIPT}")"
  echo "${H} -> ${sbatch_output}"
done
