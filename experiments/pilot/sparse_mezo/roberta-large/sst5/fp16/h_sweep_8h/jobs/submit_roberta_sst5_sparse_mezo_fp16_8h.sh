#!/bin/bash

set -euo pipefail

SCRATCH_ROOT="/scratch/jy03364/MeZO_"
JOB_SCRIPT="${SCRATCH_ROOT}/experiments/pilot/sparse_mezo/roberta-large/sst5/fp16/h_sweep_8h/jobs/roberta_sst5_sparse_mezo_fp16_8h.sh"
H_VALUES_FILE="${SCRATCH_ROOT}/experiments/pilot/_shared/h_sweep_8h/h_values.sh"

[[ -f "${JOB_SCRIPT}" ]] || { echo "Missing job script: ${JOB_SCRIPT}" >&2; exit 1; }
[[ -f "${H_VALUES_FILE}" ]] || { echo "Missing h-values file: ${H_VALUES_FILE}" >&2; exit 1; }

source "${H_VALUES_FILE}"

for H in "${H_VALUES[@]}"; do
  job_name="pilot_roberta_sst5_sparse_fp16_h${H}"
  sbatch_output="$(sbatch --job-name="${job_name}" --export=ALL,H_VALUES_OVERRIDE="${H}" "${JOB_SCRIPT}")"
  echo "${H} -> ${sbatch_output}"
done
