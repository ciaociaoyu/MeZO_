#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
JOB_SCRIPT="${SCRIPT_DIR}/roberta_sst5_sparse_mezo_fp16_8h.sh"
LOCAL_LOG_DIR="${EXPERIMENT_ROOT}/logs/local_l4_4way"

mkdir -p "${LOCAL_LOG_DIR}"

declare -a H_VALUES=("1e-4" "1e-5" "1e-6" "1e-7")

for idx in "${!H_VALUES[@]}"; do
  h="${H_VALUES[$idx]}"
  gpu="${idx}"
  launcher_log="${LOCAL_LOG_DIR}/h_${h}_gpu${gpu}.launcher.log"
  pid_file="${LOCAL_LOG_DIR}/h_${h}_gpu${gpu}.pid"

  nohup bash -lc "
    set -euo pipefail
    cd '${EXPERIMENT_ROOT}'
    export CUDA_VISIBLE_DEVICES='${gpu}'
    export H_VALUES_OVERRIDE='${h}'
    export OMP_NUM_THREADS=1
    export MKL_NUM_THREADS=1
    export OPENBLAS_NUM_THREADS=1
    export NUMEXPR_NUM_THREADS=1
    export TOKENIZERS_PARALLELISM=false
    bash '${JOB_SCRIPT}'
  " >"${launcher_log}" 2>&1 &

  echo $! > "${pid_file}"
  echo "launched h=${h} gpu=${gpu} pid=$(cat "${pid_file}") log=${launcher_log}"
done
