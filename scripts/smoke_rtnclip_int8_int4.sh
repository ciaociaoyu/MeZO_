#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

CONDA_ENV="${CONDA_ENV:-ciao}"
TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/runs/smoke_rtnclip_int8_int4_sharedgrid_${TIMESTAMP}}"

if [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
  # shellcheck disable=SC1091
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV}"
fi

mkdir -p "${OUTPUT_DIR}"
export DATALOADER_SHUFFLE=True
export TOKENIZERS_PARALLELISM=false

python tools/smoke_rtnclip_roberta_sst5.py \
  --model_id roberta-large \
  --output_dir "${OUTPUT_DIR}" \
  --seed 16 \
  --data_seed 16 \
  --batch_size 64 \
  --steps 50 \
  --h 1e-3 \
  --group_size 128
