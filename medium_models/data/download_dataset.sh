#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

DATASET_URL="https://nlp.cs.princeton.edu/projects/lm-bff/datasets.tar"
DATASET_TAR="datasets.tar"
FORCE_EXTRACT="${FORCE_EXTRACT:-0}"

if [[ ! -f "${DATASET_TAR}" ]]; then
  wget "${DATASET_URL}" -O "${DATASET_TAR}"
else
  echo "*** ${DATASET_TAR} already exists; skip download ***"
fi

if [[ ! -d original || "${FORCE_EXTRACT}" == "1" ]]; then
  if [[ "${FORCE_EXTRACT}" == "1" && -d original ]]; then
    echo "*** FORCE_EXTRACT=1; re-extract into existing original/ ***"
  fi
  tar xvf "${DATASET_TAR}"
else
  echo "*** original/ already exists; skip extract ***"
fi

echo "*** Use GLUE-SST-2 as default SST-2 ***"
if [[ -d original/GLUE-SST-2 ]]; then
  if [[ -d original/SST-2 && ! -d original/SST-2-original ]]; then
    mv original/SST-2 original/SST-2-original
  fi
  if [[ ! -d original/SST-2 ]]; then
    mv original/GLUE-SST-2 original/SST-2
  fi
fi

echo "*** Done ***"
