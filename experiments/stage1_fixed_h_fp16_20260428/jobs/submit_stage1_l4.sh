#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
MANIFEST="$SCRIPT_DIR/stage1_manifest.tsv"
ARRAY_SCRIPT="$SCRIPT_DIR/stage1_array_l4.sh"

NUM_RUNS="$(awk 'NR > 1 && NF > 0 {n++} END {print n+0}' "$MANIFEST")"
if [[ "$NUM_RUNS" -le 0 ]]; then
  echo "No runs found in manifest: $MANIFEST" >&2
  exit 2
fi

mkdir -p "$EXPERIMENT_ROOT/logs" "$EXPERIMENT_ROOT/results/status"

MAX_PARALLEL="${MAX_PARALLEL:-4}"
END_INDEX="$((NUM_RUNS - 1))"
ARRAY_SPEC="${INDICES:-0-${END_INDEX}}"

echo "Submitting Stage 1 fixed-h FP16 runs to L4"
echo "runs=$NUM_RUNS array=${ARRAY_SPEC}%${MAX_PARALLEL}"
sbatch --array="${ARRAY_SPEC}%${MAX_PARALLEL}" "$ARRAY_SCRIPT"
