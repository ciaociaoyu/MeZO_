#!/usr/bin/env bash
set -euo pipefail

cd /scratch/jy03364/MeZO_
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate ciao

export RESULT_ROOT=/scratch/jy03364/MeZO_/runs/future_probe_window_training_20260512_235627
export CUDA_VISIBLE_DEVICES=0
export SPARSE_MODE=exact_random
export CHECKPOINT_PROBE_NUM_DIRECTIONS=16
export CHECKPOINT_PROBE_NUM_BATCHES=1
export DENSE_SEEDS=0
export SPARSE_SCREEN_SEEDS=0

mkdir -p "$RESULT_ROOT/logs"

{
  echo "[launch] $(date -Is) start all_screen"
  echo "[launch] RESULT_ROOT=$RESULT_ROOT"
  echo "[launch] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

  MODE=all_screen bash scripts/run_future_probe_window_training.sh

  echo "[launch] $(date -Is) summarize all_screen"
  python scripts/summarize_future_probe_window_training.py "$RESULT_ROOT"
  python scripts/plot_future_probe_window_training.py "$RESULT_ROOT"

  promoted_specs="$(
    python - "$RESULT_ROOT/summary_sparse.csv" <<'PY'
import csv
import math
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.exists():
    raise SystemExit("")

rows = []
with path.open(newline="") as f:
    for row in csv.DictReader(f):
        try:
            best = float(row.get("best_acc") or "nan")
            p = float(row.get("sparse_rate") or "nan")
            h_active = float(row.get("h_active") or "nan")
            lr = float(row.get("lr") or "nan")
            seed = int(float(row.get("seed") or 0))
        except ValueError:
            continue
        if math.isfinite(best) and math.isfinite(p) and math.isfinite(h_active) and math.isfinite(lr):
            rows.append((best, p, h_active, lr, seed))

rows.sort(reverse=True)
print(",".join(f"{p:g}:{h:g}:{lr:g}:{seed:d}" for _, p, h, lr, seed in rows[:4]))
PY
  )"

  if [[ -n "$promoted_specs" ]]; then
    echo "[launch] $(date -Is) start sparse_promote specs=$promoted_specs"
    PROMOTED_SPARSE_SPECS="$promoted_specs" MODE=sparse_promote bash scripts/run_future_probe_window_training.sh
    echo "[launch] $(date -Is) summarize sparse_promote"
    python scripts/summarize_future_probe_window_training.py "$RESULT_ROOT"
    python scripts/plot_future_probe_window_training.py "$RESULT_ROOT"
  else
    echo "[launch] $(date -Is) no sparse promoted specs found"
  fi

  echo "[launch] $(date -Is) complete"
} 2>&1 | tee -a "$RESULT_ROOT/local_tmux.log"
