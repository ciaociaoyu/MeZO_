#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
STATUS_DIR="$EXPERIMENT_ROOT/results/status"

echo "Slurm jobs matching stage1-fp16-l4:"
squeue -u "$USER" -n stage1-fp16-l4 -o "%.18i %.9P %.20j %.8T %.10M %.9l %.6D %R" || true

echo
echo "Local status files:"
if compgen -G "$STATUS_DIR/*.json" > /dev/null; then
  python - "$STATUS_DIR" <<'PY'
import glob
import json
import os
import sys

status_dir = sys.argv[1]
rows = []
for path in sorted(glob.glob(os.path.join(status_dir, "*.json")), key=lambda p: int(os.path.splitext(os.path.basename(p))[0])):
    with open(path, "r", encoding="utf-8") as f:
        r = json.load(f)
    rows.append(r)

print("idx\tstate\texit\tmodel\ttask\tmethod\th\tupdated")
for r in rows:
    print(
        f"{r.get('index')}\t{r.get('state')}\t{r.get('exit_code')}\t"
        f"{r.get('model_key')}\t{r.get('task_key')}\t{r.get('method')}\t"
        f"{r.get('h')}\t{r.get('updated_at')}"
    )
PY
else
  echo "No status files yet."
fi

