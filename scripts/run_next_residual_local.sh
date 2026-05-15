#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

timestamp() {
  date +"%Y%m%d_%H%M%S"
}

RESULT_ROOT="${RESULT_ROOT:-${ROOT_DIR}/runs/next_residual_local_$(timestamp)}"
CONDA_ENV="${CONDA_ENV:-ciao}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
DATA_SEED="${DATA_SEED:-16}"
BS="${BS:-64}"
EVAL_EVERY="${EVAL_EVERY:-100}"
SEED="${SEED:-0}"
AUTO_PROMOTE="${AUTO_PROMOTE:-0}"

mkdir -p "${RESULT_ROOT}/logs"

if [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
  # Keep local H100 runs independent from the caller's shell environment.
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV}"
fi

write_manifest() {
  python - "${RESULT_ROOT}" "${SEED}" "${EVAL_EVERY}" <<'PY'
import json
import sys
from pathlib import Path
root = Path(sys.argv[1])
seed = int(sys.argv[2])
eval_every = int(float(sys.argv[3]))
cases = [
    {
        "family": "residual",
        "run_name": "residual_grid_round_lr3e-5_step500",
        "seed": seed,
        "precision_mode": "int8",
        "zo_quantization": "int8",
        "update_backend": "residual_grid",
        "direction_type": "dense",
        "h_raw": 3e-3,
        "h_active": 3e-3,
        "sparse_rate": 1.0,
        "sparse_mode": "none",
        "sparse_rescale": "none",
        "lr": "3e-5",
        "steps": 500,
        "eval_every": eval_every,
        "residual_commit_mode": "round",
        "residual_dtype": "fp32",
        "residual_max_code_step": 0,
        "zo_update_norm_clip": 0,
    },
    {
        "family": "residual",
        "run_name": "residual_grid_round_step1_lr1e-4_clip5_step500",
        "seed": seed,
        "precision_mode": "int8",
        "zo_quantization": "int8",
        "update_backend": "residual_grid",
        "direction_type": "dense",
        "h_raw": 3e-3,
        "h_active": 3e-3,
        "sparse_rate": 1.0,
        "sparse_mode": "none",
        "sparse_rescale": "none",
        "lr": "1e-4",
        "steps": 500,
        "eval_every": eval_every,
        "residual_commit_mode": "round",
        "residual_dtype": "fp32",
        "residual_max_code_step": 1,
        "zo_update_norm_clip": 5,
    },
    {
        "family": "residual",
        "run_name": "residual_grid_stoch_step1_lr3e-4_clip10_step500",
        "seed": seed,
        "precision_mode": "int8",
        "zo_quantization": "int8",
        "update_backend": "residual_grid",
        "direction_type": "dense",
        "h_raw": 3e-3,
        "h_active": 3e-3,
        "sparse_rate": 1.0,
        "sparse_mode": "none",
        "sparse_rescale": "none",
        "lr": "3e-4",
        "steps": 500,
        "eval_every": eval_every,
        "residual_commit_mode": "stochastic",
        "residual_dtype": "fp32",
        "residual_max_code_step": 1,
        "zo_update_norm_clip": 10,
    },
]
manifest = {
    "manifest_schema_version": 1,
    "mode": "residual_stage_a",
    "cases": cases,
}
root.mkdir(parents=True, exist_ok=True)
(root / "config_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
for name in ["commands.txt", "job_ids.txt", "summary.csv", "summary.md"]:
    path = root / name
    if not path.exists():
        path.write_text("", encoding="utf-8")
PY
}

run_case() {
  local idx="${1:?missing case index}"
  python - "${RESULT_ROOT}/config_manifest.json" "${idx}" > "${RESULT_ROOT}/case_${idx}.env" <<'PY'
import json
import shlex
import sys
from pathlib import Path
manifest = json.loads(Path(sys.argv[1]).read_text())
case = manifest["cases"][int(sys.argv[2])]
for key, value in case.items():
    print(f"{key.upper()}={shlex.quote(str(value))}")
PY
  # shellcheck disable=SC1090
  source "${RESULT_ROOT}/case_${idx}.env"
  local log_path="${RESULT_ROOT}/logs/${RUN_NAME}.log"
  {
    echo "[$(date '+%F %T')] run_name=${RUN_NAME} case=${idx}"
    echo "RESULT_ROOT=${RESULT_ROOT}"
    echo "seed=${SEED} h=${H_RAW} lr=${LR} commit=${RESIDUAL_COMMIT_MODE} max_code_step=${RESIDUAL_MAX_CODE_STEP} clip=${ZO_UPDATE_NORM_CLIP}"
    (
      cd "${ROOT_DIR}/medium_models"
      TASK=SST-5 \
      K=16 \
      SEED="${SEED}" \
      DATA_SEED="${DATA_SEED}" \
      DATASET_MODE=full \
      FULL_DEV_RATIO=0.1 \
      BS="${BS}" \
      LR="${LR}" \
      EPS="${H_RAW}" \
      WD=0 \
      STEP="${STEPS}" \
      EVAL_STEP="${EVAL_EVERY}" \
      MODEL=roberta-large \
      USE_H=False \
      USE_C=False \
      DATALOADER_SHUFFLE=False \
      EFFICIENT_ZERO_ORDER=True \
      EXTRA_TAG=next-residual-local \
      TOKENIZERS_PARALLELISM=false \
      CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
      bash ./mezo.sh \
        --result_root "${RESULT_ROOT}" \
        --job_name "${RUN_NAME}" \
        --dataset_mode full \
        --precision_mode "${PRECISION_MODE}" \
        --zo_quantization "${ZO_QUANTIZATION}" \
        --zo_update_backend "${UPDATE_BACKEND}" \
        --direction_type "${DIRECTION_TYPE}" \
        --sparse_rate "${SPARSE_RATE}" \
        --sparse_mode "${SPARSE_MODE}" \
        --sparse_rescale "${SPARSE_RESCALE}" \
        --zo_h "${H_RAW}" \
        --residual_commit_mode "${RESIDUAL_COMMIT_MODE}" \
        --residual_dtype "${RESIDUAL_DTYPE}" \
        --residual_max_code_step "${RESIDUAL_MAX_CODE_STEP}" \
        --zo_update_norm_clip "${ZO_UPDATE_NORM_CLIP}" \
        --int8_freeze_scale True \
        --log_update_stats_every 1 \
        --save_update_stats_jsonl update_stats.jsonl \
        --random_prediction_guard_enabled False \
        --save_strategy no \
        --save_at_last False \
        --no_predict
    )
    echo "[$(date '+%F %T')] completed ${RUN_NAME}"
  } > >(tee -a "${log_path}") 2>&1
  python scripts/summarize_next_experiments.py "${RESULT_ROOT}" || true
  python - "${RESULT_ROOT}" "${RUN_NAME}" <<'PY'
import json
import math
import sys
from pathlib import Path
root = Path(sys.argv[1])
run_name = sys.argv[2]
path = root / run_name / "seed0" / "update_stats.jsonl"
rows = []
if path.exists():
    for line in path.read_text().splitlines():
        try:
            rows.append(json.loads(line))
        except Exception:
            pass
if not rows:
    raise SystemExit(0)
bad = []
last = rows[-1]
loss = last.get("train_loss")
if isinstance(loss, (int, float)) and (not math.isfinite(loss) or loss > 20):
    bad.append(f"bad train_loss={loss}")
scale_drift = last.get("scale_drift_max")
if isinstance(scale_drift, (int, float)) and scale_drift != 0:
    bad.append(f"scale_drift_max={scale_drift}")
if last.get("residual_commit_mode") == "round" and last.get("residual_max_code_step") == 0:
    viol = last.get("unsaturated_residual_bound_violation_frac")
    if isinstance(viol, (int, float)) and viol > 0:
        bad.append(f"residual_bound_violation_frac={viol}")
if bad:
    raise SystemExit("; ".join(bad))
PY
}

write_manifest
{
  echo "# Next residual local commands"
  echo "RESULT_ROOT='${RESULT_ROOT}' CUDA_VISIBLE_DEVICES='${CUDA_VISIBLE_DEVICES}' bash scripts/run_next_residual_local.sh"
  echo "python scripts/summarize_next_experiments.py '${RESULT_ROOT}'"
} > "${RESULT_ROOT}/commands.txt"

{
  echo "created_at=$(date -Is)"
  echo "result_root=${RESULT_ROOT}"
  echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES}"
  echo "auto_promote=${AUTO_PROMOTE}"
  nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv,noheader || true
} | tee "${RESULT_ROOT}/local_gpu_status.txt"

if [[ "${1:-}" == "--run-case" ]]; then
  run_case "${2:?missing case index}"
  exit 0
fi

num_cases="$(python - "${RESULT_ROOT}/config_manifest.json" <<'PY'
import json, sys
print(len(json.load(open(sys.argv[1]))["cases"]))
PY
)"
echo "local_sequential_stage_a cases=${num_cases}" | tee "${RESULT_ROOT}/job_ids.txt"
for ((i = 0; i < num_cases; i++)); do
  run_case "${i}"
done

python scripts/summarize_next_experiments.py "${RESULT_ROOT}"

if [[ "${AUTO_PROMOTE}" != "1" ]]; then
  echo "AUTO_PROMOTE=${AUTO_PROMOTE}; residual promotion not launched." | tee -a "${RESULT_ROOT}/job_ids.txt"
  exit 0
fi

python - "${RESULT_ROOT}/summary_residual.csv" <<'PY' > "${RESULT_ROOT}/promoted_residual_cases.txt"
import csv
import math
import sys
rows = []
with open(sys.argv[1], newline="") as f:
    for row in csv.DictReader(f):
        if str(row.get("completed")).lower() != "true":
            continue
        if str(row.get("nan_occurred")).lower() == "true":
            continue
        try:
            acc = float(row.get("best_eval_acc") or "nan")
            loss = float(row.get("last_eval_loss") or "nan")
        except ValueError:
            continue
        if math.isfinite(acc) and math.isfinite(loss):
            rows.append((acc, -loss, row["run_name"]))
for _, _, name in sorted(rows, reverse=True)[:2]:
    print(name)
PY
echo "AUTO_PROMOTE=1 selected residual cases:" | tee -a "${RESULT_ROOT}/job_ids.txt"
cat "${RESULT_ROOT}/promoted_residual_cases.txt" | tee -a "${RESULT_ROOT}/job_ids.txt"
echo "Promotion execution is intentionally manual: rerun selected configs with STEPS=2000 after inspecting summary." | tee -a "${RESULT_ROOT}/job_ids.txt"
