#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

EXPERIMENT_ID="${EXPERIMENT_ID:-latest_main_roberta_sst5_fp32_fp16_hsweep_seed16_bs64_ckpt1k}"
EXP_ROOT="${EXP_ROOT:-${ROOT_DIR}/experiments/main_latest/mezo/roberta-large/sst5/fp32_fp16_h_sweep_11h_seed16_bs64_ckpt1k_20260517}"
LANE_ID="${LANE_ID:-${SLURM_ARRAY_TASK_ID:-0}}"
LANE_MANIFEST="${LANE_MANIFEST:-${EXP_ROOT}/lane_manifests/lane${LANE_ID}.csv}"
CONDA_ENV="${CONDA_ENV:-ciao}"
DATA_ROOT="${DATA_ROOT:-data/k-shot-1k-test}"
PROBE_DIRECTIONS="${PROBE_DIRECTIONS:-50}"

if [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
  # shellcheck disable=SC1091
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV}"
fi

mkdir -p "${EXP_ROOT}/logs" "${EXP_ROOT}/summaries"

latest_step_checkpoint() {
  local run_dir="$1"
  python - "$run_dir" <<'PY'
import re
import sys
from pathlib import Path
root = Path(sys.argv[1]) / "checkpoints"
best = None
if root.exists():
    for path in root.glob("step_*"):
        m = re.match(r"step_(\d+)$", path.name)
        if m and path.is_dir():
            item = (int(m.group(1)), path)
            if best is None or item[0] > best[0]:
                best = item
print("" if best is None else str(best[1]))
PY
}

postprocess_run() {
  local run_dir="$1"
  local manifest_json="$2"
  local exit_code="$3"
  python - "$run_dir" "$manifest_json" "$exit_code" <<'PY'
import csv
import json
import math
import os
import shutil
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])
manifest = json.loads(Path(sys.argv[2]).read_text())
exit_code = int(sys.argv[3])
metrics_path = run_dir / "metrics_logs" / "metrics_adaptiveH-0_cscale-0.csv"
flat_metrics_path = run_dir / "metrics.csv"
eval_jsonl_path = run_dir / "eval_metrics.jsonl"
if metrics_path.exists():
    shutil.copyfile(metrics_path, flat_metrics_path)

def safe_float(value):
    try:
        if value in (None, ""):
            return None
        x = float(value)
        return x if math.isfinite(x) else None
    except Exception:
        return None

rows = []
if metrics_path.exists():
    with metrics_path.open(newline="") as f:
        rows = list(csv.DictReader(f))
eval_rows = [r for r in rows if str(r.get("eval_ran", "")).upper() not in {"", "NO", "NONE"} and safe_float(r.get("eval_loss")) is not None]
with eval_jsonl_path.open("w", encoding="utf-8") as f:
    for row in eval_rows:
        f.write(json.dumps(row) + "\n")

best_acc_row = None
best_loss_row = None
for row in eval_rows:
    acc = safe_float(row.get("eval_acc"))
    loss = safe_float(row.get("eval_loss"))
    if acc is not None and (best_acc_row is None or acc > safe_float(best_acc_row.get("eval_acc"))):
        best_acc_row = row
    if loss is not None and (best_loss_row is None or loss < safe_float(best_loss_row.get("eval_loss"))):
        best_loss_row = row
last_eval_row = eval_rows[-1] if eval_rows else None
last_train_row = rows[-1] if rows else None

ckpt_root = run_dir / "checkpoints"
step_ckpts = sorted(ckpt_root.glob("step_*")) if ckpt_root.exists() else []
final_ckpt = ckpt_root / "final"
best_acc_ckpt = ckpt_root / "best_acc"
best_loss_ckpt = ckpt_root / "best_loss"
checkpoint_count = sum(1 for p in [*step_ckpts, final_ckpt, best_acc_ckpt, best_loss_ckpt] if p.exists())

probe = {}
probe_path = run_dir / "checkpoint_probe_stats.jsonl"
if probe_path.exists():
    with probe_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except Exception:
                continue
            probe.update(item)

def row_step(row):
    val = safe_float((row or {}).get("global_step"))
    return None if val is None else int(val)

nan_occurred = False
for row in rows:
    for key in ("train_loss", "eval_loss", "train_acc", "eval_acc"):
        raw = row.get(key)
        if raw not in (None, ""):
            try:
                x = float(raw)
                if math.isnan(x) or math.isinf(x):
                    nan_occurred = True
            except Exception:
                pass

status = "completed" if exit_code == 0 and final_ckpt.exists() else ("failed" if exit_code != 0 else "incomplete")
summary = {
    "run_name": manifest["run_name"],
    "precision_mode": manifest["precision_mode"],
    "h": float(manifest["h"]),
    "seed": int(manifest["seed"]),
    "data_seed": int(manifest["data_seed"]),
    "dataset_mode": manifest["dataset_mode"],
    "dataloader_shuffle": str(manifest["dataloader_shuffle"]),
    "batch_size": int(manifest["batch_size"]),
    "max_steps": int(manifest["max_steps"]),
    "steps_completed": row_step(last_train_row) or 0,
    "best_eval_acc": safe_float((best_acc_row or {}).get("eval_acc")),
    "best_eval_step": row_step(best_acc_row),
    "last_eval_acc": safe_float((last_eval_row or {}).get("eval_acc")),
    "last_eval_step": row_step(last_eval_row),
    "best_eval_loss": safe_float((best_loss_row or {}).get("eval_loss")),
    "best_eval_loss_step": row_step(best_loss_row),
    "last_eval_loss": safe_float((last_eval_row or {}).get("eval_loss")),
    "last_eval_loss_step": row_step(last_eval_row),
    "final_train_loss": safe_float((last_train_row or {}).get("train_loss")),
    "final_train_acc": safe_float((last_train_row or {}).get("train_acc")),
    "nan_occurred": bool(nan_occurred),
    "checkpoint_count": int(checkpoint_count),
    "final_checkpoint_path": str(final_ckpt) if final_ckpt.exists() else None,
    "best_acc_checkpoint_path": str(best_acc_ckpt) if best_acc_ckpt.exists() else None,
    "best_loss_checkpoint_path": str(best_loss_ckpt) if best_loss_ckpt.exists() else None,
    "probe_corr_fd_true": probe.get("corr_fd_true"),
    "probe_nMSE_fd_true": probe.get("nMSE_fd_true"),
    "probe_alignment": probe.get("probe_alignment") or probe.get("probe_alignment_mean"),
    "probe_norm_ratio": probe.get("probe_norm_ratio") or probe.get("probe_norm_ratio_mean"),
    "status": status,
    "exit_code": exit_code,
}
raw_summary = run_dir / "run_summary.json"
if raw_summary.exists() and not (run_dir / "run_summary_raw.json").exists():
    shutil.copyfile(raw_summary, run_dir / "run_summary_raw.json")
raw_summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
print(json.dumps(summary, sort_keys=True))
PY
}

if [[ ! -f "${LANE_MANIFEST}" ]]; then
  echo "Missing lane manifest: ${LANE_MANIFEST}" >&2
  exit 2
fi

echo "[lane] experiment_id=${EXPERIMENT_ID}"
echo "[lane] exp_root=${EXP_ROOT}"
echo "[lane] lane_id=${LANE_ID}"
echo "[lane] lane_manifest=${LANE_MANIFEST}"

lane_summary="${EXP_ROOT}/summaries/lane${LANE_ID}_summary.jsonl"
tail -n +2 "${LANE_MANIFEST}" | while IFS=, read -r lane_id gpu_type precision h h_label run_name result_root max_steps eval_steps checkpoint_steps seed data_seed batch_size lr; do
  [[ -z "${run_name}" ]] && continue
  run_dir="${result_root}/${run_name}/seed${seed}"
  mkdir -p "${run_dir}"
  manifest_json="${run_dir}/run_manifest_row.json"
  python - "$manifest_json" "$lane_id" "$gpu_type" "$precision" "$h" "$h_label" "$run_name" "$result_root" "$max_steps" "$eval_steps" "$checkpoint_steps" "$seed" "$data_seed" "$batch_size" "$lr" <<'PY'
import json
import sys
path = sys.argv[1]
keys = ["lane_id","gpu_type","precision_mode","h","h_label","run_name","result_root","max_steps","eval_steps","checkpoint_steps","seed","data_seed","batch_size","lr"]
values = sys.argv[2:]
row = dict(zip(keys, values))
for key in ["lane_id", "max_steps", "eval_steps", "checkpoint_steps", "seed", "data_seed", "batch_size"]:
    row[key] = int(float(row[key]))
for key in ["h", "lr"]:
    row[key] = float(row[key])
row.update({
    "dataset": "SST-5",
    "dataset_mode": "full",
    "dataloader_shuffle": True,
    "model": "roberta-large",
    "method": "mezo",
    "direction_type": "dense",
})
open(path, "w", encoding="utf-8").write(json.dumps(row, indent=2) + "\n")
open(str(path).replace("run_manifest_row.json", "run_config.json"), "w", encoding="utf-8").write(json.dumps(row, indent=2) + "\n")
PY
  if [[ -f "${run_dir}/run_summary.json" ]] && [[ -d "${run_dir}/checkpoints/final" ]]; then
    echo "[lane${LANE_ID}] skip completed ${run_name}"
    continue
  fi

  resume_ckpt="$(latest_step_checkpoint "${run_dir}")"
  model_name="roberta-large"
  if [[ -n "${resume_ckpt}" && ! -d "${run_dir}/checkpoints/final" ]]; then
    model_name="${resume_ckpt}"
    echo "[lane${LANE_ID}] resuming ${run_name} from ${resume_ckpt}"
  else
    echo "[lane${LANE_ID}] starting ${run_name}"
  fi

  zo_quantization="${precision}"
  train_log="${run_dir}/train.log"
  stderr_log="${run_dir}/stderr.log"
  set +e
  (
    cd "${ROOT_DIR}/medium_models"
    TASK=SST-5 \
    K=16 \
    SEED="${seed}" \
    DATA_SEED="${data_seed}" \
    DATASET_MODE=full \
    FULL_DEV_RATIO=0.1 \
    DATA_ROOT="${DATA_ROOT}" \
    BS="${batch_size}" \
    LR="${lr}" \
    EPS="${h}" \
    WD=0 \
    OPT=sgd \
    STEP="${max_steps}" \
    EVAL_STEP="${eval_steps}" \
    MODEL="${model_name}" \
    USE_H=False \
    USE_C=False \
    DATALOADER_SHUFFLE=True \
    KEEP_CHECKPOINTS=True \
    EFFICIENT_ZERO_ORDER=True \
    ZERO_ORDER_USE_TRAINER_OPTIM=False \
    NUM_GPU=1 \
    EXTRA_TAG="${EXPERIMENT_ID}" \
    TOKENIZERS_PARALLELISM=false \
    bash ./mezo.sh \
      --result_root "${result_root}" \
      --job_name "${run_name}" \
      --dataset_mode full \
      --data_seed "${data_seed}" \
      --full_dev_ratio 0.1 \
      --precision_mode "${precision}" \
      --zo_quantization "${zo_quantization}" \
      --direction_type dense \
      --zo_h "${h}" \
      --gradient_accumulation_steps 1 \
      --save_strategy no \
      --save_steps "${checkpoint_steps}" \
      --main_save_checkpoints True \
      --main_checkpoint_steps "${checkpoint_steps}" \
      --main_save_final_checkpoint True \
      --main_save_best_acc_checkpoint True \
      --main_save_best_loss_checkpoint True \
      --checkpoint_probe_steps 0 \
      --checkpoint_probe_num_directions "${PROBE_DIRECTIONS}" \
      --checkpoint_probe_num_batches 1 \
      --checkpoint_probe_compute_true_grad True \
      --save_checkpoint_probe_stats_jsonl checkpoint_probe_stats.jsonl \
      --random_prediction_guard_enabled False \
      --zo_probe_health_guard_enabled False \
      --save_at_last True \
      --no_predict
  ) > >(tee -a "${train_log}") 2> >(tee -a "${stderr_log}" >&2)
  exit_code=$?
  set -e
  summary_json="$(postprocess_run "${run_dir}" "${manifest_json}" "${exit_code}")"
  echo "${summary_json}" >> "${lane_summary}"
  if [[ "${exit_code}" -ne 0 ]]; then
    echo "[lane${LANE_ID}] ${run_name} failed with exit_code=${exit_code}; continuing to next run." >&2
  fi
done

python scripts/summarize_latest_main_hsweep.py "${EXP_ROOT}" || true
