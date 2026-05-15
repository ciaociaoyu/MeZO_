#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

timestamp() {
  date +"%Y%m%d_%H%M%S"
}

RESULT_ROOT="${RESULT_ROOT:-${ROOT_DIR}/runs/future_probe_window_training_$(timestamp)}"
MODE="${MODE:-dense_main}"
TASK_INDEX="${TASK_INDEX:-}"
CONDA_ENV="${CONDA_ENV:-ciao}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

mkdir -p "${RESULT_ROOT}/logs"

emit_cases() {
  MODE="${MODE}" \
  DENSE_SEEDS="${DENSE_SEEDS:-0}" \
  SPARSE_SCREEN_SEEDS="${SPARSE_SCREEN_SEEDS:-0}" \
  RESIDUAL_SEEDS="${RESIDUAL_SEEDS:-0}" \
  DENSE_STEPS="${DENSE_STEPS:-2000}" \
  DENSE_EVAL_STEP="${DENSE_EVAL_STEP:-200}" \
  SPARSE_SCREEN_STEPS="${SPARSE_SCREEN_STEPS:-500}" \
  SPARSE_EVAL_STEP="${SPARSE_EVAL_STEP:-100}" \
  RESIDUAL_STEPS="${RESIDUAL_STEPS:-500}" \
  RESIDUAL_EVAL_STEP="${RESIDUAL_EVAL_STEP:-100}" \
  DENSE_LR="${DENSE_LR:-1e-5}" \
  SPARSE_MODE="${SPARSE_MODE:-exact_random}" \
  PROMOTED_SPARSE_SPECS="${PROMOTED_SPARSE_SPECS:-}" \
  python - <<'PY'
import math
import os

mode = os.environ["MODE"]

def words(name, default):
    return str(os.environ.get(name, default)).split()

def add(rows, *, run_name, family, seed, backend, direction, h_raw, p, h_active, sparse_mode,
        sparse_rescale, lr, steps, eval_step, commit_mode="", residual_dtype="", max_code_step="",
        clip="", checkpoint_steps=""):
    rows.append("|".join([
        run_name, family, str(seed), "int8", backend, direction, f"{h_raw:.17g}",
        f"{p:.17g}", f"{h_active:.17g}", sparse_mode, sparse_rescale, str(lr),
        str(steps), str(eval_step), commit_mode, residual_dtype, str(max_code_step),
        str(clip), checkpoint_steps,
    ]))

rows = []

if mode in {"dense_main", "all_screen", "all"}:
    for seed in words("DENSE_SEEDS", "0"):
        for h in [3e-4, 1e-3, 2e-3, 3e-3, 5e-3, 1e-2]:
            tag = f"{h:.0e}".replace("-", "m")
            checkpoint = "0,300,1000,2000" if h in {3e-4, 3e-3} else ""
            add(
                rows,
                run_name=f"dense_int8_fp16master_h{tag}_seed{seed}",
                family="dense",
                seed=seed,
                backend="fp16_master",
                direction="dense",
                h_raw=h,
                p=1.0,
                h_active=h,
                sparse_mode="none",
                sparse_rescale="none",
                lr=os.environ.get("DENSE_LR", "1e-5"),
                steps=os.environ.get("DENSE_STEPS", "2000"),
                eval_step=os.environ.get("DENSE_EVAL_STEP", "200"),
                checkpoint_steps=checkpoint,
            )

if mode in {"sparse_screen", "all_screen", "all"}:
    sparse_mode = os.environ.get("SPARSE_MODE", "exact_random")
    for seed in words("SPARSE_SCREEN_SEEDS", "0"):
        for p in [0.003, 0.01, 0.03]:
            for h_active in [0.003, 0.006, 0.012]:
                h_raw = h_active * math.sqrt(p)
                for lr in ["3e-6", "1e-5", "3e-5"]:
                    ptag = str(p).replace(".", "p")
                    hatag = str(h_active).replace(".", "p")
                    lrtag = str(lr).replace("-", "m")
                    checkpoint = "0,300" if abs(p - 0.01) < 1e-12 and h_active in {0.006, 0.012} and lr == "1e-5" else ""
                    add(
                        rows,
                        run_name=f"sparse_int8_p{ptag}_ha{hatag}_lr{lrtag}_seed{seed}",
                        family="sparse_screen",
                        seed=seed,
                        backend="fp16_master",
                        direction="sparse",
                        h_raw=h_raw,
                        p=p,
                        h_active=h_active,
                        sparse_mode=sparse_mode,
                        sparse_rescale="inv_sqrt_p",
                        lr=lr,
                        steps=os.environ.get("SPARSE_SCREEN_STEPS", "500"),
                        eval_step=os.environ.get("SPARSE_EVAL_STEP", "100"),
                        checkpoint_steps=checkpoint,
                    )

if mode in {"sparse_promote", "all"}:
    specs = os.environ.get("PROMOTED_SPARSE_SPECS", "").strip()
    if not specs:
        raise SystemExit("PROMOTED_SPARSE_SPECS is required for MODE=sparse_promote; format p:h_active:lr:seed[,p:h_active:lr:seed...]")
    sparse_mode = os.environ.get("SPARSE_MODE", "exact_random")
    for spec in specs.replace(",", " ").split():
        p_s, ha_s, lr, seed = spec.split(":")
        p = float(p_s)
        h_active = float(ha_s)
        h_raw = h_active * math.sqrt(p)
        ptag = p_s.replace(".", "p")
        hatag = ha_s.replace(".", "p")
        lrtag = lr.replace("-", "m")
        add(
            rows,
            run_name=f"sparse_promoted_int8_p{ptag}_ha{hatag}_lr{lrtag}_seed{seed}",
            family="sparse_promote",
            seed=seed,
            backend="fp16_master",
            direction="sparse",
            h_raw=h_raw,
            p=p,
            h_active=h_active,
            sparse_mode=sparse_mode,
            sparse_rescale="inv_sqrt_p",
            lr=lr,
            steps=os.environ.get("SPARSE_PROMOTE_STEPS", "2000"),
            eval_step=os.environ.get("SPARSE_PROMOTE_EVAL_STEP", "200"),
            checkpoint_steps="0,300,1000,2000",
        )

if mode in {"residual", "all_residual", "all"}:
    for seed in words("RESIDUAL_SEEDS", "0"):
        for name, commit, lr, max_step, clip in [
            ("round_lr3em5_step0_clip0", "round", "3e-5", "0", "0"),
            ("round_lr1em4_step1_clip5", "round", "1e-4", "1", "5"),
            ("stoch_lr3em4_step1_clip10", "stochastic", "3e-4", "1", "10"),
        ]:
            add(
                rows,
                run_name=f"residual_grid_{name}_seed{seed}",
                family="residual",
                seed=seed,
                backend="residual_grid",
                direction="dense",
                h_raw=3e-3,
                p=1.0,
                h_active=3e-3,
                sparse_mode="none",
                sparse_rescale="none",
                lr=lr,
                steps=os.environ.get("RESIDUAL_STEPS", "500"),
                eval_step=os.environ.get("RESIDUAL_EVAL_STEP", "100"),
                commit_mode=commit,
                residual_dtype="fp32",
                max_code_step=max_step,
                clip=clip,
            )

for row in rows:
    print(row)
PY
}

if [[ "${1:-}" == "--count" ]]; then
  emit_cases | wc -l
  exit 0
fi

if [[ "${1:-}" == "--case" ]]; then
  TASK_INDEX="${2:?missing case index}"
fi

mapfile -t CASES < <(emit_cases)
if (( ${#CASES[@]} == 0 )); then
  echo "No cases generated for MODE=${MODE}" >&2
  exit 1
fi

run_case() {
  local line="$1"
  IFS='|' read -r run_name family seed precision backend direction h_raw p h_active sparse_mode sparse_rescale lr steps eval_step commit_mode residual_dtype max_code_step clip checkpoint_steps <<< "${line}"

  local log_path="${RESULT_ROOT}/logs/${run_name}.log"
  {
    echo "[$(date '+%F %T')] run_name=${run_name}"
    echo "RESULT_ROOT=${RESULT_ROOT}"
    echo "family=${family} seed=${seed} h_raw=${h_raw} h_active=${h_active} p=${p} lr=${lr} steps=${steps} eval_step=${eval_step}"
    echo "backend=${backend} direction=${direction} sparse_mode=${sparse_mode} sparse_rescale=${sparse_rescale}"
    echo "checkpoint_steps=${checkpoint_steps:-none}"

    local extra_args=(
      --result_root "${RESULT_ROOT}"
      --job_name "${run_name}"
      --dataset_mode full
      --precision_mode "${precision}"
      --zo_quantization int8
      --zo_update_backend "${backend}"
      --direction_type "${direction}"
      --sparse_rate "${p}"
      --sparse_mode "${sparse_mode}"
      --sparse_rescale "${sparse_rescale}"
      --zo_h "${h_raw}"
      --log_update_stats_every 1
      --save_update_stats_jsonl update_stats.jsonl
      --random_prediction_guard_enabled False
      --save_strategy no
      --save_at_last False
      --no_predict
    )

    if [[ "${backend}" == "residual_grid" ]]; then
      extra_args+=(
        --residual_commit_mode "${commit_mode}"
        --residual_dtype "${residual_dtype}"
        --residual_max_code_step "${max_code_step}"
        --zo_update_norm_clip "${clip}"
        --int8_freeze_scale True
      )
    fi

    if [[ -n "${checkpoint_steps}" ]]; then
      extra_args+=(
        --checkpoint_probe_steps "${checkpoint_steps}"
        --checkpoint_probe_num_directions "${CHECKPOINT_PROBE_NUM_DIRECTIONS:-16}"
        --checkpoint_probe_num_batches "${CHECKPOINT_PROBE_NUM_BATCHES:-1}"
        --checkpoint_probe_compute_true_grad "${CHECKPOINT_PROBE_COMPUTE_TRUE_GRAD:-True}"
        --save_checkpoint_probe_stats_jsonl checkpoint_probe_stats.jsonl
      )
    fi

    (
      cd "${ROOT_DIR}/medium_models"
      TASK=SST-5 \
      K=16 \
      SEED="${seed}" \
      DATA_SEED="${DATA_SEED:-16}" \
      DATASET_MODE=full \
      FULL_DEV_RATIO=0.1 \
      BS="${BS:-64}" \
      LR="${lr}" \
      EPS="${h_raw}" \
      WD="${WD:-0}" \
      STEP="${steps}" \
      EVAL_STEP="${eval_step}" \
      MODEL=roberta-large \
      USE_H=False \
      USE_C=False \
      DATALOADER_SHUFFLE=False \
      EFFICIENT_ZERO_ORDER=True \
      EXTRA_TAG=future-probe-window \
      CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
      TOKENIZERS_PARALLELISM=false \
      bash ./mezo.sh "${extra_args[@]}"
    )

    echo "[$(date '+%F %T')] completed ${run_name}"
  } > >(tee -a "${log_path}") 2>&1
}

if [[ -n "${TASK_INDEX}" ]]; then
  if ! [[ "${TASK_INDEX}" =~ ^[0-9]+$ ]]; then
    echo "Invalid TASK_INDEX=${TASK_INDEX}" >&2
    exit 1
  fi
  if (( TASK_INDEX < 0 || TASK_INDEX >= ${#CASES[@]} )); then
    echo "TASK_INDEX=${TASK_INDEX} out of range 0..$((${#CASES[@]} - 1))" >&2
    exit 0
  fi
  run_case "${CASES[${TASK_INDEX}]}"
  exit 0
fi

echo "Sequential mode: running ${#CASES[@]} cases for MODE=${MODE}"
for case_line in "${CASES[@]}"; do
  run_case "${case_line}"
done
