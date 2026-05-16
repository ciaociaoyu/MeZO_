#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

timestamp() {
  date +"%Y%m%d_%H%M%S"
}

MODE="${MODE:-all}"
RESULT_ROOT="${RESULT_ROOT:-${ROOT_DIR}/runs/next_round_window_sparse_$(timestamp)}"
MAX_CONCURRENT_JOBS="${MAX_CONCURRENT_JOBS:-2}"
TIME_LIMIT="${TIME_LIMIT:-24:00:00}"
CPUS_PER_TASK="${CPUS_PER_TASK:-8}"
MEMORY="${MEMORY:-96G}"
CONDA_ENV="${CONDA_ENV:-ciao}"
SPARSE_MODE="${SPARSE_MODE:-exact_random}"
DATA_SEED="${DATA_SEED:-16}"
BS="${BS:-64}"
EVAL_EVERY_DENSE="${EVAL_EVERY_DENSE:-200}"
EVAL_EVERY_SPARSE="${EVAL_EVERY_SPARSE:-100}"
CASE_OFFSET="${CASE_OFFSET:-0}"
CASE_LIMIT="${CASE_LIMIT:-0}"

mkdir -p "${RESULT_ROOT}/slurm" "${RESULT_ROOT}/logs"

if (( MAX_CONCURRENT_JOBS < 1 )); then
  MAX_CONCURRENT_JOBS=1
fi
if (( MAX_CONCURRENT_JOBS > 3 )); then
  echo "MAX_CONCURRENT_JOBS=${MAX_CONCURRENT_JOBS} requested; capping at 3. Edit the script manually to exceed this." >&2
  MAX_CONCURRENT_JOBS=3
fi

detect_scheduler() {
  if command -v sbatch >/dev/null 2>&1 && find medium_models scripts -maxdepth 3 -type f | grep -qE '(slurm|sbatch|#SBATCH)'; then
    echo "slurm"
  elif command -v sbatch >/dev/null 2>&1; then
    echo "slurm"
  else
    echo "local"
  fi
}

detect_gpu_type() {
  if ! command -v sinfo >/dev/null 2>&1; then
    echo "local"
    return
  fi
  local info
  info="$(sinfo -h -o '%P|%G|%t|%D|%C' 2>/dev/null || true)"
  if echo "${info}" | awk -F'|' 'tolower($2) ~ /h100/ && $3 ~ /idle/ {found=1} END{exit found?0:1}'; then
    echo "h100"
  elif echo "${info}" | awk -F'|' 'tolower($2) ~ /a100/ && $3 ~ /idle/ {found=1} END{exit found?0:1}'; then
    echo "a100"
  else
    echo "l4"
  fi
}

gpu_count_hint() {
  local gpu="$1"
  if ! command -v sinfo >/dev/null 2>&1; then
    echo 0
    return
  fi
  sinfo -h -o '%G|%t|%D' 2>/dev/null \
    | awk -F'|' -v gpu="${gpu^^}" 'tolower($1) ~ tolower(gpu) && $2 ~ /idle/ {n += $3} END{print n+0}'
}

generate_manifest() {
  python - "${RESULT_ROOT}" "${MODE}" "${SPARSE_MODE}" "${EVAL_EVERY_DENSE}" "${EVAL_EVERY_SPARSE}" "${CASE_OFFSET}" "${CASE_LIMIT}" <<'PY'
import json
import math
import sys
from pathlib import Path

root = Path(sys.argv[1])
mode = sys.argv[2]
sparse_mode = sys.argv[3]
eval_dense = int(float(sys.argv[4]))
eval_sparse = int(float(sys.argv[5]))
case_offset = int(float(sys.argv[6]))
case_limit = int(float(sys.argv[7]))

cases = []


def tag_decimal(x):
    return f"{x:g}".replace(".", "p")


def tag_h(x):
    mapping = {
        2e-3: "2e-3",
        3e-3: "3e-3",
    }
    for key, value in mapping.items():
        if abs(float(x) - key) < 1e-15:
            return value
    return f"{x:.0e}"


def add(**kwargs):
    kwargs.setdefault("precision_mode", "int8")
    kwargs.setdefault("zo_quantization", "int8")
    kwargs.setdefault("update_backend", "fp16_master")
    kwargs.setdefault("sparse_rescale", "none")
    kwargs.setdefault("sparse_mode", "none")
    kwargs.setdefault("sparse_rate", 1.0)
    kwargs.setdefault("checkpoint_probe_steps", "")
    cases.append(kwargs)


if mode in {"all", "dense_missing"}:
    add(
        family="dense",
        run_name="dense_int8_fp16master_h3e-3_seed2_step2000",
        seed=2,
        direction_type="dense",
        h_raw=3e-3,
        h_active=3e-3,
        lr="1e-5",
        steps=2000,
        eval_every=eval_dense,
    )

if mode in {"all", "dense_promote"}:
    for h in [2e-3, 3e-3]:
        for seed in [0, 1, 2]:
            checkpoint = "0,500,2000,5000" if seed == 0 else ""
            add(
                family="dense",
                run_name=f"dense_int8_fp16master_h{tag_h(h)}_seed{seed}_step5000",
                seed=seed,
                direction_type="dense",
                h_raw=h,
                h_active=h,
                lr="1e-5",
                steps=5000,
                eval_every=eval_dense,
                checkpoint_probe_steps=checkpoint,
            )

if mode in {"all", "sparse_screen"}:
    for p in [0.003, 0.01, 0.03]:
        for h_active in [0.006, 0.012]:
            h_raw = h_active * math.sqrt(p)
            for lr in ["3e-6", "1e-5", "3e-5"]:
                add(
                    family="sparse_screen",
                    run_name=f"sparse_int8_fp16master_p{tag_decimal(p)}_hactive{tag_decimal(h_active)}_lr{lr}_seed0_step500",
                    seed=0,
                    direction_type="sparse",
                    sparse_rate=p,
                    sparse_mode=sparse_mode,
                    sparse_rescale="inv_sqrt_p",
                    h_raw=h_raw,
                    h_active=h_active,
                    lr=lr,
                    steps=500,
                    eval_every=eval_sparse,
                )

if mode == "sparse_promote":
    specs = [s for s in str(__import__("os").environ.get("PROMOTED_SPARSE_SPECS", "")).replace(",", " ").split() if s]
    if not specs:
        raise SystemExit("PROMOTED_SPARSE_SPECS is required for MODE=sparse_promote; format p:h_active:lr:seed[,..]")
    for spec in specs:
        p_s, ha_s, lr, seed_s = spec.split(":")
        p = float(p_s)
        h_active = float(ha_s)
        h_raw = h_active * math.sqrt(p)
        add(
            family="sparse_promote",
            run_name=f"sparse_promoted_int8_p{tag_decimal(p)}_hactive{tag_decimal(h_active)}_lr{lr}_seed{seed_s}_step2000",
            seed=int(seed_s),
            direction_type="sparse",
            sparse_rate=p,
            sparse_mode=sparse_mode,
            sparse_rescale="inv_sqrt_p",
            h_raw=h_raw,
            h_active=h_active,
            lr=lr,
            steps=2000,
            eval_every=200,
            checkpoint_probe_steps="0,500,2000",
        )

all_cases = cases
if case_offset < 0:
    raise SystemExit("CASE_OFFSET must be >= 0")
if case_limit < 0:
    raise SystemExit("CASE_LIMIT must be >= 0")
if case_offset or case_limit:
    end = None if case_limit == 0 else case_offset + case_limit
    cases = all_cases[case_offset:end]
for i, case in enumerate(cases):
    case["manifest_case_index"] = i
    case["source_case_index"] = case_offset + i

manifest = {
    "manifest_schema_version": 1,
    "mode": mode,
    "case_offset": case_offset,
    "case_limit": case_limit,
    "total_cases_before_slice": len(all_cases),
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

print_table() {
  python - "${RESULT_ROOT}/config_manifest.json" <<'PY'
import json
import sys
from pathlib import Path

manifest = json.loads(Path(sys.argv[1]).read_text())
print("idx source family          run_name                                                            steps  h_raw        h_active     p       lr      seed")
for i, case in enumerate(manifest["cases"]):
    print(f"{i:3d} {case['source_case_index']:6d} {case['family']:<15} {case['run_name']:<68} {case['steps']:<6} {case['h_raw']:<12.6g} {case['h_active']:<11.6g} {case['sparse_rate']:<7g} {case['lr']:<7} {case['seed']}")
PY
}

if [[ "${1:-}" == "--run-case" ]]; then
  bash scripts/submit_next_window_sparse.sh --run-case "${2:?missing case index}"
  exit 0
fi

scheduler="$(detect_scheduler)"
gpu_type="${GPU_TYPE:-$(detect_gpu_type)}"
partition="${SLURM_PARTITION:-gpu_p}"
case "${gpu_type}" in
  h100) gres="gpu:H100:1" ;;
  a100) gres="gpu:A100:1" ;;
  l4) gres="gpu:L4:1" ;;
  local) gres="" ;;
  *) gres="gpu:${gpu_type}:1" ;;
esac

generate_manifest
num_cases="$(python - "${RESULT_ROOT}/config_manifest.json" <<'PY'
import json
import sys
print(len(json.load(open(sys.argv[1]))["cases"]))
PY
)"
array_max=$((num_cases - 1))

{
  echo "created_at=$(date -Is)"
  echo "scheduler=${scheduler}"
  echo "mode=${MODE}"
  echo "result_root=${RESULT_ROOT}"
  echo "gpu_type=${gpu_type}"
  echo "gpu_count_hint=$(gpu_count_hint "${gpu_type}")"
  echo "partition=${partition}"
  echo "gres=${gres}"
  echo "max_concurrent_jobs=${MAX_CONCURRENT_JOBS}"
  echo "case_offset=${CASE_OFFSET}"
  echo "case_limit=${CASE_LIMIT}"
  echo "num_cases=${num_cases}"
} | tee "${RESULT_ROOT}/submission_plan.txt"

{
  echo "# Next-round window/sparse commands"
  echo "RESULT_ROOT='${RESULT_ROOT}' MODE='${MODE}' CASE_OFFSET='${CASE_OFFSET}' CASE_LIMIT='${CASE_LIMIT}' MAX_CONCURRENT_JOBS='${MAX_CONCURRENT_JOBS}' CONFIRM_SUBMIT=1 bash scripts/submit_next_round_window_sparse.sh"
  echo "python scripts/summarize_next_round.py '${RESULT_ROOT}'"
} > "${RESULT_ROOT}/commands.txt"

print_table | tee -a "${RESULT_ROOT}/submission_plan.txt"

if (( num_cases < 1 )); then
  echo "No cases selected." | tee -a "${RESULT_ROOT}/job_ids.txt"
  exit 0
fi

if [[ "${scheduler}" != "slurm" ]]; then
  echo "No supported cluster scheduler detected; not submitting." | tee -a "${RESULT_ROOT}/job_ids.txt"
  exit 0
fi

submit_script="${RESULT_ROOT}/slurm/next_round_window_sparse_${MODE}.sbatch"
cat > "${submit_script}" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=next-round-${MODE}
#SBATCH --output=${RESULT_ROOT}/slurm/%x_%A_%a.out
#SBATCH --error=${RESULT_ROOT}/slurm/%x_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=${CPUS_PER_TASK}
#SBATCH --mem=${MEMORY}
#SBATCH --time=${TIME_LIMIT}
#SBATCH --partition=${partition}
EOF
if [[ -n "${gres}" ]]; then
  echo "#SBATCH --gres=${gres}" >> "${submit_script}"
fi
if [[ -n "${SBATCH_ACCOUNT:-}" ]]; then
  echo "#SBATCH --account=${SBATCH_ACCOUNT}" >> "${submit_script}"
fi
cat >> "${submit_script}" <<EOF

set -euo pipefail
cd "${ROOT_DIR}"
source "\$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
export RESULT_ROOT="${RESULT_ROOT}"
export MODE="${MODE}"
export DATA_SEED="${DATA_SEED}"
export BS="${BS}"
bash scripts/submit_next_round_window_sparse.sh --run-case "\${SLURM_ARRAY_TASK_ID}"
EOF

echo "submit_script=${submit_script}" | tee -a "${RESULT_ROOT}/submission_plan.txt"
if [[ "${CONFIRM_SUBMIT:-0}" != "1" ]]; then
  echo "CONFIRM_SUBMIT is not 1; dry-run only. Set CONFIRM_SUBMIT=1 to submit." | tee -a "${RESULT_ROOT}/job_ids.txt"
  exit 0
fi

cmd=(sbatch --array="0-${array_max}%${MAX_CONCURRENT_JOBS}" "${submit_script}")
printf -v submit_command '%q ' "${cmd[@]}"
echo "submit_command=${submit_command}" | tee -a "${RESULT_ROOT}/job_ids.txt"
if submit_output="$("${cmd[@]}" 2>&1)"; then
  echo "${submit_output}" | tee -a "${RESULT_ROOT}/job_ids.txt"
else
  echo "${submit_output}" | tee -a "${RESULT_ROOT}/job_ids.txt"
  exit 1
fi
