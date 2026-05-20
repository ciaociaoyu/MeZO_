#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

DATE_TAG="${DATE_TAG:-$(date +%Y%m%d_%H%M%S)}"
EXP_ROOT="${EXP_ROOT:-${ROOT_DIR}/outputs/rtnclip_lowbit_roberta_sst5_seed16_${DATE_TAG}}"
CONDA_ENV="${CONDA_ENV:-ciao}"
PARTITION="${SLURM_PARTITION:-gpu_p}"
TIME_LIMIT="${TIME_LIMIT:-72:00:00}"
CPUS_PER_TASK="${CPUS_PER_TASK:-8}"
MEMORY="${MEMORY:-96G}"
LR="${LR:-1e-6}"
MAX_STEPS="${MAX_STEPS:-20000}"
EVAL_EVERY="${EVAL_EVERY:-1000}"
CHECKPOINT_STEPS="${CHECKPOINT_STEPS:-1000}"
MAX_ACTIVE_LANES="${MAX_ACTIVE_LANES:-6}"

mkdir -p "${EXP_ROOT}/"{jobs,logs,summaries,int8_hsearch,int4_probe,smoke}

usage() {
  cat <<'EOF'
Usage:
  bash scripts/submit_rtnclip_lowbit_roberta_sst5.sh --prepare
  CONFIRM_SUBMIT=1 bash scripts/submit_rtnclip_lowbit_roberta_sst5.sh --submit-int8

Smoke and INT4 probe are run directly via tools/rtnclip_roberta_sst5_batch.py.
This script prepares/submits the INT8 11-h lane training phase only.
EOF
}

MODE="${1:---prepare}"
case "${MODE}" in
  --prepare|--submit-int8|-h|--help) ;;
  *) echo "Unknown mode: ${MODE}" >&2; usage; exit 2 ;;
esac
if [[ "${MODE}" == "-h" || "${MODE}" == "--help" ]]; then
  usage
  exit 0
fi

generate_manifests() {
  python - "${EXP_ROOT}" "${MAX_STEPS}" "${EVAL_EVERY}" "${CHECKPOINT_STEPS}" "${LR}" <<'PY'
import csv
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
max_steps = int(sys.argv[2])
eval_every = int(sys.argv[3])
checkpoint_steps = int(sys.argv[4])
lr = float(sys.argv[5])

h_grid = [
    ("1e-5", 1e-5),
    ("3e-5", 3e-5),
    ("1e-4", 1e-4),
    ("3e-4", 3e-4),
    ("1e-3", 1e-3),
    ("1p5e-3", 1.5e-3),
    ("2e-3", 2e-3),
    ("3e-3", 3e-3),
    ("4e-3", 4e-3),
    ("5e-3", 5e-3),
    ("1e-2", 1e-2),
]
lane_specs = {
    0: ("H100", ["1e-5", "1e-3"]),
    1: ("H100", ["3e-5", "1p5e-3"]),
    2: ("A100", ["1e-4", "2e-3"]),
    3: ("A100", ["3e-4", "3e-3"]),
    4: ("A100", ["4e-3", "5e-3"]),
    5: ("A100", ["1e-2"]),
}
h_by_label = dict(h_grid)
fields = [
    "lane_id",
    "gpu_type",
    "phase",
    "bitwidth",
    "h",
    "h_label",
    "run_name",
    "run_dir",
    "max_steps",
    "eval_every",
    "checkpoint_steps",
    "scale_refresh_k",
    "seed",
    "data_seed",
    "batch_size",
    "lr",
]
rows = []
for lane_id, (gpu_type, labels) in lane_specs.items():
    for label in labels:
        h = h_by_label[label]
        run_name = f"int8_g128_rtnclip_k1_h{label}_seed16_bs64_ckpt1k"
        run_dir = root / "int8_hsearch" / run_name
        rows.append({
            "lane_id": lane_id,
            "gpu_type": gpu_type,
            "phase": "int8_hsearch",
            "bitwidth": 8,
            "h": f"{h:.12g}",
            "h_label": label,
            "run_name": run_name,
            "run_dir": str(run_dir),
            "max_steps": max_steps,
            "eval_every": eval_every,
            "checkpoint_steps": checkpoint_steps,
            "scale_refresh_k": 1,
            "seed": 16,
            "data_seed": 16,
            "batch_size": 64,
            "lr": f"{lr:.12g}",
        })
expected = set(label for label, _ in h_grid)
seen = set(r["h_label"] for r in rows)
if seen != expected or len(rows) != 11:
    raise SystemExit(f"bad manifest seen={sorted(seen)} expected={sorted(expected)} rows={len(rows)}")
(root / "lane_manifests").mkdir(parents=True, exist_ok=True)
with (root / "int8_hsearch_manifest.csv").open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
for lane in range(6):
    lane_rows = [r for r in rows if int(r["lane_id"]) == lane]
    with (root / "lane_manifests" / f"lane{lane}.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(lane_rows)
config = {
    "experiment": "rtnclip_lowbit_roberta_sst5_seed16",
    "model": "roberta-large",
    "dataset": "SST-5",
    "dataset_mode": "full",
    "seed": 16,
    "data_seed": 16,
    "batch_size": 64,
    "shuffle": True,
    "direction": "dense",
    "quantizer": "G128_groupwise_RTNClip_fake_quant",
    "update_backend": "fp16_master",
    "master_dtype": "fp16",
    "pair_shared_grid": True,
    "fresh_round_codes": True,
    "int8_h_grid": [{"label": label, "h": h} for label, h in h_grid],
    "int8_max_steps": max_steps,
    "lane_count": 6,
    "max_active_lanes": 6,
    "excluded": ["GPTQ", "residual_grid", "direct_int_update", "sparse", "LoRA", "RTE", "MNLI", "OPT", "Mistral"],
}
(root / "config_manifest.json").write_text(json.dumps(config, indent=2) + "\n")
readme = """# RTNClip Low-Bit RoBERTa SST-5 Seed16

Official dense RoBERTa-large / SST-5 low-bit batch using G128 groupwise RTNClip
fake quantization, shared-grid plus/minus probes, fresh perturbed integer codes,
and FP16 master updates. This is not GPTQ and does not run sparse, residual-grid,
direct INT update, LoRA, RTE/MNLI/OPT/Mistral, NF4, AWQ, or HQQ.
"""
(root / "README.md").write_text(readme)
PY
}

print_dry_run() {
  python - "${EXP_ROOT}/int8_hsearch_manifest.csv" <<'PY'
import csv
import sys
from collections import defaultdict
rows = list(csv.DictReader(open(sys.argv[1], newline="")))
by_lane = defaultdict(list)
for row in rows:
    by_lane[int(row["lane_id"])].append(row)
print("lane_id gpu_pref num_runs total_steps run_names")
for lane in sorted(by_lane):
    group = by_lane[lane]
    print(f"{lane:7d} {group[0]['gpu_type']:8s} {len(group):8d} {sum(int(r['max_steps']) for r in group):11d} {' '.join(r['run_name'] for r in group)}")
PY
}

write_commands() {
  cat > "${EXP_ROOT}/commands.txt" <<EOF
# RTNClip low-bit RoBERTa/SST-5 commands
OUTPUT_ROOT='${EXP_ROOT}' CUDA_VISIBLE_DEVICES=0 DATALOADER_SHUFFLE=True python tools/rtnclip_roberta_sst5_batch.py --output_root '${EXP_ROOT}' --checkpoint_steps 25 --eval_every 50 smoke
OUTPUT_ROOT='${EXP_ROOT}' CUDA_VISIBLE_DEVICES=0 DATALOADER_SHUFFLE=True python tools/rtnclip_roberta_sst5_batch.py --output_root '${EXP_ROOT}' probe-int4
EXP_ROOT='${EXP_ROOT}' bash scripts/submit_rtnclip_lowbit_roberta_sst5.sh --prepare
EXP_ROOT='${EXP_ROOT}' CONFIRM_SUBMIT=1 bash scripts/submit_rtnclip_lowbit_roberta_sst5.sh --submit-int8
python tools/rtnclip_roberta_sst5_batch.py --output_root '${EXP_ROOT}' summarize
EOF
}

write_sbatch() {
  local script_path="$1"
  local job_name="$2"
  local lane_id="$3"
  local gpu_kind="$4"
  local fallback_used="$5"
  cat > "${script_path}" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=${job_name}
#SBATCH --output=${EXP_ROOT}/jobs/%x_%j.out
#SBATCH --error=${EXP_ROOT}/jobs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=${CPUS_PER_TASK}
#SBATCH --mem=${MEMORY}
#SBATCH --time=${TIME_LIMIT}
#SBATCH --partition=${PARTITION}
#SBATCH --gres=gpu:${gpu_kind}:1
set -euo pipefail
cd "${ROOT_DIR}"
if [[ -f "\$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
  source "\$HOME/miniconda3/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV}"
fi
export DATALOADER_SHUFFLE=True
export TOKENIZERS_PARALLELISM=false
export REQUESTED_GPU_TYPE="${gpu_kind}"
export FALLBACK_USED="${fallback_used}"
python tools/rtnclip_roberta_sst5_batch.py \
  --output_root "${EXP_ROOT}" \
  --manifest "${EXP_ROOT}/lane_manifests/lane${lane_id}.csv" \
  --eval_every "${EVAL_EVERY}" \
  --checkpoint_steps "${CHECKPOINT_STEPS}" \
  run-manifest
EOF
}

submit_lane_with_fallback() {
  local lane_id="$1"
  local preferred_gpu="$2"
  local preferred_script="${EXP_ROOT}/jobs/rtnclip_int8_lane${lane_id}_${preferred_gpu}.sbatch"
  local fallback_script="${EXP_ROOT}/jobs/rtnclip_int8_lane${lane_id}_L4.sbatch"
  write_sbatch "${preferred_script}" "rtnclip-i8-l${lane_id}-${preferred_gpu}" "${lane_id}" "${preferred_gpu}" "0"
  write_sbatch "${fallback_script}" "rtnclip-i8-l${lane_id}-L4" "${lane_id}" "L4" "1"
  local job_id
  job_id="$(sbatch --parsable "${preferred_script}")"
  echo "lane${lane_id}_preferred_${preferred_gpu}_job=${job_id}" | tee -a "${EXP_ROOT}/jobs/job_ids.txt"
  local waited=0
  while [[ "${waited}" -lt 300 ]]; do
    local state
    state="$(squeue -h -j "${job_id}" -o "%T" 2>/dev/null | head -n 1 || true)"
    if [[ -z "${state}" || "${state}" == "RUNNING" ]]; then
      echo "lane${lane_id}_state_after_${waited}s=${state:-completed_or_left_queue}" | tee -a "${EXP_ROOT}/jobs/job_ids.txt"
      return 0
    fi
    sleep 30
    waited=$((waited + 30))
  done
  echo "lane${lane_id}_preferred_pending_after_300s=${job_id}; cancelling and submitting L4 fallback" | tee -a "${EXP_ROOT}/jobs/job_ids.txt"
  scancel "${job_id}" || true
  local fb_id
  fb_id="$(sbatch --parsable "${fallback_script}")"
  echo "lane${lane_id}_fallback_L4_job=${fb_id}" | tee -a "${EXP_ROOT}/jobs/job_ids.txt"
}

generate_manifests
write_commands
print_dry_run | tee "${EXP_ROOT}/jobs/dry_run_table.txt"

if [[ "${MODE}" == "--prepare" ]]; then
  echo "Prepared RTNClip low-bit batch at ${EXP_ROOT}"
  exit 0
fi

if [[ "${CONFIRM_SUBMIT:-0}" != "1" ]]; then
  echo "CONFIRM_SUBMIT is not 1; refusing submission." >&2
  exit 2
fi
if [[ "${MAX_ACTIVE_LANES}" -gt 6 ]]; then
  echo "MAX_ACTIVE_LANES=${MAX_ACTIVE_LANES} exceeds hard cap 6." >&2
  exit 2
fi
if ! command -v sbatch >/dev/null 2>&1; then
  echo "sbatch not found." >&2
  exit 2
fi

: > "${EXP_ROOT}/jobs/job_ids.txt"
{
  echo "submitted_at=$(date -Is)"
  echo "exp_root=${EXP_ROOT}"
  echo "partition=${PARTITION}"
  echo "max_active_lanes=6"
  echo "squeue_before=$(squeue -u "${USER}" -o '%.18i %.9P %.30j %.8T %.10M %.9l %.6D %R' | tr '\n' ';')"
  echo "sinfo_before=$(sinfo -o '%20P %10a %10l %10D %N' | head -20 | tr '\n' ';')"
} | tee -a "${EXP_ROOT}/jobs/job_ids.txt"

preferred_gpus=(H100 H100 A100 A100 A100 A100)
for lane_id in 0 1 2 3 4 5; do
  submit_lane_with_fallback "${lane_id}" "${preferred_gpus[$lane_id]}"
done

echo "Submitted up to six INT8 RTNClip lane jobs. Monitor with: squeue -u ${USER}"
