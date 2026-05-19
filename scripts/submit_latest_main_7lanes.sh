#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

EXPERIMENT_ID="${EXPERIMENT_ID:-latest_main_roberta_sst5_fp32_fp16_hsweep_seed16_bs64_ckpt1k}"
EXP_ROOT="${EXP_ROOT:-${ROOT_DIR}/experiments/main_latest/mezo/roberta-large/sst5/fp32_fp16_h_sweep_11h_seed16_bs64_ckpt1k_20260517}"
CONDA_ENV="${CONDA_ENV:-ciao}"
PARTITION="${SLURM_PARTITION:-gpu_p}"
TIME_LIMIT="${TIME_LIMIT:-72:00:00}"
CPUS_PER_TASK="${CPUS_PER_TASK:-8}"
MEMORY="${MEMORY:-96G}"
LR="${LR:-1e-6}"
DATA_ROOT="${DATA_ROOT:-data/k-shot-1k-test}"
SMOKE_ONLY=0

if [[ "${1:-}" == "--smoke" ]]; then
  SMOKE_ONLY=1
fi

mkdir -p \
  "${EXP_ROOT}/jobs" \
  "${EXP_ROOT}/logs" \
  "${EXP_ROOT}/summaries" \
  "${EXP_ROOT}/plots" \
  "${EXP_ROOT}/probe_diagnostics/fp32" \
  "${EXP_ROOT}/probe_diagnostics/fp16" \
  "${EXP_ROOT}/lane_manifests" \
  "${EXP_ROOT}/fp32/h_sweep_11h_checkpointed/results" \
  "${EXP_ROOT}/fp16/h_sweep_11h_checkpointed/results"

generate_manifests() {
  python - "${EXP_ROOT}" "${EXPERIMENT_ID}" "${LR}" <<'PY'
import csv
import json
import math
import sys
from pathlib import Path

root = Path(sys.argv[1])
experiment_id = sys.argv[2]
lr = float(sys.argv[3])
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
    0: ("H100", [("fp32", "1e-5"), ("fp16", "1e-5"), ("fp32", "2e-3"), ("fp16", "2e-3")]),
    1: ("H100", [("fp32", "3e-5"), ("fp16", "3e-5"), ("fp32", "3e-3"), ("fp16", "3e-3")]),
    2: ("H100", [("fp32", "1e-4"), ("fp16", "1e-4"), ("fp32", "4e-3"), ("fp16", "4e-3")]),
    3: ("A100", [("fp32", "3e-4"), ("fp16", "3e-4"), ("fp32", "5e-3")]),
    4: ("A100", [("fp16", "5e-3"), ("fp32", "1e-3"), ("fp16", "1e-3")]),
    5: ("A100", [("fp32", "1p5e-3"), ("fp16", "1p5e-3"), ("fp32", "1e-2")]),
    6: ("A100", [("fp16", "1e-2")]),
}
h_by_label = dict(h_grid)
rows = []
for lane_id, (gpu_type, specs) in lane_specs.items():
    for precision, h_label in specs:
        h = h_by_label[h_label]
        run_name = f"{precision}_h{h_label}_seed16_bs64_ckpt1k"
        result_root = root / precision / "h_sweep_11h_checkpointed" / "results"
        rows.append({
            "lane_id": lane_id,
            "gpu_type": gpu_type,
            "precision_mode": precision,
            "h": f"{h:.12g}",
            "h_label": h_label,
            "run_name": run_name,
            "result_root": str(result_root),
            "max_steps": 20000,
            "eval_steps": 1000,
            "checkpoint_steps": 1000,
            "seed": 16,
            "data_seed": 16,
            "batch_size": 64,
            "lr": f"{lr:.12g}",
        })
seen = {(row["precision_mode"], row["h_label"]) for row in rows}
expected = {(precision, label) for precision in ("fp32", "fp16") for label, _ in h_grid}
missing = sorted(expected - seen)
extra = sorted(seen - expected)
if missing or extra or len(rows) != 22:
    raise SystemExit(f"bad run manifest: rows={len(rows)} missing={missing} extra={extra}")
fields = ["lane_id","gpu_type","precision_mode","h","h_label","run_name","result_root","max_steps","eval_steps","checkpoint_steps","seed","data_seed","batch_size","lr"]
with (root / "run_manifest.csv").open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
for lane_id in range(7):
    lane_rows = [row for row in rows if int(row["lane_id"]) == lane_id]
    with (root / "lane_manifests" / f"lane{lane_id}.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(lane_rows)
config = {
    "experiment_id": experiment_id,
    "model": "roberta-large",
    "task": "SST-5",
    "method": "mezo",
    "dataset_mode": "full",
    "seed": 16,
    "data_seed": 16,
    "dataloader_shuffle": True,
    "batch_size": 64,
    "gradient_accumulation_steps": 1,
    "precisions": ["fp32", "fp16"],
    "h_grid": [{"label": label, "h": h} for label, h in h_grid],
    "max_steps": 20000,
    "eval_steps": 1000,
    "checkpoint_steps": 1000,
    "total_runs": len(rows),
    "excluded": ["int8", "int4", "sparse", "residual_grid", "lora", "rte", "opt", "mnli"],
}
(root / "config_manifest.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
readme = f"""# {experiment_id}

Latest checkpointed main MeZO h-sweep for RoBERTa-large / SST-5.

- Scope: FP32 and FP16 only.
- Excluded: INT8, INT4, sparse directions, residual_grid, LoRA, RTE, OPT, MNLI.
- Dataset: full SST-5 (`dataset_mode=full`), seed=16, data_seed=16.
- Dataloader shuffle: enabled explicitly by `DATALOADER_SHUFFLE=True`.
- Batch size: 64.
- h-grid: {', '.join(label for label, _ in h_grid)}.
- Checkpoints: every 1000 steps plus final, best_acc, best_loss under each run's `checkpoints/`.
- Scheduling: 7 lanes; lanes 0-2 request H100, lanes 3-6 request A100.
"""
(root / "README.md").write_text(readme, encoding="utf-8")
PY
}

print_dry_run() {
  python - "${EXP_ROOT}/run_manifest.csv" <<'PY'
import csv
import sys
from collections import defaultdict
rows = list(csv.DictReader(open(sys.argv[1], newline="")))
by_lane = defaultdict(list)
for row in rows:
    by_lane[int(row["lane_id"])].append(row)
print("lane_id gpu_type num_runs total_estimated_steps output_dir run_names")
for lane in sorted(by_lane):
    group = by_lane[lane]
    gpu = group[0]["gpu_type"]
    steps = sum(int(r["max_steps"]) for r in group)
    out = ";".join(sorted({r["result_root"] for r in group}))
    names = " ".join(r["run_name"] for r in group)
    print(f"{lane:7d} {gpu:8s} {len(group):8d} {steps:21d} {out} {names}")
PY
}

write_commands() {
  cat > "${EXP_ROOT}/commands.txt" <<EOF
# Latest main FP32/FP16 h-sweep commands
EXP_ROOT='${EXP_ROOT}' bash scripts/submit_latest_main_7lanes.sh --smoke
EXP_ROOT='${EXP_ROOT}' CONFIRM_SUBMIT=1 bash scripts/submit_latest_main_7lanes.sh
python scripts/summarize_latest_main_hsweep.py '${EXP_ROOT}'
EOF
}

generate_manifests
write_commands
print_dry_run | tee "${EXP_ROOT}/jobs/dry_run_table.txt"

if [[ "${SMOKE_ONLY}" == "1" ]]; then
  smoke_root="${EXP_ROOT}/smoke"
  mkdir -p "${smoke_root}/lane_manifests" "${smoke_root}/fp32/h_sweep_11h_checkpointed/results" "${smoke_root}/fp16/h_sweep_11h_checkpointed/results" "${smoke_root}/logs" "${smoke_root}/summaries" "${smoke_root}/plots"
  python - "${smoke_root}" "${LR}" <<'PY'
import csv
import sys
from pathlib import Path
root = Path(sys.argv[1])
lr = float(sys.argv[2])
fields = ["lane_id","gpu_type","precision_mode","h","h_label","run_name","result_root","max_steps","eval_steps","checkpoint_steps","seed","data_seed","batch_size","lr"]
rows = [
    {"lane_id":0,"gpu_type":"local","precision_mode":"fp32","h":"0.001","h_label":"1e-3","run_name":"smoke_fp32_h1e-3_seed16_bs64_ckpt25","result_root":str(root/"fp32/h_sweep_11h_checkpointed/results"),"max_steps":50,"eval_steps":25,"checkpoint_steps":25,"seed":16,"data_seed":16,"batch_size":64,"lr":f"{lr:.12g}"},
    {"lane_id":0,"gpu_type":"local","precision_mode":"fp16","h":"0.001","h_label":"1e-3","run_name":"smoke_fp16_h1e-3_seed16_bs64_ckpt25","result_root":str(root/"fp16/h_sweep_11h_checkpointed/results"),"max_steps":50,"eval_steps":25,"checkpoint_steps":25,"seed":16,"data_seed":16,"batch_size":64,"lr":f"{lr:.12g}"},
]
for path in [root/"run_manifest.csv", root/"lane_manifests/lane0.csv"]:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
PY
  EXP_ROOT="${smoke_root}" LANE_ID=0 LANE_MANIFEST="${smoke_root}/lane_manifests/lane0.csv" PROBE_DIRECTIONS=4 bash scripts/run_latest_main_lane.sh
  python scripts/summarize_latest_main_hsweep.py "${smoke_root}"
  echo "smoke_root=${smoke_root}"
  exit 0
fi

if [[ "${CONFIRM_SUBMIT:-0}" != "1" ]]; then
  echo "CONFIRM_SUBMIT is not 1; dry-run only. Set CONFIRM_SUBMIT=1 to submit seven lanes."
  exit 0
fi

if ! command -v sbatch >/dev/null 2>&1; then
  echo "sbatch not found; use scripts/run_latest_main_lane.sh manually with LANE_ID=0..6." >&2
  exit 2
fi

h100_script="${EXP_ROOT}/jobs/latest_main_h100_lanes.sbatch"
a100_script="${EXP_ROOT}/jobs/latest_main_a100_lanes.sbatch"
cat > "${h100_script}" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=latest-main-h100
#SBATCH --output=${EXP_ROOT}/jobs/%x_%A_%a.out
#SBATCH --error=${EXP_ROOT}/jobs/%x_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=${CPUS_PER_TASK}
#SBATCH --mem=${MEMORY}
#SBATCH --time=${TIME_LIMIT}
#SBATCH --partition=${PARTITION}
#SBATCH --gres=gpu:H100:1
set -euo pipefail
cd "${ROOT_DIR}"
export EXP_ROOT="${EXP_ROOT}"
export EXPERIMENT_ID="${EXPERIMENT_ID}"
export CONDA_ENV="${CONDA_ENV}"
export DATA_ROOT="${DATA_ROOT}"
export LANE_ID="\${SLURM_ARRAY_TASK_ID}"
bash scripts/run_latest_main_lane.sh
EOF
cat > "${a100_script}" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=latest-main-a100
#SBATCH --output=${EXP_ROOT}/jobs/%x_%A_%a.out
#SBATCH --error=${EXP_ROOT}/jobs/%x_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=${CPUS_PER_TASK}
#SBATCH --mem=${MEMORY}
#SBATCH --time=${TIME_LIMIT}
#SBATCH --partition=${PARTITION}
#SBATCH --gres=gpu:A100:1
set -euo pipefail
cd "${ROOT_DIR}"
export EXP_ROOT="${EXP_ROOT}"
export EXPERIMENT_ID="${EXPERIMENT_ID}"
export CONDA_ENV="${CONDA_ENV}"
export DATA_ROOT="${DATA_ROOT}"
export LANE_ID="\${SLURM_ARRAY_TASK_ID}"
bash scripts/run_latest_main_lane.sh
EOF

{
  echo "submitted_at=$(date -Is)"
  echo "experiment_id=${EXPERIMENT_ID}"
  echo "exp_root=${EXP_ROOT}"
  echo "partition=${PARTITION}"
  echo "h100_lanes=0-2"
  echo "a100_lanes=3-6"
  echo "no_int8=1"
  echo "no_sparse=1"
  echo "no_residual_grid=1"
} > "${EXP_ROOT}/jobs/job_ids.txt"

sbatch --array=0-2%3 "${h100_script}" | tee -a "${EXP_ROOT}/jobs/job_ids.txt"
sbatch --array=3-6%4 "${a100_script}" | tee -a "${EXP_ROOT}/jobs/job_ids.txt"
