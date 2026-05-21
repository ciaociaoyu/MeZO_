#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MEDIUM_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${MEDIUM_DIR}/.." && pwd)"

TASK="${TASK:-sst-5}"
DATASET_MODE="${DATASET_MODE:-full}"
K="${K:-16}"
SEED="${SEED:-16}"
DATA_SEED="${DATA_SEED:-16}"
BS="${BS:-2}"
LR="${LR:-1e-6}"
EPS="${EPS:-1e-3}"
STEP="${STEP:-2}"
EVAL_STEP="${EVAL_STEP:-1}"
MODEL="${MODEL:-roberta-large}"
DATALOADER_SHUFFLE="${DATALOADER_SHUFFLE:-True}"
H_GRID_POLICY="${H_GRID_POLICY:-continuous}"
H_WINDOW_MIN="${H_WINDOW_MIN:-1e-5}"
H_WINDOW_MAX="${H_WINDOW_MAX:-1e-2}"
H_SCHEDULE_FD_CLIP_POLICY="${H_SCHEDULE_FD_CLIP_POLICY:-none}"
H_SCHEDULE_FD_FLOOR_MIN="${H_SCHEDULE_FD_FLOOR_MIN:-1e-5}"
H_SCHEDULE_FD_CLIP_MAX="${H_SCHEDULE_FD_CLIP_MAX:-0.0}"
H_SCHEDULE_FD_INT8_POLICY="${H_SCHEDULE_FD_INT8_POLICY:-fp16_proxy_raw}"
RESULT_ROOT="${RESULT_ROOT:-${REPO_ROOT}/outputs/smoke_main_mechanism_initial_h}"
SUMMARY_DIR="${SUMMARY_DIR:-${REPO_ROOT}/medium_models/pilot_results}"

RUN_PREFIX=()
if [[ -n "${SMOKE_CONDA_ENV:-}" ]]; then
    RUN_PREFIX=(conda run -n "${SMOKE_CONDA_ENV}")
fi
PYTHON_BIN="${PYTHON_BIN:-python}"
ENV_LABEL="${SMOKE_CONDA_ENV:-current environment}"

run_py_has() {
    grep -q "$1" "${MEDIUM_DIR}/run.py"
}

guard_args=()
if run_py_has "random_prediction_guard_enabled"; then
    guard_args+=(
        --random_prediction_guard_enabled True
        --random_prediction_guard_step 1000
        --random_prediction_guard_recent_evals 2
        --random_prediction_guard_acc_tolerance 0.05
        --random_prediction_guard_loss_tolerance 0.03
        --random_prediction_guard_bad_loss_excess 0.5
    )
fi
if run_py_has "zo_probe_health_guard_enabled"; then
    guard_args+=(
        --zo_probe_health_guard_enabled True
        --zo_probe_health_guard_step 1000
        --zo_probe_health_guard_max_bad_probes 3
    )
fi

echo "[preflight] checking existing default h=1e-3 outputs; this smoke will not rerun them"
mkdir -p "${SUMMARY_DIR}"
DEFAULT_PREFLIGHT="${SUMMARY_DIR}/main_mechanism_initial_h_default_preflight.txt"
{
    find "${REPO_ROOT}/outputs" "${REPO_ROOT}/experiments" -maxdepth 8 -type d \
        \( -iname '*h1e-3*' -o -iname '*eps1e-3*' -o -iname '*default*' \) 2>/dev/null | sort | head -200 || true
} >"${DEFAULT_PREFLIGHT}"
head -50 "${DEFAULT_PREFLIGHT}" || true

echo "[preflight] planned h values"
cd "${REPO_ROOT}"
"${RUN_PREFIX[@]}" "${PYTHON_BIN}" medium_models/tools/print_h_schedule.py \
    --format csv --steps 1 --zero_order_eps "${EPS}" --precision_mode fp32 \
    --h_schedule mezo_default --h_schedule_grid_policy "${H_GRID_POLICY}"
"${RUN_PREFIX[@]}" "${PYTHON_BIN}" medium_models/tools/print_h_schedule.py \
    --format csv --steps 2 --zero_order_eps "${EPS}" --precision_mode fp32 \
    --h_schedule fixed_small --h_schedule_grid_policy "${H_GRID_POLICY}" \
    --h_schedule_window_min "${H_WINDOW_MIN}" --h_schedule_window_max "${H_WINDOW_MAX}"
for precision in fp32 fp16 int8; do
    "${RUN_PREFIX[@]}" "${PYTHON_BIN}" medium_models/tools/print_h_schedule.py \
        --format csv --steps 1 --zero_order_eps "${EPS}" --precision_mode "${precision}" \
        --h_schedule fd_eps13_raw \
        --h_schedule_grid_policy "${H_GRID_POLICY}" \
        --h_schedule_window_min "${H_WINDOW_MIN}" \
        --h_schedule_window_max "${H_WINDOW_MAX}" \
        --h_schedule_fd_clip_policy "${H_SCHEDULE_FD_CLIP_POLICY}" \
        --h_schedule_fd_floor_min "${H_SCHEDULE_FD_FLOOR_MIN}" \
        --h_schedule_fd_clip_max "${H_SCHEDULE_FD_CLIP_MAX}" \
        --h_schedule_fd_int8_policy "${H_SCHEDULE_FD_INT8_POLICY}"
done

echo "[1/4] test h schedule helper in ${ENV_LABEL}"
if "${RUN_PREFIX[@]}" "${PYTHON_BIN}" -c "import pytest" >/dev/null 2>&1; then
    "${RUN_PREFIX[@]}" "${PYTHON_BIN}" -m pytest medium_models/tests/test_h_schedules.py
else
    echo "pytest is not installed for ${ENV_LABEL}; using unittest fallback." >&2
    "${RUN_PREFIX[@]}" "${PYTHON_BIN}" -m unittest medium_models.tests.test_h_schedules
fi

echo "[2/4] checking INT8 command invariants"
if ! run_py_has "groupwise_symmetric"; then
    echo "run.py does not expose groupwise_symmetric INT8 quantization" >&2
    exit 2
fi
if ! run_py_has "zo_update_backend"; then
    echo "run.py does not expose zo_update_backend for FP16 master INT8 updates" >&2
    exit 2
fi

echo "[3/4] running 2-step MeZO smoke matrix"
mkdir -p "${RESULT_ROOT}"
policies=(fixed_small_1e-5 fd_eps13_raw)
precisions=(fp32 fp16 int8)
for policy_tag in "${policies[@]}"; do
    for precision in "${precisions[@]}"; do
        if [[ "${policy_tag}" == "fixed_small_1e-5" ]]; then
            schedule="fixed_small"
        else
            schedule="fd_eps13_raw"
        fi
        tag="mainmech-initialh-${policy_tag}-${precision}"
        echo "[smoke] ${tag}"
        schedule_args=()
        if [[ "${schedule}" == "fixed_small" ]]; then
            schedule_args+=(--h_schedule_h0 1e-5)
        fi
        extra_args=()
        if [[ "${precision}" == "int8" ]]; then
            extra_args+=(
                --quantization_algorithm groupwise_symmetric
                --quantization_group_size 128
                --zo_update_backend fp16_master
            )
        fi
        (
            cd "${MEDIUM_DIR}"
            "${RUN_PREFIX[@]}" env \
                JOB_NAME="${tag}" \
                RESULT_ROOT="${RESULT_ROOT}" \
                TASK="${TASK}" \
                DATASET_MODE="${DATASET_MODE}" \
                K="${K}" \
                SEED="${SEED}" \
                DATA_SEED="${DATA_SEED}" \
                BS="${BS}" \
                LR="${LR}" \
                EPS="${EPS}" \
                STEP="${STEP}" \
                EVAL_STEP="${EVAL_STEP}" \
                MODEL="${MODEL}" \
                USE_H=False \
                USE_C=False \
                DATALOADER_SHUFFLE="${DATALOADER_SHUFFLE}" \
                EXTRA_TAG="${tag}" \
                TAG="${tag}" \
                bash mezo.sh \
                    --precision_mode "${precision}" \
                    --h_schedule "${schedule}" \
                    --h_schedule_grid_policy "${H_GRID_POLICY}" \
                    --h_schedule_window_min "${H_WINDOW_MIN}" \
                    --h_schedule_window_max "${H_WINDOW_MAX}" \
                    --h_schedule_fd_clip_policy "${H_SCHEDULE_FD_CLIP_POLICY}" \
                    --h_schedule_fd_floor_min "${H_SCHEDULE_FD_FLOOR_MIN}" \
                    --h_schedule_fd_clip_max "${H_SCHEDULE_FD_CLIP_MAX}" \
                    --h_schedule_fd_int8_policy "${H_SCHEDULE_FD_INT8_POLICY}" \
                    "${schedule_args[@]}" \
                    "${guard_args[@]}" \
                    "${extra_args[@]}"
        )
    done
done

echo "[4/4] verifying smoke outputs and writing summary"
"${PYTHON_BIN}" - "${RESULT_ROOT}" "${SUMMARY_DIR}" "${STEP}" "${DEFAULT_PREFLIGHT}" <<'PY'
import csv
import json
import math
import sys
from pathlib import Path

result_root = Path(sys.argv[1])
summary_dir = Path(sys.argv[2])
expected_step = int(sys.argv[3])
default_preflight = Path(sys.argv[4])
summary_dir.mkdir(parents=True, exist_ok=True)

rows = []
failures = []
expected = {
    ("fixed_small_1e-5", "fp32"): (1e-5, True, False),
    ("fixed_small_1e-5", "fp16"): (1e-5, True, False),
    ("fixed_small_1e-5", "int8"): (1e-5, True, False),
    ("fd_eps13_raw", "fp32"): (0.004921565763652325, True, False),
    ("fd_eps13_raw", "fp16"): (0.0992431640625, True, True),
    ("fd_eps13_raw", "int8"): (0.0992431640625, False, True),
}

for policy in ("fixed_small_1e-5", "fd_eps13_raw"):
    for precision in ("fp32", "fp16", "int8"):
        tag = f"mainmech-initialh-{policy}-{precision}"
        result_dir = result_root / tag / "seed16"
        h_csv = result_dir / "metrics_logs" / "h_schedule.csv"
        notes = []
        status = "completed"
        data = []
        if not result_dir.is_dir():
            status = "failed"
            notes.append("missing result_dir")
        if not h_csv.is_file():
            status = "failed"
            notes.append("missing h_schedule.csv")
        else:
            with h_csv.open() as f:
                data = list(csv.DictReader(f))
            if not data:
                status = "failed"
                notes.append("empty h_schedule.csv")

        first = data[0] if data else {}
        last = data[-1] if data else {}
        first_h = float(first.get("final_h", "nan")) if first else math.nan
        last_h = float(last.get("final_h", "nan")) if last else math.nan
        raw_h = float(first.get("raw_h", "nan")) if first else math.nan
        final_step_seen = int(last.get("global_step", -1)) if last else -1

        try:
            if data:
                expected_h, expected_principled, expected_out = expected[(policy, precision)]
                if abs(first_h - expected_h) > 1e-10:
                    raise AssertionError(f"final_h {first_h} != expected {expected_h}")
                if abs(raw_h - expected_h) > 1e-10:
                    raise AssertionError(f"raw_h {raw_h} != expected {expected_h}")
                actual_principled = str(first.get("fd_principled", "")).lower() in {"true", "1"}
                if actual_principled != expected_principled:
                    raise AssertionError(f"fd_principled {actual_principled} != expected {expected_principled}")
                actual_out = str(first.get("out_of_window_raw", "")).lower() in {"true", "1"}
                if actual_out != expected_out:
                    raise AssertionError(f"out_of_window_raw {actual_out} != expected {expected_out}")
                if policy == "fd_eps13_raw" and precision in {"fp16", "int8"} and first.get("cap_reason", ""):
                    raise AssertionError("raw FD fp16/int8 should not be capped")
                if policy == "fd_eps13_raw" and precision == "int8" and "no machine-epsilon analogue" not in first.get("fd_exception_reason", ""):
                    raise AssertionError("int8 FD proxy missing exception reason")
        except Exception as exc:
            status = "failed"
            notes.append(str(exc))

        if status != "completed":
            failures.append(tag)
        rows.append({
            "policy": policy,
            "precision": precision,
            "status": status,
            "result_dir": str(result_dir),
            "final_step_seen": final_step_seen if final_step_seen >= 0 else "NA",
            "first_h": first_h if not math.isnan(first_h) else "NA",
            "last_h": last_h if not math.isnan(last_h) else "NA",
            "raw_h": raw_h if not math.isnan(raw_h) else "NA",
            "final_h": first_h if not math.isnan(first_h) else "NA",
            "fd_principled": first.get("fd_principled", "NA"),
            "fd_exception_reason": first.get("fd_exception_reason", "NA"),
            "cap_reason": first.get("cap_reason", "NA"),
            "h_schedule_csv_exists": h_csv.is_file(),
            "notes": "; ".join(notes) if notes else "",
        })

csv_path = summary_dir / "main_mechanism_initial_h_smoke_summary.csv"
with csv_path.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)

default_paths = [line.strip() for line in default_preflight.read_text().splitlines() if line.strip()]
md_path = summary_dir / "main_mechanism_initial_h_smoke_summary.md"
completed = [f"{r['policy']}-{r['precision']}" for r in rows if r["status"] == "completed"]
with md_path.open("w") as f:
    f.write("# Main Mechanism Initial-h Baselines Smoke Summary\n\n")
    f.write(f"- Result root: `{result_root}`\n")
    f.write(f"- Expected max steps: `{expected_step}`\n")
    f.write(f"- Completed: {', '.join(completed) if completed else 'none'}\n")
    f.write(f"- Failed: {', '.join(failures) if failures else 'none'}\n")
    f.write(f"- Default h=1e-3 paths found in preflight: `{len(default_paths)}`\n")
    f.write("- Default h=1e-3 was not rerun by this smoke.\n")
    f.write("- Fixed small final h: `1e-5` for FP32/FP16/INT8.\n")
    f.write("- Raw FD FP32 final h: `0.004921565763652325`.\n")
    f.write("- Raw FD FP16 final h: `0.0992431640625`, intentionally out of window and uncapped.\n")
    f.write("- Raw FD INT8 final h: `0.0992431640625`, FP16 proxy, not principled, intentionally out of window and uncapped.\n\n")
    f.write("Raw outputs are under the ignored `outputs/` tree and are not intended for commit.\n")

print(json.dumps({"summary_csv": str(csv_path), "summary_md": str(md_path), "failures": failures}, indent=2))
if failures:
    raise SystemExit(1)
PY

echo "smoke_main_mechanism_initial_h_baselines.sh completed"
