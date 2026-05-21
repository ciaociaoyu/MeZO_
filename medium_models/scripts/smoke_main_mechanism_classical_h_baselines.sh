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
SPALL_H0="${SPALL_H0:-1e-3}"
SPALL_GAMMA="${SPALL_GAMMA:-0.101}"
RESULT_ROOT="${RESULT_ROOT:-${REPO_ROOT}/outputs/smoke_main_mechanism_classical_h}"
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
    guard_args+=(--random_prediction_guard_enabled True)
fi
if run_py_has "zo_probe_health_guard_enabled"; then
    guard_args+=(--zo_probe_health_guard_enabled True)
fi

echo "[preflight] checking existing default h=1e-3 outputs; this smoke will not rerun them"
find "${REPO_ROOT}/outputs" "${REPO_ROOT}/experiments" -maxdepth 6 -type d \
    \( -iname '*h1e-3*' -o -iname '*eps1e-3*' -o -iname '*default*' \) 2>/dev/null | head -50 || true

echo "[1/4] test h schedule helper in ${ENV_LABEL}"
cd "${REPO_ROOT}"
if "${RUN_PREFIX[@]}" "${PYTHON_BIN}" -c "import pytest" >/dev/null 2>&1; then
    "${RUN_PREFIX[@]}" "${PYTHON_BIN}" -m pytest medium_models/tests/test_h_schedules.py
else
    echo "pytest is not installed for ${ENV_LABEL}; using unittest fallback." >&2
    "${RUN_PREFIX[@]}" "${PYTHON_BIN}" -m unittest medium_models.tests.test_h_schedules
fi

echo "[2/4] print_h_schedule diagnostics"
"${RUN_PREFIX[@]}" "${PYTHON_BIN}" medium_models/tools/print_h_schedule.py \
    --format csv --steps 1 --zero_order_eps "${EPS}" --precision_mode fp32 \
    --h_schedule fd_eps13 --h_schedule_fd_clip_min "${H_WINDOW_MIN}" --h_schedule_fd_clip_max "${H_WINDOW_MAX}"
"${RUN_PREFIX[@]}" "${PYTHON_BIN}" medium_models/tools/print_h_schedule.py \
    --format csv --steps 1 --zero_order_eps "${EPS}" --precision_mode fp16 \
    --h_schedule fd_eps13 --h_schedule_fd_clip_min "${H_WINDOW_MIN}" --h_schedule_fd_clip_max "${H_WINDOW_MAX}"
"${RUN_PREFIX[@]}" "${PYTHON_BIN}" medium_models/tools/print_h_schedule.py \
    --format csv --steps 1 --zero_order_eps "${EPS}" --precision_mode int8 \
    --h_schedule fd_eps13 --h_schedule_fd_clip_min "${H_WINDOW_MIN}" --h_schedule_fd_clip_max "${H_WINDOW_MAX}" \
    --h_schedule_fd_int8_policy capped_stress
"${RUN_PREFIX[@]}" "${PYTHON_BIN}" medium_models/tools/print_h_schedule.py \
    --format csv --steps 5 --include_steps 20000 --zero_order_eps "${EPS}" --precision_mode fp32 \
    --h_schedule spall_ck --h_schedule_h0 "${SPALL_H0}" --h_schedule_gamma "${SPALL_GAMMA}" \
    --h_schedule_window_min "${H_WINDOW_MIN}" --h_schedule_window_max "${H_WINDOW_MAX}" \
    --h_schedule_grid_policy "${H_GRID_POLICY}"

echo "[3/4] running 2-step MeZO smoke matrix"
mkdir -p "${RESULT_ROOT}" "${SUMMARY_DIR}"
policies=(fd_eps13 spall_ck)
precisions=(fp32 fp16 int8)
for policy in "${policies[@]}"; do
    for precision in "${precisions[@]}"; do
        tag="mainmech-classicalh-${policy}-${precision}"
        echo "[smoke] ${tag}"
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
                    --h_schedule "${policy}" \
                    --h_schedule_grid_policy "${H_GRID_POLICY}" \
                    --h_schedule_window_min "${H_WINDOW_MIN}" \
                    --h_schedule_window_max "${H_WINDOW_MAX}" \
                    --h_schedule_fd_clip_min "${H_WINDOW_MIN}" \
                    --h_schedule_fd_clip_max "${H_WINDOW_MAX}" \
                    --h_schedule_fd_int8_policy capped_stress \
                    --h_schedule_h0 "${SPALL_H0}" \
                    --h_schedule_gamma "${SPALL_GAMMA}" \
                    "${guard_args[@]}" \
                    "${extra_args[@]}"
        )
    done
done

echo "[4/4] verifying smoke outputs and writing summary"
"${PYTHON_BIN}" - "${RESULT_ROOT}" "${SUMMARY_DIR}" "${STEP}" <<'PY'
import csv
import math
import sys
from pathlib import Path

result_root = Path(sys.argv[1])
summary_dir = Path(sys.argv[2])
expected_step = int(sys.argv[3])
summary_dir.mkdir(parents=True, exist_ok=True)

rows = []
failures = []
for policy in ("fd_eps13", "spall_ck"):
    for precision in ("fp32", "fp16", "int8"):
        tag = f"mainmech-classicalh-{policy}-{precision}"
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
        final_h = first_h
        final_step_seen = int(last.get("global_step", -1)) if last else -1

        try:
            if policy == "fd_eps13" and data:
                if precision == "fp32":
                    expected = float(2.0 ** (-23.0 / 3.0))
                    if abs(first_h - expected) > 1e-8:
                        raise AssertionError(f"fd fp32 final_h {first_h} != {expected}")
                elif precision == "fp16":
                    if abs(first_h - 1e-2) > 1e-12:
                        raise AssertionError(f"fd fp16 final_h {first_h} != 1e-2")
                    if "fp16" not in first.get("cap_reason", ""):
                        raise AssertionError("fd fp16 missing cap_reason")
                elif precision == "int8":
                    if abs(first_h - 1e-2) > 1e-12:
                        raise AssertionError(f"fd int8 final_h {first_h} != 1e-2")
                    if str(first.get("fd_principled", "")).lower() not in {"false", "0"}:
                        raise AssertionError("fd int8 should set fd_principled=false")
            if policy == "spall_ck" and data:
                if abs(first_h - 1e-3) > 1e-12:
                    raise AssertionError(f"spall first_h {first_h} != 1e-3")
                if len(data) > 1 and not (last_h < first_h):
                    raise AssertionError(f"spall last_h {last_h} should be < first_h {first_h}")
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
            "final_h": final_h if not math.isnan(final_h) else "NA",
            "fd_principled": first.get("fd_principled", "NA"),
            "fd_exception_reason": first.get("fd_exception_reason", "NA"),
            "cap_reason": first.get("cap_reason", "NA"),
            "h_schedule_csv_exists": h_csv.is_file(),
            "notes": "; ".join(notes) if notes else "",
        })

csv_path = summary_dir / "main_mechanism_classical_h_smoke_summary.csv"
with csv_path.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)

md_path = summary_dir / "main_mechanism_classical_h_smoke_summary.md"
completed = [f"{r['policy']}-{r['precision']}" for r in rows if r["status"] == "completed"]
with md_path.open("w") as f:
    f.write("# Main Mechanism Classical h Baselines Smoke Summary\n\n")
    f.write(f"- Result root: `{result_root}`\n")
    f.write(f"- Expected max steps: `{expected_step}`\n")
    f.write(f"- Completed: {', '.join(completed) if completed else 'none'}\n")
    f.write(f"- Failed: {', '.join(failures) if failures else 'none'}\n")
    f.write("- FD fp32 final h: `np.finfo(np.float32).eps ** (1/3)`\n")
    f.write("- FD fp16 final h: capped to `1e-2`\n")
    f.write("- FD int8 final h: capped stress `1e-2`, not principled\n")
    f.write("- Spall c_k: starts at `1e-3` and decays continuously with gamma `0.101`\n\n")
    f.write("Raw outputs are under the ignored `outputs/` tree and are not intended for commit.\n")

print(json.dumps({"summary_csv": str(csv_path), "summary_md": str(md_path), "failures": failures}, indent=2))
if failures:
    raise SystemExit(1)
PY

echo "smoke_main_mechanism_classical_h_baselines.sh completed"
