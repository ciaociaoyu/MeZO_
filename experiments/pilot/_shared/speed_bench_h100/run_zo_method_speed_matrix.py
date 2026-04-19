#!/usr/bin/env python3
"""Run a resumable ZO-method speed matrix across repo-supported model paths."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional


REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT
    / "experiments"
    / "pilot"
    / "_shared"
    / "speed_bench_h100"
    / "zo_method_matrix_20260418"
)
SUMMARY_NAME = "summary.jsonl"

METHODS = ("mezo", "sparse_mezo", "lozo", "hizoo")
MODELS = ("roberta-large", "opt-1.3b", "mistral-7b")
PRECISIONS = ("fp16", "int8")
TASKS = ("MNLI", "SST-5", "BoolQ")


@dataclass(frozen=True)
class Case:
    model: str
    task: str
    method: str
    precision: str

    @property
    def key(self) -> str:
        return "|".join((self.model, self.task, self.method, self.precision))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Directory that will store per-run artifacts and the matrix summary.",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=list(MODELS),
        choices=list(MODELS),
        help="Subset of models to benchmark.",
    )
    parser.add_argument(
        "--methods",
        nargs="*",
        default=list(METHODS),
        choices=list(METHODS),
        help="Subset of methods to benchmark.",
    )
    parser.add_argument(
        "--precisions",
        nargs="*",
        default=list(PRECISIONS),
        choices=list(PRECISIONS),
        help="Subset of precision modes to benchmark.",
    )
    parser.add_argument(
        "--tasks",
        nargs="*",
        default=list(TASKS),
        choices=list(TASKS),
        help="Subset of tasks to benchmark.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rerun cases even if a completed row already exists in summary.jsonl.",
    )
    return parser.parse_args()


def require_path(path: Path, kind: str) -> None:
    if kind == "file" and not path.is_file():
        raise FileNotFoundError(f"missing required file: {path}")
    if kind == "dir" and not path.is_dir():
        raise FileNotFoundError(f"missing required directory: {path}")


def validate_repo_layout() -> None:
    require_path(REPO_ROOT, "dir")
    require_path(REPO_ROOT / "medium_models", "dir")
    require_path(REPO_ROOT / "medium_models" / "mezo.sh", "file")
    require_path(REPO_ROOT / "large_models", "dir")
    require_path(REPO_ROOT / "large_models" / "run.py", "file")


def load_existing_rows(summary_path: Path) -> Dict[str, Dict]:
    rows: Dict[str, Dict] = {}
    if not summary_path.exists():
        return rows
    with summary_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            key = row.get("case_key")
            if key:
                rows[key] = row
    return rows


def append_row(summary_path: Path, row: Dict) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")


def iter_cases(args: argparse.Namespace) -> Iterable[Case]:
    for model in args.models:
        for task in args.tasks:
            for method in args.methods:
                for precision in args.precisions:
                    yield Case(model=model, task=task, method=method, precision=precision)


def roberta_supported(case: Case) -> Optional[str]:
    if case.precision == "int8" and case.method in {"lozo", "hizoo"}:
        return (
            "medium_models has no separate load_int8 path, and lozo/hizoo are "
            "explicitly incompatible with QuZO low-bit perturbations"
        )
    return None


def support_reason(case: Case) -> Optional[str]:
    if case.model == "roberta-large":
        return roberta_supported(case)
    return None


def case_env(case: Case) -> str:
    if case.model == "roberta-large":
        return "ciao"
    if case.model == "mistral-7b":
        return "mezo-mistral"
    return "mezo-env"


def medium_task_cli(task: str) -> str:
    return task


def large_task_cli(task: str) -> str:
    if task == "SST-5":
        return "SST5"
    return task


def case_output_root(output_root: Path, case: Case) -> Path:
    return output_root / case.model / case.task / case.precision / case.method


def medium_run_summary_path(case_dir: Path) -> Path:
    return case_dir / "run" / "seed16" / "run_summary.json"


def large_run_summary_path(case_dir: Path) -> Path:
    return case_dir / "run_summary.json"


def common_sparse_args(include_refresh: bool) -> List[str]:
    args = [
        "--sparse_ratio",
        "0.25",
        "--sparse_mask_strategy",
        "percentile_per_layer",
        "--sparse_scope",
        "trainable_only",
        "--sparse_log_active_fraction",
        "True",
    ]
    if include_refresh:
        args.extend(["--sparse_mask_refresh_steps", "0"])
    return args


def shell_join(parts: List[str]) -> str:
    return " ".join(shlex.quote(part) for part in parts)


def build_medium_command(case: Case, case_dir: Path) -> str:
    efficient_zero_order = "False" if case.method in {"lozo", "hizoo"} else "True"
    args: List[str] = [
        "--result_root",
        str(case_dir),
        "--job_name",
        "run",
        "--measure_perf_tail_window_steps",
        "3",
        "--zo_probe_every",
        "0",
        "--zo_method",
        case.method,
    ]
    if case.precision == "fp16":
        args.extend(["--zo_two_point_precision", "fp16"])
        if case.method in {"lozo", "hizoo"}:
            args.extend(["--zo_quantization_bits", "32"])
        else:
            args.extend(["--zo_quantization_bits", "16"])
    else:
        args.extend(["--zo_two_point_precision", "fp16", "--zo_quantization", "int8"])
    if case.method == "sparse_mezo":
        args.extend(common_sparse_args(include_refresh=True))

    env_bits = [
        "TASK=" + medium_task_cli(case.task),
        "K=16",
        "SEED=16",
        "DATA_SEED=16",
        "DATASET_MODE=full",
        "FULL_DEV_RATIO=0.1",
        "BS=32",
        "LR=1e-6",
        "WD=0",
        "STEP=5",
        "EVAL_STEP=5000",
        "MODEL=roberta-large",
        "USE_H=False",
        "USE_C=False",
        "DATALOADER_SHUFFLE=False",
        "EPS=1e-4",
        f"EFFICIENT_ZERO_ORDER={efficient_zero_order}",
        f"EXTRA_TAG=zo-matrix-{case.method}-{case.precision}-{case.task.lower().replace('-', '')}",
    ]
    mezo_cmd = shell_join(args)
    return (
        f"cd {shlex.quote(str(REPO_ROOT / 'medium_models'))} && "
        + " ".join(env_bits)
        + f" bash ./mezo.sh {mezo_cmd}"
    )


def build_large_command(case: Case, case_dir: Path) -> str:
    args: List[str] = [
        "python",
        "large_models/run.py",
        "--model_name",
        case.model,
        "--task_name",
        large_task_cli(case.task),
        "--dataset_mode",
        "full",
        "--num_k",
        "16",
        "--data_seed",
        "42",
        "--train_set_seed",
        "42",
        "--trainer",
        "zo",
        "--learning_rate",
        "1e-6",
        "--zo_eps",
        "1e-4",
        "--num_train_epochs",
        "1",
        "--max_steps",
        "4",
        "--per_device_train_batch_size",
        "16",
        "--gradient_accumulation_steps",
        "1",
        "--lr_scheduler_type",
        "constant",
        "--save_strategy",
        "no",
        "--no_eval",
        "--logging_steps",
        "1",
        "--zo_probe_every",
        "0",
        "--measure_perf_tail_window_steps",
        "2",
        "--output_dir",
        str(case_dir),
        "--overwrite_output_dir",
        "--tag",
        f"zo-matrix-{case.model}-{case.task}-{case.method}-{case.precision}",
        "--train_as_classification",
        "--zo_method",
        case.method,
    ]
    if case.precision == "fp16":
        args.extend(["--load_float16"])
        if case.method in {"lozo", "hizoo"}:
            args.extend(["--zo_quantization_bits", "32"])
        else:
            args.extend(["--zo_quantization_bits", "16"])
    else:
        args.extend(["--load_int8", "--zo_quantization_bits", "32"])
    if case.method == "sparse_mezo":
        args.extend(common_sparse_args(include_refresh=False))

    return f"cd {shlex.quote(str(REPO_ROOT))} && {shell_join(args)}"


def build_command(case: Case, case_dir: Path) -> str:
    if case.model == "roberta-large":
        return build_medium_command(case, case_dir)
    return build_large_command(case, case_dir)


def run_summary_path(case: Case, case_dir: Path) -> Path:
    if case.model == "roberta-large":
        return medium_run_summary_path(case_dir)
    return large_run_summary_path(case_dir)


def parse_run_summary(summary_path: Path) -> Dict:
    with summary_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    artifacts = data.get("artifacts", {})
    tail_perf = artifacts.get("tail_perf_metrics") or {}
    train_metrics = artifacts.get("train_metrics") or data.get("train") or {}
    sparse_stats = artifacts.get("sparse_mezo_last_stats") or {}
    return {
        "tail_perf_wallclock_per_step": tail_perf.get("tail_perf_wallclock_per_step"),
        "tail_perf_samples_per_second": tail_perf.get("tail_perf_samples_per_second"),
        "tail_perf_max_gpu_memory_gb": tail_perf.get("tail_perf_max_gpu_memory_gb"),
        "tail_perf_measured_steps": tail_perf.get("tail_perf_measured_steps"),
        "tail_perf_window_steps": tail_perf.get("tail_perf_window_steps"),
        "train_steps_per_second": train_metrics.get("train_steps_per_second"),
        "train_samples_per_second": train_metrics.get("train_samples_per_second"),
        "train_runtime": train_metrics.get("train_runtime"),
        "sparse_active_fraction": sparse_stats.get("active_fraction"),
    }


def parse_large_metrics_jsonl(case_dir: Path) -> Optional[Dict]:
    metric_files = sorted(case_dir.glob("metrics_*.jsonl"))
    if not metric_files:
        return None
    values: Dict[str, float] = {}
    for line in metric_files[0].read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        metric_name = row.get("metric")
        if metric_name:
            values[str(metric_name)] = row.get("value")
    if not values:
        return None
    return {
        "tail_perf_wallclock_per_step": values.get("tail_perf_wallclock_per_step"),
        "tail_perf_samples_per_second": values.get("tail_perf_samples_per_second"),
        "tail_perf_max_gpu_memory_gb": values.get("tail_perf_max_gpu_memory_gb"),
        "tail_perf_measured_steps": values.get("tail_perf_measured_steps"),
        "tail_perf_window_steps": values.get("tail_perf_window_steps"),
        "train_steps_per_second": values.get("train_steps_per_second"),
        "train_samples_per_second": values.get("train_samples_per_second"),
        "train_runtime": values.get("train_runtime"),
        "sparse_active_fraction": None,
        "metrics_jsonl_path": str(metric_files[0]),
    }


def collect_existing_case(case: Case, output_root: Path) -> Optional[Dict]:
    case_dir = case_output_root(output_root, case)
    summary_path = run_summary_path(case, case_dir)
    row = {
        "case_key": case.key,
        "model": case.model,
        "task": case.task,
        "method": case.method,
        "precision": case.precision,
        "env": case_env(case),
        "output_dir": str(case_dir),
        "command_path": str(case_dir / "command.sh"),
        "log_path": str(case_dir / "combined.log"),
        "run_summary_path": str(summary_path),
    }
    if summary_path.exists():
        row["status"] = "completed"
        row["exit_code"] = 0
        row.update(parse_run_summary(summary_path))
        return row
    if case.model != "roberta-large":
        metrics = parse_large_metrics_jsonl(case_dir)
        if metrics is not None:
            row["status"] = "completed"
            row["exit_code"] = 0
            row.update(metrics)
            return row
    return None


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def run_case(case: Case, output_root: Path) -> Dict:
    env_name = case_env(case)
    case_dir = case_output_root(output_root, case)
    command = build_command(case, case_dir)
    summary_path = run_summary_path(case, case_dir)
    log_path = case_dir / "combined.log"
    cmd_path = case_dir / "command.sh"
    write_text(cmd_path, command + "\n")

    started = time.time()
    completed = subprocess.run(
        ["conda", "run", "-n", env_name, "bash", "-c", command],
        cwd=str(REPO_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    wall_seconds = time.time() - started
    write_text(log_path, completed.stdout or "")

    row = {
        "case_key": case.key,
        "model": case.model,
        "task": case.task,
        "method": case.method,
        "precision": case.precision,
        "env": env_name,
        "output_dir": str(case_dir),
        "command_path": str(cmd_path),
        "log_path": str(log_path),
        "run_summary_path": str(summary_path),
        "wall_seconds": wall_seconds,
        "exit_code": completed.returncode,
    }
    if completed.returncode != 0:
        row["status"] = "error"
        row["error_tail"] = (completed.stdout or "")[-4000:]
        return row
    harvested = collect_existing_case(case, output_root)
    if harvested is not None:
        harvested["wall_seconds"] = wall_seconds
        return harvested
    row["status"] = "missing_run_summary"
    row["error_tail"] = (completed.stdout or "")[-4000:]
    return row


def main() -> int:
    validate_repo_layout()
    args = parse_args()
    output_root = args.output_root.resolve()
    summary_path = output_root / SUMMARY_NAME
    existing = load_existing_rows(summary_path)

    for case in iter_cases(args):
        existing_row = existing.get(case.key)
        if existing_row and existing_row.get("status") == "completed" and not args.force:
            print(f"[skip] completed {case.key}", flush=True)
            continue
        if not args.force:
            harvested = collect_existing_case(case, output_root)
            if harvested is not None:
                append_row(summary_path, harvested)
                existing[case.key] = harvested
                print(f"[recover] completed {case.key}", flush=True)
                continue
        unsupported_reason = support_reason(case)
        if unsupported_reason:
            row = {
                "case_key": case.key,
                "model": case.model,
                "task": case.task,
                "method": case.method,
                "precision": case.precision,
                "status": "unsupported",
                "reason": unsupported_reason,
            }
            append_row(summary_path, row)
            existing[case.key] = row
            print(f"[unsupported] {case.key}: {unsupported_reason}", flush=True)
            continue

        print(f"[run] {case.key}", flush=True)
        row = run_case(case, output_root)
        append_row(summary_path, row)
        existing[case.key] = row
        print(
            f"[done] {case.key} status={row['status']} "
            f"tail_sps={row.get('tail_perf_samples_per_second')}",
            flush=True,
        )
    print(f"[summary] {summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
