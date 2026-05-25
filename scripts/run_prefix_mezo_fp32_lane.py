#!/usr/bin/env python
"""Run one lane of original FP32 Prefix-MeZO full-data jobs."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List


REPO_ROOT = Path(__file__).resolve().parents[1]
TASK_SHELL = {
    "sst-2": "SST-2",
    "sst-5": "SST-5",
    "rte": "RTE",
    "mnli": "MNLI",
    "trec": "trec",
}


def read_rows(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_json(path: Path, data: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def parse_float(row: Dict[str, str], key: str, default: float) -> float:
    try:
        value = row.get(key, "")
        return float(value) if value not in ("", None) else float(default)
    except Exception:
        return float(default)


def parse_int(row: Dict[str, str], key: str, default: int) -> int:
    try:
        value = row.get(key, "")
        return int(float(value)) if value not in ("", None) else int(default)
    except Exception:
        return int(default)


def run_complete(run_dir: Path, target_steps: int) -> bool:
    summary = run_dir / "run_summary.json"
    if not summary.exists():
        return False
    try:
        data = json.loads(summary.read_text(encoding="utf-8"))
    except Exception:
        return False
    step_candidates = [
        data.get("global_step"),
        data.get("steps_completed"),
        data.get("max_steps"),
    ]
    for value in step_candidates:
        try:
            if value is not None and int(float(value)) >= int(target_steps):
                return True
        except Exception:
            pass
    return False


def copy_if_exists(src: Path, dst: Path) -> None:
    if src.exists() and src.is_file():
        shutil.copy2(src, dst)


def command_for(row: Dict[str, str], output_root: Path) -> List[str]:
    result_root = output_root.resolve() / "results"
    return [
        "bash",
        "mezo.sh",
        "--result_root",
        str(result_root),
        "--job_name",
        row["run_name"],
        "--prefix_tuning",
        "--num_prefix",
        str(parse_int(row, "num_prefix", 5)),
        "--no_reparam",
        "--prefix_init_by_real_act",
        "--precision_mode",
        "fp32",
        "--zo_quantization",
        "fp32",
        "--zo_quantization_bits",
        "32",
        "--zo_two_point_precision",
        "fp32",
        "--main_save_checkpoints",
        "--main_checkpoint_steps",
        str(parse_int(row, "checkpoint_steps", 1000)),
        "--main_save_final_checkpoint",
        "--main_save_best_acc_checkpoint",
        "--main_save_best_loss_checkpoint",
        "--save_at_last",
    ]


def run_row(row: Dict[str, str], output_root: Path) -> int:
    task = row["task_name"].strip().lower()
    run_name = row["run_name"]
    max_steps = parse_int(row, "max_steps", 20000)
    run_dir = output_root.resolve() / "results" / run_name / "seed16"
    run_dir.mkdir(parents=True, exist_ok=True)

    manifest_row = dict(row)
    manifest_row.update(
        {
            "run_dir": str(run_dir),
            "actual_forward_path": "original_medium_models_mezo_fp32",
            "precision_mode": "fp32",
            "zo_quantization": "fp32",
            "zo_quantization_bits": 32,
            "zo_two_point_precision": "fp32",
            "model_storage_fp16_expected": False,
            "quantizer": "none",
            "direct_int_update": False,
            "residual_grid": False,
            "sparse": False,
            "prefix_tuning": True,
            "prefix_init_by_real_act": True,
            "no_reparam": True,
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }
    )
    write_json(run_dir / "run_manifest_row.json", manifest_row)
    write_json(run_dir / "run_config.json", manifest_row)

    cmd = command_for(row, output_root)
    resume_env = (
        f"TASK={TASK_SHELL[task]} K=16 SEED=16 DATA_SEED=16 DATASET_MODE=full "
        f"DATALOADER_SHUFFLE=True BS=64 LR={row['lr']} EPS={row['h']} "
        f"STEP={max_steps} EVAL_STEP={row['eval_every']} MODEL=roberta-large "
        f"USE_H=False USE_C=False KEEP_CHECKPOINTS=True "
        f"RESULT_ROOT={output_root.resolve() / 'results'} JOB_NAME={run_name}"
    )
    (run_dir / "resume_command.txt").write_text(
        f"cd {REPO_ROOT / 'medium_models'} && {resume_env} {' '.join(cmd)}\n",
        encoding="utf-8",
    )

    if run_complete(run_dir, max_steps):
        with (run_dir / "train.log").open("a", encoding="utf-8") as log:
            log.write(f"[{datetime.now().isoformat(timespec='seconds')}] skip complete {run_name}\n")
        return 0

    env = os.environ.copy()
    env.update(
        {
            "TASK": TASK_SHELL[task],
            "K": "16",
            "SEED": "16",
            "DATA_SEED": "16",
            "DATASET_MODE": "full",
            "DATALOADER_SHUFFLE": "True",
            "BS": str(parse_int(row, "batch_size", 64)),
            "LR": str(parse_float(row, "lr", 1e-2)),
            "EPS": str(parse_float(row, "h", 1e-3)),
            "STEP": str(max_steps),
            "EVAL_STEP": str(parse_int(row, "eval_every", 1000)),
            "MODEL": "roberta-large",
            "USE_H": "False",
            "USE_C": "False",
            "KEEP_CHECKPOINTS": "True",
            "ZERO_ORDER_USE_TRAINER_OPTIM": "False",
            "EFFICIENT_ZERO_ORDER": "True",
            "OPT": "sgd",
            "RESULT_ROOT": str(output_root.resolve() / "results"),
            "JOB_NAME": run_name,
        }
    )
    log_path = run_dir / "train.log"
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"[{datetime.now().isoformat(timespec='seconds')}] START {run_name}\n")
        log.write(f"COMMAND: {' '.join(cmd)}\n")
        log.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=str(REPO_ROOT / "medium_models"),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            log.write(line)
        code = proc.wait()
        log.write(f"[{datetime.now().isoformat(timespec='seconds')}] END {run_name} exit_code={code}\n")

    metrics_dir = run_dir / "metrics_logs"
    copy_if_exists(metrics_dir / "training_metrics.csv", run_dir / "metrics.csv")
    copy_if_exists(metrics_dir / "eval_metrics.jsonl", run_dir / "eval_metrics.jsonl")
    if code != 0 and not (run_dir / "run_summary.json").exists():
        write_json(
            run_dir / "run_summary.json",
            {
                **manifest_row,
                "status": "failed",
                "exit_code": code,
                "steps_completed": 0,
                "error_message": f"medium_models/mezo.sh exited {code}",
            },
        )
    return code


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()

    manifest = Path(args.manifest)
    output_root = Path(args.output_root)
    rows = read_rows(manifest)
    status = 0
    for row in rows:
        status = max(status, run_row(row, output_root))
    return status


if __name__ == "__main__":
    raise SystemExit(main())
