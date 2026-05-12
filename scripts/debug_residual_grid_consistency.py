#!/usr/bin/env python3
"""Run the residual_grid consistency diagnostic through the project's MeZO entrypoint."""

from __future__ import annotations

import argparse
import os
import subprocess
from datetime import datetime
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    repo_root = Path(__file__).resolve().parents[1]
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    parser.add_argument("--repo-root", default=str(repo_root))
    parser.add_argument("--output-root", default=str(repo_root / "runs" / f"int8_residual_consistency_{ts}"))
    parser.add_argument("--debug-save-dir", default="")
    parser.add_argument("--cuda-visible-devices", default=os.environ.get("CUDA_VISIBLE_DEVICES", "0"))
    parser.add_argument("--layer-regex", default="")
    parser.add_argument("--debug-num-tensors", type=int, default=8)
    parser.add_argument("--debug-dump-tensor-stats", action="store_true")
    parser.add_argument("--int8-scale-floor", type=float, default=0.0)
    parser.add_argument("extra_args", nargs=argparse.REMAINDER)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    run_root = Path(args.output_root).resolve()
    debug_save_dir = Path(args.debug_save_dir).resolve() if args.debug_save_dir else run_root / "debug"
    run_root.mkdir(parents=True, exist_ok=True)
    (run_root / "logs").mkdir(parents=True, exist_ok=True)
    debug_save_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.cuda_visible_devices)
    env.update(
        {
            "TASK": "SST-5",
            "K": "16",
            "SEED": "16",
            "DATA_SEED": "16",
            "DATASET_MODE": "full",
            "FULL_DEV_RATIO": "0.1",
            "BS": "64",
            "LR": "1e-5",
            "EPS": "3e-3",
            "WD": "0",
            "STEP": "1",
            "EVAL_STEP": "100000",
            "MODEL": "roberta-large",
            "USE_H": "False",
            "USE_C": "False",
            "DATALOADER_SHUFFLE": "False",
            "EFFICIENT_ZERO_ORDER": "True",
            "EXTRA_TAG": "int8-residual-debug",
        }
    )
    command = [
        "bash",
        "./mezo.sh",
        "--result_root",
        str(run_root),
        "--job_name",
        "residual_grid_consistency_debug",
        "--dataset_mode",
        "full",
        "--zo_quantization",
        "int8",
        "--zo_two_point_precision",
        "fp16",
        "--zo_h",
        "3e-3",
        "--zo_update_backend",
        "residual_grid",
        "--residual_dtype",
        "fp32",
        "--residual_commit_mode",
        "round",
        "--residual_max_code_step",
        "0",
        "--int8_freeze_scale",
        "True",
        "--int8_scale_floor",
        str(args.int8_scale_floor),
        "--zo_update_norm_clip",
        "0",
        "--log_update_stats_every",
        "1",
        "--save_update_stats_jsonl",
        "update_stats.jsonl",
        "--zo_probe_every",
        "0",
        "--debug_residual_grid_consistency",
        "True",
        "--debug_save_dir",
        str(debug_save_dir),
        "--debug_num_tensors",
        str(args.debug_num_tensors),
        "--random_prediction_guard_enabled",
        "False",
        "--save_strategy",
        "no",
        "--no_predict",
    ]
    if args.layer_regex:
        command.extend(["--debug_layer_regex", args.layer_regex])
    if args.debug_dump_tensor_stats:
        command.extend(["--debug_dump_tensor_stats", "True"])
    command.extend(args.extra_args)

    manifest = run_root / "debug_command.txt"
    manifest.write_text(
        " ".join(command) + "\n"
        + f"CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']}\n"
        + f"debug_save_dir={debug_save_dir}\n",
        encoding="utf-8",
    )
    print(f"RUN_ROOT={run_root}")
    print(f"DEBUG_SAVE_DIR={debug_save_dir}")
    print("COMMAND=" + " ".join(command))
    return subprocess.call(command, cwd=str(repo_root / "medium_models"), env=env)


if __name__ == "__main__":
    raise SystemExit(main())
