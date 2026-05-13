#!/usr/bin/env python3
"""Launch one MeZO probe-window diagnostic run through the existing medium_models entrypoint."""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path


def str_bool(value: bool) -> str:
    return "True" if bool(value) else "False"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", default="roberta-large")
    parser.add_argument("--task_name", default="SST-5")
    parser.add_argument("--precision_mode", choices=["fp32", "fp16", "bf16", "int8"], required=True)
    parser.add_argument("--quant_bits", type=int, default=None)
    parser.add_argument("--zo_h", type=float, default=3e-3)
    parser.add_argument("--h_list", default="")
    parser.add_argument("--num_probe_directions", type=int, default=50)
    parser.add_argument("--num_probe_batches", type=int, default=1)
    parser.add_argument("--direction_type", choices=["dense", "sparse"], default="dense")
    parser.add_argument("--sparse_rate", type=float, default=1.0)
    parser.add_argument("--sparse_mode", choices=["none", "exact_random", "bernoulli"], default="none")
    parser.add_argument("--sparse_rescale", choices=["none", "inv_sqrt_p"], default="none")
    parser.add_argument("--compute_true_grad_directional", type=lambda x: str(x).lower() in {"1", "true", "yes", "y"}, default=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--seed", type=int, default=16)
    parser.add_argument("--data_seed", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--conda_env", default=os.environ.get("CONDA_ENV", "ciao"))
    parser.add_argument("--cuda_visible_devices", default=os.environ.get("CUDA_VISIBLE_DEVICES", "0"))
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    medium_dir = repo_root / "medium_models"
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    h_list = args.h_list.strip() or f"{args.zo_h:.17g}"
    sparse_mode = args.sparse_mode
    sparse_rate = args.sparse_rate
    sparse_rescale = args.sparse_rescale
    if args.direction_type == "dense":
        sparse_mode = "none"
        sparse_rate = 1.0
        sparse_rescale = "none"
    elif sparse_mode == "none":
        sparse_mode = "bernoulli"

    extra_args = [
        "--output_dir",
        str(output_dir),
        "--overwrite_output_dir",
        "--probe_window_diagnostics_only",
        "True",
        "--probe_window_h_list",
        h_list,
        "--precision_mode",
        args.precision_mode,
        "--zo_h",
        f"{args.zo_h:.17g}",
        "--num_probe_directions",
        str(args.num_probe_directions),
        "--num_probe_batches",
        str(args.num_probe_batches),
        "--direction_type",
        args.direction_type,
        "--sparse_rate",
        f"{sparse_rate:.17g}",
        "--sparse_mode",
        sparse_mode,
        "--sparse_rescale",
        sparse_rescale,
        "--compute_true_grad_directional",
        str_bool(args.compute_true_grad_directional),
        "--save_probe_stats_jsonl",
        "probe_stats.jsonl",
        "--random_prediction_guard_enabled",
        "False",
        "--save_strategy",
        "no",
        "--no_predict",
    ]
    if args.quant_bits is not None:
        extra_args.extend(["--quant_bits", str(args.quant_bits)])
    if args.precision_mode == "int8":
        extra_args.extend(["--zo_update_backend", "fp16_master"])

    env = os.environ.copy()
    env.update(
        {
            "TASK": args.task_name,
            "K": "16",
            "SEED": str(args.seed),
            "DATA_SEED": str(args.data_seed if args.data_seed is not None else args.seed),
            "DATASET_MODE": "full",
            "FULL_DEV_RATIO": "0.1",
            "BS": str(args.batch_size),
            "LR": "0",
            "EPS": f"{args.zo_h:.17g}",
            "WD": "0",
            "STEP": "1",
            "EVAL_STEP": "100000",
            "MODEL": args.model_name_or_path,
            "USE_H": "False",
            "USE_C": "False",
            "DATALOADER_SHUFFLE": "False",
            "EFFICIENT_ZERO_ORDER": "True",
            "EXTRA_TAG": "probe-window",
            "CUDA_VISIBLE_DEVICES": args.cuda_visible_devices,
        }
    )

    cmd = [
        "bash",
        "./mezo.sh",
        "--result_root",
        str(output_dir.parent),
        "--job_name",
        output_dir.name,
        "--dataset_mode",
        "full",
        *extra_args,
    ]
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(medium_dir), env=env, check=True)


if __name__ == "__main__":
    main()
