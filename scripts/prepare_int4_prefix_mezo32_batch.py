#!/usr/bin/env python
"""Prepare full-data INT4-base Prefix-MeZO FP32-prefix training manifests."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, List, Sequence


TASKS = ["sst-2", "sst-5", "rte", "mnli", "trec"]
POLICIES = [
    ("fixed_small", 1e-5, "1e-5"),
    ("mezo_default", 1e-3, "1e-3"),
    ("hstar_ours", None, None),
]


def read_rows(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: List[Dict[str, object]], columns: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def columns(rows: List[Dict[str, object]]) -> List[str]:
    keys: List[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    return keys


def split_lanes(rows: List[Dict[str, object]], n_lanes: int) -> List[List[Dict[str, object]]]:
    lanes = [[] for _ in range(n_lanes)]
    for idx, row in enumerate(rows):
        lanes[idx % n_lanes].append(row)
    return lanes


def h_label_from_value(h: float) -> str:
    known = {1e-5: "1e-5", 1e-3: "1e-3"}
    for value, label in known.items():
        if abs(float(h) - value) <= max(abs(value) * 1e-9, 1e-15):
            return label
    return f"{h:.10g}".replace(".", "p").replace("-", "m")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--hstar_summary", required=True)
    parser.add_argument("--lanes", type=int, default=3)
    args = parser.parse_args()

    output_root = Path(args.output_root)
    hstar_summary = Path(args.hstar_summary)
    hstar_rows = {row["task_name"]: row for row in read_rows(hstar_summary) if row.get("direction_mode") == "prefix"}
    rows: List[Dict[str, object]] = []
    for task in TASKS:
        if task not in hstar_rows:
            raise RuntimeError(f"Missing prefix h-star row for {task} in {hstar_summary}")
        src = hstar_rows[task]
        hstar = float(src["hstar_cont"])
        if not math.isfinite(hstar) or hstar <= 0.0:
            raise RuntimeError(f"Invalid hstar_cont={src.get('hstar_cont')!r} for {task}")
        for policy, fixed_h, fixed_label in POLICIES:
            if policy == "hstar_ours":
                h = hstar
                h_label = h_label_from_value(h)
            else:
                h = float(fixed_h)
                h_label = str(fixed_label)
            run_name = f"int4_prefix_mezo32_{task.replace('-', '')}_{policy}_h{h_label}_seed16_full_bs64_step20k"
            rows.append(
                {
                    "run_name": run_name,
                    "run_dir": str(output_root / "int4_hsearch" / "prefix_mezo32" / run_name),
                    "phase": "int4_prefix_mezo32_full_data_20k",
                    "task_name": task,
                    "dataset": task,
                    "dataset_mode": "full",
                    "data_dir": "",
                    "num_k": 16,
                    "seed": 16,
                    "data_seed": 16,
                    "batch_size": 64,
                    "bitwidth": 4,
                    "h": h,
                    "h_label": h_label,
                    "h_policy": policy,
                    "max_steps": 20000,
                    "scale_refresh_k": 1,
                    "lr": 1e-6,
                    "eval_every": 1000,
                    "checkpoint_steps": 1000,
                    "eval_batch_size": 64,
                    "eval_batches": -1,
                    "diag_every": 100,
                    "quant_log_every": 1000,
                    "log_every": 100,
                    "direction_mode": "prefix",
                    "prefix_num": 5,
                    "prefix_precision": "fp32",
                    "prefix_init_strategy": "real_act_with_random_fallback",
                    "master_dtype": "fp32",
                    "hstar_source": str(hstar_summary),
                    "selector_name": src.get("selector_name", "simple2pt_corrected"),
                    "hstar_cont": src.get("hstar_cont", ""),
                    "hstar_nearest_grid": src.get("hstar_nearest_grid", ""),
                    "hstar_used": h if policy == "hstar_ours" else "",
                    "hstar_used_policy": "continuous_unsnapped_prefix_mezo32" if policy == "hstar_ours" else "",
                    "Delta_mode": src.get("Delta_mode", ""),
                    "Delta_value": src.get("Delta_value", ""),
                    "G_mode": src.get("G_mode", ""),
                    "G_value": src.get("G_value", ""),
                    "L_mode": src.get("L_mode", ""),
                    "L_hat": src.get("L_hat", ""),
                    "L_lowbit_q90": src.get("L_lowbit_q90", ""),
                    "h2_L": src.get("h2_L", ""),
                    "d_trainable": src.get("d_trainable", ""),
                    "perturbed_parameter_scope": "prefix_parameters_only",
                    "quantized_forward_scope": "base_Linear.weight_only_prefix_fp32",
                    "notes": (
                        "MeZO-style prefix: num_prefix=5, no_reparam, real-act init with random fallback; "
                        "full data; prefix/master parameters FP32; frozen base Linear weights use INT4 RTNClip forward."
                    ),
                }
            )

    cols = columns(rows)
    write_csv(output_root / "manifests" / "prefix_mezo32_manifest.csv", rows, cols)
    for idx, lane in enumerate(split_lanes(rows, int(args.lanes))):
        write_csv(output_root / "manifests" / f"prefix_mezo32_lane{idx}.csv", lane, cols)
    print(f"Wrote {len(rows)} prefix rows to {output_root} across {args.lanes} lanes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
