#!/usr/bin/env python
"""Standard dense INT4 RTNClip preliminary h screen.

This wrapper intentionally runs only the standard two-point MeZO update through
the shared-grid RTNClip fake-quantized forward oracle implemented in
``rtnclip_roberta_sst5_batch.py``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Sequence, Tuple

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = REPO_ROOT / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import rtnclip_roberta_sst5_batch as batch  # noqa: E402
import smoke_rtnclip_roberta_sst5 as smoke  # noqa: E402


REQUIRED_H_GRID: List[Tuple[str, float]] = [
    ("2e-4", 2e-4),
    ("3e-4", 3e-4),
    ("5e-4", 5e-4),
    ("7e-4", 7e-4),
    ("1e-3", 1e-3),
    ("1p5e-3", 1.5e-3),
    ("2e-3", 2e-3),
]

ANCHOR_H_GRID: List[Tuple[str, float]] = [
    ("3e-3", 3e-3),
    ("5e-3", 5e-3),
]

SUMMARY_COLUMNS = [
    "run_name",
    "h_label",
    "h",
    "target_steps",
    "status",
    "steps_completed",
    "best_eval_acc",
    "best_eval_step",
    "last_eval_acc",
    "last_eval_step",
    "best_eval_loss",
    "last_eval_loss",
    "final_train_loss",
    "d_h_finite_rate",
    "active_frac",
    "code_change_frac",
    "alignment",
    "norm_ratio",
    "delta_q_norm",
    "ideal_displacement_norm",
    "clip_frac",
    "saturation_frac_w",
    "saturation_frac_w_plus",
    "saturation_frac_w_minus",
    "alpha_lt_1_frac",
    "update_norm_last",
    "nan_flag",
    "seconds_per_step",
    "peak_gpu_mem",
    "perturbed_parameter_scope",
    "quantized_forward_scope",
    "update_variant",
    "run_dir",
    "resume_command",
    "selection_status",
    "notes",
]


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def append_line(path: Path, line: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(line.rstrip() + "\n")


def write_csv(path: Path, rows: Sequence[Dict[str, object]], columns: Sequence[str] = SUMMARY_COLUMNS) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def command_output(cmd: Sequence[str]) -> str:
    try:
        return subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT).strip()
    except Exception as exc:
        return f"unavailable: {exc}"


def default_output_root() -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return REPO_ROOT / "outputs" / f"rtnclip_int4_standard_screen_seed16_{stamp}"


def make_train_args(args, output_root: Path) -> SimpleNamespace:
    return SimpleNamespace(
        output_root=str(output_root),
        lr=float(args.lr),
        eval_every=int(args.eval_every),
        checkpoint_steps=int(args.checkpoint_steps),
        eval_batch_size=int(args.eval_batch_size),
        eval_batches=int(args.eval_batches),
        diag_every=int(args.diag_every),
        quant_log_every=int(args.quant_log_every),
        log_every=int(args.log_every),
    )


def h_run_name(h_label: str) -> str:
    return f"int4_standard_g128_rtnclip_h{h_label}_seed16_bs64"


def run_dir_for(root: Path, h_label: str) -> Path:
    return root / "phase1_1k_runs" / h_run_name(h_label)


def run_h(root: Path, train_args: SimpleNamespace, h_label: str, h: float, target_steps: int) -> Dict[str, object]:
    run_name = h_run_name(h_label)
    run_dir = run_dir_for(root, h_label)
    summary = batch.train_one(
        train_args,
        run_dir,
        run_name,
        4,
        h,
        h_label,
        int(target_steps),
        1,
        "int4_standard_screen",
        None,
    )
    summary["target_steps"] = int(target_steps)
    summary["run_dir"] = str(run_dir)
    resume = run_dir / "resume_command.txt"
    summary["resume_command"] = resume.read_text(encoding="utf-8").strip() if resume.exists() else ""
    return summary


def row_from_summary(summary: Dict[str, object], target_steps: int, selection_status: str = "", notes: str = "") -> Dict[str, object]:
    row = dict(summary)
    row["target_steps"] = int(target_steps)
    row["nan_flag"] = row.get("status") == "failed"
    row["selection_status"] = selection_status
    row["notes"] = notes
    return row


def is_finite_number(value) -> bool:
    try:
        return math.isfinite(float(value))
    except Exception:
        return False


def passes_1k_rule(row: Dict[str, object]) -> Tuple[bool, str]:
    if row.get("status") != "complete":
        return False, "not complete"
    if int(row.get("steps_completed", 0) or 0) < 1000:
        return False, "under 1k steps"
    best = row.get("best_eval_acc")
    last = row.get("last_eval_acc")
    active = row.get("active_frac")
    loss = row.get("final_train_loss")
    if not is_finite_number(best) or float(best) < 0.30:
        return False, "best_eval_acc < 0.30"
    if not is_finite_number(last) or float(last) < float(best) - 0.06:
        return False, "last_eval_acc collapsed"
    if not is_finite_number(active) or float(active) < 1e-3:
        return False, "active_frac near zero"
    if not is_finite_number(loss) or float(loss) > 20.0:
        return False, "loss exploded"
    return True, "kept"


def passes_stability_rule(row: Dict[str, object], target_steps: int) -> Tuple[bool, str]:
    if row.get("status") != "complete":
        return False, "not complete"
    if int(row.get("steps_completed", 0) or 0) < int(target_steps):
        return False, f"under {target_steps} steps"
    best = row.get("best_eval_acc")
    last = row.get("last_eval_acc")
    loss = row.get("final_train_loss")
    active = row.get("active_frac")
    if not is_finite_number(best) or not is_finite_number(last):
        return False, "missing eval"
    if float(best) < 0.30:
        return False, "best_eval_acc < 0.30"
    if float(last) < float(best) - 0.06:
        return False, "last_eval_acc collapsed"
    if not is_finite_number(active) or float(active) < 1e-3:
        return False, "active_frac near zero"
    if not is_finite_number(loss) or float(loss) > 20.0:
        return False, "loss exploded"
    return True, "stable"


def choose_2k_candidates(rows: List[Dict[str, object]], h_grid: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
    kept = []
    for row in rows:
        ok, reason = passes_1k_rule(row)
        row["selection_status"] = "keep" if ok else "drop"
        row["notes"] = reason
        if ok:
            kept.append(row)
    if not kept:
        return []
    kept_sorted = sorted(kept, key=lambda r: (float(r["best_eval_acc"]), float(r.get("last_eval_acc") or -1)), reverse=True)
    selected_labels = [str(r["h_label"]) for r in kept_sorted[:3]]
    best_label = str(kept_sorted[0]["h_label"])
    labels = [label for label, _ in h_grid]
    if best_label in labels:
        idx = labels.index(best_label)
        neighbor_labels = []
        if idx > 0:
            neighbor_labels.append(labels[idx - 1])
        if idx + 1 < len(labels):
            neighbor_labels.append(labels[idx + 1])
        by_label = {str(r["h_label"]): r for r in rows}
        neighbor_labels = sorted(
            [label for label in neighbor_labels if label not in selected_labels and label in by_label],
            key=lambda label: float(by_label[label].get("best_eval_acc") or -1.0),
            reverse=True,
        )
        if neighbor_labels:
            selected_labels.append(neighbor_labels[0])
    h_by_label = dict(h_grid)
    return [(label, h_by_label[label]) for label in selected_labels if label in h_by_label]


def write_screen_md(path: Path, title: str, rows: List[Dict[str, object]], target_steps: int) -> None:
    lines = [
        f"# {title}",
        "",
        f"Target steps: `{target_steps}`",
        "",
        "| h | status | steps | best_acc | last_acc | loss | active_frac | alignment | norm_ratio | selection | notes |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for row in sorted(rows, key=lambda r: float(r.get("h") or 0.0)):
        lines.append(
            f"| {row.get('h_label')} | {row.get('status')} | {row.get('steps_completed')} | "
            f"{row.get('best_eval_acc')} | {row.get('last_eval_acc')} | {row.get('final_train_loss')} | "
            f"{row.get('active_frac')} | {row.get('alignment')} | {row.get('norm_ratio')} | "
            f"{row.get('selection_status', '')} | {row.get('notes', '')} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_recommendation(path: Path, rows_1k: List[Dict[str, object]], rows_ext: List[Dict[str, object]], selected_2k: List[Tuple[str, float]], selected_5k: List[Tuple[str, float]], h_grid: List[Tuple[str, float]]) -> None:
    stable_ext = [r for r in rows_ext if passes_stability_rule(r, int(r.get("target_steps", 2000)))[0]]
    stable_5k = [r for r in stable_ext if int(r.get("steps_completed", 0) or 0) >= 5000]
    best_pool = stable_5k or stable_ext or [r for r in rows_1k if passes_1k_rule(r)[0]]
    best = max(best_pool, key=lambda r: float(r.get("best_eval_acc") or -1.0)) if best_pool else None
    h5 = next((r for r in rows_ext if r.get("h_label") == "5e-4"), None) or next((r for r in rows_1k if r.get("h_label") == "5e-4"), None)
    h1e3 = next((r for r in rows_ext if r.get("h_label") == "1e-3"), None) or next((r for r in rows_1k if r.get("h_label") == "1e-3"), None)
    official_grid = ", ".join(label for label, _ in REQUIRED_H_GRID)
    if best is None:
        recommended_grid = official_grid
        launch = "no, current screen did not identify a stable candidate"
    else:
        labels = [label for label, _ in h_grid]
        best_label = str(best["h_label"])
        idx = labels.index(best_label) if best_label in labels else -1
        narrow = []
        for j in range(max(0, idx - 2), min(len(labels), idx + 3)):
            if labels[j] in {label for label, _ in REQUIRED_H_GRID}:
                narrow.append(labels[j])
        if best_label == "5e-4":
            recommended_grid = "3e-4, 5e-4, 7e-4, 1e-3"
        else:
            recommended_grid = ", ".join(narrow or [best_label])
        launch = "yes, use the narrow grid first" if stable_ext else "not yet, extend at least one candidate further"
    lines = [
        "# Recommended INT4 Standard H Sweep",
        "",
        f"1. Does standard INT4 continue training beyond 1k? {'yes' if stable_ext else 'not confirmed'}",
        f"2. Which h is most stable? `{best.get('h_label') if best else None}`",
        f"3. Is h=5e-4 still best? {'yes' if best and best.get('h_label') == '5e-4' else 'no'}"
        + (f" (h=5e-4 best_acc={h5.get('best_eval_acc')}, last_acc={h5.get('last_eval_acc')})" if h5 else ""),
        f"4. Does h=1e-3 collapse after 1k again? {'yes' if h1e3 and is_finite_number(h1e3.get('best_eval_acc')) and is_finite_number(h1e3.get('last_eval_acc')) and float(h1e3['last_eval_acc']) < float(h1e3['best_eval_acc']) - 0.06 else 'no hard collapse; monitor mild downward drift if present'}",
        f"5. Should we launch a 5k/10k narrow sweep? {launch}",
        f"6. Exact h grid to use: `{recommended_grid}`",
        "",
        f"Official preliminary grid was: `{official_grid}`.",
        f"2k candidates: `{', '.join(label for label, _ in selected_2k) or 'none'}`.",
        f"5k candidates: `{', '.join(label for label, _ in selected_5k) or 'none'}`.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args) -> None:
    os.environ["DATALOADER_SHUFFLE"] = "True"
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; this screen is intended for the local H100.")
    output_root = Path(args.output_root) if args.output_root else default_output_root()
    output_root.mkdir(parents=True, exist_ok=True)
    h_grid = list(REQUIRED_H_GRID)
    if args.include_anchors:
        h_grid.extend(ANCHOR_H_GRID)
    train_args = make_train_args(args, output_root)

    env = smoke.collect_env(REPO_ROOT)
    write_json(output_root / "env.json", env)
    write_json(
        output_root / "config_manifest.json",
        {
            "experiment": "int4_standard_dense_preliminary_h_screen",
            "model": "roberta-large",
            "dataset": "SST-5",
            "seed": 16,
            "data_seed": 16,
            "batch_size": 64,
            "shuffle": True,
            "sampler": "RandomSampler",
            "direction": "dense",
            "update_variant": "standard",
            "quantizer": "G128_groupwise_RTNClip_fake_quant",
            "bitwidth": 4,
            "group_size": 128,
            "scale_refresh_k": 1,
            "pair_shared_grid": True,
            "fresh_round_codes": True,
            "grid_source": "unperturbed_fp16_master_weight",
            "update_backend": "fp16_master",
            "master_dtype": "fp16",
            "perturbed_parameter_scope": "full_dense_all_trainable",
            "quantized_forward_scope": "Linear.weight_only",
            "h_grid": [{"h_label": label, "h": h} for label, h in h_grid],
            "checkpoint_steps": int(args.checkpoint_steps),
            "eval_every": int(args.eval_every),
            "lr": float(args.lr),
            "hostname": command_output(["hostname"]),
            "date": datetime.now().isoformat(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            "nvidia_smi": command_output(["nvidia-smi"]),
            "git_commit": command_output(["git", "rev-parse", "HEAD"]),
        },
    )
    append_line(output_root / "commands.txt", " ".join(sys.argv))

    manifest_rows = []
    for label, h in h_grid:
        manifest_rows.append({"h_label": label, "h": h, "run_name": h_run_name(label), "run_dir": str(run_dir_for(output_root, label)), "phase1_steps": 1000})
    write_csv(output_root / "run_manifest.csv", manifest_rows, ["h_label", "h", "run_name", "run_dir", "phase1_steps"])

    phase1_rows: List[Dict[str, object]] = []
    for label, h in h_grid:
        summary = run_h(output_root, train_args, label, h, 1000)
        row = row_from_summary(summary, 1000)
        phase1_rows.append(row)
        write_csv(output_root / "int4_standard_screen_1k_summary.csv", phase1_rows)
        write_screen_md(output_root / "int4_standard_screen_1k_summary.md", "INT4 Standard 1k Screen", phase1_rows, 1000)

    selected_2k = choose_2k_candidates(phase1_rows, h_grid)
    write_csv(output_root / "int4_standard_screen_1k_summary.csv", phase1_rows)
    write_screen_md(output_root / "int4_standard_screen_1k_summary.md", "INT4 Standard 1k Screen", phase1_rows, 1000)

    extend_rows: List[Dict[str, object]] = []
    for label, h in selected_2k:
        summary = run_h(output_root, train_args, label, h, 2000)
        ok, reason = passes_stability_rule(summary, 2000)
        row = row_from_summary(summary, 2000, "stable_2k" if ok else "drop_after_2k", reason)
        extend_rows.append(row)
        write_csv(output_root / "int4_standard_extend_summary.csv", extend_rows)
        write_screen_md(output_root / "int4_standard_extend_summary.md", "INT4 Standard Extension", extend_rows, 2000)

    write_csv(output_root / "int4_standard_extend_2k_snapshot.csv", extend_rows)
    selected_5k_rows = [r for r in extend_rows if passes_stability_rule(r, 2000)[0]]
    selected_5k_rows = sorted(selected_5k_rows, key=lambda r: (float(r.get("best_eval_acc") or -1.0), float(r.get("last_eval_acc") or -1.0)), reverse=True)[:2]
    selected_5k = [(str(r["h_label"]), float(r["h"])) for r in selected_5k_rows]
    for label, h in selected_5k:
        summary = run_h(output_root, train_args, label, h, 5000)
        ok, reason = passes_stability_rule(summary, 5000)
        row = row_from_summary(summary, 5000, "stable_5k" if ok else "drop_after_5k", reason)
        extend_rows = [r for r in extend_rows if str(r.get("h_label")) != label]
        extend_rows.append(row)
        write_csv(output_root / "int4_standard_extend_summary.csv", extend_rows)
        write_screen_md(output_root / "int4_standard_extend_summary.md", "INT4 Standard Extension", extend_rows, 5000)

    write_csv(output_root / "int4_standard_extend_summary.csv", extend_rows)
    write_screen_md(output_root / "int4_standard_extend_summary.md", "INT4 Standard Extension", extend_rows, 5000 if selected_5k else 2000)
    write_recommendation(output_root / "recommended_int4_hsweep.md", phase1_rows, extend_rows, selected_2k, selected_5k, h_grid)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_root", default="")
    parser.add_argument("--include_anchors", action="store_true")
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--eval_every", type=int, default=500)
    parser.add_argument("--checkpoint_steps", type=int, default=500)
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--eval_batches", type=int, default=-1)
    parser.add_argument("--diag_every", type=int, default=100)
    parser.add_argument("--quant_log_every", type=int, default=500)
    parser.add_argument("--log_every", type=int, default=100)
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("run")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.cmd == "run":
        run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
