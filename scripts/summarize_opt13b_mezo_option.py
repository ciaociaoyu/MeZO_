#!/usr/bin/env python
"""Summarize OPT-1.3B INT4 MeZO-option runs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List


def read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def read_eval(path: Path) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    best_acc = None
    best_acc_step = None
    best_loss = None
    best_loss_step = None
    for row in rows:
        if row.get("eval_acc") is not None and (best_acc is None or float(row["eval_acc"]) > best_acc):
            best_acc = float(row["eval_acc"])
            best_acc_step = int(row.get("step", 0))
        if row.get("eval_loss") is not None and (best_loss is None or float(row["eval_loss"]) < best_loss):
            best_loss = float(row["eval_loss"])
            best_loss_step = int(row.get("step", 0))
    last = rows[-1] if rows else {}
    return {
        "eval_count": len(rows),
        "best_eval_acc": best_acc,
        "best_eval_step": best_acc_step,
        "best_eval_loss": best_loss,
        "best_eval_loss_step": best_loss_step,
        "last_eval_acc": last.get("eval_acc"),
        "last_eval_loss": last.get("eval_loss"),
        "last_eval_step": last.get("step"),
    }


def read_last_metric(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return rows[-1] if rows else {}


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: List[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_md(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = [
        "task",
        "h_label",
        "h",
        "status",
        "steps_completed",
        "best_eval_acc",
        "best_eval_step",
        "last_eval_acc",
        "last_eval_step",
        "final_train_loss",
    ]
    lines = ["# OPT-1.3B INT4 MeZO-option summary", "", "|" + "|".join(cols) + "|", "|" + "|".join(["---"] * len(cols)) + "|"]
    for row in rows:
        lines.append("|" + "|".join(str(row.get(col, "")) for col in cols) + "|")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--output_csv", default="")
    parser.add_argument("--output_md", default="")
    args = parser.parse_args()

    root = Path(args.root)
    rows: List[Dict[str, Any]] = []
    for run_config_path in sorted(root.glob("*/*/run_config.json")):
        run_dir = run_config_path.parent
        cfg = read_json(run_config_path)
        summary = read_json(run_dir / "run_summary.json")
        eval_summary = read_eval(run_dir / "eval_metrics.jsonl")
        last_metric = read_last_metric(run_dir / "metrics.csv")
        inferred_status = summary.get("status")
        if not inferred_status:
            inferred_status = "partial_with_eval" if eval_summary.get("eval_count", 0) else ("running" if last_metric else "created")
        row: Dict[str, Any] = {
            "run_dir": str(run_dir),
            "task": cfg.get("task"),
            "h_label": cfg.get("h_label"),
            "h": cfg.get("h"),
            "lr": cfg.get("lr"),
            "batch_size": cfg.get("batch_size"),
            "max_seq_len": cfg.get("max_seq_len"),
            "task_path": cfg.get("task_path"),
            "status": inferred_status,
            "steps_completed": summary.get("steps_completed", last_metric.get("step")),
            "final_train_loss": summary.get("final_train_loss", last_metric.get("train_loss")),
            "last_d_h": last_metric.get("d_h"),
            "last_update_norm": last_metric.get("update_norm"),
            **eval_summary,
        }
        if summary:
            for key in ("runtime_sec", "peak_gpu_memory_mb", "nan_occurred"):
                row[key] = summary.get(key)
        rows.append(row)
    rows.sort(key=lambda r: (str(r.get("task")), str(r.get("h_label"))))
    output_csv = Path(args.output_csv) if args.output_csv else root / "summary_mezo_option.csv"
    output_md = Path(args.output_md) if args.output_md else root / "summary_mezo_option.md"
    write_csv(output_csv, rows)
    write_md(output_md, rows)
    print(f"wrote {output_csv}")
    print(f"wrote {output_md}")
    for row in rows:
        print(
            f"{row.get('task')} {row.get('h_label')} status={row.get('status')} "
            f"steps={row.get('steps_completed')} best={row.get('best_eval_acc')} "
            f"last={row.get('last_eval_acc')}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
