#!/usr/bin/env python3
"""Summarize the latest FP32/FP16 checkpointed main h-sweep."""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional


SUMMARY_FIELDS = [
    "run_name",
    "precision_mode",
    "h",
    "seed",
    "data_seed",
    "batch_size",
    "max_steps",
    "steps_completed",
    "best_eval_acc",
    "best_eval_step",
    "last_eval_acc",
    "last_eval_step",
    "best_eval_loss",
    "last_eval_loss",
    "final_train_loss",
    "final_train_acc",
    "nan_occurred",
    "checkpoint_count",
    "final_checkpoint_path",
    "best_acc_checkpoint_path",
    "best_loss_checkpoint_path",
    "probe_corr_fd_true",
    "probe_nMSE_fd_true",
    "probe_alignment",
    "probe_norm_ratio",
    "status",
]


def safe_float(value: Any) -> Optional[float]:
    try:
        if value in (None, ""):
            return None
        x = float(value)
    except Exception:
        return None
    return x if math.isfinite(x) else None


def read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: Iterable[Dict[str, Any]], fields: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def run_dir_for(row: Dict[str, str]) -> Path:
    return Path(row["result_root"]) / row["run_name"] / f"seed{row['seed']}"


def load_summary(row: Dict[str, str]) -> Dict[str, Any]:
    run_dir = run_dir_for(row)
    path = run_dir / "run_summary.json"
    if path.exists():
        try:
            data = json.loads(path.read_text())
        except Exception:
            data = {}
    else:
        data = {}
    out = {field: data.get(field) for field in SUMMARY_FIELDS}
    out.update(
        {
            "run_name": row["run_name"],
            "precision_mode": row["precision_mode"],
            "h": row["h"],
            "seed": row["seed"],
            "data_seed": row["data_seed"],
            "batch_size": row["batch_size"],
            "max_steps": row["max_steps"],
            "status": data.get("status") or ("missing" if not path.exists() else "incomplete"),
        }
    )
    return out


def checkpoint_inventory(rows: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in rows:
        run_dir = run_dir_for(row)
        ckpt_root = run_dir / "checkpoints"
        expected = [f"step_{i}" for i in range(1000, int(row["max_steps"]) + 1, 1000)]
        expected += ["final", "best_acc", "best_loss"]
        for label in expected:
            path = ckpt_root / label
            size = 0
            if path.exists():
                for item in path.rglob("*"):
                    if item.is_file():
                        try:
                            size += item.stat().st_size
                        except OSError:
                            pass
            out.append(
                {
                    "run_name": row["run_name"],
                    "precision_mode": row["precision_mode"],
                    "h": row["h"],
                    "checkpoint_step": label,
                    "checkpoint_path": str(path),
                    "exists": path.exists(),
                    "size_gb": f"{size / (1024 ** 3):.6f}",
                    "resumable_status": "ok" if (path / "main_checkpoint_metadata.json").exists() else ("missing" if not path.exists() else "partial"),
                }
            )
    return out


def best(rows: List[Dict[str, Any]], precision: str, field: str) -> Optional[Dict[str, Any]]:
    candidates = [r for r in rows if r.get("precision_mode") == precision and safe_float(r.get(field)) is not None]
    if not candidates:
        return None
    return max(candidates, key=lambda r: safe_float(r.get(field)) or -float("inf"))


def write_plot_csvs(root: Path, rows: List[Dict[str, Any]]) -> None:
    plot_dir = root / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    acc_fields = ["precision_mode", "h", "best_eval_acc", "last_eval_acc", "best_eval_step"]
    loss_fields = ["precision_mode", "h", "best_eval_loss", "last_eval_loss"]
    probe_fields = ["precision_mode", "h", "probe_corr_fd_true", "probe_nMSE_fd_true", "probe_alignment", "probe_norm_ratio"]
    mse_acc_fields = ["precision_mode", "h", "probe_nMSE_fd_true", "probe_corr_fd_true", "best_eval_acc", "last_eval_acc", "best_eval_loss", "last_eval_loss"]
    write_csv(plot_dir / "plot_training_acc_vs_h.csv", rows, acc_fields)
    write_csv(plot_dir / "plot_training_loss_vs_h.csv", rows, loss_fields)
    write_csv(plot_dir / "plot_probe_vs_h.csv", rows, probe_fields)
    write_csv(plot_dir / "plot_mse_vs_acc.csv", rows, mse_acc_fields)
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    def line_plot(y_key: str, name: str, ylabel: str) -> None:
        fig, ax = plt.subplots(figsize=(7, 4))
        for precision in ["fp32", "fp16"]:
            pts = sorted(
                ((safe_float(r.get("h")), safe_float(r.get(y_key))) for r in rows if r.get("precision_mode") == precision),
                key=lambda x: -1 if x[0] is None else x[0],
            )
            pts = [(x, y) for x, y in pts if x is not None and y is not None]
            if pts:
                ax.plot([x for x, _ in pts], [y for _, y in pts], marker="o", label=precision)
        ax.set_xscale("log")
        ax.set_xlabel("h")
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(plot_dir / name)
        plt.close(fig)

    line_plot("best_eval_acc", "acc_vs_h.png", "best_eval_acc")
    line_plot("last_eval_loss", "loss_vs_h.png", "last_eval_loss")
    line_plot("probe_nMSE_fd_true", "nmse_vs_h.png", "probe_nMSE_fd_true")
    line_plot("probe_corr_fd_true", "corr_vs_h.png", "probe_corr_fd_true")


def main() -> int:
    root = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path("experiments/main_latest/mezo/roberta-large/sst5/fp32_fp16_h_sweep_11h_seed16_bs64_ckpt1k_20260517").resolve()
    rows = read_csv(root / "run_manifest.csv")
    summaries = [load_summary(row) for row in rows]
    summary_dir = root / "summaries"
    summary_dir.mkdir(parents=True, exist_ok=True)
    write_csv(summary_dir / "summary_all.csv", summaries, SUMMARY_FIELDS)
    write_csv(summary_dir / "summary_fp32.csv", [r for r in summaries if r.get("precision_mode") == "fp32"], SUMMARY_FIELDS)
    write_csv(summary_dir / "summary_fp16.csv", [r for r in summaries if r.get("precision_mode") == "fp16"], SUMMARY_FIELDS)
    by_h_fields = ["h", "fp32_best_eval_acc", "fp32_last_eval_acc", "fp16_best_eval_acc", "fp16_last_eval_acc"]
    h_values = sorted({str(r.get("h")) for r in summaries}, key=lambda x: float(x))
    by_h = []
    for h in h_values:
        item = {"h": h}
        for precision in ["fp32", "fp16"]:
            match = next((r for r in summaries if r.get("precision_mode") == precision and str(r.get("h")) == h), {})
            item[f"{precision}_best_eval_acc"] = match.get("best_eval_acc")
            item[f"{precision}_last_eval_acc"] = match.get("last_eval_acc")
        by_h.append(item)
    write_csv(summary_dir / "summary_by_h.csv", by_h, by_h_fields)
    inventory = checkpoint_inventory(rows)
    write_csv(summary_dir / "checkpoint_inventory.csv", inventory, ["run_name", "precision_mode", "h", "checkpoint_step", "checkpoint_path", "exists", "size_gb", "resumable_status"])
    failed = [r for r in summaries if r.get("status") != "completed"]
    write_csv(summary_dir / "failed_or_incomplete_runs.csv", failed, SUMMARY_FIELDS)
    write_plot_csvs(root, summaries)

    completed = [r for r in summaries if r.get("status") == "completed"]
    lines = [
        "# Latest Main FP32/FP16 H-Sweep Summary",
        "",
        f"Experiment root: `{root}`",
        f"Completed runs: {len(completed)} / {len(summaries)}",
        f"Failed/incomplete runs: {len(failed)}",
        "",
    ]
    for precision in ["fp32", "fp16"]:
        b_acc = best(summaries, precision, "best_eval_acc")
        l_acc = best(summaries, precision, "last_eval_acc")
        lines.append(f"## {precision.upper()}")
        lines.append(f"- Best by best_eval_acc: `{(b_acc or {}).get('run_name')}` h=`{(b_acc or {}).get('h')}` acc=`{(b_acc or {}).get('best_eval_acc')}`")
        lines.append(f"- Best by last_eval_acc: `{(l_acc or {}).get('run_name')}` h=`{(l_acc or {}).get('h')}` acc=`{(l_acc or {}).get('last_eval_acc')}`")
        lines.append("")
    lines.extend(
        [
            "## Contract Checks",
            "- Scope is FP32/FP16 only; no INT8/INT4/sparse/residual-grid runs are in the manifest.",
            "- Manifest uses RoBERTa-large and full SST-5 only.",
            "- Launcher sets `DATALOADER_SHUFFLE=True`; run logs should contain RandomSampler lines from the Trainer override.",
            "- FP16 rows use `precision_mode=fp16` and `zo_quantization=fp16`; BF16 is not used.",
            "",
            "## Files",
            "- `summaries/summary_all.csv`",
            "- `summaries/checkpoint_inventory.csv`",
            "- `summaries/failed_or_incomplete_runs.csv`",
            "- `plots/plot_training_acc_vs_h.csv`",
            "- `plots/plot_training_loss_vs_h.csv`",
            "- `plots/plot_probe_vs_h.csv`",
            "- `plots/plot_mse_vs_acc.csv`",
        ]
    )
    (summary_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (summary_dir / "final_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"summary_all={summary_dir / 'summary_all.csv'}")
    print(f"failed={len(failed)} completed={len(completed)} total={len(summaries)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
