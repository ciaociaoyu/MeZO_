#!/usr/bin/env python3
"""Summarize the FP32/FP16 RoBERTa SST-5 left-tail h-sweep extension."""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


DEFAULT_PARENT = Path(
    "experiments/main_latest/mezo/roberta-large/sst5/"
    "fp32_fp16_h_sweep_11h_seed16_bs64_ckpt1k_20260517"
)

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
    "last5_eval_acc_mean",
    "last5_eval_acc_std",
    "last5_eval_loss_mean",
    "last5_eval_loss_std",
    "final_train_loss",
    "final_train_acc",
    "last5_train_acc_mean",
    "last5_train_acc_std",
    "nan_occurred",
    "first_nan_step",
    "checkpoint_count",
    "final_checkpoint_path",
    "best_acc_checkpoint_path",
    "best_loss_checkpoint_path",
    "probe_corr_fd_true",
    "probe_nMSE_fd_true",
    "probe_sign_agreement",
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


def safe_int(value: Any) -> Optional[int]:
    x = safe_float(value)
    return None if x is None else int(x)


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


def read_metrics(run_dir: Path) -> List[Dict[str, str]]:
    for path in [run_dir / "metrics.csv", run_dir / "metrics_logs" / "metrics_adaptiveH-0_cscale-0.csv"]:
        rows = read_csv(path)
        if rows:
            return rows
    return []


def mean_std(values: List[Optional[float]]) -> tuple[Optional[float], Optional[float]]:
    xs = [x for x in values if x is not None]
    if not xs:
        return None, None
    avg = sum(xs) / len(xs)
    if len(xs) == 1:
        return avg, 0.0
    return avg, math.sqrt(sum((x - avg) ** 2 for x in xs) / len(xs))


def row_step(row: Optional[Dict[str, Any]]) -> Optional[int]:
    if not row:
        return None
    return safe_int(row.get("global_step") or row.get("step"))


def checkpoint_count(run_dir: Path, max_steps: int) -> tuple[int, Dict[str, Optional[str]]]:
    ckpt_root = run_dir / "checkpoints"
    labels = [f"step_{i}" for i in range(1000, max_steps + 1, 1000)]
    labels += ["final", "best_acc", "best_loss"]
    count = sum(1 for label in labels if (ckpt_root / label).exists())
    return count, {
        "final_checkpoint_path": str(ckpt_root / "final") if (ckpt_root / "final").exists() else None,
        "best_acc_checkpoint_path": str(ckpt_root / "best_acc") if (ckpt_root / "best_acc").exists() else None,
        "best_loss_checkpoint_path": str(ckpt_root / "best_loss") if (ckpt_root / "best_loss").exists() else None,
    }


def probe_stats(run_dir: Path) -> Dict[str, Any]:
    path = run_dir / "checkpoint_probe_stats.jsonl"
    out: Dict[str, Any] = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
        except Exception:
            continue
        out.update(item)
    return out


def compute_summary(row: Dict[str, str]) -> Dict[str, Any]:
    run_dir = run_dir_for(row)
    summary_path = run_dir / "run_summary.json"
    data: Dict[str, Any] = {}
    if summary_path.exists():
        try:
            data = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception:
            data = {}

    metrics = read_metrics(run_dir)
    eval_rows = [
        r for r in metrics
        if str(r.get("eval_ran", "")).upper() not in {"", "NO", "NONE"}
        and safe_float(r.get("eval_loss")) is not None
    ]
    last_train = metrics[-1] if metrics else None
    best_acc = max(eval_rows, key=lambda r: safe_float(r.get("eval_acc")) or -float("inf"), default=None)
    best_loss = min(eval_rows, key=lambda r: safe_float(r.get("eval_loss")) or float("inf"), default=None)
    last_eval = eval_rows[-1] if eval_rows else None
    last5_eval = eval_rows[-5:]
    last5_acc_mean, last5_acc_std = mean_std([safe_float(r.get("eval_acc")) for r in last5_eval])
    last5_loss_mean, last5_loss_std = mean_std([safe_float(r.get("eval_loss")) for r in last5_eval])
    last5_train_acc_mean, last5_train_acc_std = mean_std([safe_float(r.get("train_acc")) for r in metrics[-5:]])

    nan_occurred = bool(data.get("nan_occurred", False))
    first_nan_step = data.get("first_nan_step")
    for r in metrics:
        for key in ("train_loss", "eval_loss", "train_acc", "eval_acc"):
            raw = r.get(key)
            if raw in (None, ""):
                continue
            try:
                x = float(raw)
            except Exception:
                continue
            if math.isnan(x) or math.isinf(x):
                nan_occurred = True
                if first_nan_step in (None, ""):
                    first_nan_step = row_step(r)

    max_steps = int(float(row["max_steps"]))
    ckpt_count, ckpt_paths = checkpoint_count(run_dir, max_steps)
    probe = probe_stats(run_dir)
    status = data.get("status")
    if not status:
        status = "completed" if ckpt_paths["final_checkpoint_path"] else ("missing" if not run_dir.exists() else "incomplete")

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
            "steps_completed": data.get("steps_completed") or row_step(last_train) or 0,
            "best_eval_acc": data.get("best_eval_acc") if data.get("best_eval_acc") is not None else safe_float((best_acc or {}).get("eval_acc")),
            "best_eval_step": data.get("best_eval_step") if data.get("best_eval_step") is not None else row_step(best_acc),
            "last_eval_acc": data.get("last_eval_acc") if data.get("last_eval_acc") is not None else safe_float((last_eval or {}).get("eval_acc")),
            "last_eval_step": data.get("last_eval_step") if data.get("last_eval_step") is not None else row_step(last_eval),
            "best_eval_loss": data.get("best_eval_loss") if data.get("best_eval_loss") is not None else safe_float((best_loss or {}).get("eval_loss")),
            "last_eval_loss": data.get("last_eval_loss") if data.get("last_eval_loss") is not None else safe_float((last_eval or {}).get("eval_loss")),
            "last5_eval_acc_mean": data.get("last5_eval_acc_mean") if data.get("last5_eval_acc_mean") is not None else last5_acc_mean,
            "last5_eval_acc_std": data.get("last5_eval_acc_std") if data.get("last5_eval_acc_std") is not None else last5_acc_std,
            "last5_eval_loss_mean": data.get("last5_eval_loss_mean") if data.get("last5_eval_loss_mean") is not None else last5_loss_mean,
            "last5_eval_loss_std": data.get("last5_eval_loss_std") if data.get("last5_eval_loss_std") is not None else last5_loss_std,
            "final_train_loss": data.get("final_train_loss") if data.get("final_train_loss") is not None else safe_float((last_train or {}).get("train_loss")),
            "final_train_acc": data.get("final_train_acc") if data.get("final_train_acc") is not None else safe_float((last_train or {}).get("train_acc")),
            "last5_train_acc_mean": data.get("last5_train_acc_mean") if data.get("last5_train_acc_mean") is not None else last5_train_acc_mean,
            "last5_train_acc_std": data.get("last5_train_acc_std") if data.get("last5_train_acc_std") is not None else last5_train_acc_std,
            "nan_occurred": nan_occurred,
            "first_nan_step": first_nan_step,
            "checkpoint_count": data.get("checkpoint_count") if data.get("checkpoint_count") is not None else ckpt_count,
            "probe_corr_fd_true": data.get("probe_corr_fd_true") if data.get("probe_corr_fd_true") is not None else probe.get("corr_fd_true"),
            "probe_nMSE_fd_true": data.get("probe_nMSE_fd_true") if data.get("probe_nMSE_fd_true") is not None else probe.get("nMSE_fd_true"),
            "probe_sign_agreement": data.get("probe_sign_agreement") if data.get("probe_sign_agreement") is not None else probe.get("sign_agreement"),
            "probe_alignment": data.get("probe_alignment") if data.get("probe_alignment") is not None else probe.get("probe_alignment") or probe.get("probe_alignment_mean"),
            "probe_norm_ratio": data.get("probe_norm_ratio") if data.get("probe_norm_ratio") is not None else probe.get("probe_norm_ratio") or probe.get("probe_norm_ratio_mean"),
            "status": status,
        }
    )
    out.update({key: data.get(key) or value for key, value in ckpt_paths.items()})
    return out


def checkpoint_inventory(rows: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in rows:
        run_dir = run_dir_for(row)
        ckpt_root = run_dir / "checkpoints"
        max_steps = int(float(row["max_steps"]))
        labels = [f"step_{i}" for i in range(1000, max_steps + 1, 1000)]
        labels += ["final", "best_acc", "best_loss"]
        for label in labels:
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


def write_plots(root: Path, rows: List[Dict[str, Any]], suffix: str = "") -> None:
    plot_dir = root / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    tag = f"_{suffix}" if suffix else ""
    acc_fields = ["precision_mode", "h", "best_eval_acc", "last_eval_acc", "best_eval_step"]
    loss_fields = ["precision_mode", "h", "best_eval_loss", "last_eval_loss"]
    probe_fields = ["precision_mode", "h", "probe_corr_fd_true", "probe_nMSE_fd_true", "probe_sign_agreement", "probe_alignment", "probe_norm_ratio"]
    mse_acc_fields = ["precision_mode", "h", "probe_nMSE_fd_true", "probe_corr_fd_true", "best_eval_acc", "last_eval_acc", "best_eval_loss", "last_eval_loss"]
    write_csv(plot_dir / f"plot_training_acc_vs_h{tag}.csv", rows, acc_fields)
    write_csv(plot_dir / f"plot_training_loss_vs_h{tag}.csv", rows, loss_fields)
    write_csv(plot_dir / f"plot_probe_vs_h{tag}.csv", rows, probe_fields)
    write_csv(plot_dir / f"plot_mse_vs_acc{tag}.csv", rows, mse_acc_fields)
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    def line_plot(y_key: str, name: str, ylabel: str) -> None:
        fig, ax = plt.subplots(figsize=(7, 4))
        for precision in ["fp32", "fp16"]:
            pts = sorted(
                ((safe_float(r.get("h")), safe_float(r.get(y_key))) for r in rows if r.get("precision_mode") == precision),
                key=lambda x: float("inf") if x[0] is None else x[0],
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

    line_plot("best_eval_acc", f"acc_vs_h{tag}.png", "best_eval_acc")
    line_plot("last_eval_loss", f"loss_vs_h{tag}.png", "last_eval_loss")
    line_plot("probe_corr_fd_true", f"corr_vs_h{tag}.png", "probe_corr_fd_true")
    line_plot("probe_nMSE_fd_true", f"nMSE_vs_h{tag}.png", "probe_nMSE_fd_true")


def load_parent_rows(parent: Path) -> List[Dict[str, Any]]:
    rows = read_csv(parent / "summaries" / "summary_all.csv")
    for row in rows:
        row.setdefault("source", "previous_11h")
    return rows


def write_summary_md(root: Path, rows: List[Dict[str, Any]], merged_rows: List[Dict[str, Any]]) -> None:
    completed = [r for r in rows if r.get("status") == "completed"]
    failed = [r for r in rows if r.get("status") != "completed"]
    nan_rows = [r for r in rows if str(r.get("nan_occurred")).lower() == "true"]
    lines = [
        "# FP32/FP16 Left-Tail H-Sweep Summary",
        "",
        f"Output root: `{root}`",
        f"Completed runs: {len(completed)} / {len(rows)}",
        f"Failed/incomplete runs: {len(failed)}",
        f"NaN-marked runs: {len(nan_rows)}",
        "",
        "## Run Table",
        "",
        "| precision | h | status | steps | best_acc | last_acc | best_loss | last_loss | nan |",
        "|---|---:|---|---:|---:|---:|---:|---:|---|",
    ]
    for r in sorted(rows, key=lambda x: (x.get("precision_mode"), safe_float(x.get("h")) or 0.0)):
        lines.append(
            f"| {r.get('precision_mode')} | {r.get('h')} | {r.get('status')} | "
            f"{r.get('steps_completed')} | {r.get('best_eval_acc')} | {r.get('last_eval_acc')} | "
            f"{r.get('best_eval_loss')} | {r.get('last_eval_loss')} | {r.get('nan_occurred')} |"
        )
    lines += [
        "",
        "## Interpretation Notes",
        "",
        "- This report is updated whenever the summarizer runs; incomplete jobs remain marked as missing/incomplete.",
        "- Left-tail conclusions should only be drawn after all 12 runs complete.",
        f"- Merged rows currently available: {len(merged_rows)}.",
        "",
    ]
    (root / "summaries" / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (root / "summaries" / "final_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    root = Path(sys.argv[1]).resolve()
    parent = Path(sys.argv[2]).resolve() if len(sys.argv) > 2 else (Path.cwd() / DEFAULT_PARENT).resolve()
    manifest_rows = read_csv(root / "run_manifest.csv")
    summaries = [compute_summary(row) for row in manifest_rows]

    summary_dir = root / "summaries"
    summary_dir.mkdir(parents=True, exist_ok=True)
    write_csv(summary_dir / "summary_all.csv", summaries, SUMMARY_FIELDS)
    write_csv(summary_dir / "summary_fp32.csv", [r for r in summaries if r.get("precision_mode") == "fp32"], SUMMARY_FIELDS)
    write_csv(summary_dir / "summary_fp16.csv", [r for r in summaries if r.get("precision_mode") == "fp16"], SUMMARY_FIELDS)
    write_csv(summary_dir / "checkpoint_inventory.csv", checkpoint_inventory(manifest_rows), ["run_name", "precision_mode", "h", "checkpoint_step", "checkpoint_path", "exists", "size_gb", "resumable_status"])
    failed = [r for r in summaries if r.get("status") != "completed"]
    write_csv(summary_dir / "failed_or_incomplete_runs.csv", failed, SUMMARY_FIELDS)

    parent_rows = load_parent_rows(parent)
    for row in summaries:
        row["source"] = "left_tail"
    merged = parent_rows + summaries
    merged_fields = list(dict.fromkeys(["source", *SUMMARY_FIELDS]))
    write_csv(summary_dir / "merged_fp32_fp16_h_sweep_1e-9_to_1e-2.csv", merged, merged_fields)

    write_plots(root, summaries)
    write_plots(root, merged, "extended")
    write_summary_md(root, summaries, merged)
    print(f"wrote summaries under {summary_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
