#!/usr/bin/env python3
import csv
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(
    "/Users/jichaoyu/Documents/GitHub/MeZO/medium_models/sh_file/sst5/bs32/h_precision_sweep/result/sst5-bs32-h-precision-sweep"
)
OUT_DIR = ROOT.parent / "figures"
SUMMARY_CSV = OUT_DIR / "h_precision_summary.csv"
JOBS_DIR = ROOT.parent.parent / "jobs"


def _parse_eval_file(path: Path) -> Dict[str, float]:
    out = {"eval_loss": float("nan"), "eval_acc": float("nan")}
    if not path.exists():
        return out
    text = path.read_text(encoding="utf-8", errors="ignore")
    m_loss = re.search(r"eval_loss\s*=\s*([0-9eE+\-.]+|nan)", text, flags=re.IGNORECASE)
    m_acc = re.search(r"eval_acc\s*=\s*([0-9eE+\-.]+|nan)", text, flags=re.IGNORECASE)
    if m_loss:
        try:
            out["eval_loss"] = float(m_loss.group(1))
        except Exception:
            out["eval_loss"] = float("nan")
    if m_acc:
        try:
            out["eval_acc"] = float(m_acc.group(1))
        except Exception:
            out["eval_acc"] = float("nan")
    return out


def _parse_probe_file(path: Path) -> Dict[str, float]:
    out = {
        "probe_rows": 0,
        "probe_mae_mean": float("nan"),
        "probe_rmse_mean": float("nan"),
        "probe_sign_acc_mean": float("nan"),
        "probe_corr_mean": float("nan"),
    }
    if not path.exists():
        return out

    maes, rmses, sign_accs, corrs = [], [], [], []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                maes.append(float(row["mae"]))
                rmses.append(float(row["rmse"]))
                sign_accs.append(float(row["sign_acc"]))
                c = float(row["corr"])
                if math.isfinite(c):
                    corrs.append(c)
            except Exception:
                continue

    out["probe_rows"] = len(maes)
    if maes:
        out["probe_mae_mean"] = float(np.mean(maes))
        out["probe_rmse_mean"] = float(np.mean(rmses))
        out["probe_sign_acc_mean"] = float(np.mean(sign_accs))
        out["probe_corr_mean"] = float(np.mean(corrs)) if corrs else float("nan")
    return out


def _safe_float(x: str) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def _run_key_from_path_text(path_text: str) -> Optional[Tuple[str, str, str]]:
    m = re.search(r"/(fp(?:16|32))/eps_([^/]+)/seed(\d+)", path_text)
    if not m:
        return None
    return (m.group(1), m.group(2), f"seed{m.group(3)}")


def _parse_train_loss_from_job(path: Path) -> Optional[Dict[str, float]]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    run_key: Optional[Tuple[str, str, str]] = None

    m_out = re.search(r"Output dir:\s*(\S+)", text)
    if m_out:
        run_key = _run_key_from_path_text(m_out.group(1))
    if run_key is None:
        m_out_alt = re.search(r"output_dir=([^,\s]+)", text)
        if m_out_alt:
            run_key = _run_key_from_path_text(m_out_alt.group(1))
    if run_key is None:
        return None

    losses: List[Tuple[int, float]] = []
    for line in text.splitlines():
        if "{'loss':" not in line or "'global_step':" not in line:
            continue
        m_loss = re.search(r"'loss':\s*([0-9eE+\-.]+|nan)", line, flags=re.IGNORECASE)
        m_step = re.search(r"'global_step':\s*(\d+)", line)
        if not m_loss or not m_step:
            continue
        losses.append((int(m_step.group(1)), _safe_float(m_loss.group(1))))

    if not losses:
        return None

    max_step = max(step for step, _ in losses)
    final_candidates = [loss for step, loss in losses if step == max_step]
    final_train_loss = final_candidates[-1] if final_candidates else float("nan")
    finite_losses = [loss for _, loss in losses if math.isfinite(loss)]
    train_loss_last100_mean = float(np.mean(finite_losses[-100:])) if finite_losses else float("nan")

    return {
        "key": run_key,
        "max_global_step": float(max_step),
        "train_loss_final": final_train_loss,
        "train_loss_last100_mean": train_loss_last100_mean,
        "log_file": str(path),
        "mtime": path.stat().st_mtime,
    }


def _collect_train_loss_map() -> Dict[Tuple[str, str, str], Dict[str, float]]:
    out: Dict[Tuple[str, str, str], Dict[str, float]] = {}
    if not JOBS_DIR.exists():
        return out

    for job_file in sorted(JOBS_DIR.glob("sst5_bs32_h_precision_sweep_*.out")):
        parsed = _parse_train_loss_from_job(job_file)
        if not parsed:
            continue
        key = parsed["key"]
        prev = out.get(key)
        if prev is None:
            out[key] = parsed
            continue
        same_step = parsed["max_global_step"] == prev["max_global_step"]
        better = (parsed["max_global_step"] > prev["max_global_step"]) or (
            same_step and parsed["mtime"] >= prev["mtime"]
        )
        if better:
            out[key] = parsed
    return out


def collect_records() -> List[Dict[str, float]]:
    records: List[Dict[str, float]] = []
    train_map = _collect_train_loss_map()
    for precision in ("fp32", "fp16"):
        pdir = ROOT / precision
        if not pdir.exists():
            continue
        for eps_dir in sorted(pdir.iterdir(), key=lambda p: float(p.name.replace("eps_", ""))):
            if not eps_dir.is_dir() or not eps_dir.name.startswith("eps_"):
                continue
            eps = eps_dir.name.replace("eps_", "")
            for run_dir in sorted(eps_dir.glob("seed*")):
                if not run_dir.is_dir():
                    continue
                eval_metrics = _parse_eval_file(run_dir / "eval_results_sst-5.txt")
                test_metrics = _parse_eval_file(run_dir / "test_results_sst-5.txt")
                probe_metrics = _parse_probe_file(run_dir / "zo_directional_probe.csv")
                train_metrics = train_map.get((precision, eps, run_dir.name), {})
                records.append(
                    {
                        "precision": precision,
                        "eps": eps,
                        "eps_float": float(eps),
                        "seed": run_dir.name,
                        "dev_loss": eval_metrics["eval_loss"],
                        "dev_acc": eval_metrics["eval_acc"],
                        "test_loss": test_metrics["eval_loss"],
                        "test_acc": test_metrics["eval_acc"],
                        "train_loss_final": float(train_metrics.get("train_loss_final", float("nan"))),
                        "train_loss_last100_mean": float(
                            train_metrics.get("train_loss_last100_mean", float("nan"))
                        ),
                        "train_max_global_step": int(train_metrics.get("max_global_step", 0)),
                        "train_log_file": str(train_metrics.get("log_file", "")),
                        **probe_metrics,
                    }
                )
    return records


def write_summary(records: List[Dict[str, float]]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fields = [
        "precision",
        "eps",
        "eps_float",
        "seed",
        "train_loss_final",
        "train_loss_last100_mean",
        "train_max_global_step",
        "train_log_file",
        "dev_loss",
        "dev_acc",
        "test_loss",
        "test_acc",
        "probe_rows",
        "probe_mae_mean",
        "probe_rmse_mean",
        "probe_sign_acc_mean",
        "probe_corr_mean",
    ]
    with SUMMARY_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in sorted(records, key=lambda r: (r["precision"], r["eps_float"])):
            writer.writerow(row)


def _series(records: List[Dict[str, float]], precision: str, metric: str):
    rs = [r for r in records if r["precision"] == precision]
    rs.sort(key=lambda r: r["eps_float"])
    xs = np.array([r["eps_float"] for r in rs], dtype=float)
    ys = np.array([r[metric] for r in rs], dtype=float)
    return xs, ys


def plot_overview(records: List[Dict[str, float]]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    for precision, color in (("fp32", "#1f77b4"), ("fp16", "#d62728")):
        xs, ys = _series(records, precision, "test_acc")
        axes[0].plot(xs, ys, marker="o", linewidth=2, markersize=5, label=precision, color=color)
    axes[0].set_xscale("log")
    axes[0].set_xlabel("h (zero_order_eps)")
    axes[0].set_ylabel("Test Accuracy")
    axes[0].set_title("SST-5 Test Acc vs h")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    for precision, color in (("fp32", "#1f77b4"), ("fp16", "#d62728")):
        xs, ys = _series(records, precision, "probe_mae_mean")
        mask = np.isfinite(ys) & (ys > 0)
        axes[1].plot(xs[mask], ys[mask], marker="o", linewidth=2, markersize=5, label=precision, color=color)
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("h (zero_order_eps)")
    axes[1].set_ylabel("Probe MAE")
    axes[1].set_title("Directional Probe MAE vs h")
    axes[1].grid(True, which="both", alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(OUT_DIR / "h_precision_overview.png", dpi=200)
    fig.savefig(OUT_DIR / "h_precision_overview.svg")
    plt.close(fig)


def plot_deltas(records: List[Dict[str, float]]) -> None:
    fp32 = {r["eps_float"]: r for r in records if r["precision"] == "fp32"}
    fp16 = {r["eps_float"]: r for r in records if r["precision"] == "fp16"}
    xs = sorted(set(fp32).intersection(fp16))
    d_test = np.array([fp16[x]["test_acc"] - fp32[x]["test_acc"] for x in xs], dtype=float)
    d_dev = np.array([fp16[x]["dev_acc"] - fp32[x]["dev_acc"] for x in xs], dtype=float)

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.plot(xs, d_test, marker="o", linewidth=2, label="Delta Test Acc (fp16 - fp32)")
    ax.plot(xs, d_dev, marker="s", linewidth=2, label="Delta Dev Acc (fp16 - fp32)")
    ax.axhline(0.0, color="black", linewidth=1, alpha=0.7)
    ax.set_xscale("log")
    ax.set_xlabel("h (zero_order_eps)")
    ax.set_ylabel("Accuracy Delta")
    ax.set_title("Accuracy Delta: fp16 - fp32")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "h_precision_delta_acc.png", dpi=200)
    fig.savefig(OUT_DIR / "h_precision_delta_acc.svg")
    plt.close(fig)


def plot_h_loss(records: List[Dict[str, float]]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    for precision, color in (("fp32", "#1f77b4"), ("fp16", "#d62728")):
        xs, ys = _series(records, precision, "train_loss_final")
        mask = np.isfinite(ys)
        axes[0].plot(xs[mask], ys[mask], marker="o", linewidth=2, markersize=5, label=precision, color=color)
    axes[0].set_xscale("log")
    axes[0].set_xlabel("h (zero_order_eps)")
    axes[0].set_ylabel("Train Loss (final step)")
    axes[0].set_title("Train Loss vs h")
    axes[0].set_ylim(0.0, 2.0)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    for precision, color in (("fp32", "#1f77b4"), ("fp16", "#d62728")):
        xs, ys = _series(records, precision, "test_loss")
        mask = np.isfinite(ys) & (ys > 0)
        axes[1].plot(xs[mask], ys[mask], marker="o", linewidth=2, markersize=5, label=precision, color=color)
    axes[1].set_xscale("log")
    axes[1].set_xlabel("h (zero_order_eps)")
    axes[1].set_ylabel("Test Loss")
    axes[1].set_title("Test Loss vs h")
    axes[1].set_ylim(0.0, 2.0)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(OUT_DIR / "h_loss_train_test.png", dpi=200)
    fig.savefig(OUT_DIR / "h_loss_train_test.svg")
    plt.close(fig)


def main() -> None:
    records = collect_records()
    if not records:
        raise RuntimeError(f"No records found under: {ROOT}")
    write_summary(records)
    plot_overview(records)
    plot_deltas(records)
    plot_h_loss(records)
    print(f"[ok] summary: {SUMMARY_CSV}")
    print(f"[ok] figures: {OUT_DIR}")


if __name__ == "__main__":
    main()
