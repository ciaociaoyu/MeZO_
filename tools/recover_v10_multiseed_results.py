#!/usr/bin/env python3
"""Recover V10 medium supplement multi-seed summaries from trainer logs.

The 2026-06-27 V10 medium supplement jobs produced empty eval_metrics.jsonl
files, but the trainer stderr logs contain the actual evaluation curves. This
script reconstructs the high-precision SST-5 multi-seed run table from those
logs and records the low-bit sparse/prefix failure mode separately.
"""

from __future__ import annotations

import ast
import json
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "v10_medium_supplement_results_20260627"
HP_ROOT = (
    OUT
    / "raw_runs"
    / "high_precision_sst5_fp32_multiseed"
    / "fp32"
    / "multiseed_plateau"
    / "results"
)
AGG = OUT / "aggregates"
FIG = OUT / "figures"


EVAL_RE = re.compile(r"INFO:src\.trainer:(\{'eval_loss': [^}]+, 'eval_acc': [^}]+\})")
MISSING_DATA_RE = re.compile(r"FileNotFoundError: \[Errno 2\] No such file or directory: '([^']+)'")


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def parse_eval_curve(stderr_path: Path, eval_steps: int, max_steps: int) -> list[dict]:
    text = stderr_path.read_text(encoding="utf-8", errors="ignore")
    records: list[dict] = []
    for match in EVAL_RE.finditer(text):
        try:
            rec = ast.literal_eval(match.group(1))
        except Exception:
            continue
        records.append({"eval_loss": float(rec["eval_loss"]), "eval_acc": float(rec["eval_acc"])})

    # The runner prints a final validation after training, duplicating the last
    # training-time eval. Keep the training-time curve only.
    if len(records) >= 2 and records[-1] == records[-2]:
        records = records[:-1]

    for i, rec in enumerate(records, start=1):
        rec["eval_index"] = i
        rec["step"] = min(i * eval_steps, max_steps)
    return records


def recover_high_precision() -> tuple[pd.DataFrame, pd.DataFrame]:
    curve_rows: list[dict] = []
    run_rows: list[dict] = []

    for stderr_path in sorted(HP_ROOT.glob("*/seed*/stderr.log")):
        run_dir = stderr_path.parent
        cfg = read_json(run_dir / "run_config.json")
        manifest = read_json(run_dir / "run_manifest_row.json")
        meta = {**manifest, **cfg}
        eval_steps = int(meta.get("eval_steps") or 1000)
        max_steps = int(meta.get("max_steps") or 20000)
        curve = parse_eval_curve(stderr_path, eval_steps=eval_steps, max_steps=max_steps)
        if not curve:
            continue

        base = {
            "model": meta.get("model", "roberta-large"),
            "task": meta.get("dataset", "SST-5"),
            "precision": meta.get("precision_mode", "fp32"),
            "mode": meta.get("direction_type", "dense"),
            "quantizer": "none",
            "h": float(meta.get("h")),
            "h_label": meta.get("h_label"),
            "seed": int(meta.get("seed")),
            "data_seed": int(meta.get("data_seed", meta.get("seed"))),
            "train_seed": int(meta.get("seed")),
            "direction_seed": int(meta.get("seed")),
            "batch_size": int(meta.get("batch_size", 64)),
            "lr": float(meta.get("lr", 1e-6)),
            "run_name": meta.get("run_name", run_dir.parent.name),
            "run_type": "full",
            "max_steps": max_steps,
            "eval_steps": eval_steps,
            "source_path": str(run_dir.relative_to(ROOT)),
            "stderr_log": str(stderr_path.relative_to(ROOT)),
            "recovery_source": "stderr.log trainer eval records",
        }

        for rec in curve:
            curve_rows.append({**base, **rec})

        best = max(curve, key=lambda r: r["eval_acc"])
        last = curve[-1]
        best_loss = min(curve, key=lambda r: r["eval_loss"])
        run_rows.append(
            {
                **base,
                "status": "complete_recovered_from_stderr",
                "n_eval_records": len(curve),
                "steps": int(last["step"]),
                "best_eval_acc": best["eval_acc"],
                "best_eval_step": int(best["step"]),
                "final_eval_acc": last["eval_acc"],
                "final_eval_step": int(last["step"]),
                "best_eval_loss": best_loss["eval_loss"],
                "best_eval_loss_step": int(best_loss["step"]),
                "final_eval_loss": last["eval_loss"],
            }
        )

    curves = pd.DataFrame(curve_rows)
    runs = pd.DataFrame(run_rows).sort_values(["h", "seed"]).reset_index(drop=True)
    return curves, runs


def aggregate_runs(runs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for h, group in runs.groupby("h", sort=True):
        rows.append(
            {
                "precision": group["precision"].iloc[0],
                "task": group["task"].iloc[0],
                "h": h,
                "h_label": group["h_label"].iloc[0],
                "n_runs": len(group),
                "seeds": ",".join(str(int(s)) for s in sorted(group["seed"].unique())),
                "best_eval_acc_mean": group["best_eval_acc"].mean(),
                "best_eval_acc_std": group["best_eval_acc"].std(ddof=1),
                "final_eval_acc_mean": group["final_eval_acc"].mean(),
                "final_eval_acc_std": group["final_eval_acc"].std(ddof=1),
                "best_eval_loss_mean": group["best_eval_loss"].mean(),
                "final_eval_loss_mean": group["final_eval_loss"].mean(),
            }
        )
    return pd.DataFrame(rows)


def variance_components(runs: pd.DataFrame, metric: str = "best_eval_acc", analysis_set: str = "all_h") -> pd.DataFrame:
    pivot = runs.pivot_table(index="h", columns="seed", values=metric, aggfunc="mean")
    pivot = pivot.dropna(axis=0, how="any").dropna(axis=1, how="any")
    if pivot.empty or pivot.shape[0] < 2 or pivot.shape[1] < 2:
        return pd.DataFrame(
            [
                {
                    "metric": metric,
                    "analysis_set": analysis_set,
                    "status": "insufficient_paired_data",
                    "n_h": pivot.shape[0],
                    "n_seeds": pivot.shape[1],
                }
            ]
        )

    grand = pivot.to_numpy().mean()
    h_effect = pivot.mean(axis=1) - grand
    seed_effect = pivot.mean(axis=0) - grand
    residual = pivot - grand - h_effect.to_frame().values - seed_effect.values

    # Descriptive variance components for the balanced paired grid. These are
    # not inferential p-value estimates; they quantify observed effect sizes.
    var_h = float((h_effect**2).mean())
    var_seed = float((seed_effect**2).mean())
    var_resid = float((residual.to_numpy() ** 2).mean())
    total = var_h + var_seed + var_resid
    return pd.DataFrame(
        [
            {
                "metric": metric,
                "analysis_set": analysis_set,
                "status": "descriptive_balanced_two_way",
                "n_h": pivot.shape[0],
                "n_seeds": pivot.shape[1],
                "var_h_policy": var_h,
                "var_seed_direction": var_seed,
                "var_residual_interaction": var_resid,
                "share_h_policy": var_h / total if total else math.nan,
                "share_seed_direction": var_seed / total if total else math.nan,
                "share_residual_interaction": var_resid / total if total else math.nan,
                "h_values": ",".join(f"{h:.8g}" for h in pivot.index),
                "seeds": ",".join(str(int(s)) for s in pivot.columns),
            }
        ]
    )


def write_plots(runs: pd.DataFrame, var_df: pd.DataFrame) -> None:
    FIG.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    for seed, group in runs.sort_values("h").groupby("seed"):
        ax.plot(group["h"], group["best_eval_acc"], marker="o", linewidth=1.5, label=f"seed {seed}")
    ax.set_xscale("log")
    ax.set_xlabel("h")
    ax.set_ylabel("Best dev accuracy")
    ax.set_title("RoBERTa-large / SST-5 FP32 multi-seed plateau")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8)
    for ext in ["pdf", "png"]:
        fig.savefig(FIG / f"high_precision_seed_lines_accuracy_vs_h_recovered.{ext}", bbox_inches="tight", dpi=180)
    plt.close(fig)

    plot_var = var_df[var_df["analysis_set"] == "all_h"] if "analysis_set" in var_df else var_df
    if plot_var.empty:
        plot_var = var_df

    if not plot_var.empty and plot_var["status"].iloc[0] == "descriptive_balanced_two_way":
        labels = ["h policy", "seed/direction", "residual"]
        vals = [
            float(plot_var["var_h_policy"].iloc[0]),
            float(plot_var["var_seed_direction"].iloc[0]),
            float(plot_var["var_residual_interaction"].iloc[0]),
        ]
        shares = [
            float(plot_var["share_h_policy"].iloc[0]),
            float(plot_var["share_seed_direction"].iloc[0]),
            float(plot_var["share_residual_interaction"].iloc[0]),
        ]
        fig, ax = plt.subplots(figsize=(5.8, 4.0))
        bars = ax.bar(labels, vals, color=["#4c78a8", "#f58518", "#54a24b"])
        ax.set_ylabel("Observed variance component")
        ax.set_title("Descriptive variance decomposition")
        ax.grid(True, axis="y", alpha=0.25)
        for bar, share in zip(bars, shares):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{share:.1%}", ha="center", va="bottom")
        for ext in ["pdf", "png"]:
            fig.savefig(FIG / f"high_precision_variance_components_recovered.{ext}", bbox_inches="tight", dpi=180)
        plt.close(fig)


def lowbit_failure_rows() -> pd.DataFrame:
    rows = []
    for log_path in sorted((OUT / "jobs").glob("v10-med_*.err")):
        text = log_path.read_text(encoding="utf-8", errors="ignore")
        for missing in MISSING_DATA_RE.findall(text):
            rows.append(
                {
                    "status": "failed_before_training",
                    "failure_type": "missing_dataset_split",
                    "missing_path": missing,
                    "log_path": str(log_path.relative_to(ROOT)),
                }
            )
    if not rows:
        return pd.DataFrame(
            [
                {
                    "status": "not_detected",
                    "failure_type": "",
                    "missing_path": "",
                    "log_path": "",
                }
            ]
        )
    return pd.DataFrame(rows).drop_duplicates().reset_index(drop=True)


def write_summary(runs: pd.DataFrame, by_h: pd.DataFrame, var_df: pd.DataFrame, failures: pd.DataFrame) -> None:
    note = [
        "# V10 Medium Multi-Seed Recovery Summary",
        "",
        "This recovery targets the 2026-06-27 V10 medium supplement queue.",
        "",
        "## What Was Recovered",
        "",
        f"- Recovered {len(runs)} completed RoBERTa-large/SST-5 FP32 high-precision runs from trainer stderr logs.",
        "- The emitted `eval_metrics.jsonl` files are empty, so `run_summary.json` is not reliable for these jobs.",
        "- Each recovered run has 20 training-time dev evaluations; the duplicated final validation line was removed from curves.",
        "- h values: " + ", ".join(f"{h:.8g}" for h in sorted(runs["h"].unique())),
        "- seeds: " + ", ".join(str(int(s)) for s in sorted(runs["seed"].unique())),
        "",
        "## High-Precision Aggregate",
        "",
        "```",
        by_h.to_string(index=False),
        "```",
        "",
        "## Variance Components",
        "",
        "```",
        var_df.to_string(index=False),
        "```",
        "",
        "## Low-Bit Multi-Seed Status",
        "",
        "The sparse/prefix INT4 seed32/seed64 jobs did not produce valid training results. They failed before training because the runner looked for missing seed-specific full-data directories such as `full-32` and `full-64`.",
        "",
        "```",
        failures.head(20).to_string(index=False),
        "```",
        "",
        "The earlier `v10_supplement_results_20260626` sparse/prefix aggregates are seed16-only despite their historical filenames; they should not be described as multi-seed confirmation.",
    ]
    (OUT / "MULTISEED_RECOVERY_SUMMARY.md").write_text("\n".join(note) + "\n", encoding="utf-8")


def main() -> None:
    AGG.mkdir(parents=True, exist_ok=True)
    curves, runs = recover_high_precision()
    if runs.empty:
        raise SystemExit("No high-precision runs recovered")

    by_h = aggregate_runs(runs)
    var_all = variance_components(runs, "best_eval_acc", analysis_set="all_h")
    inner = runs[runs["h"].isin([1e-5, 1e-4, 1e-3])].copy()
    var_inner = variance_components(inner, "best_eval_acc", analysis_set="inner_plateau_1e-5_1e-4_1e-3")
    var_df = pd.concat([var_all, var_inner], ignore_index=True)
    failures = lowbit_failure_rows()

    curves.to_csv(AGG / "high_precision_sst5_multiseed_recovered_eval_curves.csv", index=False)
    runs.to_csv(AGG / "high_precision_sst5_multiseed_recovered_runs.csv", index=False)
    by_h.to_csv(AGG / "high_precision_sst5_multiseed_recovered_by_h.csv", index=False)
    var_df.to_csv(AGG / "high_precision_variance_components_recovered.csv", index=False)
    failures.to_csv(AGG / "lowbit_multiseed_failure_reasons.csv", index=False)
    write_plots(runs, var_df)
    write_summary(runs, by_h, var_df, failures)

    print(f"Recovered {len(runs)} runs")
    print(by_h.to_string(index=False))
    print(var_df.to_string(index=False))


if __name__ == "__main__":
    main()
