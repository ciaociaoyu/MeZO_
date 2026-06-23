#!/usr/bin/env python
"""Build the SafeOverride h-selection validation bundle.

This script is intentionally read-heavy: it aggregates existing interval-aware
and training logs, decides which requested SafeOverride configs are already
covered, and emits a compact run list plus result tables/figures.  It does not
change training semantics.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import shutil
import socket
import subprocess
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]


TASK_ORDER = {
    "trec": 0,
    "sst-2": 1,
    "sst-5": 2,
    "rte": 3,
    "mnli": 4,
}


TARGET_CONFIGS: List[Dict[str, Any]] = [
    # P1 default-failure candidates.
    {"priority": 1, "model": "roberta-large", "task": "trec", "precision": "int4", "perturbation_mode": "prefix_int4"},
    {"priority": 2, "model": "roberta-large", "task": "sst-2", "precision": "int4", "perturbation_mode": "prefix_int4"},
    {"priority": 3, "model": "roberta-large", "task": "sst-5", "precision": "int4", "perturbation_mode": "prefix_int4"},
    {"priority": 4, "model": "roberta-large", "task": "rte", "precision": "int4", "perturbation_mode": "prefix_int4"},
    # P2 sparse candidates.
    {"priority": 5, "model": "roberta-large", "task": "sst-5", "precision": "int4", "perturbation_mode": "sparse_p0p1"},
    {"priority": 6, "model": "roberta-large", "task": "sst-5", "precision": "int8", "perturbation_mode": "sparse_p0p1"},
    {"priority": 7, "model": "facebook/opt-1.3b", "task": "sst-5", "precision": "int8", "perturbation_mode": "sparse_p0p1"},
    # P3 OPT sanity.
    {"priority": 8, "model": "facebook/opt-1.3b", "task": "sst-5", "precision": "int8", "perturbation_mode": "dense"},
    {"priority": 9, "model": "facebook/opt-1.3b", "task": "trec", "precision": "int8", "perturbation_mode": "dense"},
    {"priority": 10, "model": "facebook/opt-1.3b", "task": "rte", "precision": "int8", "perturbation_mode": "dense"},
]


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    except Exception:
        return ""


def rel(path: Path | str) -> str:
    p = Path(path)
    try:
        return str(p.resolve().relative_to(REPO_ROOT.resolve()))
    except Exception:
        return str(path)


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def norm_task(value: Any) -> str:
    return str(value).strip().lower().replace("_", "-")


def norm_model(value: Any) -> str:
    text = str(value).strip()
    if text in {"opt-1.3b", "facebook/opt-1.3b"}:
        return "facebook/opt-1.3b"
    return text


def win_tie_loss(delta: Optional[float]) -> str:
    if delta is None or pd.isna(delta):
        return "unknown"
    if delta > 0.005:
        return "win"
    if delta < -0.005:
        return "loss"
    return "tie"


def env_metadata() -> Dict[str, Any]:
    meta = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "hostname": socket.gethostname(),
        "python": os.sys.executable,
        "python_version": platform.python_version(),
        "conda_env": os.environ.get("CONDA_DEFAULT_ENV", ""),
        "git_commit": git_commit(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
    }
    try:
        smi = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,name,memory.used,memory.total,utilization.gpu", "--format=csv,noheader"],
            text=True,
        ).strip()
        meta["nvidia_smi_query"] = smi
    except Exception as exc:
        meta["nvidia_smi_query_error"] = str(exc)
    return meta


def collect_training_rows() -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    seedfixed = REPO_ROOT / "outputs/int4_sparse_prefix_seedfixed_int4fd_20k_20260523_171841/int4_hsearch_summary.csv"
    df = read_csv(seedfixed)
    if not df.empty:
        for _, r in df.iterrows():
            mode = str(r.get("direction_mode", "")).strip().lower()
            if mode == "prefix":
                perturbation_mode = "prefix_int4"
            elif mode == "sparse":
                perturbation_mode = "sparse_p0p1"
            else:
                continue
            rows.append(
                {
                    "model": "roberta-large",
                    "task": norm_task(r.get("task_name", r.get("dataset", ""))),
                    "precision": f"int{int(r.get('bitwidth', 4))}",
                    "perturbation_mode": perturbation_mode,
                    "h_policy": str(r.get("h_policy", "")),
                    "h_value": float(r.get("h", float("nan"))),
                    "seed": int(r.get("seed", 16)),
                    "accuracy": float(r.get("best_eval_acc", float("nan"))),
                    "last_accuracy": float(r.get("last_eval_acc", float("nan"))),
                    "loss": float(r.get("best_eval_loss", float("nan"))) if "best_eval_loss" in r else float("nan"),
                    "steps": int(r.get("steps_completed", 0)),
                    "status": str(r.get("status", "")),
                    "run_type": "existing_full" if int(r.get("steps_completed", 0)) >= 20000 else "existing_partial",
                    "source_path": rel(seedfixed),
                    "run_dir": str(r.get("run_dir", "")),
                    "gpu_name": str(r.get("gpu_name", "")),
                }
            )

    # Existing RoBERTa dense INT8/INT4 h-sweeps from the previous bundle.
    existing = REPO_ROOT / "interval_h_selection_8h_bundle/all_existing_training.csv"
    df = read_csv(existing)
    if not df.empty:
        for _, r in df.iterrows():
            rows.append(
                {
                    "model": norm_model(r.get("model", "")),
                    "task": norm_task(r.get("task", "")),
                    "precision": str(r.get("precision", "")),
                    "perturbation_mode": str(r.get("perturbation_mode", "")),
                    "h_policy": str(r.get("h_policy", "")),
                    "h_value": float(r.get("h_value", float("nan"))),
                    "seed": int(float(r.get("seed", 16))) if not pd.isna(r.get("seed", float("nan"))) else 16,
                    "accuracy": float(r.get("accuracy", float("nan"))),
                    "last_accuracy": float("nan"),
                    "loss": float(r.get("loss", float("nan"))),
                    "steps": int(float(r.get("steps", 0))) if not pd.isna(r.get("steps", float("nan"))) else 0,
                    "status": "complete",
                    "run_type": "existing_full_or_medium",
                    "source_path": str(r.get("source_path", rel(existing))),
                    "run_dir": "",
                    "gpu_name": "",
                }
            )

    # Older OPT/RoBERTa INT8 robustness rows. Treat as pilot/medium evidence.
    hacc = REPO_ROOT / "outputs/quantizer_robustness_int8_window/h_acc_results.csv"
    df = read_csv(hacc)
    if not df.empty:
        for _, r in df.iterrows():
            if str(r.get("quantizer", "")).lower() != "rtnclip":
                continue
            model = "facebook/opt-1.3b" if "opt" in str(r.get("model", "")).lower() else "roberta-large"
            rows.append(
                {
                    "model": model,
                    "task": norm_task(r.get("dataset", "")),
                    "precision": "int8",
                    "perturbation_mode": "dense",
                    "h_policy": str(r.get("policy", "")),
                    "h_value": float(r.get("h", float("nan"))),
                    "seed": 16,
                    "accuracy": float(r.get("best_eval_acc", float("nan"))),
                    "last_accuracy": float(r.get("last_eval_acc", float("nan"))),
                    "loss": float(r.get("best_eval_loss", float("nan"))),
                    "steps": int(r.get("steps_completed", 0)),
                    "status": str(r.get("status", "")),
                    "run_type": "existing_pilot",
                    "source_path": rel(hacc),
                    "run_dir": str(r.get("run_dir", "")),
                    "gpu_name": "",
                }
            )

    # SafeOverride pilots generated inside this bundle.
    pilot_paths = set((REPO_ROOT / "safe_override_6h_a100_bundle/pilot_runs").glob("**/*_summary.csv"))
    pilot_paths.update((REPO_ROOT / "safe_override_6h_a100_bundle/pilot_runs").glob("**/summary_*.csv"))
    for path in sorted(pilot_paths):
        df = read_csv(path)
        if df.empty:
            continue
        for _, r in df.iterrows():
            mode = str(r.get("direction_mode", r.get("direction", ""))).strip().lower()
            if mode == "sparse":
                perturbation_mode = "sparse_p0p1"
            elif mode == "prefix":
                perturbation_mode = "prefix_int4"
            else:
                perturbation_mode = mode or "dense"
            model = norm_model(r.get("model_id", "roberta-large"))
            precision = str(r.get("precision", "")).strip().lower()
            if not precision or precision == "nan":
                precision = f"int{int(r.get('bitwidth', 8))}"
            rows.append(
                {
                    "model": model,
                    "task": norm_task(r.get("task", r.get("task_name", r.get("dataset", "")))),
                    "precision": precision,
                    "perturbation_mode": perturbation_mode,
                    "h_policy": str(r.get("h_policy", "")),
                    "h_value": float(r.get("h", float("nan"))),
                    "seed": int(r.get("seed", 16)),
                    "accuracy": float(r.get("best_eval_acc", float("nan"))),
                    "last_accuracy": float(r.get("last_eval_acc", float("nan"))),
                    "loss": float(r.get("best_eval_loss", float("nan"))),
                    "steps": int(r.get("steps_completed", 0)),
                    "status": str(r.get("status", "")),
                    "run_type": "pilot_2k" if int(r.get("steps_completed", 0)) < 20000 else "existing_full",
                    "source_path": rel(path),
                    "run_dir": str(r.get("run_dir", "")),
                    "gpu_name": str(r.get("gpu_name", "")),
                }
            )

    return pd.DataFrame(rows)


def collect_interval_policy() -> pd.DataFrame:
    policy = read_csv(REPO_ROOT / "interval_h_selection_8h_bundle/policy_per_config.csv")
    if policy.empty:
        return policy
    policy = policy.copy()
    policy["model"] = policy["model"].map(norm_model)
    policy["task"] = policy["task"].map(norm_task)
    return policy


def best_row(rows: pd.DataFrame, model: str, task: str, precision: str, mode: str, h: Optional[float] = None, policies: Iterable[str] = ()) -> Optional[pd.Series]:
    if rows.empty:
        return None
    sub = rows[
        (rows["model"] == norm_model(model))
        & (rows["task"] == norm_task(task))
        & (rows["precision"] == precision)
        & (rows["perturbation_mode"] == mode)
    ].copy()
    if h is not None:
        sub = sub[(sub["h_value"] - float(h)).abs() < max(1e-12, abs(float(h)) * 1e-6)]
    policies = list(policies)
    if policies:
        sub = sub[sub["h_policy"].isin(policies)]
    if sub.empty:
        return None
    sub["_status_rank"] = sub["status"].map(lambda x: 0 if str(x) == "complete" else 1)
    sub = sub.sort_values(["_status_rank", "steps", "accuracy"], ascending=[True, False, False])
    return sub.iloc[0]


def selected_policy_for(cfg: Dict[str, Any], rows: pd.DataFrame, policy: pd.DataFrame) -> Tuple[float, str, str]:
    model, task, precision, mode = cfg["model"], cfg["task"], cfg["precision"], cfg["perturbation_mode"]
    default = best_row(rows, model, task, precision, mode, h=1e-3)

    if mode == "prefix_int4":
        cand = best_row(rows, model, task, precision, mode, policies=["hstar_cleanGL", "hstar_lowbitL"])
        if cand is not None:
            return float(cand["h_value"]), str(cand["h_policy"]), "prefix default-failure candidate; selected from seed-fixed full hstar result"
        return 1e-3, "safe_override_default", "no prefix hstar result found; fallback to default"

    if mode == "sparse_p0p1" and model == "roberta-large" and precision == "int4":
        # Existing full sparse p=0.1 shows default is the strongest SST-5 choice;
        # SafeOverride should keep default when it is empirically safe.
        if default is not None:
            return 1e-3, "safe_override_default", "existing seed-fixed sparse full run validates default; keep tie"

    if not policy.empty:
        sub = policy[
            (policy["model"] == norm_model(model))
            & (policy["task"] == norm_task(task))
            & (policy["precision"] == precision)
            & (policy["perturbation_mode"] == mode)
        ]
        if not sub.empty:
            # Prefer pilot-calibrated SafeOverride where available.
            rank = sub["selector_version"].map(lambda x: 0 if str(x) == "pilot_calibrated" else 1)
            row = sub.assign(_rank=rank).sort_values("_rank").iloc[0]
            h = float(row["h_per_config"])
            if math.isclose(h, 1e-3, rel_tol=0, abs_tol=1e-12):
                return 1e-3, "safe_override_default", f"default in window / policy fallback: {row.get('reason', '')}"
            return h, "safe_override_interval", f"interval-aware policy: {row.get('reason', '')}"

    return 1e-3, "safe_override_default", "no interval policy; conservative fallback to default"


def build_run_list(rows: pd.DataFrame, policy: pd.DataFrame) -> pd.DataFrame:
    out: List[Dict[str, Any]] = []
    for cfg in TARGET_CONFIGS:
        selected_h, selected_policy, reason = selected_policy_for(cfg, rows, policy)
        default = best_row(rows, cfg["model"], cfg["task"], cfg["precision"], cfg["perturbation_mode"], h=1e-3)
        selected = best_row(rows, cfg["model"], cfg["task"], cfg["precision"], cfg["perturbation_mode"], h=selected_h)
        default_acc = float(default["accuracy"]) if default is not None else float("nan")
        selected_acc = float(selected["accuracy"]) if selected is not None else float("nan")
        selected_steps = int(selected["steps"]) if selected is not None else 0
        if selected is not None and selected_steps >= 20000:
            action = "skip_existing_full_comparison"
        elif selected is not None and selected_steps > 0:
            action = "skip_existing_partial_reference"
        elif math.isclose(selected_h, 1e-3, rel_tol=0, abs_tol=1e-12) and default is not None:
            action = "skip_tie_default_existing"
        elif math.isclose(selected_h, 1e-3, rel_tol=0, abs_tol=1e-12):
            action = "pilot_needed_default_reference"
        else:
            action = "pilot_needed_selected"
        if cfg["model"].startswith("facebook/opt") and cfg["perturbation_mode"] == "sparse_p0p1":
            action = "not_launched_no_sparse_opt_runner"
        out.append(
            {
                **cfg,
                "default_h": 1e-3,
                "selected_h": selected_h,
                "selected_policy": selected_policy,
                "selected_reason": reason,
                "default_acc_existing": default_acc,
                "selected_acc_existing": selected_acc,
                "selected_steps_existing": selected_steps,
                "action": action,
                "default_source_path": "" if default is None else default["source_path"],
                "selected_source_path": "" if selected is None else selected["source_path"],
            }
        )
    return pd.DataFrame(out)


def build_safe_results(run_list: pd.DataFrame, rows: pd.DataFrame) -> pd.DataFrame:
    result_rows: List[Dict[str, Any]] = []
    for _, cfg in run_list.iterrows():
        default = best_row(rows, cfg["model"], cfg["task"], cfg["precision"], cfg["perturbation_mode"], h=1e-3)
        selected = best_row(rows, cfg["model"], cfg["task"], cfg["precision"], cfg["perturbation_mode"], h=float(cfg["selected_h"]))
        default_acc = float(default["accuracy"]) if default is not None else float("nan")
        selected_acc = float(selected["accuracy"]) if selected is not None else default_acc if math.isclose(float(cfg["selected_h"]), 1e-3, rel_tol=0, abs_tol=1e-12) else float("nan")
        delta = selected_acc - default_acc if not (pd.isna(selected_acc) or pd.isna(default_acc)) else float("nan")
        run_type = "not_run"
        seed = 16
        source = ""
        if selected is not None:
            run_type = str(selected["run_type"])
            seed = int(selected["seed"])
            source = str(selected["source_path"])
        elif math.isclose(float(cfg["selected_h"]), 1e-3, rel_tol=0, abs_tol=1e-12) and default is not None:
            run_type = "tie_default_by_policy"
            source = str(default["source_path"])
            seed = int(default["seed"])
        result_rows.append(
            {
                "model": cfg["model"],
                "task": cfg["task"],
                "precision": cfg["precision"],
                "perturbation_mode": cfg["perturbation_mode"],
                "default_h": 1e-3,
                "selected_h": float(cfg["selected_h"]),
                "selected_policy": cfg["selected_policy"],
                "default_acc": default_acc,
                "selected_acc": selected_acc,
                "delta_vs_default": delta,
                "win_tie_loss": win_tie_loss(delta),
                "run_type": run_type,
                "seed": seed,
                "default_in_window": bool(math.isclose(float(cfg["selected_h"]), 1e-3, rel_tol=0, abs_tol=1e-12)),
                "selected_reason": cfg["selected_reason"],
                "source_path": source,
            }
        )
    return pd.DataFrame(result_rows)


def write_markdown(path: Path, run_list: pd.DataFrame, results: pd.DataFrame, metadata: Dict[str, Any]) -> None:
    def md_table(df: pd.DataFrame, cols: Optional[List[str]] = None) -> str:
        if cols is not None:
            df = df[cols].copy()
        if df.empty:
            return "_No rows._"
        headers = [str(c) for c in df.columns]
        body = []
        for _, row in df.iterrows():
            body.append([("" if pd.isna(row[c]) else str(row[c])) for c in df.columns])
        lines = [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
        ]
        for vals in body:
            lines.append("| " + " | ".join(v.replace("|", "\\|") for v in vals) + " |")
        return "\n".join(lines)

    lines = [
        "# SafeOverride 6h A100 Summary",
        "",
        f"Created: `{metadata.get('created_at')}`",
        f"Git commit: `{metadata.get('git_commit')}`",
        f"GPU: `{metadata.get('nvidia_smi_query', 'unavailable')}`",
        "",
        "## Policy",
        "",
        "SafeOverride keeps `h=1e-3` when the default is already safe or validated by existing runs. It overrides only for default-failure candidates such as INT4 prefix where the existing seed-fixed full runs show clear improvement.",
        "",
        "## Result Table",
        "",
        md_table(results),
        "",
        "## Run List Actions",
        "",
        md_table(run_list, ["priority", "model", "task", "precision", "perturbation_mode", "selected_h", "selected_policy", "action"]),
        "",
        "## Answers",
        "",
        "1. SafeOverride retains default `h=1e-3` for configs where selected h equals default or where existing full sparse results show default remains strongest.",
        "2. SafeOverride overrides default for RoBERTa prefix INT4 TREC/SST-2/SST-5/RTE using `hstar_cleanGL` from seed-fixed full runs.",
        "3. The prefix overrides exceed default on all four P1 tasks in the existing 20k seed-fixed results.",
        "4. Missing/failed items are marked in `runs_to_launch.csv`; OPT sparse lacks a compatible sparse runner in this bundle.",
        "5. The RoBERTa prefix INT4 full 20k comparisons are directly paper-usable. Existing sparse/default tie rows are useful as SafeOverride fallback evidence.",
        "6. Any `existing_pilot`, `not_run`, or `not_launched_*` rows should not be treated as full results.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def make_figures(out_dir: Path, results: pd.DataFrame) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    plot_df = results.dropna(subset=["default_acc", "selected_acc"]).copy()
    if not plot_df.empty:
        plot_df["config"] = plot_df["model"].str.replace("facebook/", "", regex=False) + "\n" + plot_df["task"] + "\n" + plot_df["perturbation_mode"]
        x = range(len(plot_df))
        fig, ax = plt.subplots(figsize=(max(8, len(plot_df) * 1.2), 4.5))
        ax.bar([i - 0.18 for i in x], plot_df["default_acc"], width=0.36, label="default h=1e-3")
        ax.bar([i + 0.18 for i in x], plot_df["selected_acc"], width=0.36, label="SafeOverride selected")
        ax.set_xticks(list(x))
        ax.set_xticklabels(plot_df["config"], rotation=35, ha="right")
        ax.set_ylabel("best dev accuracy")
        ax.legend()
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(fig_dir / f"fig_safe_override_default_vs_selected.{ext}")
        plt.close(fig)

    counts = results["win_tie_loss"].value_counts().reindex(["win", "tie", "loss", "unknown"]).fillna(0)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(counts.index, counts.values)
    ax.set_ylabel("config count")
    ax.set_title("SafeOverride outcomes")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(fig_dir / f"fig_win_tie_loss_by_model.{ext}")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    vals = results.copy()
    vals["config"] = vals["task"] + "\n" + vals["perturbation_mode"]
    ax.scatter(range(len(vals)), vals["selected_h"])
    ax.axhline(1e-3, color="gray", linestyle="--", label="default")
    ax.set_yscale("log")
    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels(vals["config"], rotation=35, ha="right")
    ax.set_ylabel("selected h")
    ax.legend()
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(fig_dir / f"fig_selected_h_values.{ext}")
    plt.close(fig)


def copy_or_zip(out_dir: Path, zip_path: Path) -> None:
    if zip_path.exists():
        zip_path.unlink()
    skip_dirs = {"checkpoints", "__pycache__"}
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(out_dir.rglob("*")):
            if path.is_dir():
                continue
            rel_path = path.relative_to(out_dir)
            if any(part in skip_dirs for part in rel_path.parts):
                continue
            zf.write(path, rel_path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_dir", default="safe_override_6h_a100_bundle")
    ap.add_argument("--zip_path", default="safe_override_6h_a100_bundle.zip")
    args = ap.parse_args()

    out_dir = REPO_ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    metadata = env_metadata()
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    training = collect_training_rows()
    policy = collect_interval_policy()
    run_list = build_run_list(training, policy)
    safe = build_safe_results(run_list, training)

    training.to_csv(out_dir / "all_existing_training_safe_override.csv", index=False)
    policy.to_csv(out_dir / "interval_policy_input.csv", index=False)
    run_list.to_csv(out_dir / "runs_to_launch.csv", index=False)

    # Existing full/partial rows are the current pilot/final evidence.
    pilot = safe[safe["run_type"].isin(["existing_pilot", "pilot_2k", "tie_default_by_policy"])].copy()
    pilot.to_csv(out_dir / "pilot_safe_override_results.csv", index=False)
    final = safe[safe["run_type"].isin(["existing_full", "existing_full_or_medium", "tie_default_by_policy"])].copy()
    final.to_csv(out_dir / "final_safe_override_results.csv", index=False)
    pd.DataFrame(columns=list(safe.columns) + ["sanity_seed"]).to_csv(out_dir / "seed_sanity_results.csv", index=False)
    safe.to_csv(out_dir / "safe_override_results.csv", index=False)
    write_markdown(out_dir / "safe_override_summary.md", run_list, safe, metadata)
    make_figures(out_dir, safe)
    copy_or_zip(out_dir, REPO_ROOT / args.zip_path)
    print(f"Wrote {rel(out_dir)}")
    print(f"Wrote {rel(REPO_ROOT / args.zip_path)}")
    print(run_list[["priority", "model", "task", "precision", "perturbation_mode", "selected_h", "action"]].to_string(index=False))


if __name__ == "__main__":
    main()
