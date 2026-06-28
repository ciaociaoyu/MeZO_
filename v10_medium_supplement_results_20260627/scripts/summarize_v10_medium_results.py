#!/usr/bin/env python3
from __future__ import annotations
import csv, json, math, sys
from pathlib import Path
import pandas as pd
import numpy as np

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.cwd()
AG = OUT / "aggregates"
FIG = OUT / "figures"
AG.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))

def collect_lowbit():
    rows = []
    root = OUT / "raw_runs" / "roberta_int4_sparse_prefix_multiseed"
    for p in root.glob("int4_hsearch/**/run_summary.json"):
        try:
            d = read_json(p)
        except Exception:
            continue
        cfg = {}
        cfgp = p.parent / "run_config.json"
        if cfgp.exists():
            try: cfg = read_json(cfgp)
            except Exception: cfg = {}
        rows.append({**cfg, **d, "source_path": str(p)})
    return pd.DataFrame(rows)

def collect_hp():
    rows = []
    root = OUT / "raw_runs" / "high_precision_sst5_fp32_multiseed"
    for p in root.glob("fp32/multiseed_plateau/results/**/run_summary.json"):
        try: rows.append({**read_json(p), "source_path": str(p)})
        except Exception: pass
    return pd.DataFrame(rows)

hp = collect_hp()
lb = collect_lowbit()
if not hp.empty:
    hp.to_csv(OUT / "raw_runs" / "high_precision_sst5_multiseed_runs.csv", index=False)
    summ = hp.groupby(["h"]).agg(
        n=("best_eval_acc","count"),
        best_dev_acc_mean=("best_eval_acc","mean"),
        best_dev_acc_std=("best_eval_acc","std"),
        final_dev_acc_mean=("last_eval_acc","mean"),
        final_dev_acc_std=("last_eval_acc","std"),
    ).reset_index()
    summ.to_csv(AG / "high_precision_sst5_multiseed_summary.csv", index=False)
    # Simple two-way variance components.
    tab = hp.dropna(subset=["h","seed","best_eval_acc"]).copy()
    if tab["h"].nunique() > 1 and tab["seed"].nunique() > 1:
        grand = tab["best_eval_acc"].mean()
        hm = tab.groupby("h")["best_eval_acc"].mean()
        sm = tab.groupby("seed")["best_eval_acc"].mean()
        resid = tab.apply(lambda r: r["best_eval_acc"] - hm.loc[r["h"]] - sm.loc[r["seed"]] + grand, axis=1)
        var = pd.DataFrame([{
            "metric": "best_eval_acc",
            "num_runs": len(tab),
            "num_h": tab["h"].nunique(),
            "num_seed": tab["seed"].nunique(),
            "var_h_policy": float(np.var(hm - grand)),
            "var_seed_direction": float(np.var(sm - grand)),
            "var_residual_interaction": float(np.var(resid)),
        }])
    else:
        var = pd.DataFrame([{"metric":"best_eval_acc","status":"insufficient_completed_runs"}])
    var.to_csv(AG / "high_precision_variance_components.csv", index=False)
if not lb.empty:
    lb.to_csv(OUT / "raw_runs" / "lowbit_sparse_prefix_multiseed_runs.csv", index=False)
    prefix = lb[lb.get("direction_mode","").astype(str).eq("prefix")].copy()
    sparse = lb[lb.get("direction_mode","").astype(str).eq("sparse")].copy()
    for name, df in [("prefix_int4_multiseed_summary.csv", prefix), ("sparse_int4_multiseed_summary.csv", sparse)]:
        if df.empty:
            pd.DataFrame([{"status":"no_completed_runs"}]).to_csv(AG / name, index=False)
            continue
        group_cols = ["task_name","h_policy","h"]
        s = df.groupby(group_cols).agg(
            n=("best_eval_acc","count"),
            best_dev_acc_mean=("best_eval_acc","mean"),
            best_dev_acc_std=("best_eval_acc","std"),
            final_dev_acc_mean=("last_eval_acc","mean"),
            final_dev_acc_std=("last_eval_acc","std"),
        ).reset_index()
        s.to_csv(AG / name, index=False)

try:
    import matplotlib.pyplot as plt
    if not hp.empty:
        fig, ax = plt.subplots(figsize=(6,4))
        for seed, g in hp.groupby("seed"):
            g = g.sort_values("h")
            ax.plot(g["h"], g["best_eval_acc"], marker="o", label=f"seed{seed}")
        ax.set_xscale("log"); ax.set_xlabel("h"); ax.set_ylabel("best dev acc"); ax.grid(True, alpha=.25); ax.legend(fontsize=7)
        for ext in ["pdf","png"]: fig.savefig(FIG / f"high_precision_seed_lines_accuracy_vs_h.{ext}", bbox_inches="tight", dpi=180)
        plt.close(fig)
    for mode_name, filename in [("prefix", "prefix_int4_policy_bars_with_seed_dots"), ("sparse", "sparse_int4_policy_bars_with_seed_dots")]:
        df = lb[lb.get("direction_mode","").astype(str).eq(mode_name)] if not lb.empty else pd.DataFrame()
        if df.empty: continue
        fig, ax = plt.subplots(figsize=(8,4))
        labels=[]; vals=[]; xs=[]; i=0
        for (task, pol), g in df.groupby(["task_name","h_policy"]):
            labels.append(f"{task}\n{pol}")
            vals.append(g["best_eval_acc"].mean())
            xs.append(i)
            ax.scatter([i]*len(g), g["best_eval_acc"], color="black", s=14, zorder=3)
            i += 1
        ax.bar(xs, vals, alpha=.75)
        ax.set_xticks(xs); ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
        ax.set_ylabel("best dev acc"); ax.grid(True, axis="y", alpha=.25)
        for ext in ["pdf","png"]: fig.savefig(FIG / f"{filename}.{ext}", bbox_inches="tight", dpi=180)
        plt.close(fig)
except Exception as exc:
    (OUT / "figure_generation_error.txt").write_text(str(exc)+"\n")

notes = ["# V10 Medium Supplement Paper Update Notes", ""]
notes.append("Run this summarizer after Slurm jobs finish. Completed rows are aggregated only from run_summary.json files.")
notes.append("")
notes.append(f"High precision completed rows: {len(hp)}")
notes.append(f"Low-bit completed rows: {len(lb)}")
(OUT / "paper_update_notes.md").write_text("\n".join(notes)+"\n", encoding="utf-8")
print(f"wrote summaries under {OUT}")
