#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson corr; return nan if degenerate."""
    if a.size < 2 or b.size < 2:
        return float("nan")
    sa = float(np.std(a))
    sb = float(np.std(b))
    if sa <= 0.0 or sb <= 0.0:
        return float("nan")
    c = np.corrcoef(a, b)[0, 1]
    if np.isnan(c):
        return float("nan")
    return float(c)


def load_hprobe_jsonl(jsonl_path: Path) -> pd.DataFrame:
    rows = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)

            dt = np.asarray(r.get("d_true_list", []), dtype=float)
            df = np.asarray(r.get("d_fd_list", []), dtype=float)

            # derived metrics
            sign_match = float("nan")
            corr = float("nan")
            delta_pred = float("nan")

            if dt.size > 0 and df.size == dt.size:
                # sign match: exact sign equality (including 0)
                sign_match = float(np.mean(np.sign(dt) == np.sign(df)))
                corr = safe_corr(dt, df)

                # first-order predicted ΔL for the update rule used in probe:
                # update uses average over directions: θ' = θ - lr*(1/M) Σ d_fd_j u_j
                # so g_true · update = -lr*(1/M) Σ d_true_j * d_fd_j
                M = int(r.get("ndir", dt.size)) if int(r.get("ndir", dt.size)) > 0 else dt.size
                lr = float(r.get("lr", float("nan")))
                delta_pred = -lr * float(np.sum(dt * df)) / float(M)

            rows.append(
                {
                    "global_step": int(r.get("global_step", 0)),
                    "h": float(r.get("h", float("nan"))),
                    "lr": float(r.get("lr", float("nan"))),
                    "unit_u": bool(r.get("unit_u", True)),
                    "ndir": int(r.get("ndir", dt.size if dt.size else 0)),

                    "probe_loss": float(r.get("probe_loss", float("nan"))),
                    "eval_loss": float(r.get("eval_loss", float("nan"))),
                    "loss_after": float(r.get("loss_after", float("nan"))),
                    "deltaL": float(r.get("deltaL", float("nan"))),

                    "grad_true_norm": float(r.get("grad_true_norm", float("nan"))),

                    "d_true_mean": float(r.get("d_true_mean", float("nan"))),
                    "d_true_std": float(r.get("d_true_std", float("nan"))),
                    "d_fd_mean": float(r.get("d_fd_mean", float("nan"))),
                    "d_fd_std": float(r.get("d_fd_std", float("nan"))),

                    "e_d_mean": float(r.get("e_d_mean", float("nan"))),
                    "e_d_std": float(r.get("e_d_std", float("nan"))),
                    "e_d_abs_mean": float(r.get("e_d_abs_mean", float("nan"))),
                    "e_d_abs_std": float(r.get("e_d_abs_std", float("nan"))),

                    # derived:
                    "sign_match": sign_match,
                    "corr": corr,
                    "delta_pred": delta_pred,
                }
            )

    df = pd.DataFrame(rows)
    return df


def ensure_outdir(outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)


def summarize_by_h(df: pd.DataFrame) -> pd.DataFrame:
    agg = (
        df.groupby("h")
        .agg(
            e_abs_mean=("e_d_abs_mean", "mean"),
            e_abs_std=("e_d_abs_mean", "std"),
            corr_mean=("corr", "mean"),
            corr_std=("corr", "std"),
            sign_match_mean=("sign_match", "mean"),
            deltaL_mean=("deltaL", "mean"),
            deltaL_std=("deltaL", "std"),
            delta_pred_mean=("delta_pred", "mean"),
            delta_pred_std=("delta_pred", "std"),
        )
        .reset_index()
        .sort_values("h")
    )
    return agg


def best_h_per_step(df: pd.DataFrame) -> pd.DataFrame:
    # pick per step the row with minimal mean abs error
    idx = df.groupby("global_step")["e_d_abs_mean"].idxmin()
    best = df.loc[idx, ["global_step", "h", "e_d_abs_mean", "corr", "sign_match", "deltaL", "delta_pred"]].copy()
    best = best.sort_values("global_step").reset_index(drop=True)
    return best


def plot_error_vs_h(agg: pd.DataFrame, outdir: Path) -> None:
    x = agg["h"].to_numpy()
    y = agg["e_abs_mean"].to_numpy()

    plt.figure()
    plt.plot(x, y, marker="o")
    plt.xscale("log")
    plt.yscale("log")
    plt.grid(True, which="both", linestyle="--", alpha=0.4)
    plt.title("Directional-derivative FD error vs h")
    plt.xlabel("h")
    plt.ylabel("Mean |d_true - d_fd| (across steps)")
    plt.tight_layout()
    plt.savefig(outdir / "error_vs_h.png", dpi=160)
    plt.close()


def plot_quality_vs_h(agg: pd.DataFrame, outdir: Path) -> None:
    x = agg["h"].to_numpy()
    c = agg["corr_mean"].to_numpy()
    s = agg["sign_match_mean"].to_numpy()

    plt.figure()
    plt.plot(x, c, marker="o", label="corr(d_true, d_fd)")
    plt.plot(x, s, marker="s", label="sign match")
    plt.xscale("log")
    plt.ylim(-0.1, 1.05)
    plt.grid(True, which="both", linestyle="--", alpha=0.4)
    plt.title("Quality metrics vs h")
    plt.xlabel("h")
    plt.ylabel("Average over steps")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "quality_vs_h.png", dpi=160)
    plt.close()


def plot_deltaL_vs_h(agg: pd.DataFrame, outdir: Path) -> None:
    x = agg["h"].to_numpy()
    y = agg["deltaL_mean"].to_numpy()

    plt.figure()
    plt.plot(x, y, marker="o")
    plt.xscale("log")
    plt.grid(True, which="both", linestyle="--", alpha=0.4)
    plt.title("Virtual-step ΔL vs h (note: float quantization likely)")
    plt.xlabel("h")
    plt.ylabel("Mean ΔL (loss_after - loss_base)")
    plt.tight_layout()
    plt.savefig(outdir / "deltaL_vs_h.png", dpi=160)
    plt.close()


def plot_delta_pred_vs_h(agg: pd.DataFrame, outdir: Path) -> None:
    x = agg["h"].to_numpy()
    y = agg["delta_pred_mean"].to_numpy()

    plt.figure()
    plt.plot(x, y, marker="o")
    plt.xscale("log")
    plt.grid(True, which="both", linestyle="--", alpha=0.4)
    plt.title("Predicted ΔL ≈ -lr * <g_true, g_fd>")
    plt.xlabel("h")
    plt.ylabel("Mean predicted ΔL (1st-order)")
    plt.tight_layout()
    plt.savefig(outdir / "delta_pred_vs_h.png", dpi=160)
    plt.close()


def plot_probe_loss_curve(df: pd.DataFrame, outdir: Path) -> None:
    # probe_loss should be identical across h for the same step; take the first row per step
    base = df.sort_values(["global_step", "h"]).groupby("global_step", as_index=False).first()

    plt.figure()
    plt.plot(base["global_step"], base["probe_loss"], marker="o", markersize=3)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.title("Probe batch loss over training")
    plt.xlabel("global_step")
    plt.ylabel("probe_loss on fixed B_probe")
    plt.tight_layout()
    plt.savefig(outdir / "probe_loss_curve.png", dpi=160)
    plt.close()


def plot_error_heatmap(df: pd.DataFrame, outdir: Path) -> None:
    piv = df.pivot(index="global_step", columns="h", values="e_d_abs_mean").sort_index()
    mat = np.log10(piv.to_numpy() + 1e-30)

    plt.figure(figsize=(10, 5))
    plt.imshow(mat, aspect="auto", origin="lower")
    plt.title("Heatmap: error vs (step, h)")
    plt.xlabel("h")
    plt.ylabel("global_step")
    # ticks
    hs = piv.columns.to_numpy()
    plt.xticks(ticks=np.arange(len(hs)), labels=[f"{h:g}" for h in hs], rotation=45, ha="right")
    steps = piv.index.to_numpy()
    # show a few y ticks
    if len(steps) > 1:
        yt = np.linspace(0, len(steps) - 1, num=min(8, len(steps))).astype(int)
        plt.yticks(ticks=yt, labels=[str(int(steps[i])) for i in yt])

    cb = plt.colorbar()
    cb.set_label("log10 mean |error|")
    plt.tight_layout()
    plt.savefig(outdir / "error_heatmap.png", dpi=160)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", type=str, required=True, help="Path to hprobe.jsonl")
    ap.add_argument("--outdir", type=str, default="hprobe_analysis", help="Output directory for csv/png")
    args = ap.parse_args()

    jsonl_path = Path(args.jsonl)
    outdir = Path(args.outdir)
    ensure_outdir(outdir)

    df = load_hprobe_jsonl(jsonl_path)

    print(f"[load] rows={len(df)} steps={df['global_step'].nunique()} unique_h={df['h'].nunique()}")
    if df["probe_loss"].notna().all() and df["eval_loss"].notna().all():
        same_eval = bool((df["probe_loss"] == df["eval_loss"]).all())
        if same_eval:
            print("[note] probe_loss == eval_loss for all rows -> B_eval likely equals B_probe in this run.")

    agg = summarize_by_h(df)
    best = best_h_per_step(df)

    agg.to_csv(outdir / "summary_by_h.csv", index=False)
    best.to_csv(outdir / "best_h_by_step.csv", index=False)

    # thresholds
    thr_corr = float("nan")
    thr_sign = float("nan")
    for _, row in agg.sort_values("h").iterrows():
        if math.isnan(thr_corr) and float(row["corr_mean"]) >= 0.99:
            thr_corr = float(row["h"])
        if math.isnan(thr_sign) and float(row["sign_match_mean"]) >= 0.95:
            thr_sign = float(row["h"])
    print(f"[threshold] smallest h with avg corr>=0.99: {thr_corr}")
    print(f"[threshold] smallest h with avg sign_match>=0.95: {thr_sign}")

    # how often each h is best by abs error
    counts = best["h"].value_counts().sort_index()
    print("[best by e_abs_mean] counts:")
    for h, c in counts.items():
        print(f"  h={h:g} : {int(c)} steps")

    # plots
    plot_error_vs_h(agg, outdir)
    plot_quality_vs_h(agg, outdir)
    plot_deltaL_vs_h(agg, outdir)
    plot_delta_pred_vs_h(agg, outdir)
    plot_probe_loss_curve(df, outdir)
    plot_error_heatmap(df, outdir)

    print(f"[done] wrote csv/png to: {outdir.resolve()}")


if __name__ == "__main__":
    main()