#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyze sst5 bs32 h-loss experiments.

What it does:
1) For each h (eps) run, parse metrics_logs/*.csv to get a "converged" eval loss.
   - Plot: converged eval loss vs h (log-x).
2) For each h run, parse hprobe.jsonl and aggregate per-run metrics:
   - error: (d_true - d_fd) statistics
   - param_changed_ratio_mean (quantization/precision proxy)
   - delta_zero_frac, h_eff_mean/h, slope a in d_fd ≈ a * d_true
   - Plot each metric vs h (log-x).

Usage:
  python anl_hloss.py --root /path/to/results/sst5_bs32_hloss --out out_hloss
"""

import argparse
import csv
import json
import math
import os
import re
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt


DEBUG = False
# Exclude extreme eps points from summary plots (per-run curves still include them)
EXCLUDE_EPS_FOR_PLOTS = []


def dprint(*args):
    if DEBUG:
        print("[debug]", *args)


def _eps_match(x: float, t: float) -> bool:
    tol = max(1e-12, abs(t) * 1e-6)
    return abs(x - t) <= tol


def _is_excluded_eps(x: float, exclude: Optional[List[float]]) -> bool:
    if not exclude:
        return False
    for t in exclude:
        if _eps_match(x, t):
            return True
    return False


def safe_float(x):
    try:
        return float(x)
    except Exception:
        return None


def mean(xs: List[float]) -> Optional[float]:
    vals = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    if not vals:
        return None
    return sum(vals) / len(vals)


def parse_eps_from_dirname(name: str) -> Optional[float]:
    """
    Parse eps from a directory name like:
      sst5_bs32_hloss_eps1e-3
      sst5_bs32_hloss_eps3e-8
    """
    m = re.search(r"eps([0-9]+(?:\.[0-9]+)?(?:e[+-]?[0-9]+)?)", name)
    if not m:
        dprint("parse_eps_from_dirname: no match:", name)
        return None
    val = safe_float(m.group(1))
    dprint("parse_eps_from_dirname:", name, "->", val)
    return val


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def count_lines(path: str) -> int:
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return sum(1 for _ in f)
    except Exception:
        return 0


def discover_runs(root: str) -> Dict[str, dict]:
    """
    Return dict keyed by run_dir (seed directory).
    Each value has: eps, metrics_csv_paths, hprobe_path, eval_last5_path.
    Eps is parsed strictly from the parent directory name:
      .../sst5_bs32_hloss_eps1e-3/seed16/
    """
    runs: Dict[str, dict] = {}

    dprint("discover_runs: root=", root)
    for base, _, files in os.walk(root):
        for fn in files:
            fpath = os.path.join(base, fn)

            # Determine run_dir (seed dir) and eps_dir (parent)
            if fn == "hprobe.jsonl" or fn == "eval_loss_last5.json":
                run_dir = base
            elif fn.endswith(".csv") and os.path.basename(base) == "metrics_logs":
                run_dir = os.path.dirname(base)
            else:
                continue

            eps_dir = os.path.basename(os.path.dirname(run_dir))
            eps = parse_eps_from_dirname(eps_dir)
            if eps is None:
                dprint("skip (no eps):", run_dir, "eps_dir=", eps_dir)
                continue

            runs.setdefault(run_dir, {"eps": eps, "metrics": [], "hprobe": None, "eval_last5": None})

            if fn == "hprobe.jsonl":
                runs[run_dir]["hprobe"] = fpath
            elif fn == "eval_loss_last5.json":
                runs[run_dir]["eval_last5"] = fpath
            elif fn.endswith(".csv") and os.path.basename(base) == "metrics_logs":
                runs[run_dir]["metrics"].append(fpath)

    # Also include run dirs even if files are missing (fallback by directory naming)
    for base, dirs, _ in os.walk(root):
        if os.path.basename(base).startswith("seed"):
            eps_dir = os.path.basename(os.path.dirname(base))
            eps = parse_eps_from_dirname(eps_dir)
            if eps is None:
                continue
            runs.setdefault(base, {"eps": eps, "metrics": [], "hprobe": None, "eval_last5": None})

    dprint("discover_runs: found", len(runs), "runs")
    return runs


def pick_metrics_csv(paths: List[str]) -> Optional[str]:
    if not paths:
        return None
    # choose the file with the most lines (most complete)
    paths = sorted(paths)
    best = max(paths, key=count_lines)
    dprint("pick_metrics_csv:", best)
    return best


def parse_metrics_csv(path: str) -> dict:
    """
    Parse metrics_logs CSV and compute:
      - eval_losses list (rows with eval_loss)
      - eval_loss_avg5 list (if column exists)
      - converged_eval_loss: last non-nan eval_loss_avg5 OR mean(last 5 eval_loss)
    """
    eval_losses: List[float] = []
    eval_loss_avg5: List[float] = []
    last_eval_loss = None
    last_eval_step = None

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f)
        for row in reader:
            eval_ran = str(row.get("eval_ran", "")).strip()
            eval_loss = safe_float(row.get("eval_loss", None))
            eval_avg5 = safe_float(row.get("eval_loss_avg5", None))
            step = safe_float(row.get("global_step", None))

            if eval_avg5 is not None:
                eval_loss_avg5.append(eval_avg5)
            if eval_loss is not None:
                eval_losses.append(eval_loss)
                last_eval_loss = eval_loss
                last_eval_step = step

    conv = None
    if eval_loss_avg5:
        conv = eval_loss_avg5[-1]
    elif len(eval_losses) >= 5:
        conv = sum(eval_losses[-5:]) / 5.0
    elif eval_losses:
        conv = sum(eval_losses) / len(eval_losses)

    dprint("parse_metrics_csv:", path, "eval_count=", len(eval_losses), "last_eval_loss=", last_eval_loss, "converged=", conv)
    return {
        "metrics_csv": path,
        "eval_count": len(eval_losses),
        "last_eval_loss": last_eval_loss,
        "last_eval_step": last_eval_step,
        "converged_eval_loss": conv,
    }


def parse_metrics_timeseries(path: str) -> dict:
    """
    Return time series from metrics_logs CSV:
      - steps_train, train_loss
      - steps_eval, eval_loss (only when eval_ran != NO)
    """
    steps_train: List[int] = []
    train_loss: List[float] = []
    steps_eval: List[int] = []
    eval_loss: List[float] = []

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f)
        for row in reader:
            step = safe_float(row.get("global_step", None))
            if step is None:
                continue
            step_i = int(step)

            tl = safe_float(row.get("train_loss", None))
            if tl is not None:
                steps_train.append(step_i)
                train_loss.append(float(tl))

            ev = safe_float(row.get("eval_loss", None))
            eval_ran = str(row.get("eval_ran", "")).strip().upper()
            if ev is not None and eval_ran != "NO":
                steps_eval.append(step_i)
                eval_loss.append(float(ev))

    dprint("parse_metrics_timeseries:", path, "train_pts=", len(steps_train), "eval_pts=", len(steps_eval))
    return {
        "steps_train": steps_train,
        "train_loss": train_loss,
        "steps_eval": steps_eval,
        "eval_loss": eval_loss,
    }


def _safe_label(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", s)


def plot_loss_curve(run_dir: str, eps: float, series: dict, out_dir: str) -> None:
    steps_train = series.get("steps_train", [])
    train_loss = series.get("train_loss", [])
    steps_eval = series.get("steps_eval", [])
    eval_loss = series.get("eval_loss", [])

    if not steps_train and not steps_eval:
        return

    plt.figure()
    if steps_train and train_loss:
        plt.plot(steps_train, train_loss, label="train_loss", linewidth=1.2)
    if steps_eval and eval_loss:
        plt.plot(steps_eval, eval_loss, label="eval_loss", marker="o", linestyle="--", linewidth=1.0, markersize=3)

    plt.xlabel("global_step")
    plt.ylabel("loss")
    plt.title(f"Loss vs step (eps={eps:g})")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()

    eps_dir = os.path.basename(os.path.dirname(run_dir))
    seed_dir = os.path.basename(run_dir)
    tag = _safe_label(f"{eps_dir}_{seed_dir}")
    out_path = os.path.join(out_dir, f"loss_curve_{tag}.png")
    plt.savefig(out_path, dpi=200)
    plt.close()


def load_hprobe(path: str) -> List[dict]:
    rows = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    dprint("load_hprobe:", path, "rows=", len(rows))
    return rows


def aggregate_hprobe(rows: List[dict]) -> dict:
    # per-row derived metrics
    e_d_mean = []
    e_d_abs = []
    pcr = []
    delta_zero = []
    h_eff_over_h = []
    slope_a = []
    dir_corr = []
    dir_sign = []

    for r in rows:
        h = safe_float(r.get("h", None))
        em = safe_float(r.get("e_d_mean", None))
        ea = safe_float(r.get("e_d_abs_mean", None))
        p = safe_float(r.get("param_changed_ratio_mean", None))
        dz = safe_float(r.get("delta_zero_frac", None))
        he = safe_float(r.get("h_eff_mean", None))
        if em is not None:
            e_d_mean.append(em)
        if ea is not None:
            e_d_abs.append(ea)
        if p is not None:
            pcr.append(p)
        if dz is not None:
            delta_zero.append(dz)
        if h is not None and h > 0.0 and he is not None:
            h_eff_over_h.append(he / h)

        # slope a from d_true_list / d_fd_list
        dt = r.get("d_true_list", None)
        df = r.get("d_fd_list", None)
        if isinstance(dt, list) and isinstance(df, list) and len(dt) == len(df) and len(dt) > 0:
            try:
                num = 0.0
                denom = 0.0
                for t, f in zip(dt, df):
                    t = float(t)
                    f = float(f)
                    num += t * f
                    denom += t * t
                if denom > 0.0:
                    slope_a.append(num / denom)
            except Exception:
                pass

        # direction consistency metrics (if present)
        dc = safe_float(r.get("dir_corr", None))
        ds = safe_float(r.get("dir_sign_match", None))
        if dc is not None:
            dir_corr.append(dc)
        if ds is not None:
            dir_sign.append(ds)

    out = {
        "e_d_mean_mean": mean(e_d_mean),
        "e_d_abs_mean": mean(e_d_abs),
        "param_changed_ratio_mean": mean(pcr),
        "delta_zero_frac_mean": mean(delta_zero),
        "h_eff_over_h_mean": mean(h_eff_over_h),
        "slope_a_mean": mean(slope_a),
        "dir_corr_mean": mean(dir_corr),
        "dir_sign_match_mean": mean(dir_sign),
        "n_probe_rows": len(rows),
    }
    dprint("aggregate_hprobe:", "n_rows=", len(rows), "e_d_abs_mean=", out.get("e_d_abs_mean"))
    return out


def save_scatter(x, y, xlabel, ylabel, title, outpath, xlog=True):
    plt.figure()
    plt.scatter(x, y)
    if xlog:
        plt.xscale("log")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def _quantile(vals: List[float], q: float) -> Optional[float]:
    vals = [v for v in vals if v is not None and not (isinstance(v, float) and math.isnan(v))]
    if not vals:
        return None
    vals = sorted(vals)
    if q <= 0:
        return float(vals[0])
    if q >= 1:
        return float(vals[-1])
    pos = (len(vals) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(vals[lo])
    frac = pos - lo
    return float(vals[lo] * (1 - frac) + vals[hi] * frac)


def save_line(x, y, xlabel, ylabel, title, outpath, xlog=True, ylim: Optional[Tuple[float, float]] = None):
    plt.figure()
    plt.plot(x, y, marker="o")
    if xlog:
        plt.xscale("log")
    if ylim is not None:
        plt.ylim(ylim[0], ylim[1])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def write_csv(path: str, rows: List[dict]) -> None:
    keys = []
    seen = set()
    for r in rows:
        for k in r.keys():
            if k not in seen:
                seen.add(k)
                keys.append(k)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, required=True, help="Root results dir (e.g., results/sst5_bs32_hloss)")
    ap.add_argument("--out", type=str, default="out_hloss", help="Output directory")
    ap.add_argument("--debug", action="store_true", help="Enable debug logs")
    args = ap.parse_args()

    global DEBUG
    DEBUG = bool(args.debug)

    root = os.path.abspath(args.root)
    out = os.path.abspath(args.out)
    ensure_dir(out)
    curves_dir = os.path.join(out, "per_run_loss_curves")
    ensure_dir(curves_dir)

    runs = discover_runs(root)
    if not runs:
        print("No runs found under root:", root)
        return

    summary_rows = []
    for run_dir, info in sorted(runs.items(), key=lambda kv: kv[1]["eps"]):
        row = {
            "run_dir": run_dir,
            "eps": info["eps"],
        }
        dprint("run:", run_dir, "eps=", info["eps"])

        # metrics logs
        metrics_csv = pick_metrics_csv(info.get("metrics", []))
        if metrics_csv:
            m = parse_metrics_csv(metrics_csv)
            row.update(m)
            series = parse_metrics_timeseries(metrics_csv)
            plot_loss_curve(run_dir, info["eps"], series, curves_dir)
        else:
            dprint("no metrics_csv for", run_dir)

        # eval_last5.json (optional)
        eval_last5_mean = None
        if info.get("eval_last5"):
            try:
                with open(info["eval_last5"], "r", encoding="utf-8") as f:
                    last5 = json.load(f)
                eval_last5_mean = last5.get("eval_loss_last5_mean")
                row["eval_loss_last5_mean"] = eval_last5_mean
                row["eval_loss_last5_count"] = last5.get("eval_loss_last5_count")
                dprint("eval_last5:", info["eval_last5"], "mean=", eval_last5_mean)
            except Exception:
                pass
        else:
            row["eval_loss_last5_mean"] = None
            row["eval_loss_last5_count"] = None
            dprint("no eval_loss_last5.json for", run_dir)

        # choose final eval loss (prefer eval_loss_last5_mean if available)
        final_eval = None
        if isinstance(eval_last5_mean, (int, float)):
            final_eval = float(eval_last5_mean)
        elif isinstance(row.get("converged_eval_loss"), (int, float)):
            final_eval = float(row.get("converged_eval_loss"))
        elif isinstance(row.get("last_eval_loss"), (int, float)):
            final_eval = float(row.get("last_eval_loss"))
        row["final_eval_loss"] = final_eval

        # hprobe
        if info.get("hprobe"):
            rows = load_hprobe(info["hprobe"])
            agg = aggregate_hprobe(rows)
            row.update(agg)
        else:
            dprint("no hprobe.jsonl for", run_dir)

        summary_rows.append(row)

    # write summary (per-run)
    write_csv(os.path.join(out, "summary.csv"), summary_rows)

    # aggregate to per-eps (mean across seeds)
    def aggregate_by_eps(metric_key: str, exclude_eps: Optional[List[float]] = None) -> List[Tuple[float, float]]:
        buckets: Dict[float, List[float]] = {}
        for r in summary_rows:
            x = r.get("eps")
            y = r.get(metric_key)
            if isinstance(x, (int, float)) and isinstance(y, (int, float)):
                if _is_excluded_eps(float(x), exclude_eps):
                    continue
                buckets.setdefault(float(x), []).append(float(y))
        pairs = []
        for k, vals in buckets.items():
            if vals:
                pairs.append((k, sum(vals) / len(vals)))
        out_pairs = sorted(pairs, key=lambda t: t[0])
        dprint("aggregate_by_eps:", metric_key, "pairs=", out_pairs)
        return out_pairs

    # plot: converged eval loss vs h (per-eps mean)
    pairs = aggregate_by_eps("final_eval_loss", exclude_eps=EXCLUDE_EPS_FOR_PLOTS)
    if pairs:
        xs_s, ys_s = zip(*pairs)
        ymin = max(min(ys_s), 0.0)
        ymax = min(max(ys_s), 2.0)
        ylim = (ymin, ymax) if ymax > ymin else (0.0, 2.0)
        save_line(
            xs_s,
            ys_s,
            xlabel="h (eps)",
            ylabel="converged eval loss (mean over seeds)",
            title="Converged eval loss vs h",
            outpath=os.path.join(out, "h_vs_converged_eval_loss.png"),
            xlog=True,
            ylim=ylim,
        )

    # plot: error vs h (per-eps mean)
    def plot_metric(metric_key: str, title: str, fname: str, ylim: Optional[Tuple[float, float]] = None, annotate: bool = False):
        pairs = aggregate_by_eps(metric_key, exclude_eps=EXCLUDE_EPS_FOR_PLOTS)
        if not pairs:
            return
        x_s, y_s = zip(*pairs)

        if not annotate:
            save_line(x_s, y_s, "h (eps)", metric_key, title, os.path.join(out, fname), xlog=True, ylim=ylim)
            return

        plt.figure()
        plt.plot(x_s, y_s, marker="o")
        plt.xscale("log")
        if ylim is not None:
            plt.ylim(ylim[0], ylim[1])
        plt.xlabel("h (eps)")
        plt.ylabel(metric_key)
        plt.title(title)
        plt.grid(True, linestyle="--", alpha=0.4)
        for x, y in zip(x_s, y_s):
            try:
                plt.text(x, y, f"{y:.4f}", fontsize=8, ha="center", va="bottom")
            except Exception:
                pass
        plt.tight_layout()
        plt.savefig(os.path.join(out, fname), dpi=200)
        plt.close()

    plot_metric("e_d_mean_mean", "Mean error (d_true - d_fd) vs h", "h_vs_error_mean.png")
    plot_metric("e_d_abs_mean", "Mean |error| vs h", "h_vs_error_abs_mean.png", ylim=(0.0, 20.0))
    plot_metric("param_changed_ratio_mean", "param_changed_ratio_mean vs h", "h_vs_param_changed_ratio.png", annotate=True)
    plot_metric("delta_zero_frac_mean", "delta_zero_frac vs h", "h_vs_delta_zero_frac.png")
    plot_metric("h_eff_over_h_mean", "h_eff_mean / h vs h", "h_vs_h_eff_over_h.png")
    plot_metric("slope_a_mean", "d_fd ≈ a * d_true (slope a) vs h", "h_vs_slope_a.png")
    plot_metric("dir_corr_mean", "Directional correlation vs h", "h_vs_dir_corr.png", ylim=(0.0, 1.0))
    plot_metric("dir_sign_match_mean", "Directional sign match vs h", "h_vs_dir_sign_match.png", ylim=(0.0, 1.0))

    print("Done. Outputs in:", out)


if __name__ == "__main__":
    main()
