#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import math
import os
import re
from typing import List, Optional, Tuple


def safe_float(x: str) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None


def parse_eps_from_dirname(name: str) -> Optional[float]:
    m = re.search(r"eps([0-9]+(?:\.[0-9]+)?(?:e[+-]?[0-9]+)?)", name)
    if not m:
        return None
    return safe_float(m.group(1))


def discover_metric_csvs(root: str) -> List[Tuple[float, str, str]]:
    """
    Return (eps, eps_dir_name, metrics_csv_path)
    """
    out = []
    for base, _, files in os.walk(root):
        if "metrics_logs" not in base:
            continue
        for fn in files:
            if not fn.endswith(".csv"):
                continue
            csv_path = os.path.join(base, fn)
            run_dir = os.path.dirname(base)               # .../seed16
            eps_dir = os.path.basename(os.path.dirname(run_dir))  # sst5_bs32_hloss_eps...
            eps = parse_eps_from_dirname(eps_dir)
            if eps is None:
                continue
            out.append((eps, eps_dir, csv_path))
    out.sort(key=lambda x: x[0])
    return out


def read_metrics_curve(csv_path: str):
    steps_train, train_loss = [], []
    steps_eval, eval_loss = [], []

    with open(csv_path, "r", encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f)
        for row in reader:
            step = safe_float(row.get("global_step", ""))
            if step is None:
                continue
            step_i = int(step)

            t = safe_float(row.get("train_loss", ""))
            if t is not None and math.isfinite(t):
                steps_train.append(step_i)
                train_loss.append(t)

            e = safe_float(row.get("eval_loss", ""))
            eval_ran = (row.get("eval_ran", "") or "").strip().upper()
            if e is not None and math.isfinite(e) and eval_ran != "NO":
                steps_eval.append(step_i)
                eval_loss.append(e)

    return steps_train, train_loss, steps_eval, eval_loss


def plot_one(eps: float, eps_dir: str, csv_path: str, outdir: str):
    steps_train, train_loss, steps_eval, eval_loss = read_metrics_curve(csv_path)
    if not steps_train and not steps_eval:
        return

    x_all = []
    y_all = []
    if steps_train:
        x_all.extend(steps_train)
        y_all.extend(train_loss)
    if steps_eval:
        x_all.extend(steps_eval)
        y_all.extend(eval_loss)
    if not x_all or not y_all:
        return

    x_min, x_max = min(x_all), max(x_all)
    y_min, y_max = min(y_all), max(y_all)
    if x_max == x_min:
        x_max = x_min + 1
    if y_max == y_min:
        y_max = y_min + 1e-6

    # SVG canvas config
    w, h = 1200, 700
    ml, mr, mt, mb = 90, 30, 70, 80
    pw, ph = w - ml - mr, h - mt - mb

    def sx(x):
        return ml + (float(x) - x_min) / (x_max - x_min) * pw

    def sy(y):
        return mt + (1.0 - (float(y) - y_min) / (y_max - y_min)) * ph

    def polyline(xs, ys):
        pts = []
        for x, y in zip(xs, ys):
            if not (math.isfinite(x) and math.isfinite(y)):
                continue
            pts.append(f"{sx(x):.2f},{sy(y):.2f}")
        return " ".join(pts)

    train_pts = polyline(steps_train, train_loss) if steps_train else ""
    eval_pts = polyline(steps_eval, eval_loss) if steps_eval else ""

    # y ticks
    y_ticks = []
    for i in range(6):
        t = y_min + (y_max - y_min) * i / 5.0
        y_ticks.append(t)

    # x ticks
    x_ticks = []
    for i in range(6):
        t = x_min + (x_max - x_min) * i / 5.0
        x_ticks.append(int(round(t)))

    lines = []
    lines.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">')
    lines.append('<rect x="0" y="0" width="100%" height="100%" fill="white"/>')

    # Title
    lines.append(f'<text x="{w/2:.1f}" y="35" text-anchor="middle" font-size="24" font-family="Arial">Train/Eval Loss vs Step (eps={eps:g})</text>')

    # Plot border
    lines.append(f'<rect x="{ml}" y="{mt}" width="{pw}" height="{ph}" fill="none" stroke="#222" stroke-width="1.5"/>')

    # Grid + y tick labels
    for yv in y_ticks:
        yy = sy(yv)
        lines.append(f'<line x1="{ml}" y1="{yy:.2f}" x2="{ml+pw}" y2="{yy:.2f}" stroke="#ddd" stroke-width="1"/>')
        lines.append(f'<text x="{ml-10}" y="{yy+4:.2f}" text-anchor="end" font-size="12" font-family="Arial">{yv:.4g}</text>')

    # x ticks + labels
    for xv in x_ticks:
        xx = sx(xv)
        lines.append(f'<line x1="{xx:.2f}" y1="{mt}" x2="{xx:.2f}" y2="{mt+ph}" stroke="#eee" stroke-width="1"/>')
        lines.append(f'<text x="{xx:.2f}" y="{mt+ph+22}" text-anchor="middle" font-size="12" font-family="Arial">{xv}</text>')

    # Axes labels
    lines.append(f'<text x="{ml+pw/2:.1f}" y="{h-20}" text-anchor="middle" font-size="16" font-family="Arial">global_step</text>')
    lines.append(f'<text x="25" y="{mt+ph/2:.1f}" text-anchor="middle" font-size="16" font-family="Arial" transform="rotate(-90 25 {mt+ph/2:.1f})">loss</text>')

    # Curves
    if train_pts:
        lines.append(f'<polyline points="{train_pts}" fill="none" stroke="#1f77b4" stroke-width="2"/>')
    if eval_pts:
        lines.append(f'<polyline points="{eval_pts}" fill="none" stroke="#d62728" stroke-width="2" stroke-dasharray="8 6"/>')
        # eval points markers
        for x, y in zip(steps_eval, eval_loss):
            if not (math.isfinite(x) and math.isfinite(y)):
                continue
            lines.append(f'<circle cx="{sx(x):.2f}" cy="{sy(y):.2f}" r="2.4" fill="#d62728"/>')

    # Legend
    lx, ly = ml + 10, mt + 15
    lines.append(f'<line x1="{lx}" y1="{ly}" x2="{lx+35}" y2="{ly}" stroke="#1f77b4" stroke-width="2"/>')
    lines.append(f'<text x="{lx+45}" y="{ly+4}" font-size="13" font-family="Arial">train_loss</text>')
    lines.append(f'<line x1="{lx}" y1="{ly+22}" x2="{lx+35}" y2="{ly+22}" stroke="#d62728" stroke-width="2" stroke-dasharray="8 6"/>')
    lines.append(f'<text x="{lx+45}" y="{ly+26}" font-size="13" font-family="Arial">eval_loss</text>')

    lines.append("</svg>")

    out_path = os.path.join(outdir, f"curve_{eps_dir}.svg")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default=".", help="h-loss root directory")
    ap.add_argument("--outdir", type=str, default="plots_train_eval_by_eps", help="output plot directory")
    args = ap.parse_args()

    root = os.path.abspath(args.root)
    outdir = os.path.join(root, args.outdir) if not os.path.isabs(args.outdir) else args.outdir
    os.makedirs(outdir, exist_ok=True)

    entries = discover_metric_csvs(root)
    if not entries:
        print(f"[warn] no metrics csv found under: {root}")
        return

    for eps, eps_dir, csv_path in entries:
        plot_one(eps, eps_dir, csv_path, outdir)

    print(f"[done] wrote {len(entries)} curve files to: {outdir}")


if __name__ == "__main__":
    main()
