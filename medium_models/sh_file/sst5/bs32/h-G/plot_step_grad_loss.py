#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plot:
1) grad L1 / L2 vs training step (log-scale y)
2) train/eval loss vs training step (linear y)

The output is a self-contained SVG (no third-party dependency required).

Example:
  python plot_step_grad_loss.py \
    --run_dir medium_models/sh_file/sst5/bs32/h-G/sst5_bs32_hloss_eps3e-5_noprobe_gradnorm/seed16 \
    --out medium_models/sh_file/sst5/bs32/h-G/step_grad_l1_l2_loss.svg
"""

import argparse
import csv
import math
import os
from collections import deque
from typing import Dict, List, Optional, Tuple


def safe_float(x: str) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None


def moving_average(values: List[float], window: int) -> List[float]:
    if window <= 1 or len(values) <= 1:
        return list(values)
    out: List[float] = []
    q: deque = deque()
    s = 0.0
    for v in values:
        q.append(v)
        s += v
        if len(q) > window:
            s -= q.popleft()
        out.append(s / float(len(q)))
    return out


def aggregate_mean_by_step(pairs: List[Tuple[int, float]]) -> Tuple[List[int], List[float]]:
    by_step: Dict[int, List[float]] = {}
    for step, val in pairs:
        by_step.setdefault(step, []).append(val)
    steps = sorted(by_step.keys())
    vals = [sum(by_step[s]) / float(len(by_step[s])) for s in steps]
    return steps, vals


def read_grad_csv(path: str) -> Tuple[List[int], List[float], List[float]]:
    l1_pairs: List[Tuple[int, float]] = []
    l2_pairs: List[Tuple[int, float]] = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f)
        for row in reader:
            step_f = safe_float(row.get("global_step", ""))
            l1 = safe_float(row.get("grad_l1_norm", ""))
            l2 = safe_float(row.get("grad_l2_norm", ""))
            if step_f is None or l1 is None or l2 is None:
                continue
            if not (math.isfinite(l1) and math.isfinite(l2)):
                continue
            step = int(step_f)
            l1_pairs.append((step, l1))
            l2_pairs.append((step, l2))

    s1, l1_vals = aggregate_mean_by_step(l1_pairs)
    s2, l2_vals = aggregate_mean_by_step(l2_pairs)

    # Merge step sets to avoid mismatch after aggregation.
    all_steps = sorted(set(s1) | set(s2))
    l1_map = {s: v for s, v in zip(s1, l1_vals)}
    l2_map = {s: v for s, v in zip(s2, l2_vals)}

    steps: List[int] = []
    l1_out: List[float] = []
    l2_out: List[float] = []
    for s in all_steps:
        if s in l1_map and s in l2_map:
            steps.append(s)
            l1_out.append(l1_map[s])
            l2_out.append(l2_map[s])
    return steps, l1_out, l2_out


def read_metrics_csv(path: str) -> Tuple[List[int], List[float], List[int], List[float]]:
    train_pairs: List[Tuple[int, float]] = []
    eval_pairs: List[Tuple[int, float]] = []

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f)
        for row in reader:
            step_f = safe_float(row.get("global_step", ""))
            if step_f is None:
                continue
            step = int(step_f)

            train_loss = safe_float(row.get("train_loss", ""))
            if train_loss is not None and math.isfinite(train_loss):
                train_pairs.append((step, train_loss))

            eval_loss = safe_float(row.get("eval_loss", ""))
            eval_ran = (row.get("eval_ran", "") or "").strip().upper()
            if eval_loss is not None and math.isfinite(eval_loss) and eval_ran != "NO":
                eval_pairs.append((step, eval_loss))

    train_steps, train_loss = aggregate_mean_by_step(train_pairs)
    eval_steps, eval_loss = aggregate_mean_by_step(eval_pairs)
    return train_steps, train_loss, eval_steps, eval_loss


def log_ticks(vmin: float, vmax: float) -> List[float]:
    lo = int(math.floor(math.log10(vmin)))
    hi = int(math.ceil(math.log10(vmax)))
    ticks = [10 ** e for e in range(lo, hi + 1)]
    ticks = [t for t in ticks if vmin <= t <= vmax]
    if not ticks:
        ticks = [vmin, vmax]
    return ticks


def lin_ticks(vmin: float, vmax: float, n: int = 6) -> List[float]:
    if vmax == vmin:
        return [vmin]
    return [vmin + (vmax - vmin) * i / float(max(1, n - 1)) for i in range(n)]


def fmt_sci(v: float) -> str:
    if v <= 0:
        return "0"
    e = int(math.floor(math.log10(v)))
    m = v / (10 ** e)
    if abs(m - round(m)) < 1e-9:
        m_str = str(int(round(m)))
    else:
        m_str = f"{m:.1f}".rstrip("0").rstrip(".")
    return f"{m_str}e{e}"


def fmt_num(v: float) -> str:
    if abs(v) >= 1e4 or (0 < abs(v) < 1e-2):
        return f"{v:.2e}"
    return f"{v:.4f}".rstrip("0").rstrip(".")


def polyline_points(xs: List[int], ys: List[float], sx, sy) -> str:
    pts: List[str] = []
    for x, y in zip(xs, ys):
        if not (math.isfinite(x) and math.isfinite(y)):
            continue
        pts.append(f"{sx(x):.2f},{sy(y):.2f}")
    return " ".join(pts)


def build_svg(
    out_path: str,
    grad_steps: List[int],
    grad_l1: List[float],
    grad_l2: List[float],
    train_steps: List[int],
    train_loss: List[float],
    eval_steps: List[int],
    eval_loss: List[float],
    run_label: str,
) -> None:
    if not grad_steps:
        raise ValueError("No gradient points to plot.")
    if not train_steps:
        raise ValueError("No train_loss points to plot.")

    all_x = grad_steps + train_steps + eval_steps
    x_min, x_max = min(all_x), max(all_x)
    if x_min == x_max:
        x_max = x_min + 1

    grad_positive = [v for v in (grad_l1 + grad_l2) if v > 0 and math.isfinite(v)]
    if not grad_positive:
        raise ValueError("All grad values are non-positive or invalid; cannot use log scale.")
    g_min, g_max = min(grad_positive), max(grad_positive)
    if g_min == g_max:
        g_max = g_min * 10.0

    loss_vals = [v for v in (train_loss + eval_loss) if math.isfinite(v)]
    y2_min, y2_max = min(loss_vals), max(loss_vals)
    if y2_min == y2_max:
        y2_max = y2_min + 1e-6
    pad = (y2_max - y2_min) * 0.08
    y2_min -= pad
    y2_max += pad

    # Canvas
    w, h = 1400, 980
    ml, mr, mt, mb = 100, 45, 80, 85
    gap = 85
    pw = w - ml - mr
    ph = (h - mt - mb - gap) / 2.0

    x0, x1 = ml, ml + pw
    y1t, y1b = mt, mt + ph
    y2t, y2b = y1b + gap, y1b + gap + ph

    def sx(x: float) -> float:
        return x0 + (x - x_min) / float(x_max - x_min) * pw

    def sy_grad(v: float) -> float:
        v = max(v, 1e-30)
        lo, hi = math.log10(g_min), math.log10(g_max)
        t = 0.5 if hi == lo else (math.log10(v) - lo) / (hi - lo)
        return y1b - t * (y1b - y1t)

    def sy_loss(v: float) -> float:
        t = (v - y2_min) / float(y2_max - y2_min)
        return y2b - t * (y2b - y2t)

    x_ticks = [int(round(x_min + (x_max - x_min) * i / 5.0)) for i in range(6)]
    g_ticks = log_ticks(g_min, g_max)
    l_ticks = lin_ticks(y2_min, y2_max, n=6)

    l1_pts = polyline_points(grad_steps, grad_l1, sx, sy_grad)
    l2_pts = polyline_points(grad_steps, grad_l2, sx, sy_grad)
    tr_pts = polyline_points(train_steps, train_loss, sx, sy_loss)
    ev_pts = polyline_points(eval_steps, eval_loss, sx, sy_loss)

    lines: List[str] = []
    lines.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">')
    lines.append('<rect x="0" y="0" width="100%" height="100%" fill="white"/>')
    lines.append("<style>")
    lines.append(".title { font: 24px sans-serif; fill: #222; }")
    lines.append(".subtitle { font: 14px sans-serif; fill: #555; }")
    lines.append(".axis { stroke: #333; stroke-width: 1.2; }")
    lines.append(".grid { stroke: #e5e5e5; stroke-width: 1; }")
    lines.append(".tick { font: 12px sans-serif; fill: #333; }")
    lines.append(".label { font: 14px sans-serif; fill: #222; }")
    lines.append("</style>")

    lines.append(f'<text class="title" x="{ml}" y="38">Step vs Grad(L1/L2) and Loss</text>')
    lines.append(f'<text class="subtitle" x="{ml}" y="60">{run_label}</text>')

    # Panel borders
    lines.append(f'<rect x="{x0}" y="{y1t}" width="{pw}" height="{ph}" fill="none" class="axis"/>')
    lines.append(f'<rect x="{x0}" y="{y2t}" width="{pw}" height="{ph}" fill="none" class="axis"/>')

    # X grid/ticks (both panels)
    for xt in x_ticks:
        xx = sx(xt)
        lines.append(f'<line class="grid" x1="{xx:.2f}" y1="{y1t}" x2="{xx:.2f}" y2="{y1b}"/>')
        lines.append(f'<line class="grid" x1="{xx:.2f}" y1="{y2t}" x2="{xx:.2f}" y2="{y2b}"/>')
        lines.append(f'<line class="axis" x1="{xx:.2f}" y1="{y2b}" x2="{xx:.2f}" y2="{y2b+5}"/>')
        lines.append(f'<text class="tick" x="{xx:.2f}" y="{y2b+24}" text-anchor="middle">{xt}</text>')

    # Grad y ticks
    for yt in g_ticks:
        yy = sy_grad(yt)
        lines.append(f'<line class="grid" x1="{x0}" y1="{yy:.2f}" x2="{x1}" y2="{yy:.2f}"/>')
        lines.append(f'<line class="axis" x1="{x0-5}" y1="{yy:.2f}" x2="{x0}" y2="{yy:.2f}"/>')
        lines.append(f'<text class="tick" x="{x0-10}" y="{yy+4:.2f}" text-anchor="end">{fmt_sci(yt)}</text>')

    # Loss y ticks
    for yt in l_ticks:
        yy = sy_loss(yt)
        lines.append(f'<line class="grid" x1="{x0}" y1="{yy:.2f}" x2="{x1}" y2="{yy:.2f}"/>')
        lines.append(f'<line class="axis" x1="{x0-5}" y1="{yy:.2f}" x2="{x0}" y2="{yy:.2f}"/>')
        lines.append(f'<text class="tick" x="{x0-10}" y="{yy+4:.2f}" text-anchor="end">{fmt_num(yt)}</text>')

    # Curves
    if l1_pts:
        lines.append(f'<polyline points="{l1_pts}" fill="none" stroke="#1f77b4" stroke-width="1.3"/>')
    if l2_pts:
        lines.append(f'<polyline points="{l2_pts}" fill="none" stroke="#ff7f0e" stroke-width="1.3"/>')
    if tr_pts:
        lines.append(f'<polyline points="{tr_pts}" fill="none" stroke="#2ca02c" stroke-width="1.4"/>')
    if ev_pts:
        lines.append(f'<polyline points="{ev_pts}" fill="none" stroke="#d62728" stroke-width="1.4" stroke-dasharray="7 5"/>')
        for x, y in zip(eval_steps, eval_loss):
            lines.append(f'<circle cx="{sx(x):.2f}" cy="{sy_loss(y):.2f}" r="2.4" fill="#d62728"/>')

    # Labels
    lines.append(f'<text class="label" x="{x0}" y="{y1t-10}">Grad Norm (log scale)</text>')
    lines.append(f'<text class="label" x="{x0}" y="{y2t-10}">Loss (linear scale)</text>')
    lines.append(f'<text class="label" x="{(x0+x1)/2:.2f}" y="{h-25}" text-anchor="middle">Training Step</text>')

    # Legends
    # Top panel legend
    lx, ly = x1 - 250, y1t + 16
    lines.append(f'<rect x="{lx}" y="{ly}" width="235" height="48" fill="white" stroke="#ccc"/>')
    lines.append(f'<line x1="{lx+10}" y1="{ly+16}" x2="{lx+42}" y2="{ly+16}" stroke="#1f77b4" stroke-width="2"/>')
    lines.append(f'<text class="tick" x="{lx+50}" y="{ly+20}">grad_l1_norm</text>')
    lines.append(f'<line x1="{lx+10}" y1="{ly+34}" x2="{lx+42}" y2="{ly+34}" stroke="#ff7f0e" stroke-width="2"/>')
    lines.append(f'<text class="tick" x="{lx+50}" y="{ly+38}">grad_l2_norm</text>')

    # Bottom panel legend
    lx2, ly2 = x1 - 250, y2t + 16
    lines.append(f'<rect x="{lx2}" y="{ly2}" width="235" height="52" fill="white" stroke="#ccc"/>')
    lines.append(f'<line x1="{lx2+10}" y1="{ly2+16}" x2="{lx2+42}" y2="{ly2+16}" stroke="#2ca02c" stroke-width="2"/>')
    lines.append(f'<text class="tick" x="{lx2+50}" y="{ly2+20}">train_loss</text>')
    lines.append(f'<line x1="{lx2+10}" y1="{ly2+36}" x2="{lx2+42}" y2="{ly2+36}" stroke="#d62728" stroke-width="2" stroke-dasharray="7 5"/>')
    lines.append(f'<text class="tick" x="{lx2+50}" y="{ly2+40}">eval_loss</text>')

    lines.append("</svg>")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def resolve_inputs(args) -> Tuple[str, str, str]:
    if args.run_dir:
        run_dir = os.path.abspath(args.run_dir)
        grad_csv = os.path.join(run_dir, "metrics_logs", "grad_norms.csv")
        metrics_csv = args.metrics_csv
        if not metrics_csv:
            metrics_logs = os.path.join(run_dir, "metrics_logs")
            if os.path.isdir(metrics_logs):
                cands = [x for x in os.listdir(metrics_logs) if x.startswith("metrics_") and x.endswith(".csv")]
                cands.sort()
                if cands:
                    metrics_csv = os.path.join(metrics_logs, cands[0])
        if not metrics_csv:
            raise FileNotFoundError("Cannot auto-detect metrics_*.csv. Please pass --metrics_csv.")
        out = args.out or os.path.join(run_dir, "metrics_logs", "step_grad_l1_l2_loss.svg")
    else:
        if not args.grad_csv or not args.metrics_csv:
            raise ValueError("Either --run_dir OR both --grad_csv and --metrics_csv are required.")
        grad_csv = args.grad_csv
        metrics_csv = args.metrics_csv
        out = args.out or os.path.join(os.path.dirname(os.path.abspath(grad_csv)), "step_grad_l1_l2_loss.svg")

    return os.path.abspath(grad_csv), os.path.abspath(metrics_csv), os.path.abspath(out)


def main():
    ap = argparse.ArgumentParser(description="Plot step-vs-grad(L1/L2) and step-vs-loss into one SVG.")
    ap.add_argument("--run_dir", type=str, default="", help="Path to one run dir (contains metrics_logs/).")
    ap.add_argument("--grad_csv", type=str, default="", help="Path to grad_norms.csv.")
    ap.add_argument("--metrics_csv", type=str, default="", help="Path to metrics_*.csv.")
    ap.add_argument("--out", type=str, default="", help="Output SVG path.")
    ap.add_argument("--smooth", type=int, default=1, help="Moving-average window size (>=1).")
    args = ap.parse_args()

    grad_csv, metrics_csv, out_path = resolve_inputs(args)
    if not os.path.isfile(grad_csv):
        raise FileNotFoundError(f"grad csv not found: {grad_csv}")
    if not os.path.isfile(metrics_csv):
        raise FileNotFoundError(f"metrics csv not found: {metrics_csv}")

    grad_steps, grad_l1, grad_l2 = read_grad_csv(grad_csv)
    train_steps, train_loss, eval_steps, eval_loss = read_metrics_csv(metrics_csv)

    # Optional smoothing (same step axis, value-only smoothing)
    window = max(1, int(args.smooth))
    grad_l1_sm = moving_average(grad_l1, window)
    grad_l2_sm = moving_average(grad_l2, window)
    train_loss_sm = moving_average(train_loss, window)

    run_label = (
        f"grad_csv={os.path.basename(grad_csv)} | "
        f"metrics_csv={os.path.basename(metrics_csv)} | "
        f"smooth={window}"
    )

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    build_svg(
        out_path=out_path,
        grad_steps=grad_steps,
        grad_l1=grad_l1_sm,
        grad_l2=grad_l2_sm,
        train_steps=train_steps,
        train_loss=train_loss_sm,
        eval_steps=eval_steps,
        eval_loss=eval_loss,
        run_label=run_label,
    )

    print(f"[done] wrote: {out_path}")
    print(f"[info] grad points={len(grad_steps)}, train points={len(train_steps)}, eval points={len(eval_steps)}")


if __name__ == "__main__":
    main()

