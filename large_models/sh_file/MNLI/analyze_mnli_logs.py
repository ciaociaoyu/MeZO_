#!/usr/bin/env python3
import argparse
import csv
import glob
import html
import math
import os
import re
from dataclasses import dataclass
from typing import List, Optional


EPS_PATTERNS = [
    re.compile(r"Selected h\(zo_eps\)\s*=\s*([0-9eE\.\-]+)"),
    re.compile(r"(?:^|[\s,])eps=([0-9eE\.\-]+)"),
]

ACC_PATTERN = re.compile(
    r"'accuracy':\s*(?:np\.float64\()?\s*([0-9]*\.?[0-9]+)\)?\s*,\s*'valid_mismatched_accuracy':\s*(?:np\.float64\()?\s*([0-9]*\.?[0-9]+)\)?"
)


@dataclass
class Record:
    log_file: str
    eps: float
    accuracy: float
    mismatched_accuracy: float


def parse_one_log(path: str) -> Optional[Record]:
    eps_value = None
    acc = None
    mm_acc = None

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if eps_value is None:
                for p in EPS_PATTERNS:
                    m = p.search(line)
                    if m:
                        try:
                            eps_value = float(m.group(1))
                        except ValueError:
                            eps_value = None
                        break
            if acc is None:
                m2 = ACC_PATTERN.search(line)
                if m2:
                    acc = float(m2.group(1))
                    mm_acc = float(m2.group(2))

    if eps_value is None or acc is None or mm_acc is None:
        return None
    return Record(log_file=path, eps=eps_value, accuracy=acc, mismatched_accuracy=mm_acc)


def dedup_records(records: List[Record]) -> List[Record]:
    # If multiple logs share one eps, keep the one with the best matched accuracy.
    best = {}
    for r in records:
        prev = best.get(r.eps)
        if prev is None or r.accuracy > prev.accuracy:
            best[r.eps] = r
    return [best[k] for k in sorted(best.keys())]


def save_csv(records: List[Record], out_csv: str) -> None:
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["eps", "accuracy", "valid_mismatched_accuracy", "log_file"])
        for r in records:
            w.writerow([f"{r.eps:.10g}", f"{r.accuracy:.6f}", f"{r.mismatched_accuracy:.6f}", r.log_file])


def _polyline(points):
    return " ".join([f"{x:.2f},{y:.2f}" for x, y in points])


def save_svg(records: List[Record], out_svg: str) -> None:
    os.makedirs(os.path.dirname(out_svg), exist_ok=True)

    width, height = 920, 560
    left, right, top, bottom = 90, 40, 45, 80
    plot_w = width - left - right
    plot_h = height - top - bottom

    xs = [r.eps for r in records]
    log_xs = [math.log10(x) for x in xs]
    ys_all = [r.accuracy for r in records] + [r.mismatched_accuracy for r in records]

    min_x, max_x = min(log_xs), max(log_xs)
    if min_x == max_x:
        min_x -= 0.5
        max_x += 0.5

    min_y = max(0.0, min(ys_all) - 0.02)
    max_y = min(1.0, max(ys_all) + 0.02)
    if min_y == max_y:
        min_y = max(0.0, min_y - 0.05)
        max_y = min(1.0, max_y + 0.05)

    def x_to_px(log_x):
        return left + (log_x - min_x) / (max_x - min_x) * plot_w

    def y_to_px(y):
        return top + (max_y - y) / (max_y - min_y) * plot_h

    matched_pts = [(x_to_px(math.log10(r.eps)), y_to_px(r.accuracy)) for r in records]
    mismatched_pts = [(x_to_px(math.log10(r.eps)), y_to_px(r.mismatched_accuracy)) for r in records]

    # Grid/ticks
    y_ticks = 6
    x_ticks = sorted(set(int(round(v)) for v in log_xs))
    if len(x_ticks) == 1:
        x_ticks = [x_ticks[0] - 1, x_ticks[0], x_ticks[0] + 1]

    lines = []
    lines.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
    )
    lines.append('<rect x="0" y="0" width="100%" height="100%" fill="white"/>')
    lines.append('<text x="460" y="26" text-anchor="middle" font-size="20" font-family="Arial">MNLI MeZO Sweep (h vs accuracy)</text>')

    # Horizontal grid + labels
    for i in range(y_ticks + 1):
        y_val = min_y + (max_y - min_y) * i / y_ticks
        py = y_to_px(y_val)
        lines.append(f'<line x1="{left}" y1="{py:.2f}" x2="{width-right}" y2="{py:.2f}" stroke="#e6e6e6" stroke-width="1"/>')
        lines.append(f'<text x="{left-10}" y="{py+4:.2f}" text-anchor="end" font-size="12" font-family="Arial">{y_val:.3f}</text>')

    # Vertical grid + labels (log10 scale)
    for xt in x_ticks:
        px = x_to_px(float(xt))
        lines.append(f'<line x1="{px:.2f}" y1="{top}" x2="{px:.2f}" y2="{height-bottom}" stroke="#f0f0f0" stroke-width="1"/>')
        lines.append(f'<text x="{px:.2f}" y="{height-bottom+22}" text-anchor="middle" font-size="12" font-family="Arial">1e{xt}</text>')

    # Axes
    lines.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="#222" stroke-width="1.5"/>')
    lines.append(f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="#222" stroke-width="1.5"/>')
    lines.append(f'<text x="{left-58}" y="{top+plot_h/2:.2f}" transform="rotate(-90 {left-58},{top+plot_h/2:.2f})" text-anchor="middle" font-size="13" font-family="Arial">Accuracy</text>')
    lines.append(f'<text x="{left+plot_w/2:.2f}" y="{height-20}" text-anchor="middle" font-size="13" font-family="Arial">h / zo_eps (log scale)</text>')

    # Lines
    lines.append(f'<polyline points="{_polyline(matched_pts)}" fill="none" stroke="#1f77b4" stroke-width="2.3"/>')
    lines.append(f'<polyline points="{_polyline(mismatched_pts)}" fill="none" stroke="#ff7f0e" stroke-width="2.3"/>')

    # Points + labels
    for r, (px, py) in zip(records, matched_pts):
        lines.append(f'<circle cx="{px:.2f}" cy="{py:.2f}" r="3.8" fill="#1f77b4"/>')
        lines.append(f'<text x="{px:.2f}" y="{py-8:.2f}" text-anchor="middle" font-size="10" font-family="Arial" fill="#1f77b4">{r.accuracy:.3f}</text>')
    for r, (px, py) in zip(records, mismatched_pts):
        lines.append(f'<rect x="{px-3.3:.2f}" y="{py-3.3:.2f}" width="6.6" height="6.6" fill="#ff7f0e"/>')
        lines.append(f'<text x="{px:.2f}" y="{py+15:.2f}" text-anchor="middle" font-size="10" font-family="Arial" fill="#ff7f0e">{r.mismatched_accuracy:.3f}</text>')

    # Legend
    lx, ly = width - right - 245, top + 10
    lines.append(f'<rect x="{lx}" y="{ly}" width="235" height="54" fill="white" stroke="#ccc"/>')
    lines.append(f'<line x1="{lx+14}" y1="{ly+18}" x2="{lx+50}" y2="{ly+18}" stroke="#1f77b4" stroke-width="2.3"/>')
    lines.append(f'<circle cx="{lx+32}" cy="{ly+18}" r="3.8" fill="#1f77b4"/>')
    lines.append(f'<text x="{lx+60}" y="{ly+22}" font-size="12" font-family="Arial">validation_matched</text>')
    lines.append(f'<line x1="{lx+14}" y1="{ly+38}" x2="{lx+50}" y2="{ly+38}" stroke="#ff7f0e" stroke-width="2.3"/>')
    lines.append(f'<rect x="{lx+29}" y="{ly+35}" width="6.6" height="6.6" fill="#ff7f0e"/>')
    lines.append(f'<text x="{lx+60}" y="{ly+42}" font-size="12" font-family="Arial">validation_mismatched</text>')

    # Notes
    note = f"n={len(records)} run(s)"
    lines.append(f'<text x="{left}" y="{height-8}" font-size="11" font-family="Arial" fill="#666">{html.escape(note)}</text>')
    lines.append("</svg>")

    with open(out_svg, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse MNLI logs and draw sweep chart (pure stdlib SVG).")
    parser.add_argument(
        "--jobs-glob",
        default="/Users/jichaoyu/Documents/GitHub/MeZO/large_models/sh_file/MNLI/**/jobs/*.out",
        help="Glob pattern for job .out files (supports **).",
    )
    parser.add_argument(
        "--out-csv",
        default="/Users/jichaoyu/Documents/GitHub/MeZO/large_models/sh_file/MNLI/mnli_h_sweep_summary.csv",
    )
    parser.add_argument(
        "--out-svg",
        default="/Users/jichaoyu/Documents/GitHub/MeZO/large_models/sh_file/MNLI/mnli_h_sweep_plot.svg",
    )
    args = parser.parse_args()

    files = sorted(glob.glob(args.jobs_glob, recursive=True))
    records = []
    for p in files:
        rec = parse_one_log(p)
        if rec is not None:
            records.append(rec)

    if not records:
        print("No parseable MNLI result logs found.")
        return

    records = dedup_records(records)
    save_csv(records, args.out_csv)
    save_svg(records, args.out_svg)

    print(f"Parsed logs: {len(records)}")
    print(f"CSV saved: {args.out_csv}")
    print(f"SVG saved: {args.out_svg}")
    for r in records:
        print(
            f"eps={r.eps:.10g} matched={r.accuracy:.6f} mismatched={r.mismatched_accuracy:.6f} file={r.log_file}"
        )


if __name__ == "__main__":
    main()
