#!/usr/bin/env python3
import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean, pstdev


FIELDS = [
    "p",
    "h_raw",
    "h_active",
    "sparse_mode",
    "sparse_rescale",
    "num_probe_directions",
    "probe_active_frac_mean",
    "probe_alignment_mean",
    "probe_norm_ratio_mean",
    "fd_zero_ratio",
    "fd_mean",
    "fd_std",
    "corr_fd_true",
    "nMSE_fd_true",
]


def finite_float(value):
    try:
        out = float(value)
    except Exception:
        return None
    return out if math.isfinite(out) else None


def collect_values(rows, key):
    vals = []
    for row in rows:
        val = finite_float(row.get(key))
        if val is not None:
            vals.append(val)
    return vals


def summarize_group(rows):
    first = rows[0]
    fd_vals = collect_values(rows, "fd_mean")
    corr_vals = collect_values(rows, "corr")
    nmse_vals = collect_values(rows, "nmse")
    return {
        "p": finite_float(first.get("p", first.get("direction_sparse_rate", 1.0))),
        "h_raw": finite_float(first.get("h_raw", first.get("eps"))),
        "h_active": finite_float(first.get("h_active")),
        "sparse_mode": first.get("sparse_mode", "none"),
        "sparse_rescale": first.get("sparse_rescale", "none"),
        "num_probe_directions": int(finite_float(first.get("num_probe_directions", first.get("probe_num_seeds", 0))) or 0),
        "probe_active_frac_mean": (mean(collect_values(rows, "probe_active_frac")) if collect_values(rows, "probe_active_frac") else None),
        "probe_alignment_mean": (mean(collect_values(rows, "probe_alignment")) if collect_values(rows, "probe_alignment") else None),
        "probe_norm_ratio_mean": (mean(collect_values(rows, "probe_norm_ratio")) if collect_values(rows, "probe_norm_ratio") else None),
        "fd_zero_ratio": (mean(collect_values(rows, "fd_zero_ratio")) if collect_values(rows, "fd_zero_ratio") else None),
        "fd_mean": (mean(fd_vals) if fd_vals else None),
        "fd_std": (pstdev(fd_vals) if len(fd_vals) > 1 else 0.0 if fd_vals else None),
        "corr_fd_true": (mean(corr_vals) if corr_vals else None),
        "nMSE_fd_true": (mean(nmse_vals) if nmse_vals else None),
    }


def read_jsonl(path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            row["_source"] = str(path)
            rows.append(row)
    return rows


def write_markdown(rows, path):
    with path.open("w", encoding="utf-8") as f:
        f.write("| " + " | ".join(FIELDS) + " |\n")
        f.write("| " + " | ".join(["---"] * len(FIELDS)) + " |\n")
        for row in rows:
            vals = []
            for field in FIELDS:
                val = row.get(field)
                if isinstance(val, float):
                    vals.append(f"{val:.6g}")
                else:
                    vals.append("" if val is None else str(val))
            f.write("| " + " | ".join(vals) + " |\n")


def maybe_plot(rows, out_dir):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return []
    plots = []
    specs = [
        ("h_raw", "probe_alignment_mean", "probe_alignment_vs_h_raw.png"),
        ("h_active", "probe_alignment_mean", "probe_alignment_vs_h_active.png"),
        ("h_active", "probe_norm_ratio_mean", "probe_norm_ratio_vs_h_active.png"),
        ("h_active", "fd_zero_ratio", "fd_zero_ratio_vs_h_active.png"),
        ("h_active", "corr_fd_true", "corr_vs_h_active.png"),
        ("h_active", "nMSE_fd_true", "nmse_vs_h_active.png"),
    ]
    groups = {}
    for row in rows:
        groups.setdefault(row.get("p"), []).append(row)
    for x_key, y_key, filename in specs:
        fig, ax = plt.subplots(figsize=(6, 4))
        drew = False
        for p, group in sorted(groups.items(), key=lambda kv: float(kv[0] or 0.0)):
            pts = []
            for row in group:
                x = finite_float(row.get(x_key))
                y = finite_float(row.get(y_key))
                if x is not None and y is not None:
                    pts.append((x, y))
            if not pts:
                continue
            pts.sort()
            ax.plot([x for x, _ in pts], [y for _, y in pts], marker="o", label=f"p={p:g}")
            drew = True
        if drew:
            ax.set_xscale("log")
            ax.set_xlabel(x_key)
            ax.set_ylabel(y_key)
            ax.grid(True, which="both", alpha=0.3)
            ax.legend()
            out = out_dir / filename
            fig.tight_layout()
            fig.savefig(out, dpi=160)
            plots.append(str(out))
        plt.close(fig)
    return plots


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_root", type=Path)
    args = parser.parse_args()
    run_root = args.run_root
    logs = sorted(run_root.glob("**/probe_stats.jsonl"))
    all_rows = []
    for path in logs:
        all_rows.extend(read_jsonl(path))
    groups = {}
    for row in all_rows:
        p = finite_float(row.get("p", row.get("direction_sparse_rate", 1.0)))
        h = finite_float(row.get("h_raw", row.get("eps")))
        mode = row.get("sparse_mode", "none")
        rescale = row.get("sparse_rescale", "none")
        groups.setdefault((p, h, mode, rescale), []).append(row)
    summary = [summarize_group(rows) for _, rows in sorted(groups.items(), key=lambda kv: (kv[0][0] or 0.0, kv[0][1] or 0.0))]
    csv_path = run_root / "summary.csv"
    md_path = run_root / "summary.md"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(summary)
    write_markdown(summary, md_path)
    plots = maybe_plot(summary, run_root)
    print(f"rows={len(summary)}")
    print(f"summary_csv={csv_path}")
    print(f"summary_md={md_path}")
    for plot in plots:
        print(f"plot={plot}")


if __name__ == "__main__":
    main()
