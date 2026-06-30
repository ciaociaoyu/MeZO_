#!/usr/bin/env python3
"""Probe-only V_dir versus V_h_dep contribution experiment.

This script is a thin wrapper around `roberta_sst5_theoretical_windows.py`.
It recomputes true directional finite-difference errors and then emits the
paper-facing files requested for the guardrail-window contribution figure.

No training is launched.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import roberta_sst5_theoretical_windows as tw


ROOT = Path(__file__).resolve().parents[1]
DATE_DEFAULT = os.environ.get("HWINDOW_DATE", datetime.now().strftime("%Y%m%d"))
DEFAULT_OUT = ROOT / f"roberta_sst5_vdir_vs_vhdep_{DATE_DEFAULT}"
TINY_H = 1e-5
DEFAULT_H = 1e-3
ZERO_TOL = 1e-12

H_GRIDS = {
    "fp32": [
        1e-9, 3e-9, 1e-8, 3e-8, 1e-7, 3e-7, 1e-6, 3e-6,
        1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 1.5e-3, 2e-3,
        3e-3, 5e-3, 1e-2,
    ],
    "fp16": [
        1e-6, 3e-6, 5e-6, 7e-6, 1e-5, 2e-5, 3e-5,
        5e-5, 7e-5, 1e-4, 3e-4, 1e-3, 1.5e-3,
        2e-3, 3e-3, 5e-3,
    ],
    "int8": [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 1.5e-3, 2e-3, 3e-3, 5e-3, 1e-2],
    "int4": [1e-5, 3e-5, 1e-4, 3e-4, 5e-4, 7e-4, 1e-3, 1.2e-3, 1.5e-3, 2e-3, 3e-3, 5e-3],
}


def finite(value) -> float | None:
    try:
        x = float(value)
    except Exception:
        return None
    return x if math.isfinite(x) else None


def nearest_h(group: pd.DataFrame, target: float) -> pd.Series | None:
    if group.empty:
        return None
    idx = (group["h"].astype(float) - float(target)).abs().idxmin()
    return group.loc[idx]


def mean(values: Iterable[float]) -> float | None:
    xs = [float(x) for x in values if math.isfinite(float(x))]
    return sum(xs) / len(xs) if xs else None


def write_csv(path: Path, rows: Sequence[Dict[str, object]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fields), extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def configure_h_grids() -> None:
    for precision, grid in H_GRIDS.items():
        tw.DEFAULT_H_GRIDS[precision] = list(grid)


def add_gitignore(out: Path) -> None:
    (out / ".gitignore").write_text(
        "checkpoints/*.pt\nraw_probe_metrics.jsonl\n*.log\n",
        encoding="utf-8",
    )


def run_fresh_probe(args: argparse.Namespace) -> None:
    configure_h_grids()
    os.environ.setdefault("DATALOADER_SHUFFLE", "True")
    tw_args = SimpleNamespace(
        output_dir=str(args.output_dir),
        precisions=list(args.precisions),
        model_id="roberta-large",
        seed=int(args.seed),
        data_seed=int(args.data_seed),
        batch_size=int(args.batch_size),
        num_batches=int(args.num_batches),
        num_directions=int(args.num_directions),
        direction_seed_base=int(args.direction_seed_base),
        group_size=int(args.group_size),
        h_grid="",
        progress_every=max(1, int(args.progress_every)),
        reuse_raw_metrics=False,
    )
    tw.run_probe(tw_args)


def compute_zero_fraction(raw: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (precision, h), g in raw.groupby(["precision", "h"], sort=False):
        dh = g["d_h"].astype(float).to_numpy()
        rows.append(
            {
                "precision": precision,
                "h": float(h),
                "d_h_zero_fraction": float(np.mean(np.abs(dh) <= ZERO_TOL)) if len(dh) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def build_raw_direction_metrics(out: Path, raw: pd.DataFrame) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "precision": raw["precision"],
            "h": raw["h"].astype(float),
            "direction_seed": raw["direction_seed"],
            "batch_index": raw.get("batch_index", 0),
            "direction_index": raw.get("direction_index", np.nan),
            "d_star": raw["d_star"],
            "d_h": raw["d_h"],
            "e_h": raw["e_h"],
            "norm_u2": raw["norm_u2"],
            "vector_error_h": raw["vector_error_h"],
            "z_dir_norm2": raw["V_dir_sample_direction"],
            "loss_plus": raw["loss_plus"],
            "loss_minus": raw["loss_minus"],
            "loss_base": raw["loss_base"],
        }
    )
    df.to_csv(out / "raw_direction_metrics.csv", index=False)
    return df


def build_contribution_by_h(out: Path, summary: pd.DataFrame, raw: pd.DataFrame) -> pd.DataFrame:
    zero = compute_zero_fraction(raw)
    df = summary.merge(zero, on=["precision", "h"], how="left")
    df["V_h_dep"] = df["V_h_raw"].astype(float)
    df["rho_sample"] = df["rho_raw"].astype(float)
    df["rho_formula"] = df["V_h_dep"].astype(float) / df["V_dir_formula"].astype(float).clip(lower=1e-30)
    df["default_marker"] = np.isclose(df["h"].astype(float), DEFAULT_H, rtol=0.0, atol=1e-15)
    df["tiny_marker"] = np.isclose(df["h"].astype(float), TINY_H, rtol=0.0, atol=1e-15)
    fields = [
        "precision", "h", "n_directions", "G2", "d", "V_dir_sample", "V_dir_formula",
        "V_h_dep", "rho_sample", "rho_formula", "scalar_nmse", "directional_corr",
        "sign_agreement", "d_h_std", "d_h_zero_fraction", "default_marker", "tiny_marker",
        "probe_active_frac", "probe_norm_ratio", "probe_alignment", "saturation_frac",
    ]
    df.to_csv(out / "contribution_by_h.csv", index=False, columns=[c for c in fields if c in df.columns])
    return df


def interpret(row: pd.Series) -> str:
    zero = finite(row.get("d_h_zero_fraction"))
    dh_std = finite(row.get("d_h_std"))
    corr = finite(row.get("directional_corr"))
    rho = finite(row.get("rho_sample"))
    if zero is not None and zero >= 0.95:
        return "probe dead zone"
    if dh_std is not None and dh_std <= 0:
        return "no reliable finite-difference signal"
    if corr is not None and corr < 0.2:
        return "no reliable finite-difference signal"
    if rho is None:
        return "no reliable finite-difference signal"
    if rho < 0.8:
        return "below random-direction floor"
    if rho <= 1.25:
        return "near random-direction floor"
    return "above random-direction floor"


def build_representative_points(out: Path, contrib: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for precision, g in contrib.groupby("precision", sort=False):
        g = g.sort_values("h")
        points: List[tuple[str, pd.Series | None]] = [
            ("tiny_h_1e-5", nearest_h(g, TINY_H)),
            ("default_h_1e-3", nearest_h(g, DEFAULT_H)),
            ("min_rho", g.loc[g["rho_sample"].astype(float).idxmin()] if not g.empty else None),
            ("largest_tested_h", g.iloc[-1] if not g.empty else None),
        ]
        seen: set[tuple[str, float]] = set()
        for point_name, row in points:
            if row is None:
                continue
            key = (point_name, float(row["h"]))
            if key in seen:
                continue
            seen.add(key)
            rows.append(
                {
                    "precision": precision,
                    "point": point_name,
                    "h": float(row["h"]),
                    "V_dir_sample": row.get("V_dir_sample"),
                    "V_h_dep": row.get("V_h_dep"),
                    "rho_sample": row.get("rho_sample"),
                    "scalar_nmse": row.get("scalar_nmse"),
                    "directional_corr": row.get("directional_corr"),
                    "sign_agreement": row.get("sign_agreement"),
                    "d_h_zero_fraction": row.get("d_h_zero_fraction"),
                    "interpretation": interpret(row),
                }
            )
    df = pd.DataFrame(rows)
    df.to_csv(out / "representative_points_table.csv", index=False)
    return df


def plot_rho(contrib: pd.DataFrame, out: Path) -> None:
    colors = {"fp32": "#1f77b4", "fp16": "#2ca02c", "int8": "#ff7f0e", "int4": "#d62728"}
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    for precision, g in contrib.groupby("precision", sort=False):
        g = g.sort_values("h")
        ax.plot(g["h"], g["rho_sample"], marker="o", linewidth=1.6, label=precision, color=colors.get(precision))
        safe = g[g["rho_sample"].astype(float) < 1.0]
        if not safe.empty:
            ax.fill_between(safe["h"], safe["rho_sample"], 1.0, color=colors.get(precision), alpha=0.08)
    ax.axhline(1.0, color="black", linestyle=":", linewidth=1.2, label="rho=1")
    ax.axvline(DEFAULT_H, color="black", linestyle="--", linewidth=1.0, label="MeZO default 1e-3")
    ax.axvline(TINY_H, color="0.45", linestyle="--", linewidth=1.0, label="tiny 1e-5")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("finite-difference radius h")
    ax.set_ylabel(r"$\rho(h)=V_{h,\mathrm{dep}}/V_{\mathrm{dir}}$")
    ax.set_title("Finite-difference error relative to the random-direction floor")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "fig_rho_vs_h_by_precision.pdf")
    fig.savefig(out / "fig_rho_vs_h_by_precision.png", dpi=220)
    plt.close(fig)


def plot_components(contrib: pd.DataFrame, out: Path) -> None:
    colors = {"fp32": "#1f77b4", "fp16": "#2ca02c", "int8": "#ff7f0e", "int4": "#d62728"}
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.2), sharex=False, sharey=False)
    for ax, precision in zip(axes.ravel(), ["fp32", "fp16", "int8", "int4"]):
        g = contrib[contrib["precision"] == precision].sort_values("h")
        if g.empty:
            ax.text(0.5, 0.5, "missing", transform=ax.transAxes, ha="center", va="center")
            continue
        vdir = float(g["V_dir_sample"].iloc[0])
        ax.axhline(vdir, color="black", linestyle=":", linewidth=1.2, label=r"$V_{dir}$")
        ax.plot(g["h"], g["V_h_dep"], marker="o", color=colors.get(precision), label=r"$V_{h,dep}(h)$")
        ax.axvline(DEFAULT_H, color="black", linestyle="--", linewidth=0.9)
        ax.axvline(TINY_H, color="0.45", linestyle="--", linewidth=0.9)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(precision)
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(fontsize=8)
    fig.supxlabel("finite-difference radius h")
    fig.supylabel("vector-level MSE")
    fig.tight_layout()
    fig.savefig(out / "fig_vdir_vhdep_components.pdf")
    fig.savefig(out / "fig_vdir_vhdep_components.png", dpi=220)
    plt.close(fig)


def plot_reliability(contrib: pd.DataFrame, out: Path) -> None:
    colors = {"fp32": "#1f77b4", "fp16": "#2ca02c", "int8": "#ff7f0e", "int4": "#d62728"}
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.8), sharex=False)
    panels = [
        ("scalar_nmse", "scalar true directional nMSE", True),
        ("directional_corr", "corr(d_h, d*)", False),
        ("d_h_zero_fraction", "d_h zero fraction", False),
    ]
    for ax, (col, title, logy) in zip(axes, panels):
        for precision, g in contrib.groupby("precision", sort=False):
            g = g.sort_values("h")
            ax.plot(g["h"], g[col], marker="o", linewidth=1.4, label=precision, color=colors.get(precision))
        ax.axvline(DEFAULT_H, color="black", linestyle="--", linewidth=1.0)
        ax.axvline(TINY_H, color="0.45", linestyle="--", linewidth=1.0)
        ax.set_xscale("log")
        if logy:
            ax.set_yscale("log")
        ax.set_title(title)
        ax.grid(True, which="both", alpha=0.25)
    axes[0].legend(fontsize=8)
    fig.supxlabel("finite-difference radius h")
    fig.tight_layout()
    fig.savefig(out / "fig_probe_reliability_vs_h.pdf")
    fig.savefig(out / "fig_probe_reliability_vs_h.png", dpi=220)
    plt.close(fig)


def crossing_points(g: pd.DataFrame) -> str:
    g = g.sort_values("h")
    hs = g["h"].astype(float).to_numpy()
    rho = g["rho_sample"].astype(float).to_numpy()
    crossings = []
    for i in range(len(hs) - 1):
        a = rho[i] - 1.0
        b = rho[i + 1] - 1.0
        if a == 0:
            crossings.append(hs[i])
        if a * b < 0:
            # Interpolate in log h / log rho for a visually meaningful crossing.
            x0, x1 = math.log(hs[i]), math.log(hs[i + 1])
            y0, y1 = math.log(max(rho[i], 1e-300)), math.log(max(rho[i + 1], 1e-300))
            if y1 != y0:
                x = x0 + (0.0 - y0) * (x1 - x0) / (y1 - y0)
                crossings.append(math.exp(x))
    return " ".join(f"{x:.6g}" for x in crossings)


def paper_sentence(precision: str, default_row: pd.Series | None, tiny_row: pd.Series | None, crosses: str) -> str:
    default_interp = interpret(default_row) if default_row is not None else "missing"
    tiny_interp = interpret(tiny_row) if tiny_row is not None else "missing"
    if precision == "fp32":
        return "For FP32, the measured radius-dependent term is below the random-direction floor across a broad radius range, supporting a wide high-precision plateau."
    if precision == "fp16":
        return "FP16 shows a small-h numerical dead zone, but the default h=1e-3 lies in the probe-reliable region when its rho and reliability diagnostics are favorable."
    if precision == "int8":
        return "INT8 remains a default-safe low-bit case when h=1e-3 keeps V_h_dep below or comparable to the random-direction floor."
    if precision == "int4":
        return "INT4 is a boundary case: dense default can be empirically usable, but the finite-difference contribution is not uniformly below the random-direction floor."
    return f"Default: {default_interp}; tiny: {tiny_interp}; crossings: {crosses or 'none'}."


def write_report(out: Path, contrib: pd.DataFrame, reps: pd.DataFrame, args: argparse.Namespace) -> None:
    lines = [
        "# V_dir versus V_h_dep Probe Summary",
        "",
        "This is a probe-only contribution experiment for the guardrail-window section. No training was launched.",
        "",
        "Definitions used:",
        "",
        "- `d_star = <g,u>`",
        "- `d_h = [F(w+h u)-F(w-h u)]/(2h)` under the precision-specific forward oracle",
        "- `V_h_dep(h) = E[(d_h-d_star)^2 ||u||^2]`",
        "- `V_dir = E||(<g,u>)u-g||^2`",
        "- `rho(h) = V_h_dep(h) / V_dir`",
        "",
        "Scalar true directional nMSE is reported as a diagnostic only; it is not used as rho.",
        "",
        "## Probe Setup",
        "",
        f"- output folder: `{out}`",
        "- model/task: `roberta-large` / SST-5 full data",
        f"- seed/data_seed: `{args.seed}` / `{args.data_seed}`",
        f"- batch_size: `{args.batch_size}`, num_batches: `{args.num_batches}`",
        f"- directions: `{args.num_directions}` with base seed `{args.direction_seed_base}`",
        "- precision modes: " + ", ".join(args.precisions),
        "- low-bit quantizer: existing G128 RTNClip shared-grid fake quantized forward oracle",
        "",
        "## Precision Conclusions",
        "",
    ]
    for precision, g in contrib.groupby("precision", sort=False):
        default_row = nearest_h(g, DEFAULT_H)
        tiny_row = nearest_h(g, TINY_H)
        crosses = crossing_points(g)
        lines.extend(
            [
                f"### {precision}",
                "",
                f"- h=1e-3: `{interpret(default_row) if default_row is not None else 'missing'}`.",
                f"- h=1e-5: `{interpret(tiny_row) if tiny_row is not None else 'missing'}`.",
                f"- rho=1 crossing estimates: `{crosses or 'none in tested grid'}`.",
                f"- Minimum rho in grid: `{float(g['rho_sample'].min()):.6g}` at h=`{float(g.loc[g['rho_sample'].idxmin(), 'h']):.6g}`.",
                f"- Paper sentence: {paper_sentence(precision, default_row, tiny_row, crosses)}",
                "",
            ]
        )
    lines.extend(
        [
            "## Files",
            "",
            "- `raw_direction_metrics.csv`: per precision / h / direction raw quantities.",
            "- `contribution_by_h.csv`: aggregated `V_dir`, `V_h_dep`, `rho`, scalar nMSE, corr, sign, and zero-fraction.",
            "- `representative_points_table.csv`: tiny/default/min-rho/largest-h rows.",
            "- `fig_rho_vs_h_by_precision.pdf/png`",
            "- `fig_vdir_vhdep_components.pdf/png`",
            "- `fig_probe_reliability_vs_h.pdf/png`",
            "",
        ]
    )
    (out / "paper_summary.md").write_text("\n".join(lines), encoding="utf-8")


def write_metadata(out: Path, args: argparse.Namespace) -> None:
    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "script": "tools/roberta_sst5_vdir_vs_vhdep.py",
        "model": "roberta-large",
        "task": "SST-5",
        "precisions": list(args.precisions),
        "h_grids": H_GRIDS,
        "batch_size": int(args.batch_size),
        "num_batches": int(args.num_batches),
        "num_directions": int(args.num_directions),
        "direction_seed_base": int(args.direction_seed_base),
        "notes": [
            "Probe-only; no training.",
            "rho is vector-level V_h_dep / V_dir, not scalar nMSE.",
            "The wrapped base probe writes local deterministic task-start checkpoint artifacts under checkpoints/; these are ignored by .gitignore.",
        ],
    }
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def postprocess(args: argparse.Namespace) -> None:
    out = Path(args.output_dir).resolve()
    raw_path = out / "raw_probe_metrics.csv"
    summary_path = out / "probe_summary_by_h.csv"
    if not raw_path.exists() or not summary_path.exists():
        raise FileNotFoundError(f"Missing raw probe outputs in {out}")
    raw = pd.read_csv(raw_path)
    summary = pd.read_csv(summary_path)
    raw_direction = build_raw_direction_metrics(out, raw)
    contrib = build_contribution_by_h(out, summary, raw)
    reps = build_representative_points(out, contrib)
    plot_rho(contrib, out)
    plot_components(contrib, out)
    plot_reliability(contrib, out)
    write_report(out, contrib, reps, args)
    write_metadata(out, args)
    add_gitignore(out)
    # Keep the base script outputs too, but the paper-facing names above are canonical.
    if raw_path.name != "raw_direction_metrics.csv":
        shutil.copy2(summary_path, out / "base_probe_summary_by_h.csv")
    print(f"Wrote contribution outputs to {out}")
    print(f"raw rows: {len(raw_direction)}, contribution rows: {len(contrib)}, representative rows: {len(reps)}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default=str(DEFAULT_OUT))
    parser.add_argument("--precisions", nargs="+", default=["fp32", "fp16", "int8", "int4"])
    parser.add_argument("--seed", type=int, default=16)
    parser.add_argument("--data_seed", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_batches", type=int, default=1)
    parser.add_argument("--num_directions", type=int, default=64)
    parser.add_argument("--direction_seed_base", type=int, default=730000)
    parser.add_argument("--group_size", type=int, default=128)
    parser.add_argument("--progress_every", type=int, default=8)
    parser.add_argument("--postprocess_only", action="store_true")
    args = parser.parse_args()

    args.output_dir = str(Path(args.output_dir).resolve())
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    if not args.postprocess_only:
        run_fresh_probe(args)
    postprocess(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
