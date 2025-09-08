#!/usr/bin/env python3
"""
DrawpFigure.py

Usage:
  python DrawpFigure.py /path/to/metrics.csv [--xcol STEP_COL] [--out /path/to/output.png] [--title "Figure Title"]

This script reads a CSV file, finds a column named 'train_loss' (case-insensitive),
chooses a sensible x-axis (given via --xcol or guessed from common step/epoch names or the row index),
and saves a line chart next to the CSV by default: <basename>_train_loss.png
"""

import argparse
import sys
from pathlib import Path
from typing import Optional, Union

import pandas as pd
import matplotlib.pyplot as plt


COMMON_XCANDS = [
    "step", "global_step", "iterations", "iteration", "iter",
    "epoch", "epoch_id", "step_idx", "step_index",
]


def find_column_case_insensitive(columns, target):
    """Return the real column name in `columns` matching `target` case-insensitively, else None."""
    lower_map = {c.lower(): c for c in columns}
    return lower_map.get(target.lower())


def guess_x_column(df, preferred=None):
    """Pick an x-axis column.

    Priority:
      1) `preferred` if provided and exists
      2) First match from COMMON_XCANDS
      3) None (use index)
    """
    if preferred:
        real = find_column_case_insensitive(df.columns, preferred)
        if real is not None:
            return real
        # fallthrough to guess if provided but missing
    for cand in COMMON_XCANDS:
        real = find_column_case_insensitive(df.columns, cand)
        if real is not None:
            return real
    return None  # use index


def resolve_train_loss_column(df):
    # try exact match first
    exact = find_column_case_insensitive(df.columns, "train_loss")
    if exact is not None:
        return exact
    # try contains-like fallbacks (e.g., "loss" when only training loss is recorded)
    lowers = {c.lower(): c for c in df.columns}
    for key in ["train_loss", "training_loss", "loss_train", "loss"]:
        if key in lowers:
            return lowers[key]
    return None


def plot_train_loss(
    csv_path: Union[str, Path],
    xcol: Optional[str],
    out_path: Optional[Union[str, Path]],
    title: Optional[str],
):
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    # Read CSV
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        raise RuntimeError(f"Failed to read CSV '{csv_path}': {e}")

    if df.empty:
        raise ValueError("CSV is empty — no data to plot.")

    # Find y column (train_loss)
    ycol = resolve_train_loss_column(df)
    if ycol is None:
        raise ValueError("Could not find 'train_loss' (case-insensitive) in the CSV columns: "
                         f"{list(df.columns)}")

    # Decide x axis
    real_xcol = guess_x_column(df, xcol)

    # Build output path
    if out_path is None:
        out_path = csv_path.with_name(csv_path.stem + "_train_loss.png")
    else:
        out_path = Path(out_path)

    # Plot
    plt.figure()
    if real_xcol is not None:
        x = df[real_xcol]
        plt.plot(x, df[ycol])
        plt.xlabel(real_xcol)
    else:
        plt.plot(df.index, df[ycol])
        plt.xlabel("index")

    plt.ylabel(ycol)
    plt.title(title if title else f"Train Loss from {csv_path.name}")
    plt.grid(True, linewidth=0.5, alpha=0.6)
    plt.tight_layout()

    # Save
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()

    return out_path


def generate_train_loss_figure(
    csv: Union[str, Path],
    xcol: Optional[str] = None,
    out: Optional[Union[str, Path]] = None,
    title: Optional[str] = None,
) -> Path:
    """Programmatic API: generate the train_loss figure and return the saved Path.

    Examples
    --------
    >>> from DrawpFigure import generate_train_loss_figure
    >>> p = generate_train_loss_figure("/path/to/metrics.csv", xcol="step")
    >>> print(p)
    """
    return plot_train_loss(csv, xcol, out, title)


def parse_args(argv):
    p = argparse.ArgumentParser(description="Plot train_loss line chart from a CSV.")
    p.add_argument("csv", type=str, help="Path to the metrics CSV file.")
    p.add_argument("--xcol", type=str, default=None, help="Column to use as x-axis (optional).")
    p.add_argument("--out", type=str, default=None, help="Output image path (optional).")
    p.add_argument("--title", type=str, default=None, help="Custom chart title (optional).")
    return p.parse_args(argv)


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    try:
        saved = plot_train_loss(args.csv, args.xcol, args.out, args.title)
        print(f"Saved train_loss figure to: {saved}")
    except Exception as e:
        print(f"[ERROR] {e}")
        sys.exit(1)