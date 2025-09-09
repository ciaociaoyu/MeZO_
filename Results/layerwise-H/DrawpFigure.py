#!/usr/bin/env python3
"""
DrawpFigure.py

This script provides a programmatic API to plot training loss from a CSV file.

Run this file directly; configuration is set in MAIN_CFG below.

You can configure a maximum y-axis limit (ymax) if needed.
"""

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
    ma_window: Optional[int],
    ymax: Optional[float],
    ymin: Optional[float],
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

    y = df[ycol]
    if ma_window is not None and isinstance(ma_window, int) and ma_window > 1:
        y = y.rolling(window=ma_window, min_periods=1).mean()

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
        plt.plot(x, y)
        plt.xlabel(real_xcol)
    else:
        plt.plot(df.index, y)
        plt.xlabel("index")

    plt.ylabel(ycol)
    final_title = title if title else f"Train Loss from {csv_path.name}"
    if ma_window is not None and isinstance(ma_window, int) and ma_window > 1:
        final_title += f" (MA window={ma_window})"
    plt.title(final_title)
    plt.grid(True, linewidth=0.5, alpha=0.6)
    plt.tight_layout()

    if ymax is not None or ymin is not None:
        plt.ylim(bottom=ymin, top=ymax)

    # Save
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()

    return out_path


# ---- Simple in-file configuration (no CLI) ----
MAIN_CFG = {
    # Set your CSV path here. Example defaults to the file beside this script or to your known dataset path.
    "csv": Path("metrics_adaptiveH-0_cscale-0_layerwiseH-0.csv"),  # change if needed
    # Optional x-axis column; set to None to auto-guess
    "xcol": None,  # e.g., "step" or "epoch"
    # Optional custom output path; set to None to save next to CSV as *_train_loss.png
    "out": None,
    # Optional chart title; set to None to auto-generate
    "title": None,
    # Moving-average window; use 1 or None to disable smoothing
    "ma_window": 50,
    # Optional maximum y-axis limit
    "ymax": 0.7,
    # Optional minimum y-axis limit
    "ymin": 0,
}


def main():
    try:
        saved = plot_train_loss(
            MAIN_CFG["csv"],
            MAIN_CFG.get("xcol"),
            MAIN_CFG.get("out"),
            MAIN_CFG.get("title"),
            MAIN_CFG.get("ma_window"),
            MAIN_CFG.get("ymax"),
            MAIN_CFG.get("ymin"),
        )
        print(f"Saved train_loss figure to: {saved}")
    except Exception as e:
        print(f"[ERROR] {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()