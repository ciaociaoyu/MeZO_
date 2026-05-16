#!/usr/bin/env python3
"""Summarize next-round dense, sparse, and residual experiment runs.

This is a thin compatibility wrapper around summarize_next_experiments.py. It
keeps the explicit best/last metric fields and also writes the next-round
filename requested by the experiment plan: summary_sparse_promoted.csv.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import summarize_next_experiments  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    forwarded = [args.run_root]
    if args.no_plots:
        forwarded.append("--no-plots")

    old_argv = sys.argv[:]
    try:
        sys.argv = ["summarize_next_experiments.py", *forwarded]
        rc = summarize_next_experiments.main()
    finally:
        sys.argv = old_argv

    root = Path(args.run_root).resolve()
    promoted = root / "summary_promoted.csv"
    sparse_promoted = root / "summary_sparse_promoted.csv"
    if promoted.exists():
        shutil.copyfile(promoted, sparse_promoted)
    elif not sparse_promoted.exists():
        sparse_promoted.write_text("", encoding="utf-8")

    return rc


if __name__ == "__main__":
    raise SystemExit(main())
