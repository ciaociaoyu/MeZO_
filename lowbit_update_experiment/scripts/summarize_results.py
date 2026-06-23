#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def summarize(input_dir: str | Path) -> None:
    root = Path(input_dir)
    rows = [json.loads(line) for line in (root / "results.jsonl").read_text().splitlines() if line.strip()]
    df = pd.DataFrame(rows)
    df.to_csv(root / "summary.csv", index=False)
    if not df.empty:
        key_cols = ["update_rule"]
        if "target_layer_name" in df:
            key_cols.append("target_layer_name")
        best = df.sort_values("delta_train_loss", ascending=True).groupby(key_cols, as_index=False).head(1)
        best.to_csv(root / "best_by_rule.csv", index=False)
        plots = root / "plots"
        plots.mkdir(exist_ok=True)
        try:
            import matplotlib.pyplot as plt

            for x, y, name in [
                ("active_fraction", "delta_train_loss", "loss_delta_vs_active_fraction.png"),
                ("norm_ratio", "cosine_intended_actual", "cosine_vs_norm_ratio.png"),
            ]:
                if x in df and y in df:
                    plt.figure(figsize=(7, 5))
                    for rule, sub in df.groupby("update_rule"):
                        plt.scatter(sub[x], sub[y], s=14, label=rule, alpha=0.7)
                    plt.xlabel(x)
                    plt.ylabel(y)
                    plt.legend(fontsize=7)
                    plt.tight_layout()
                    plt.savefig(plots / name, dpi=160)
                    plt.close()
            if "update_rule" in df:
                plt.figure(figsize=(9, 5))
                df.boxplot(column="delta_train_loss", by="update_rule", rot=45)
                plt.suptitle("")
                plt.tight_layout()
                plt.savefig(plots / "train_delta_by_rule.png", dpi=160)
                plt.close()
        except Exception as exc:
            (plots / "plot_error.txt").write_text(repr(exc) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    args = parser.parse_args()
    summarize(args.input_dir)
    print(f"wrote summary files under {args.input_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
